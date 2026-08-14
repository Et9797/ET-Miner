"""
CUDA Kernel for Direct CSR-to-Bitvec Conversion on GPU

Phase 2 GPU Pipeline Optimization:
- Instead of building bitvecs on CPU (3GB) and transferring to GPU
- Transfer CSR data (~100MB) to GPU and build bitvecs there
- Expected: 3.6s → 0.02s transfer, 4s → 0.5s build

The kernel converts CSR (Compressed Sparse Row) format directly to
column-oriented bitvector format on the GPU, enabling massive parallelism
for the conversion.

Memory layout:
- Input CSR:
  - indptr: [n_rows + 1] row pointers
  - indices: [nnz] column indices
- Output bitvecs: [n_cols, n_u64s] where n_u64s = ceil(n_rows / 64)
  - Each column gets a bitvector with 1 bit per row
  - Bit i in bitvec[col] is set if row i contains column col

"""

import numpy as np
import threading
from typing import Optional

from loguru import logger

__all__ = [
    'build_bitvecs_gpu',
    'build_bitvecs_from_gpu_arrays',
    'build_bitvecs_row_split_from_arrays',
    'get_csr_to_bitvec_kernel',
    'clear_csr_kernel_cache',
    'PinnedBufferPool',
    'get_default_pinned_pool',
]


# =============================================================================
# Pinned Memory Buffer Pool (Phase 3 optimization: 3x faster transfers)
# =============================================================================

class PinnedBufferPool:
    """
    Reusable pinned memory buffer pool for fast GPU transfers.

    Pinned memory enables DMA (Direct Memory Access) transfers at ~37 GB/s
    vs ~11 GB/s for unpinned memory. This pool pre-allocates and reuses
    buffers to avoid allocation overhead in the hot path.

    Usage:
        pool = PinnedBufferPool()

        for chunk in chunks:
            # Get buffers (automatically sized)
            indptr_buf, indices_buf = pool.get_buffers(indptr, indices)

            # Transfer to GPU (3x faster!)
            indptr_gpu = cp.asarray(indptr_buf)
            indices_gpu = cp.asarray(indices_buf)
    """

    def __init__(self, initial_size: int = 0):
        """
        Initialize the buffer pool.

        Args:
            initial_size: Pre-allocate buffers of this size (bytes).
                         If 0, buffers are allocated on first use.
        """
        self._lock = threading.Lock()
        self._indptr_buf = None
        self._indices_buf = None
        self._indptr_size = 0
        self._indices_size = 0
        self._initialized = False

        if initial_size > 0:
            self._allocate(initial_size, initial_size)

    def _allocate(self, indptr_bytes: int, indices_bytes: int):
        """Allocate or reallocate pinned buffers."""
        import cupy as cp

        # Only reallocate if needed (grow-only)
        if indptr_bytes > self._indptr_size:
            self._indptr_buf = cp.cuda.alloc_pinned_memory(indptr_bytes)
            self._indptr_size = indptr_bytes

        if indices_bytes > self._indices_size:
            self._indices_buf = cp.cuda.alloc_pinned_memory(indices_bytes)
            self._indices_size = indices_bytes

        self._initialized = True

    def get_buffers(
        self,
        indptr: np.ndarray,
        indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get pinned memory buffers with data copied in.

        Args:
            indptr: CSR indptr array
            indices: CSR indices array

        Returns:
            Tuple of (pinned_indptr, pinned_indices) numpy arrays
            backed by pinned memory.
        """
        with self._lock:
            # Ensure int64 dtype
            if indptr.dtype != np.int64:
                indptr = indptr.astype(np.int64)
            if indices.dtype != np.int64:
                indices = indices.astype(np.int64)

            # Allocate/resize if needed
            if indptr.nbytes > self._indptr_size or indices.nbytes > self._indices_size:
                self._allocate(indptr.nbytes, indices.nbytes)

            # Create numpy views into pinned memory
            pinned_indptr = np.frombuffer(
                self._indptr_buf, dtype=np.int64, count=len(indptr)
            )
            pinned_indices = np.frombuffer(
                self._indices_buf, dtype=np.int64, count=len(indices)
            )

            # Copy data to pinned buffers (fast memcpy)
            np.copyto(pinned_indptr, indptr)
            np.copyto(pinned_indices, indices)

            return pinned_indptr, pinned_indices

    def clear(self):
        """Release all buffers."""
        with self._lock:
            self._indptr_buf = None
            self._indices_buf = None
            self._indptr_size = 0
            self._indices_size = 0
            self._initialized = False


# Global default pool (lazy initialized)
_default_pinned_pool: Optional[PinnedBufferPool] = None


def get_default_pinned_pool() -> PinnedBufferPool:
    """Get the default global pinned buffer pool."""
    global _default_pinned_pool
    if _default_pinned_pool is None:
        _default_pinned_pool = PinnedBufferPool()
    return _default_pinned_pool


# =============================================================================
# CUDA Kernel Source
# =============================================================================

CSR_TO_BITVEC_KERNEL = r'''
// Convert CSR format to column-oriented bitvectors on GPU
// Each thread handles one row, setting bits for all items in that row
//
// NOTE: Uses long long for ALL index calculations to support >2B elements
// This is critical for large-scale mining (10B+ transactions)

extern "C" __global__
void csr_to_bitvec(
    const long long* __restrict__ csr_indptr,   // [n_rows + 1] row pointers
    const long long* __restrict__ csr_indices,  // [nnz] column indices
    unsigned long long* __restrict__ bitvecs,   // [n_cols, n_u64s] output
    const long long n_rows,
    const long long n_cols,
    const long long n_u64s
) {
    // Each thread processes one row
    long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // Get column indices for this row from CSR format
    long long start = csr_indptr[row];
    long long end = csr_indptr[row + 1];

    // Calculate which u64 word and which bit within that word
    // This row maps to bit (row % 64) in word (row / 64)
    long long word_idx = row / 64;
    unsigned long long bit_mask = 1ULL << (row % 64);

    // For each column in this row, set the corresponding bit
    for (long long i = start; i < end; i++) {
        long long col = csr_indices[i];
        // bitvecs layout: [col, word_idx] = bitvecs[col * n_u64s + word_idx]
        // Use atomicOr for thread-safe bit setting (multiple rows may write same word)
        atomicOr(&bitvecs[col * n_u64s + word_idx], bit_mask);
    }
}
'''


# =============================================================================
# Kernel Cache (same pattern as cuda_kernels.py)
# =============================================================================

_csr_kernel_cache = {}


def clear_csr_kernel_cache():
    """Clear compiled kernel cache. Call after modifying kernel source."""
    global _csr_kernel_cache
    _csr_kernel_cache = {}


def get_csr_to_bitvec_kernel():
    """Get compiled CUDA kernel for CSR→bitvec conversion, caching for reuse.

    Returns:
        Compiled CuPy RawKernel ready for execution.

    Raises:
        ImportError: If CuPy is not available.
    """
    import cupy as cp

    if 'csr_to_bitvec' not in _csr_kernel_cache:
        _csr_kernel_cache['csr_to_bitvec'] = cp.RawKernel(
            CSR_TO_BITVEC_KERNEL,
            'csr_to_bitvec'
        )

    return _csr_kernel_cache['csr_to_bitvec']


# =============================================================================
# Python Wrapper
# =============================================================================

def build_bitvecs_gpu(
    csr_indptr: np.ndarray,
    csr_indices: np.ndarray,
    n_rows: int,
    n_cols: int,
    device_id: int = 0,
    buffer_pool: Optional[PinnedBufferPool] = None,
) -> "cp.ndarray":
    """
    Build bitvecs directly on GPU from CSR data.

    This is the key optimization for Phase 2 GPU pipeline:
    - Transfer CSR (~100MB) instead of bitvecs (~3GB) to GPU
    - Build bitvecs in parallel on GPU using thousands of CUDA threads

    Phase 3 optimization: Pinned memory for 3x faster transfers!
    - Normal transfer: ~12 GB/s
    - Pinned transfer: ~37 GB/s (with buffer_pool)

    Args:
        csr_indptr: numpy array of row pointers [n_rows + 1], dtype int64/int32
        csr_indices: numpy array of column indices [nnz], dtype int64/int32
        n_rows: number of rows (transactions)
        n_cols: number of columns (items)
        device_id: GPU device to use (default 0)
        buffer_pool: Optional PinnedBufferPool for 3x faster transfers.
                    If None, uses unpinned transfers. For best performance,
                    create one pool and reuse it for all chunks.

    Returns:
        CuPy array of shape [n_cols, n_u64s] on GPU, where n_u64s = ceil(n_rows/64)
        Each element is a uint64 containing 64 transaction bits for that column.

    Raises:
        ImportError: If CuPy is not available.
        RuntimeError: If GPU initialization fails.

    Example:
        >>> import numpy as np
        >>> from et_miner.cuda_csr_bitvec import PinnedBufferPool
        >>> # Create pool ONCE, reuse for all chunks
        >>> pool = PinnedBufferPool()
        >>> indptr = np.array([0, 2, 3, 6], dtype=np.int64)
        >>> indices = np.array([0, 2, 1, 0, 1, 3], dtype=np.int64)
        >>> bitvecs = build_bitvecs_gpu(indptr, indices, n_rows=3, n_cols=4, buffer_pool=pool)
    """
    import cupy as cp

    # Set device
    with cp.cuda.Device(device_id):
        # Calculate output dimensions
        n_u64s = (n_rows + 63) // 64  # ceil(n_rows / 64)

        # Overflow check: n_cols * n_u64s must fit in int64 for kernel indexing
        max_int64 = (1 << 63) - 1
        if n_cols > 0 and n_u64s > max_int64 // n_cols:
            raise OverflowError(
                f"Bitvec dimensions overflow int64: n_cols={n_cols} * n_u64s={n_u64s} "
                f"exceeds {max_int64}. Reduce n_rows or n_cols."
            )

        # Ensure input arrays are int64 for kernel compatibility
        # (kernel uses long long for all indices)
        if csr_indptr.dtype != np.int64:
            csr_indptr = csr_indptr.astype(np.int64)
        if csr_indices.dtype != np.int64:
            csr_indices = csr_indices.astype(np.int64)

        # Transfer CSR data to GPU
        # Phase 3: Use pinned memory pool for 3x faster DMA transfers
        if buffer_pool is not None:
            # Use provided buffer pool (reuses memory, avoids allocation overhead)
            pinned_indptr, pinned_indices = buffer_pool.get_buffers(csr_indptr, csr_indices)
            indptr_gpu = cp.asarray(pinned_indptr)
            indices_gpu = cp.asarray(pinned_indices)
        else:
            # Standard unpinned transfer (slower but simpler)
            indptr_gpu = cp.asarray(csr_indptr)
            indices_gpu = cp.asarray(csr_indices)

        # Allocate output bitvecs on GPU, zero-initialized
        # Zero-init is critical: atomicOr only sets bits, doesn't clear them
        bitvecs_gpu = cp.zeros((n_cols, n_u64s), dtype=cp.uint64)

        # Handle edge case: no transactions
        if n_rows == 0:
            return bitvecs_gpu

        # Get compiled kernel
        kernel = get_csr_to_bitvec_kernel()

        # Kernel launch config
        # One thread per row, 256 threads per block (standard choice)
        block_size = 256
        grid_size = (n_rows + block_size - 1) // block_size

        # CUDA max grid X dimension is 2^31-1
        max_grid = (1 << 31) - 1
        grid_size = min(grid_size, max_grid)

        # Launch kernel
        # Parameters: indptr, indices, bitvecs, n_rows, n_cols, n_u64s
        kernel(
            (grid_size,), (block_size,),
            (
                indptr_gpu,
                indices_gpu,
                bitvecs_gpu,
                np.int64(n_rows),
                np.int64(n_cols),
                np.int64(n_u64s),
            )
        )

        # Synchronize to ensure kernel completes before returning
        cp.cuda.Stream.null.synchronize()

        return bitvecs_gpu


def build_bitvecs_from_gpu_arrays(
    indptr_gpu,   # CuPy int64 array [n_rows + 1]
    indices_gpu,  # CuPy int64 array [nnz]
    n_rows: int,
    n_cols: int,
) -> "cp.ndarray":
    """Build bitvecs from CSR arrays already resident on GPU.

    Same CUDA kernel as build_bitvecs_gpu() but skips the CPU→GPU transfer.
    Used by the GPU-resident null model where indptr/indices persist in VRAM
    across permutations.

    Args:
        indptr_gpu: CuPy int64 array of row pointers [n_rows + 1], already on GPU.
        indices_gpu: CuPy int64 array of column indices [nnz], already on GPU.
        n_rows: Number of rows (transactions).
        n_cols: Number of columns (items).

    Returns:
        CuPy array of shape [n_cols, n_u64s] on GPU, dtype uint64.
    """
    import cupy as cp

    n_u64s = (n_rows + 63) // 64

    max_int64 = (1 << 63) - 1
    if n_cols > 0 and n_u64s > max_int64 // n_cols:
        raise OverflowError(
            f"Bitvec dimensions overflow int64: n_cols={n_cols} * n_u64s={n_u64s} "
            f"exceeds {max_int64}. Reduce n_rows or n_cols."
        )

    bitvecs_gpu = cp.zeros((n_cols, n_u64s), dtype=cp.uint64)

    if n_rows == 0:
        return bitvecs_gpu

    kernel = get_csr_to_bitvec_kernel()

    block_size = 256
    grid_size = min((n_rows + block_size - 1) // block_size, (1 << 31) - 1)

    kernel(
        (grid_size,), (block_size,),
        (
            indptr_gpu,
            indices_gpu,
            bitvecs_gpu,
            np.int64(n_rows),
            np.int64(n_cols),
            np.int64(n_u64s),
        )
    )

    cp.cuda.Stream.null.synchronize()
    return bitvecs_gpu


def build_bitvecs_gpu_from_scipy(
    csr: "scipy.sparse.csr_matrix",
    device_id: int = 0,
    buffer_pool: Optional[PinnedBufferPool] = None,
) -> "cp.ndarray":
    """
    Build GPU bitvecs directly from a scipy CSR matrix.

    Convenience wrapper that extracts CSR components and calls build_bitvecs_gpu.
    This replaces the CPU bitvec build + transfer pattern.

    Args:
        csr: scipy.sparse.csr_matrix (transactions x items)
        device_id: GPU device to use
        buffer_pool: Optional PinnedBufferPool for 3x faster transfers.
                    For best performance, create one pool and reuse for all chunks.

    Returns:
        CuPy array [n_cols, n_u64s] on GPU

    Example:
        >>> from scipy.sparse import csr_matrix
        >>> from et_miner.cuda_csr_bitvec import PinnedBufferPool
        >>> pool = PinnedBufferPool()  # Create once, reuse
        >>> csr = csr_matrix([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
        >>> bitvecs = build_bitvecs_gpu_from_scipy(csr, buffer_pool=pool)
    """
    n_rows, n_cols = csr.shape

    # Extract CSR components (these are views, not copies)
    indptr = csr.indptr.astype(np.int64)
    indices = csr.indices.astype(np.int64)

    return build_bitvecs_gpu(indptr, indices, n_rows, n_cols, device_id, buffer_pool)


def build_bitvecs_row_split(
    csr: "scipy.sparse.csr_matrix",
    n_gpus: int,
) -> list[tuple["cp.ndarray", int, int]]:
    """Build row-split bitvecs across multiple GPUs from scipy CSR matrix.

    Splits transaction rows evenly across GPUs. Each GPU builds its own
    bitvec from its portion of the CSR data. Total VRAM usage is split
    n_gpus ways, enabling datasets that don't fit on a single GPU.

    Args:
        csr: scipy CSR matrix of shape (n_transactions, n_items).
        n_gpus: Number of GPUs to distribute across.

    Returns:
        List of (bitvec_gpu, device_id, n_local_rows) tuples.
        Each bitvec_gpu is on its respective GPU.
    """
    import cupy as cp

    n_rows, n_cols = csr.shape
    available_gpus = cp.cuda.runtime.getDeviceCount()
    n_gpus = min(n_gpus, available_gpus)
    rows_per_gpu = (n_rows + n_gpus - 1) // n_gpus

    indptr = csr.indptr.astype(np.int64)
    indices = csr.indices.astype(np.int64)

    bitvecs_list = []
    for gpu_id in range(n_gpus):
        start = gpu_id * rows_per_gpu
        end = min(start + rows_per_gpu, n_rows)
        if start >= end:
            break

        # Extract local CSR slice (offset indptr to start from 0)
        nnz_start = int(indptr[start])
        nnz_end = int(indptr[end])
        local_indptr = indptr[start:end + 1] - nnz_start
        local_indices = indices[nnz_start:nnz_end]
        local_n_rows = end - start

        logger.debug(f"  [GPU {gpu_id}] Building bitvec: {local_n_rows:,} rows, {len(local_indices):,} nnz, ~{n_cols * ((local_n_rows + 63) // 64) * 8 / 1e9:.1f} GB")

        bitvec = build_bitvecs_gpu(
            local_indptr, local_indices, local_n_rows, n_cols, device_id=gpu_id,
        )
        bitvecs_list.append((bitvec, gpu_id, local_n_rows))

    return bitvecs_list


def build_bitvecs_row_split_from_arrays(
    indptr: np.ndarray,
    indices: np.ndarray,
    n_rows: int,
    n_cols: int,
    n_gpus: int,
) -> list[tuple["cp.ndarray", int, int]]:
    """Build row-split bitvecs across multiple GPUs from raw CSR arrays.

    Same as build_bitvecs_row_split but accepts numpy arrays directly instead
    of a scipy CSR matrix. Used by the GPU-resident null model where CSR arrays
    are built from shuffled items without constructing a scipy matrix.

    Args:
        indptr: numpy int64 array of row pointers [n_rows + 1].
        indices: numpy int64 array of column indices [nnz].
        n_rows: Number of rows (transactions).
        n_cols: Number of columns (items).
        n_gpus: Number of GPUs to distribute across.

    Returns:
        List of (bitvec_gpu, device_id, n_local_rows) tuples.
    """
    import cupy as cp

    available_gpus = cp.cuda.runtime.getDeviceCount()
    n_gpus = min(n_gpus, available_gpus)
    rows_per_gpu = (n_rows + n_gpus - 1) // n_gpus

    if indptr.dtype != np.int64:
        indptr = indptr.astype(np.int64)
    if indices.dtype != np.int64:
        indices = indices.astype(np.int64)

    bitvecs_list = []
    for gpu_id in range(n_gpus):
        start = gpu_id * rows_per_gpu
        end = min(start + rows_per_gpu, n_rows)
        if start >= end:
            break

        nnz_start = int(indptr[start])
        nnz_end = int(indptr[end])
        local_indptr = indptr[start:end + 1] - nnz_start
        local_indices = indices[nnz_start:nnz_end]
        local_n_rows = end - start

        logger.debug(f"  [GPU {gpu_id}] Building bitvec: {local_n_rows:,} rows, {len(local_indices):,} nnz, ~{n_cols * ((local_n_rows + 63) // 64) * 8 / 1e9:.1f} GB")

        bitvec = build_bitvecs_gpu(
            local_indptr, local_indices, local_n_rows, n_cols, device_id=gpu_id,
        )
        bitvecs_list.append((bitvec, gpu_id, local_n_rows))

    return bitvecs_list
