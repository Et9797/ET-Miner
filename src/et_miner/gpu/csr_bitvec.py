"""Column bitvectors from a CSR on the GPU, whole or split by rows across devices.

The ``csr_to_bitvec`` kernel (``gpu/kernels/_src/csr_to_bitvec.cu``) sets bit
``row`` of column ``col``'s bitvector for every CSR entry; the output is a
uint64 array of shape ``(n_cols, ceil(n_rows / 64))``, zero-initialised, so a
repeated entry sets its bit once.

Usage:
    from et_miner.gpu.csr_bitvec import build_bitvecs_gpu, build_bitvecs_row_split

    bitvecs = build_bitvecs_gpu(indptr, indices, n_rows, n_cols, device_id=0)
    shards = build_bitvecs_row_split(csr, n_gpus=2)  # [(bitvecs, device_id, n_rows)]
"""

import numpy as np
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import cupy as cp
    import scipy.sparse

__all__ = [
    'build_bitvecs_gpu',
    'build_bitvecs_from_gpu_arrays',
    'build_bitvecs_row_split_from_arrays',
    'get_csr_to_bitvec_kernel',
    'clear_csr_kernel_cache',
]


# =============================================================================
# Kernel access (source lives in gpu/kernels/_src/csr_to_bitvec.cu)
# =============================================================================


def clear_csr_kernel_cache():
    """Clear compiled kernel cache. Call after modifying kernel source."""
    from et_miner.gpu.kernels.loader import clear_kernel_cache

    clear_kernel_cache()


def get_csr_to_bitvec_kernel():
    """Get compiled CUDA kernel for CSR→bitvec conversion, caching for reuse.

    Returns:
        Compiled CuPy RawKernel ready for execution.

    Raises:
        ImportError: If CuPy is not available.
    """
    from et_miner.gpu.kernels.loader import get_cuda_kernel

    return get_cuda_kernel("csr_to_bitvec")


# =============================================================================
# Python Wrapper
# =============================================================================

def build_bitvecs_gpu(
    csr_indptr: np.ndarray,
    csr_indices: np.ndarray,
    n_rows: int,
    n_cols: int,
    device_id: int = 0,
) -> "cp.ndarray":
    """Build the column bitvectors of a CSR on one GPU: the CSR is copied over, the bitvectors built there.

    Args:
        csr_indptr: numpy array of row pointers [n_rows + 1], dtype int64/int32
        csr_indices: numpy array of column indices [nnz], dtype int64/int32
        n_rows: number of rows (transactions)
        n_cols: number of columns (items)
        device_id: GPU device to use (default 0)

    Returns:
        CuPy array of shape [n_cols, n_u64s] on GPU, where n_u64s = ceil(n_rows/64)
        Each element is a uint64 containing 64 transaction bits for that column.

    Raises:
        ImportError: If CuPy is not available.
        OverflowError: If n_cols * n_u64s does not fit int64.

    Example:
        >>> import numpy as np
        >>> indptr = np.array([0, 2, 3, 6], dtype=np.int64)
        >>> indices = np.array([0, 2, 1, 0, 1, 3], dtype=np.int64)
        >>> bitvecs = build_bitvecs_gpu(indptr, indices, n_rows=3, n_cols=4)
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

    Same CUDA kernel as build_bitvecs_gpu() but skips the CPU→GPU transfer,
    for callers that keep the CSR in VRAM (the AlphaFold null-model
    experiment permutes it there between runs).

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
) -> "cp.ndarray":
    """``build_bitvecs_gpu`` on a scipy CSR matrix (transactions x items)."""
    n_rows, n_cols = csr.shape

    # Extract CSR components (these are views, not copies)
    indptr = csr.indptr.astype(np.int64)
    indices = csr.indices.astype(np.int64)

    return build_bitvecs_gpu(indptr, indices, n_rows, n_cols, device_id)


def _row_split_cuts(n_rows: int, n_gpus: int, balance: str = "rows") -> list[tuple[int, int]]:
    """Contiguous [start, end) row ranges per GPU. Pure Python — unit-tested.

    Equal row counts (``balance="rows"``, the only mode): dense-kernel cost
    and bitvec bytes scale with ROWS (every kernel strip-mines all
    ceil(rows/64) words). The nnz-balanced cut (``balance="nnz"``, equal
    cumulative nnz per shard) was removed after it tied the rows split on
    both regimes built for it (`bench/results/2026-09-28-consolidation-2gpu/`);
    asking for it raises. Row ranges stay contiguous — round-robin would
    break the row-contiguity that CSR slicing depends on.
    """
    _check_balance(balance)
    if n_rows <= 0 or n_gpus <= 0:
        return []
    n_gpus = min(n_gpus, n_rows)
    rows_per_gpu = (n_rows + n_gpus - 1) // n_gpus
    cuts = [min(i * rows_per_gpu, n_rows) for i in range(n_gpus + 1)]
    return [(s, e) for s, e in zip(cuts[:-1], cuts[1:]) if e > s]


def _check_balance(balance) -> None:
    if balance == "rows":
        return
    if balance == "nnz":
        raise ValueError(
            "balance='nnz' was removed: the multi-GPU row split is by equal row counts "
            "(balance='rows', the default; nnz-balanced cuts won no regime). Pass 'rows' or None."
        )
    raise ValueError(f"balance must be 'rows' or None, got {balance!r} (the split is by equal row counts)")


def build_bitvecs_row_split_from_arrays(
    indptr: np.ndarray,
    indices: np.ndarray,
    n_rows: int,
    n_cols: int,
    n_gpus: int,
    balance: str | None = None,
) -> list[tuple["cp.ndarray", int, int]]:
    """Build row-split bitvecs across multiple GPUs from raw CSR arrays.

    Splits transaction rows across GPUs (contiguous ranges) and builds each
    shard's bitvec on its own device in parallel (one thread per GPU; the
    kernel loader's cache lock makes the first compile on each device safe).
    Total VRAM usage is split n_gpus ways, enabling datasets that don't fit
    on a single GPU.

    Args:
        indptr: numpy int64 array of row pointers [n_rows + 1].
        indices: numpy int64 array of column indices [nnz].
        n_rows: Number of rows (transactions).
        n_cols: Number of columns (items).
        n_gpus: Number of GPUs to distribute across.
        balance: "rows" (equal row counts) or None (the same). "nnz" was
            removed and raises (see _row_split_cuts).

    Returns:
        List of (bitvec_gpu, device_id, n_local_rows) tuples.
    """
    from concurrent.futures import ThreadPoolExecutor

    from et_miner import _env

    balance = balance or "rows"
    _check_balance(balance)
    _env.reject_removed_knobs()

    import cupy as cp

    available_gpus = cp.cuda.runtime.getDeviceCount()
    n_gpus = min(n_gpus, available_gpus)

    if indptr.dtype != np.int64:
        indptr = indptr.astype(np.int64)
    if indices.dtype != np.int64:
        indices = indices.astype(np.int64)

    ranges = _row_split_cuts(n_rows, n_gpus, balance=balance)

    def _build_shard(gpu_id, start, end):
        nnz_start = int(indptr[start])
        nnz_end = int(indptr[end])
        local_indptr = indptr[start : end + 1] - nnz_start
        local_indices = indices[nnz_start:nnz_end]
        local_n_rows = end - start
        logger.debug(
            f"  [GPU {gpu_id}] Building bitvec: {local_n_rows:,} rows, {len(local_indices):,} nnz, "
            f"~{n_cols * ((local_n_rows + 63) // 64) * 8 / 1e9:.1f} GB ({balance}-balanced)"
        )
        bitvec = build_bitvecs_gpu(local_indptr, local_indices, local_n_rows, n_cols, device_id=gpu_id)
        return (bitvec, gpu_id, local_n_rows)

    with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
        futures = [pool.submit(_build_shard, gpu_id, s, e) for gpu_id, (s, e) in enumerate(ranges)]
        return [f.result() for f in futures]


def build_bitvecs_row_split(
    csr: "scipy.sparse.csr_matrix",
    n_gpus: int,
    balance: str | None = None,
) -> list[tuple["cp.ndarray", int, int]]:
    """Build row-split bitvecs across multiple GPUs from a scipy CSR matrix.

    Thin wrapper over build_bitvecs_row_split_from_arrays — see there for
    the split semantics and parallel shard builds.

    Args:
        csr: scipy CSR matrix of shape (n_transactions, n_items).
        n_gpus: Number of GPUs to distribute across.
        balance: "rows" | None; "nnz" was removed and raises.

    Returns:
        List of (bitvec_gpu, device_id, n_local_rows) tuples.
    """
    n_rows, n_cols = csr.shape
    return build_bitvecs_row_split_from_arrays(
        csr.indptr.astype(np.int64),
        csr.indices.astype(np.int64),
        n_rows,
        n_cols,
        n_gpus,
        balance=balance,
    )
