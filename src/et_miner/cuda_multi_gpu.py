"""
Multi-GPU Support for ET-Miner with Row-Split Strategy.

Distributes transaction data across multiple GPUs using row splitting:
- Each GPU holds a subset of rows (transactions)
- Itemset counting runs in parallel on all GPUs
- Partial counts are summed across GPUs

This approach scales linearly with GPU count for large datasets.
"""

import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    import cupy as cp

# Lock for thread-safe CUDA device context switching
# Required when multiple threads access different GPUs concurrently
_cuda_context_lock = threading.Lock()

__all__ = [
    "warmup_cuda_multi_gpu",
    "generate_bitvecs_multi_gpu",
    "count_itemsets_multi_gpu",
    "generate_bitvecs_cpu_fallback",
    "count_itemsets_cpu_fallback",
]


def warmup_cuda_multi_gpu(n_gpus: int, n_cols: int = 100) -> None:
    """
    Warmup CUDA context on all GPUs (MUST call before timing).

    First CUDA call has ~500ms-2s overhead for context init, kernel JIT,
    and memory pool setup. This would skew benchmark results.

    Args:
        n_gpus: Number of GPUs to warmup
        n_cols: Number of columns for warmup test (default 100)
    """
    import cupy as cp
    from et_miner.cuda_csr_build import generate_bitvecs_gpu
    from et_miner.cuda_kernels import count_itemsets_cuda

    # Limit to available GPUs
    available_gpus = cp.cuda.runtime.getDeviceCount()
    n_gpus = min(n_gpus, available_gpus)

    def warmup_single_gpu(device_id: int) -> None:
        """Warmup a single GPU with thread-safe device context switching."""
        # Acquire lock for device context initialization to prevent race conditions
        # when multiple threads switch device contexts simultaneously
        with _cuda_context_lock:
            with cp.cuda.Device(device_id):
                # 1. Force context init
                cp.zeros(1)

                # 2. Warmup RNG
                cp.random.seed(42 + device_id)
                _ = cp.random.randint(0, 100, size=100, dtype=cp.int64)

                # 3. Warmup generation kernel
                _, _, bitvecs = generate_bitvecs_gpu(
                    n_rows=1000,
                    n_cols=n_cols,
                    avg_items_per_row=10,
                    seed=42 + device_id,
                    device_id=device_id,
                )

                # 4. Warmup counting kernel
                test_itemsets = [np.array([0, 1], dtype=np.int64)]
                _ = count_itemsets_cuda(bitvecs, test_itemsets)

                # 5. Sync stream and free memory pool
                cp.cuda.Stream.null.synchronize()
                cp.get_default_memory_pool().free_all_blocks()

    # Warmup all GPUs in parallel
    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = [executor.submit(warmup_single_gpu, i) for i in range(n_gpus)]
        for future in as_completed(futures):
            future.result()  # Raise any exceptions


def generate_bitvecs_multi_gpu(
    n_rows: int,
    n_cols: int,
    avg_items_per_row: int,
    n_gpus: int,
    seed: int = 42,
) -> List[Tuple["cp.ndarray", int, int]]:
    """
    Generate bitvecs distributed across GPUs (row-split).

    Rows are split evenly across GPUs, with the last GPU handling
    any remainder. Each GPU generates its data independently with
    a unique seed.

    Args:
        n_rows: Total number of rows (transactions)
        n_cols: Number of columns (items)
        avg_items_per_row: Average items per transaction
        n_gpus: Number of GPUs to use
        seed: Base random seed (each GPU uses seed + device_id)

    Returns:
        List of (bitvecs_gpu, device_id, n_rows_on_gpu) tuples.
        bitvecs_gpu is a CuPy array on the corresponding GPU.
    """
    import cupy as cp
    from et_miner.cuda_csr_build import generate_bitvecs_gpu

    # Limit to available GPUs
    available_gpus = cp.cuda.runtime.getDeviceCount()
    n_gpus = min(n_gpus, available_gpus)

    # Calculate rows per GPU (ceiling division)
    rows_per_gpu = (n_rows + n_gpus - 1) // n_gpus

    results = [None] * n_gpus

    def generate_on_gpu(device_id: int) -> Tuple["cp.ndarray", int, int]:
        """Generate bitvecs on a single GPU with thread-safe device context."""
        # Calculate rows for this GPU (last GPU may have fewer)
        start_row = device_id * rows_per_gpu
        end_row = min(start_row + rows_per_gpu, n_rows)
        gpu_rows = end_row - start_row

        if gpu_rows <= 0:
            return None

        # Acquire lock for thread-safe device context switching
        with _cuda_context_lock:
            with cp.cuda.Device(device_id):
                _, _, bitvecs = generate_bitvecs_gpu(
                    n_rows=gpu_rows,
                    n_cols=n_cols,
                    avg_items_per_row=avg_items_per_row,
                    seed=seed + device_id,
                    device_id=device_id,
                )

                cp.cuda.Stream.null.synchronize()

                return (bitvecs, device_id, gpu_rows)

    # Generate in parallel on all GPUs
    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = {executor.submit(generate_on_gpu, i): i for i in range(n_gpus)}

        for future in as_completed(futures):
            device_id = futures[future]
            result = future.result()
            if result is not None:
                results[device_id] = result

    # Filter out None results (for case where n_gpus > needed)
    return [r for r in results if r is not None]


def count_itemsets_multi_gpu(
    bitvecs_per_gpu: List[Tuple["cp.ndarray", int, int]],
    itemsets: List[np.ndarray],
) -> np.ndarray:
    """
    Count itemsets across GPUs with row-split, sum results.

    Each GPU counts the itemsets for its portion of the data,
    then partial counts are summed across all GPUs.

    Args:
        bitvecs_per_gpu: List of (bitvecs, device_id, n_rows) tuples
            from generate_bitvecs_multi_gpu()
        itemsets: List of numpy arrays with item indices

    Returns:
        Numpy array of total counts across all GPUs
    """
    import cupy as cp
    from et_miner.cuda_kernels import count_itemsets_cuda

    n_itemsets = len(itemsets)
    n_gpus = len(bitvecs_per_gpu)

    # Store partial counts from each GPU
    partial_counts = [None] * n_gpus

    def count_on_gpu(gpu_idx: int) -> np.ndarray:
        """Count itemsets on a single GPU with thread-safe device context."""
        bitvecs, device_id, n_rows = bitvecs_per_gpu[gpu_idx]

        # Acquire lock for thread-safe device context switching
        with _cuda_context_lock:
            with cp.cuda.Device(device_id):
                counts = count_itemsets_cuda(bitvecs, itemsets)
                cp.cuda.Stream.null.synchronize()
                return counts

    # Count in parallel on all GPUs
    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = {executor.submit(count_on_gpu, i): i for i in range(n_gpus)}

        for future in as_completed(futures):
            gpu_idx = futures[future]
            partial_counts[gpu_idx] = future.result()

    # Sum partial counts across all GPUs
    total_counts = np.zeros(n_itemsets, dtype=np.int64)
    for counts in partial_counts:
        if counts is not None:
            total_counts += counts

    return total_counts


# =============================================================================
# CPU Fallback Functions (for testing without GPU)
# =============================================================================

# Lookup table for popcount (8-bit)
_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount_u64(x: np.uint64) -> int:
    """Count set bits in a uint64 using lookup table."""
    count = 0
    for _ in range(8):
        count += _POPCOUNT_TABLE[int(x & 0xFF)]
        x >>= 8
    return count


def _popcount_array(arr: np.ndarray) -> int:
    """Total set bits in a uint64 array via byte-wise lookup."""
    byte_view = arr.view(np.uint8)
    return int(np.sum(_POPCOUNT_TABLE[byte_view]))


def generate_bitvecs_cpu_fallback(
    n_rows: int,
    n_cols: int,
    avg_items_per_row: int,
    seed: int = 42,
) -> np.ndarray:
    """
    CPU bitvec generation for testing (SLOW but doesn't need GPU).

    Generates random transaction data and packs into bitvec format.
    This is intentionally slow - use GPU functions for real benchmarks.

    Args:
        n_rows: Number of transactions
        n_cols: Number of items
        avg_items_per_row: Average items per transaction
        seed: Random seed

    Returns:
        np.ndarray of shape (n_cols, ceil(n_rows/64)) dtype uint64
    """
    rng = np.random.default_rng(seed)

    # Calculate number of uint64s needed to pack n_rows bits
    n_u64s = (n_rows + 63) // 64

    # Initialize bitvecs: (n_cols, n_u64s)
    bitvecs = np.zeros((n_cols, n_u64s), dtype=np.uint64)

    # Calculate probability of item being present
    # avg_items_per_row = n_cols * p => p = avg_items_per_row / n_cols
    item_prob = min(avg_items_per_row / n_cols, 1.0)

    # Generate transactions row by row (slow but simple)
    for row in range(n_rows):
        # Which u64 does this row belong to?
        u64_idx = row // 64
        bit_pos = row % 64
        bit_mask = np.uint64(1 << bit_pos)

        # Generate items for this transaction
        items_present = rng.random(n_cols) < item_prob

        # Set bits for items present
        for col in np.where(items_present)[0]:
            bitvecs[col, u64_idx] |= bit_mask

    return bitvecs


def count_itemsets_cpu_fallback(
    bitvecs: np.ndarray,
    itemsets: List[np.ndarray],
    n_rows: int,
) -> np.ndarray:
    """
    CPU itemset counting for testing.

    Performs AND of bitvec columns and popcount.
    This is intentionally not optimized - use GPU for real benchmarks.

    Args:
        bitvecs: np.ndarray shape (n_cols, n_u64s) dtype uint64
        itemsets: List of numpy arrays with item indices
        n_rows: Total number of rows (for masking partial last u64)

    Returns:
        np.ndarray of counts (int64)
    """
    n_itemsets = len(itemsets)
    counts = np.zeros(n_itemsets, dtype=np.int64)

    n_u64s = bitvecs.shape[1]

    # Create mask for the last u64 (may have fewer than 64 valid bits)
    last_valid_bits = n_rows % 64
    if last_valid_bits == 0:
        last_mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    else:
        last_mask = np.uint64((1 << last_valid_bits) - 1)

    for i, itemset in enumerate(itemsets):
        if len(itemset) == 0:
            # Empty itemset matches all rows
            counts[i] = n_rows
            continue

        # AND all columns in the itemset
        # Start with all bits set
        result = np.copy(bitvecs[itemset[0]])

        for item_idx in itemset[1:]:
            result = np.bitwise_and(result, bitvecs[item_idx])

        # Apply mask to last u64
        if n_u64s > 0:
            result[-1] &= last_mask

        # Count bits
        counts[i] = _popcount_array(result)

    return counts
