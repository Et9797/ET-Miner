"""Batched itemset counting: the general AND+popcount kernel family."""

from __future__ import annotations

import numpy as np

from .loader import get_cuda_kernel


def count_itemsets_cuda(
    bitvecs_gpu,  # CuPy array [n_cols, n_u64s]
    itemsets: list[np.ndarray],
    use_batch: bool = True,
) -> np.ndarray:
    """
    Count itemset support using custom CUDA kernels.

    10x faster than CuPy bitwise_and.reduce + unpackbits!

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s)
        itemsets: List of numpy arrays with item indices
        use_batch: Use batched kernel (faster for many itemsets)

    Returns:
        Numpy array of counts
    """

    n_cols, n_u64s = bitvecs_gpu.shape
    n_itemsets = len(itemsets)

    if use_batch and n_itemsets > 10:
        return _count_batch(bitvecs_gpu, itemsets, n_cols, n_u64s)
    else:
        return _count_single(bitvecs_gpu, itemsets, n_cols, n_u64s)


def _count_single(bitvecs_gpu, itemsets, n_cols, n_u64s):
    """Count itemsets one at a time."""
    import cupy as cp

    kernel = get_cuda_kernel("count_itemset_fused")
    counts = np.zeros(len(itemsets), dtype=np.int64)

    # Kernel config
    # Note: Kernel uses grid-stride loop, so all data is processed regardless of grid size.
    # Higher grid size = more parallelism. CUDA x-dimension supports up to 2^31-1 blocks.
    block_size = 256
    blocks_needed = (n_u64s + block_size - 1) // block_size
    # Cap at 2^20 (~1M blocks) for safety on older GPUs while allowing massive parallelism
    grid_size = min(blocks_needed, 1 << 20)

    # Cast to int64 to match kernel's long long parameter
    n_u64s_i64 = np.int64(n_u64s)

    for i, itemset in enumerate(itemsets):
        if len(itemset) == 0:
            continue

        items_gpu = cp.array(itemset, dtype=cp.int32)
        count_gpu = cp.zeros(1, dtype=cp.uint64)

        kernel(
            (grid_size,),
            (block_size,),
            (bitvecs_gpu, items_gpu, np.int32(len(itemset)), n_u64s_i64, np.int32(n_cols), count_gpu),
        )

        # CRITICAL: Synchronize before reading result to ensure kernel completion
        cp.cuda.Stream.null.synchronize()
        counts[i] = int(count_gpu.get()[0])

    return counts


def _count_batch(bitvecs_gpu, itemsets, n_cols, n_u64s):
    """Count all itemsets in one kernel launch."""
    import cupy as cp

    kernel = get_cuda_kernel("count_itemsets_batch")
    n_itemsets = len(itemsets)

    # Flatten itemsets
    all_items = []
    offsets = [0]
    for itemset in itemsets:
        all_items.extend(itemset.tolist())
        offsets.append(len(all_items))

    all_items_gpu = cp.array(all_items, dtype=cp.int32)
    offsets_gpu = cp.array(offsets, dtype=cp.int64)  # FIXED: int32 -> int64 for >2B elements

    return _launch_batch_kernel(
        bitvecs_gpu,
        all_items_gpu,
        offsets_gpu,
        n_itemsets,
        n_u64s,
    )


def _count_batch_prebuilt(bitvecs_gpu, items_flat_np, offsets_np, n_itemsets, n_u64s):
    """Count itemsets using pre-built flat numpy arrays (avoids Python loop).

    For multi-GPU recount: build arrays ONCE, each GPU only does DMA transfer.
    Eliminates O(N) Python list-building that was serialized by the GIL.
    """
    import cupy as cp

    items_gpu = cp.array(items_flat_np, dtype=cp.int32)
    offsets_gpu = cp.array(offsets_np, dtype=cp.int64)  # FIXED: int32 -> int64 for >2B elements

    return _launch_batch_kernel(
        bitvecs_gpu,
        items_gpu,
        offsets_gpu,
        n_itemsets,
        n_u64s,
    )


def _launch_batch_kernel(bitvecs_gpu, all_items_gpu, offsets_gpu, n_itemsets, n_u64s):
    """Shared kernel launch logic for batch counting."""
    import cupy as cp

    kernel = get_cuda_kernel("count_itemsets_batch")
    counts_gpu = cp.zeros(n_itemsets, dtype=cp.uint64)

    # 2D grid: x for u64s, y for itemsets
    # Kernel uses grid-stride loop for x, so all u64 words are processed.
    # CUDA y-dimension max is 65535 — chunk launches for > 65535 itemsets.
    _MAX_GRID_Y = 65535
    block_size = 256
    blocks_needed_x = (n_u64s + block_size - 1) // block_size
    grid_x = min(blocks_needed_x, 1 << 16)
    n_u64s_i64 = np.int64(n_u64s)

    for chunk_start in range(0, n_itemsets, _MAX_GRID_Y):
        chunk_end = min(chunk_start + _MAX_GRID_Y, n_itemsets)
        chunk_size = chunk_end - chunk_start

        # Slice into the pre-allocated GPU arrays using offsets
        chunk_offsets = offsets_gpu[chunk_start : chunk_end + 1]
        chunk_counts = counts_gpu[chunk_start:chunk_end]

        kernel(
            (grid_x, chunk_size),
            (block_size,),
            (
                bitvecs_gpu,
                all_items_gpu,
                chunk_offsets,
                n_u64s_i64,
                np.int32(bitvecs_gpu.shape[0]),
                np.int64(chunk_size),
                chunk_counts,
            ),
        )

    cp.cuda.Stream.null.synchronize()
    return counts_gpu.get().astype(np.int64)


