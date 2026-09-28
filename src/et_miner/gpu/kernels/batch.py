"""Batched itemset counting: the general AND+popcount kernel family."""

from __future__ import annotations

import numpy as np

from .loader import _assert_bitvecs, get_cuda_kernel


def count_itemsets_cuda(bitvecs_gpu, itemsets: list[np.ndarray]) -> np.ndarray:
    """Support count of every itemset, one batched AND + popcount launch.

    Args:
        bitvecs_gpu: CuPy uint64 array of shape (n_cols, n_u64s).
        itemsets: Non-empty int arrays of column indices, of any lengths.

    Returns:
        int64 NumPy array of counts, one per itemset.
    """
    import cupy as cp

    _assert_bitvecs("count_itemsets_cuda", bitvecs_gpu)
    n_cols, n_u64s = bitvecs_gpu.shape
    if any(len(s) == 0 for s in itemsets):
        raise ValueError("count_itemsets_cuda: every itemset needs at least one item")
    if not itemsets or n_u64s == 0:
        return np.zeros(len(itemsets), dtype=np.int64)
    with cp.cuda.Device(bitvecs_gpu.device.id):
        return _count_batch(bitvecs_gpu, itemsets, n_cols, n_u64s)


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


