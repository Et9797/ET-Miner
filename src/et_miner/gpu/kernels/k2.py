"""K=2 dense pair counting on the per-candidate kernel (the tiled one is in shared_tiled)."""

from __future__ import annotations

import numpy as np

from .loader import _assert_bitvecs, _grid_dims, get_cuda_kernel


def count_pairs_k2_per_candidate(bitvecs_gpu, freq_item_cols, n_u64s, chunk_start=0, chunk_size=None):
    """Dense K=2 counting, one thread per pair: support counts for a pair range.

    No threshold filtering — outputs counts for every pair in
    [chunk_start, chunk_start + chunk_size), chunk-relative (mirrors
    count_k3plus_per_candidate). For row-split multi-GPU: sum arrays across
    GPUs = exact global counts. Unlike the tiled kernel it serves any pair
    range, so it counts the pair space when that must be chunked.

    Memory: chunk_size × 4 bytes (int32 — counts are bounded by
    n_transactions, which the row-split caller guards to < 2^31; the
    ≤8-GPU sum of per-shard partials is bounded by the same
    n_transactions, so the NCCL int32 SUM cannot overflow either).
    For 35K items unchunked: 609M pairs × 4 = 2.44 GB.

    Returns CuPy array (stays in VRAM). Caller sums on GPU, only transfers
    the final freq_indices to CPU, and widens to int64 host-side.

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s).
        freq_item_cols: List/array of frequent item column indices (sorted).
        n_u64s: Number of uint64 words per bitvector.
        chunk_start: First pair index to process (default: 0).
        chunk_size: Number of pairs to process (default: all remaining).

    Returns:
        CuPy int32 array of shape (chunk_size,) with counts — stays in VRAM.
    """
    import cupy as cp

    n_freq = len(freq_item_cols)
    n_pairs = n_freq * (n_freq - 1) // 2
    if chunk_size is None:
        chunk_size = n_pairs - chunk_start

    _assert_bitvecs("count_pairs_k2_per_candidate", bitvecs_gpu)
    with cp.cuda.Device(bitvecs_gpu.device.id):
        freq_items_gpu = cp.array(freq_item_cols, dtype=cp.int32)
        result_counts = cp.zeros(chunk_size, dtype=cp.int32)

        kernel = get_cuda_kernel("count_pairs_k2_dense")
        block_size = 256
        grid = _grid_dims(chunk_size)

        kernel(
            grid,
            (block_size,),
            (bitvecs_gpu, freq_items_gpu, np.int64(n_u64s), np.int32(n_freq), result_counts, np.int64(chunk_start)),
        )
        cp.cuda.Stream.null.synchronize()

    return result_counts  # stays in VRAM — no .get()


