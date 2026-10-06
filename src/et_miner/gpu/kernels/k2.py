"""K=2 pair counting: the per-candidate kernel over bitvecs, and the row-wise kernel over
each shard's rows (the tiled bitvec kernel is in shared_tiled)."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from .loader import _assert_bitvecs, _assert_dtype, _assert_home, _grid_dims, get_cuda_kernel


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




#: Pair range up to which ``count_pairs_k2_rows`` privatizes the counters in
#: shared memory (``K2_ROWS_SHARED_PAIRS`` in ``_src/pairs_k2_rows.cu``: 44 KB
#: of int32, under the 48 KB static limit). A larger range uses global atomics.
K2_ROWS_SHARED_PAIRS = 11264


class K2Rows(NamedTuple):
    """One shard's rows as frequent-column positions, on its device."""

    row_ptr: object  # CuPy int64 (n_rows + 1,)
    pos: object  # CuPy int32 (nnz,), ascending within each row
    n_rows: int


def upload_k2_rows(row_ptr, pos, device_id: int) -> K2Rows:
    """Copy one shard's host CSR of frequent-column positions to ``device_id``."""
    import cupy as cp

    with cp.cuda.Device(device_id):
        return K2Rows(
            cp.asarray(np.ascontiguousarray(row_ptr, dtype=np.int64)),
            cp.asarray(np.ascontiguousarray(pos, dtype=np.int32)),
            len(row_ptr) - 1,
        )


def pair_j(pair_idx: int) -> int:
    """The larger position b of pair index ``pair_idx`` (b*(b-1)/2 <= pair_idx < b*(b+1)/2), exactly."""
    return (1 + math.isqrt(1 + 8 * int(pair_idx))) // 2


def count_pairs_k2_rows(rows: K2Rows, chunk_start: int, chunk_size: int):
    """Row-wise K=2 counting of the pair range [chunk_start, chunk_start + chunk_size).

    Every row of the shard adds 1 to each pair of its positions inside the
    range; the counts are chunk-relative, in the pair layout of
    ``count_pairs_k2_per_candidate``, so the per-GPU arrays sum to the global
    counts the same way. A range of at most ``K2_ROWS_SHARED_PAIRS`` pairs is
    counted in shared memory per block, a larger one with global atomics.

    Returns:
        CuPy int32 array of shape (chunk_size,) on the shard's device.
    """
    import cupy as cp

    _assert_home("count_pairs_k2_rows", row_ptr=rows.row_ptr, pos=rows.pos)
    _assert_dtype("count_pairs_k2_rows", row_ptr=(rows.row_ptr, "int64"), pos=(rows.pos, "int32"))
    with cp.cuda.Device(rows.row_ptr.device.id):
        out = cp.zeros(chunk_size, dtype=cp.int32)
        if chunk_size <= 0 or rows.n_rows == 0:
            return out
        shared = chunk_size <= K2_ROWS_SHARED_PAIRS
        kernel = get_cuda_kernel("count_pairs_k2_rows_shared" if shared else "count_pairs_k2_rows")
        n_sm = cp.cuda.Device().attributes["MultiProcessorCount"]
        warps_per_block = 256 // 32
        blocks = max(1, min(-(-rows.n_rows // warps_per_block), n_sm * (2 if shared else 32)))
        kernel(
            (blocks,),
            (256,),
            (
                rows.row_ptr,
                rows.pos,
                np.int64(rows.n_rows),
                np.int64(chunk_start),
                np.int64(chunk_size),
                np.int64(pair_j(chunk_start)),
                np.int64(pair_j(chunk_start + chunk_size - 1)),
                out,
            ),
        )
        cp.cuda.Stream.null.synchronize()
    return out
