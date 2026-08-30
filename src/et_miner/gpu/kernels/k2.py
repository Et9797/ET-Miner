"""Fused k=2 pair counting kernels (pair generation + count + filter on GPU)."""

from __future__ import annotations

import numpy as np

from .loader import _get_device_lock, _grid_dims, _warn_result_truncation, get_cuda_kernel


def count_pairs_fused_k2(bitvecs_gpu, freq_item_cols, n_u64s, min_count):
    """Fused k=2 kernel: generate pairs + AND + popcount + filter in ONE launch.

    Replaces the entire Python pipeline of:
      _generate_candidates() -> np.array per candidate -> _count_batch() -> filter

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s) with packed bitvectors.
        freq_item_cols: List/array of frequent item column indices (sorted).
        n_u64s: Number of uint64 words per bitvector.
        min_count: Minimum support count threshold.

    Returns:
        Tuple of (pairs, counts) where:
          pairs: list of (col_i, col_j) tuples (column indices, not item IDs)
          counts: numpy array of int64 support counts
    """
    import cupy as cp

    n_freq = len(freq_item_cols)
    if n_freq < 2:
        return [], np.array([], dtype=np.int64)

    n_pairs = n_freq * (n_freq - 1) // 2

    # Upload frequent item column indices to GPU (tiny: just n_freq ints)
    freq_items_gpu = cp.array(freq_item_cols, dtype=cp.int32)

    # Pre-allocate output buffers. Upper bound: min(n_pairs, reasonable cap).
    # If more results than max_results, the kernel safely stops writing.
    max_results = min(n_pairs, 10_000_000)
    result_i = cp.empty(max_results, dtype=cp.int64)
    result_j = cp.empty(max_results, dtype=cp.int64)
    result_count = cp.empty(max_results, dtype=cp.int64)
    n_results = cp.zeros(1, dtype=cp.int64)

    kernel = get_cuda_kernel("count_pairs_fused_k2")

    # Grid: 1 block per pair. Block: 256 threads for u64 word parallelism.
    # Uses 2D grid for >2.15B pairs (CUDA grid X max = 2^31-1).
    block_size = 256
    grid = _grid_dims(n_pairs)
    kernel(
        grid,
        (block_size,),
        (
            bitvecs_gpu,
            freq_items_gpu,
            np.int64(n_u64s),
            np.int32(n_freq),
            np.int64(min_count),
            result_i,
            result_j,
            result_count,
            n_results,
            np.int64(max_results),
            np.int64(0),
        ),  # pair_offset=0 for single GPU
    )
    cp.cuda.Stream.null.synchronize()

    n = int(n_results.get()[0])
    if n == 0:
        return [], np.array([], dtype=np.int64)

    # Clamp to max_results — warn on truncation (P0: silent data loss)
    n = _warn_result_truncation(n, max_results, "filtered_kernel")

    # Convert freq_items indices back to column indices
    ri = result_i[:n].get()
    rj = result_j[:n].get()
    rc = result_count[:n].get()

    pairs = [(int(freq_item_cols[ri[idx]]), int(freq_item_cols[rj[idx]])) for idx in range(n)]
    counts = rc

    return pairs, counts


def count_pairs_fused_k2_multi_gpu(bitvecs_gpu, freq_item_cols, n_u64s, min_count, n_gpus):
    """Multi-GPU fused k=2: split pairs across GPUs for linear scaling.

    Each GPU gets the FULL bitvec matrix but processes only its subset of pairs.
    This is pair-index splitting (not row splitting), which gives near-linear
    speedup because there's zero inter-GPU communication during computation.

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s) on GPU 0.
        freq_item_cols: List/array of frequent item column indices (sorted).
        n_u64s: Number of uint64 words per bitvector.
        min_count: Minimum support count threshold.
        n_gpus: Number of GPUs to use.

    Returns:
        Tuple of (pairs, counts) - same format as count_pairs_fused_k2().
    """
    import cupy as cp
    from concurrent.futures import ThreadPoolExecutor

    n_freq = len(freq_item_cols)
    if n_freq < 2:
        return [], np.array([], dtype=np.int64)

    n_pairs = n_freq * (n_freq - 1) // 2

    # Limit GPUs to available and sensible count
    available_gpus = cp.cuda.runtime.getDeviceCount()
    n_gpus = min(n_gpus, available_gpus, n_pairs)  # no point in more GPUs than pairs

    if n_gpus <= 1:
        return count_pairs_fused_k2(bitvecs_gpu, freq_item_cols, n_u64s, min_count)

    # Calculate pair ranges per GPU
    pairs_per_gpu = (n_pairs + n_gpus - 1) // n_gpus
    freq_items_np = np.array(freq_item_cols, dtype=np.int32)

    # Get kernel reference (compile once, reuse across GPUs)
    kernel = get_cuda_kernel("count_pairs_fused_k2")

    # Get bitvecs as numpy for replication to other GPUs
    bitvecs_np = bitvecs_gpu.get()

    def _run_on_gpu(device_id):
        """Run pair subset on a single GPU (thread-safe for free-threading)."""
        pair_start = device_id * pairs_per_gpu
        pair_end = min(pair_start + pairs_per_gpu, n_pairs)
        n_gpu_pairs = pair_end - pair_start

        if n_gpu_pairs <= 0:
            return [], np.array([], dtype=np.int64)

        # Phase 1: Device setup + data allocation (per-device lock).
        with _get_device_lock(device_id):
            with cp.cuda.Device(device_id):
                stream = cp.cuda.Stream(non_blocking=True)
                with stream:
                    # Replicate data to this GPU
                    if device_id == 0:
                        bv_gpu = bitvecs_gpu  # already on GPU 0
                    else:
                        bv_gpu = cp.array(bitvecs_np, dtype=cp.uint64)

                    freq_gpu = cp.array(freq_items_np, dtype=cp.int32)

                    # Allocate output buffers on this GPU
                    max_results = min(n_gpu_pairs, 10_000_000)
                    result_i = cp.empty(max_results, dtype=cp.int64)
                    result_j = cp.empty(max_results, dtype=cp.int64)
                    result_count = cp.empty(max_results, dtype=cp.int64)
                    n_results = cp.zeros(1, dtype=cp.int64)

                    block_size = 256
                    grid = _grid_dims(n_gpu_pairs)
                    kernel(
                        grid,
                        (block_size,),
                        (
                            bv_gpu,
                            freq_gpu,
                            np.int64(n_u64s),
                            np.int32(n_freq),
                            np.int64(min_count),
                            result_i,
                            result_j,
                            result_count,
                            n_results,
                            np.int64(max_results),
                            np.int64(pair_start),
                        ),
                    )

        # Phase 2: Wait for kernel completion (unlocked — GPUs run in parallel).
        stream.synchronize()

        # Phase 3: Retrieve results (per-device lock).
        with _get_device_lock(device_id):
            with cp.cuda.Device(device_id):
                n = int(n_results.get()[0])
                if n == 0:
                    return [], np.array([], dtype=np.int64)

                n = _warn_result_truncation(n, max_results, "filtered_kernel")
                ri = result_i[:n].get()
                rj = result_j[:n].get()
                rc = result_count[:n].get()

        pairs = [(int(freq_item_cols[ri[idx]]), int(freq_item_cols[rj[idx]])) for idx in range(n)]
        return pairs, rc

    # Launch on all GPUs in parallel
    all_pairs = []
    all_counts = []

    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = {executor.submit(_run_on_gpu, i): i for i in range(n_gpus)}
        results = {}
        for future in futures:
            results[futures[future]] = future.result()

    # Merge results in GPU order (deterministic)
    for device_id in range(n_gpus):
        if device_id in results:
            pairs, counts = results[device_id]
            if len(pairs) > 0:
                all_pairs.extend(pairs)
                all_counts.append(counts)

    if not all_counts:
        return [], np.array([], dtype=np.int64)

    return all_pairs, np.concatenate(all_counts)




def count_pairs_k2_allcounts(bitvecs_gpu, freq_item_cols, n_u64s, chunk_start=0, chunk_size=None):
    """Dense K=2 counting: support counts for pairs in a candidate range.

    No threshold filtering — outputs counts for every pair in
    [chunk_start, chunk_start + chunk_size), chunk-relative (mirrors
    count_k3plus_allcounts). For row-split multi-GPU: sum arrays across
    GPUs = exact global counts.

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


