"""GPU-resident kernels: frequent itemsets stay on-device between levels
(zero-transfer mining); k2 and k3+ variants plus prefix-group construction."""

from __future__ import annotations

import numpy as np

from .loader import _assert_k_supported, _get_device_lock, _grid_dims, _warn_result_truncation, get_cuda_kernel


def build_prefix_groups_gpu(prev_freq_gpu):
    """Build prefix group structures from sorted 2D CuPy array, entirely on GPU.

    Args:
        prev_freq_gpu: CuPy array of shape (n_freq, k_prev) with sorted frequent itemsets.

    Returns:
        Tuple of (group_starts, group_sizes, cumulative_pairs, total_candidates)
        where group_starts/sizes/cumulative_pairs are CuPy int32/int64 arrays in VRAM,
        and total_candidates is a Python int (single 8-byte PCIe transfer).
    """
    import cupy as cp

    n, k_prev = prev_freq_gpu.shape

    if n < 2:
        empty_i64 = cp.empty(0, dtype=cp.int64)
        empty_cp = cp.array([0], dtype=cp.int64)
        return empty_i64, empty_i64, empty_cp, 0

    # Detect boundaries where prefix (first k_prev-1 cols) changes
    if k_prev == 1:
        # K=2 case: every row is its own "group" (single item), no prefix grouping
        # This path shouldn't be called for K=2 but handle gracefully
        empty_i64 = cp.empty(0, dtype=cp.int64)
        empty_cp = cp.array([0], dtype=cp.int64)
        return empty_i64, empty_i64, empty_cp, 0

    if k_prev == 2:
        diff = prev_freq_gpu[1:, 0] != prev_freq_gpu[:-1, 0]
    else:
        diff = cp.any(prev_freq_gpu[1:, :-1] != prev_freq_gpu[:-1, :-1], axis=1)

    boundary_mask = cp.concatenate([cp.array([True]), diff])
    group_starts = cp.where(boundary_mask)[0].astype(cp.int64)
    group_ends = cp.concatenate([group_starts[1:], cp.array([n], dtype=cp.int64)])
    group_sizes = (group_ends - group_starts).astype(cp.int64)

    # Filter groups with >= 2 suffixes
    valid = group_sizes >= 2
    group_starts = group_starts[valid]
    group_sizes = group_sizes[valid]

    if len(group_starts) == 0:
        empty_i64 = cp.empty(0, dtype=cp.int64)
        empty_cp = cp.array([0], dtype=cp.int64)
        return empty_i64, empty_i64, empty_cp, 0

    # Compute cumulative pairs for candidate indexing
    pairs = (group_sizes.astype(cp.int64) * (group_sizes.astype(cp.int64) - 1)) // 2
    cumulative_pairs = cp.concatenate([cp.array([0], dtype=cp.int64), cp.cumsum(pairs)])
    total_candidates = int(cumulative_pairs[-1])  # 8 bytes PCIe

    return group_starts, group_sizes, cumulative_pairs, total_candidates


def count_k3plus_gpu_resident(bitvecs_gpu, prev_freq_gpu, n_u64s, min_count, max_results=10_000_000):
    """GPU-resident K>=3: count + filter + decode, all in VRAM.

    Takes a sorted 2D CuPy array of frequent (k-1)-itemsets, builds prefix groups
    on GPU, runs the counting kernel, decodes results on GPU, and returns CuPy arrays.

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s) with packed bitvectors.
        prev_freq_gpu: CuPy array of shape (n_freq, k_prev) with sorted frequent itemsets.
        n_u64s: Number of uint64 words per bitvector.
        min_count: Minimum support count threshold.
        max_results: Output buffer capacity.

    Returns:
        Tuple of (freq_itemsets_gpu, counts_gpu) where:
          freq_itemsets_gpu: CuPy array (n_results, k) of frequent k-itemsets in VRAM
          counts_gpu: CuPy array (n_results,) of support counts in VRAM
        Returns (None, None) if no frequent itemsets found.
    """
    _assert_k_supported(int(prev_freq_gpu.shape[1]) + 1, "count_k3plus_gpu_resident")
    import cupy as cp

    n_freq, k_prev = prev_freq_gpu.shape

    # Build prefix groups entirely on GPU
    group_starts, group_sizes, cumulative_pairs, total_candidates = build_prefix_groups_gpu(prev_freq_gpu)

    if total_candidates == 0:
        return None, None

    # Allocate output buffers in VRAM
    max_results = min(total_candidates, max_results)
    result_indices = cp.empty(max_results, dtype=cp.int64)  # FIXED: int32 -> int64 for >2B candidate indices
    result_counts = cp.empty(max_results, dtype=cp.int64)
    n_results = cp.zeros(1, dtype=cp.int64)

    # Run counting kernel
    kernel = get_cuda_kernel("count_k3plus_gpu_resident")
    block_size = 256
    grid = _grid_dims(total_candidates)
    n_groups = len(group_starts)

    kernel(
        grid,
        (block_size,),
        (
            bitvecs_gpu,
            prev_freq_gpu,
            np.int32(k_prev),
            group_starts,
            group_sizes,
            cumulative_pairs,
            np.int64(n_u64s),
            np.int64(n_groups),
            np.int64(total_candidates),
            np.int64(min_count),
            result_indices,
            result_counts,
            n_results,
            np.int64(max_results),
            np.int64(0),
        ),
    )
    cp.cuda.Stream.null.synchronize()

    n = int(n_results.get()[0])  # 4 bytes PCIe
    if n == 0:
        return None, None

    n = _warn_result_truncation(n, max_results, "filtered_kernel")

    # Decode results on GPU: result indices -> full k-itemset rows
    k = k_prev + 1
    output_itemsets = cp.empty((n, k), dtype=cp.int32)
    output_counts = cp.empty(n, dtype=cp.int64)

    decode_kernel = get_cuda_kernel("decode_candidates_gpu")
    decode_blocks = (n + 255) // 256
    decode_kernel(
        (decode_blocks,),
        (256,),
        (
            result_indices[:n],
            result_counts[:n],
            prev_freq_gpu,
            np.int32(k_prev),
            group_starts,
            group_sizes,
            cumulative_pairs,
            np.int64(n_groups),
            np.int64(n),
            output_itemsets,
            output_counts,
        ),
    )
    cp.cuda.Stream.null.synchronize()

    # Sort lexicographically on GPU for next iteration
    # lexsort: last key is primary, so reverse column order for lex sort
    sort_keys = cp.stack([output_itemsets[:, i] for i in range(k - 1, -1, -1)])
    sort_order = cp.lexsort(sort_keys)
    output_itemsets = output_itemsets[sort_order]
    output_counts = output_counts[sort_order]

    return output_itemsets, output_counts


def count_pairs_fused_k2_gpu_resident(bitvecs_gpu, freq_cols_gpu, n_u64s, min_count):
    """GPU-resident K=2: fused pair counting, returns CuPy arrays in VRAM.

    Uses the existing K=2 kernel but keeps results as CuPy arrays instead of
    converting to Python lists.

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s) with packed bitvectors.
        freq_cols_gpu: CuPy int32 array of frequent item column indices (sorted).
        n_u64s: Number of uint64 words per bitvector.
        min_count: Minimum support count threshold.

    Returns:
        Tuple of (pair_itemsets_gpu, counts_gpu) where:
          pair_itemsets_gpu: CuPy array (n_results, 2) of column index pairs in VRAM
          counts_gpu: CuPy array (n_results,) of support counts in VRAM
        Returns (None, None) if no frequent pairs found.
    """
    import cupy as cp

    n_freq = len(freq_cols_gpu)
    if n_freq < 2:
        return None, None

    n_pairs = n_freq * (n_freq - 1) // 2

    # Pre-allocate output buffers in VRAM
    max_results = min(n_pairs, 10_000_000)
    result_i = cp.empty(max_results, dtype=cp.int64)
    result_j = cp.empty(max_results, dtype=cp.int64)
    result_count = cp.empty(max_results, dtype=cp.int64)
    n_results = cp.zeros(1, dtype=cp.int64)

    kernel = get_cuda_kernel("count_pairs_fused_k2")

    block_size = 256
    grid = _grid_dims(n_pairs)
    kernel(
        grid,
        (block_size,),
        (
            bitvecs_gpu,
            freq_cols_gpu,
            np.int64(n_u64s),
            np.int32(n_freq),
            np.int64(min_count),
            result_i,
            result_j,
            result_count,
            n_results,
            np.int64(max_results),
            np.int64(0),
        ),
    )
    cp.cuda.Stream.null.synchronize()

    n = int(n_results.get()[0])  # 4 bytes PCIe
    if n == 0:
        return None, None

    n = _warn_result_truncation(n, max_results, "filtered_kernel")

    # Convert freq_items indices to column indices via GPU fancy indexing
    col_i = freq_cols_gpu[result_i[:n]]  # VRAM -> VRAM
    col_j = freq_cols_gpu[result_j[:n]]  # VRAM -> VRAM
    pair_itemsets = cp.stack([col_i, col_j], axis=1)  # (n, 2) in VRAM
    counts = result_count[:n]  # VRAM slice

    # Sort lexicographically on GPU (col_j secondary, col_i primary)
    sort_order = cp.lexsort(cp.stack([pair_itemsets[:, 1], pair_itemsets[:, 0]]))
    pair_itemsets = pair_itemsets[sort_order]
    counts = counts[sort_order]

    return pair_itemsets, counts


def count_pairs_fused_k2_gpu_resident_multi_gpu(bitvecs_gpu, freq_cols_gpu, n_u64s, min_count, n_gpus):
    """Multi-GPU GPU-resident K=2: split pairs across GPUs, merge results in VRAM.

    Each GPU processes its slice of the pair index range. Results are gathered
    to GPU 0, where fancy indexing + sorting happen entirely in VRAM.

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s) on GPU 0.
        freq_cols_gpu: CuPy int32 array of frequent item column indices (sorted, GPU 0).
        n_u64s: Number of uint64 words per bitvector.
        min_count: Minimum support count threshold.
        n_gpus: Number of GPUs to use.

    Returns:
        Tuple of (pair_itemsets_gpu, counts_gpu) CuPy arrays on GPU 0,
        or (None, None) if no frequent pairs found.
    """
    import cupy as cp
    from concurrent.futures import ThreadPoolExecutor

    n_freq = len(freq_cols_gpu)
    if n_freq < 2:
        return None, None

    n_pairs = n_freq * (n_freq - 1) // 2

    available_gpus = cp.cuda.runtime.getDeviceCount()
    n_gpus = min(n_gpus, available_gpus, n_pairs)

    if n_gpus <= 1:
        return count_pairs_fused_k2_gpu_resident(bitvecs_gpu, freq_cols_gpu, n_u64s, min_count)

    pairs_per_gpu = (n_pairs + n_gpus - 1) // n_gpus
    freq_items_np = freq_cols_gpu.get()
    bitvecs_np = bitvecs_gpu.get()
    kernel = get_cuda_kernel("count_pairs_fused_k2")

    def _run_on_gpu(device_id):
        pair_start = device_id * pairs_per_gpu
        pair_end = min(pair_start + pairs_per_gpu, n_pairs)
        n_gpu_pairs = pair_end - pair_start

        if n_gpu_pairs <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        with _get_device_lock(device_id):
            with cp.cuda.Device(device_id):
                stream = cp.cuda.Stream(non_blocking=True)
                with stream:
                    if device_id == 0:
                        bv_gpu = bitvecs_gpu
                    else:
                        bv_gpu = cp.array(bitvecs_np, dtype=cp.uint64)

                    freq_gpu = cp.array(freq_items_np, dtype=cp.int32)

                    gpu_max = min(n_gpu_pairs, 10_000_000)
                    result_i = cp.empty(gpu_max, dtype=cp.int64)
                    result_j = cp.empty(gpu_max, dtype=cp.int64)
                    result_count = cp.empty(gpu_max, dtype=cp.int64)
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
                            np.int64(gpu_max),
                            np.int64(pair_start),
                        ),
                    )

        stream.synchronize()

        with _get_device_lock(device_id):
            with cp.cuda.Device(device_id):
                n = int(n_results.get()[0])
                if n == 0:
                    return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
                n = _warn_result_truncation(
                    n, gpu_max, f"filtered_kernel (device {device_id})", k=2
                )
                ri = result_i[:n].get()
                rj = result_j[:n].get()
                rc = result_count[:n].get()

        return ri, rj, rc

    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = {executor.submit(_run_on_gpu, i): i for i in range(n_gpus)}
        gpu_results = {}
        for future in futures:
            gpu_results[futures[future]] = future.result()

    all_ri, all_rj, all_rc = [], [], []
    for device_id in range(n_gpus):
        if device_id in gpu_results:
            ri, rj, rc = gpu_results[device_id]
            if len(ri) > 0:
                all_ri.append(ri)
                all_rj.append(rj)
                all_rc.append(rc)

    if not all_ri:
        return None, None

    ri_merged = np.concatenate(all_ri)
    rj_merged = np.concatenate(all_rj)
    rc_merged = np.concatenate(all_rc)

    # Build + sort on GPU 0 in VRAM
    with cp.cuda.Device(0):
        ri_gpu = cp.array(ri_merged, dtype=cp.int64)  # FIXED: int32 -> int64 for >2B candidate indices
        rj_gpu = cp.array(rj_merged, dtype=cp.int64)

        col_i = freq_cols_gpu[ri_gpu]
        col_j = freq_cols_gpu[rj_gpu]
        pair_itemsets = cp.stack([col_i, col_j], axis=1)
        counts = cp.array(rc_merged, dtype=cp.int64)

        sort_order = cp.lexsort(cp.stack([pair_itemsets[:, 1], pair_itemsets[:, 0]]))
        pair_itemsets = pair_itemsets[sort_order]
        counts = counts[sort_order]

    return pair_itemsets, counts


def count_k3plus_gpu_resident_multi_gpu(bitvecs_gpu, prev_freq_gpu, n_u64s, min_count, n_gpus, max_results=10_000_000):
    """Multi-GPU GPU-resident K>=3: split candidates across GPUs, decode + sort on GPU 0.

    Each GPU processes its slice of candidates. Results (indices + counts) are
    gathered to GPU 0, where decode and sort happen entirely in VRAM.

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s) on GPU 0.
        prev_freq_gpu: CuPy array of shape (n_freq, k_prev) on GPU 0.
        n_u64s: Number of uint64 words per bitvector.
        min_count: Minimum support count threshold.
        n_gpus: Number of GPUs to use.
        max_results: Output buffer capacity per GPU.

    Returns:
        Tuple of (freq_itemsets_gpu, counts_gpu) CuPy arrays on GPU 0,
        or (None, None) if no frequent itemsets found.
    """
    _assert_k_supported(int(prev_freq_gpu.shape[1]) + 1, "count_k3plus_gpu_resident_multi_gpu")
    import cupy as cp
    from concurrent.futures import ThreadPoolExecutor

    n_freq, k_prev = prev_freq_gpu.shape

    # Build prefix groups on GPU 0
    group_starts, group_sizes, cumulative_pairs, total_candidates = build_prefix_groups_gpu(prev_freq_gpu)

    if total_candidates == 0:
        return None, None

    available_gpus = cp.cuda.runtime.getDeviceCount()
    n_gpus = min(n_gpus, available_gpus, total_candidates)

    if n_gpus <= 1:
        return count_k3plus_gpu_resident(bitvecs_gpu, prev_freq_gpu, n_u64s, min_count, max_results)

    # Get numpy copies for replication to other GPUs
    bitvecs_np = bitvecs_gpu.get()
    prev_freq_np = prev_freq_gpu.get()
    group_starts_np = group_starts.get()
    group_sizes_np = group_sizes.get()
    cumulative_pairs_np = cumulative_pairs.get()
    n_groups = len(group_starts)

    cands_per_gpu = (total_candidates + n_gpus - 1) // n_gpus
    kernel = get_cuda_kernel("count_k3plus_gpu_resident")

    def _run_on_gpu(device_id):
        cand_start = device_id * cands_per_gpu
        cand_end = min(cand_start + cands_per_gpu, total_candidates)
        n_gpu_cands = cand_end - cand_start

        if n_gpu_cands <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        with _get_device_lock(device_id):
            with cp.cuda.Device(device_id):
                stream = cp.cuda.Stream(non_blocking=True)
                with stream:
                    if device_id == 0:
                        bv_gpu = bitvecs_gpu
                        pf_gpu = prev_freq_gpu
                        gs_gpu = group_starts
                        gsz_gpu = group_sizes
                        cp_gpu = cumulative_pairs
                    else:
                        bv_gpu = cp.array(bitvecs_np, dtype=cp.uint64)
                        pf_gpu = cp.array(prev_freq_np, dtype=cp.int32)
                        gs_gpu = cp.array(group_starts_np, dtype=cp.int64)
                        gsz_gpu = cp.array(group_sizes_np, dtype=cp.int64)
                        cp_gpu = cp.array(cumulative_pairs_np, dtype=cp.int64)

                    gpu_max = min(n_gpu_cands, max_results)
                    res_indices = cp.empty(gpu_max, dtype=cp.int64)  # FIXED: int32 -> int64 for >2B candidate indices
                    res_counts = cp.empty(gpu_max, dtype=cp.int64)
                    n_res = cp.zeros(1, dtype=cp.int64)

                    block_size = 256
                    grid = _grid_dims(n_gpu_cands)
                    kernel(
                        grid,
                        (block_size,),
                        (
                            bv_gpu,
                            pf_gpu,
                            np.int32(k_prev),
                            gs_gpu,
                            gsz_gpu,
                            cp_gpu,
                            np.int64(n_u64s),
                            np.int64(n_groups),
                            np.int64(total_candidates),
                            np.int64(min_count),
                            res_indices,
                            res_counts,
                            n_res,
                            np.int64(gpu_max),
                            np.int64(cand_start),
                        ),
                    )

        stream.synchronize()

        with _get_device_lock(device_id):
            with cp.cuda.Device(device_id):
                n = int(n_res.get()[0])
                if n == 0:
                    return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
                n = _warn_result_truncation(
                    n,
                    gpu_max,
                    f"filtered_kernel (device {device_id})",
                    k=int(prev_freq_gpu.shape[1]) + 1,
                )
                ri = res_indices[:n].get()
                rc = res_counts[:n].get()

        return ri, rc

    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = {executor.submit(_run_on_gpu, i): i for i in range(n_gpus)}
        gpu_results = {}
        for future in futures:
            gpu_results[futures[future]] = future.result()

    all_ri, all_rc = [], []
    for device_id in range(n_gpus):
        if device_id in gpu_results:
            ri, rc = gpu_results[device_id]
            if len(ri) > 0:
                all_ri.append(ri)
                all_rc.append(rc)

    if not all_ri:
        return None, None

    ri_merged = np.concatenate(all_ri)
    rc_merged = np.concatenate(all_rc)

    # Decode + sort on GPU 0 (group arrays still in VRAM from build_prefix_groups_gpu)
    n = len(ri_merged)
    k = k_prev + 1

    with cp.cuda.Device(0):
        result_indices_gpu = cp.array(ri_merged, dtype=cp.int64)  # FIXED: int32 -> int64 for >2B candidate indices
        result_counts_gpu = cp.array(rc_merged, dtype=cp.int64)

        output_itemsets = cp.empty((n, k), dtype=cp.int32)
        output_counts = cp.empty(n, dtype=cp.int64)

        decode_kernel = get_cuda_kernel("decode_candidates_gpu")
        decode_blocks = (n + 255) // 256
        decode_kernel(
            (decode_blocks,),
            (256,),
            (
                result_indices_gpu,
                result_counts_gpu,
                prev_freq_gpu,
                np.int32(k_prev),
                group_starts,
                group_sizes,
                cumulative_pairs,
                np.int64(n_groups),
                np.int64(n),
                output_itemsets,
                output_counts,
            ),
        )
        cp.cuda.Stream.null.synchronize()

        sort_keys = cp.stack([output_itemsets[:, i] for i in range(k - 1, -1, -1)])
        sort_order = cp.lexsort(sort_keys)
        output_itemsets = output_itemsets[sort_order]
        output_counts = output_counts[sort_order]

    return output_itemsets, output_counts


