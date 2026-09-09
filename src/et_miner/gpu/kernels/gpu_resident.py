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
        where group_starts/sizes/cumulative_pairs are CuPy int32/int64 arrays in
        VRAM **on prev_freq_gpu's device**, and total_candidates is a Python int
        (single 8-byte PCIe transfer).

    Device: every allocation here follows the input, not the ambient current
    device. It used to follow the ambient one, so calling it from a thread
    parked on device 0 with `prev_freq_gpu` on device 1 built `cp.array([True])`
    and friends on the wrong card and raised on the first comparison. Fixed
    here rather than at the call site so the property holds for every caller.
    N20.
    """
    import cupy as cp

    with cp.cuda.Device(int(prev_freq_gpu.device.id)):
        return _build_prefix_groups_gpu_impl(prev_freq_gpu)


def _build_prefix_groups_gpu_impl(prev_freq_gpu):
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
    to the caller's device, where fancy indexing + sorting happen in VRAM.

    Contract: all inputs must be resident on ONE device -- the wrapper
    replicates them to the others. That device used to be hardcoded as 0.

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s). Its device is the
            home device for every other input and for the returned arrays.
        freq_cols_gpu: CuPy int32 array of frequent item column indices
            (sorted), on the home device.
        n_u64s: Number of uint64 words per bitvector.
        min_count: Minimum support count threshold.
        n_gpus: Number of GPUs to use.

    Returns:
        Tuple of (pair_itemsets_gpu, counts_gpu) CuPy arrays on the caller's
        device, or (None, None) if no frequent pairs found.
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
    # The device the caller's arrays actually live on. Every one of these
    # wrappers hardcoded `if device_id == 0`, i.e. "the caller's bitvecs are on
    # GPU 0". When they are not, device 0 aliases a foreign array -- the
    # "device where the array resides (0) is different from the current device
    # (1)" fault -- and the real home device re-uploads a copy of what it
    # already holds. Latent while every in-tree route builds on device 0;
    # gpu/dispatch reaches these from callers that need not. N10
    _home = int(bitvecs_gpu.device.id)
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
                    if device_id == _home:
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

    # Build + sort on the HOME device in VRAM.
    #
    # N20 -- a second device assumption, distinct from N10 and in the output
    # half rather than the input alias. This tail forced device 0 and then
    # indexed `freq_cols_gpu`, which belongs to the caller and lives wherever
    # the caller put it. With bitvecs on device 1 that raises "the device where
    # the array resides (1) is different from the current device (0)". Fixing
    # only the `if device_id == 0` alias leaves this live, which is how it was
    # found: the N10 test still failed here.
    with cp.cuda.Device(_home):
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
    """Multi-GPU GPU-resident K>=3: split candidates across GPUs, decode + sort on the home device.

    Each GPU processes its slice of candidates. Results (indices + counts) are
    gathered to the caller's device, where decode and sort happen in VRAM.

    Contract: all inputs must be resident on ONE device -- the wrapper
    replicates them to the others and aliases them on that one. It used to
    assume that device was 0 and is now taken from `bitvecs_gpu.device.id`; a
    mixed-device call raises rather than being silently repaired by a transfer,
    because that is a caller bug and a hidden copy makes it unattributable.

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s). Its device is the
            home device for every other input.
        prev_freq_gpu: CuPy array of shape (n_freq, k_prev), on the home device.
        n_u64s: Number of uint64 words per bitvector.
        min_count: Minimum support count threshold.
        n_gpus: Number of GPUs to use.
        max_results: Output buffer capacity per GPU.

    Returns:
        Tuple of (freq_itemsets_gpu, counts_gpu) CuPy arrays on the caller's
        device, or (None, None) if no frequent itemsets found.
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
    # The device the caller's arrays actually live on. Every one of these
    # wrappers hardcoded `if device_id == 0`, i.e. "the caller's bitvecs are on
    # GPU 0". When they are not, device 0 aliases a foreign array -- the
    # "device where the array resides (0) is different from the current device
    # (1)" fault -- and the real home device re-uploads a copy of what it
    # already holds. Latent while every in-tree route builds on device 0;
    # gpu/dispatch reaches these from callers that need not. N10
    _home = int(bitvecs_gpu.device.id)
    prev_freq_np = prev_freq_gpu.get()
    group_starts_np = group_starts.get()
    group_sizes_np = group_sizes.get()
    cumulative_pairs_np = cumulative_pairs.get()
    n_groups = len(group_starts)

    # All five inputs are aliased together on the home device below, so they
    # must live together. Asserted rather than transferred: a mixed-device
    # caller is a caller bug, and silently copying would hide it behind a
    # transfer nobody attributes. See this function's docstring for the
    # contract that makes the assert the documented behaviour.
    for _name, _arr in (
        ("prev_freq_gpu", prev_freq_gpu),
        ("group_starts", group_starts),
        ("group_sizes", group_sizes),
        ("cumulative_pairs", cumulative_pairs),
    ):
        if int(_arr.device.id) != _home:
            raise ValueError(
                f"count_k3plus_gpu_resident_multi_gpu: {_name} is on device "
                f"{int(_arr.device.id)} but bitvecs_gpu is on device {_home}. "
                "All inputs must be resident on one device; the wrapper "
                "replicates them to the others."
            )

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
                    if device_id == _home:
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

    # Decode + sort on the HOME device (group arrays still in VRAM from
    # build_prefix_groups_gpu, which built them there). N20, as above: the
    # decode kernel is handed prev_freq_gpu, group_starts, group_sizes and
    # cumulative_pairs, all caller-side or derived from caller-side arrays.
    n = len(ri_merged)
    k = k_prev + 1

    with cp.cuda.Device(_home):
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


