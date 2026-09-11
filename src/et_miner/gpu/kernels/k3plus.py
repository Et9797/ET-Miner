"""Fused k>=3 counting kernels: candidate generation, prefix groups, dense
allcounts for row-split multi-GPU, and CSR tidset intersection."""

from __future__ import annotations

from collections import namedtuple

import numpy as np
from loguru import logger

from .loader import _assert_k_supported, _get_device_lock, _grid_dims, _warn_result_truncation, get_cuda_kernel


def count_itemsets_fused_k3plus(bitvecs_gpu, candidates, n_u64s, min_count, max_results=10_000_000):
    """Fused k>=3 kernel: count + filter candidates in ONE launch.

    Replaces the Python pipeline of:
      np.array per candidate -> flatten -> _count_batch() -> Python filter

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s) with packed bitvectors.
        candidates: list of tuples of int column indices, e.g. [(0,1,3), (0,2,4)].
        n_u64s: Number of uint64 words per bitvector.
        min_count: Minimum support count threshold.
        max_results: Output buffer capacity.

    Returns:
        Tuple of (frequent_candidates, counts) where:
          frequent_candidates: list of tuples (column index tuples) that met min_count
          counts: numpy array of int64 support counts
    """
    _assert_k_supported(len(candidates[0]) if candidates else None, "count_itemsets_fused_k3plus")
    import cupy as cp

    n_candidates = len(candidates)
    if n_candidates == 0:
        return [], np.array([], dtype=np.int64)

    # Flatten candidates -> all_items + offsets (same pattern as _count_batch)
    all_items = []
    offsets = [0]
    for candidate in candidates:
        all_items.extend(candidate)
        offsets.append(len(all_items))

    all_items_gpu = cp.array(all_items, dtype=cp.int32)
    offsets_gpu = cp.array(offsets, dtype=cp.int64)  # FIXED: int32 -> int64 for >2B elements

    # Output buffers
    max_results = min(n_candidates, max_results)
    result_indices = cp.empty(max_results, dtype=cp.int64)  # FIXED: int32 -> int64 for >2B candidate indices
    result_counts = cp.empty(max_results, dtype=cp.int64)
    n_results = cp.zeros(1, dtype=cp.int64)

    kernel = get_cuda_kernel("count_itemsets_fused_k3plus")

    # Grid: 1 block per candidate. Block: 256 threads for u64 word parallelism.
    block_size = 256
    grid = _grid_dims(n_candidates)
    n_cols = bitvecs_gpu.shape[0]
    kernel(
        grid,
        (block_size,),
        (
            bitvecs_gpu,
            all_items_gpu,
            offsets_gpu,
            np.int64(n_u64s),
            np.int32(n_cols),
            np.int64(n_candidates),
            np.int64(min_count),
            result_indices,
            result_counts,
            n_results,
            np.int64(max_results),
            np.int64(0),
        ),  # candidate_offset=0 for single GPU
    )
    cp.cuda.Stream.null.synchronize()

    n = int(n_results.get()[0])
    if n == 0:
        return [], np.array([], dtype=np.int64)

    n = _warn_result_truncation(n, max_results, "filtered_kernel")

    # Map result indices back to candidate tuples
    ri = result_indices[:n].get()
    rc = result_counts[:n].get()

    frequent = [candidates[int(ri[idx])] for idx in range(n)]
    return frequent, rc


def count_itemsets_fused_k3plus_multi_gpu(bitvecs_gpu, candidates, n_u64s, min_count, n_gpus, max_results=10_000_000):
    """Multi-GPU fused k>=3: split candidates across GPUs for linear scaling.

    Each GPU gets the FULL bitvec matrix but processes only its subset of candidates.
    Candidate-index splitting (same pattern as pair-index splitting for k=2).

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s) on GPU 0.
        candidates: list of tuples of int column indices.
        n_u64s: Number of uint64 words per bitvector.
        min_count: Minimum support count threshold.
        n_gpus: Number of GPUs to use.
        max_results: Output buffer capacity per GPU.

    Returns:
        Tuple of (frequent_candidates, counts) - same format as single-GPU version.
    """
    _assert_k_supported(len(candidates[0]) if candidates else None, "count_itemsets_fused_k3plus_multi_gpu")
    import cupy as cp
    from concurrent.futures import ThreadPoolExecutor

    n_candidates = len(candidates)
    if n_candidates == 0:
        return [], np.array([], dtype=np.int64)

    available_gpus = cp.cuda.runtime.getDeviceCount()
    n_gpus = min(n_gpus, available_gpus, n_candidates)

    if n_gpus <= 1:
        return count_itemsets_fused_k3plus(bitvecs_gpu, candidates, n_u64s, min_count, max_results)

    # Flatten ALL candidates (shared across GPUs — each GPU uses candidate_offset to index)
    all_items = []
    offsets = [0]
    for candidate in candidates:
        all_items.extend(candidate)
        offsets.append(len(all_items))

    all_items_np = np.array(all_items, dtype=np.int32)
    offsets_np = np.array(offsets, dtype=np.int64)
    bitvecs_np = bitvecs_gpu.get()
    # The device the caller's arrays actually live on. Every one of these
    # wrappers hardcoded `if device_id == 0`, i.e. "the caller's bitvecs are on
    # GPU 0". When they are not, device 0 aliases a foreign array -- the
    # "device where the array resides (0) is different from the current device
    # (1)" fault -- and the real home device re-uploads a copy of what it
    # already holds. Latent while every in-tree route builds on device 0;
    # gpu/dispatch reaches these from callers that need not. N10
    _home = int(bitvecs_gpu.device.id)
    n_cols = bitvecs_gpu.shape[0]

    candidates_per_gpu = (n_candidates + n_gpus - 1) // n_gpus
    kernel = get_cuda_kernel("count_itemsets_fused_k3plus")

    def _run_on_gpu(device_id):
        cand_start = device_id * candidates_per_gpu
        cand_end = min(cand_start + candidates_per_gpu, n_candidates)
        n_gpu_candidates = cand_end - cand_start

        if n_gpu_candidates <= 0:
            return [], np.array([], dtype=np.int64)

        with _get_device_lock(device_id):
            with cp.cuda.Device(device_id):
                stream = cp.cuda.Stream(non_blocking=True)
                with stream:
                    if device_id == _home:
                        bv_gpu = bitvecs_gpu
                    else:
                        bv_gpu = cp.array(bitvecs_np, dtype=cp.uint64)

                    items_gpu = cp.array(all_items_np, dtype=cp.int32)
                    offs_gpu = cp.array(offsets_np, dtype=cp.int64)  # FIXED: int32 -> int64 (was negating the fix!)

                    gpu_max = min(n_gpu_candidates, max_results)
                    res_indices = cp.empty(gpu_max, dtype=cp.int64)  # FIXED: int32 -> int64 for >2B candidate indices
                    res_counts = cp.empty(gpu_max, dtype=cp.int64)
                    n_res = cp.zeros(1, dtype=cp.int64)

                    block_size = 256
                    grid = _grid_dims(n_gpu_candidates)
                    kernel(
                        grid,
                        (block_size,),
                        (
                            bv_gpu,
                            items_gpu,
                            offs_gpu,
                            np.int64(n_u64s),
                            np.int32(n_cols),
                            np.int64(n_candidates),
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
                    return [], np.array([], dtype=np.int64)

                n = _warn_result_truncation(
                    n,
                    gpu_max,
                    f"filtered_kernel (device {device_id})",
                    k=len(candidates[0]) if candidates else None,
                )
                ri = res_indices[:n].get()
                rc = res_counts[:n].get()

        frequent = [candidates[int(ri[idx])] for idx in range(n)]
        return frequent, rc

    all_frequent = []
    all_counts = []

    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = {executor.submit(_run_on_gpu, i): i for i in range(n_gpus)}
        results = {}
        for future in futures:
            results[futures[future]] = future.result()

    for device_id in range(n_gpus):
        if device_id in results:
            frequent, counts = results[device_id]
            if len(frequent) > 0:
                all_frequent.extend(frequent)
                all_counts.append(counts)

    if not all_counts:
        return [], np.array([], dtype=np.int64)

    return all_frequent, np.concatenate(all_counts)


def count_k3plus_fully_fused(bitvecs_gpu, prev_frequent, k, n_u64s, min_count, max_results=10_000_000):
    """Fully-fused k>=3: candidate GENERATION + count + filter in ONE kernel launch.

    Eliminates ALL Python candidate generation overhead. Prefix groups are
    built on CPU (tiny), transferred to GPU, and candidates are generated
    on-the-fly using triangular number inverse (same trick as fused k=2).

    Correctness without Apriori pruning: support is anti-monotone, so any
    candidate with a non-frequent subset also has support < min_count and
    will be filtered by the kernel's in-GPU min_count check.

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s) with packed bitvectors.
        prev_frequent: List of frequent (k-1)-itemsets as tuples of column indices.
        k: Current itemset size being generated.
        n_u64s: Number of uint64 words per bitvector.
        min_count: Minimum support count threshold.
        max_results: Output buffer capacity.

    Returns:
        Tuple of (frequent_candidates, counts) where:
          frequent_candidates: list of tuples (column index tuples) that met min_count
          counts: numpy array of int64 support counts
    """
    _assert_k_supported(k, "count_k3plus_fully_fused")
    import cupy as cp
    import math

    # Build prefix groups from prev_frequent
    groups_info = build_k3plus_groups(prev_frequent)
    if groups_info is None:
        return [], np.array([], dtype=np.int64)
    groups = groups_info.groups
    group_prefix_items = groups_info.prefix_items
    group_prefix_offsets = groups_info.prefix_offsets
    group_suffixes = groups_info.suffixes
    group_suffix_offsets = groups_info.suffix_offsets
    cumulative_pairs = groups_info.cumulative_pairs
    total_candidates = groups_info.total_candidates

    # Transfer to GPU — O(n_frequent) not O(n_candidates)
    gpi_gpu = cp.asarray(group_prefix_items)
    gpo_gpu = cp.asarray(group_prefix_offsets)
    gs_gpu = cp.asarray(group_suffixes)
    gso_gpu = cp.asarray(group_suffix_offsets)
    cp_gpu = cp.asarray(cumulative_pairs)

    # Output buffers
    max_results = min(total_candidates, max_results)
    result_indices = cp.empty(max_results, dtype=cp.int64)  # FIXED: int32 -> int64 for >2B candidate indices
    result_counts = cp.empty(max_results, dtype=cp.int64)
    n_results = cp.zeros(1, dtype=cp.int64)

    kernel = get_cuda_kernel("count_k3plus_from_groups")

    block_size = 256
    grid = _grid_dims(total_candidates)
    kernel(
        grid,
        (block_size,),
        (
            bitvecs_gpu,
            gpi_gpu,
            gpo_gpu,
            gs_gpu,
            gso_gpu,
            cp_gpu,
            np.int64(n_u64s),
            np.int64(len(groups)),
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

    n = int(n_results.get()[0])
    if n == 0:
        return [], np.array([], dtype=np.int64)

    n = _warn_result_truncation(n, max_results, "filtered_kernel")

    # Decode results: linear cand_idx -> candidate tuple (CPU, only for frequent results)
    ri = result_indices[:n].get()
    rc = result_counts[:n].get()

    cumulative_pairs_np = np.array(cumulative_pairs, dtype=np.int64)
    frequent = []
    for idx in range(n):
        cand_idx = int(ri[idx])
        # Binary search for group
        g = int(np.searchsorted(cumulative_pairs_np, cand_idx, side="right")) - 1
        pair_idx = cand_idx - cumulative_pairs[g]

        # Triangular inverse
        j_val = int(0.5 + math.sqrt(0.25 + 2.0 * pair_idx))
        i_val = pair_idx - j_val * (j_val - 1) // 2

        # Reconstruct candidate
        prefix = tuple(group_prefix_items[group_prefix_offsets[g] : group_prefix_offsets[g + 1]])
        suf_start = group_suffix_offsets[g]
        suffix_i = group_suffixes[suf_start + i_val]
        suffix_j = group_suffixes[suf_start + j_val]
        frequent.append(prefix + (suffix_i, suffix_j))

    return frequent, rc


def count_k3plus_fully_fused_multi_gpu(
    bitvecs_gpu, prev_frequent, k, n_u64s, min_count, n_gpus, max_results=10_000_000
):
    """Multi-GPU fully-fused k>=3: candidate gen + count + filter split across GPUs.

    Each GPU gets the FULL bitvec matrix + group structure but processes only
    its slice of the candidate index range via candidate_offset.

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s) on GPU 0.
        prev_frequent: List of frequent (k-1)-itemsets as tuples of column indices.
        k: Current itemset size being generated.
        n_u64s: Number of uint64 words per bitvector.
        min_count: Minimum support count threshold.
        n_gpus: Number of GPUs to use.
        max_results: Output buffer capacity per GPU.

    Returns:
        Tuple of (frequent_candidates, counts) - same format as single-GPU version.
    """
    _assert_k_supported(k, "count_k3plus_fully_fused_multi_gpu")
    import cupy as cp
    import math
    from concurrent.futures import ThreadPoolExecutor

    # Build prefix groups (CPU, shared across GPUs)
    groups_info = build_k3plus_groups(prev_frequent)
    if groups_info is None:
        return [], np.array([], dtype=np.int64)
    groups = groups_info.groups
    group_prefix_items = groups_info.prefix_items
    group_prefix_offsets = groups_info.prefix_offsets
    group_suffixes = groups_info.suffixes
    group_suffix_offsets = groups_info.suffix_offsets
    cumulative_pairs = groups_info.cumulative_pairs
    total_candidates = groups_info.total_candidates

    # Limit GPUs
    available_gpus = cp.cuda.runtime.getDeviceCount()
    n_gpus = min(n_gpus, available_gpus, total_candidates)

    if n_gpus <= 1:
        return count_k3plus_fully_fused(bitvecs_gpu, prev_frequent, k, n_u64s, min_count, max_results)

    # Numpy arrays already from build_k3plus_groups, ready for replication to each GPU
    gpi_np = group_prefix_items
    gpo_np = group_prefix_offsets
    gs_np = group_suffixes
    gso_np = group_suffix_offsets
    cp_np = cumulative_pairs
    bitvecs_np = bitvecs_gpu.get()
    # The device the caller's arrays actually live on. Every one of these
    # wrappers hardcoded `if device_id == 0`, i.e. "the caller's bitvecs are on
    # GPU 0". When they are not, device 0 aliases a foreign array -- the
    # "device where the array resides (0) is different from the current device
    # (1)" fault -- and the real home device re-uploads a copy of what it
    # already holds. Latent while every in-tree route builds on device 0;
    # gpu/dispatch reaches these from callers that need not. N10
    _home = int(bitvecs_gpu.device.id)

    cands_per_gpu = (total_candidates + n_gpus - 1) // n_gpus
    kernel = get_cuda_kernel("count_k3plus_from_groups")
    n_groups = len(groups)

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
                    else:
                        bv_gpu = cp.array(bitvecs_np, dtype=cp.uint64)

                    gpi_gpu = cp.array(gpi_np, dtype=cp.int32)
                    gpo_gpu = cp.array(gpo_np, dtype=cp.int64)
                    gs_gpu = cp.array(gs_np, dtype=cp.int32)
                    gso_gpu = cp.array(gso_np, dtype=cp.int64)
                    cp_gpu_arr = cp.array(cp_np, dtype=cp.int64)

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
                            gpi_gpu,
                            gpo_gpu,
                            gs_gpu,
                            gso_gpu,
                            cp_gpu_arr,
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
                    n, gpu_max, f"filtered_kernel (device {device_id})", k=k
                )
                ri = res_indices[:n].get()
                rc = res_counts[:n].get()

        return ri, rc

    # Launch on all GPUs
    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = {executor.submit(_run_on_gpu, i): i for i in range(n_gpus)}
        gpu_results = {}
        for future in futures:
            gpu_results[futures[future]] = future.result()

    # Merge + decode results
    all_ri = []
    all_rc = []
    for device_id in range(n_gpus):
        if device_id in gpu_results:
            ri, rc = gpu_results[device_id]
            if len(ri) > 0:
                all_ri.append(ri)
                all_rc.append(rc)

    if not all_ri:
        return [], np.array([], dtype=np.int64)

    ri_merged = np.concatenate(all_ri)
    rc_merged = np.concatenate(all_rc)

    # Decode all results (CPU)
    cumulative_pairs_np = np.array(cumulative_pairs, dtype=np.int64)
    frequent = []
    for idx in range(len(ri_merged)):
        cand_idx = int(ri_merged[idx])
        g = int(np.searchsorted(cumulative_pairs_np, cand_idx, side="right")) - 1
        pair_idx = cand_idx - cumulative_pairs[g]

        j_val = int(0.5 + math.sqrt(0.25 + 2.0 * pair_idx))
        i_val = pair_idx - j_val * (j_val - 1) // 2

        prefix = tuple(group_prefix_items[group_prefix_offsets[g] : group_prefix_offsets[g + 1]])
        suf_start = group_suffix_offsets[g]
        suffix_i = group_suffixes[suf_start + i_val]
        suffix_j = group_suffixes[suf_start + j_val]
        frequent.append(prefix + (suffix_i, suffix_j))

    return frequent, rc_merged




# ── Dense counting for row-split multi-GPU ────────────────────────────
# Eliminates the recount phase by outputting counts for ALL candidates.
# In row-split mode, all GPUs generate the same candidates (same prev_frequent),
# so element-wise sum of dense count arrays = exact global counts.

# suffix_src_rows (int64, parallel to `suffixes`, default None): the row of the
# builder's input flat array each suffix slot came from. Opt-in via
# build_k3plus_groups_from_flat(with_src_rows=True) — 8 B per prev row that
# only the sparse-CSR path needs (its kernels map a candidate's two suffix
# slots back to prev-level CSR rows). Trailing field with a default keeps
# every keyword constructor and the positional arity of the first seven intact.
K3PlusGroups = namedtuple(
    "K3PlusGroups",
    [
        "prefix_items",
        "prefix_offsets",
        "suffixes",
        "suffix_offsets",
        "cumulative_pairs",
        "total_candidates",
        "groups",
        "suffix_src_rows",
    ],
    defaults=(None,),
)

_STALE_RUST_WARNED = False


def _warn_stale_rust_once(what: str) -> None:
    """One-time warning when the installed et_miner_rust predates an API it
    is being called with; callers then take the numpy/Python fallbacks."""
    global _STALE_RUST_WARNED
    if not _STALE_RUST_WARNED:
        _STALE_RUST_WARNED = True
        logger.warning(
            f"et_miner_rust is stale ({what}) — rebuild with "
            "`cd rust_ext && uv run maturin develop --release`; using numpy/Python fallbacks meanwhile"
        )


def build_k3plus_groups(prev_frequent):
    """Build prefix group arrays from prev_frequent (k-1)-itemsets.

    CPU-only, O(n_frequent). Returns numpy arrays ready for GPU upload.

    Args:
        prev_frequent: List of frequent (k-1)-itemsets as tuples of column indices.

    Returns:
        K3PlusGroups namedtuple or None if no valid groups.
    """
    prefix_groups: dict[tuple[int, ...], list[int]] = {}
    for itemset in prev_frequent:
        prefix = itemset[:-1]
        suffix = itemset[-1]
        if prefix in prefix_groups:
            prefix_groups[prefix].append(suffix)
        else:
            prefix_groups[prefix] = [suffix]

    groups = [(prefix, sorted(suffixes)) for prefix, suffixes in prefix_groups.items() if len(suffixes) >= 2]

    if not groups:
        return None

    group_prefix_items = []
    group_prefix_offsets = [0]
    group_suffixes = []
    group_suffix_offsets = [0]
    cumulative_pairs = [0]

    total_candidates = 0
    for prefix, suffixes in groups:
        group_prefix_items.extend(prefix)
        group_prefix_offsets.append(len(group_prefix_items))
        group_suffixes.extend(suffixes)
        group_suffix_offsets.append(len(group_suffixes))
        n_pairs = len(suffixes) * (len(suffixes) - 1) // 2
        total_candidates += n_pairs
        cumulative_pairs.append(total_candidates)

    if total_candidates == 0:
        return None

    return K3PlusGroups(
        prefix_items=np.array(group_prefix_items, dtype=np.int32),
        prefix_offsets=np.array(group_prefix_offsets, dtype=np.int64),
        suffixes=np.array(group_suffixes, dtype=np.int32),
        suffix_offsets=np.array(group_suffix_offsets, dtype=np.int64),
        cumulative_pairs=np.array(cumulative_pairs, dtype=np.int64),
        total_candidates=total_candidates,
        groups=groups,
    )




def upload_k3plus_groups(groups_info, device_id, *, with_src_rows: bool = False):
    """Upload K3+ group data to GPU once, keep resident across chunks — ~40 GB at K=8.

    Includes "ctp" (cumulative tile-pairs) for the shared/tiled kernel;
    negligible extra bytes (one int64 per group + 1) for the legacy path.
    "tc" carries total_candidates for range checks. With ``with_src_rows``
    the sparse-CSR kernels' "gsr" (``suffix_src_rows``, int64) is uploaded
    too — it requires groups built with ``with_src_rows=True``.
    """
    import cupy as cp

    from .shared_tiled import compute_cumulative_tilepairs

    with cp.cuda.Device(device_id):
        out = {
            "gpi": cp.array(groups_info.prefix_items, dtype=cp.int32),
            "gpo": cp.array(groups_info.prefix_offsets, dtype=cp.int64),
            "gs": cp.array(groups_info.suffixes, dtype=cp.int32),
            "gso": cp.array(groups_info.suffix_offsets, dtype=cp.int64),
            "cp": cp.array(groups_info.cumulative_pairs, dtype=cp.int64),
            "ctp": cp.array(compute_cumulative_tilepairs(groups_info.suffix_offsets), dtype=cp.int64),
            "tc": int(groups_info.total_candidates),
        }
        if with_src_rows:
            gsr = getattr(groups_info, "suffix_src_rows", None)
            if gsr is None:
                raise ValueError(
                    "upload_k3plus_groups(with_src_rows=True) needs groups built with "
                    "build_k3plus_groups_from_flat(..., with_src_rows=True)"
                )
            out["gsr"] = cp.array(gsr, dtype=cp.int64)
        return out


def count_k3plus_allcounts(bitvecs_gpu, groups_info, n_u64s, chunk_start=0, chunk_size=None, groups_gpu=None, variant=None):
    """Dense K>=3 counting: returns support count for candidates in range.

    Takes pre-built groups_info from build_k3plus_groups().
    No threshold filtering — outputs counts for every candidate in range.

    Supports candidate-range chunking: when chunk_start/chunk_size are set,
    only processes candidates [chunk_start, chunk_start + chunk_size).
    The CUDA kernel uses candidate_offset for chunk-relative output indexing:
    result_counts[cand_idx - candidate_offset] instead of result_counts[cand_idx].

    When groups_gpu is provided, skips group data upload (already resident).
    This is critical for K=8+: ~40 GB group data uploaded once, not per chunk.

    Memory: chunk_size × 4 bytes (int32, not total_candidates × 8 — counts
    are bounded by n_transactions, guarded to < 2^31 by the row-split caller,
    and the ≤8-GPU partial sum is bounded by the same n_transactions).

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s).
        groups_info: K3PlusGroups namedtuple from build_k3plus_groups().
        n_u64s: Number of uint64 words per bitvector.
        chunk_start: First candidate index to process (default: 0).
        chunk_size: Number of candidates to process (default: all).
        groups_gpu: Pre-uploaded group data dict from upload_k3plus_groups().
            If None, uploads fresh (backward compatible legacy path).
        variant: "legacy" | "shared" | None (None resolves
            ET_MINER_KERNEL_VARIANT). The shared/tiled kernel requires
            group-aligned chunks — callers route mega-group sub-chunks
            here with variant="legacy" (see plan_group_chunks).

    Returns:
        CuPy int32 array of shape (chunk_size,) with counts — stays in VRAM.
    """
    from .shared_tiled import _assert_k_cap

    _assert_k_cap(groups_info)
    import cupy as cp

    if variant is None:
        from et_miner.gpu.dispatch import resolved_kernel_variant

        variant = resolved_kernel_variant()
    if variant == "shared":
        from .shared_tiled import count_shared_tiled_allcounts

        return count_shared_tiled_allcounts(
            bitvecs_gpu, groups_info, n_u64s, chunk_start=chunk_start, chunk_size=chunk_size, groups_gpu=groups_gpu
        )

    tc = groups_info.total_candidates
    if chunk_size is None:
        chunk_size = tc - chunk_start

    if groups_gpu is None:
        # Legacy path: upload fresh (backward compat for existing callers)
        groups_gpu = upload_k3plus_groups(groups_info, int(cp.cuda.Device()))

    result_counts = cp.zeros(chunk_size, dtype=cp.int32)

    kernel = get_cuda_kernel("count_k3plus_dense")
    block_size = 256
    grid = _grid_dims(chunk_size)

    kernel(
        grid,
        (block_size,),
        (
            bitvecs_gpu,
            groups_gpu["gpi"],
            groups_gpu["gpo"],
            groups_gpu["gs"],
            groups_gpu["gso"],
            groups_gpu["cp"],
            np.int64(n_u64s),
            np.int64(len(groups_info.cumulative_pairs) - 1),
            np.int64(chunk_start + chunk_size),
            result_counts,
            np.int64(chunk_start),
        ),
    )
    cp.cuda.Stream.null.synchronize()

    return result_counts  # stays in VRAM — no .get()




def build_k3plus_groups_from_flat(freq_flat, *, with_src_rows: bool = False):
    """Build prefix group arrays from flat (n_freq, k) numpy array.

    Uses Rust/Rayon parallel sort when available (10-100x faster, 9x less memory).
    Falls back to vectorized numpy if Rust extension not built (or predates
    ``with_src_rows`` — a one-time warning names the rebuild command).

    Args:
        freq_flat: numpy int32 array of shape (n_freq, k).
        with_src_rows: also fill ``suffix_src_rows`` (int64, parallel to
            ``suffixes``): the row of ``freq_flat`` each suffix slot came
            from. Off by default — 8 B per row that only the sparse-CSR
            path needs.

    Returns:
        K3PlusGroups namedtuple or None if no valid groups.
    """
    n, k = freq_flat.shape
    if n < 2:
        return None

    # ── Rust fast path: Rayon parallel sort, GIL-free ──────────────
    try:
        from et_miner.backends import get_rust_ext

        et_miner_rust = get_rust_ext()
        if hasattr(et_miner_rust, "build_k3plus_groups_from_flat"):
            arr = np.ascontiguousarray(freq_flat, dtype=np.int32)
            try:
                result = et_miner_rust.build_k3plus_groups_from_flat(arr, with_src_rows)
            except TypeError:  # pre-0.2.0 wheel: no with_src_rows argument
                result = None
                _warn_stale_rust_once("build_k3plus_groups_from_flat has no with_src_rows")
            else:
                if result is None:
                    return None
                if len(result) != 7:  # pre-0.2.0 wheel: 6-tuple
                    _warn_stale_rust_once("build_k3plus_groups_from_flat returned a 6-tuple")
                    result = None
            if result is not None:
                prefix_items, prefix_offsets, suffixes, suffix_offsets, cumulative_pairs, src_rows, total = result
                return K3PlusGroups(
                    prefix_items=prefix_items,
                    prefix_offsets=prefix_offsets,
                    suffixes=suffixes,
                    suffix_offsets=suffix_offsets,
                    cumulative_pairs=cumulative_pairs,
                    total_candidates=int(total),
                    groups=None,
                    suffix_src_rows=np.asarray(src_rows, dtype=np.int64) if with_src_rows else None,
                )
    except (ImportError, Exception):
        pass  # Fall through to numpy

    return _build_k3plus_groups_numpy(freq_flat, with_src_rows=with_src_rows)


def _build_k3plus_groups_numpy(freq_flat, *, with_src_rows: bool = False):
    """Vectorized numpy group builder (the no-Rust fallback; directly testable).

    Sorts by the FULL row (prefix columns, then suffix) like the Rust
    builder, so suffixes are ascending within every group.
    """
    n, k = freq_flat.shape
    if n < 2:
        return None
    prefixes = freq_flat[:, :-1]  # (n, k-1) — group key
    suffixes_col = freq_flat[:, -1]  # (n,) — suffix values

    # Lexicographic sort by prefix, then suffix. np.lexsort sorts by its LAST
    # key first, so the suffix goes first (least significant) and the prefix
    # columns follow in reverse order.
    sort_keys = (suffixes_col,) + tuple(prefixes[:, i] for i in range(prefixes.shape[1] - 1, -1, -1))
    order = np.lexsort(sort_keys)
    prefixes_sorted = prefixes[order]
    suffixes_sorted = suffixes_col[order]

    # Find group boundaries: where prefix changes
    diff = np.any(prefixes_sorted[1:] != prefixes_sorted[:-1], axis=1)
    boundary_mask = np.concatenate([[True], diff])
    boundaries = np.where(boundary_mask)[0]
    group_starts = boundaries
    group_ends = np.concatenate([boundaries[1:], [n]])
    group_sizes = group_ends - group_starts

    # Filter groups with >= 2 suffixes
    valid = group_sizes >= 2
    if not np.any(valid):
        return None

    valid_starts = group_starts[valid]
    valid_ends = group_ends[valid]
    valid_sizes = group_sizes[valid]
    n_groups = len(valid_starts)

    # Build flat prefix items array
    prefix_len = prefixes_sorted.shape[1]
    group_prefix_items = prefixes_sorted[valid_starts].ravel().astype(np.int32)
    group_prefix_offsets = np.arange(0, (n_groups + 1) * prefix_len, prefix_len, dtype=np.int64)

    # Build flat suffix array — vectorized gather, no Python loop
    total_suffixes = int(valid_sizes.sum())
    group_suffix_offsets = np.zeros(n_groups + 1, dtype=np.int64)
    np.cumsum(valid_sizes, out=group_suffix_offsets[1:])

    # Vectorized index construction: repeat group starts, add range offsets
    offsets_within = np.arange(total_suffixes, dtype=np.int64)
    group_ids = np.searchsorted(group_suffix_offsets[1:], offsets_within, side="right")
    src_indices = valid_starts[group_ids] + offsets_within - group_suffix_offsets[:-1][group_ids]
    group_suffixes = suffixes_sorted[src_indices].astype(np.int32)
    # Sorted position → original row of freq_flat, for the slots we kept.
    suffix_src_rows = order[src_indices].astype(np.int64) if with_src_rows else None

    # Cumulative pairs
    pairs_per_group = valid_sizes * (valid_sizes - 1) // 2
    cumulative_pairs = np.zeros(n_groups + 1, dtype=np.int64)
    np.cumsum(pairs_per_group, out=cumulative_pairs[1:])
    total_candidates = int(cumulative_pairs[-1])

    if total_candidates == 0:
        return None

    return K3PlusGroups(
        prefix_items=group_prefix_items,
        prefix_offsets=group_prefix_offsets,
        suffixes=group_suffixes,
        suffix_offsets=group_suffix_offsets,  # int64 — prevents overflow at >2B suffixes
        cumulative_pairs=cumulative_pairs,
        total_candidates=total_candidates,
        groups=None,  # not needed for dense counting
        suffix_src_rows=suffix_src_rows,
    )
