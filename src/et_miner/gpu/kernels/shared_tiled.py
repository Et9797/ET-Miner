"""Wrappers for the shared/tiled counting kernels (shared_tiled.cu).

Prefix sharing + suffix-pair tiling: one block per (tile A, tile B) pair
within a prefix group; the prefix AND is staged once per block per word and
up to T*T pair counts are served from shared memory. Output is
bit-identical to the legacy kernels (same triangular candidate indexing),
which the equivalence tests assert.

Constraints the callers must honor:
- the DENSE variant requires group-aligned candidate chunks (a tile-pair's
  candidates scatter across its whole group, so a partial group cannot be
  served); `plan_group_chunks` guarantees this and routes groups larger than
  a chunk to the per-candidate kernel, or, on one GPU, the row-split miner
  counts them with the FUSED variant;
- K <= 62 (same `s_items` cap as the legacy kernels — asserted here);
- the FUSED variant uses the overflow-safe protocol: the kernel counts
  past capacity with writes dropped, and the wrapper re-allocates to the
  exact reported size and re-runs — silent truncation is impossible.
"""

from __future__ import annotations

import numpy as np

from .loader import _assert_bitvecs, _assert_k_supported, _grid_dims, get_cuda_kernel

#: Suffixes per tile — fixed with the kernel's TILE_T (and blockDim 256).
TILE_T = 32

_BLOCK = 256


def compute_cumulative_tilepairs(suffix_offsets) -> np.ndarray:
    """Prefix sums of per-group tile-pair counts (0 for pairless groups)."""
    sizes = np.diff(np.asarray(suffix_offsets, dtype=np.int64))
    nt = (sizes + TILE_T - 1) // TILE_T
    tp = np.where(sizes >= 2, nt * (nt + 1) // 2, 0)
    out = np.zeros(len(sizes) + 1, dtype=np.int64)
    np.cumsum(tp, out=out[1:])
    return out


def _assert_k_cap(groups_info) -> None:
    """The groups_info-shaped form of loader._assert_k_supported.

    A prefix of length p yields candidates of length p+2, so the cap on the
    prefix is MAX_SUPPORTED_K - 2. Delegating keeps one rule rather than two
    copies of a constant.
    """
    gpo = np.asarray(groups_info.prefix_offsets, dtype=np.int64)
    if len(gpo) > 1:
        max_prefix = int(np.max(np.diff(gpo)))
        _assert_k_supported(max_prefix + 2, "shared/tiled kernel")


def _tilepair_range(groups_info, ctp: np.ndarray, chunk_start: int, chunk_end: int) -> tuple[int, int]:
    """Map a group-ALIGNED candidate range to its tile-pair range."""
    cp_arr = np.asarray(groups_info.cumulative_pairs, dtype=np.int64)
    g0 = int(np.searchsorted(cp_arr, chunk_start, side="left"))
    g1 = int(np.searchsorted(cp_arr, chunk_end, side="left"))
    if g0 >= len(cp_arr) or cp_arr[g0] != chunk_start or g1 >= len(cp_arr) or cp_arr[g1] != chunk_end:
        raise ValueError(
            f"shared dense kernel requires group-aligned chunks; "
            f"[{chunk_start}, {chunk_end}) does not land on group boundaries"
        )
    return int(ctp[g0]), int(ctp[g1])


def count_shared_tiled_allcounts(
    bitvecs_gpu, groups_info, n_u64s, chunk_start=0, chunk_size=None, groups_gpu=None
):
    """Dense counting via the tiled kernel — drop-in for
    count_k3plus_allcounts on group-aligned chunks (int32, chunk-relative,
    bit-identical candidate layout)."""
    import cupy as cp

    tc = groups_info.total_candidates
    if chunk_size is None:
        chunk_size = tc - chunk_start
    _assert_k_cap(groups_info)
    _assert_bitvecs(
        "count_shared_tiled_allcounts", bitvecs_gpu, **({} if groups_gpu is None else {"groups_gpu": groups_gpu["gpi"]})
    )
    device_id = bitvecs_gpu.device.id

    if groups_gpu is None:
        from .k3plus import upload_k3plus_groups

        groups_gpu = upload_k3plus_groups(groups_info, device_id)
    ctp = compute_cumulative_tilepairs(groups_info.suffix_offsets)
    tp0, tp1 = _tilepair_range(groups_info, ctp, chunk_start, chunk_start + chunk_size)

    with cp.cuda.Device(device_id):
        if "ctp" not in groups_gpu:
            groups_gpu["ctp"] = cp.array(ctp, dtype=cp.int64)
        result_counts = cp.zeros(chunk_size, dtype=cp.int32)
        if tp1 > tp0:
            kernel = get_cuda_kernel("count_shared_tiled_dense")
            kernel(
                _grid_dims(tp1 - tp0),
                (_BLOCK,),
                (
                    bitvecs_gpu,
                    groups_gpu["gpi"],
                    groups_gpu["gpo"],
                    groups_gpu["gs"],
                    groups_gpu["gso"],
                    groups_gpu["cp"],
                    groups_gpu["ctp"],
                    np.int64(n_u64s),
                    np.int64(len(groups_info.cumulative_pairs) - 1),
                    np.int64(tp0),
                    np.int64(tp1),
                    np.int64(chunk_start),
                    result_counts,
                ),
            )
            cp.cuda.Stream.null.synchronize()
    return result_counts


def k2_groups(freq_item_cols):
    """K=2 as one synthetic empty-prefix group over the frequent items."""
    from .k3plus import K3PlusGroups

    suffixes = np.asarray(freq_item_cols, dtype=np.int32)
    n = len(suffixes)
    return K3PlusGroups(
        prefix_items=np.empty(0, dtype=np.int32),
        prefix_offsets=np.array([0, 0], dtype=np.int64),
        suffixes=suffixes,
        suffix_offsets=np.array([0, n], dtype=np.int64),
        cumulative_pairs=np.array([0, n * (n - 1) // 2], dtype=np.int64),
        total_candidates=n * (n - 1) // 2,
        groups=None,
    )


def count_pairs_k2_shared(bitvecs_gpu, freq_item_cols, n_u64s):
    """Tiled dense K=2 over the WHOLE pair space (single chunk only — a
    partial pair range cannot be tile-served; a pair space chunked across
    several GPUs runs on the per-candidate kernel via the planner's
    use_legacy flag, and on one GPU count_tiled_fused counts it whole)."""
    return count_shared_tiled_allcounts(bitvecs_gpu, k2_groups(freq_item_cols), n_u64s)


def count_tiled_fused(bitvecs_gpu, groups_info, n_u64s, min_count, initial_capacity=1 << 24):
    """Fused tiled counting over whole groups: the survivors' candidate indices
    (ascending, into ``groups_info``'s candidate space) and int64 counts.

    Needs no dense count array, so a group of any size fits; the output grows
    with the survivors only (12 B each), re-run once at the exact size when
    ``initial_capacity`` is exceeded.
    """
    import cupy as cp

    from .k3plus import upload_k3plus_groups

    _assert_k_cap(groups_info)
    _assert_bitvecs("count_tiled_fused", bitvecs_gpu)
    ctp = compute_cumulative_tilepairs(groups_info.suffix_offsets)
    tp_total = int(ctp[-1])
    if tp_total == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    device_id = bitvecs_gpu.device.id
    groups_gpu = upload_k3plus_groups(groups_info, device_id)
    with cp.cuda.Device(device_id):
        return _run_fused(bitvecs_gpu, groups_info, groups_gpu, tp_total, n_u64s, min_count, initial_capacity)


def _run_fused(bitvecs_gpu, groups_info, groups_gpu, tp_total, n_u64s, min_count, initial_capacity):
    import cupy as cp

    kernel = get_cuda_kernel("count_shared_tiled_fused")
    capacity = max(1, min(int(initial_capacity), groups_info.total_candidates))
    out_idx = out_cnt = None
    n = 0
    for _attempt in range(2):
        n_results = cp.zeros(1, dtype=cp.uint64)
        out_idx = cp.empty(capacity, dtype=cp.int64)
        out_cnt = cp.empty(capacity, dtype=cp.int32)
        kernel(
            _grid_dims(tp_total),
            (_BLOCK,),
            (
                bitvecs_gpu,
                groups_gpu["gpi"],
                groups_gpu["gpo"],
                groups_gpu["gs"],
                groups_gpu["gso"],
                groups_gpu["cp"],
                groups_gpu["ctp"],
                np.int64(n_u64s),
                np.int64(len(groups_info.cumulative_pairs) - 1),
                np.int64(0),
                np.int64(tp_total),
                np.int32(min_count),
                out_idx,
                out_cnt,
                n_results,
                np.int64(capacity),
            ),
        )
        cp.cuda.Stream.null.synchronize()
        n = int(n_results.get()[0])
        if n <= capacity:
            break
        # Overflow-safe protocol: the kernel kept counting with writes
        # dropped; drop the buffers, re-run with the exact reported size.
        out_idx = out_cnt = None
        capacity = n
    else:
        raise RuntimeError(f"fused capacity did not converge (last n={n})")

    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    idx = out_idx[:n].get()
    cnt = out_cnt[:n].get().astype(np.int64)
    order = np.argsort(idx)
    return idx[order], cnt[order]
