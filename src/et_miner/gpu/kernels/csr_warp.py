"""Warp-cooperative CSR tidset intersection kernels (the sparse-CSR mining path).

A *shard* is a device-resident CSR of sorted-unique int32 transaction ids per
frequent itemset: ``offsets_gpu`` int64 ``(n_itemsets + 1)`` and
``indices_gpu`` int32. Candidates are enumerated in-kernel from the resident
group arrays (``upload_k3plus_groups(..., with_src_rows=True)``): the shared
decode (``_src/_decode_common.cu``) maps a candidate index to its two suffix
slots and ``suffix_src_rows`` maps those to shard rows — exactly the
enumeration ``decode.py::decode_k3plus_flat`` performs on the host, so
survivor ``i`` of a count pass is decoded row ``i`` and materialized row ``i``.

One warp per candidate, ``blockDim.x == 256`` (8 candidates per block).
Counts are int32 (bounded by ``n_transactions``, guarded ``< 2**31`` by the
miners).
"""

from __future__ import annotations

import numpy as np

from .loader import _grid_dims, get_cuda_kernel

#: Candidates per 256-thread block (8 warps) — coupled to ``#define CANDS_PER_BLOCK``
#: in ``_src/csr_warp.cu``.
CANDS_PER_BLOCK = 8
_BLOCK = (256,)


def _grid_for(n: int):
    return _grid_dims((int(n) + CANDS_PER_BLOCK - 1) // CANDS_PER_BLOCK)


def _group_args(groups_gpu):
    """(cp, gso, gsr, n_groups) from an ``upload_k3plus_groups`` dict."""
    gsr = groups_gpu.get("gsr")
    if gsr is None:
        raise ValueError(
            "CSR kernels need group arrays uploaded with upload_k3plus_groups(..., with_src_rows=True) "
            "from groups built with with_src_rows=True"
        )
    cp_arr = groups_gpu["cp"]
    return cp_arr, groups_gpu["gso"], gsr, np.int64(len(cp_arr) - 1)


def _check_shard(offsets_gpu, indices_gpu):
    import cupy as cp

    if offsets_gpu.dtype != cp.int64 or indices_gpu.dtype != cp.int32:
        raise TypeError(f"CSR shard must be int64 offsets + int32 indices, got {offsets_gpu.dtype}/{indices_gpu.dtype}")


def _prepare_ids(cand_ids_gpu, groups_gpu):
    """Contiguous int64 candidate ids, range-checked against total_candidates."""
    import cupy as cp

    ids = cp.ascontiguousarray(cand_ids_gpu, dtype=cp.int64)
    n = int(ids.size)
    tc = groups_gpu.get("tc")
    if n > 0 and tc is not None:
        lo, hi = int(ids.min()), int(ids.max())
        if lo < 0 or hi >= int(tc):
            raise ValueError(f"candidate ids must lie in [0, {int(tc)}), got [{lo}, {hi}]")
    return ids, n


def count_csr_range(offsets_gpu, indices_gpu, groups_gpu, chunk_start: int, chunk_size: int):
    """Partial intersection counts on this shard for candidates
    ``[chunk_start, chunk_start + chunk_size)`` — CuPy int32 ``(chunk_size,)``."""
    import cupy as cp

    _check_shard(offsets_gpu, indices_gpu)
    n = int(chunk_size)
    out = cp.zeros(max(n, 0), dtype=cp.int32)
    if n <= 0:
        return out
    tc = groups_gpu.get("tc")
    if tc is not None and (int(chunk_start) < 0 or int(chunk_start) + n > int(tc)):
        raise ValueError(f"chunk [{chunk_start}, {int(chunk_start) + n}) exceeds total_candidates={int(tc)}")
    cp_arr, gso, gsr, n_groups = _group_args(groups_gpu)
    get_cuda_kernel("csr_count_range")(
        _grid_for(n),
        _BLOCK,
        (indices_gpu, offsets_gpu, cp_arr, gso, gsr, n_groups, np.int64(chunk_start), np.int64(n), out),
    )
    cp.cuda.Stream.null.synchronize()
    return out


def count_csr_gather(offsets_gpu, indices_gpu, groups_gpu, cand_ids_gpu):
    """Intersection counts on this shard for explicit candidate ids — CuPy int32 ``(n,)``."""
    import cupy as cp

    _check_shard(offsets_gpu, indices_gpu)
    ids, n = _prepare_ids(cand_ids_gpu, groups_gpu)
    out = cp.zeros(n, dtype=cp.int32)
    if n == 0:
        return out
    cp_arr, gso, gsr, n_groups = _group_args(groups_gpu)
    get_cuda_kernel("csr_count_gather")(
        _grid_for(n),
        _BLOCK,
        (indices_gpu, offsets_gpu, cp_arr, gso, gsr, n_groups, ids, np.int64(n), out),
    )
    cp.cuda.Stream.null.synchronize()
    return out


def write_csr_gather(offsets_gpu, indices_gpu, groups_gpu, cand_ids_gpu, out_offsets_gpu, out_indices_gpu):
    """Write the sorted intersections of explicit candidate ids into a new CSR.

    ``out_offsets_gpu`` (int64, ``n + 1``) must be the exclusive scan of
    ``count_csr_gather`` over the same ids on the same shard, and
    ``out_indices_gpu`` (int32) must hold ``out_offsets_gpu[-1]`` entries.
    """
    import cupy as cp

    _check_shard(offsets_gpu, indices_gpu)
    ids, n = _prepare_ids(cand_ids_gpu, groups_gpu)
    if n == 0:
        return
    if out_offsets_gpu.dtype != cp.int64 or int(out_offsets_gpu.size) != n + 1:
        raise ValueError(f"out_offsets must be int64 of length n+1={n + 1}, got {out_offsets_gpu.dtype}/{out_offsets_gpu.size}")
    if out_indices_gpu.dtype != cp.int32:
        raise TypeError(f"out_indices must be int32, got {out_indices_gpu.dtype}")
    cp_arr, gso, gsr, n_groups = _group_args(groups_gpu)
    get_cuda_kernel("csr_write_gather")(
        _grid_for(n),
        _BLOCK,
        (indices_gpu, offsets_gpu, cp_arr, gso, gsr, n_groups, ids, np.int64(n), out_offsets_gpu, out_indices_gpu),
    )
    cp.cuda.Stream.null.synchronize()
