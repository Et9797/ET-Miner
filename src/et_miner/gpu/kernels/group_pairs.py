"""Wrapper for the group kernel (group_pairs.cu): one block per prefix group.

A block classifies its group's pairs with the subset test, lists the pairs to
count, stages the prefix AND and the used suffix rows one word tile at a time,
and counts only the listed pairs. It writes the entries the per-candidate
kernel writes, in the same chunk-relative layout.

Constraints the callers must honor:
- every group has at most ``GROUP_MAX_SUFFIXES`` suffixes (checked here);
- candidate chunks are group-aligned, as for the tiled kernel;
- K <= 62 (the shared prefix cache of the other K>=3 kernels).
"""

from __future__ import annotations

import numpy as np

from .loader import _assert_bitvecs, _grid_dims, get_cuda_kernel

#: Suffixes a prefix group may have — fixed with the kernel's GP_MAX_SUFFIXES.
GROUP_MAX_SUFFIXES = 64

_BLOCK = 256


def _group_range(groups_info, chunk_start: int, chunk_end: int) -> tuple[int, int]:
    """Map a group-aligned candidate range to its group range."""
    cp_arr = np.asarray(groups_info.cumulative_pairs, dtype=np.int64)
    g0 = int(np.searchsorted(cp_arr, chunk_start, side="left"))
    g1 = int(np.searchsorted(cp_arr, chunk_end, side="left"))
    if g0 >= len(cp_arr) or cp_arr[g0] != chunk_start or g1 >= len(cp_arr) or cp_arr[g1] != chunk_end:
        raise ValueError(
            f"group kernel requires group-aligned chunks; [{chunk_start}, {chunk_end}) does not land on group boundaries"
        )
    return g0, g1


def count_group_pairs(
    bitvecs_gpu, groups_info, n_u64s, chunk_start=0, chunk_size=None, groups_gpu=None, index=None, untouched=0
):
    """Dense counting via the group kernel — drop-in for count_k3plus_per_candidate
    on group-aligned chunks of groups with at most ``GROUP_MAX_SUFFIXES`` suffixes
    (int32, chunk-relative, the same candidate layout, ``index`` and ``untouched``
    semantics, and the same written entries)."""
    import cupy as cp

    from .k3plus import upload_k3plus_groups
    from .shared_tiled import _assert_k_cap
    from .subset_index import kernel_args

    tc = groups_info.total_candidates
    if chunk_size is None:
        chunk_size = tc - chunk_start
    _assert_k_cap(groups_info)
    _assert_bitvecs(
        "count_group_pairs", bitvecs_gpu, **({} if groups_gpu is None else {"groups_gpu": groups_gpu["gpi"]})
    )
    g0, g1 = _group_range(groups_info, chunk_start, chunk_start + chunk_size)
    sizes = np.diff(np.asarray(groups_info.suffix_offsets, dtype=np.int64)[g0 : g1 + 1])
    if len(sizes) and int(sizes.max()) > GROUP_MAX_SUFFIXES:
        raise ValueError(f"group kernel takes groups of at most {GROUP_MAX_SUFFIXES} suffixes; got {int(sizes.max())}")
    device_id = bitvecs_gpu.device.id
    if groups_gpu is None:
        groups_gpu = upload_k3plus_groups(groups_info, device_id)

    with cp.cuda.Device(device_id):
        result_counts = (
            cp.zeros(chunk_size, dtype=cp.int32) if untouched == 0 else cp.full(chunk_size, untouched, dtype=cp.int32)
        )
        if g1 > g0:
            get_cuda_kernel("count_group_pairs")(
                _grid_dims(g1 - g0),
                (_BLOCK,),
                (
                    bitvecs_gpu,
                    groups_gpu["gpi"],
                    groups_gpu["gpo"],
                    groups_gpu["gs"],
                    groups_gpu["gso"],
                    groups_gpu["cp"],
                    np.int64(n_u64s),
                    np.int64(g0),
                    np.int64(g1),
                    np.int64(chunk_start),
                    result_counts,
                    *kernel_args(index, device_id),
                ),
            )
            cp.cuda.Stream.null.synchronize()
    return result_counts
