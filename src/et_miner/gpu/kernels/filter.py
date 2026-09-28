"""Threshold filtering of dense GPU count arrays into survivor (index, count) pairs.

The row-split dense paths count every candidate into a chunk-sized int32
array in VRAM and then need the indices whose count clears the support
threshold. One boolean index or ``cp.where`` over the whole array allocates
temporaries proportional to the chunk (the documented K=8 OOM at 10.68B
candidates), and shipping the whole array to the host costs a full-array
D2H (80 GB at 10B candidates). ``threshold_filter`` runs ``cp.nonzero``
over slices of ``SLICE_ELEMS`` elements instead: the temporaries are bounded
per slice, only survivors cross PCIe, and the result is ordered by
construction. Measured on an RTX A4000 (CuPy 14.1), a 64M-element slice
takes 1.1 B/element of extra VRAM at a 1% pass rate and 13 B/element
(794 MiB) when every element survives, under the chunk budget's safety
margin of max(1 GiB, 4% of VRAM) (``row_split_chunks``). A slice that still
does not fit the device is filtered on the host (slice-wise D2H; up to
24 B/element of host RAM for the slice, its int64 indices, the gathered
counts and their int64 copy), so a degenerate chunk degrades to a slower
path instead of failing.

This was one of three implementations selected by ``ET_MINER_FILTER_IMPL``:
a two-launch ``compact_threshold`` kernel with a host sort, this sliced
filter, and the pre-2026 whole-array path. On two GPUs the three tied in
every regime (`bench/results/2026-09-28-consolidation-2gpu/`), so the
consolidation's rule 5 kept the smallest; the knob now raises
(``et_miner._env.reject_removed_knobs``).

Every call returns ``(indices, counts)`` as int64 NumPy arrays with indices
strictly ascending — the order the decode helpers and the sorted-by-row
closed-pruning contract in ``row_split`` depend on.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

#: Slice length: bounds the per-slice temporaries of ``cp.nonzero`` and the
#: gather (13 B/element with every element surviving: 794 MiB at 64M) and
#: the host fallback's staging (4 B/element).
SLICE_ELEMS = 64_000_000

#: Measured worst case per element of a slice on top of the counts (every
#: element surviving): the bool mask (1 B), the int64 survivor indices (8 B)
#: and the gathered int32 counts (4 B). ``SLICE_ELEMS`` times this stays
#: under the chunk budget's margin floor (``row_split_chunks.MARGIN_FLOOR_BYTES``).
WORST_CASE_BYTES_PER_ELEMENT = 13


def _empty_result() -> tuple[np.ndarray, np.ndarray]:
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)


def _filter_slice_on_device(view, threshold: int) -> tuple[np.ndarray, np.ndarray]:
    import cupy as cp

    mask = view >= threshold
    idx_gpu = cp.nonzero(mask)[0]
    del mask
    if len(idx_gpu) == 0:
        return _empty_result()
    return idx_gpu.get().astype(np.int64, copy=False), view[idx_gpu].get().astype(np.int64)


def _filter_slice_on_host(view, threshold: int) -> tuple[np.ndarray, np.ndarray]:
    """The fallback for a slice that does not fit the device: D2H, then NumPy."""
    chunk = view.get()
    idx = np.nonzero(chunk >= threshold)[0].astype(np.int64, copy=False)
    return idx, chunk[idx].astype(np.int64)


def threshold_filter(counts_gpu, threshold: int) -> tuple[np.ndarray, np.ndarray]:
    """Filter a dense GPU count array to (indices, counts) of survivors.

    Args:
        counts_gpu: CuPy integer array of per-candidate counts (int32 on the
            dense paths; any integer dtype works).
        threshold: Minimum count (inclusive — matches ``count >= min_count``).

    Returns:
        ``(indices, counts)`` int64 NumPy arrays, indices strictly ascending.
    """
    import cupy as cp

    n = len(counts_gpu)
    if n == 0:
        return _empty_result()
    threshold = int(threshold)
    idx_parts: list[np.ndarray] = []
    cnt_parts: list[np.ndarray] = []
    for start in range(0, n, SLICE_ELEMS):
        view = counts_gpu[start : min(start + SLICE_ELEMS, n)]
        try:
            idx, cnt = _filter_slice_on_device(view, threshold)
        except cp.cuda.memory.OutOfMemoryError:
            cp.get_default_memory_pool().free_all_blocks()
            logger.debug(
                f"threshold filter: the {len(view):,}-element slice at {start:,} does not fit the device "
                "— filtering it on the host"
            )
            idx, cnt = _filter_slice_on_host(view, threshold)
        if len(idx):
            idx_parts.append(idx + start)
            cnt_parts.append(cnt)
    if not idx_parts:
        return _empty_result()
    return np.concatenate(idx_parts), np.concatenate(cnt_parts)
