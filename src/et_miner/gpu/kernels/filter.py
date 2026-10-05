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
takes 1.1 B/element of extra VRAM at a 1% pass rate and, when every
element survives, 12 B/element live (732 MiB) with 13 B/element held by
the pool (794 MiB), under the chunk budget's safety
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

The compacted multi-GPU reduce adds two steps around it: ``compact_written``
moves the entries a counting kernel wrote (the rest hold ``UNTOUCHED``) to the
front of a chunk's array, in order and in place, slice by slice with the same
per-slice temporaries; ``threshold_filter_compacted`` filters the summed
prefix and maps the survivors back to chunk positions through a packed bitmask
of the written entries (1 bit per chunk entry, kept on the reducing GPU only).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from loguru import logger

#: Slice length: bounds the per-slice temporaries of ``cp.nonzero`` and the
#: gather (13 B/element held by the pool with every element surviving:
#: 794 MiB at 64M) and the host fallback's staging (up to 24 B/element).
SLICE_ELEMS = 64_000_000

#: Measured worst case per element of a slice on top of the counts (every
#: element surviving): the bool mask (1 B), the int64 survivor indices (8 B)
#: and the gathered int32 counts (4 B). ``SLICE_ELEMS`` times this stays
#: under the chunk budget's margin floor (``row_split_chunks.MARGIN_FLOOR_BYTES``).
WORST_CASE_BYTES_PER_ELEMENT = 13

#: Fill of the dense entries a counting kernel did not write (the dense
#: wrappers' ``untouched=``); counts and inferred counts are never negative.
UNTOUCHED = -1


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


class Compacted(NamedTuple):
    """Where ``compact_written`` moved a chunk's written entries."""

    #: Written entries, now the array's first ``n`` elements in position order.
    n: int
    #: The chunk's length.
    size: int
    #: Slice length the compaction ran with.
    slice_elems: int
    #: Written entries per slice (int64, host).
    slice_counts: np.ndarray
    #: Packed bits (CuPy uint8, big-endian bit order) of the written positions, or None.
    mask: object


def _compact_slice_on_device(counts_gpu, start: int, length: int, out: int, mask) -> int:
    import cupy as cp

    view = counts_gpu[start : start + length]
    written = view != UNTOUCHED
    if mask is not None:
        mask[start // 8 : (start + length + 7) // 8] = cp.packbits(written)
    idx = cp.flatnonzero(written)
    del written
    m = len(idx)
    if m:
        counts_gpu[out : out + m] = view[idx]
    return m


def _compact_slice_on_host(counts_gpu, start: int, length: int, out: int, mask) -> int:
    """The fallback for a slice whose temporaries do not fit the device: D2H, NumPy, H2D."""
    host = counts_gpu[start : start + length].get()
    written = host != UNTOUCHED
    if mask is not None:
        mask[start // 8 : (start + length + 7) // 8].set(np.packbits(written))
    vals = host[written]
    if len(vals):
        counts_gpu[out : out + len(vals)].set(vals)
    return len(vals)


def compact_written(counts_gpu, *, keep_mask: bool) -> Compacted:
    """Move a chunk's written entries (not ``UNTOUCHED``) to its front, in order, in place.

    Slices run in ascending order and a slice's entries land at or before its
    own start, so no write reaches a slice that has not been read. Temporaries
    are one slice's (at most ``WORST_CASE_BYTES_PER_ELEMENT`` per element, plus
    the slice's packed mask); a slice that does not fit is compacted on the
    host. ``keep_mask`` keeps the written positions as packed bits (1 bit per
    entry) for ``threshold_filter_compacted``. Must run under the array's device.
    """
    import cupy as cp

    slice_elems = SLICE_ELEMS
    if slice_elems % 8:
        raise ValueError(f"SLICE_ELEMS must be a multiple of 8 for the packed mask, got {slice_elems}")
    n = len(counts_gpu)
    mask = cp.empty((n + 7) // 8, dtype=cp.uint8) if keep_mask else None
    slice_counts = np.zeros(-(-n // slice_elems), dtype=np.int64)
    out = 0
    for s, start in enumerate(range(0, n, slice_elems)):
        length = min(slice_elems, n - start)
        try:
            m = _compact_slice_on_device(counts_gpu, start, length, out, mask)
        except cp.cuda.memory.OutOfMemoryError:
            # Nothing was moved yet: the device path writes only after its gather.
            cp.get_default_memory_pool().free_all_blocks()
            logger.debug(f"compaction: the {length:,}-element slice at {start:,} does not fit the device — on the host")
            m = _compact_slice_on_host(counts_gpu, start, length, out, mask)
        slice_counts[s] = m
        out += m
    return Compacted(out, n, slice_elems, slice_counts, mask)


def _pick_written(mask, start: int, length: int, picked: np.ndarray) -> np.ndarray:
    """Slice positions of a slice's ``picked``-th written entries (on the host when the device is full)."""
    import cupy as cp

    packed = mask[start // 8 : (start + length + 7) // 8]
    try:
        return cp.flatnonzero(cp.unpackbits(packed)[:length])[cp.asarray(picked)].get()
    except cp.cuda.memory.OutOfMemoryError:
        cp.get_default_memory_pool().free_all_blocks()
        return np.flatnonzero(np.unpackbits(packed.get())[:length])[picked]


def threshold_filter_compacted(counts_gpu, compacted: Compacted, threshold: int) -> tuple[np.ndarray, np.ndarray]:
    """``threshold_filter`` of a compacted chunk; the survivors' indices are chunk positions.

    ``counts_gpu[:compacted.n]`` holds the written entries (summed across
    devices) and ``compacted`` is what ``compact_written(keep_mask=True)``
    returned for this array. Must run under the array's device.
    """
    if compacted.mask is None:
        raise ValueError("threshold_filter_compacted needs the mask of compact_written(keep_mask=True)")
    idx, cnt = threshold_filter(counts_gpu[: compacted.n], threshold)
    if not len(idx):
        return idx, cnt
    step = compacted.slice_elems
    starts = np.zeros(len(compacted.slice_counts) + 1, dtype=np.int64)
    np.cumsum(compacted.slice_counts, out=starts[1:])
    bounds = np.searchsorted(idx, starts)
    pos = np.empty_like(idx)
    for s in range(len(compacted.slice_counts)):
        lo, hi = int(bounds[s]), int(bounds[s + 1])
        if lo == hi:
            continue
        start = s * step
        pos[lo:hi] = _pick_written(compacted.mask, start, min(step, compacted.size - start), idx[lo:hi] - starts[s])
        pos[lo:hi] += start
    return pos, cnt
