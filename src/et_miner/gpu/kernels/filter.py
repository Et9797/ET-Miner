"""Threshold filtering of dense GPU count arrays into survivor (index, count) pairs.

The row-split dense paths count every candidate into a chunk-sized int32
array in VRAM and then need the indices whose count clears the support
threshold. Doing that with one boolean-index/`cp.where` over the full array
allocates a hidden int64 prefix-sum (~8 B/element — the documented K=8 OOM),
and shipping the whole array to the CPU instead costs a full-array pageable
D2H (80 GB at 10B candidates). This module holds three interchangeable
implementations, selected by ``ET_MINER_FILTER_IMPL`` (see ``et_miner._env``)
or per call:

- ``compact`` (default): the ``compact_threshold`` CUDA kernel. Two
  launches — an exact count-only pass, then an exact-capacity scatter into
  structure-of-arrays survivor buffers (12 B/survivor) — so peak extra VRAM
  is 12 B × n_survivors and only survivors cross PCIe. Atomic append is
  unordered, so survivors are sorted ascending on the host (~32-48 B ×
  n_survivors of host RAM for argsort + gather); when that host headroom
  cannot be verified or is insufficient, the call falls back to the sliced
  CPU valve *before* any sort is attempted.
- ``cupy``: sliced ``cp.nonzero`` — ≤64M-element slices bound the hidden
  prefix-sum temporary at ~512 MB/slice; ordered by construction. Pure
  CuPy escape hatch and the ordering cross-check for ``compact``.
- ``cpu``: the pre-2026 ``safe_threshold_filter`` behavior, verbatim, as
  the A/B baseline (full-array D2H above ``max_gpu_elements``, whole-array
  ``cp.where`` below it).

Every implementation returns ``(indices, counts)`` as int64 NumPy arrays
with indices strictly ascending — the order the decode helpers and the
sorted-by-row closed-pruning contract in ``row_split`` depend on.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from et_miner import _env

from .loader import get_cuda_kernel

#: Slice length for the sliced implementations. Bounds the hidden int64
#: prefix-sum temporary of `cp.nonzero` at ~512 MB and the valve's host
#: staging at ~256 MB (int32) per slice.
SLICE_ELEMS = 64_000_000

#: Host bytes per survivor the compact path's post-D2H sort needs at peak:
#: 12 B survivor pair + 8 B argsort index array + 12 B gathered copies,
#: rounded up for slack.
HOST_SORT_BYTES_PER_SURVIVOR = 48

_COMPACT_BLOCK = 256
_COMPACT_GRID = 4096  # grid-stride kernel: fixed grid, no 2^31 grid math


def _empty_result() -> tuple[np.ndarray, np.ndarray]:
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)


def _host_ram_available() -> int | None:
    """Available host RAM in bytes, or None when psutil is not installed."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _filter_cpu_sliced(counts_gpu, threshold: int) -> tuple[np.ndarray, np.ndarray]:
    """The valve: slice-wise D2H + host filter. Bounded, ordered, never OOMs."""
    idx_parts: list[np.ndarray] = []
    cnt_parts: list[np.ndarray] = []
    n = len(counts_gpu)
    for start in range(0, n, SLICE_ELEMS):
        chunk = counts_gpu[start : min(start + SLICE_ELEMS, n)].get()
        mask = chunk >= threshold
        idx = np.nonzero(mask)[0]
        if len(idx):
            idx_parts.append(idx.astype(np.int64) + start)
            cnt_parts.append(chunk[idx].astype(np.int64))
        del chunk, mask, idx
    if not idx_parts:
        return _empty_result()
    return np.concatenate(idx_parts), np.concatenate(cnt_parts)


def _filter_cupy_sliced(counts_gpu, threshold: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliced on-GPU cp.nonzero: bounded temporaries, ordered by construction."""
    import cupy as cp

    idx_parts: list[np.ndarray] = []
    cnt_parts: list[np.ndarray] = []
    n = len(counts_gpu)
    for start in range(0, n, SLICE_ELEMS):
        view = counts_gpu[start : min(start + SLICE_ELEMS, n)]
        mask = view >= threshold
        idx_gpu = cp.nonzero(mask)[0]
        if len(idx_gpu):
            idx_parts.append(idx_gpu.get().astype(np.int64) + start)
            cnt_parts.append(view[idx_gpu].get().astype(np.int64))
        del mask, idx_gpu
    if not idx_parts:
        return _empty_result()
    return np.concatenate(idx_parts), np.concatenate(cnt_parts)


def _filter_cpu_legacy(counts_gpu, threshold: int, max_gpu_elements: int) -> tuple[np.ndarray, np.ndarray]:
    """Pre-2026 safe_threshold_filter behavior, verbatim — the A/B baseline."""
    import cupy as cp

    n = len(counts_gpu)
    if n > max_gpu_elements:
        counts_cpu = counts_gpu.get()
        del counts_gpu
        cp.get_default_memory_pool().free_all_blocks()
        mask = counts_cpu >= threshold
        indices = np.where(mask)[0]
        filtered_counts = counts_cpu[indices]
        return indices.astype(np.int64), filtered_counts.astype(np.int64)
    mask = counts_gpu >= threshold
    indices_gpu = cp.where(mask)[0]
    indices = indices_gpu.get()
    filtered_counts = counts_gpu[indices_gpu].get()
    return indices.astype(np.int64), filtered_counts.astype(np.int64)


def _filter_compact(counts_gpu, threshold: int) -> tuple[np.ndarray, np.ndarray]:
    """Two-launch compact_threshold kernel; falls back to the valve when the
    survivor buffers or the host sort workspace would not fit."""
    import cupy as cp

    n = len(counts_gpu)
    kernel = get_cuda_kernel("compact_threshold")
    grid = (min(_COMPACT_GRID, max(1, (n + _COMPACT_BLOCK - 1) // _COMPACT_BLOCK)),)
    n_out = cp.zeros(1, dtype=cp.uint64)
    dummy_idx = cp.empty(1, dtype=cp.int64)
    dummy_cnt = cp.empty(1, dtype=cp.int32)

    # Pass 1: count-only (capacity=0) — exact survivor count, zero writes.
    kernel(
        grid,
        (_COMPACT_BLOCK,),
        (counts_gpu, np.int64(n), np.int32(threshold), dummy_idx, dummy_cnt, n_out, np.int64(0)),
    )
    cp.cuda.Stream.null.synchronize()
    n_surv = int(n_out.get()[0])
    if n_surv == 0:
        return _empty_result()

    # Feasibility: 12 B × n_surv survivor buffers in VRAM, ~48 B × n_surv
    # host sort workspace. Unknown host RAM (no psutil) is treated
    # conservatively: only allow sorts small enough to be safe anywhere.
    free_vram, _ = cp.cuda.Device().mem_info
    need_vram = n_surv * 12
    host_avail = _host_ram_available()
    need_host = n_surv * HOST_SORT_BYTES_PER_SURVIVOR
    host_ok = (host_avail is not None and need_host < host_avail * 0.8) or (
        host_avail is None and n_surv <= SLICE_ELEMS
    )
    if need_vram > free_vram * 0.9 or not host_ok:
        logger.warning(
            f"  compact filter: {n_surv:,} survivors exceed workspace "
            f"(VRAM need {need_vram / 1e9:.1f} GB / free {free_vram / 1e9:.1f} GB, "
            f"host need {need_host / 1e9:.1f} GB) — using sliced CPU valve"
        )
        return _filter_cpu_sliced(counts_gpu, threshold)

    # Pass 2: exact-capacity scatter.
    out_idx = cp.empty(n_surv, dtype=cp.int64)
    out_cnt = cp.empty(n_surv, dtype=cp.int32)
    n_out[:] = 0
    kernel(
        grid,
        (_COMPACT_BLOCK,),
        (counts_gpu, np.int64(n), np.int32(threshold), out_idx, out_cnt, n_out, np.int64(n_surv)),
    )
    cp.cuda.Stream.null.synchronize()
    n_surv2 = int(n_out.get()[0])
    if n_surv2 != n_surv:
        raise RuntimeError(f"compact_threshold pass disagreement: {n_surv} then {n_surv2} survivors")

    indices = out_idx.get()
    counts = out_cnt.get()
    del out_idx, out_cnt, n_out, dummy_idx, dummy_cnt

    # Atomic append is unordered — restore the ascending-index contract.
    order = np.argsort(indices)
    return indices[order], counts[order].astype(np.int64)


def compact_threshold_filter(counts_gpu, threshold: int, impl: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Filter a dense GPU count array to (indices, counts) of survivors.

    Args:
        counts_gpu: CuPy int32 (or int64) array of per-candidate counts.
        threshold: Minimum count (inclusive — matches ``count >= min_count``).
        impl: Override the ``ET_MINER_FILTER_IMPL`` selection for this call.

    Returns:
        ``(indices, counts)`` int64 NumPy arrays, indices strictly ascending.
    """
    import cupy as cp

    chosen = impl or _env.filter_impl()
    if chosen not in ("compact", "cupy", "cpu"):
        raise ValueError(f"ET_MINER_FILTER_IMPL must be 'compact', 'cupy', or 'cpu' — got {chosen!r}")
    if len(counts_gpu) == 0:
        return _empty_result()
    if chosen == "compact" and counts_gpu.dtype != cp.int32:
        # The kernel is int32-only (the dense counting path emits int32);
        # other dtypes route to the sliced CuPy implementation.
        logger.debug(f"compact filter: dtype {counts_gpu.dtype} not int32 — using sliced cupy impl")
        chosen = "cupy"

    if chosen == "compact":
        return _filter_compact(counts_gpu, int(threshold))
    if chosen == "cupy":
        return _filter_cupy_sliced(counts_gpu, int(threshold))
    return _filter_cpu_legacy(counts_gpu, int(threshold), max_gpu_elements=100_000_000)
