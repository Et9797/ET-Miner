"""Per-level time split of the row-split GPU miner, by wrapping its inner steps.

Wraps the functions the miner calls inside a level with wall-clock timers and
attributes the time to the level whose ``level_callback`` fires next. The
miner's code and results are unchanged; every wrapped call returns what the
original returns. The kernel wrappers synchronize before returning and the
filter copies survivors to the host, so wall time is device time for them.

Usage (inside a bench child, after warm-up, before the timed call):

    split = LevelSplit()
    split.install()
    try:
        apriori(..., level_callback=split.callback(level_cb))
    finally:
        split.uninstall()
    split.levels  # {k: {phase: seconds, ...}}

Phases:
    group_build    build_k3plus_groups_from_flat, select_k3plus_groups
    group_upload   upload_k3plus_groups (host-to-device group arrays)
    budget         compute_chunk_budget (VRAM probe)
    count_percand  per-candidate kernel launches (dense chunks)
    count_tiled    tiled kernel launches (dense chunks)
    count_fused    count_tiled_fused (one-GPU oversize groups; includes its filter)
    reduce         reduce_sum_to_gpu0 (no-op on one GPU)
    filter         threshold_filter (survivors to the host)
    decode         decode_k2_pairs_flat, decode_k3plus_flat
    sort           _rows_sorted and the level-end lexsort
    free_prune     _prune_non_free_mask
    other          the level's time minus every phase above
    level          the level's time as level_callback reports it
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict

PHASES = (
    "group_build", "group_upload", "budget", "count_percand", "count_tiled", "count_fused",
    "reduce", "filter", "decode", "sort", "free_prune",
)

_MINER = "_apriori_row_split_multi_gpu"


def _union_length(intervals: list[tuple[float, float]]) -> float:
    """Total length covered by possibly overlapping intervals (concurrent GPU threads)."""
    total = 0.0
    end = float("-inf")
    for lo, hi in sorted(intervals):
        if hi <= end:
            continue
        total += hi - max(lo, end)
        end = hi
    return total


class LevelSplit:
    """Installs and removes the timing wrappers; collects per-level phase seconds."""

    def __init__(self) -> None:
        self.levels: dict[int, dict[str, float]] = {}
        self._pending: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self._saved: list[tuple[object, str, object]] = []

    def _record(self, phase: str, t0: float) -> None:
        self._pending[phase].append((t0, time.perf_counter()))

    def _timed(self, phase: str, fn):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self._record(phase, t0)

        return wrapper

    def _timed_dense_level(self, fn):
        """Times each chunk launch inside run_chunked_dense_level, by kernel."""

        def wrapper(bitvecs_list, chunks, launch_chunk, *args, **kwargs):
            def timed_launch(bitvec_gpu, device_id, chunk):
                t0 = time.perf_counter()
                try:
                    return launch_chunk(bitvec_gpu, device_id, chunk)
                finally:
                    self._record("count_percand" if chunk.per_candidate else "count_tiled", t0)

            return fn(bitvecs_list, chunks, timed_launch, *args, **kwargs)

        return wrapper

    def _timed_lexsort(self, fn):
        def wrapper(*args, **kwargs):
            if sys._getframe(1).f_code.co_name != _MINER:
                return fn(*args, **kwargs)
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self._record("sort", t0)

        return wrapper

    def _patch(self, owner, name: str, replacement) -> None:
        self._saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def install(self) -> None:
        import numpy as np

        import et_miner.gpu.kernels as kernels
        import et_miner.gpu.kernels.filter as filter_mod
        import et_miner.gpu.nccl as nccl
        import et_miner.gpu.row_split as row_split

        for name in ("build_k3plus_groups_from_flat", "select_k3plus_groups"):
            self._patch(kernels, name, self._timed("group_build", getattr(kernels, name)))
        self._patch(kernels, "upload_k3plus_groups", self._timed("group_upload", kernels.upload_k3plus_groups))
        self._patch(kernels, "count_tiled_fused", self._timed("count_fused", kernels.count_tiled_fused))
        for name in ("decode_k2_pairs_flat", "decode_k3plus_flat"):
            self._patch(kernels, name, self._timed("decode", getattr(kernels, name)))
        self._patch(filter_mod, "threshold_filter", self._timed("filter", filter_mod.threshold_filter))
        self._patch(nccl, "reduce_sum_to_gpu0", self._timed("reduce", nccl.reduce_sum_to_gpu0))
        self._patch(row_split, "compute_chunk_budget", self._timed("budget", row_split.compute_chunk_budget))
        self._patch(row_split, "run_chunked_dense_level", self._timed_dense_level(row_split.run_chunked_dense_level))
        self._patch(row_split, "_rows_sorted", self._timed("sort", row_split._rows_sorted))
        self._patch(row_split, "_prune_non_free_mask", self._timed("free_prune", row_split._prune_non_free_mask))
        self._patch(np, "lexsort", self._timed_lexsort(np.lexsort))

    def uninstall(self) -> None:
        while self._saved:
            owner, name, original = self._saved.pop()
            setattr(owner, name, original)

    def callback(self, level_cb=None):
        """A level_callback that closes the level's split, then calls ``level_cb``."""

        def cb(k, n_candidates, n_frequent, ms):
            phases = {p: round(_union_length(self._pending.get(p, [])), 6) for p in PHASES}
            level_s = ms / 1000.0
            phases["other"] = round(level_s - sum(phases.values()), 6)
            phases["level"] = round(level_s, 6)
            self.levels[int(k)] = phases
            self._pending = defaultdict(list)
            if level_cb is not None:
                level_cb(k, n_candidates, n_frequent, ms)

        return cb
