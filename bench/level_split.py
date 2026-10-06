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
    k2_dispatch    _k2_row_pairs (the K=2 dispatch's r from the host CSR)
    k2_input       the row-wise K=2 kernel's input: host CSR of frequent
                   positions per shard, and its upload
    count_rows     row-wise K=2 kernel launches (dense chunks)
    count_percand  per-candidate kernel launches (dense chunks)
    count_tiled    tiled kernel launches (dense chunks)
    count_fused    count_tiled_fused (one-GPU oversize groups; includes its filter)
    transition     convert_shards_to_csr (the dense-to-ESCO switch: tidsets
                   built from the bitvecs)
    count_csr      CSR kernel launches of an ESCO level (sparse chunks)
    compact        compact_written (the compacted reduce moving each GPU's
                   written entries to the front of its chunk array)
    reduce         reduce_sum_to_gpu0 (no-op on one GPU)
    filter         threshold_filter, threshold_filter_compacted (survivors
                   to the host)
    decode         decode_k2_pairs_flat, decode_k3plus_flat
    sort           _rows_sorted and the level-end lexsort
    free_prune     _prune_non_free_mask
    other          the level's time minus every phase above
    level          the level's time as level_callback reports it
    materialize    materialize_survivors (an ESCO level's survivors' tidsets),
                   which runs after the level's callback: outside ``level``
                   and ``other``, added to the level it materializes
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict

PHASES = (
    "group_build", "group_upload", "budget", "k2_dispatch", "k2_input", "count_rows", "count_percand", "count_tiled", "count_fused",
    "transition", "count_csr", "compact", "reduce", "filter", "decode", "sort", "free_prune",
)
#: Phases that run after the level's callback, credited to the level just closed.
AFTER_LEVEL = ("materialize",)

_MINER = "_apriori_row_split_multi_gpu"
_ROWS_LAUNCH = "_k2_rows_on_gpu"


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
        self._closed: int | None = None

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

    def _timed_after_level(self, phase: str, fn):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                if self._closed is not None:
                    level = self.levels[self._closed]
                    level[phase] = round(level[phase] + time.perf_counter() - t0, 6)

        return wrapper

    def _timed_dense_level(self, fn, phase: str | None = None):
        """Times each chunk launch inside run_chunked_dense_level, by kernel (or all as ``phase``)."""

        def wrapper(bitvecs_list, chunks, launch_chunk, *args, **kwargs):
            rows = launch_chunk.__name__ == _ROWS_LAUNCH

            def timed_launch(bitvec_gpu, device_id, chunk):
                t0 = time.perf_counter()
                try:
                    return launch_chunk(bitvec_gpu, device_id, chunk)
                finally:
                    kernel = "count_rows" if rows else "count_percand" if chunk.per_candidate else "count_tiled"
                    self._record(phase or kernel, t0)

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
        import et_miner.gpu.sparse_csr as sparse_csr

        for name in ("build_k3plus_groups_from_flat", "select_k3plus_groups"):
            self._patch(kernels, name, self._timed("group_build", getattr(kernels, name)))
        self._patch(kernels, "upload_k3plus_groups", self._timed("group_upload", kernels.upload_k3plus_groups))
        self._patch(kernels, "count_tiled_fused", self._timed("count_fused", kernels.count_tiled_fused))
        for name in ("decode_k2_pairs_flat", "decode_k3plus_flat"):
            self._patch(kernels, name, self._timed("decode", getattr(kernels, name)))
        self._patch(filter_mod, "threshold_filter", self._timed("filter", filter_mod.threshold_filter))
        self._patch(
            filter_mod, "threshold_filter_compacted", self._timed("filter", filter_mod.threshold_filter_compacted)
        )
        self._patch(filter_mod, "compact_written", self._timed("compact", filter_mod.compact_written))
        self._patch(nccl, "reduce_sum_to_gpu0", self._timed("reduce", nccl.reduce_sum_to_gpu0))
        self._patch(row_split, "compute_chunk_budget", self._timed("budget", row_split.compute_chunk_budget))
        self._patch(row_split, "_k2_row_pairs", self._timed("k2_dispatch", row_split._k2_row_pairs))
        self._patch(row_split, "_k2_row_shards", self._timed("k2_input", row_split._k2_row_shards))
        self._patch(row_split, "_upload_k2_rows", self._timed("k2_input", row_split._upload_k2_rows))
        self._patch(row_split, "convert_shards_to_csr", self._timed("transition", row_split.convert_shards_to_csr))
        self._patch(
            row_split, "materialize_survivors", self._timed_after_level("materialize", row_split.materialize_survivors)
        )
        self._patch(row_split, "run_chunked_dense_level", self._timed_dense_level(row_split.run_chunked_dense_level))
        self._patch(
            sparse_csr,
            "run_chunked_dense_level",
            self._timed_dense_level(sparse_csr.run_chunked_dense_level, "count_csr"),
        )
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
            phases.update(dict.fromkeys(AFTER_LEVEL, 0.0))
            self.levels[int(k)] = phases
            self._closed = int(k)
            self._pending = defaultdict(list)
            if level_cb is not None:
                level_cb(k, n_candidates, n_frequent, ms)

        return cb
