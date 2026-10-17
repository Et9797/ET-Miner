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
    split.span    # {phase: seconds} before K=1 and after the last level

Phases:
    group_build    build_k3plus_groups_from_flat, select_k3plus_groups
    group_upload   upload_k3plus_groups (host-to-device group arrays)
    budget         compute_chunk_budget (VRAM probe)
    k2_dispatch    _k2_row_pairs (the K=2 dispatch's r from the host CSR),
                   or DeviceRows.pairs (from the device entries)
    k2_input       the row-wise K=2 kernel's input: host CSR of frequent
                   positions per shard, and its upload; or DeviceRows.upload
    k3_graph       build_k3_graph (row-wise K=3's pair graph and bitmap on
                   each device)
    count_rows     row-wise K=2 and K=3 kernel launches (dense chunks)
    count_percand  per-candidate kernel launches (dense chunks)
    count_group    group kernel launches (dense chunks)
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
    decode         decode_k2_pairs_flat, decode_k3plus_flat, decode_k3_slots
    sort           _rows_sorted and the level-end lexsort
    free_prune     _prune_non_free_mask
    other          the level's time minus every phase above
    level          the level's time as level_callback reports it
    materialize    materialize_survivors (an ESCO level's survivors' tidsets),
                   which runs after the level's callback: outside ``level``
                   and ``other``, added to the level it materializes

Span phases (``span``, transactions input on the row-split route; the call
starts at ``start()`` and ends at ``finish()``):
    input_count    the transaction count (first Polars collect of
                   core.matrix._build_csr_from_transactions)
    input_k1       the host K=1: explode, group_by, threshold (second collect)
    input_explode  explode and join onto the frequent items (third collect)
    input_csr      the rest of _build_csr_from_transactions: COO to CSR
    input_widen    the CSR's int64 copies in build_bitvecs_row_split
    input_h2d      cupy.asarray of each shard's CSR in build_bitvecs_gpu
    input_bitvec   the rest of build_bitvecs_gpu: allocation, kernel, sync
    input_dev_chunks  the device input layer's chunk entries
                   (input_layer._chunk_entries: Arrow views, or Polars'
                   explode for a chunk with nulls)
    input_dev_copy the device layer's ids into pinned buffers and onto the
                   devices (_Uploader.copy and .finish)
    input_dev_k1   the device layer's occurrence bincount (_occurrences)
    input_dev_bitvec  the device layer's lookup upload, bitvector
                   allocation and ids_to_bitvec (_scatter_bitvecs)
    input_other    from the call to K=1's start, minus the input phases
                   (argument checks, the column's collect, NCCL setup,
                   dispatch)
    input          from the call to K=1's start
    frame          _build_deferred_frame and _as_item_dtype (the result frame)
    tail_other     from the last level's callback to the return, minus frame
    tail           from the last level's callback to the return
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict

PHASES = (
    "group_build", "group_upload", "budget", "k2_dispatch", "k2_input", "k3_graph", "count_rows", "count_percand", "count_group", "count_tiled",
    "count_fused",
    "transition", "count_csr", "compact", "reduce", "filter", "decode", "sort", "free_prune",
)
#: Phases that run after the level's callback, credited to the level just closed.
AFTER_LEVEL = ("materialize",)

SPAN_PHASES = (
    "input_count", "input_k1", "input_explode", "input_csr", "input_widen", "input_h2d", "input_bitvec",
    "input_dev_chunks", "input_dev_copy", "input_dev_k1", "input_dev_bitvec",
    "input_other", "input", "frame", "tail_other", "tail",
)
_DEVICE_SPANS = ("input_dev_chunks", "input_dev_copy", "input_dev_k1", "input_dev_bitvec")
_COLLECTS = ("input_count", "input_k1", "input_explode")

_MINER = "_apriori_row_split_multi_gpu"
_ROWS_LAUNCHES = ("_k2_rows_on_gpu", "_k3_rows_on_gpu")


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


def _kernel_phase(chunk, rows: bool) -> str:
    if rows:
        return "count_rows"
    if chunk.per_candidate:
        return "count_percand"
    return "count_group" if chunk.group else "count_tiled"


class LevelSplit:
    """Installs and removes the timing wrappers; collects per-level phase seconds."""

    def __init__(self) -> None:
        self.levels: dict[int, dict[str, float]] = {}
        self._pending: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self._saved: list[tuple[object, str, object]] = []
        self._closed: int | None = None
        self.span: dict[str, float] = {}
        self._span: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self._t_start: float | None = None
        self._first_level_start: float | None = None
        self._last_level_end: float | None = None

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
            rows = launch_chunk.__name__ in _ROWS_LAUNCHES

            def timed_launch(bitvec_gpu, device_id, chunk):
                t0 = time.perf_counter()
                try:
                    return launch_chunk(bitvec_gpu, device_id, chunk)
                finally:
                    self._record(phase or _kernel_phase(chunk, rows), t0)

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

    def _timed_span(self, phase: str, fn):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self._span[phase].append((t0, time.perf_counter()))

        return wrapper

    def _timed_csr_build(self, fn):
        """Times _build_csr_from_transactions and, inside it, each Polars collect in call order."""
        import polars as pl

        def wrapper(*args, **kwargs):
            original = pl.LazyFrame.collect
            calls = []

            def collect(lf, *a, **kw):
                phase = _COLLECTS[min(len(calls), len(_COLLECTS) - 1)]
                calls.append(phase)
                t0 = time.perf_counter()
                try:
                    return original(lf, *a, **kw)
                finally:
                    self._span[phase].append((t0, time.perf_counter()))

            pl.LazyFrame.collect = collect
            try:
                return self._timed_span("_csr_build", fn)(*args, **kwargs)
            finally:
                pl.LazyFrame.collect = original

        return wrapper

    def _timed_bitvec_gpu(self, fn):
        """Times build_bitvecs_gpu and, inside it, cupy.asarray (the host-to-device copies)."""
        import cupy as cp

        def wrapper(*args, **kwargs):
            original = cp.asarray
            cp.asarray = self._timed_span("input_h2d", original)
            try:
                return self._timed_span("_bitvec_gpu", fn)(*args, **kwargs)
            finally:
                cp.asarray = original

        return wrapper

    def start(self) -> None:
        """Marks the start of the timed call."""
        self._t_start = time.perf_counter()

    def finish(self) -> None:
        """Marks the end of the timed call and computes ``span``."""
        t_end = time.perf_counter()
        u = {p: _union_length(v) for p, v in self._span.items()}
        span = {p: u.get(p, 0.0) for p in ("input_count", "input_k1", "input_explode", "input_h2d", *_DEVICE_SPANS)}
        span["input_csr"] = u.get("_csr_build", 0.0) - sum(u.get(p, 0.0) for p in _COLLECTS)
        span["input_widen"] = u.get("_bitvec_split", 0.0) - u.get("_bitvec_arrays", 0.0)
        span["input_bitvec"] = u.get("_bitvec_gpu", 0.0) - u.get("input_h2d", 0.0)
        if self._t_start is not None and self._first_level_start is not None:
            span["input"] = self._first_level_start - self._t_start
            span["input_other"] = span["input"] - sum(
                span[p] for p in ("input_count", "input_k1", "input_explode", "input_csr", "input_widen",
                                  "input_h2d", "input_bitvec", *_DEVICE_SPANS)
            )
        span["frame"] = u.get("frame", 0.0)
        if self._last_level_end is not None:
            span["tail"] = t_end - self._last_level_end
            span["tail_other"] = span["tail"] - span["frame"]
        self.span = {p: round(span[p], 6) for p in SPAN_PHASES if p in span}

    def _patch(self, owner, name: str, replacement) -> None:
        self._saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def install(self) -> None:
        import numpy as np

        import importlib

        import et_miner.core.matrix as matrix
        import et_miner.gpu.csr_bitvec as csr_bitvec
        import et_miner.gpu.input_layer as input_layer
        import et_miner.gpu.kernels as kernels
        import et_miner.gpu.kernels.filter as filter_mod
        import et_miner.gpu.nccl as nccl
        import et_miner.gpu.row_split as row_split
        import et_miner.gpu.sparse_csr as sparse_csr

        for name in ("build_k3plus_groups_from_flat", "select_k3plus_groups"):
            self._patch(kernels, name, self._timed("group_build", getattr(kernels, name)))
        self._patch(kernels, "upload_k3plus_groups", self._timed("group_upload", kernels.upload_k3plus_groups))
        self._patch(kernels, "count_tiled_fused", self._timed("count_fused", kernels.count_tiled_fused))
        self._patch(kernels, "build_k3_graph", self._timed("k3_graph", kernels.build_k3_graph))
        for name in ("decode_k2_pairs_flat", "decode_k3plus_flat", "decode_k3_slots"):
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
        self._patch(
            matrix, "_build_csr_from_transactions", self._timed_csr_build(matrix._build_csr_from_transactions)
        )
        self._patch(
            csr_bitvec, "build_bitvecs_row_split", self._timed_span("_bitvec_split", csr_bitvec.build_bitvecs_row_split)
        )
        self._patch(
            csr_bitvec,
            "build_bitvecs_row_split_from_arrays",
            self._timed_span("_bitvec_arrays", csr_bitvec.build_bitvecs_row_split_from_arrays),
        )
        self._patch(csr_bitvec, "build_bitvecs_gpu", self._timed_bitvec_gpu(csr_bitvec.build_bitvecs_gpu))
        self._patch(input_layer, "_chunk_entries", self._timed_span("input_dev_chunks", input_layer._chunk_entries))
        for name in ("copy", "finish"):
            self._patch(input_layer._Uploader, name, self._timed_span("input_dev_copy", getattr(input_layer._Uploader, name)))
        self._patch(input_layer, "_occurrences", self._timed_span("input_dev_k1", input_layer._occurrences))
        self._patch(input_layer, "_scatter_bitvecs", self._timed_span("input_dev_bitvec", input_layer._scatter_bitvecs))
        self._patch(input_layer.DeviceRows, "pairs", self._timed("k2_dispatch", input_layer.DeviceRows.pairs))
        self._patch(input_layer.DeviceRows, "upload", self._timed("k2_input", input_layer.DeviceRows.upload))
        self._patch(row_split, "_build_deferred_frame", self._timed_span("frame", row_split._build_deferred_frame))
        # et_miner.core re-exports the apriori function under the module's name.
        apriori_mod = importlib.import_module("et_miner.core.apriori")
        self._patch(apriori_mod, "_as_item_dtype", self._timed_span("frame", apriori_mod._as_item_dtype))

    def uninstall(self) -> None:
        while self._saved:
            owner, name, original = self._saved.pop()
            setattr(owner, name, original)

    def callback(self, level_cb=None):
        """A level_callback that closes the level's split, then calls ``level_cb``."""

        def cb(k, n_candidates, n_frequent, ms):
            now = time.perf_counter()
            if self._first_level_start is None:
                self._first_level_start = now - ms / 1000.0
            self._last_level_end = now
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
