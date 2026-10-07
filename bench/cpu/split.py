"""Per-level time split of the CPU route and of efficient-apriori, by wrapping their inner steps.

Wraps the functions each miner calls inside a level with wall-clock timers and
attributes the time to the level whose end is recorded next. Results are
unchanged: every wrapper returns what the original returns, and each costs a
few microseconds per call (one call per level or per batch, never per
candidate).

Usage (inside a bench child, after warm-up, around the timed call):

    split = CpuSplit()
    split.install()
    try:
        apriori(..., level_callback=split.callback(level_cb))
    finally:
        split.uninstall()
    split.levels   # {k: {phase: seconds, ...}}
    split.tail     # phases after the last level (result frame build)

    ea = EaSplit()
    ea.install()
    try:
        itemsets_from_transactions(...)
    finally:
        ea.uninstall()
    ea.levels(t_end)

CPU-route phases (seconds, union of intervals; nested phases are also counted
inside their parent):
    matrix_build    build_boolean_matrix (transaction count, K=1 counts, list.contains matrix)
    cand_gen        _generate_candidates
    count           count_support_batched, which contains:
      len_filter      _filter_transactions_by_length (sum_horizontal + filtered copy)
      density         _estimate_density
      to_csr          _polars_to_sparse_csr
      k2_sparse       _count_support_sparse_k2_batch / _parallel (contains matmul)
      matmul          _sparse_matmul
      kgt2            _count_support_sparse_k_gt_2 (contains rust_call)
      rust_call       the Rust counting call (_call_with_budget)
      polars_count    count_support_vectorized (contains polars_collect)
      polars_collect  LazyFrame.collect inside count_support_vectorized
    other           the level's time minus cand_gen and count (inference, dict
                    merge, threshold filter, emit, free-set test)
    level           the level's time as level_callback reports it
    span            wall time since the previous level closed (K=1: since install);
                    for K=1 it holds matrix_build and the max-length scan

efficient-apriori phases:
    index           TransactionManager construction (one row-id set per item)
    cand_gen        apriori_gen (join + prune) for level k
    count           everything else of level k: the transaction_indices_sc loop
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from level_split import _union_length  # noqa: E402

PHASES = (
    "matrix_build", "cand_gen", "count", "len_filter", "density", "to_csr", "k2_sparse", "matmul",
    "kgt2", "rust_call", "polars_count", "polars_collect",
)
#: Top-level phases of a level; "other" is the level's time minus these.
TOP = ("cand_gen", "count")


class CpuSplit:
    """Installs and removes the CPU-route timing wrappers; collects per-level phase seconds."""

    def __init__(self) -> None:
        self.levels: dict[int, dict[str, float]] = {}
        self.tail: dict[str, float] = {}
        self._pending: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self._saved: list[tuple[object, str, object]] = []
        self._t_last: float | None = None

    def _timed(self, phase: str, fn):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self._pending[phase].append((t0, time.perf_counter()))

        return wrapper

    def _timed_vectorized(self, fn):
        """count_support_vectorized, with LazyFrame.collect timed while it runs."""
        import polars as pl

        def wrapper(*args, **kwargs):
            original = pl.LazyFrame.collect
            pl.LazyFrame.collect = self._timed("polars_collect", original)
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                pl.LazyFrame.collect = original
                self._pending["polars_count"].append((t0, time.perf_counter()))

        return wrapper

    def _patch(self, owner, name: str, replacement) -> None:
        self._saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def install(self) -> None:
        import importlib

        # By module path: et_miner.core re-exports the function `apriori`, which shadows the submodule.
        ap = importlib.import_module("et_miner.core.apriori")
        mx = importlib.import_module("et_miner.core.matrix")
        sp = importlib.import_module("et_miner.core.sparse")

        self._patch(ap, "build_boolean_matrix", self._timed("matrix_build", ap.build_boolean_matrix))
        self._patch(ap, "_generate_candidates", self._timed("cand_gen", ap._generate_candidates))
        self._patch(ap, "count_support_batched", self._timed("count", ap.count_support_batched))
        self._patch(ap, "_build_result_df", self._timed("result_df", ap._build_result_df))
        self._patch(mx, "_filter_transactions_by_length", self._timed("len_filter", mx._filter_transactions_by_length))
        self._patch(mx, "count_support_vectorized", self._timed_vectorized(mx.count_support_vectorized))
        self._patch(sp, "_estimate_density", self._timed("density", sp._estimate_density))
        self._patch(sp, "_polars_to_sparse_csr", self._timed("to_csr", sp._polars_to_sparse_csr))
        for name in ("_count_support_sparse_k2_batch", "_count_support_sparse_k2_parallel"):
            self._patch(sp, name, self._timed("k2_sparse", getattr(sp, name)))
        self._patch(sp, "_sparse_matmul", self._timed("matmul", sp._sparse_matmul))
        self._patch(sp, "_count_support_sparse_k_gt_2", self._timed("kgt2", sp._count_support_sparse_k_gt_2))
        self._patch(sp, "_call_with_budget", self._timed("rust_call", sp._call_with_budget))
        self._t_last = time.perf_counter()

    def uninstall(self) -> None:
        while self._saved:
            owner, name, original = self._saved.pop()
            setattr(owner, name, original)
        self.tail = {p: round(_union_length(v), 6) for p, v in self._pending.items() if v}

    def callback(self, level_cb=None):
        """A level_callback that closes the level's split, then calls ``level_cb``."""

        def cb(k, n_candidates, n_frequent, ms):
            now = time.perf_counter()
            phases = {p: round(_union_length(self._pending.get(p, [])), 6) for p in PHASES}
            level_s = ms / 1000.0
            phases["other"] = round(level_s - sum(phases[p] for p in TOP), 6)
            phases["level"] = round(level_s, 6)
            phases["span"] = round(now - self._t_last, 6)
            self._t_last = now
            self.levels[int(k)] = phases
            self._pending = defaultdict(list)
            if level_cb is not None:
                level_cb(k, n_candidates, n_frequent, ms)

        return cb


class EaSplit:
    """Times efficient-apriori's index build and each level's candidate generation and counting."""

    def __init__(self) -> None:
        self.events: list[tuple[str, float, float, int]] = []
        self._saved: list[tuple[object, str, object]] = []

    def _patch(self, owner, name: str, replacement) -> None:
        self._saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def install(self) -> None:
        import efficient_apriori.itemsets as eai

        init = eai.TransactionManager.__init__
        gen = eai.apriori_gen

        def timed_init(this, *args, **kwargs):
            t0 = time.perf_counter()
            init(this, *args, **kwargs)
            self.events.append(("index", t0, time.perf_counter(), 0))

        def timed_gen(itemsets):
            t0 = time.perf_counter()
            out = list(gen(itemsets))
            self.events.append(("cand_gen", t0, time.perf_counter(), len(out)))
            return out

        self._patch(eai.TransactionManager, "__init__", timed_init)
        self._patch(eai, "apriori_gen", timed_gen)

    def uninstall(self) -> None:
        while self._saved:
            owner, name, original = self._saved.pop()
            setattr(owner, name, original)

    def levels(self, t_end: float) -> dict:
        """{"index": s, "levels": {k: {"cand_gen", "count", "n_candidates"}}} from the recorded events.

        Level k's counting runs from the end of its apriori_gen call to the start
        of level k+1's call (the last level: to ``t_end``, the return of the
        timed call). K=1's count is the time between the index build and the
        first apriori_gen call.
        """
        index = next((e for e in self.events if e[0] == "index"), None)
        gens = [e for e in self.events if e[0] == "cand_gen"]
        out: dict = {"index": round(index[2] - index[1], 6) if index else None, "levels": {}}
        if index is None:
            return out
        first_end = gens[0][1] if gens else t_end
        out["levels"][1] = {"cand_gen": 0.0, "count": round(first_end - index[2], 6), "n_candidates": None}
        for i, (_, g0, g1, n) in enumerate(gens):
            nxt = gens[i + 1][1] if i + 1 < len(gens) else t_end
            out["levels"][i + 2] = {"cand_gen": round(g1 - g0, 6), "count": round(nxt - g1, 6), "n_candidates": n}
        return out
