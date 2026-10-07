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

CPU-route phases (core/cpu_miner.py; seconds, union of intervals):
    matrix_build    build_transaction_csr (K=1 counts and the CSR)
    k2_gram         count_pairs (scipy Gram)
    k2_bitvec       count_pairs_bitvec (K=2 on bitvectors)
    bitvec_build    build_bitvecs (row-space bitvectors; may run inside k2_bitvec's level)
    count           count_candidates (K>=3 prefix groups)
    subsets         _Level.subset_positions (Pascal and free-set look-ups)
    other           the level's time minus the phases above: candidate
                    generation, Pascal and free-set arithmetic, bookkeeping
                    (the profile's kN_candidate_gen phases time the generation)
    level           the level's time as level_callback reports it
    span            wall time since the previous level closed (K=1: since install);
                    for K=1 it holds matrix_build and the max-length scan
    tail: emit      _emit (the result frame), after the last level

Rows recorded before the array miner (Phase 0, bench/results/2026-10-07-cpu-baseline)
carry the old engine's phases (build_boolean_matrix, count_support_batched and
their parts); see this file at commit b40a1dd.

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

PHASES = ("matrix_build", "k2_gram", "k2_bitvec", "bitvec_build", "count", "subsets")
#: Phases inside a level's timed span; "other" is the level's time minus these.
TOP = ("k2_gram", "k2_bitvec", "bitvec_build", "count", "subsets")


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

    def _patch(self, owner, name: str, replacement) -> None:
        self._saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def install(self) -> None:
        import importlib

        cm = importlib.import_module("et_miner.core.cpu_miner")
        for name, phase in (("build_transaction_csr", "matrix_build"), ("count_pairs", "k2_gram"),
                            ("count_pairs_bitvec", "k2_bitvec"), ("build_bitvecs", "bitvec_build"),
                            ("count_candidates", "count"), ("_emit", "emit")):
            self._patch(cm, name, self._timed(phase, getattr(cm, name)))
        self._patch(cm._Level, "subset_positions", self._timed("subsets", cm._Level.subset_positions))
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
