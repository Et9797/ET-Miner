"""Phase 4 of the route consolidation: every surviving default config on the final tree.

Each config keeps the ``base_id`` of the arm that survived it in Phase 2 (``bench/routes_matrix.py``), with no
pin, so the final rows join the Phase 2 rows by id: ``C1-default`` for the in-core GPU regimes, ``C1-coded`` for
the cast inputs (the coded device layer is the default now), ``D1-ported`` / ``E2-ported`` for SON on the GPU,
``D1-default`` for SON on the CPU, ``F1-default`` and ``R1-default`` for the CPU route and the Rust miner.
Left out: the arms Phase 3 removed, and the defaults that timed out in Phase 2 (the CPU route on dsl and
dslxl, apriori_from_csr on deepk20), whose rows are lower bounds already.
"""

from __future__ import annotations

from routes_matrix import CPU_IN_CORE, GPU_DEFAULT, THREADS, _cfg, _reps, _son

#: Phase 2 defaults that timed out (their lower bounds stand; rerunning them only spends the budget).
TIMED_OUT = {("dsl", "F"), ("dslxl", "F"), ("deepk20", "RUST")}


def final_configs(n_dev: int) -> list[dict]:
    n_jobs = int(THREADS)
    two = n_dev >= 2
    gpu = [_cfg(w, "C", "default") for w in GPU_DEFAULT]
    gpu += [_cfg("dsl", "C", "default", option=o) for o in ("infer", "free")]
    gpu += [_cfg(w, "C-bitvecs", "default") for w in ("oom2", "sk2ml2", "dsl", "oom2ml3", "sk2ml3")]
    gpu += [_cfg(w, "C", "coded", variant=v) for w, v in (("dsl", "str"), ("or002", "str"), ("dsl", "wide"))]
    if two:
        gpu += [_cfg(w, "C", "default", n_gpus=2) for w in ("oom2", "sk2ml2", "dsl", "or002")]
        gpu += [_cfg(w, "C-bitvecs", "default", n_gpus=2) for w in ("oom2", "dsl", "oom2ml3", "sk2ml3")]
    son = []
    for w in ("deepk", "dsl", "dslxl"):
        son.append(_son(w, "D", "ported"))
        if two:
            son.append(_son(w, "E", "ported", n_gpus=2))
    son += [_son(w, "D", "default", use_gpu=False) for w in ("deepk", "skew", "wide", "or005", "dsl")]
    cpu = []
    for w in CPU_IN_CORE:
        if (w, "F") not in TIMED_OUT:
            cpu.append(_cfg(w, "F", "default", n_jobs=n_jobs))
        if (w, "RUST") not in TIMED_OUT:
            cpu.append(_cfg(w, "RUST", "default"))
    cpu += [_cfg("dsl", "F", "default", option=o, n_jobs=n_jobs) for o in ("infer", "free")]
    return _reps(son + gpu + cpu)
