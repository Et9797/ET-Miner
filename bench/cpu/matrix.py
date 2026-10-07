"""The CPU-tier baseline matrix (`bench/runner.py --mode cpu-baseline`).

Arms × thread settings × workloads, pre-registered in bench/cpu/PROTOCOL.md.
Every config pins the four thread pools (plus OpenBLAS) and ``n_jobs``, so no
number depends on a default. efficient-apriori runs single-threaded once per
rep. Order: rep-major; within a rep the workloads run in a fixed order and the
arms of a workload run forward on even reps and reversed on odd reps.

Reps 1 and 2 are budget-gated (``rep_budget``): after rep 0 the runner keeps
them for the cheapest base configs whose doubled rep-0 process time fits the
remaining campaign budget (`rep_budget_skip`).

Usage: imported by bench/runner.py; `build_cpu_baseline_matrix()` returns the configs.
"""

from __future__ import annotations

REPS = 3
#: Campaign budget in process seconds (rep 0 plus the reps it admits).
BUDGET_S = 6300
#: Per-config cap: a cap hit is a lower bound and is never repeated.
TIMEOUT_F_S = 600
TIMEOUT_EA_S = 900

THREADS = {"T1": 1, "T4": 4}

#: id -> (dataset, min_support, max_length)
WORKLOADS = {
    "smoke": ("smoke", 0.01, None),
    "deepk": ("deep_k", 0.02, None),
    "skew": ("skewed_rows", 0.02, None),
    "wide": ("wide_vocab", 0.004, None),
    "or005": ("online_retail", 0.005, None),
    "or003": ("online_retail", 0.003, None),
    "or002": ("online_retail", 0.002, None),
    "or0001k2": ("online_retail", 0.0001, 2),
}

#: arm -> apriori(sparse=...)
F_ARMS = {"F-polars": False, "F-auto": None, "F-sparse": True}


def _env(n: int) -> dict:
    s = str(n)
    return {
        "POLARS_MAX_THREADS": s,
        "RAYON_NUM_THREADS": s,
        "MKL_NUM_THREADS": s,
        "OMP_NUM_THREADS": s,
        "OPENBLAS_NUM_THREADS": s,
    }


def _base_configs(workload: str) -> list[dict]:
    dataset, min_support, max_length = WORKLOADS[workload]
    common = {"mode": "cpu", "dataset": dataset, "preset": dataset, "min_support": min_support,
              "max_length": max_length, "n_gpus": 0, "rep_budget": True}
    out = []
    for arm, sparse in F_ARMS.items():
        for t, n in THREADS.items():
            out.append({**common, "base_id": f"{workload}-{arm}-{t}", "route": "F", "arm": arm, "threads": t,
                        "sparse": sparse, "n_jobs": n, "env": _env(n), "timeout_s": TIMEOUT_F_S})
    out.append({**common, "base_id": f"{workload}-EA", "route": "EA", "arm": "EA", "threads": "T1",
                "env": _env(1), "timeout_s": TIMEOUT_EA_S})
    return out


def build_cpu_baseline_matrix() -> list[dict]:
    out = []
    for rep in range(REPS):
        for w in WORKLOADS:
            base = _base_configs(w)
            if rep % 2:
                base = base[::-1]
            out.extend({**c, "id": f"{c['base_id']}#r{rep}", "rep": rep} for c in base)
    return out


def rep_budget_skip(cfg: dict, rows: list[dict]) -> str | None:
    """None when a rep>=1 config is admitted by the budget, else why not.

    Rule (PROTOCOL.md, "Repetitions"): rep 0 always runs. Once every rep-0
    config has a row, take each base config's rep-0 process seconds (a cap hit
    or an error is never repeated), sort ascending, and admit reps 1-2 for the
    longest prefix whose doubled cost fits ``BUDGET_S`` minus the rep-0 total.
    """
    if not cfg.get("rep_budget") or cfg.get("rep", 0) == 0:
        return None
    rep0 = {}
    for r in rows:
        c = r.get("config", {})
        if c.get("mode") == "cpu" and c.get("rep") == 0:
            rep0[c["base_id"]] = r
    expected = {c["base_id"] for c in build_cpu_baseline_matrix() if c["rep"] == 0}
    if not expected <= rep0.keys():
        return "skipped: rep 0 incomplete"
    row = rep0[cfg["base_id"]]
    if row.get("status") != "ok":
        return f"skipped: rep 0 {row.get('status')}"
    spent = sum(float(r.get("proc_s") or 0) for r in rep0.values())
    remaining = BUDGET_S - spent
    admitted: set[str] = set()
    used = 0.0
    for base_id, r in sorted(rep0.items(), key=lambda kv: float(kv[1].get("proc_s") or 0)):
        if r.get("status") != "ok":
            continue
        cost = 2 * float(r.get("proc_s") or 0)
        if used + cost > remaining:
            break
        admitted.add(base_id)
        used += cost
    if cfg["base_id"] in admitted:
        return None
    return f"skipped: budget (rep-0 total {spent:.0f}s of {BUDGET_S}s; reps 1-2 cover {len(admitted)} configs)"
