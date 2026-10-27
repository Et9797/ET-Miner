"""The route consolidation's matrices (`bench/runner.py --mode routes|routes-cpu|routes-census|routes-final`).

Every config names its route (``bench/routes_run.py``) and arm, pins every thread pool to ``THREADS``, sets
``NCCL_P2P_DISABLE=1``, restricts one-GPU configs to device 0 and pins the arm by env (GPU) or by module
overrides (CPU). A regime is (workload, input variant, option set, GPU count); its default config runs every
knob unset and is the "full dispatch" or "today" arm of every decision point that names the regime
(bench/routes/PROTOCOL.md maps arms to decision points). Config ids are
``<regime>-<route><n>-<arm>#r<rep>``; every config of one (dataset, min_support, max_length, free sets)
group must give one signature.

Modes:
    routes-cpu     Phase 2a: the CPU decision points (C1-C3, SON's CPU passes), no GPU
    routes         Phase 2b: the GPU decision points (G1-G3, G5-G7, S1, S2)
    routes-census  one rep of every GPU regime's default config, with the level split (harness check)
    routes-final   Phase 4: every surviving default config on the final tree

Ordering: one-off axes first, every config's rep 0 before any rep 1, rep-major, so a budget stop sheds reps.
"""

from __future__ import annotations

#: Thread pools: the box's cgroup quota (17.92 CPUs on the 2x RTX 3060 / 2x E5-2680 v4 box; its 28
#: physical cores would oversubscribe the quota).
THREADS = "17"
POOLS = {k: THREADS for k in ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                              "OPENBLAS_NUM_THREADS")}
REPS = 3
TIMEOUT_S = 900

#: id -> (dataset, min_support, max_length)
WORKLOADS = {
    "smoke": ("smoke", 0.01, None),
    "deepk": ("deep_k", 0.02, None),
    "deepk15": ("deep_k", 0.15, None),
    "skew": ("skewed_rows", 0.02, None),
    "wide": ("wide_vocab", 0.004, None),
    "deepk20": ("deep_k20", 0.01, None),
    "oom2": ("oom_regression", 0.00003, 2),
    "oom2ml3": ("oom_regression", 0.00003, 3),
    "sk2ml2": ("stress_k2", 0.000015, 2),
    "sk2ml3": ("stress_k2", 0.000015, 3),
    "dsl": ("deep_sparse_large", 0.015, None),
    "dslxl": ("deep_sparse_xl", 0.015, None),
    "or005": ("online_retail", 0.005, None),
    "or003": ("online_retail", 0.003, None),
    "or002": ("online_retail", 0.002, None),
    "or0001k2": ("online_retail", 0.0001, 2),
}

#: Option sets: label -> config keys.
OPTIONS = {
    "": {},
    "infer": {"use_generator_pruning": True},
    "free": {"prune_equal_support": True},
    "noprune": {"prune_apriori": False},
}

#: SON regimes: workload -> chunk_size (each runs exactly 4 chunks).
SON = {"dsl": 5_000_000, "deepk": 250_000, "dslxl": 15_000_000, "skew": 250_000, "or005": 9_106, "wide": 25_000}


def _regime(workload: str, variant: str = "int", option: str = "", tag: str = "") -> str:
    parts = [workload] + ([variant] if variant != "int" else []) + ([option] if option else []) + ([tag] if tag else [])
    return "-".join(parts)


def _cfg(workload: str, route: str, arm: str, *, n_gpus: int = 1, variant: str = "int", option: str = "",
         env: dict | None = None, overrides: dict | None = None, tag: str = "", timeout_s: int = TIMEOUT_S,
         **kw) -> dict:
    dataset, min_support, max_length = WORKLOADS[workload]
    regime = _regime(workload, "bv" if route == "C-bitvecs" else variant, option, tag)
    route_tag = {"C": "C", "C-bitvecs": "C", "D": "D", "E": "E", "F": "F", "RUST": "R"}[route]
    full_env = {**POOLS, "NCCL_P2P_DISABLE": "1", **(env or {})}
    cpu = route in ("F", "RUST") or (route == "D" and not kw.get("use_gpu", True))
    if not cpu and n_gpus == 1:
        full_env["CUDA_VISIBLE_DEVICES"] = "0"
    if cpu:
        full_env["CUDA_VISIBLE_DEVICES"] = ""
    return {
        "base_id": f"{regime}-{route_tag}{n_gpus}-{arm}",
        "mode": "routes",
        "route": route,
        "arm": arm,
        "regime": regime,
        "workload": workload,
        "dataset": dataset,
        "preset": dataset,
        "min_support": min_support,
        "max_length": max_length,
        "n_gpus": n_gpus,
        "input": variant,
        "env": full_env,
        "overrides": overrides or {},
        "cpu_config": cpu,
        "level_split": route in ("C", "C-bitvecs"),
        "allow_fallback": [],
        "timeout_s": timeout_s,
        **OPTIONS[option],
        **kw,
    }


def _son(workload: str, route: str, arm: str, *, n_gpus: int = 1, env: dict | None = None,
         overrides: dict | None = None, use_gpu: bool = True, timeout_s: int = 1800) -> dict:
    return _cfg(workload, route, arm, n_gpus=n_gpus, env=env, overrides=overrides, tag="son", timeout_s=timeout_s,
                chunk_size=SON[workload], expect_chunks=4, use_gpu=use_gpu, n_jobs=int(THREADS))


# ── Phase 2a: CPU ───────────────────────────────────────────────────────────

CPU_IN_CORE = ("smoke", "deepk", "skew", "wide", "deepk20", "or005", "or003", "or002", "or0001k2", "dsl")
#: Regimes with a K>=3 level (C2).
CPU_DEEP = ("smoke", "deepk", "skew", "wide", "deepk20", "or005", "or003", "or002", "dsl")
INF = float("inf")
CPU_ARMS = {
    # C1, K=2
    "gram": {"et_miner.core.cpu_miner:GRAM_BITVEC_RATIO": 0.0},
    "bitvec": {"et_miner.core.cpu_miner:GRAM_BITVEC_RATIO": INF},
    # C2, K>=3 per prefix group
    "popcount": {"et_miner.core.cpu_miner:PROJ_MIN_SUFFIXES": [[INF, 10**9]]},
    "proj": {"et_miner.core.cpu_miner:PROJ_MIN_SUFFIXES": [[INF, 0]]},
    # C2, SON pass 2 at K>=3
    "percand": {"et_miner.core.cpu_miner:PER_CANDIDATE_MAX_WORDS": 10**12},
    "groups": {"et_miner.core.cpu_miner:PER_CANDIDATE_MAX_WORDS": -1,
               "et_miner.core.cpu_miner:PER_CANDIDATE_MEAN_GROUP": 0.0},
}


def build_cpu() -> list[dict]:
    n_jobs = int(THREADS)
    one_off = [_cfg("dslxl", "F", "default", n_jobs=n_jobs, timeout_s=1800)]
    son = [
        _son(w, "D", arm, use_gpu=False, overrides=CPU_ARMS.get(arm))
        for w in ("deepk", "skew", "wide", "or005", "dsl")
        for arm in ("default", "percand", "groups")
    ]
    base = []
    for w in CPU_IN_CORE:
        base.append(_cfg(w, "F", "default", n_jobs=n_jobs))
        base.append(_cfg(w, "RUST", "default"))
        base += [_cfg(w, "F", arm, n_jobs=n_jobs, overrides=CPU_ARMS[arm]) for arm in ("gram", "bitvec")]
        if w in CPU_DEEP:
            base += [_cfg(w, "F", arm, n_jobs=n_jobs, overrides=CPU_ARMS[arm]) for arm in ("popcount", "proj")]
    for option in ("infer", "free"):
        base.append(_cfg("dsl", "F", "default", option=option, n_jobs=n_jobs))
    return _reps(one_off, one_rep=True) + _reps(son + base)


# ── Phase 2b: GPU ───────────────────────────────────────────────────────────

#: Integer regimes on the device layer, their default config (every knob unset).
GPU_DEFAULT = ("smoke", "deepk", "deepk15", "skew", "wide", "deepk20", "or005", "or003", "or002", "or0001k2",
               "oom2", "oom2ml3", "sk2ml2", "sk2ml3", "dsl", "dslxl")
LEAVE = {
    "no-rows": {"ET_MINER_K2_KERNEL": "dense"},
    "no-k2shared": {"ET_MINER_LEAVE_OUT": "k2-shared"},
    "no-k2percand": {"ET_MINER_LEAVE_OUT": "k2-percand"},
    "no-k2tiled": {"ET_MINER_LEAVE_OUT": "k2-tiled"},
    "no-k2fused": {"ET_MINER_LEAVE_OUT": "k2-fused"},
    "no-percand": {"ET_MINER_LEAVE_OUT": "percand"},
    "no-group": {"ET_MINER_LEAVE_OUT": "group"},
    "no-tiled": {"ET_MINER_LEAVE_OUT": "tiled"},
    "no-fused": {"ET_MINER_LEAVE_OUT": "fused"},
}
#: G3: (workload, variant, option, GPU counts, arms). "bv" is the bitvecs= input.
G3 = (
    ("oom2", "int", "", (1, 2), ("no-rows",)),
    ("sk2ml2", "int", "", (1, 2), ("no-rows",)),
    ("wide", "int", "", (1,), ("no-rows", "no-k2tiled")),
    ("or0001k2", "int", "", (1,), ("no-rows", "no-k2tiled")),
    ("or002", "int", "", (1,), ("no-rows", "no-k2tiled")),
    ("deepk15", "int", "", (1,), ("no-k2percand",)),
    ("dsl", "int", "", (1, 2), ("no-k2tiled",)),
    ("dslxl", "int", "", (1,), ("no-k2tiled",)),
    ("oom2", "bv", "", (1, 2), ("no-k2tiled",)),
    ("sk2ml2", "bv", "", (1, 2), ("no-k2fused",)),
)
#: Members only one GPU runs: on several GPUs their work goes per candidate whatever ET_MINER_LEAVE_OUT says.
ONE_GPU_ONLY = ("no-k2fused", "no-fused")
#: G3's shared-memory row-wise K=2: dsl with the row-wise kernel pinned (its 3,570 pairs fit shared memory).
G3_SHARED = ("dsl", {"ET_MINER_K2_KERNEL": "rows"})
#: G5: (workload, variant, option, GPU counts, arms).
G5 = (
    ("dsl", "int", "", (1, 2), ("no-percand", "no-group", "no-tiled")),
    ("dsl", "int", "infer", (1,), ("no-percand", "no-group", "no-tiled")),
    ("dsl", "int", "free", (1,), ("no-percand", "no-group", "no-tiled")),
    ("dslxl", "int", "", (1,), ("no-percand", "no-group", "no-tiled")),
    ("or002", "int", "", (1, 2), ("no-percand", "no-group", "no-tiled")),
    ("or003", "int", "", (1,), ("no-percand", "no-group", "no-tiled")),
    ("deepk", "int", "", (1,), ("no-percand", "no-group", "no-tiled")),
    ("skew", "int", "", (1,), ("no-percand", "no-group", "no-tiled")),
    ("deepk20", "int", "", (1,), ("no-percand", "no-group", "no-tiled")),
    ("dsl", "bv", "", (1, 2), ("no-percand", "no-group", "no-tiled")),
    ("oom2ml3", "bv", "", (1, 2), ("no-percand", "no-group", "no-tiled")),
    ("sk2ml3", "bv", "", (1, 2), ("no-percand", "no-group", "no-tiled", "no-fused")),
    ("dsl", "int", "noprune", (1,), ("no-percand", "no-group", "no-tiled")),
    ("oom2ml3", "int", "noprune", (1,), ("no-percand", "no-group", "no-tiled")),
)
G1_INT = ("deepk", "skew", "or002", "oom2ml3", "sk2ml3", "dsl", "dslxl")
G1_CAST = (("dsl", "str"), ("or002", "str"), ("dsl", "wide"))
G2 = ("smoke", "deepk", "skew", "wide", "deepk20", "or005", "or003", "or002", "oom2ml3", "sk2ml3", "dsl", "dslxl")
G6 = ("deepk", "skew", "or003", "or002", "oom2ml3", "sk2ml3", "dsl")
G7 = ("deepk", "or002", "oom2ml3", "sk2ml3", "dsl")


def _gpu_one_off(n_dev: int) -> list[dict]:
    """The configs of one decision point each (SON on the GPU, ESCO), and the input cast variants."""
    out = []
    for w in ("deepk", "dsl", "dslxl"):
        out.append(_son(w, "D", "boolean", env={"ET_MINER_SON_GPU": "boolean"}))
        out.append(_son(w, "D", "ported", env={"ET_MINER_SON_GPU": "ported"}))
        if n_dev >= 2:
            out.append(_son(w, "E", "boolean", n_gpus=2, env={"ET_MINER_SON_GPU": "boolean"}))
            out.append(_son(w, "E", "ported", n_gpus=2, env={"ET_MINER_SON_GPU": "ported"}))
    for w in G6:
        out.append(_cfg(w, "C", "esco-auto", sparse_from_k="auto"))
        out.append(_cfg(w, "C", "esco3", sparse_from_k=3))
    for w, variant in G1_CAST:
        for arm in ("host", "rebuilt", "coded"):
            out.append(_cfg(w, "C", arm, variant=variant, env={"ET_MINER_INPUT_LAYER": arm}))
    return out


def _gpu_main(n_dev: int) -> list[dict]:
    gpus = (1, 2) if n_dev >= 2 else (1,)
    out = [_cfg(w, "C", "default") for w in GPU_DEFAULT]
    out += [_cfg(w, "C", "default", option=o) for w in ("dsl",) for o in ("infer", "free")]
    out += [_cfg(w, "C", "default", option="noprune") for w in G7]
    for w in G1_INT:
        out += [_cfg(w, "C", arm, env={"ET_MINER_INPUT_LAYER": arm}) for arm in ("host", "rebuilt")]
    out += [_cfg(w, "C", "keep", env={"ET_MINER_ROW_ORDER": "keep"}) for w in G2]
    for o in ("infer", "free"):
        out.append(_cfg("dsl", "C", "keep", option=o, env={"ET_MINER_ROW_ORDER": "keep"}))
    w, env = G3_SHARED
    out.append(_cfg(w, "C", "rows", env=env))
    out.append(_cfg(w, "C", "rows-global", env={**env, **LEAVE["no-k2shared"]}))
    route = {"int": "C", "bv": "C-bitvecs"}
    for w, variant, option, ns, arms in G3 + G5:
        for n in ns:
            arms_n = [a for a in arms if n == 1 or a not in ONE_GPU_ONLY]
            if n not in gpus or not arms_n:
                continue
            r = route[variant]
            if variant == "bv" or n == 2:
                out.append(_cfg(w, r, "default", n_gpus=n, option=option))
            out += [_cfg(w, r, a, n_gpus=n, option=option, env=LEAVE[a]) for a in arms_n]
    seen, unique = set(), []
    for c in out:
        if c["base_id"] not in seen:
            seen.add(c["base_id"])
            unique.append(c)
    return unique


#: Regimes whose default wall is under 1 s on the 3060 box (bench/routes census): their reps 1-2 run last, the
#: first a budget stop sheds.
SUBSECOND = ("smoke", "deepk", "deepk15", "skew", "wide", "or005", "or003", "or002", "oom2", "oom2ml3")


def _reps(configs: list[dict], one_rep: bool = False) -> list[dict]:
    """Rep-major: every rep 0, then reps 1-2 of the regimes at or above 1 s, then those of the sub-second ones."""
    rep0 = [{**c, "id": f"{c['base_id']}#r0", "rep": 0} for c in configs]
    if one_rep:
        return rep0
    fast = [c for c in configs if c["workload"] in SUBSECOND and c["route"] in ("C", "C-bitvecs")]
    slow = [c for c in configs if c not in fast]
    return rep0 + [
        {**c, "id": f"{c['base_id']}#r{rep}", "rep": rep} for group in (slow, fast) for rep in range(1, REPS) for c in group
    ]


def rep0_failed_skip(cfg: dict, rows: list[dict]) -> str | None:
    """Why a rep >= 1 config does not run (its rep 0 timed out or failed: never repeated), else None."""
    if cfg.get("mode") != "routes" or cfg.get("rep", 0) == 0:
        return None
    rep0 = next((r for r in reversed(rows) if r.get("id") == f"{cfg['base_id']}#r0"), None)
    if rep0 is not None and rep0.get("status") != "ok":
        return f"skipped: rep 0 {str(rep0.get('status'))[:80]}"
    return None


def build_routes_matrix(mode: str, n_dev: int) -> list[dict]:
    if mode == "routes-cpu":
        return build_cpu()
    if mode == "routes-census":
        census = [_cfg(w, "C", "default") for w in GPU_DEFAULT]
        census += [_cfg(w, "C-bitvecs", "default") for w in ("oom2", "sk2ml2", "dsl", "oom2ml3", "sk2ml3")]
        census += [_cfg(w, "C", "default", option="noprune") for w in ("dsl", "oom2ml3")]
        return _reps(census, one_rep=True)
    if mode == "routes-final":
        from routes.final import final_configs

        return final_configs(n_dev)
    return _reps(_gpu_one_off(n_dev) + _gpu_main(n_dev))
