"""The GPU-layer consolidation matrix (`bench/runner.py --mode consolidation`).

Every config names its route (see bench/consolidation_run.py for what each
route calls) and pins every thread pool and every kernel/route knob it
depends on, so no number depends on a default. bench/consolidation/PROTOCOL.md
maps the configs to the decision points and fixes the decision rule.

Ordering: every config's rep 0 runs before any rep 1, and the one-off axes
(DP5-DP10) run before the kernel/miner axes, so a `--max-hours` stop sheds
reps rather than whole axes; the smoke and deep_k reps 1-2 run last, so they
are the first reps shed. The stress_k2 max_length=3 configs are conditional
(`requires_within`): the runner runs one only if its max_length=2 twin's K=2
median is within 2x of the fastest K=2 median among configs using the same
number of GPUs. The SON configs on deep_sparse_large run once, capped at
10 minutes (`single_rep`).
"""

from __future__ import annotations

THREADS = "6"
BASE_ENV = {
    "POLARS_MAX_THREADS": THREADS,
    "RAYON_NUM_THREADS": THREADS,
    "MKL_NUM_THREADS": THREADS,
    "OMP_NUM_THREADS": THREADS,
}
REPS = 3

#: id -> (dataset, min_support, max_length)
WORKLOADS = {
    "smoke": ("smoke", 0.01, None),
    "deepk": ("deep_k", 0.02, None),
    "skew": ("skewed_rows", 0.02, None),
    "oom2": ("oom_regression", 0.00003, 2),
    "sk2ml2": ("stress_k2", 0.000015, 2),
    "sk2ml3": ("stress_k2", 0.000015, 3),
    "dsl": ("deep_sparse_large", 0.015, None),
    "wide": ("wide_vocab", 0.004, None),
    "or005": ("online_retail", 0.005, None),
    "or003": ("online_retail", 0.003, None),
    "or002": ("online_retail", 0.002, None),
    "oom2ml3": ("oom_regression", 0.00003, 3),
    **{
        f"or{label}-{depth}": ("online_retail", support, max_length)
        for label, support in (("0001", 0.0001), ("00005", 0.00005))
        for depth, max_length in (("k2", 2), ("k3", 3), ("k4", 4))
    },
}

DEEP = ("smoke", "deepk", "skew", "dsl", "or003", "or002")
IN_CORE = ("smoke", "deepk", "skew", "oom2", "sk2ml2", "dsl", "wide", "or005", "or003", "or002")
WIDE_K2 = ("oom2", "sk2ml2")
CPU = ("smoke", "deepk", "skew", "wide", "or005", "or003")
SON = {"dsl": (5_000_000, 600), "deepk": (250_000, 1800)}
LAST_REPS = ("smoke", "deepk")

#: kernel variant -> the env that pins it (tiny-group routing at its default)
VARIANTS = {
    "shared": {"ET_MINER_KERNEL_VARIANT": "shared", "ET_MINER_TILED_MIN_GROUP_PAIRS": "64"},
    "legacy": {"ET_MINER_KERNEL_VARIANT": "legacy", "ET_MINER_TILED_MIN_GROUP_PAIRS": "64"},
    "tiny0": {"ET_MINER_KERNEL_VARIANT": "shared", "ET_MINER_TILED_MIN_GROUP_PAIRS": "0"},
}


def _cfg(name: str, workload: str, route: str, *, n_gpus: int = 1, variant: str | None = None,
         env: dict | None = None, timeout_s: int = 1800, **kw) -> dict:
    dataset, min_support, max_length = WORKLOADS[workload]
    full_env = {**BASE_ENV, **(VARIANTS[variant] if variant else {}), **(env or {})}
    allow = list(kw.pop("allow_fallback", []))
    if route in ("A", "B", "D") or (route == "C" and n_gpus == 1):
        full_env["CUDA_VISIBLE_DEVICES"] = "0"
    if route == "C" and n_gpus == 1:
        full_env["ET_MINER_DISABLE_NCCL"] = "1"
    if full_env.get("ET_MINER_DISABLE_NCCL") == "1":
        allow.append("NCCL unavailable")
    if full_env.get("ET_MINER_DISABLE_RUST") == "1":
        allow.append("disabled by ET_MINER_DISABLE_RUST")
    return {
        "base_id": f"{workload}-{name}",
        "mode": "consolidation",
        "route": route,
        "dataset": dataset,
        "preset": dataset,
        "min_support": min_support,
        "max_length": max_length,
        "n_gpus": n_gpus,
        "env": full_env,
        "allow_fallback": sorted(set(allow)),
        "timeout_s": timeout_s,
        **kw,
    }


def _one_off_axes() -> list[dict]:
    out: list[dict] = []
    # DP10: survivor filter implementation and row balance, on the row-split miner.
    for w, v in (("sk2ml2", "shared"), ("dsl", "legacy")):
        for impl in ("cupy", "cpu"):
            out.append(_cfg(f"C2-{v}-filter-{impl}", w, "C", n_gpus=2, variant=v,
                            env={"ET_MINER_FILTER_IMPL": impl}))
    for w in ("skew", "dsl"):
        out.append(_cfg("C2-legacy-balance-nnz", w, "C", n_gpus=2, variant="legacy",
                        env={"ET_MINER_ROW_BALANCE": "nnz"}))
    # DP9: Rust host roles on the GPU path, and prune_apriori=False.
    for w in DEEP:
        for v in ("legacy", "shared"):
            out.append(_cfg(f"C1-{v}-norust", w, "C", variant=v, env={"ET_MINER_DISABLE_RUST": "1"}))
            out.append(_cfg(f"C1-{v}-noprune", w, "C", variant=v, prune_apriori=False))
    for w in ("smoke", "deepk", "dsl", "or002"):
        out.append(_cfg("C1-legacy-free", w, "C", variant="legacy", prune_equal_support=True))
        out.append(_cfg("C1-legacy-free-norust", w, "C", variant="legacy", prune_equal_support=True,
                        env={"ET_MINER_DISABLE_RUST": "1"}))
    # DP8: CPU tier.
    for w in CPU:
        out.append(_cfg("F-polars", w, "F", sparse=False, n_jobs=6))
        out.append(_cfg("F-sparse", w, "F", sparse=True, n_jobs=6))
        out.append(_cfg("F-sparse-norust", w, "F", sparse=True, n_jobs=6, env={"ET_MINER_DISABLE_RUST": "1"}))
        out.append(_cfg("F-auto", w, "F", sparse=None, n_jobs=6))
    # DP7: SON, forced to four chunks.
    for w, (chunk, timeout) in SON.items():
        once = w == "dsl"
        out.append(_cfg("D1-resident", w, "D", chunk_size=chunk, expect_chunks=4, gpu_resident=True,
                        timeout_s=timeout, single_rep=once))
        out.append(_cfg("D1-gpu", w, "D", chunk_size=chunk, expect_chunks=4, timeout_s=timeout, single_rep=once))
        out.append(_cfg("E2", w, "E", n_gpus=2, chunk_size=chunk, expect_chunks=4, timeout_s=timeout,
                        single_rep=once))
    out.append(_cfg("D1-cpu", "deepk", "D", chunk_size=SON["deepk"][0], expect_chunks=4, use_gpu=False))
    # DP5: layout.
    for w in DEEP:
        for v in ("legacy", "shared"):
            out.append(_cfg(f"C1-{v}-auto", w, "C", variant=v, sparse_from_k="auto"))
            out.append(_cfg(f"C1-{v}-k3", w, "C", variant=v, sparse_from_k=3))
            out.append(_cfg(f"A1-{v}-auto", w, "A", variant=v, sparse_from_k="auto"))
    return out


def _main_axes() -> list[dict]:
    out: list[dict] = []
    # DP1-DP4: in-core miners and kernels on one GPU.
    for w in IN_CORE:
        for v in ("shared", "legacy"):
            out.append(_cfg(f"A1-{v}", w, "A", variant=v))
        out.append(_cfg("B1", w, "B"))
        out.append(_cfg("C1-shared", w, "C", variant="shared"))
        if w != "sk2ml2":
            out.append(_cfg("C1-legacy", w, "C", variant="legacy"))
        if w not in WIDE_K2:
            out.append(_cfg("C1-tiny0", w, "C", variant="tiny0"))
    # DP6: multi-GPU.
    for w in ("smoke", "deepk", "skew", "oom2", "sk2ml2", "dsl"):
        for v in ("shared", "legacy"):
            out.append(_cfg(f"C2-{v}", w, "C", n_gpus=2, variant=v))
    for w in WIDE_K2:
        out.append(_cfg("Asplit2-shared", w, "A-split", n_gpus=2, variant="shared"))
        out.append(_cfg("Bsplit2", w, "B-split", n_gpus=2))
    # stress_k2 to K=3, for the candidates within 2x of the K=2 winner of their GPU class.
    ml3 = [("A1-shared", "A", 1, "shared"), ("A1-legacy", "A", 1, "legacy"), ("B1", "B", 1, None),
           ("C1-shared", "C", 1, "shared"), ("C2-shared", "C", 2, "shared"), ("C2-legacy", "C", 2, "legacy"),
           ("Asplit2-shared", "A-split", 2, "shared"), ("Bsplit2", "B-split", 2, None)]
    for name, route, n, v in ml3:
        out.append(_cfg(name, "sk2ml3", route, n_gpus=n, variant=v, timeout_s=3600,
                        requires_within={"twin": f"sk2ml2-{name}", "factor": 2.0, "level": 2}))
    return out


def build_consolidation_matrix(n_dev: int) -> list[dict]:
    """Every config x rep, rep-major, one-off axes first, smoke/deep_k reps 1-2 last."""
    base = _one_off_axes() + _main_axes()
    if n_dev < 2:
        base = [c for c in base if c["n_gpus"] == 1]

    def _last(c: dict) -> bool:
        return c["base_id"].split("-", 1)[0] in LAST_REPS

    rounds = [(0, base)]
    rounds += [(rep, [c for c in base if not _last(c)]) for rep in range(1, REPS)]
    rounds += [(rep, [c for c in base if _last(c)]) for rep in range(1, REPS)]
    cfgs = []
    for rep, members in rounds:
        for c in members:
            if rep and c.get("single_rep"):
                continue
            cfgs.append({**c, "id": f"{c['base_id']}#r{rep}", "rep": rep})
    return cfgs


def build_supplement_matrix(n_dev: int) -> list[dict]:
    """Configs run after the campaign and reported apart from it (`--mode supplement`).

    The campaign left no regime with wide K>=3 groups in which both K>=3
    kernels ran on the row-split miner: the max_length=3 gate skipped
    C1-shared on stress_k2, whose K=2 falls back to the per-candidate kernel
    once the pair counts need more than one chunk, and the 2-GPU configs were
    lost with a device. oom_regression to K=3 is such a regime on one GPU.
    """
    base = [
        _cfg("A1-shared", "oom2ml3", "A", variant="shared"),
        _cfg("C1-shared", "oom2ml3", "C", variant="shared"),
        _cfg("C1-shared-noprune", "oom2ml3", "C", variant="shared", prune_apriori=False),
        _cfg("C1-legacy", "oom2ml3", "C", variant="legacy"),
    ]
    return [{**c, "id": f"{c['base_id']}#r{rep}", "rep": rep} for rep in range(REPS) for c in base]


#: Pins every prefix group on the per-candidate kernel (no group reaches it).
_NO_GROUP = str(10**12)


def build_verify_matrix(n_dev: int) -> list[dict]:
    """The reduced re-run on the consolidated tree (`--mode verify`).

    The surviving in-core miner at its default dispatch on every in-core
    regime plus the two max_length=3 ones, the same miner with each kernel
    pinned where the dispatch has to pick, SON on one GPU and a CPU control.
    """
    tiled = {"ET_MINER_TILED_MIN_GROUP_PAIRS": "0"}
    per_candidate = {"ET_MINER_TILED_MIN_GROUP_PAIRS": _NO_GROUP}
    base = [_cfg("C1", w, "C") for w in IN_CORE + ("oom2ml3",)]
    base.append(_cfg("C1", "sk2ml3", "C", timeout_s=3600, single_rep=True))
    for w in ("dsl", "or002"):
        base.append(_cfg("C1-tiled", w, "C", env=tiled))
        base.append(_cfg("C1-percand", w, "C", env=per_candidate))
    base.append(_cfg("C1-tiled", "oom2ml3", "C", env=tiled))
    base.append(_cfg("D1", "deepk", "D", chunk_size=SON["deepk"][0], expect_chunks=4))
    base.append(_cfg("D1", "dsl", "D", chunk_size=SON["dsl"][0], expect_chunks=4, timeout_s=900, single_rep=True))
    for w in ("deepk", "or003"):
        base.append(_cfg("F-auto", w, "F", sparse=None, n_jobs=6))
    if n_dev >= 2:
        base += [_cfg("C2", w, "C", n_gpus=2) for w in ("smoke", "deepk", "oom2", "sk2ml2", "dsl")]
        base.append(_cfg("C2", "sk2ml3", "C", n_gpus=2, timeout_s=3600, single_rep=True))
        base.append(_cfg("E2", "deepk", "E", n_gpus=2, chunk_size=SON["deepk"][0], expect_chunks=4))
    cfgs = []
    for rep in range(REPS):
        for c in base:
            if rep and c.get("single_rep"):
                continue
            cfgs.append({**c, "id": f"{c['base_id']}#r{rep}", "rep": rep})
    return cfgs


#: Workloads of the candidate-waste measurement; oom_regression to K=3 follows, and stress_k2
#: to K=3 runs once, last.
WASTE = ("smoke", "deepk", "skew", "or003", "or002", "dsl")


def build_waste_matrix(n_dev: int) -> list[dict]:
    """The candidate-waste measurement (`--mode waste`).

    The row-split miner on one GPU at its default dispatch, complete lattice
    and free-sets, each level's time split recorded (`level_split.py`). Rep 0
    of every config dumps its lattice for `bench/candidate_waste.py`.
    """
    base = []
    for w in WASTE:
        base.append(_cfg("C1-split", w, "C", level_split=True))
        base.append(_cfg("C1-split-free", w, "C", level_split=True, prune_equal_support=True))
    base.append(_cfg("C1-split", "oom2ml3", "C", level_split=True))
    base.append(_cfg("C1-split", "sk2ml3", "C", level_split=True, timeout_s=3600, single_rep=True))
    cfgs = []
    for rep in range(REPS):
        for c in base:
            if rep and c.get("single_rep"):
                continue
            dump = {"dump_lattice": True} if rep == 0 else {}
            cfgs.append({**c, **dump, "id": f"{c['base_id']}#r{rep}", "rep": rep})
    return cfgs


#: The three arms of the pruning campaign (bench/pruning/PROTOCOL.md).
PRUNING_ARMS = {
    "off": {"prune_apriori": False},
    "prune": {"prune_apriori": True},
    "infer": {"prune_apriori": True, "use_generator_pruning": True},
}


def build_pruning_matrix(n_dev: int) -> list[dict]:
    """The candidate-pruning campaign (`--mode pruning`, bench/pruning/PROTOCOL.md).

    Every config records its level split. Rep-major; within a rep the short
    workloads run first; the single-rep configs run in rep 0 only.
    """

    def arm(name: str, workload: str, a: str, *, n_gpus: int = 1, free: bool = False, **kw) -> dict:
        extra = {"prune_equal_support": True} if free else {}
        return _cfg(name, workload, "C", n_gpus=n_gpus, level_split=True, **PRUNING_ARMS[a], **extra, **kw)

    short, long_ = [], []
    for w in ("smoke", "deepk", "skew", "or003", "or002"):
        short += [arm(f"C1-{a}", w, a) for a in PRUNING_ARMS]
        short += [arm(f"C1-free-{a}", w, a, free=True) for a in ("off", "prune")]
    for w in ("deepk", "or002"):
        short += [arm(f"C1-esco-{a}", w, a, sparse_from_k=3, expect_transition=True) for a in PRUNING_ARMS]
    long_ += [arm(f"C1-{a}", "dsl", a) for a in PRUNING_ARMS]
    long_ += [arm(f"C1-free-{a}", "dsl", a, free=True) for a in ("off", "prune")]
    long_ += [arm(f"C1-{a}", "oom2ml3", a) for a in PRUNING_ARMS]
    long_.append(arm("C1-prune", "sk2ml3", "prune", timeout_s=3600))
    long_.append(arm("C1-off", "sk2ml3", "off", timeout_s=3600, single_rep=True))
    if n_dev >= 2:
        long_ += [arm(f"C2-{a}", "dsl", a, n_gpus=2) for a in PRUNING_ARMS]
        long_ += [arm(f"C2-{a}", "oom2ml3", a, n_gpus=2) for a in ("off", "prune")]
        long_.append(arm("C2-prune", "sk2ml3", "prune", n_gpus=2, timeout_s=3600, single_rep=True))
    cfgs = []
    for rep in range(REPS):
        for c in short + long_:
            if rep and c.get("single_rep"):
                continue
            cfgs.append({**c, "id": f"{c['base_id']}#r{rep}", "rep": rep})
    return cfgs


def build_esco_matrix(n_dev: int, *, retail_low: bool = False) -> list[dict]:
    """Dense and ESCO on the same workloads, repeats and result-signature gate.

    Low-support Retail runs cover K=2, K=3 and K=4. Repeated baskets alone
    imply at least 2**53-1 / 2**93-1 itemsets at the two supports, so the
    comparison uses explicit depth limits. Shallow runs precede deeper ones;
    OOM/timeouts fail the gate rather than count as wins.
    """
    if retail_low:
        workloads = tuple(f"or{label}-{depth}" for depth in ("k2", "k3", "k4")
                          for label in ("0001", "00005"))
    else:
        workloads = DEEP + ("wide",)
    base = []
    for w in workloads:
        for n in ([1, 2] if n_dev >= 2 else [1]):
            prefix = f"C{n}"
            base.append(_cfg(f"{prefix}-dense", w, "C", n_gpus=n))
            base.append(_cfg(f"{prefix}-dense-tiled", w, "C", n_gpus=n,
                             env={"ET_MINER_TILED_MIN_GROUP_PAIRS": "0"}))
            base.append(_cfg(f"{prefix}-dense-percand", w, "C", n_gpus=n,
                             env={"ET_MINER_TILED_MIN_GROUP_PAIRS": _NO_GROUP}))
            base.append(_cfg(f"{prefix}-esco-auto", w, "C", n_gpus=n, sparse_from_k="auto"))
            base.append(_cfg(f"{prefix}-esco-k3", w, "C", n_gpus=n, sparse_from_k=3,
                             expect_transition=WORKLOADS[w][2] != 2))
        if w in CPU or (retail_low and WORKLOADS[w][2] == 2):
            base.append(_cfg("F-sparse", w, "F", sparse=True, n_jobs=6))
        if retail_low and WORKLOADS[w][2] == 2:
            base.append(_cfg("EA", w, "EA"))
    return [{**c, "id": f"{c['base_id']}#r{rep}", "rep": rep} for rep in range(REPS) for c in base]
