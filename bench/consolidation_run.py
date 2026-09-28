"""Consolidation-mode child: one route, one dataset, one timed run.

Called by ``child_run.py`` for configs with ``"mode": "consolidation"``. The
route is named explicitly rather than inferred from ``n_gpus``:

    C          row-split miner         apriori(use_gpu=True, n_gpus=N)
    C-bitvecs  row-split from bitvecs  apriori(bitvecs=..., n_gpus=N); the
                                       bitvec build is timed separately
    D        SON, one GPU or CPU       apriori(streaming=True, chunk_size=...)
    E        SON, multi-GPU            apriori(streaming=True, n_gpus=N, chunk_size=...)
    F        CPU                       apriori(use_gpu=False, sparse=..., n_jobs=...)

Before the timed call the child compiles every registered kernel on each
device the route uses and runs the same route on a 20,000-row slice of the
``smoke`` preset. A fallback logged during the timed call (see
``FALLBACK_PATTERNS``) fails the config unless the config allows it.
"""

from __future__ import annotations

import json
import os
import platform
import re
import resource
import sys
import time
from pathlib import Path

from child_run import REPO, VramSampler, result_signatures

#: Log messages that mean a slower or different code path ran than the one the
#: config names. Matched case-insensitively against every record at DEBUG and up.
FALLBACK_PATTERNS = (
    r"falling back",
    r"fallback",
    r"sliced CPU valve",
    r"NCCL init failed",
    r"NCCL unavailable",
    r"failed for chunk",
    r"chunk mining failed",
    r"counting failed",
    r"bitvec build failed",
    r"is stale",
    r"is not installed",
)
_FALLBACK_RE = re.compile("|".join(FALLBACK_PATTERNS), re.IGNORECASE)

WARMUP_ROWS = 20_000


def _devices(cfg: dict) -> list[int]:
    if cfg["route"] == "F":
        return []
    n = int(cfg.get("n_gpus", 1))
    return list(range(n))


def _load(name: str):
    """(DataFrame with an `items` column, n_rows, sidecar or None)."""
    import polars as pl

    if name == "online_retail":
        df = pl.read_parquet(REPO / "datasets" / "online_retail_ii" / "transactions.parquet")
        df = df.select(pl.col("items").cast(pl.List(pl.Int64)))
        return df, df.height, None
    if name == "alphafold":
        df = pl.read_parquet(os.environ["ET_BENCH_ALPHAFOLD"]).select("items")
        return df, df.height, None
    from et_miner.synthetic import PRESETS

    df = pl.read_parquet(REPO / "datasets" / "synth" / f"{name}.parquet")
    sidecar = json.loads((REPO / "datasets" / "synth" / f"{name}.json").read_text())
    return df, PRESETS[name].n_rows, sidecar


def _bitvecs(df, min_support: float):
    """(bitvecs on device 0, col_to_item, n_rows) — the `bitvecs=` input."""
    import cupy as cp

    from et_miner.core.matrix import _build_csr_from_transactions
    from et_miner.gpu.bitvec import _build_gpu_bitvec_matrix

    csr, idx_to_item, n = _build_csr_from_transactions(df.lazy(), min_support, "items")
    with cp.cuda.Device(0):
        bv = _build_gpu_bitvec_matrix(csr)
    return bv, idx_to_item, n


def _mine(cfg: dict, df, min_support: float, max_length, level_cb, progress_cb, timings: dict):
    from et_miner.core.apriori import apriori

    route = cfg["route"]
    n_gpus = int(cfg.get("n_gpus", 1))
    common = dict(min_support=min_support, max_length=max_length)
    if route == "C-bitvecs":
        t0 = time.perf_counter()
        bv, col_to_item, n = _bitvecs(df, min_support)
        timings["bitvec_build_s"] = round(time.perf_counter() - t0, 3)
        t0 = time.perf_counter()
        res = apriori(bitvecs=(bv, col_to_item, n), n_gpus=n_gpus, level_callback=level_cb, **common)
        timings["mine_s"] = round(time.perf_counter() - t0, 3)
        return res
    if route == "C":
        return apriori(df, use_gpu=True, n_gpus=n_gpus,
                       prune_equal_support=cfg.get("prune_equal_support", False),
                       level_callback=level_cb, **common)
    if route in ("D", "E"):
        return apriori(df, streaming=True, chunk_size=int(cfg["chunk_size"]), n_gpus=n_gpus if route == "E" else 1,
                       use_gpu=cfg.get("use_gpu", True),
                       show_progress=False, progress_callback=progress_cb, **common)
    if route == "F":
        return apriori(df, use_gpu=False, sparse=cfg.get("sparse"), n_jobs=int(cfg.get("n_jobs", 1)),
                       level_callback=level_cb, **common)
    raise ValueError(f"unknown route {route!r}")


def _warmup(cfg: dict) -> None:
    import polars as pl

    if cfg["route"] != "F":
        import cupy as cp

        from et_miner.gpu.kernels.loader import _KERNEL_FILES, get_cuda_kernel

        for d in _devices(cfg):
            with cp.cuda.Device(d):
                for name in sorted(_KERNEL_FILES):
                    get_cuda_kernel(name).kernel
    df = pl.read_parquet(REPO / "datasets" / "synth" / "smoke.parquet").head(WARMUP_ROWS)
    warm = {**cfg}
    if cfg["route"] in ("D", "E"):
        warm["chunk_size"] = WARMUP_ROWS // 4
    _mine(warm, df, 0.02, 3, None, None, {})


def _group_stats(res) -> dict:
    """Per level K>=2, the prefix-group shape the level is generated from.

    Computed from the emitted (K-1)-level, so it describes what every miner
    knows before level K runs (C's Apriori prune only shrinks it further).
    """
    import numpy as np
    import polars as pl

    lens = res.select(pl.col("itemset").list.len().alias("k"), "itemset")
    out: dict[int, dict] = {}
    max_k = int(lens["k"].max() or 0)
    for k_prev in range(1, max_k + 1):
        level = lens.filter(pl.col("k") == k_prev)
        if k_prev == 1:
            sizes = np.array([level.height], dtype=np.int64)
        else:
            sizes = (
                level.select(pl.col("itemset").list.head(k_prev - 1).alias("prefix"))
                .group_by("prefix")
                .len()["len"]
                .to_numpy()
                .astype(np.int64)
            )
        sizes = sizes[sizes >= 2]
        if len(sizes) == 0:
            continue
        pairs = sizes * (sizes - 1) // 2
        total = int(pairs.sum())
        out[k_prev + 1] = {
            "groups": int(len(sizes)),
            "candidates": total,
            "median_suffixes": float(np.median(sizes)),
            "max_suffixes": int(sizes.max()),
            "frac_groups_lt64_pairs": round(float((pairs < 64).mean()), 4),
            "frac_cands_in_ge64_pair_groups": round(float(pairs[pairs >= 64].sum() / total), 4) if total else 0.0,
        }
    return out


def _host() -> dict:
    model = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "nproc": len(os.sched_getaffinity(0)),
        "cpu_count": os.cpu_count(),
        "cpu_model": model,
        "python": platform.python_version(),
        "threads": {k: os.environ.get(k) for k in
                    ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")},
    }


def run_consolidation(cfg: dict) -> int:
    from loguru import logger

    import et_miner  # noqa: F401 -- import before replacing the sinks

    fallbacks: list[str] = []
    allowed = re.compile("|".join(cfg["allow_fallback"]), re.IGNORECASE) if cfg.get("allow_fallback") else None

    def _sink(message):
        text = message.record["message"]
        if _FALLBACK_RE.search(text) and not (allowed and allowed.search(text)):
            fallbacks.append(f"{message.record['level'].name}: {text[:300]}")

    logger.remove()
    logger.add(sys.stderr, level=cfg.get("log_level", "INFO"))

    dataset = cfg["dataset"]
    df, n_rows, sidecar = _load(dataset)
    min_support = float(cfg["min_support"])
    max_length = cfg.get("max_length")

    status = "ok"
    try:
        _warmup(cfg)
    except Exception as e:  # noqa: BLE001 -- recorded, parent decides
        status = f"warmup error: {type(e).__name__}: {e}"

    levels: list[dict] = []
    events: list[tuple[str, int, int, float]] = []
    timings: dict = {}

    def level_cb(k, n_cand, n_freq, ms):
        levels.append({"k": k, "n_candidates": int(n_cand), "n_frequent": int(n_freq), "ms": float(ms)})

    def progress_cb(phase, chunk_idx, n_chunks, metrics):
        events.append((phase, int(chunk_idx), int(n_chunks), time.perf_counter()))

    sink_id = logger.add(_sink, level="DEBUG")
    sampler = VramSampler()
    sampler.start()
    signatures: dict = {}
    motifs_ok = True
    t0 = time.perf_counter()
    wall_s = 0.0
    try:
        if status != "ok":
            raise RuntimeError(status)
        res = _mine(cfg, df, min_support, max_length, level_cb, progress_cb, timings)
        wall_s = time.perf_counter() - t0
        bad = [list(x) for x in res["itemset"].head(100_000).to_list() if list(x) != sorted(x)]
        if bad:
            raise AssertionError(f"itemsets not ascending, e.g. {bad[:3]}")
        signatures = result_signatures(res, n_rows)
        timings["group_stats"] = _group_stats(res)
        if sidecar and sidecar.get("planted") and max_length is None and not cfg.get("prune_equal_support"):
            counts = {
                tuple(sorted(int(i) for i in s)): round(sup * n_rows)
                for s, sup in zip(res["itemset"].to_list(), res["support"].to_list())
            }
            for entry in sidecar["planted"]:
                got = counts.get(tuple(sorted(entry["items"])))
                if got is None or got < entry["planted_count"]:
                    motifs_ok = False
                    print(f"MOTIF FAILURE: {entry['items']} mined={got} planted={entry['planted_count']}",
                          file=sys.stderr)
        if cfg["route"] in ("D", "E"):
            n_chunks = max((e[2] for e in events), default=0)
            p1 = [e[3] for e in events if e[0] == "pass1"]
            p2 = [e[3] for e in events if e[0] == "pass2"]
            timings["n_chunks"] = n_chunks
            timings["pass1_s"] = round(max(p1) - t0, 3) if p1 else None
            timings["pass2_s"] = round(max(p2) - max(p1), 3) if p1 and p2 else None
            if cfg.get("expect_chunks") and n_chunks != cfg["expect_chunks"]:
                raise AssertionError(f"expected {cfg['expect_chunks']} SON chunks, ran {n_chunks}")
        if fallbacks:
            status = f"fallback: {fallbacks[0]}"
    except Exception as e:  # noqa: BLE001 -- recorded, parent decides
        if status == "ok":
            status = f"error: {type(e).__name__}: {e}"
        wall_s = time.perf_counter() - t0
    finally:
        sampler.stop()
        sampler.join(timeout=3)
        logger.remove(sink_id)

    payload = json.dumps(
        {
            "id": cfg["id"],
            "config": cfg,
            "status": status,
            "wall_s": round(wall_s, 3),
            "levels": levels,
            "levels_ms_k2plus": round(sum(lv["ms"] for lv in levels if lv["k"] >= 2), 1),
            "timings": timings,
            "peak_vram_mb": sampler.peak_mb,
            "peak_rss_mb": round(sampler.peak_rss_mb, 1),
            "ru_maxrss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "throttle_reasons": sorted(sampler.throttle_reasons),
            "throttled_devices": sampler.throttled_devices(),
            "fallbacks": fallbacks[:20],
            "motifs_ok": motifs_ok,
            "host": _host(),
            **signatures,
        }
    )
    if cfg.get("result_path"):
        Path(cfg["result_path"]).write_text(payload + "\n")
    print(payload)
    if status != "ok":
        return 1
    return 0 if motifs_ok else 3
