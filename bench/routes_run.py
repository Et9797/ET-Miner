"""Routes-mode child: one route and arm on one dataset, one timed call.

Called by ``child_run.py`` for configs with ``"mode": "routes"``. Routes:

    C          row-split miner          apriori(use_gpu=True, n_gpus=N, ...)
    C-bitvecs  row-split from bitvecs   apriori(bitvecs=..., n_gpus=N); the bitvector build is timed
                                        separately (``bitvec_build_s``) and is outside ``wall_s``
    D          SON, one device          apriori(streaming=True, chunk_size=..., use_gpu=...)
    E          SON, several GPUs        apriori(streaming=True, n_gpus=N, chunk_size=...)
    F          CPU array miner          apriori(use_gpu=False, n_jobs=...)
    RUST       all-Rust miner           apriori_from_csr on the frame's CSR; ``wall_s`` includes the CSR
                                        (``csr_s``)

Config keys beyond the consolidation child's (``consolidation_run.py``):

    input       "int" (default), "str" (every item cast to String) or "wide" (every id times WIDE, a span
                beyond 2**22); cast after the read, outside ``wall_s``; the signature maps the items back
    overrides   {"module:ATTR": value} set before the warm-up (the CPU arms; lists become tuples)
    use_gpu     SON's passes (route D)
    n_jobs      the CPU route's and SON's CPU passes' workers

Before the timed call the child compiles every registered kernel on every device a GPU route uses and runs
the same route and arm on a 20,000-row slice of ``smoke``. A logged fallback fails the config
(``consolidation_run.FALLBACK_PATTERNS``). The row records ``wall_s``, ``cpu_s`` (user + system seconds of
the process during the call), the level split (row-split routes) or the SON passes, per-device peak VRAM,
peak RSS, throttle flags, the emitted item dtype, the host and the result signature.
"""

from __future__ import annotations

import importlib
import json
import math
import re
import resource
import sys
import time
from pathlib import Path

from child_run import VramSampler, result_signatures
from consolidation_run import _FALLBACK_RE, WARMUP_ROWS, _group_stats, _host, _load

#: The "wide" input multiplies every id by this prime: ids up to 10**4 span beyond 2**22 and stay in Int64.
WIDE = 10_000_019


def _cpu_seconds() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def _cgroup() -> dict:
    out = {}
    for name in ("cpu.max", "memory.max"):
        try:
            out[name] = Path("/sys/fs/cgroup", name).read_text().strip()
        except OSError:
            pass
    return out


def _apply_overrides(overrides: dict) -> None:
    for key, value in overrides.items():
        module, attr = key.split(":")
        if isinstance(value, list):
            value = tuple(tuple(v) if isinstance(v, list) else v for v in value)
        mod = importlib.import_module(module)
        if not hasattr(mod, attr):
            raise AttributeError(f"override {key}: {module} has no {attr}")
        setattr(mod, attr, value)


def _cast(df, variant: str):
    import polars as pl

    if variant == "str":
        return df.select(pl.col("items").list.eval(pl.element().cast(pl.String)))
    if variant == "wide":
        return df.select(pl.col("items").list.eval(pl.element().cast(pl.Int64) * WIDE))
    return df


def _decoded(res, variant: str):
    import polars as pl

    if variant == "wide":
        return res.with_columns(pl.col("itemset").list.eval(pl.element() // WIDE))
    if variant == "str":
        return res.with_columns(pl.col("itemset").list.eval(pl.element().cast(pl.Int64)))
    return res


def _bitvecs(df, min_support: float):
    import cupy as cp

    from et_miner.core.matrix import _build_csr_from_transactions
    from et_miner.gpu.bitvec import _build_gpu_bitvec_matrix

    csr, idx_to_item, n = _build_csr_from_transactions(df.lazy(), min_support, "items")
    with cp.cuda.Device(0):
        bv = _build_gpu_bitvec_matrix(csr)
    return bv, idx_to_item, n


def _rust(df, min_support: float, max_length, timings: dict):
    import numpy as np
    import polars as pl

    from et_miner import apriori_from_csr

    t0 = time.perf_counter()
    col = df.get_column("items")
    lens = col.list.len().fill_null(0).to_numpy().astype(np.int64)
    indptr = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=indptr[1:])
    indices = col.explode().drop_nulls().to_numpy().astype(np.int64)
    n_cols = int(indices.max()) + 1 if len(indices) else 0
    timings["csr_s"] = round(time.perf_counter() - t0, 3)
    t1 = time.perf_counter()
    itemsets, counts = apriori_from_csr(indptr, indices, len(lens), n_cols, min_support, max_length or 0)
    timings["mine_s"] = round(time.perf_counter() - t1, 3)
    n = len(lens)
    return pl.DataFrame(
        {"itemset": pl.Series(itemsets, dtype=pl.List(pl.Int64)), "support": np.asarray(counts, dtype=np.float64) / n}
    )


def _mine(cfg: dict, df, min_support: float, max_length, level_cb, progress_cb, timings: dict):
    from et_miner.core.apriori import apriori

    route = cfg["route"]
    n_gpus = int(cfg.get("n_gpus", 1))
    common = dict(min_support=min_support, max_length=max_length)
    options = dict(
        prune_equal_support=cfg.get("prune_equal_support", False),
        use_generator_pruning=cfg.get("use_generator_pruning", False),
    )
    if route == "RUST":
        return _rust(df, min_support, max_length, timings)
    if route == "C-bitvecs":
        t0 = time.perf_counter()
        bv, col_to_item, n = _bitvecs(df, min_support)
        timings["bitvec_build_s"] = round(time.perf_counter() - t0, 3)
        timings["t_mine"] = time.perf_counter()
        return apriori(bitvecs=(bv, col_to_item, n), n_gpus=n_gpus, level_callback=level_cb,
                       prune_apriori=cfg.get("prune_apriori", True), sparse_from_k=cfg.get("sparse_from_k"),
                       **options, **common)
    if route == "C":
        return apriori(df, use_gpu=True, n_gpus=n_gpus, prune_apriori=cfg.get("prune_apriori", True),
                       sparse_from_k=cfg.get("sparse_from_k"), level_callback=level_cb, **options, **common)
    if route in ("D", "E"):
        return apriori(df, streaming=True, chunk_size=int(cfg["chunk_size"]), n_gpus=n_gpus if route == "E" else 1,
                       use_gpu=cfg.get("use_gpu", True) or route == "E", n_jobs=int(cfg.get("n_jobs", 1)),
                       show_progress=False, progress_callback=progress_cb, **common)
    if route == "F":
        return apriori(df, use_gpu=False, n_jobs=int(cfg.get("n_jobs", 1)), level_callback=level_cb,
                       **options, **common)
    raise ValueError(f"unknown route {route!r}")


def _gpu_route(cfg: dict) -> bool:
    return cfg["route"] in ("C", "C-bitvecs", "E") or (cfg["route"] == "D" and cfg.get("use_gpu", True))


def _warmup(cfg: dict) -> None:
    import polars as pl

    from child_run import REPO

    if _gpu_route(cfg):
        import cupy as cp

        from et_miner.gpu.kernels.loader import _KERNEL_FILES, get_cuda_kernel

        for d in range(int(cfg.get("n_gpus", 1))):
            with cp.cuda.Device(d):
                for name in sorted(_KERNEL_FILES):
                    get_cuda_kernel(name).kernel
    df = _cast(pl.read_parquet(REPO / "datasets" / "synth" / "smoke.parquet").head(WARMUP_ROWS), cfg.get("input", "int"))
    warm = {**cfg}
    if cfg["route"] in ("D", "E"):
        warm["chunk_size"] = WARMUP_ROWS // 4
    _mine(warm, df, 0.02, 3, None, None, {})


def run_routes(cfg: dict) -> int:
    from loguru import logger

    import et_miner  # noqa: F401 -- import before replacing the sinks

    fallbacks: list[str] = []
    transitions: list[str] = []
    allowed = re.compile("|".join(cfg["allow_fallback"]), re.IGNORECASE) if cfg.get("allow_fallback") else None

    def _sink(message):
        text = message.record["message"]
        if "DENSITY TRANSITION" in text:
            transitions.append(text)
        if _FALLBACK_RE.search(text) and not (allowed and allowed.search(text)):
            fallbacks.append(f"{message.record['level'].name}: {text[:300]}")

    logger.remove()
    logger.add(sys.stderr, level=cfg.get("log_level", "INFO"))

    status = "ok"
    timings: dict = {}
    try:
        _apply_overrides(cfg.get("overrides", {}))
    except Exception as e:  # noqa: BLE001 -- recorded, parent decides
        status = f"override error: {type(e).__name__}: {e}"
    variant = cfg.get("input", "int")
    t_read = time.perf_counter()
    df, n_rows, sidecar = _load(cfg["dataset"])
    timings["read_s"] = round(time.perf_counter() - t_read, 3)
    t_cast = time.perf_counter()
    df = _cast(df, variant)
    timings["cast_s"] = round(time.perf_counter() - t_cast, 3)
    min_support = float(cfg["min_support"])
    max_length = cfg.get("max_length")

    if status == "ok":
        try:
            _warmup(cfg)
        except Exception as e:  # noqa: BLE001 -- recorded, parent decides
            status = f"warmup error: {type(e).__name__}: {e}"

    levels: list[dict] = []
    events: list[tuple[str, int, int, float]] = []

    def level_cb(k, n_cand, n_freq, ms):
        levels.append({"k": k, "n_candidates": int(n_cand), "n_frequent": int(n_freq), "ms": float(ms)})

    def progress_cb(phase, chunk_idx, n_chunks, metrics):
        events.append((phase, int(chunk_idx), int(n_chunks), time.perf_counter()))

    split = None
    if cfg.get("level_split") and cfg["route"] in ("C", "C-bitvecs"):
        from level_split import LevelSplit

        split = LevelSplit()
        level_cb = split.callback(level_cb)

    sink_id = logger.add(_sink, level="DEBUG")
    sampler = VramSampler()
    sampler.start()
    signatures: dict = {}
    motifs_ok = True
    item_dtype = None
    cpu0 = _cpu_seconds()
    t0 = time.perf_counter()
    wall_s = 0.0
    try:
        if status != "ok":
            raise RuntimeError(status)
        if split is not None:
            split.install()
            split.start()
        res = _mine(cfg, df, min_support, max_length, level_cb, progress_cb, timings)
        wall_s = time.perf_counter() - timings.pop("t_mine", t0)
        if split is not None:
            split.finish()
            split.uninstall()
            timings["level_split"] = split.levels
            timings["span"] = split.span
        timings["density_transitions"] = transitions
        if cfg.get("expect_transition") and not transitions:
            raise AssertionError("ESCO config completed without entering the sparse CSR path")
        item_dtype = str(res.schema["itemset"])
        bad = [list(x) for x in res["itemset"].head(100_000).to_list() if list(x) != sorted(x)]
        if bad:
            raise AssertionError(f"itemsets not ascending, e.g. {bad[:3]}")
        res = _decoded(res, variant)
        signatures = result_signatures(res, n_rows)
        timings["group_stats"] = _group_stats(res)
        if sidecar and sidecar.get("planted") and max_length is None and not cfg.get("prune_equal_support"):
            counts = {
                tuple(sorted(int(i) for i in s)): round(sup * n_rows)
                for s, sup in zip(res["itemset"].to_list(), res["support"].to_list())
            }
            from et_miner.core.result import _min_count

            # A workload above the preset's support (deepk15) leaves motifs planted below its min_count out.
            min_count = _min_count(min_support, n_rows)
            for entry in sidecar["planted"]:
                if entry["planted_count"] < min_count:
                    continue
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
        cpu_s = _cpu_seconds() - cpu0
        timings["density_transitions"] = transitions
        if split is not None:
            split.uninstall()
        sampler.stop()
        sampler.join(timeout=3)
        logger.remove(sink_id)

    payload = json.dumps(
        {
            "id": cfg["id"],
            "config": cfg,
            "status": status,
            "wall_s": round(wall_s, 3),
            "cpu_s": round(cpu_s, 3),
            "levels": levels,
            "levels_ms_k2plus": round(sum(lv["ms"] for lv in levels if lv["k"] >= 2), 1),
            "timings": timings,
            "item_dtype": item_dtype,
            "peak_vram_mb": sampler.peak_mb,
            "peak_rss_mb": round(sampler.peak_rss_mb, 1),
            "ru_maxrss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "throttle_reasons": sorted(sampler.throttle_reasons),
            "throttled_devices": sampler.throttled_devices(),
            "fallbacks": fallbacks[:20],
            "motifs_ok": motifs_ok,
            "host": {**_host(), "cgroup": _cgroup(),
                     "threads_extra": {k: __import__("os").environ.get(k) for k in ("OPENBLAS_NUM_THREADS",)}},
            **signatures,
        },
        default=lambda o: None if isinstance(o, float) and not math.isfinite(o) else str(o),
    )
    if cfg.get("result_path"):
        Path(cfg["result_path"]).write_text(payload + "\n")
    print(payload)
    if status != "ok":
        return 1
    return 0 if motifs_ok else 3
