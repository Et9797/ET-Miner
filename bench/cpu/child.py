"""CPU-tier bench child: one arm, one dataset, one timed mining call from in-memory input.

Called by ``bench/child_run.py`` for configs with ``"mode": "cpu"``. Each
config names an arm:

    F    ET-Miner's CPU route   apriori(df, use_gpu=False, sparse=..., n_jobs=..., profile=True)
    EA   efficient-apriori       itemsets_from_transactions(tuples, (min_count - 0.5) / N, max_length)

Both are timed from their own in-memory input format: a Polars DataFrame with
an ``items`` list column for F, a list of tuples for EA. EA's conversion
(``to_list()`` then ``tuple`` per row) is timed separately as ``convert_s``.
EA's max_length is the config's, or the longest transaction when the config
has none (the exact bound; efficient-apriori defaults to 8). The result frame
is built after the timer stops.

Before the timed call the child runs the same arm on a 20,000-row slice of
``smoke``. The row records wall time, per-level phases (``bench/cpu/split.py``
and the ProfilingSession of ``profile=True``), process CPU seconds during the
call, peak RSS (sampled every 0.1 s during the call, and the process
high-water mark), the result signature, the prefix-group shape per level, and
any fallback the run logged. Each closed level is also written to
``<result>.partial.json``, so a run killed at its timeout keeps the levels it
finished.

Usage: invoked by bench/runner.py --mode cpu-baseline; not run by hand.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from child_run import result_signatures  # noqa: E402
from consolidation_run import _FALLBACK_RE, _group_stats, _host, _load  # noqa: E402

WARMUP_ROWS = 20_000


class RssSampler(threading.Thread):
    """Samples this process's RSS every 0.1 s until stopped."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        import psutil

        self._proc = psutil.Process()
        self.start_mb = self._proc.memory_info().rss / 2**20
        self.peak_mb = self.start_mb
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                self.peak_mb = max(self.peak_mb, self._proc.memory_info().rss / 2**20)
            except Exception:
                pass
            self._halt.wait(0.1)

    def stop(self) -> None:
        self._halt.set()


def _ea_min_support(min_support: float, n_rows: int) -> float:
    from et_miner.core.result import _min_count

    return (_min_count(min_support, n_rows) - 0.5) / n_rows


def _ea_frame(itemsets: dict, n_rows: int):
    import polars as pl

    rows = [(list(key), count / n_rows) for level in itemsets.values() for key, count in level.items()]
    return pl.DataFrame(rows, schema={"itemset": pl.List(pl.Int64), "support": pl.Float64}, orient="row")


def _mine(cfg: dict, df, min_support: float, max_length, level_cb, out: dict):
    """Run the arm once; returns the result frame. Fills ``out`` with timings."""
    if cfg["route"] == "EA":
        from efficient_apriori.itemsets import itemsets_from_transactions

        import polars as pl

        n = df.height
        ml = max_length or int(df.select(pl.col("items").list.len().max()).item() or 1)
        t0 = time.perf_counter()
        tx = [tuple(r) for r in df["items"].to_list()]
        out["convert_s"] = round(time.perf_counter() - t0, 6)
        split = out.get("_ea_split")
        if split is not None:
            split.install()
        try:
            t0 = time.perf_counter()
            itemsets, _ = itemsets_from_transactions(tx, _ea_min_support(min_support, n), max_length=ml)
            t1 = time.perf_counter()
        finally:
            if split is not None:
                split.uninstall()
        out["mine_s"] = t1 - t0
        out["_t_end"] = t1
        return _ea_frame(itemsets, n)

    from et_miner.core.apriori import apriori

    t0 = time.perf_counter()
    res, session = apriori(
        df, min_support=min_support, max_length=max_length, use_gpu=False, sparse=cfg.get("sparse"),
        n_jobs=int(cfg.get("n_jobs", 1)), level_callback=level_cb, profile=True,
    )
    out["mine_s"] = time.perf_counter() - t0
    out["profile"] = session.to_dict() if session is not None else None
    return res


def _warmup(cfg: dict) -> None:
    import polars as pl

    from consolidation_run import REPO

    df = pl.read_parquet(REPO / "datasets" / "synth" / "smoke.parquet").head(WARMUP_ROWS)
    _mine(cfg, df, 0.02, 3, None, {})


def run_cpu(cfg: dict) -> int:
    from loguru import logger

    import et_miner  # noqa: F401 -- import before replacing the sinks
    from split import CpuSplit, EaSplit

    fallbacks: list[str] = []

    def _sink(message):
        text = message.record["message"]
        if _FALLBACK_RE.search(text):
            fallbacks.append(f"{message.record['level'].name}: {text[:300]}")

    logger.remove()
    logger.add(sys.stderr, level=cfg.get("log_level", "INFO"))

    df, n_rows, sidecar = _load(cfg["dataset"])
    min_support = float(cfg["min_support"])
    max_length = cfg.get("max_length")
    partial = Path(cfg["result_path"] + ".partial.json") if cfg.get("result_path") else None

    status = "ok"
    try:
        _warmup(cfg)
    except Exception as e:  # noqa: BLE001 -- recorded, parent decides
        status = f"warmup error: {type(e).__name__}: {e}"

    levels: list[dict] = []
    out: dict = {}
    split = EaSplit() if cfg["route"] == "EA" else CpuSplit()

    def level_cb(k, n_cand, n_freq, ms):
        levels.append({"k": k, "n_candidates": int(n_cand), "n_frequent": int(n_freq), "ms": float(ms)})
        if partial is not None:
            partial.write_text(json.dumps({"levels": levels, "split": split.levels}) + "\n")

    sink_id = logger.add(_sink, level="DEBUG")
    signatures: dict = {}
    timings: dict = {}
    motifs_ok = True
    wall_s = 0.0
    sampler = RssSampler()
    sampler.start()
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    try:
        if status != "ok":
            raise RuntimeError(status)
        if cfg["route"] == "EA":
            out["_ea_split"] = split
            res = _mine(cfg, df, min_support, max_length, None, out)
            ea = split.levels(out["_t_end"])
            timings["ea_index_s"] = ea["index"]
            timings["split"] = ea["levels"]
            levels = [{"k": int(k), "n_candidates": v["n_candidates"], "ms": 1000 * (v["cand_gen"] + v["count"])}
                      for k, v in ea["levels"].items()]
        else:
            split.install()
            try:
                res = _mine(cfg, df, min_support, max_length, split.callback(level_cb), out)
            finally:
                split.uninstall()
            timings["split"] = split.levels
            timings["tail"] = split.tail
            timings["profile"] = out.get("profile")
        wall_s = out["mine_s"]
        ru1 = resource.getrusage(resource.RUSAGE_SELF)
        timings["cpu_s"] = round((ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime), 3)
        if "convert_s" in out:
            timings["convert_s"] = out["convert_s"]
        bad = [list(x) for x in res["itemset"].head(100_000).to_list() if list(x) != sorted(x)]
        if bad:
            raise AssertionError(f"itemsets not ascending, e.g. {bad[:3]}")
        signatures = result_signatures(res, n_rows)
        timings["group_stats"] = _group_stats(res)
        if sidecar and sidecar.get("planted") and max_length is None:
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
        if fallbacks:
            status = f"fallback: {fallbacks[0]}"
    except Exception as e:  # noqa: BLE001 -- recorded, parent decides
        if status == "ok":
            status = f"error: {type(e).__name__}: {e}"
        wall_s = out.get("mine_s", 0.0)
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
            "rss_start_mb": round(sampler.start_mb, 1),
            "peak_rss_mb": round(sampler.peak_mb, 1),
            "ru_maxrss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "fallbacks": fallbacks[:20],
            "motifs_ok": motifs_ok,
            "host": _host(),
            "polars": __import__("polars").__version__,
            "rust": __import__("et_miner.backends", fromlist=["RUST_INSTALLED"]).RUST_INSTALLED,
            **signatures,
        }
    )
    if cfg.get("result_path"):
        Path(cfg["result_path"]).write_text(payload + "\n")
    print(payload)
    if status != "ok":
        return 1
    return 0 if motifs_ok else 3


def main() -> int:
    cfg = json.loads(sys.argv[1])
    for k, v in cfg.get("env", {}).items():
        os.environ[k] = str(v)
    return run_cpu(cfg)


if __name__ == "__main__":
    sys.exit(main())

