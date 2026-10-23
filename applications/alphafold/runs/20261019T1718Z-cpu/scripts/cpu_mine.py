"""In-core CPU mining run with the pinned ET-Miner engine (public `apriori` only).

Loads the `items` column, asserts the min_count conversion, mines every level
on THREADS threads with no device, and writes canonical per-K tables plus a
meta JSON (per-level times, wall, CPU seconds, peak RSS, K distribution,
signature). Per-level progress is appended to OUT/levels.jsonl as it happens.

Usage:
    run_py.sh cpu_mine.py DATA.parquet OUT_DIR --min-count M [options]

Options:
    --min-support S          round support to use if it hits M (else (M - 0.5) / N)
    --head N                 mine only the first N rows (as validate_row_split.py)
    --use-generator-pruning  count inference (exact; off by default)
    --label TEXT             run id recorded in the meta JSON
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as h  # noqa: E402

THREADS = h.check_env()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--min-count", type=int, required=True)
    ap.add_argument("--min-support", type=float)
    ap.add_argument("--head", type=int)
    ap.add_argument("--use-generator-pruning", action="store_true")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    from et_miner import apriori

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = h.RunStats()
    df = h.load_items(a.data, a.head)
    n_rows = df.height
    s = h.min_support_for(a.min_count, n_rows, a.min_support)
    load_s = stats.snapshot()["wall_s"]
    levels = []
    log = open(out / "levels.jsonl", "w")

    def on_level(k, n_candidates, n_frequent, duration_ms):
        rec = {"k": k, "candidates": int(n_candidates), "frequent": int(n_frequent),
               "ms": float(duration_ms), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        levels.append(rec)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    t0 = time.perf_counter()
    res = apriori(df, min_support=s, max_length=None, use_gpu=False, n_jobs=THREADS,
                  level_callback=on_level, show_progress=False,
                  use_generator_pruning=a.use_generator_pruning)
    mine_s = time.perf_counter() - t0
    run = stats.snapshot()
    del df
    tables = h.normalize(res, n_rows, sort_items=False)
    del res
    h.write_tables(tables, out / "parquet", n_rows)
    meta = {
        "label": a.label, "data": str(Path(a.data).resolve()), "head": a.head, "n_rows": n_rows,
        "min_count": a.min_count, "min_support": s, "max_length": None,
        "use_generator_pruning": a.use_generator_pruning, "n_jobs": THREADS,
        "load_s": load_s, "mine_wall_s": mine_s, "process": run, "levels": levels,
        "signature": h.signature(tables), "env": h.env_record(THREADS),
    }
    meta["k_max"] = max(tables) if tables else 0
    meta["process_end"] = stats.snapshot()
    h.dump_json(meta, out / "meta.json")
    print(json.dumps({"label": a.label, "itemsets": meta["signature"]["itemsets"], "k_max": meta["k_max"],
                      "mine_wall_s": round(mine_s, 2), "peak_rss_mb": round(run["peak_rss_mb"])}))


if __name__ == "__main__":
    main()
