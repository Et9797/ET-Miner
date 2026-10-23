"""CPU SON streaming run with the pinned ET-Miner engine (public `apriori_streaming` only).

Loads the `items` column, asserts the min_count conversion and the chunk
count, runs both SON passes on THREADS threads with no device, and writes the
same canonical per-K tables and meta JSON as cpu_mine.py, with per-pass and
per-chunk times from the progress callback (also appended to OUT/progress.jsonl).

Usage:
    run_py.sh cpu_son.py DATA.parquet OUT_DIR --min-count M --chunk-size C --chunks K [options]

Options:
    --min-support S            round support to use if it hits M (else (M - 0.5) / N)
    --local-support-factor F   SON local factor (default 0.9, as the GPU run)
    --head N                   use only the first N rows
    --label TEXT               run id recorded in the meta JSON
"""

import argparse
import json
import math
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
    ap.add_argument("--chunk-size", type=int, required=True)
    ap.add_argument("--chunks", type=int, required=True)
    ap.add_argument("--min-support", type=float)
    ap.add_argument("--local-support-factor", type=float, default=0.9)
    ap.add_argument("--head", type=int)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    from et_miner import apriori_streaming

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = h.RunStats()
    df = h.load_items(a.data, a.head)
    n_rows = df.height
    s = h.min_support_for(a.min_count, n_rows, a.min_support)
    assert math.ceil(n_rows / a.chunk_size) == a.chunks, f"N={n_rows}, chunk {a.chunk_size} != {a.chunks} chunks"
    assert a.chunks >= 2, "one chunk falls back to apriori() and is not a SON run"
    load_s = stats.snapshot()["wall_s"]
    events = []
    log = open(out / "progress.jsonl", "w")
    t0 = time.perf_counter()

    def on_progress(phase, chunk_idx, n_chunks, metrics):
        rec = {"phase": phase, "chunk": int(chunk_idx), "n_chunks": int(n_chunks),
               "t_s": time.perf_counter() - t0, "metrics": metrics,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        events.append(rec)
        log.write(json.dumps(rec, default=str) + "\n")
        log.flush()

    res = apriori_streaming(df, min_support=s, max_length=None, chunk_size=a.chunk_size,
                            local_support_factor=a.local_support_factor, use_gpu=False,
                            n_jobs=THREADS, show_progress=False, progress_callback=on_progress)
    mine_s = time.perf_counter() - t0
    run = stats.snapshot()
    seen = {e["n_chunks"] for e in events}
    assert seen == {a.chunks}, f"progress callback reported chunk counts {seen}, expected {a.chunks}"
    p1 = [e for e in events if e["phase"] == "pass1"]
    assert len(p1) == a.chunks, f"{len(p1)} pass-1 chunk events"
    p2 = [e for e in events if e["phase"] == "pass2"]
    pass1_s = p1[-1]["t_s"]
    del df
    tables = h.normalize(res, n_rows, sort_items=False)
    del res
    h.write_tables(tables, out / "parquet", n_rows)
    meta = {
        "label": a.label, "data": str(Path(a.data).resolve()), "head": a.head, "n_rows": n_rows,
        "min_count": a.min_count, "min_support": s, "chunk_size": a.chunk_size, "chunks": a.chunks,
        "local_support_factor": a.local_support_factor, "n_jobs": THREADS,
        "load_s": load_s, "mine_wall_s": mine_s, "pass1_s": pass1_s, "pass2_s": mine_s - pass1_s,
        "pass2_chunks_read": len(p2), "process": run, "progress": events,
        "signature": h.signature(tables), "env": h.env_record(THREADS),
    }
    meta["k_max"] = max(tables) if tables else 0
    meta["process_end"] = stats.snapshot()
    h.dump_json(meta, out / "meta.json")
    print(json.dumps({"label": a.label, "itemsets": meta["signature"]["itemsets"], "k_max": meta["k_max"],
                      "mine_wall_s": round(mine_s, 2), "pass1_s": round(pass1_s, 2),
                      "peak_rss_mb": round(run["peak_rss_mb"])}))


if __name__ == "__main__":
    main()
