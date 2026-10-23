"""efficient-apriori oracle run, written as canonical per-K tables.

Follows the oracle call convention of ET-Miner's CLAUDE.md: min_support =
(min_count - 0.5) / N and an explicit max_length (the longest row), so the
oracle neither shifts the integer boundary nor truncates deep itemsets.

Usage:
    run_py.sh ea_oracle.py DATA.parquet OUT_DIR --min-count M [--head N] [--label TEXT]

Options:
    --min-count M   absolute threshold
    --head N        first N rows only
    --label TEXT    id recorded in the meta JSON
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
    ap.add_argument("--head", type=int)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    import polars as pl
    from efficient_apriori import apriori as ea_apriori

    out = Path(a.out)
    stats = h.RunStats()
    df = h.load_items(a.data, a.head)
    n_rows = df.height
    rows = [tuple(sorted(set(r))) for r in df["items"].to_list()]
    max_len = max(len(r) for r in rows)
    del df
    t0 = time.perf_counter()
    itemsets, _ = ea_apriori(rows, min_support=(a.min_count - 0.5) / n_rows, min_confidence=1.0,
                             max_length=max_len)
    mine_s = time.perf_counter() - t0
    run = stats.snapshot()
    recs = [(list(t), c) for level in itemsets.values() for t, c in level.items()]
    res = pl.DataFrame({"itemset": [r[0] for r in recs], "count": [r[1] for r in recs]},
                       schema={"itemset": pl.List(pl.Int64), "count": pl.Int64})
    tables = h.normalize(res)
    assert all(int(t["count"].min()) >= a.min_count for t in tables.values())
    h.write_tables(tables, out / "parquet", n_rows)
    meta = {"label": a.label, "data": str(Path(a.data).resolve()), "head": a.head, "n_rows": n_rows,
            "min_count": a.min_count, "ea_min_support": (a.min_count - 0.5) / n_rows, "max_length": max_len,
            "mine_wall_s": mine_s, "process": run, "signature": h.signature(tables), "k_max": max(tables)}
    h.dump_json(meta, out / "meta.json")
    print(json.dumps({"label": a.label, "itemsets": meta["signature"]["itemsets"], "k_max": meta["k_max"],
                      "max_length": max_len, "mine_wall_s": round(mine_s, 1)}))


if __name__ == "__main__":
    main()
