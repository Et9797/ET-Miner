"""Association rules from a CPU itemset table with the engine's public `generate_rules`.

Counts all rules, rules with lift >= 5 and rules with lift >= 100 (as
son_run.py did), and writes the rules as a parquet table plus a meta JSON.

Usage:
    run_py.sh rules.py ITEMSET_DIR OUT_DIR --n N [--min-confidence C] [--label TEXT]

Options:
    --n N               transaction count of the mined set (support = count / N)
    --min-confidence C  default 0.5
    --label TEXT        id recorded in the meta JSON
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
    ap.add_argument("itemsets")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--min-confidence", type=float, default=0.5)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    import polars as pl

    from et_miner import generate_rules

    out = Path(a.out)
    stats = h.RunStats()
    tables = h.load_tables(a.itemsets, a.n)
    fi = pl.concat([t for _, t in sorted(tables.items())]).select(
        "itemset", support=pl.col("count").cast(pl.Float64) / a.n)
    t0 = time.perf_counter()
    rules = generate_rules(fi, min_confidence=a.min_confidence)
    gen_s = time.perf_counter() - t0
    lifts = [r.lift for r in rules]
    rdf = pl.DataFrame({"lhs": [sorted(r.lhs) for r in rules], "rhs": [sorted(r.rhs) for r in rules],
                        "support": [r.support for r in rules], "confidence": [r.confidence for r in rules],
                        "lift": lifts}, schema={"lhs": pl.List(pl.Int64), "rhs": pl.List(pl.Int64),
                                                "support": pl.Float64, "confidence": pl.Float64,
                                                "lift": pl.Float64})
    out.mkdir(parents=True, exist_ok=True)
    rdf.sort(["lhs", "rhs"]).write_parquet(out / "rules.parquet", compression="zstd")
    meta = {"label": a.label, "itemsets": str(Path(a.itemsets).resolve()), "n_itemsets": fi.height,
            "min_confidence": a.min_confidence, "n_rules": len(rules),
            "n_rules_lift_ge5": sum(x >= 5 for x in lifts), "n_rules_lift_ge100": sum(x >= 100 for x in lifts),
            "max_lift": max(lifts) if lifts else None, "generate_s": gen_s, "process": stats.snapshot(),
            "env": h.env_record(THREADS)}
    h.dump_json(meta, out / "meta.json")
    print(json.dumps({k: meta[k] for k in ("label", "n_rules", "n_rules_lift_ge5", "n_rules_lift_ge100",
                                           "max_lift")}))


if __name__ == "__main__":
    main()
