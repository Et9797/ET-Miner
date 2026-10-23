#!/usr/bin/env python3
"""Mine one support threshold with the streaming SON algorithm and record
the result summary.

Loads the ≥2-item transaction parquet exactly as the committed experiment
scripts do (`utils.load_transactions(min_items=2)`), runs
`et_miner.apriori_streaming` with the same defaults as
experiments/experiment_direct_vs_son.py (chunk_size 40,000,000, local
support factor 0.9, GPU on), and writes a JSON with the itemset count,
K-distribution, max K, wall-clock seconds, and optionally the number of
association rules at a confidence threshold.

Usage:
    son_run.py --data PARQUET --min-support S --output OUT.json
               [--chunk-size N] [--local-support-factor F] [--max-length K]
               [--rules-min-confidence C] [--itemsets-out PARQUET]

Options:
    --data                 transaction parquet with an `items` list column
    --min-support          relative support threshold (e.g. 0.001)
    --output               JSON summary path
    --chunk-size           SON chunk size in transactions (default 40000000)
    --local-support-factor local threshold factor (default 0.9)
    --max-length           maximum itemset length (default: unlimited)
    --rules-min-confidence when given, generate rules from the result and
                           count them
    --itemsets-out         optional parquet path for the mined itemsets
    --item-mapping         item mapping parquet; enables the cross-domain rule
                           count (rules whose items span >1 feature category)
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "applications" / "alphafold"))

from utils import k_distribution, load_transactions  # noqa: E402

from et_miner import apriori_streaming  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--min-support", type=float, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--chunk-size", type=int, default=40_000_000)
    ap.add_argument("--local-support-factor", type=float, default=0.9)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--rules-min-confidence", type=float, default=None)
    ap.add_argument("--itemsets-out", default=None)
    ap.add_argument("--item-mapping", default=None)
    a = ap.parse_args()

    df = load_transactions(a.data, min_items=2)
    n = len(df)
    min_count = math.ceil(a.min_support * n)
    t0 = time.perf_counter()
    result = apriori_streaming(
        df, min_support=a.min_support, chunk_size=a.chunk_size,
        local_support_factor=a.local_support_factor, use_gpu=True,
        max_length=a.max_length, show_progress=False,
    )
    elapsed = time.perf_counter() - t0
    k_dist = k_distribution(result)
    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data": a.data, "n_transactions": n, "min_support": a.min_support,
        "min_count": min_count, "chunk_size": a.chunk_size,
        "local_support_factor": a.local_support_factor, "max_length": a.max_length,
        "itemsets": len(result), "max_k": max(k_dist) if k_dist else 0,
        "k_distribution": {str(k): v for k, v in sorted(k_dist.items())},
        "time_seconds": round(elapsed, 2),
    }
    if a.itemsets_out:
        result.write_parquet(a.itemsets_out)
    if a.rules_min_confidence is not None:
        from et_miner.core.rules import generate_rules
        t1 = time.perf_counter()
        rules = generate_rules(result, min_confidence=a.rules_min_confidence)
        out["rules_min_confidence"] = a.rules_min_confidence
        out["n_rules"] = len(rules)
        out["rules_time_seconds"] = round(time.perf_counter() - t1, 2)
        if len(rules):
            out["rules_max"] = {f: max(float(getattr(r, f)) for r in rules)
                                for f in ("lift", "confidence", "support") if hasattr(rules[0], f)}
            lifts = [float(r.lift) for r in rules]
            out["n_rules_lift_ge5"] = sum(1 for x in lifts if x >= 5)
            out["n_rules_lift_ge100"] = sum(1 for x in lifts if x >= 100)
            if a.item_mapping:
                import polars as pl
                m = pl.read_parquet(a.item_mapping)
                cat = dict(zip(m["item_id"].to_list(), m["feature_category"].to_list()))
                def cats(r):
                    items = list(getattr(r, "lhs", ())) + list(getattr(r, "rhs", ()))
                    return {cat.get(int(i), "?") for i in items}
                out["n_rules_cross_domain"] = sum(1 for r in rules if len(cats(r)) > 1)
                out["cross_domain_definition"] = "lhs ∪ rhs spans more than one feature_category"
    json.dump(out, open(a.output, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
