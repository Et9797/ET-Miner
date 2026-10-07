"""Where efficient-apriori spends its counting time, and what its rule step adds.

Two measurements on one workload:

1. Intersection share. EA counts a candidate with
   ``TransactionManager.transaction_indices_sc``: sort the items by tidset
   size, then intersect Python sets smallest-first, stopping once the support
   drops below the minimum. On a systematic sample of one level's candidates
   this times (a) that call as EA makes it and (b) the same sequence of
   ``set.intersection`` calls alone, so (b)/(a) is the share of counting spent
   intersecting. The level's candidates come from EA's own ``apriori_gen``
   over the level below, mined by EA first.
2. What ``efficient_apriori.apriori`` adds. It calls
   ``itemsets_from_transactions(..., output_transaction_ids=True)``, which
   recomputes every frequent itemset's full row-id set, then
   ``generate_rules_apriori``. Both are timed against the plain miner, at
   ``min_confidence=1.0`` (the call the 2026-10-01 Retail baseline timed).

Usage:
    uv run python bench/cpu/ea_profile.py --workload wide --level 2 --sample 500000

Options:
    --workload  an id of bench/cpu/matrix.py WORKLOADS
    --level     the level whose candidates are sampled (default 2)
    --sample    candidates sampled systematically from that level (0 = all)
    --rules     also time the plain miner, the miner with row ids, and the rule step
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from consolidation_run import _load  # noqa: E402
from cpu.matrix import WORKLOADS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True, choices=list(WORKLOADS))
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--sample", type=int, default=500_000)
    ap.add_argument("--rules", action="store_true")
    args = ap.parse_args()

    import polars as pl
    from efficient_apriori.itemsets import TransactionManager, apriori_gen, itemsets_from_transactions
    from efficient_apriori.rules import generate_rules_apriori

    from et_miner.core.result import _min_count

    dataset, min_support, max_length = WORKLOADS[args.workload]
    df, n, _ = _load(dataset)
    tx = [tuple(r) for r in df["items"].to_list()]
    s = (_min_count(min_support, n) - 0.5) / n
    ml = max_length or int(df.select(pl.col("items").list.len().max()).item())
    out: dict = {"workload": args.workload, "level": args.level}

    below, _ = itemsets_from_transactions(tx, s, max_length=args.level - 1)
    manager = TransactionManager(tx)
    cands = list(apriori_gen(sorted(below[args.level - 1].keys())))
    step = max(1, len(cands) // args.sample) if args.sample else 1
    sample = cands[::step]
    out.update(n_candidates=len(cands), sampled=len(sample))

    t0 = time.perf_counter()
    for c in sample:
        manager.transaction_indices_sc(c, min_support=s)
    out["sc_s"] = round(time.perf_counter() - t0, 4)

    by_item = manager.indices_by_item
    n_rows = len(manager)
    t0 = time.perf_counter()
    for c in sample:
        items = sorted(c, key=lambda it: len(by_item[it]), reverse=True)
        idx = by_item[items.pop()]
        if len(idx) / n_rows < s:
            continue
        while items:
            idx = idx.intersection(by_item[items.pop()])
            if len(idx) / n_rows < s:
                break
    out["sort_and_intersect_s"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    for c in sample:
        items = sorted(c, key=lambda it: len(by_item[it]), reverse=True)
        idx = by_item[items.pop()]
        if len(idx) / n_rows < s:
            continue
    out["sort_only_s"] = round(time.perf_counter() - t0, 4)
    out["intersect_share_est"] = round(1 - out["sort_only_s"] / out["sc_s"], 3)

    if args.rules:
        t0 = time.perf_counter()
        itemsets_from_transactions(tx, s, max_length=ml)
        out["itemsets_s"] = round(time.perf_counter() - t0, 3)
        t0 = time.perf_counter()
        itemsets, num = itemsets_from_transactions(tx, s, max_length=ml, output_transaction_ids=True)
        out["itemsets_with_tids_s"] = round(time.perf_counter() - t0, 3)
        raw = {k: {i: c.itemset_count for i, c in v.items()} for k, v in itemsets.items()}
        t0 = time.perf_counter()
        rules = list(generate_rules_apriori(raw, 1.0, num, 0))
        out["rules_s"] = round(time.perf_counter() - t0, 3)
        out["n_rules"] = len(rules)
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
