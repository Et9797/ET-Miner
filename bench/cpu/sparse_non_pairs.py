"""Times count_support_batched's sparse non-pair counter on three synthetic matrices and records its counts.

One JSON line per (case, n_jobs): seconds, the itemset count, and a signature
of the result (sum of counts, sha256 of the sorted (itemset, count) pairs), so
rows from different counters can be checked for identical output.

Cases (seed 7, boolean matrices, density per column):
    small  3,000 x 14, 0.35: every K=0, K=1, K=3..6 itemset, plus a repeated
           item, a repeated itemset and a single item
    mid    200,000 x 40, 0.12: 20,000 sampled K=3 and K=4 itemsets each
    wide   500,000 x 120, 0.05: 30,000 sampled K=3 itemsets

Runs against whichever et_miner is importable: ``_count_support_sparse_non_pairs``
when present, else the older ``_count_support_sparse_k_gt_2`` (scipy, or the Rust
counter when an et_miner_rust < 0.4.0 is built).

Usage:
    uv run python bench/cpu/sparse_non_pairs.py --label "this tree" --out rows.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time

import numpy as np
import polars as pl

from et_miner.core import sparse
from et_miner.core.matrix import _polars_to_sparse_csr

CASES = [
    ("small", 3_000, 14, 0.35, (0, 1, 3, 4, 5, 6), None),
    ("mid", 200_000, 40, 0.12, (3, 4), 20_000),
    ("wide", 500_000, 120, 0.05, (3,), 30_000),
]


def _counter():
    new = getattr(sparse, "_count_support_sparse_non_pairs", None)
    if new is not None:
        return "_count_support_sparse_non_pairs", lambda csr, c2i, its, nj: new(csr, c2i, its, nj)
    old = sparse._count_support_sparse_k_gt_2
    return "_count_support_sparse_k_gt_2", lambda csr, c2i, its, nj: old(csr, c2i, its, False, nj)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="the counter and tree this run measures")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fn_name, count = _counter()
    rng = np.random.default_rng(7)
    for name, n_rows, n_cols, density, ks, cap in CASES:
        dense = rng.random((n_rows, n_cols)) < density
        csr, c2i = _polars_to_sparse_csr(pl.DataFrame({f"i_{j}": dense[:, j] for j in range(n_cols)}))
        cols = list(c2i)
        itemsets: list[tuple[str, ...]] = []
        for k in ks:
            comb = list(itertools.combinations(cols, k))
            if cap and len(comb) > cap:
                comb = [comb[i] for i in rng.choice(len(comb), cap, replace=False)]
            itemsets += comb
        if name == "small":
            itemsets += [("i_0", "i_0", "i_1"), ("i_3", "i_1", "i_2"), ("i_3", "i_1", "i_2"), ("i_5",)]
        for n_jobs in (1, 4):
            t = time.perf_counter()
            res = count(csr, c2i, itemsets, n_jobs)
            s = time.perf_counter() - t
            pairs = sorted((list(k), int(v)) for k, v in res.items())
            row = {
                "label": args.label,
                "counter": fn_name,
                "case": name,
                "shape": [n_rows, n_cols],
                "n_jobs": n_jobs,
                "itemsets": len(itemsets),
                "distinct": len(res),
                "sum_counts": sum(v for _, v in pairs),
                "sha256": hashlib.sha256(json.dumps(pairs).encode()).hexdigest()[:16],
                "s": round(s, 3),
            }
            print(json.dumps(row), flush=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
