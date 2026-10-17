"""Distinct rows after projection (M11, weighted row deduplication).

Per lattice dump: the rows that hold at least one frequent item and how many
of them are distinct once projected onto the frequent items; then per level k
>= 2, the rows holding at least k of level k-1's items and how many distinct
projections onto those items they form. Rows are compared by two independent
64-bit sums of random item keys (a row is a set, so its keys sum in any order);
a collision would merge two rows, with probability about rows^2 / 2^128.

Usage: uv run python bench/moonshots/dedup_stakes.py LATTICE --dataset NAME > OUT.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from candidate_waste import load_lattice  # noqa: E402
from rows import frequent_rows, row_ids  # noqa: E402


def distinct(rid: np.ndarray, pos: np.ndarray, keep_item: np.ndarray, min_items: int, n_rows: int, keys) -> tuple[int, int]:
    """(rows with >= min_items kept items, distinct projections among them)."""
    m = keep_item[pos]
    r, p = rid[m], pos[m]
    if len(r) == 0:
        return 0, 0
    starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
    per = np.diff(np.r_[starts, len(r)])
    h1 = np.add.reduceat(keys[0][p], starts)
    h2 = np.add.reduceat(keys[1][p], starts)
    sel = per >= min_items
    pairs = np.stack([h1[sel], h2[sel]], axis=1)
    return int(sel.sum()), int(len(np.unique(pairs, axis=0))) if len(pairs) else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lattice")
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    lat = load_lattice(Path(args.lattice))
    indptr, pos, n_rows = frequent_rows(args.dataset, lat.col_to_item)
    rid = row_ids(indptr)
    rng = np.random.default_rng(12345)
    keys = rng.integers(0, 2**63, size=(2, lat.n_cols), dtype=np.int64).astype(np.uint64) * np.uint64(2) + np.uint64(1)
    rows, uniq = distinct(rid, pos, np.ones(lat.n_cols, dtype=bool), 1, n_rows, keys)
    print(json.dumps({"level": "frequent", "n_rows": n_rows, "rows": rows, "distinct": uniq,
                      "ratio": rows / max(uniq, 1)}), flush=True)
    for k in range(2, max(lat.items) + 2):
        if len(lat.items.get(k - 1, [])) < 2:
            break
        active = np.zeros(lat.n_cols, dtype=bool)
        active[np.unique(lat.items[k - 1])] = True
        rows, uniq = distinct(rid, pos, active, k, n_rows, keys)
        print(json.dumps({"level": k, "active_items": int(active.sum()), "rows": rows, "distinct": uniq,
                          "ratio": rows / max(uniq, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
