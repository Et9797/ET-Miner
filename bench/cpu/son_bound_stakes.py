"""Untimed stakes for SON's partition upper bound: how many pass-2 candidates it would drop.

Pass 1 mines every chunk completely at its local min_count ``m_i``, so an
itemset a chunk did not emit occurs in at most ``m_i - 1`` of its rows. Per
union candidate X:

    bound(X) = sum of X's local counts over the chunks that emitted it
             + sum of (m_i - 1) over the chunks that did not

and X's global count is at most bound(X). A candidate with
``bound < min_count`` cannot be globally frequent, and pass 2 need not count
it. A candidate whose unseen chunks all have ``m_i = 1`` (no slack left) has
its exact count already; pass 2 only has to count the rest, and those only in
the chunks that did not emit them.

Per workload and local support factor, this prints and appends one JSON row:
per K the union size, the candidates left after the bound, those with an exact
count from pass 1, those left to count, the (candidate, chunk) pairs left to
count, and the globally frequent ones (in-core ``apriori``). It fails if a
globally frequent itemset is missing from the union, if its bound is below
min_count or below its in-core count, or if an exact count differs from the
in-core count.

Usage:
    uv run python bench/cpu/son_bound_stakes.py --workloads smoke,deepk --factors 0.9,1.0 --out rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bench"))
sys.path.insert(0, str(REPO / "bench" / "cpu"))

from consolidation_run import _load  # noqa: E402
from matrix import WORKLOADS  # noqa: E402
from son_stakes import INPUT_WORKLOADS, SON_WORKLOADS, _rev  # noqa: E402


def _local(lf: pl.LazyFrame, local_s: float, max_length: int | None):
    """(items, [(sets, counts) per K]) of one chunk as SON's pass 1 mines it, or None."""
    from et_miner.core import cpu_miner as cm
    from et_miner.core.result import _min_count

    tc = cm.build_transaction_csr(lf, local_s, "items")
    if tc is None:
        return None
    ones = np.arange(tc.n_cols, dtype=np.int32)[:, None]
    emitted = [(ones, tc.counts)]
    cm._mine_levels(
        tc, ones, cm._Level(ones, tc.counts, tc.counts < tc.n_rows), emitted,
        max(1, _min_count(local_s, tc.n_rows)),
        min(max_length or math.inf, int(np.diff(tc.indptr).max()), tc.n_cols),
        False, False, True, None, None, None, 1,
    )
    return tc.items, emitted


def _keys(rows: np.ndarray, base: int) -> np.ndarray:
    """Lexicographic-order int64 keys of int32 rows: packed, or dense ranks when packing overflows."""
    from et_miner.core import cpu_miner as cm

    keys = cm._pack(rows, base)
    if keys is None:
        keys = np.unique(rows, axis=0, return_inverse=True)[1].ravel().astype(np.int64)
    return keys


def stakes(df: pl.DataFrame, min_support: float, max_length: int | None, n_chunks: int, factor: float) -> dict:
    from et_miner import apriori
    from et_miner.core.result import _min_count

    n_total = df.height
    chunk_size = math.ceil(n_total / n_chunks)
    local_s = min_support * factor
    total_slack = 0
    n_chunks_all = 0
    chunks = []
    for off in range(0, n_total, chunk_size):
        n = min(chunk_size, n_total - off)
        slack = max(1, _min_count(local_s, n)) - 1
        total_slack += slack
        n_chunks_all += 1
        local = _local(df.lazy().slice(off, n), local_s, max_length)
        if local is not None:
            chunks.append((*local, slack))
    items = pl.concat([c[0] for c in chunks]).unique().sort()
    base = len(items)

    min_count = _min_count(min_support, n_total)
    truth = apriori(df, min_support=min_support, max_length=max_length, show_progress=False)
    truth_rows: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k, part in truth.with_columns(k=pl.col("itemset").list.len()).partition_by("k", as_dict=True).items():
        flat = pl.Series(part["itemset"].list.sort().explode())
        ids = items.search_sorted(flat).to_numpy().astype(np.int32)
        if not (items.gather(pl.Series(ids)) == flat).all():
            raise RuntimeError("a frequent item is missing from the union")
        counts = np.rint(part["support"].to_numpy() * n_total).astype(np.int64)
        truth_rows[int(k[0])] = (ids.reshape(-1, int(k[0])), counts)

    per_k = {}
    for k in sorted({sets.shape[1] for _, levels, _ in chunks for sets, _ in levels if len(sets)}):
        rows, counts, slacks = [], [], []
        for its, levels, slack in chunks:
            remap = items.search_sorted(its).to_numpy().astype(np.int32)
            for sets, cnt in levels:
                if sets.shape[1] == k and len(sets):
                    rows.append(remap[sets])
                    counts.append(cnt.astype(np.int64))
                    slacks.append(np.full(len(sets), slack, dtype=np.int64))
        t_rows, t_counts = truth_rows.get(k, (np.empty((0, k), np.int32), np.empty(0, np.int64)))
        rows = np.concatenate(rows)
        keys = _keys(np.concatenate([rows, t_rows]), base)
        uniq, inv = np.unique(keys[: len(rows)], return_inverse=True)
        t_keys = keys[len(rows) :]
        known = np.bincount(inv, weights=np.concatenate(counts), minlength=len(uniq)).astype(np.int64)
        seen_slack = np.bincount(inv, weights=np.concatenate(slacks), minlength=len(uniq)).astype(np.int64)
        n_seen = np.bincount(inv, minlength=len(uniq))
        bound = known + total_slack - seen_slack
        keep = bound >= min_count

        pos = np.searchsorted(uniq, t_keys)
        if len(t_keys) and ((pos >= len(uniq)).any() or (uniq[np.minimum(pos, len(uniq) - 1)] != t_keys).any()):
            raise RuntimeError(f"K={k}: a frequent itemset is missing from the union")
        if (bound[pos] < t_counts).any() or not keep[pos].all():
            raise RuntimeError(f"K={k}: a frequent itemset's bound is below its count")
        exact = bound == known
        if exact[pos].any() and (known[pos][exact[pos]] != t_counts[exact[pos]]).any():
            raise RuntimeError(f"K={k}: an exact pass-1 count differs from the in-core count")
        per_k[k] = {
            "union": int(len(uniq)),
            "after_bound": int(keep.sum()),
            "exact": int((keep & exact).sum()),
            "to_count": int((keep & ~exact).sum()),
            "pairs_to_count": int((n_chunks_all - n_seen)[keep & ~exact].sum()),
            "frequent": int(len(t_keys)),
        }
    if set(truth_rows) - set(per_k):
        raise RuntimeError(f"frequent lengths {sorted(set(truth_rows) - set(per_k))} have no candidates")
    tot = {f: sum(v[f] for v in per_k.values()) for f in ("union", "after_bound", "exact", "to_count", "pairs_to_count", "frequent")}
    return {
        "factor": factor, "chunks": n_chunks, "chunk_size": chunk_size, "n_items": base,
        "min_count": min_count, "total_slack": total_slack, "per_k": per_k, "total": tot,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", default=",".join(SON_WORKLOADS))
    ap.add_argument("--factors", default="0.9")
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--out")
    args = ap.parse_args()

    from loguru import logger

    logger.remove()
    for w in args.workloads.split(","):
        dataset, min_support, max_length = {**WORKLOADS, **INPUT_WORKLOADS}[w]
        df, _, _ = _load(dataset)
        for factor in map(float, args.factors.split(",")):
            row = {"workload": w, **stakes(df, min_support, max_length, args.chunks, factor)}
            row |= {"rev": _rev(), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            t = row["total"]
            print(
                f"{w} f={factor}: union {t['union']:,} -> bound {t['after_bound']:,} "
                f"({t['after_bound'] / max(1, t['union']):.1%}), exact {t['exact']:,}, to count {t['to_count']:,} "
                f"({t['pairs_to_count']:,} chunk pairs), frequent {t['frequent']:,}",
                flush=True,
            )
            if args.out:
                with open(args.out, "a") as f:
                    f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
