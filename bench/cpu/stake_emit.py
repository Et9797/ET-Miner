"""Stake for emitting a lattice from arrays: the result frame built from per-level (n, k) int arrays.

Builds, for each workload, the campaign's level sizes as int32 (n_k, k) arrays
of column indices plus int64 counts (random content of the right shape), and
times what an array-based CPU route would do to emit them: map columns to item
ids, turn each level into a List[Int64] column via a fixed-width reshape,
divide counts by N, and concatenate. The baseline's equivalent is the
``tail`` (``_build_result_df``) plus the per-level ``results.append`` loop
inside ``other``.

Usage:
    uv run python bench/cpu/stake_emit.py bench/results/2026-10-07-cpu-baseline

Options:
    DIR   results directory; reads raw.jsonl, appends to stakes.jsonl
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl


def emit(levels: list[tuple[np.ndarray, np.ndarray]], item_ids: np.ndarray, n_rows: int) -> pl.DataFrame:
    frames = []
    for sets, counts in levels:
        n, k = sets.shape
        flat = pl.Series(item_ids[sets.ravel()], dtype=pl.Int64)
        col = flat.reshape((n, k)).arr.to_list() if k > 1 else flat.reshape((n, 1)).arr.to_list()
        frames.append(pl.DataFrame({"itemset": col, "support": counts / n_rows}))
    return pl.concat(frames)


def main() -> int:
    d = Path(sys.argv[1])
    rows = [json.loads(x) for x in (d / "raw.jsonl").read_text().splitlines()]
    seen = set()
    rng = np.random.default_rng(0)
    for r in rows:
        c = r.get("config", {})
        if r.get("status") != "ok" or c.get("route") != "F" or c["base_id"].split("-", 1)[0] in seen:
            continue
        w = c["base_id"].split("-", 1)[0]
        seen.add(w)
        n_items = r["levels"][0]["n_frequent"]
        item_ids = np.sort(rng.choice(10 * n_items, n_items, replace=False)).astype(np.int64)
        levels = []
        for lv in r["levels"]:
            n, k = lv["n_frequent"], lv["k"]
            if n == 0:
                continue
            sets = np.sort(rng.integers(0, n_items, size=(n, k), dtype=np.int32), axis=1)
            levels.append((sets, rng.integers(1, 1000, size=n).astype(np.int64)))
        n_rows = 10**6
        t0 = time.perf_counter()
        frame = emit(levels, item_ids, n_rows)
        dt = time.perf_counter() - t0
        assert frame.height == sum(len(s) for s, _ in levels)
        out = {"workload": w, "threads": 1, "lever": "L5", "variant": "emit_arrays", "k": None, "s": round(dt, 6),
               "n_itemsets": frame.height}
        print(json.dumps(out))
        with (d / "stakes.jsonl").open("a") as f:
            f.write(json.dumps(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
