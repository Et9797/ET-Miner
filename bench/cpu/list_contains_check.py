"""Exactness of Polars `list.contains` under the in-memory and streaming engines.

Builds the boolean item matrix the way `core.matrix.build_boolean_matrix` does
(one `list.contains` expression per item) under both engines and compares every
column sum with the item's true count from an explode + group_by. Prints one
JSON line per dataset with the number of mismatching columns per engine.

Usage:
    uv run python bench/cpu/list_contains_check.py [--datasets smoke,deep_k,stress_k2] [--max-items 600]

Options:
    --datasets   synthetic presets under datasets/synth/ (generated if missing)
    --max-items  items checked per dataset, spread evenly over the frequency ranking
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[2]


def _load(name: str) -> pl.DataFrame:
    path = REPO / "datasets" / "synth" / f"{name}.parquet"
    if not path.exists():
        from et_miner.synthetic import PRESETS, generate_transactions

        df, _ = generate_transactions(PRESETS[name])
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
    return pl.read_parquet(path)


def check(name: str, max_items: int) -> dict:
    """Compare list.contains column sums per engine against explode counts.

    Steps: true counts from explode + group_by (in-memory engine); pick up to
    `max_items` items spread over the frequency ranking; build the boolean
    columns from a LazyFrame scan of the parquet file under each engine and sum
    them; count mismatches.
    """
    df = _load(name)
    truth = (
        df.lazy().select(pl.col("items").explode().alias("item")).drop_nulls()
        .group_by("item").agg(pl.len().alias("n")).sort("n", descending=True)
        .collect(engine="in-memory")
    )
    idx = np.unique(np.linspace(0, truth.height - 1, min(max_items, truth.height)).astype(int))
    picked = truth[idx]
    items = picked["item"].to_list()
    true_counts = dict(zip(items, picked["n"].to_list()))
    lf = pl.scan_parquet(REPO / "datasets" / "synth" / f"{name}.parquet")
    exprs = [pl.col("items").list.contains(i).alias(f"c{i}") for i in items]
    out = {"dataset": name, "rows": df.height, "items_checked": len(items), "polars": pl.__version__}
    for engine in ("in-memory", "streaming"):
        t0 = time.perf_counter()
        sums = lf.select(exprs).collect(engine=engine).sum().row(0)
        out[f"{engine}_s"] = round(time.perf_counter() - t0, 2)
        bad = [(i, s, true_counts[i]) for i, s in zip(items, sums) if s != true_counts[i]]
        out[f"{engine}_mismatches"] = len(bad)
        out[f"{engine}_examples"] = bad[:3]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="smoke,deep_k,skewed_rows,stress_k2")
    ap.add_argument("--max-items", type=int, default=600)
    args = ap.parse_args()
    for name in args.datasets.split(","):
        print(json.dumps(check(name, args.max_items)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
