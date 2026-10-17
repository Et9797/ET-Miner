"""Mine the proposed ``deep_k20`` preset (M4's target regime) on the CPU and dump its lattice.

The spec is generated in memory (it is a proposal, not in ``PRESETS``), mined
with ``apriori(use_gpu=False)``, and written as ``<OUT>.lattice.parquet`` with
its sidecar, the input of ``component_stakes.py``. Prints the mining time and
the itemsets per level as one JSON line.

Usage: uv run python bench/moonshots/preset_check.py OUT_PREFIX
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl

from et_miner import apriori
from et_miner.synthetic import SynthSpec, generate_transactions

DEEP_K20 = SynthSpec(
    name="deep_k20", n_rows=1_000_000, vocab_size=4_000, zipf_a=1.1, row_len_mean=10, row_len_max=48,
    motif_count=2, motif_size=20, motif_penetration=0.011, min_support=0.01, seed=2020,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    args = ap.parse_args()
    spec = DEEP_K20
    df, _ = generate_transactions(spec)
    t0 = time.perf_counter()
    res = apriori(df, min_support=spec.min_support, use_gpu=False)
    mine_s = time.perf_counter() - t0
    res.select(
        pl.col("itemset").list.eval(pl.element().sort()).cast(pl.List(pl.Int64)),
        (pl.col("support") * spec.n_rows).round().cast(pl.Int64).alias("count"),
    ).write_parquet(f"{args.out}.lattice.parquet")
    meta = {"dataset": spec.name, "min_support": spec.min_support, "max_length": None, "prune_equal_support": False,
            "n_rows": spec.n_rows, "min_count": spec.min_count}
    with open(f"{args.out}.lattice.json", "w") as f:
        json.dump(meta, f, indent=2)
    k = res["itemset"].list.len().to_numpy()
    levels, counts = np.unique(k, return_counts=True)
    print(json.dumps({"mine_s": round(mine_s, 1), "itemsets": res.height, "max_k": int(k.max()),
                      "per_k": {int(a): int(b) for a, b in zip(levels, counts)}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
