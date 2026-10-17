"""A lattice dump with its items relabelled in another order (M10's replay input).

Writes a copy of a ``.lattice.parquet`` dump (and its sidecar) whose item ids
are the items' ranks under the order: ``asc`` / ``desc`` (ascending /
descending K=1 count, ties by item id). Every itemset is re-sorted, so the
miner's column order of the copy is the new order, and ``replay_stakes.py`` or
``k3_census.py`` replay the levels as the miner would generate them in it.

Usage: uv run python bench/moonshots/relabel_lattice.py LATTICE ORDER OUT_PREFIX
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl


def main() -> int:
    src, order, out = sys.argv[1], sys.argv[2], sys.argv[3]
    df = pl.read_parquet(src)
    k1 = df.filter(pl.col("itemset").list.len() == 1)
    items = k1["itemset"].list.first().to_numpy().astype(np.int64)
    counts = k1["count"].to_numpy()
    key = {"asc": counts, "desc": -counts}[order]
    ranked = items[np.lexsort((items, key))]
    mapping = pl.DataFrame({"old": ranked, "new": np.arange(len(ranked), dtype=np.int64)})
    relabelled = (
        df.with_row_index("row")
        .explode("itemset")
        .join(mapping, left_on="itemset", right_on="old", how="left")
        .group_by("row", maintain_order=True)
        .agg(pl.col("new").sort().alias("itemset"), pl.col("count").first())
        .select("itemset", "count")
    )
    relabelled.write_parquet(f"{out}.lattice.parquet")
    shutil.copy(str(src).removesuffix(".parquet") + ".json", f"{out}.lattice.json")
    meta = json.loads(Path(f"{out}.lattice.json").read_text())
    meta["relabelled"] = order
    Path(f"{out}.lattice.json").write_text(json.dumps(meta, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
