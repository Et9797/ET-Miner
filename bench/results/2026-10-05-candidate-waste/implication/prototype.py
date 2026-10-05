"""Prototype of the proposed implication-heavy synthetic preset (not in PRESETS).

Generates a smoke-shaped dataset with ``et_miner.synthetic.generate_csr`` and
closes every row under an item hierarchy: items ``i >= fanout`` have the parent
``i // fanout`` (items ``0..fanout-1`` are roots), and each drawn item brings
its ancestors up to ``depth`` steps. A child therefore always implies its
parent, so {child, parent} is never free and Pascal's inference applies from
K=3 on. Mines the complete lattice and the free-sets on one GPU and writes
lattice dumps that ``bench/candidate_waste.py classify`` reads.

Usage:
    uv run python bench/results/2026-10-05-candidate-waste/implication/prototype.py OUT_DIR

Options:
    OUT_DIR   directory for ``impl-C1.lattice.{parquet,json}``,
              ``impl-C1-free.lattice.{parquet,json}`` and the run summaries
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

FANOUT = 8
DEPTH = 2
SPEC_OVERRIDES = {"name": "implication_proto", "n_rows": 200_000}


def ancestors_closure(indptr: np.ndarray, indices: np.ndarray, n_rows: int, vocab: int):
    """Rows closed under the parent map, as unique sorted CSR (int64 indptr, int32 items)."""
    rows = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(indptr))
    items = indices.astype(np.int64)
    all_rows, all_items = [rows], [items]
    cur = items
    for _ in range(DEPTH):
        has_parent = cur >= FANOUT
        cur = np.where(has_parent, cur // FANOUT, -1)
        keep = cur >= 0
        all_rows.append(rows[keep])
        all_items.append(cur[keep])
        rows, cur = rows[keep], cur[keep]
    keys = np.unique(np.concatenate(all_rows) * vocab + np.concatenate(all_items))
    out_rows, out_items = keys // vocab, (keys % vocab).astype(np.int32)
    new_indptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(np.bincount(out_rows, minlength=n_rows), out=new_indptr[1:])
    return new_indptr, out_items


def main(out_dir: Path) -> int:
    import polars as pl
    import pyarrow as pa

    from et_miner import apriori
    from et_miner.core.result import _min_count
    from et_miner.synthetic import PRESETS, generate_csr

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from level_split import LevelSplit

    spec = replace(PRESETS["smoke"], **SPEC_OVERRIDES)
    data = generate_csr(spec)
    indptr, items = ancestors_closure(data.indptr, data.indices, spec.n_rows, spec.vocab_size)
    df = pl.DataFrame({"items": pl.Series("items", pa.LargeListArray.from_arrays(indptr, items.astype(np.int64)))})
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"spec": {**SPEC_OVERRIDES, "fanout": FANOUT, "depth": DEPTH, "base": "smoke"},
               "nnz_before": int(data.indptr[-1]), "nnz_after": int(indptr[-1])}
    for name, free in (("impl-C1", False), ("impl-C1-free", True)):
        levels = []
        split = LevelSplit()
        split.install()
        t0 = time.perf_counter()
        try:
            res = apriori(df, min_support=spec.min_support, use_gpu=True, n_gpus=1, prune_equal_support=free,
                          level_callback=split.callback(
                              lambda k, n, f, ms: levels.append({"k": k, "n_candidates": n, "n_frequent": f, "ms": ms})))
        finally:
            split.uninstall()
        wall = time.perf_counter() - t0
        res.select(
            pl.col("itemset").list.eval(pl.element().sort()).cast(pl.List(pl.Int64)),
            (pl.col("support") * spec.n_rows).round().cast(pl.Int64).alias("count"),
        ).write_parquet(out_dir / f"{name}.lattice.parquet")
        meta = {"dataset": spec.name, "min_support": spec.min_support, "max_length": None,
                "prune_equal_support": free, "n_rows": spec.n_rows,
                "min_count": _min_count(spec.min_support, spec.n_rows)}
        (out_dir / f"{name}.lattice.json").write_text(json.dumps(meta, indent=2) + "\n")
        (out_dir / f"{name}_r0.result.json").write_text(json.dumps(
            {"id": f"{name}#r0", "wall_s": wall, "levels": levels, "n_itemsets": res.height,
             "timings": {"level_split": split.levels}}) + "\n")
        summary[name] = {"wall_s": round(wall, 3), "n_itemsets": res.height}
    (out_dir / "implication_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
