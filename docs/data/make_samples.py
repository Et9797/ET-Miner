"""Generate the small datasets the tutorials read from ``docs/data/``.

Run from the repository root:

    uv run python docs/data/make_samples.py

Outputs (all deterministic, fixed seeds):

- ``toy_8x6.parquet``: the 8-row, 6-item table built by hand in
  ``tier1-polars/01-the-problem-and-the-data.ipynb``.
- ``smoke.parquet``: the ``smoke`` preset from ``et_miner.synthetic.PRESETS``
  (60,000 rows, seed 42). This is the shared sample of all three tiers and
  the same dataset the repository's tier-equivalence test mines.
- ``smoke.json``: the preset parameters and the exact minimum count, so a
  notebook can check it read the file it expected.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import polars as pl

from et_miner.synthetic import PRESETS, generate_transactions

HERE = Path(__file__).resolve().parent

# Items are single letters so the table can be checked by eye:
# a = bread, b = butter, c = milk, d = eggs, e = coffee, f = tea.
TOY_ROWS = [
    ["a", "b", "c"],
    ["a", "b"],
    ["a", "c", "d"],
    ["b", "c"],
    ["a", "b", "c", "e"],
    ["a", "b", "d"],
    ["c", "e", "f"],
    ["a", "b", "c", "f"],
]


def write_toy() -> Path:
    path = HERE / "toy_8x6.parquet"
    pl.DataFrame({"items": TOY_ROWS}).write_parquet(path)
    return path


def write_smoke() -> tuple[Path, Path]:
    spec = PRESETS["smoke"]
    df, data = generate_transactions(spec)
    parquet_path = HERE / "smoke.parquet"
    df.write_parquet(parquet_path)
    meta = {k: v for k, v in dataclasses.asdict(spec).items() if isinstance(v, (int, float, str, type(None)))}
    meta["min_count"] = spec.min_count
    meta["n_cols"] = int(data.n_cols)
    meta_path = HERE / "smoke.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return parquet_path, meta_path


if __name__ == "__main__":
    for p in (write_toy(), *write_smoke()):
        print(f"wrote {p.relative_to(HERE.parent.parent)} ({p.stat().st_size:,} bytes)")
