"""A dataset's rows as ascending positions of its frequent items (the moonshot stake scripts' input).

Usage:
    from rows import frequent_rows
    indptr, pos, n_rows = frequent_rows("deep_sparse_large", lat.col_to_item)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def frequent_rows(dataset: str, col_to_item: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """(indptr int64, positions int64, n_rows): each row's items found in ``col_to_item``, ascending and unique.

    Null rows and null items hold nothing; a duplicate item counts once.
    """
    from consolidation_run import _load

    df, n_rows, _ = _load(dataset)
    items = df["items"]
    lens = items.list.len().fill_null(0).to_numpy().astype(np.int64)
    flat = items.explode(empty_as_null=True).to_numpy()
    row = np.repeat(np.arange(n_rows, dtype=np.int64), lens)
    if len(flat) != len(row):
        raise ValueError(f"{dataset}: exploded {len(flat):,} entries for {len(row):,} list slots")
    present = ~np.isnan(flat) if flat.dtype.kind == "f" else np.ones(len(flat), dtype=bool)
    flat = flat[present].astype(np.int64)
    row = row[present]
    pos = np.searchsorted(col_to_item, flat)
    keep = (pos < len(col_to_item)) & (col_to_item[np.minimum(pos, len(col_to_item) - 1)] == flat)
    row, pos = row[keep], pos[keep]
    order = np.lexsort((pos, row))
    row, pos = row[order], pos[order]
    dup = np.zeros(len(row), dtype=bool)
    dup[1:] = (row[1:] == row[:-1]) & (pos[1:] == pos[:-1])
    row, pos = row[~dup], pos[~dup]
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(np.bincount(row, minlength=n_rows), out=indptr[1:])
    return indptr, pos.astype(np.int64), n_rows


def row_ids(indptr: np.ndarray) -> np.ndarray:
    """The row of every CSR entry."""
    return np.repeat(np.arange(len(indptr) - 1, dtype=np.int64), np.diff(indptr))
