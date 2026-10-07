"""The array-based CPU miner (core/cpu_miner.py): each building block against a brute force,
and the route end to end against efficient-apriori (oracle convention, CLAUDE.md)."""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import polars as pl
import pytest

from et_miner import apriori
from et_miner.core import cpu_miner
from et_miner.core.result import _min_count


def _random_rows(seed: int, n_rows: int = 400, n_items: int = 25, messy: bool = True) -> list:
    """Rows of item ids; with ``messy`` also unsorted rows, repeats, empty and null lists, null items."""
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, n_items + 1) ** 0.8
    p /= p.sum()
    rows: list = []
    for _ in range(n_rows):
        row = rng.choice(n_items, size=int(rng.integers(0, 9)), replace=False, p=p).tolist()
        if messy:
            r = rng.random()
            if r < 0.05:
                row = None
            elif r < 0.10 and row:
                row = row + [row[0]]
            elif r < 0.13:
                row = row + [None]
            rng.shuffle(row) if row and None not in row else None
        else:
            row = sorted(row)
        rows.append(row)
    return rows


def _presence(rows: list, items: list) -> np.ndarray:
    col = {it: j for j, it in enumerate(items)}
    m = np.zeros((len(rows), len(items)), dtype=bool)
    for i, row in enumerate(rows):
        for it in row or []:
            if it in col:
                m[i, col[it]] = True
    return m


def _oracle(rows: list, min_support: float, max_length: int | None = None) -> dict[tuple, int]:
    ea = pytest.importorskip("efficient_apriori")
    tx = [tuple(x for x in (r or []) if x is not None) for r in rows]
    n = len(tx)
    ml = max_length or max((len(set(t)) for t in tx), default=1) or 1
    itemsets, _ = ea.itemsets_from_transactions(tx, (_min_count(min_support, n) - 0.5) / n, max_length=ml)
    return {tuple(sorted(k)): v for level in itemsets.values() for k, v in level.items()}


def _mined(df: pl.DataFrame, min_support: float, **kw) -> dict[tuple, int]:
    res = apriori(df, min_support=min_support, **kw)
    n = df.height
    return {tuple(sorted(s)): round(p * n) for s, p in zip(res["itemset"].to_list(), res["support"].to_list())}


# ── L2: the CSR build ───────────────────────────────────────────────────────


@pytest.mark.parametrize("chunk_nnz", [3, 50, 4_000_000])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_csr_matches_the_presence_matrix(monkeypatch, seed, chunk_nnz):
    monkeypatch.setattr(cpu_miner, "CSR_CHUNK_NNZ", chunk_nnz)
    rows = _random_rows(seed)
    df = pl.DataFrame({"items": rows}, schema={"items": pl.List(pl.Int64)})
    min_support = 0.03
    n = len(rows)
    tc = cpu_miner.build_transaction_csr(df.lazy(), min_support, "items")
    present = _presence(rows, sorted({x for r in rows for x in (r or []) if x is not None}))
    all_items = sorted({x for r in rows for x in (r or []) if x is not None})
    keep = present.sum(axis=0) >= _min_count(min_support, n)
    want_items = [it for it, k in zip(all_items, keep) if k]
    assert tc.items.to_list() == want_items
    m = tc.to_scipy().toarray().astype(bool)
    assert np.array_equal(m, present[:, keep])
    assert np.array_equal(tc.counts, present[:, keep].sum(axis=0))
    for i in range(n):
        row = tc.indices[tc.indptr[i] : tc.indptr[i + 1]]
        assert np.all(np.diff(row) > 0), f"row {i} not strictly increasing: {row}"


def test_csr_handles_string_items():
    rows = [["b", "a"], ["a", "c"], ["a", "b", "c"], [], None, ["c"]]
    tc = cpu_miner.build_transaction_csr(pl.DataFrame({"items": rows}).lazy(), 0.3, "items")
    assert tc.items.to_list() == ["a", "b", "c"]
    assert tc.counts.tolist() == [3, 2, 3]


def test_an_item_frequent_only_through_repeats_is_dropped():
    """Explode counts item 9 three times; it is in one row, below min_count 2."""
    rows = [[1, 2], [1, 2], [9, 9, 9]]
    tc = cpu_miner.build_transaction_csr(pl.DataFrame({"items": rows}).lazy(), 0.5, "items")
    assert tc.items.to_list() == [1, 2]
    assert tc.n_scanned == 3
    assert tc.counts.tolist() == [2, 2]


def test_nothing_frequent_returns_none():
    tc = cpu_miner.build_transaction_csr(pl.DataFrame({"items": [[1], [2], [3]]}).lazy(), 0.9, "items")
    assert tc is None


# ── the route end to end ────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("min_support", [0.02, 0.05])
def test_cpu_route_matches_the_oracle_on_messy_rows(seed, min_support):
    rows = _random_rows(seed)
    df = pl.DataFrame({"items": rows}, schema={"items": pl.List(pl.Int64)})
    assert _mined(df, min_support) == _oracle(rows, min_support)


@pytest.mark.parametrize("max_length", [1, 2, 3])
def test_cpu_route_honours_max_length(max_length):
    rows = _random_rows(5, messy=False)
    df = pl.DataFrame({"items": rows})
    got = _mined(df, 0.03, max_length=max_length)
    assert got == _oracle(rows, 0.03, max_length)
    assert max(len(s) for s in got) <= max_length


def test_cpu_route_keeps_string_items():
    rows = [list(t) for t in itertools.chain.from_iterable([[("a", "b", "c")] * 5, [("a", "b")] * 3, [("c",)] * 2])]
    res = apriori(pl.DataFrame({"items": rows}), min_support=0.3)
    assert res.schema["itemset"] == pl.List(pl.String)
    got = {tuple(s): round(p * len(rows)) for s, p in zip(res["itemset"].to_list(), res["support"].to_list())}
    assert got == _oracle(rows, 0.3)


def test_sparse_is_deprecated_on_the_cpu_route_and_changes_nothing():
    df = pl.DataFrame({"items": _random_rows(7, messy=False)})
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        plain = _mined(df, 0.03)
    for value in (True, False):
        with pytest.warns(DeprecationWarning, match="sparse= no longer selects"):
            assert _mined(df, 0.03, sparse=value) == plain
