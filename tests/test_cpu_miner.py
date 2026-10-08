"""The array-based CPU miner (core/cpu_miner.py): each building block against a brute force,
and the route end to end against efficient-apriori (oracle convention, CLAUDE.md)."""

from __future__ import annotations

import itertools
import warnings
from concurrent.futures import ThreadPoolExecutor

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


def test_item_counts_are_int64():
    """Polars counts in UInt32, which wraps past 2**32 entries; the CSR's size is summed from these counts."""
    column = pl.Series("items", [[1, 2], [1], [2, 2]])
    lens = column.list.len().to_numpy().astype(np.int64)
    assert cpu_miner._count_items(column, lens).schema["count"] == pl.Int64
    assert cpu_miner._count_items(column.clear(), lens[:0]).schema["count"] == pl.Int64


def test_csr_and_route_with_an_int64_indptr(monkeypatch):
    monkeypatch.setattr(cpu_miner, "INDPTR32_LIMIT", 0)
    rows = _random_rows(3)
    df = pl.DataFrame({"items": rows}, schema={"items": pl.List(pl.Int64)})
    tc = cpu_miner.build_transaction_csr(df.lazy(), 0.03, "items")
    assert tc.indptr.dtype == np.int64
    assert np.array_equal(tc.to_scipy().toarray().astype(bool), _presence(rows, tc.items.to_list()))
    assert _mined(df, 0.03) == _oracle(rows, 0.03)


# ── L1: K=2 from one Gram matrix ────────────────────────────────────────────


def _random_csr(seed: int, n_rows: int = 600, n_cols: int = 30):
    rng = np.random.default_rng(seed)
    dense = rng.random((n_rows, n_cols)) < rng.uniform(0.02, 0.4, size=n_cols)
    from scipy.sparse import csr_matrix

    return dense, csr_matrix(dense.astype(np.int32))


@pytest.mark.parametrize("budget", [cpu_miner.GRAM_BUDGET_BYTES, 12 * 30 * 4, 1])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_count_pairs_matches_a_brute_force(monkeypatch, seed, budget):
    """Whole product, blocks of 4 columns, and one column per block give the same pairs."""
    monkeypatch.setattr(cpu_miner, "GRAM_BUDGET_BYTES", budget)
    dense, m = _random_csr(seed)
    rng = np.random.default_rng(seed + 100)
    gen = rng.random(dense.shape[1]) < 0.8
    min_count = 15
    want = {}
    for a, b in itertools.combinations(range(dense.shape[1]), 2):
        c = int((dense[:, a] & dense[:, b]).sum())
        if gen[a] and gen[b] and c >= min_count:
            want[(a, b)] = c
    pairs, counts = cpu_miner.count_pairs(m, gen, min_count)
    assert dict(zip(map(tuple, pairs.tolist()), counts.tolist())) == want
    assert pairs.dtype == np.int32 and counts.dtype == np.int64
    assert np.all(np.diff(pairs[:, 0]) >= 0) and len(pairs) == len(set(map(tuple, pairs.tolist())))
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    assert np.array_equal(order, np.arange(len(pairs))), "pairs must come out lexsorted"


@pytest.mark.parametrize("seed", [0, 1])
def test_count_pairs_bitvec_matches_the_gram(monkeypatch, seed):
    monkeypatch.setattr(cpu_miner, "AND_CHUNK_BYTES", 64)
    dense, m = _random_csr(seed, n_rows=700, n_cols=20)
    gen = np.flatnonzero(np.random.default_rng(seed).random(20) < 0.8)
    mask = np.zeros(20, dtype=bool)
    mask[gen] = True
    p1, c1 = cpu_miner.count_pairs(m, mask, 12)
    p2, c2 = cpu_miner.count_pairs_bitvec(cpu_miner.build_bitvecs(m.indptr, m.indices, 20), gen, 12)
    assert np.array_equal(p1, p2) and np.array_equal(c1, c2)


@pytest.mark.parametrize("ratio", [0.0, 1e12])
def test_both_k2_paths_give_the_route_the_same_result(monkeypatch, ratio):
    monkeypatch.setattr(cpu_miner, "GRAM_BITVEC_RATIO", ratio)
    df = pl.DataFrame({"items": _dense_rows(4, n_rows=500)})
    assert _mined(df, 0.05) == _brute_lattice(df["items"].to_list(), 0.05)[0]


@pytest.mark.parametrize("ratio", [0.0, 1e12])
def test_min_support_zero_keeps_the_occurring_itemsets_on_both_k2_paths(monkeypatch, ratio):
    """At min_count 0 the bitvectors would keep pairs that never co-occur, which the Gram cannot see."""
    monkeypatch.setattr(cpu_miner, "GRAM_BITVEC_RATIO", ratio)
    df = pl.DataFrame({"items": [[1], [2], [3, 4, 5]]})
    want = {s: 1 for s in [(1,), (2,), (3,), (4,), (5,), (3, 4), (3, 5), (4, 5), (3, 4, 5)]}
    assert _mined(df, 0.0) == want


def test_count_pairs_on_fewer_than_two_columns():
    _, m = _random_csr(0, n_cols=1)
    pairs, counts = cpu_miner.count_pairs(m, np.ones(1, dtype=bool), 1)
    assert pairs.shape == (0, 2) and len(counts) == 0


# ── L4: K>=3 candidates on sorted int32 arrays ──────────────────────────────


def _random_level(seed: int, k: int, n_cols: int = 14, n: int = 300) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = {tuple(sorted(rng.choice(n_cols, size=k, replace=False).tolist())) for _ in range(n)}
    arr = np.array(sorted(rows), dtype=np.int32)
    return arr


def _reference_candidates(level: np.ndarray, k: int) -> np.ndarray:
    from et_miner.core.candidates import _generate_candidates

    names = [f"i_{i:02d}" for i in range(64)]
    out = _generate_candidates([tuple(names[c] for c in row) for row in level.tolist()], k)
    arr = np.array([[int(c[2:]) for c in t] for t in out], dtype=np.int32).reshape(-1, k)
    return arr[np.lexsort(arr.T[::-1])] if len(arr) else arr


@pytest.mark.parametrize("chunk", [cpu_miner.CAND_CHUNK, 7, 1])
@pytest.mark.parametrize("k", [3, 4, 5, 6])
@pytest.mark.parametrize("seed", [0, 1])
def test_generate_candidates_matches_the_reference(monkeypatch, seed, k, chunk):
    monkeypatch.setattr(cpu_miner, "CAND_CHUNK", chunk)
    level = _random_level(seed, k - 1)
    got = list(cpu_miner.generate_candidates(level, k, 14))
    got = np.concatenate(got) if got else np.empty((0, k), dtype=np.int32)
    assert np.array_equal(got, _reference_candidates(level, k))


@pytest.mark.parametrize("path", ["mask", "keys", "bytes"])
def test_every_subset_test_path_agrees(monkeypatch, path):
    """The pair mask (K=3), packed keys, and the byte-wise fallback give the same candidates."""
    if path == "keys":
        monkeypatch.setattr(cpu_miner, "PAIR_MASK_BYTES", 0)
    if path == "bytes":
        monkeypatch.setattr(cpu_miner, "PAIR_MASK_BYTES", 0)
        monkeypatch.setattr(cpu_miner, "_pack", lambda rows, base: None)
    for k in (3, 4):
        level = _random_level(3, k - 1)
        got = np.concatenate(list(cpu_miner.generate_candidates(level, k, 14)))
        assert np.array_equal(got, _reference_candidates(level, k))


def test_generate_candidates_on_tiny_levels():
    assert list(cpu_miner.generate_candidates(np.empty((0, 2), dtype=np.int32), 3, 5)) == []
    assert list(cpu_miner.generate_candidates(np.array([[0, 1]], dtype=np.int32), 3, 5)) == []
    lone = np.array([[0, 1], [0, 2]], dtype=np.int32)
    assert list(cpu_miner.generate_candidates(lone, 3, 5)) == []
    tri = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int32)
    assert np.array_equal(np.concatenate(list(cpu_miner.generate_candidates(tri, 3, 5))), [[0, 1, 2]])


# ── L3: K>=3 counting per prefix group ──────────────────────────────────────


@pytest.mark.parametrize("chunk", [cpu_miner.BITVEC_CHUNK, 5])
@pytest.mark.parametrize("n_rows", [1, 63, 64, 65, 600])
def test_bitvecs_match_the_dense_matrix(monkeypatch, n_rows, chunk):
    monkeypatch.setattr(cpu_miner, "BITVEC_CHUNK", chunk)
    dense, m = _random_csr(n_rows, n_rows=n_rows, n_cols=9)
    bv = cpu_miner.build_bitvecs(m.indptr, m.indices, 9)
    bits = np.unpackbits(bv.view(np.uint8), axis=1, bitorder="little")[:, :n_rows].astype(bool)
    assert np.array_equal(bits, dense.T)
    assert not np.unpackbits(bv.view(np.uint8), axis=1, bitorder="little")[:, n_rows:].any()


@pytest.mark.parametrize("chunk", [cpu_miner.BITVEC_CHUNK, 5, 70])
def test_bitvecs_over_a_row_subset(monkeypatch, chunk):
    """Rows are renumbered in order; chunk boundaries fall on 64-row words."""
    monkeypatch.setattr(cpu_miner, "BITVEC_CHUNK", chunk)
    dense, m = _random_csr(8, n_rows=500, n_cols=7)
    rows = np.flatnonzero(np.random.default_rng(1).random(500) < 0.6)
    bv = cpu_miner.build_bitvecs(m.indptr, m.indices, 7, rows)
    bits = np.unpackbits(bv.view(np.uint8), axis=1, bitorder="little")[:, : len(rows)].astype(bool)
    assert np.array_equal(bits, dense[rows].T)


@pytest.mark.parametrize("budget", [cpu_miner.GRAM_BUDGET_BYTES, 12 * 4 * 2, 1])
@pytest.mark.parametrize("chunk", [cpu_miner.BITVEC_CHUNK, 7])
def test_row_space_gram_pairs_match_a_dense_product(monkeypatch, chunk, budget):
    """The whole Gram, row blocks of two and one row per block give the same pair counts."""
    monkeypatch.setattr(cpu_miner, "BITVEC_CHUNK", chunk)
    dense, m = _random_csr(9, n_rows=300, n_cols=10)
    space = cpu_miner.RowSpace(m.indptr, m.indices, 10)
    rows = np.flatnonzero(dense[:, 2])
    suffix = np.array([1, 4, 5, 8, 9])
    ia, ib = (np.array(x) for x in zip(*itertools.combinations(range(1, 5), 2)))
    x = dense[rows][:, suffix].astype(np.int64)
    assert np.array_equal(space.gram_pairs(rows, suffix, ia, ib, budget), (x.T @ x)[ia, ib])
    assert len(space.gram_pairs(rows, suffix, ia[:0], ib[:0], budget)) == 0


@pytest.mark.parametrize("table", [False, True])
def test_popcount_with_and_without_bitwise_count(monkeypatch, table):
    if table:
        monkeypatch.setattr(cpu_miner, "_HAS_BITWISE_COUNT", False)
    a = np.random.default_rng(0).integers(0, 2**63, size=(17, 5), dtype=np.uint64)
    want = [sum(bin(int(x)).count("1") for x in row) for row in a]
    assert cpu_miner.popcount_rows(a).tolist() == want


def _brute_counts(dense: np.ndarray, cands: np.ndarray) -> np.ndarray:
    return np.array([int(np.logical_and.reduce(dense[:, list(c)], axis=1).sum()) for c in cands], dtype=np.int64)


def _all_candidates(n_cols: int, k: int) -> np.ndarray:
    return np.array(list(itertools.combinations(range(n_cols), k)), dtype=np.int32)


@pytest.mark.parametrize("path", ["bitvec", "proj", "tidset", "compacted"])
@pytest.mark.parametrize("k", [3, 4, 5])
def test_count_candidates_matches_a_brute_force(monkeypatch, path, k):
    dense, m = _random_csr(11, n_rows=700, n_cols=12)
    cands = _all_candidates(12, k)
    if path == "bitvec":
        monkeypatch.setattr(cpu_miner, "PROJ_MIN_SUFFIXES", ((float("inf"), 10**9),))
    if path == "proj":
        monkeypatch.setattr(cpu_miner, "PROJ_MIN_SUFFIXES", ((float("inf"), 0),))
    if path == "tidset":
        monkeypatch.setattr(cpu_miner, "BITVEC_BUDGET_BYTES", 0)
    rows = None
    if path == "compacted":
        rows = np.flatnonzero(dense.sum(axis=1) >= k)
    space = cpu_miner.RowSpace(m.indptr, m.indices, 12, rows)
    assert (space.bitvecs is None) == (path == "tidset")
    got = cpu_miner.count_candidates(cands, k, space)
    assert np.array_equal(got, _brute_counts(dense, cands))


@pytest.mark.parametrize("path", ["proj", "tidset"])
def test_projection_under_a_tiny_gram_budget_matches_a_brute_force(monkeypatch, path):
    """One Gram row per block bounds the projection's memory and leaves its counts unchanged."""
    monkeypatch.setattr(cpu_miner, "GRAM_BUDGET_BYTES", 1)
    if path == "proj":
        monkeypatch.setattr(cpu_miner, "PROJ_MIN_SUFFIXES", ((float("inf"), 0),))
    else:
        monkeypatch.setattr(cpu_miner, "BITVEC_BUDGET_BYTES", 0)
    dense, m = _random_csr(13, n_rows=700, n_cols=12)
    for k in (3, 4):
        cands = _all_candidates(12, k)
        got = cpu_miner.count_candidates(cands, k, cpu_miner.RowSpace(m.indptr, m.indices, 12))
        assert np.array_equal(got, _brute_counts(dense, cands))
    rows = _random_rows(2, messy=False)
    assert _mined(pl.DataFrame({"items": rows}), 0.02) == _oracle(rows, 0.02)


def test_count_candidates_on_a_group_with_an_empty_prefix():
    """A prefix with no rows gives zero counts on both counters."""
    dense = np.zeros((70, 5), dtype=bool)
    dense[:, 1:] = True
    from scipy.sparse import csr_matrix

    m = csr_matrix(dense.astype(np.int32))
    cands = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 4]], dtype=np.int32)
    for steps in (((float("inf"), 10**9),), ((float("inf"), 0),)):
        cpu_miner.PROJ_MIN_SUFFIXES, saved = steps, cpu_miner.PROJ_MIN_SUFFIXES
        try:
            got = cpu_miner.count_candidates(cands, 3, cpu_miner.RowSpace(m.indptr, m.indices, 5))
        finally:
            cpu_miner.PROJ_MIN_SUFFIXES = saved
        assert got.tolist() == [0, 0, 0]


# ── Counting given itemsets (SON's pass 2) ──────────────────────────────────


@pytest.mark.parametrize("packed", [True, False])
def test_sum_rows_matches_numpy(monkeypatch, packed):
    rng = np.random.default_rng(3)
    sets = np.sort(rng.integers(0, 40, size=(500, 3)), axis=1).astype(np.int32)
    w = rng.integers(1, 9, size=500).astype(np.int32)
    if not packed:
        monkeypatch.setattr(cpu_miner, "_pack", lambda rows, base: None)
    got, (summed, ones) = cpu_miner.sum_rows(sets, 40, [w, np.ones(500, dtype=np.int64)])
    want, inverse, n = np.unique(sets, axis=0, return_inverse=True, return_counts=True)
    assert got.dtype == np.int32
    assert summed.dtype == np.int32
    assert np.array_equal(got, want)
    assert np.array_equal(summed, np.bincount(inverse.ravel(), weights=w).astype(np.int32))
    assert np.array_equal(ones, n)


def test_sum_rows_of_nothing():
    got, (w,) = cpu_miner.sum_rows(np.empty((0, 2), dtype=np.int32), 5, [np.empty(0, dtype=np.int32)])
    assert got.shape == (0, 2)
    assert len(w) == 0


def test_count_per_candidate_matches_a_brute_force(monkeypatch):
    """Blocks of one candidate and one block of all give the same counts."""
    dense, m = _random_csr(21, n_rows=700, n_cols=12)
    bv = cpu_miner.build_bitvecs(m.indptr, m.indices, 12)
    for k in (2, 3, 4):
        cands = _all_candidates(12, k)
        for chunk in (1, cpu_miner.AND_CHUNK_BYTES):
            monkeypatch.setattr(cpu_miner, "AND_CHUNK_BYTES", chunk)
            assert np.array_equal(cpu_miner.count_per_candidate(cands, bv), _brute_counts(dense, cands))


def _sparse_candidates(n_cols: int, seed: int) -> dict[int, np.ndarray]:
    """Lexsorted samples of every length 1-4: groups of one candidate and groups of many."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in (1, 2, 3, 4):
        every = _all_candidates(n_cols, k)
        out[k] = every[np.sort(rng.choice(len(every), size=min(len(every), 40 * k), replace=False))]
    return out


_COUNT_PATHS = {
    "bitvec-percand": {"PER_CANDIDATE_MAX_WORDS": 10**9},
    "gram-groups": {"GRAM_BITVEC_RATIO": 0.0, "PER_CANDIDATE_MAX_WORDS": 0, "PER_CANDIDATE_MEAN_GROUP": 0.0},
    "no-bitvecs": {"BITVEC_BUDGET_BYTES": 0},
    "gram-tiny-budget": {"GRAM_BITVEC_RATIO": 0.0, "GRAM_BUDGET_BYTES": 1},
}


@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize("path", sorted(_COUNT_PATHS))
def test_count_itemsets_matches_a_brute_force(monkeypatch, path, pooled):
    for name, value in _COUNT_PATHS[path].items():
        monkeypatch.setattr(cpu_miner, name, value)
    dense, m = _random_csr(23, n_rows=900, n_cols=14)
    dense[::3] = False  # short rows, so the K>=3 row spaces compact
    from scipy.sparse import csr_matrix

    m = csr_matrix(dense.astype(np.int32))
    cands = _sparse_candidates(14, 5)
    pool = ThreadPoolExecutor(3) if pooled else None
    try:
        got = cpu_miner.count_itemsets(m.indptr, m.indices, 14, cands, pool, 3 if pooled else 1)
    finally:
        if pool is not None:
            pool.shutdown()
    assert sorted(got) == [1, 2, 3, 4]
    for k, c in cands.items():
        assert np.array_equal(got[k], _brute_counts(dense, c)), f"k={k}"


@pytest.mark.parametrize(
    ("max_words", "spread", "want"),
    [
        (10**9, False, "count_per_candidate"),  # few words: per candidate whatever the groups
        (0, True, "count_per_candidate"),  # many words, one candidate per prefix group
        (0, False, "count_candidates"),  # many words, one large prefix group
    ],
)
def test_count_itemsets_dispatches_on_words_and_group_size(monkeypatch, max_words, spread, want):
    monkeypatch.setattr(cpu_miner, "PER_CANDIDATE_MAX_WORDS", max_words)
    dense, m = _random_csr(29, n_rows=300, n_cols=12)
    every = _all_candidates(12, 3)
    cands = every[np.r_[True, every[1:, 0] != every[:-1, 0]]] if spread else every[every[:, 0] == 0]
    calls = []
    for name in ("count_per_candidate", "count_candidates"):
        real = getattr(cpu_miner, name)
        monkeypatch.setattr(cpu_miner, name, lambda *a, _r=real, _n=name, **k: calls.append(_n) or _r(*a, **k))
    got = cpu_miner.count_itemsets(m.indptr, m.indices, 12, {3: cands})
    assert calls == [want]
    assert np.array_equal(got[3], _brute_counts(dense, cands))


# ── L5: array levels, free-sets, Pascal, emission ───────────────────────────


@pytest.mark.parametrize("fallback", [False, True])
def test_level_index_finds_rows_on_both_paths(monkeypatch, fallback):
    if fallback:
        monkeypatch.setattr(cpu_miner, "_pack", lambda rows, base: None)
    level = _random_level(4, 3)
    index = cpu_miner._LevelIndex(level, 14)
    assert np.array_equal(index.find(level), np.arange(len(level)))
    absent = np.array([[13, 13, 13]], dtype=np.int32)
    assert index.find(absent).tolist() == [-1]


def _brute_lattice(rows: list, min_support: float) -> tuple[dict, set]:
    """(complete frequent lattice {itemset: count}, free-sets) by enumeration."""
    n = len(rows)
    sets = [set(r) for r in rows]
    items = sorted({x for r in sets for x in r})
    min_count = _min_count(min_support, n)
    counts: dict = {}
    level = [(i,) for i in items]
    while level:
        nxt = []
        for c in level:
            v = sum(1 for r in sets if r.issuperset(c))
            if v >= min_count:
                counts[c] = v
                nxt.append(c)
        level = sorted({tuple(sorted(set(a) | {b[-1]})) for a in nxt for b in nxt if a[:-1] == b[:-1] and a[-1] < b[-1]})
    free = {c for c, v in counts.items() if v != n
            and not any(counts.get(s) == v for r in range(1, len(c)) for s in itertools.combinations(c, r))}
    return counts, free


def _dense_rows(seed: int, n_rows: int = 250, n_items: int = 10) -> list:
    """Rows from correlated items, so many itemsets share counts (non-free) and Pascal has work."""
    rng = np.random.default_rng(seed)
    base = rng.random((n_rows, n_items)) < 0.45
    base[:, 1] |= base[:, 0]
    base[:, 3] = base[:, 2]
    base[: n_rows // 2, 5] = True
    return [sorted(np.flatnonzero(r).tolist()) for r in base]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_free_sets_and_pascal_match_a_brute_force(seed):
    rows = _dense_rows(seed)
    df = pl.DataFrame({"items": rows})
    counts, free = _brute_lattice(rows, 0.1)
    assert _mined(df, 0.1) == counts
    assert _mined(df, 0.1, use_generator_pruning=True) == counts
    want_free = {c: counts[c] for c in free}
    assert _mined(df, 0.1, prune_equal_support=True) == want_free
    assert _mined(df, 0.1, prune_equal_support=True, use_generator_pruning=True) == want_free


def test_pascal_infers_and_the_profile_says_so():
    df = pl.DataFrame({"items": _dense_rows(5)})
    _, session = apriori(df, min_support=0.1, use_generator_pruning=True, profile=True)
    inferred = sum(p.extra.get("n_inferred", 0) for p in session.phases)
    assert inferred > 0
    names = [p.name for p in session.phases]
    assert names[:4] == ["matrix_build", "k1_support", "k2_candidate_gen", "k2_support_count"]
    assert "k3_candidate_gen" in names and "k3_support_count" in names


def test_level_callback_matches_the_previous_engine_on_smoke():
    """Candidates per level after the subset test, as the campaign recorded them for the old engine."""
    from et_miner.synthetic import PRESETS, generate_transactions

    df, _ = generate_transactions(PRESETS["smoke"])
    seen = []
    apriori(df, min_support=0.01, level_callback=lambda k, c, f, ms: seen.append((k, c, f)))
    assert seen == [(1, 118, 118), (2, 6903, 290), (3, 560, 202), (4, 95, 63), (5, 18, 18), (6, 3, 3)]


@pytest.mark.parametrize("dtype,want", [
    (pl.Int32, pl.Int64), (pl.Int64, pl.Int64), (pl.UInt16, pl.Int64), (pl.UInt64, pl.UInt64),
    (pl.String, pl.String), (pl.Categorical, pl.String),
])
def test_emitted_item_types(dtype, want):
    raw = [[1, 2, 3], [1, 2], [2, 3], [1, 2, 3]]
    if dtype in (pl.String, pl.Categorical):
        raw = [[str(x) for x in r] for r in raw]
    df = pl.DataFrame({"items": raw}, schema={"items": pl.List(dtype)})
    res = apriori(df, min_support=0.5)
    assert res.schema["itemset"] == pl.List(want)
    assert res.schema["support"] == pl.Float64
    assert res.height == 7


def test_an_empty_result_keeps_the_schema():
    res = apriori(pl.DataFrame({"items": [[1], [2]]}), min_support=0.9)
    assert res.height == 0 and res.schema == {"itemset": pl.List(pl.Int64), "support": pl.Float64}


# ── n_jobs > 1: the same counts from a thread pool ──────────────────────────


class _RecordingPool(ThreadPoolExecutor):
    """A thread pool that records the functions it runs, so a test can show it reached the pooled kernels."""

    def __init__(self, n_workers: int) -> None:
        super().__init__(n_workers)
        self.ran: set[str] = set()

    def submit(self, fn, /, *args, **kwargs):
        self.ran.add(fn.__name__)
        return super().submit(fn, *args, **kwargs)


@pytest.mark.parametrize("path", ["bitvec", "proj", "tidset"])
def test_pool_variants_match_the_sequential_kernels(monkeypatch, path):
    """The thresholds that keep small inputs on the calling thread are lowered so every kernel uses the pool."""
    monkeypatch.setattr(cpu_miner, "PARALLEL_GRAM_WORK", 0)
    monkeypatch.setattr(cpu_miner, "HEAVY_GROUP_WORK", 0)
    monkeypatch.setattr(cpu_miner, "BITVEC_CHUNK", 50)
    if path == "proj":
        monkeypatch.setattr(cpu_miner, "PROJ_MIN_SUFFIXES", ((float("inf"), 0),))
        monkeypatch.setattr(cpu_miner, "GRAM_BUDGET_BYTES", 12 * 16 * 4)
    if path == "tidset":
        monkeypatch.setattr(cpu_miner, "BITVEC_BUDGET_BYTES", 0)
    dense, m = _random_csr(21, n_rows=900, n_cols=16)
    gen = np.ones(16, dtype=bool)
    with _RecordingPool(4) as pool:
        p1, c1 = cpu_miner.count_pairs(m, gen, 10)
        p4, c4 = cpu_miner.count_pairs(m, gen, 10, pool, 4)
        assert np.array_equal(p1, p4) and np.array_equal(c1, c4)
        assert "block" in pool.ran
        if path != "tidset":
            assert np.array_equal(cpu_miner.build_bitvecs(m.indptr, m.indices, 16),
                                  cpu_miner.build_bitvecs(m.indptr, m.indices, 16, pool=pool))
            assert "fill" in pool.ran
        cands = _all_candidates(16, 4)
        space = cpu_miner.RowSpace(m.indptr, m.indices, 16, pool=pool)
        assert (space.bitvecs is None) == (path == "tidset")
        assert np.array_equal(cpu_miner.count_candidates(cands, 4, space, pool, 4), _brute_counts(dense, cands))
        assert "_count_groups" in pool.ran


@pytest.mark.parametrize("kw", [{}, {"prune_equal_support": True}, {"use_generator_pruning": True}])
def test_n_jobs_does_not_change_the_result(kw):
    df = pl.DataFrame({"items": _dense_rows(9, n_rows=600)})
    assert _mined(df, 0.05, n_jobs=4, **kw) == _mined(df, 0.05, n_jobs=1, **kw)


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


def test_streaming_single_chunk_fallback_warns_about_sparse_once_and_changes_nothing():
    df = pl.DataFrame({"items": _random_rows(7, messy=False)})
    plain = _mined(df, 0.03)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert _mined(df, 0.03, streaming=True, sparse=None) == plain
    for value in (True, False):
        with pytest.warns(DeprecationWarning, match="sparse= no longer selects") as record:
            assert _mined(df, 0.03, streaming=True, sparse=value) == plain
        assert sum(issubclass(w.category, DeprecationWarning) for w in record) == 1
