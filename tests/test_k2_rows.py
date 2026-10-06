"""Row-wise K=2 counting (kernels/k2.py::count_pairs_k2_rows) against NumPy pair counts."""

from itertools import combinations

import numpy as np
import pytest

from et_miner.gpu.kernels.k2 import pair_j


def _tri(t: int) -> int:
    return t * (t - 1) // 2


def test_pair_j_inverts_the_pair_layout():
    for idx in range(20_000):
        b = pair_j(idx)
        assert _tri(b) <= idx < _tri(b + 1), idx
    for b in (2, 3, 35_000, 1 << 20, (1 << 31) - 1):
        assert pair_j(_tri(b)) == b
        assert pair_j(_tri(b + 1) - 1) == b


def _random_rows(rng, n_rows: int, n_pos: int, max_len: int):
    lens = rng.integers(0, max_len + 1, size=n_rows)
    lens[rng.integers(0, n_rows)] = n_pos  # one row holding every position
    rows = [np.sort(rng.choice(n_pos, size=int(m), replace=False)) for m in lens]
    row_ptr = np.zeros(n_rows + 1, dtype=np.int64)
    row_ptr[1:] = np.cumsum([len(r) for r in rows])
    return row_ptr, np.concatenate(rows).astype(np.int32)


def _numpy_pair_counts(row_ptr, pos, n_pos: int) -> np.ndarray:
    counts = np.zeros(_tri(n_pos), dtype=np.int64)
    for r in range(len(row_ptr) - 1):
        for a, b in combinations(pos[row_ptr[r]:row_ptr[r + 1]].tolist(), 2):
            counts[_tri(b) + a] += 1
    return counts


def _ranges(n_pairs: int, rng) -> list[tuple[int, int]]:
    out = [(0, n_pairs), (0, 1), (n_pairs - 1, 1), (_tri(7) + 3, 50)]
    for _ in range(6):
        lo = int(rng.integers(0, n_pairs))
        out.append((lo, int(rng.integers(1, n_pairs - lo + 1))))
    return out


@pytest.mark.gpu
@pytest.mark.parametrize("variant", ["shared", "global"])
@pytest.mark.parametrize("n_pos", [40, 200])
def test_row_wise_counts_match_numpy(monkeypatch, variant, n_pos):
    pytest.importorskip("cupy")
    from et_miner.gpu.kernels import k2

    if variant == "global":
        monkeypatch.setattr(k2, "K2_ROWS_SHARED_PAIRS", 0)
    rng = np.random.default_rng(n_pos)
    row_ptr, pos = _random_rows(rng, 3_000, n_pos, 30)
    expected = _numpy_pair_counts(row_ptr, pos, n_pos)
    n_pairs = _tri(n_pos)
    rows = k2.upload_k2_rows(row_ptr, pos, 0)
    for lo, size in _ranges(n_pairs, rng):
        if variant == "shared" and size > k2.K2_ROWS_SHARED_PAIRS:
            continue
        got = k2.count_pairs_k2_rows(rows, lo, size)
        assert got.dtype == np.int32
        np.testing.assert_array_equal(got.get(), expected[lo:lo + size], err_msg=f"range [{lo}, {lo + size})")


@pytest.mark.gpu
def test_row_wise_shards_sum_to_the_global_counts():
    """Any row partition sums to the global counts, as the cross-GPU reduce requires."""
    cp = pytest.importorskip("cupy")
    from et_miner.gpu.kernels import k2

    n_dev = min(2, cp.cuda.runtime.getDeviceCount())
    rng = np.random.default_rng(7)
    row_ptr, pos = _random_rows(rng, 2_001, 60, 25)
    expected = _numpy_pair_counts(row_ptr, pos, 60)
    cut = 1_000
    shards = [(row_ptr[: cut + 1], pos[: row_ptr[cut]]), (row_ptr[cut:] - row_ptr[cut], pos[row_ptr[cut]:])]
    total = np.zeros(_tri(60), dtype=np.int64)
    for i, (rp, p) in enumerate(shards):
        total += k2.count_pairs_k2_rows(k2.upload_k2_rows(rp, p, i % n_dev), 0, _tri(60)).get()
    np.testing.assert_array_equal(total, expected)


@pytest.mark.gpu
def test_row_wise_empty_shard_and_short_rows():
    pytest.importorskip("cupy")
    from et_miner.gpu.kernels import k2

    rows = k2.upload_k2_rows(np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int32), 0)
    assert k2.count_pairs_k2_rows(rows, 0, 10).get().tolist() == [0] * 10
    rows = k2.upload_k2_rows(np.array([0, 1, 1, 3]), np.array([4, 0, 3], dtype=np.int32), 0)
    got = k2.count_pairs_k2_rows(rows, 0, _tri(5)).get()
    assert got.sum() == 1 and got[_tri(3) + 0] == 1


def test_row_shards_map_columns_to_frequent_positions():
    """A column outside the frequent list is dropped; positions are indices into it."""
    from scipy.sparse import csr_matrix

    from et_miner.gpu.row_split import _k2_row_shards

    rows = [[0, 2, 3], [1], [0, 1, 2, 3], [], [3, 2]]
    indptr = np.cumsum([0] + [len(r) for r in rows])
    csr = csr_matrix((np.ones(indptr[-1]), np.concatenate([np.array(r, dtype=np.int64) for r in rows]), indptr),
                     shape=(5, 4))
    assert not csr.has_canonical_format  # row 4 is unsorted
    (p0, q0), (p1, q1) = _k2_row_shards(csr, np.array([0, 2, 3]), [3, 2])
    assert p0.tolist() == [0, 3, 3, 6] and q0.tolist() == [0, 1, 2, 0, 1, 2]
    assert p1.tolist() == [0, 0, 2] and q1.tolist() == [1, 2]
    (p, q), = _k2_row_shards(csr, np.arange(4), [5])
    assert p.tolist() == [0, 3, 4, 8, 8, 10] and q.tolist() == [0, 2, 3, 1, 0, 1, 2, 3, 2, 3]
    with pytest.raises(ValueError, match="shards hold"):
        _k2_row_shards(csr, np.arange(4), [4])


@pytest.mark.gpu
def test_bitvecs_input_counts_k2_dense_whatever_the_pin(monkeypatch):
    """bitvecs= input has no rows to count, so a row-wise pin keeps the bitvec kernels."""
    pytest.importorskip("cupy")
    import polars as pl

    from et_miner import apriori
    from et_miner.gpu import kernels
    from et_miner.gpu.bitvec import _build_gpu_bitvec_matrix
    from scipy.sparse import csr_matrix

    rng = np.random.default_rng(3)
    dense = rng.random((500, 12)) < 0.4
    csr = csr_matrix(dense.astype(np.int8))
    calls = []
    monkeypatch.setattr(kernels, "count_pairs_k2_rows", lambda *a, **k: calls.append("rows"))
    for name in ("count_pairs_k2_shared", "count_pairs_k2_per_candidate"):
        def spy(*a, _real=getattr(kernels, name), **k):
            calls.append("dense")
            return _real(*a, **k)

        monkeypatch.setattr(kernels, name, spy)
    monkeypatch.setenv("ET_MINER_K2_KERNEL", "rows")
    got = apriori(None, min_support=0.1, bitvecs=(_build_gpu_bitvec_matrix(csr), {c: c for c in range(12)}, 500),
                  use_gpu=True, max_length=2)
    want = apriori(pl.DataFrame({"items": [np.flatnonzero(r).tolist() for r in dense]}), min_support=0.1,
                   max_length=2)
    assert calls and set(calls) == {"dense"}, calls
    assert sorted(map(tuple, got["itemset"].to_list())) == sorted(map(tuple, want["itemset"].to_list()))


def test_k2_kernel_knob_rejects_unknown_values(monkeypatch):
    from et_miner import _env

    monkeypatch.setenv("ET_MINER_K2_KERNEL", "Rows")
    assert _env.k2_kernel() == "rows"
    monkeypatch.setenv("ET_MINER_K2_KERNEL", "atomic")
    with pytest.raises(ValueError, match="ET_MINER_K2_KERNEL"):
        _env.k2_kernel()


def test_row_pairs_count_the_frequent_columns_of_each_row():
    """The dispatch's Σ C(len, 2) equals the one of the shards the row-wise kernel gets."""
    from scipy.sparse import csr_matrix

    from et_miner.gpu.row_split import _k2_row_pairs, _k2_row_shards

    rows = [[0, 2, 3], [1], [0, 1, 2, 3], [], [3, 2]]
    indptr = np.cumsum([0] + [len(r) for r in rows])
    csr = csr_matrix((np.ones(indptr[-1]), np.concatenate([np.array(r, dtype=np.int64) for r in rows]), indptr),
                     shape=(5, 4))
    for freq, want in ((np.array([0, 2, 3]), 3 + 0 + 3 + 0 + 1), (np.arange(4), 3 + 0 + 6 + 0 + 1)):
        assert _k2_row_pairs(csr, freq) == want
        shard_lens = [np.diff(p) for p, _ in _k2_row_shards(csr, freq, [2, 3])]
        assert sum(int((n * (n - 1) // 2).sum()) for n in shard_lens) == want


def test_dispatch_counts_rows_strictly_below_the_crossover(monkeypatch):
    from et_miner.gpu import row_split_chunks
    from et_miner.gpu.row_split_chunks import k2_counts_rows

    monkeypatch.delenv("ET_MINER_K2_KERNEL", raising=False)
    monkeypatch.setattr(row_split_chunks, "K2_ROWS_MAX_R", 0.004)
    # 1,000 pairs over 128 rows (2 words): rows below 8 row pairs.
    assert k2_counts_rows(7, 1_000, 128)
    assert not k2_counts_rows(8, 1_000, 128)
    assert not k2_counts_rows(0, 0, 128)


@pytest.mark.parametrize(("pin", "row_pairs", "want"), [("dense", 0, False), ("rows", 10**12, True)])
def test_the_pin_overrides_the_dispatch(monkeypatch, pin, row_pairs, want):
    from et_miner.gpu.row_split_chunks import k2_counts_rows

    monkeypatch.setenv("ET_MINER_K2_KERNEL", pin)
    assert k2_counts_rows(row_pairs, 1_000, 128) is want


@pytest.mark.gpu
@pytest.mark.parametrize(("n_items", "row_len", "want"), [(300, 2, "rows"), (12, 5, "dense")])
def test_unpinned_k2_follows_r(monkeypatch, n_items, row_len, want):
    """r = 1.4e-3 (300 items, pairs per row 1) counts row-wise, r = 9.5 (12 items) dense."""
    pytest.importorskip("cupy")
    import polars as pl

    from et_miner import apriori
    from et_miner.gpu import kernels

    rows = [sorted({(i * (7 * j + 1) + j) % n_items for j in range(row_len)}) for i in range(1_000)]
    rows = [r for r in rows if len(r) == row_len]
    calls = []
    for name in ("count_pairs_k2_rows", "count_pairs_k2_shared", "count_pairs_k2_per_candidate"):
        def spy(*a, _real=getattr(kernels, name), _name=name, **k):
            calls.append("rows" if _name == "count_pairs_k2_rows" else "dense")
            return _real(*a, **k)

        monkeypatch.setattr(kernels, name, spy)
    monkeypatch.delenv("ET_MINER_K2_KERNEL", raising=False)
    df = pl.DataFrame({"items": rows})
    got = apriori(df, min_support=0.5 / len(rows), use_gpu=True, max_length=2)
    want_set = apriori(df, min_support=0.5 / len(rows), max_length=2)
    assert calls and set(calls) == {want}, calls
    assert sorted(zip(map(tuple, got["itemset"].to_list()), got["support"].to_list())) == sorted(
        zip(map(tuple, want_set["itemset"].to_list()), want_set["support"].to_list()))
