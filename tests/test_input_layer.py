"""The device input layer (gpu/input_layer.py) against the host CSR path, and end to end against the oracle.

The bitvectors must be the host path's bytes: the same shards, the same
columns and the same bits; so must the row-wise K=2 rows cut from the device
entries, and the row pairs its dispatch reads. The edge cases are the
brief's: null lists and items, repeated items, unsorted and empty rows, ids
at the span limit, negative ids, UInt32 ids above 2**31, and row counts that
are not a multiple of 64.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from et_miner.core.cpu_miner import INT_SPAN_LIMIT
from et_miner.core.result import _min_count
from tests._oracle import ea_counts, mined_counts

REPO = Path(__file__).resolve().parents[1]
_SMALL_PRESETS = ("smoke", "wide_vocab")
_LARGE_PRESETS = ("deep_k", "skewed_rows", "oom_regression", "stress_k2", "deep_sparse_large", "deep_k20")


def _gpu_count() -> int:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


def _n_gpus_params():
    return [1, pytest.param(2, marks=pytest.mark.multigpu)]


def _needs(n_gpus: int) -> None:
    if _gpu_count() < n_gpus:
        pytest.skip(f"needs {n_gpus} CUDA devices")


def _random_column(
    seed: int,
    n_rows: int,
    dtype,
    vocab: np.ndarray,
    *,
    nulls: bool = False,
    repeats: bool = False,
    unsorted: bool = False,
) -> pl.Series:
    """Rows drawn from ``vocab`` with skewed weights (so some items are frequent), empty rows included."""
    rng = np.random.default_rng(seed)
    weights = 1.0 / np.arange(1, len(vocab) + 1)
    weights /= weights.sum()
    rows: list = []
    for _ in range(n_rows):
        m = int(rng.integers(0, 9))
        row = rng.choice(vocab, size=m, p=weights).tolist()
        if not repeats:
            row = sorted(set(row))
        elif m and rng.random() < 0.3:
            row = row + row[: int(rng.integers(1, m + 1))]
        if unsorted:
            rng.shuffle(row)
        elif repeats:
            row.sort()
        if nulls and row and rng.random() < 0.1:
            row.insert(int(rng.integers(0, len(row) + 1)), None)
        if nulls and rng.random() < 0.05:
            row = None
        rows.append(row)
    return pl.Series("items", rows, dtype=pl.List(dtype))


def _tie_support(column: pl.Series) -> float:
    """A min_support whose min_count equals some item's occurrence count: a tie at the threshold."""
    counts = column.explode().drop_nulls().value_counts().get_column("count").sort()
    n = len(column)
    c = int(counts[len(counts) // 2])
    s = c / n
    while _min_count(s, n) > c:
        s = np.nextafter(s, 0.0)
    assert _min_count(s, n) == c
    return float(s)


def _host(column: pl.Series, min_support: float, n_gpus: int):
    from et_miner.core.matrix import _build_csr_from_transactions
    from et_miner.gpu.csr_bitvec import build_bitvecs_row_split

    built = _build_csr_from_transactions(column.to_frame("items").lazy(), min_support, "items")
    if built is None:
        return None
    csr, col_to_item, n = built
    return csr, build_bitvecs_row_split(csr, n_gpus), col_to_item, n


def _freq_cols(bitvecs_list, min_count: int, choice: str) -> list[int]:
    import cupy as cp

    from et_miner.gpu.kernels import column_popcounts

    counts = 0
    for bv, did, _ in bitvecs_list:
        with cp.cuda.Device(did):
            counts = counts + column_popcounts(bv).get()
    if choice == "frequent":
        return np.flatnonzero(counts >= min_count).tolist()
    if choice == "alternate":
        return list(range(0, len(counts), 2))
    return list(range(len(counts)))


def _free_pools() -> None:
    import cupy as cp

    for d in range(cp.cuda.runtime.getDeviceCount()):
        with cp.cuda.Device(d):
            cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _assert_equal(got: np.ndarray, want: np.ndarray, what: str) -> None:
    assert got.dtype == want.dtype and got.shape == want.shape, (
        f"{what}: {got.dtype}{got.shape} != {want.dtype}{want.shape}"
    )
    if not np.array_equal(got, want):
        first = np.flatnonzero((got != want).ravel())[:5]
        raise AssertionError(f"{what}: differs at flat indices {first.tolist()}")


def _assert_same_input(column: pl.Series, min_support: float, n_gpus: int, cols: str = "frequent") -> None:
    """Device-built bitvectors, mapping and K=2 rows equal the host path's.

    The host path's results are copied to host memory and released before the
    device path runs, so the largest presets fit one card.
    """
    import cupy as cp
    import cupyx

    from et_miner.gpu.input_layer import build_device_input
    from et_miner.gpu.row_split import _HostRows

    host = _host(column, min_support, n_gpus)
    if host is None:
        assert build_device_input(column, min_support, n_gpus, keep_rows=True) is None
        return
    csr, bitvecs_list, col_to_item, n = host
    shards = [(d, r) for _, d, r in bitvecs_list]
    want_bitvecs = []
    for bv, _, _ in bitvecs_list:
        assert bv.flags.c_contiguous
        want_bitvecs.append(bv.get(out=cupyx.empty_pinned(bv.shape, dtype=bv.dtype)))
    del bv
    freq_cols = _freq_cols(bitvecs_list, _min_count(min_support, n), cols)
    host_rows = _HostRows(csr, bitvecs_list)
    want_pairs = host_rows.pairs(freq_cols)
    want_sums = host_rows.pairs_triples(freq_cols)
    want_rows = {d: (k.row_ptr.get(), k.pos.get(), k.n_rows) for d, k in host_rows.upload(freq_cols).items()}
    del host, csr, bitvecs_list, host_rows
    _free_pools()

    dev = build_device_input(column, min_support, n_gpus, keep_rows=True)
    assert dev.n_transactions == n
    assert dev.col_to_item == col_to_item
    assert [(d, r) for _, d, r in dev.bitvecs_list] == shards
    for (got, d, _), want in zip(dev.bitvecs_list, want_bitvecs):
        assert got.flags.c_contiguous and got.dtype == want.dtype and got.shape == want.shape
        step = max(1, (256 << 20) // max(1, want[:1].nbytes))
        with cp.cuda.Device(d):
            for c0 in range(0, len(want), step):
                same = cp.array_equal(got[c0 : c0 + step], cp.asarray(want[c0 : c0 + step]))
                assert bool(same), f"bitvecs of the shard on device {d} differ in columns {c0}..{c0 + step}"
    del got, want_bitvecs
    dev.bitvecs_list = []

    assert dev.rows is not None
    assert dev.rows.pairs(freq_cols) == want_pairs
    assert dev.rows.pairs_triples(freq_cols) == want_sums
    got_rows = dev.rows.upload(freq_cols)
    assert list(got_rows) == list(want_rows)
    for did, (row_ptr, pos, n_rows) in want_rows.items():
        got = got_rows[did]
        assert got.n_rows == n_rows
        assert got.row_ptr.device.id == did and got.pos.device.id == did
        _assert_equal(got.row_ptr.get(), row_ptr, f"K=2 row offsets on device {did}")
        _assert_equal(got.pos.get(), pos, f"K=2 positions on device {did}")
    del got_rows, dev
    _free_pools()


def _preset_column(name: str) -> tuple[pl.Series, float]:
    from et_miner.synthetic import PRESETS, generate_transactions

    path = REPO / "datasets" / "synth" / f"{name}.parquet"
    df = pl.read_parquet(path) if path.exists() else generate_transactions(PRESETS[name])[0]
    return df.get_column("items"), PRESETS[name].min_support


# ── bitvectors, mapping and K=2 rows equal the host path's ──────────────────


@pytest.mark.gpu
@pytest.mark.parametrize("n_gpus", _n_gpus_params())
@pytest.mark.parametrize("name", _SMALL_PRESETS)
def test_small_presets_match_the_host_path(name, n_gpus):
    _needs(n_gpus)
    _assert_same_input(*_preset_column(name), n_gpus)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("n_gpus", _n_gpus_params())
@pytest.mark.parametrize("name", _LARGE_PRESETS)
def test_large_presets_match_the_host_path(name, n_gpus):
    _needs(n_gpus)
    _assert_same_input(*_preset_column(name), n_gpus)


_DTYPE_VOCABS = {
    "Int8": (pl.Int8, np.array([-128, -127, -3, 0, 5, 126, 127])),
    "Int16": (pl.Int16, np.array([-32768, -5, 0, 9, 32767])),
    "Int32": (pl.Int32, np.array([-(2**31), -(2**31) + 7, -(2**31) + 1_000, -(2**31) + 2**22 - 1])),
    "Int64": (pl.Int64, np.array([2**40, 2**40 + 1, 2**40 + 77, 2**40 + 2**22 - 1])),
    "UInt8": (pl.UInt8, np.array([0, 1, 2, 200, 255])),
    "UInt16": (pl.UInt16, np.array([3, 4, 60_000, 65_535])),
    "UInt32": (pl.UInt32, np.array([2**31 - 2, 2**31, 2**31 + 5, 2**31 + 2**22 - 3])),
}


@pytest.mark.gpu
@pytest.mark.parametrize("n_gpus", _n_gpus_params())
@pytest.mark.parametrize("dtype", sorted(_DTYPE_VOCABS))
def test_every_served_dtype_matches_the_host_path(dtype, n_gpus):
    """Negative ids, UInt32 ids above 2**31 and ids at both ends of a 2**22 span."""
    _needs(n_gpus)
    pl_dtype, vocab = _DTYPE_VOCABS[dtype]
    column = _random_column(5, 1_001, pl_dtype, vocab)
    _assert_same_input(column, 0.05, n_gpus)


@pytest.mark.gpu
@pytest.mark.parametrize("n_gpus", _n_gpus_params())
@pytest.mark.parametrize(
    "shape",
    ["nulls", "repeats", "unsorted", "repeats-unsorted", "nulls-repeats-unsorted"],
)
@pytest.mark.parametrize("cols", ["frequent", "alternate", "every"])
def test_irregular_rows_match_the_host_path(shape, cols, n_gpus):
    """Null lists and items, repeated items and unsorted rows; the K=2 rows over several column sets."""
    _needs(n_gpus)
    vocab = np.arange(-20, 40, dtype=np.int64)
    column = _random_column(
        len(shape),
        2_003,
        pl.Int64,
        vocab,
        nulls="nulls" in shape,
        repeats="repeats" in shape,
        unsorted="unsorted" in shape,
    )
    _assert_same_input(column, _tie_support(column), n_gpus, cols)


@pytest.mark.gpu
@pytest.mark.parametrize("n_gpus", _n_gpus_params())
@pytest.mark.parametrize("n_rows", [1, 63, 64, 65, 255, 257, 4_097])
def test_row_counts_off_word_boundaries_match_the_host_path(n_rows, n_gpus):
    _needs(n_gpus)
    column = _random_column(n_rows, n_rows, pl.Int32, np.arange(7, dtype=np.int64))
    _assert_same_input(column, 1 / n_rows, n_gpus)


@pytest.mark.gpu
@pytest.mark.parametrize("n_gpus", _n_gpus_params())
def test_all_ones_words_match_the_host_path(n_gpus):
    """An item in every row: every word of its column is all ones."""
    _needs(n_gpus)
    rows = [[0, 1 + (r % 3)] for r in range(1_024)]
    _assert_same_input(pl.Series("items", rows, dtype=pl.List(pl.Int64)), 0.2, n_gpus)


@pytest.mark.gpu
@pytest.mark.parametrize("n_gpus", _n_gpus_params())
def test_chunked_and_sliced_columns_match_the_host_path(n_gpus):
    """Several Arrow chunks, one with null items (exploded by Polars), and a slice with a nonzero offset."""
    _needs(n_gpus)
    vocab = np.arange(0, 30, dtype=np.int64)
    parts = [
        _random_column(1, 700, pl.Int64, vocab),
        _random_column(2, 501, pl.Int64, vocab, nulls=True),
        _random_column(3, 333, pl.Int64, vocab, repeats=True, unsorted=True),
    ]
    column = pl.concat(parts, rechunk=False)
    assert column.n_chunks() == 3
    _assert_same_input(column, 0.03, n_gpus)
    _assert_same_input(column.slice(123, 1_200), 0.03, n_gpus)


@pytest.mark.gpu
def test_span_limit_is_served_and_beyond_it_is_not():
    from et_miner.gpu.input_layer import NOT_SERVED, build_device_input

    at_limit = pl.Series("items", [[0, INT_SPAN_LIMIT - 1]] * 70 + [[0]] * 6, dtype=pl.List(pl.Int64))
    _assert_same_input(at_limit, 0.5, 1)
    beyond = pl.Series("items", [[0, INT_SPAN_LIMIT]] * 4, dtype=pl.List(pl.Int64))
    assert build_device_input(beyond, 0.5, 1, keep_rows=True) is NOT_SERVED
    strings = pl.Series("items", [["a", "b"]] * 4)
    assert build_device_input(strings, 0.5, 1, keep_rows=True) is NOT_SERVED


@pytest.mark.gpu
@pytest.mark.parametrize(
    "rows",
    [[], [None, None], [[], []], [[None], [None, None]]],
    ids=["no-rows", "null-lists", "empty-lists", "null-items"],
)
def test_no_frequent_item_gives_none(rows):
    from et_miner.gpu.input_layer import build_device_input

    column = pl.Series("items", rows, dtype=pl.List(pl.Int64))
    assert _host(column, 0.5, 1) is None
    assert build_device_input(column, 0.5, 1, keep_rows=True) is None


@pytest.mark.gpu
def test_rows_are_not_kept_when_they_do_not_fit(monkeypatch):
    from et_miner.gpu import input_layer

    monkeypatch.setattr(input_layer, "KEEP_ROWS_BYTES", 1 << 50)
    column = _random_column(9, 500, pl.Int64, np.arange(10, dtype=np.int64))
    built = input_layer.build_device_input(column, 0.05, 1, keep_rows=True)
    assert built.rows is None
    assert input_layer.build_device_input(column, 0.05, 1, keep_rows=False).rows is None


@pytest.mark.gpu
@pytest.mark.parametrize("n_gpus", _n_gpus_params())
@pytest.mark.parametrize("block", [1, 5, 64])
def test_unsorted_rows_canonicalize_in_blocks(monkeypatch, block, n_gpus):
    """Blocks of a few entries: rows longer than a block, blocks that end inside a row, empty and null rows."""
    _needs(n_gpus)
    from et_miner.gpu import input_layer

    monkeypatch.setattr(input_layer, "CANON_BLOCK_ENTRIES", block)
    column = _random_column(
        17, 1_501, pl.Int64, np.arange(-3, 30, dtype=np.int64), nulls=True, repeats=True, unsorted=True
    )
    _assert_same_input(column, _tie_support(column), n_gpus, "every")


@pytest.mark.gpu
def test_unsorted_rows_are_cut_within_the_keep_decision(monkeypatch):
    """A card with just the room the keep decision asks for: canonicalizing and cutting unsorted rows fit it.

    The shard holds many blocks, so the room is about ``KEEP_ROWS_BYTES`` per
    entry, below what sorting every entry at once takes.
    """
    import cupy as cp

    from et_miner.gpu import input_layer

    monkeypatch.setattr(input_layer, "CANON_BLOCK_ENTRIES", 1 << 16)
    rng = np.random.default_rng(0)
    n_rows, width = 200_000, 15
    column = pl.Series("items", rng.integers(0, 60, size=n_rows * width)).reshape((n_rows, width))
    column = column.cast(pl.List(pl.Int64))
    built = input_layer.build_device_input(column, 0.01, 1, keep_rows=True)
    pool = cp.get_default_memory_pool()
    held = pool.used_bytes()
    (shard,) = built.rows._shards
    assert not shard.canonical
    need = input_layer._keep_bytes(shard)
    del built, shard
    _free_pools()
    pool.set_limit(size=held + need + (1 << 20))
    try:
        built = input_layer.build_device_input(column, 0.01, 1, keep_rows=True)
        assert built.rows is not None
        cols = list(range(len(built.col_to_item)))
        assert built.rows.pairs(cols, keep=True) > 0
        assert built.rows.upload(cols)[0].n_rows == n_rows
    finally:
        pool.set_limit(size=0)
        _free_pools()


@pytest.mark.gpu
def test_a_row_longer_than_a_block_is_cut_within_the_keep_decision(monkeypatch):
    """One unsorted row much longer than a sorting block: the keep decision budgets the block it makes."""
    import cupy as cp

    from et_miner.gpu import input_layer

    monkeypatch.setattr(input_layer, "CANON_BLOCK_ENTRIES", 1 << 12)
    rng = np.random.default_rng(3)
    rows = [rng.integers(0, 30, size=3).tolist() for _ in range(2_000)]
    rows.insert(700, rng.integers(0, 30, size=200_000).tolist())
    column = pl.Series("items", rows, dtype=pl.List(pl.Int64))
    built = input_layer.build_device_input(column, 0.01, 1, keep_rows=True)
    pool = cp.get_default_memory_pool()
    held = pool.used_bytes()
    (shard,) = built.rows._shards
    assert not shard.canonical
    need = input_layer._keep_bytes(shard)
    del built, shard
    _free_pools()
    pool.set_limit(size=held + need + (1 << 20))
    try:
        built = input_layer.build_device_input(column, 0.01, 1, keep_rows=True)
        assert built.rows is not None
        cols = list(range(len(built.col_to_item)))
        assert built.rows.pairs(cols, keep=True) is not None, "the rows ran out of the memory the decision gave them"
    finally:
        pool.set_limit(size=0)
        _free_pools()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "layer, where", [("device", "canonicalize"), ("device", "upload"), ("host", "upload")]
)
def test_rows_out_of_device_memory_give_way_to_the_bitvectors(monkeypatch, layer, where):
    """An OutOfMemoryError while the rows are cut or uploaded drops them; K=2 and K=3 count from the bitvectors."""
    import cupy as cp

    from et_miner import apriori
    from et_miner.gpu import input_layer, kernels, row_split

    def no_room(*a, **k):
        raise cp.cuda.memory.OutOfMemoryError(1 << 40, 0)

    launches = []
    for name in ("count_pairs_k2_rows", "count_k3_rows"):
        real = getattr(kernels, name)
        monkeypatch.setattr(kernels, name, lambda *a, _real=real, **k: launches.append(1) or _real(*a, **k))
    monkeypatch.setenv("ET_MINER_INPUT_LAYER", layer)
    monkeypatch.setenv("ET_MINER_K2_KERNEL", "rows")
    monkeypatch.setenv("ET_MINER_K3_KERNEL", "rows")
    column = _random_column(
        41, 1_777, pl.Int64, np.arange(0, 25, dtype=np.int64), nulls=True, repeats=True, unsorted=True
    )
    s = _tie_support(column)
    want = ea_counts(column.to_list(), s)
    res = apriori(column.to_frame(), min_support=s, use_gpu=True)
    assert launches, "the row-wise kernels did not run without the fault: the test would be vacuous"
    assert mined_counts(res, len(column)) == want

    if where == "canonicalize":
        monkeypatch.setattr(input_layer, "_canonicalize", no_room)
    elif layer == "device":
        monkeypatch.setattr(kernels, "K2Rows", no_room)
    else:
        monkeypatch.setattr(row_split, "_upload_k2_rows", no_room)
    launches.clear()
    res = apriori(column.to_frame(), min_support=s, use_gpu=True)
    assert not launches
    assert mined_counts(res, len(column)) == want


@pytest.mark.gpu
def test_the_vram_guard_drops_the_kept_rows_before_it_trips(monkeypatch):
    """The kept entries only speed up K=2 and K=3: a max_vram_gb the bitvectors fit passes as on the host layer."""
    from et_miner import apriori
    from et_miner.gpu import input_layer

    kept = []
    real = input_layer.build_device_input
    monkeypatch.setattr(
        input_layer, "build_device_input", lambda *a, **k: kept.append((built := real(*a, **k)).rows is not None) or built
    )
    rng = np.random.default_rng(1)
    n_rows, width = 1_000_000, 12
    column = pl.Series("items", rng.integers(0, 20, size=n_rows * width)).reshape((n_rows, width))
    frame = column.cast(pl.List(pl.Int64)).to_frame()
    results = {}
    for layer in ("host", "device"):
        monkeypatch.setenv("ET_MINER_INPUT_LAYER", layer)
        res = apriori(frame, min_support=0.05, use_gpu=True, max_length=3, max_vram_gb=0.02)
        results[layer] = sorted(zip(map(tuple, res["itemset"].to_list()), res["support"].to_list()))
        _free_pools()
    assert kept == [True], "the device layer did not keep its rows"
    assert results["device"] == results["host"] and results["host"]


# ── end to end: apriori through the device input layer ──────────────────────


@pytest.mark.gpu
@pytest.mark.parametrize("max_length", [1, 2])
def test_min_support_zero_makes_no_column_of_an_absent_id(monkeypatch, max_length):
    """At min_count 0 the columns are the ids that occur, as on the host layer, not every id of the span."""
    from et_miner import apriori

    frame = pl.DataFrame({"items": pl.Series([[1, 5], [5], [1, 5, 9]], dtype=pl.List(pl.Int64))})
    results = {}
    for layer in ("host", "device"):
        monkeypatch.setenv("ET_MINER_INPUT_LAYER", layer)
        res = apriori(frame, min_support=0.0, use_gpu=True, max_length=max_length)
        results[layer] = sorted(zip(map(tuple, res["itemset"].to_list()), res["support"].to_list()))
    assert results["device"] == results["host"]
    assert {i for itemset, _ in results["device"] for i in itemset} == {1, 5, 9}


@pytest.mark.gpu
@pytest.mark.parametrize("n_gpus", _n_gpus_params())
@pytest.mark.parametrize("k2", ["rows", "dense"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mining_through_the_device_layer_matches_the_oracle(monkeypatch, seed, k2, n_gpus):
    """Ties at min_count, empty and null rows, repeated items, unsorted rows; the row-wise and dense K=2."""
    _needs(n_gpus)
    from et_miner import apriori
    from et_miner.gpu import input_layer

    built = []
    real = input_layer.build_device_input
    monkeypatch.setattr(input_layer, "build_device_input", lambda *a, **k: built.append(1) or real(*a, **k))
    monkeypatch.setenv("ET_MINER_INPUT_LAYER", "device")
    monkeypatch.setenv("ET_MINER_K2_KERNEL", k2)
    column = _random_column(
        100 + seed,
        777 + 64 * seed,
        pl.Int64,
        np.arange(-5, 25, dtype=np.int64),
        nulls=True,
        repeats=seed > 0,
        unsorted=seed > 1,
    )
    s = _tie_support(column)
    res = apriori(column.to_frame(), min_support=s, use_gpu=True, n_gpus=n_gpus)
    assert built, "the device input layer did not run"
    assert mined_counts(res, len(column)) == ea_counts(column.to_list(), s)


@pytest.mark.gpu
def test_host_pin_skips_the_device_layer_and_unserved_columns_fall_back(monkeypatch):
    from et_miner import apriori
    from et_miner.gpu import input_layer

    built = []
    real = input_layer.build_device_input
    monkeypatch.setattr(input_layer, "build_device_input", lambda *a, **k: built.append(1) or real(*a, **k))
    column = _random_column(4, 600, pl.Int64, np.arange(0, 12, dtype=np.int64))
    expected = ea_counts(column.to_list(), 0.1)
    monkeypatch.setenv("ET_MINER_INPUT_LAYER", "host")
    assert mined_counts(apriori(column.to_frame(), min_support=0.1, use_gpu=True), 600) == expected
    assert not built
    monkeypatch.setenv("ET_MINER_INPUT_LAYER", "device")
    wide = column.list.eval(pl.element() * INT_SPAN_LIMIT)
    want = {tuple(i * INT_SPAN_LIMIT for i in s): c for s, c in expected.items()}
    assert mined_counts(apriori(wide.to_frame(), min_support=0.1, use_gpu=True), 600) == want
    assert built, "an unserved column must still be offered to the device layer first"


def test_input_layer_knob_is_validated(monkeypatch):
    from et_miner import _env

    monkeypatch.setenv("ET_MINER_INPUT_LAYER", "gpu")
    with pytest.raises(ValueError, match="ET_MINER_INPUT_LAYER"):
        _env.input_layer()
    monkeypatch.setenv("ET_MINER_INPUT_LAYER", " Device ")
    assert _env.input_layer() == "device"


@pytest.mark.gpu
def test_without_pyarrow_every_chunk_is_exploded(monkeypatch):
    """The route runs without pyarrow (its result frame has a fallback too): the chunks go through Polars."""
    import builtins

    real_import = builtins.__import__

    def no_pyarrow(name, *args, **kwargs):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError("injected: no pyarrow")
        return real_import(name, *args, **kwargs)

    column = pl.concat(
        [
            _random_column(1, 300, pl.Int64, np.arange(12, dtype=np.int64)),
            _random_column(2, 200, pl.Int64, np.arange(12, dtype=np.int64), repeats=True),
        ],
        rechunk=False,
    )
    from et_miner.gpu.input_layer import build_device_input

    want = build_device_input(column, 0.05, 1, keep_rows=False)
    monkeypatch.setattr(builtins, "__import__", no_pyarrow)
    got = build_device_input(column, 0.05, 1, keep_rows=False)
    assert got.col_to_item == want.col_to_item
    np.testing.assert_array_equal(got.bitvecs_list[0][0].get(), want.bitvecs_list[0][0].get())
