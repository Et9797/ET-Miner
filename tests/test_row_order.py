"""M5b: the lex row order of the device input layer and the kernels' tile-occupancy skip.

Counts do not depend on the row order, and a tile an item does not occupy
contributes zero to every candidate holding the item. So the bitvectors under
"lex" are a per-shard row permutation of those under "keep", and every count,
with or without the occupancy skip, is the same.
"""

from __future__ import annotations

import itertools
from collections import Counter

import numpy as np
import polars as pl
import pytest

from et_miner.core.result import _min_count
from tests._oracle import ea_counts, mined_counts


def _gpu_count() -> int:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


def _needs(n_gpus: int) -> None:
    if _gpu_count() < n_gpus:
        pytest.skip(f"needs {n_gpus} CUDA devices")


def _n_gpus_params():
    return [1, pytest.param(2, marks=pytest.mark.multigpu)]


def _clustered_bitvecs(seed: int, n_cols: int, n_rows: int) -> np.ndarray:
    """Columns that hold rows in a few 2,048-row tiles only, and some that hold rows everywhere."""
    rng = np.random.default_rng(seed)
    words = -(-n_rows // 64)
    bv = np.zeros((n_cols, words), dtype=np.uint64)
    n_tiles = -(-n_rows // 2048)
    for c in range(n_cols):
        tiles = rng.choice(n_tiles, size=max(1, n_tiles // 3), replace=False) if c % 3 else np.arange(n_tiles)
        rows = np.concatenate([np.arange(t * 2048, min(n_rows, (t + 1) * 2048)) for t in tiles])
        rows = rng.choice(rows, size=len(rows) // 2, replace=False)
        np.bitwise_or.at(bv[c], rows // 64, np.left_shift(np.uint64(1), (rows % 64).astype(np.uint64)))
    return bv


@pytest.mark.gpu
@pytest.mark.parametrize("n_rows", [64, 2_047, 2_048, 2_049, 131_072 + 999])
@pytest.mark.parametrize("block", ["cols", "bytes"])
def test_tile_occupancy_matches_numpy(monkeypatch, block, n_rows):
    """Blocks of three columns, or of one column when the byte bound allows no more."""
    import cupy as cp

    from et_miner.gpu.kernels import occupancy

    monkeypatch.setattr(occupancy, "OCC_BLOCK_COLS" if block == "cols" else "OCC_BLOCK_BYTES", 3 if block == "cols" else 1)
    bv = _clustered_bitvecs(n_rows, 7, n_rows)
    occ = occupancy.tile_occupancy(cp.asarray(bv))
    n_tiles = -(-bv.shape[1] // 32)
    for c in range(7):
        for t in range(n_tiles):
            want = bool(bv[c, t * 32 : (t + 1) * 32].any())
            got = bool((int(occ.bits[c, t // 64]) >> (t % 64)) & 1)
            assert got == want, (c, t)
    assert occ.words == max(1, -(-n_tiles // 64))


@pytest.mark.gpu
@pytest.mark.parametrize("infer", [False, True])
def test_kernels_count_the_same_with_the_occupancy_skip(infer):
    """Per-candidate and group kernels, with and without the subset index, on clustered columns."""
    import cupy as cp

    from et_miner.gpu.kernels import (
        SUBSET_INFER,
        SUBSET_PRUNE,
        build_k3plus_groups_from_flat,
        count_group_pairs,
        count_k3plus_per_candidate,
        tile_occupancy,
        upload_subset_index,
    )

    n_rows = 50_000
    bv_host = _clustered_bitvecs(1, 12, n_rows)
    bv = cp.asarray(bv_host)
    occ = tile_occupancy(bv)
    pairs = np.array(list(itertools.combinations(range(12), 2)), dtype=np.int32)
    counts = np.array([int(np.bitwise_count(bv_host[a] & bv_host[b]).sum()) for a, b in pairs], dtype=np.int64)
    index = None
    if infer:
        free = np.ones(len(pairs), dtype=bool)
        free[::5] = False
        index = upload_subset_index(
            pairs, counts, free, mode=SUBSET_PRUNE | SUBSET_INFER, device_id=0, write_inferred=True
        )
    groups = build_k3plus_groups_from_flat(pairs)
    plain = count_k3plus_per_candidate(bv, groups, bv.shape[1], index=index).get()
    skipped = count_k3plus_per_candidate(bv, groups, bv.shape[1], index=index, occ=occ).get()
    grouped = count_group_pairs(bv, groups, bv.shape[1], index=index, occ=occ).get()
    np.testing.assert_array_equal(skipped, plain)
    np.testing.assert_array_equal(grouped, plain)


def _random_column(seed: int, n_rows: int, *, repeats: bool) -> pl.Series:
    rng = np.random.default_rng(seed)
    weights = 1.0 / np.arange(1, 31) ** 0.8
    weights /= weights.sum()
    rows = []
    for i in range(n_rows):
        m = 0 if i % 11 == 0 else int(rng.integers(1, 10))
        row = rng.choice(30, size=m, p=weights).tolist()
        rows.append(row + row[:2] if repeats and m and i % 5 == 0 else sorted(set(row)))
    return pl.Series("items", rows, dtype=pl.List(pl.Int64))


@pytest.mark.gpu
@pytest.mark.parametrize("n_gpus", _n_gpus_params())
@pytest.mark.parametrize("repeats", [False, True])
def test_lex_bitvecs_are_a_row_permutation_of_keep(n_gpus, repeats):
    """Per shard: the same column sums and pair counts, and the rows sorted by their keys."""
    _needs(n_gpus)
    from et_miner.gpu.input_layer import build_device_input

    column = _random_column(7, 10_001, repeats=repeats)
    keep = build_device_input(column, 0.01, n_gpus, keep_rows=False, row_order="keep")
    lex = build_device_input(column, 0.01, n_gpus, keep_rows=False, row_order="lex")
    assert keep.col_to_item == lex.col_to_item and lex.row_order == "lex"
    for (a, da, na), (b, db, nb) in zip(keep.bitvecs_list, lex.bitvecs_list):
        assert (da, na) == (db, nb)
        a, b = a.get(), b.get()
        np.testing.assert_array_equal(np.bitwise_count(a).sum(axis=1), np.bitwise_count(b).sum(axis=1))
        for i, j in itertools.combinations(range(len(a)), 2):
            assert int(np.bitwise_count(a[i] & a[j]).sum()) == int(np.bitwise_count(b[i] & b[j]).sum())
        assert not np.array_equal(a, b), "the lex order left the rows where they were"


def _tie_support(column: pl.Series) -> float:
    rows = [sorted(set(r)) for r in column.to_list()]
    counts = sorted(Counter(p for r in rows for p in itertools.combinations(r, 3)).values())
    c, n = counts[len(counts) // 2], len(rows)
    s = c / n
    while _min_count(s, n) > c:
        s = np.nextafter(s, 0.0)
    return float(s)


_PINS = {
    "dispatch": {},
    "per-candidate": {"ET_MINER_TILED_MIN_GROUP_PAIRS": str(10**12), "ET_MINER_SMALL_GROUP_KERNEL": "percand"},
    "group": {"ET_MINER_TILED_MIN_GROUP_PAIRS": str(10**12), "ET_MINER_SMALL_GROUP_KERNEL": "group"},
    "forced chunks": {"ET_MINER_MAX_CHUNK_CANDS": "41"},
}


@pytest.mark.gpu
@pytest.mark.parametrize("n_gpus", _n_gpus_params())
@pytest.mark.parametrize("pin", sorted(_PINS))
@pytest.mark.parametrize("option", ["plain", "count inference", "free sets"])
def test_mining_under_the_lex_order_matches_keep_and_the_oracle(monkeypatch, option, pin, n_gpus):
    _needs(n_gpus)
    from et_miner import apriori

    for name, value in _PINS[pin].items():
        monkeypatch.setenv(name, value)
    column = _random_column(11, 4_099, repeats=True)
    s = _tie_support(column)
    kw = {"use_generator_pruning": option == "count inference", "prune_equal_support": option == "free sets"}
    results = {}
    from et_miner.gpu import input_layer

    lex_bytes = input_layer.LEX_BYTES_PER_ROW
    for order in ("keep", "lex"):
        # keep: the lex sort's no-room fallback, forced by leaving it no room
        monkeypatch.setattr(input_layer, "LEX_BYTES_PER_ROW", 1 << 50 if order == "keep" else lex_bytes)
        res = apriori(column.to_frame(), min_support=s, use_gpu=True, n_gpus=n_gpus, **kw)
        results[order] = sorted(zip(map(tuple, res["itemset"].to_list()), res["support"].to_list()))
    assert results["lex"] == results["keep"]
    if option != "free sets":
        got = {s_: round(p * len(column)) for s_, p in results["lex"]}
        assert got == ea_counts(column.to_list(), s)
        assert (
            mined_counts(
                pl.DataFrame(
                    {"itemset": [list(x) for x, _ in results["lex"]], "support": [p for _, p in results["lex"]]}
                ),
                len(column),
            )
            == got
        )


@pytest.mark.gpu
def test_without_room_for_the_occupancy_every_tile_is_counted(monkeypatch):
    import cupy as cp

    from et_miner import apriori
    from et_miner.gpu import kernels

    tried = []

    def no_room(bitvecs_gpu):
        tried.append(1)
        raise cp.cuda.memory.OutOfMemoryError(1 << 40, 0)

    monkeypatch.setattr(kernels, "tile_occupancy", no_room)
    for name, value in _PINS["per-candidate"].items():
        monkeypatch.setenv(name, value)
    column = _random_column(11, 4_099, repeats=True)
    s = _tie_support(column)
    res = apriori(column.to_frame(), min_support=s, use_gpu=True)
    assert tried, "no launch asked for the occupancy"
    assert mined_counts(res, len(column)) == ea_counts(column.to_list(), s)


@pytest.mark.gpu
def test_the_lex_order_keeps_the_rows_where_its_sort_does_not_fit(monkeypatch):
    from et_miner.gpu import input_layer
    from et_miner.gpu.input_layer import build_device_input

    column = _random_column(11, 4_099, repeats=True)
    assert build_device_input(column, 0.01, 1, keep_rows=False, row_order="lex").row_order == "lex"
    monkeypatch.setattr(input_layer, "LEX_BYTES_PER_ROW", 1 << 50)
    assert build_device_input(column, 0.01, 1, keep_rows=False, row_order="lex").row_order == "keep"
