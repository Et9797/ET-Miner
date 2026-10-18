"""Row-wise K=3 (gpu/kernels/k3_rows.py) against brute force, and end to end against the oracle.

The slot layout gives pair (u, v) one slot per partner w of v; a slot counts
the rows holding (u, v, w) when (u, w) is frequent too and stays 0 otherwise.
The end-to-end cases are the brief's: ties at min_count, empty rows,
repeated items, rows of 64 active items, row counts that are not a multiple
of 64 or 256, one GPU and two.
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


def _random_rows(seed: int, n_rows: int, n_cols: int, max_len: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = 1.0 / np.arange(1, n_cols + 1)
    weights /= weights.sum()
    return [
        np.sort(rng.choice(n_cols, size=min(int(rng.integers(0, max_len + 1)), n_cols), replace=False, p=weights))
        for _ in range(n_rows)
    ]


def _level(rows, min_count: int) -> np.ndarray:
    counts = Counter(p for r in rows for p in itertools.combinations(r.tolist(), 2))
    return np.array(sorted(p for p, c in counts.items() if c >= min_count), dtype=np.int32).reshape(-1, 2)


def test_layout_holds_one_slot_per_partner_of_v():
    from et_miner.gpu.kernels import decode_k3_slots, k3_layout

    flat = _level(_random_rows(1, 2_000, 50, 10), 20)
    layout = k3_layout(flat)
    pairs = {tuple(p) for p in flat.tolist()}
    partners = {c: sorted(b for a, b in pairs if a == c) for c in np.unique(flat).tolist()}
    want = [(a, b, w) for a, b in sorted(pairs) for w in partners.get(b, [])]
    got = [tuple(t) for t in decode_k3_slots(np.arange(layout.n_slots), layout).tolist()]
    assert got == want
    degree = Counter(a for a, _ in pairs)
    assert layout.n_generated == sum(d * (d - 1) // 2 for d in degree.values())


def _rows_on(rows, layout, device_id: int):
    from et_miner.gpu.kernels import upload_k2_rows

    pos_of_col = {c: i for i, c in enumerate(layout.active_cols.tolist())}
    kept = [np.array([pos_of_col[c] for c in r.tolist() if c in pos_of_col], dtype=np.int32) for r in rows]
    row_ptr = np.zeros(len(rows) + 1, dtype=np.int64)
    row_ptr[1:] = np.cumsum([len(r) for r in kept])
    pos = np.concatenate(kept) if kept else np.zeros(0, dtype=np.int32)
    return upload_k2_rows(row_ptr, pos, device_id)


def _expected(rows, layout) -> np.ndarray:
    from et_miner.gpu.kernels import decode_k3_slots

    pairs = {tuple(p) for p in layout.pair_cols.tolist()}
    triples = Counter(t for r in rows for t in itertools.combinations(r.tolist(), 3))
    slots = decode_k3_slots(np.arange(layout.n_slots), layout).tolist()
    return np.array([triples[tuple(t)] if (t[0], t[2]) in pairs else 0 for t in slots], dtype=np.int64)


@pytest.mark.gpu
@pytest.mark.parametrize("bitmap", [True, False])
@pytest.mark.parametrize("seed", [0, 1])
def test_counts_match_brute_force(monkeypatch, bitmap, seed):
    from et_miner.gpu.kernels import build_k3_graph, count_k3_rows, k3_layout
    from et_miner.gpu.kernels import k3_rows as mod

    if not bitmap:
        monkeypatch.setattr(mod, "PAIR_BITS_MAX_BYTES", 0)
    rows = _random_rows(seed, 3_001, 70, 14)
    layout = k3_layout(_level(rows, 25))
    graph = build_k3_graph(layout, 0)
    assert (graph.bits_words > 0) == bitmap
    want = _expected(rows, layout)
    k2 = _rows_on(rows, layout, 0)
    np.testing.assert_array_equal(count_k3_rows(k2, graph, 0, layout.n_slots).get(), want)
    rng = np.random.default_rng(seed)
    for _ in range(8):
        lo = int(rng.integers(0, layout.n_slots))
        size = int(rng.integers(1, layout.n_slots - lo + 1))
        np.testing.assert_array_equal(count_k3_rows(k2, graph, lo, size).get(), want[lo : lo + size])


@pytest.mark.gpu
def test_rows_of_64_active_items_count_every_triple():
    """A row with K3_ROWS_MAX_LEN active items: 64 neighbour masks, every one full."""
    from et_miner.gpu.kernels import K3_ROWS_MAX_LEN, build_k3_graph, count_k3_rows, k3_layout

    full = np.arange(K3_ROWS_MAX_LEN)
    rows = [full] * 3 + _random_rows(4, 500, K3_ROWS_MAX_LEN, 9)
    layout = k3_layout(_level(rows, 3))
    graph = build_k3_graph(layout, 0)
    got = count_k3_rows(_rows_on(rows, layout, 0), graph, 0, layout.n_slots).get()
    np.testing.assert_array_equal(got, _expected(rows, layout))


@pytest.mark.gpu
def test_shards_sum_to_the_global_counts():
    """Any row partition sums to the global counts, as the cross-GPU reduce requires."""
    from et_miner.gpu.kernels import build_k3_graph, count_k3_rows, k3_layout

    devices = [0, 1] if _gpu_count() >= 2 else [0, 0]
    rows = _random_rows(7, 2_049, 60, 12)
    layout = k3_layout(_level(rows, 15))
    parts = [rows[:777], rows[777:]]
    total = sum(
        count_k3_rows(_rows_on(part, layout, d), build_k3_graph(layout, d), 0, layout.n_slots).get()
        for part, d in zip(parts, devices)
    )
    np.testing.assert_array_equal(total, _expected(rows, layout))


def test_statistic_adds_the_estimated_increments_to_the_row_pairs():
    from et_miner.gpu import row_split_chunks as rsc

    words = -(-640_000 // 64)
    s = rsc.k3_rows_statistic
    assert s(1_000, 4_000, 1_000, 50, 640_000) == pytest.approx(5_000 / (50 * words)), "every row pair frequent"
    assert s(1_000, 4_000, 500, 50, 640_000) == pytest.approx(2_000 / (50 * words)), "rho = 1/2: a quarter"
    assert s(1_000, 4_000, 3_000, 50, 640_000) == pytest.approx(5_000 / (50 * words)), "rho capped at 1"
    assert s(1_000, 4_000, 1_000, 0, 640_000) == float("inf"), "no slots"
    assert s(0, 0, 0, 50, 640_000) == float("inf"), "no row pairs"


def test_dispatch_follows_the_pin_then_the_statistic_and_one_chunk(monkeypatch):
    from et_miner.gpu import row_split_chunks as rsc

    below, above = rsc.K3_ROWS_MAX_S / 2, rsc.K3_ROWS_MAX_S * 2
    monkeypatch.delenv("ET_MINER_K3_KERNEL", raising=False)
    assert rsc.k3_counts_rows(below, 500, 1_000)
    assert not rsc.k3_counts_rows(below, 2_000, 1_000), "slots beyond one chunk"
    assert not rsc.k3_counts_rows(above, 500, 1_000), "s above the crossover"
    assert not rsc.k3_counts_rows(rsc.K3_ROWS_MAX_S, 500, 1_000), "s at the crossover"
    assert not rsc.k3_counts_rows(float("inf"), 0, 1_000), "no slots"
    monkeypatch.setenv("ET_MINER_K3_KERNEL", "rows")
    assert rsc.k3_counts_rows(above, 2_000, 1_000)
    monkeypatch.setenv("ET_MINER_K3_KERNEL", "dense")
    assert not rsc.k3_counts_rows(below, 500, 1_000)
    monkeypatch.setenv("ET_MINER_K3_KERNEL", "tiled")
    with pytest.raises(ValueError, match="ET_MINER_K3_KERNEL"):
        rsc.k3_counts_rows(below, 500, 1_000)


# ── end to end ───────────────────────────────────────────────────────────────


def _frame(seed: int, n_rows: int, *, repeats: bool, long_row: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [r.tolist() for r in _random_rows(seed, n_rows, 40, 12)]
    for i in range(0, n_rows, 7):
        rows[i] = []
    if repeats:
        for i in range(1, n_rows, 5):
            rows[i] = rows[i] + rows[i][: int(rng.integers(0, len(rows[i]) + 1))]
    if long_row:
        rows[3] = list(range(long_row))
    return pl.DataFrame({"items": pl.Series(rows, dtype=pl.List(pl.Int64))})


def _tie_support(df: pl.DataFrame) -> float:
    """A min_support whose min_count equals some frequent pair's count (a tie at the threshold)."""
    rows = [sorted(set(r)) for r in df["items"].to_list()]
    counts = sorted(Counter(p for r in rows for p in itertools.combinations(r, 2)).values())
    c, n = counts[int(len(counts) * 0.97)], len(rows)
    s = c / n
    while _min_count(s, n) > c:
        s = np.nextafter(s, 0.0)
    return float(s)


def _spy(monkeypatch) -> list:
    from et_miner.gpu import kernels

    calls = []
    real = kernels.count_k3_rows
    monkeypatch.setattr(kernels, "count_k3_rows", lambda *a, **k: calls.append(1) or real(*a, **k))
    return calls


@pytest.mark.gpu
@pytest.mark.parametrize("n_gpus", _n_gpus_params())
@pytest.mark.parametrize("layer", ["host", "device"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_row_wise_k3_matches_the_oracle(monkeypatch, seed, layer, n_gpus):
    _needs(n_gpus)
    from et_miner import apriori

    calls = _spy(monkeypatch)
    monkeypatch.setenv("ET_MINER_K3_KERNEL", "rows")
    monkeypatch.setenv("ET_MINER_INPUT_LAYER", layer)
    df = _frame(seed, 1_000 + 257 * seed, repeats=seed > 0)
    s = _tie_support(df)
    res = apriori(df, min_support=s, max_length=4, use_gpu=True, n_gpus=n_gpus)
    assert calls, "row-wise K=3 did not run"
    assert mined_counts(res, df.height) == ea_counts(df["items"].to_list(), s, max_length=4)


@pytest.mark.gpu
@pytest.mark.parametrize("layer", ["host", "device"])
def test_row_wise_k3_pin_holds_with_the_dense_k2_pin(monkeypatch, layer):
    from et_miner import apriori

    calls = _spy(monkeypatch)
    monkeypatch.setenv("ET_MINER_K3_KERNEL", "rows")
    monkeypatch.setenv("ET_MINER_K2_KERNEL", "dense")
    monkeypatch.setenv("ET_MINER_INPUT_LAYER", layer)
    df = _frame(1, 1_257, repeats=True)
    s = _tie_support(df)
    res = apriori(df, min_support=s, max_length=4, use_gpu=True)
    assert calls, "row-wise K=3 did not run"
    assert mined_counts(res, df.height) == ea_counts(df["items"].to_list(), s, max_length=4)


@pytest.mark.gpu
@pytest.mark.parametrize("layer", ["host", "device"])
def test_dense_k2_and_k3_levels_run_without_the_rows(monkeypatch, layer):
    """When the dispatch leaves K=2 and K=3 dense, their positions and rows go before the dense levels."""
    from et_miner import apriori
    from et_miner.gpu import input_layer, kernels, row_split, row_split_chunks

    events = []
    for cls in (input_layer.DeviceRows, row_split._HostRows):
        for name in ("drop_positions", "release"):
            real = getattr(cls, name)
            monkeypatch.setattr(cls, name, lambda self, _real=real, _name=name: events.append(_name) or _real(self))
    real_groups = kernels.build_k3plus_groups_from_flat
    monkeypatch.setattr(
        kernels,
        "build_k3plus_groups_from_flat",
        lambda flat, *a, **k: events.append(f"K={flat.shape[1] + 1}") or real_groups(flat, *a, **k),
    )
    monkeypatch.setattr(row_split_chunks, "K2_ROWS_MAX_R", 0.0)
    monkeypatch.setattr(row_split_chunks, "K3_ROWS_MAX_S", 0.0)
    monkeypatch.setenv("ET_MINER_INPUT_LAYER", layer)
    df = _frame(2, 1_514, repeats=True)
    s = _tie_support(df)
    res = apriori(df, min_support=s, max_length=4, use_gpu=True)
    assert "drop_positions" in events and "K=3" in events
    assert events.index("drop_positions") < events.index("release") < events.index("K=3")
    assert mined_counts(res, df.height) == ea_counts(df["items"].to_list(), s, max_length=4)


@pytest.mark.gpu
def test_a_row_beyond_64_active_items_counts_from_the_bitvecs(monkeypatch):
    from et_miner import apriori

    calls = _spy(monkeypatch)
    monkeypatch.setenv("ET_MINER_K3_KERNEL", "rows")
    df = _frame(3, 800, repeats=False, long_row=70)
    df = pl.concat([df] + [df.slice(3, 1)] * 200)
    s = 150 / df.height  # every pair of the long row (201 copies) is frequent, so all 70 items are active
    res = apriori(df, min_support=s, max_length=3, use_gpu=True)
    assert not calls
    assert mined_counts(res, df.height) == ea_counts(df["items"].to_list(), s, max_length=3)


@pytest.mark.gpu
@pytest.mark.parametrize("n_gpus", _n_gpus_params())
@pytest.mark.parametrize("option", ["free sets", "count inference", "forced chunks"])
def test_row_wise_k3_options_match_the_dense_level(monkeypatch, option, n_gpus):
    """Free sets, count inference and chunked slots give the dense K=3's result exactly."""
    _needs(n_gpus)
    from et_miner import apriori

    df = _frame(5, 1_500, repeats=True)
    s = _tie_support(df)
    kw = {"prune_equal_support": option == "free sets", "use_generator_pruning": option == "count inference"}
    if option == "forced chunks":
        monkeypatch.setenv("ET_MINER_MAX_CHUNK_CANDS", "97")
    results = {}
    for k3 in ("dense", "rows"):
        monkeypatch.setenv("ET_MINER_K3_KERNEL", k3)
        calls = _spy(monkeypatch)
        res = apriori(df, min_support=s, use_gpu=True, n_gpus=n_gpus, **kw)
        assert bool(calls) == (k3 == "rows")
        results[k3] = sorted(zip(map(tuple, res["itemset"].to_list()), res["support"].to_list()))
    assert results["rows"] == results["dense"]
