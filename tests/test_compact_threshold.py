"""GPU tests for the compact_threshold survivor filter (gpu/kernels/filter.py).

Every implementation (compact kernel / sliced cupy / legacy cpu) must agree
exactly with a NumPy reference — same survivors, same counts, ascending
indices — across sizes and pass rates including the 100%-survivor
exact-alloc path and the forced host-workspace valve.
"""

import numpy as np
import pytest

cp = pytest.importorskip("cupy", reason="cupy not installed")

pytestmark = pytest.mark.gpu

from et_miner.gpu.kernels import filter as filter_mod
from et_miner.gpu.kernels.filter import compact_threshold_filter

IMPLS = ("compact", "cupy", "cpu")


def _reference(counts: np.ndarray, threshold: int):
    idx = np.nonzero(counts >= threshold)[0].astype(np.int64)
    return idx, counts[idx].astype(np.int64)


def _make_counts(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 1000, size=n, dtype=np.int32)


@pytest.mark.parametrize("n", [0, 1, 17, 4_096, 5_000_000])
@pytest.mark.parametrize("impl", IMPLS)
def test_matches_numpy_reference(n, impl):
    counts = _make_counts(n)
    threshold = 500
    if n:
        counts[n // 2] = threshold  # exact-boundary value must survive (>=)
    idx, cnt = compact_threshold_filter(cp.asarray(counts), threshold, impl=impl)
    ref_idx, ref_cnt = _reference(counts, threshold)
    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_array_equal(cnt, ref_cnt)
    assert idx.dtype == np.int64 and cnt.dtype == np.int64


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("threshold,expect_all", [(0, True), (1001, False)])
def test_extreme_pass_rates(impl, threshold, expect_all):
    counts = _make_counts(1_000_000, seed=3)
    idx, cnt = compact_threshold_filter(cp.asarray(counts), threshold, impl=impl)
    if expect_all:
        # 100%-survivor path: compact must take the exact-alloc second pass.
        assert len(idx) == len(counts)
        np.testing.assert_array_equal(idx, np.arange(len(counts), dtype=np.int64))
        np.testing.assert_array_equal(cnt, counts.astype(np.int64))
    else:
        assert len(idx) == 0 and len(cnt) == 0


def test_indices_strictly_ascending():
    counts = _make_counts(2_000_000, seed=5)
    idx, _ = compact_threshold_filter(cp.asarray(counts), 900, impl="compact")
    assert len(idx) > 0
    assert np.all(np.diff(idx) > 0)


def test_crosses_slice_boundaries():
    """Survivors straddling the 64M slice size of the sliced impls."""
    n = filter_mod.SLICE_ELEMS + 1_000
    counts = np.zeros(n, dtype=np.int32)
    hot = np.array([0, filter_mod.SLICE_ELEMS - 1, filter_mod.SLICE_ELEMS, n - 1], dtype=np.int64)
    counts[hot] = 7
    counts_gpu = cp.asarray(counts)
    for impl in IMPLS:
        idx, cnt = compact_threshold_filter(counts_gpu, 7, impl=impl)
        np.testing.assert_array_equal(idx, hot)
        np.testing.assert_array_equal(cnt, np.full(4, 7, dtype=np.int64))
    del counts_gpu
    cp.get_default_memory_pool().free_all_blocks()


def test_int64_input_still_correct():
    """The kernel is int32-only; int64 inputs must route and stay correct."""
    counts = _make_counts(100_000, seed=7).astype(np.int64)
    idx, cnt = compact_threshold_filter(cp.asarray(counts), 800, impl="compact")
    ref_idx, ref_cnt = _reference(counts.astype(np.int32), 800)
    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_array_equal(cnt, ref_cnt)


def test_forced_valve_equivalence(monkeypatch):
    """Deterministically starve the host-workspace check: the compact impl
    must fall back to the sliced-D2H valve and still return exactly what
    every other impl returns."""
    monkeypatch.setattr(filter_mod, "_host_ram_available", lambda: 1024)  # 1 KiB
    counts = _make_counts(300_000, seed=11)
    counts_gpu = cp.asarray(counts)
    got = compact_threshold_filter(counts_gpu, 400, impl="compact")
    ref_idx, ref_cnt = _reference(counts, 400)
    np.testing.assert_array_equal(got[0], ref_idx)
    np.testing.assert_array_equal(got[1], ref_cnt)
    for impl in ("cupy", "cpu"):
        other = compact_threshold_filter(counts_gpu, 400, impl=impl)
        np.testing.assert_array_equal(got[0], other[0])
        np.testing.assert_array_equal(got[1], other[1])


def test_env_toggle_routes(monkeypatch):
    counts = _make_counts(10_000, seed=13)
    counts_gpu = cp.asarray(counts)
    baseline = compact_threshold_filter(counts_gpu, 500, impl="compact")
    for env_impl in IMPLS:
        monkeypatch.setenv("ET_MINER_FILTER_IMPL", env_impl)
        got = compact_threshold_filter(counts_gpu, 500)
        np.testing.assert_array_equal(got[0], baseline[0])
        np.testing.assert_array_equal(got[1], baseline[1])


def test_unknown_impl_rejected():
    with pytest.raises(ValueError, match="ET_MINER_FILTER_IMPL"):
        compact_threshold_filter(cp.zeros(4, dtype=cp.int32), 1, impl="bogus")
