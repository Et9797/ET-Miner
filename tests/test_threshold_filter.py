"""GPU tests for the survivor filter (gpu/kernels/filter.py).

``threshold_filter`` must agree exactly with a NumPy reference — same
survivors, same counts, ascending indices — across sizes and pass rates,
across the 64M slice boundary, for int64 input, and when a slice has to
fall back to the host because it does not fit the device.
"""

import numpy as np
import pytest

cp = pytest.importorskip("cupy", reason="cupy not installed")

pytestmark = pytest.mark.gpu

from et_miner.gpu.kernels import filter as filter_mod
from et_miner.gpu.kernels.filter import threshold_filter


def _reference(counts: np.ndarray, threshold: int):
    idx = np.nonzero(counts >= threshold)[0].astype(np.int64)
    return idx, counts[idx].astype(np.int64)


def _make_counts(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 1000, size=n, dtype=np.int32)


@pytest.mark.parametrize("n", [0, 1, 17, 4_096, 5_000_000])
def test_matches_numpy_reference(n):
    counts = _make_counts(n)
    threshold = 500
    if n:
        counts[n // 2] = threshold  # exact-boundary value must survive (>=)
    idx, cnt = threshold_filter(cp.asarray(counts), threshold)
    ref_idx, ref_cnt = _reference(counts, threshold)
    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_array_equal(cnt, ref_cnt)
    assert idx.dtype == np.int64 and cnt.dtype == np.int64


@pytest.mark.parametrize("threshold,expect_all", [(0, True), (1001, False)])
def test_extreme_pass_rates(threshold, expect_all):
    counts = _make_counts(1_000_000, seed=3)
    idx, cnt = threshold_filter(cp.asarray(counts), threshold)
    if expect_all:
        assert len(idx) == len(counts)
        np.testing.assert_array_equal(idx, np.arange(len(counts), dtype=np.int64))
        np.testing.assert_array_equal(cnt, counts.astype(np.int64))
    else:
        assert len(idx) == 0 and len(cnt) == 0


def test_indices_strictly_ascending():
    counts = _make_counts(2_000_000, seed=5)
    idx, _ = threshold_filter(cp.asarray(counts), 900)
    assert len(idx) > 0
    assert np.all(np.diff(idx) > 0)


def test_crosses_slice_boundaries():
    """Survivors straddling the 64M slice size."""
    n = filter_mod.SLICE_ELEMS + 1_000
    counts = np.zeros(n, dtype=np.int32)
    hot = np.array([0, filter_mod.SLICE_ELEMS - 1, filter_mod.SLICE_ELEMS, n - 1], dtype=np.int64)
    counts[hot] = 7
    counts_gpu = cp.asarray(counts)
    idx, cnt = threshold_filter(counts_gpu, 7)
    np.testing.assert_array_equal(idx, hot)
    np.testing.assert_array_equal(cnt, np.full(4, 7, dtype=np.int64))
    del counts_gpu
    cp.get_default_memory_pool().free_all_blocks()


def test_int64_input_still_correct():
    counts = _make_counts(100_000, seed=7).astype(np.int64)
    idx, cnt = threshold_filter(cp.asarray(counts), 800)
    ref_idx, ref_cnt = _reference(counts, 800)
    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_array_equal(cnt, ref_cnt)


def test_slice_that_does_not_fit_the_device_is_filtered_on_the_host(monkeypatch):
    """Force the device path out of memory: the slice must fall back to the
    host and return exactly the reference."""
    counts = _make_counts(300_000, seed=11)
    counts_gpu = cp.asarray(counts)
    real = filter_mod._filter_slice_on_device
    calls = {"device": 0, "host": 0}

    def exploding(view, threshold):
        calls["device"] += 1
        raise cp.cuda.memory.OutOfMemoryError(13 * len(view), 0)

    real_host = filter_mod._filter_slice_on_host

    def counting_host(view, threshold):
        calls["host"] += 1
        return real_host(view, threshold)

    monkeypatch.setattr(filter_mod, "_filter_slice_on_device", exploding)
    monkeypatch.setattr(filter_mod, "_filter_slice_on_host", counting_host)
    idx, cnt = threshold_filter(counts_gpu, 400)
    ref_idx, ref_cnt = _reference(counts, 400)
    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_array_equal(cnt, ref_cnt)
    assert calls == {"device": 1, "host": 1}
    monkeypatch.setattr(filter_mod, "_filter_slice_on_device", real)
    on_device = threshold_filter(counts_gpu, 400)
    np.testing.assert_array_equal(on_device[0], idx)
    np.testing.assert_array_equal(on_device[1], cnt)


def test_host_fallback_under_a_real_pool_limit():
    """A pool limit just above the counts leaves no room for the device
    path's temporaries; the filter must still return the reference."""
    counts = _make_counts(4_000_000, seed=17)  # 16 MB of counts
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    counts_gpu = cp.asarray(counts)
    previous_limit = pool.get_limit()
    try:
        pool.set_limit(size=pool.used_bytes() + 2 * (1 << 20))  # 2 MiB of headroom
        idx, cnt = threshold_filter(counts_gpu, 300)
    finally:
        pool.set_limit(size=previous_limit)
    ref_idx, ref_cnt = _reference(counts, 300)
    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_array_equal(cnt, ref_cnt)
