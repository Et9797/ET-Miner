"""The compacted multi-GPU reduce against the dense one.

``run_chunked_dense_level(compact=True)`` reduces only the entries the
counting kernels wrote (the rest hold ``UNTOUCHED``) and maps the survivors
back to chunk positions. On randomized skip masks, with written zeros (the
inferred entries of the devices that do not write inferred counts), several
chunks and several slices per chunk, it must return exactly what the dense
reduce returns when the skipped entries hold 0. The partials live on one
device (staged reduce) or on two (NCCL, and the staged fallback).
"""

import numpy as np
import pytest

from et_miner import _env
from et_miner.gpu.row_split_chunks import CHUNK_BYTES_PER_CANDIDATE, chunk_budget_from_bytes

GIB = 1 << 30
MIN_COUNT = 40


def _gpu_count() -> int:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


needs_two = pytest.mark.skipif(_gpu_count() < 2, reason="needs 2 CUDA devices")


def test_reduce_knob_parses_and_rejects(monkeypatch):
    monkeypatch.delenv("ET_MINER_REDUCE", raising=False)
    assert _env.reduce_mode() is None
    monkeypatch.setenv("ET_MINER_REDUCE", " Compact ")
    assert _env.reduce_mode() == "compact"
    monkeypatch.setenv("ET_MINER_REDUCE", "sparse")
    with pytest.raises(ValueError, match="ET_MINER_REDUCE"):
        _env.reduce_mode()


def test_budget_reserves_one_mask_bit_per_candidate():
    kw = dict(avail_bytes=20 * GIB, total_vram_bytes=24 * GIB, group_data_bytes=2 * GIB)
    dense = chunk_budget_from_bytes(**kw)
    compact = chunk_budget_from_bytes(**kw, compact_reduce=True)
    usable = 20 * GIB - 2 * GIB - GIB
    per_candidate = CHUNK_BYTES_PER_CANDIDATE + 2
    assert dense == usable // per_candidate
    assert compact == usable * 8 // (8 * per_candidate + 1)
    assert compact * (per_candidate + 1 / 8) <= usable < (compact + 1) * (per_candidate + 1 / 8)


def _partials(n: int, n_dev: int, skip: float, seed: int):
    """Per-device partial counts with the same written set; skipped entries are -1.

    Some written entries are 0 on every device but one, as the inferred
    entries are; the per-device values are small so sums straddle MIN_COUNT.
    """
    rng = np.random.default_rng(seed)
    written = rng.random(n) >= skip
    parts = [np.where(written, rng.integers(0, 2 * MIN_COUNT // n_dev + 2, n), -1).astype(np.int32)
             for _ in range(n_dev)]
    inferred = written & (rng.random(n) < 0.2)
    for p in parts[1:]:
        p[inferred] = 0
    parts[0][inferred] = rng.integers(0, 2 * MIN_COUNT, int(inferred.sum()))
    return parts


def _run(parts, device_ids, chunks, *, compact, comms=None):
    """run_chunked_dense_level over host partials; the bitvec slot carries the shard number."""
    import cupy as cp

    from et_miner.gpu.row_split_chunks import run_chunked_dense_level

    host = [p if compact else np.maximum(p, 0) for p in parts]

    def launch(shard, device_id, chunk):
        with cp.cuda.Device(device_id):
            return cp.asarray(host[shard][chunk.start : chunk.start + chunk.size])

    shards = [(i, did, 0) for i, did in enumerate(device_ids)]
    return run_chunked_dense_level(shards, chunks, launch, MIN_COUNT, comms, comms is not None, compact=compact)


def _chunks(n, size):
    from et_miner.gpu.row_split_chunks import plan_candidate_chunks

    return plan_candidate_chunks(n, size)


def _reference(parts):
    total = sum(np.maximum(p, 0).astype(np.int64) for p in parts)
    idx = np.nonzero(total >= MIN_COUNT)[0].astype(np.int64)
    return idx, total[idx]


def _assert_compact_equals_dense(parts, device_ids, chunks, comms=None):
    dense = _run(parts, device_ids, chunks, compact=False, comms=comms)
    compact = _run(parts, device_ids, chunks, compact=True, comms=comms)
    ref = _reference(parts)
    for got, label in ((dense, "dense"), (compact, "compact")):
        np.testing.assert_array_equal(got[0], ref[0], err_msg=f"{label} indices")
        np.testing.assert_array_equal(got[1], ref[1], err_msg=f"{label} counts")
        assert got[0].dtype == np.int64 and got[1].dtype == np.int64


@pytest.mark.gpu
@pytest.mark.parametrize("skip", [0.0, 0.5, 0.97, 1.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_compact_equals_dense_on_one_device(monkeypatch, skip, seed):
    """Two shards on device 0 (the staged reduce); 8-element-aligned slices of 1,024."""
    from et_miner.gpu.kernels import filter as filter_mod

    monkeypatch.setattr(filter_mod, "SLICE_ELEMS", 1024)
    n = 10_007
    parts = _partials(n, 2, skip, seed)
    for size in (n, 3_000, 1_024, 999):
        _assert_compact_equals_dense(parts, [0, 0], _chunks(n, size))


@pytest.mark.gpu
@pytest.mark.parametrize("n", [1, 7, 8, 9, 1_023, 1_025])
def test_compact_written_moves_the_written_entries_in_order(monkeypatch, n):
    import cupy as cp

    from et_miner.gpu.kernels import filter as filter_mod
    from et_miner.gpu.kernels.filter import compact_written

    monkeypatch.setattr(filter_mod, "SLICE_ELEMS", 512)
    rng = np.random.default_rng(n)
    host = np.where(rng.random(n) < 0.6, rng.integers(0, 100, n), -1).astype(np.int32)
    written = host != -1
    counts = cp.asarray(host)
    out = compact_written(counts, keep_mask=True)
    assert out.n == int(written.sum()) and out.size == n and out.slice_elems == 512
    np.testing.assert_array_equal(counts[: out.n].get(), host[written])
    np.testing.assert_array_equal(cp.unpackbits(out.mask).get()[:n].astype(bool), written)
    np.testing.assert_array_equal(out.slice_counts, [written[s : s + 512].sum() for s in range(0, n, 512)])
    assert compact_written(cp.asarray(host), keep_mask=False).mask is None


@pytest.mark.gpu
def test_compacted_filter_maps_survivors_back_to_chunk_positions(monkeypatch):
    import cupy as cp

    from et_miner.gpu.kernels import filter as filter_mod
    from et_miner.gpu.kernels.filter import compact_written, threshold_filter_compacted

    monkeypatch.setattr(filter_mod, "SLICE_ELEMS", 64)
    host = np.full(1_000, -1, dtype=np.int32)
    host[[0, 63, 64, 500, 999]] = [50, 10, 70, 40, 90]
    counts = cp.asarray(host)
    kept = compact_written(counts, keep_mask=True)
    idx, cnt = threshold_filter_compacted(counts, kept, 40)
    assert idx.tolist() == [0, 64, 500, 999] and cnt.tolist() == [50, 70, 40, 90]
    with pytest.raises(ValueError, match="keep_mask=True"):
        threshold_filter_compacted(counts, kept._replace(mask=None), 40)


@pytest.mark.gpu
def test_different_written_entries_raise_before_the_reduce(monkeypatch):
    from et_miner.gpu import nccl

    def never(*a, **k):
        raise AssertionError("the reduce ran on mismatched buffers")

    monkeypatch.setattr(nccl, "reduce_sum_to_gpu0", never)
    parts = _partials(5_000, 2, 0.5, 7)
    parts[1][np.nonzero(parts[1] == -1)[0][0]] = 3
    with pytest.raises(RuntimeError, match="wrote different entries"):
        _run(parts, [0, 0], _chunks(5_000, 5_000), compact=True)


@pytest.mark.gpu
@pytest.mark.multigpu
@needs_two
@pytest.mark.parametrize("use_nccl", [True, False])
@pytest.mark.parametrize("skip", [0.5, 0.97])
def test_compact_equals_dense_on_two_devices(monkeypatch, use_nccl, skip):
    from et_miner.gpu.kernels import filter as filter_mod
    from et_miner.gpu.nccl import _init_nccl

    monkeypatch.setattr(filter_mod, "SLICE_ELEMS", 4_096)
    comms = None
    if use_nccl:
        comms, ok = _init_nccl([0, 1])
        if not ok:
            pytest.skip("NCCL unavailable")
    n = 50_001
    parts = _partials(n, 2, skip, 11)
    for size in (n, 12_345):
        _assert_compact_equals_dense(parts, [0, 1], _chunks(n, size), comms=comms)


@pytest.mark.gpu
def test_the_reduce_sums_only_the_written_entries(monkeypatch):
    from et_miner.gpu import nccl

    sizes = []
    real = nccl.reduce_sum_to_gpu0

    def spy(arrays, *a, **k):
        sizes.append([int(x.size) for x in arrays])
        return real(arrays, *a, **k)

    monkeypatch.setattr(nccl, "reduce_sum_to_gpu0", spy)
    n = 9_000
    parts = _partials(n, 2, 0.9, 5)
    chunks = _chunks(n, 2_500)
    _run(parts, [0, 0], chunks, compact=True)
    written = parts[0] != -1
    want = [int(written[c.start : c.start + c.size].sum()) for c in chunks]
    assert sizes == [[w, w] for w in want if w]


@pytest.mark.gpu
def test_a_slice_that_does_not_fit_is_compacted_on_the_host(monkeypatch):
    import cupy as cp

    from et_miner.gpu.kernels import filter as filter_mod

    monkeypatch.setattr(filter_mod, "SLICE_ELEMS", 1024)
    real = filter_mod._compact_slice_on_device
    calls = {"device": 0, "failed": 0}

    def every_other(counts_gpu, start, *a):
        calls["device"] += 1
        if (start // 1024) % 2:
            calls["failed"] += 1
            raise cp.cuda.memory.OutOfMemoryError(1024 * 13, 0)
        return real(counts_gpu, start, *a)

    monkeypatch.setattr(filter_mod, "_compact_slice_on_device", every_other)
    n = 10_007
    _assert_compact_equals_dense(_partials(n, 2, 0.8, 3), [0, 0], _chunks(n, n))
    assert calls["failed"] >= 4


@pytest.mark.gpu
def test_compaction_under_a_real_pool_limit(monkeypatch):
    """Headroom for the mask but not for a slice's temporaries: both steps take the host path."""
    import cupy as cp

    from et_miner.gpu.kernels import filter as filter_mod
    from et_miner.gpu.kernels.filter import compact_written, threshold_filter_compacted

    host_calls = []

    def on_host(*a, _real=filter_mod._compact_slice_on_host):
        host_calls.append(a[1])
        return _real(*a)

    monkeypatch.setattr(filter_mod, "_compact_slice_on_host", on_host)

    n = 4_000_000
    host = _partials(n, 1, 0.7, 13)[0]
    written = host != -1
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    counts = cp.asarray(host)
    previous_limit = pool.get_limit()
    try:
        pool.set_limit(size=pool.used_bytes() + 2 * (1 << 20))
        kept = compact_written(counts, keep_mask=True)
        idx, cnt = threshold_filter_compacted(counts, kept, MIN_COUNT // 2)
    finally:
        pool.set_limit(size=previous_limit)
    assert host_calls == [0], "the slice did not take the host path"
    assert kept.n == int(written.sum())
    ref = np.nonzero(written & (host >= MIN_COUNT // 2))[0]
    np.testing.assert_array_equal(idx, ref)
    np.testing.assert_array_equal(cnt, host[ref])
