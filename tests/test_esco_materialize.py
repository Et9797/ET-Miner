"""ESCO materialization sized from the count pass ("reuse") against the recount.

``materialize_survivors(peer_counts=...)`` must build exactly the shards the
recount builds — on one shard and on two, with survivors in any order, and
when some peer counts are zero the way an inferred count leaves them — and
must raise when the counts do not match the shards. The chunk loop must hand
back the peers' partial counts of the survivors.
"""

import numpy as np
import pytest

from et_miner import _env

MIN_COUNT = 6
N_TIDS = 3_000


def _gpu_count() -> int:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


def test_materialize_knob_parses_and_rejects(monkeypatch):
    monkeypatch.delenv("ET_MINER_ESCO_MATERIALIZE", raising=False)
    assert _env.esco_materialize() is None
    monkeypatch.setenv("ET_MINER_ESCO_MATERIALIZE", "Reuse")
    assert _env.esco_materialize() == "reuse"
    monkeypatch.setenv("ET_MINER_ESCO_MATERIALIZE", "lazy")
    with pytest.raises(ValueError, match="ET_MINER_ESCO_MATERIALIZE"):
        _env.esco_materialize()


def test_keeps_peer_arrays_follows_the_reduce_path():
    from et_miner.gpu.nccl import keeps_peer_arrays

    class WithReduce:
        def reduce(self):
            pass

    assert keeps_peer_arrays(None)
    assert keeps_peer_arrays([WithReduce(), WithReduce()])
    assert not keeps_peer_arrays([object(), object()])


def _level(seed: int):
    """A previous level (pairs), their tidsets over N_TIDS rows, and its K=3 groups."""
    from et_miner.gpu.kernels import build_k3plus_groups_from_flat

    rng = np.random.default_rng(seed)
    prev = set()
    for a in range(10):
        for b in rng.choice(np.arange(a + 1, 30), size=int(rng.integers(2, 7)), replace=False):
            prev.add((a, int(b)))
    flat = np.array(sorted(prev), np.int32)
    rows = [np.sort(rng.choice(N_TIDS, size=int(rng.integers(0, 900)), replace=False)).astype(np.int32)
            for _ in range(len(flat))]
    return rows, build_k3plus_groups_from_flat(flat, with_src_rows=True)


def _shards(rows, device_ids):
    """Rows split by tid range into one shard per entry of ``device_ids`` (shard-local tids)."""
    import cupy as cp

    from et_miner.gpu.sparse_csr import CsrShard

    cuts = np.linspace(0, N_TIDS, len(device_ids) + 1).astype(int)
    shards = []
    for lo, hi, did in zip(cuts[:-1], cuts[1:], device_ids):
        local = [(r[(r >= lo) & (r < hi)] - lo).astype(np.int32) for r in rows]
        offsets = np.zeros(len(local) + 1, np.int64)
        np.cumsum([len(r) for r in local], out=offsets[1:])
        with cp.cuda.Device(did):
            shards.append(CsrShard(did, int(hi - lo), cp.asarray(offsets), cp.asarray(np.concatenate(local))))
    return shards


def _case(seed: int, device_ids):
    """(shards, groups_gpu, survivors, their global counts, per-shard counts (n_shards, n))."""
    import cupy as cp

    from et_miner.gpu.kernels import count_csr_range, upload_k3plus_groups

    rows, groups = _level(seed)
    shards = _shards(rows, device_ids)
    groups_gpu = {did: upload_k3plus_groups(groups, did, with_src_rows=True) for did in set(device_ids)}
    tc = groups.total_candidates
    per_shard = []
    for s in shards:
        with cp.cuda.Device(s.device_id):
            per_shard.append(count_csr_range(s.offsets, s.indices, groups_gpu[s.device_id], 0, tc).get())
    per_shard = np.array(per_shard, dtype=np.int64)
    total = per_shard.sum(axis=0)
    surv = np.nonzero(total >= MIN_COUNT)[0]
    assert len(surv) > 20 and (per_shard[:, surv] > 0).all(axis=0).any()
    order = np.random.default_rng(seed).permutation(len(surv))  # the level-end lexsort reorders survivors
    surv = surv[order]
    return shards, groups_gpu, surv, total[surv], per_shard[:, surv]


def _host(shards):
    return [(s.offsets.get(), s.indices.get()) for s in shards]


def _assert_same_shards(a, b):
    assert len(a) == len(b)
    for (off_a, idx_a), (off_b, idx_b) in zip(_host(a), _host(b)):
        np.testing.assert_array_equal(off_a, off_b)
        np.testing.assert_array_equal(idx_a, idx_b)


def _devices(n_shards, two_devices):
    return list(range(n_shards)) if two_devices else [0] * n_shards


@pytest.mark.gpu
@pytest.mark.parametrize("n_shards", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 1])
def test_reuse_builds_the_recount_shards(n_shards, seed):
    from et_miner.gpu.sparse_csr import materialize_survivors

    shards, ggpu, surv, expected, per_shard = _case(seed, _devices(n_shards, False))
    recount = materialize_survivors(shards, ggpu, surv, expected)
    reuse = materialize_survivors(shards, ggpu, surv, expected, peer_counts=per_shard[1:])
    _assert_same_shards(reuse, recount)


@pytest.mark.gpu
def test_reuse_recounts_survivors_without_peer_counts(monkeypatch):
    """Zero peer counts (an inferred survivor's) are recounted on every shard, and only those."""
    from et_miner.gpu import sparse_csr
    from et_miner.gpu.sparse_csr import materialize_survivors

    shards, ggpu, surv, expected, per_shard = _case(2, [0, 0])
    peers = per_shard[1:].copy()
    inferred = np.random.default_rng(5).random(len(surv)) < 0.3
    peers[:, inferred] = 0
    recount = materialize_survivors(shards, ggpu, surv, expected)
    gathered = []

    def spy(*a, _real=sparse_csr.count_csr_gather):
        gathered.append(int(a[3].size))
        return _real(*a)

    monkeypatch.setattr(sparse_csr, "count_csr_gather", spy)
    reuse = materialize_survivors(shards, ggpu, surv, expected, peer_counts=peers)
    _assert_same_shards(reuse, recount)
    assert gathered == [int((peers == 0).all(axis=0).sum())] * 2


@pytest.mark.gpu
@pytest.mark.parametrize("n_shards", [1, 2])
def test_reuse_skips_the_gather_count(monkeypatch, n_shards):
    from et_miner.gpu import sparse_csr
    from et_miner.gpu.sparse_csr import materialize_survivors

    shards, ggpu, surv, expected, per_shard = _case(3, [0] * n_shards)
    keep = (per_shard[1:] > 0).all(axis=0) if n_shards > 1 else np.ones(len(surv), bool)

    def never(*a, **k):
        raise AssertionError("reuse counted survivors that have peer counts")

    monkeypatch.setattr(sparse_csr, "count_csr_gather", never)
    materialize_survivors(shards, ggpu, surv[keep], expected[keep], peer_counts=per_shard[1:, keep])


@pytest.mark.gpu
@pytest.mark.parametrize("n_shards", [1, 2])
def test_counts_that_do_not_match_the_shards_raise(n_shards):
    from et_miner.gpu.sparse_csr import materialize_survivors

    shards, ggpu, surv, expected, per_shard = _case(4, [0] * n_shards)
    if n_shards == 1:
        bad_expected = expected.copy()
        bad_expected[3] += 1
        with pytest.raises(RuntimeError, match="do not fill the slots"):
            materialize_survivors(shards, ggpu, surv, bad_expected, peer_counts=per_shard[1:])
        return
    peers = per_shard[1:].copy()
    i = int(np.nonzero(peers[0] > 1)[0][0])
    peers[0, i] -= 1
    with pytest.raises(RuntimeError, match="do not fill the slots"):
        materialize_survivors(shards, ggpu, surv, expected, peer_counts=peers)
    peers[0, i] = expected[i] + 1
    with pytest.raises(RuntimeError, match="exceed the survivors' counts"):
        materialize_survivors(shards, ggpu, surv, expected, peer_counts=peers)


@pytest.mark.gpu
@pytest.mark.multigpu
@pytest.mark.skipif(_gpu_count() < 2, reason="needs 2 CUDA devices")
def test_reuse_builds_the_recount_shards_on_two_devices():
    from et_miner.gpu.sparse_csr import materialize_survivors

    shards, ggpu, surv, expected, per_shard = _case(6, _devices(2, True))
    recount = materialize_survivors(shards, ggpu, surv, expected)
    reuse = materialize_survivors(shards, ggpu, surv, expected, peer_counts=per_shard[1:])
    _assert_same_shards(reuse, recount)


def _chunk_loop(parts, chunks, **kw):
    import cupy as cp

    from et_miner.gpu.row_split_chunks import run_chunked_dense_level

    def launch(shard, device_id, chunk):
        with cp.cuda.Device(device_id):
            return cp.asarray(parts[shard][chunk.start : chunk.start + chunk.size])

    return run_chunked_dense_level([(i, 0, 0) for i in range(len(parts))], chunks, launch, MIN_COUNT, None, False,
                                   **kw)


@pytest.mark.gpu
@pytest.mark.parametrize("n_shards", [1, 2, 3])
def test_the_chunk_loop_returns_the_peers_partial_counts(n_shards):
    from et_miner.gpu.row_split_chunks import plan_candidate_chunks

    rng = np.random.default_rng(n_shards)
    parts = [rng.integers(0, 2 * MIN_COUNT // n_shards + 2, 5_000).astype(np.int32) for _ in range(n_shards)]
    idx, cnt, peers = _chunk_loop(parts, plan_candidate_chunks(5_000, 1_234), peer_counts=True)
    total = np.sum(parts, axis=0, dtype=np.int64)
    np.testing.assert_array_equal(idx, np.nonzero(total >= MIN_COUNT)[0])
    np.testing.assert_array_equal(cnt, total[idx])
    assert peers.shape == (n_shards - 1, len(idx)) and peers.dtype == np.int64
    want = np.array([p[idx] for p in parts[1:]], np.int64) if n_shards > 1 else np.empty((0, len(idx)), np.int64)
    np.testing.assert_array_equal(peers, want)


@pytest.mark.gpu
def test_no_peer_counts_when_the_reduce_overwrites_them(monkeypatch):
    from et_miner.gpu import nccl
    from et_miner.gpu.row_split_chunks import plan_candidate_chunks

    monkeypatch.setattr(nccl, "keeps_peer_arrays", lambda comms: False)
    parts = [np.full(100, MIN_COUNT, np.int32)] * 2
    *_, peers = _chunk_loop(parts, plan_candidate_chunks(100, 100), peer_counts=True)
    assert peers is None
    with pytest.raises(ValueError, match="dense reduce"):
        _chunk_loop(parts, plan_candidate_chunks(100, 100), peer_counts=True, compact=True)
