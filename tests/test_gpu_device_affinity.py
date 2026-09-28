"""Kernel wrappers follow their inputs' device, not the ambient one.

Every bitvec wrapper pins its launch to the device its bitvecs live on
(`loader._assert_bitvecs` checks the inputs first). Called from a thread parked
on device 0 with its inputs on device 1, a wrapper that launched on the ambient
device would hand the kernel two cards' pointers -- CUDA_ERROR_ILLEGAL_ADDRESS,
which poisons the context process-wide. The fixture builds on device 1 on
purpose; nothing else in the suite does.

Also here: `bitvecs=` on several GPUs, which the row-split miner shards across
devices, must mine what one device mines.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _device_count() -> int:
    try:
        return cp.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


N_COLS = 24
N_ROWS = 4096
MIN_COUNT = 400


def _packed() -> np.ndarray:
    """A deterministic transaction matrix packed into (N_COLS, n_u64s) bitvecs."""
    rng = np.random.default_rng(3)
    n_u64s = (N_ROWS + 63) // 64
    dense = (rng.random((N_COLS, N_ROWS)) < 0.35).astype(np.uint8)
    # A couple of guaranteed-frequent columns so k>=3 has candidates.
    dense[0, :] = 1
    dense[1, : int(N_ROWS * 0.9)] = 1
    dense[2, : int(N_ROWS * 0.8)] = 1
    packed = np.zeros((N_COLS, n_u64s), dtype=np.uint64)
    for c in range(N_COLS):
        bits = np.packbits(dense[c], bitorder="little")
        buf = np.zeros(n_u64s * 8, dtype=np.uint8)
        buf[: len(bits)] = bits
        packed[c] = buf.view(np.uint64)
    return packed


def _groups():
    from et_miner.gpu.kernels import build_k3plus_groups_from_flat

    prev = np.array([(i, j) for i in range(8) for j in range(i + 1, 9)], dtype=np.int32)
    return build_k3plus_groups_from_flat(prev)


def _entry_points():
    """(name, call(bitvecs) -> host ndarray of counts) for every bitvec wrapper."""
    from et_miner.gpu.kernels import (
        count_itemsets_cuda,
        count_k3plus_allcounts,
        count_pairs_k2_allcounts,
        count_pairs_k2_shared,
        count_shared_tiled_allcounts,
        count_tiled_fused,
    )

    n_u64s = (N_ROWS + 63) // 64
    cols = list(range(N_COLS))
    groups = _groups()
    itemsets = [np.array([0, 1, c], dtype=np.int32) for c in range(2, N_COLS)]
    return [
        ("count_pairs_k2_allcounts", lambda bv: count_pairs_k2_allcounts(bv, cols, n_u64s, variant="legacy").get()),
        ("count_pairs_k2_shared", lambda bv: count_pairs_k2_shared(bv, cols, n_u64s).get()),
        ("count_k3plus_allcounts", lambda bv: count_k3plus_allcounts(bv, groups, n_u64s, variant="legacy").get()),
        ("count_shared_tiled_allcounts", lambda bv: count_shared_tiled_allcounts(bv, groups, n_u64s).get()),
        ("count_tiled_fused", lambda bv: np.concatenate(count_tiled_fused(bv, groups, n_u64s, MIN_COUNT))),
        ("count_itemsets_cuda", lambda bv: count_itemsets_cuda(bv, itemsets)),
    ]


needs_two = pytest.mark.skipif(_device_count() < 2, reason="needs 2 CUDA devices")


@pytest.mark.gpu
@pytest.mark.multigpu
@needs_two
@pytest.mark.parametrize("name", [n for n, _ in _entry_points()])
def test_wrappers_follow_their_inputs_not_the_ambient_device(name):
    """CONTROL: drop a wrapper's `with cp.cuda.Device(...)` and this aborts the
    process rather than failing -- so it asserts on the counts, which is what
    the wrapper being on the right device buys."""
    call = dict(_entry_points())[name]
    packed = _packed()
    with cp.cuda.Device(0):
        want = call(cp.asarray(packed))
    with cp.cuda.Device(1):
        bv1 = cp.asarray(packed)
    with cp.cuda.Device(0):
        got = call(bv1)
    np.testing.assert_array_equal(got, want)


@pytest.mark.gpu
@pytest.mark.multigpu
@needs_two
@pytest.mark.parametrize("wrapper", ["count_k3plus_allcounts", "count_shared_tiled_allcounts"])
def test_groups_on_another_device_raise_rather_than_being_repaired(wrapper):
    from et_miner.gpu import kernels

    groups = _groups()
    with cp.cuda.Device(1):
        bv1 = cp.asarray(_packed())
    groups_dev0 = kernels.upload_k3plus_groups(groups, 0)
    with pytest.raises(ValueError, match="must be resident on one device"):
        getattr(kernels, wrapper)(bv1, groups, bv1.shape[1], groups_gpu=groups_dev0)


@pytest.mark.gpu
def test_assert_home_rejects_a_host_array_with_a_readable_error():
    """A host array reaching a device-only path is a ValueError, not AttributeError.

    NumPy 2 gives `ndarray.device == "cpu"` and NumPy 1 has no `.device` at
    all; both used to surface as `AttributeError: 'str' object has no attribute
    'id'` or similar from inside the validator, which reads like a bug in the
    check rather than a bug in the call.
    """
    from et_miner.gpu.kernels.loader import _assert_home

    with cp.cuda.Device(0):
        on_gpu = cp.zeros((4, 2), dtype=cp.uint64)

    with pytest.raises(ValueError, match="not a CuPy array resident on a CUDA device"):
        _assert_home("ctx", bitvecs_gpu=on_gpu, groups_gpu=np.zeros((4, 2), dtype=np.int32))


@pytest.mark.gpu
@pytest.mark.multigpu
@needs_two
@pytest.mark.parametrize("home", [0, 1])
def test_prebuilt_bitvecs_on_two_gpus_mine_what_one_gpu_mines(home):
    """bitvecs= with n_gpus=2 is sharded by 64-row words across both devices,
    wherever the caller's array lives."""
    from et_miner.core.apriori import apriori

    packed = _packed()
    col_to_item = {c: 100 + c for c in range(N_COLS)}
    with cp.cuda.Device(home):
        bv = cp.asarray(packed)
    one = apriori(bitvecs=(bv, col_to_item, N_ROWS), min_support=MIN_COUNT / N_ROWS, use_gpu=True, n_gpus=1)
    two = apriori(bitvecs=(bv, col_to_item, N_ROWS), min_support=MIN_COUNT / N_ROWS, use_gpu=True, n_gpus=2)

    def as_set(df):
        return {(tuple(r), round(s * N_ROWS)) for r, s in zip(df["itemset"].to_list(), df["support"].to_list())}

    assert as_set(two) == as_set(one) and len(as_set(one)) > N_COLS
    np.testing.assert_array_equal(bv.get(), packed)  # the caller's array is read-only to the engine


@pytest.mark.gpu
@pytest.mark.parametrize("max_chunk", [None, "7"])
def test_two_shards_on_one_device_mine_what_one_shard_mines(monkeypatch, max_chunk):
    """The multi-shard path -- per-shard counts, the staged reduce, the shard
    cut at word boundaries -- on a one-GPU box: both shards on device 0. A
    forced chunk cap also sends the pair space and the prefix groups through
    the per-candidate sub-chunks that several GPUs use."""
    from et_miner.gpu.row_split import _apriori_row_split_multi_gpu, shard_prebuilt_bitvecs

    monkeypatch.setenv("ET_MINER_DISABLE_NCCL", "1")
    if max_chunk:
        monkeypatch.setenv("ET_MINER_MAX_CHUNK_CANDS", max_chunk)
    packed = _packed()
    col_to_item = {c: 100 + c for c in range(N_COLS)}
    with cp.cuda.Device(0):
        bv = cp.asarray(packed)

    def mine(shards):
        df = _apriori_row_split_multi_gpu(None, col_to_item, N_ROWS, MIN_COUNT / N_ROWS, None, len(shards),
                                          bitvecs_list=shards)
        return {(tuple(r), round(s * N_ROWS)) for r, s in zip(df["itemset"].to_list(), df["support"].to_list())}

    one = mine(shard_prebuilt_bitvecs(bv, N_ROWS, 1))
    two = mine(shard_prebuilt_bitvecs(bv, N_ROWS, 2, devices=[0, 0]))
    assert two == one and len(one) > N_COLS
