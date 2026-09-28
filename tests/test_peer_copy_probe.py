"""The peer-copy probe and the host-staged reduce it switches on.

On a box whose PCIe P2P drops device-to-device writes, a cross-device copy
returns success with the destination untouched: the staged reduce would
then sum only GPU 0's shard, and NCCL's P2P transport hangs. The probe
detects that once per device pair; the reduce goes through host memory and
NCCL is started with P2P disabled.
"""

import numpy as np
import pytest

cp = pytest.importorskip("cupy", reason="cupy not installed")

pytestmark = pytest.mark.gpu

from et_miner.gpu import nccl as nccl_mod
from et_miner.gpu.nccl import _staged_reduce_to_gpu0, peer_copy_works

needs_two = pytest.mark.skipif(cp.cuda.runtime.getDeviceCount() < 2, reason="needs 2 CUDA devices")


def test_same_device_is_never_probed(monkeypatch):
    monkeypatch.setattr(nccl_mod, "_peer_copy_ok", {})
    assert peer_copy_works(0, 0) is True
    assert nccl_mod._peer_copy_ok == {(0, 0): True}


@pytest.mark.multigpu
@needs_two
def test_probe_is_a_cached_bool(monkeypatch):
    monkeypatch.setattr(nccl_mod, "_peer_copy_ok", {})
    first = peer_copy_works(0, 1)
    assert first in (True, False)
    assert nccl_mod._peer_copy_ok[(0, 1)] is first
    monkeypatch.setitem(nccl_mod._peer_copy_ok, (0, 1), not first)
    assert peer_copy_works(0, 1) is (not first)


@pytest.mark.multigpu
@needs_two
@pytest.mark.parametrize("n", [1, 1000, 6903, 3_000_007])
def test_host_staged_reduce_sums_both_shards(monkeypatch, n):
    """With the direct copy declared broken, the reduce must still equal
    the NumPy sum of both shards, whatever the box's P2P does."""
    monkeypatch.setattr(nccl_mod, "peer_copy_works", lambda dst, src: False)
    rng = np.random.default_rng(n)
    a = rng.integers(0, 1000, n, dtype=np.int32)
    b = rng.integers(0, 1000, n, dtype=np.int32)
    with cp.cuda.Device(0):
        ga = cp.asarray(a)
    with cp.cuda.Device(1):
        gb = cp.asarray(b)
    _staged_reduce_to_gpu0([ga, gb], [0, 1])
    with cp.cuda.Device(0):
        np.testing.assert_array_equal(ga.get(), a + b)


@pytest.mark.multigpu
@needs_two
def test_reduce_on_this_box_matches_numpy():
    """Whichever path the real probe picks here, the sum must be right."""
    rng = np.random.default_rng(1)
    a = rng.integers(0, 1000, 4096, dtype=np.int32)
    b = rng.integers(0, 1000, 4096, dtype=np.int32)
    with cp.cuda.Device(0):
        ga = cp.asarray(a)
    with cp.cuda.Device(1):
        gb = cp.asarray(b)
    _staged_reduce_to_gpu0([ga, gb], [0, 1])
    with cp.cuda.Device(0):
        np.testing.assert_array_equal(ga.get(), a + b)


@pytest.mark.multigpu
@needs_two
def test_failed_probe_disables_nccl_p2p_before_init(monkeypatch):
    monkeypatch.setattr(nccl_mod, "peer_copy_works", lambda dst, src: False)
    monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
    monkeypatch.delenv("ET_MINER_DISABLE_NCCL", raising=False)
    comms, ok = nccl_mod._init_nccl([0, 1])
    assert ok and comms is not None and len(comms) == 2
    assert nccl_mod.os.environ.get("NCCL_P2P_DISABLE") == "1"


@pytest.mark.multigpu
@needs_two
def test_a_callers_setting_is_kept(monkeypatch):
    monkeypatch.setattr(nccl_mod, "peer_copy_works", lambda dst, src: False)
    monkeypatch.setenv("NCCL_P2P_DISABLE", "0")
    monkeypatch.delenv("ET_MINER_DISABLE_NCCL", raising=False)
    nccl_mod._init_nccl([0, 1])
    assert nccl_mod.os.environ.get("NCCL_P2P_DISABLE") == "0"


@pytest.mark.multigpu
@needs_two
@pytest.mark.parametrize("home", [0, 1])
def test_prebuilt_shards_are_the_host_slices_when_copies_do_not_land(monkeypatch, home):
    from et_miner.gpu.row_split import shard_prebuilt_bitvecs

    monkeypatch.setattr(nccl_mod, "peer_copy_works", lambda dst, src: False)
    import et_miner.gpu.row_split as rs

    monkeypatch.setattr(rs, "peer_copy_works", lambda dst, src: False, raising=False)
    rng = np.random.default_rng(home)
    packed = rng.integers(0, 2**63 - 1, size=(5, 7), dtype=np.int64).astype(np.uint64)
    with cp.cuda.Device(home):
        bv = cp.asarray(packed)
    shards = shard_prebuilt_bitvecs(bv, 7 * 64, 2)
    assert [d for _, d, _ in shards] == [0, 1]
    lo = 0
    for arr, dev, rows in shards:
        with cp.cuda.Device(dev):
            got = arr.get()
        np.testing.assert_array_equal(got, packed[:, lo : lo + got.shape[1]])
        assert rows == got.shape[1] * 64
        lo += got.shape[1]
    assert lo == 7
