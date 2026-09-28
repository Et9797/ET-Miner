"""Cross-device copies never issue a device-to-device write unless opted in.

On a box whose PCIe P2P drops writes, a direct copy returns success with the
destination untouched, and small copies land while large ones do not, so no
probe can certify a device pair. ``gpu.nccl.copy_between_devices`` therefore
stages through host memory by default; the staged reduce and the
``bitvecs=`` shard copy go through it, and NCCL is created under
``NCCL_P2P_LEVEL=NVL`` unless the caller chose. The two-GPU tests here use
real transfers of at least 8M int32 elements and no monkeypatching.
"""

import os

import numpy as np
import pytest

cp = pytest.importorskip("cupy", reason="cupy not installed")

pytestmark = pytest.mark.gpu

from et_miner import _env
from et_miner.gpu import nccl as nccl_mod
from et_miner.gpu.nccl import _staged_reduce_to_gpu0, copy_between_devices, reduce_sum_to_gpu0

needs_two = pytest.mark.skipif(cp.cuda.runtime.getDeviceCount() < 2, reason="needs 2 CUDA devices")


def _checksums(a: np.ndarray) -> tuple[int, int]:
    """Exact int64 sum and a position-weighted uint64 sum (mod 2**64): the
    second catches misplaced or permuted writes the first misses."""
    x = a.astype(np.int64)
    s0 = int(x.sum())
    s1 = int(((np.arange(1, len(x) + 1, dtype=np.uint64)) * x.astype(np.uint64)).sum(dtype=np.uint64))
    return s0, s1


def _never_direct(dst, src):
    raise AssertionError("a direct device-to-device copy was issued on the default path")


def test_knob_default_off(monkeypatch):
    monkeypatch.delenv("ET_MINER_DIRECT_D2D", raising=False)
    assert _env.direct_d2d() is False
    monkeypatch.setenv("ET_MINER_DIRECT_D2D", "0")
    assert _env.direct_d2d() is False
    monkeypatch.setenv("ET_MINER_DIRECT_D2D", "1")
    assert _env.direct_d2d() is True


def test_same_device_copy_is_a_device_assignment(monkeypatch):
    monkeypatch.setattr(nccl_mod, "_direct_copy", _never_direct)
    monkeypatch.setattr(nccl_mod, "_host_staged_copy", _never_direct)
    a = np.arange(1000, dtype=np.int32)
    with cp.cuda.Device(0):
        src = cp.asarray(a)
        dst = cp.zeros(1000, dtype=cp.int32)
        copy_between_devices(dst, src)
        np.testing.assert_array_equal(dst.get(), a)


def test_shape_or_dtype_mismatch_is_rejected():
    with cp.cuda.Device(0):
        with pytest.raises(ValueError, match="shape/dtype"):
            copy_between_devices(cp.zeros(3, dtype=cp.int32), cp.zeros(4, dtype=cp.int32))


@pytest.mark.multigpu
@needs_two
def test_default_copy_goes_through_the_host(monkeypatch):
    monkeypatch.delenv("ET_MINER_DIRECT_D2D", raising=False)
    monkeypatch.setattr(nccl_mod, "_direct_copy", _never_direct)
    rng = np.random.default_rng(7)
    a = rng.integers(-1000, 1000, 8_000_003, dtype=np.int32)
    with cp.cuda.Device(1):
        src = cp.asarray(a)
    with cp.cuda.Device(0):
        dst = cp.full(a.shape, -1, dtype=cp.int32)
    copy_between_devices(dst, src)
    with cp.cuda.Device(0):
        got = dst.get()
    np.testing.assert_array_equal(got, a)
    assert _checksums(got) == _checksums(a)


@pytest.mark.multigpu
@needs_two
def test_opt_in_routes_to_the_direct_copy(monkeypatch):
    """The knob selects the direct path; the direct copy itself is replaced
    here so the test never issues a P2P write on a box that drops them."""
    monkeypatch.setenv("ET_MINER_DIRECT_D2D", "1")
    calls = []

    def fake_direct(dst, src):
        calls.append((int(dst.device.id), int(src.device.id)))
        nccl_mod._host_staged_copy(dst, src)

    monkeypatch.setattr(nccl_mod, "_direct_copy", fake_direct)
    with cp.cuda.Device(1):
        src = cp.arange(4096, dtype=cp.int32)
    with cp.cuda.Device(0):
        dst = cp.zeros(4096, dtype=cp.int32)
    copy_between_devices(dst, src)
    assert calls == [(0, 1)]
    with cp.cuda.Device(0):
        np.testing.assert_array_equal(dst.get(), np.arange(4096, dtype=np.int32))


@pytest.mark.multigpu
@needs_two
@pytest.mark.parametrize("n", [1, 4096, 8_000_003, nccl_mod.STAGING_BYTES // 4 + 5])  # the last spans two staging slices
def test_staged_reduce_sums_both_shards(n):
    """Real transfers, no monkeypatching: the reduce must equal the NumPy sum."""
    rng = np.random.default_rng(n)
    a = rng.integers(0, 1000, n, dtype=np.int32)
    b = rng.integers(0, 1000, n, dtype=np.int32)
    with cp.cuda.Device(0):
        ga = cp.asarray(a)
    with cp.cuda.Device(1):
        gb = cp.asarray(b)
    _staged_reduce_to_gpu0([ga, gb], [0, 1])
    with cp.cuda.Device(0):
        got = ga.get()
    np.testing.assert_array_equal(got, a + b)
    assert _checksums(got) == _checksums(a + b)


@pytest.mark.multigpu
@needs_two
def test_default_reduce_issues_no_direct_copy(monkeypatch):
    monkeypatch.delenv("ET_MINER_DIRECT_D2D", raising=False)
    monkeypatch.setattr(nccl_mod, "_direct_copy", _never_direct)
    rng = np.random.default_rng(11)
    a = rng.integers(0, 1000, 8_000_000, dtype=np.int32)
    b = rng.integers(0, 1000, 8_000_000, dtype=np.int32)
    with cp.cuda.Device(0):
        ga = cp.asarray(a)
    with cp.cuda.Device(1):
        gb = cp.asarray(b)
    reduce_sum_to_gpu0([ga, gb], [0, 1], comms=None)
    with cp.cuda.Device(0):
        np.testing.assert_array_equal(ga.get(), a + b)


@pytest.mark.multigpu
@needs_two
@pytest.mark.parametrize("home", [0, 1])
def test_prebuilt_shards_are_the_host_slices(monkeypatch, home):
    from et_miner.gpu.row_split import shard_prebuilt_bitvecs

    monkeypatch.delenv("ET_MINER_DIRECT_D2D", raising=False)
    monkeypatch.setattr(nccl_mod, "_direct_copy", _never_direct)
    rng = np.random.default_rng(home)
    packed = rng.integers(0, 2**63 - 1, size=(64, 40_001), dtype=np.int64).astype(np.uint64)  # 20 MiB
    with cp.cuda.Device(home):
        bv = cp.asarray(packed)
    shards = shard_prebuilt_bitvecs(bv, 40_001 * 64, 2)
    assert [d for _, d, _ in shards] == [0, 1]
    lo = 0
    for arr, dev, rows in shards:
        with cp.cuda.Device(dev):
            got = arr.get()
        np.testing.assert_array_equal(got, packed[:, lo : lo + got.shape[1]])
        assert rows == got.shape[1] * 64
        lo += got.shape[1]
    assert lo == 40_001
    with cp.cuda.Device(home):
        np.testing.assert_array_equal(bv.get(), packed)


class _RecordingCommunicator:
    """Stands in for cupy.cuda.nccl.NcclCommunicator and records the P2P policy NCCL would read."""

    seen: list = []

    def __init__(self, n, uid, rank):
        from cupy.cuda import nccl as _nccl

        _RecordingCommunicator.seen.append(os.environ.get("NCCL_P2P_LEVEL"))
        self._real = _nccl.__dict__["_RealCommunicator"](n, uid, rank)

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture
def recording_communicator(monkeypatch):
    from cupy.cuda import nccl as _nccl

    real = _nccl.NcclCommunicator
    monkeypatch.setitem(_nccl.__dict__, "_RealCommunicator", real)
    monkeypatch.setattr(_nccl, "NcclCommunicator", _RecordingCommunicator)
    _RecordingCommunicator.seen = []
    return _RecordingCommunicator


@pytest.mark.multigpu
@needs_two
def test_nccl_is_created_under_nvl_only_and_the_variable_is_restored(monkeypatch, recording_communicator):
    monkeypatch.delenv("NCCL_P2P_LEVEL", raising=False)
    monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
    monkeypatch.delenv("ET_MINER_DISABLE_NCCL", raising=False)
    comms, ok = nccl_mod._init_nccl([0, 1])
    assert ok and comms is not None and len(comms) == 2
    assert recording_communicator.seen == ["NVL", "NVL"]
    assert "NCCL_P2P_LEVEL" not in os.environ
    assert "NCCL_P2P_DISABLE" not in os.environ


@pytest.mark.multigpu
@needs_two
@pytest.mark.parametrize("var,value", [("NCCL_P2P_DISABLE", "1"), ("NCCL_P2P_LEVEL", "PXB")])
def test_a_callers_nccl_setting_wins(monkeypatch, recording_communicator, var, value):
    monkeypatch.delenv("NCCL_P2P_LEVEL", raising=False)
    monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
    monkeypatch.delenv("ET_MINER_DISABLE_NCCL", raising=False)
    monkeypatch.setenv(var, value)
    nccl_mod._init_nccl([0, 1])
    expected = value if var == "NCCL_P2P_LEVEL" else None
    assert recording_communicator.seen == [expected, expected]
    assert os.environ.get(var) == value
