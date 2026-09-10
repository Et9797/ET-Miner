"""N10 -- five multi-GPU wrappers assumed the caller's bitvecs were on device 0.

Each fan-out wrapper aliases the caller's array on one device and uploads a
copy to the others, and each chose that device with a literal:

    if device_id == 0:
        bv_gpu = bitvecs_gpu     # "already on GPU 0"
    else:
        bv_gpu = cp.array(bitvecs_np, ...)

With the caller's bitvecs on device 1, device 0 aliases a foreign array -- the
"device where the array resides (0) is different from the current device (1)"
fault seen in a real campaign log -- while device 1 re-uploads a copy of what
it already holds.

The path is latent today: every in-tree route builds bitvecs on device 0. It
stops being latent in PR 10, which routes k>=3 through gpu/dispatch from a
caller that need not. So the fixture builds on device 1 on purpose; nothing
else in the suite does.

CONTROL: with `_home` reverted to the literal 0, every case here either raises
the cross-device error or returns different counts.
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


pytestmark = pytest.mark.skipif(_device_count() < 2, reason="needs 2 CUDA devices")

# Enough columns that the fan-out wrappers are worth calling, small enough to
# stay well inside a 3090 while a second copy exists on the other card.
N_COLS = 24
N_ROWS = 4096
MIN_COUNT = 400


def _bitvecs_on(device_id: int):
    """Pack a deterministic transaction matrix into bitvecs on `device_id`."""
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
    with cp.cuda.Device(device_id):
        return cp.asarray(packed), n_u64s


def _assert_two_devices():
    """Ground rule: assert the capability inside the test, never infer it."""
    assert cp.cuda.runtime.getDeviceCount() == 2, "this suite is written for exactly 2 devices"


def _sorted_pairs(pairs, counts):
    return sorted(zip([tuple(p) for p in pairs], [int(c) for c in np.asarray(counts).tolist()]))


class TestFanOutHonoursTheCallersDevice:
    @pytest.mark.gpu
    @pytest.mark.multigpu
    def test_k2_fused(self):
        from et_miner.gpu.kernels import count_pairs_fused_k2, count_pairs_fused_k2_multi_gpu

        _assert_two_devices()
        freq_cols = list(range(N_COLS))

        bv0, n_u64s = _bitvecs_on(0)
        with cp.cuda.Device(0):
            want = _sorted_pairs(*count_pairs_fused_k2(bv0, freq_cols, n_u64s, MIN_COUNT))

        bv1, _ = _bitvecs_on(1)
        got = _sorted_pairs(*count_pairs_fused_k2_multi_gpu(bv1, freq_cols, n_u64s, MIN_COUNT, 2))

        assert got == want, "multi-GPU k=2 disagreed when the caller's bitvecs were on device 1"

    @pytest.mark.gpu
    @pytest.mark.multigpu
    def test_k3plus_fully_fused(self):
        from et_miner.gpu.kernels import count_k3plus_fully_fused, count_k3plus_fully_fused_multi_gpu

        _assert_two_devices()
        prev = [(i, j) for i in range(8) for j in range(i + 1, 9)]

        bv0, n_u64s = _bitvecs_on(0)
        with cp.cuda.Device(0):
            w_items, w_counts = count_k3plus_fully_fused(bv0, prev, 3, n_u64s, MIN_COUNT)
        want = _sorted_pairs(w_items, w_counts)

        bv1, _ = _bitvecs_on(1)
        g_items, g_counts = count_k3plus_fully_fused_multi_gpu(bv1, prev, 3, n_u64s, MIN_COUNT, 2)
        assert _sorted_pairs(g_items, g_counts) == want

    @pytest.mark.gpu
    @pytest.mark.multigpu
    def test_k3plus_candidate_list(self):
        from et_miner.gpu.kernels import count_itemsets_fused_k3plus, count_itemsets_fused_k3plus_multi_gpu

        _assert_two_devices()
        cands = [(i, j, k) for i in range(6) for j in range(i + 1, 7) for k in range(j + 1, 8)]

        bv0, n_u64s = _bitvecs_on(0)
        with cp.cuda.Device(0):
            w_items, w_counts = count_itemsets_fused_k3plus(bv0, cands, n_u64s, MIN_COUNT)
        want = _sorted_pairs(w_items, w_counts)

        bv1, _ = _bitvecs_on(1)
        g_items, g_counts = count_itemsets_fused_k3plus_multi_gpu(bv1, cands, n_u64s, MIN_COUNT, 2)
        assert _sorted_pairs(g_items, g_counts) == want

    @pytest.mark.gpu
    @pytest.mark.multigpu
    def test_k2_gpu_resident(self):
        from et_miner.gpu.kernels import (
            count_pairs_fused_k2_gpu_resident,
            count_pairs_fused_k2_gpu_resident_multi_gpu,
        )

        _assert_two_devices()

        bv0, n_u64s = _bitvecs_on(0)
        with cp.cuda.Device(0):
            cols0 = cp.arange(N_COLS, dtype=cp.int32)
            w_items, w_counts = count_pairs_fused_k2_gpu_resident(bv0, cols0, n_u64s, MIN_COUNT)
            want = None if w_items is None else _sorted_pairs(cp.asnumpy(w_items), cp.asnumpy(w_counts))

        bv1, _ = _bitvecs_on(1)
        with cp.cuda.Device(1):
            cols1 = cp.arange(N_COLS, dtype=cp.int32)
        g_items, g_counts = count_pairs_fused_k2_gpu_resident_multi_gpu(bv1, cols1, n_u64s, MIN_COUNT, 2)
        got = None if g_items is None else _sorted_pairs(cp.asnumpy(g_items), cp.asnumpy(g_counts))

        assert got == want

    @pytest.mark.gpu
    @pytest.mark.multigpu
    def test_k3plus_gpu_resident(self):
        from et_miner.gpu.kernels import (
            count_k3plus_gpu_resident,
            count_k3plus_gpu_resident_multi_gpu,
        )

        _assert_two_devices()
        prev_np = np.array([(i, j) for i in range(8) for j in range(i + 1, 9)], dtype=np.int32)

        bv0, n_u64s = _bitvecs_on(0)
        with cp.cuda.Device(0):
            prev0 = cp.asarray(prev_np)
            w_items, w_counts = count_k3plus_gpu_resident(bv0, prev0, n_u64s, MIN_COUNT)
            want = None if w_items is None else _sorted_pairs(cp.asnumpy(w_items), cp.asnumpy(w_counts))

        bv1, _ = _bitvecs_on(1)
        with cp.cuda.Device(1):
            prev1 = cp.asarray(prev_np)
        g_items, g_counts = count_k3plus_gpu_resident_multi_gpu(bv1, prev1, n_u64s, MIN_COUNT, 2)
        got = None if g_items is None else _sorted_pairs(cp.asnumpy(g_items), cp.asnumpy(g_counts))

        assert got == want


@pytest.mark.gpu
@pytest.mark.multigpu
@pytest.mark.parametrize("n_gpus", [1, 2])
def test_mixed_device_inputs_raise_rather_than_being_repaired(n_gpus):
    """The co-residency contract, asserted -- on BOTH sides of the routing.

    `count_k3plus_gpu_resident_multi_gpu` aliases the caller's arrays together
    on the home device. Transferring a stray one would hide a caller bug behind
    a copy nobody attributes, so it raises -- and because that makes it a
    crash, the wrapper's docstring states the contract.

    CONTROL, and the reason `n_gpus` is parametrised: restore the pre-fix shape
    -- the check back below `if n_gpus <= 1: return ...` where it first
    shipped, and no check on `count_k3plus_gpu_resident` -- and `n_gpus=1`
    aborts the process with `CUDA_ERROR_ILLEGAL_ADDRESS` (measured). Not a
    clean failure: that poisons the CUDA context process-wide.

    Two guards defend this branch now, so a narrower control does NOT fail:
    move only the wrapper's check back below the routing, leaving
    `count_k3plus_gpu_resident`'s in place, and the fall-through still raises
    from there. Both are kept -- the wrapper's states its own contract, the
    callee's is what makes the crash unreachable.

    That branch is not a corner: it is every call on a one-GPU host and every
    level with a single candidate, including `dispatch.py:230`, the caller the
    fix was written for. `n_gpus=2` passed throughout and proved nothing
    about it.
    """
    from et_miner.gpu.kernels import count_k3plus_gpu_resident_multi_gpu

    _assert_two_devices()
    prev_np = np.array([(i, j) for i in range(8) for j in range(i + 1, 9)], dtype=np.int32)

    bv1, n_u64s = _bitvecs_on(1)
    with cp.cuda.Device(0):
        prev_wrong_device = cp.asarray(prev_np)

    with pytest.raises(ValueError, match="must be resident on one device"):
        count_k3plus_gpu_resident_multi_gpu(bv1, prev_wrong_device, n_u64s, MIN_COUNT, n_gpus)


@pytest.mark.gpu
@pytest.mark.multigpu
def test_k2_mixed_device_inputs_raise_rather_than_being_repaired():
    """Same contract on the K=2 wrapper, which had no check at all.

    Lower severity than the K>=3 case and the assertion says so: measured
    without the guard, this raises CuPy's own `ValueError: The device where the
    array resides (0) is different from the current device (1)` from inside the
    fan-out, not an illegal access. So the guard buys attribution -- which
    array, whose contract -- rather than turning a crash into an error. It is
    here because the contract is identical and a reader should not have to
    work out which of the two wrappers enforces it.
    """
    from et_miner.gpu.kernels import count_pairs_fused_k2_gpu_resident_multi_gpu

    _assert_two_devices()
    cols_np = np.arange(16, dtype=np.int32)

    bv1, n_u64s = _bitvecs_on(1)
    with cp.cuda.Device(0):
        cols_wrong_device = cp.asarray(cols_np)

    with pytest.raises(ValueError, match="must be resident on one device"):
        count_pairs_fused_k2_gpu_resident_multi_gpu(bv1, cols_wrong_device, n_u64s, MIN_COUNT, 2)


@pytest.mark.gpu
@pytest.mark.multigpu
def test_k3plus_single_gpu_follows_its_inputs_not_the_ambient_device():
    """N20's real blast radius: the single-GPU K>=3 body, called off-device.

    Both inputs on device 1, the calling thread parked on device 0 -- exactly
    what `gpu/dispatch` produces once k>=3 fans out. `count_k3plus_gpu_resident`
    allocated its output buffers and launched on the *ambient* device while
    reading device-1 pointers.

    It used to be caught by accident: `build_prefix_groups_gpu` ran first, on
    the ambient device, and raised `ValueError: The device where the array
    resides (1) is different from the current device (0)` before any launch.
    Making that helper device-following fixed the helper and deleted the
    accident, and the body reached the kernel with two cards' pointers ->
    `CUDA_ERROR_ILLEGAL_ADDRESS`, which poisons the context process-wide. A
    lost campaign, not a lost level, and no test failed.

    CONTROL: drop the `with cp.cuda.Device(...)` from the wrapper and this
    aborts the process rather than failing -- so assert on the counts, which is
    what the wrapper being on the right device actually buys.
    """
    from et_miner.gpu.kernels import count_k3plus_gpu_resident

    _assert_two_devices()
    prev_np = np.array([(i, j) for i in range(8) for j in range(i + 1, 9)], dtype=np.int32)

    bv0, n_u64s = _bitvecs_on(0)
    with cp.cuda.Device(0):
        prev0 = cp.asarray(prev_np)
        w_items, w_counts = count_k3plus_gpu_resident(bv0, prev0, n_u64s, MIN_COUNT)
        want = None if w_items is None else _sorted_pairs(cp.asnumpy(w_items), cp.asnumpy(w_counts))

    bv1, _ = _bitvecs_on(1)
    with cp.cuda.Device(1):
        prev1 = cp.asarray(prev_np)

    # Ambient device 0, both inputs on device 1.
    with cp.cuda.Device(0):
        g_items, g_counts = count_k3plus_gpu_resident(bv1, prev1, n_u64s, MIN_COUNT)
        got = None if g_items is None else _sorted_pairs(cp.asnumpy(g_items), cp.asnumpy(g_counts))

    assert got == want
    if g_items is not None:
        assert int(g_items.device.id) == 1
        assert int(g_counts.device.id) == 1
