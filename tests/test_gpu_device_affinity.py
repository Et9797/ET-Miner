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


@pytest.mark.gpu
@pytest.mark.multigpu
def test_k2_single_gpu_follows_its_inputs_not_the_ambient_device():
    """The K=2 twin of N20, which the first remediation left untreated.

    `count_pairs_fused_k2_gpu_resident` allocated its result buffers on the
    AMBIENT device and handed the kernel `bitvecs_gpu` from another card. Three
    reviewers found it independently and one measured it aborting with
    `cudaErrorIllegalAddress` -- on inputs that are co-resident, i.e. inputs the
    co-residency guard passes. Co-residency and ambient-pinning are two
    different properties and neither implies the other.

    Reachable directly and through the multi-GPU wrapper's `n_gpus <= 1`
    fall-through, which `dispatch.py::dispatch_k2_gpu_resident` takes on its
    single-GPU branch. It was unreachable off-device from in-tree callers only
    because the K=1 popcount upstream is a CuPy ElementwiseKernel that raises
    first -- an accident of the caller, and relying on exactly that accident is
    what the K>=3 version of this fix was blocked for.

    CONTROL: drop the `with cp.cuda.Device(...)` from the wrapper and this
    aborts the process rather than failing, so it asserts on counts and on the
    device the results come back on.
    """
    from et_miner.gpu.kernels import count_pairs_fused_k2_gpu_resident

    _assert_two_devices()
    cols_np = np.arange(24, dtype=np.int32)

    bv0, n_u64s = _bitvecs_on(0)
    with cp.cuda.Device(0):
        cols0 = cp.asarray(cols_np)
        w_items, w_counts = count_pairs_fused_k2_gpu_resident(bv0, cols0, n_u64s, MIN_COUNT)
        want = None if w_items is None else _sorted_pairs(cp.asnumpy(w_items), cp.asnumpy(w_counts))

    bv1, _ = _bitvecs_on(1)
    with cp.cuda.Device(1):
        cols1 = cp.asarray(cols_np)

    # Ambient device 0, both inputs on device 1.
    with cp.cuda.Device(0):
        g_items, g_counts = count_pairs_fused_k2_gpu_resident(bv1, cols1, n_u64s, MIN_COUNT)
        got = None if g_items is None else _sorted_pairs(cp.asnumpy(g_items), cp.asnumpy(g_counts))

    assert got == want
    assert want is not None, "fixture must produce frequent pairs or this asserts nothing"
    assert int(g_items.device.id) == 1
    assert int(g_counts.device.id) == 1


@pytest.mark.gpu
@pytest.mark.multigpu
@pytest.mark.parametrize(
    "fn_name, second_kwarg",
    [
        ("count_k3plus_gpu_resident", "prev_freq_gpu"),
        ("count_pairs_fused_k2_gpu_resident", "freq_cols_gpu"),
    ],
)
def test_single_gpu_entry_points_reject_mixed_device_inputs(fn_name, second_kwarg):
    """The co-residency guard on the SINGLE-GPU entry points, called directly.

    These two guards were previously unreachable from any test: the multi-GPU
    parametrisation cannot get here, because the wrapper's own check fires
    above the routing and raises first. So the deviation that added them was
    shipped unpinned. This calls them directly, which is also how
    `gpu/dispatch.py` reaches them on a single-GPU host.

    CONTROL: delete either `_assert_home` call and the mixed-device input
    reaches the kernel instead of raising.
    """
    import et_miner.gpu.kernels as K

    fn = getattr(K, fn_name)
    _assert_two_devices()

    if second_kwarg == "prev_freq_gpu":
        second_np = np.array([(i, j) for i in range(8) for j in range(i + 1, 9)], dtype=np.int32)
    else:
        second_np = np.arange(16, dtype=np.int32)

    bv1, n_u64s = _bitvecs_on(1)
    with cp.cuda.Device(0):
        second_wrong_device = cp.asarray(second_np)

    with pytest.raises(ValueError, match="must be resident on one device"):
        fn(bv1, second_wrong_device, n_u64s, MIN_COUNT)


@pytest.mark.gpu
def test_assert_home_rejects_a_host_array_with_a_readable_error():
    """A host array reaching a device-only path is a ValueError, not AttributeError.

    NumPy 2 gives `ndarray.device == "cpu"` and NumPy 1 has no `.device` at
    all; both used to surface as `AttributeError: 'str' object has no attribute
    'id'` or similar from inside the validator, which reads like a bug in the
    check rather than a bug in the call.

    This pins the validator. It does NOT pin the guard ORDER in the wrappers --
    see the test below, and do not extend this one to try.
    """
    from et_miner.gpu.kernels.loader import _assert_home

    with cp.cuda.Device(0):
        on_gpu = cp.zeros((4, 2), dtype=cp.uint64)

    with pytest.raises(ValueError, match="not a CuPy array resident on a CUDA device"):
        _assert_home("ctx", bitvecs_gpu=on_gpu, prev_freq_gpu=np.zeros((4, 2), dtype=np.int32))


@pytest.mark.gpu
@pytest.mark.parametrize(
    "bad_second",
    [
        pytest.param([(0, 1), (0, 2), (1, 2)], id="host-list-of-tuples"),
        pytest.param("cupy-1d", id="1d-device-array"),
    ],
)
def test_wrapper_reports_a_malformed_prev_freq_before_reading_its_shape(bad_second):
    """The guard ORDER, pinned at the wrapper -- which needs an input that can tell.

    `count_k3plus_gpu_resident` used to run `_assert_k_supported(...shape[1]...)`
    before `_assert_home`, so a malformed `prev_freq_gpu` raised out of the
    k-cap guard instead of the validator added to name it.

    The obvious test does not catch that. A host 2-D ndarray has `.shape[1]`,
    so it reaches the readable ValueError in EITHER order -- which is exactly
    what the validator test above passes, and why extending that one would pin
    nothing. Only inputs without a usable `.shape[1]` distinguish the orders:

        host list of tuples -> AttributeError: 'list' object has no attribute 'shape'
        1-D cupy array      -> IndexError: tuple index out of range

    The list case is a plausible mis-call rather than a contrived one:
    `k3plus.py::count_k3plus_fully_fused` takes precisely a list of candidate
    tuples, one module away, with a near-identical signature.

    The 1-D case is NOT fixed by the ordering, and finding that out is why it
    is parametrised here rather than folded into the list case: a 1-D array is
    co-resident, so `_assert_home` passes it legitimately and there is no
    device fault to report. It needed an explicit rank check, which is what
    the second parameter pins.

    CONTROL: restore `_assert_k_supported` above `_assert_home` and the list
    case raises AttributeError; delete the `ndim != 2` check and the 1-D case
    raises IndexError.
    """
    from et_miner.gpu.kernels import count_k3plus_gpu_resident

    with cp.cuda.Device(0):
        bitvecs_gpu = cp.zeros((4, 2), dtype=cp.uint64)
        second = cp.arange(4, dtype=cp.int32) if bad_second == "cupy-1d" else bad_second

    with pytest.raises(ValueError, match="not a CuPy array resident on a CUDA device|must be 2-D"):
        count_k3plus_gpu_resident(bitvecs_gpu, second, 2, 1)
