"""Rank and dtype preconditions on the gpu-resident entry points.

Separate from `test_gpu_device_affinity.py`, whose docstring scopes it to N10
and N20 -- wrappers assuming the caller's arrays live on device 0. These are
the two preconditions that are NOT about which device an input is on.

The dtype guard is the one with nothing behind it:

* a mixed-device input aborts loudly (CUDA_ERROR_ILLEGAL_ADDRESS);
* a wrong-rank input trips an IndexError somewhere downstream, which is
  unreadable but is at least an exception;
* a wrong-DTYPE input does neither. The K>=3 kernels read `prev_freq_gpu`
  through an `int*` cast, so an int64 array of the correct shape is
  reinterpreted pairwise and returns plausible garbage silently -- measured
  before this guard as 56 itemsets of [[0, 0, 0]] at a uniform count of 46,
  from an entry point the package exports.

What stands between the exported entry points and that result is one
hand-written cast: `cp.where(freq_mask)[0]` in `gpu/mining.py` returns int64
natively and is `.astype(cp.int32)`'d on the spot. That cast was untested and
unguarded; `test_the_k1_seed_really_is_int32` pins it directly, so a future
edit dropping the `.astype` fails here and not in a campaign.

CONTROL: drop the `_assert_dtype` call from an entry point and its dtype case
returns a result instead of raising. Drop `_assert_rank` and the rank cases
raise IndexError or produce a wrong-shaped result rather than a named error.
"""

from __future__ import annotations

import pytest

cp = pytest.importorskip("cupy")

pytestmark = pytest.mark.gpu


def _gpu_count() -> int:
    try:
        return cp.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


if _gpu_count() == 0:
    pytest.skip("no CUDA device", allow_module_level=True)


@pytest.fixture
def inputs():
    """Co-resident, correctly ranked, correctly typed -- so every failure below
    is the one variable the case changes, and nothing else."""
    with cp.cuda.Device(0):
        bitvecs = cp.zeros((8, 4), dtype=cp.uint64)
        bitvecs[:, 0] = cp.arange(8, dtype=cp.uint64) | 0xFF
        prev_freq = cp.array([[0, 1], [0, 2], [1, 2]], dtype=cp.int32)
        freq_cols = cp.arange(4, dtype=cp.int32)
    return bitvecs, prev_freq, freq_cols


def test_the_guards_admit_correct_inputs(inputs):
    """The negative cases below prove nothing if the positive one cannot run."""
    from et_miner.gpu.kernels import count_k3plus_gpu_resident, count_pairs_fused_k2_gpu_resident

    bitvecs, prev_freq, freq_cols = inputs
    count_k3plus_gpu_resident(bitvecs, prev_freq, 4, 1)
    count_pairs_fused_k2_gpu_resident(bitvecs, freq_cols, 4, 1)


@pytest.mark.parametrize("entry", ["single", "multi"])
def test_k3plus_rejects_an_int64_prev_freq(inputs, entry):
    """The silent-garbage case. Right device, right rank, right shape."""
    from et_miner.gpu.kernels.gpu_resident import (
        count_k3plus_gpu_resident,
        count_k3plus_gpu_resident_multi_gpu,
    )

    bitvecs, prev_freq, _ = inputs
    with pytest.raises(ValueError, match=r"prev_freq_gpu must be int32, got int64"):
        if entry == "single":
            count_k3plus_gpu_resident(bitvecs, prev_freq.astype(cp.int64), 4, 1)
        else:
            count_k3plus_gpu_resident_multi_gpu(bitvecs, prev_freq.astype(cp.int64), 4, 1, _gpu_count())


@pytest.mark.parametrize("entry", ["single", "multi"])
def test_k2_rejects_an_int64_freq_cols(inputs, entry):
    from et_miner.gpu.kernels.gpu_resident import (
        count_pairs_fused_k2_gpu_resident,
        count_pairs_fused_k2_gpu_resident_multi_gpu,
    )

    bitvecs, _, freq_cols = inputs
    with pytest.raises(ValueError, match=r"freq_cols_gpu must be int32, got int64"):
        if entry == "single":
            count_pairs_fused_k2_gpu_resident(bitvecs, freq_cols.astype(cp.int64), 4, 1)
        else:
            count_pairs_fused_k2_gpu_resident_multi_gpu(bitvecs, freq_cols.astype(cp.int64), 4, 1, _gpu_count())


def test_every_entry_point_rejects_a_non_uint64_bitvec(inputs):
    """bitvecs_gpu is the input all four share, so it is checked on all four."""
    from et_miner.gpu.kernels.gpu_resident import (
        count_k3plus_gpu_resident,
        count_k3plus_gpu_resident_multi_gpu,
        count_pairs_fused_k2_gpu_resident,
        count_pairs_fused_k2_gpu_resident_multi_gpu,
    )

    bitvecs, prev_freq, freq_cols = inputs
    bad = bitvecs.astype(cp.int64)
    n = _gpu_count()
    for fn, second, extra in [
        (count_k3plus_gpu_resident, prev_freq, ()),
        (count_k3plus_gpu_resident_multi_gpu, prev_freq, (n,)),
        (count_pairs_fused_k2_gpu_resident, freq_cols, ()),
        (count_pairs_fused_k2_gpu_resident_multi_gpu, freq_cols, (n,)),
    ]:
        with pytest.raises(ValueError, match=r"bitvecs_gpu must be uint64, got int64"):
            fn(bad, second, 4, 1, *extra)


def test_k2_rejects_a_2d_freq_cols(inputs):
    """The K=2 rank guard is `ndim != 1`, not the K>=3 `ndim != 2`. A 2-D
    `freq_cols_gpu` is co-resident and int32, so only the rank check sees it."""
    from et_miner.gpu.kernels.gpu_resident import count_pairs_fused_k2_gpu_resident

    bitvecs, _, freq_cols = inputs
    with pytest.raises(ValueError, match=r"freq_cols_gpu must be 1-D, got 2-D"):
        count_pairs_fused_k2_gpu_resident(bitvecs, freq_cols.reshape(2, 2), 4, 1)


def test_the_k1_seed_really_is_int32():
    """`gpu/mining.py`'s `cp.where(freq_mask)[0].astype(cp.int32)` is the only
    thing making the K=1 seed int32, and `cp.where` returns int64 natively.
    Pinned on the expression rather than on a mined result: at K=1 a dropped
    cast changes no output, so nothing else in the suite would notice."""
    with cp.cuda.Device(0):
        col_counts = cp.array([5, 0, 7, 3], dtype=cp.int64)
        freq_mask = col_counts >= 3
        assert cp.where(freq_mask)[0].dtype == cp.int64, "premise: cp.where is int64 natively"
        seed = cp.where(freq_mask)[0].astype(cp.int32).reshape(-1, 1)

    from et_miner.gpu.kernels.loader import _assert_dtype, _assert_rank

    _assert_rank("k1 seed", prev_freq_gpu=(seed, 2))
    _assert_dtype("k1 seed", prev_freq_gpu=(seed, "int32"))
