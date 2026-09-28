"""Input guards on the bitvec kernel entry points, on a device.

Every exported wrapper that launches on a caller's bitvecs runs
`loader._assert_bitvecs` first: the bitvecs must be a 2-D uint64 CuPy array,
and the launch is pinned to the device they live on. The dtype check is the
one with nothing behind it: the kernels read the bitvecs through an
`unsigned long long*`, so a wrong-dtype array of the right shape would count
plausible garbage without an exception. Which wrappers call the guard is
checked without a device in `test_kernel_guard_claims.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

pytestmark = pytest.mark.gpu


def _entry_points():
    """(name, call) for every guarded wrapper, each on a 2-group K=3 level over 8 columns."""
    from et_miner.gpu.kernels import (
        K3PlusGroups,
        count_itemsets_cuda,
        count_k3plus_per_candidate,
        count_pairs_k2_per_candidate,
        count_pairs_k2_shared,
        count_shared_tiled_allcounts,
        count_tiled_fused,
    )

    groups = K3PlusGroups(
        prefix_items=np.array([0, 1], dtype=np.int32),
        prefix_offsets=np.array([0, 1, 2], dtype=np.int64),
        suffixes=np.array([2, 3, 4, 5, 6, 7], dtype=np.int32),
        suffix_offsets=np.array([0, 3, 6], dtype=np.int64),
        cumulative_pairs=np.array([0, 3, 6], dtype=np.int64),
        total_candidates=6,
        groups=None,
    )
    cols = list(range(8))
    return [
        ("count_pairs_k2_per_candidate", lambda bv: count_pairs_k2_per_candidate(bv, cols, 4)),
        ("count_pairs_k2_shared", lambda bv: count_pairs_k2_shared(bv, cols, 4)),
        ("count_k3plus_per_candidate", lambda bv: count_k3plus_per_candidate(bv, groups, 4)),
        ("count_shared_tiled_allcounts", lambda bv: count_shared_tiled_allcounts(bv, groups, 4)),
        ("count_tiled_fused", lambda bv: count_tiled_fused(bv, groups, 4, 1)),
        ("count_itemsets_cuda", lambda bv: count_itemsets_cuda(bv, [np.array([0, 1], dtype=np.int32)])),
    ]


@pytest.fixture
def bitvecs():
    with cp.cuda.Device(0):
        bv = cp.zeros((8, 4), dtype=cp.uint64)
        bv[:, 0] = cp.arange(8, dtype=cp.uint64) | 0xFF
    return bv


@pytest.mark.parametrize("name", [n for n, _ in _entry_points()])
def test_the_guard_admits_correct_inputs(bitvecs, name):
    """The negative cases below prove nothing if the positive one cannot run."""
    dict(_entry_points())[name](bitvecs)


@pytest.mark.parametrize("name", [n for n, _ in _entry_points()])
def test_a_non_uint64_bitvec_is_rejected(bitvecs, name):
    with pytest.raises(ValueError, match=r"bitvecs_gpu must be uint64, got int64"):
        dict(_entry_points())[name](bitvecs.astype(cp.int64))


@pytest.mark.parametrize("name", [n for n, _ in _entry_points()])
def test_a_1d_bitvec_is_rejected(bitvecs, name):
    with pytest.raises(ValueError, match=r"bitvecs_gpu must be 2-D, got 1-D"):
        dict(_entry_points())[name](bitvecs.ravel())


@pytest.mark.parametrize("name", [n for n, _ in _entry_points()])
def test_a_host_bitvec_is_rejected_with_a_readable_error(bitvecs, name):
    with pytest.raises(ValueError, match=r"bitvecs_gpu is not a CuPy array resident on a CUDA device"):
        dict(_entry_points())[name](bitvecs.get())
