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

One hand-written cast feeds that path: `cp.where(freq_mask)[0]` in
`gpu/mining.py` returns int64 natively and is `.astype(cp.int32)`'d on the spot
to build the K=1 seed.

WHAT CATCHES A DROPPED `.astype`, measured rather than reasoned. Deleting it at
`gpu/mining.py:974` fails
`tests/test_gpu_resident_e2e.py::test_gpu_resident_matches_cpu_reference`, where
the `_assert_dtype` guard on `count_pairs_fused_k2_gpu_resident` raises on the
seed -- the guard working as designed, since an end-to-end run is what supplies
it a real seed. The BEHAVIOURAL backstop is therefore that e2e test, not
anything in this file.

Under the same mutation this file used to stay entirely green: the test that
claimed to pin the cast "directly" rebuilt the expression locally and never
imported `gpu.mining`. It now reads the module's own source and goes red
(1 failed, 12 passed, re-measured after the change), so the claim and the check
finally agree. That is a source check, not a behavioural one, and the docstring
on the test says so.

CONTROL: drop the `_assert_dtype` call from an entry point and its dtype case
returns a result instead of raising. Drop `_assert_rank` and the rank cases
raise IndexError or produce a wrong-shaped result rather than a named error.
Both were checked by mutation; so was the `gpu/mining.py` claim above.
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


def test_the_k1_seed_cast_is_still_in_mining():
    """Read off `gpu.mining`'s own source, because that is the file the claim is
    about. A local rebuild of the expression -- what this test used to do --
    stays green when the cast is deleted from the module.

    Source inspection and not behaviour: the cast is 60 lines inside
    `_apriori_from_bitvecs_gpu_resident`, reachable only with a real bitvec
    matrix, and the behavioural check already exists one layer out
    (`test_gpu_resident_e2e.py`, via the `_assert_dtype` guard). This pins the
    expression so a reader of THIS file is not told a cast is checked here when
    it is not."""
    import inspect

    from et_miner.gpu import mining

    src = inspect.getsource(mining._apriori_from_bitvecs_gpu_resident)
    assert "cp.where(freq_mask)[0].astype(cp.int32)" in src, (
        "the K=1 seed cast is gone or reshaped. `cp.where` is int64 natively, so "
        "without it the seed reaches count_pairs_fused_k2_gpu_resident as int64 "
        "and _assert_dtype raises there -- see test_gpu_resident_e2e.py."
    )


def test_the_guards_accept_a_correctly_built_k1_seed():
    """The other half: an int32 (n,1) seed passes both guards. Without this the
    test above could be satisfied by a cast to the WRONG width."""
    with cp.cuda.Device(0):
        col_counts = cp.array([5, 0, 7, 3], dtype=cp.int64)
        freq_mask = col_counts >= 3
        assert cp.where(freq_mask)[0].dtype == cp.int64, "premise: cp.where is int64 natively"
        seed = cp.where(freq_mask)[0].astype(cp.int32).reshape(-1, 1)

    from et_miner.gpu.kernels.loader import _assert_dtype, _assert_rank

    _assert_rank("k1 seed", prev_freq_gpu=(seed, 2))
    _assert_dtype("k1 seed", prev_freq_gpu=(seed, "int32"))


# --------------------------------------------------------------------------
# Structural closure for the module docstrings' set claims
# --------------------------------------------------------------------------
#
# `gpu_resident.py`'s docstring and `loader.py::_assert_home`'s both quantify
# over "the entry points that guard their inputs". Successive rewordings of that
# sentence were each false about `build_prefix_groups_gpu`, which this package
# exports and which calls no guard. These tests make the two tuples the claim,
# so a new export cannot join the family by being described as part of it.
#
# CPU-only on purpose -- no `cp` use below, so they run on a box with no
# device. They are in this file's `pytestmark = pytest.mark.gpu` scope
# regardless; splitting them out would separate the check from the claim it
# closes, and the campaign gate runs with devices present.


def _exported_from_gpu_resident() -> set[str]:
    """Names `kernels/__init__.py` re-exports that are DEFINED in
    `gpu_resident`, read off the objects rather than parsed out of the import
    statement.

    `__module__` is the load-bearing filter, not `dir()`: `gpu_resident` itself
    imports `get_cuda_kernel` and the `_assert_*` helpers from `loader`, so
    `dir()` sees them and identity against `kernels` matches -- the first
    version of this helper reported `get_cuda_kernel` as an unclassified
    gpu-resident entry point. It says "defined in", so it must test that."""
    from et_miner.gpu import kernels
    from et_miner.gpu.kernels import gpu_resident

    return {
        n
        for n in kernels.__all__
        if getattr(getattr(kernels, n, None), "__module__", None) == gpu_resident.__name__
    }


def test_every_exported_entry_point_is_classified_exactly_once():
    from et_miner.gpu.kernels import gpu_resident

    guarded = set(gpu_resident._GUARDED_ENTRY_POINTS)
    exempt = set(gpu_resident._EXEMPT_ENTRY_POINTS)
    exported = _exported_from_gpu_resident()

    assert exported, "premise: the package must re-export something from gpu_resident"
    assert guarded & exempt == set(), f"classified both ways: {sorted(guarded & exempt)}"
    assert exported - (guarded | exempt) == set(), (
        f"exported but unclassified: {sorted(exported - (guarded | exempt))}. Add each to "
        "_GUARDED_ENTRY_POINTS (and give it the three guards) or to _EXEMPT_ENTRY_POINTS "
        "with the reason -- the module docstring quantifies over those two tuples."
    )
    assert (guarded | exempt) - exported == set(), (
        f"classified but not exported: {sorted((guarded | exempt) - exported)}. The tuples "
        "describe this module's PUBLIC surface; a renamed or withdrawn export must leave them."
    )


def test_the_guarded_entry_points_really_call_all_three_guards():
    """Membership checked against the function's source, not declared. Adding a
    name to `_GUARDED_ENTRY_POINTS` without adding the guards fails here."""
    import inspect

    from et_miner.gpu.kernels import gpu_resident

    for name in gpu_resident._GUARDED_ENTRY_POINTS:
        src = inspect.getsource(getattr(gpu_resident, name))
        for guard in ("_assert_home(", "_assert_rank(", "_assert_dtype("):
            assert guard in src, f"{name} is in _GUARDED_ENTRY_POINTS but its source has no {guard}"


def test_the_exempt_entry_points_call_no_guard_and_say_why():
    """The other direction: an exempt name that HAS acquired guards is a stale
    exemption, and belongs in the guarded tuple instead."""
    import inspect

    from et_miner.gpu.kernels import gpu_resident

    for name, reason in gpu_resident._EXEMPT_ENTRY_POINTS.items():
        src = inspect.getsource(getattr(gpu_resident, name))
        for guard in ("_assert_home(", "_assert_rank(", "_assert_dtype("):
            assert guard not in src, (
                f"{name} is listed exempt but calls {guard} -- move it to _GUARDED_ENTRY_POINTS."
            )
        assert reason.strip(), f"{name} is exempt with no stated reason"


def test_the_k2_and_k3_families_are_both_represented():
    """A guarded tuple that lost its K=2 half, or its multi-GPU half, would
    still pass every test above while the docstring's "differing only in the
    rank each requires" stopped covering anything."""
    from et_miner.gpu.kernels import gpu_resident

    guarded = set(gpu_resident._GUARDED_ENTRY_POINTS)
    assert any("k2" in n for n in guarded) and any("k3plus" in n for n in guarded)
    assert any(n.endswith("_multi_gpu") for n in guarded)
    assert any(not n.endswith("_multi_gpu") for n in guarded)


def _src_files() -> list:
    """Every .py under the package, so a claim about call sites can be checked
    against the tree rather than against the last time someone grepped."""
    from pathlib import Path

    import et_miner

    return sorted(Path(et_miner.__file__).parent.rglob("*.py"))


def _call_sites(symbol: str) -> set[str]:
    """`<relative path>:<lineno>` for every call to `symbol`, excluding its own
    `def` line and any line that is only a comment or an import."""
    import re
    from pathlib import Path

    import et_miner

    root = Path(et_miner.__file__).parent
    pat = re.compile(rf"\b{re.escape(symbol)}\s*\(")
    out = set()
    for f in _src_files():
        for i, line in enumerate(f.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("#", "import ", "from ")) or stripped.startswith(f"def {symbol}"):
                continue
            if pat.search(line):
                out.add(f"{f.relative_to(root)}:{i}")
    return out


def test_assert_home_is_called_only_from_the_guarded_entry_points():
    """`loader.py::_assert_home` says its callers are EXACTLY
    `_GUARDED_ENTRY_POINTS`. "Exactly" is two claims -- every guarded one calls
    it (above) and nothing else does (here). Only asserting the first is how a
    set claim stays half-true."""
    import inspect

    from et_miner.gpu.kernels import gpu_resident

    guarded_src = "\n".join(inspect.getsource(getattr(gpu_resident, n)) for n in gpu_resident._GUARDED_ENTRY_POINTS)
    n_in_guarded = guarded_src.count("_assert_home(")
    sites = _call_sites("_assert_home")

    assert len(sites) == n_in_guarded, (
        f"_assert_home has {len(sites)} call sites tree-wide ({sorted(sites)}) but "
        f"{n_in_guarded} inside _GUARDED_ENTRY_POINTS. If a new caller is legitimate, the "
        "'callers are exactly _GUARDED_ENTRY_POINTS' sentence in loader.py is now false."
    )
    assert all(s.startswith("gpu/kernels/gpu_resident.py:") for s in sites), sorted(sites)


def test_build_prefix_groups_gpu_call_sites_match_its_exemption():
    """The exemption reason for `build_prefix_groups_gpu` counts its call sites
    and says one of them (`gpu/dispatch.py`) reaches it BEFORE any guard. A
    counted claim in prose goes stale the moment a caller is added."""
    from et_miner.gpu.kernels import gpu_resident

    sites = _call_sites("build_prefix_groups_gpu")
    files = {s.rsplit(":", 1)[0] for s in sites}

    assert files == {"gpu/kernels/gpu_resident.py", "gpu/dispatch.py"}, sorted(sites)
    assert len(sites) == 3, (
        f"the exemption reason says THREE call sites; found {len(sites)}: {sorted(sites)}. "
        "Update _EXEMPT_ENTRY_POINTS['build_prefix_groups_gpu'] with it."
    )
    assert "gpu/dispatch.py" in gpu_resident._EXEMPT_ENTRY_POINTS["build_prefix_groups_gpu"]
