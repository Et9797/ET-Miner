"""ET_MINER_LEAVE_OUT: the row-split miner without one counting arm mines the same lattice.

Each case runs the smoke preset on one GPU with one arm left out (with forced
chunks where the arm only runs beyond one chunk), compares itemsets and
absolute counts with the CPU route, and asserts through spies on the kernel
wrappers that the arm did not run and the arm taking its place did.
"""

from __future__ import annotations

import pytest

from et_miner.core.apriori import apriori
from et_miner.synthetic import PRESETS, generate_transactions

pytestmark = pytest.mark.gpu

SPEC = PRESETS["smoke"]
_WRAPPERS = (
    "count_pairs_k2_shared",
    "count_pairs_k2_per_candidate",
    "count_k3plus_per_candidate",
    "count_group_pairs",
    "count_shared_tiled_allcounts",
    "count_tiled_fused",
)
_DENSE = {"ET_MINER_K2_KERNEL": "dense", "ET_MINER_K3_KERNEL": "dense"}
#: Below smoke's pair space (1,000+ pairs) and its largest prefix groups.
_CHUNK = {"ET_MINER_MAX_CHUNK_CANDS": "40"}

#: case -> (ET_MINER_LEAVE_OUT, extra env, max_length, wrappers that must run, wrappers that must not)
CASES = {
    "k2-percand": ("k2-percand", {"ET_MINER_TILED_MIN_GROUP_PAIRS": "100000"}, 2, ("count_pairs_k2_shared",),
                   ("count_pairs_k2_per_candidate",)),
    "k2-tiled": ("k2-tiled", {}, 2, ("count_pairs_k2_per_candidate",), ("count_pairs_k2_shared",)),
    "k2-fused": ("k2-fused", _CHUNK, 2, ("count_pairs_k2_per_candidate",), ("count_tiled_fused",)),
    "percand": ("percand", {}, None, ("count_group_pairs",), ("count_k3plus_per_candidate",)),
    "group": ("group", {}, None, ("count_k3plus_per_candidate",), ("count_group_pairs",)),
    "tiled": ("tiled", {}, None, ("count_group_pairs",), ("count_shared_tiled_allcounts",)),
    "fused": ("fused,k2-fused", _CHUNK, None, ("count_k3plus_per_candidate",), ("count_tiled_fused",)),
}


@pytest.fixture(scope="module")
def smoke():
    df, _ = generate_transactions(SPEC)
    return df, {n: _counted(apriori(df, min_support=SPEC.min_support, max_length=n)) for n in (2, None)}


def _counted(res) -> set[tuple[tuple[int, ...], int]]:
    return {(tuple(s), round(p * SPEC.n_rows)) for s, p in zip(res["itemset"].to_list(), res["support"].to_list())}


@pytest.mark.parametrize("arm", list(CASES))
def test_leave_one_arm_out_mines_the_same_lattice(smoke, monkeypatch, arm):
    from et_miner.gpu import kernels

    df, lattices = smoke
    left_out, env, max_length, runs, never = CASES[arm]
    want = lattices[max_length]
    calls = dict.fromkeys(_WRAPPERS, 0)
    for name in _WRAPPERS:
        def spy(*a, _real=getattr(kernels, name), _name=name, **k):
            calls[_name] += 1
            return _real(*a, **k)

        monkeypatch.setattr(kernels, name, spy)
    for name, value in {**_DENSE, **env, "ET_MINER_LEAVE_OUT": left_out}.items():
        monkeypatch.setenv(name, value)
    got = _counted(apriori(df, min_support=SPEC.min_support, max_length=max_length, use_gpu=True))
    assert got == want, f"{arm}: {len(got ^ want)} itemsets differ"
    assert all(calls[n] for n in runs), f"{arm}: {calls}"
    assert not any(calls[n] for n in never), f"{arm}: {calls}"


def test_row_wise_k2_without_shared_memory_uses_global_atomics(smoke, monkeypatch):
    from et_miner.gpu.kernels import k2

    df, lattices = smoke
    want = lattices[None]
    launched = []

    def kernel_spy(name, _real=k2.get_cuda_kernel):
        launched.append(name)
        return _real(name)

    monkeypatch.setattr(k2, "get_cuda_kernel", kernel_spy)
    monkeypatch.setenv("ET_MINER_K2_KERNEL", "rows")
    monkeypatch.setenv("ET_MINER_LEAVE_OUT", "k2-shared")
    assert _counted(apriori(df, min_support=SPEC.min_support, use_gpu=True)) == want
    assert set(launched) == {"count_pairs_k2_rows"}


def test_an_unknown_arm_raises(monkeypatch):
    monkeypatch.setenv("ET_MINER_LEAVE_OUT", "tiled,nope")
    from et_miner import _env

    with pytest.raises(ValueError, match="nope"):
        _env.leave_out()
