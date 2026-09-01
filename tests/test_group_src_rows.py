"""CPU tests for ``K3PlusGroups.suffix_src_rows`` — the suffix-slot → prev-row
permutation the sparse-CSR kernels use to enumerate candidates in-kernel.

Every builder (Rust, numpy fallback, the public function with Rust absent)
and every apriori-prune implementation (Rust, Python fallback) must satisfy:
for each group g and suffix slot s, ``prev_flat[rows[s]] == prefix_g + (suffix_s,)``,
rows unique, dtype int64, suffixes strictly ascending within a group.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import et_miner.backends as backends
from et_miner.gpu.kernels.k3plus import K3PlusGroups, _build_k3plus_groups_numpy, build_k3plus_groups_from_flat
from et_miner.gpu.mining import _prune_groups_apriori

KS = [2, 3, 4, 5]


def _random_prev_flat(k: int, seed: int, n_prefixes: int = 40) -> np.ndarray:
    """Sorted (k)-itemsets over random prefixes with 1..8 suffixes each — so
    the table has singleton groups AND groups of four or more — returned in
    shuffled row order (the builders must not assume sorted input)."""
    rng = np.random.default_rng(seed)
    rows: set[tuple[int, ...]] = set()
    for _ in range(n_prefixes):
        prefix = tuple(sorted(int(x) for x in rng.choice(200, size=k - 1, replace=False)))
        n_suf = int(rng.integers(1, 9))
        pool = np.arange(prefix[-1] + 1, 260)
        for s in rng.choice(pool, size=min(n_suf, len(pool)), replace=False):
            rows.add(prefix + (int(s),))
    flat = np.array(sorted(rows), dtype=np.int32)
    rng.shuffle(flat, axis=0)
    return flat


def _clique_plus_noise(k: int, seed: int) -> np.ndarray:
    """All (k-1)-subsets of the clique {0..7} — every apriori subset check
    succeeds, so those groups survive pruning, and they have up to 7
    suffixes — plus noise groups that share a (k-2)-prefix but whose
    prefix-drop subsets are absent, so every noise candidate is pruned.
    Shuffled row order."""
    rng = np.random.default_rng(seed)
    rows = {tuple(c) for c in itertools.combinations(range(8), k - 1)}
    for _ in range(10):
        prefix = tuple(sorted(int(x) for x in rng.choice(np.arange(20, 200), size=k - 2, replace=False)))
        lo = (prefix[-1] + 1) if prefix else 20
        for s in rng.choice(np.arange(lo, 260), size=int(rng.integers(3, 6)), replace=False):
            rows.add(prefix + (int(s),))
    flat = np.array(sorted(rows), dtype=np.int32)
    rng.shuffle(flat, axis=0)
    return flat


def _assert_slots_map_to_rows(groups: K3PlusGroups, prev_flat: np.ndarray) -> None:
    rows = groups.suffix_src_rows
    assert rows is not None
    assert rows.dtype == np.int64
    assert len(rows) == len(groups.suffixes)
    assert len(np.unique(rows)) == len(rows), "a prev row can occupy only one suffix slot"
    n_groups = len(groups.suffix_offsets) - 1
    for g in range(n_groups):
        p0, p1 = int(groups.prefix_offsets[g]), int(groups.prefix_offsets[g + 1])
        prefix = tuple(int(x) for x in groups.prefix_items[p0:p1])
        s0, s1 = int(groups.suffix_offsets[g]), int(groups.suffix_offsets[g + 1])
        for slot in range(s0, s1):
            got = tuple(int(x) for x in prev_flat[rows[slot]])
            assert got == prefix + (int(groups.suffixes[slot]),), (g, slot, got)
        assert np.all(np.diff(groups.suffixes[s0:s1]) > 0), "suffixes must be strictly ascending within a group"


def _has_fresh_rust() -> bool:
    ext = backends.get_rust_ext()
    if ext is None:
        return False
    version = tuple(int(x) for x in str(getattr(ext, "__version__", "0.0.0")).split(".")[:2])
    return version >= (0, 2)


requires_rust = pytest.mark.skipif(not _has_fresh_rust(), reason="et_miner_rust >= 0.2.0 not built")


@requires_rust
@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("seed", [0, 1])
def test_rust_builder_rows(k, seed):
    flat = _random_prev_flat(k, seed)
    groups = build_k3plus_groups_from_flat(flat, with_src_rows=True)
    assert groups is not None
    _assert_slots_map_to_rows(groups, flat)
    # some group must be big enough to exercise j-major decode order
    assert int(np.diff(groups.suffix_offsets).max()) >= 4


@requires_rust
@pytest.mark.parametrize("k", KS)
def test_rust_builder_default_has_no_rows(k):
    groups = build_k3plus_groups_from_flat(_random_prev_flat(k, 3))
    assert groups is not None
    assert groups.suffix_src_rows is None


@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("seed", [0, 1])
def test_numpy_builder_rows(k, seed):
    flat = _random_prev_flat(k, seed)
    groups = _build_k3plus_groups_numpy(flat, with_src_rows=True)
    assert groups is not None
    _assert_slots_map_to_rows(groups, flat)
    assert _build_k3plus_groups_numpy(flat).suffix_src_rows is None


@requires_rust
@pytest.mark.parametrize("k", KS)
def test_numpy_builder_matches_rust(k):
    """Both builders sort by the full row, so the group arrays are identical."""
    flat = _random_prev_flat(k, 7)
    a = build_k3plus_groups_from_flat(flat, with_src_rows=True)
    b = _build_k3plus_groups_numpy(flat, with_src_rows=True)
    for field in ("prefix_items", "prefix_offsets", "suffixes", "suffix_offsets", "cumulative_pairs", "suffix_src_rows"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field), err_msg=field)
    assert a.total_candidates == b.total_candidates


@pytest.mark.parametrize("k", KS)
def test_public_builder_falls_back_without_rust(monkeypatch, k):
    monkeypatch.setattr(backends, "_rust_ext", None)
    flat = _random_prev_flat(k, 5)
    groups = build_k3plus_groups_from_flat(flat, with_src_rows=True)
    assert groups is not None
    _assert_slots_map_to_rows(groups, flat)


@pytest.mark.parametrize("k", [3, 4, 5])
def test_prune_implementations_carry_rows_and_agree(k):
    flat = _clique_plus_noise(k, 11)
    prev_set = set(map(tuple, flat.tolist()))
    groups = _build_k3plus_groups_numpy(flat, with_src_rows=True)
    assert groups is not None

    python_pruned = _prune_groups_apriori(groups, prev_set, k, prev_flat_np=None)
    assert python_pruned is not None
    assert python_pruned.total_candidates < groups.total_candidates, "fixture must actually prune something"
    _assert_slots_map_to_rows(python_pruned, flat)

    if _has_fresh_rust():
        rust_pruned = _prune_groups_apriori(groups, prev_set, k, prev_flat_np=flat)
        assert rust_pruned is not None
        _assert_slots_map_to_rows(rust_pruned, flat)
        for field in (
            "prefix_items",
            "prefix_offsets",
            "suffixes",
            "suffix_offsets",
            "cumulative_pairs",
            "suffix_src_rows",
        ):
            np.testing.assert_array_equal(getattr(rust_pruned, field), getattr(python_pruned, field), err_msg=field)
        assert rust_pruned.total_candidates == python_pruned.total_candidates


@pytest.mark.parametrize("k", [3, 4])
def test_prune_without_rows_keeps_none(k):
    flat = _clique_plus_noise(k, 13)
    prev_set = set(map(tuple, flat.tolist()))
    groups = _build_k3plus_groups_numpy(flat)  # no rows requested
    assert groups.suffix_src_rows is None
    assert _prune_groups_apriori(groups, prev_set, k, prev_flat_np=None).suffix_src_rows is None
    if _has_fresh_rust():
        assert _prune_groups_apriori(groups, prev_set, k, prev_flat_np=flat).suffix_src_rows is None


def test_keyword_construction_defaults_rows_to_none():
    g = K3PlusGroups(
        prefix_items=np.array([1], np.int32),
        prefix_offsets=np.array([0, 1], np.int64),
        suffixes=np.array([2, 3], np.int32),
        suffix_offsets=np.array([0, 2], np.int64),
        cumulative_pairs=np.array([0, 1], np.int64),
        total_candidates=1,
        groups=None,
    )
    assert g.suffix_src_rows is None
    assert len(g) == 8
