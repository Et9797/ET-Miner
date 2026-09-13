"""``prune_apriori`` is its own switch, decoupled from ``prune_equal_support``.

The row-split miner's K>=3 Apriori subset test used to be wired to the free-set
gate at the dispatch site (``prune_apriori=prune_equal_support``). A run that
asked for the complete lattice on that route -- ``n_gpus>1`` or ``anchor_items``
with ``prune_equal_support=False`` -- therefore counted every suffix extension
of every frequent group with no downward closure at all. Exact, since the test
only removes candidates that cannot be frequent, but the K>=3 candidate count
grew with the lattice instead of being cut by it. The test now defaults to on
and is threaded through ``apriori(prune_apriori=...)``.

Exactness is the hard assertion: itemsets and counts must not move, pruned or
not. Whether the prune ENGAGED is observed directly by spying on
``gpu.row_split._prune_groups_apriori`` -- the row-split miner runs its GPU
workers as threads of the calling process, so the spy sees every call, and its
before/after candidate counts are the per-level figures the engine's DEBUG log
reports. Off the row-split miner the switch does not exist, so ``False`` is
refused there like every other parameter/route mismatch.
"""

from __future__ import annotations

import inspect
import itertools

import numpy as np
import polars as pl
import pytest

from et_miner.core.apriori import apriori

MIN_SUPPORT = 0.03
MAX_K = 5


def _gpu_count() -> int:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


def _fixture(n_rows: int = 4000, n_base: int = 14, seed: int = 7):
    """Nested vocabulary (a child item implies its parent) plus themed noise:
    frequent pairs are plentiful, frequent triples are not, so the K>=3 group
    candidates include ones with an infrequent (k-1)-subset and the subset test
    has something to remove."""
    rng = np.random.default_rng(seed)
    parent = {i: i % 4 for i in range(4, n_base)}
    themes = [rng.choice(range(4, n_base), size=5, replace=False) for _ in range(8)]
    matrix = np.zeros((n_rows, n_base), dtype=bool)
    rows = []
    for i in range(n_rows):
        items = {int(x) for x in themes[i % 8][rng.random(5) < 0.9]}
        items |= {parent[x] for x in items}
        items |= {int(x) for x in rng.choice(n_base, size=int(rng.integers(0, 3)), replace=False)}
        items.add(1)
        rows.append(sorted(items))
        matrix[i, list(items)] = True
    df = pl.DataFrame({"items": rows}, schema={"items": pl.List(pl.Int64)})
    return df, matrix


def _brute_force(matrix, min_support=MIN_SUPPORT, max_k=MAX_K):
    """(complete frequent lattice, free-sets) as {itemset: count} / {itemset}."""
    n_rows, n_cols = matrix.shape
    min_count = int(np.ceil(min_support * n_rows))
    counts = {}
    for k in range(1, max_k + 1):
        for c in itertools.combinations(range(n_cols), k):
            count = int(np.logical_and.reduce(matrix[:, list(c)], axis=1).sum())
            if count >= min_count:
                counts[c] = count
    free = {
        c
        for c, v in counts.items()
        if v != n_rows and not any(counts.get(s) == v for r in range(1, len(c)) for s in itertools.combinations(c, r))
    }
    return counts, free


def _mined(df: pl.DataFrame, **kwargs) -> dict[tuple[int, ...], int]:
    result = apriori(df, min_support=MIN_SUPPORT, max_length=MAX_K, item_col="items", **kwargs)
    return {
        tuple(sorted(int(i) for i in row["itemset"])): round(float(row["support"]) * df.height)
        for row in result.iter_rows(named=True)
    }


@pytest.fixture
def prune_calls(monkeypatch) -> list[tuple[int, int, int]]:
    """Spy on the row-split miner's subset test: (k, candidates before, after)."""
    from et_miner.gpu import row_split

    real = row_split._prune_groups_apriori
    calls: list[tuple[int, int, int]] = []

    def spy(groups_info, prev_frequent_set, k, prev_flat_np=None):
        before = int(groups_info.total_candidates)
        out = real(groups_info, prev_frequent_set, k, prev_flat_np=prev_flat_np)
        calls.append((k, before, int(out.total_candidates) if out is not None else 0))
        return out

    monkeypatch.setattr(row_split, "_prune_groups_apriori", spy)
    return calls


def _assert_pruned_something(calls: list[tuple[int, int, int]]) -> None:
    assert calls, "the subset test never ran: the row-split miner mined with no downward closure"
    assert any(after < before for _, before, after in calls), (
        f"the subset test ran but removed nothing at any level: {calls}"
    )


# ── the contract, CPU-checkable ──────────────────────────────────────────────


def test_default_is_on_and_independent_of_the_free_set_gate():
    params = inspect.signature(apriori).parameters
    assert params["prune_apriori"].default is True
    assert params["prune_equal_support"].default is False


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    return _fixture()[0]


class TestRefusedOffTheRowSplitMiner:
    """Validation runs above the routing, so none of these touches a device."""

    MATCH = "prune_apriori=False requires the row-split miner"

    def test_cpu_route(self, df):
        with pytest.raises(ValueError, match=self.MATCH):
            apriori(df, min_support=MIN_SUPPORT, prune_apriori=False)

    def test_streaming_route(self, df):
        with pytest.raises(ValueError, match=self.MATCH):
            apriori(df, min_support=MIN_SUPPORT, streaming=True, chunk_size=1000, prune_apriori=False)

    def test_single_gpu_bitvec_route(self, df):
        with pytest.raises(ValueError, match=self.MATCH):
            apriori(df, min_support=MIN_SUPPORT, use_gpu=True, n_gpus=1, prune_apriori=False)

    def test_gpu_resident_route(self, df):
        with pytest.raises(ValueError, match=self.MATCH):
            apriori(df, min_support=MIN_SUPPORT, use_gpu=True, gpu_resident=True, prune_apriori=False)

    def test_explicit_true_is_accepted_on_the_cpu_route(self, df):
        """The default, passed explicitly, must not be over-restricted."""
        counts, _ = _brute_force(_fixture()[1])
        assert _mined(df, prune_apriori=True) == counts


# ── the row-split miner, where the switch lives ───────────────────────────────


def _needs_two_gpus() -> None:
    if _gpu_count() < 2:
        pytest.skip("needs 2 CUDA devices: the complete-lattice route to the row-split miner is n_gpus>1")


@pytest.mark.gpu
@pytest.mark.multigpu
def test_complete_lattice_is_exact_and_pruned_by_default(prune_calls):
    _needs_two_gpus()
    df, matrix = _fixture()
    counts, _ = _brute_force(matrix)
    assert _mined(df, use_gpu=True, n_gpus=2) == counts
    _assert_pruned_something(prune_calls)


@pytest.mark.gpu
@pytest.mark.multigpu
def test_prune_apriori_off_is_exact_and_never_runs_the_subset_test(prune_calls):
    """The pre-fix shape of a complete-lattice row-split run, now opt-in: same
    itemsets, same counts, no subset test."""
    _needs_two_gpus()
    df, matrix = _fixture()
    counts, _ = _brute_force(matrix)
    assert _mined(df, use_gpu=True, n_gpus=2, prune_apriori=False) == counts
    assert prune_calls == []


@pytest.mark.gpu
@pytest.mark.multigpu
def test_free_set_run_still_prunes_by_default(prune_calls):
    """Every existing prune_equal_support=True caller resolved prune_apriori to
    True before the split; the default keeps that run byte-identical."""
    _needs_two_gpus()
    df, matrix = _fixture()
    counts, free = _brute_force(matrix)
    mined = _mined(df, use_gpu=True, n_gpus=2, prune_equal_support=True)
    assert set(mined) == free
    assert all(mined[c] == counts[c] for c in mined)
    _assert_pruned_something(prune_calls)


@pytest.mark.gpu
@pytest.mark.multigpu
def test_free_set_run_with_the_subset_test_off_is_exact(prune_calls):
    _needs_two_gpus()
    df, matrix = _fixture()
    counts, free = _brute_force(matrix)
    mined = _mined(df, use_gpu=True, n_gpus=2, prune_equal_support=True, prune_apriori=False)
    assert set(mined) == free
    assert all(mined[c] == counts[c] for c in mined)
    assert prune_calls == []
