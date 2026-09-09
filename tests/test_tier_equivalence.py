"""The mandated cross-tier correctness gate (see CLAUDE.md).

Every smoke/validation run asserts, on the ``smoke`` synthetic preset:

    Tier 1 Polars == Tier 2 Rust (sparse=True)
        == single-GPU legacy == multi-GPU legacy == shared multi-GPU
        == single-GPU sparse CSR (sparse_from_k=3) == multi-GPU sparse CSR
        == efficient-apriori (the canonical oracle)

Comparisons are exact on itemsets AND absolute counts — never weakened to
count-only or tolerance checks. Oracle boundary handling: the miner keeps
``count >= ceil(min_support * N)`` while efficient-apriori keeps
``support >= min_support`` in float, so the oracle is called with
``min_support = (min_count - 0.5) / N`` (exact for integer counts) and an
explicit ``max_length`` (efficient-apriori silently defaults to 8).

CPU-tier assertions run everywhere (they gate CI); GPU-tier assertions are
gpu-marked and self-skip below the needed device count.
"""

import inspect
from itertools import chain, combinations

import numpy as np
import pytest

from et_miner.core.apriori import apriori
from et_miner.synthetic import PRESETS, generate_transactions

ea_apriori = pytest.importorskip("efficient_apriori", reason="efficient_apriori not installed").apriori

SPEC = PRESETS["smoke"]
#: Explicit — efficient-apriori defaults to max_length=8 and would silently
#: truncate deeper itemsets from the oracle.
ORACLE_MAX_LENGTH = 32

CountedSet = set[tuple[tuple[int, ...], int]]


def _result_to_counted_set(result_df) -> CountedSet:
    """et-miner result DataFrame → {(sorted itemset, absolute count)}."""
    out: CountedSet = set()
    for itemset, sup in zip(result_df["itemset"].to_list(), result_df["support"].to_list()):
        out.add((tuple(sorted(int(i) for i in itemset)), round(sup * SPEC.n_rows)))
    return out


def _assert_ascending(result_df, label: str) -> None:
    """Every emitted itemset must be an ASCENDING tuple of item ids.

    This is the property _result_to_counted_set launders away: it canonicalises
    with sorted() before comparing, so a tier emitting [10, 2] compares equal to
    one emitting [2, 10] and the gate passes 9/9 either way. The equivalence
    assertions above are deliberately left alone (CLAUDE.md forbids weakening
    them, and the canonicalisation also insulates the gate from efficient-apriori
    emitting in an undocumented order); this is the ADDITIVE half that makes the
    producer's own contract checkable.

    It is also the only in-code guard on the row-split route, which is the one
    route that already got the order right -- and therefore the one most worth
    protecting from regression.
    """
    bad = [list(x) for x in result_df["itemset"].to_list() if list(x) != sorted(x)]
    assert not bad, f"{label}: {len(bad)} itemsets not in ascending item-id order, e.g. {bad[:5]}"


def _counted(result_df, label: str) -> CountedSet:
    """Assert the emission-order contract, then reduce to the counted set."""
    _assert_ascending(result_df, label)
    return _result_to_counted_set(result_df)


def _assert_counted_sets_equal(got: CountedSet, expected: CountedSet, label: str) -> None:
    got_keys = {k for k, _ in got}
    exp_keys = {k for k, _ in expected}
    missing = exp_keys - got_keys
    extra = got_keys - exp_keys
    assert not missing, f"{label}: missing itemsets: {sorted(missing)[:10]}"
    assert not extra, f"{label}: extra itemsets: {sorted(extra)[:10]}"
    got_counts = dict(got)
    exp_counts = dict(expected)
    mismatches = [(k, exp_counts[k], got_counts[k]) for k in exp_keys if got_counts[k] != exp_counts[k]]
    assert not mismatches, f"{label}: count mismatches (itemset, expected, got): {mismatches[:10]}"


@pytest.fixture(scope="session")
def smoke_dataset():
    df, data = generate_transactions(SPEC)
    return df, data


@pytest.fixture(scope="session")
def oracle_set(smoke_dataset) -> CountedSet:
    """efficient-apriori mined once per session, boundary-safe."""
    _, data = smoke_dataset
    rows = np.split(data.indices, data.indptr[1:-1])
    transactions = [tuple(int(x) for x in r) for r in rows]
    ea_support = (SPEC.min_count - 0.5) / SPEC.n_rows
    itemsets, _ = ea_apriori(
        transactions, min_support=ea_support, min_confidence=1.0, max_length=ORACLE_MAX_LENGTH
    )
    out: CountedSet = set()
    for _k, sets_of_k in itemsets.items():
        for fset, count in sets_of_k.items():
            out.add((tuple(sorted(int(i) for i in fset)), int(count)))
    return out


@pytest.fixture(scope="session")
def tier1_set(smoke_dataset) -> CountedSet:
    df, _ = smoke_dataset
    return _counted(apriori(df, min_support=SPEC.min_support, item_col="items"), "Tier 1 Polars")


def _gpu_count() -> int:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


# ── CPU half of the chain (runs everywhere, gates CI) ──────────────────────


def test_tier1_polars_matches_oracle(tier1_set, oracle_set):
    _assert_counted_sets_equal(tier1_set, oracle_set, "Tier 1 Polars vs efficient-apriori")


def test_tier2_rust_matches_oracle(smoke_dataset, oracle_set):
    df, _ = smoke_dataset
    got = _counted(apriori(df, min_support=SPEC.min_support, item_col="items", sparse=True), "Tier 2 Rust/sparse")
    _assert_counted_sets_equal(got, oracle_set, "Tier 2 Rust/sparse vs efficient-apriori")


def test_planted_motifs_recovered(tier1_set, smoke_dataset):
    """Oracle-independent ground truth: every planted motif and every subset
    must be mined with count >= the planted count (>=, never == — background
    noise can only add occurrences)."""
    _, data = smoke_dataset
    counts = dict(tier1_set)
    for motif, planted_count in data.planted:
        subsets = chain.from_iterable(combinations(motif, r) for r in range(1, len(motif) + 1))
        for sub in subsets:
            got = counts.get(tuple(sub))
            assert got is not None, f"planted subset {sub} of motif {motif} not mined"
            assert got >= planted_count, f"subset {sub}: mined {got} < planted {planted_count}"


def test_fpgrowth_second_oracle_agrees(smoke_dataset):
    """Optional independent cross-check of the oracle itself: mlxtend
    fpgrowth on a subsample must agree with efficient-apriori exactly."""
    fpgrowth_mod = pytest.importorskip("mlxtend.frequent_patterns", reason="mlxtend not installed")
    import pandas as pd

    _, data = smoke_dataset
    n_sub = 10_000
    sub_indptr = data.indptr[: n_sub + 1]
    sub_items = data.indices[: sub_indptr[-1]]

    onehot = np.zeros((n_sub, data.n_cols), dtype=bool)
    onehot[np.repeat(np.arange(n_sub), np.diff(sub_indptr)), sub_items] = True
    min_count = SPEC.min_count  # same absolute count on the subsample
    fp = fpgrowth_mod.fpgrowth(
        pd.DataFrame(onehot), min_support=(min_count - 0.5) / n_sub, use_colnames=True
    )
    fp_set = {
        (tuple(sorted(int(i) for i in row)), round(sup * n_sub))
        for row, sup in zip(fp["itemsets"], fp["support"])
    }

    rows = np.split(sub_items, sub_indptr[1:-1])
    transactions = [tuple(int(x) for x in r) for r in rows]
    itemsets, _ = ea_apriori(
        transactions, min_support=(min_count - 0.5) / n_sub, min_confidence=1.0, max_length=ORACLE_MAX_LENGTH
    )
    ea_set = {
        (tuple(sorted(int(i) for i in fs)), int(c)) for d in itemsets.values() for fs, c in d.items()
    }
    _assert_counted_sets_equal(fp_set, ea_set, "fpgrowth vs efficient-apriori (subsample)")


# ── GPU half of the chain (box campaign; auto-skipped without devices) ─────


@pytest.mark.gpu
def test_single_gpu_legacy_matches_oracle(smoke_dataset, oracle_set):
    df, _ = smoke_dataset
    got = _counted(apriori(df, min_support=SPEC.min_support, item_col="items", use_gpu=True), "single-GPU legacy")
    _assert_counted_sets_equal(got, oracle_set, "single-GPU legacy vs efficient-apriori")


@pytest.mark.gpu
def test_multi_gpu_legacy_matches_oracle(smoke_dataset, oracle_set):
    if _gpu_count() < 2:
        pytest.skip("needs 2 CUDA devices")
    df, _ = smoke_dataset
    got = _counted(
        apriori(df, min_support=SPEC.min_support, item_col="items", use_gpu=True, n_gpus=2),
        "multi-GPU legacy",
    )
    _assert_counted_sets_equal(got, oracle_set, "multi-GPU legacy vs efficient-apriori")


@pytest.mark.gpu
def test_multi_gpu_shared_matches_oracle(smoke_dataset, oracle_set, monkeypatch):
    if _gpu_count() < 2:
        pytest.skip("needs 2 CUDA devices")
    from et_miner.gpu.kernels import count_k3plus_allcounts

    if "variant" not in inspect.signature(count_k3plus_allcounts).parameters:
        pytest.skip("shared kernel variant not wired yet")
    monkeypatch.setenv("ET_MINER_KERNEL_VARIANT", "shared")
    df, _ = smoke_dataset
    got = _counted(
        apriori(df, min_support=SPEC.min_support, item_col="items", use_gpu=True, n_gpus=2),
        "shared multi-GPU",
    )
    _assert_counted_sets_equal(got, oracle_set, "shared multi-GPU vs efficient-apriori")


@pytest.mark.gpu
def test_single_gpu_sparse_matches_oracle(smoke_dataset, oracle_set):
    """sparse_from_k=3 forces the dense→sparse CSR transition on the dense-shaped
    smoke preset, so every level from K=3 runs on the GPU-resident shard."""
    df, _ = smoke_dataset
    got = _counted(
        apriori(df, min_support=SPEC.min_support, item_col="items", use_gpu=True, sparse_from_k=3),
        "single-GPU sparse CSR (sparse_from_k=3)",
    )
    _assert_counted_sets_equal(got, oracle_set, "single-GPU sparse CSR (sparse_from_k=3) vs efficient-apriori")


@pytest.mark.gpu
def test_multi_gpu_sparse_matches_oracle(smoke_dataset, oracle_set):
    if _gpu_count() < 2:
        pytest.skip("needs 2 CUDA devices")
    df, _ = smoke_dataset
    got = _counted(
        apriori(df, min_support=SPEC.min_support, item_col="items", use_gpu=True, n_gpus=2, sparse_from_k=3),
        "multi-GPU sparse CSR (sparse_from_k=3)",
    )
    _assert_counted_sets_equal(got, oracle_set, "multi-GPU sparse CSR (sparse_from_k=3) vs efficient-apriori")
