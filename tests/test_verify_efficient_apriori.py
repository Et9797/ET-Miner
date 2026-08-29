"""Correctness verification: et-miner GPU pipeline vs efficient-apriori ground truth.

Validates that the fused k=2 CUDA kernel + standard k>=3 path produce
identical results to the reference efficient-apriori implementation. The
whole module requires a CUDA device (``pytest.mark.gpu``): every case runs
apriori(use_gpu=True). CPU-path equivalence is covered by
tests/test_integration.py.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

ea_apriori = pytest.importorskip("efficient_apriori", reason="efficient_apriori not installed").apriori
from et_miner import apriori as pa_apriori

pytestmark = pytest.mark.gpu

REAL_DATASET = Path(__file__).resolve().parent.parent / "datasets" / "online_retail_ii" / "transactions.parquet"


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def ea_to_set(itemsets_dict: dict, n_trans: int) -> set[tuple]:
    """Convert efficient-apriori output to a comparable set of (sorted_tuple, support).

    efficient-apriori returns {k: {frozenset: absolute_count}}.
    """
    result = set()
    for k, itemsets in itemsets_dict.items():
        if k < 1:
            continue
        for fset, count in itemsets.items():
            key = tuple(sorted(fset))
            support = round(count / n_trans, 6)
            result.add((key, support))
    return result


def pa_to_set(result_df: pl.DataFrame) -> set[tuple]:
    """Convert et-miner output DataFrame to a comparable set of (sorted_tuple, support)."""
    result = set()
    itemsets = result_df.get_column("itemset").to_list()
    supports = result_df.get_column("support").to_list()
    for itemset, support in zip(itemsets, supports):
        result.add((tuple(sorted(itemset)), round(support, 6)))
    return result


def assert_results_match(ea_set: set[tuple], pa_set: set[tuple], tol: float = 0.001, label: str = "") -> None:
    """Assert both implementations found the same itemsets with matching supports."""
    ea_keys = {item[0] for item in ea_set}
    pa_keys = {item[0] for item in pa_set}

    missing_from_pa = ea_keys - pa_keys
    extra_in_pa = pa_keys - ea_keys
    assert not missing_from_pa, f"{label}: missing from et-miner: {sorted(missing_from_pa)[:10]}"
    assert not extra_in_pa, f"{label}: extra in et-miner: {sorted(extra_in_pa)[:10]}"

    ea_lookup = dict(ea_set)
    pa_lookup = dict(pa_set)
    mismatches = [
        (key, ea_lookup[key], pa_lookup[key])
        for key in ea_keys & pa_keys
        if abs(ea_lookup[key] - pa_lookup[key]) > tol
    ]
    assert not mismatches, f"{label}: support mismatches: {mismatches[:10]}"


def itemsets_by_k(result_set: set[tuple]) -> dict[int, set[tuple]]:
    """Group a result set by itemset size k."""
    groups: dict[int, set[tuple]] = {}
    for key, sup in result_set:
        groups.setdefault(len(key), set()).add((key, sup))
    return groups


def _generate_synthetic_data(
    n_transactions: int = 100_000,
    n_items: int = 500,
    avg_items_per_txn: int = 10,
    seed: int = 42,
) -> list[list[int]]:
    """Generate reproducible synthetic transaction data."""
    rng = np.random.RandomState(seed)
    transactions = []
    for _ in range(n_transactions):
        n_items_in_txn = min(max(1, rng.poisson(avg_items_per_txn)), n_items)
        transactions.append(sorted(rng.choice(n_items, size=n_items_in_txn, replace=False).tolist()))
    return transactions


def _load_dataset() -> tuple[list[list[int]], pl.DataFrame]:
    """Real dataset when generated (see datasets/prepare_online_retail.py), else synthetic."""
    if REAL_DATASET.exists():
        df = pl.read_parquet(REAL_DATASET)
        if "items" in df.columns:
            return df.get_column("items").to_list(), df
    transactions = _generate_synthetic_data()
    return transactions, pl.DataFrame({"items": transactions})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("min_support", [0.3, 0.4])
def test_small_dataset(min_support):
    """Exact per-k match on a tiny hand-verifiable dataset."""
    transactions_raw = [[1, 2, 3], [2, 3, 4], [1, 3, 5], [2, 3], [1, 2, 3, 4]]
    n_trans = len(transactions_raw)

    ea_itemsets, _ = ea_apriori([tuple(t) for t in transactions_raw], min_support=min_support)
    ea_by_k = itemsets_by_k(ea_to_set(ea_itemsets, n_trans))

    pa_result = pa_apriori(pl.DataFrame({"items": transactions_raw}), min_support=min_support, use_gpu=True)
    pa_by_k = itemsets_by_k(pa_to_set(pa_result))

    for k in sorted(set(ea_by_k) | set(pa_by_k)):
        assert_results_match(
            ea_by_k.get(k, set()), pa_by_k.get(k, set()), label=f"s={min_support} k={k}"
        )


@pytest.mark.slow
@pytest.mark.parametrize("min_support", [0.01, 0.005])
def test_large_dataset(min_support):
    """Full-result match on the real dataset (or the synthetic fallback)."""
    transactions_raw, pa_df = _load_dataset()
    n_trans = len(transactions_raw)

    ea_itemsets, _ = ea_apriori([tuple(t) for t in transactions_raw], min_support=min_support)
    ea_set = ea_to_set(ea_itemsets, n_trans)

    pa_result = pa_apriori(pa_df, min_support=min_support, use_gpu=True)
    pa_set = pa_to_set(pa_result)

    assert_results_match(ea_set, pa_set, label=f"s={min_support}")


def test_k3_validation():
    """k=3 itemsets specifically must match (denser dataset guarantees some exist)."""
    rng = np.random.RandomState(123)
    item_pool = list(range(1, 21))
    dense_txns = []
    for _ in range(1000):
        size = min(max(3, rng.poisson(5)), len(item_pool))
        dense_txns.append(sorted(rng.choice(item_pool, size=size, replace=False).tolist()))

    n_trans = len(dense_txns)
    min_support = 0.05

    ea_itemsets, _ = ea_apriori([tuple(t) for t in dense_txns], min_support=min_support)
    ea_by_k = itemsets_by_k(ea_to_set(ea_itemsets, n_trans))

    pa_result = pa_apriori(pl.DataFrame({"items": dense_txns}), min_support=min_support, use_gpu=True)
    pa_by_k = itemsets_by_k(pa_to_set(pa_result))

    assert ea_by_k.get(3), "expected ground-truth k=3 itemsets from the dense dataset"
    for k in [1, 2, 3]:
        assert_results_match(ea_by_k.get(k, set()), pa_by_k.get(k, set()), label=f"k={k}")
