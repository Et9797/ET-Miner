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

ea = pytest.importorskip("efficient_apriori", reason="efficient_apriori not installed")
from et_miner import apriori as pa_apriori
from et_miner.core.result import _min_count

pytestmark = pytest.mark.gpu

REAL_DATASET = Path(__file__).resolve().parent.parent / "datasets" / "online_retail_ii" / "transactions.parquet"


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def ea_counts(transactions: list, min_support: float) -> dict[tuple, int]:
    """efficient-apriori's {sorted itemset: count} under the oracle convention of CLAUDE.md.

    ``(min_count - 0.5) / N`` makes its float threshold keep exactly the counts
    the miner keeps (``count >= min_count``), and ``max_length`` is the longest
    transaction, since efficient-apriori otherwise stops at 8.
    """
    tx = [tuple(t) for t in transactions]
    n = len(tx)
    max_length = max(len(set(t)) for t in tx)
    itemsets, _ = ea.itemsets_from_transactions(tx, (_min_count(min_support, n) - 0.5) / n, max_length=max_length)
    return {tuple(sorted(k)): v for level in itemsets.values() for k, v in level.items()}


def pa_counts(result_df: pl.DataFrame, n_trans: int) -> dict[tuple, int]:
    """et-miner's {sorted itemset: count}."""
    itemsets = result_df.get_column("itemset").to_list()
    supports = result_df.get_column("support").to_list()
    return {tuple(sorted(s)): round(p * n_trans) for s, p in zip(itemsets, supports)}


def assert_results_match(ea_counts: dict[tuple, int], pa_counts: dict[tuple, int], label: str = "") -> None:
    """Assert both implementations found the same itemsets with the same absolute counts."""
    missing_from_pa = ea_counts.keys() - pa_counts.keys()
    extra_in_pa = pa_counts.keys() - ea_counts.keys()
    assert not missing_from_pa, f"{label}: missing from et-miner: {sorted(missing_from_pa)[:10]}"
    assert not extra_in_pa, f"{label}: extra in et-miner: {sorted(extra_in_pa)[:10]}"
    mismatches = [(key, c, pa_counts[key]) for key, c in ea_counts.items() if pa_counts[key] != c]
    assert not mismatches, f"{label}: count mismatches: {mismatches[:10]}"


def itemsets_by_k(counts: dict[tuple, int]) -> dict[int, dict[tuple, int]]:
    """Group a result by itemset size k."""
    groups: dict[int, dict[tuple, int]] = {}
    for key, c in counts.items():
        groups.setdefault(len(key), {})[key] = c
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

    ea_by_k = itemsets_by_k(ea_counts(transactions_raw, min_support))

    pa_result = pa_apriori(pl.DataFrame({"items": transactions_raw}), min_support=min_support, use_gpu=True)
    pa_by_k = itemsets_by_k(pa_counts(pa_result, n_trans))

    for k in sorted(set(ea_by_k) | set(pa_by_k)):
        assert_results_match(ea_by_k.get(k, {}), pa_by_k.get(k, {}), label=f"s={min_support} k={k}")


@pytest.mark.slow
@pytest.mark.parametrize("min_support", [0.01, 0.005])
def test_large_dataset(min_support):
    """Full-result match on the real dataset (or the synthetic fallback)."""
    transactions_raw, pa_df = _load_dataset()
    n_trans = len(transactions_raw)

    pa_result = pa_apriori(pa_df, min_support=min_support, use_gpu=True)

    assert_results_match(
        ea_counts(transactions_raw, min_support), pa_counts(pa_result, n_trans), label=f"s={min_support}"
    )


def test_k3_validation():
    """k=3 itemsets specifically must match (denser dataset guarantees some exist)."""
    rng = np.random.RandomState(123)
    item_pool = list(range(1, 21))
    dense_txns = []
    for _ in range(1000):
        size = min(max(3, rng.poisson(5)), len(item_pool))
        dense_txns.append(sorted(rng.choice(item_pool, size=size, replace=False).tolist()))

    n_trans = len(dense_txns)
    # 0.02, not 0.05: at 0.05 this dataset provably has ZERO k=3 itemsets
    # (verified with efficient-apriori directly), so the ground-truth assert
    # below can never hold — a latent bug exposed the first time this
    # gpu-marked test ran on a real device. At 0.02 there are ~459.
    min_support = 0.02

    ea_by_k = itemsets_by_k(ea_counts(dense_txns, min_support))

    pa_result = pa_apriori(pl.DataFrame({"items": dense_txns}), min_support=min_support, use_gpu=True)
    pa_by_k = itemsets_by_k(pa_counts(pa_result, n_trans))

    assert ea_by_k.get(3), "expected ground-truth k=3 itemsets from the dense dataset"
    for k in [1, 2, 3]:
        assert_results_match(ea_by_k.get(k, {}), pa_by_k.get(k, {}), label=f"k={k}")
