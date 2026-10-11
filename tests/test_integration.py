"""Integration tests for et-miner."""

import polars as pl
import pytest

from et_miner import apriori, generate_rules
from et_miner.core.result import _min_count


def _ea_counts(transactions: list[tuple], min_support: float) -> dict[tuple, int]:
    """efficient-apriori's {sorted itemset: count} under the oracle convention of CLAUDE.md.

    ``(min_count - 0.5) / N`` and the longest transaction as ``max_length``
    (efficient-apriori otherwise stops at 8).
    """
    ea = pytest.importorskip("efficient_apriori")
    n = len(transactions)
    max_length = max(len(set(t)) for t in transactions)
    itemsets, _ = ea.itemsets_from_transactions(
        transactions, (_min_count(min_support, n) - 0.5) / n, max_length=max_length
    )
    return {tuple(sorted(k)): v for level in itemsets.values() for k, v in level.items()}


class TestFullPipeline:
    """End-to-end pipeline tests."""

    def test_apriori_to_rules_pipeline(self, sample_transactions):
        """Test full pipeline: apriori -> generate_rules."""
        # Step 1: Find frequent itemsets
        itemsets = apriori(sample_transactions, min_support=0.5)

        assert itemsets.height > 0

        # Step 2: Generate rules
        rules = generate_rules(itemsets, min_confidence=0.5)

        assert isinstance(rules, list)
        # With 5 itemsets including pairs, should generate some rules
        assert len(rules) > 0

        # Verify rules reference items from our transactions
        all_items = {1, 2, 3, 4, 5}
        for rule in rules:
            assert all(item in all_items for item in rule.lhs)
            assert all(item in all_items for item in rule.rhs)

    def test_pipeline_with_large_dataset(self, large_transactions):
        """Test pipeline scales to larger datasets."""
        itemsets = apriori(large_transactions, min_support=0.1)

        assert itemsets.height > 0

        rules = generate_rules(itemsets, min_confidence=0.5)

        # Should complete without error
        assert isinstance(rules, list)


class TestMatchesEfficientApriori:
    """Tests comparing output with efficient-apriori."""

    @pytest.fixture
    def comparison_transactions(self):
        """Transactions in both formats for comparison."""
        items_list = [
            [1, 2, 3],
            [2, 3, 4],
            [1, 3, 5],
            [2, 3],
            [1, 2, 3, 4],
        ]
        return {
            "polars": pl.DataFrame({"items": items_list}),
            "tuples": [tuple(items) for items in items_list]
        }

    def test_itemset_counts_match(self, comparison_transactions):
        """Test that et-miner finds same number of itemsets."""
        min_support = 0.4

        # Run efficient-apriori
        ea_count = len(_ea_counts(comparison_transactions["tuples"], min_support))

        # Run et-miner
        pa_result = apriori(comparison_transactions["polars"], min_support)
        pa_count = pa_result.height

        assert pa_count == ea_count, f"Mismatch: EA={ea_count}, PA={pa_count}"

    def test_itemsets_match(self, comparison_transactions):
        """Test that et-miner finds exactly the same itemsets."""
        min_support = 0.4

        # Run efficient-apriori
        ea_set = set(_ea_counts(comparison_transactions["tuples"], min_support))

        # Run et-miner
        pa_result = apriori(comparison_transactions["polars"], min_support)
        pa_set = {tuple(sorted(row["itemset"])) for row in pa_result.iter_rows(named=True)}

        assert pa_set == ea_set, f"Itemsets differ:\nEA only: {ea_set - pa_set}\nPA only: {pa_set - ea_set}"

    def test_supports_match(self, comparison_transactions):
        """Test that absolute counts match between implementations."""
        min_support = 0.4
        n_transactions = len(comparison_transactions["tuples"])

        ea_counts = _ea_counts(comparison_transactions["tuples"], min_support)
        pa_result = apriori(comparison_transactions["polars"], min_support)
        pa_counts = {
            tuple(sorted(row["itemset"])): round(row["support"] * n_transactions)
            for row in pa_result.iter_rows(named=True)
        }

        assert pa_counts == ea_counts
