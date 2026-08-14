"""Integration tests for et-miner."""

import polars as pl
import pytest

from et_miner import apriori, generate_rules


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
        pytest.importorskip("efficient_apriori")
        from efficient_apriori import apriori as ea_apriori

        min_support = 0.4

        # Run efficient-apriori
        ea_itemsets, _ = ea_apriori(comparison_transactions["tuples"], min_support)
        ea_count = sum(len(v) for v in ea_itemsets.values())

        # Run et-miner
        pa_result = apriori(comparison_transactions["polars"], min_support)
        pa_count = pa_result.height

        assert pa_count == ea_count, f"Mismatch: EA={ea_count}, PA={pa_count}"

    def test_itemsets_match(self, comparison_transactions):
        """Test that et-miner finds exactly the same itemsets."""
        pytest.importorskip("efficient_apriori")
        from efficient_apriori import apriori as ea_apriori

        min_support = 0.4

        # Run efficient-apriori
        ea_itemsets, _ = ea_apriori(comparison_transactions["tuples"], min_support)
        ea_set = set()
        for size_dict in ea_itemsets.values():
            for itemset in size_dict.keys():
                ea_set.add(tuple(sorted(itemset)))

        # Run et-miner
        pa_result = apriori(comparison_transactions["polars"], min_support)
        pa_set = {tuple(sorted(row["itemset"])) for row in pa_result.iter_rows(named=True)}

        assert pa_set == ea_set, f"Itemsets differ:\nEA only: {ea_set - pa_set}\nPA only: {pa_set - ea_set}"

    def test_supports_match(self, comparison_transactions):
        """Test that support values match between implementations."""
        pytest.importorskip("efficient_apriori")
        from efficient_apriori import apriori as ea_apriori

        min_support = 0.4
        n_transactions = len(comparison_transactions["tuples"])

        # Run efficient-apriori (returns absolute counts, not fractions)
        ea_itemsets, _ = ea_apriori(comparison_transactions["tuples"], min_support)
        ea_supports = {}
        for size_dict in ea_itemsets.values():
            for itemset, count in size_dict.items():
                # Convert count to fraction for comparison
                ea_supports[tuple(sorted(itemset))] = count / n_transactions

        # Run et-miner (returns fractions)
        pa_result = apriori(comparison_transactions["polars"], min_support)
        pa_supports = {
            tuple(sorted(row["itemset"])): row["support"]
            for row in pa_result.iter_rows(named=True)
        }

        for itemset in ea_supports:
            assert itemset in pa_supports, f"Missing itemset: {itemset}"
            assert abs(ea_supports[itemset] - pa_supports[itemset]) < 0.001, \
                f"Support mismatch for {itemset}: EA={ea_supports[itemset]}, PA={pa_supports[itemset]}"
