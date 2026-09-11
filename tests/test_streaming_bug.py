"""Test for Polars streaming engine bug with list.contains().

This module tests that the boolean matrix construction produces accurate
True-counts. The streaming engine has a known bug where list.contains()
undercounts True values by approximately 2%, which causes items near the
support threshold to be incorrectly filtered out.

Regression test for: https://github.com/pola-rs/polars/issues/XXXXX
"""
import polars as pl

from et_miner import apriori, apriori_streaming
from et_miner.core.matrix import build_boolean_matrix, count_support_batched


class TestStreamingBugRegression:
    """Tests to catch the Polars streaming engine list.contains() bug."""

    def test_matrix_column_counts_match_freq1_counts(self):
        """Verify boolean matrix True-counts match original frequent item counts.

        This test catches the streaming engine bug where list.contains() undercounts.
        We create items with exact known counts and verify the matrix reflects them.
        """
        # Create dataset with items near support threshold
        # Item 999 appears in exactly 50 transactions (5% support)
        # Items 1, 2 appear in all 1000 transactions
        # Item 3 appears in 950 transactions
        transactions = pl.LazyFrame({
            "items": [[999, 1, 2]] * 50 + [[1, 2, 3]] * 950
        })

        # Build matrix at 4.9% threshold (item 999 should be frequent with 5%)
        matrix, col_to_item, n_tx = build_boolean_matrix(transactions, min_support=0.049)

        # Verify transaction count
        assert n_tx == 1000, f"Expected 1000 transactions, got {n_tx}"

        # Item 999 should be in the matrix (5% support >= 4.9% threshold)
        assert 999 in col_to_item.values(), "Item 999 should be frequent at 4.9% threshold"

        # Find the column for item 999
        col_999 = [k for k, v in col_to_item.items() if v == 999][0]

        # Matrix column sum should match actual count (50)
        actual_count = matrix.get_column(col_999).sum()
        assert actual_count == 50, f"Expected 50 True values for item 999, got {actual_count}"

    def test_all_items_counts_are_exact(self):
        """Verify all item counts in matrix match expected values exactly."""
        # Create transactions with known exact counts
        # Item A: 100 occurrences (10%)
        # Item B: 200 occurrences (20%)
        # Item C: 500 occurrences (50%)
        transactions = pl.LazyFrame({
            "items": (
                [["A", "B", "C"]] * 100  # A=100, B=100, C=100
                + [["B", "C"]] * 100      # B=200, C=200
                + [["C"]] * 300           # C=500
                + [[]] * 500              # Empty transactions
            )
        })

        matrix, col_to_item, n_tx = build_boolean_matrix(transactions, min_support=0.05)

        assert n_tx == 1000

        # All items should be frequent (A=10%, B=20%, C=50%, all >= 5%)
        item_to_col = {v: k for k, v in col_to_item.items()}

        # Verify exact counts
        assert matrix.get_column(item_to_col["A"]).sum() == 100, "Item A should have exactly 100 True values"
        assert matrix.get_column(item_to_col["B"]).sum() == 200, "Item B should have exactly 200 True values"
        assert matrix.get_column(item_to_col["C"]).sum() == 500, "Item C should have exactly 500 True values"

    def test_borderline_items_not_lost(self):
        """Test that items exactly at the support threshold are not lost.

        The streaming bug causes items exactly at threshold to fall below
        due to undercounting, making them disappear from results.
        """
        # Item at exactly 5% support (50 out of 1000)
        transactions = pl.LazyFrame({
            "items": [["rare"]] * 50 + [["common"]] * 950
        })

        # Threshold at exactly 5%
        matrix, col_to_item, n_tx = build_boolean_matrix(transactions, min_support=0.05)

        # Both items should be frequent
        assert "rare" in col_to_item.values(), "Item 'rare' at exactly 5% should be frequent"
        assert "common" in col_to_item.values(), "Item 'common' at 95% should be frequent"

        # Verify counts are exact
        item_to_col = {v: k for k, v in col_to_item.items()}
        assert matrix.get_column(item_to_col["rare"]).sum() == 50
        assert matrix.get_column(item_to_col["common"]).sum() == 950

    def test_large_scale_counts_accurate(self):
        """Test that counts remain accurate at larger scale.

        The ~2% undercount becomes more significant at scale, potentially
        causing dozens of items to be incorrectly filtered out.
        """
        import random
        random.seed(42)

        # Create 10,000 transactions with various items
        n_transactions = 10_000
        data = []
        for i in range(n_transactions):
            items = []
            # Item 'always' in every transaction
            items.append("always")
            # Item 'half' in 50% of transactions
            if i % 2 == 0:
                items.append("half")
            # Item 'threshold' in exactly 500 transactions (5%)
            if i < 500:
                items.append("threshold")
            data.append(items)

        transactions = pl.LazyFrame({"items": data})
        matrix, col_to_item, n_tx = build_boolean_matrix(transactions, min_support=0.049)

        item_to_col = {v: k for k, v in col_to_item.items()}

        # Verify exact counts at scale
        assert matrix.get_column(item_to_col["always"]).sum() == 10_000, "Item 'always' should have 10,000 True values"
        assert matrix.get_column(item_to_col["half"]).sum() == 5_000, "Item 'half' should have 5,000 True values"
        assert matrix.get_column(item_to_col["threshold"]).sum() == 500, "Item 'threshold' should have 500 True values"


class TestMixedLengthFilterRegression:
    """The transaction-length filter must be exact for every itemset in a batch.

    count_support_batched auto-detected the filter as k=len(itemsets[0]).
    Whichever itemset happened to be first set the row filter for the whole
    call, so a longer itemset landing first dropped shorter transactions and
    undercounted the shorter itemsets — and a caller building the list from a
    set got a different answer per run (set-iteration order).

    It now uses the batch MINIMUM: a transaction shorter than k cannot contain
    any k-itemset, so filtering at the minimum is exact for every itemset in the
    batch.

    Historical note, because this test used to assert the opposite. Its body and
    its name pinned the undercount as expected behaviour (`assert
    buggy[("i_0",)] < fixed[("i_0",)]`) while its own summary line asked for the
    correct behaviour — and it was green, so anyone auditing the suite found a
    passing test that appeared to sanction the defect. Fixing the filter turned
    it red, and that was the fix working.
    """

    def test_length_filter_is_exact_on_a_mixed_length_batch(self):
        """A mixed-length batch with a long itemset first must not undercount.

        Six transactions: 'a' appears in 5, 'b' in 4, {a,b,c} in 2. With a
        3-itemset first, the old filter dropped every transaction shorter than
        3 items and returned 1 for 'a' — a 75% undercount.
        """
        df = pl.DataFrame(
            {"items": [["a"], ["a"], ["a", "b", "c"], ["a", "b", "c"], ["a", "b"], ["b", "c"]]}
        )
        matrix, _, _ = build_boolean_matrix(df.lazy(), min_support=0.0)
        # i_0=a, i_1=b, i_2=c; longest itemset first is what used to trigger k=3
        itemsets = [("i_0", "i_1", "i_2"), ("i_0",), ("i_1",)]

        filtered = count_support_batched(matrix, itemsets, 6, enable_length_filter=True)
        unfiltered = count_support_batched(matrix, itemsets, 6, enable_length_filter=False)

        # Ground truth: a in 5 transactions, b in 4, {a,b,c} in 2.
        assert unfiltered[("i_0",)] == 5
        assert unfiltered[("i_1",)] == 4
        assert unfiltered[("i_0", "i_1", "i_2")] == 2
        # The filter is an optimisation, so it must not change any answer.
        assert filtered == unfiltered

    def test_result_is_independent_of_batch_order(self):
        """The defect's sharpest edge: the same batch, permuted, counted
        differently — so a caller passing a set got a different answer per run."""
        df = pl.DataFrame(
            {"items": [["a"], ["a"], ["a", "b", "c"], ["a", "b", "c"], ["a", "b"], ["b", "c"]]}
        )
        matrix, _, _ = build_boolean_matrix(df.lazy(), min_support=0.0)
        long_first = [("i_0", "i_1", "i_2"), ("i_0",), ("i_1",)]
        short_first = [("i_0",), ("i_1",), ("i_0", "i_1", "i_2")]

        assert count_support_batched(matrix, long_first, 6) == count_support_batched(
            matrix, short_first, 6
        )

    def test_son_streaming_matches_direct_on_mixed_length_data(self):
        """SON (streaming) must reproduce direct apriori on mixed-length data."""
        df = pl.DataFrame(
            {"items": [["a"], ["a"], ["a", "b", "c"], ["a", "b", "c"], ["a", "b"], ["b", "c"]]}
        )
        direct = apriori(df, min_support=0.3, item_col="items").sort("itemset")
        son = apriori_streaming(
            df.lazy(), min_support=0.3, item_col="items", chunk_size=2
        ).sort("itemset")

        d = {tuple(r["itemset"]): round(r["support"], 6) for r in direct.to_dicts()}
        s = {tuple(r["itemset"]): round(r["support"], 6) for r in son.to_dicts()}
        assert d == s
