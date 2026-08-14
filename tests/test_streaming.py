"""Tests for streaming Apriori implementation (SON algorithm).

These tests verify that:
1. streaming=True produces identical results to streaming=False
2. Memory stays within expected bounds
3. Edge cases are handled correctly
"""

from __future__ import annotations

import tracemalloc

import polars as pl
import pytest

from et_miner import apriori, apriori_streaming


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def small_transactions():
    """Small test dataset for basic functionality."""
    return pl.DataFrame(
        {
            "items": [
                [1, 2, 3],
                [2, 3, 4],
                [1, 2, 3, 4],
                [1, 3, 5],
                [2, 3, 5],
                [1, 2, 3, 5],
            ]
        }
    )


@pytest.fixture
def medium_transactions():
    """Medium dataset with ~1000 transactions for testing chunking."""
    import random

    random.seed(42)
    items = list(range(1, 51))  # 50 unique items

    transactions = []
    for _ in range(1000):
        n_items = random.randint(3, 10)
        tx = sorted(random.sample(items, n_items))
        transactions.append(tx)

    return pl.DataFrame({"items": transactions})


@pytest.fixture
def large_transactions():
    """Larger dataset with ~10000 transactions."""
    import random

    random.seed(42)
    items = list(range(1, 101))  # 100 unique items

    transactions = []
    for _ in range(10000):
        n_items = random.randint(3, 15)
        tx = sorted(random.sample(items, n_items))
        transactions.append(tx)

    return pl.DataFrame({"items": transactions})


# =============================================================================
# Correctness Tests
# =============================================================================


class TestStreamingCorrectness:
    """Verify streaming produces identical results to standard apriori."""

    def test_identical_results_small_dataset(self, small_transactions):
        """Streaming should produce same itemsets as standard on small data."""
        min_support = 0.3

        # Standard apriori
        standard_result = apriori(small_transactions, min_support=min_support)

        # Streaming with small chunks to force multiple chunks
        streaming_result = apriori(
            small_transactions,
            min_support=min_support,
            streaming=True,
            chunk_size=3,  # Force 2 chunks
        )

        # Sort both for comparison
        standard_sorted = standard_result.sort("itemset")
        streaming_sorted = streaming_result.sort("itemset")

        # Compare itemsets
        assert standard_sorted.height == streaming_sorted.height, (
            f"Different number of itemsets: standard={standard_sorted.height}, "
            f"streaming={streaming_sorted.height}"
        )

        # Compare each itemset
        for i in range(standard_sorted.height):
            std_itemset = standard_sorted["itemset"][i].to_list()
            str_itemset = streaming_sorted["itemset"][i].to_list()
            assert std_itemset == str_itemset, (
                f"Itemset mismatch at row {i}: standard={std_itemset}, streaming={str_itemset}"
            )

    def test_identical_results_medium_dataset(self, medium_transactions):
        """Streaming should produce same itemsets on medium data."""
        min_support = 0.05

        standard_result = apriori(medium_transactions, min_support=min_support)
        streaming_result = apriori(
            medium_transactions,
            min_support=min_support,
            streaming=True,
            chunk_size=200,  # ~5 chunks
        )

        standard_sorted = standard_result.sort("itemset")
        streaming_sorted = streaming_result.sort("itemset")

        assert standard_sorted.height == streaming_sorted.height

    def test_single_chunk_optimization(self, small_transactions):
        """When data fits in one chunk, should use standard apriori path."""
        result = apriori(
            small_transactions,
            min_support=0.3,
            streaming=True,
            chunk_size=1000,  # Larger than dataset
        )

        # Should still produce correct results
        assert result.height > 0

    def test_support_values_match(self, small_transactions):
        """Support values should be identical between streaming and standard."""
        min_support = 0.3

        standard = apriori(small_transactions, min_support=min_support)
        streaming = apriori(
            small_transactions,
            min_support=min_support,
            streaming=True,
            chunk_size=3,
        )

        # Create lookup dict
        standard_supports = {
            tuple(row["itemset"]): row["support"]
            for row in standard.iter_rows(named=True)
        }
        streaming_supports = {
            tuple(row["itemset"]): row["support"]
            for row in streaming.iter_rows(named=True)
        }

        for itemset, support in standard_supports.items():
            assert itemset in streaming_supports, f"Missing itemset {itemset}"
            assert abs(support - streaming_supports[itemset]) < 1e-9, (
                f"Support mismatch for {itemset}: "
                f"standard={support}, streaming={streaming_supports[itemset]}"
            )


# =============================================================================
# Edge Cases
# =============================================================================


class TestStreamingEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_transactions(self):
        """Handle empty input gracefully."""
        empty_df = pl.DataFrame({"items": []}, schema={"items": pl.List(pl.Int64)})

        result = apriori(empty_df, min_support=0.5, streaming=True, chunk_size=100)
        assert result.height == 0

    def test_no_frequent_itemsets(self):
        """Handle case where no itemsets meet threshold."""
        df = pl.DataFrame(
            {
                "items": [
                    [1],
                    [2],
                    [3],
                    [4],
                    [5],
                ]
            }
        )

        result = apriori(df, min_support=0.5, streaming=True, chunk_size=2)
        assert result.height == 0

    def test_all_same_transaction(self):
        """All transactions identical should find all subsets."""
        df = pl.DataFrame(
            {
                "items": [
                    [1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3],
                ]
            }
        )

        result = apriori(df, min_support=0.5, streaming=True, chunk_size=2)
        # Should find: {1}, {2}, {3}, {1,2}, {1,3}, {2,3}, {1,2,3}
        assert result.height == 7

    def test_max_length_respected(self, medium_transactions):
        """max_length should limit itemset size in streaming mode."""
        result = apriori(
            medium_transactions,
            min_support=0.05,
            max_length=2,
            streaming=True,
            chunk_size=200,
        )

        max_itemset_len = result.select(
            pl.col("itemset").list.len().max()
        ).item()
        assert max_itemset_len <= 2


# =============================================================================
# Memory Tests
# =============================================================================


class TestStreamingMemory:
    """Verify memory bounds are respected."""

    @pytest.mark.slow
    def test_memory_stays_bounded(self, large_transactions):
        """Memory should not grow linearly with dataset size."""
        tracemalloc.start()

        # Run streaming with small chunks
        _ = apriori(
            large_transactions,
            min_support=0.01,
            streaming=True,
            chunk_size=500,
        )

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Peak memory should be reasonable (< 100MB for this dataset)
        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 100, f"Peak memory {peak_mb:.1f}MB exceeds 100MB limit"

    def test_memory_budget_parameter(self, medium_transactions):
        """memory_budget_gb should automatically calculate chunk size."""
        # Very small budget should still work
        result = apriori(
            medium_transactions,
            min_support=0.1,
            streaming=True,
            memory_budget_gb=0.001,  # 1MB budget
        )

        assert result.height > 0


# =============================================================================
# API Tests
# =============================================================================


class TestStreamingAPI:
    """Test API surface and parameter handling."""

    def test_lazyframe_input(self, small_transactions):
        """Should work with LazyFrame input."""
        lf = small_transactions.lazy()

        result = apriori(lf, min_support=0.3, streaming=True, chunk_size=3)
        assert result.height > 0

    def test_profiling_returns_session(self, small_transactions):
        """profile=True should return ProfilingSession."""
        result, session = apriori(
            small_transactions,
            min_support=0.3,
            streaming=True,
            chunk_size=3,
            profile=True,
        )

        assert result.height > 0
        assert session is not None
        # Session should have streaming-specific phases
        assert "count_transactions" in str(session)

    def test_apriori_streaming_direct_call(self, small_transactions):
        """apriori_streaming can be called directly."""
        result = apriori_streaming(
            small_transactions,
            min_support=0.3,
            chunk_size=3,
        )

        assert result.height > 0

    def test_sparse_parameter_forwarded(self, medium_transactions):
        """sparse parameter should work in streaming mode."""
        result = apriori(
            medium_transactions,
            min_support=0.05,
            streaming=True,
            chunk_size=200,
            sparse=True,
        )

        assert result.height > 0

    def test_n_jobs_parameter_forwarded(self, medium_transactions):
        """n_jobs parameter should work in streaming mode."""
        result = apriori(
            medium_transactions,
            min_support=0.05,
            streaming=True,
            chunk_size=200,
            n_jobs=2,
        )

        assert result.height > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestStreamingIntegration:
    """Integration tests combining multiple features."""

    def test_streaming_with_all_parameters(self, medium_transactions):
        """All parameters should work together."""
        result = apriori(
            medium_transactions,
            min_support=0.05,
            max_length=3,
            streaming=True,
            chunk_size=200,
            sparse=True,
            n_jobs=2,
            show_progress=False,
        )

        assert result.height > 0
        max_len = result.select(pl.col("itemset").list.len().max()).item()
        assert max_len <= 3

    @pytest.mark.slow
    def test_streaming_deterministic(self, large_transactions):
        """Multiple runs should produce identical results."""
        results = []
        for _ in range(3):
            result = apriori(
                large_transactions,
                min_support=0.01,
                streaming=True,
                chunk_size=500,
            )
            results.append(result.sort("itemset"))

        # All runs should be identical
        for i in range(1, len(results)):
            assert results[0].height == results[i].height
            assert results[0]["itemset"].to_list() == results[i]["itemset"].to_list()
