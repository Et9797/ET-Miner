"""Tests for the apriori() function."""

import time

import numpy as np
import polars as pl
import pytest

from et_miner import apriori
from et_miner.core.matrix import (
    build_boolean_matrix,
    count_support_batched,
    count_support_vectorized,
)


class TestAprioriBasic:
    """Basic functionality tests for apriori()."""

    def test_apriori_basic(self, sample_transactions):
        """Test basic apriori with standard parameters."""
        result = apriori(sample_transactions, min_support=0.5)

        # Should find items 1, 2, 3 and pairs [1,3], [2,3]
        assert result.height == 5

        # Check schema
        assert "itemset" in result.columns
        assert "support" in result.columns
        assert result.schema["itemset"] == pl.List(pl.Int64)
        assert result.schema["support"] == pl.Float64

        # Verify specific supports
        itemsets = {tuple(row["itemset"]): row["support"] for row in result.iter_rows(named=True)}
        assert (3,) in itemsets
        assert itemsets[(3,)] == 1.0  # Item 3 in all transactions
        assert (2, 3) in itemsets
        assert itemsets[(2, 3)] == 0.75

    def test_apriori_high_support_empty_result(self, sample_transactions):
        """Test that min_support=1.0 returns only items in all transactions."""
        result = apriori(sample_transactions, min_support=1.0)

        # Only item 3 appears in all 4 transactions
        assert result.height == 1
        itemsets = [tuple(row["itemset"]) for row in result.iter_rows(named=True)]
        assert (3,) in itemsets

    def test_apriori_low_support_all_frequent(self, sample_transactions):
        """Test that min_support=0 returns all possible itemsets."""
        result = apriori(sample_transactions, min_support=0.0)

        # Should include items 4 and 5 (support=0.25)
        itemsets = [tuple(row["itemset"]) for row in result.iter_rows(named=True)]
        assert (4,) in itemsets
        assert (5,) in itemsets

    def test_apriori_support_at_boundary(self, sample_transactions):
        """Test min_support exactly at item support boundary."""
        # Item 1 has support=0.5
        result = apriori(sample_transactions, min_support=0.5)
        itemsets = [tuple(row["itemset"]) for row in result.iter_rows(named=True)]
        assert (1,) in itemsets

        # With support just above 0.5, item 1 should be excluded
        result = apriori(sample_transactions, min_support=0.51)
        itemsets = [tuple(row["itemset"]) for row in result.iter_rows(named=True)]
        assert (1,) not in itemsets


class TestAprioriParameters:
    """Tests for apriori() parameters."""

    def test_apriori_max_length(self, sample_transactions):
        """Test max_length parameter limits itemset size."""
        result = apriori(sample_transactions, min_support=0.5, max_length=1)

        lengths = [len(row["itemset"]) for row in result.iter_rows(named=True)]
        assert all(length == 1 for length in lengths)
        assert result.height == 3  # Items 1, 2, 3

    def test_apriori_max_length_two(self, sample_transactions):
        """Test max_length=2 allows pairs but not triplets."""
        result = apriori(sample_transactions, min_support=0.25, max_length=2)

        lengths = [len(row["itemset"]) for row in result.iter_rows(named=True)]
        assert max(lengths) <= 2

    def test_apriori_custom_column(self, custom_column_transactions):
        """Test item_col parameter with custom column name."""
        result = apriori(
            custom_column_transactions,
            min_support=0.5,
            item_col="products"
        )

        assert result.height > 0
        itemsets = [tuple(row["itemset"]) for row in result.iter_rows(named=True)]
        assert (20,) in itemsets  # Item 20 appears in all 3 transactions


class TestAprioriEdgeCases:
    """Edge case tests for apriori()."""

    def test_apriori_single_transaction(self, single_transaction):
        """Test with only one transaction."""
        result = apriori(single_transaction, min_support=0.5)

        # All items have support=1.0 in single transaction
        assert result.height > 0
        supports = result["support"].to_list()
        assert all(s == 1.0 for s in supports)

    def test_apriori_empty_dataframe(self, empty_transactions):
        """Test with empty DataFrame."""
        result = apriori(empty_transactions, min_support=0.5)

        assert result.height == 0
        assert "itemset" in result.columns
        assert "support" in result.columns

    def test_apriori_output_schema(self, sample_transactions):
        """Verify output DataFrame has correct schema."""
        result = apriori(sample_transactions, min_support=0.5)

        expected_schema = {
            "itemset": pl.List(pl.Int64),
            "support": pl.Float64
        }
        assert result.schema == expected_schema

    def test_apriori_itemsets_sorted(self, sample_transactions):
        """Verify itemsets are internally sorted."""
        result = apriori(sample_transactions, min_support=0.5)

        for row in result.iter_rows(named=True):
            itemset = row["itemset"]
            assert itemset == sorted(itemset), f"Itemset {itemset} is not sorted"

    def test_invalid_min_support_raises_error(self, sample_transactions):
        """Test that invalid min_support values raise ValueError."""
        import pytest

        with pytest.raises(ValueError, match="must be in range"):
            apriori(sample_transactions, min_support=1.1)

        with pytest.raises(ValueError, match="must be in range"):
            apriori(sample_transactions, min_support=-0.1)


class TestAprioriPruning:
    """Tests for candidate pruning (downward closure property)."""

    def test_pruning_eliminates_invalid_candidates(self):
        """Test that candidates with non-frequent subsets are pruned.

        Scenario: {A,B} and {A,C} are frequent, but {B,C} is NOT frequent.
        Therefore {A,B,C} should be pruned (not generated as frequent).
        """
        # Transactions designed so {1,2} and {1,3} are frequent but {2,3} is not
        transactions = pl.DataFrame({
            "items": [
                [1, 2],      # Has {1,2}
                [1, 2],      # Has {1,2}
                [1, 3],      # Has {1,3}
                [1, 3],      # Has {1,3}
                [2],         # Only 2, no pair with 3
                [3],         # Only 3, no pair with 2
            ]
        })

        result = apriori(transactions, min_support=0.3)

        # {1,2} should be frequent (appears in 2/6 = 0.33)
        # {1,3} should be frequent (appears in 2/6 = 0.33)
        # {2,3} should NOT be frequent (appears in 0/6 = 0.0)
        # Therefore {1,2,3} should be PRUNED (not in result)

        itemsets = [tuple(sorted(row["itemset"])) for row in result.iter_rows(named=True)]

        assert (1, 2) in itemsets, "Expected {1,2} to be frequent"
        assert (1, 3) in itemsets, "Expected {1,3} to be frequent"
        assert (2, 3) not in itemsets, "Expected {2,3} to NOT be frequent"
        assert (1, 2, 3) not in itemsets, "Expected {1,2,3} to be PRUNED"


class TestAprioriLazyFrame:
    """Tests for LazyFrame input support."""

    def test_lazyframe_input(self, sample_transactions):
        """Test with LazyFrame input."""
        result = apriori(sample_transactions.lazy(), min_support=0.5)
        assert result.height > 0

    def test_lazyframe_from_parquet(self, sample_parquet_path):
        """apriori() accepts LazyFrame from scan_parquet."""
        lf = pl.scan_parquet(sample_parquet_path)
        result = apriori(lf, min_support=0.4)

        assert result.height > 0
        assert "itemset" in result.columns
        assert "support" in result.columns


class TestBooleanMatrix:
    """Tests for boolean matrix construction."""

    def test_basic_matrix(self, sample_transactions):
        """Test basic matrix construction."""
        matrix, col_to_item, n = build_boolean_matrix(
            sample_transactions.lazy(), 0.5
        )

        # Should have 4 transactions
        assert n == 4

        # Should have frequent items (support >= 0.5): 1, 2, 3
        assert len(col_to_item) == 3

        # Matrix should have 4 rows (transactions) and 3 cols (frequent items)
        assert matrix.height == 4
        assert len(matrix.columns) == 3

    def test_matrix_values(self, sample_transactions):
        """Test matrix boolean values are correct."""
        matrix, col_to_item, n = build_boolean_matrix(
            sample_transactions.lazy(), 0.5
        )

        # Find column for item 3 (appears in all transactions)
        item_to_col = {v: k for k, v in col_to_item.items()}
        col_3 = item_to_col[3]

        # Item 3 should be True in all rows
        assert matrix[col_3].sum() == 4

    def test_empty_result_high_threshold(self, sample_transactions):
        """Test with high threshold returning no frequent items."""
        matrix, col_to_item, n = build_boolean_matrix(
            sample_transactions.lazy(), 1.1
        )

        assert len(col_to_item) == 0
        assert matrix.height == 0

    def test_empty_transactions(self, empty_transactions):
        """Test with empty input."""
        matrix, col_to_item, n = build_boolean_matrix(
            empty_transactions.lazy(), 0.5
        )

        assert n == 0
        assert len(col_to_item) == 0


class TestSupportCounting:
    """Tests for support counting functions."""

    def test_single_itemset(self, sample_transactions):
        """Test counting support for single itemset."""
        matrix, col_to_item, n = build_boolean_matrix(
            sample_transactions.lazy(), 0.25
        )

        cols = list(col_to_item.keys())
        counts = count_support_vectorized(matrix, [(cols[0],)])

        assert len(counts) == 1
        assert all(isinstance(v, int) for v in counts.values())

    def test_pair_itemset(self, sample_transactions):
        """Test counting support for pair itemsets."""
        matrix, col_to_item, n = build_boolean_matrix(
            sample_transactions.lazy(), 0.25
        )

        cols = list(col_to_item.keys())
        if len(cols) >= 2:
            pairs = [(cols[0], cols[1])]
            counts = count_support_vectorized(matrix, pairs)

            assert len(counts) == 1
            # Count should be non-negative
            assert counts[pairs[0]] >= 0

    def test_batched_matches_unbatched(self, sample_transactions):
        """Test that batched counting matches unbatched."""
        matrix, col_to_item, n = build_boolean_matrix(
            sample_transactions.lazy(), 0.25
        )

        cols = list(col_to_item.keys())
        if len(cols) >= 3:
            itemsets = [
                (cols[0],),
                (cols[1],),
                (cols[2],),
                (cols[0], cols[1]),
                (cols[1], cols[2]),
            ]

            unbatched = count_support_vectorized(matrix, itemsets)
            batched = count_support_batched(matrix, itemsets, n, batch_size=2)

            assert unbatched == batched


class TestAprioriPerformance:
    """Performance tests for apriori algorithm."""

    def test_performance_reasonable(self):
        """Test apriori completes in reasonable time."""
        rng = np.random.default_rng(42)
        transactions = [
            rng.choice(50, size=8, replace=False).tolist() for _ in range(5000)
        ]
        df = pl.DataFrame({"items": transactions})

        start = time.time()
        result = apriori(df, min_support=0.1)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 30 seconds)
        assert elapsed < 30, f"Apriori took {elapsed:.2f}s"
        assert result.height > 0

    def test_batch_size_parameter(self, large_transactions):
        """Test batch_size parameter works."""
        # Small batch size should still produce correct results
        result_small = apriori(
            large_transactions, min_support=0.1, batch_size=100
        )
        result_large = apriori(
            large_transactions, min_support=0.1, batch_size=10000
        )

        assert result_small.height == result_large.height

        # Same itemsets
        small_itemsets = set(
            tuple(row["itemset"]) for row in result_small.iter_rows(named=True)
        )
        large_itemsets = set(
            tuple(row["itemset"]) for row in result_large.iter_rows(named=True)
        )
        assert small_itemsets == large_itemsets


class TestBitvecsIntegration:
    """Integration tests for bitvecs parameter and GPU/CPU fallback."""

    def test_apriori_bitvecs_validation_no_transactions(self):
        """Test error when neither transactions nor bitvecs provided."""
        with pytest.raises(ValueError, match="Either 'transactions' or 'bitvecs' must be provided"):
            apriori(transactions=None, bitvecs=None, min_support=0.1)

    def test_apriori_bitvecs_validation_both_provided(self, sample_transactions):
        """Test error when both transactions and bitvecs provided."""
        fake_bitvecs = (None, {}, 2)  # Will fail bitvecs validation anyway
        with pytest.raises(ValueError, match="Cannot provide both"):
            apriori(transactions=sample_transactions, bitvecs=fake_bitvecs, min_support=0.1)

    def test_apriori_bitvecs_validation_not_tuple(self):
        """Test error when bitvecs is not a tuple."""
        with pytest.raises(ValueError, match="bitvecs must be a tuple"):
            apriori(transactions=None, bitvecs="not a tuple", min_support=0.1)

    def test_apriori_bitvecs_validation_wrong_length(self):
        """Test error when bitvecs tuple has wrong number of elements."""
        with pytest.raises(ValueError, match="bitvecs must be a tuple"):
            apriori(transactions=None, bitvecs=(None, {}), min_support=0.1)  # 2-tuple instead of 3

    def test_apriori_bitvecs_validation_not_cupy(self):
        """Test error when bitvecs_gpu is not a CuPy array (no __cuda_array_interface__)."""
        # Use a numpy array - it has no __cuda_array_interface__
        fake_array = np.zeros((10, 2), dtype=np.uint64)
        fake_bitvecs = (fake_array, {i: i for i in range(10)}, 100)
        with pytest.raises(TypeError, match="bitvecs_gpu must be a CuPy array"):
            apriori(transactions=None, bitvecs=fake_bitvecs, min_support=0.1)

    def test_apriori_bitvecs_validation_wrong_shape(self):
        """Test error when bitvecs shape doesn't match n_transactions."""
        # Create a mock object that has __cuda_array_interface__ but wrong shape
        class MockCuPyArray:
            """Mock CuPy array for testing validation without actual GPU."""

            def __init__(self, shape):
                self.shape = shape
                self.ndim = len(shape)
                # Required for CuPy array interface detection
                self.__cuda_array_interface__ = {
                    "shape": shape,
                    "typestr": "<u8",
                    "version": 3,
                }

        # n_transactions=100 should require ceil(100/64)=2 u64s
        # But we provide shape with only 1 u64
        mock_array = MockCuPyArray(shape=(10, 1))  # Wrong: should be (10, 2)
        fake_bitvecs = (mock_array, {i: i for i in range(10)}, 100)

        with pytest.raises(ValueError, match=r"bitvecs_gpu\.shape\[1\].*must equal"):
            apriori(transactions=None, bitvecs=fake_bitvecs, min_support=0.1)

    def test_cpu_fallback_consistency(self):
        """Test that CPU fallback produces consistent results with same seed."""
        from et_miner.gpu.multi_gpu import (
            generate_bitvecs_cpu_fallback,
            count_itemsets_cpu_fallback,
        )

        n_rows = 1000
        n_cols = 50
        avg_items = 10
        seed = 42

        # Generate bitvecs twice with same seed
        bitvecs1 = generate_bitvecs_cpu_fallback(n_rows, n_cols, avg_items, seed=seed)
        bitvecs2 = generate_bitvecs_cpu_fallback(n_rows, n_cols, avg_items, seed=seed)

        # Should be identical
        np.testing.assert_array_equal(bitvecs1, bitvecs2)

        # Count same itemsets on both
        itemsets = [
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([0, 1, 2], dtype=np.int64),
        ]

        counts1 = count_itemsets_cpu_fallback(bitvecs1, itemsets, n_rows)
        counts2 = count_itemsets_cpu_fallback(bitvecs2, itemsets, n_rows)

        # Results should be identical
        np.testing.assert_array_equal(counts1, counts2)

    def test_cpu_fallback_monotonicity(self):
        """Test that larger itemsets have <= support than their subsets (anti-monotonicity)."""
        from et_miner.gpu.multi_gpu import (
            generate_bitvecs_cpu_fallback,
            count_itemsets_cpu_fallback,
        )

        n_rows = 1000
        n_cols = 50
        avg_items = 10

        bitvecs = generate_bitvecs_cpu_fallback(n_rows, n_cols, avg_items, seed=42)

        # Test anti-monotonicity: count({A}) >= count({A,B}) >= count({A,B,C})
        itemsets = [
            np.array([0], dtype=np.int64),         # {A}
            np.array([0, 1], dtype=np.int64),      # {A, B}
            np.array([0, 1, 2], dtype=np.int64),   # {A, B, C}
        ]

        counts = count_itemsets_cpu_fallback(bitvecs, itemsets, n_rows)

        # Anti-monotonicity: superset count <= subset count
        assert counts[0] >= counts[1], f"count({{A}})={counts[0]} should >= count({{A,B}})={counts[1]}"
        assert counts[1] >= counts[2], f"count({{A,B}})={counts[1]} should >= count({{A,B,C}})={counts[2]}"

        # Also test another path: {B} >= {A,B}
        itemsets_b = [
            np.array([1], dtype=np.int64),         # {B}
            np.array([0, 1], dtype=np.int64),      # {A, B}
        ]
        counts_b = count_itemsets_cpu_fallback(bitvecs, itemsets_b, n_rows)
        assert counts_b[0] >= counts_b[1], f"count({{B}})={counts_b[0]} should >= count({{A,B}})={counts_b[1]}"
