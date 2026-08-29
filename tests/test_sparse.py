"""Tests for scipy sparse matrix support.

These tests verify that the sparse counting strategy produces identical
results to the standard Polars approach, and that the automatic strategy
selection works correctly.
"""

import random

import numpy as np
import polars as pl
import pytest

from et_miner.core.matrix import (
    _choose_counting_strategy,
    _estimate_density,
    _polars_to_sparse_csr,
    build_boolean_matrix,
    count_support_batched,
    count_support_sparse,
    count_support_vectorized,
)


class TestPolarsToSparseCsr:
    """Tests for _polars_to_sparse_csr conversion."""

    def test_basic_conversion(self):
        """Test basic Polars DataFrame to CSR conversion."""
        df = pl.DataFrame({
            "i_0": [True, False, True],
            "i_1": [True, True, False],
            "i_2": [False, True, True],
        })

        csr, col_to_idx = _polars_to_sparse_csr(df)

        # Check dimensions
        assert csr.shape == (3, 3)

        # Check column mapping
        assert col_to_idx == {"i_0": 0, "i_1": 1, "i_2": 2}

        # Check values - convert to dense for comparison
        dense = csr.toarray()
        expected = np.array([
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ], dtype=np.uint8)
        np.testing.assert_array_equal(dense, expected)

    def test_empty_matrix(self):
        """Test conversion of empty matrix."""
        df = pl.DataFrame({
            "i_0": pl.Series([], dtype=pl.Boolean),
            "i_1": pl.Series([], dtype=pl.Boolean),
        })

        csr, col_to_idx = _polars_to_sparse_csr(df)

        assert csr.shape == (0, 2)
        assert col_to_idx == {"i_0": 0, "i_1": 1}

    def test_all_false_matrix(self):
        """Test conversion of matrix with all False values."""
        df = pl.DataFrame({
            "i_0": [False, False, False],
            "i_1": [False, False, False],
        })

        csr, col_to_idx = _polars_to_sparse_csr(df)

        assert csr.shape == (3, 2)
        assert csr.nnz == 0  # No non-zero elements

    def test_all_true_matrix(self):
        """Test conversion of matrix with all True values."""
        df = pl.DataFrame({
            "i_0": [True, True],
            "i_1": [True, True],
        })

        csr, col_to_idx = _polars_to_sparse_csr(df)

        assert csr.shape == (2, 2)
        assert csr.nnz == 4
        np.testing.assert_array_equal(
            csr.toarray(),
            np.array([[1, 1], [1, 1]], dtype=np.uint8)
        )


class TestCountSupportSparse:
    """Tests for sparse support counting."""

    @pytest.fixture
    def simple_matrix(self):
        """Simple boolean matrix for testing."""
        return pl.DataFrame({
            "i_0": [True, False, True, True],   # items: A
            "i_1": [True, True, False, True],   # items: B
            "i_2": [False, True, True, True],   # items: C
        })

    def test_single_item_support(self, simple_matrix):
        """Test support counting for single items."""
        itemsets = [("i_0",), ("i_1",), ("i_2",)]
        result = count_support_sparse(simple_matrix, itemsets)

        assert result[("i_0",)] == 3
        assert result[("i_1",)] == 3
        assert result[("i_2",)] == 3

    def test_pair_itemset_support(self, simple_matrix):
        """Test support counting for 2-itemsets."""
        itemsets = [("i_0", "i_1"), ("i_0", "i_2"), ("i_1", "i_2")]
        result = count_support_sparse(simple_matrix, itemsets)

        # Verify manually:
        # i_0 & i_1: rows 0, 3 -> 2
        # i_0 & i_2: rows 2, 3 -> 2
        # i_1 & i_2: rows 1, 3 -> 2
        assert result[("i_0", "i_1")] == 2
        assert result[("i_0", "i_2")] == 2
        assert result[("i_1", "i_2")] == 2

    def test_triple_itemset_support(self, simple_matrix):
        """Test support counting for 3-itemsets."""
        itemsets = [("i_0", "i_1", "i_2")]
        result = count_support_sparse(simple_matrix, itemsets)

        # i_0 & i_1 & i_2: only row 3 -> 1
        assert result[("i_0", "i_1", "i_2")] == 1

    def test_empty_itemsets(self, simple_matrix):
        """Test with empty itemsets list."""
        result = count_support_sparse(simple_matrix, [])
        assert result == {}

    def test_matches_vectorized(self, simple_matrix):
        """Verify sparse results exactly match vectorized results."""
        itemsets = [
            ("i_0",), ("i_1",), ("i_2",),
            ("i_0", "i_1"), ("i_0", "i_2"), ("i_1", "i_2"),
            ("i_0", "i_1", "i_2"),
        ]

        sparse_result = count_support_sparse(simple_matrix, itemsets)
        vectorized_result = count_support_vectorized(simple_matrix, itemsets)

        assert sparse_result == vectorized_result


class TestSparseIntegration:
    """Integration tests for sparse support in count_support_batched."""

    @pytest.fixture
    def large_sparse_matrix(self):
        """Generate a larger sparse boolean matrix."""
        np.random.seed(42)
        n_rows = 1000
        n_cols = 100
        density = 0.03  # 3% density

        # Generate sparse random data
        data = {}
        for i in range(n_cols):
            col_data = np.random.random(n_rows) < density
            data[f"i_{i}"] = col_data.tolist()

        return pl.DataFrame(data)

    def test_forced_sparse_matches_polars(self, large_sparse_matrix):
        """Test that forced sparse=True produces identical results to Polars."""
        # Generate some itemsets
        itemsets = [
            ("i_0", "i_1"),
            ("i_0", "i_2"),
            ("i_1", "i_2"),
            ("i_5", "i_10"),
            ("i_0", "i_1", "i_2"),
        ]

        polars_result = count_support_batched(
            large_sparse_matrix, itemsets, n_transactions=1000, sparse=False
        )
        sparse_result = count_support_batched(
            large_sparse_matrix, itemsets, n_transactions=1000, sparse=True
        )

        assert polars_result == sparse_result

    def test_auto_detection_chooses_polars_for_dense(self, large_sparse_matrix):
        """Test that auto-detection chooses Polars for smaller matrices."""
        # 100 columns at 3% density won't trigger sparse (needs >2000 items)
        itemsets = [("i_0", "i_1")]

        # This should use Polars internally (auto-detection)
        result = count_support_batched(
            large_sparse_matrix, itemsets, n_transactions=1000, sparse=None
        )

        # Just verify it works - we can't easily check which strategy was used
        assert ("i_0", "i_1") in result


class TestEstimateDensity:
    """Tests for _estimate_density function."""

    def test_zero_density(self):
        """Test with all False values."""
        df = pl.DataFrame({
            "i_0": [False, False, False],
            "i_1": [False, False, False],
        })
        density = _estimate_density(df)
        assert density == 0.0

    def test_full_density(self):
        """Test with all True values."""
        df = pl.DataFrame({
            "i_0": [True, True, True],
            "i_1": [True, True, True],
        })
        density = _estimate_density(df)
        assert density == 1.0

    def test_partial_density(self):
        """Test with 50% density."""
        df = pl.DataFrame({
            "i_0": [True, False, True, False],
            "i_1": [False, True, False, True],
        })
        density = _estimate_density(df)
        assert density == 0.5

    def test_empty_matrix(self):
        """Test with empty matrix."""
        df = pl.DataFrame({
            "i_0": pl.Series([], dtype=pl.Boolean),
        })
        density = _estimate_density(df)
        assert density == 0.0


class TestChooseCountingStrategy:
    """Tests for _choose_counting_strategy function."""

    def test_small_matrix_chooses_polars(self):
        """Test that small matrices use Polars."""
        strategy = _choose_counting_strategy(
            n_items=100, n_transactions=10_000, density=0.01
        )
        assert strategy == "polars"

    def test_dense_matrix_below_k2_threshold_chooses_polars(self):
        """Test that dense matrices below the k2-explosion threshold use Polars.

        The k2 candidate-explosion rule (>100K pairs -> sparse) precedes the
        density check, so density only decides the outcome when n_items keeps
        k2 pairs under 100K (n_items <= 447). At 447 items and 20% density,
        neither the k2 rule nor the item+density rule fires -> Polars.
        """
        strategy = _choose_counting_strategy(
            n_items=447, n_transactions=1_000_000, density=0.20  # k2=99,681 < 100K, dense
        )
        assert strategy == "polars"

    def test_k2_explosion_overrides_density(self):
        """Test that the k2-explosion rule fires regardless of density.

        At 3000 items, k2 = 4.5M pairs > 100K, so sparse is chosen even at high
        (20%) density — the candidate explosion dominates the strategy choice.
        """
        strategy = _choose_counting_strategy(
            n_items=3000, n_transactions=1_000_000, density=0.20
        )
        assert strategy == "sparse"

    def test_large_sparse_matrix_chooses_sparse(self):
        """Test that large sparse matrices use scipy.

        Updated Jan 2026: New thresholds are >500 items + <10% density, OR >1GB size.
        """
        # 10K items × 5M tx = 50G bits = 6.25 GB → triggers >1GB threshold
        strategy = _choose_counting_strategy(
            n_items=10_000, n_transactions=5_000_000, density=0.01  # 1% density
        )
        assert strategy == "sparse"

    def test_extreme_sparse_chooses_sparse(self):
        """Test that extreme sparse workloads use scipy."""
        # 25K items × 2M tx = 50GB+ dense
        strategy = _choose_counting_strategy(
            n_items=25_000, n_transactions=2_000_000, density=0.001
        )
        assert strategy == "sparse"

    def test_borderline_cases(self):
        """Test borderline cases around the thresholds.

        The k2-explosion rule (>100K pairs -> sparse) fires first, so it caps the
        Polars region at n_items <= 447 (k2=99,681). Above that, sparse is chosen
        by the k2 rule before density or size are considered.
        """
        # k2 boundary: 447 items = 99,681 pairs (< 100K) -> below all rules -> Polars
        strategy = _choose_counting_strategy(
            n_items=447, n_transactions=100_000, density=0.05
        )
        assert strategy == "polars"

        # 448 items = 100,128 pairs (> 100K) -> k2-explosion rule -> sparse
        strategy = _choose_counting_strategy(
            n_items=448, n_transactions=100_000, density=0.11
        )
        assert strategy == "sparse"

        # Under 1GB estimated size AND under the k2 threshold - should use Polars
        # 400 items × 1M tx = 400M bits = 50MB, k2=79,800 < 100K
        strategy = _choose_counting_strategy(
            n_items=400, n_transactions=1_000_000, density=0.05
        )
        assert strategy == "polars"


class TestSparseWithRealData:
    """Tests using the build_boolean_matrix + sparse pipeline."""

    def test_full_pipeline_sparse_forced(self, sample_transactions):
        """Test the full pipeline with forced sparse counting."""
        matrix, col_to_item, n_tx = build_boolean_matrix(
            sample_transactions.lazy(), min_support=0.5
        )

        # Get column names for frequent items
        col_names = list(col_to_item.keys())

        # Generate itemsets
        itemsets = [(col,) for col in col_names]
        if len(col_names) >= 2:
            itemsets.append((col_names[0], col_names[1]))

        # Count with both methods
        polars_counts = count_support_batched(matrix, itemsets, n_tx, sparse=False)
        sparse_counts = count_support_batched(matrix, itemsets, n_tx, sparse=True)

        assert polars_counts == sparse_counts


@pytest.fixture
def sample_transactions():
    """Basic sample transactions."""
    return pl.DataFrame({
        "items": [[1, 2, 3], [2, 3, 4], [1, 3, 5], [2, 3]]
    })


class TestParallelSparseSupport:
    """Tests for parallel k>2 itemset support counting."""

    def test_parallel_matches_sequential_small(self):
        """Test parallel results match sequential for small datasets."""
        from et_miner.core.matrix import (
            _count_support_sparse_k_gt_2,
            _polars_to_sparse_csr,
        )

        # Create a dataset with multiple k>2 itemsets
        df = pl.DataFrame({
            "i_0": [True, True, False, True, True],
            "i_1": [True, True, True, False, True],
            "i_2": [True, False, True, True, True],
            "i_3": [False, True, True, True, False],
        })

        csr, col_to_idx = _polars_to_sparse_csr(df)

        # Generate k=3 and k=4 itemsets
        itemsets = [
            ("i_0", "i_1", "i_2"),
            ("i_0", "i_1", "i_3"),
            ("i_0", "i_2", "i_3"),
            ("i_1", "i_2", "i_3"),
            ("i_0", "i_1", "i_2", "i_3"),
        ]

        # Sequential (n_jobs=1)
        seq_results = _count_support_sparse_k_gt_2(
            csr, col_to_idx, itemsets, show_progress=False, n_jobs=1
        )

        # Parallel (n_jobs=2)
        par_results = _count_support_sparse_k_gt_2(
            csr, col_to_idx, itemsets, show_progress=False, n_jobs=2
        )

        assert seq_results == par_results

    def test_parallel_matches_sequential_large(self):
        """Test parallel results match sequential for larger datasets with many itemsets."""
        from et_miner.core.matrix import (
            _count_support_sparse_k_gt_2,
            _polars_to_sparse_csr,
            _PARALLEL_THRESHOLD,
        )
        import random

        random.seed(42)

        # Create a dataset with enough itemsets to trigger parallel processing
        n_cols = 15
        n_rows = 100

        # Generate random boolean data
        data = {
            f"i_{i}": [random.random() < 0.4 for _ in range(n_rows)]
            for i in range(n_cols)
        }
        df = pl.DataFrame(data)

        csr, col_to_idx = _polars_to_sparse_csr(df)

        # Generate enough k=3 itemsets to exceed threshold
        from itertools import combinations
        col_names = list(col_to_idx.keys())
        itemsets = list(combinations(col_names, 3))

        # If not enough itemsets, skip this test
        if len(itemsets) < _PARALLEL_THRESHOLD:
            pytest.skip(f"Not enough itemsets ({len(itemsets)}) to test parallel threshold")

        # Sequential
        seq_results = _count_support_sparse_k_gt_2(
            csr, col_to_idx, itemsets, show_progress=False, n_jobs=1
        )

        # Parallel with auto workers
        par_results = _count_support_sparse_k_gt_2(
            csr, col_to_idx, itemsets, show_progress=False, n_jobs=-1
        )

        assert seq_results == par_results

    def test_n_jobs_api_parameter(self):
        """Test that n_jobs parameter flows through the full API."""
        from et_miner import apriori

        df = pl.DataFrame({
            "items": [
                [1, 2, 3, 4],
                [2, 3, 4, 5],
                [1, 2, 3, 5],
                [3, 4, 5, 6],
                [1, 3, 5, 6],
            ]
        })

        # Sequential
        result_seq = apriori(df, min_support=0.4, sparse=True, n_jobs=1)

        # Parallel
        result_par = apriori(df, min_support=0.4, sparse=True, n_jobs=2)

        # Both should give identical results
        assert result_seq.shape == result_par.shape
        assert result_seq["support"].to_list() == result_par["support"].to_list()


class TestGilDetection:
    """Tests for GIL detection utilities."""

    def test_is_gil_disabled_returns_bool(self):
        """Test that _is_gil_disabled returns a boolean."""
        from et_miner.core.matrix import _is_gil_disabled

        result = _is_gil_disabled()
        assert isinstance(result, bool)

    def test_get_effective_workers(self):
        """Test _get_effective_workers helper."""
        from et_miner.core.matrix import _get_effective_workers
        import os

        # -1 should use all CPUs
        assert _get_effective_workers(-1) == (os.cpu_count() or 1)

        # Positive int should use that many
        assert _get_effective_workers(4) == 4

        # 1 should return 1
        assert _get_effective_workers(1) == 1

        # 0 should return 1 (minimum)
        assert _get_effective_workers(0) == 1


class TestSparseMkl:
    """Tests for MKL-accelerated sparse matrix multiplication."""

    def test_sparse_matmul_correctness(self):
        """_sparse_matmul produces same result as scipy @ operator."""
        from et_miner.core.matrix import _sparse_matmul
        from scipy.sparse import csr_matrix

        # Create test matrix (5 transactions x 4 items)
        data = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        row = np.array([0, 0, 1, 1, 2, 2, 3, 4])
        col = np.array([0, 1, 1, 2, 0, 2, 3, 0])
        A = csr_matrix((data, (row, col)), shape=(5, 4))

        # Compare _sparse_matmul(A.T, A) with A.T @ A
        result_mkl = _sparse_matmul(A.T, A)
        result_scipy = A.T @ A

        # Convert to dense for comparison
        np.testing.assert_allclose(
            result_mkl.toarray(),
            result_scipy.toarray(),
            rtol=1e-5,
        )

    def test_sparse_matmul_handles_int_dtype(self):
        """_sparse_matmul handles integer matrices (converts to float)."""
        from et_miner.core.matrix import _sparse_matmul
        from scipy.sparse import csr_matrix

        # Create int32 sparse matrix
        data = np.array([1, 1, 1, 1], dtype=np.int32)
        row = np.array([0, 0, 1, 2])
        col = np.array([0, 1, 1, 0])
        A = csr_matrix((data, (row, col)), shape=(3, 2))

        # Should not raise an error and produce correct result
        result = _sparse_matmul(A.T, A)
        expected = A.T @ A

        np.testing.assert_allclose(
            result.toarray(),
            expected.toarray(),
            rtol=1e-5,
        )

    def test_sparse_matmul_fallback_works(self):
        """_sparse_matmul falls back to scipy when MKL unavailable."""
        from et_miner.core.matrix import _sparse_matmul
        from scipy.sparse import csr_matrix

        # Create a simple matrix - this test verifies the function works
        # regardless of whether MKL is installed or not
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        row = np.array([0, 1, 2])
        col = np.array([0, 0, 1])
        A = csr_matrix((data, (row, col)), shape=(3, 2))

        # Should work with either MKL or scipy fallback
        result = _sparse_matmul(A.T, A)
        expected = A.T @ A

        np.testing.assert_allclose(
            result.toarray(),
            expected.toarray(),
            rtol=1e-5,
        )


class TestAdaptiveParallelConfig:
    """Tests for adaptive parallel configuration."""

    def test_returns_required_keys(self):
        """Config dict contains workers, chunk_size, gil_disabled."""
        from et_miner.core.matrix import _get_adaptive_parallel_config

        config = _get_adaptive_parallel_config(n_itemsets=5000, n_workers=4)

        assert 'workers' in config
        assert 'chunk_size' in config
        assert 'gil_disabled' in config
        assert isinstance(config['workers'], int)
        assert isinstance(config['chunk_size'], int)
        assert isinstance(config['gil_disabled'], bool)

    def test_workers_capped(self):
        """Workers are capped at reasonable maximum."""
        from et_miner.core.matrix import _get_adaptive_parallel_config

        # With n_workers=100, should be capped at 8 (GIL) or 12 (no GIL)
        config = _get_adaptive_parallel_config(n_itemsets=10000, n_workers=100)

        # Workers should be capped at max 12 (free-threading cap)
        assert config['workers'] <= 12

        # With standard GIL Python, cap is 8
        if not config['gil_disabled']:
            assert config['workers'] <= 8

    def test_chunk_size_minimum(self):
        """Chunk size has minimum floor."""
        from et_miner.core.matrix import _get_adaptive_parallel_config

        # With small n_itemsets, chunk_size should still have a minimum
        config = _get_adaptive_parallel_config(n_itemsets=10, n_workers=4)

        # Minimum is 50 for free-threading, 100 for GIL Python
        if config['gil_disabled']:
            assert config['chunk_size'] >= 50
        else:
            assert config['chunk_size'] >= 100

    def test_chunk_size_scales_with_itemsets(self):
        """Chunk size scales appropriately with number of itemsets."""
        from et_miner.core.matrix import _get_adaptive_parallel_config

        config_small = _get_adaptive_parallel_config(n_itemsets=1000, n_workers=4)
        config_large = _get_adaptive_parallel_config(n_itemsets=100000, n_workers=4)

        # Larger itemset count should result in larger chunk size
        assert config_large['chunk_size'] >= config_small['chunk_size']

    def test_config_with_single_worker(self):
        """Config still works with single worker."""
        from et_miner.core.matrix import _get_adaptive_parallel_config

        config = _get_adaptive_parallel_config(n_itemsets=5000, n_workers=1)

        assert config['workers'] == 1
        assert config['chunk_size'] >= 50  # Still has minimum


# =============================================================================
# Sparse Mode Correctness Tests (merged from test_sparse_correctness.py)
# =============================================================================


class TestSparseModeCorrectness:
    """Verify sparse and non-sparse modes produce identical results.

    These tests prevent regression of the uint8 overflow bug that caused sparse
    mode to give wrong counts for datasets with >255 transactions per itemset.
    """

    @pytest.fixture
    def patterned_dataset(self):
        """Dataset met bekende patronen voor betrouwbare k>1 itemsets."""
        random.seed(42)
        transactions = []

        # Pattern [1,2,3] in 57% van transacties
        for _ in range(400):
            base = [1, 2, 3]
            extras = random.sample(range(10, 30), random.randint(0, 3))
            transactions.append(sorted(base + extras))

        # Pattern [4,5,6,7] in 29% van transacties
        for _ in range(200):
            base = [4, 5, 6, 7]
            extras = random.sample(range(30, 50), random.randint(0, 2))
            transactions.append(sorted(base + extras))

        # Random noise
        for _ in range(100):
            transactions.append(sorted(random.sample(range(1, 50), random.randint(2, 5))))

        return pl.DataFrame({"items": transactions})

    @pytest.mark.parametrize("min_support", [0.20, 0.10, 0.05])
    def test_sparse_vs_nonsparse_count(self, patterned_dataset, min_support):
        """sparse=True en sparse=False moeten identieke counts geven."""
        from et_miner import apriori

        sparse = apriori(patterned_dataset, min_support=min_support, sparse=True)
        nonsparse = apriori(patterned_dataset, min_support=min_support, sparse=False)

        assert sparse.height == nonsparse.height, (
            f"Count mismatch at support={min_support}: "
            f"sparse={sparse.height}, non-sparse={nonsparse.height}"
        )

    @pytest.mark.parametrize("min_support", [0.20, 0.10, 0.05])
    def test_sparse_vs_nonsparse_exact_itemsets(self, patterned_dataset, min_support):
        """sparse=True en sparse=False moeten exact dezelfde itemsets vinden."""
        from et_miner import apriori

        sparse = apriori(patterned_dataset, min_support=min_support, sparse=True)
        nonsparse = apriori(patterned_dataset, min_support=min_support, sparse=False)

        sparse_set = {
            tuple(sorted(r["itemset"]))
            for r in sparse.iter_rows(named=True)
        }
        nonsparse_set = {
            tuple(sorted(r["itemset"]))
            for r in nonsparse.iter_rows(named=True)
        }

        assert sparse_set == nonsparse_set, (
            f"Itemset mismatch at support={min_support}. "
            f"Only in sparse: {sparse_set - nonsparse_set}, "
            f"Only in non-sparse: {nonsparse_set - sparse_set}"
        )

    @pytest.mark.parametrize("min_support", [0.20, 0.10])
    def test_sparse_vs_nonsparse_supports_match(self, patterned_dataset, min_support):
        """Support values moeten identiek zijn tussen sparse en non-sparse."""
        from et_miner import apriori

        sparse = apriori(patterned_dataset, min_support=min_support, sparse=True)
        nonsparse = apriori(patterned_dataset, min_support=min_support, sparse=False)

        # Build dicts keyed by itemset
        sparse_supports = {
            tuple(sorted(r["itemset"])): r["support"]
            for r in sparse.iter_rows(named=True)
        }
        nonsparse_supports = {
            tuple(sorted(r["itemset"])): r["support"]
            for r in nonsparse.iter_rows(named=True)
        }

        for itemset, sparse_sup in sparse_supports.items():
            nonsparse_sup = nonsparse_supports.get(itemset)
            assert nonsparse_sup is not None, f"Itemset {itemset} missing from non-sparse"
            assert abs(sparse_sup - nonsparse_sup) < 1e-9, (
                f"Support mismatch for {itemset}: "
                f"sparse={sparse_sup}, non-sparse={nonsparse_sup}"
            )


class TestUint8OverflowRegression:
    """Specifieke tests voor de uint8 overflow bug (counts > 255)."""

    def test_large_transaction_count_no_overflow(self):
        """Dataset met >255 transacties per itemset moet correct tellen."""
        from et_miner import apriori

        # 500 transacties met exact dezelfde items - zou overflow geven met uint8
        transactions = [[1, 2, 3]] * 500
        df = pl.DataFrame({"items": transactions})

        sparse = apriori(df, min_support=0.5, sparse=True)
        nonsparse = apriori(df, min_support=0.5, sparse=False)

        assert sparse.height == nonsparse.height
        assert sparse.height > 0, "Should find frequent itemsets"

        # Check dat counts correct zijn (niet 500 % 256 = 244)
        for row in sparse.iter_rows(named=True):
            if len(row["itemset"]) == 2:
                # Support voor k=2 paren moet 1.0 zijn (500/500)
                assert row["support"] == 1.0, f"Expected support 1.0, got {row['support']}"

    def test_boundary_256_transactions(self):
        """Exact 256 transacties - edge case voor uint8."""
        from et_miner import apriori

        transactions = [[1, 2]] * 256
        df = pl.DataFrame({"items": transactions})

        sparse = apriori(df, min_support=0.5, sparse=True)
        nonsparse = apriori(df, min_support=0.5, sparse=False)

        assert sparse.height == nonsparse.height

    def test_boundary_257_transactions(self):
        """257 transacties - eerste overflow met uint8."""
        from et_miner import apriori

        transactions = [[1, 2]] * 257
        df = pl.DataFrame({"items": transactions})

        sparse = apriori(df, min_support=0.5, sparse=True)
        nonsparse = apriori(df, min_support=0.5, sparse=False)

        assert sparse.height == nonsparse.height
        # Met uint8 bug zou sparse 1 itemset vinden (257%256=1 < threshold)
        # Nu moeten beide alle itemsets vinden
