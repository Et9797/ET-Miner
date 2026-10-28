"""The CPU counting API (count_support_batched) and the boolean matrix's CSR conversion.

The scipy sparse counter went with route consolidation C4; count_support_batched counts with Polars.
"""

import random

import numpy as np
import polars as pl
import pytest

from et_miner.core.matrix import (
    _polars_to_sparse_csr,
    build_boolean_matrix,
    count_support_batched,
)


def _counter_counts(df: pl.DataFrame, min_support: float) -> dict[tuple, int]:
    """count_support_batched of every itemset the CPU route mines, keyed by sorted item ids.

    The whole frame is one boolean matrix, so one counter call sees every row
    holding an itemset: a counter that wraps at 256 shows wherever 256+ rows
    hold one.
    """
    from et_miner import apriori

    matrix, col_to_item, n = build_boolean_matrix(df.lazy(), min_support)
    col_of = {item: col for col, item in col_to_item.items()}
    itemsets = [tuple(col_of[i] for i in sorted(s)) for s in apriori(df, min_support=min_support)["itemset"].to_list()]
    counts = count_support_batched(matrix, itemsets, n, enable_length_filter=False)
    return {tuple(sorted(col_to_item[c] for c in cols)): int(v) for cols, v in counts.items()}


def _mined_counts(df: pl.DataFrame, min_support: float) -> dict[tuple, int]:
    from et_miner import apriori

    res = apriori(df, min_support=min_support)
    return {tuple(sorted(s)): round(p * df.height) for s, p in zip(res["itemset"].to_list(), res["support"].to_list())}


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


class TestCounter:
    def test_a_repeated_item_is_held_by_a_row_of_its_distinct_items(self):
        """The length filter keys on distinct items: ("a", "a", "b") is held by the 2-item row {a, b}."""
        df = pl.DataFrame({"items": [["a", "b"], ["a", "b", "c"], ["a"]]})
        matrix, col_to_item, n = build_boolean_matrix(df.lazy(), 0.01)
        item_to_col = {v: k for k, v in col_to_item.items()}
        itemset = (item_to_col["a"], item_to_col["a"], item_to_col["b"])

        assert count_support_batched(matrix, [itemset], n) == {itemset: 2}

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
        result_seq = apriori(df, min_support=0.4, streaming=True, chunk_size=2, n_jobs=1)

        # Parallel
        result_par = apriori(df, min_support=0.4, streaming=True, chunk_size=2, n_jobs=2)

        # Both should give identical results
        assert result_seq.shape == result_par.shape
        assert result_seq["support"].to_list() == result_par["support"].to_list()


class TestCounterCorrectness:
    """The counter gives the exact count of every mined itemset (a uint8 counter would wrap past 255)."""

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
    def test_counter_matches_the_mined_counts(self, patterned_dataset, min_support):
        expected = _mined_counts(patterned_dataset, min_support)
        assert len(expected) > 3
        assert _counter_counts(patterned_dataset, min_support) == expected


class TestUint8OverflowRegression:
    """Specifieke tests voor de uint8 overflow bug (counts > 255)."""

    def test_large_transaction_count_no_overflow(self):
        """500 identieke transacties: elke count is 500, niet 500 % 256 = 244."""
        df = pl.DataFrame({"items": [[1, 2, 3]] * 500})
        counts = _counter_counts(df, 0.5)
        assert len(counts) == 7
        assert set(counts.values()) == {500}

    @pytest.mark.parametrize("n", [255, 256, 257])
    def test_counts_around_256(self, n):
        """n transacties met [1,2]; met de uint8 bug telt 256 als 0 en 257 als 1."""
        df = pl.DataFrame({"items": [[1, 2]] * n + [[3]]})
        assert _counter_counts(df, 0.5) == _mined_counts(df, 0.5) == {(1,): n, (2,): n, (1, 2): n}


@pytest.mark.parametrize("kw", [{"sparse": True}, {"sparse": False}, {"n_jobs": 4}])
def test_the_removed_sparse_counter_arguments_raise(kw):
    matrix, col_to_item, n = build_boolean_matrix(pl.DataFrame({"items": [[1, 2], [1, 2, 3]]}).lazy(), 0.5)
    with pytest.raises(ValueError, match="sparse= and n_jobs= were removed"):
        count_support_batched(matrix, [(c,) for c in col_to_item], n, **kw)
