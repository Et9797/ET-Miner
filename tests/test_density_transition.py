"""Tests for the measured-density dense→sparse transition (gpu/density.py).

Pure decision logic — no GPU required. The GPU loops consume these
decisions; end-to-end sparse-mode behavior is covered by the gpu-marked
kernel tests.
"""

import numpy as np
import polars as pl
import pytest

from et_miner.core.apriori import apriori
from et_miner.gpu.density import (
    DENSITY_CROSSOVER,
    MIN_SPARSE_K,
    SPARSE_AUTO,
    should_transition_to_sparse,
    validate_sparse_from_k,
)


class TestValidateSparseFromK:
    """Public-argument validation."""

    @pytest.mark.parametrize("value", [None, 3, 4, 100, SPARSE_AUTO])
    def test_valid_values_pass_through(self, value):
        assert validate_sparse_from_k(value) is value

    @pytest.mark.parametrize("value", ["AUTO", "Auto", "always", ""])
    def test_other_strings_rejected(self, value):
        with pytest.raises(ValueError, match="sparse_from_k"):
            validate_sparse_from_k(value)

    @pytest.mark.parametrize("value", [3.5, True, False, [4]])
    def test_other_types_rejected(self, value):
        with pytest.raises(TypeError, match="sparse_from_k"):
            validate_sparse_from_k(value)


class TestShouldTransitionFixedK:
    """Explicit int sparse_from_k keeps the fixed-K semantics."""

    def test_none_never_transitions(self):
        for k in range(2, 12):
            assert should_transition_to_sparse(None, k, n_transactions=1000, mean_count=1.0) is False

    def test_fixed_k_triggers_at_k(self):
        assert should_transition_to_sparse(4, 3, n_transactions=1000) is False
        assert should_transition_to_sparse(4, 4, n_transactions=1000) is True
        assert should_transition_to_sparse(4, 5, n_transactions=1000) is True

    def test_fixed_k_ignores_density(self):
        """Explicit int overrides measurement — dense data still transitions."""
        # mean_count == n_transactions: as dense as it gets
        assert should_transition_to_sparse(4, 4, n_transactions=1000, mean_count=1000.0) is True

    def test_fixed_k_floored_at_min_sparse_k(self):
        """CSR needs K>=3 groups; K=2 fused kernel is always faster."""
        for sparse_from_k in (0, 1, 2):
            assert should_transition_to_sparse(sparse_from_k, 2, n_transactions=1000) is False
            assert should_transition_to_sparse(sparse_from_k, MIN_SPARSE_K, n_transactions=1000) is True


class TestShouldTransitionAuto:
    """Auto mode: transition when tidsets become smaller than bitvecs."""

    N = 3_200_000  # crossover count = N/32 = 100_000

    def test_below_crossover_transitions(self):
        assert should_transition_to_sparse(SPARSE_AUTO, 3, n_transactions=self.N, mean_count=99_999.0) is True

    def test_at_crossover_stays_dense(self):
        """Strict inequality: equal cost keeps the dense layout (no conversion)."""
        assert should_transition_to_sparse(SPARSE_AUTO, 3, n_transactions=self.N, mean_count=100_000.0) is False

    def test_above_crossover_stays_dense(self):
        assert should_transition_to_sparse(SPARSE_AUTO, 3, n_transactions=self.N, mean_count=500_000.0) is False

    def test_crossover_fraction_is_bitvec_vs_tidset_bytes(self):
        """4 bytes/tid crosses n_rows/8 bytes/bitvec at mean support n/32."""
        assert DENSITY_CROSSOVER == pytest.approx((1 / 8) / 4)

    def test_never_below_min_sparse_k(self):
        """Even extremely sparse data waits for K=3 (fused K=2 kernel wins)."""
        assert should_transition_to_sparse(SPARSE_AUTO, 2, n_transactions=self.N, mean_count=1.0) is False
        assert should_transition_to_sparse(SPARSE_AUTO, 3, n_transactions=self.N, mean_count=1.0) is True

    def test_unknown_counts_defer(self):
        """No counts (e.g. first level after resume) — stay dense this level."""
        assert should_transition_to_sparse(SPARSE_AUTO, 5, n_transactions=self.N, mean_count=None) is False

    def test_degenerate_n_transactions(self):
        assert should_transition_to_sparse(SPARSE_AUTO, 5, n_transactions=0, mean_count=1.0) is False


class TestAutoVersusFixedEquivalence:
    """On synthetic per-level count arrays, auto fires exactly where a
    correctly hand-picked fixed K would."""

    @staticmethod
    def _first_sparse_level(sparse_from_k, level_mean_counts, n_transactions):
        """Simulate the loop: first k (starting at 2) decided sparse.

        level_mean_counts[k] holds the mean count of level k-1's frequent
        itemsets, as measured at the top of level k.
        """
        for k in sorted(level_mean_counts):
            if should_transition_to_sparse(
                sparse_from_k, k, n_transactions=n_transactions, mean_count=level_mean_counts[k]
            ):
                return k  # sticky: callers never leave sparse mode
        return None

    def test_monotone_decay_matches_fixed(self):
        n = 1_000_000  # crossover at mean_count 31_250
        # Support decays with k; crosses n/32 between k=4 and k=5
        means = {2: 400_000.0, 3: 120_000.0, 4: 40_000.0, 5: 25_000.0, 6: 8_000.0}
        assert self._first_sparse_level(SPARSE_AUTO, means, n) == 5
        assert self._first_sparse_level(5, means, n) == 5

    def test_sparse_from_the_start(self):
        """Data already sparse at K=2 → auto fires at the K=3 floor."""
        n = 1_000_000
        means = {2: 1_000.0, 3: 500.0, 4: 100.0}
        assert self._first_sparse_level(SPARSE_AUTO, means, n) == MIN_SPARSE_K

    def test_dense_forever_never_fires(self):
        """Dense data (e.g. online-retail shaped) never converts in auto."""
        n = 10_000
        means = {k: 5_000.0 for k in range(2, 10)}
        assert self._first_sparse_level(SPARSE_AUTO, means, n) is None


class TestResumeCountRoundTrip:
    """Counts reconstructed from the parquet support column must be exact.

    The flush writes support = count / n_transactions (float64); resume
    recovers count = rint(support * n_transactions). Verify the float64
    round trip is lossless across the int32 transaction-count range."""

    @pytest.mark.parametrize("n_transactions", [4, 1_000, 205_000_000, 2_147_483_647])
    def test_round_trip_exact(self, n_transactions):
        rng = np.random.default_rng(42)
        counts = rng.integers(1, n_transactions + 1, size=1_000, dtype=np.int64)
        # Include boundary values
        counts[:3] = (1, n_transactions, max(1, n_transactions // 32))
        supports = counts / n_transactions  # what _flush_k_parquet stores
        recovered = np.rint(supports * n_transactions).astype(np.int64)
        np.testing.assert_array_equal(recovered, counts)


class TestAprioriParameterWiring:
    """sparse_from_k validation happens at the public API, CPU path included."""

    @pytest.fixture
    def df(self):
        return pl.DataFrame({"items": [[1, 2, 3], [2, 3], [1, 3], [2, 3, 4]]})

    def test_invalid_string_rejected_early(self, df):
        with pytest.raises(ValueError, match="sparse_from_k"):
            apriori(df, min_support=0.5, sparse_from_k="bogus")

    def test_invalid_type_rejected_early(self, df):
        with pytest.raises(TypeError, match="sparse_from_k"):
            apriori(df, min_support=0.5, sparse_from_k=3.5)

    def test_auto_accepted_on_cpu_path(self, df):
        """CPU tier ignores sparse_from_k — 'auto' must not change results."""
        base = apriori(df, min_support=0.5).sort("support", "itemset")
        auto = apriori(df, min_support=0.5, sparse_from_k=SPARSE_AUTO).sort("support", "itemset")
        assert base.equals(auto)
