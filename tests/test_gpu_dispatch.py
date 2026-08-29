"""Tests for gpu_dispatch module — auto-dispatch logic for multi-GPU k=2.

Tests the decision logic (should_use_multi_gpu) using mocks so no GPU required.
"""
from unittest.mock import patch


from et_miner.gpu.dispatch import (
    PAIR_COUNT_THRESHOLD,
    should_use_multi_gpu,
)


class TestShouldUseMultiGpu:
    """Test should_use_multi_gpu decision logic."""

    @patch("et_miner.gpu.dispatch.get_gpu_count", return_value=4)
    def test_above_threshold_multi_gpu(self, mock_gpus):
        """Above threshold + multiple GPUs -> True."""
        # Find n_items that gives us pairs above threshold
        # comb(5500, 2) = 15_122_250 > 15M threshold
        assert should_use_multi_gpu(5500) is True

    @patch("et_miner.gpu.dispatch.get_gpu_count", return_value=4)
    def test_below_threshold_multi_gpu(self, mock_gpus):
        """Below threshold -> False even with multiple GPUs."""
        # comb(1000, 2) = 499_500 << 15M threshold
        assert should_use_multi_gpu(1000) is False

    @patch("et_miner.gpu.dispatch.get_gpu_count", return_value=1)
    def test_single_gpu_always_false(self, mock_gpus):
        """Single GPU -> always False regardless of pair count."""
        # Even with huge pair count, single GPU means no multi-GPU dispatch
        assert should_use_multi_gpu(10_000) is False

    @patch("et_miner.gpu.dispatch.get_gpu_count", return_value=0)
    def test_no_gpu_always_false(self, mock_gpus):
        """No GPUs -> False."""
        assert should_use_multi_gpu(10_000) is False

    @patch("et_miner.gpu.dispatch.get_gpu_count", return_value=4)
    def test_exact_threshold_boundary(self, mock_gpus):
        """At exact threshold boundary -> True (>= comparison)."""
        # Find n where comb(n, 2) == PAIR_COUNT_THRESHOLD exactly is unlikely,
        # so test just above and just below
        # comb(5477, 2) = 14_996_026 < 15M
        # comb(5478, 2) = 15_001_503 >= 15M
        assert should_use_multi_gpu(5477) is False
        assert should_use_multi_gpu(5478) is True

    @patch("et_miner.gpu.dispatch.get_gpu_count", return_value=4)
    def test_zero_items(self, mock_gpus):
        """Zero frequent items -> False."""
        assert should_use_multi_gpu(0) is False

    @patch("et_miner.gpu.dispatch.get_gpu_count", return_value=4)
    def test_one_item(self, mock_gpus):
        """One frequent item (0 pairs) -> False."""
        assert should_use_multi_gpu(1) is False

    @patch("et_miner.gpu.dispatch.get_gpu_count", return_value=2)
    def test_two_gpus_above_threshold(self, mock_gpus):
        """Two GPUs + above threshold -> True."""
        assert should_use_multi_gpu(5500) is True


class TestThresholdValue:
    """Sanity checks on the threshold constant."""

    def test_threshold_is_positive(self):
        assert PAIR_COUNT_THRESHOLD > 0

    def test_threshold_is_reasonable(self):
        """Threshold should be in the millions (overhead justification range)."""
        assert 1_000_000 <= PAIR_COUNT_THRESHOLD <= 100_000_000
