"""Tests for gpu.dispatch — the device-count decision for the k=2 fan-out.

`_resolve_gpus` is driven with a mocked device count, so no GPU is required.
"""
from math import comb
from unittest.mock import patch


from et_miner.gpu.dispatch import (
    PAIR_COUNT_THRESHOLD,
    _resolve_gpus,
)


def _k2_devices(n_frequent_items: int, n_gpus: int | None = None) -> int:
    return _resolve_gpus(n_gpus, comb(n_frequent_items, 2), PAIR_COUNT_THRESHOLD, "k=2")


class TestResolveGpusK2:
    """The k=2 fan-out engages at >= PAIR_COUNT_THRESHOLD pairs on > 1 device."""

    @patch("et_miner.gpu.dispatch.get_gpu_count", return_value=4)
    def test_above_threshold_multi_gpu(self, mock_gpus):
        # comb(5500, 2) = 15_122_250 > 15M threshold
        assert _k2_devices(5500) == 4

    @patch("et_miner.gpu.dispatch.get_gpu_count", return_value=4)
    def test_below_threshold_multi_gpu(self, mock_gpus):
        # comb(1000, 2) = 499_500 << 15M threshold
        assert _k2_devices(1000) == 1

    @patch("et_miner.gpu.dispatch.get_gpu_count", return_value=1)
    def test_single_gpu_always_one(self, mock_gpus):
        assert _k2_devices(10_000) == 1

    @patch("et_miner.gpu.dispatch.get_gpu_count", return_value=0)
    def test_no_gpu_resolves_to_one(self, mock_gpus):
        assert _k2_devices(10_000) == 1

    @patch("et_miner.gpu.dispatch.get_gpu_count", return_value=4)
    def test_exact_threshold_boundary(self, mock_gpus):
        # comb(5477, 2) = 14_996_026 < 15M; comb(5478, 2) = 15_001_503 >= 15M
        assert _k2_devices(5477) == 1
        assert _k2_devices(5478) == 4

    @patch("et_miner.gpu.dispatch.get_gpu_count", return_value=4)
    def test_zero_items(self, mock_gpus):
        assert _k2_devices(0) == 1

    @patch("et_miner.gpu.dispatch.get_gpu_count", return_value=4)
    def test_one_item(self, mock_gpus):
        assert _k2_devices(1) == 1

    @patch("et_miner.gpu.dispatch.get_gpu_count", return_value=2)
    def test_two_gpus_above_threshold(self, mock_gpus):
        assert _k2_devices(5500) == 2

    @patch("et_miner.gpu.dispatch.get_gpu_count", return_value=4)
    def test_caller_budget_caps_the_device_count(self, mock_gpus):
        assert _k2_devices(5500, n_gpus=2) == 2
        assert _k2_devices(5500, n_gpus=1) == 1


class TestThresholdValue:
    """Sanity checks on the threshold constant."""

    def test_threshold_is_positive(self):
        assert PAIR_COUNT_THRESHOLD > 0

    def test_threshold_is_reasonable(self):
        """Threshold should be in the millions (overhead justification range)."""
        assert 1_000_000 <= PAIR_COUNT_THRESHOLD <= 100_000_000


class TestKernelVariantResolution:
    """resolved_kernel_variant (CPU-only decision logic)."""

    def test_auto_resolves_to_shared(self, monkeypatch):
        from et_miner.gpu.dispatch import resolved_kernel_variant

        monkeypatch.delenv("ET_MINER_KERNEL_VARIANT", raising=False)
        assert resolved_kernel_variant() == "shared"
        monkeypatch.setenv("ET_MINER_KERNEL_VARIANT", "auto")
        assert resolved_kernel_variant() == "shared"

    def test_explicit_values(self, monkeypatch):
        from et_miner.gpu.dispatch import resolved_kernel_variant

        for v in ("legacy", "shared"):
            monkeypatch.setenv("ET_MINER_KERNEL_VARIANT", v)
            assert resolved_kernel_variant() == v

    def test_invalid_rejected(self, monkeypatch):
        import pytest

        from et_miner.gpu.dispatch import resolved_kernel_variant

        monkeypatch.setenv("ET_MINER_KERNEL_VARIANT", "turbo")
        with pytest.raises(ValueError, match="ET_MINER_KERNEL_VARIANT"):
            resolved_kernel_variant()
