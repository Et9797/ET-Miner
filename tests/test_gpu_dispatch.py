"""Tests for gpu.dispatch — the kernel-variant resolver (CPU-only decision logic)."""


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
