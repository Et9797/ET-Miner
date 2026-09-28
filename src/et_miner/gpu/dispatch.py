"""Kernel-variant selection for the row-split miner's dense counting."""

from et_miner import _env


def resolved_kernel_variant() -> str:
    """ET_MINER_KERNEL_VARIANT with "auto" resolved to a concrete choice.

    "auto" currently means the shared/tiled kernel; this indirection is the
    single point to flip if the benchmark campaign favors legacy.
    """
    v = _env.kernel_variant()
    if v not in ("auto", "legacy", "shared"):
        raise ValueError(f"ET_MINER_KERNEL_VARIANT must be auto/legacy/shared, got {v!r}")
    return "shared" if v == "auto" else v
