"""Single source of truth for optional-backend detection (CuPy / Rust).

Every module that needs to know whether the Rust extension or CuPy is
available imports from here instead of probing on its own. This module
imports nothing from et_miner, so it can be imported from anywhere in the
package with no cycle risk.

The module-scope try-imports below are the only places in the package where
`cupy` or `et_miner_rust` is imported unguarded at module scope; call sites
that need the cupy module object itself do a local `import cupy as cp` after
checking the flags here (re-importing an already-loaded module is a dict
lookup).
"""

from __future__ import annotations

# The one place the extension's build recipe is written in source. It was a
# literal string at each warning site, and that is how a spelling change reached
# five call sites and missed bench/setup_box.sh entirely.
#
# `-m rust_ext/Cargo.toml` from the repo root rather than `cd rust_ext`: with a
# pyproject beside its Cargo.toml maturin reads rust_ext's own metadata, while a
# cwd inside rust_ext makes uv discover it as a separate project and build a
# second venv there. Running from the root is what removes that failure rather
# than documenting against it.
#
# In a conda-ambient shell prefix this with `env -u CONDA_PREFIX`: uv exports
# VIRTUAL_ENV and maturin refuses when both are set. Unsetting is required --
# `CONDA_PREFIX=` still reads as set. bench/setup_box.sh carries the prefix
# because it runs unattended in whatever shell an operator has.
BUILD_COMMAND = "uv run maturin develop --release -m rust_ext/Cargo.toml"

try:
    import cupy as _cp

    CUPY_INSTALLED = True
    _CUPY_VERSION: str | None = _cp.__version__
except ImportError:
    _cp = None  # type: ignore[assignment]
    CUPY_INSTALLED = False
    _CUPY_VERSION = None

try:
    import et_miner_rust as _rust_ext

    RUST_INSTALLED = True
    _RUST_VERSION: str | None = getattr(_rust_ext, "__version__", "unknown")
except ImportError:
    _rust_ext = None  # type: ignore[assignment]
    RUST_INSTALLED = False
    _RUST_VERSION = None


def has_cupy() -> bool:
    """True iff CuPy is installed AND at least one CUDA device is usable."""
    if not CUPY_INSTALLED:
        return False
    try:
        return _cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def get_gpu_count() -> int:
    """Number of visible CUDA devices (0 when CuPy is absent or CUDA fails)."""
    if not CUPY_INSTALLED:
        return 0
    try:
        return _cp.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


def get_cupy_version() -> str | None:
    return _CUPY_VERSION


def has_rust_extension() -> bool:
    return RUST_INSTALLED


def get_rust_version() -> str | None:
    return _RUST_VERSION


def get_rust_ext():
    """The et_miner_rust module, or None when the extension is not built.

    Canonical accessor — do not `import et_miner_rust` elsewhere in the
    package.
    """
    return _rust_ext


def rust_has(attr: str) -> bool:
    """Capability probe for optional symbols in the Rust extension.

    Newer extension builds may expose functions older wheels lack; call
    sites probe per-symbol instead of assuming a version.
    """
    return _rust_ext is not None and hasattr(_rust_ext, attr)


__all__ = [
    "BUILD_COMMAND",
    "CUPY_INSTALLED",
    "RUST_INSTALLED",
    "has_cupy",
    "get_gpu_count",
    "get_cupy_version",
    "has_rust_extension",
    "get_rust_version",
    "get_rust_ext",
    "rust_has",
]
