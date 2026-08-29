"""Type safety utilities for billion-scale frequent itemset mining.

Prevents int32 overflow bugs that caused K=8 mining crashes.
The threshold is 2^31-1 (2,147,483,647) — any index, offset, or count
exceeding this MUST use int64.

Key insight: Python int64 uploads do NOT protect C kernels expecting int32.
The CUDA kernel signature must match: `const long long*` for int64 data.
"""

import warnings
import numpy as np

__all__ = [
    "INT32_MAX",
    "safe_offset_dtype",
    "assert_fits_int32",
    "warn_if_large",
]

INT32_MAX = 2**31 - 1  # 2,147,483,647


def safe_offset_dtype(max_value: int, gpu: bool = False):
    """Return int32 or int64 dtype (NumPy or CuPy) based on max_value vs INT32_MAX."""
    if gpu:
        import cupy as cp

        return cp.int64 if max_value > INT32_MAX else cp.int32
    return np.int64 if max_value > INT32_MAX else np.int32


def assert_fits_int32(value: int, name: str) -> None:
    """Raise OverflowError if value exceeds INT32_MAX — call before int32 CUDA kernel params."""
    if value > INT32_MAX:
        raise OverflowError(f"{name}={value:,} exceeds int32 max ({INT32_MAX:,}). Use int64 or chunk the operation.")


def warn_if_large(value: int, threshold: int, name: str) -> None:
    """Emit RuntimeWarning if value exceeds threshold — early int32 overflow detection."""
    if value > threshold:
        warnings.warn(
            f"{name}={value:,} is large (>{threshold:,}). Monitor for potential int32 overflow.",
            RuntimeWarning,
            stacklevel=2,
        )
