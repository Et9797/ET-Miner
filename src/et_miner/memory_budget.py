"""GPU memory budget estimation for billion-scale frequent itemset mining.

Prevents hidden OOM crashes from CuPy operations that allocate temporary
arrays without warning. The K=8 crash was caused by boolean indexing
creating an 85GB hidden prefix-sum array.

Key insight: CuPy boolean indexing `array[bool_mask]` allocates:
  - The bool mask itself: N × 1 byte
  - Hidden int64 prefix-sum for output positions: N × 8 bytes

For 10B elements, that's 10GB mask + 80GB prefix-sum = 90GB hidden allocation.
"""

import warnings
import numpy as np
from loguru import logger

__all__ = [
    'estimate_boolean_index_memory',
    'check_vram_budget',
    'safe_threshold_filter',
    'set_mempool_limit',
    'get_mempool_stats',
]


def estimate_boolean_index_memory(n_elements: int) -> int:
    """Estimate peak memory for array[bool_mask] operation.

    CuPy boolean indexing creates hidden temporaries:
    1. Bool mask: n_elements × 1 byte
    2. Prefix-sum array (int64): n_elements × 8 bytes

    The prefix-sum is used to determine output positions — it's the
    "hidden" allocation that caused the K=8 85GB OOM crash.

    Args:
        n_elements: Number of elements in the source array.

    Returns:
        Estimated peak memory in bytes for the boolean indexing operation.

    Example:
        >>> estimate_boolean_index_memory(10_680_000_000)  # K=8 candidates
        96120000000  # ~96 GB for mask + prefix-sum alone
    """
    mask_bytes = n_elements * 1        # bool mask
    prefix_sum_bytes = n_elements * 8  # int64 prefix-sum for output positions
    return mask_bytes + prefix_sum_bytes


def check_vram_budget(required_bytes: int, safety_margin: float = 0.1) -> None:
    """Raise if required memory exceeds available VRAM.

    Call this BEFORE launching operations that might OOM. Better to fail
    fast with a clear message than crash mid-kernel with cryptic errors.

    Args:
        required_bytes: Estimated memory requirement in bytes.
        safety_margin: Extra headroom as fraction (default 10%).

    Raises:
        MemoryError: If required memory exceeds available VRAM.

    Example:
        >>> check_vram_budget(100_000_000_000)  # 100 GB
        MemoryError: Operation requires 100.0GB + 10% margin,
        but only 85.0GB available. Consider: (1) chunked processing,
        (2) CPU fallback, (3) free unused arrays.
    """
    import cupy as cp

    mempool = cp.get_default_memory_pool()
    used = mempool.used_bytes()
    total = cp.cuda.Device().mem_info[1]  # total VRAM
    available = total - used
    required_with_margin = int(required_bytes * (1 + safety_margin))

    if required_with_margin > available:
        raise MemoryError(
            f"Operation requires {required_bytes/1e9:.1f}GB + {safety_margin*100:.0f}% margin, "
            f"but only {available/1e9:.1f}GB available. "
            f"Consider: (1) chunked processing, (2) CPU fallback, (3) free unused arrays."
        )


def safe_threshold_filter(counts_gpu, threshold: int, max_gpu_elements: int = 100_000_000):
    """Threshold filter with automatic CPU fallback for large arrays.

    This is the CORRECT pattern for filtering billion-scale count arrays.
    For small arrays (<100M), GPU filtering is fine. For large arrays,
    CPU filtering avoids the hidden 8× prefix-sum allocation.

    The K=8 fix used this exact pattern: transfer to CPU, filter with NumPy,
    return indices. No hidden 85GB allocation.

    Args:
        counts_gpu: CuPy array of support counts.
        threshold: Minimum count threshold.
        max_gpu_elements: Above this, use CPU fallback (default 100M).

    Returns:
        Tuple of (indices, filtered_counts) as NumPy arrays.

    Example:
        >>> # Safe for 10B elements — uses CPU path
        >>> indices, counts = safe_threshold_filter(huge_counts_gpu, min_support)
    """
    import cupy as cp

    n = len(counts_gpu)

    if n > max_gpu_elements:
        # CPU path — no hidden GPU temporaries
        counts_cpu = counts_gpu.get()

        # Free GPU memory immediately
        del counts_gpu
        cp.get_default_memory_pool().free_all_blocks()

        # Filter on CPU (2TB RAM has plenty of headroom)
        mask = counts_cpu >= threshold
        indices = np.where(mask)[0]
        filtered_counts = counts_cpu[indices]

        return indices, filtered_counts
    else:
        # GPU path — safe for small arrays
        mask = counts_gpu >= threshold
        indices_gpu = cp.where(mask)[0]
        indices = indices_gpu.get()
        filtered_counts = counts_gpu[indices_gpu].get()

        return indices, filtered_counts


def set_mempool_limit(limit_gb: float = None) -> None:
    """Set a hard limit on CuPy's GPU memory pool.

    Without this, CuPy will try to allocate until CUDA OOM.
    Setting a limit provides earlier, clearer failure.

    Args:
        limit_gb: Memory limit in GB. If None, uses 90% of total VRAM.

    Example:
        >>> set_mempool_limit(130)  # Cap at 130GB on H200
    """
    import cupy as cp

    mempool = cp.get_default_memory_pool()

    if limit_gb is None:
        total = cp.cuda.Device().mem_info[1]
        limit_bytes = int(total * 0.9)  # 90% of total VRAM
    else:
        limit_bytes = int(limit_gb * 1e9)

    mempool.set_limit(size=limit_bytes)

    logger.info(f"[memory_budget] Set mempool limit to {limit_bytes/1e9:.1f}GB")


def get_mempool_stats() -> dict:
    """Get current GPU memory pool statistics.

    Useful for debugging memory issues and understanding allocation patterns.

    Returns:
        Dict with used_bytes, total_bytes, limit_bytes, free_bytes.
    """
    import cupy as cp

    mempool = cp.get_default_memory_pool()
    total = cp.cuda.Device().mem_info[1]
    used = mempool.used_bytes()
    limit = mempool.get_limit()

    return {
        'used_bytes': used,
        'used_gb': used / 1e9,
        'total_bytes': total,
        'total_gb': total / 1e9,
        'limit_bytes': limit if limit > 0 else total,
        'limit_gb': (limit if limit > 0 else total) / 1e9,
        'free_bytes': total - used,
        'free_gb': (total - used) / 1e9,
    }
