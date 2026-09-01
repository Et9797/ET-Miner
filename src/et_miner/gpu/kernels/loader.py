"""CUDA kernel loading, compilation cache, and shared launch helpers.

Kernel sources live as .cu files in the adjacent _src/ directory (shipped as
package data) and are compiled on first use via cupy.RawKernel, keyed by the
CUDA entry-point name. Also holds the per-device locks and grid-dimension
helpers shared by every kernel-wrapper module.
"""

from __future__ import annotations

import threading
from importlib import resources

from loguru import logger

# Per-device locks for thread-safe CUDA operations on each GPU independently.
# Allows true parallel kernel execution across GPUs (the old single global lock
# serialized ALL GPU work, making multi-GPU slower than single-GPU).
_cuda_device_locks: dict = {}
_cuda_locks_init = threading.Lock()


def _get_device_lock(device_id: int) -> threading.Lock:
    """Get or create a lock for a specific CUDA device."""
    if device_id not in _cuda_device_locks:
        with _cuda_locks_init:
            if device_id not in _cuda_device_locks:
                _cuda_device_locks[device_id] = threading.Lock()
    return _cuda_device_locks[device_id]


def _warn_result_truncation(n_actual: int, max_results: int, context: str = ""):
    """Warn or raise on result buffer truncation."""
    if n_actual > max_results:
        overflow_pct = (n_actual - max_results) / n_actual * 100
        msg = f"Result truncation: {n_actual:,} found but buffer={max_results:,} ({overflow_pct:.1f}% lost). {context}"
        if overflow_pct > 5:
            raise RuntimeError(msg + " Use allcounts path or increase max_results.")
        logger.warning(msg)
    return min(n_actual, max_results)


_MAX_GRID_X = 2_147_483_647  # 2^31 - 1


def _grid_dims(n_blocks):
    """Compute CUDA grid dimensions for n_blocks, using 2D grid if needed."""
    if n_blocks <= _MAX_GRID_X:
        return (int(n_blocks),)
    grid_y = (n_blocks + _MAX_GRID_X - 1) // _MAX_GRID_X
    if grid_y > 65535:
        raise ValueError(f"n_blocks={n_blocks} exceeds max 2D CUDA grid (2^31-1 × 65535)")
    return (int(_MAX_GRID_X), int(grid_y))


# CUDA entry point -> .cu file in _src/ (cache keys are the entry-point names)
_KERNEL_FILES: dict[str, str] = {
    "count_itemset_fused": "itemset_count.cu",
    "count_itemsets_batch": "itemset_count.cu",
    "count_pairs_fused_k2": "pairs_k2.cu",
    "count_itemsets_fused_k3plus": "k3plus_fused.cu",
    "count_k3plus_from_groups": "k3plus_fullyfused.cu",
    "count_k3plus_gpu_resident": "k3plus_gpu_resident.cu",
    "decode_candidates_gpu": "decode_candidates.cu",
    "count_pairs_k2_dense": "pairs_k2_dense.cu",
    "count_k3plus_dense": "k3plus_dense.cu",
    "compact_threshold": "compact_threshold.cu",
    "count_shared_tiled_dense": "shared_tiled.cu",
    "count_shared_tiled_fused": "shared_tiled.cu",
    "count_k3plus_sampled": "k3plus_sampled.cu",
    "count_k3plus_indirect": "k3plus_indirect.cu",
    "csr_count_range": "csr_warp.cu",
    "csr_count_gather": "csr_warp.cu",
    "csr_write_gather": "csr_warp.cu",
    "bitvec_extract_tids": "bitvec_extract_tids.cu",
    "fill_row_ids": "fill_row_ids.cu",
    "bootstrap_copy": "bootstrap_copy.cu",
    "csr_to_bitvec": "csr_to_bitvec.cu",
}

# Shared preludes prepended at NVRTC compile time: .cu file -> list of _src/
# snippets it needs. Keeps one copy of code that must stay identical across
# kernels (the candidate decode that mirrors decode.py::decode_k3plus_flat).
_KERNEL_PRELUDES: dict[str, list[str]] = {
    "k3plus_dense.cu": ["_decode_common.cu"],
    "csr_warp.cu": ["_decode_common.cu"],
}

_source_cache: dict[str, str] = {}
_kernel_cache: dict = {}
_kernel_cache_lock = threading.Lock()


def _read_src(cu_name: str) -> str:
    return (resources.files("et_miner.gpu.kernels") / "_src" / cu_name).read_text(encoding="utf-8")


def get_kernel_source(cu_name: str) -> str:
    """CUDA C source of a _src/*.cu file as compiled: its preludes
    (``_KERNEL_PRELUDES``) followed by the file itself (cached)."""
    if cu_name not in _source_cache:
        parts = [_read_src(p) for p in _KERNEL_PRELUDES.get(cu_name, [])]
        parts.append(_read_src(cu_name))
        _source_cache[cu_name] = "\n".join(parts)
    return _source_cache[cu_name]


def clear_kernel_cache():
    """Clear compiled kernel cache. Call after modifying kernel source."""
    global _kernel_cache
    with _kernel_cache_lock:
        _kernel_cache = {}


def get_cuda_kernel(name: str = "count_itemset_fused"):
    """Get compiled CUDA kernel, caching for reuse. Thread-safe."""
    import cupy as cp

    with _kernel_cache_lock:
        if name not in _kernel_cache:
            if name not in _KERNEL_FILES:
                raise ValueError(f"Unknown kernel: {name}")
            _kernel_cache[name] = cp.RawKernel(get_kernel_source(_KERNEL_FILES[name]), name)

    return _kernel_cache[name]


def get_popcount_kernel():
    """Get a popcount ElementwiseKernel that uses __popcll hardware intrinsic.

    This is the FASTEST way to count set bits on NVIDIA GPUs - it compiles directly
    to the native POPC instruction. Much faster than software bit-twiddling algorithms.

    Usage:
        kernel = get_popcount_kernel()
        popcounts = kernel(bitvecs.view(cp.uint64))  # Returns popcount per u64

    Returns:
        CuPy ElementwiseKernel that takes uint64 input and returns uint64 popcount.
    """
    import cupy as cp

    with _kernel_cache_lock:
        if "popcount_u64" not in _kernel_cache:
            _kernel_cache["popcount_u64"] = cp.ElementwiseKernel(
                "uint64 x", "uint64 y", "y = __popcll(x)", "popcount_u64"
            )
    return _kernel_cache["popcount_u64"]

