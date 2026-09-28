"""CUDA kernel loading, compilation cache, and shared launch helpers.

Kernel sources live as .cu files in the adjacent _src/ directory (shipped as
package data) and are compiled on first use via cupy.RawKernel, keyed by the
CUDA entry-point name. Also holds the per-device locks and grid-dimension
helpers shared by every kernel-wrapper module.
"""

from __future__ import annotations

import threading
from importlib import resources


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


#: Host-side cap on K for every K>=3 kernel. The group kernels cache the
#: candidate in `__shared__ int s_items[64]` (prefix under
#: `threadIdx.x < prefix_len && threadIdx.x < 62`, then the two suffixes at
#: s_items[prefix_len] and s_items[prefix_len+1]) and `shared_tiled.cu` stages
#: the prefix in `s_pref[62]`, so every kernel is exact to K=64; at K=65 a
#: prefix slot is read without being written and the second suffix lands one
#: past the array, in the block reduction. The cap sits two below that bound.
#: Reaching K=63 needs a frequent 62-itemset, which no run completes. The cap
#: is enforced here, on the host: widening the device guards would buy
#: unreachable capacity and leave K>=65 corrupt.
MAX_SUPPORTED_K = 62


def _assert_k_supported(k: int | None, context: str = "") -> None:
    """Raise before launching a K>=3 kernel that would read uninitialised shared memory."""
    if k is not None and k > MAX_SUPPORTED_K:
        where = f" ({context})" if context else ""
        raise ValueError(
            f"K={k} exceeds the kernel cap of {MAX_SUPPORTED_K}{where}: the K>=3 "
            "kernels cache the candidate in a fixed 64-slot shared array, and "
            "beyond this they read an uninitialised slot as a column index and "
            "write one past the array into the block reduction."
        )


def _assert_home(context: str, **arrays) -> None:
    """Raise before a wrapper launches on GPU arrays that do not share a device.

    The FIRST keyword is the home device; every later one must match it. Pass
    them by keyword because the keyword is what the error names. Callers pin
    their launch to the home device, so the check also catches a host array
    reaching a device-only path, with a readable error instead of an
    AttributeError. Mix the devices and the kernel is handed pointers from two
    cards -- CUDA_ERROR_ILLEGAL_ADDRESS, which poisons the context
    process-wide.

    It raises instead of transferring: a mixed-device call is a caller bug, and
    repairing it with a hidden copy makes it unattributable.
    """
    home_name = home_id = None
    for name, arr in arrays.items():
        dev = getattr(arr, "device", None)
        dev_id = getattr(dev, "id", None)
        if dev_id is None:
            # NumPy 2 gives ndarray.device == "cpu"; NumPy 1 has no .device at
            # all. Either way this is a host array reaching a device-only path,
            # and a validator should say that rather than AttributeError.
            raise ValueError(
                f"{context}: {name} is not a CuPy array resident on a CUDA "
                f"device (got {type(arr).__name__}, device={dev!r})."
            )
        dev_id = int(dev_id)
        if home_name is None:
            home_name, home_id = name, dev_id
        elif dev_id != home_id:
            raise ValueError(
                f"{context}: {name} is on device {dev_id} but {home_name} is "
                f"on device {home_id}. All inputs must be resident on one "
                "device; this route aliases them together on it."
            )



def _assert_rank(context: str, **arrays) -> None:
    """Raise before a kernel indexes a device array of the wrong rank.

    Each keyword is `name=(array, expected_ndim)`; the keyword is what the
    error names.
    """
    for name, spec in arrays.items():
        arr, want = spec
        got = getattr(arr, "ndim", None)
        if got is None:
            raise ValueError(
                f"{context}: {name} has no `.ndim` (got {type(arr).__name__}); "
                "this route takes device arrays, not host sequences."
            )
        if int(got) != int(want):
            raise ValueError(
                f"{context}: {name} must be {want}-D, got {got}-D with shape "
                f"{tuple(getattr(arr, 'shape', ()))}."
            )


def _assert_dtype(context: str, **arrays) -> None:
    """Raise before a kernel reinterprets a device array of the wrong dtype.

    Each keyword is `name=(array, expected_dtype)`, the dtype written as the
    string the error should print. It raises rather than casting, per the rule
    stated on `_assert_home`, and because nothing else would catch it: the
    kernels read the bitvecs through an `unsigned long long*`, so a uint32
    array of the right shape counts plausible garbage without an exception.
    """
    for name, spec in arrays.items():
        arr, want = spec
        got = getattr(arr, "dtype", None)
        if got is None:
            raise ValueError(
                f"{context}: {name} has no `.dtype` (got {type(arr).__name__}); "
                "this route takes device arrays, not host sequences."
            )
        if got != want:
            raise ValueError(
                f"{context}: {name} must be {want}, got {got}. This route raises "
                "rather than casting -- see `_assert_home` for why."
            )


def _assert_bitvecs(context: str, bitvecs_gpu, **same_device) -> None:
    """The input guards every bitvec kernel wrapper runs before it launches.

    ``bitvecs_gpu`` must be a 2-D uint64 CuPy array; any ``same_device`` array
    must live on its device, which is the device the wrapper launches on.
    """
    _assert_home(context, bitvecs_gpu=bitvecs_gpu, **same_device)
    _assert_rank(context, bitvecs_gpu=(bitvecs_gpu, 2))
    _assert_dtype(context, bitvecs_gpu=(bitvecs_gpu, "uint64"))


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
    "count_pairs_k2_dense": "pairs_k2_dense.cu",
    "count_k3plus_dense": "k3plus_dense.cu",
    "compact_threshold": "compact_threshold.cu",
    "count_shared_tiled_dense": "shared_tiled.cu",
    "count_shared_tiled_fused": "shared_tiled.cu",
    "csr_count_range": "csr_warp.cu",
    "csr_count_gather": "csr_warp.cu",
    "csr_write_gather": "csr_warp.cu",
    "bitvec_extract_tids": "bitvec_extract_tids.cu",
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


def column_popcounts(bitvecs_gpu, max_temp_bytes: int = 1 << 28):
    """Per-column set-bit counts of an (n_cols, n_u64s) bitvec matrix (device int64).

    Popcounts a block of columns at a time, so the uint64 temporary stays
    within ``max_temp_bytes`` (at least one column) instead of matching the
    whole matrix. Allocates on the current device.
    """
    import cupy as cp

    n_cols, n_u64s = bitvecs_gpu.shape
    kernel = get_popcount_kernel()
    cols_per_block = max(1, max_temp_bytes // max(1, n_u64s * 8))
    out = cp.empty(n_cols, dtype=cp.int64)
    for start in range(0, n_cols, cols_per_block):
        end = min(start + cols_per_block, n_cols)
        out[start:end] = kernel(bitvecs_gpu[start:end].view(cp.uint64)).sum(axis=1, dtype=cp.int64)
    return out
