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


def _warn_result_truncation(
    n_actual: int,
    max_results: int,
    context: str = "",
    *,
    k: int | None = None,
    knob: str = "max_results",
):
    """Raise on result-buffer overflow. Never truncate silently.

    This used to tolerate an overflow of up to 5% with a `logger.warning` and
    return the truncated count. There is no percentage of silently-lost frequent
    itemsets that is acceptable for a miner whose contract is exactness, and the
    tolerance was worse than it looks on three counts:

      - the kernels append via atomicAdd, so the DROPPED SET IS
        NON-DETERMINISTIC. Three repeats at 3% overflow kept three different
        sets, pairwise differing by 30-32 itemsets. Those routes disagreed with
        the row-split path, with the CPU tiers, and with themselves run twice --
        so the tier-equivalence chain CLAUDE.md mandates could not hold at that
        scale even against itself.
      - the truncated level feeds candidate generation, so the loss compounds at
        every deeper K.
      - a silent 60% loss cannot be caught by a log grep, and 5% of a 10M-result
        buffer is 500,000 itemsets.

    The raise lands hours into a run, so the message has to be actionable: it
    names the level, the overflow, the exact knob to raise, and the K to resume
    from. See `resume_from_k` on the row-split miner.
    """
    if n_actual <= max_results:
        return n_actual

    overflow_pct = (n_actual - max_results) / n_actual * 100
    where = f" at K={k}" if k is not None else ""
    # Name ONLY remedies reachable on the route that raises. These kernels are
    # reached from _apriori_from_bitvecs / _apriori_from_bitvecs_gpu_resident,
    # where `max_results` is not a public apriori() parameter and both
    # `output_dir` and `resume_from_k` are refused by _validate_route_support --
    # so an earlier version of this message advised two impossible things and
    # one knob the caller cannot set.
    raise RuntimeError(
        f"Result truncation{where}: {n_actual:,} frequent itemsets found but the "
        f"result buffer holds {max_results:,} ({overflow_pct:.1f}% would be lost, "
        f"non-deterministically, because the kernels append via atomicAdd). "
        f"{context} "
        f"Re-run on the row-split miner, which sizes exactly to the survivor "
        f"count and has no ceiling: pass n_gpus>1 or prune_equal_support=True to "
        f"apriori(). That route also supports output_dir, so each level is "
        f"flushed as it completes. Otherwise lower max_length"
        + (f" (this is K={k})" if k is not None else "")
        + "."
    )


#: Every K>=3 kernel caches the candidate in `__shared__ int s_items[64]`,
#: immediately followed by `__shared__ unsigned long long warp_sums[8]`.
#:
#: The three GROUP kernels fill s_items under `threadIdx.x < prefix_len &&
#: threadIdx.x < 62`, and thread 0 then separately writes s_items[prefix_len]
#: and s_items[prefix_len+1] -- which at prefix_len == 62 land exactly on slots
#: 62 and 63. So they are correct to K=64 and break at K=65, where slot 62 is
#: never written but IS read as a column index, and s_items[prefix_len+1]
#: writes index 64: one past the array, into warp_sums, corrupting the block
#: reduction too.
#:
#: count_itemsets_fused_k3plus caches the whole itemset with no separate suffix
#: write, and its guard `threadIdx.x < k && threadIdx.x < 62` leaves slots 62-63
#: unwritten while the AND loop reads s_items[0..k-1] -- so it is correct only
#: to K=62, as its own comment says.
#:
#: 62 is therefore the uniform host-side cap. It costs nothing: reaching K=63
#: requires a frequent 62-itemset, i.e. all 2**62 of its subsets frequent, which
#: no run completes. Enforcing it on the host is the whole fix -- do NOT widen
#: the device guards, which buys unreachable capacity and leaves K>=65 silently
#: corrupt while making the code look repaired.
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
    """Raise before a wrapper aliases GPU arrays that do not share a device.

    The FIRST keyword is the home device; every later one must match it. Pass
    them by keyword because the keyword is what the error names, and pass only
    arrays the caller supplied -- arrays derived from those follow by
    construction, and naming a derived one points the error at an array the
    caller never chose.

    Two things depend on the inputs living together. The multi-GPU wrappers
    alias the caller's arrays on one device and upload copies to the others.
    The three gpu-resident entry points that call THIS function pin their
    launch to the home device while helpers like `build_prefix_groups_gpu`
    follow their own input. Mix the devices and the kernel is handed pointers
    from two cards -- CUDA_ERROR_ILLEGAL_ADDRESS, which poisons the context
    process-wide rather than costing one level.

    That pinning is NOT a module-wide property, and reading it as one is how
    the K=2 twin of N20 shipped. `k2.py::count_pairs_fused_k2`,
    `k3plus.py::count_itemsets_fused_k3plus`, `k3plus.py::count_k3plus_fully_fused`
    and `shared_tiled.py::count_pairs_k2_shared_fused` all still allocate and
    launch on the AMBIENT device; the last of those is what `gpu/dispatch.py`
    selects by default when `ET_MINER_KERNEL_VARIANT` is unset, and both it and
    the `k2.py` one were measured aborting with `cudaErrorIllegalAddress` from
    ambient device 0 with bitvecs on device 1. They take host lists rather than
    device arrays for their second argument, so they have no home to infer and
    this function cannot guard them -- which is the reason they may be deferred
    and equally the reason this docstring may not generalise over them.

    It raises instead of transferring: a mixed-device call is a caller bug, and
    repairing it with a hidden copy makes it unattributable. That is this
    module's rule, not a project-wide one -- `gpu/csr_build.py` deliberately
    repairs with `cp.asarray`, because it takes an explicit target `device_id`
    ("put it here") where these wrappers infer home from the data.

    Callers must invoke this ABOVE any routing. Below a
    `if n_gpus <= 1: return ...` it never runs on a single-GPU host, which is
    the placement mistake `core/apriori.py::_validate_route_support` already
    records in its own comment.
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
    "k3plus_fullyfused.cu": ["_decode_common.cu"],
    "k3plus_gpu_resident.cu": ["_decode_common.cu"],
    "decode_candidates.cu": ["_decode_common.cu"],
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

