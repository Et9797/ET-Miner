"""Auto-dispatch for multi-GPU pair counting (k=2) and candidate counting (k>=3).

Thresholds calibrated from RTX 3090 4x benchmarks (2026-02).
Pair/candidate count is the universal heuristic — support thresholds are
dataset-dependent, but work count directly determines computational load.
"""

from math import comb

from loguru import logger

from et_miner import _env
from et_miner.backends import get_gpu_count


def _resolve_gpus(n_gpus: int | None, work: int, threshold: int, what: str) -> int:
    """How many devices this launch will actually use, logged.

    Two things were wrong here and they compounded.

    The route was chosen from the ambient device count and the caller's n_gpus
    was ignored, so `n_gpus=1` on a two-GPU box still landed on the fan-out
    kernels -- which additionally round-trip the whole bitvec matrix through
    host RAM, and which used to truncate results silently. A parameter that
    reads as a resource cap was in practice a correctness-relevant route
    selector the caller could not set. On a shared box it also meant a job told
    to use one card took both.

    And nothing recorded which way it went: this module had NO logging at all,
    while five sites downstream silently clamped with
    `min(n_gpus, available, ...)`. So a campaign that lost itemsets to a clamp
    on a 2-GPU box produced a log indistinguishable from the 1-GPU run that
    would have raised, and anyone reproducing it on one card got a clean run and
    concluded the data had changed.
    """
    available = get_gpu_count()
    budget = available if n_gpus is None else max(1, min(int(n_gpus), available))
    use_multi = budget > 1 and work >= threshold
    resolved = budget if use_multi else 1
    logger.debug(
        f"    dispatch {what}: {work:,} units, threshold {threshold:,}, "
        f"caller n_gpus={n_gpus if n_gpus is not None else 'ambient'}, "
        f"available={available} -> running on {resolved} device(s)"
    )
    return resolved


def resolved_kernel_variant() -> str:
    """ET_MINER_KERNEL_VARIANT with "auto" resolved to a concrete choice.

    "auto" currently means the shared/tiled kernel; this indirection is the
    single point to flip if the benchmark campaign favors legacy.
    """
    v = _env.kernel_variant()
    if v not in ("auto", "legacy", "shared"):
        raise ValueError(f"ET_MINER_KERNEL_VARIANT must be auto/legacy/shared, got {v!r}")
    return "shared" if v == "auto" else v

# Break-even threshold with ~20% safety margin.
# Calibrated from RTX 3090 4x scaling data:
#   6.4M pairs  -> 0.37x (overhead kills)
#   28.2M pairs -> 1.34x (break-even zone)
#   97.2M pairs -> 2.54x (meaningful scaling)
# Crossover ~15-20M pairs. UPDATE after Phase 2 ultra-low benchmark data.
PAIR_COUNT_THRESHOLD = 15_000_000


def should_use_multi_gpu(n_frequent_items: int, n_gpus: int | None = None) -> bool:
    n_pairs = comb(n_frequent_items, 2)
    return _resolve_gpus(n_gpus, n_pairs, PAIR_COUNT_THRESHOLD, "k=2") > 1


def dispatch_k2(bitvecs_gpu, freq_cols, n_u64s, min_count, n_gpus=None):
    """Dispatch k=2 pair counting to single or multi-GPU automatically.

    Drop-in replacement for the if/else block in apriori.py.
    Power users control GPU selection via CUDA_VISIBLE_DEVICES env var.

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s) with packed bitvectors.
        freq_cols: Sorted list of frequent item column indices.
        n_u64s: Number of uint64 words per bitvector.
        min_count: Minimum support count threshold.

    Returns:
        Tuple of (pairs, counts) — same as count_pairs_fused_k2().
    """
    from .kernels import count_pairs_fused_k2, count_pairs_fused_k2_multi_gpu

    n_freq = len(freq_cols)
    _n = _resolve_gpus(n_gpus, comb(n_freq, 2), PAIR_COUNT_THRESHOLD, "k=2")
    if _n > 1:
        return count_pairs_fused_k2_multi_gpu(bitvecs_gpu, freq_cols, n_u64s, min_count, _n)
    if resolved_kernel_variant() == "shared":
        from .kernels.shared_tiled import count_pairs_k2_shared_fused

        return count_pairs_k2_shared_fused(bitvecs_gpu, freq_cols, n_u64s, min_count)
    return count_pairs_fused_k2(bitvecs_gpu, freq_cols, n_u64s, min_count)


# K>=3 candidate threshold for multi-GPU.
# RTX 3090: overhead ~2ms per GPU. Break-even at ~50K candidates.
# UPDATE after Phase 4 benchmark data.
CANDIDATE_COUNT_THRESHOLD_K3 = 500_000


def should_use_multi_gpu_k3(n_candidates: int, n_gpus: int | None = None) -> bool:
    return _resolve_gpus(n_gpus, n_candidates, CANDIDATE_COUNT_THRESHOLD_K3, "k>=3") > 1


def dispatch_k3plus(bitvecs_gpu, candidates, n_u64s, min_count, n_gpus=None):
    """Dispatch k>=3 candidate counting to single or multi-GPU automatically.

    Drop-in replacement for count_itemsets_cuda() + Python filtering.

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s) with packed bitvectors.
        candidates: list of tuples of int column indices, e.g. [(0,1,3), (0,2,4)].
        n_u64s: Number of uint64 words per bitvector.
        min_count: Minimum support count threshold.

    Returns:
        Tuple of (frequent_candidates, counts) — only candidates meeting min_count.
    """
    from .kernels import (
        count_itemsets_fused_k3plus,
        count_itemsets_fused_k3plus_multi_gpu,
    )

    _n = _resolve_gpus(n_gpus, len(candidates), CANDIDATE_COUNT_THRESHOLD_K3, "k>=3")
    if _n > 1:
        return count_itemsets_fused_k3plus_multi_gpu(bitvecs_gpu, candidates, n_u64s, min_count, _n)
    return count_itemsets_fused_k3plus(bitvecs_gpu, candidates, n_u64s, min_count)


def dispatch_k3plus_fused(bitvecs_gpu, prev_frequent, k, n_u64s, min_count, n_gpus=None):
    """Fully-fused k>=3: candidate gen + count + filter all on GPU.

    Eliminates CPU-side candidate generation entirely. Prefix groups are built
    on CPU (O(n_frequent)), transferred to GPU, and candidates are generated
    on-the-fly in the kernel via triangular number inverse.

    Automatically routes to multi-GPU when beneficial.

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s) with packed bitvectors.
        prev_frequent: List of frequent (k-1)-itemsets as tuples of column indices.
        k: Current itemset size being generated.
        n_u64s: Number of uint64 words per bitvector.
        min_count: Minimum support count threshold.

    Returns:
        Tuple of (frequent_candidates, counts) — only candidates meeting min_count.
    """
    from .kernels import count_k3plus_fully_fused, count_k3plus_fully_fused_multi_gpu

    # Estimate candidate count for multi-GPU threshold
    prefix_groups: dict[tuple, int] = {}
    for itemset in prev_frequent:
        prefix = itemset[:-1]
        prefix_groups[prefix] = prefix_groups.get(prefix, 0) + 1
    n_cands = sum(g * (g - 1) // 2 for g in prefix_groups.values())

    _n = _resolve_gpus(n_gpus, n_cands, CANDIDATE_COUNT_THRESHOLD_K3, f"k={k}")
    if _n > 1:
        return count_k3plus_fully_fused_multi_gpu(bitvecs_gpu, prev_frequent, k, n_u64s, min_count, _n)
    if resolved_kernel_variant() == "shared":
        from .kernels.shared_tiled import count_k3plus_shared_fused

        return count_k3plus_shared_fused(bitvecs_gpu, prev_frequent, k, n_u64s, min_count)
    return count_k3plus_fully_fused(bitvecs_gpu, prev_frequent, k, n_u64s, min_count)


def dispatch_k2_gpu_resident(bitvecs_gpu, freq_cols_gpu, n_u64s, min_count, n_gpus=None):
    """GPU-resident k=2 dispatch. Auto-routes to multi-GPU when beneficial.

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s) with packed bitvectors.
        freq_cols_gpu: CuPy int32 array of frequent item column indices (sorted, in VRAM).
        n_u64s: Number of uint64 words per bitvector.
        min_count: Minimum support count threshold.

    Returns:
        Tuple of (pair_itemsets_gpu, counts_gpu) CuPy arrays in VRAM,
        or (None, None) if no frequent pairs.
    """
    from .kernels import (
        count_pairs_fused_k2_gpu_resident,
        count_pairs_fused_k2_gpu_resident_multi_gpu,
    )

    n_freq = len(freq_cols_gpu)
    _n = _resolve_gpus(n_gpus, comb(n_freq, 2), PAIR_COUNT_THRESHOLD, "k=2 gpu-resident")
    if _n > 1:
        return count_pairs_fused_k2_gpu_resident_multi_gpu(
            bitvecs_gpu, freq_cols_gpu, n_u64s, min_count, _n
        )
    return count_pairs_fused_k2_gpu_resident(bitvecs_gpu, freq_cols_gpu, n_u64s, min_count)


def dispatch_k3plus_gpu_resident(bitvecs_gpu, prev_freq_gpu, n_u64s, min_count, n_gpus=None):
    """GPU-resident k>=3 dispatch. Auto-routes to multi-GPU when beneficial.

    Estimates candidate count from prefix groups, routes to multi-GPU
    when above threshold and multiple GPUs are available.

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s) with packed bitvectors.
        prev_freq_gpu: CuPy array (n_freq, k_prev) of sorted frequent itemsets in VRAM.
        n_u64s: Number of uint64 words per bitvector.
        min_count: Minimum support count threshold.

    Returns:
        Tuple of (freq_itemsets_gpu, counts_gpu) CuPy arrays in VRAM,
        or (None, None) if no frequent itemsets.
    """
    from .kernels import (
        count_k3plus_gpu_resident,
        count_k3plus_gpu_resident_multi_gpu,
        build_prefix_groups_gpu,
    )

    # Estimate candidate count for multi-GPU threshold
    _, _, _, total_candidates = build_prefix_groups_gpu(prev_freq_gpu)

    _n = _resolve_gpus(n_gpus, int(total_candidates), CANDIDATE_COUNT_THRESHOLD_K3,
                       "k>=3 gpu-resident")
    if _n > 1:
        return count_k3plus_gpu_resident_multi_gpu(bitvecs_gpu, prev_freq_gpu, n_u64s, min_count, _n)
    return count_k3plus_gpu_resident(bitvecs_gpu, prev_freq_gpu, n_u64s, min_count)
