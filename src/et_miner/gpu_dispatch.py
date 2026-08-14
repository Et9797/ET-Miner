"""Auto-dispatch for multi-GPU pair counting (k=2) and candidate counting (k>=3).

Thresholds calibrated from RTX 3090 4x benchmarks (2026-02).
Pair/candidate count is the universal heuristic — support thresholds are
dataset-dependent, but work count directly determines computational load.
"""

from math import comb

from .gpu_utils import get_gpu_count

# Break-even threshold with ~20% safety margin.
# Calibrated from RTX 3090 4x scaling data:
#   6.4M pairs  -> 0.37x (overhead kills)
#   28.2M pairs -> 1.34x (break-even zone)
#   97.2M pairs -> 2.54x (meaningful scaling)
# Crossover ~15-20M pairs. UPDATE after Phase 2 ultra-low benchmark data.
PAIR_COUNT_THRESHOLD = 15_000_000


def should_use_multi_gpu(n_frequent_items: int) -> bool:
    n_pairs = comb(n_frequent_items, 2)
    return get_gpu_count() > 1 and n_pairs >= PAIR_COUNT_THRESHOLD


def dispatch_k2(bitvecs_gpu, freq_cols, n_u64s, min_count):
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
    from .cuda_kernels import count_pairs_fused_k2, count_pairs_fused_k2_multi_gpu

    n_freq = len(freq_cols)
    if should_use_multi_gpu(n_freq):
        return count_pairs_fused_k2_multi_gpu(bitvecs_gpu, freq_cols, n_u64s, min_count, get_gpu_count())
    return count_pairs_fused_k2(bitvecs_gpu, freq_cols, n_u64s, min_count)


# K>=3 candidate threshold for multi-GPU.
# RTX 3090: overhead ~2ms per GPU. Break-even at ~50K candidates.
# UPDATE after Phase 4 benchmark data.
CANDIDATE_COUNT_THRESHOLD_K3 = 500_000


def should_use_multi_gpu_k3(n_candidates: int) -> bool:
    return get_gpu_count() > 1 and n_candidates >= CANDIDATE_COUNT_THRESHOLD_K3


def dispatch_k3plus(bitvecs_gpu, candidates, n_u64s, min_count):
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
    from .cuda_kernels import (
        count_itemsets_fused_k3plus,
        count_itemsets_fused_k3plus_multi_gpu,
    )

    if should_use_multi_gpu_k3(len(candidates)):
        return count_itemsets_fused_k3plus_multi_gpu(bitvecs_gpu, candidates, n_u64s, min_count, get_gpu_count())
    return count_itemsets_fused_k3plus(bitvecs_gpu, candidates, n_u64s, min_count)


def dispatch_k3plus_fused(bitvecs_gpu, prev_frequent, k, n_u64s, min_count):
    """Fully-fused k>=3: candidate gen + count + filter all on GPU.

    Eliminates _generate_candidates_int() entirely. Prefix groups are built
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
    from .cuda_kernels import count_k3plus_fully_fused, count_k3plus_fully_fused_multi_gpu

    # Estimate candidate count for multi-GPU threshold
    prefix_groups: dict[tuple, int] = {}
    for itemset in prev_frequent:
        prefix = itemset[:-1]
        prefix_groups[prefix] = prefix_groups.get(prefix, 0) + 1
    n_cands = sum(g * (g - 1) // 2 for g in prefix_groups.values())

    if n_cands >= CANDIDATE_COUNT_THRESHOLD_K3 and get_gpu_count() > 1:
        return count_k3plus_fully_fused_multi_gpu(bitvecs_gpu, prev_frequent, k, n_u64s, min_count, get_gpu_count())
    return count_k3plus_fully_fused(bitvecs_gpu, prev_frequent, k, n_u64s, min_count)


SAMPLED_PREFILTER_THRESHOLD = 1_000_000  # >1M candidates = worth sampling


def dispatch_k3plus_sampled(bitvecs_gpu, prev_frequent, k, n_u64s, min_count, sample_stride=None):
    """Sampled popcount pre-filter for K>=3: reject ~68% of candidates cheaply.

    Two-phase: sampled pass rejects ~68% of candidates, then indirect kernel
    exact-recounts only survivors. Falls back to fully-fused when below threshold.

    Adaptive stride: at high K (sparse bitvectors), use smaller stride for more
    accurate estimates. Rate-distortion tradeoff — Shannon in a CUDA kernel.

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s) with packed bitvectors.
        prev_frequent: List of frequent (k-1)-itemsets as tuples of column indices.
        k: Current itemset size being generated.
        n_u64s: Number of uint64 words per bitvector.
        min_count: Minimum support count threshold.
        sample_stride: Sample every Nth u64 word. None = adaptive based on K.

    Returns:
        Tuple of (frequent_candidates, counts) — only candidates meeting min_count.
    """
    from .cuda_kernels import count_k3plus_sampled_prefilter, count_k3plus_fully_fused

    # Adaptive stride: dense K=3 benefits from aggressive sampling,
    # sparse K=7+ needs more samples for statistical reliability.
    if sample_stride is None:
        if k <= 3:
            sample_stride = 8  # 12.5% sample — plenty for dense bitvecs
        elif k <= 5:
            sample_stride = 4  # 25% sample — moderate density
        else:
            sample_stride = 2  # 50% sample — sparse, need accuracy

    # Estimate candidate count
    prefix_groups: dict[tuple, int] = {}
    for itemset in prev_frequent:
        prefix = itemset[:-1]
        prefix_groups[prefix] = prefix_groups.get(prefix, 0) + 1
    n_cands = sum(g * (g - 1) // 2 for g in prefix_groups.values())

    if n_cands < SAMPLED_PREFILTER_THRESHOLD or n_u64s < sample_stride:
        # Not enough candidates or bitvecs too small for sampling
        return count_k3plus_fully_fused(bitvecs_gpu, prev_frequent, k, n_u64s, min_count)

    return count_k3plus_sampled_prefilter(bitvecs_gpu, prev_frequent, k, n_u64s, min_count, sample_stride=sample_stride)


def dispatch_k2_gpu_resident(bitvecs_gpu, freq_cols_gpu, n_u64s, min_count):
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
    from .cuda_kernels import (
        count_pairs_fused_k2_gpu_resident,
        count_pairs_fused_k2_gpu_resident_multi_gpu,
    )

    n_freq = len(freq_cols_gpu)
    if should_use_multi_gpu(n_freq):
        return count_pairs_fused_k2_gpu_resident_multi_gpu(
            bitvecs_gpu, freq_cols_gpu, n_u64s, min_count, get_gpu_count()
        )
    return count_pairs_fused_k2_gpu_resident(bitvecs_gpu, freq_cols_gpu, n_u64s, min_count)


def dispatch_k3plus_gpu_resident(bitvecs_gpu, prev_freq_gpu, n_u64s, min_count):
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
    from .cuda_kernels import (
        count_k3plus_gpu_resident,
        count_k3plus_gpu_resident_multi_gpu,
        build_prefix_groups_gpu,
    )

    # Estimate candidate count for multi-GPU threshold
    _, _, _, total_candidates = build_prefix_groups_gpu(prev_freq_gpu)

    if total_candidates >= CANDIDATE_COUNT_THRESHOLD_K3 and get_gpu_count() > 1:
        return count_k3plus_gpu_resident_multi_gpu(bitvecs_gpu, prev_freq_gpu, n_u64s, min_count, get_gpu_count())
    return count_k3plus_gpu_resident(bitvecs_gpu, prev_freq_gpu, n_u64s, min_count)
