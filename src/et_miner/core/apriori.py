"""Apriori algorithm implementation for association rule mining.

Fast and memory-efficient implementation using boolean matrix representation and
vectorized operations, with optional GPU acceleration via CUDA kernels.

Support counting uses column bitwise AND operations:

    support({A,B}) = (col("A") & col("B")).sum()

Key ideas:
1. Boolean matrix representation (not TID-lists)
2. Column bitwise AND operations for support (no Python loops in counting)
3. GPU-ready via CUDA bitvector kernels (CuPy)

The GPU inner loops live in et_miner.gpu (mining, row_split); parquet flush
and GCS upload in et_miner.io; candidate generation in .candidates. This
module keeps parameter validation, the CPU mining loop, and the routing in
apriori() that picks a tier.
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Callable
from math import comb
from typing import Any, Literal

import polars as pl
from loguru import logger

from et_miner.gpu.density import validate_sparse_from_k

from .candidates import _generate_candidates
from .matrix import (
    build_boolean_matrix,
    count_support_batched,
)
from .profiling import ProfilingSession
from .result import (
    _build_result_df,
    _empty_result,
    _min_count,
)

def _warn_complexity(
    n_frequent_items: int,
    min_support: float,
) -> None:
    """Warn user if computation might be expensive."""
    # Estimate k=2 candidates (n choose 2)
    n_pairs = comb(n_frequent_items, 2) if n_frequent_items > 1 else 0

    # Warning thresholds (tuned from benchmarks)
    if n_pairs > 10_000_000:
        warnings.warn(
            f"Very large computation ahead!\n"
            f"  Frequent items: {n_frequent_items:,}\n"
            f"  Candidate pairs: {n_pairs:,}\n"
            f"  Estimated time: >30 minutes\n"
            f"  Consider increasing min_support (currently {min_support})",
            UserWarning,
            stacklevel=3,
        )
    elif n_pairs > 1_000_000:
        warnings.warn(
            f"Large computation: {n_pairs:,} candidate pairs.\n  This may take several minutes.",
            UserWarning,
            stacklevel=3,
        )


def _validate_parameters(
    min_support: float,
    max_length: int | None,
    batch_size: int | None,
    sparse_from_k: int | str | None = None,
) -> None:
    """Validate apriori parameters with clear error messages."""
    # min_support: must be numeric in [0.0, 1.0]
    if not isinstance(min_support, (int, float)):
        raise TypeError(f"min_support must be numeric, got {type(min_support).__name__}")
    if not 0.0 <= min_support <= 1.0:
        raise ValueError(
            f"min_support must be in range [0.0, 1.0], got {min_support}\n"
            f"  Hint: Support is a proportion (e.g., 0.01 = 1%)"
        )

    # max_length: must be positive int or None
    if max_length is not None:
        if not isinstance(max_length, int):
            raise TypeError(f"max_length must be int or None, got {type(max_length).__name__}")
        if max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {max_length}")

    # batch_size: must be positive int or None
    if batch_size is not None:
        if not isinstance(batch_size, int):
            raise TypeError(f"batch_size must be int or None, got {type(batch_size).__name__}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    # sparse_from_k: int K-level, "auto", or None
    validate_sparse_from_k(sparse_from_k)


def _prune_equal_support(
    current_frequent: list[tuple[str, ...]],
    current_supports: dict[tuple[str, ...], float],
    prev_supports: dict[tuple[str, ...], float],
    tolerance: float = 1e-9,
) -> list[tuple[str, ...]]:
    """Prune itemsets with equal support to their subsets (non-closed itemsets).

    Based on BitApriori [Zheng 2010]: if support({A,B,C}) == support({A,B}),
    then C appears in ALL transactions containing {A,B}. These non-closed
    itemsets add no information for rule mining and can be pruned to reduce
    candidate generation in subsequent iterations.

    Args:
        current_frequent: Frequent k-itemsets from current iteration.
        current_supports: Support values for current k-itemsets.
        prev_supports: Support values for (k-1)-itemsets from previous iteration.
        tolerance: Floating point comparison tolerance.

    Returns:
        List of closed itemsets (those with support different from all subsets).
    """
    if not prev_supports:
        return current_frequent

    closed = []
    for itemset in current_frequent:
        is_closed = True
        current_sup = current_supports[itemset]

        # Check if support equals any (k-1) subset
        for i in range(len(itemset)):
            subset = itemset[:i] + itemset[i + 1 :]
            if subset in prev_supports:
                if abs(current_sup - prev_supports[subset]) < tolerance:
                    is_closed = False
                    break

        if is_closed:
            closed.append(itemset)

    return closed




def _infer_count_from_subsets(
    candidate: tuple[str, ...],
    prev_counts: dict[tuple[str, ...], int],
) -> int | None:
    """Infer candidate count when ALL subsets have EQUAL count (APPROXIMATION).

    WARNING: This is mathematically unsound. Anti-monotonicity only guarantees
    support(superset) <= min(subset_supports), NOT equality when all subsets
    match. Counterexample: support({A,B})=support({A,C})=support({B,C})=30
    but support({A,B,C})=15. Use with caution — may overcount.

    Only used in CPU path when use_generator_pruning=True (default: False).
    GPU production path does NOT use this function.

    Args:
        candidate: Candidate k-itemset to check.
        prev_counts: Integer counts for (k-1)-itemsets from previous iteration.

    Returns:
        Inferred count if ALL subsets have equal count, None otherwise.
        NOTE: Returned count is an UPPER BOUND, not exact.
    """
    k = len(candidate)
    if k < 2:
        return None

    # Collect counts for ALL (k-1)-subsets
    subset_counts: list[int] = []
    for i in range(k):
        subset = candidate[:i] + candidate[i + 1 :]
        if subset not in prev_counts:
            # If ANY subset is missing (not frequent), we can't infer
            return None
        subset_counts.append(prev_counts[subset])

    # Check if ALL subsets have the SAME count
    unique_counts = set(subset_counts)
    if len(unique_counts) == 1:
        # All subsets have equal count -> candidate must have this count too
        inferred = subset_counts[0]
        logger.debug(f"INFERRED {candidate}: all {k} subsets have count={inferred}")
        return inferred

    # Not inferable
    logger.debug("Cannot infer {}: subsets have different counts {}", candidate, subset_counts)

    return None




def apriori(
    transactions: pl.LazyFrame | pl.DataFrame | None = None,
    min_support: float = 0.5,
    max_length: int | None = None,
    item_col: str = "items",
    use_gpu: bool = False,
    batch_size: int | None = 10_000,
    profile: bool = False,
    show_progress: bool = False,
    warn_complexity: bool = True,
    prune_equal_support: bool = False,
    use_generator_pruning: bool = False,
    sparse: bool | None = None,
    n_jobs: int = 1,
    enable_length_filter: bool = True,
    # Streaming parameters (SON algorithm)
    streaming: bool = False,
    chunk_size: int = 10_000_000,
    n_gpus: int = 1,
    memory_budget_gb: float | None = None,
    progress_callback: Callable[[str, int, int, dict[str, Any]], None] | None = None,
    level_callback: Callable[[int, int, int, float], None] | None = None,
    # GPU-resident mode (zero PCIe round trips)
    gpu_resident: bool = False,
    # Pre-built bitvector input (GPU fast path)
    bitvecs: tuple | None = None,
    # Per-K Parquet flush (prevents CPU RAM OOM at K=7+ scale)
    output_dir: str | None = None,
    # Resume from K=N+1 by loading K=N frequent itemsets from parquet
    resume_from_k: int | None = None,
    # Memory guard limits for exhaustive mining
    max_ram_gb: float = 800.0,
    max_vram_gb: float = 70.0,
    # V3: dense→sparse CSR transition — int K-level, "auto" = measured density
    sparse_from_k: int | Literal["auto"] | None = None,
    # V3 B6: restrict candidates to anchor neighborhoods (two-phase mining)
    anchor_items: set | None = None,
) -> pl.DataFrame | tuple[pl.DataFrame, ProfilingSession]:
    """Find frequent itemsets using fully vectorized boolean matrix operations.

    Transactions become a boolean matrix; support counting becomes vectorized AND + sum.

    Args:
        transactions: Transaction data with item lists.
        min_support: Minimum support threshold (0.0-1.0).
        max_length: Maximum itemset length (None = unlimited).
        item_col: Column name with item lists.
        use_gpu: Use GPU acceleration if available (requires polars[gpu]).
        batch_size: Candidates per batch for memory control. None = no batching.
        profile: If True, return profiling metrics alongside results.
        show_progress: If True, display progress bar (requires tqdm).
        warn_complexity: If True, warn when candidate pairs > 1M.
        prune_equal_support: Prune non-closed itemsets (equal support to subset).
            Based on BitApriori [Zheng 2010]. 3-50x speedup on dense datasets.
        use_generator_pruning: Infer support from (k-1)-subsets instead of counting.
            Based on Pascal/generator pruning [Bastide et al. 2000]. 10-40% speedup,
            no impact on results (unlike prune_equal_support).
        sparse: Scipy CSR matrix usage. True = force, False = Polars, None = auto
            (switches at >100K k=2 candidates or >500 items <10% density).
        n_jobs: Parallel workers for sparse k>2 counting. 1=sequential, -1=all CPUs.
            Only active when sparse mode is used.
        enable_length_filter: Filter transactions shorter than k before counting
            k-itemset support. Set to False to disable.
        streaming: Use SON algorithm for chunked processing. Memory becomes
            O(chunk_size × n_items) instead of O(total × n_items).
        chunk_size: Transactions per chunk when streaming=True. Default 10M.
        n_gpus: GPUs for multi-GPU streaming (default 1). Requires CuPy.
        memory_budget_gb: Auto-calculate chunk_size to fit this budget.
            Overrides chunk_size. Only used when streaming=True.
        progress_callback: Streaming progress callback
            (phase, chunk_idx, n_chunks, metrics).
        level_callback: Per-level callback
            (k, n_candidates, n_frequent, duration_ms).
        bitvecs: Pre-built GPU bitvectors tuple (bitvecs_gpu, col_to_item, n_transactions).
            Skips DataFrame conversion; transactions must be None when provided.
        sparse_from_k: GPU paths only — when to switch support counting from
            dense bitvectors to sparse CSR tidsets. An int fixes the K-level;
            "auto" transitions when the previous level's measured mean support
            drops below n_transactions/32 (the point where tidsets become
            smaller than bitvectors); None (default) never switches.

    Returns:
        DataFrame with itemset (List[Int64]) and support (Float64) columns.
        If profile=True: tuple of (DataFrame, ProfilingSession).

    Example:
        >>> result = apriori(df, min_support=0.5)
        >>> result = apriori(df, min_support=0.001, use_gpu=True)
        >>> result, session = apriori(df, min_support=0.1, profile=True)
        >>> result = apriori(df, min_support=0.0001, sparse=True, n_jobs=-1)
        >>> result = apriori(huge_df, min_support=0.001, streaming=True, n_gpus=8)
    """
    _validate_parameters(min_support, max_length, batch_size, sparse_from_k)

    # Route to bitvecs fast path if pre-built GPU bitvectors provided
    if bitvecs is not None:
        # Validate: cannot provide both bitvecs and transactions
        if transactions is not None:
            raise ValueError(
                "Cannot provide both 'transactions' and 'bitvecs'. "
                "Use 'bitvecs' for pre-built GPU bitvectors or 'transactions' for DataFrame input."
            )

        # Unpack and validate bitvecs tuple
        if not isinstance(bitvecs, tuple) or len(bitvecs) != 3:
            raise ValueError("bitvecs must be a tuple of (bitvecs_gpu, col_to_item, n_transactions)")

        bitvecs_gpu, col_to_item, n_trans = bitvecs

        # Validate bitvecs_gpu is a CuPy array
        if not hasattr(bitvecs_gpu, "__cuda_array_interface__"):
            raise TypeError(
                f"bitvecs_gpu must be a CuPy array (has __cuda_array_interface__), got {type(bitvecs_gpu).__name__}"
            )

        # Validate shape
        if bitvecs_gpu.ndim != 2:
            raise ValueError(f"bitvecs_gpu must be 2D (n_cols, n_u64s), got {bitvecs_gpu.ndim}D")

        # Validate n_u64s matches n_transactions
        expected_n_u64s = math.ceil(n_trans / 64)
        if bitvecs_gpu.shape[1] != expected_n_u64s:
            raise ValueError(
                f"bitvecs_gpu.shape[1] ({bitvecs_gpu.shape[1]}) must equal "
                f"ceil(n_transactions/64) = ceil({n_trans}/64) = {expected_n_u64s}"
            )

        # Validate col_to_item keys match n_cols
        if len(col_to_item) != bitvecs_gpu.shape[0]:
            raise ValueError(
                f"col_to_item has {len(col_to_item)} keys but bitvecs_gpu has {bitvecs_gpu.shape[0]} columns"
            )

        # Route to GPU-resident or standard bitvecs fast path
        if gpu_resident:
            from et_miner.gpu.mining import _apriori_from_bitvecs_gpu_resident

            return _apriori_from_bitvecs_gpu_resident(
                bitvecs_gpu,
                col_to_item,
                n_trans,
                min_support,
                max_length,
                profile,
                level_callback,
            )
        from et_miner.gpu.mining import _apriori_from_bitvecs

        return _apriori_from_bitvecs(
            bitvecs_gpu,
            col_to_item,
            n_trans,
            min_support,
            max_length,
            batch_size,
            profile,
            level_callback,
            n_gpus=n_gpus,
            max_ram_gb=max_ram_gb,
            max_vram_gb=max_vram_gb,
            sparse_from_k=sparse_from_k,
        )

    # Validate transactions is provided when bitvecs is not
    if transactions is None:
        raise ValueError("Either 'transactions' or 'bitvecs' must be provided")

    # Route to streaming implementation if requested
    if streaming:
        # Multi-GPU streaming if n_gpus > 1
        if n_gpus > 1:
            from et_miner.streaming.multi_gpu import apriori_streaming_multi_gpu

            return apriori_streaming_multi_gpu(
                transactions,
                min_support=min_support,
                max_length=max_length,
                item_col=item_col,
                n_gpus=n_gpus,
                chunk_size=chunk_size,
                batch_size=batch_size,
                show_progress=show_progress,
                sparse=sparse,
                n_jobs=n_jobs,
                progress_callback=progress_callback,
            )

        # Single-GPU/CPU streaming
        from et_miner.streaming.son import apriori_streaming

        return apriori_streaming(
            transactions,
            min_support=min_support,
            max_length=max_length,
            item_col=item_col,
            chunk_size=chunk_size,
            memory_budget_gb=memory_budget_gb,
            use_gpu=use_gpu,
            gpu_resident=gpu_resident,
            batch_size=batch_size,
            profile=profile,
            show_progress=show_progress,
            sparse=sparse,
            n_jobs=n_jobs,
            progress_callback=progress_callback,
        )

    lf = transactions.lazy() if isinstance(transactions, pl.DataFrame) else transactions

    # ── GPU direct path: transactions → CSR → GPU bitvecs ──────────────
    # Bypasses the dense boolean matrix entirely.
    # For 205M × 1006: ~5 GB CPU + ~25 GB GPU  instead of  206 GB CPU.
    if use_gpu:
        from et_miner.gpu.bitvec import _build_gpu_bitvec_matrix

        from .matrix import _build_csr_from_transactions

        csr_result = _build_csr_from_transactions(lf, min_support, item_col)
        if csr_result is None:
            if profile:
                return _empty_result(), ProfilingSession()
            return _empty_result()

        csr, idx_to_item, n_trans = csr_result

        # Multi-GPU row-split path, or single-GPU with anchor filtering
        if n_gpus > 1 or anchor_items is not None:
            from et_miner.gpu.row_split import _apriori_row_split_multi_gpu

            return _apriori_row_split_multi_gpu(
                csr,
                idx_to_item,
                n_trans,
                min_support,
                max_length,
                n_gpus,
                level_callback,
                output_dir=output_dir,
                resume_from_k=resume_from_k,
                prune_closed=prune_equal_support,  # V3: reuse existing flag
                prune_apriori=prune_equal_support,  # V3: Apriori subset pruning
                sparse_from_k=sparse_from_k,  # V3: density transition K-level
                anchor_items=anchor_items,  # V3 B6: two-phase anchor filtering
            )

        bitvecs_gpu = _build_gpu_bitvec_matrix(csr)
        del csr  # free CPU CSR memory

        if gpu_resident:
            from et_miner.gpu.mining import _apriori_from_bitvecs_gpu_resident

            return _apriori_from_bitvecs_gpu_resident(
                bitvecs_gpu,
                idx_to_item,
                n_trans,
                min_support,
                max_length,
                profile,
                level_callback,
            )
        from et_miner.gpu.mining import _apriori_from_bitvecs

        return _apriori_from_bitvecs(
            bitvecs_gpu,
            idx_to_item,
            n_trans,
            min_support,
            max_length,
            batch_size,
            profile,
            level_callback,
            n_gpus=n_gpus,
            max_ram_gb=max_ram_gb,
            max_vram_gb=max_vram_gb,
            sparse_from_k=sparse_from_k,
        )

    # ── CPU path: dense boolean matrix ─────────────────────────────────
    session = ProfilingSession() if profile else None

    # Phase 1: Build boolean matrix (includes frequent 1-itemset mining)
    if session:
        session.start_phase("matrix_build")

    matrix, col_to_item, n_trans = build_boolean_matrix(lf, min_support, item_col)

    if session:
        session.end_phase(n_frequent_items=len(col_to_item), n_transactions=n_trans)

    # Compute max possible k from transaction lengths
    max_tx_length = (lf.select(pl.col(item_col).list.len().max()).collect(engine="streaming").item()) or 0

    effective_max_length = min(
        max_length if max_length else float("inf"),
        max_tx_length,
        len(col_to_item),  # Can't have more items than columns
    )

    if not col_to_item:
        if profile:
            return _empty_result(), session
        return _empty_result()

    item_cols = list(col_to_item.keys())

    # Phase 2: Get exact 1-itemset supports from matrix column sums
    _k1_start = time.perf_counter()
    if session:
        session.start_phase("k1_support")

    # 1-itemset counting: always use streaming engine (simple column sums, no GPU benefit)
    one_itemset_exprs = [pl.col(c).sum().alias(c) for c in item_cols]
    one_counts = matrix.lazy().select(one_itemset_exprs).collect(engine="streaming")

    results: list[tuple[list[int], float]] = []
    prev_frequent: list[tuple[str, ...]] = []

    # Filter on integer count to avoid float precision issues at boundaries.
    # math.ceil ensures we don't include items below threshold.
    min_count_threshold = _min_count(min_support, n_trans)
    for col in item_cols:
        count = one_counts.get_column(col).item()
        if count >= min_count_threshold:
            support = count / n_trans
            results.append(([col_to_item[col]], support))
            prev_frequent.append((col,))

    if session:
        session.end_phase(n_frequent=len(prev_frequent))

    # Call level callback for k=1
    if level_callback:
        _k1_duration_ms = (time.perf_counter() - _k1_start) * 1000
        level_callback(1, len(item_cols), len(prev_frequent), _k1_duration_ms)

    if not prev_frequent:
        if profile:
            return _empty_result(), session
        return _empty_result()

    # Warn about potentially expensive computation
    if warn_complexity:
        _warn_complexity(len(prev_frequent), min_support)

    # Initialize prev_supports for equal-support pruning
    # Use integer count filtering for consistency
    prev_supports: dict[tuple[str, ...], float] = {
        (col,): one_counts.get_column(col).item() / n_trans
        for col in item_cols
        if one_counts.get_column(col).item() >= min_count_threshold
    }

    # Initialize prev_counts for generator-based pruning (uses integer counts for precision)
    prev_counts: dict[tuple[str, ...], int] = {
        (col,): one_counts.get_column(col).item()
        for col in item_cols
        if one_counts.get_column(col).item() >= min_count_threshold
    }

    # Phase 3: k >= 2 with vectorized support counting
    k = 2

    while k <= effective_max_length and len(prev_frequent) >= k:
        _k_start = time.perf_counter()

        # Generate candidates using Apriori join
        if session:
            session.start_phase(f"k{k}_candidate_gen")

        candidates = _generate_candidates(prev_frequent, k)

        if session:
            session.end_phase(n_candidates=len(candidates))

        if not candidates:
            break

        # Support inference + counting
        if session:
            session.start_phase(f"k{k}_support_count")

        # Phase A: Try to infer count from subsets (Pascal/generator pruning)
        inferred_counts: dict[tuple[str, ...], int] = {}
        needs_counting: list[tuple[str, ...]] = []

        if use_generator_pruning and prev_counts and k >= 2:
            for candidate in candidates:
                inferred_count = _infer_count_from_subsets(candidate, prev_counts)
                if inferred_count is not None:
                    # Count can be inferred - no need to count!
                    inferred_counts[candidate] = inferred_count
                else:
                    needs_counting.append(candidate)
        else:
            needs_counting = candidates

        # Log inference results
        if use_generator_pruning and len(inferred_counts) > 0:
            skip_rate = 100 * len(inferred_counts) / len(candidates)
            logger.debug(
                f"[k={k}] Generator pruning: {len(inferred_counts)}/{len(candidates)} "
                f"candidates ({skip_rate:.1f}%) skipped via count inference"
            )

        # Phase B: Count support for remaining candidates
        if needs_counting:
            counted = count_support_batched(
                matrix,
                needs_counting,
                n_trans,
                batch_size,
                use_gpu,
                show_progress,
                sparse,
                n_jobs,
                enable_length_filter=enable_length_filter,
            )
        else:
            counted = {}

        # Combine inferred and counted (all are integer counts now)
        all_counts = {**inferred_counts, **counted}

        # Filter to frequent itemsets using integer count comparison.
        # This avoids float precision issues at boundaries.
        current_frequent: list[tuple[str, ...]] = []
        current_supports: dict[tuple[str, ...], float] = {}
        current_counts: dict[tuple[str, ...], int] = {}
        n_skipped = len(inferred_counts)

        for itemset in candidates:
            count = all_counts[itemset]

            if count >= min_count_threshold:
                support = count / n_trans
                current_frequent.append(itemset)
                current_supports[itemset] = support
                current_counts[itemset] = count
                item_list = [col_to_item[c] for c in itemset]
                results.append((item_list, support))

        if session:
            session.end_phase(
                n_frequent=len(current_frequent),
                n_inferred=n_skipped if use_generator_pruning else 0,
            )

        # Call level callback for k>=2
        if level_callback:
            _k_duration_ms = (time.perf_counter() - _k_start) * 1000
            level_callback(k, len(candidates), len(current_frequent), _k_duration_ms)

        if not current_frequent:
            break

        # Apply equal-support pruning if enabled
        if prune_equal_support and prev_supports:
            current_frequent = _prune_equal_support(current_frequent, current_supports, prev_supports)

        prev_supports = current_supports
        prev_counts = current_counts
        prev_frequent = current_frequent
        k += 1

    result_df = _build_result_df(results)
    if profile:
        return result_df, session
    return result_df


