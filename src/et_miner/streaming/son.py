"""Streaming Apriori implementation using SON algorithm.

The SON (Savasere-Omiecinski-Navathe) algorithm enables processing datasets
that don't fit in memory by using a two-pass approach:

Pass 1 - Local Mining:
    For each chunk of the dataset, find locally frequent itemsets using a
    slightly lowered support threshold (0.9× by default). This ensures we
    don't miss any globally frequent itemsets due to sampling variance.

Pass 2 - Global Counting:
    Count the actual support for all candidate itemsets (union of local
    frequent itemsets) across the entire dataset.

Memory Guarantee:
    Memory usage is O(chunk_size × n_items) instead of O(dataset_size × n_items).
    For 1B transactions with 10M chunk size: 100× memory reduction.

Reference:
    Savasere, A., Omiecinski, E. R., & Navathe, S. B. (1995).
    "An Efficient Algorithm for Mining Association Rules in Large Databases"
    VLDB 1995.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from et_miner._compat import HAS_TQDM, tqdm
from et_miner.core.matrix import (
    build_boolean_matrix,
    count_support_batched,
)
from et_miner.core.result import (
    _build_result_df,
    _empty_result,
    _min_count,
)
from et_miner.core.profiling import ProfilingSession


if TYPE_CHECKING:
    pass

from loguru import logger


def _estimate_chunk_size_from_memory(
    memory_budget_gb: float,
    n_items_estimate: int = 1000,
    items_per_transaction: int = 10,
) -> int:
    """Estimate chunk size from memory budget.

    Args:
        memory_budget_gb: Maximum memory to use in GB.
        n_items_estimate: Estimated number of frequent items.
        items_per_transaction: Estimated average items per transaction.

    Returns:
        Recommended chunk size (number of transactions).
    """
    # Boolean matrix: n_transactions × n_items × 1 bit (Polars bit-packed)
    # Plus overhead for intermediate operations (~2× safety factor)
    bytes_per_transaction = (n_items_estimate / 8) * 2  # bit-packed with 2× safety

    memory_budget_bytes = memory_budget_gb * 1024**3
    chunk_size = int(memory_budget_bytes / max(bytes_per_transaction, 1))

    # Clamp to reasonable range
    return max(100_000, min(chunk_size, 100_000_000))  # 100K to 100M


def apriori_streaming(
    transactions: pl.LazyFrame | pl.DataFrame,
    min_support: float = 0.5,
    max_length: int | None = None,
    item_col: str = "items",
    chunk_size: int = 40_000_000,
    memory_budget_gb: float | None = None,
    local_support_factor: float = 0.9,
    use_gpu: bool = False,
    gpu_resident: bool = False,
    batch_size: int | None = 10_000,
    profile: bool = False,
    show_progress: bool = True,
    sparse: bool | None = None,
    n_jobs: int = 1,
    progress_callback: Callable[[str, int, int, dict[str, Any]], None] | None = None,
) -> pl.DataFrame | tuple[pl.DataFrame, ProfilingSession]:
    """Find frequent itemsets using streaming SON algorithm.

    Processes the dataset in chunks, enabling analysis of datasets that don't
    fit in memory. Uses the SON (Savasere-Omiecinski-Navathe) algorithm:

    1. Pass 1 - Local Mining: Find locally frequent itemsets in each chunk
       with a lowered support threshold (local_support_factor × min_support).

    2. Pass 2 - Global Counting: Count support for all candidate itemsets
       across the full dataset.

    Memory usage is O(chunk_size × n_items), not O(total_transactions × n_items).

    Args:
        transactions: Transaction data with item lists (LazyFrame recommended).
        min_support: Minimum global support threshold (0.0-1.0).
        max_length: Maximum itemset length (None = unlimited).
        item_col: Column name with item lists.
        chunk_size: Number of transactions per chunk (default 40M).
        memory_budget_gb: If set, automatically calculate chunk_size to stay
            within this memory budget. Overrides chunk_size parameter.
        local_support_factor: Factor to lower local support threshold (default 0.9).
            Lower values reduce false negatives but increase candidates.
        use_gpu: Use GPU acceleration if available.
        batch_size: Candidates per batch for memory control in counting phase.
        profile: If True, return profiling metrics alongside results.
        show_progress: If True, display progress bars (requires tqdm).
        sparse: Control scipy sparse matrix usage.
        n_jobs: Number of parallel workers for support counting.
        progress_callback: Optional callback for progress updates. Called with:
            (phase: str, chunk_idx: int, n_chunks: int, metrics: dict)
            where phase is "pass1" or "pass2", and metrics contains
            {candidates, items, memory_gb} for pass1 or {counted, memory_gb} for pass2.

    Returns:
        If profile=False: DataFrame with columns [itemset, support]
        If profile=True: Tuple of (DataFrame, ProfilingSession)

    Example:
        >>> # Process 1 billion transactions in 10M chunks
        >>> df = pl.scan_parquet("huge_dataset/*.parquet")
        >>> result = apriori_streaming(
        ...     df,
        ...     min_support=0.001,
        ...     chunk_size=10_000_000,
        ...     show_progress=True,
        ... )
    """
    lf = transactions.lazy() if isinstance(transactions, pl.DataFrame) else transactions
    session = ProfilingSession() if profile else None

    # Phase 0: Get total transaction count
    if session:
        session.start_phase("count_transactions")

    n_total = lf.select(pl.len()).collect(engine="streaming").item()

    if session:
        session.end_phase(n_transactions=n_total)

    if n_total == 0:
        result = _empty_result()
        if profile:
            return result, session
        return result

    # Determine effective chunk size
    if memory_budget_gb is not None:
        effective_chunk_size = _estimate_chunk_size_from_memory(memory_budget_gb)
        logger.info(
            "Memory budget %.1f GB → chunk size %d",
            memory_budget_gb,
            effective_chunk_size,
        )
    else:
        effective_chunk_size = chunk_size

    # Single chunk optimization: use standard apriori if data fits
    if n_total <= effective_chunk_size:
        logger.info(
            "Dataset (%d) fits in single chunk (%d), using standard apriori",
            n_total,
            effective_chunk_size,
        )
        # Import here to avoid circular dependency
        from et_miner.core.apriori import apriori

        return apriori(
            transactions,
            min_support=min_support,
            max_length=max_length,
            item_col=item_col,
            use_gpu=use_gpu,
            batch_size=batch_size,
            profile=profile,
            show_progress=show_progress,
            sparse=sparse,
            n_jobs=n_jobs,
        )

    # Calculate number of chunks
    n_chunks = math.ceil(n_total / effective_chunk_size)
    local_min_support = min_support * local_support_factor

    # Pre-calculate chunk sizes to avoid redundant counting
    # This eliminates ~3 streaming counts per chunk (major I/O reduction)
    chunk_sizes = []
    for chunk_idx in range(n_chunks):
        offset = chunk_idx * effective_chunk_size
        remaining = n_total - offset
        chunk_sizes.append(min(effective_chunk_size, remaining))

    logger.info(
        "SON streaming: %d total transactions, %d chunks of %d, local_support=%.6f (%.1f%% of %.6f)",
        n_total,
        n_chunks,
        effective_chunk_size,
        local_min_support,
        local_support_factor * 100,
        min_support,
    )

    # =========================================================================
    # PASS 1: Local frequent itemset mining
    # =========================================================================
    if session:
        session.start_phase("pass1_local_mining")

    # Collect all locally frequent itemsets (candidates for global counting)
    # Using a set to deduplicate across chunks
    candidate_itemsets: set[tuple[int, ...]] = set()
    all_items: set[int] = set()

    # Create chunk iterator
    chunk_iter = range(n_chunks)
    if show_progress and HAS_TQDM:
        chunk_iter = tqdm(
            chunk_iter,
            desc="Pass 1: Local mining",
            unit="chunk",
            total=n_chunks,
        )

    for chunk_idx in chunk_iter:
        offset = chunk_idx * effective_chunk_size

        # Extract chunk using slice
        chunk_lf = lf.slice(offset, effective_chunk_size)

        # Use pre-calculated chunk size (avoids redundant streaming count!)
        chunk_n = chunk_sizes[chunk_idx]
        if chunk_n == 0:
            continue

        # Build boolean matrix for this chunk with LOCAL support threshold
        try:
            matrix, col_to_item, _ = build_boolean_matrix(
                chunk_lf,
                local_min_support,
                item_col,
            )
        except Exception as e:
            logger.warning("Chunk %d failed: %s", chunk_idx, e)
            continue

        if not col_to_item:
            continue

        # Track all items seen
        all_items.update(col_to_item.values())

        # Mine frequent itemsets: GPU-resident fast path or standard CPU/GPU path
        local_frequent = None
        if gpu_resident:
            local_frequent = _mine_chunk_gpu_resident(
                matrix,
                col_to_item,
                chunk_n,
                local_min_support,
                max_length,
            )
            if local_frequent is None:
                logger.warning(
                    "GPU-resident failed for chunk %d, falling back to CPU",
                    chunk_idx,
                )

        if local_frequent is None:
            local_frequent = _mine_chunk_frequent(
                matrix,
                col_to_item,
                chunk_n,
                local_min_support,
                max_length,
                batch_size,
                use_gpu,
                sparse,
                n_jobs,
            )

        # Add to global candidates
        for itemset in local_frequent:
            candidate_itemsets.add(itemset)

        if show_progress and HAS_TQDM:
            chunk_iter.set_postfix(  # type: ignore
                candidates=len(candidate_itemsets),
                items=len(all_items),
            )

        # Progress callback for external monitoring
        if progress_callback:
            progress_callback(
                "pass1",
                chunk_idx,
                n_chunks,
                {
                    "candidates": len(candidate_itemsets),
                    "items": len(all_items),
                    "memory_gb": _get_memory_gb(),
                },
            )

    if session:
        session.end_phase(
            n_candidates=len(candidate_itemsets),
            n_items=len(all_items),
        )

    if not candidate_itemsets:
        result = _empty_result()
        if profile:
            return result, session
        return result

    logger.info(
        "Pass 1 complete: %d candidate itemsets from %d items",
        len(candidate_itemsets),
        len(all_items),
    )

    # =========================================================================
    # PASS 2: Global support counting
    # =========================================================================
    if session:
        session.start_phase("pass2_global_counting")

    # We need to count support for all candidates across the FULL dataset
    # OPTIMIZATION: Only build columns for items that appear in candidates
    # This can significantly reduce matrix size for sparse candidate sets
    candidate_items: set[int] = set()
    for itemset in candidate_itemsets:
        candidate_items.update(itemset)

    logger.info(
        "Pass 2 optimization: %d/%d items needed for candidates (%.1f%% reduction)",
        len(candidate_items),
        len(all_items),
        (1 - len(candidate_items) / len(all_items)) * 100 if all_items else 0,
    )

    global_counts: dict[tuple[int, ...], int] = {itemset: 0 for itemset in candidate_itemsets}

    # Create item -> column name mapping for global counting (only candidate items!)
    sorted_items = sorted(candidate_items)  # Only items in candidates
    item_to_col = {item: f"i_{idx}" for idx, item in enumerate(sorted_items)}

    # Convert candidate itemsets to column-name tuples
    candidate_cols: list[tuple[str, ...]] = [
        tuple(item_to_col[item] for item in itemset) for itemset in candidate_itemsets
    ]

    # Map back from col tuples to original itemsets
    col_to_itemset = {col_tuple: itemset for col_tuple, itemset in zip(candidate_cols, candidate_itemsets)}

    # Second pass: count support across all chunks
    chunk_iter2 = range(n_chunks)
    if show_progress and HAS_TQDM:
        chunk_iter2 = tqdm(
            chunk_iter2,
            desc="Pass 2: Global counting",
            unit="chunk",
            total=n_chunks,
        )

    for chunk_idx in chunk_iter2:
        offset = chunk_idx * effective_chunk_size

        # Extract chunk
        chunk_lf = lf.slice(offset, effective_chunk_size)
        # Use pre-calculated chunk size (avoids redundant streaming count!)
        chunk_n = chunk_sizes[chunk_idx]
        if chunk_n == 0:
            continue

        # Build boolean matrix with ALL items (not just locally frequent)
        # We need consistent column mapping across chunks
        matrix = _build_matrix_for_items(
            chunk_lf,
            item_col,
            sorted_items,
            item_to_col,
        )

        if matrix.height == 0:
            continue

        # Count support for all candidates in this chunk
        # GPU-resident fast path: bitvec AND+popcount on GPU
        gpu_chunk_counts = None
        if gpu_resident:
            gpu_chunk_counts = _count_candidates_gpu(
                matrix,
                list(candidate_itemsets),
                sorted_items,
            )
            if gpu_chunk_counts is None:
                logger.warning(
                    "GPU counting failed for chunk %d, falling back to CPU",
                    chunk_idx,
                )

        if gpu_chunk_counts is not None:
            # GPU path: counts are keyed by item-ID tuples directly
            for itemset, count in gpu_chunk_counts.items():
                global_counts[itemset] += count
        else:
            # CPU path: counts are keyed by column-name tuples
            chunk_counts = count_support_batched(
                matrix,
                candidate_cols,
                chunk_n,
                batch_size,
                use_gpu,
                False,  # No progress for individual chunks
                sparse,
                n_jobs,
                # Candidates are mixed-length (all K pooled in SON Pass 2); the
                # length filter auto-picks k=len(itemsets[0]) and would drop
                # transactions shorter than that, undercounting shorter itemsets.
                enable_length_filter=False,
            )
            for col_tuple, count in chunk_counts.items():
                itemset = col_to_itemset[col_tuple]
                global_counts[itemset] += count

        # Progress callback for external monitoring
        if progress_callback:
            progress_callback(
                "pass2",
                chunk_idx,
                n_chunks,
                {
                    "counted": chunk_idx + 1,
                    "memory_gb": _get_memory_gb(),
                },
            )

    if session:
        session.end_phase(n_counted=len(global_counts))

    # =========================================================================
    # Filter to globally frequent itemsets
    # =========================================================================
    min_count_threshold = _min_count(min_support, n_total)

    results: list[tuple[list[int], float]] = []
    for itemset, count in global_counts.items():
        if count >= min_count_threshold:
            support = count / n_total
            results.append((list(itemset), support))

    logger.info(
        "Pass 2 complete: %d/%d candidates are globally frequent (support >= %.6f)",
        len(results),
        len(candidate_itemsets),
        min_support,
    )

    result_df = _build_result_df(results)

    if profile:
        return result_df, session
    return result_df


def _build_bitvecs_for_chunk(
    matrix: pl.DataFrame,
    col_to_item: dict[str, int] | None = None,
) -> tuple[Any, dict[int, int] | None] | None:
    """Boolean matrix -> CSR -> GPU bitvecs. Returns (bitvecs_gpu, idx_to_item) or None.

    Chains existing functions to build GPU bitvectors from a Polars boolean matrix.
    Returns None on any failure so callers can gracefully fall back to CPU.
    """
    try:
        from et_miner.core.matrix import _polars_to_sparse_csr
        from et_miner.gpu.bitvec import _build_gpu_bitvec_matrix

        csr, col_name_to_idx = _polars_to_sparse_csr(matrix)
        bitvecs_gpu = _build_gpu_bitvec_matrix(csr)
        idx_to_item = {col_name_to_idx[col]: col_to_item[col] for col in col_to_item} if col_to_item else None
        return bitvecs_gpu, idx_to_item
    except Exception as e:
        logger.warning("GPU bitvec build failed: %s", e)
        return None


def _mine_chunk_gpu_resident(
    matrix: pl.DataFrame,
    col_to_item: dict[str, int],
    n_transactions: int,
    min_support: float,
    max_length: int | None,
) -> list[tuple[int, ...]] | None:
    """Mine chunk with GPU-resident Apriori. Returns list of itemsets or None on failure."""
    try:
        import cupy as cp
        from et_miner.core.apriori import _apriori_from_bitvecs_gpu_resident

        result = _build_bitvecs_for_chunk(matrix, col_to_item)
        if result is None:
            return None

        bitvecs_gpu, idx_to_item = result
        result_df = _apriori_from_bitvecs_gpu_resident(
            bitvecs_gpu,
            idx_to_item,
            n_transactions,
            min_support,
            max_length,
            profile=False,
            level_callback=None,
        )
        itemsets = [tuple(x) for x in result_df["itemset"].to_list()]
        del bitvecs_gpu
        cp.get_default_memory_pool().free_all_blocks()
        return itemsets
    except Exception as e:
        logger.warning("GPU-resident chunk mining failed: %s", e)
        return None


def _count_candidates_gpu(
    matrix: pl.DataFrame,
    candidate_itemsets: list[tuple[int, ...]],
    sorted_items: list[int],
) -> dict[tuple[int, ...], int] | None:
    """Count all candidates on GPU via bitvec AND+popcount. Returns counts dict or None."""
    try:
        import cupy as cp
        from et_miner.gpu.kernels import count_itemsets_cuda

        result = _build_bitvecs_for_chunk(matrix)
        if result is None:
            return None

        bitvecs_gpu, _ = result

        # Map item IDs -> bitvec column indices
        # sorted_items order = bitvec column order (from _build_matrix_for_items)
        item_to_idx = {item: idx for idx, item in enumerate(sorted_items)}
        itemsets_np = [np.array([item_to_idx[i] for i in itemset], dtype=np.int32) for itemset in candidate_itemsets]

        counts = count_itemsets_cuda(bitvecs_gpu, itemsets_np)
        del bitvecs_gpu
        cp.get_default_memory_pool().free_all_blocks()

        return {itemset: int(counts[i]) for i, itemset in enumerate(candidate_itemsets)}
    except Exception as e:
        logger.warning("GPU candidate counting failed: %s", e)
        return None


def _mine_chunk_frequent(
    matrix: pl.DataFrame,
    col_to_item: dict[str, int],
    n_transactions: int,
    min_support: float,
    max_length: int | None,
    batch_size: int | None,
    use_gpu: bool | str,
    sparse: bool | None,
    n_jobs: int,
) -> list[tuple[int, ...]]:
    """Mine frequent itemsets from a single chunk's boolean matrix.

    This is a simplified version of the main apriori() logic, optimized for
    the streaming use case where we just need the itemsets (not supports).

    Returns:
        List of frequent itemsets as tuples of item IDs.
    """
    from et_miner.core.apriori import _generate_candidates

    min_count_threshold = _min_count(min_support, n_transactions)
    item_cols = list(col_to_item.keys())

    # Get 1-itemset counts
    one_itemset_exprs = [pl.col(c).sum().alias(c) for c in item_cols]
    one_counts = matrix.lazy().select(one_itemset_exprs).collect(engine="streaming")

    # Filter frequent 1-itemsets
    frequent_itemsets: list[tuple[int, ...]] = []
    prev_frequent: list[tuple[str, ...]] = []

    for col in item_cols:
        count = one_counts.get_column(col).item()
        if count >= min_count_threshold:
            frequent_itemsets.append((col_to_item[col],))
            prev_frequent.append((col,))

    if not prev_frequent:
        return frequent_itemsets

    # Compute max possible k
    max_tx_length = matrix.select(pl.sum_horizontal(pl.all()).max()).item() or 0
    effective_max_length = min(
        max_length if max_length else float("inf"),
        max_tx_length,
        len(col_to_item),
    )

    # Mine k >= 2
    k = 2
    while k <= effective_max_length and len(prev_frequent) >= k:
        candidates = _generate_candidates(prev_frequent, k)
        if not candidates:
            break

        # Count support for candidates
        counts = count_support_batched(
            matrix,
            candidates,
            n_transactions,
            batch_size,
            use_gpu,
            False,
            sparse,
            n_jobs,
        )

        # Filter to frequent
        current_frequent: list[tuple[str, ...]] = []
        for candidate in candidates:
            count = counts[candidate]
            if count >= min_count_threshold:
                current_frequent.append(candidate)
                itemset = tuple(col_to_item[c] for c in candidate)
                frequent_itemsets.append(itemset)

        if not current_frequent:
            break

        prev_frequent = current_frequent
        k += 1

    return frequent_itemsets


def _build_matrix_for_items(
    transactions: pl.LazyFrame,
    item_col: str,
    items: list[int],
    item_to_col: dict[int, str],
) -> pl.DataFrame:
    """Boolean matrix with a column per item in `items` — used in SON Pass 2 for consistent mapping."""
    exprs = [pl.col(item_col).list.contains(item).alias(item_to_col[item]) for item in items]

    # Don't use engine="streaming" due to list.contains bug
    return transactions.select(exprs).collect()


def _get_memory_gb() -> float:
    """Get current process memory usage in GB.

    Returns:
        Memory usage in gigabytes, or 0.0 if psutil unavailable.
    """
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e9
    except ImportError:
        return 0.0
