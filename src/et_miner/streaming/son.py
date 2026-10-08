"""Streaming Apriori with the SON algorithm: mining a dataset chunk by chunk.

The SON (Savasere-Omiecinski-Navathe) algorithm makes two passes over the
chunks:

Pass 1 - Local mining:
    Mine each chunk at a lowered support threshold (``local_support_factor`` x
    min_support, 0.9 by default). Every globally frequent itemset is locally
    frequent in at least one chunk, so the union of the local results holds
    them all.

Pass 2 - Global counting:
    Count every candidate of that union over every chunk and keep those whose
    total reaches the global threshold.

On the CPU both passes run on the array miner (core/cpu_miner.py): pass 1
mines each chunk's CSR level by level, and the union is kept as one int32
array per length, deduplicated chunk by chunk; pass 2 maps each chunk onto the
candidate items and counts the candidates with ``count_itemsets``. Memory
holds one chunk's CSR and the candidate arrays, never the whole dataset. With
``use_gpu=True`` each chunk is mined on the row-split GPU miner and pass 2
counts with the batched itemset kernel.

Usage:
    from et_miner.streaming.son import apriori_streaming

    result = apriori_streaming(pl.scan_parquet("big/*.parquet"), min_support=0.001, chunk_size=10_000_000)

Options: the parameters of ``apriori_streaming``.

Reference:
    Savasere, A., Omiecinski, E. R., & Navathe, S. B. (1995).
    "An Efficient Algorithm for Mining Association Rules in Large Databases"
    VLDB 1995.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import polars as pl

from et_miner._compat import HAS_TQDM, tqdm
from et_miner.core.cpu_miner import (
    _emit,
    _Level,
    _map_rows,
    _mine_levels,
    _workers,
    build_transaction_csr,
    count_itemsets,
    unique_rows,
)
from et_miner.core.matrix import build_boolean_matrix
from et_miner.core.result import (
    _build_result_df,
    _empty_result,
    _min_count,
)
from et_miner.core.profiling import ProfilingSession

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
        use_gpu: Mine each chunk on the GPU with the row-split miner and count
            pass 2 with the batched itemset kernel; otherwise both passes run
            on the CPU's array miner.
        gpu_resident: Removed; True raises ValueError. ``use_gpu=True`` keeps
            the candidates on the GPU without it.
        batch_size: Not read by either pass; passed on to ``apriori()`` when
            the data fits one chunk.
        profile: If True, return profiling metrics alongside results.
        show_progress: If True, display progress bars (requires tqdm).
        sparse: Deprecated on the CPU passes, where it selects no counter (a
            non-None value warns and is ignored); the GPU passes do not read it.
        n_jobs: Threads for the CPU passes (-1 = every CPU); 1 runs them on
            the calling thread.
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
    if gpu_resident:
        raise ValueError(
            "gpu_resident was removed: with use_gpu=True, SON mines each chunk on the "
            "row-split miner and counts pass 2 with the batched kernel. Drop the argument."
        )
    if sparse is not None and not use_gpu:
        warnings.warn(
            "sparse= no longer selects a counting engine under streaming=True: SON's CPU passes count "
            "from a CSR of each chunk whatever its value, and the argument is ignored there. It will be "
            "removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
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
            "Memory budget {:.1f} GB → chunk size {}",
            memory_budget_gb,
            effective_chunk_size,
        )
    else:
        effective_chunk_size = chunk_size

    # Single chunk optimization: use standard apriori if data fits
    if n_total <= effective_chunk_size:
        logger.info(
            "Dataset ({}) fits in single chunk ({}), using standard apriori",
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
        "SON streaming: {} total transactions, {} chunks of {}, local_support={:.6f} ({:.1f}% of {:.6f})",
        n_total,
        n_chunks,
        effective_chunk_size,
        local_min_support,
        local_support_factor * 100,
        min_support,
    )

    if not use_gpu:
        result_df = _son_cpu(
            lf,
            item_col,
            effective_chunk_size,
            chunk_sizes,
            n_total,
            min_support,
            local_min_support,
            max_length,
            n_jobs,
            session,
            show_progress,
            progress_callback,
        )
        return (result_df, session) if profile else result_df

    # =========================================================================
    # PASS 1 (GPU): Local frequent itemset mining
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

        # Build boolean matrix for this chunk with LOCAL support threshold.
        # A failure propagates: a skipped chunk contributes no candidates, and
        # an itemset frequent only there would be missing from the result.
        matrix, col_to_item, _ = build_boolean_matrix(
            chunk_lf,
            local_min_support,
            item_col,
        )

        if not col_to_item:
            continue

        # Track all items seen
        all_items.update(col_to_item.values())

        local_frequent = _mine_chunk_gpu(matrix, col_to_item, chunk_n, local_min_support, max_length)

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
        "Pass 1 complete: {} candidate itemsets from {} items",
        len(candidate_itemsets),
        len(all_items),
    )

    # =========================================================================
    # PASS 2 (GPU): Global support counting
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
        "Pass 2 optimization: {}/{} items needed for candidates ({:.1f}% reduction)",
        len(candidate_items),
        len(all_items),
        (1 - len(candidate_items) / len(all_items)) * 100 if all_items else 0,
    )

    global_counts: dict[tuple[int, ...], int] = {itemset: 0 for itemset in candidate_itemsets}

    # Create item -> column name mapping for global counting (only candidate items!)
    sorted_items = sorted(candidate_items)  # Only items in candidates
    item_to_col = {item: f"i_{idx}" for idx, item in enumerate(sorted_items)}
    candidate_list = list(candidate_itemsets)

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

        for itemset, count in _count_candidates_gpu(matrix, candidate_list, sorted_items).items():
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
        "Pass 2 complete: {}/{} candidates are globally frequent (support >= {:.6f})",
        len(results),
        len(candidate_itemsets),
        min_support,
    )

    result_df = _build_result_df(results)

    if profile:
        return result_df, session
    return result_df


def _mine_chunk_gpu(
    matrix: pl.DataFrame,
    col_to_item: dict[str, int],
    n_transactions: int,
    min_support: float,
    max_length: int | None,
) -> list[tuple[int, ...]]:
    """Mine one chunk on the current device with the row-split miner.

    The miner runs on column indices and the items are mapped back here, so
    items of any type (strings, ids beyond int32, an empty basket's None)
    mine as they do on the CPU route.
    """
    import cupy as cp

    from et_miner.core.matrix import _polars_to_sparse_csr
    from et_miner.gpu.bitvec import _build_gpu_bitvec_matrix
    from et_miner.gpu.row_split import _apriori_row_split_multi_gpu

    csr, col_name_to_idx = _polars_to_sparse_csr(matrix)
    items = [None] * len(col_name_to_idx)
    for col, idx in col_name_to_idx.items():
        items[idx] = col_to_item[col]
    device_id = cp.cuda.Device().id
    bitvecs_gpu = _build_gpu_bitvec_matrix(csr)
    del csr
    try:
        result_df = _apriori_row_split_multi_gpu(
            None,
            {idx: idx for idx in range(len(items))},
            n_transactions,
            min_support,
            max_length,
            1,
            bitvecs_list=[(bitvecs_gpu, device_id, n_transactions)],
        )
    finally:
        del bitvecs_gpu
        cp.get_default_memory_pool().free_all_blocks()
    return [tuple(items[c] for c in cols) for cols in result_df["itemset"].to_list()]


def _count_candidates_gpu(
    matrix: pl.DataFrame,
    candidate_itemsets: list[tuple[int, ...]],
    sorted_items: list[int],
) -> dict[tuple[int, ...], int]:
    """Count mixed-length candidates on the current device with the batched kernel."""
    import cupy as cp

    from et_miner.core.matrix import _polars_to_sparse_csr
    from et_miner.gpu.bitvec import _build_gpu_bitvec_matrix
    from et_miner.gpu.kernels import count_itemsets_cuda

    # sorted_items order = matrix column order = bitvec column order (_build_matrix_for_items)
    csr, _ = _polars_to_sparse_csr(matrix)
    bitvecs_gpu = _build_gpu_bitvec_matrix(csr)
    del csr
    item_to_idx = {item: idx for idx, item in enumerate(sorted_items)}
    itemsets_np = [np.array([item_to_idx[i] for i in itemset], dtype=np.int32) for itemset in candidate_itemsets]
    try:
        counts = count_itemsets_cuda(bitvecs_gpu, itemsets_np)
    finally:
        del bitvecs_gpu
        cp.get_default_memory_pool().free_all_blocks()
    return {itemset: int(counts[i]) for i, itemset in enumerate(candidate_itemsets)}


def _chunks(n_chunks: int, desc: str, show_progress: bool):
    """range(n_chunks), wrapped in a tqdm bar when progress is shown."""
    if show_progress and HAS_TQDM:
        return tqdm(range(n_chunks), desc=desc, unit="chunk", total=n_chunks)
    return range(n_chunks)


class _CandidateUnion:
    """The union of the chunks' locally frequent itemsets, deduplicated chunk by chunk.

    Items get ids in the order they are first seen (``seen``), so a chunk's
    column ids map onto them without knowing the later chunks; each length
    keeps one lexsorted int32 array of distinct rows of ids. ``finish``
    renumbers the ids in sorted item order.
    """

    def __init__(self) -> None:
        self.seen: pl.Series | None = None
        self.levels: dict[int, np.ndarray] = {}

    @property
    def n_items(self) -> int:
        return 0 if self.seen is None else len(self.seen)

    def __len__(self) -> int:
        return sum(len(v) for v in self.levels.values())

    def add(self, items: pl.Series, levels: list[np.ndarray]) -> None:
        """Add one chunk: its frequent items (column order) and its levels as column ids."""
        self.seen = (
            items if self.seen is None else pl.concat([self.seen, items.filter(~items.is_in(self.seen.implode()))])
        )
        ids = pl.Series(np.arange(len(self.seen), dtype=np.int32))
        remap = items.replace_strict(self.seen, ids, return_dtype=pl.Int32).to_numpy()
        for sets in levels:
            if len(sets) == 0:
                continue
            k = sets.shape[1]
            rows = np.sort(remap[sets], axis=1)
            if k in self.levels:
                rows = np.concatenate([self.levels[k], rows])
            self.levels[k] = unique_rows(rows, len(self.seen))

    def finish(self) -> tuple[pl.Series, dict[int, np.ndarray]]:
        """(the items in sorted order, per length the lexsorted candidates as column ids into them)."""
        if self.seen is None:
            raise ValueError("no chunk was added")
        order = self.seen.arg_sort().to_numpy()
        rank = np.empty(len(order), dtype=np.int32)
        rank[order] = np.arange(len(order), dtype=np.int32)
        items = self.seen.gather(pl.Series(order))
        return items, {k: unique_rows(np.sort(rank[v], axis=1), len(items)) for k, v in sorted(self.levels.items())}


def _local_levels(
    chunk_lf: pl.LazyFrame,
    local_min_support: float,
    max_length: int | None,
    item_col: str,
    pool: ThreadPoolExecutor | None,
    workers: int,
) -> tuple[pl.Series, list[np.ndarray]] | None:
    """One chunk's locally frequent itemsets: (its frequent items, a lexsorted int32 array of column ids per K).

    Steps: the chunk's CSR at the local threshold (``build_transaction_csr``),
    then the array miner's levels on it (``_mine_levels``, as ``mine_cpu``
    runs them). None when no item is locally frequent.
    """
    tc = build_transaction_csr(chunk_lf, local_min_support, item_col)
    if tc is None:
        return None
    ones = np.arange(tc.n_cols, dtype=np.int32)[:, None]
    emitted = [(ones, tc.counts)]
    _mine_levels(
        tc,
        gen=ones,
        prev=_Level(ones, tc.counts, tc.counts < tc.n_rows),
        emitted=emitted,
        min_count=max(1, _min_count(local_min_support, tc.n_rows)),
        effective_max_length=min(max_length or math.inf, int(np.diff(tc.indptr).max()), tc.n_cols),
        prune_equal_support=False,
        use_generator_pruning=False,
        enable_length_filter=True,
        level_callback=None,
        session=None,
        pool=pool,
        workers=workers,
    )
    return tc.items, [sets for sets, _ in emitted]


def _count_chunk(
    chunk_lf: pl.LazyFrame,
    item_col: str,
    items: pl.Series,
    cands: dict[int, np.ndarray],
    pool: ThreadPoolExecutor | None,
    workers: int,
) -> dict[int, np.ndarray]:
    """One chunk's count of every candidate: its rows mapped onto the candidate items (``_map_rows``), then ``count_itemsets``."""
    column = chunk_lf.select(pl.col(item_col)).collect(engine="in-memory").get_column(item_col)
    indptr, indices = _map_rows(column, items, int(column.list.len().fill_null(1).sum()))
    del column
    return count_itemsets(indptr, indices, len(items), cands, pool, workers)


def _son_cpu(
    lf: pl.LazyFrame,
    item_col: str,
    chunk_size: int,
    chunk_sizes: list[int],
    n_total: int,
    min_support: float,
    local_min_support: float,
    max_length: int | None,
    n_jobs: int,
    session: ProfilingSession | None,
    show_progress: bool,
    progress_callback: Callable[[str, int, int, dict[str, Any]], None] | None,
) -> pl.DataFrame:
    """SON's two passes on the CPU's array miner.

    Steps:

    1. Pass 1 (profile phase ``pass1_local_mining``): per chunk, the locally
       frequent itemsets (``_local_levels``), added to ``_CandidateUnion``,
       which deduplicates them chunk by chunk.
    2. The union's items in sorted order and, per length, its lexsorted
       candidates as column ids into them.
    3. Pass 2 (``pass2_global_counting``): per chunk, the count of every
       candidate (``_count_chunk``), summed in int64.
    4. Keep the candidates whose total reaches the global min_count and build
       the result as the CPU route does (``_emit``).

    A chunk that fails raises: a skipped chunk would drop its candidates or its
    counts while support is still divided by every row.
    """
    n_chunks = len(chunk_sizes)
    workers = _workers(n_jobs)
    pool = ThreadPoolExecutor(workers) if workers > 1 else None
    try:
        if session:
            session.start_phase("pass1_local_mining")
        union = _CandidateUnion()
        bar = _chunks(n_chunks, "Pass 1: Local mining", show_progress)
        for chunk_idx in bar:
            local = _local_levels(
                lf.slice(chunk_idx * chunk_size, chunk_size), local_min_support, max_length, item_col, pool, workers
            )
            if local is not None:
                union.add(*local)
            if show_progress and HAS_TQDM:
                bar.set_postfix(candidates=len(union), items=union.n_items)  # type: ignore[union-attr]
            if progress_callback:
                metrics = {"candidates": len(union), "items": union.n_items, "memory_gb": _get_memory_gb()}
                progress_callback("pass1", chunk_idx, n_chunks, metrics)
        n_candidates = len(union)
        if session:
            session.end_phase(n_candidates=n_candidates, n_items=union.n_items)
        if n_candidates == 0:
            return _empty_result()
        items, cands = union.finish()
        logger.info("Pass 1 complete: {} candidate itemsets from {} items", n_candidates, len(items))

        if session:
            session.start_phase("pass2_global_counting")
        totals = {k: np.zeros(len(c), dtype=np.int64) for k, c in cands.items()}
        for chunk_idx in _chunks(n_chunks, "Pass 2: Global counting", show_progress):
            counts = _count_chunk(lf.slice(chunk_idx * chunk_size, chunk_size), item_col, items, cands, pool, workers)
            for k, c in counts.items():
                totals[k] += c
            if progress_callback:
                progress_callback(
                    "pass2", chunk_idx, n_chunks, {"counted": chunk_idx + 1, "memory_gb": _get_memory_gb()}
                )
        if session:
            session.end_phase(n_counted=n_candidates)
    finally:
        if pool is not None:
            pool.shutdown()

    min_count = _min_count(min_support, n_total)
    levels = [(cands[k][totals[k] >= min_count], totals[k][totals[k] >= min_count]) for k in cands]
    logger.info(
        "Pass 2 complete: {}/{} candidates are globally frequent (support >= {:.6f})",
        sum(len(c) for c, _ in levels),
        n_candidates,
        min_support,
    )
    return _emit(levels, items, n_total)


def _build_matrix_for_items(
    transactions: pl.LazyFrame,
    item_col: str,
    items: list[int],
    item_to_col: dict[int, str],
) -> pl.DataFrame:
    """Boolean matrix with a column per item in `items` — used in SON Pass 2 for consistent mapping."""
    exprs = [pl.col(item_col).list.contains(item).alias(item_to_col[item]) for item in items]

    # In-memory engine: faster than streaming for one list.contains per item (bench/cpu/list_contains_check.py).
    return transactions.select(exprs).collect(engine="in-memory")


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
