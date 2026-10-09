"""Multi-GPU Streaming Apriori implementation using SON algorithm.

This module extends the single-GPU streaming implementation to leverage multiple
GPUs for parallel chunk processing. The architecture follows a wave-based approach:

Pass 1 (Local Mining):
┌─────────────┬─────────────┬─────────────┬─────────────┐
│  Chunk 0    │  Chunk 1    │  Chunk 2    │  Chunk 3    │
│  GPU 0      │  GPU 1      │  GPU 2      │  GPU 3      │
└─────────────┴─────────────┴─────────────┴─────────────┘
       ↓              ↓              ↓              ↓
                 [Merge candidate_itemsets]
                        ↓
Pass 2 (Global Counting):
┌─────────────┬─────────────┬─────────────┬─────────────┐
│  Chunk 0    │  Chunk 1    │  Chunk 2    │  Chunk 3    │
│  GPU 0      │  GPU 1      │  GPU 2      │  GPU 3      │
└─────────────┴─────────────┴─────────────┴─────────────┘
       ↓              ↓              ↓              ↓
                    [Sum global_counts]

Each wave processes n_gpus chunks simultaneously using ThreadPoolExecutor.
GPU memory is cleaned after each chunk to prevent accumulation.

Reference:
    Savasere, A., Omiecinski, E. R., & Navathe, S. B. (1995).
    "An Efficient Algorithm for Mining Association Rules in Large Databases"
    VLDB 1995.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import polars as pl

from et_miner._compat import HAS_TQDM, tqdm

# SON-specific helpers live in son.py; generic ones in matrix.py (foundation layer)
from et_miner.streaming.son import (
    LOCAL_SUPPORT_FACTOR,
    _build_matrix_for_items,
    _check_local_support_factor,
    _count_candidates_gpu,
    _estimate_chunk_size_from_memory,
    _get_memory_gb,
    _mine_chunk_gpu,
)
from et_miner.core.matrix import build_boolean_matrix
from et_miner.core.result import (
    _build_result_df,
    _empty_result,
    _min_count,
)
from et_miner.backends import CUPY_INSTALLED, get_gpu_count, has_cupy
from et_miner.exceptions import MiningError


if TYPE_CHECKING:
    pass


from loguru import logger


def _get_gpu_memory_info(device_id: int = 0) -> tuple[float, float]:
    """Get GPU memory info (free, total) in GB.

    Args:
        device_id: GPU device ID.

    Returns:
        Tuple of (free_gb, total_gb).
    """
    if not CUPY_INSTALLED:
        return (0.0, 0.0)
    try:
        import cupy as cp

        with cp.cuda.Device(device_id):
            free, total = cp.cuda.runtime.memGetInfo()
            return free / 1e9, total / 1e9
    except Exception:
        return (0.0, 0.0)


def _cleanup_gpu_memory(device_id: int) -> None:
    """Free all cached GPU memory for a device.

    Args:
        device_id: GPU device ID to clean.
    """
    if not CUPY_INSTALLED:
        return
    try:
        import cupy as cp

        with cp.cuda.Device(device_id):
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as e:
        logger.warning("Failed to cleanup GPU {} memory: {}", device_id, e)


# =============================================================================
# Multi-GPU Streaming Apriori
# =============================================================================


def apriori_streaming_multi_gpu(
    transactions: pl.LazyFrame | pl.DataFrame,
    min_support: float = 0.5,
    max_length: int | None = None,
    item_col: str = "items",
    n_gpus: int = 8,
    chunk_size: int = 10_000_000,
    memory_budget_gb: float | None = None,
    local_support_factor: float = LOCAL_SUPPORT_FACTOR,
    batch_size: int | None = 10_000,
    show_progress: bool = True,
    sparse: bool | None = None,
    n_jobs: int = 1,
    progress_callback: Callable[[str, int, int, dict[str, Any]], None] | None = None,
) -> pl.DataFrame:
    """Find frequent itemsets using multi-GPU streaming SON algorithm.

    Processes the dataset in chunks across multiple GPUs, enabling analysis of
    massive datasets that don't fit in memory. Uses wave-based processing where
    each wave processes n_gpus chunks simultaneously.

    The SON (Savasere-Omiecinski-Navathe) algorithm:

    1. Pass 1 - Local Mining: Find locally frequent itemsets in each chunk
       with a lowered support threshold (local_support_factor × min_support),
       one chunk per GPU on the row-split miner.

    2. Pass 2 - Global Counting: Count support for all candidate itemsets
       across the full dataset with the batched itemset kernel, again one
       chunk per GPU.

    Memory usage is O(chunk_size × n_items × n_gpus), with GPU memory cleaned
    after each chunk to prevent accumulation.

    Args:
        transactions: Transaction data with item lists (LazyFrame recommended).
        min_support: Minimum global support threshold (0.0-1.0).
        max_length: Maximum itemset length (None = unlimited).
        item_col: Column name with item lists.
        n_gpus: Number of GPUs to use (default 8).
        chunk_size: Number of transactions per chunk (default 10M).
        local_support_factor: Factor to lower local support threshold (default
            ``LOCAL_SUPPORT_FACTOR``, 0.95), in (0, 1]; every value gives the same
            result. Lower values make pass 1 mine more local itemsets, and pass 2
            counts every one of them. Near 0 every chunk's local min_count is 1,
            and pass 1 mines each chunk's whole lattice.
        batch_size: Accepted for parity with apriori_streaming; the GPU
            passes do not read it.
        show_progress: If True, display progress bars (requires tqdm).
        sparse: Accepted for parity with apriori_streaming; the GPU passes
            do not read it.
        n_jobs: Accepted for parity with apriori_streaming; the GPU passes do
            not read it.
        progress_callback: Optional callback for progress updates. Called with:
            (phase: str, chunk_idx: int, n_chunks: int, metrics: dict)
            where phase is "pass1" or "pass2", and metrics contains
            {candidates, items, memory_gb, gpu_id} for pass1 or
            {counted, memory_gb, gpu_id} for pass2.

    Returns:
        DataFrame with columns [itemset, support]

    Raises:
        RuntimeError: If CuPy is not available or no GPUs are detected.

    Example:
        >>> # Process 1 trillion transactions on 8 H200 GPUs
        >>> df = pl.scan_parquet("huge_dataset/*.parquet")
        >>> result = apriori_streaming_multi_gpu(
        ...     df,
        ...     min_support=0.001,
        ...     n_gpus=8,
        ...     chunk_size=100_000_000,  # 100M per chunk
        ...     show_progress=True,
        ... )
    """
    _check_local_support_factor(local_support_factor)

    # Validate GPU availability
    if not has_cupy():
        raise MiningError(
            "apriori_streaming_multi_gpu requires CuPy and a CUDA device. "
            "Install GPU support with: pip install 'et-miner[gpu]'"
        )

    available_gpus = get_gpu_count()
    if available_gpus == 0:
        raise MiningError("No CUDA GPUs detected")

    import cupy as cp

    effective_n_gpus = min(n_gpus, available_gpus)
    if effective_n_gpus < n_gpus:
        logger.warning(
            "Requested {} GPUs but only {} available, using {}",
            n_gpus,
            available_gpus,
            effective_n_gpus,
        )

    # Log GPU memory info
    for gpu_id in range(effective_n_gpus):
        free_gb, total_gb = _get_gpu_memory_info(gpu_id)
        logger.info("GPU {}: {:.1f} GB free / {:.1f} GB total", gpu_id, free_gb, total_gb)

    lf = transactions.lazy() if isinstance(transactions, pl.DataFrame) else transactions

    # Phase 0: Get total transaction count
    n_total = lf.select(pl.len()).collect(engine="streaming").item()

    if n_total == 0:
        return _empty_result()

    # memory_budget_gb overrides chunk_size, in streaming/son.py's ORDER: resolve
    # the effective chunk size first, THEN test whether the data fits in one.
    #
    # Placed after the single-chunk shortcut instead, the parameter was still
    # dropped on every dataset below the 10M default -- which is exactly the
    # small-budget case, on the one path whose reason to exist is not exceeding
    # memory. A budget of 0.25 GB derives a ~1M-row chunk, so a 5M-row dataset
    # must run as five chunks and previously ran as one.
    if memory_budget_gb is not None:
        chunk_size = _estimate_chunk_size_from_memory(memory_budget_gb)
        logger.info(
            "Memory budget {:.3f} GB -> chunk size {}", memory_budget_gb, chunk_size
        )

    # Single chunk optimization: use standard apriori if data fits
    if n_total <= chunk_size:
        logger.info(
            "Dataset ({}) fits in single chunk ({}), using standard apriori",
            n_total,
            chunk_size,
        )
        from et_miner.core.apriori import apriori

        return apriori(
            transactions,
            min_support=min_support,
            max_length=max_length,
            item_col=item_col,
            use_gpu=True,
            n_gpus=effective_n_gpus,
            batch_size=batch_size,
            show_progress=show_progress,
            sparse=sparse,
            n_jobs=n_jobs,
        )

    # Calculate number of chunks and waves
    n_chunks = math.ceil(n_total / chunk_size)
    n_waves = math.ceil(n_chunks / effective_n_gpus)
    local_min_support = min_support * local_support_factor

    # Pre-calculate chunk sizes
    chunk_sizes = []
    for chunk_idx in range(n_chunks):
        offset = chunk_idx * chunk_size
        remaining = n_total - offset
        chunk_sizes.append(min(chunk_size, remaining))

    logger.info(
        "Multi-GPU SON streaming: {} total transactions, {} chunks, {} waves, "
        "{} GPUs, local_support={:.6f} ({:.1f}% of {:.6f})",
        n_total,
        n_chunks,
        n_waves,
        effective_n_gpus,
        local_min_support,
        local_support_factor * 100,
        min_support,
    )

    # =========================================================================
    # PASS 1: Local frequent itemset mining (parallel across GPUs)
    # =========================================================================
    candidate_itemsets: set[tuple[int, ...]] = set()
    all_items: set[int] = set()
    candidates_lock = threading.Lock()

    def process_chunk_pass1(chunk_idx: int, gpu_id: int) -> tuple[set[tuple[int, ...]], set[int]]:
        """Process a single chunk for Pass 1 on a specific GPU.

        Returns:
            Tuple of (local_frequent_itemsets, local_items)
        """
        try:
            # Set GPU device for this thread
            with cp.cuda.Device(gpu_id):
                offset = chunk_idx * chunk_size
                chunk_lf = lf.slice(offset, chunk_size)
                chunk_n = chunk_sizes[chunk_idx]

                if chunk_n == 0:
                    return set(), set()

                # Build boolean matrix for this chunk with LOCAL support threshold
                matrix, col_to_item, _ = build_boolean_matrix(
                    chunk_lf,
                    local_min_support,
                    item_col,
                )

                if not col_to_item:
                    return set(), set()

                local_items = set(col_to_item.values())

                local_frequent = _mine_chunk_gpu(matrix, col_to_item, chunk_n, local_min_support, max_length)

                return set(local_frequent), local_items

        finally:
            # Clean up GPU memory after processing
            _cleanup_gpu_memory(gpu_id)

    # Process chunks in waves
    wave_iter = range(n_waves)
    if show_progress and HAS_TQDM:
        wave_iter = tqdm(
            wave_iter,
            desc="Pass 1: Local mining (waves)",
            unit="wave",
            total=n_waves,
        )

    for wave_idx in wave_iter:
        start_chunk = wave_idx * effective_n_gpus
        end_chunk = min(start_chunk + effective_n_gpus, n_chunks)

        logger.debug(
            "Wave {}: chunks {}-{} on GPUs 0-{}",
            wave_idx,
            start_chunk,
            end_chunk - 1,
            effective_n_gpus - 1,
        )

        with ThreadPoolExecutor(max_workers=effective_n_gpus) as executor:
            futures = {}
            for i, chunk_idx in enumerate(range(start_chunk, end_chunk)):
                gpu_id = i % effective_n_gpus
                future = executor.submit(process_chunk_pass1, chunk_idx, gpu_id)
                futures[future] = (chunk_idx, gpu_id)

            # A failed chunk propagates: a dropped chunk contributes no
            # candidates, and an itemset frequent only there would be lost.
            for future in as_completed(futures):
                chunk_idx, gpu_id = futures[future]
                local_itemsets, local_items = future.result()

                with candidates_lock:
                    candidate_itemsets.update(local_itemsets)
                    all_items.update(local_items)

                # Progress callback
                if progress_callback:
                    progress_callback(
                        "pass1",
                        chunk_idx,
                        n_chunks,
                        {
                            "candidates": len(candidate_itemsets),
                            "items": len(all_items),
                            "memory_gb": _get_memory_gb(),
                            "gpu_id": gpu_id,
                        },
                    )

        if show_progress and HAS_TQDM:
            wave_iter.set_postfix(  # type: ignore
                candidates=len(candidate_itemsets),
                items=len(all_items),
            )

    if not candidate_itemsets:
        return _empty_result()

    logger.info(
        "Pass 1 complete: {} candidate itemsets from {} items",
        len(candidate_itemsets),
        len(all_items),
    )

    # =========================================================================
    # PASS 2: Global support counting (parallel across GPUs)
    # =========================================================================
    candidate_items: set[int] = set()
    for itemset in candidate_itemsets:
        candidate_items.update(itemset)

    logger.info(
        "Pass 2 optimization: {}/{} items needed for candidates ({:.1f}% reduction)",
        len(candidate_items),
        len(all_items),
        (1 - len(candidate_items) / len(all_items)) * 100 if all_items else 0,
    )

    # Thread-safe global counts
    global_counts: dict[tuple[int, ...], int] = {
        itemset: 0 for itemset in candidate_itemsets
    }
    counts_lock = threading.Lock()

    # Create item -> column name mapping
    sorted_items = sorted(candidate_items)
    item_to_col = {item: f"i_{idx}" for idx, item in enumerate(sorted_items)}

    candidate_list = list(candidate_itemsets)

    def process_chunk_pass2(chunk_idx: int, gpu_id: int) -> dict[tuple[int, ...], int]:
        """Process a single chunk for Pass 2 on a specific GPU.

        Returns:
            Dict mapping itemsets to their counts in this chunk.
        """
        try:
            with cp.cuda.Device(gpu_id):
                offset = chunk_idx * chunk_size
                chunk_lf = lf.slice(offset, chunk_size)
                chunk_n = chunk_sizes[chunk_idx]

                if chunk_n == 0:
                    return {}

                # Build boolean matrix with consistent column mapping
                matrix = _build_matrix_for_items(
                    chunk_lf,
                    item_col,
                    sorted_items,
                    item_to_col,
                )

                if matrix.height == 0:
                    return {}

                return _count_candidates_gpu(matrix, candidate_list, sorted_items)

        finally:
            _cleanup_gpu_memory(gpu_id)

    # Process Pass 2 in waves
    wave_iter2 = range(n_waves)
    if show_progress and HAS_TQDM:
        wave_iter2 = tqdm(
            wave_iter2,
            desc="Pass 2: Global counting (waves)",
            unit="wave",
            total=n_waves,
        )

    for wave_idx in wave_iter2:
        start_chunk = wave_idx * effective_n_gpus
        end_chunk = min(start_chunk + effective_n_gpus, n_chunks)

        logger.debug(
            "Wave {}: chunks {}-{} on GPUs 0-{}",
            wave_idx,
            start_chunk,
            end_chunk - 1,
            effective_n_gpus - 1,
        )

        with ThreadPoolExecutor(max_workers=effective_n_gpus) as executor:
            futures = {}
            for i, chunk_idx in enumerate(range(start_chunk, end_chunk)):
                gpu_id = i % effective_n_gpus
                future = executor.submit(process_chunk_pass2, chunk_idx, gpu_id)
                futures[future] = (chunk_idx, gpu_id)

            # A failed chunk propagates: its counts would be missing while
            # support is still divided by the full row count.
            for future in as_completed(futures):
                chunk_idx, gpu_id = futures[future]
                chunk_counts = future.result()

                # Thread-safe accumulation
                with counts_lock:
                    for itemset, count in chunk_counts.items():
                        global_counts[itemset] += count

                # Progress callback
                if progress_callback:
                    progress_callback(
                        "pass2",
                        chunk_idx,
                        n_chunks,
                        {
                            "counted": chunk_idx + 1,
                            "memory_gb": _get_memory_gb(),
                            "gpu_id": gpu_id,
                        },
                    )

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

    return _build_result_df(results)
