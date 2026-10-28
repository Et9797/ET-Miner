"""Multi-GPU Streaming Apriori implementation using SON algorithm.

The chunks run one after another, each row-split over every GPU: pass 1 mines
a chunk on the row-split miner at the local support, the union of the chunks'
levels is pruned by the partition bound, and pass 2 counts the surviving
candidates on each chunk with the batched kernel, the GPUs' counts summed
(``streaming/son.py::_son_arrays``, shared with single-GPU SON). The
wave-based variant (one chunk per GPU, a Polars boolean matrix per chunk) lost
every measured workload (route consolidation S2: dslxl 12.7 s against 306 s)
and is gone.

Reference:
    Savasere, A., Omiecinski, E. R., & Navathe, S. B. (1995).
    "An Efficient Algorithm for Mining Association Rules in Large Databases"
    VLDB 1995.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import polars as pl


# SON-specific helpers live in son.py; generic ones in matrix.py (foundation layer)
from et_miner.streaming.son import (
    LOCAL_SUPPORT_FACTOR,
    _check_chunk_size,
    _check_local_support_factor,
    _estimate_chunk_size_from_memory,
    _son_arrays,
)
from et_miner.core.result import (
    _empty_result,
    _item_dtype,
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

    Processes the dataset in chunks, each row-split over every GPU, enabling
    analysis of datasets that don't fit in memory.

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
        sparse: Removed; only None is accepted (any other value raises).
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
    if sparse is not None:
        raise ValueError(
            "sparse= was removed with the scipy sparse counter (route consolidation C4): no route reads it. "
            "Drop the argument."
        )
    _check_local_support_factor(local_support_factor)
    if memory_budget_gb is None:
        _check_chunk_size(chunk_size)

    # Validate GPU availability
    if not has_cupy():
        raise MiningError(
            "apriori_streaming_multi_gpu requires CuPy and a CUDA device. "
            "Install GPU support with: pip install 'et-miner[gpu]'"
        )

    available_gpus = get_gpu_count()
    if available_gpus == 0:
        raise MiningError("No CUDA GPUs detected")


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
        return _empty_result(_item_dtype(lf, item_col))

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
            n_jobs=n_jobs,
        )

    n_chunks = math.ceil(n_total / chunk_size)
    local_min_support = min_support * local_support_factor

    # Pre-calculate chunk sizes
    chunk_sizes = []
    for chunk_idx in range(n_chunks):
        offset = chunk_idx * chunk_size
        remaining = n_total - offset
        chunk_sizes.append(min(chunk_size, remaining))

    logger.info(
        "Multi-GPU SON streaming: {} total transactions, {} chunks, each row-split over "
        "{} GPUs, local_support={:.6f} ({:.1f}% of {:.6f})",
        n_total,
        n_chunks,
        effective_n_gpus,
        local_min_support,
        local_support_factor * 100,
        min_support,
    )

    return _son_arrays(
        lf, item_col, chunk_size, chunk_sizes, n_total, min_support, local_min_support, max_length, n_jobs,
        None, show_progress, progress_callback, gpus=effective_n_gpus,
    )
