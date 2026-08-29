"""Boolean transaction matrix for vectorized Apriori.

This module provides functions to convert transactions to a boolean matrix
representation where rows are transactions and columns are frequent items.
This enables fully vectorized support counting via column AND operations.

Key idea: support({A,B}) = (col("A") & col("B")).sum()

This is:
- Fully SIMD-vectorized (no Python loops for support counting)
- GPU-parallelizable via CUDA bitvector kernels (CuPy)
- O(T × C / P) where P = parallelism factor

For extreme sparse workloads (>2000 items, <5% density), an automatic fallback
to scipy sparse CSR matrices is available (scipy is a core dependency).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from loguru import logger

from et_miner._compat import HAS_TQDM, tqdm
from et_miner.backends import CUPY_INSTALLED
from et_miner.core.result import _min_count

# Scipy for sparse matrix operations
from scipy.sparse import coo_matrix, csr_matrix


if TYPE_CHECKING:
    from scipy.sparse import csr_matrix as CSRMatrix


# Streaming chunk size for wide boolean matrices (Polars 1.37+ width-aware chunking)
# Smaller chunks reduce memory pressure when matrix has many columns (1 per item)
_STREAMING_CHUNK_SIZE = 25_000



# =============================================================================
# Numpy Bitmap SIMD Operations (for K>2 itemsets)
# =============================================================================


def _filter_transactions_by_length(
    matrix: pl.DataFrame,
    k: int,
) -> tuple[pl.DataFrame, int]:
    """Filter transactions that are too short to contain k-itemsets.

    Transactions with fewer than k items cannot contribute to k-itemset support
    counts, so filtering them reduces computation without affecting results.

    Args:
        matrix: Boolean transaction matrix (rows=transactions, cols=items).
        k: Minimum itemset size to support.

    Returns:
        Tuple of:
            - filtered_matrix: Matrix with short transactions removed
            - n_filtered: Number of transactions that were filtered out
    """
    if k <= 1:
        # No filtering needed for k=1
        return matrix, 0

    # Row sum = number of items per transaction
    tx_lengths = matrix.select(pl.sum_horizontal(pl.all())).to_series()
    mask = tx_lengths >= k
    n_filtered = (~mask).sum()

    if n_filtered == 0:
        return matrix, 0

    return matrix.filter(mask), n_filtered


def build_boolean_matrix(
    transactions: pl.LazyFrame,
    min_support: float,
    item_col: str = "items",
) -> tuple[pl.DataFrame, dict[str, int], int]:
    """Convert transactions to boolean matrix for vectorized operations.

    Transforms horizontal transaction data into a boolean matrix where:
    - Each row represents a transaction
    - Each column represents a frequent item (columns named i_0, i_1, ...)
    - Cell value is True if item is in transaction, False otherwise

    Uses vectorized list.contains() for memory-efficient matrix construction.
    While theoretically O(n × m) where m = number of frequent items, this approach
    uses 100-850× LESS memory than explode+pivot due to minimal intermediate allocations.

    Args:
        transactions: LazyFrame with item lists.
        min_support: Minimum support threshold for 1-itemsets.
        item_col: Name of the column containing item lists.

    Returns:
        Tuple of:
            - matrix: Boolean DataFrame (rows=transactions, cols=frequent items)
            - col_to_item: Mapping from column names to item IDs
            - n_transactions: Total transaction count

    Example:
        >>> lf = pl.LazyFrame({"items": [[1, 2], [2, 3], [1, 2, 3]]})
        >>> matrix, col_to_item, n = build_boolean_matrix(lf, 0.5)
        >>> # matrix has columns i_0, i_1, i_2 for frequent items
    """
    # Step 1: Transaction count (streaming)
    n_transactions = transactions.select(pl.len()).collect(engine="streaming").item()

    # Step 2: Frequent 1-itemsets (streaming, vectorized)
    # Use integer count filtering to avoid float precision issues at boundaries.
    # math.ceil ensures we don't include items below threshold due to float rounding.
    min_count = _min_count(min_support, n_transactions)
    freq_1 = (
        transactions.select(pl.col(item_col).explode().alias("item"))
        .group_by("item")
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") >= min_count)
        .with_columns((pl.col("count") / n_transactions).alias("support"))
        .sort("item")
        .collect(engine="streaming")
    )

    if freq_1.height == 0:
        return pl.DataFrame(), {}, n_transactions

    # Step 3: Item -> column mapping
    item_ids = freq_1.get_column("item").to_list()
    col_names = [f"i_{idx}" for idx in range(len(item_ids))]
    col_to_item = dict(zip(col_names, item_ids))

    # Step 4: Build boolean matrix via vectorized list.contains
    # This approach uses MINIMAL intermediate memory compared to explode+pivot.
    # Testing showed pivot uses 100-850× more memory due to intermediate allocations,
    # while list.contains streams efficiently with only the final matrix in memory.
    #
    # Trade-off: O(T × m) vs O(T × k) where m = frequent items, k = avg items/tx
    # In practice, list.contains is faster AND uses dramatically less memory.

    exprs = [pl.col(item_col).list.contains(item_id).alias(col_name) for col_name, item_id in col_to_item.items()]

    # NOTE: Do NOT use engine="streaming" here. The streaming engine has a bug
    # with list.contains() that undercounts True values by ~2%. This causes items
    # near the support threshold to be incorrectly filtered out.
    # Verified on 2.5M transactions: streaming counted 4,073 vs correct 4,156.
    # See: docs/UNIMPLEMENTED_IDEAS.md for full analysis.
    matrix = transactions.select(exprs).collect()

    return matrix, col_to_item, n_transactions


def count_support_vectorized(
    matrix: pl.DataFrame,
    itemsets: list[tuple[str, ...]],
) -> dict[tuple[str, ...], int]:
    """Count support for itemsets using vectorized column AND operations.

    Computes support for all itemsets in a single vectorized pass:
        support({A,B,C}) = (col(A) & col(B) & col(C)).sum()

    This is fully vectorized with no Python loops for counting.

    Args:
        matrix: Boolean DataFrame from build_boolean_matrix().
        itemsets: List of itemsets as tuples of column names.

    Returns:
        Dictionary mapping itemsets to support counts.

    Example:
        >>> counts = count_support_vectorized(matrix, [("i_0", "i_1")])
        >>> # Returns {("i_0", "i_1"): 42} meaning 42 transactions contain both
    """
    if not itemsets:
        return {}

    # Build expressions for ALL itemsets at once
    # For K=2: Use direct & operator (12% faster than all_horizontal)
    # For K>2: Use all_horizontal() for efficient multi-column AND
    exprs = []
    for itemset in itemsets:
        alias = "__".join(itemset)
        if len(itemset) == 2:
            # Direct & is faster for K=2 (avoids all_horizontal overhead)
            expr = (pl.col(itemset[0]) & pl.col(itemset[1])).sum().alias(alias)
        else:
            expr = pl.all_horizontal([pl.col(c) for c in itemset]).sum().alias(alias)
        exprs.append(expr)

    # Execute all support counts in a single vectorized pass
    # Always use streaming engine (GPU path is handled by count_support_gpu_bitvec)
    with pl.Config(streaming_chunk_size=_STREAMING_CHUNK_SIZE):
        counts_df = matrix.lazy().select(exprs).collect(engine="streaming")

    # Extract results back to dictionary
    results = {}
    for itemset in itemsets:
        key = "__".join(itemset)
        results[itemset] = counts_df.get_column(key).item()

    return results


def _calculate_optimal_batch_size(
    n_itemsets: int,
    n_transactions: int,
    avg_itemset_size: int = 3,
) -> int:
    """Calculate optimal batch size based on memory budget.

    Dynamically determines batch size to target approximately 500MB per batch,
    avoiding OOM errors while maintaining good throughput.

    Args:
        n_itemsets: Total number of itemsets to process.
        n_transactions: Number of transactions in the matrix.
        avg_itemset_size: Average number of items per itemset (default 3).

    Returns:
        Optimal batch size clamped to range [100, 10_000].

    Note:
        Memory estimation accounts for Polars' bit-packed boolean storage
        (8 booleans per byte). This is the actual in-memory representation
        used by Polars for boolean arrays in the Arrow format.
    """
    # Estimated memory per itemset: n_transactions × avg_itemset_size × sizeof(bool)
    # Polars uses bit-packed booleans (8 per byte) in Arrow format for storage.
    # The // 8 accounts for this efficient packing.
    bytes_per_itemset = (n_transactions * avg_itemset_size) // 8

    if bytes_per_itemset == 0:
        return 10_000  # Default fallback for very small data

    # Target: max 500MB per batch for good throughput with memory headroom
    memory_budget_bytes = 500 * 1024 * 1024  # 500MB
    optimal_batch = memory_budget_bytes // bytes_per_itemset

    # Clamp to reasonable range: min 100 (avoid too many batches), max 10_000 (original default)
    return max(100, min(optimal_batch, 10_000))


def count_support_batched(
    matrix: pl.DataFrame,
    itemsets: list[tuple[str, ...]],
    n_transactions: int,
    batch_size: int | str | None = 10_000,
    use_gpu: bool | str = False,
    show_progress: bool = False,
    sparse: bool | None = None,
    n_jobs: int = 1,
    min_transaction_length: int | None = None,
    enable_length_filter: bool = True,
) -> dict[tuple[str, ...], int]:
    """Count support for itemsets with optional GPU acceleration.

    GPU mode: Uses custom CUDA kernels via count_support_gpu_bitvec() for
    maximum throughput (~130M+ rows/sec on H200).
    CPU mode: Batched processing to control memory, with optional sparse mode.
    Sparse mode: Uses scipy CSR for extreme sparse workloads (auto-detected),
        with optional parallelization for k>2 itemsets.

    For extreme sparse workloads (>500 items, <10% density, or >1GB estimated size),
    the function automatically switches to scipy sparse matrices which can reduce
    memory usage by 25-50x.

    Args:
        matrix: Boolean DataFrame from build_boolean_matrix().
        itemsets: List of itemsets as tuples of column names.
        n_transactions: Total transaction count.
        batch_size: Number of itemsets per batch (used for CPU mode).
            - int: Fixed batch size
            - "auto": Dynamically calculate based on memory budget (500MB target)
            - None: Process all at once (no batching)
        use_gpu: If True, use custom CUDA kernels for support counting.
            Requires CuPy to be installed. Raises ImportError if unavailable.
            For backward compatibility, also accepts string values:
            - "gpu": equivalent to use_gpu=True
            - "streaming" or other: equivalent to use_gpu=False
        show_progress: If True, display progress bar (requires tqdm).
        sparse: If True, force scipy sparse. If False, force Polars.
            If None (default), auto-detect based on workload characteristics.
        n_jobs: Number of parallel workers for sparse k>2 counting.
            - -1: Use all available CPUs
            - 1: Sequential execution
            - >1: Use that many workers
            Only used in sparse mode. On Python 3.13t/3.14t (no GIL), achieves
            true parallelism for significant speedups on large workloads.
        min_transaction_length: Minimum transaction length filter (k).
            If provided, filters out transactions with fewer than k items
            before support counting. Auto-detected from itemsets if None.
        enable_length_filter: If True, apply transaction length filtering
            (default: True). Set to False to disable optimization.

    Returns:
        Dictionary mapping itemsets to support counts.

    Raises:
        ImportError: If use_gpu=True but CuPy is not installed.
    """
    # Backward compatibility: accept string "engine" values
    # "gpu" -> use_gpu=True, anything else -> use_gpu=False
    if isinstance(use_gpu, str):
        use_gpu = use_gpu == "gpu"
    # Apply transaction length filtering if enabled
    filtered_matrix = matrix
    actual_n_transactions = n_transactions

    if enable_length_filter and itemsets:
        # Auto-detect k from itemsets if not provided
        k = min_transaction_length
        if k is None:
            k = len(itemsets[0]) if itemsets else 2

        filtered_matrix, n_filtered = _filter_transactions_by_length(matrix, k)
        actual_n_transactions = filtered_matrix.height

        # Log filtering results
        if n_filtered > 0:
            logger.debug(
                "Transaction length filter: removed %d/%d transactions (%.1f%%) with < %d items",
                n_filtered,
                n_transactions,
                100.0 * n_filtered / n_transactions,
                k,
            )

    # GPU path: use_gpu=True means custom CUDA kernels, period.
    if use_gpu:
        if not CUPY_INSTALLED:
            raise ImportError(
                "GPU support requires CuPy. Install with: pip install et-miner[gpu]\nor: pip install cupy-cuda12x"
            )
        from et_miner.gpu.bitvec import count_support_gpu_bitvec

        return count_support_gpu_bitvec(filtered_matrix, itemsets, show_progress)

    from et_miner.core.sparse import _choose_counting_strategy, _estimate_density, count_support_sparse

    # Determine counting strategy for CPU path
    if sparse is None:
        # Auto-detect based on workload
        n_items = len(filtered_matrix.columns)
        density = _estimate_density(filtered_matrix)
        strategy = _choose_counting_strategy(n_items, actual_n_transactions, density)
    else:
        strategy = "sparse" if sparse else "polars"

    # Use sparse strategy if selected
    if strategy == "sparse":
        return count_support_sparse(filtered_matrix, itemsets, show_progress, n_jobs)

    # Resolve batch_size: "auto" calculates based on memory budget
    effective_batch_size: int | None
    if batch_size == "auto":
        effective_batch_size = _calculate_optimal_batch_size(len(itemsets), actual_n_transactions)
    else:
        effective_batch_size = batch_size

    # batch_size=None or batch larger than candidates: single pass
    if effective_batch_size is None or effective_batch_size >= len(itemsets):
        return count_support_vectorized(filtered_matrix, itemsets)

    # CPU/streaming: simple batched processing
    # NOTE: Previous implementation used pl.collect_all() with chunking, attempting
    # to enable CSE (Common Subexpression Elimination) optimization. However,
    # benchmarks showed single-select is ~3x faster than collect_all() due to:
    # 1. collect_all() overhead for creating/managing multiple LazyFrames
    # 2. CSE benefits don't materialize for independent boolean AND operations
    # 3. Simpler code path with less Python overhead
    #
    # The current approach processes batches sequentially with single select(),
    # which is both faster and simpler.

    results = {}
    batches = [itemsets[i : i + effective_batch_size] for i in range(0, len(itemsets), effective_batch_size)]

    batch_iter = batches
    if show_progress and HAS_TQDM:
        batch_iter = tqdm(batches, desc="Counting support", unit="batch")

    for batch in batch_iter:
        # Process each batch with a single select() call
        # This is simpler and ~3x faster than collect_all() with multiple LazyFrames
        batch_results = count_support_vectorized(filtered_matrix, batch)
        results.update(batch_results)

    return results


# =============================================================================
# Scipy Sparse Matrix Support (for extreme sparse workloads)
# =============================================================================


def _polars_to_sparse_csr(
    matrix: pl.DataFrame,
) -> tuple[CSRMatrix, dict[str, int]]:
    """Convert Polars boolean DataFrame to scipy CSR WITHOUT dense intermediate.

    Strategy: Build COO format directly from column data, then convert to CSR.
    This avoids the 8× memory explosion of to_numpy() on booleans.

    The key insight is that we extract True indices per column using Polars'
    efficient arg_where(), then concatenate into COO format. This way we never
    materialize the full dense boolean array.

    Args:
        matrix: Boolean DataFrame (rows=transactions, cols=items).

    Returns:
        Tuple of:
            - csr: scipy CSR matrix with uint8 data (1 for True)
            - col_name_to_idx: Mapping from column names to column indices

    Raises:
        ImportError: If scipy is not installed.

    Memory profile:
        For 5000 items × 1M transactions at 1% density:
        - Dense would be: 5000 × 1M × 1 byte = 5 GB
        - Sparse stores: 50M non-zeros × (4 bytes row + 4 bytes col + 1 byte data) ≈ 450 MB
    """
    col_name_to_idx = {name: i for i, name in enumerate(matrix.columns)}
    n_rows = matrix.height
    n_cols = len(matrix.columns)

    # Collect all (row, col) pairs where value is True
    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []

    # Batch all arg_where calls in a single select for efficiency
    # This reduces Polars overhead from O(n_columns) selects to O(1)
    # Use implode() to create list columns since arg_where returns variable lengths
    all_indices = matrix.select([pl.arg_where(pl.col(name)).implode().alias(name) for name in matrix.columns])

    # Process all columns from the single result
    for col_idx, col_name in enumerate(matrix.columns):
        # Get indices where this column is True using Polars' efficient arg_where
        # This is O(n) and doesn't require unpacking the bit-packed booleans
        # Use explode() to convert from list column back to regular series
        # drop_nulls() handles empty lists which explode to a single null value
        true_indices = all_indices.get_column(col_name).explode().drop_nulls().to_numpy()

        if len(true_indices) > 0:
            rows_list.append(true_indices.astype(np.int64))
            cols_list.append(np.full(len(true_indices), col_idx, dtype=np.int64))

    # Build COO then convert to CSR (both operations are O(nnz))
    if rows_list:
        all_rows = np.concatenate(rows_list)
        all_cols = np.concatenate(cols_list)
    else:
        all_rows = np.array([], dtype=np.int64)
        all_cols = np.array([], dtype=np.int64)

    # Use int32 to avoid overflow in M.T @ M matrix multiply (uint8 overflows at 256)
    data = np.ones(len(all_rows), dtype=np.int32)

    coo = coo_matrix((data, (all_rows, all_cols)), shape=(n_rows, n_cols))
    return coo.tocsr(), col_name_to_idx


def _build_csr_from_transactions(
    transactions: pl.LazyFrame,
    min_support: float,
    item_col: str = "items",
) -> tuple[csr_matrix, dict[int, int], int] | None:
    """Build CSR sparse matrix directly from transactions, bypassing dense boolean.

    The standard path (build_boolean_matrix → _polars_to_sparse_csr) creates a
    dense Polars boolean DataFrame with n_rows × n_items booleans.  For 205M × 1006
    that is ~206 GB — more than any machine has.

    This function builds CSR format straight from the Parquet list column via
    Polars explode + inner-join, using O(nnz) CPU memory (~5 GB for 214M AlphaFold)
    instead of O(n_rows × n_items).

    Pipeline: LazyFrame → explode → join freq items → COO → CSR

    Args:
        transactions: LazyFrame with a list column of item IDs.
        min_support: Minimum support threshold (0.0–1.0).
        item_col: Name of the list column.

    Returns:
        Tuple of (csr_matrix, col_to_item, n_transactions) or None if no
        frequent items survive the support threshold.  col_to_item maps
        bitvector column index (int) → original item ID (int).
    """
    # ── transaction count ───────────────────────────────────────────────
    n_transactions = transactions.select(pl.len()).collect(engine="streaming").item()
    min_count = _min_count(min_support, n_transactions)

    logger.info(
        "Direct CSR path: {} transactions, min_count={}",
        f"{n_transactions:,}",
        min_count,
    )

    # ── frequent 1-itemsets ─────────────────────────────────────────────
    freq_1 = (
        transactions.select(pl.col(item_col).explode())
        .group_by(item_col)
        .agg(pl.len().alias("__count"))
        .filter(pl.col("__count") >= min_count)
        .sort(item_col)
        .collect(engine="streaming")
    )

    if freq_1.height == 0:
        return None

    item_ids = freq_1.get_column(item_col).to_list()
    n_items = len(item_ids)
    col_to_item: dict[int, int] = dict(enumerate(item_ids))

    logger.info(f"Direct CSR path: {n_items} frequent items")

    # ── explode + join → (row_idx, col_idx) COO pairs ──────────────────
    item_mapping = pl.DataFrame(
        {
            item_col: item_ids,
            "__col_idx": np.arange(n_items, dtype=np.int64),
        }
    ).lazy()

    exploded = (
        transactions.with_row_index("__row_idx")
        .explode(item_col)
        .join(item_mapping, on=item_col, how="inner")
        .select(
            pl.col("__row_idx").cast(pl.Int64),
            pl.col("__col_idx"),
        )
        .collect(engine="streaming")
    )

    if exploded.height == 0:
        return None

    logger.info(f"Direct CSR path: {exploded.height:,} non-zeros")

    # ── COO → CSR ──────────────────────────────────────────────────────
    row_indices = exploded.get_column("__row_idx").to_numpy()
    col_indices = exploded.get_column("__col_idx").to_numpy()
    del exploded

    data = np.ones(len(row_indices), dtype=np.int32)
    csr = coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_transactions, n_items),
    ).tocsr()
    del row_indices, col_indices, data

    return csr, col_to_item, n_transactions

