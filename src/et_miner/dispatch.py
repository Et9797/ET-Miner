"""Framework-agnostic input dispatch layer for et-miner.

This module provides the `to_csr()` function that converts any supported input
format to CSR arrays, making et-miner work with Polars, Pandas, lists, or raw arrays.

Supported input formats:
- polars.DataFrame with items column (list of item IDs per transaction)
- pandas.DataFrame with items column
- list[list[int]] - direct transaction lists
- scipy.sparse.csr_matrix - passthrough
- tuple (indptr, indices, n_rows, n_cols) - raw CSR arrays

Example:
    >>> from et_miner.dispatch import to_csr
    >>> import polars as pl
    >>>
    >>> # Polars DataFrame
    >>> df = pl.DataFrame({"items": [[0, 1], [1, 2], [0, 2]]})
    >>> indptr, indices, n_rows, n_cols, item_map = to_csr(df)
    >>>
    >>> # Raw list
    >>> transactions = [[0, 1], [1, 2], [0, 2]]
    >>> indptr, indices, n_rows, n_cols, item_map = to_csr(transactions, n_items=3)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union
import numpy as np

if TYPE_CHECKING:
    import polars as pl
    import pandas as pd
    from scipy.sparse import csr_matrix

# Type alias for CSR result
CSRResult = tuple[np.ndarray, np.ndarray, int, int, dict[Any, int] | None]


def to_csr(
    data: Any,
    *,
    item_col: str = "items",
    n_items: int | None = None,
) -> CSRResult:
    """Convert any supported input format to CSR arrays.

    This is the core dispatch function that makes et-miner framework-agnostic.
    It inspects the input type and routes to the appropriate conversion function.

    Args:
        data: Input data in one of the supported formats:
            - polars.DataFrame: Must have a column with lists of item IDs
            - pandas.DataFrame: Must have a column with lists of item IDs
            - list[list[int]]: Direct transaction lists
            - scipy.sparse.csr_matrix: Passthrough
            - tuple: (indptr, indices, n_rows, n_cols) raw CSR arrays
        item_col: Column name containing items (for DataFrame inputs)
        n_items: Number of unique items (required for list input, auto-detected otherwise)

    Returns:
        Tuple of:
            - indptr: CSR row pointers (np.ndarray, dtype=int64)
            - indices: CSR column indices (np.ndarray, dtype=int64)
            - n_rows: Number of transactions
            - n_cols: Number of items
            - item_map: Mapping from original item IDs to column indices, or None

    Raises:
        TypeError: If input type is not supported
        ValueError: If required parameters are missing

    Example:
        >>> # From Polars
        >>> indptr, indices, n_rows, n_cols, _ = to_csr(polars_df, item_col="items")
        >>>
        >>> # From raw lists (requires n_items)
        >>> indptr, indices, n_rows, n_cols, _ = to_csr([[0,1], [1,2]], n_items=3)
        >>>
        >>> # From scipy CSR
        >>> indptr, indices, n_rows, n_cols, _ = to_csr(scipy_csr)
    """
    # Check for tuple first (raw CSR arrays)
    if isinstance(data, tuple) and len(data) >= 4:
        return _from_raw_csr(data)

    # Check for scipy sparse matrix
    if _is_scipy_csr(data):
        return _from_scipy_csr(data)

    # Check for Polars DataFrame
    if _is_polars_dataframe(data):
        return _from_polars(data, item_col)

    # Check for Pandas DataFrame
    if _is_pandas_dataframe(data):
        return _from_pandas(data, item_col)

    # Check for list of lists
    if isinstance(data, (list, tuple)) and (len(data) == 0 or isinstance(data[0], (list, tuple))):
        if n_items is None:
            raise ValueError(
                "n_items is required for list input. "
                "Pass the total number of unique items."
            )
        return _from_list(data, n_items)

    raise TypeError(
        f"Unsupported input type: {type(data).__name__}. "
        f"Supported types: polars.DataFrame, pandas.DataFrame, "
        f"list[list[int]], scipy.sparse.csr_matrix, tuple(indptr, indices, n_rows, n_cols)"
    )


def _is_polars_dataframe(obj: Any) -> bool:
    """Check if object is a Polars DataFrame without importing Polars."""
    return type(obj).__module__.startswith("polars") and type(obj).__name__ in ("DataFrame", "LazyFrame")


def _is_pandas_dataframe(obj: Any) -> bool:
    """Check if object is a Pandas DataFrame without importing Pandas."""
    return type(obj).__module__.startswith("pandas") and type(obj).__name__ == "DataFrame"


def _is_scipy_csr(obj: Any) -> bool:
    """Check if object is a scipy CSR matrix without importing scipy."""
    return (
        type(obj).__module__.startswith("scipy.sparse")
        and "csr" in type(obj).__name__.lower()
    )


def _from_raw_csr(data: tuple) -> CSRResult:
    """Handle raw CSR tuple input."""
    indptr, indices = data[0], data[1]
    n_rows, n_cols = data[2], data[3]

    # Ensure correct dtypes
    indptr = np.asarray(indptr, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)

    return indptr, indices, int(n_rows), int(n_cols), None


def _from_scipy_csr(csr: Any) -> CSRResult:
    """Convert scipy CSR matrix to arrays."""
    n_rows, n_cols = csr.shape
    indptr = np.asarray(csr.indptr, dtype=np.int64)
    indices = np.asarray(csr.indices, dtype=np.int64)

    return indptr, indices, n_rows, n_cols, None


def _from_list(transactions: list | tuple, n_items: int) -> CSRResult:
    """Convert list of transactions to CSR using Rust."""
    try:
        from et_miner_rust import build_csr_from_transactions
    except ImportError:
        # Fallback to pure Python if Rust extension not available
        return _from_list_python(transactions, n_items)

    # Convert to list of list[int64] for Rust
    transactions_i64 = [
        [int(item) for item in tx] for tx in transactions
    ]

    indptr, indices = build_csr_from_transactions(transactions_i64, n_items)
    n_rows = len(transactions)

    return indptr, indices, n_rows, n_items, None


def _from_list_python(transactions: list | tuple, n_items: int) -> CSRResult:
    """Pure Python fallback for list→CSR conversion."""
    n_rows = len(transactions)

    # Build CSR directly
    indptr = [0]
    indices = []

    for tx in transactions:
        # Sort and deduplicate items
        items = sorted(set(int(item) for item in tx if 0 <= item < n_items))
        indices.extend(items)
        indptr.append(len(indices))

    indptr_arr = np.array(indptr, dtype=np.int64)
    indices_arr = np.array(indices, dtype=np.int64)

    return indptr_arr, indices_arr, n_rows, n_items, None


def _from_polars(df: Any, item_col: str) -> CSRResult:
    """Convert Polars DataFrame with items column to CSR."""
    import polars as pl

    # Handle LazyFrame
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    if item_col not in df.columns:
        raise ValueError(f"Column '{item_col}' not found in DataFrame. Available: {df.columns}")

    # Get unique items and create mapping
    items_series = df.get_column(item_col)

    # Explode to get all unique items
    all_items = items_series.explode().drop_nulls().unique().sort()
    item_list = all_items.to_list()
    item_map = {item: idx for idx, item in enumerate(item_list)}
    n_items = len(item_map)

    # Convert transactions to integer indices
    transactions = []
    for row in items_series:
        if row is None:
            transactions.append([])
        else:
            tx = [item_map[item] for item in row if item in item_map]
            transactions.append(tx)

    # Use Rust for CSR conversion
    return _from_list(transactions, n_items)[:-1] + (item_map,)


def _from_pandas(df: Any, item_col: str) -> CSRResult:
    """Convert Pandas DataFrame with items column to CSR."""
    if item_col not in df.columns:
        raise ValueError(f"Column '{item_col}' not found in DataFrame. Available: {list(df.columns)}")

    items_series = df[item_col]

    # Get unique items and create mapping
    all_items = set()
    for row in items_series:
        if row is not None and hasattr(row, "__iter__"):
            all_items.update(row)

    item_list = sorted(all_items)
    item_map = {item: idx for idx, item in enumerate(item_list)}
    n_items = len(item_map)

    # Convert transactions to integer indices
    transactions = []
    for row in items_series:
        if row is None or not hasattr(row, "__iter__"):
            transactions.append([])
        else:
            tx = [item_map[item] for item in row if item in item_map]
            transactions.append(tx)

    # Use Rust for CSR conversion
    return _from_list(transactions, n_items)[:-1] + (item_map,)


def format_output(
    itemsets: list[list[int]],
    counts: list[int],
    n_transactions: int,
    item_map: dict[Any, int] | None,
    return_format: str = "auto",
    input_type: type | None = None,
) -> Any:
    """Format Apriori results based on requested output format.

    Args:
        itemsets: List of itemsets (as column indices)
        counts: Support counts for each itemset
        n_transactions: Total number of transactions
        item_map: Mapping from original items to indices (for reverse lookup)
        return_format: Output format - "auto", "polars", "pandas", "list", "dict"
        input_type: Original input type (for "auto" format detection)

    Returns:
        Results in requested format
    """
    # Create reverse mapping if available
    idx_to_item = {v: k for k, v in item_map.items()} if item_map else None

    # Convert itemsets back to original item IDs if possible
    if idx_to_item:
        itemsets_original = [
            [idx_to_item[idx] for idx in itemset]
            for itemset in itemsets
        ]
    else:
        itemsets_original = itemsets

    # Calculate support
    supports = [count / n_transactions for count in counts]

    # Auto-detect output format based on input type
    if return_format == "auto":
        if input_type and hasattr(input_type, '__module__'):
            if input_type.__module__.startswith("polars"):
                return_format = "polars"
            elif input_type.__module__.startswith("pandas"):
                return_format = "pandas"
            else:
                return_format = "dict"
        else:
            return_format = "dict"

    if return_format == "dict":
        return {
            "itemsets": itemsets_original,
            "counts": counts,
            "support": supports,
        }

    if return_format == "list":
        return list(zip(itemsets_original, counts, supports))

    if return_format == "polars":
        import polars as pl
        return pl.DataFrame({
            "itemset": itemsets_original,
            "count": counts,
            "support": supports,
        })

    if return_format == "pandas":
        import pandas as pd
        return pd.DataFrame({
            "itemset": itemsets_original,
            "count": counts,
            "support": supports,
        })

    raise ValueError(f"Unknown return_format: {return_format}")
