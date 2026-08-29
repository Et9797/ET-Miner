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

import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
from loguru import logger

from et_miner._compat import HAS_TQDM, tqdm
from et_miner.backends import CUPY_INSTALLED, RUST_INSTALLED, get_rust_ext

# Scipy for sparse matrix operations
from scipy.sparse import coo_matrix, csr_matrix


if TYPE_CHECKING:
    import cupy as cp
    from scipy.sparse import csr_matrix as CSRMatrix


# =============================================================================
# Support threshold & result helpers (shared by the direct and SON paths)
# =============================================================================


def _min_count(min_support: float, n_transactions: int) -> int:
    """Calculate minimum count threshold from support."""
    return math.ceil(min_support * n_transactions)


def _empty_result() -> pl.DataFrame:
    """Return empty result DataFrame with correct schema."""
    return pl.DataFrame(schema={"itemset": pl.List(pl.Int64), "support": pl.Float64})


def _build_result_df(results: list[tuple[list, float]]) -> pl.DataFrame:
    """Build result DataFrame from collected itemsets.

    Preserves the original item types (int, str, etc.) from the input data.
    """
    if not results:
        return _empty_result()
    # Don't force cast to Int64 - let Polars infer the type from the actual items
    return pl.DataFrame(
        {
            "itemset": [r[0] for r in results],
            "support": [float(r[1]) for r in results],
        }
    )


# =============================================================================
# MKL Library Path Setup
# =============================================================================


def _setup_mkl_library_path() -> None:
    """Auto-discover and add MKL library path to LD_LIBRARY_PATH.

    pip installs MKL libraries to {venv}/lib/ which is not in the standard
    library search path. This function extends LD_LIBRARY_PATH at module
    load time so sparse_dot_mkl can find libmkl_rt.so.2.

    Does nothing if MKL is not installed or path already set.
    """
    venv_lib = os.path.join(sys.prefix, "lib")
    mkl_lib = os.path.join(venv_lib, "libmkl_rt.so.2")

    if not os.path.exists(mkl_lib):
        return

    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if venv_lib in current_ld_path:
        return  # Already set

    if current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{venv_lib}:{current_ld_path}"
    else:
        os.environ["LD_LIBRARY_PATH"] = venv_lib


# Run at module load time
_setup_mkl_library_path()


# =============================================================================
# MKL Thread Configuration
# =============================================================================


def _configure_mkl_threads(n_threads: int | None = None) -> int:
    """Set MKL thread count. n_threads=None falls back to MKL_NUM_THREADS/OMP_NUM_THREADS/cpu_count."""
    try:
        from sparse_dot_mkl import mkl_set_num_threads, mkl_get_max_threads

        if n_threads is None:
            n_threads = int(os.environ.get("MKL_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)))

        mkl_set_num_threads(n_threads)
        return mkl_get_max_threads()
    except ImportError:
        return 1


def _configure_mkl_for_parallel(n_workers: int) -> int:
    """Reduce MKL threads per worker to avoid oversubscription when using ThreadPoolExecutor."""
    cpu_count = os.cpu_count() or 1
    mkl_threads_per_worker = max(1, cpu_count // n_workers)
    return _configure_mkl_threads(mkl_threads_per_worker)


def _restore_mkl_threads() -> int:
    return _configure_mkl_threads(None)


def _init_mkl() -> None:
    _configure_mkl_threads()


# Configure MKL at module load
_init_mkl()


# =============================================================================
# Free-Threading (GIL) Detection
# =============================================================================


def _is_gil_disabled() -> bool:
    """True on Python 3.13+ free-threading builds (enables real ThreadPoolExecutor parallelism)."""
    if hasattr(sys, "_is_gil_enabled"):
        return not sys._is_gil_enabled()
    return False


def _sparse_matmul(A: csr_matrix, B: csr_matrix) -> csr_matrix:
    """A @ B via MKL if available, else scipy. MKL requires float dtype — non-float inputs get cast to float32."""
    try:
        from sparse_dot_mkl import dot_product_mkl

        if A.dtype not in (np.float32, np.float64):
            A = A.astype(np.float32)
        if B.dtype not in (np.float32, np.float64):
            B = B.astype(np.float32)
        return dot_product_mkl(A, B)
    except ImportError:
        return A @ B


def _get_adaptive_parallel_config(n_itemsets: int, n_workers: int) -> dict[str, int | bool]:
    """Adaptive workers + chunk_size. GIL mode caps at 8 workers (lock contention); free-threading uses all cores."""
    gil_disabled = _is_gil_disabled()

    if gil_disabled:
        effective_workers = n_workers
        chunk_size = max(50, n_itemsets // (effective_workers * 4))
    else:
        effective_workers = min(n_workers, 8)
        chunk_size = max(100, n_itemsets // effective_workers)

    return {
        "workers": effective_workers,
        "chunk_size": chunk_size,
        "gil_disabled": gil_disabled,
    }


def _get_effective_workers(n_jobs: int) -> int:
    """n_jobs=-1 → all CPUs, else max(1, n_jobs)."""
    if n_jobs == -1:
        return os.cpu_count() or 1
    return max(1, n_jobs)


def _count_single_itemset(
    csr: "CSRMatrix",
    col_to_idx: dict[str, int],
    itemset: tuple[str, ...],
) -> tuple[tuple[str, ...], int]:
    """Count support for a single k>2 itemset.

    This function is designed to be called in parallel - it's a pure function
    with no side effects, operating on read-only shared data (the CSR matrix).

    Args:
        csr: Sparse CSR matrix (read-only, shared across threads).
        col_to_idx: Column name to index mapping (read-only).
        itemset: Single itemset to count.

    Returns:
        Tuple of (itemset, count) for easy dict construction.
    """
    col_indices = [col_to_idx[col] for col in itemset]
    # Row sum == k means all k items are present
    subset = csr[:, col_indices]
    row_sums = np.asarray(subset.sum(axis=1)).ravel()
    return itemset, int((row_sums == len(col_indices)).sum())


# Minimum itemsets to justify parallelization overhead
# Lowered from 1000 to 100 for more aggressive parallelization on multi-core systems
_PARALLEL_THRESHOLD = 100


# Streaming chunk size for wide boolean matrices (Polars 1.37+ width-aware chunking)
# Smaller chunks reduce memory pressure when matrix has many columns (1 per item)
_STREAMING_CHUNK_SIZE = 25_000


def _log_parallel_decision(
    itemset_type: str,
    count: int,
    threshold: tuple[int, int] | int,
    n_workers: int,
    use_parallel: bool,
) -> None:
    """Log the parallel/sequential decision for support counting.

    This is useful for diagnosing whether workloads trigger parallel code paths.
    Enable with: logger.enable("et_miner") or LOGURU_LEVEL=DEBUG

    Args:
        itemset_type: "k=2" or "k>2" for itemset size category.
        count: Number of itemsets being processed.
        threshold: Threshold(s) for parallel decision (tuple for k=2, int for k>2).
        n_workers: Number of worker threads configured.
        use_parallel: Whether parallel execution was chosen.
    """
    strategy = "PARALLEL" if use_parallel else "SEQUENTIAL"
    if isinstance(threshold, tuple):
        threshold_str = f"threshold={threshold[0]}-{threshold[1]}"
    else:
        threshold_str = f"threshold>={threshold}"

    # Report MKL status for sparse matrix operations
    try:
        from sparse_dot_mkl import mkl_get_max_threads

        mkl_threads = mkl_get_max_threads()
        mkl_status = f"MKL available ({mkl_threads} threads)"
    except ImportError:
        mkl_status = "MKL not available (scipy fallback)"

    logger.debug(f"[{itemset_type}] {count} itemsets, {threshold_str}, workers={n_workers} → {strategy}, {mkl_status}")


def _count_support_sparse_k_gt_2(
    csr: "CSRMatrix",
    col_to_idx: dict[str, int],
    itemsets: list[tuple[str, ...]],
    show_progress: bool = False,
    n_jobs: int = 1,
    use_rust: bool | None = None,
) -> dict[tuple[str, ...], int]:
    """Count support for k>2 itemsets using numpy, with optional parallelization.

    Strategy selection (in order of preference):
    1. Rust extension (if available and suitable) - 10-50x faster
    2. Python parallel (ThreadPoolExecutor) - 3-5x faster with free-threading
    3. Python sequential - baseline

    Uses column slicing and row sums: row_sum == k means all k items present.

    When n_jobs > 1 and there are enough itemsets (>100), this function uses
    ThreadPoolExecutor for parallel execution. On Python 3.13+/3.14+ with
    free-threading (no GIL), this achieves true parallelism. On older Python
    versions, parallelism still helps because scipy releases the GIL during
    its C operations.

    Args:
        csr: Sparse CSR matrix (transactions × items).
        col_to_idx: Mapping from column names to matrix indices.
        itemsets: List of k>2 itemsets to count.
        show_progress: If True, display progress bar (requires tqdm).
        n_jobs: Number of parallel workers.
            - -1: Use all available CPUs
            - 1: Sequential execution
            - >1: Use that many workers
        use_rust: Override Rust extension usage.
            - None: Auto-detect (use if available and suitable)
            - True: Force Rust (raises if not available)
            - False: Disable Rust, use Python

    Returns:
        Dictionary mapping itemsets to support counts.
    """
    if not itemsets:
        return {}

    n_items = csr.shape[1]

    # Strategy 1: Try Rust extension (fastest path)
    if use_rust is True:
        return _count_support_sparse_k_gt_2_rust(csr, col_to_idx, itemsets, show_progress)
    elif use_rust is None and _should_use_rust(len(itemsets), n_items):
        logger.debug(f"[k>2] Using Rust extension ({len(itemsets)} itemsets, {n_items} items)")
        return _count_support_sparse_k_gt_2_rust(csr, col_to_idx, itemsets, show_progress)

    # Strategy 2/3: Python parallel or sequential
    n_workers = _get_effective_workers(n_jobs)
    use_parallel = n_workers > 1 and len(itemsets) >= _PARALLEL_THRESHOLD

    # Log the decision for diagnostics (enable with LOGURU_LEVEL=DEBUG)
    _log_parallel_decision("k>2", len(itemsets), _PARALLEL_THRESHOLD, n_workers, use_parallel)

    # Use sequential for small workloads or when n_jobs=1
    if not use_parallel:
        return _count_support_sparse_k_gt_2_sequential(csr, col_to_idx, itemsets, show_progress)

    # Parallel execution
    return _count_support_sparse_k_gt_2_parallel(csr, col_to_idx, itemsets, show_progress, n_workers)


def _count_support_sparse_k_gt_2_sequential(
    csr: "CSRMatrix",
    col_to_idx: dict[str, int],
    itemsets: list[tuple[str, ...]],
    show_progress: bool = False,
) -> dict[tuple[str, ...], int]:
    """Sequential support counting for k>2 itemsets.

    This is the original implementation, used when:
    - n_jobs=1 is explicitly requested
    - The workload is too small to benefit from parallelization
    """
    results: dict[tuple[str, ...], int] = {}

    itemset_iter: list[tuple[str, ...]] | tqdm = itemsets
    if show_progress and HAS_TQDM:
        itemset_iter = tqdm(itemsets, desc="Counting support (k>2)", unit="itemset")

    for itemset in itemset_iter:
        col_indices = [col_to_idx[col] for col in itemset]
        subset = csr[:, col_indices]
        row_sums = np.asarray(subset.sum(axis=1)).ravel()
        results[itemset] = int((row_sums == len(col_indices)).sum())

    return results


def _count_support_sparse_k_gt_2_parallel(
    csr: "CSRMatrix",
    col_to_idx: dict[str, int],
    itemsets: list[tuple[str, ...]],
    show_progress: bool = False,
    n_workers: int = 4,
) -> dict[tuple[str, ...], int]:
    """Parallel support counting for k>2 itemsets using ThreadPoolExecutor.

    Key insight: Each itemset count is completely independent, making this
    embarrassingly parallel. The CSR matrix is read-only and shared safely
    across all threads.

    Performance characteristics:
    - Python 3.13t/3.14t (no GIL): True parallelism, ~N× speedup
    - Python 3.10-3.13 (GIL): Still benefits because scipy releases GIL
      during C operations (sparse matrix slicing, sum)

    Args:
        csr: Sparse CSR matrix (shared read-only across threads).
        col_to_idx: Column mapping (shared read-only).
        itemsets: Itemsets to count.
        show_progress: Display progress bar.
        n_workers: Number of worker threads.

    Returns:
        Dictionary mapping itemsets to support counts.
    """
    results: dict[tuple[str, ...], int] = {}

    # Coordinate MKL threads with ThreadPoolExecutor to avoid oversubscription
    # e.g., 24 cores with 8 workers → 3 MKL threads each (8 × 3 = 24)
    mkl_threads = _configure_mkl_for_parallel(n_workers)
    logger.debug(f"[k>2] Configured MKL: {mkl_threads} threads/worker × {n_workers} workers")

    try:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(_count_single_itemset, csr, col_to_idx, itemset): itemset for itemset in itemsets
            }

            # Process results as they complete
            if show_progress and HAS_TQDM:
                completed = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Counting support (k>2, {n_workers} threads)",
                    unit="itemset",
                )
            else:
                completed = as_completed(futures)

            for future in completed:
                itemset, count = future.result()
                results[itemset] = count
    finally:
        # Restore full MKL threading after parallel section
        _restore_mkl_threads()

    return results


# =============================================================================
# Rust Extension Accelerated Counting
# =============================================================================


def _count_support_sparse_k_gt_2_rust(
    csr: "CSRMatrix",
    col_to_idx: dict[str, int],
    itemsets: list[tuple[str, ...]],
    show_progress: bool = False,
    use_simd: bool = True,
) -> dict[tuple[str, ...], int]:
    """Count support for k>2 itemsets using the Rust extension.

    As of v0.3.0, uses SIMD-accelerated bitset intersection for ~400-800x
    speedup over Python (5-8x faster than the previous Rust sparse method).

    The new SIMD path:
    1. Converts CSR to CSC format (one-time cost, amortized over itemsets)
    2. Builds column bitvecs (1 bit per row = 312KB for 2.5M rows)
    3. Uses SIMD-width bitwise AND for set intersection
    4. Native popcount instruction for counting

    Performance characteristics (SIMD, 2.5M rows, 12 threads):
    - 500 itemsets: 0.19s (vs 0.99s sparse = 5.1x, vs Python = 400x)
    - 5000 itemsets: 1.23s (vs 9.16s sparse = 7.4x, vs Python = 600x)

    Args:
        csr: Sparse CSR matrix (used directly, no conversion!).
        col_to_idx: Column name to index mapping.
        itemsets: List of k>2 itemsets to count.
        show_progress: Ignored (Rust handles internally).
        use_simd: Use SIMD bitvec implementation (default True).

    Returns:
        Dictionary mapping itemsets to support counts.
    """
    if not RUST_INSTALLED:
        raise RuntimeError("Rust extension not available")
    rust = get_rust_ext()

    if not itemsets:
        return {}

    # Convert itemsets to column indices
    itemsets_indices = [[col_to_idx[col] for col in itemset] for itemset in itemsets]

    # Prepare CSR arrays for Rust (zero-copy views!)
    # Rust extension expects i64 for trillion-scale support (>2.1B indices)
    indptr = csr.indptr.astype(np.int64)
    indices = csr.indices.astype(np.int64)
    n_rows = csr.shape[0]
    n_cols = csr.shape[1]

    # Use SIMD bitvec implementation if available and requested
    _use_simd = use_simd and hasattr(rust, "count_itemsets_simd")

    if _use_simd:
        logger.debug(
            f"[k>2 RUST SIMD] {len(itemsets)} itemsets, "
            f"{n_rows:,} transactions, "
            f"{n_cols} items, "
            f"{rust.get_num_threads()} threads"
        )
        counts = rust.count_itemsets_simd(indptr, indices, n_rows, n_cols, itemsets_indices)
    else:
        logger.debug(
            f"[k>2 RUST SPARSE] {len(itemsets)} itemsets, "
            f"{n_rows:,} transactions, "
            f"{n_cols} items, "
            f"{rust.get_num_threads()} threads"
        )
        counts = rust.count_itemsets_sparse(indptr, indices, n_rows, itemsets_indices)

    # Build result dict
    return {itemset: int(count) for itemset, count in zip(itemsets, counts)}


# Minimum itemsets to use Rust (below this, Python overhead ~same as Rust call overhead)
_RUST_MIN_ITEMSETS = 1  # Rust is now ALWAYS faster due to zero conversion overhead


def _should_use_rust(n_itemsets: int, n_items: int) -> bool:
    # v0.2.0+: Rust sparse CSR has zero conversion overhead, always faster when available
    return RUST_INSTALLED


# =============================================================================
# GPU Bitvec Acceleration (CuPy)
# =============================================================================


def _build_gpu_bitvec_matrix(
    csr: "CSRMatrix",
    use_cuda_kernel: bool = True,
) -> "cp.ndarray":
    """Build column bitvecs and transfer to GPU.

    Two paths available:
    1. CUDA kernel (default): Transfer CSR to GPU, build bitvecs on GPU
       - Faster for large matrices (avoids CPU→GPU bitvec transfer)
       - CSR is ~30x smaller than bitvecs
    2. Rust path: Build bitvecs on CPU, transfer to GPU
       - Fallback if CUDA kernel unavailable

    Args:
        csr: scipy CSR matrix (transactions × items).
        use_cuda_kernel: Use CUDA CSR→bitvec kernel (default True).

    Returns:
        CuPy array of shape [n_cols, ceil(n_rows/64)] containing packed bitvecs.

    Raises:
        RuntimeError: If CuPy or Rust extension not available.
    """
    if not CUPY_INSTALLED:
        raise RuntimeError("CuPy not available")
    import cupy as cp

    n_rows = csr.shape[0]
    n_cols = csr.shape[1]

    # Try CUDA kernel path (Phase 2 optimization)
    if use_cuda_kernel:
        try:
            from et_miner.cuda_csr_bitvec import build_bitvecs_gpu_from_scipy

            return build_bitvecs_gpu_from_scipy(csr)
        except ImportError:
            logger.debug("CUDA CSR→bitvec kernel not available, using Rust path")
        except Exception as e:
            logger.warning(f"CUDA kernel failed: {e}, falling back to Rust path")

    # Fallback: Rust bitvec build + transfer
    if not RUST_INSTALLED:
        raise RuntimeError("Rust extension not available")

    # Extract CSR components
    indptr = csr.indptr.astype(np.int64)
    indices = csr.indices.astype(np.int64)

    # Build bitvecs in Rust (fast CSR→CSC→bitvec)
    bitvecs_cpu = get_rust_ext().build_column_bitvecs_u64(indptr, indices, n_rows, n_cols)

    # Transfer to GPU
    return cp.asarray(bitvecs_cpu)


def _count_support_gpu_bitvec(
    bitvecs_gpu: "cp.ndarray",
    itemsets_indices: list[list[int]],
    n_rows: int,
) -> np.ndarray:
    """Count itemset support using GPU bitvec AND + popcount.

    For each itemset, performs bitwise AND across all column bitvecs,
    then counts set bits using popcount. This is extremely fast on GPU
    due to massive parallelism (thousands of itemsets processed simultaneously).

    Args:
        bitvecs_gpu: GPU array [n_cols, n_u64s] of packed column bitvecs.
        itemsets_indices: List of itemsets as column index lists.
        n_rows: Original number of rows (for validation).

    Returns:
        numpy array of support counts (one per itemset).

    Example:
        >>> bitvecs = _build_gpu_bitvec_matrix(csr)
        >>> counts = _count_support_gpu_bitvec(bitvecs, [[0, 1, 2], [1, 2, 3]], 1000000)
    """
    if not CUPY_INSTALLED:
        raise RuntimeError("CuPy not available")
    import cupy as cp

    n_itemsets = len(itemsets_indices)
    if n_itemsets == 0:
        return np.array([], dtype=np.uint32)

    n_u64s = bitvecs_gpu.shape[1]

    # Process itemsets: AND columns together, then hardware popcount
    # Uses _popcount_u64_array() which calls cuda_kernels.get_popcount_kernel()
    # for native __popcll hardware intrinsic (compiles to GPU POPC instruction)
    counts = cp.zeros(n_itemsets, dtype=cp.uint32)

    for i, cols in enumerate(itemsets_indices):
        if len(cols) == 0:
            counts[i] = n_rows
            continue

        # Start with first column
        result = bitvecs_gpu[cols[0]].copy()

        # AND with remaining columns
        for col in cols[1:]:
            result &= bitvecs_gpu[col]

        # Hardware popcount using __popcll intrinsic
        counts[i] = _popcount_u64_array(result)

    return cp.asnumpy(counts)


def _popcount_u64_array(arr: "cp.ndarray") -> int:
    """Count total set bits in a u64 array using GPU hardware intrinsic.

    Uses the native __popcll instruction via CuPy ElementwiseKernel for
    maximum performance. This compiles directly to the GPU's POPC instruction.

    Args:
        arr: CuPy array of u64 values.

    Returns:
        Total count of set bits across all u64s.
    """
    from et_miner.cuda_kernels import get_popcount_kernel

    # Use hardware __popcll intrinsic - much faster than software bit manipulation
    kernel = get_popcount_kernel()
    popcounts = kernel(arr)  # Returns popcount per u64
    return int(popcounts.sum())


def count_support_gpu_bitvec(
    matrix: pl.DataFrame,
    itemsets: list[tuple[str, ...]],
    show_progress: bool = False,
) -> dict[tuple[str, ...], int]:
    """Count itemset support using GPU-accelerated bitvec operations.

    This is the fastest counting method available, using:
    1. Rust for fast CSR→bitvec conversion (5x faster than scipy)
    2. GPU for massively parallel AND + popcount operations

    Expected throughput: ~500K itemsets/sec on H200 GPU
    (vs ~50K/s Rust SIMD CPU, ~10K/s Python)

    Args:
        matrix: Boolean DataFrame from build_boolean_matrix().
        itemsets: List of itemsets as tuples of column names.
        show_progress: Ignored (GPU is fast enough that progress isn't needed).

    Returns:
        Dictionary mapping itemsets to support counts.

    Raises:
        RuntimeError: If CuPy or Rust extension not available.

    Example:
        >>> counts = count_support_gpu_bitvec(matrix, [("i_0", "i_1"), ("i_1", "i_2")])
    """
    if not CUPY_INSTALLED:
        raise RuntimeError("CuPy not available - install with: pip install cupy-cuda12x")
    if not RUST_INSTALLED:
        raise RuntimeError("Rust extension not available")
    if not itemsets:
        return {}

    # Convert to sparse CSR (efficient for GPU transfer)
    csr, col_to_idx = _polars_to_sparse_csr(matrix)
    n_rows = csr.shape[0]
    n_cols = csr.shape[1]

    logger.debug(f"[GPU BITVEC] {len(itemsets)} itemsets, {n_rows:,} transactions, {n_cols} items")

    # Build bitvecs and transfer to GPU
    bitvecs_gpu = _build_gpu_bitvec_matrix(csr)

    # Convert itemsets to column indices
    itemsets_indices = [[col_to_idx[col] for col in itemset] for itemset in itemsets]

    # Count on GPU
    counts = _count_support_gpu_bitvec(bitvecs_gpu, itemsets_indices, n_rows)

    # Build result dict
    return {itemset: int(count) for itemset, count in zip(itemsets, counts)}


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
        return count_support_gpu_bitvec(filtered_matrix, itemsets, show_progress)

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


def _count_support_sparse_k2_batch(
    csr: CSRMatrix,
    col_to_idx: dict[str, int],
    itemsets_k2: list[tuple[str, str]],
) -> dict[tuple[str, str], int]:
    """Count ALL k=2 itemset supports in one sparse matrix multiply.

    Uses M.T @ M where result[i,j] = count of transactions containing both i and j.
    This replaces O(n_pairs) scipy operations with O(1) matrix multiply.

    The key insight is that for a boolean matrix M where M[t,i] = 1 if transaction t
    contains item i, the product M.T @ M gives us:
        (M.T @ M)[i,j] = sum_t(M[t,i] * M[t,j]) = count of transactions with both i and j

    Args:
        csr: CSR boolean matrix (transactions × items).
        col_to_idx: Mapping from column names to indices.
        itemsets_k2: List of 2-itemsets as (col_a, col_b) tuples.

    Returns:
        Dict mapping each itemset to its support count.
    """
    if not itemsets_k2:
        return {}

    # Extract only columns we need (reduces matrix size for multiplication)
    needed_cols = set()
    for a, b in itemsets_k2:
        needed_cols.add(col_to_idx[a])
        needed_cols.add(col_to_idx[b])

    col_list = sorted(needed_cols)
    col_remap = {old: new for new, old in enumerate(col_list)}

    # Extract subset and compute co-occurrence in ONE operation
    # subset: (n_transactions × n_needed_cols)
    # cooccur: (n_needed_cols × n_needed_cols) - stays SPARSE!
    subset = csr[:, col_list]
    cooccur = _sparse_matmul(subset.T, subset)

    # Extract results for requested pairs
    results: dict[tuple[str, str], int] = {}
    for a, b in itemsets_k2:
        i, j = col_remap[col_to_idx[a]], col_remap[col_to_idx[b]]
        results[(a, b)] = int(cooccur[i, j])

    return results


def _count_k2_chunk(
    args: tuple["CSRMatrix", dict[str, int], list[tuple[str, str]]],
) -> dict[tuple[str, str], int]:
    """Worker function: count supports for a chunk of k=2 pairs using M.T@M.

    Each worker processes a subset of pairs, extracting only the columns needed
    for that chunk. This results in smaller intermediate matrices and better
    cache utilization compared to processing all pairs at once.

    Args:
        args: Tuple of (csr_matrix, col_to_idx, chunk_of_pairs).

    Returns:
        Dict mapping each pair in the chunk to its support count.
    """
    csr, col_to_idx, chunk = args

    if not chunk:
        return {}

    # Get columns needed for this chunk only
    needed_cols = set()
    for a, b in chunk:
        needed_cols.add(col_to_idx[a])
        needed_cols.add(col_to_idx[b])

    col_list = sorted(needed_cols)
    col_remap = {old: new for new, old in enumerate(col_list)}

    # Extract subset and compute co-occurrence
    subset = csr[:, col_list]
    cooccur = _sparse_matmul(subset.T, subset)

    results: dict[tuple[str, str], int] = {}
    for a, b in chunk:
        i, j = col_remap[col_to_idx[a]], col_remap[col_to_idx[b]]
        results[(a, b)] = int(cooccur[i, j])

    return results


def _count_support_sparse_k2_parallel(
    csr: "CSRMatrix",
    col_to_idx: dict[str, int],
    itemsets_k2: list[tuple[str, str]],
    n_jobs: int = -1,
) -> dict[tuple[str, str], int]:
    """Count k=2 itemset supports using parallel chunking.

    For large numbers of pairs (>500) with large matrices (>100k transactions),
    parallel chunking outperforms single batch M.T@M by processing smaller
    subsets of columns per worker. This reduces memory pressure and improves
    cache utilization.

    Benchmark results (410k transactions, 1118 items, 624k pairs):
    - Batch M.T@M: ~15s
    - Parallel (12 workers): ~5s (3x faster)

    Args:
        csr: CSR boolean matrix (transactions × items).
        col_to_idx: Mapping from column names to indices.
        itemsets_k2: List of 2-itemsets as (col_a, col_b) tuples.
        n_jobs: Number of parallel workers (-1 for all CPUs).

    Returns:
        Dict mapping each itemset to its support count.
    """
    if not itemsets_k2:
        return {}

    n_workers = os.cpu_count() or 4 if n_jobs == -1 else n_jobs
    n_pairs = len(itemsets_k2)

    # Chunk size: balance between parallelism and per-chunk efficiency
    # At least 500 pairs per chunk to amortize overhead
    chunk_size = max(500, n_pairs // n_workers)
    chunks = [itemsets_k2[i : i + chunk_size] for i in range(0, n_pairs, chunk_size)]

    # Coordinate MKL threads with ThreadPoolExecutor
    mkl_threads = _configure_mkl_for_parallel(n_workers)
    logger.debug(
        "k=2 PARALLEL: %d pairs, %d chunks, %d workers, %d MKL threads/worker",
        n_pairs,
        len(chunks),
        n_workers,
        mkl_threads,
    )

    results: dict[tuple[str, str], int] = {}
    try:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_count_k2_chunk, (csr, col_to_idx, chunk)) for chunk in chunks]
            for future in as_completed(futures):
                results.update(future.result())
    finally:
        _restore_mkl_threads()

    return results


def count_support_sparse(
    matrix: pl.DataFrame,
    itemsets: list[tuple[str, ...]],
    show_progress: bool = False,
    n_jobs: int = 1,
) -> dict[tuple[str, ...], int]:
    """Count support using scipy sparse matrix.

    Optimized for sparse data where Polars streaming would use too much memory.
    Converts the boolean matrix to CSR format once, then counts support for
    all itemsets using efficient sparse operations.

    For k=2 itemsets: Uses batch M.T @ M matrix multiplication (O(1) operation)
    For k>2 itemsets: Checks row sums of subset columns == k (per-itemset),
        with optional parallelization via ThreadPoolExecutor.

    Args:
        matrix: Boolean DataFrame from build_boolean_matrix().
        itemsets: List of itemsets as tuples of column names.
        show_progress: If True, display progress bar (requires tqdm).
        n_jobs: Number of parallel workers for k>2 itemsets.
            - -1: Use all available CPUs
            - 1: Sequential execution
            - >1: Use that many workers
            On Python 3.13t/3.14t (no GIL), this achieves true parallelism.

    Returns:
        Dictionary mapping itemsets to support counts.

    Raises:
        ImportError: If scipy is not installed.
    """
    if not itemsets:
        return {}

    # Convert to sparse once (the expensive part)
    csr, col_to_idx = _polars_to_sparse_csr(matrix)

    # Separate k=2 itemsets (batch-optimized) from k>2 (per-itemset)
    k2_itemsets: list[tuple[str, str]] = []
    other_itemsets: list[tuple[str, ...]] = []

    for itemset in itemsets:
        if len(itemset) == 2:
            k2_itemsets.append(itemset)  # type: ignore[arg-type]
        else:
            other_itemsets.append(itemset)

    results: dict[tuple[str, ...], int] = {}

    # k=2: Choose strategy based on workload size
    # Benchmarks (benchmark_k2_real_data.py) on 410k transactions, 624k pairs:
    # - Batch M.T@M: ~15s
    # - Parallel chunking (12 workers): ~5s (3x faster)
    # Parallel wins when pairs * transactions is large (memory pressure).
    if k2_itemsets:
        n_pairs = len(k2_itemsets)
        n_transactions = csr.shape[0]
        # Lowered thresholds from 500/50k to 100/10k for more aggressive parallelization
        use_parallel = n_jobs != 1 and n_pairs > 100 and n_transactions > 10_000

        if use_parallel:
            logger.debug(
                "k=2 strategy: PARALLEL (pairs=%d, txns=%d, n_jobs=%d)",
                n_pairs,
                n_transactions,
                n_jobs,
            )
            k2_results = _count_support_sparse_k2_parallel(csr, col_to_idx, k2_itemsets, n_jobs)
        else:
            logger.debug(
                "k=2 strategy: BATCH (pairs=%d, txns=%d)",
                n_pairs,
                n_transactions,
            )
            k2_results = _count_support_sparse_k2_batch(csr, col_to_idx, k2_itemsets)
        results.update(k2_results)  # type: ignore[arg-type]

    # k>2: Per-itemset counting (parallelized when n_jobs > 1)
    if other_itemsets:
        k_gt_2_results = _count_support_sparse_k_gt_2(csr, col_to_idx, other_itemsets, show_progress, n_jobs)
        results.update(k_gt_2_results)

    return results


def _estimate_density(
    matrix: pl.DataFrame,
    sample_cols: int = 10,
) -> float:
    """Estimate matrix density by sampling columns.

    Density = proportion of True values in the matrix.
    We sample a few columns to avoid scanning the entire matrix.

    Args:
        matrix: Boolean DataFrame.
        sample_cols: Number of columns to sample (default 10).

    Returns:
        Estimated density as float between 0 and 1.
    """
    if matrix.height == 0 or len(matrix.columns) == 0:
        return 0.0

    # Sample columns with uniform spread (not first-N, which biases toward
    # low item IDs that may be high-frequency in discovery-ordered datasets)
    n_cols = len(matrix.columns)
    cols_to_check = min(sample_cols, n_cols)
    step = max(1, n_cols // cols_to_check)
    sample_columns = [matrix.columns[i] for i in range(0, n_cols, step)][:cols_to_check]

    total_true = matrix.select([pl.col(c).sum() for c in sample_columns]).row(0)
    return sum(total_true) / (cols_to_check * matrix.height)


def _choose_counting_strategy(
    n_items: int,
    n_transactions: int,
    density: float,
) -> Literal["polars", "sparse"]:
    """Automatically choose optimal counting strategy based on workload.

    Uses sparse when ANY of these conditions are met:
    1. k=2 candidate pairs > 100K (n*(n-1)/2 > 100K, i.e. n > ~448)
    2. Many items (>500) AND low density (<10%)
    3. Large estimated dense size (>1GB)

    Args:
        n_items: Number of frequent items (columns).
        n_transactions: Number of transactions (rows).
        density: Estimated proportion of True values.

    Returns:
        "polars" for standard Polars processing, "sparse" for scipy CSR.
    """
    # Estimate dense matrix size (Polars bit-packed = 1 bit per bool)
    # But for support counting we need to access individual bools,
    # so we estimate the effective working memory
    estimated_dense_bytes = (n_items * n_transactions) / 8  # bit-packed
    estimated_dense_gb = estimated_dense_bytes / (1024**3)

    # Sparse is better when: many items + low density + would be large
    #
    # UPDATED Jan 2026: More aggressive sparse mode activation based on benchmarks
    # showing sparse_true is faster at 2.5M transactions. Previous thresholds were
    # too conservative (>2000 items, <5% density, >4GB).
    #
    # New thresholds (lowered significantly):
    # - >500 items: sparse indexing overhead becomes worthwhile at lower item counts
    # - <10% density: sparse format beneficial at higher densities than before
    # - >1GB: memory savings justify conversion at smaller sizes
    k2_candidates = n_items * (n_items - 1) // 2
    if k2_candidates > 100_000:
        logger.debug(
            "[auto-detect] k2 candidate explosion: %d pairs from %d items -> sparse",
            k2_candidates,
            n_items,
        )
        return "sparse"
    if n_items > 500 and density < 0.10:
        logger.debug(
            "[auto-detect] n_items=%d, density=%.4f -> sparse",
            n_items,
            density,
        )
        return "sparse"
    if estimated_dense_gb > 1:
        logger.debug(
            "[auto-detect] estimated_dense_gb=%.2f -> sparse",
            estimated_dense_gb,
        )
        return "sparse"

    logger.debug(
        "[auto-detect] n_items=%d, density=%.4f, k2=%d, dense_gb=%.2f -> polars",
        n_items,
        density,
        k2_candidates,
        estimated_dense_gb,
    )
    return "polars"
