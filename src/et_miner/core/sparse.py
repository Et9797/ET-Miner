"""Scipy/Rust sparse support counting, strategy selection, and MKL setup.

Holds the sparse CSR counting paths (scipy matmul, threaded scipy, and the
Rust SIMD/sparse fast paths), the polars-vs-sparse strategy chooser, and the
MKL library-path/thread configuration that runs at import time (importing
this module configures MKL, exactly as the pre-split matrix module did).
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
from loguru import logger

from et_miner._compat import HAS_TQDM, tqdm
from et_miner.backends import RUST_INSTALLED, get_rust_ext
from et_miner.core.matrix import _polars_to_sparse_csr
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix as CSRMatrix


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
        "k=2 PARALLEL: {} pairs, {} chunks, {} workers, {} MKL threads/worker",
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
                "k=2 strategy: PARALLEL (pairs={}, txns={}, n_jobs={})",
                n_pairs,
                n_transactions,
                n_jobs,
            )
            k2_results = _count_support_sparse_k2_parallel(csr, col_to_idx, k2_itemsets, n_jobs)
        else:
            logger.debug(
                "k=2 strategy: BATCH (pairs={}, txns={})",
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
            "[auto-detect] k2 candidate explosion: {} pairs from {} items -> sparse",
            k2_candidates,
            n_items,
        )
        return "sparse"
    if n_items > 500 and density < 0.10:
        logger.debug(
            "[auto-detect] n_items={}, density={:.4f} -> sparse",
            n_items,
            density,
        )
        return "sparse"
    if estimated_dense_gb > 1:
        logger.debug(
            "[auto-detect] estimated_dense_gb={:.2f} -> sparse",
            estimated_dense_gb,
        )
        return "sparse"

    logger.debug(
        "[auto-detect] n_items={}, density={:.4f}, k2={}, dense_gb={:.2f} -> polars",
        n_items,
        density,
        k2_candidates,
        estimated_dense_gb,
    )
    return "polars"
