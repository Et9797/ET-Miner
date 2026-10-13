"""Scipy sparse support counting, strategy selection, and MKL setup.

Holds the sparse CSR counting paths (pairs by an MKL/scipy matmul, every
other itemset on the CPU array miner's counter), the polars-vs-sparse strategy
chooser, and MKL setup.

Importing this module points sparse_dot_mkl at the pip-installed MKL
(``MKL_RT``) and extends LD_LIBRARY_PATH for child processes, provided it is
imported before sparse_dot_mkl is. It deliberately does **not** set MKL's thread count any
more: this module is imported lazily from inside count_support_batched, so
doing that at import fired mid-run and silently overrode a host application's
own MKL configuration. Threads are configured only around the parallel
sections that need it, and restored afterwards.
"""

from __future__ import annotations

import itertools
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
from loguru import logger

from et_miner.core.cpu_miner import _workers, count_itemsets
from et_miner.core.matrix import _polars_to_sparse_csr
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix as CSRMatrix


# =============================================================================
# MKL Library Path Setup
# =============================================================================


#: Directories that may hold a pip- or conda-installed libmkl_rt.
def _mkl_search_dirs() -> list[str]:
    import site

    dirs = [os.path.join(sys.prefix, "lib")]
    try:
        dirs += [os.path.join(p, "..", "..") for p in site.getsitepackages()]
    except AttributeError:  # virtualenv without getsitepackages
        pass
    seen, out = set(), []
    for d in dirs:
        real = os.path.normpath(d)
        if real not in seen and os.path.isdir(real):
            seen.add(real)
            out.append(real)
    return out


def _setup_mkl_library_path() -> None:
    """Point sparse_dot_mkl at the pip-installed MKL, and put its directory on
    LD_LIBRARY_PATH for child processes.

    pip installs MKL to ``{venv}/lib/``, which is not on the standard library
    search path, so sparse_dot_mkl cannot dlopen it without help. Two variables
    are set, and this function used to set only the second:

    * ``MKL_RT`` -- the absolute path of the highest ``libmkl_rt.so.N`` found.
      sparse_dot_mkl tries this variable first and dlopens the path directly,
      and MKL's runtime then resolves its threading and kernel layers from the
      same directory. Set as a default only: an ``MKL_RT`` already in the
      environment is a deliberate choice and is kept.
    * ``LD_LIBRARY_PATH`` -- extended with the directory. This reaches CHILD
      processes only. The dynamic loader reads the variable once, at process
      start, so an edit from inside a running interpreter changes nothing for
      that interpreter's own dlopen. Measured on the dev box (2026-09-11):
      with this edit alone, the MKL that loaded was
      ``/opt/conda/lib/libmkl_rt.so.2`` -- the unpinned system copy -- whether
      or not LD_LIBRARY_PATH was preset; on a CI runner with no system MKL,
      sparse_dot_mkl did not import at all (``mkl_rt not found``). With
      ``MKL_RT`` and no LD_LIBRARY_PATH, the venv's ``libmkl_rt.so.3`` and its
      layers load and the matmul agrees with scipy exactly.
      ``tests/test_sparse_mkl.py::TestThePinnedMklIsTheOneLoaded`` holds both
      halves.

    This used to test exactly one filename at exactly one location --
    ``{sys.prefix}/lib/libmkl_rt.so.2`` -- and return silently when it missed.
    It misses on both halves in a common layout: a venv can ship
    ``libmkl_rt.so.3`` while the loadable MKL is a ``.so.2`` elsewhere entirely.
    Because the miss was silent, nothing in the output distinguished "the path
    was already fine" from "the probe found nothing", and the numerics then ran
    against whichever unpinned system MKL happened to load -- which matters,
    because _sparse_matmul's accuracy depends on it.

    Now: glob every ``libmkl_rt.so*`` under each candidate directory, and log
    the outcome either way.
    """
    import glob
    import re

    def _soname_version(path: str) -> int:
        # `libmkl_rt.so.3` ranks above `libmkl_rt.so.2`; the unversioned name,
        # when present, is a symlink to one of them and ranks below both.
        m = re.search(r"\.so\.(\d+)$", path)
        return int(m.group(1)) if m else -1

    for lib_dir in _mkl_search_dirs():
        matches = sorted(glob.glob(os.path.join(lib_dir, "libmkl_rt.so*")))
        if not matches:
            continue

        chosen = max(matches, key=_soname_version)
        if "MKL_RT" in os.environ:
            logger.debug(f"MKL: MKL_RT preset to {os.environ['MKL_RT']}; leaving it, not {chosen}")
        else:
            os.environ["MKL_RT"] = chosen
            logger.debug(f"MKL: MKL_RT set to {chosen}")

        current = os.environ.get("LD_LIBRARY_PATH", "")
        if lib_dir in current.split(os.pathsep):
            logger.debug(f"MKL: {lib_dir} already on LD_LIBRARY_PATH ({os.path.basename(matches[0])})")
            return

        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}{os.pathsep}{current}" if current else lib_dir
        logger.debug(f"MKL: added {lib_dir} to LD_LIBRARY_PATH for {[os.path.basename(m) for m in matches]}")
        return

    # loguru formats with str.format, not %-style: the old "%s" printed literally
    # and the search-dir list never appeared in the diagnostic.
    logger.debug(
        "MKL: no libmkl_rt.so* found under {}; sparse_dot_mkl will import only if "
        "MKL is already on the loader path, and _sparse_matmul falls back to scipy if not",
        _mkl_search_dirs(),
    )


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


#: The host's MKL thread count, captured the first time ET-Miner changes it.
#: None means "we have not touched it yet".
_saved_mkl_threads: int | None = None


def _current_mkl_threads() -> int | None:
    try:
        from sparse_dot_mkl import mkl_get_max_threads

        return mkl_get_max_threads()
    except ImportError:
        return None


def _configure_mkl_for_parallel(n_workers: int) -> int:
    """Reduce MKL threads per worker to avoid oversubscription under ThreadPoolExecutor.

    Records the host's setting on first use so _restore_mkl_threads can put back
    the value we displaced rather than re-deriving one from the environment.
    """
    global _saved_mkl_threads
    if _saved_mkl_threads is None:
        _saved_mkl_threads = _current_mkl_threads()

    cpu_count = os.cpu_count() or 1
    mkl_threads_per_worker = max(1, cpu_count // n_workers)
    return _configure_mkl_threads(mkl_threads_per_worker)


def _restore_mkl_threads() -> int:
    """Put back the thread count ET-Miner displaced.

    This used to call _configure_mkl_threads(None), which re-derives a value
    from MKL_NUM_THREADS / OMP_NUM_THREADS / cpu_count -- so a host application
    that had called mkl_set_num_threads(2) programmatically got cpu_count back,
    and the `finally` blocks that call this did not actually restore anything.
    """
    if _saved_mkl_threads is None:
        return _current_mkl_threads() or 1
    return _configure_mkl_threads(_saved_mkl_threads)


# NOTE: MKL's thread count is deliberately NOT configured at import.
#
# This module is imported lazily from inside count_support_batched
# (core/matrix.py), so a module-scope mkl_set_num_threads() fired mid-run, at
# first count, and silently overrode a host application's own MKL
# configuration -- oversubscribing every other MKL consumer in the process
# (measured: a host setting of 2 became 24 on import). Threads are now
# configured only around the parallel sections that need it, and restored
# afterwards.


def _sparse_matmul(A: csr_matrix, B: csr_matrix) -> csr_matrix:
    """A @ B via MKL if available, else scipy.

    MKL requires a float dtype, so integer inputs are cast — to **float64**, not
    float32. This product is a support COUNT, and float32 has a 24-bit
    significand: once an accumulator reaches 2**24 = 16,777,216, ``acc + 1``
    rounds back to ``acc`` and the sum sticks there. Measured on 20M rows with
    two fully populated columns, the true count 20,000,000 came back as
    16,777,216 — a 16.1% **under**-count, and since the level filter is
    ``count >= min_count`` that silently drops genuinely frequent pairs and the
    loss cascades into every higher K. Between 2**24 and 2**25 the representable
    spacing is 2, so counts in that range are wrong by ±1 even without
    saturating.

    float64 is exact for every integer below 2**53, which is far above the
    ``n_transactions < 2**31`` this engine guards. The cost is 2x the value-array
    bytes inside the matmul; the alternative — skipping MKL for integer dtypes
    and letting scipy evaluate ``A @ B`` in int32 — is equally exact but gives up
    the MKL path this module exists for.

    The scipy fallback below keeps int32 and was always exact, so before this
    change the two branches of one function disagreed at scale depending purely
    on whether an optional import resolved.
    """
    try:
        from sparse_dot_mkl import dot_product_mkl

        if A.dtype not in (np.float32, np.float64):
            A = A.astype(np.float64)
        if B.dtype not in (np.float32, np.float64):
            B = B.astype(np.float64)
        return dot_product_mkl(A, B)
    except ImportError:
        return A @ B


def _count_support_sparse_non_pairs(
    csr: "CSRMatrix",
    col_to_idx: dict[str, int],
    itemsets: list[tuple[str, ...]],
    n_jobs: int = 1,
) -> dict[tuple[str, ...], int]:
    """Counts of the itemsets that are not pairs (K=0, K=1, K>=3) on the CPU array miner's counter.

    Each itemset becomes its distinct column ids, ascending, so a repeated item
    counts once. The CSR is narrowed to the columns the itemsets name, as SON's
    pass 2 maps its chunks onto the candidate items, so the bitvectors scale
    with those columns rather than the matrix width. A stored entry counts only
    when it is nonzero, as in the pair product. Per length the itemsets are
    deduplicated and lexsorted (``np.unique``) and counted by
    ``cpu_miner.count_itemsets``: a bincount at K=1, column bitvectors within
    ``BITVEC_BUDGET_BYTES`` or a projection over prefix tidsets above it.
    The empty itemset counts every row. With ``n_jobs != 1`` the bitvector
    build and the heavy prefix groups run on a thread pool.
    """
    if not itemsets:
        return {}
    ids = [sorted({col_to_idx[c] for c in it}) for it in itemsets]
    by_k: dict[int, list[int]] = {}
    for i, row in enumerate(ids):
        by_k.setdefault(len(row), []).append(i)

    results: dict[tuple[str, ...], int] = {itemsets[i]: csr.shape[0] for i in by_k.pop(0, [])}
    if not by_k:
        return results
    used = np.unique(np.fromiter(itertools.chain.from_iterable(ids), dtype=np.int64))
    m = csr[:, used] if len(used) < csr.shape[1] else csr
    if not m.has_canonical_format or not m.data.all():
        m = m.copy() if m is csr else m
        m.sum_duplicates()
        m.eliminate_zeros()

    cands: dict[int, np.ndarray] = {}
    inverse: dict[int, np.ndarray] = {}
    for k, members in by_k.items():
        local = np.searchsorted(used, np.array([ids[i] for i in members], dtype=np.int64)).astype(np.int32)
        cands[k], inv = np.unique(local, axis=0, return_inverse=True)
        inverse[k] = inv.ravel()
    workers = _workers(n_jobs)
    pool = ThreadPoolExecutor(workers) if workers > 1 else None
    try:
        counts = count_itemsets(m.indptr, m.indices, len(used), cands, pool, workers)
    finally:
        if pool is not None:
            pool.shutdown()
    for k, members in by_k.items():
        for i, c in zip(members, counts[k][inverse[k]].tolist()):
            results[itemsets[i]] = c
    return results


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

    n_workers = _workers(n_jobs)
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
    For the other itemsets (k=1, k>2): the CPU array miner's counter
        (``_count_support_sparse_non_pairs``), on a thread pool when n_jobs != 1.

    Args:
        matrix: Boolean DataFrame from build_boolean_matrix().
        itemsets: List of itemsets as tuples of column names.
        show_progress: Accepted for the caller's signature; the sparse path
            shows no progress bar.
        n_jobs: Number of parallel workers.
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

    # Pairs go to the M.T @ M product; every other length to the array miner's counter
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
        use_parallel = _workers(n_jobs) > 1 and n_pairs > 100 and n_transactions > 10_000

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

    if other_itemsets:
        results.update(_count_support_sparse_non_pairs(csr, col_to_idx, other_itemsets, n_jobs))

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
