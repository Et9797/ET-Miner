"""GPU bitvec support counting (CuPy).

Packs each item column into a u64 bitvector on the GPU, then counts itemset
support with fused AND + hardware popcount. Import-safe without CuPy; the
functions raise RuntimeError at call time when CuPy is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from loguru import logger

from et_miner.backends import CUPY_INSTALLED, RUST_INSTALLED, get_rust_ext
from et_miner.core.matrix import _polars_to_sparse_csr

if TYPE_CHECKING:
    import cupy as cp
    from scipy.sparse import csr_matrix as CSRMatrix


# =============================================================================
# GPU Bitvec Acceleration (CuPy)
# =============================================================================


def _build_gpu_bitvec_matrix(
    csr: "CSRMatrix",
    use_cuda_kernel: bool = True,
) -> "cp.ndarray":
    """Build column bitvecs on the current CUDA device.

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
            from et_miner.gpu.csr_bitvec import build_bitvecs_gpu_from_scipy

            return build_bitvecs_gpu_from_scipy(csr, device_id=cp.cuda.Device().id)
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


def count_support_gpu_bitvec(
    matrix: pl.DataFrame,
    itemsets: list[tuple[str, ...]],
    show_progress: bool = False,
) -> dict[tuple[str, ...], int]:
    """Count itemset support on the current CUDA device.

    Builds the bitvecs with the CUDA CSR→bitvec kernel (the Rust builder only
    when that kernel fails), then counts every itemset in one batched
    AND + popcount launch (count_itemsets_cuda).

    Args:
        matrix: Boolean DataFrame from build_boolean_matrix().
        itemsets: List of itemsets as tuples of column names.
        show_progress: Ignored (GPU is fast enough that progress isn't needed).

    Returns:
        Dictionary mapping itemsets to support counts.

    Raises:
        RuntimeError: If CuPy is not available, or the CUDA bitvec build fails
            and the Rust extension is not available.

    Example:
        >>> counts = count_support_gpu_bitvec(matrix, [("i_0", "i_1"), ("i_1", "i_2")])
    """
    if not CUPY_INSTALLED:
        raise RuntimeError("CuPy not available - install with: pip install cupy-cuda12x")
    if not itemsets:
        return {}

    from et_miner.gpu.kernels import count_itemsets_cuda

    csr, col_to_idx = _polars_to_sparse_csr(matrix)
    n_rows = csr.shape[0]
    logger.debug(f"[GPU BITVEC] {len(itemsets)} itemsets, {n_rows:,} transactions, {csr.shape[1]} items")
    bitvecs_gpu = _build_gpu_bitvec_matrix(csr)

    # The empty itemset is in every transaction; the kernel would count it as 0.
    counted = [s for s in itemsets if s]
    counts = count_itemsets_cuda(
        bitvecs_gpu, [np.array([col_to_idx[col] for col in s], dtype=np.int32) for s in counted]
    )
    result = {s: int(c) for s, c in zip(counted, counts)}
    result.update({s: n_rows for s in itemsets if not s})
    return result
