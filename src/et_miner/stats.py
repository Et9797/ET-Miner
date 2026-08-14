"""Shared statistical utilities."""

from __future__ import annotations

import numpy as np
from loguru import logger


def bh_fdr_adjust(pvalues: np.ndarray, alpha: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction.

    Args:
        pvalues: Raw p-values array.
        alpha: Significance threshold (default 0.01).

    Returns:
        (adjusted_pvalues, significant_mask) — both same shape as input.
    """
    n = len(pvalues)
    if n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=bool)

    # NaN p-values (from Fisher edge cases) corrupt BH via minimum.accumulate propagation
    pvalues = np.where(np.isnan(pvalues), 1.0, pvalues)

    sorted_idx = np.argsort(pvalues)
    ranks = np.arange(1, n + 1)
    adjusted = np.empty(n, dtype=np.float64)
    adjusted[sorted_idx] = np.minimum(1.0, pvalues[sorted_idx] * n / ranks)

    # Enforce monotonicity: adjusted[i] <= adjusted[i+1] when sorted by p-value
    adjusted_sorted = adjusted[sorted_idx]
    adjusted_sorted[:] = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted[sorted_idx] = adjusted_sorted

    return adjusted, adjusted < alpha


_fisher_kernel_cache = None


def _get_fisher_kernel():
    """Lazy singleton for the CUDA Fisher kernel — avoids recompilation on every call."""
    global _fisher_kernel_cache
    if _fisher_kernel_cache is not None:
        return _fisher_kernel_cache
    import cupy as cp
    _fisher_kernel_cache = cp.RawKernel(r"""
    extern "C" __global__
    void fisher_pvalue(const double* w, const double* l,
                       double* out, long long N,
                       long long n_win, long long M_total) {
        long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= N) return;

        double a  = w[i];
        double K  = w[i] + l[i];          // success states in population
        double Md = (double)M_total;
        double nd = (double)n_win;

        double mean = nd * K / Md;
        double var  = nd * K * (Md - K) * (Md - nd) / (Md * Md * (Md - 1.0));
        if (var <= 0.0) {
            out[i] = (a > mean) ? 0.0 : 1.0;
            return;
        }
        double z = (a - 0.5 - mean) / sqrt(var);   // continuity correction
        out[i] = 0.5 * erfc(z / sqrt(2.0));
    }
    """, "fisher_pvalue")
    return _fisher_kernel_cache


def _scipy_fisher_fallback(w_sup, l_sup, M, n):
    """Scipy hypergeom fallback for when CuPy/CUDA is unavailable."""
    from scipy.stats import hypergeom
    K = (w_sup + l_sup).astype(int)
    return hypergeom.sf(w_sup.astype(int) - 1, M, K, n)


def fisher_gpu(
    w_sup: np.ndarray,
    l_sup: np.ndarray,
    n_winners: int,
    n_losers: int,
) -> np.ndarray:
    """GPU-accelerated Fisher's Exact Test via hypergeometric normal approximation.

    Falls back to scipy.stats.hypergeom vectorized if CuPy unavailable.

    Args:
        w_sup: Winner support counts per rule.
        l_sup: Loser support counts per rule.
        n_winners: Total winner transactions.
        n_losers: Total loser transactions.

    Returns:
        Array of one-sided p-values (greater alternative).
    """
    M = n_winners + n_losers  # population size
    n = n_winners              # number of draws

    try:
        import cupy as cp
        kernel = _get_fisher_kernel()

        total = len(w_sup)
        result = np.empty(total, dtype=np.float64)
        chunk = 50_000_000  # 50M elements per chunk for VRAM

        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            d_w = cp.asarray(w_sup[start:end].astype(np.float64))
            d_l = cp.asarray(l_sup[start:end].astype(np.float64))
            d_out = cp.empty(end - start, dtype=cp.float64)

            block = 256
            grid = (int((end - start + block - 1) // block),)
            kernel(grid, (block,), (d_w, d_l, d_out, np.int64(end - start), np.int64(n), np.int64(M)))

            result[start:end] = cp.asnumpy(d_out)
            del d_w, d_l, d_out
            cp.get_default_memory_pool().free_all_blocks()

        return result

    except ImportError:
        logger.warning("CuPy not available, falling back to scipy hypergeom (slow)")
        return _scipy_fisher_fallback(w_sup, l_sup, M, n)
    except Exception as e:
        logger.error(f"CUDA Fisher kernel failed: {e} — falling back to scipy (slow)")
        return _scipy_fisher_fallback(w_sup, l_sup, M, n)
