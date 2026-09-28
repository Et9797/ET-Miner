"""Back-compat shim for the dense-count threshold filter.

Historical home of the K=8 OOM mitigation (CuPy boolean indexing on a
10.68B-candidate array allocates a hidden ~8 B/element int64 prefix-sum;
the original fix relocated the cost to a full-array D2H). The filter now
lives in ``et_miner.gpu.kernels.filter`` (``threshold_filter``: sliced
``cp.nonzero``, only survivors cross PCIe) and the chunk byte model in
``et_miner.gpu.row_split_chunks``. Only the ``safe_threshold_filter``
name is preserved here for existing callers.
"""

__all__ = ["safe_threshold_filter"]


def safe_threshold_filter(counts_gpu, threshold: int, max_gpu_elements: int = 100_000_000):
    """Deprecated shim — delegates to et_miner.gpu.kernels.filter.

    The historical implementation (full-array D2H above ``max_gpu_elements``,
    whole-array ``cp.where`` below it) is gone; ``threshold_filter`` slices
    the array instead and ``max_gpu_elements`` is ignored.

    Returns:
        Tuple of (indices, filtered_counts) as int64 NumPy arrays, indices
        ascending.
    """
    from et_miner.gpu.kernels.filter import threshold_filter

    return threshold_filter(counts_gpu, threshold)
