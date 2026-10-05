"""CUDA kernel wrappers, split by family; sources live in _src/*.cu.

Re-exports the full public surface so `from et_miner.gpu.kernels import X`
works for every kernel wrapper regardless of which family module holds it.
Import-safe without CuPy.
"""

from .batch import count_itemsets_cuda
from .csr_warp import count_csr_gather, count_csr_range, write_csr_gather
from .decode import decode_k2_pairs_flat, decode_k3plus_flat
from .filter import threshold_filter
from .shared_tiled import (
    compute_cumulative_tilepairs,
    count_pairs_k2_shared,
    count_shared_tiled_allcounts,
    count_tiled_fused,
    k2_groups,
)
from .k2 import K2_ROWS_SHARED_PAIRS, K2Rows, count_pairs_k2_per_candidate, count_pairs_k2_rows, upload_k2_rows
from .k3plus import (
    K3PlusGroups,
    build_k3plus_groups_from_flat,
    count_k3plus_per_candidate,
    select_k3plus_groups,
    upload_k3plus_groups,
)
from .subset_index import SUBSET_INFER, SUBSET_PRUNE, index_nbytes, upload_subset_index
from .loader import (
    clear_kernel_cache,
    column_popcounts,
    get_cuda_kernel,
    get_kernel_source,
    get_popcount_kernel,
)

__all__ = [
    "count_csr_gather",
    "count_csr_range",
    "write_csr_gather",
    "count_itemsets_cuda",
    "get_cuda_kernel",
    "get_kernel_source",
    "clear_kernel_cache",
    "get_popcount_kernel",
    "column_popcounts",
    "count_pairs_k2_per_candidate",
    "count_pairs_k2_rows",
    "upload_k2_rows",
    "K2Rows",
    "K2_ROWS_SHARED_PAIRS",
    "count_k3plus_per_candidate",
    "upload_k3plus_groups",
    "build_k3plus_groups_from_flat",
    "select_k3plus_groups",
    "K3PlusGroups",
    "decode_k2_pairs_flat",
    "decode_k3plus_flat",
    "threshold_filter",
    "compute_cumulative_tilepairs",
    "count_shared_tiled_allcounts",
    "count_pairs_k2_shared",
    "count_tiled_fused",
    "k2_groups",
    "SUBSET_PRUNE",
    "SUBSET_INFER",
    "index_nbytes",
    "upload_subset_index",
]
