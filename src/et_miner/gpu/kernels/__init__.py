"""CUDA kernel wrappers, split by family; sources live in _src/*.cu.

Re-exports the full public surface so `from et_miner.gpu.kernels import X`
works for every kernel wrapper regardless of which family module holds it.
Import-safe without CuPy.
"""

from .batch import count_itemsets_cuda
from .csr_warp import CANDS_PER_BLOCK, count_csr_gather, count_csr_range, write_csr_gather
from .decode import decode_k2_pairs_flat, decode_k3plus_candidates, decode_k3plus_flat
from .filter import compact_threshold_filter
from .shared_tiled import (
    compute_cumulative_tilepairs,
    count_k3plus_shared_fused,
    count_pairs_k2_shared,
    count_pairs_k2_shared_fused,
    count_shared_tiled_allcounts,
)
from .gpu_resident import (
    build_prefix_groups_gpu,
    count_k3plus_gpu_resident,
    count_k3plus_gpu_resident_multi_gpu,
    count_pairs_fused_k2_gpu_resident,
    count_pairs_fused_k2_gpu_resident_multi_gpu,
)
from .k2 import (
    count_pairs_fused_k2,
    count_pairs_fused_k2_multi_gpu,
    count_pairs_k2_allcounts,
)
from .k3plus import (
    K3PlusGroups,
    build_k3plus_groups,
    build_k3plus_groups_from_flat,
    count_itemsets_fused_k3plus,
    count_itemsets_fused_k3plus_multi_gpu,
    count_k3plus_allcounts,
    count_k3plus_fully_fused,
    count_k3plus_fully_fused_multi_gpu,
    upload_k3plus_groups,
)
from .loader import (
    clear_kernel_cache,
    get_cuda_kernel,
    get_kernel_source,
    get_popcount_kernel,
)

__all__ = [
    "count_itemsets_cuda",
    "get_cuda_kernel",
    "get_kernel_source",
    "clear_kernel_cache",
    "get_popcount_kernel",
    "count_pairs_fused_k2",
    "count_pairs_fused_k2_multi_gpu",
    "count_itemsets_fused_k3plus",
    "count_itemsets_fused_k3plus_multi_gpu",
    "count_k3plus_fully_fused",
    "count_k3plus_fully_fused_multi_gpu",
    "count_pairs_fused_k2_gpu_resident",
    "count_k3plus_gpu_resident",
    "build_prefix_groups_gpu",
    "count_pairs_fused_k2_gpu_resident_multi_gpu",
    "count_k3plus_gpu_resident_multi_gpu",
    "count_pairs_k2_allcounts",
    "count_k3plus_allcounts",
    "CANDS_PER_BLOCK",
    "count_csr_range",
    "count_csr_gather",
    "write_csr_gather",
    "upload_k3plus_groups",
    "build_k3plus_groups",
    "build_k3plus_groups_from_flat",
    "K3PlusGroups",
    "decode_k2_pairs_flat",
    "decode_k3plus_candidates",
    "decode_k3plus_flat",
    "compact_threshold_filter",
    "compute_cumulative_tilepairs",
    "count_shared_tiled_allcounts",
    "count_k3plus_shared_fused",
    "count_pairs_k2_shared",
    "count_pairs_k2_shared_fused",
]
