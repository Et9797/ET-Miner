"""ET-Miner: Efficient Transaction Miner.

A blazing fast Apriori implementation with Python, Rust backend, and multi-GPU support.

The top level exposes the mining API plus capability probes. GPU plumbing,
ramdisk helpers, and benchmark harnesses live in their subpackages
(``et_miner.gpu``, ``et_miner.streaming``) and are imported from there.

Example:
    >>> from et_miner import apriori, generate_rules
    >>> import polars as pl
    >>>
    >>> df = pl.DataFrame({"items": [[1, 2], [2, 3], [1, 3]]})
    >>> itemsets = apriori(df, min_support=0.5)
    >>> rules = generate_rules(itemsets, min_confidence=0.7)
"""

__version__ = "0.1.0"
__author__ = "E. Ahmic"

from et_miner import backends
from et_miner.exceptions import MiningError

# Core API
from et_miner.apriori import apriori
from et_miner.rules import (
    Rule,
    generate_rules,
    generate_rules_drop1,
    generate_rules_top_n,
    compute_self_sufficiency,
)
from et_miner.streaming import apriori_streaming
from et_miner.streaming_multi_gpu import apriori_streaming_multi_gpu

# Matrix operations (advanced)
from et_miner.matrix import (
    build_boolean_matrix,
    count_support_batched,
    count_support_sparse,
    count_support_vectorized,
)

# Config & logging
from et_miner.config import Config, load_config
from et_miner._logging import configure_logging, logger

# Progress & profiling
from et_miner.profiling import ProfilingSession
from et_miner.progress import (
    LevelStats,
    ProgressTracker,
    create_live_level_printer,
    create_silent_tracker,
)

# Capability surface — honest values from the single detection point.
# HAS_GPU: cupy is importable (GPU modules are usable); HAS_MULTI_GPU: more
# than one CUDA device is actually visible right now.
has_cupy = backends.has_cupy
get_gpu_count = backends.get_gpu_count
has_rust_extension = backends.has_rust_extension
get_rust_version = backends.get_rust_version
HAS_GPU = backends.CUPY_INSTALLED
HAS_MULTI_GPU = backends.get_gpu_count() > 1
HAS_RUST = backends.has_rust_extension()

# CSR Direct Mode (Rust)
if HAS_RUST:
    from et_miner_rust import apriori_from_csr
else:

    def apriori_from_csr(*args, **kwargs):
        """Stub raised when the Rust extension is not built."""
        raise MiningError(
            "apriori_from_csr requires the Rust extension. Build it with: "
            "cd rust_ext && maturin develop --release (see README, Tier 2)"
        )


__all__ = [
    # Core
    "apriori",
    "generate_rules",
    "generate_rules_drop1",
    "generate_rules_top_n",
    "compute_self_sufficiency",
    "Rule",
    "apriori_streaming",
    "apriori_streaming_multi_gpu",
    # Matrix
    "build_boolean_matrix",
    "count_support_batched",
    "count_support_sparse",
    "count_support_vectorized",
    # Progress
    "ProfilingSession",
    "LevelStats",
    "ProgressTracker",
    "create_live_level_printer",
    "create_silent_tracker",
    # Rust
    "apriori_from_csr",
    # Capability probes
    "has_cupy",
    "get_gpu_count",
    "has_rust_extension",
    "get_rust_version",
    "HAS_GPU",
    "HAS_MULTI_GPU",
    "HAS_RUST",
    # Config & logging
    "Config",
    "load_config",
    "configure_logging",
    "logger",
    # Meta
    "__version__",
    "__author__",
]
