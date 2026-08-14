"""ET-Miner: Efficient Transaction Miner.

A blazing fast Apriori implementation with Python, Rust backend, and multi-GPU support.

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

# Core API - flat imports
from et_miner.apriori import apriori
from et_miner.rules import (
    Rule,
    generate_rules,
    generate_rules_drop1,
    generate_rules_top_n,
    compute_self_sufficiency,
)
from et_miner.streaming import apriori_streaming

# Matrix operations (advanced)
from et_miner.matrix import (
    build_boolean_matrix,
    count_support_batched,
    count_support_sparse,
    count_support_vectorized,
)

# Config & logging
from et_miner._compat import HAS_TQDM
from et_miner.config import Config, load_config
from et_miner._logging import HAS_LOGURU, configure_logging, logger

# Progress & profiling
from et_miner.profiling import ProfilingSession
from et_miner.progress import (
    LevelStats,
    ProgressTracker,
    create_live_level_printer,
    create_silent_tracker,
)

# Multi-GPU streaming (optional, requires cupy)
try:
    from et_miner.streaming_multi_gpu import (
        apriori_streaming_multi_gpu,
        get_gpu_count,
        has_cupy,
    )
    HAS_MULTI_GPU = True
except ImportError:
    HAS_MULTI_GPU = False
    apriori_streaming_multi_gpu = None
    get_gpu_count = None
    has_cupy = None

# GPU acceleration (optional, requires cupy)
try:
    from et_miner.cuda_csr_bitvec import build_bitvecs_gpu
    from et_miner.cuda_csr_build import generate_bitvecs_gpu, generate_csr_gpu
    from et_miner.cuda_multi_gpu import (
        warmup_cuda_multi_gpu,
        generate_bitvecs_multi_gpu,
        count_itemsets_multi_gpu,
    )
    from et_miner.auto_chunk import auto_chunk_size, ChunkConfig
    from et_miner.gpu_dispatch import dispatch_k2, should_use_multi_gpu
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    build_bitvecs_gpu = None
    generate_bitvecs_gpu = None
    generate_csr_gpu = None
    warmup_cuda_multi_gpu = None
    generate_bitvecs_multi_gpu = None
    count_itemsets_multi_gpu = None
    auto_chunk_size = None
    ChunkConfig = None
    dispatch_k2 = None
    should_use_multi_gpu = None

# CSR Direct Mode (Rust)
try:
    from et_miner_rust import apriori_from_csr
    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    apriori_from_csr = None

# Ramdisk data generator (optional, requires root for mount operations)
try:
    from et_miner.ramdisk_generator import (
        setup_ramdisk,
        generate_wave_to_disk,
        load_wave_from_disk,
        cleanup_ramdisk,
        get_ramdisk_info,
        delete_wave,
        list_waves,
    )
    HAS_RAMDISK = True
except ImportError:
    HAS_RAMDISK = False
    setup_ramdisk = None
    generate_wave_to_disk = None
    load_wave_from_disk = None
    cleanup_ramdisk = None
    get_ramdisk_info = None
    delete_wave = None
    list_waves = None

# CUDA Streams pipeline for GPU optimization (optional, requires cupy)
try:
    from et_miner.async_pipeline import (
        StreamContext,
        WaveData,
        run_streams_benchmark,
        run_streams_benchmark_sequential,
        compare_pipelines,
    )
    HAS_ASYNC_PIPELINE = True
except ImportError:
    HAS_ASYNC_PIPELINE = False
    StreamContext = None
    WaveData = None
    run_streams_benchmark = None
    run_streams_benchmark_sequential = None
    compare_pipelines = None

# Safeguards for billion-scale mining (type safety + memory budget)
from et_miner.type_safety import (
    INT32_MAX,
    safe_offset_dtype,
    assert_fits_int32,
    warn_if_large,
)
from et_miner.memory_budget import (
    estimate_boolean_index_memory,
    check_vram_budget,
    safe_threshold_filter,
    set_mempool_limit,
    get_mempool_stats,
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
    # Multi-GPU
    "apriori_streaming_multi_gpu",
    "get_gpu_count",
    "has_cupy",
    "HAS_MULTI_GPU",
    # GPU
    "build_bitvecs_gpu",
    "generate_bitvecs_gpu",
    "generate_csr_gpu",
    "warmup_cuda_multi_gpu",
    "generate_bitvecs_multi_gpu",
    "count_itemsets_multi_gpu",
    "auto_chunk_size",
    "ChunkConfig",
    "dispatch_k2",
    "should_use_multi_gpu",
    "HAS_GPU",
    # Rust
    "apriori_from_csr",
    "HAS_RUST",
    # Ramdisk
    "setup_ramdisk",
    "generate_wave_to_disk",
    "load_wave_from_disk",
    "cleanup_ramdisk",
    "get_ramdisk_info",
    "delete_wave",
    "list_waves",
    "HAS_RAMDISK",
    # CUDA Streams Pipeline
    "StreamContext",
    "WaveData",
    "run_streams_benchmark",
    "run_streams_benchmark_sequential",
    "compare_pipelines",
    "HAS_ASYNC_PIPELINE",
    # Config & logging
    "Config",
    "load_config",
    "configure_logging",
    "logger",
    "HAS_LOGURU",
    "HAS_TQDM",
    # Safeguards (type safety + memory budget)
    "INT32_MAX",
    "safe_offset_dtype",
    "assert_fits_int32",
    "warn_if_large",
    "estimate_boolean_index_memory",
    "check_vram_budget",
    "safe_threshold_filter",
    "set_mempool_limit",
    "get_mempool_stats",
    # Meta
    "__version__",
    "__author__",
]
