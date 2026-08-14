"""
Auto-Tuning for GPU Chunk Sizes in ET-Miner.

Automatically detects optimal chunk sizes based on available GPU memory,
enabling hands-off trillion-scale benchmarking.

Memory Formula:
    bitvec_bytes = n_cols * ceil(n_rows / 64) * 8
    csr_indptr_bytes = (n_rows + 1) * 8
    csr_indices_bytes = n_rows * avg_items_per_row * 8
    total = bitvec + csr_indptr + csr_indices
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import math

from loguru import logger

__all__ = [
    'ChunkConfig',
    'auto_chunk_size',
    'estimate_memory_usage',
    'get_available_gpus',
]


@dataclass
class ChunkConfig:
    """Configuration for GPU chunk processing."""
    chunk_size: int  # Rows per chunk per GPU
    n_gpus: int
    safety_margin: float
    estimated_memory_gb: float
    max_available_gb: float  # Smallest GPU's free memory

    # Optional: per-GPU details
    gpu_details: List[dict] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"ChunkConfig(\n"
            f"  chunk_size={self.chunk_size:,} rows ({self.chunk_size/1e6:.0f}M),\n"
            f"  n_gpus={self.n_gpus},\n"
            f"  estimated_memory={self.estimated_memory_gb:.1f}GB per GPU,\n"
            f"  available_memory={self.max_available_gb:.1f}GB (smallest GPU),\n"
            f"  safety_margin={self.safety_margin:.0%}\n"
            f")"
        )

    @property
    def rows_per_iteration(self) -> int:
        """Total rows processed per iteration across all GPUs."""
        return self.chunk_size * self.n_gpus

    def iterations_for(self, total_rows: int) -> int:
        """Calculate number of iterations needed for given total rows."""
        return math.ceil(total_rows / self.rows_per_iteration)

    def eta_seconds(self, total_rows: int, throughput_rows_per_sec: float) -> float:
        """Estimate time to completion."""
        return total_rows / throughput_rows_per_sec


def estimate_memory_usage(
    n_rows: int,
    n_cols: int = 10_000,
    avg_items_per_row: int = 10,
) -> dict:
    """
    Estimate GPU memory usage for a given chunk size.

    Memory components:
    - Bitvecs: n_cols * ceil(n_rows / 64) * 8 bytes
    - CSR indptr: (n_rows + 1) * 8 bytes
    - CSR indices: n_rows * avg_items_per_row * 8 bytes

    Args:
        n_rows: Number of rows in chunk
        n_cols: Number of items/columns
        avg_items_per_row: Average non-zeros per row

    Returns:
        Dict with memory breakdown in bytes and GB
    """
    n_u64s = math.ceil(n_rows / 64)

    bitvec_bytes = n_cols * n_u64s * 8
    csr_indptr_bytes = (n_rows + 1) * 8
    csr_indices_bytes = n_rows * avg_items_per_row * 8

    total_bytes = bitvec_bytes + csr_indptr_bytes + csr_indices_bytes

    return {
        'n_rows': n_rows,
        'bitvec_bytes': bitvec_bytes,
        'bitvec_gb': bitvec_bytes / (1024**3),
        'csr_indptr_bytes': csr_indptr_bytes,
        'csr_indptr_gb': csr_indptr_bytes / (1024**3),
        'csr_indices_bytes': csr_indices_bytes,
        'csr_indices_gb': csr_indices_bytes / (1024**3),
        'total_bytes': total_bytes,
        'total_gb': total_bytes / (1024**3),
    }


def get_available_gpus(device_ids: Optional[List[int]] = None) -> List[dict]:
    """
    Get information about available GPUs.

    Args:
        device_ids: Specific GPU IDs to use, or None for all

    Returns:
        List of dicts with GPU info (id, name, free_gb, total_gb)
    """
    try:
        import cupy as cp
    except ImportError:
        return []

    n_gpus = cp.cuda.runtime.getDeviceCount()

    if device_ids is None:
        device_ids = list(range(n_gpus))
    else:
        device_ids = [d for d in device_ids if d < n_gpus]

    gpus = []
    for device_id in device_ids:
        with cp.cuda.Device(device_id):
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            free, total = cp.cuda.runtime.memGetInfo()

            gpus.append({
                'id': device_id,
                'name': props['name'].decode() if isinstance(props['name'], bytes) else props['name'],
                'free_bytes': free,
                'free_gb': free / (1024**3),
                'total_bytes': total,
                'total_gb': total / (1024**3),
            })

    return gpus


def auto_chunk_size(
    n_cols: int = 10_000,
    avg_items_per_row: int = 10,
    device_ids: Optional[List[int]] = None,
    safety_margin: float = 0.20,
    max_chunk_size: int = 100_000_000,  # 100M cap
    min_chunk_size: int = 1_000_000,    # 1M minimum
    round_to: int = 5_000_000,          # Round to 5M
) -> ChunkConfig:
    """
    Auto-detect optimal chunk size based on GPU memory.

    Process:
    1. Query free memory on all available GPUs
    2. Use the SMALLEST free memory (bottleneck GPU)
    3. Apply safety margin (default 20%)
    4. Calculate max rows that fit within available memory
    5. Round to nearest `round_to` value

    Args:
        n_cols: Number of columns/items
        avg_items_per_row: Average items per transaction
        device_ids: Specific GPUs to use (None = all available)
        safety_margin: Fraction of memory to keep free (0.20 = 20%)
        max_chunk_size: Maximum allowed chunk size
        min_chunk_size: Minimum chunk size
        round_to: Round chunk size to nearest this value

    Returns:
        ChunkConfig with optimal settings

    Example:
        >>> config = auto_chunk_size()
        >>> print(config)
        ChunkConfig(
          chunk_size=45,000,000 rows (45M),
          n_gpus=2,
          estimated_memory=55.2GB per GPU,
          available_memory=79.5GB (smallest GPU),
          safety_margin=35%
        )

        >>> # Use in benchmark
        >>> for i in range(config.iterations_for(total_rows)):
        ...     process_chunk(config.chunk_size)
    """
    # Get available GPUs
    gpus = get_available_gpus(device_ids)

    if not gpus:
        raise RuntimeError(
            "No GPUs available! Install CuPy and ensure CUDA is configured."
        )

    n_gpus = len(gpus)

    # Use smallest available memory as the bottleneck
    min_free_bytes = min(gpu['free_bytes'] for gpu in gpus)
    min_free_gb = min_free_bytes / (1024**3)

    # Apply safety margin
    usable_bytes = int(min_free_bytes * (1 - safety_margin))

    # Calculate max chunk size that fits
    # Solve for n_rows: total_bytes(n_rows) <= usable_bytes
    #
    # total_bytes = n_cols * ceil(n_rows/64) * 8 + (n_rows+1) * 8 + n_rows * avg_items * 8
    # Approximate: n_cols * n_rows/64 * 8 + n_rows * 8 + n_rows * avg_items * 8
    # = n_rows * (n_cols/8 + 8 + avg_items * 8)
    # = n_rows * bytes_per_row

    bytes_per_row = (n_cols / 8) + 8 + (avg_items_per_row * 8)

    max_rows = int(usable_bytes / bytes_per_row)

    # Apply bounds
    chunk_size = max(min_chunk_size, min(max_rows, max_chunk_size))

    # Round to nearest round_to
    chunk_size = round(chunk_size / round_to) * round_to
    chunk_size = max(min_chunk_size, chunk_size)  # Ensure minimum after rounding

    # Calculate actual memory usage with this chunk size
    memory_estimate = estimate_memory_usage(chunk_size, n_cols, avg_items_per_row)

    return ChunkConfig(
        chunk_size=chunk_size,
        n_gpus=n_gpus,
        safety_margin=safety_margin,
        estimated_memory_gb=memory_estimate['total_gb'],
        max_available_gb=min_free_gb,
        gpu_details=gpus,
    )


def print_memory_breakdown(
    n_rows: int,
    n_cols: int = 10_000,
    avg_items_per_row: int = 10,
) -> None:
    """Print detailed memory breakdown for given chunk size."""
    mem = estimate_memory_usage(n_rows, n_cols, avg_items_per_row)

    logger.info(f"Memory Breakdown for {n_rows:,} rows:")
    logger.info(f"  Bitvecs ({n_cols} cols × {math.ceil(n_rows/64):,} u64s): {mem['bitvec_gb']:.2f} GB")
    logger.info(f"  CSR indptr ({n_rows+1:,} entries):                      {mem['csr_indptr_gb']:.2f} GB")
    logger.info(f"  CSR indices (~{n_rows * avg_items_per_row:,} entries):           {mem['csr_indices_gb']:.2f} GB")
    logger.info(f"  ─────────────────────────────────────────────────────")
    logger.info(f"  TOTAL:                                               {mem['total_gb']:.2f} GB")
