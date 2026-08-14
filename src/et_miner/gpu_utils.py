"""GPU Utilities for ET-Miner: Pinned Memory Transfers, Multi-GPU Distribution, and Memory Management.

Provides:
- Pinned memory transfers for 2x faster CPU→GPU transfers.
- Multi-GPU distribution helpers
- Memory management utilities
"""

from typing import Optional, List, Tuple
import numpy as np

__all__ = [
    "transfer_pinned",
    "get_gpu_count",
    "get_gpu_memory_info",
    "distribute_to_gpus",
    "cleanup_gpu_memory",
]


def transfer_pinned(array: np.ndarray) -> "cp.ndarray":
    """
    Transfer numpy array to GPU using pinned (page-locked) memory.

    Pinned memory enables DMA transfers that bypass CPU,
    achieving ~2x faster transfer speeds for large arrays.

    Args:
        array: Numpy array to transfer

    Returns:
        CuPy array on GPU

    Example:
        >>> bitvecs_gpu = transfer_pinned(bitvecs_np)  # 2x faster!
    """
    import cupy as cp

    # Allocate pinned memory
    pinned_mem = cp.cuda.alloc_pinned_memory(array.nbytes)

    # Create numpy array backed by pinned memory
    # Note: specify count to handle potential buffer over-allocation
    pinned_array = np.frombuffer(
        pinned_mem,
        dtype=array.dtype,
        count=array.size,  # Fix: exact element count, ignore extra padding
    ).reshape(array.shape)

    # Copy to pinned memory
    np.copyto(pinned_array, array)

    # Transfer to GPU via DMA
    gpu_array = cp.asarray(pinned_array)

    return gpu_array


def get_gpu_count() -> int:
    """Get number of available CUDA GPUs."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount()
    except ImportError:
        return 0
    except Exception:
        return 0


def get_gpu_memory_info(device_id: int = 0) -> Tuple[int, int]:
    """Return (free_bytes, total_bytes) for the given CUDA device."""
    import cupy as cp

    with cp.cuda.Device(device_id):
        free, total = cp.cuda.runtime.memGetInfo()
        return free, total


def distribute_to_gpus(
    array: np.ndarray,
    n_gpus: Optional[int] = None,
    axis: int = 0,
) -> List["cp.ndarray"]:
    """
    Distribute a numpy array across multiple GPUs.

    Args:
        array: Numpy array to distribute
        n_gpus: Number of GPUs (auto-detect if None)
        axis: Axis along which to split

    Returns:
        List of CuPy arrays, one per GPU

    Example:
        >>> # Split 8000 columns across 8 GPUs (1000 each)
        >>> gpu_arrays = distribute_to_gpus(bitvecs, n_gpus=8, axis=0)
    """
    import cupy as cp

    if n_gpus is None:
        n_gpus = get_gpu_count()

    if n_gpus == 0:
        raise RuntimeError("No GPUs available")

    # Split array
    chunks = np.array_split(array, n_gpus, axis=axis)

    # Transfer each chunk to its GPU
    gpu_arrays = []
    for i, chunk in enumerate(chunks):
        with cp.cuda.Device(i):
            gpu_arrays.append(cp.asarray(chunk))

    return gpu_arrays


def cleanup_gpu_memory(device_id: Optional[int] = None):
    """
    Free all cached GPU memory.

    Args:
        device_id: Specific GPU to clean, or all if None
    """
    import cupy as cp

    if device_id is not None:
        with cp.cuda.Device(device_id):
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
    else:
        n_gpus = get_gpu_count()
        for i in range(n_gpus):
            with cp.cuda.Device(i):
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()


def count_itemsets_batched(
    bitvecs_gpu: "cp.ndarray",
    itemsets: List[np.ndarray],
) -> np.ndarray:
    """
    Count itemset support using batched GPU operations.

    Uses fused AND reduction to minimize kernel launches.

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s)
        itemsets: List of numpy arrays, each containing item indices

    Returns:
        Numpy array of support counts
    """
    import cupy as cp

    counts = np.zeros(len(itemsets), dtype=np.int64)

    for i, itemset in enumerate(itemsets):
        if len(itemset) == 0:
            continue

        if len(itemset) == 1:
            result = bitvecs_gpu[itemset[0]]
        else:
            # Fused AND reduction - single kernel call!
            item_bitvecs = bitvecs_gpu[list(itemset)]
            result = cp.bitwise_and.reduce(item_bitvecs, axis=0)

        # Popcount
        counts[i] = int(cp.sum(cp.unpackbits(result.view(cp.uint8))))

    return counts
