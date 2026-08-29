"""NCCL collective helpers for multi-GPU mining (all-reduce of count arrays)."""

from __future__ import annotations

from loguru import logger


def _init_nccl(device_ids):
    """Initialize NCCL communicators for multi-GPU all-reduce.

    Ring topology: O(data/n_gpus) bandwidth vs O(n_gpus × data) for sequential D2D.
    Every GPU gets the result → parallel filtering, no GPU 0 bottleneck.

    Returns (comms, True) on success, (None, False) on failure.
    """
    try:
        import cupy as cp
        from cupy.cuda import nccl as _nccl
        from concurrent.futures import ThreadPoolExecutor

        n = len(device_ids)
        uid = _nccl.get_unique_id()
        comms = [None] * n

        def _init_rank(rank):
            with cp.cuda.Device(device_ids[rank]):
                comms[rank] = _nccl.NcclCommunicator(n, uid, rank)

        with ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(_init_rank, range(n)))

        return comms, True
    except Exception as e:
        logger.warning(f"NCCL init failed: {e}")
        return None, False


def _nccl_allreduce_sum(gpu_arrays, comms, device_ids):
    """In-place NCCL all-reduce SUM across GPUs.

    After call, every GPU has the global sum in its own array.
    All ranks must participate simultaneously (collective operation).
    """
    import cupy as cp
    from concurrent.futures import ThreadPoolExecutor

    NCCL_INT32 = 2
    NCCL_UINT32 = 3
    NCCL_INT64 = 4
    NCCL_UINT64 = 5
    NCCL_SUM = 0
    _NCCL_DTYPE_MAP = {
        cp.int32: NCCL_INT32,
        cp.uint32: NCCL_UINT32,
        cp.int64: NCCL_INT64,
        cp.uint64: NCCL_UINT64,
    }

    def _reduce_rank(rank):
        with cp.cuda.Device(device_ids[rank]):
            stream = cp.cuda.get_current_stream()
            arr = gpu_arrays[rank]
            nccl_dtype = _NCCL_DTYPE_MAP.get(arr.dtype.type)
            if nccl_dtype is None:
                raise TypeError(f"Unsupported dtype for NCCL allreduce: {arr.dtype}")
            comms[rank].allReduce(
                arr.data.ptr,
                arr.data.ptr,  # in-place
                arr.size,
                nccl_dtype,
                NCCL_SUM,
                stream.ptr,
            )
            stream.synchronize()

    with ThreadPoolExecutor(max_workers=len(device_ids)) as pool:
        list(pool.map(_reduce_rank, range(len(device_ids))))


