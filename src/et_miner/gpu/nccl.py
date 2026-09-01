"""NCCL collective helpers for multi-GPU mining (reduction of count arrays).

The dense row-split levels reduce per-GPU partial int32 count arrays onto
GPU 0, which alone filters survivors. Preferred path is ncclReduce to root
0 (half the traffic of allReduce — no rank needs the sum but rank 0); when
NCCL is unavailable (or disabled via ET_MINER_DISABLE_NCCL=1), a bounded
staged device-to-device fallback adds peers slice-wise through one fixed
staging buffer on GPU 0 instead of materializing full peer copies.
"""

from __future__ import annotations

from loguru import logger

from et_miner import _env

#: Fixed staging buffer for the non-NCCL fallback reduce — one allocation
#: on GPU 0, reserved by the chunk budget (row_split_chunks) so slice-wise
#: peer adds never surprise the allocator.
STAGING_BYTES = 512 * (1 << 20)


def _init_nccl(device_ids):
    """Initialize NCCL communicators for multi-GPU reduction.

    Ring topology: O(data/n_gpus) bandwidth vs O(n_gpus × data) for
    sequential D2D. Honors ET_MINER_DISABLE_NCCL=1 (forces the staged
    fallback — for tests and for boxes where NCCL misbehaves).

    Returns (comms, True) on success, (None, False) on failure/disable.
    """
    if _env.disable_nccl():
        logger.info("NCCL disabled via ET_MINER_DISABLE_NCCL — using staged D2D reduce")
        return None, False
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


def _nccl_reduce_sum_to_root(gpu_arrays, comms, device_ids):
    """NCCL reduce SUM to rank 0 — only GPU 0's array holds the result.

    Half the PCIe traffic of allReduce; the dense filter runs on GPU 0
    only, so no other rank needs the sum. Raises AttributeError when the
    CuPy binding lacks ``reduce`` (caller falls back to allReduce).
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
    if not hasattr(comms[0], "reduce"):
        raise AttributeError("CuPy NcclCommunicator has no reduce binding")

    def _reduce_rank(rank):
        with cp.cuda.Device(device_ids[rank]):
            stream = cp.cuda.get_current_stream()
            arr = gpu_arrays[rank]
            nccl_dtype = _NCCL_DTYPE_MAP.get(arr.dtype.type)
            if nccl_dtype is None:
                raise TypeError(f"Unsupported dtype for NCCL reduce: {arr.dtype}")
            # recvbuf is only read on the root; every rank passes its own
            # pointer (in-place on rank 0).
            comms[rank].reduce(
                arr.data.ptr,
                arr.data.ptr,
                arr.size,
                nccl_dtype,
                NCCL_SUM,
                0,  # root rank
                stream.ptr,
            )
            stream.synchronize()

    with ThreadPoolExecutor(max_workers=len(device_ids)) as pool:
        list(pool.map(_reduce_rank, range(len(device_ids))))


def _staged_reduce_to_gpu0(gpu_arrays, device_ids):
    """Non-NCCL fallback: slice-wise peer adds through one staging buffer.

    UVA ``copy_from_device`` handles the cross-device copy with or without
    peer access (the driver stages through the host when P2P is absent —
    the vast.ai case). Peak extra VRAM on GPU 0 is the fixed STAGING_BYTES
    buffer, never a full peer copy of the chunk.
    """
    import cupy as cp

    dev0 = device_ids[0]
    with cp.cuda.Device(dev0):
        target = gpu_arrays[0]
        n = int(target.size)
        itemsize = target.dtype.itemsize
        staging_elems = max(1, min(n, STAGING_BYTES // itemsize))
        staging = cp.empty(staging_elems, dtype=target.dtype)
        for i in range(1, len(gpu_arrays)):
            src = gpu_arrays[i]
            for start in range(0, n, staging_elems):
                m = min(staging_elems, n - start)
                staging[:m].data.copy_from_device(src[start : start + m].data, m * itemsize)
                target[start : start + m] += staging[:m]
        del staging


def reduce_sum_to_gpu0(gpu_arrays, device_ids, comms=None):
    """Sum per-GPU partial arrays into ``gpu_arrays[0]`` (in place).

    With ``comms``: ncclReduce to root 0, falling back to in-place
    allReduce when the binding lacks ``reduce``. Without: the bounded
    staged D2D fallback. Non-root arrays are left in an unspecified state
    — callers must only consume ``gpu_arrays[0]`` afterwards.

    A single array is already the sum: return without touching NCCL or the
    staged fallback (which would otherwise allocate up to STAGING_BYTES on
    the lone device to add nothing — the 1-GPU miner and 1-GPU boxes).
    """
    if len(gpu_arrays) <= 1:
        return
    if comms is not None:
        try:
            _nccl_reduce_sum_to_root(gpu_arrays, comms, device_ids)
            return
        except AttributeError:
            logger.warning("NCCL reduce binding unavailable — falling back to allReduce")
            _nccl_allreduce_sum(gpu_arrays, comms, device_ids)
            return
    _staged_reduce_to_gpu0(gpu_arrays, device_ids)


