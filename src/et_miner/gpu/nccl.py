"""NCCL collective helpers for multi-GPU mining (reduction of count arrays).

The dense row-split levels reduce per-GPU partial int32 count arrays onto
GPU 0, which alone filters survivors. Preferred path is ncclReduce to root
0 (half the traffic of allReduce — no rank needs the sum but rank 0); when
NCCL is unavailable (or disabled via ET_MINER_DISABLE_NCCL=1), a bounded
staged device-to-device fallback adds peers slice-wise through one fixed
staging buffer on GPU 0 instead of materializing full peer copies.

Some boxes report peer access between two devices and then lose the
writes: a device-to-device copy returns success with the destination
untouched, and NCCL's P2P transport hangs at the first collective (seen on
a Ryzen AM4 host with two RTX A4000s behind the CPU's host bridge,
`bench/results/2026-09-28-consolidation-2gpu/nccl-hang/`). ``peer_copy_works``
probes each device pair once with a 4 KiB pattern; when the copy does not
land, the staged fallback goes through host memory and NCCL is started with
``NCCL_P2P_DISABLE=1`` (its SHM transport), unless the caller set that
variable already.
"""

from __future__ import annotations

import os

from loguru import logger

from et_miner import _env

_peer_copy_ok: dict[tuple[int, int], bool] = {}


def peer_copy_works(dst_device: int, src_device: int) -> bool:
    """Whether a device-to-device copy from ``src_device`` lands on ``dst_device``.

    Probed once per pair per process (4 KiB pattern, both devices
    synchronized before the check) and cached. A failed probe is logged once.
    """
    import numpy as np

    key = (int(dst_device), int(src_device))
    if key in _peer_copy_ok:
        return _peer_copy_ok[key]
    if key[0] == key[1]:
        _peer_copy_ok[key] = True
        return True
    import cupy as cp

    pattern = np.arange(1024, dtype=np.int32) * 7 + 3
    with cp.cuda.Device(key[1]):
        src = cp.asarray(pattern)
        cp.cuda.Device().synchronize()
    with cp.cuda.Device(key[0]):
        dst = cp.full(pattern.size, -1, dtype=cp.int32)
        dst.data.copy_from_device(src.data, pattern.nbytes)
        cp.cuda.Device().synchronize()
        ok = bool(np.array_equal(dst.get(), pattern))
    _peer_copy_ok[key] = ok
    if not ok:
        logger.warning(
            f"device-to-device copies from GPU {key[1]} to GPU {key[0]} do not land (PCIe P2P drops "
            "them on this box): the staged reduce goes through host memory and NCCL runs with "
            "NCCL_P2P_DISABLE=1"
        )
    return ok

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

        # NCCL reads NCCL_P2P_DISABLE once, at its first communicator; a
        # box where copies do not land in either direction of some pair
        # must not use the P2P transport (the ring sends both ways).
        if any(not peer_copy_works(a, b) for a in device_ids for b in device_ids if a != b):
            os.environ.setdefault("NCCL_P2P_DISABLE", "1")
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
    the vast.ai case); a pair whose copies do not land (``peer_copy_works``)
    is staged through host memory explicitly. Peak extra VRAM on GPU 0 is
    the fixed STAGING_BYTES buffer, never a full peer copy of the chunk.
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
            direct = peer_copy_works(dev0, device_ids[i])
            for start in range(0, n, staging_elems):
                m = min(staging_elems, n - start)
                if direct:
                    staging[:m].data.copy_from_device(src[start : start + m].data, m * itemsize)
                else:
                    with cp.cuda.Device(device_ids[i]):
                        host = src[start : start + m].get()
                    staging[:m].set(host)
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


