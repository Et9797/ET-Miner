"""NCCL collective helpers for multi-GPU mining (reduction of count arrays).

The dense row-split levels reduce per-GPU partial int32 count arrays onto
GPU 0, which alone filters survivors. Preferred path is ncclReduce to root
0 (half the traffic of allReduce — no rank needs the sum but rank 0); when
NCCL is unavailable (or disabled via ET_MINER_DISABLE_NCCL=1), a bounded
staged device-to-device fallback adds peers slice-wise through one fixed
staging buffer on GPU 0 instead of materializing full peer copies.

Some boxes report peer access between two devices and then lose the
writes: a device-to-device copy returns success with the destination
untouched, or written elsewhere, and NCCL's P2P transport hangs at the
first collective (a Ryzen AM4 host with two RTX A4000s behind the CPU's
host bridge; `bench/results/2026-09-28-consolidation-2gpu/nccl-hang/`). No
probe can certify such a box: small copies land while large ones drop. So
``copy_between_devices`` stages every cross-device copy through host memory
unless ``ET_MINER_DIRECT_D2D=1`` opts in, and ``_init_nccl`` creates its
communicators under ``NCCL_P2P_LEVEL=NVL`` (P2P over NVLink only, SHM
elsewhere) unless the caller set ``NCCL_P2P_LEVEL`` or ``NCCL_P2P_DISABLE``.
NCCL reads that variable once per process, at its first communicator, so
the setting has no effect when another library initialised NCCL first; it
is put back afterwards so it leaks into nothing.
"""

from __future__ import annotations

import contextlib
import os

from loguru import logger

from et_miner import _env



def _host_staged_copy(dst, src) -> None:
    """``src`` → host → ``dst``: no device-to-device write is issued."""
    import cupy as cp

    with cp.cuda.Device(src.device.id):
        host = src.get()
    with cp.cuda.Device(dst.device.id):
        dst.set(host)
        cp.cuda.Device().synchronize()


def _direct_copy(dst, src) -> None:
    """The opt-in device-to-device copy (``ET_MINER_DIRECT_D2D=1``)."""
    import cupy as cp

    with cp.cuda.Device(src.device.id):
        src = cp.ascontiguousarray(src)
    with cp.cuda.Device(dst.device.id):
        dst.data.copy_from_device(src.data, src.nbytes)
        cp.cuda.Device().synchronize()


def copy_between_devices(dst, src) -> None:
    """Copy ``src`` into ``dst`` (same shape and dtype; ``dst`` contiguous).

    Through host memory by default. ``ET_MINER_DIRECT_D2D=1`` issues a direct
    device-to-device copy instead, for NVLink or a known-good PCIe switch
    only: a direct copy that does not land corrupts silently, and no probe
    can tell such a pair apart (see the module docstring). Both callers, the
    non-NCCL reduce and the ``bitvecs=`` shard copy, are off the hot path.
    On one device the copy is a plain device assignment.
    """
    import cupy as cp

    if dst.shape != src.shape or dst.dtype != src.dtype:
        raise ValueError(f"copy_between_devices: shape/dtype mismatch {dst.shape}/{dst.dtype} vs {src.shape}/{src.dtype}")
    if int(dst.device.id) == int(src.device.id):
        with cp.cuda.Device(dst.device.id):
            dst[...] = src
        return
    if _env.direct_d2d():
        _direct_copy(dst, src)
    else:
        _host_staged_copy(dst, src)


@contextlib.contextmanager
def _nccl_p2p_policy():
    """``NCCL_P2P_LEVEL=NVL`` while the communicators are created, unless the caller chose.

    NCCL reads its P2P settings once per process, at its first
    communicator; the variable is put back afterwards so it does not leak
    into child processes or other NCCL users of this one.
    """
    if any(os.environ.get(k, "").strip() for k in ("NCCL_P2P_LEVEL", "NCCL_P2P_DISABLE")):
        yield
        return
    previous = os.environ.get("NCCL_P2P_LEVEL")
    os.environ["NCCL_P2P_LEVEL"] = "NVL"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("NCCL_P2P_LEVEL", None)
        else:
            os.environ["NCCL_P2P_LEVEL"] = previous


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

        with _nccl_p2p_policy(), ThreadPoolExecutor(max_workers=n) as pool:
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

    Every slice reaches GPU 0 through ``copy_between_devices`` (host memory
    unless ``ET_MINER_DIRECT_D2D=1``). Peak extra VRAM on GPU 0 is the fixed
    STAGING_BYTES buffer, never a full peer copy of the chunk.
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
                copy_between_devices(staging[:m], src[start : start + m])
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


