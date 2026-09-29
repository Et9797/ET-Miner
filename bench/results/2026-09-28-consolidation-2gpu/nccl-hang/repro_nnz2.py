"""Warm-up repro with a traced ncclReduce: per rank the arguments, exceptions and a bounded sync."""
import faulthandler
import sys
import threading
import time
faulthandler.dump_traceback_later(60, exit=True)
import cupy as cp  # noqa: E402  (armed the hang dump first)
import polars as pl  # noqa: E402  (armed the hang dump first)
import et_miner.gpu.nccl as nm  # noqa: E402  (armed the hang dump first)
from et_miner import apriori  # noqa: E402  (armed the hang dump first)

T0 = time.perf_counter()
def _log(msg):
    print(f"[trace +{time.perf_counter() - T0:7.3f}s tid={threading.get_ident() % 100000:5d}] {msg}", flush=True)

def _traced_reduce_sum_to_root(gpu_arrays, comms, device_ids):
    from concurrent.futures import ThreadPoolExecutor
    NCCL_INT32, NCCL_SUM = 2, 0
    def _reduce_rank(rank):
        try:
            with cp.cuda.Device(device_ids[rank]):
                stream = cp.cuda.get_current_stream()
                arr = gpu_arrays[rank]
                _log(f"rank {rank}: comm rank_id={comms[rank].rank_id()} comm_dev={comms[rank].device_id()} "
                     f"arr dev={arr.device.id} size={arr.size} dtype={arr.dtype} ptr={arr.data.ptr:#x} stream={stream.ptr:#x}")
                assert arr.dtype == cp.int32
                comms[rank].reduce(arr.data.ptr, arr.data.ptr, arr.size, NCCL_INT32, NCCL_SUM, 0, stream.ptr)
                _log(f"rank {rank}: reduce enqueued")
                ev = cp.cuda.Event()
                ev.record(stream)
                t = time.perf_counter()
                while not ev.done:
                    if time.perf_counter() - t > 20:
                        _log(f"rank {rank}: SYNC TIMED OUT after 20 s (kernel never completed)")
                        return "timeout"
                    time.sleep(0.005)
                _log(f"rank {rank}: sync done in {time.perf_counter() - t:.3f}s")
                return "ok"
        except BaseException as e:
            _log(f"rank {rank}: RAISED {type(e).__name__}: {e}")
            raise
    with ThreadPoolExecutor(max_workers=len(device_ids)) as pool:
        futs = [pool.submit(_reduce_rank, r) for r in range(len(device_ids))]
        res = [f.result() for f in futs]
    _log(f"reduce results: {res}")
    if "timeout" in res:
        raise RuntimeError("NCCL reduce hang detected (bounded sync)")

nm._nccl_reduce_sum_to_root = _traced_reduce_sum_to_root

repo = sys.argv[1]
df = pl.read_parquet(f"{repo}/datasets/synth/smoke.parquet").head(20000)
t0 = time.perf_counter()
res = apriori(df, use_gpu=True, n_gpus=2, min_support=0.02, max_length=3)
n = getattr(res, "height", None)
print(f"OK {time.perf_counter() - t0:.2f}s itemsets={n}", flush=True)
