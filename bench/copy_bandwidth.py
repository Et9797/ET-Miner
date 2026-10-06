"""Device-to-device copy bandwidth per GPU: the yardstick of phase B's intersection model.

On every visible GPU, copies one buffer into another with an asynchronous
device-to-device memcpy on the null stream: once untimed, then `--reps` times
between CUDA events. GB/s counts the bytes read plus the bytes written
(2 × nbytes per copy). Prints one JSON line:
{"gbps": {device: median GB/s}, "nbytes": ..., "reps": ...}.
bench/runner.py runs it before `--mode o5-calibration` (PROTOCOL-B.md, reading 4).

Usage: uv run python bench/copy_bandwidth.py [--nbytes N] [--reps R]
"""

from __future__ import annotations

import argparse
import json
import statistics

_D2D = 3  # cudaMemcpyDeviceToDevice


def measure(device_id: int, nbytes: int = 1 << 30, reps: int = 10) -> float:
    """Median GB/s (read + written) of `reps` device-to-device copies of `nbytes` on one GPU."""
    import cupy as cp

    with cp.cuda.Device(device_id):
        src = cp.ones(nbytes, dtype=cp.uint8)
        dst = cp.empty_like(src)
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        times = []
        for i in range(reps + 1):
            start.record()
            cp.cuda.runtime.memcpyAsync(dst.data.ptr, src.data.ptr, nbytes, _D2D, 0)
            stop.record()
            stop.synchronize()
            if i:
                times.append(cp.cuda.get_elapsed_time(start, stop) / 1000)
        del src, dst
        cp.get_default_memory_pool().free_all_blocks()
    return 2 * nbytes / statistics.median(times) / 1e9


def main() -> int:
    import cupy as cp

    ap = argparse.ArgumentParser()
    ap.add_argument("--nbytes", type=int, default=1 << 30)
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()
    gbps = {str(d): round(measure(d, args.nbytes, args.reps), 1) for d in range(cp.cuda.runtime.getDeviceCount())}
    print(json.dumps({"gbps": gbps, "nbytes": args.nbytes, "reps": args.reps}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
