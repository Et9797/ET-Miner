"""On-device preflight: fail in minute one, not hour two.

Compiles EVERY kernel registered in gpu/kernels/loader.py on every device
(NVRTC compiles per compute capability — this is where an sm_86
incompatibility would surface), smoke-launches the critical wrappers on
tiny data, and prints the device/NCCL/P2P//dev/shm/rust matrix.

Exit code 0 = ready for the campaign.
"""

from __future__ import annotations

import os
import sys

import numpy as np


def main() -> int:
    failures: list[str] = []

    try:
        import cupy as cp
    except Exception as e:
        print(f"FATAL: cupy import failed: {e}")
        return 2

    n_dev = cp.cuda.runtime.getDeviceCount()
    drv = cp.cuda.runtime.driverGetVersion()
    print(f"devices: {n_dev} | driver {drv // 1000}.{drv % 1000 // 10} | cupy {cp.__version__}")
    for d in range(n_dev):
        props = cp.cuda.runtime.getDeviceProperties(d)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        free, total = cp.cuda.Device(d).mem_info
        print(
            f"  [{d}] {name} sm_{props['major']}{props['minor']} "
            f"{free / 1e9:.1f}/{total / 1e9:.1f} GB free"
        )

    # P2P matrix
    if n_dev > 1:
        for a in range(n_dev):
            row = []
            for b in range(n_dev):
                row.append("-" if a == b else str(cp.cuda.runtime.deviceCanAccessPeer(a, b)))
            print(f"  P2P from {a}: {' '.join(row)}")

    # /dev/shm — NCCL's SHM transport needs real room when P2P is absent
    try:
        shm = os.statvfs("/dev/shm")
        shm_gb = shm.f_bsize * shm.f_blocks / 1e9
        print(f"/dev/shm: {shm_gb:.2f} GB")
        if shm_gb < 1.0:
            print("  WARNING: small /dev/shm (Docker default is 64 MB) — raise --shm-size or NCCL may hang; NCCL_SHM_DISABLE=1 is the fallback")
    except Exception:
        pass

    # NCCL availability + init across all devices
    try:
        from cupy.cuda import nccl as _nccl  # noqa: F401

        from et_miner.gpu.nccl import _init_nccl

        comms, ok = _init_nccl(list(range(n_dev)))
        print(f"NCCL: available, init {'OK' if ok else 'FAILED (staged D2D fallback will be used)'}")
        has_reduce = ok and hasattr(comms[0], "reduce")
        print(f"NCCL reduce-to-root binding: {'yes' if has_reduce else 'no (allReduce fallback)'}")
    except Exception as e:
        print(f"NCCL: unavailable ({e}) — staged D2D fallback will be used")

    # Rust extension
    try:
        from et_miner.backends import get_rust_ext

        print(f"rust ext: {'present' if get_rust_ext() else 'MISSING (numpy fallbacks active)'}")
    except Exception as e:
        print(f"rust ext: MISSING ({e})")

    # Compile every registered kernel on every device
    from et_miner.gpu.kernels.loader import _KERNEL_FILES, get_cuda_kernel

    print(f"compiling {len(_KERNEL_FILES)} kernels on {n_dev} device(s)...")
    for d in range(n_dev):
        with cp.cuda.Device(d):
            for name in sorted(_KERNEL_FILES):
                try:
                    get_cuda_kernel(name).kernel  # forces NVRTC compile for this arch
                except Exception as e:
                    failures.append(f"[dev {d}] {name}: {e}")
    if failures:
        print("KERNEL COMPILE FAILURES:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("  all kernels compiled")

    # Smoke-launch the critical wrappers on tiny data (device 0)
    with cp.cuda.Device(0):
        try:
            from et_miner.gpu.kernels import get_popcount_kernel

            bv = cp.zeros((8, 4), dtype=cp.uint64)
            bv[0, 0] = 0b1011
            assert int(get_popcount_kernel()(bv.view(cp.uint64)).sum()) == 3

            from et_miner.gpu.kernels import count_pairs_k2_allcounts

            counts = count_pairs_k2_allcounts(bv, [0, 1, 2], 4)
            assert counts.shape == (3,)

            from et_miner.gpu.kernels.filter import compact_threshold_filter

            arr = cp.asarray(np.array([5, 1, 7, 7, 0], dtype=np.int32))
            idx, cnt = compact_threshold_filter(arr, 5, impl="compact")
            assert idx.tolist() == [0, 2, 3] and cnt.tolist() == [5, 7, 7]

            print("critical-wrapper smoke launches: OK")
        except Exception as e:
            print(f"CRITICAL WRAPPER LAUNCH FAILED: {type(e).__name__}: {e}")
            return 1

    print("selfcheck: READY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
