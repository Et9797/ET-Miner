"""On-device preflight: fail in minute one, not hour two.

Compiles EVERY kernel registered in gpu/kernels/loader.py on every device
(NVRTC compiles per compute capability — this is where an sm_86
incompatibility would surface), launches each of them on tiny data with a
known answer, and prints the device/NCCL/P2P//dev/shm/rust matrix.

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
        if any("Failed to find CUDA headers" in f for f in failures):
            print("REMEDY: uv pip install 'cupy-cuda12x[ctk]'   # CUDA header wheels (runtime images ship none)")
            print("    or: export CUDA_PATH=/usr/local/cuda     # when the image has a toolkit")
        return 1
    print("  all kernels compiled")

    # Smoke-launch every registered kernel on tiny data with a known answer,
    # on every device (the loader compiles per device).
    for d in range(n_dev):
        with cp.cuda.Device(d):
            try:
                _smoke_launch(cp)
            except Exception as e:
                print(f"[dev {d}] KERNEL LAUNCH FAILED: {type(e).__name__}: {e}")
                return 1
    print(f"  every kernel launched with the expected answer on {n_dev} device(s)")

    print("selfcheck: READY")
    return 0


def _smoke_launch(cp) -> None:
    """One launch of each registered kernel (and the popcount ElementwiseKernel)."""
    from et_miner.gpu.csr_bitvec import build_bitvecs_gpu
    from et_miner.gpu.kernels import (
        K3PlusGroups,
        count_itemsets_cuda,
        count_csr_range,
        count_csr_gather,
        write_csr_gather,
        count_k3plus_per_candidate,
        count_pairs_k2_per_candidate,
        count_pairs_k2_rows,
        count_pairs_k2_shared,
        count_shared_tiled_allcounts,
        count_tiled_fused,
        get_popcount_kernel,
        upload_k2_rows,
    )
    from et_miner.gpu.kernels import k2 as k2_mod
    from et_miner.gpu.kernels.loader import _KERNEL_FILES

    launched = set()
    # csr_to_bitvec: rows {0, 2}, {1}, {0, 1, 2}, {} over 3 columns
    bv = build_bitvecs_gpu(np.array([0, 2, 3, 6, 6]), np.array([0, 2, 1, 0, 1, 2]), 4, 3,
                           device_id=cp.cuda.Device().id)
    assert bv[:, 0].get().tolist() == [0b0101, 0b0110, 0b0101]
    launched.add("csr_to_bitvec")
    assert int(get_popcount_kernel()(bv.view(cp.uint64)).sum()) == 6

    full = cp.full((4, 4), 0xFFFFFFFFFFFFFFFF, dtype=cp.uint64)  # every row in every column: counts are 256
    assert count_pairs_k2_per_candidate(full, [0, 1, 2], 4).tolist() == [256] * 3
    launched.add("count_pairs_k2_dense")
    assert count_pairs_k2_shared(full, [0, 1, 2], 4).tolist() == [256] * 3
    launched.add("count_shared_tiled_dense")
    # the same rows as frequent positions: pairs (0,1), (0,2), (1,2) occur 1, 2, 1 times
    rows = upload_k2_rows(np.array([0, 2, 3, 6, 6]), np.array([0, 2, 1, 0, 1, 2]), cp.cuda.Device().id)
    assert count_pairs_k2_rows(rows, 0, 3).tolist() == [1, 2, 1]
    launched.add("count_pairs_k2_rows_shared")
    shared_pairs, k2_mod.K2_ROWS_SHARED_PAIRS = k2_mod.K2_ROWS_SHARED_PAIRS, 0
    try:
        assert count_pairs_k2_rows(rows, 1, 2).tolist() == [2, 1]
    finally:
        k2_mod.K2_ROWS_SHARED_PAIRS = shared_pairs
    launched.add("count_pairs_k2_rows")
    groups = K3PlusGroups(np.array([0], np.int32), np.array([0, 1], np.int64), np.array([1, 2, 3], np.int32),
                          np.array([0, 3], np.int64), np.array([0, 3], np.int64), 3, None)
    assert count_k3plus_per_candidate(full, groups, 4).tolist() == [256] * 3
    launched.add("count_k3plus_dense")
    assert count_shared_tiled_allcounts(full, groups, 4).tolist() == [256] * 3
    idx, cnt = count_tiled_fused(full, groups, 4, 1)
    assert idx.tolist() == [0, 1, 2] and cnt.tolist() == [256] * 3
    launched.add("count_shared_tiled_fused")
    assert count_itemsets_cuda(full, [np.array([0, 1], np.int32), np.array([1, 2, 3], np.int32)]).tolist() == [256, 256]
    launched.add("count_itemsets_batch")
    # The (k-1)-subset test: candidates 0+{1,2}, 0+{1,3}, 0+{2,3} against a level
    # without {1,3} (one candidate skipped), without any suffix pair (the tile is
    # skipped), and with {1,2} not free (that candidate's count is inferred).
    from et_miner.gpu.kernels import SUBSET_INFER, SUBSET_PRUNE, upload_subset_index

    dev = cp.cuda.Device().id
    pairs = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]], np.int32)
    partial = upload_subset_index(pairs, [256] * 5, mode=SUBSET_PRUNE, device_id=dev, write_inferred=True)
    bare = upload_subset_index(pairs[:3], [256] * 3, mode=SUBSET_PRUNE, device_id=dev, write_inferred=True)
    infer = upload_subset_index(np.vstack([pairs, [[1, 3]]])[[0, 1, 2, 3, 5, 4]], [256, 256, 256, 100, 256, 256],
                                [True, True, True, False, True, True], mode=SUBSET_PRUNE | SUBSET_INFER,
                                device_id=dev, write_inferred=True)
    assert count_k3plus_per_candidate(full, groups, 4, index=partial).tolist() == [256, 0, 256]
    assert count_k3plus_per_candidate(full, groups, 4, index=infer).tolist() == [100, 256, 256]
    assert count_shared_tiled_allcounts(full, groups, 4, index=bare).tolist() == [0, 0, 0]
    assert count_tiled_fused(full, groups, 4, 1, index=bare)[0].tolist() == []
    # The sparse parents are {0,1}, {0,2}, {0,3}; all candidate
    # intersections contain only transaction 0.
    did = cp.cuda.Device().id
    from et_miner.gpu.kernels import build_k3plus_groups_from_flat, get_cuda_kernel, upload_k3plus_groups

    sparse_groups = build_k3plus_groups_from_flat(np.array([[0, 1], [0, 2], [0, 3]], np.int32), with_src_rows=True)
    uploaded = upload_k3plus_groups(sparse_groups, did, with_src_rows=True)
    offsets = cp.array([0, 2, 4, 6], dtype=cp.int64)
    tids = cp.array([0, 1, 0, 2, 0, 3], dtype=cp.int32)
    assert count_csr_range(offsets, tids, uploaded, 0, 3).tolist() == [1, 1, 1]
    assert count_csr_range(offsets, tids, uploaded, 0, 3, index=partial).tolist() == [1, 0, 1]
    launched.add("csr_count_range")
    ids = cp.array([2, 0, 1], dtype=cp.int64)
    assert count_csr_gather(offsets, tids, uploaded, ids).tolist() == [1, 1, 1]
    launched.add("csr_count_gather")
    out_offsets = cp.arange(4, dtype=cp.int64)
    out_tids = cp.empty(3, dtype=cp.int32)
    write_csr_gather(offsets, tids, uploaded, ids, out_offsets, out_tids)
    assert out_tids.tolist() == [0, 0, 0]
    launched.add("csr_write_gather")
    extracted = cp.empty(6, dtype=cp.int32)
    get_cuda_kernel("bitvec_extract_tids")(
        (1,), (256,), (bv, offsets, extracted, np.int64(3), np.int64(1), np.int64(0))
    )
    assert extracted.tolist() == [0, 2, 1, 2, 0, 2]
    launched.add("bitvec_extract_tids")
    missing = set(_KERNEL_FILES) - launched
    assert not missing, f"registered kernels this selfcheck never launches: {sorted(missing)}"


if __name__ == "__main__":
    sys.exit(main())
