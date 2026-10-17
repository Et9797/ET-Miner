"""Instruction-rate probes for the GPU counting kernels (moonshots Phase 0).

Standalone CUDA probes compiled with cupy.RawKernel; none is a registered
kernel. On every visible device it measures:

- ``popc32`` / ``popc64``: ``__popc`` / ``__popcll`` results per second, on
  register operands;
- ``lop3``: 32-bit three-input logic ops per second;
- ``tiled_inner``: the tiled kernel's inner loop (``shared_tiled.cu``: the
  staged prefix AND, one suffix row per lane, four register-blocked pairs per
  thread) on dense shared-memory tiles, in AND+popcount slot-words per second;
- ``bmma_m16n8k256`` / ``bmma_m8n8k128``: ``mma.sync ... .b1.b1.s32.and.popc``
  on register operands (sm_80+), in slot-words per second (one slot-word is one
  pair over 64 transactions, 64 bit-MACs);
- ``bmma_exact``: the m16n8k256 tile product against a ``__popcll`` reference on
  random and all-ones tiles, word counts not a multiple of 4 padded with zero;
- ``kernel_tiled`` / ``kernel_group`` / ``kernel_percand``: the registered
  counting kernels on synthetic groups (``KERNEL_SHAPES``) over zero, sparse and
  dense bitvectors: staging alone against staging plus counting;
- ``copy``: device-to-device copy bandwidth (``bench/copy_bandwidth.py``);
- ``h2d_pageable`` / ``h2d_pinned``: host-to-device copy bandwidth.

Each rate is the median of ``--reps`` timed launches after one untimed launch,
with the SM clock read from NVML after the run. One JSON line per (device,
probe).

Usage: uv run python bench/moonshots/microbench.py [--out FILE] [--reps R]
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

_SRC = r"""
extern "C" __global__ void mb_popc32(const unsigned int* seed, unsigned int* out, const int iters) {
    unsigned int v[16];
    unsigned int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = 0; j < 16; j++) v[j] = seed[(t + 7 * j) & 1023];
    for (int it = 0; it < iters; it++) {
        #pragma unroll
        for (int j = 0; j < 16; j += 4) {
            acc0 += __popc(v[j]); acc1 += __popc(v[j + 1]); acc2 += __popc(v[j + 2]); acc3 += __popc(v[j + 3]);
        }
        #pragma unroll
        for (int j = 0; j < 16; j++) v[j] += 0x9E3779B9u;
    }
    out[t] = acc0 + acc1 + acc2 + acc3;
}

extern "C" __global__ void mb_popc64(const unsigned long long* seed, unsigned int* out, const int iters) {
    unsigned long long v[16];
    unsigned int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = 0; j < 16; j++) v[j] = seed[(t + 7 * j) & 1023];
    for (int it = 0; it < iters; it++) {
        #pragma unroll
        for (int j = 0; j < 16; j += 4) {
            acc0 += __popcll(v[j]); acc1 += __popcll(v[j + 1]); acc2 += __popcll(v[j + 2]); acc3 += __popcll(v[j + 3]);
        }
        #pragma unroll
        for (int j = 0; j < 16; j++) v[j] ^= (v[j] << 1);
    }
    out[t] = acc0 + acc1 + acc2 + acc3;
}

extern "C" __global__ void mb_lop3(const unsigned int* seed, unsigned int* out, const int iters) {
    unsigned int v[16];
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = 0; j < 16; j++) v[j] = seed[(t + 7 * j) & 1023];
    for (int it = 0; it < iters; it++) {
        #pragma unroll
        for (int j = 0; j < 16; j++) v[j] = (v[j] & v[(j + 1) & 15]) ^ v[(j + 5) & 15];
    }
    unsigned int r = 0;
    for (int j = 0; j < 16; j++) r ^= v[j];
    out[t] = r;
}

// The inner loop of shared_tiled.cu on one staged 32-word tile, repeated.
extern "C" __global__ void mb_tiled_inner(const unsigned long long* seed, unsigned int* out, const int iters) {
    __shared__ unsigned long long s_pa[33];
    __shared__ unsigned long long s_sufA[32][33];
    __shared__ unsigned long long s_sufB[32][33];
    for (int i = threadIdx.x; i < 32 * 33; i += blockDim.x) {
        s_sufA[i / 33][i % 33] = seed[(blockIdx.x * 7 + i) & 4095] | 1ULL;
        s_sufB[i / 33][i % 33] = seed[(blockIdx.x * 13 + i + 2048) & 4095] | 1ULL;
    }
    if (threadIdx.x < 33) s_pa[threadIdx.x] = ~0ULL;
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    for (int it = 0; it < iters; it++) {
        for (int w = 0; w < 32; w++) {
            const unsigned long long pb = s_pa[w] & s_sufB[lane][w];
            if (pb == 0ULL) continue;
            acc0 += (unsigned int)__popcll(pb & s_sufA[warp][w]);
            acc1 += (unsigned int)__popcll(pb & s_sufA[warp + 8][w]);
            acc2 += (unsigned int)__popcll(pb & s_sufA[warp + 16][w]);
            acc3 += (unsigned int)__popcll(pb & s_sufA[warp + 24][w]);
        }
        __syncthreads();
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc0 + acc1 + acc2 + acc3;
}

"""

_SRC_BMMA = r"""
#define BMMA16(c, a, b)                                                                              \
    asm volatile("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "                        \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"                              \
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])                                    \
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]))

#define BMMA8(c, a, b)                                                                               \
    asm volatile("mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc "                         \
                 "{%0,%1}, {%2}, {%3}, {%0,%1};"                                                     \
                 : "+r"(c[0]), "+r"(c[1]) : "r"(a), "r"(b))

// Independent accumulators so the rate is throughput, not latency.
extern "C" __global__ void mb_bmma16(const unsigned int* seed, int* out, const int iters) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int a[4], b[2];
    for (int j = 0; j < 4; j++) a[j] = seed[(t + j) & 1023];
    for (int j = 0; j < 2; j++) b[j] = seed[(t + 17 * j + 5) & 1023];
    int c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0}, c2[4] = {0, 0, 0, 0}, c3[4] = {0, 0, 0, 0};
    for (int it = 0; it < iters; it++) {
        BMMA16(c0, a, b); BMMA16(c1, a, b); BMMA16(c2, a, b); BMMA16(c3, a, b);
    }
    out[t] = c0[0] + c1[1] + c2[2] + c3[3];
}

extern "C" __global__ void mb_bmma8(const unsigned int* seed, int* out, const int iters) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int a = seed[t & 1023], b = seed[(t + 5) & 1023];
    int c0[2] = {0, 0}, c1[2] = {0, 0}, c2[2] = {0, 0}, c3[2] = {0, 0};
    for (int it = 0; it < iters; it++) {
        BMMA8(c0, a, b); BMMA8(c1, a, b); BMMA8(c2, a, b); BMMA8(c3, a, b);
    }
    out[t] = c0[0] + c1[1] + c2[0] + c3[1];
}

// One warp: C[16][8] = popcount(A[r] & B[c]) over n_words 64-bit words (rows of
// A and B are n_words long, padded on the host to a multiple of 4 with zeros).
// Fragments per the PTX ISA m16n8k256 .b1 layout: g = lane >> 2, q = lane & 3;
// a0/a2 row g, a1/a3 row g + 8, b0/b1 column g; a0, a1, b0 hold 32-bit word q
// of the 256-bit chunk and a2, a3, b1 word q + 4.
extern "C" __global__ void mb_bmma16_tile(const unsigned int* A, const unsigned int* B, const int n_chunks, int* C) {
    const int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
    const int row_u32 = n_chunks * 8;
    int c[4] = {0, 0, 0, 0};
    for (int ch = 0; ch < n_chunks; ch++) {
        unsigned int a[4], b[2];
        a[0] = A[g * row_u32 + ch * 8 + q];
        a[1] = A[(g + 8) * row_u32 + ch * 8 + q];
        a[2] = A[g * row_u32 + ch * 8 + q + 4];
        a[3] = A[(g + 8) * row_u32 + ch * 8 + q + 4];
        b[0] = B[g * row_u32 + ch * 8 + q];
        b[1] = B[g * row_u32 + ch * 8 + q + 4];
        BMMA16(c, a, b);
    }
    C[g * 8 + 2 * q] = c[0];
    C[g * 8 + 2 * q + 1] = c[1];
    C[(g + 8) * 8 + 2 * q] = c[2];
    C[(g + 8) * 8 + 2 * q + 1] = c[3];
}
"""

THREADS = 256


def _sm_clock_mhz(device_id: int) -> tuple[int | None, int | None, str | None]:
    """(current SM clock, max SM clock, throttle reasons) from nvidia-smi, or Nones."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_id}",
                "--query-gpu=clocks.sm,clocks.max.sm,clocks_throttle_reasons.active",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        cur, mx, reasons = (x.strip() for x in out.split(","))
        return int(cur), int(mx), reasons
    except Exception:
        return None, None, None


def _time_launch(cp, fn, reps: int) -> float:
    """Median seconds of ``reps`` timed calls of ``fn`` after one untimed call."""
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    fn()
    cp.cuda.Device().synchronize()
    times = []
    for _ in range(reps):
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, stop) / 1000)
    return statistics.median(times)


def _rate_probes(cp, mod, bmma, n_sm: int, reps: int) -> list[dict]:
    """The register and shared-memory rate probes: one dict per probe."""
    rng = np.random.default_rng(1)
    seed32 = cp.asarray(rng.integers(0, 2**32, size=1024, dtype=np.uint64).astype(np.uint32))
    seed64 = cp.asarray(rng.integers(0, 2**63, size=4096, dtype=np.uint64) | np.uint64(1 << 63))
    blocks = n_sm * 16
    n_threads = blocks * THREADS
    out = cp.empty(n_threads, dtype=cp.uint32)
    probes = [
        # name, kernel, seed, iters, units per thread per iteration, unit
        ("popc32", "mb_popc32", seed32, 4096, 16, "popc32/s"),
        ("popc64", "mb_popc64", seed64, 2048, 16, "popc64/s"),
        ("lop3", "mb_lop3", seed32, 4096, 16, "lop3/s"),
        ("bmma_m16n8k256", "mb_bmma16", seed32, 4096, 4 * 16 * 8 * 4 / 32, "slot-words/s"),
        ("bmma_m8n8k128", "mb_bmma8", seed32, 4096, 4 * 8 * 8 * 2 / 32, "slot-words/s"),
    ]
    rows = []
    for name, kname, seed, iters, units, unit in probes:
        if kname.startswith("mb_bmma") and bmma is None:
            continue
        kernel = (bmma if kname.startswith("mb_bmma") else mod).get_function(kname)
        try:
            t = _time_launch(cp, lambda: kernel((blocks,), (THREADS,), (seed, out, np.int32(iters))), reps)
        except Exception as exc:  # an unsupported instruction fails to compile or launch
            rows.append({"probe": name, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append({"probe": name, "status": "ok", "seconds": t, "rate": n_threads * iters * units / t, "unit": unit})
    # tiled_inner: per block per iteration 32 words x 1,024 slots
    kernel = mod.get_function("mb_tiled_inner")
    tblocks, iters = n_sm * 8, 2048
    tout = cp.empty(tblocks * THREADS, dtype=cp.uint32)
    t = _time_launch(cp, lambda: kernel((tblocks,), (THREADS,), (seed64, tout, np.int32(iters))), reps)
    rows.append(
        {"probe": "tiled_inner", "status": "ok", "seconds": t, "rate": tblocks * iters * 32 * 1024 / t,
         "unit": "slot-words/s"}
    )
    return rows


def _bmma_exact(cp, mod) -> dict:
    """m16n8k256 tile products against np.bitwise_count on random, all-ones and ragged tiles."""
    rng = np.random.default_rng(7)
    kernel = mod.get_function("mb_bmma16_tile")
    cases = []
    for n_words in (1, 2, 3, 4, 5, 7, 8, 63, 64, 65, 129, 1001):
        for fill in ("random", "ones", "sparse"):
            if fill == "random":
                a = rng.integers(0, 2**64, size=(16, n_words), dtype=np.uint64)
                b = rng.integers(0, 2**64, size=(8, n_words), dtype=np.uint64)
            elif fill == "ones":
                a = np.full((16, n_words), np.uint64(2**64 - 1))
                b = np.full((8, n_words), np.uint64(2**64 - 1))
            else:
                a = rng.integers(0, 2**64, size=(16, n_words), dtype=np.uint64) & rng.integers(
                    0, 2**64, size=(16, n_words), dtype=np.uint64
                )
                b = rng.integers(0, 2**64, size=(8, n_words), dtype=np.uint64) & rng.integers(
                    0, 2**64, size=(8, n_words), dtype=np.uint64
                )
            want = np.bitwise_count(a[:, None, :] & b[None, :, :]).sum(axis=2).astype(np.int64)
            n_chunks = -(-n_words // 4)
            pad = n_chunks * 4 - n_words
            ap = np.pad(a, ((0, 0), (0, pad))).view(np.uint32)
            bp = np.pad(b, ((0, 0), (0, pad))).view(np.uint32)
            c = cp.zeros((16, 8), dtype=cp.int32)
            kernel((1,), (32,), (cp.asarray(ap), cp.asarray(bp), np.int32(n_chunks), c))
            got = cp.asnumpy(c).astype(np.int64)
            cases.append({"n_words": n_words, "fill": fill, "equal": bool(np.array_equal(got, want))})
    return {"probe": "bmma_exact", "status": "ok", "cases": len(cases),
            "all_equal": all(c["equal"] for c in cases), "failures": [c for c in cases if not c["equal"]]}


def _groups(rng, k: int, m: int, n_groups: int, n_cols: int):
    from et_miner.gpu.kernels import K3PlusGroups

    p = k - 2
    pairs = m * (m - 1) // 2
    return K3PlusGroups(
        prefix_items=rng.integers(0, n_cols, size=n_groups * p).astype(np.int32),
        prefix_offsets=np.arange(0, (n_groups + 1) * p, p, dtype=np.int64),
        suffixes=np.concatenate([np.sort(rng.choice(n_cols, size=m, replace=False)) for _ in range(n_groups)]).astype(
            np.int32
        ),
        suffix_offsets=np.arange(0, (n_groups + 1) * m, m, dtype=np.int64),
        cumulative_pairs=np.arange(0, (n_groups + 1) * pairs, pairs, dtype=np.int64),
        total_candidates=n_groups * pairs,
        groups=None,
    )


#: (kernel, K, suffixes per group, groups, words): the tiled kernel at sk2ml3's K=3 shape (3 tile-pairs per
#: group), the group and per-candidate kernels at a deep dsl-like shape.
KERNEL_SHAPES = (("tiled", 3, 64, 1_000, 31_250), ("group", 8, 16, 1_000, 312_500), ("percand", 8, 4, 1_000, 312_500))


def _kernels(cp, reps: int) -> list[dict]:
    """The registered counting kernels on synthetic groups over zero, sparse and dense bitvectors.

    Zero bitvectors leave only the staging (and the per-candidate kernel's first
    row); dense ones make every slot reach the popcount; sparse ones (1e-3 of
    the words non-zero) sit between. Times per launch and per unit of work:
    the tiled kernel per tile-pair per word, the others per counted pair per word.
    """
    from et_miner.gpu.kernels import (
        count_group_pairs,
        count_k3plus_per_candidate,
        count_shared_tiled_allcounts,
        upload_k3plus_groups,
    )

    count = {"tiled": count_shared_tiled_allcounts, "group": count_group_pairs, "percand": count_k3plus_per_candidate}
    n_cols = 600
    rng = np.random.default_rng(3)
    rows = []
    for kernel, k, m, n_groups, n_u64s in KERNEL_SHAPES:
        groups = _groups(rng, k, m, n_groups, n_cols)
        units = n_groups * (3 if kernel == "tiled" else m * (m - 1) // 2)
        for fill in ("zero", "sparse", "dense"):
            if fill == "zero":
                bv = cp.zeros((n_cols, n_u64s), dtype=cp.uint64)
            else:
                host = rng.integers(0, 2**64, size=(n_cols, n_u64s), dtype=np.uint64)
                if fill == "sparse":
                    host &= np.where(rng.random((n_cols, n_u64s)) < 1e-3, np.uint64(2**64 - 1), np.uint64(0))
                else:
                    host |= np.uint64(1)
                bv = cp.asarray(host)
                del host
            ggpu = upload_k3plus_groups(groups, bv.device.id)
            t = _time_launch(cp, functools.partial(count[kernel], bv, groups, n_u64s, groups_gpu=ggpu), reps)
            row = {"probe": f"kernel_{kernel}", "fill": fill, "k": k, "suffixes": m, "words": n_u64s,
                   "status": "ok", "seconds": t, "ns_per_unit_word": t / (units * n_u64s) * 1e9,
                   "unit": "tile-pair" if kernel == "tiled" else "pair"}
            if kernel == "tiled":
                row["staged_gbps"] = units * n_u64s * ((k - 2) + 2 * 32) * 8 / t / 1e9
            rows.append(row)
            del bv, ggpu
            cp.get_default_memory_pool().free_all_blocks()
    return rows


def _h2d(cp, reps: int, nbytes: int = 1 << 28) -> list[dict]:
    """Host-to-device copy bandwidth from pageable and from pinned host memory."""
    import time

    host = np.ones(nbytes, dtype=np.uint8)
    dev = cp.empty(nbytes, dtype=cp.uint8)
    pinned_mem = cp.cuda.alloc_pinned_memory(nbytes)
    pinned = np.frombuffer(pinned_mem, dtype=np.uint8, count=nbytes)
    pinned[:] = 1
    rows = []
    for name, src in (("h2d_pageable", host), ("h2d_pinned", pinned)):
        times = []
        for i in range(reps + 1):
            cp.cuda.Device().synchronize()
            t0 = time.perf_counter()
            dev.set(src)
            cp.cuda.Device().synchronize()
            if i:
                times.append(time.perf_counter() - t0)
        rows.append({"probe": name, "status": "ok", "rate": nbytes / statistics.median(times), "unit": "B/s"})
    del dev, pinned, pinned_mem
    cp.get_default_memory_pool().free_all_blocks()
    return rows


def run_device(device_id: int, reps: int) -> list[dict]:
    import cupy as cp

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from copy_bandwidth import measure

    with cp.cuda.Device(device_id):
        attrs = cp.cuda.Device(device_id).attributes
        n_sm = int(attrs["MultiProcessorCount"])
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        head = {
            "device": device_id,
            "name": props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
            "cc": f"{props['major']}.{props['minor']}",
            "n_sm": n_sm,
        }
        mod = cp.RawModule(code=_SRC)
        try:
            bmma = cp.RawModule(code=_SRC_BMMA)
            bmma.get_function("mb_bmma16")
        except Exception as exc:  # b1 mma needs sm_80+ and an NVRTC that emits it
            bmma = None
            rows = [{"probe": "bmma_compile", "status": "error", "error": f"{type(exc).__name__}: {exc}"}]
        else:
            rows = []
        rows += _rate_probes(cp, mod, bmma, n_sm, reps)
        clk, clk_max, reasons = _sm_clock_mhz(device_id)
        try:
            if bmma is not None:
                rows.append(_bmma_exact(cp, bmma))
        except Exception as exc:
            rows.append({"probe": "bmma_exact", "status": "error", "error": f"{type(exc).__name__}: {exc}"})
        rows.extend(_kernels(cp, reps))
        rows.append({"probe": "copy", "status": "ok", "rate": measure(device_id, 1 << 30, reps) * 1e9, "unit": "B/s"})
        rows.extend(_h2d(cp, reps))
    for r in rows:
        r.update(head, sm_clock_mhz=clk, sm_clock_max_mhz=clk_max, throttle_reasons=reasons)
        if clk and r.get("unit", "").endswith("/s") and r.get("unit") != "B/s":
            r["per_clock_per_sm"] = r["rate"] / (n_sm * clk * 1e6)
    return rows


def main() -> int:
    import cupy as cp

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    rows = []
    for d in range(cp.cuda.runtime.getDeviceCount()):
        rows.extend(run_device(d, args.reps))
    text = "\n".join(json.dumps(r) for r in rows)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
