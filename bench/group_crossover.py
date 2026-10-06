"""The K>=3 kernel sweep of phase C (bench/optimizations/PROTOCOL-C.md, "Step 1: kernel sweep").

Synthetic prefix groups (`kernel_crossover.py`'s correlated random bitvecs over
600 columns, a random prefix of K-2 items and m sorted suffixes per group),
every pair counted (no subset index), 1,000 groups per point so every kernel
launches at least 1,000 blocks. One process per (K, words) line, on device 0:
per point an untimed launch of each kernel, then 3 interleaved reps
(per-candidate, group, tiled, ...); a kernel's value is the median launch time.
The per-candidate kernel is timed up to m = 24, and beyond that only while it
beat the group kernel at the line's previous point. Every timed kernel must
return the same count array.

The table (`table`): per K at 312,500 words, `GROUP_TILED_MIN_PAIRS[k]` where
t_group / t_tiled first crosses 1 going up in m, `GROUP_MIN_PAIRS[k]` where
t_percand / t_group does, each the geometric mean of the two adjacent pair
counts C(m, 2); K=7 from its neighbours, K >= 9 the K=8 value. The 31,250-word
lines must cross within one grid step of the 312,500-word ones.

Rows append to <out> (default: `group_crossover.jsonl` in the runner's
per-revision campaign directory, stamped with the revision); a line with ok
rows at this revision is skipped, so the sweep resumes.

Usage: uv run python bench/group_crossover.py [--out FILE] [--max-gpu-hours H] [--analyze]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

KS = (3, 4, 5, 6, 8)
WORDS = (312_500, 31_250)
TABLE_WORDS = 312_500
SUFFIXES = (2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 48, 64)
PERCAND_UP_TO = 24
N_GROUPS = 1_000
N_COLS = 600
REPS = 3
KERNELS = ("percand", "group", "tiled")
TIMEOUT_S = 1800
ENV = {
    "CUDA_VISIBLE_DEVICES": "0",
    "POLARS_MAX_THREADS": "6",
    "RAYON_NUM_THREADS": "6",
    "MKL_NUM_THREADS": "6",
    "OMP_NUM_THREADS": "6",
}


def lines() -> list[dict]:
    return [{"id": f"K{k}-W{w}", "k": k, "n_u64s": w} for w in WORDS for k in KS]


def _child(line: dict) -> list[dict]:
    import cupy as cp
    from kernel_crossover import _groups, _random_bitvecs

    from et_miner.gpu.kernels import (
        count_group_pairs,
        count_k3plus_per_candidate,
        count_shared_tiled_allcounts,
        upload_k3plus_groups,
    )

    count = {"percand": count_k3plus_per_candidate, "group": count_group_pairs, "tiled": count_shared_tiled_allcounts}
    k, n_u64s = line["k"], line["n_u64s"]
    rng = np.random.default_rng(k * 1_000_003 + n_u64s)
    bitvecs = _random_bitvecs(cp, N_COLS, n_u64s)
    out = []
    percand_ahead = True
    for m in SUFFIXES:
        groups = _groups(k, m, N_GROUPS, N_COLS, rng)
        gpu = upload_k3plus_groups(groups, 0)
        kernels = [kn for kn in KERNELS if kn != "percand" or m <= PERCAND_UP_TO or percand_ahead]

        def launch(kernel, groups=groups, gpu=gpu):
            cp.cuda.Device(0).synchronize()
            t0 = time.perf_counter()
            counts = count[kernel](bitvecs, groups, n_u64s, groups_gpu=gpu)
            cp.cuda.Device(0).synchronize()
            return time.perf_counter() - t0, counts

        arrays = {kn: launch(kn)[1].get() for kn in kernels}
        if any(not np.array_equal(arrays[kn], arrays["tiled"]) for kn in kernels):
            raise RuntimeError(f"K={k} words={n_u64s} m={m}: the kernels disagree")
        times: dict[str, list[float]] = {kn: [] for kn in kernels}
        for _ in range(REPS):
            for kn in kernels:
                times[kn].append(round(launch(kn)[0], 6))
        med = {kn: statistics.median(v) for kn, v in times.items()}
        if "percand" in med:
            percand_ahead = med["percand"] < med["group"]
        out.append(
            {
                "id": f"{line['id']}-m{m}",
                "line": line["id"],
                "k": k,
                "n_u64s": n_u64s,
                "suffixes": m,
                "pairs": m * (m - 1) // 2,
                "groups": N_GROUPS,
                "status": "ok",
                "times_s": times,
                "median_s": med,
            }
        )
        del gpu
        cp.get_default_memory_pool().free_all_blocks()
    return out


def _crossing(points: list[dict], fast: str, slow: str) -> tuple[int | None, list[int]]:
    """The first m where `slow` stops being slower than `fast`, going up in m: (pairs threshold, every crossing's
    grid index). None when `fast` is faster at every point that times both."""
    both = [p for p in sorted(points, key=lambda p: p["suffixes"]) if fast in p["median_s"] and slow in p["median_s"]]
    ahead = [p["median_s"][fast] < p["median_s"][slow] for p in both]
    flips = [i for i in range(1, len(both)) if ahead[i] != ahead[i - 1]]
    if not both or all(ahead):
        return None, flips
    if not ahead[0]:
        return 0, flips  # a later flip back shows up as a crossing in `flips`
    a, b = both[flips[0] - 1], both[flips[0]]
    return round(math.sqrt(a["pairs"] * b["pairs"])), flips


def table(rows: list[dict], n_u64s: int = TABLE_WORDS) -> dict:
    """GROUP_MIN_PAIRS and GROUP_TILED_MIN_PAIRS per K from the ok rows at `n_u64s` words, with the findings."""
    from et_miner.gpu.kernels.group_pairs import GROUP_MAX_SUFFIXES
    from et_miner.gpu.row_split_chunks import TILED_MIN_GROUP_PAIRS

    above_cap = GROUP_MAX_SUFFIXES * (GROUP_MAX_SUFFIXES + 1) // 2
    floor, tiled_from, notes = {}, {}, []
    for k in KS:
        pts = [r for r in rows if r.get("status") == "ok" and r["k"] == k and r["n_u64s"] == n_u64s]
        if not pts:
            continue
        t, t_flips = _crossing(pts, "group", "tiled")
        f, f_flips = _crossing(pts, "percand", "group")
        if len(t_flips) > (0 if t == 0 else 1) or len(f_flips) > (0 if f == 0 else 1):
            notes.append(f"(should) K={k}: more than one crossing (group/tiled {t_flips}, percand/group {f_flips})")
        if t == 0:
            notes.append(f"K={k}: the group kernel never beats the tiled kernel; today's two-way dispatch holds")
            floor[k] = tiled_from[k] = TILED_MIN_GROUP_PAIRS.get(k, TILED_MIN_GROUP_PAIRS[8])
            continue
        tiled_from[k] = above_cap if t is None else t
        floor[k] = tiled_from[k] if f is None else f
        if f is None:
            notes.append(f"K={k}: the per-candidate kernel beats the group kernel at every point that times both")
    for tab in (floor, tiled_from):
        if 6 in tab and 8 in tab:
            tab[7] = round(math.sqrt(tab[6] * tab[8]))
    return {
        "GROUP_MIN_PAIRS": dict(sorted(floor.items())),
        "GROUP_TILED_MIN_PAIRS": dict(sorted(tiled_from.items())),
        "notes": notes,
    }


def word_check(rows: list[dict]) -> list[str]:
    """(must-fix) findings where a 31,250-word crossover lies more than one grid step from the 312,500-word one."""
    out = []
    for k in KS:
        for fast, slow in (("group", "tiled"), ("percand", "group")):
            idx = {}
            for w in WORDS:
                pts = [r for r in rows if r.get("status") == "ok" and r["k"] == k and r["n_u64s"] == w]
                both = [
                    p
                    for p in sorted(pts, key=lambda p: p["suffixes"])
                    if fast in p["median_s"] and slow in p["median_s"]
                ]
                ahead = [p["median_s"][fast] < p["median_s"][slow] for p in both]
                idx[w] = next(
                    (SUFFIXES.index(both[i]["suffixes"]) for i in range(1, len(both)) if ahead[i] != ahead[i - 1]),
                    None if all(ahead) else -1,
                )
            a, b = idx.get(WORDS[0]), idx.get(WORDS[1])
            if (a is None) != (b is None) or (a is not None and b is not None and abs(a - b) > 1):
                out.append(
                    f"(must-fix) K={k} {fast}/{slow}: crossing at grid step {a} ({WORDS[0]} words) "
                    f"vs {b} ({WORDS[1]} words)"
                )
    return out


def _run_line(line: dict, rev: str) -> list[dict]:
    print(f"→ {line['id']}", flush=True)
    t = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, __file__, "--child", json.dumps(line)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            env={**os.environ, **ENV},
        )
        last = proc.stdout.strip().splitlines()[-1:] or [""]
        rows = (
            json.loads(last[0])
            if proc.returncode == 0 and last[0].startswith("[")
            else [
                {
                    "id": line["id"],
                    "line": line["id"],
                    "k": line["k"],
                    "n_u64s": line["n_u64s"],
                    "status": f"error (rc={proc.returncode}): {proc.stderr.strip().splitlines()[-1:]}",
                }
            ]
        )
    except subprocess.TimeoutExpired:
        rows = [{"id": line["id"], "line": line["id"], "k": line["k"], "n_u64s": line["n_u64s"], "status": "timeout"}]
    proc_s = round(time.time() - t, 1)
    return [{**r, "rev": rev, "line_proc_s": proc_s} for r in rows]


def _report(rows: list[dict]) -> None:
    for r in sorted(rows, key=lambda r: (-r["n_u64s"], r["k"], r.get("suffixes", 0))):
        if r.get("status") != "ok":
            print(f"  {r['id']:<20} {r.get('status')}")
            continue
        med = r["median_s"]
        cells = "  ".join(f"{kn}={med[kn] * 1e3:>9.2f} ms" if kn in med else f"{kn}={'—':>12}" for kn in KERNELS)
        print(f"  {r['id']:<20} pairs={r['pairs']:<5} {cells}  group/tiled={med['group'] / med['tiled']:.3f}")
    print(json.dumps(table(rows), indent=1))
    print(json.dumps({"31,250 words": table(rows, WORDS[1])}, indent=1))
    for finding in word_check(rows):
        print(finding)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-gpu-hours", type=float, default=None)
    ap.add_argument("--analyze", action="store_true", help="only print the analysis of the rows in --out")
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        print(json.dumps(_child(json.loads(args.child))))
        return 0

    from runner import _campaign_out, _git_rev

    here = _git_rev()
    if here == "unknown":
        print("cannot determine the revision; fix git and re-run")
        return 2
    out = Path(args.out) if args.out else _campaign_out(here) / "group_crossover.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
    if not args.analyze:
        gpu_s = 0.0
        for line in lines():
            if any(r["line"] == line["id"] and r.get("status") == "ok" and r.get("rev") == here for r in rows):
                continue
            if args.max_gpu_hours is not None and gpu_s > args.max_gpu_hours * 3600:
                print(f"max-gpu-hours reached ({gpu_s / 3600:.2f}) — stopping (resume with the same command)")
                return 1
            new = _run_line(line, here)
            gpu_s += new[0]["line_proc_s"]
            rows += new
            with out.open("a") as f:
                f.writelines(json.dumps(r) + "\n" for r in new)
            print(f"   {new[0]['status']} ({new[0]['line_proc_s']} s)", flush=True)
        print(f"GPU-hours this invocation: {gpu_s / 3600:.3f}")
    _report([r for r in rows if r.get("rev") == here])
    return 0


if __name__ == "__main__":
    sys.exit(main())
