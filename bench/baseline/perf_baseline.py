#!/usr/bin/env python3
"""Performance baseline for the BUGS_FOUND remediation programme (PR 0b).

At least five queued fixes are explicitly slower:

  #4  int32 -> float64 for MKL doubles the value array in the matmul
  #13 taking the batch MINIMUM length filters fewer transactions
  #18 honouring n_jobs=1 gives up rayon's every-core default
  #22 the anchor filter becomes emit-only, so Phase 2 mines the full lattice
  #24 any result-buffer overflow now raises instead of completing

The #22 warning -- "anyone who reverts this when Phase 2 slows down has
restored a 99.3% silent data loss to fix a performance problem" -- is only
checkable against a measured before. This produces that before, and the same
matrix re-run after each PR produces the delta.

Each config runs in a fresh process via bench/child_run.py (fresh CUDA
context, context-free nvidia-smi VRAM peak, ru_maxrss RSS peak, per-K
timings from level_callback).

    uv run python bench/baseline/perf_baseline.py --out bench/baseline/perf-baseline.json
    uv run python bench/baseline/perf_baseline.py --compare bench/baseline/perf-baseline.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def matrix() -> list[dict]:
    """Fixed configs, one per code path a slow fix touches."""
    return [
        # --- GPU row-split: the #22 / #24 surface -------------------------
        {"id": "smoke-gpu1", "preset": "smoke", "n_gpus": 1},
        {"id": "smoke-gpu2", "preset": "smoke", "n_gpus": 2},
        {"id": "deepk-gpu2", "preset": "deep_k", "n_gpus": 2},
        {"id": "deepk-gpu2-sparse", "preset": "deep_k", "n_gpus": 2, "sparse_from_k": 3},
        # deeper lattice: where a per-level regression compounds
        {"id": "deepk-gpu2-low", "preset": "deep_k", "n_gpus": 2, "min_support": 0.005},
        # mine_two_phase with DISTINCT supports -- the only configuration in
        # which the anchor filter does anything, and therefore the only one
        # that measures what #22 costs.
        {"id": "twophase-deepk", "preset": "deep_k", "two_phase": True, "n_gpus": 2,
         "min_support": 0.02, "phase2_support": 0.005, "max_length": 5},
        # --- CPU tier: the #4 / #13 / #18 surface -------------------------
        {"id": "smoke-cpu-polars", "preset": "smoke", "route": "cpu", "sparse": False},
        {"id": "smoke-cpu-sparse-j1", "preset": "smoke", "route": "cpu", "sparse": True, "n_jobs": 1},
        {"id": "smoke-cpu-sparse-j8", "preset": "smoke", "route": "cpu", "sparse": True, "n_jobs": 8},
        # deep_k on the CPU sparse path is where the MKL matmul actually bites
        {"id": "deepk-cpu-sparse-j8", "preset": "deep_k", "route": "cpu", "sparse": True,
         "n_jobs": 8, "max_length": 4},
        # stress_k2: 2M rows x 35k items, so the k=2 sparse matmul dominates --
        # that is the operation #4 changes from float32 to float64. min_support
        # is raised well above the preset default (1.5e-5, ~612M candidate
        # pairs) so the config stays matmul-bound but finishes in minutes: this
        # matrix is re-run after every PR, so an hours-long config is useless
        # for the delta it exists to produce.
        {"id": "stressk2-cpu-sparse", "preset": "stress_k2", "route": "cpu", "sparse": True,
         "n_jobs": 8, "max_length": 2, "min_support": 0.002},
    ]


def run_one(cfg: dict, timeout_s: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "result.json"
        payload = {**cfg, "result_path": str(out)}
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(REPO / "bench" / "child_run.py"), json.dumps(payload)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=REPO,
        )
        elapsed = time.perf_counter() - t0
        if out.exists():
            rec = json.loads(out.read_text())
        else:
            rec = {
                "id": cfg["id"],
                "config": cfg,
                "status": f"no result file (rc={proc.returncode})",
                "stderr_tail": proc.stderr[-2000:],
            }
        rec["parent_wall_s"] = round(elapsed, 3)
        return rec


def summarize(rec: dict) -> str:
    if rec.get("status") != "ok":
        return f"  {rec['id']:24} FAILED  {str(rec.get('status'))[:70]}"
    vram = max(rec.get("peak_vram_mb", {}).values(), default=0)
    return (
        f"  {rec['id']:24} {rec['wall_s']:>8.2f}s  "
        f"itemsets={rec.get('n_itemsets', 0):>9,}  "
        f"rss={rec.get('peak_rss_mb', 0):>8.0f}MB  vram={vram:>6}MB  "
        f"K={len(rec.get('levels', []))}"
    )


def compare(base: list[dict], now: list[dict]) -> int:
    by_id = {r["id"]: r for r in base}
    print(f"\n{'config':24} {'before':>10} {'after':>10} {'delta':>9}   itemsets")
    worst = 0.0
    for rec in now:
        old = by_id.get(rec["id"])
        if not old or old.get("status") != "ok" or rec.get("status") != "ok":
            print(f"{rec['id']:24}  (not comparable)")
            continue
        b, a = old["wall_s"], rec["wall_s"]
        pct = (a - b) / b * 100 if b else 0.0
        worst = max(worst, pct)
        same = old.get("itemset_hash") == rec.get("itemset_hash")
        mark = "same" if same else f"CHANGED {old.get('n_itemsets')}->{rec.get('n_itemsets')}"
        print(f"{rec['id']:24} {b:>9.2f}s {a:>9.2f}s {pct:>+8.1f}%   {mark}")
    print(f"\nworst regression: {worst:+.1f}%")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, help="write results JSON here")
    ap.add_argument("--compare", type=Path, help="compare against a previous results JSON")
    ap.add_argument("--timeout", type=int, default=1800, help="per-config timeout (s)")
    ap.add_argument("--only", action="append", help="run only these config ids")
    args = ap.parse_args()

    cfgs = matrix()
    if args.only:
        cfgs = [c for c in cfgs if c["id"] in set(args.only)]

    results = []
    for cfg in cfgs:
        print(f"running {cfg['id']} ...", flush=True)
        try:
            rec = run_one(cfg, args.timeout)
        except subprocess.TimeoutExpired:
            rec = {"id": cfg["id"], "config": cfg, "status": f"timeout after {args.timeout}s"}
        results.append(rec)
        print(summarize(rec), flush=True)

    if args.out:
        args.out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    if args.compare:
        return compare(json.loads(args.compare.read_text()), results)
    return 0 if all(r.get("status") == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
