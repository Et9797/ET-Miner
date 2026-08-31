#!/usr/bin/env python3
"""Experiment: Direct GPU vs SON streaming — controlled triplicate comparison.

Runs both Direct GPU and SON streaming at identical support threshold,
multiple times (default 3), to isolate the method effect with proper
statistics. Both methods are run LIVE (no hardcoded reference values).

Previous single-run result: 21.4x speedup, 95.2% SON miss rate.
This experiment: triplicate runs with mean ± std for both methods.
"""

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from et_miner import apriori
from et_miner.streaming import apriori_streaming

from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_transactions, k_distribution


def main():
    parser = argparse.ArgumentParser(
        description="Direct GPU vs SON comparison at identical support (triplicate)",
    )
    parser.add_argument(
        "--data", default="/workspace/data/transactions_35k.parquet",
        help="Path to transactions parquet (default: %(default)s)",
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of runs per method (default: %(default)s)",
    )
    parser.add_argument(
        "--min-support", type=float, default=0.00001,
        help="Support threshold (default: 0.00001 = 0.001%%)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=40_000_000,
        help="SON chunk size (default: %(default)s)",
    )
    parser.add_argument(
        "--local-support-factor", type=float, default=0.9,
        help="SON local support factor (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", default="results_35k",
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--n-gpus", type=int, default=1,
        help="Number of GPUs for row-split mining (default: %(default)s)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
    )
    args = parser.parse_args()

    # Load and filter
    df = load_transactions(args.data, min_items=2)
    n_transactions = len(df)
    min_support = args.min_support
    min_count = math.ceil(min_support * n_transactions)

    logger.info(f"\n{'='*72}")
    logger.info(f"Experiment: Direct GPU vs SON (Triplicate)")
    logger.info(f"{'='*72}")
    logger.info(f"  min_support:     {min_support} ({min_support*100:.3f}%)")
    logger.info(f"  min_count:       {min_count:,}")
    logger.info(f"  n_transactions:  {n_transactions:,}")
    logger.info(f"  runs per method: {args.runs}")
    logger.info(f"  SON chunk_size:  {args.chunk_size:,}")
    logger.info(f"  SON local_factor: {args.local_support_factor}")

    # ── Direct GPU runs ──────────────────────────────────────────────
    direct_runs = []
    for run in range(args.runs):
        logger.info(f"\n{'─'*60}")
        logger.info(f"Direct GPU run {run + 1}/{args.runs}")
        logger.info(f"{'─'*60}")

        t0 = time.perf_counter()
        result = apriori(df, min_support=min_support, use_gpu=True, max_length=None,
                         n_gpus=args.n_gpus)
        elapsed = time.perf_counter() - t0

        n_itemsets = len(result)
        k_dist = k_distribution(result)
        max_k = max(k_dist.keys()) if k_dist else 0

        direct_runs.append({
            "run": run + 1,
            "itemsets": n_itemsets,
            "time_seconds": round(elapsed, 2),
            "max_k": max_k,
            "k_distribution": {str(k): v for k, v in k_dist.items()},
        })
        logger.info(f"  {n_itemsets:,} itemsets, max K={max_k}, {elapsed:.1f}s")

    # ── SON streaming runs ───────────────────────────────────────────
    son_runs = []
    for run in range(args.runs):
        logger.info(f"\n{'─'*60}")
        logger.info(f"SON streaming run {run + 1}/{args.runs}")
        logger.info(f"{'─'*60}")

        t0 = time.perf_counter()
        result = apriori_streaming(
            df,
            min_support=min_support,
            chunk_size=args.chunk_size,
            local_support_factor=args.local_support_factor,
            use_gpu=True,
            show_progress=True,
        )
        elapsed = time.perf_counter() - t0

        n_itemsets = len(result)
        k_dist = k_distribution(result)
        max_k = max(k_dist.keys()) if k_dist else 0

        son_runs.append({
            "run": run + 1,
            "itemsets": n_itemsets,
            "time_seconds": round(elapsed, 2),
            "max_k": max_k,
            "k_distribution": {str(k): v for k, v in k_dist.items()},
        })
        logger.info(f"  {n_itemsets:,} itemsets, max K={max_k}, {elapsed:.1f}s")

    # ── Aggregate statistics ─────────────────────────────────────────
    direct_itemsets = [r["itemsets"] for r in direct_runs]
    direct_times = [r["time_seconds"] for r in direct_runs]
    son_itemsets = [r["itemsets"] for r in son_runs]
    son_times = [r["time_seconds"] for r in son_runs]

    direct_mean_time = float(np.mean(direct_times))
    direct_std_time = float(np.std(direct_times, ddof=1))
    son_mean_time = float(np.mean(son_times))
    son_std_time = float(np.std(son_times, ddof=1))

    speedup = son_mean_time / direct_mean_time if direct_mean_time > 0 else float("inf")

    direct_mean_itemsets = float(np.mean(direct_itemsets))
    son_mean_itemsets = float(np.mean(son_itemsets))
    miss_rate = (1 - son_mean_itemsets / direct_mean_itemsets) * 100 if direct_mean_itemsets > 0 else 0

    # Check itemset consistency across runs
    direct_cv = float(np.std(direct_itemsets, ddof=1) / np.mean(direct_itemsets) * 100) if direct_mean_itemsets > 0 else 0
    son_cv = float(np.std(son_itemsets, ddof=1) / np.mean(son_itemsets) * 100) if son_mean_itemsets > 0 else 0

    logger.info(f"\n{'='*72}")
    logger.info(f"Summary (mean ± std across {args.runs} runs)")
    logger.info(f"{'='*72}")
    logger.info(f"  {'Method':<15} {'Itemsets':>20} {'Time (s)':>18} {'Max K':>8}")
    logger.info(f"  {'-'*65}")
    logger.info(f"  {'Direct GPU':<15} {direct_mean_itemsets:>12,.0f} ± {np.std(direct_itemsets, ddof=1):>5,.0f}"
          f"  {direct_mean_time:>8.1f} ± {direct_std_time:.1f}"
          f"  {max(r['max_k'] for r in direct_runs):>5}")
    logger.info(f"  {'SON':<15} {son_mean_itemsets:>12,.0f} ± {np.std(son_itemsets, ddof=1):>5,.0f}"
          f"  {son_mean_time:>8.1f} ± {son_std_time:.1f}"
          f"  {max(r['max_k'] for r in son_runs):>5}")
    logger.info(f"\n  Speedup (Direct vs SON): {speedup:.1f}x")
    logger.info(f"  SON miss rate: {miss_rate:.1f}%")
    logger.info(f"  Direct itemset CV: {direct_cv:.2f}% (expect 0 for exact algorithm)")
    logger.info(f"  SON itemset CV:    {son_cv:.2f}%")

    if direct_cv < 0.01:
        logger.info(f"  → Direct GPU is deterministic ✓")
    else:
        logger.info(f"  → WARNING: Direct GPU shows variance (expected 0)")

    # ── Save results ─────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"experiment_direct_vs_son_{timestamp}.json"

    output = {
        "experiment": "direct_gpu_vs_son_triplicate",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "min_support": min_support,
            "support_pct": f"{min_support*100:.4f}%",
            "n_transactions": n_transactions,
            "min_count": min_count,
            "n_runs": args.runs,
            "chunk_size": args.chunk_size,
            "local_support_factor": args.local_support_factor,
        },
        "direct_gpu_runs": direct_runs,
        "son_runs": son_runs,
        "aggregate": {
            "direct_gpu": {
                "mean_itemsets": round(direct_mean_itemsets, 1),
                "std_itemsets": round(float(np.std(direct_itemsets, ddof=1)), 1),
                "cv_itemsets_pct": round(direct_cv, 4),
                "mean_time_seconds": round(direct_mean_time, 2),
                "std_time_seconds": round(direct_std_time, 2),
            },
            "son": {
                "mean_itemsets": round(son_mean_itemsets, 1),
                "std_itemsets": round(float(np.std(son_itemsets, ddof=1)), 1),
                "cv_itemsets_pct": round(son_cv, 4),
                "mean_time_seconds": round(son_mean_time, 2),
                "std_time_seconds": round(son_std_time, 2),
            },
            "speedup": round(speedup, 2),
            "miss_rate_pct": round(miss_rate, 2),
        },
    }

    output_file.write_text(json.dumps(output, indent=2))
    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
