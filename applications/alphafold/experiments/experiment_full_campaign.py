#!/usr/bin/env python3
"""Experiment: Full mining campaign across all 6 support thresholds (triplicate).

Reproduces Table 2 from the paper with proper statistics: each threshold
is run N times (default 3) using Direct GPU row-split mining.

NOTE: At 35K features, SON streaming is infeasible — each 40M-row chunk
requires ~175 GB bitvec, exceeding single-GPU VRAM (143 GB on H200).
All thresholds use Direct GPU with multi-GPU row-split.

Thresholds:
  Base   0.1%      direct GPU
  Super  0.01%     direct GPU
  Power  0.001%    direct GPU
  Blitz  0.0001%   direct GPU
  Ultra  0.00002%  direct GPU
  Opus   0.00001%  direct GPU
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

from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_transactions, k_distribution

# Campaign thresholds: (name, min_support)
# All use Direct GPU — SON infeasible at 35K features (chunk bitvec > single GPU VRAM)
THRESHOLDS = [
    ("Base",  0.001),       # 0.1%
    ("Super", 0.0001),      # 0.01%
    ("Power", 0.00001),     # 0.001%
    ("Blitz", 0.000001),    # 0.0001%
    ("Ultra", 0.0000002),   # 0.00002%
    ("Opus",  0.0000001),   # 0.00001%
]


def run_single(
    df: pl.DataFrame,
    min_support: float,
    n_gpus: int = 1,
) -> dict:
    """Run a single Direct GPU mining operation and return structured results."""
    t0 = time.perf_counter()

    result = apriori(
        df, min_support=min_support, use_gpu=True, max_length=None,
        n_gpus=n_gpus,
    )

    elapsed = time.perf_counter() - t0
    n_itemsets = len(result)
    k_dist = k_distribution(result)
    max_k = max(k_dist.keys()) if k_dist else 0

    return {
        "itemsets": n_itemsets,
        "time_seconds": round(elapsed, 2),
        "max_k": max_k,
        "k_distribution": {str(k): v for k, v in k_dist.items()},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Full mining campaign across 6 support thresholds (triplicate)",
    )
    parser.add_argument(
        "--data", default="/workspace/data/transactions_35k.parquet",
        help="Path to transactions parquet (default: %(default)s)",
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of runs per threshold (default: %(default)s)",
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

    df = load_transactions(args.data, min_items=2)
    n_transactions = len(df)

    logger.info(f"\n{'='*72}")
    logger.info(f"Full Mining Campaign (Triplicate)")
    logger.info(f"{'='*72}")
    logger.info(f"  n_transactions:  {n_transactions:,}")
    logger.info(f"  runs per threshold: {args.runs}")
    logger.info(f"  thresholds: {len(THRESHOLDS)}")

    all_results = {}
    total_t0 = time.perf_counter()

    for name, min_support in THRESHOLDS:
        min_count = math.ceil(min_support * n_transactions)
        support_pct = f"{min_support * 100:.4f}%"

        logger.info(f"\n{'='*60}")
        logger.info(f"Threshold: {name} ({support_pct}, min_count={min_count:,}, direct)")
        logger.info(f"{'='*60}")

        runs = []
        for run_idx in range(args.runs):
            logger.info(f"  Run {run_idx + 1}/{args.runs}...", end=" ", flush=True)
            result = run_single(df, min_support, n_gpus=args.n_gpus)
            runs.append(result)
            logger.info(f"{result['itemsets']:,} itemsets, K={result['max_k']}, "
                  f"{result['time_seconds']:.1f}s")

        # Aggregate
        itemsets_list = [r["itemsets"] for r in runs]
        times_list = [r["time_seconds"] for r in runs]
        max_k_list = [r["max_k"] for r in runs]

        mean_itemsets = float(np.mean(itemsets_list))
        std_itemsets = float(np.std(itemsets_list, ddof=1)) if len(itemsets_list) > 1 else 0.0
        cv_itemsets = std_itemsets / mean_itemsets * 100 if mean_itemsets > 0 else 0

        agg = {
            "name": name,
            "min_support": min_support,
            "support_pct": support_pct,
            "min_count": min_count,
            "method": "direct",
            "runs": runs,
            "aggregate": {
                "mean_itemsets": round(mean_itemsets, 1),
                "std_itemsets": round(std_itemsets, 1),
                "cv_itemsets_pct": round(cv_itemsets, 4),
                "mean_time_seconds": round(float(np.mean(times_list)), 2),
                "std_time_seconds": round(float(np.std(times_list, ddof=1)) if len(times_list) > 1 else 0.0, 2),
                "max_k": max(max_k_list),
                "min_k": min(max_k_list),
            },
        }
        all_results[name] = agg

        logger.info(f"  → Mean: {mean_itemsets:,.0f} ± {std_itemsets:,.0f} itemsets "
              f"(CV={cv_itemsets:.2f}%), "
              f"K={max(max_k_list)}, "
              f"{np.mean(times_list):.1f} ± {np.std(times_list, ddof=1) if len(times_list) > 1 else 0.0:.1f}s")

    total_time = time.perf_counter() - total_t0

    # ── Summary table ────────────────────────────────────────────────
    logger.info(f"\n{'='*80}")
    logger.info(f"Campaign Summary")
    logger.info(f"{'='*80}")
    logger.info(f"  {'Run':<8} {'Support':>10} {'Min Prot':>10} {'Itemsets':>15} "
          f"{'Max K':>7} {'Time':>10}")
    logger.info(f"  {'-'*65}")

    for name, _ in THRESHOLDS:
        r = all_results[name]
        a = r["aggregate"]
        time_str = f"{a['mean_time_seconds']:.1f}s"
        if args.runs > 1:
            time_str += f" ±{a['std_time_seconds']:.1f}"
        logger.info(f"  {name:<8} {r['support_pct']:>10} {r['min_count']:>10,} "
              f"{a['mean_itemsets']:>12,.0f} ±{a['std_itemsets']:>4,.0f} "
              f"{a['max_k']:>5} {time_str:>12}")

    logger.info(f"\n  Total campaign time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # ── Save results ─────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"experiment_full_campaign_{timestamp}.json"

    output = {
        "experiment": "full_mining_campaign_triplicate",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "n_transactions": n_transactions,
            "n_runs": args.runs,
            "n_gpus": args.n_gpus,
            "method": "direct_gpu_row_split",
        },
        "thresholds": all_results,
        "total_time_seconds": round(total_time, 2),
    }

    output_file.write_text(json.dumps(output, indent=2))
    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
