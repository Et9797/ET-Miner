#!/usr/bin/env python3
"""Run ET-miner frequent itemset mining on AlphaFold structural features.

Discovers structural motifs (co-occurring feature combinations) across
protein structures using GPU-accelerated Apriori algorithm.
"""

import argparse
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_transactions, load_item_names

import polars as pl

from et_miner import apriori

from loguru import logger

DEFAULT_INPUT = "data/processed/alphafold_transactions_nr30.parquet"
DEFAULT_MAPPING = "data/processed/item_mapping.parquet"
DEFAULT_OUTPUT = "results"


def print_dataset_stats(df: pl.DataFrame) -> dict:
    """Print and return dataset statistics."""
    items_col = df.get_column("items")
    lengths = items_col.list.len()
    all_items = items_col.explode().drop_nulls()

    n_transactions = len(df)
    n_unique = all_items.n_unique()
    avg_len = lengths.mean()

    logger.info(f"\n{'='*50}")
    logger.info(f"Dataset Statistics")
    logger.info(f"{'='*50}")
    logger.info(f"Transactions:       {n_transactions:>10,}")
    logger.info(f"Unique items:       {n_unique:>10,}")
    logger.info(f"Avg items/txn:      {avg_len:>10.1f}")
    logger.info(f"Min items/txn:      {lengths.min():>10}")
    logger.info(f"Max items/txn:      {lengths.max():>10}")

    # Item frequency distribution
    freq = all_items.value_counts().sort("count", descending=True)
    logger.info(f"\nTop-10 most frequent items:")
    for row in freq.head(10).iter_rows(named=True):
        pct = row["count"] / n_transactions * 100
        logger.info(f"  Item {row['items']:>3}: {row['count']:>8,} ({pct:.1f}%)")
    logger.info("")

    return {
        "n_transactions": n_transactions,
        "n_unique_items": n_unique,
        "avg_items_per_transaction": round(avg_len, 2),
    }


def decode_itemsets(result: pl.DataFrame, mapping: dict[int, str]) -> list[dict]:
    """Convert mining results to decoded itemset list."""
    itemsets = []
    for row in result.iter_rows(named=True):
        item_ids = sorted(row["itemset"])
        features = [mapping.get(i, f"unknown_{i}") for i in item_ids]
        itemsets.append({
            "item_ids": item_ids,
            "features": features,
            "support": round(row["support"], 6),
            "size": len(item_ids),
        })
    # Sort by support descending, then by size descending
    itemsets.sort(key=lambda x: (-x["support"], -x["size"]))
    return itemsets


def run_baseline(df: pl.DataFrame, min_support: float, use_gpu: bool,
                 max_length: int, n_runs: int, real_count: int) -> dict:
    """Random shuffle baseline to assess statistical significance."""
    all_items = df.select(pl.col("items").explode()).to_series().to_list()
    baseline_counts = []

    for run in range(n_runs):
        logger.info(f"Baseline run {run + 1}/{n_runs}")
        shuffled = df.with_columns(
            pl.col("items").map_elements(
                lambda x: sorted(random.sample(all_items, min(len(x), len(all_items)))),
                return_dtype=pl.List(pl.Int64),
            )
        )
        baseline_result = apriori(shuffled, min_support=min_support,
                                  use_gpu=use_gpu, max_length=max_length)
        baseline_counts.append(len(baseline_result))

    mean_baseline = sum(baseline_counts) / len(baseline_counts)
    ratio = real_count / max(mean_baseline, 1)

    if len(baseline_counts) > 1:
        std_baseline = statistics.stdev(baseline_counts)
        z_score = (real_count - mean_baseline) / max(std_baseline, 1)
    else:
        z_score = float("inf") if real_count > mean_baseline else 0.0

    logger.info(f"\nBaseline Analysis ({n_runs} runs):")
    logger.info(f"  Real:     {real_count} itemsets")
    logger.info(f"  Baseline: {mean_baseline:.1f} (avg)")
    logger.info(f"  Ratio:    {ratio:.1f}x")
    logger.info(f"  Z-score:  {z_score:.2f}")

    return {
        "mean_itemsets": round(mean_baseline, 2),
        "std_itemsets": round(std_baseline, 2) if len(baseline_counts) > 1 else 0.0,
        "ratio": round(ratio, 2),
        "z_score": round(z_score, 2),
        "n_runs": n_runs,
    }


def run_sweep(df: pl.DataFrame, thresholds: list[float], use_gpu: bool,
              max_length: int, mapping: dict[int, str]) -> None:
    """Run mining at multiple support thresholds and print comparison."""
    header = f"{'Support':>8} | {'Itemsets':>8}"
    for k in range(2, max_length + 1):
        header += f" | {'Size-' + str(k):>6}"
    header += f" | {'Time (s)':>8}"
    logger.info(f"\n{header}")
    logger.info("-" * len(header))

    for threshold in sorted(thresholds, reverse=True):
        t0 = time.perf_counter()
        result = apriori(df, min_support=threshold, use_gpu=use_gpu,
                         max_length=max_length)
        elapsed = time.perf_counter() - t0

        total = len(result)
        sizes = {}
        if total > 0:
            for row in result.iter_rows(named=True):
                k = len(row["itemset"])
                sizes[k] = sizes.get(k, 0) + 1

        row_str = f"{threshold:>8.4f} | {total:>8}"
        for k in range(2, max_length + 1):
            row_str += f" | {sizes.get(k, 0):>6}"
        row_str += f" | {elapsed:>8.1f}"
        logger.info(row_str)


def print_summary(itemsets: list[dict], top_n: int = 20) -> None:
    """Print summary of discovered motifs."""
    if not itemsets:
        logger.info("\nNo itemsets found.")
        return

    # Count by size
    by_size: dict[int, int] = {}
    for it in itemsets:
        by_size[it["size"]] = by_size.get(it["size"], 0) + 1

    logger.info(f"\n{'='*50}")
    logger.info(f"Mining Results Summary")
    logger.info(f"{'='*50}")
    logger.info(f"Total itemsets: {len(itemsets)}")
    for k in sorted(by_size):
        logger.info(f"  Size {k}: {by_size[k]}")

    logger.info(f"\nTop-{top_n} itemsets by support:")
    for i, it in enumerate(itemsets[:top_n]):
        features_str = ", ".join(it["features"])
        logger.info(f"  {i+1:>3}. [{features_str}]  support={it['support']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run ET-miner on AlphaFold structural features",
        epilog="""Examples:
  python run_mining.py --use-gpu
  python run_mining.py --support 0.05 --use-gpu
  python run_mining.py --sweep 0.1,0.05,0.01,0.005 --use-gpu
  python run_mining.py --baseline --baseline-runs 10 --use-gpu
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Input transactions parquet (default: %(default)s)")
    parser.add_argument("--item-mapping", default=DEFAULT_MAPPING,
                        help="Item mapping parquet (default: %(default)s)")
    parser.add_argument("--support", type=float, default=0.01,
                        help="Minimum support threshold (default: %(default)s)")
    parser.add_argument("--max-length", type=int, default=4,
                        help="Maximum itemset length (default: %(default)s)")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Use GPU acceleration")
    parser.add_argument("--sweep", type=str, default=None,
                        help="Comma-separated support thresholds for sweep")
    parser.add_argument("--baseline", action="store_true",
                        help="Run random shuffle baseline")
    parser.add_argument("--baseline-runs", type=int, default=5,
                        help="Number of baseline runs (default: %(default)s)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT,
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--n-gpus", type=int, default=1,
                        help="Number of GPUs for row-split mining (default: %(default)s)")
    parser.add_argument("--parquet-flush", action="store_true",
                        help="Flush results per K level to Parquet (prevents CPU RAM OOM)")
    parser.add_argument("--resume-from-k", type=int, default=None,
                        help="Resume from K=N+1 by loading K=N frequent itemsets from parquet")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging")
    args = parser.parse_args()

    # Load data
    df = load_transactions(args.input)
    mapping = load_item_names(args.item_mapping)
    stats = print_dataset_stats(df)

    # Support sweep mode
    if args.sweep:
        thresholds = [float(x.strip()) for x in args.sweep.split(",")]
        run_sweep(df, thresholds, args.use_gpu, args.max_length, mapping)
        return

    # Single run
    logger.info(f"Running apriori (support={args.support:.4f}, max_length={args.max_length}, gpu={args.use_gpu})")

    # Per-K Parquet flush for multi-GPU runs: prevents CPU RAM OOM at K=7+ scale
    parquet_dir = None
    if args.use_gpu and args.n_gpus > 1 and args.parquet_flush:
        parquet_dir = str(Path(args.output_dir) / "parquet")
        logger.info(f"Per-K Parquet flush enabled → {parquet_dir}")

    t0 = time.perf_counter()
    result = apriori(df, min_support=args.support, use_gpu=args.use_gpu,
                     max_length=args.max_length, n_gpus=args.n_gpus,
                     output_dir=parquet_dir,
                     resume_from_k=args.resume_from_k)
    duration = time.perf_counter() - t0
    logger.info(f"Mining completed in {duration:.1f}s, found {len(result)} itemsets")

    # When results are flushed to Parquet, skip in-memory decode
    if parquet_dir:
        logger.info(f"\nResults flushed to {parquet_dir}/frequent_k*.parquet")
        logger.info(f"Total time: {duration:.1f}s")

        # Write metadata JSON alongside Parquet files
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "input_file": args.input,
            "n_transactions": stats["n_transactions"],
            "n_unique_items": stats["n_unique_items"],
            "min_support": args.support,
            "max_length": args.max_length,
            "n_gpus": args.n_gpus,
            "use_gpu": args.use_gpu,
            "parquet_dir": parquet_dir,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(duration, 2),
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        meta_file = output_dir / f"mining_meta_{args.support}_{timestamp}.json"
        meta_file.write_text(json.dumps(metadata, indent=2))
        logger.info(f"Metadata written to {meta_file}")
        return

    # Decode results
    itemsets = decode_itemsets(result, mapping)
    print_summary(itemsets)

    # Baseline
    baseline_data = None
    if args.baseline:
        baseline_data = run_baseline(df, args.support, args.use_gpu,
                                     args.max_length, args.baseline_runs,
                                     len(result))

    # Build output
    by_size: dict[str, int] = {}
    for it in itemsets:
        by_size[str(it["size"])] = by_size.get(str(it["size"]), 0) + 1

    output = {
        "metadata": {
            "input_file": args.input,
            "n_transactions": stats["n_transactions"],
            "n_unique_items": stats["n_unique_items"],
            "min_support": args.support,
            "max_length": args.max_length,
            "use_gpu": args.use_gpu,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(duration, 2),
        },
        "summary": {
            "total_itemsets": len(itemsets),
            "by_size": by_size,
        },
        "itemsets": itemsets,
    }
    if baseline_data:
        output["baseline"] = baseline_data

    # Write output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"motifs_{args.support}_{timestamp}.json"
    output_file.write_text(json.dumps(output, indent=2))
    logger.info(f"\nResults written to {output_file}")


if __name__ == "__main__":
    main()
