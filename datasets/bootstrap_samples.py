#!/usr/bin/env python3
"""
Bootstrap sampling utility for creating larger benchmark datasets.

Creates bootstrap samples from existing transaction datasets by sampling
with replacement. This preserves the real-world data distribution while
scaling to arbitrary sizes for stress testing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from loguru import logger


def create_bootstrap_sample(
    source_path: Path,
    target_transactions: int,
    output_path: Path,
    seed: int = 42,
) -> pl.DataFrame:
    """
    Bootstrap sample real-world data to target size.

    Args:
        source_path: Path to source parquet file with transactions
        target_transactions: Desired number of transactions
        output_path: Where to write the bootstrapped dataset
        seed: Random seed for reproducibility

    Returns:
        The bootstrapped DataFrame
    """
    logger.info(f"Loading source dataset from {source_path}...")
    df = pl.read_parquet(source_path)
    source_size = df.height
    logger.info(f"Source dataset: {source_size:,} transactions")

    if target_transactions <= source_size:
        logger.info(f"Target ({target_transactions:,}) <= source, sampling without replacement")
        sampled = df.sample(n=target_transactions, seed=seed, with_replacement=False)
    else:
        logger.info(f"Target ({target_transactions:,}) > source, sampling with replacement")
        sampled = df.sample(n=target_transactions, seed=seed, with_replacement=True)

    # Ensure we have exactly the target size
    assert sampled.height == target_transactions, (
        f"Expected {target_transactions}, got {sampled.height}"
    )

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.write_parquet(output_path)
    logger.info(f"Saved: {output_path} ({target_transactions:,} transactions)")

    item_counts = sampled.select(pl.col("items").list.len())
    logger.info("Dataset statistics:")
    logger.debug(f"  Transactions: {sampled.height:,}")
    logger.debug("  Items per transaction:")
    logger.debug(f"    Min: {item_counts.min().item()}")
    logger.debug(f"    Max: {item_counts.max().item()}")
    logger.debug(f"    Mean: {item_counts.mean().item():.1f}")
    logger.debug(f"    Median: {item_counts.median().item():.0f}")

    return sampled


def find_source_dataset(datasets_dir: Path) -> Path:
    """Find the best source dataset for bootstrapping."""
    # Priority: largest available real-world dataset
    candidates = [
        datasets_dir / "all_transactions_2m.parquet",
        datasets_dir / "all_transactions.parquet",
        datasets_dir / "online_retail_ii" / "transactions.parquet",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No source dataset found. Candidates checked: {candidates}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create bootstrap samples for benchmarking"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2_500_000,
        help="Target number of transactions (default: 2,500,000)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Source parquet file (auto-detected if not specified)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet file (auto-generated if not specified)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    datasets_dir = Path(__file__).parent

    # Auto-detect source
    if args.source is None:
        source_path = find_source_dataset(datasets_dir)
    else:
        source_path = args.source
        if not source_path.exists():
            raise FileNotFoundError(f"Source not found: {source_path}")

    # Auto-generate output name
    if args.output is None:
        size_str = f"{args.size // 1_000_000}m" if args.size >= 1_000_000 else f"{args.size // 1000}k"
        output_path = datasets_dir / f"all_transactions_{size_str}.parquet"
    else:
        output_path = args.output

    # Check if output already exists
    if output_path.exists():
        logger.info(f"Output already exists: {output_path}")
        df = pl.read_parquet(output_path)
        logger.info(f"Contains {df.height:,} transactions")

        if df.height == args.size:
            logger.info("Size matches, skipping generation")
            return
        else:
            logger.warning(f"Size mismatch (have {df.height:,}, want {args.size:,}), regenerating...")

    # Create bootstrap sample
    create_bootstrap_sample(
        source_path=source_path,
        target_transactions=args.size,
        output_path=output_path,
        seed=args.seed,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
