#!/usr/bin/env python3
"""Filter K=8 frequent itemsets to self-sufficient patterns only.

Self-sufficient itemsets are those where the K-th feature genuinely adds
information beyond what the (K-1)-subset already captures. An itemset
{A,B,C,D,E,F,G,H} is self-sufficient when:

    support({A,B,C,D,E,F,G,H}) / max(support(7-subsets)) < decay_threshold

A ratio close to 1.0 means the 8th item barely reduces support → near-closed
→ biologically uninteresting (the pattern is already captured at K=7).
A ratio well below 1.0 means the 8-item combination is genuinely rare
compared to any 7-item subset → interesting combinatorial signal.

Purpose: reduce K=8 from 12B to ~3-6B itemsets, making K=9 group building
feasible on 2 TB RAM machines (192 GB vs 384 GB at full 12B).

Usage:
    # On Vast.ai with 2 TB RAM:
    python filter_self_sufficient.py \
        --k8 /workspace/results_35k/run4/parquet/frequent_k8.parquet \
        --k7 /workspace/results_35k/run3/parquet/frequent_k7.parquet \
        --output /workspace/results_35k/run4/parquet/frequent_k8_selfsufficient.parquet \
        --decay-threshold 0.95 \
        --chunk-size 2000000

    # Local analysis (data on D: drive):
    python filter_self_sufficient.py \
        --k8 /mnt/d/et-miner-data/results_35k/run4/parquet/frequent_k8.parquet \
        --k7 /mnt/d/et-miner-data/results_35k/run3/parquet/frequent_k7.parquet \
        --output /mnt/d/et-miner-data/results_35k/run4/parquet/frequent_k8_selfsufficient.parquet

Output parquet has same schema as input (itemset, support) and is directly
usable as --resume-from-k 8 input for K=9 mining.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path for et_miner imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from et_miner.core.rules import _detect_k, _iter_row_groups, _explode_drop1, _list_to_scalar_cols

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from loguru import logger


def filter_self_sufficient(
    k_parquet: Path,
    k_minus1_parquet: Path,
    output_path: Path,
    decay_threshold: float = 0.95,
    chunk_size: int = 1_000_000,
) -> dict:
    """Filter K-itemsets to only self-sufficient patterns.

    Processes K parquet in chunks, joins each chunk's drop-1 subsets with
    K-1 support values, computes max K-1 support per K-itemset, and keeps
    only itemsets where support_K / max_K-1_support < decay_threshold.

    The output parquet preserves the original schema (itemset, support)
    for direct compatibility with et-miner's --resume-from-k pipeline.

    Args:
        k_parquet: Path to K-level parquet (e.g., frequent_k8.parquet).
        k_minus1_parquet: Path to (K-1)-level parquet (e.g., frequent_k7.parquet).
        output_path: Where to write filtered parquet.
        decay_threshold: Max ratio to keep. Items with ratio >= threshold
            are considered near-closed and filtered out.
            Default 0.95 means the K-th item must reduce support by >5%.
        chunk_size: Rows per processing chunk.

    Returns:
        Dict with stats: total_input, total_output, reduction_pct, ratio_stats.
    """
    k = _detect_k(k_parquet)
    km1 = k - 1
    logger.info(f"Filtering K={k} itemsets for self-sufficiency (threshold={decay_threshold:.3f})")

    # ── Load K-1 support lookup (loaded ONCE, reused for all chunks) ──
    logger.info(f"Loading K-{km1} parquet: {k_minus1_parquet}")
    t_start = time.time()

    join_cols = [f"i{j}" for j in range(km1)]
    km1_df: pl.DataFrame = (
        pl.scan_parquet(k_minus1_parquet)
        .with_columns(_list_to_scalar_cols("itemset", km1))
        .select([*join_cols, pl.col("support").alias("km1_support")])
        .collect(engine="streaming")
    )
    logger.info(f"K-{km1} loaded: {len(km1_df):,} itemsets in {time.time() - t_start:.1f}s")

    # ── Initialize output writer ──
    # We write chunks incrementally to avoid accumulating the full filtered
    # result in memory (could still be 3-6B rows).
    writer = None
    total_input = 0
    total_output = 0
    total_near_closed = 0
    ratio_sums = 0.0
    ratio_min = float("inf")
    ratio_max = float("-inf")
    chunk_idx = 0
    t_filter_start = time.time()

    for chunk in _iter_row_groups(k_parquet, chunk_size):
        chunk_idx += 1
        n_rows = len(chunk)
        total_input += n_rows

        # Add row index for grouping back after explode
        chunk = chunk.with_row_index("_row_idx")

        # Explode into K rows per itemset (drop-1)
        exploded = _explode_drop1(chunk.drop("_row_idx"), k)

        # Reconstruct row index (cross join order: K copies per row)
        n_exploded = len(exploded)
        exploded = exploded.with_columns(
            pl.Series("_row_idx", [i // k for i in range(n_exploded)], dtype=pl.UInt32)
        )

        # Unpack antecedent into scalar columns and join with K-1
        exploded = exploded.with_columns(_list_to_scalar_cols("antecedent", km1))
        joined = exploded.join(km1_df, on=join_cols, how="inner")
        del exploded

        # Group by row → max K-1 support
        grouped = (
            joined
            .group_by("_row_idx")
            .agg(pl.col("km1_support").max().alias("max_km1_support"))
        )
        del joined

        # Warn if K-1 subsets are missing (inner join will silently drop these)
        n_expected = len(chunk)
        n_found = grouped.height
        if n_found < n_expected:
            logger.warning(
                f"Chunk {chunk_idx}: {n_expected - n_found}/{n_expected} itemsets missing K-1 subsets (dropped by inner join)"
            )

        # Join back to get original itemset + support, compute ratio
        result = (
            chunk
            .join(grouped, on="_row_idx", how="inner")
            .with_columns(
                (pl.col("support") / pl.col("max_km1_support")).alias("ratio")
            )
        )
        del grouped

        # Track stats
        ratios = result["ratio"]
        chunk_min = ratios.min()
        chunk_max = ratios.max()
        ratio_sums += ratios.sum()
        if chunk_min < ratio_min:
            ratio_min = chunk_min
        if chunk_max > ratio_max:
            ratio_max = chunk_max

        n_near_closed = result.filter(pl.col("ratio") >= decay_threshold).height
        total_near_closed += n_near_closed

        # Filter: keep only self-sufficient (ratio < threshold)
        filtered = (
            result
            .filter(pl.col("ratio") < decay_threshold)
            .select("itemset", "support")
        )
        del result, chunk

        n_kept = len(filtered)
        total_output += n_kept

        # Write chunk to parquet
        if n_kept > 0:
            arrow_chunk = filtered.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(
                    str(output_path),
                    schema=arrow_chunk.schema,
                    compression="zstd",
                )
            writer.write_table(arrow_chunk)
            del arrow_chunk
        del filtered

        # Progress logging
        pct_kept = (n_kept / n_rows * 100) if n_rows > 0 else 0
        elapsed = time.time() - t_filter_start
        rows_per_sec = total_input / elapsed if elapsed > 0 else 0
        logger.info(
            f"Chunk {chunk_idx}: {n_rows:,} → {n_kept:,} kept ({pct_kept:.1f}%) | Total: {total_output:,}/{total_input:,} | {rows_per_sec:.0f} rows/s"
        )

    # ── Finalize ──
    if writer is not None:
        writer.close()

    elapsed_total = time.time() - t_filter_start
    reduction_pct = ((total_input - total_output) / total_input * 100) if total_input > 0 else 0
    ratio_mean = ratio_sums / total_input if total_input > 0 else 0

    stats = {
        "k": k,
        "decay_threshold": decay_threshold,
        "total_input": total_input,
        "total_output": total_output,
        "total_near_closed": total_near_closed,
        "reduction_pct": reduction_pct,
        "ratio_min": ratio_min,
        "ratio_max": ratio_max,
        "ratio_mean": ratio_mean,
        "elapsed_seconds": elapsed_total,
        "output_path": str(output_path),
    }

    logger.info("=" * 60)
    logger.info("SELF-SUFFICIENT FILTERING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Input:      {total_input:,} K={k} itemsets")
    logger.info(f"Output:     {total_output:,} self-sufficient ({reduction_pct:.1f}% reduction)")
    logger.info(f"Near-closed: {total_near_closed:,} (ratio >= {decay_threshold:.3f})")
    logger.info(f"Ratio:       min={ratio_min:.6f}  mean={ratio_mean:.6f}  max={ratio_max:.6f}")
    logger.info(f"Time:        {elapsed_total:.1f}s ({total_input / elapsed_total if elapsed_total > 0 else 0:.0f} rows/s)")
    logger.info(f"Output:      {output_path}")
    logger.info("=" * 60)

    # Memory estimate for K=9 group building
    bytes_per_itemset = k * 4  # K items × 4 bytes (int32)
    k9_mem_gb = total_output * bytes_per_itemset / (1024 ** 3)
    logger.info(f"K=9 group building estimate: {total_output:,} × {k} × 4B = {k9_mem_gb:.0f} GB")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Filter K-itemsets to self-sufficient patterns.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--k8", "--k-parquet", type=Path, required=True,
        help="Path to K-level frequent itemsets parquet",
    )
    parser.add_argument(
        "--k7", "--k-minus1-parquet", type=Path, required=True,
        help="Path to (K-1)-level frequent itemsets parquet",
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Output path for filtered parquet",
    )
    parser.add_argument(
        "--decay-threshold", type=float, default=0.95,
        help="Max self-sufficiency ratio to keep (default: 0.95)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1_000_000,
        help="Rows per processing chunk (default: 1M)",
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.k8.exists():
        logger.error(f"K parquet not found: {args.k8}")
        sys.exit(1)
    if not args.k7.exists():
        logger.error(f"K-1 parquet not found: {args.k7}")
        sys.exit(1)

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    stats = filter_self_sufficient(
        k_parquet=args.k8,
        k_minus1_parquet=args.k7,
        output_path=args.output,
        decay_threshold=args.decay_threshold,
        chunk_size=args.chunk_size,
    )

    # Write stats JSON alongside output
    import json
    stats_path = args.output.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats written to: {stats_path}")


if __name__ == "__main__":
    main()
