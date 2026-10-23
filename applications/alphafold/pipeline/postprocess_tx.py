#!/usr/bin/env python3
"""
Post-process Rust build_tx output → final transactions + item_mapping parquets.

Reads the raw TSV from build_tx (118K features at min_count=8), applies a higher
min_count threshold to reduce to target feature count, remaps IDs, merges pLDDT
features from existing dataset, and writes final parquets.

Usage:
    python postprocess_tx.py \
        --raw-tsv /workspace/data/transactions_12k_raw.tsv \
        --feature-mapping /workspace/data/annotations/uniprot_expanded_feature_mapping.tsv \
        --existing-transactions /workspace/data/processed/transactions_214m.parquet \
        --existing-mapping /workspace/data/processed/item_mapping_214m.parquet \
        --min-count 3423 \
        --output /workspace/data/processed/transactions_35k.parquet \
        --mapping /workspace/data/processed/item_mapping_35k.parquet
"""

import argparse
import sys
import time
from pathlib import Path

import polars as pl
from loguru import logger


def load_feature_mapping(path: Path, min_count: int) -> tuple[dict[int, int], pl.DataFrame]:
    """Load feature mapping TSV, filter by min_count, return old_id->new_id map and filtered DataFrame."""
    t0 = time.time()
    df = pl.read_csv(path, separator="\t")
    total = df.height
    df = df.filter(pl.col("count") >= min_count).sort("feature_name")

    # Assign new contiguous IDs (sorted by feature_name for determinism)
    df = df.with_row_index("new_id").with_columns(pl.col("new_id").cast(pl.Int64))

    # Build old_id -> new_id mapping
    old_ids = df["item_id"].to_list()
    new_ids = df["new_id"].to_list()
    remap = dict(zip(old_ids, new_ids))

    elapsed = time.time() - t0
    logger.info(f"Feature mapping: {total:,} total → {df.height:,} with count >= {min_count} ({elapsed:.1f}s)")

    # Category breakdown
    for cat in df["feature_category"].unique().sort().to_list():
        sub = df.filter(pl.col("feature_category") == cat)
        logger.info(f"  {cat:<15s} {sub.height:>6,} features")

    return remap, df


def load_plddt_features(
    tx_path: Path, mapping_path: Path
) -> tuple[dict[str, list[str]], list[str]]:
    """Extract per-protein pLDDT feature names from existing transactions."""
    t0 = time.time()

    mapping_df = pl.read_parquet(mapping_path)
    plddt_rows = mapping_df.filter(pl.col("feature_name").str.contains("plddt"))
    if plddt_rows.height == 0:
        logger.info("  No pLDDT features found in existing mapping")
        return {}, []

    plddt_id_to_name = dict(zip(
        plddt_rows["item_id"].to_list(),
        plddt_rows["feature_name"].to_list(),
    ))
    plddt_feature_names = sorted(plddt_id_to_name.values())
    logger.info(f"  pLDDT features: {plddt_feature_names}")

    # list.contains() approach: 6 in-place checks, zero intermediate rows
    # (explode on 205M×5 items = 1B rows = 98 GB; list.contains = ~1.2 GB)
    tx_df = pl.read_parquet(tx_path, columns=["protein_id", "items"])
    plddt_names = sorted(plddt_id_to_name.values())

    # Add boolean flag per pLDDT feature (all 6 at once)
    tx_df = tx_df.with_columns([
        pl.col("items").list.contains(pid).alias(f"has_{name}")
        for pid, name in sorted(plddt_id_to_name.items())
    ])

    # Build list of feature names per protein using conditional concat_list
    feature_list_expr = pl.concat_list([
        pl.when(pl.col(f"has_{name}")).then(pl.lit(name))
        for name in plddt_names
    ]).list.drop_nulls()

    result = (
        tx_df
        .with_columns(feature_list_expr.alias("plddt_features"))
        .filter(pl.col("plddt_features").list.len() > 0)
        .select(["protein_id", "plddt_features"])
    )

    protein_plddt = dict(zip(
        result["protein_id"].to_list(),
        result["plddt_features"].to_list(),
    ))

    elapsed = time.time() - t0
    print(f"  Loaded pLDDT for {len(protein_plddt):,} proteins ({elapsed:.1f}s)")
    return protein_plddt, plddt_feature_names


def process_transactions(
    raw_tsv: Path,
    remap: dict[int, int],
    protein_plddt: dict[str, list[str]],
    plddt_name_to_id: dict[str, int],
) -> tuple[list[str], list[list[int]]]:
    """Stream raw TSV, remap feature IDs, merge pLDDT, return protein_ids and item_lists."""
    t0 = time.time()
    protein_ids: list[str] = []
    item_lists: list[list[int]] = []
    n_lines = 0
    n_kept = 0
    n_plddt_merged = 0

    with open(raw_tsv) as f:
        for line in f:
            n_lines += 1
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue

            protein_id = parts[0]
            old_ids = parts[1].split(",")

            # Remap: keep only IDs that survive the threshold
            new_ids = set()
            for old_id_str in old_ids:
                old_id = int(old_id_str)
                if old_id in remap:
                    new_ids.add(remap[old_id])

            # Merge pLDDT features
            if protein_id in protein_plddt:
                for feat_name in protein_plddt[protein_id]:
                    if feat_name in plddt_name_to_id:
                        new_ids.add(plddt_name_to_id[feat_name])
                n_plddt_merged += 1

            if len(new_ids) > 1:
                protein_ids.append(protein_id)
                item_lists.append(sorted(new_ids))
                n_kept += 1

            if n_lines % 5_000_000 == 0:
                elapsed = time.time() - t0
                rate = n_lines / elapsed
                logger.info(f"  {n_lines:>12,} lines | {n_kept:>10,} kept | {n_plddt_merged:>10,} pLDDT | {rate:,.0f} lines/s")

    # Skip pLDDT-only proteins: they have only 2 pLDDT features (mean + frac bin),
    # no UniProt annotations. Adding ~96M noise proteins would bloat bitvec from
    # 477 GB to 897 GB (exceeds 4×H200 VRAM) and contribute only K≤2 patterns.
    n_plddt_only = 0

    elapsed = time.time() - t0
    logger.info(f"  Done: {n_lines:,} lines → {n_kept:,} kept + {n_plddt_only:,} pLDDT-only = "
          f"{len(protein_ids):,} total ({elapsed:.1f}s)")
    return protein_ids, item_lists


def main():
    parser = argparse.ArgumentParser(
        description="Post-process Rust build_tx output → final parquets",
    )
    parser.add_argument("--raw-tsv", type=Path, required=True,
                        help="Raw transactions TSV from build_tx (protein_id\\tid1,id2,...)")
    parser.add_argument("--feature-mapping", type=Path, required=True,
                        help="Feature mapping TSV from build_tx (item_id, feature_name, category, count)")
    parser.add_argument("--existing-transactions", type=Path, required=True,
                        help="Existing transactions_214m.parquet (for pLDDT features)")
    parser.add_argument("--existing-mapping", type=Path, required=True,
                        help="Existing item_mapping_214m.parquet (for pLDDT feature names)")
    parser.add_argument("--min-count", type=int, default=3423,
                        help="Minimum feature count threshold (default: 3423 for ~35K features)")
    parser.add_argument("--output", type=Path, default=Path("transactions_35k.parquet"),
                        help="Output transactions parquet")
    parser.add_argument("--mapping", type=Path, default=Path("item_mapping_35k.parquet"),
                        help="Output item mapping parquet")

    args = parser.parse_args()

    for path, label in [
        (args.raw_tsv, "Raw TSV"), (args.feature_mapping, "Feature mapping"),
        (args.existing_transactions, "Existing transactions"),
        (args.existing_mapping, "Existing mapping"),
    ]:
        if not path.exists():
            logger.error(f"{label} not found: {path}")
            sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.mapping.parent.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    # --- Load + filter feature mapping ---
    logger.info("=" * 60)
    logger.info("Step 1: Load feature mapping + apply threshold")
    logger.info("=" * 60)
    remap, mapping_df = load_feature_mapping(args.feature_mapping, args.min_count)

    # --- Load pLDDT from existing dataset ---
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Load pLDDT features from existing dataset")
    logger.info("=" * 60)
    protein_plddt, plddt_feature_names = load_plddt_features(
        args.existing_transactions, args.existing_mapping
    )

    # Assign IDs to pLDDT features (appended after the filtered features)
    next_id = mapping_df.height
    plddt_name_to_id: dict[str, int] = {}
    plddt_mapping_rows = []
    for feat_name in sorted(plddt_feature_names):
        plddt_name_to_id[feat_name] = next_id
        plddt_mapping_rows.append({
            "new_id": next_id,
            "item_id": -1,  # no old ID
            "feature_name": feat_name,
            "feature_category": "plddt",
            "count": -1,  # count not applicable
        })
        next_id += 1

    if plddt_mapping_rows:
        plddt_df = pl.DataFrame(plddt_mapping_rows, schema=mapping_df.schema)
        mapping_df = pl.concat([mapping_df, plddt_df])

    total_features = mapping_df.height
    logger.info(f"\nTotal features: {total_features:,} ({total_features - len(plddt_feature_names):,} from TSV + {len(plddt_feature_names)} pLDDT)")

    # --- Process transactions ---
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Process transactions (remap + merge pLDDT)")
    logger.info("=" * 60)
    protein_ids, item_lists = process_transactions(
        args.raw_tsv, remap, protein_plddt, plddt_name_to_id
    )

    # --- Write parquets ---
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Write parquet files")
    logger.info("=" * 60)

    tx_df = pl.DataFrame(
        {"protein_id": protein_ids, "items": item_lists},
        schema={"protein_id": pl.Utf8, "items": pl.List(pl.Int64)},
    )
    tx_df.write_parquet(args.output)

    item_counts = tx_df["items"].list.len()
    logger.info(f"  Transactions: {tx_df.height:,} proteins → {args.output}")
    logger.info(f"  Items/protein: mean={item_counts.mean():.1f}, median={item_counts.median()}, "
          f"min={item_counts.min()}, max={item_counts.max()}")

    # Write item mapping (using new_id as the authoritative item_id)
    out_mapping = mapping_df.select([
        pl.col("new_id").alias("item_id"),
        "feature_name",
        "feature_category",
        "count",
    ]).with_columns(
        pl.col("item_id").cast(pl.Int64),
        pl.col("count").cast(pl.Int64),
    )
    out_mapping.write_parquet(args.mapping)
    logger.info(f"  Item mapping: {out_mapping.height:,} features → {args.mapping}")

    for cat in out_mapping["feature_category"].unique().sort().to_list():
        sub = out_mapping.filter(pl.col("feature_category") == cat)
        logger.info(f"    {cat:<15s} {sub.height:>6,} features")

    # VRAM estimate
    n_proteins = tx_df.height
    n_features = total_features
    bitvec_bytes = n_proteins * ((n_features + 7) // 8)
    bitvec_gb = bitvec_bytes / (1024 ** 3)
    logger.info(f"\n  VRAM estimate: {n_proteins:,} proteins × {n_features:,} features = {bitvec_gb:.1f} GB bitvec")
    gpus_needed = max(1, int(bitvec_gb / 135) + 1)
    logger.info(f"  GPUs needed: {gpus_needed} × H200 (143 GB)")

    elapsed = time.time() - t_total
    logger.info(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
