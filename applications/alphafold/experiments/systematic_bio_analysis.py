#!/usr/bin/env python3
"""Systematic biological analysis of ET-Miner V2 frequent itemsets.

Cross-references mined patterns against ChEMBL drug targets, ClinVar
pathogenic variants, and Open Targets disease associations. Produces
categorized pattern tables for the V2 paper.

Usage:
    python3 systematic_bio_analysis.py \
        --parquet-dir /mnt/hdd/research/et-miner-data/project-milky-way/parquet \
        --data-dir /mnt/hdd/research/et-miner-data \
        --output results_bio_analysis.parquet
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_item_mapping


def load_chembl_targets(data_dir: Path) -> pl.DataFrame:
    """Load ChEMBL drug target annotations."""
    return pl.read_parquet(data_dir / "chembl" / "drug_target_annotations.parquet")


def load_clinvar(data_dir: Path) -> pl.DataFrame:
    """Load ClinVar pathogenic protein annotations."""
    return pl.read_parquet(data_dir / "clinvar" / "clinvar_protein_annotations.parquet")


def load_open_targets(data_dir: Path) -> dict[str, list[dict]]:
    """Load Open Targets disease association JSONs."""
    ot_dir = data_dir / "open_targets"
    diseases = {}
    for f in sorted(ot_dir.glob("disease_*.json")):
        disease_id = f.stem.replace("disease_", "")
        with open(f) as fh:
            diseases[disease_id] = json.load(fh)
    return diseases


def load_transactions_protein_index(data_dir: Path) -> pl.DataFrame:
    """Build protein_id → item_ids index from transactions."""
    logger.info("  Loading transactions (109M proteins)...")
    t0 = time.time()
    tx = pl.scan_parquet(
        data_dir / "project-milky-way" / "transactions_35k.parquet"
    ).select("protein_id", "items").collect(engine="streaming")
    logger.info(f"  Loaded {len(tx):,} transactions in {time.time()-t0:.1f}s")
    return tx


def build_protein_drug_target_set(chembl: pl.DataFrame) -> set[str]:
    """Get set of UniProt accessions that are known drug targets."""
    return set(
        chembl.filter(pl.col("is_known_drug_target"))["uniprot_accession"].to_list()
    )


def build_protein_clinvar_set(clinvar: pl.DataFrame) -> set[str]:
    """Get set of UniProt accessions with pathogenic variants."""
    return set(clinvar["uniprot_accession"].to_list())


def build_open_targets_protein_set(ot_diseases: dict) -> dict[str, set[str]]:
    """Build disease → protein set mapping from Open Targets."""
    result = {}
    for disease_id, associations in ot_diseases.items():
        proteins = set()
        for assoc in associations:
            for pid in assoc.get("target", {}).get("proteinIds", []):
                proteins.add(pid["id"])
        result[disease_id] = proteins
    return result


def analyze_itemset_biology(
    itemset: list[int],
    support: float,
    feature_map: dict[int, dict],
    drug_target_proteins: set[str],
    clinvar_proteins: set[str],
    tx_index: pl.DataFrame,
) -> dict:
    """Analyze a single itemset's biological significance.

    Returns dict with: features, categories, n_drug_targets, n_clinvar,
    has_enzyme, has_domain, has_go, pattern_type classification.
    """
    features = []
    categories = set()
    has_enzyme = False
    has_domain = False
    has_go = False

    for item_id in itemset:
        info = feature_map.get(item_id, {})
        features.append(info.get("feature_name", f"item_{item_id}"))
        cat = info.get("feature_category", "unknown")
        categories.add(cat)
        if cat == "ec_number":
            has_enzyme = True
        elif cat == "interpro":
            has_domain = True
        elif cat == "go_term":
            has_go = True

    # Find proteins carrying this exact itemset
    item_set = set(itemset)
    matching_proteins = tx_index.filter(
        pl.col("items").list.eval(
            pl.element().is_in(list(item_set))
        ).list.sum().list.first() >= len(item_set)
    )["protein_id"].to_list()

    n_drug_targets = len(set(matching_proteins) & drug_target_proteins)
    n_clinvar = len(set(matching_proteins) & clinvar_proteins)

    # Classify pattern type
    if n_drug_targets > 0 and has_enzyme:
        pattern_type = "drug_target_enzyme"
    elif n_drug_targets > 0:
        pattern_type = "drug_target"
    elif n_clinvar > 0:
        pattern_type = "disease_associated"
    elif has_domain and has_go:
        pattern_type = "functional_module"
    elif has_domain:
        pattern_type = "structural_motif"
    else:
        pattern_type = "annotation_pattern"

    return {
        "itemset": itemset,
        "k": len(itemset),
        "support": support,
        "features": features,
        "categories": sorted(categories),
        "n_proteins": len(matching_proteins),
        "n_drug_targets": n_drug_targets,
        "n_clinvar": n_clinvar,
        "has_enzyme": has_enzyme,
        "has_domain": has_domain,
        "has_go": has_go,
        "pattern_type": pattern_type,
    }


def analyze_k_level_summary(
    parquet_path: Path,
    feature_map: dict[int, dict],
    drug_target_proteins: set[str],
    clinvar_proteins: set[str],
    k: int,
    sample_n: int = 100,
) -> dict:
    """Analyze a K-level parquet with sampling for large files.

    For K-levels with millions of itemsets, analyze a stratified sample.
    Returns summary statistics.
    """
    pf = pq.ParquetFile(str(parquet_path))
    n_rows = pf.metadata.num_rows

    logger.info(f"  K={k}: {n_rows:,} itemsets")

    # Read sample
    if n_rows <= sample_n:
        df = pl.read_parquet(parquet_path)
    else:
        # Read first row group, sample from it
        df = pl.from_arrow(pf.read_row_group(0)).head(sample_n)

    # Analyze feature category distribution
    category_counts = {}

    for row in df.iter_rows(named=True):
        items = row["itemset"]
        for item_id in items:
            info = feature_map.get(item_id, {})
            cat = info.get("feature_category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

    return {
        "k": k,
        "n_itemsets": n_rows,
        "sample_size": len(df),
        "category_distribution": category_counts,
    }


def main():
    parser = argparse.ArgumentParser(description="Systematic biological analysis")
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results_bio_analysis.json"))
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--sample-per-k", type=int, default=100)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ET-Miner V2 — Systematic Biological Analysis")
    logger.info("=" * 60)

    # Load reference databases
    logger.info("\nLoading reference databases...")
    mapping_path = args.data_dir / "project-milky-way" / "item_mapping_35k_named.parquet"
    raw_mapping = load_item_mapping(mapping_path)
    feature_map = {
        item_id: {
            "feature_name": info["name"],
            "feature_category": info["category"],
            "description": info.get("description", ""),
        }
        for item_id, info in raw_mapping.items()
    }
    logger.info(f"  Feature mapping: {len(feature_map):,} features")

    chembl = load_chembl_targets(args.data_dir)
    drug_targets = build_protein_drug_target_set(chembl)
    logger.info(f"  ChEMBL drug targets: {len(drug_targets):,} proteins")

    clinvar = load_clinvar(args.data_dir)
    clinvar_proteins = build_protein_clinvar_set(clinvar)
    logger.info(f"  ClinVar pathogenic: {len(clinvar_proteins):,} proteins")

    ot_diseases = load_open_targets(args.data_dir)
    logger.info(f"  Open Targets: {len(ot_diseases)} therapeutic areas")

    # Analyze each K-level
    logger.info("\nAnalyzing K-levels...")
    k_summaries = []
    for k in range(1, args.max_k + 1):
        pq_path = args.parquet_dir / f"frequent_k{k}.parquet"
        if not pq_path.exists():
            logger.info(f"  K={k}: not found, skipping")
            continue

        try:
            summary = analyze_k_level_summary(
                pq_path, feature_map, drug_targets, clinvar_proteins, k, args.sample_per_k
            )
            k_summaries.append(summary)
        except Exception as e:
            logger.error(f"  K={k}: ERROR — {e}")
            continue

    # Output
    results = {
        "analysis_date": time.strftime("%Y-%m-%d %H:%M"),
        "n_features": len(feature_map),
        "n_drug_targets": len(drug_targets),
        "n_clinvar_proteins": len(clinvar_proteins),
        "n_therapeutic_areas": len(ot_diseases),
        "k_summaries": k_summaries,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults written to {args.output}")

    # Print summary table
    logger.info("\n" + "=" * 60)
    logger.info("K-LEVEL SUMMARY")
    logger.info("=" * 60)
    for s in k_summaries:
        cats = s["category_distribution"]
        top_cats = sorted(cats.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{c}:{n}" for c, n in top_cats)
        logger.info(f"  K={s['k']:>2}: {s['n_itemsets']:>15,} itemsets | top: {top_str}")


if __name__ == "__main__":
    main()
