#!/usr/bin/env python3
"""Resolve opaque GO and InterPro IDs to human-readable descriptions.

Reads item_mapping_35k.parquet (35,012 features) and adds a `description`
column by:
  - GO terms (5,763): resolved via pronto + go.obo from OBO Foundry
  - InterPro (12,773): resolved via EBI entry.list TSV
  - Keywords (15,162): extracted from feature_name (strip evidence codes)
  - EC numbers (1,107): already human-readable (kept as-is)
  - Taxonomy/length/pLDDT: kept as-is

Output: item_mapping_35k_named.parquet

Usage:
    python3 resolve_feature_names.py [--data-dir /mnt/d/research/et-miner-data/project-milky-way]
"""

import argparse
import re
import sys
import urllib.request
from pathlib import Path

import polars as pl
from loguru import logger


DEFAULT_DATA_DIR = Path("/mnt/d/research/et-miner-data/project-milky-way")


def resolve_go_terms(go_ids: list[str], cache_dir: Path) -> dict[str, str]:
    """Resolve GO IDs to names via pronto + go.obo."""
    try:
        import pronto
    except ImportError:
        logger.error("pronto not installed. Run: pip3 install pronto")
        return {}

    obo_path = cache_dir / "go.obo"

    if obo_path.exists():
        logger.info(f"  Using cached {obo_path}")
        onto = pronto.Ontology(str(obo_path))
    else:
        logger.info("  Downloading go.obo...")
        obo_path.parent.mkdir(parents=True, exist_ok=True)
        # OBO Foundry blocks bare urllib — need User-Agent header
        urls = [
            "https://release.geneontology.org/2024-06-17/ontology/go.obo",
            "http://purl.obolibrary.org/obo/go.obo",
            "https://current.geneontology.org/ontology/go.obo",
        ]
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "et-miner/1.0"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    with open(obo_path, "wb") as f:
                        f.write(resp.read())
                logger.info(f"  Downloaded from {url}")
                break
            except Exception as e:
                logger.warning(f"  {url} failed: {e}")
        onto = pronto.Ontology(str(obo_path))

    logger.info(f"  Loaded {len(onto)} ontology terms")

    resolved = {}
    found = 0
    for go_id in go_ids:
        if go_id in onto:
            resolved[go_id] = onto[go_id].name
            found += 1
        else:
            resolved[go_id] = ""

    logger.info(f"  Resolved {found}/{len(go_ids)} GO terms ({100*found/len(go_ids):.1f}%)")
    return resolved


def resolve_interpro(ipr_ids: list[str], cache_dir: Path) -> dict[str, str]:
    """Resolve InterPro IDs to names via EBI entry.list."""
    entry_path = cache_dir / "interpro_entry.list"

    if not entry_path.exists():
        logger.info("  Downloading InterPro entry.list from EBI...")
        url = "https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/entry.list"
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "et-miner/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            with open(entry_path, "wb") as f:
                f.write(resp.read())

    logger.info(f"  Parsing {entry_path}")
    ipr_map = {}
    with open(entry_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                ipr_id, entry_type, name = parts[0], parts[1], parts[2]
                ipr_map[ipr_id] = f"{name} [{entry_type}]"

    logger.info(f"  Loaded {len(ipr_map)} InterPro entries")

    resolved = {}
    found = 0
    for ipr_id in ipr_ids:
        if ipr_id in ipr_map:
            resolved[ipr_id] = ipr_map[ipr_id]
            found += 1
        else:
            resolved[ipr_id] = ""

    logger.info(f"  Resolved {found}/{len(ipr_ids)} InterPro IDs ({100*found/len(ipr_ids):.1f}%)")
    return resolved


def clean_keyword(feature_name: str) -> str:
    """Extract clean keyword name by stripping evidence codes.

    'KW:calcium binding {ECO:0000256|ARBA:AR...}' -> 'calcium binding'
    """
    # Remove KW: prefix
    name = feature_name[3:] if feature_name.startswith("KW:") else feature_name
    # Strip evidence codes in curly braces
    name = re.sub(r"\s*\{[^}]*\}", "", name).strip()
    return name


def main():
    parser = argparse.ArgumentParser(description="Resolve feature names for item mapping")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    input_path = args.data_dir / "item_mapping_35k.parquet"
    output_path = args.data_dir / "item_mapping_35k_named.parquet"

    logger.info(f"Loading {input_path}...")
    df = pl.read_parquet(input_path)
    logger.info(f"  {df.shape[0]} features, categories: {df['feature_category'].unique().to_list()}")

    # --- GO terms ---
    go_mask = df["feature_category"] == "go_term"
    go_features = df.filter(go_mask)["feature_name"].to_list()
    # feature_name is like "GO:0000009", which is also the GO ID
    logger.info(f"\nResolving {len(go_features)} GO terms...")
    go_names = resolve_go_terms(go_features, args.data_dir)

    # --- InterPro ---
    ipr_mask = df["feature_category"] == "interpro"
    ipr_features = df.filter(ipr_mask)["feature_name"].to_list()
    # feature_name is like "IPR:IPR000001" -> extract "IPR000001"
    ipr_ids = [f.split(":")[-1] if ":" in f else f for f in ipr_features]
    logger.info(f"\nResolving {len(ipr_ids)} InterPro IDs...")
    ipr_names = resolve_interpro(ipr_ids, args.data_dir)
    # Map back to feature_name keys
    ipr_name_map = {feat: ipr_names.get(ipr_id, "") for feat, ipr_id in zip(ipr_features, ipr_ids)}

    # --- Keywords ---
    kw_mask = df["feature_category"] == "keyword"
    kw_features = df.filter(kw_mask)["feature_name"].to_list()
    logger.info(f"\nCleaning {len(kw_features)} keyword names...")
    kw_names = {feat: clean_keyword(feat) for feat in kw_features}
    logger.info(f"  Done. Example: '{kw_features[0]}' -> '{kw_names[kw_features[0]]}'")

    # --- Build description column ---
    descriptions = []
    for row in df.iter_rows(named=True):
        cat = row["feature_category"]
        fname = row["feature_name"]

        if cat == "go_term":
            descriptions.append(go_names.get(fname, ""))
        elif cat == "interpro":
            descriptions.append(ipr_name_map.get(fname, ""))
        elif cat == "keyword":
            descriptions.append(kw_names.get(fname, ""))
        elif cat == "ec_number":
            # EC numbers are already readable
            descriptions.append(fname.replace("EC:", "EC "))
        elif cat == "taxonomy":
            descriptions.append(fname.replace("TAX:", ""))
        elif cat == "length_bin":
            descriptions.append(fname.replace("LEN:", "length "))
        elif cat == "plddt":
            descriptions.append(fname.replace("PLDDT:", "pLDDT "))
        else:
            descriptions.append("")

    df = df.with_columns(pl.Series("description", descriptions))

    # Stats
    filled = sum(1 for d in descriptions if d)
    logger.info(f"\n{'='*60}")
    logger.info(f"Resolution summary:")
    logger.info(f"  Total features: {len(descriptions)}")
    logger.info(f"  With description: {filled} ({100*filled/len(descriptions):.1f}%)")
    logger.info(f"  Missing: {len(descriptions) - filled}")

    # Show examples per category
    for cat in df["feature_category"].unique().to_list():
        subset = df.filter(pl.col("feature_category") == cat)
        filled_cat = subset.filter(pl.col("description") != "").shape[0]
        logger.info(f"  {cat}: {filled_cat}/{subset.shape[0]} resolved")

    logger.info(f"\nSaving to {output_path}...")
    df.write_parquet(output_path)
    logger.info(f"Done! {output_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
