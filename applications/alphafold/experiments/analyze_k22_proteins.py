#!/usr/bin/env python3
"""Analysis: deepest-itemset protein identification + GO hierarchy quantification.

Part 1: Identifies the proteins sharing all features of the deepest frequent
        itemset (Major 3: name the proteins behind the deepest pattern).
Part 2: Analyzes GO term parent-child relationships to quantify hierarchy
        inflation, reporting the "independent K" after collapsing redundant
        GO true-path pairs.

Generalized (2026-07): the deepest itemset is read from the run's itemsets
parquet via --itemset-source instead of the hardcoded Table-4 K=22 set. The
hardcoded set (FALLBACK_FEATURES) remains as a fallback when no source is given,
so old invocations keep working.

Addresses reviewer concerns M2 (GO true-path inflation) and M3 (deepest-itemset
proteins never named).
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_item_names

# Fallback: the K=22 itemset features from Table 4 of the v1 paper.
# Used only when --itemset-source is not given. Each tuple: (category, feature_id, description).
FALLBACK_FEATURES = [
    # Pfam domains (2)
    ("Pfam", "PF00270", "DEAD/DEAH helicase N-terminal"),
    ("Pfam", "PF00271", "Helicase C-terminal"),
    # Molecular Function (8)
    ("MF", "GO:0005524", "ATP binding"),
    ("MF", "GO:0016787", "Hydrolase activity"),
    ("MF", "GO:0000287", "Magnesium ion binding"),
    ("MF", "GO:0003697", "ssDNA binding"),
    ("MF", "GO:0003724", "RNA helicase activity"),
    ("MF", "GO:0003725", "dsRNA binding"),
    ("MF", "GO:0003678", "DNA helicase activity"),
    ("MF", "GO:0000978", "RNA Pol II regulatory binding"),
    # Biological Process (4)
    ("BP", "GO:0030154", "Cell differentiation"),
    ("BP", "GO:0045087", "Innate immune response"),
    ("BP", "GO:0051607", "Defense response to virus"),
    ("BP", "GO:0034605", "Cellular response to heat"),
    # Cellular Component (7)
    ("CC", "GO:0005737", "Cytoplasm"),
    ("CC", "GO:0005829", "Cytosol"),
    ("CC", "GO:0005634", "Nucleus"),
    ("CC", "GO:0005739", "Mitochondrion"),
    ("CC", "GO:0030424", "Axon"),
    ("CC", "GO:0030425", "Dendrite"),
    ("CC", "GO:0016607", "Nuclear speckle"),
    # Structural property (1)
    ("Struct", "plddt_mean_medium", "pLDDT 70-90"),
]

# Categories in the af-extract item_mapping (feature_category column) → paper labels.
_CATEGORY_LABELS = {
    "pfam": "Pfam",
    "go_term": "GO",
    "interpro": "InterPro",
    "ec_number": "EC",
    "taxonomy": "Taxonomy",
    "plddt_mean": "Struct",
    "plddt_fraction": "Struct",
}


def resolve_deepest_itemset(
    itemset_source: str,
    mapping_path: str,
) -> tuple[list[tuple[str, str, str]], list[int]]:
    """Read the deepest frequent itemset from the run and resolve it to features.

    Picks the itemset with the greatest length (ties broken by highest support),
    then maps each integer item_id to (category, feature_id, description) using the
    af-extract item_mapping parquet (columns: item_id, feature_name, feature_category).

    Returns (features, item_ids): the feature list in the same shape as
    FALLBACK_FEATURES (for display / GO analysis), and the raw integer item_ids of the
    itemset (for a direct, mapping-independent protein search — see find_itemset_proteins).
    """
    logger.info(f"Resolving deepest itemset from {itemset_source}")
    df = pl.read_parquet(itemset_source, columns=["itemset", "support"])
    if df.is_empty():
        raise ValueError(f"No itemsets in {itemset_source}")

    df = df.with_columns(pl.col("itemset").list.len().alias("_k"))
    kmax = int(df["_k"].max())
    deepest = (
        df.filter(pl.col("_k") == kmax)
        .sort("support", descending=True)
        .row(0, named=True)
    )
    item_ids = list(deepest["itemset"])
    logger.info(f"  Deepest itemset: K={kmax}, support={deepest['support']:.3e}, "
                f"item_ids={item_ids}")

    mapping = pl.read_parquet(mapping_path)
    id_to_name = dict(zip(mapping["item_id"].to_list(), mapping["feature_name"].to_list()))
    id_to_cat = dict(zip(mapping["item_id"].to_list(), mapping["feature_category"].to_list()))

    unmapped = [iid for iid in item_ids if iid not in id_to_name]
    if unmapped:
        logger.warning(
            f"  {len(unmapped)}/{len(item_ids)} item_ids not in {mapping_path}: {unmapped}. "
            "Item mapping likely does not match this itemset source — downstream protein "
            "search will find nothing. Check --item-mapping is from the SAME run."
        )

    features: list[tuple[str, str, str]] = []
    for iid in item_ids:
        name = id_to_name.get(iid, f"item_{iid}")
        raw_cat = id_to_cat.get(iid, "")
        label = _CATEGORY_LABELS.get(raw_cat, raw_cat or "Unknown")
        features.append((label, name, name))
    logger.info(f"  Resolved {len(features)} features "
                f"({sum(1 for f in features if f[1].startswith('GO:'))} GO terms)")
    return features, item_ids


def find_itemset_proteins(
    data_path: str,
    features: list[tuple[str, str, str]],
    mapping_path: str | None = None,
    target_ids: list[int] | None = None,
) -> list[dict]:
    """Find proteins containing all features of the target itemset.

    Args:
        data_path: Path to transactions parquet (must have protein_id column).
        features: List of (category, feature_id, description) tuples.
        mapping_path: Path to item_mapping (integer ID → feature name).
        target_ids: Raw integer item_ids of the itemset. When given (the
            --itemset-source path), the protein search filters on these ids DIRECTLY
            — no fragile name→id round trip, no silent fall-through to a broken
            fuzzy fallback. When None, uses the name-based path (FALLBACK_FEATURES).

    Returns:
        List of dicts with protein info.
    """
    n_feat = len(features)
    logger.info(f"\n{'='*60}")
    logger.info(f"Part 1: K={n_feat} Protein Identification")
    logger.info(f"{'='*60}")

    # Load transactions WITH protein_id
    logger.info(f"Loading {data_path}...")
    t0 = time.perf_counter()
    df = pl.read_parquet(data_path)
    load_time = time.perf_counter() - t0
    logger.info(f"  Loaded {len(df):,} proteins in {load_time:.1f}s")

    if "protein_id" not in df.columns:
        logger.error("  protein_id column missing from parquet file")
        logger.error(f"  Cannot identify K={n_feat} proteins without protein_id")
        return []

    # Direct-id path (--itemset-source): filter on the raw item_ids, mapping-independent.
    # This avoids the name→id round trip that could silently drop a feature and fall
    # through to a semantically-wrong fuzzy fallback (workflow finding 14).
    if target_ids is not None:
        logger.info(f"  Filtering for proteins with all {len(target_ids)} target item_ids (direct)...")
        filter_expr = pl.lit(True)
        for item_id in sorted(target_ids):
            filter_expr = filter_expr & pl.col("items").list.contains(item_id)
        matched = df.filter(filter_expr)
        if matched.is_empty():
            logger.warning(
                f"  0 proteins contain all {len(target_ids)} item_ids {sorted(target_ids)}. "
                "A frequent itemset must have >= min_count supporting proteins, so an empty "
                "result means the transactions parquet does not match the itemset source. "
                "Check --data and --itemset-source are from the SAME run."
            )
        return [
            {
                "protein_id": row["protein_id"],
                "n_features": len(row["items"]),
                "feature_ids": sorted(row["items"]),
            }
            for row in matched.iter_rows(named=True)
        ]

    # Load item mapping to resolve integer IDs to feature names
    if mapping_path:
        id_to_feature = load_item_names(mapping_path)
        feature_to_id = {v: k for k, v in id_to_feature.items()}
    else:
        # Try to find mapping file alongside transactions
        data_dir = Path(data_path).parent
        candidates = list(data_dir.glob("*item_mapping*")) + list(data_dir.glob("*feature_mapping*"))
        if candidates:
            mapping_file = str(candidates[0])
            logger.info(f"  Found mapping: {mapping_file}")
            id_to_feature = load_item_names(mapping_file)
            feature_to_id = {v: k for k, v in id_to_feature.items()}
        else:
            logger.warning("  No item mapping found. Will search by feature count.")
            id_to_feature = {}
            feature_to_id = {}

    # Strategy 1: If we have the mapping, find the integer IDs for all features
    if feature_to_id:
        target_ids = set()
        missing = []
        for cat, feat_id, desc in features:
            if feat_id in feature_to_id:
                target_ids.add(feature_to_id[feat_id])
            else:
                # Try partial match
                matches = [k for k, v in feature_to_id.items() if feat_id in k]
                if matches:
                    target_ids.add(feature_to_id[matches[0]])
                else:
                    missing.append(feat_id)

        if missing:
            logger.warning(f"  {len(missing)} features not found in mapping: {missing}")

        logger.info(f"  Resolved {len(target_ids)}/{n_feat} features to integer IDs")

        if len(target_ids) == n_feat:
            # Filter: find proteins whose items list contains ALL target IDs
            logger.info(f"  Filtering for proteins with all {n_feat} features...")
            target_list = sorted(target_ids)

            # Build filter expression: items must contain each ID
            filter_expr = pl.lit(True)
            for item_id in target_list:
                filter_expr = filter_expr & pl.col("items").list.contains(item_id)

            matched = df.filter(filter_expr)
            logger.info(f"  Found {len(matched):,} proteins with all {n_feat} features")

            results = []
            for row in matched.iter_rows(named=True):
                results.append({
                    "protein_id": row["protein_id"],
                    "n_features": len(row["items"]),
                    "feature_ids": sorted(row["items"]),
                })

            return results

    # Strategy 2: Find proteins with at least n_feat features (fallback)
    logger.info(f"  Fallback: searching for proteins with >= {n_feat} features...")
    df_deep = df.filter(pl.col("items").list.len() >= n_feat)
    logger.info(f"  Found {len(df_deep):,} proteins with >= {n_feat} features")

    # Among these, find the set that shares the most features
    if len(df_deep) > 0 and len(df_deep) <= 100:
        # Check which proteins share ALL their features
        first_items = set(df_deep.row(0, named=True)["items"])
        matching = []
        for row in df_deep.iter_rows(named=True):
            if set(row["items"]) == first_items:
                matching.append({
                    "protein_id": row["protein_id"],
                    "n_features": len(row["items"]),
                    "feature_ids": sorted(row["items"]),
                })

        logger.info(f"  {len(matching)} proteins share identical feature set")
        return matching

    return []


def analyze_go_hierarchy(
    output_dir: str,
    go_terms_list: list[str],
    n_non_go: int,
) -> dict:
    """Analyze GO term parent-child relationships in the target itemset.

    Downloads go.obo from OBO Foundry and checks which GO term pairs
    are linked by the true-path rule (ancestor-descendant relationship).

    Args:
        output_dir: Where to cache go.obo.
        go_terms_list: GO term IDs from the itemset under analysis.
        n_non_go: Number of non-GO features in the itemset (for independent-K).

    Returns:
        Dict with hierarchy analysis results.
    """
    logger.info(f"\n{'='*60}")
    logger.info("Part 2: GO Hierarchy Analysis")
    logger.info(f"{'='*60}")

    try:
        import pronto
    except ImportError:
        logger.error("  pronto not installed. Run: uv pip install pronto")
        logger.error("  Skipping GO hierarchy analysis.")
        return {"error": "pronto not installed"}

    # Download/load GO ontology
    logger.info("  Loading Gene Ontology (go.obo)...")
    script_dir = Path(__file__).parent
    obo_path = Path(output_dir) / "go.obo"

    # Check multiple locations for go.obo
    for candidate in [obo_path, script_dir / "go.obo"]:
        if candidate.exists():
            logger.info(f"  Using cached {candidate}")
            onto = pronto.Ontology(str(candidate))
            break
    else:
        logger.info("  Downloading from OBO Foundry...")
        try:
            onto = pronto.Ontology.from_obo_library("go.obo")
            # Cache for future use
            obo_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"  Download failed: {e}")
            logger.info("  Trying direct URL...")
            import urllib.request
            url = "http://purl.obolibrary.org/obo/go.obo"
            obo_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, str(obo_path))
            onto = pronto.Ontology(str(obo_path))

    logger.info(f"  Loaded {len(onto)} ontology terms")

    # Separate GO terms by namespace
    go_terms = {}
    for go_id in go_terms_list:
        if go_id in onto:
            term = onto[go_id]
            ns = term.namespace if hasattr(term, "namespace") else "unknown"
            go_terms[go_id] = {
                "name": term.name,
                "namespace": ns,
                "ancestors": {a.id for a in term.superclasses(distance=None, with_self=False)
                              if hasattr(a, "id") and a.id.startswith("GO:")},
            }
            logger.info(f"    {go_id}: {term.name} ({ns})")
        else:
            logger.warning(f"    {go_id}: NOT FOUND in ontology")
            go_terms[go_id] = {"name": "unknown", "namespace": "unknown", "ancestors": set()}

    # Find parent-child pairs within the itemset's GO set
    k22_go_set = set(go_terms_list)
    parent_child_pairs = []

    for child_id in go_terms_list:
        child_info = go_terms.get(child_id, {})
        child_ancestors = child_info.get("ancestors", set())

        for parent_id in go_terms_list:
            if parent_id == child_id:
                continue
            if parent_id in child_ancestors:
                parent_child_pairs.append({
                    "parent": parent_id,
                    "parent_name": go_terms[parent_id]["name"],
                    "child": child_id,
                    "child_name": go_terms[child_id]["name"],
                    "namespace": go_terms[child_id].get("namespace", "unknown"),
                })

    # Compute independent K
    # A term is "redundant" if it's an ancestor of another itemset GO term
    redundant_terms = set()
    for pair in parent_child_pairs:
        redundant_terms.add(pair["parent"])

    n_go_terms = len(go_terms_list)
    n_redundant = len(redundant_terms)
    n_independent_go = n_go_terms - n_redundant
    k_independent = n_non_go + n_independent_go
    k_total = n_go_terms + n_non_go

    logger.info("\n  GO hierarchy analysis:")
    logger.info(f"    Total features in itemset: {k_total}")
    logger.info(f"    GO terms:               {n_go_terms}")
    logger.info(f"    Non-GO features:        {n_non_go}")
    logger.info(f"    Parent-child pairs:     {len(parent_child_pairs)}")
    logger.info(f"    Redundant GO terms:     {n_redundant}")
    logger.info(f"    Independent GO terms:   {n_independent_go}")
    logger.info(f"    Independent K:          {k_independent}")

    if parent_child_pairs:
        logger.info("\n  Parent → Child relationships:")
        for p in parent_child_pairs:
            logger.info(f"    {p['parent']} ({p['parent_name']}) "
                  f"→ {p['child']} ({p['child_name']}) [{p['namespace']}]")

    if redundant_terms:
        logger.info("\n  Redundant terms (ancestors present in itemset):")
        for t in sorted(redundant_terms):
            logger.info(f"    {t}: {go_terms[t]['name']}")

    return {
        "n_go_terms": n_go_terms,
        "n_non_go_features": n_non_go,
        "n_parent_child_pairs": len(parent_child_pairs),
        "parent_child_pairs": parent_child_pairs,
        "redundant_terms": sorted(redundant_terms),
        "n_redundant": n_redundant,
        "n_independent_go": n_independent_go,
        "k_independent": k_independent,
        "go_term_details": {
            go_id: {
                "name": info["name"],
                "namespace": info.get("namespace", "unknown"),
                "n_ancestors_in_k22": len(info["ancestors"] & k22_go_set),
            }
            for go_id, info in go_terms.items()
        },
    }


def load_go_evidence(annotations_path: str) -> dict[tuple[str, str], str]:
    """Load {(accession, go_id): evidence_code} from a GOA GAF-format file.

    GAF 2.x is tab-separated; comment lines start with '!'. Relevant columns
    (0-indexed): col1 = DB Object ID (accession), col4 = GO ID, col6 = Evidence Code.
    Returns an empty dict on any parse failure (evidence codes are supplementary).
    """
    evidence: dict[tuple[str, str], str] = {}
    try:
        with open(annotations_path) as fh:
            for line in fh:
                if line.startswith("!") or not line.strip():
                    continue
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 7:
                    continue
                evidence[(cols[1], cols[4])] = cols[6]
    except OSError as e:
        logger.warning(f"  Could not read GO evidence file {annotations_path}: {e}")
    logger.info(f"  Loaded {len(evidence):,} (accession, GO) evidence codes")
    return evidence


def emit_accession_table(
    proteins: list[dict],
    features: list[tuple[str, str, str]],
    output_path: Path,
    go_evidence: dict[tuple[str, str], str] | None = None,
) -> None:
    """Write a supplementary TSV: one row per matched protein, with per-GO evidence.

    Columns: protein_id, n_features, <one column per GO term with its evidence code
    (or '' if unavailable)>. Answers Major 3: name the proteins and cite the GO
    evidence behind the deepest itemset.
    """
    go_ids = [f[1] for f in features if f[1].startswith("GO:")]
    header = ["protein_id", "n_features", *go_ids]
    lines = ["\t".join(header)]
    for p in proteins:
        row = [p["protein_id"], str(p["n_features"])]
        for go_id in go_ids:
            code = go_evidence.get((p["protein_id"], go_id), "") if go_evidence else ""
            row.append(code)
        lines.append("\t".join(row))
    output_path.write_text("\n".join(lines) + "\n")
    logger.info(f"  Accession table ({len(proteins)} proteins) written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Deepest-itemset protein identification + GO hierarchy analysis",
    )
    parser.add_argument(
        "--data", default="/workspace/data/transactions_214m_base.parquet",
        help="Path to transactions parquet with protein_id column",
    )
    parser.add_argument(
        "--mapping", "--item-mapping", dest="mapping", default=None,
        help="Path to item_mapping parquet (item_id, feature_name, feature_category)",
    )
    parser.add_argument(
        "--itemset-source", default=None,
        help="Frequent-itemsets parquet to read the deepest itemset from. "
        "If omitted, falls back to the hardcoded v1 K=22 features.",
    )
    parser.add_argument(
        "--annotations", default=None,
        help="Optional GOA GAF file for per-(accession, GO) evidence codes (supplementary table).",
    )
    parser.add_argument(
        "--output-dir", default="results_214m",
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
    )
    args = parser.parse_args()

    total_t0 = time.perf_counter()

    # Resolve the target itemset: from the run's parquet, or the v1 fallback.
    if args.itemset_source:
        if not args.mapping:
            logger.error("--itemset-source requires --mapping (item_mapping parquet)")
            return
        features, target_ids = resolve_deepest_itemset(args.itemset_source, args.mapping)
        source_label = f"deepest itemset from {args.itemset_source}"
    else:
        features = FALLBACK_FEATURES
        target_ids = None  # name-based path for the hardcoded fallback
        source_label = "hardcoded v1 K=22 features (no --itemset-source given)"
    logger.info(f"Analyzing {len(features)} features: {source_label}")

    go_terms_list = [f[1] for f in features if f[1].startswith("GO:")]
    n_non_go = sum(1 for f in features if not f[1].startswith("GO:"))

    # Part 1: Protein identification
    proteins = find_itemset_proteins(args.data, features, args.mapping, target_ids=target_ids)

    # Part 2: GO hierarchy analysis
    hierarchy = analyze_go_hierarchy(args.output_dir, go_terms_list, n_non_go)

    total_time = time.perf_counter() - total_t0

    # ── Save results ─────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"analysis_deepest_itemset_{timestamp}.json"

    # Supplementary accession table (Major 3) with optional GO evidence codes.
    if proteins:
        go_evidence = load_go_evidence(args.annotations) if args.annotations else None
        emit_accession_table(
            proteins, features,
            output_dir / f"accessions_deepest_itemset_{timestamp}.tsv",
            go_evidence,
        )

    output = {
        "analysis": "deepest_itemset_proteins_and_go_hierarchy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "itemset_source": args.itemset_source or "fallback_v1_k22",
        "k": len(features),
        "features": [
            {"category": cat, "id": fid, "description": desc}
            for cat, fid, desc in features
        ],
        "proteins": proteins,
        "n_proteins": len(proteins),
        "go_hierarchy": hierarchy,
        "total_time_seconds": round(total_time, 2),
    }

    output_file.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {output_file}")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    if proteins:
        print(f"  K={len(features)} proteins found: {len(proteins)}")
        for p in proteins:
            print(f"    {p['protein_id']} ({p['n_features']} features)")
    else:
        print("  Proteins: could not identify (missing mapping or protein_id)")

    if "k_independent" in hierarchy:
        print(f"  GO parent-child pairs: {hierarchy['n_parent_child_pairs']}")
        print(f"  Independent K: {hierarchy['k_independent']} "
              f"(raw K={len(features)}, {hierarchy['n_redundant']} redundant GO terms)")


if __name__ == "__main__":
    main()
