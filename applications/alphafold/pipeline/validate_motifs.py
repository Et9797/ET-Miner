#!/usr/bin/env python3
"""Validate discovered motifs against known biology.

Cross-references frequent itemsets from ET-miner against:
  - Pfam domain co-occurrence in Swiss-Prot annotations
  - GO term consistency (shared ancestry heuristics)
  - PROSITE pattern associations
  - Structural feature + annotation consistency

Classifies each itemset as KNOWN_MATCH, PARTIALLY_KNOWN, or NOVEL.

Usage:
    python validate_motifs.py
    python validate_motifs.py --motifs results/motifs_0.01.json
    python validate_motifs.py --motifs results/motifs_0.01.json --prosite-dat data/annotations/prosite.dat
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import go_terms_related

import polars as pl

from loguru import logger


# ---------------------------------------------------------------------------
# PROSITE parser
# ---------------------------------------------------------------------------

def parse_prosite(path: Path) -> dict[str, dict]:
    """Parse prosite.dat into {accession: {id, description, cross_refs}}.

    Cross-refs (DR lines) are stored as sets of UniProt accessions.
    """
    entries = {}
    if not path.exists():
        logger.warning(f"PROSITE file not found: {path} — skipping PROSITE validation")
        return entries

    current = {}
    for line in path.open(encoding="utf-8", errors="replace"):
        if line.startswith("ID   "):
            current = {"id": line[5:].strip().rstrip(".")}
        elif line.startswith("AC   "):
            current["ac"] = line[5:].strip().rstrip(";")
        elif line.startswith("DE   "):
            current.setdefault("description", "")
            current["description"] += line[5:].strip() + " "
        elif line.startswith("DR   "):
            refs = current.setdefault("cross_refs", set())
            # DR lines: "P12345, NAME, T; P67890, NAME, T;"
            for chunk in line[5:].split(";"):
                chunk = chunk.strip()
                if chunk:
                    parts = chunk.split(",")
                    if parts:
                        refs.add(parts[0].strip())
        elif line.startswith("//"):
            if "ac" in current:
                current["description"] = current.get("description", "").strip()
                entries[current["ac"]] = current
            current = {}

    logger.info(f"Parsed {len(entries)} PROSITE entries")
    return entries


# ---------------------------------------------------------------------------
# Pfam co-occurrence matrix from annotations
# ---------------------------------------------------------------------------

def build_pfam_cooccurrence(annotations_path: Path) -> dict[tuple[str, str], int]:
    """Build pairwise Pfam domain co-occurrence counts from annotations parquet."""
    if not annotations_path.exists():
        logger.warning(f"Annotations file not found: {annotations_path} — skipping co-occurrence matrix")
        return {}

    df = pl.read_parquet(annotations_path)
    if "pfam_domains" not in df.columns:
        logger.warning("No pfam_domains column in annotations — skipping co-occurrence")
        return {}

    cooccurrence: dict[tuple[str, str], int] = Counter()
    for domains in df["pfam_domains"].to_list():
        if domains is None or len(domains) < 2:
            continue
        domains = sorted(set(domains))
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                cooccurrence[(domains[i], domains[j])] += 1

    logger.info(f"Built co-occurrence matrix: {len(cooccurrence)} unique Pfam pairs from {len(df)} proteins")
    return cooccurrence


# ---------------------------------------------------------------------------
# Interpretation generator
# ---------------------------------------------------------------------------

PLDDT_LABELS = {
    "plddt_mean_low": "low mean pLDDT",
    "plddt_mean_med": "medium mean pLDDT",
    "plddt_mean_high": "high mean pLDDT",
    "plddt_frac_high_low": "low fraction high-confidence residues",
    "plddt_frac_high_med": "moderate fraction high-confidence residues",
    "plddt_frac_high_high": "high fraction high-confidence residues",
}

SS_LABELS = {
    "mostly_helix": "predominantly helical",
    "mostly_sheet": "predominantly beta-sheet",
    "mostly_coil": "predominantly coil",
    "mixed_ss": "mixed secondary structure",
    "has_significant_helix": "significant helix content",
    "has_significant_sheet": "significant sheet content",
}


def generate_interpretation(features: list[str], categories: dict[str, int]) -> str:
    """Generate a human-readable interpretation of a feature combination."""
    parts = []

    # Structural features first
    for f in features:
        if f in PLDDT_LABELS:
            parts.append(PLDDT_LABELS[f] + " structures")
        elif f in SS_LABELS:
            parts.append(SS_LABELS[f])

    # Pfam domains
    pfam = [f for f in features if f.startswith("PF")]
    if pfam:
        parts.append("Pfam domain" + ("s " if len(pfam) > 1 else " ") + " + ".join(pfam))

    # GO terms
    go = [f for f in features if f.startswith("GO:")]
    if go:
        parts.append("GO term" + ("s " if len(go) > 1 else " ") + " + ".join(go))

    if not parts:
        return "Uncharacterized feature combination"

    return " with ".join(parts[:3])


# ---------------------------------------------------------------------------
# Main classification logic
# ---------------------------------------------------------------------------

def classify_itemset(
    features: list[str],
    pfam_cooccurrence: dict[tuple[str, str], int],
    prosite_entries: dict[str, dict],
    min_cooccurrence: int = 5,
) -> tuple[str, str]:
    """Classify an itemset and return (classification, evidence_string).

    Returns one of: KNOWN_MATCH, PARTIALLY_KNOWN, NOVEL.
    """
    SS_NAMES = {
        "mostly_helix", "mostly_sheet", "mostly_coil", "mixed_ss",
        "has_significant_helix", "has_significant_sheet",
    }
    pfam_features = [f for f in features if f.startswith("PF")]
    go_features = [f for f in features if f.startswith("GO:")]
    plddt_features = [f for f in features if f.startswith("plddt_")]
    ss_features = [f for f in features if f in SS_NAMES]

    known_pairs = 0
    total_pairs = 0
    evidence_parts = []

    # Check Pfam pair co-occurrences
    if len(pfam_features) >= 2:
        sorted_pfam = sorted(pfam_features)
        for i in range(len(sorted_pfam)):
            for j in range(i + 1, len(sorted_pfam)):
                total_pairs += 1
                pair = (sorted_pfam[i], sorted_pfam[j])
                count = pfam_cooccurrence.get(pair, 0)
                if count >= min_cooccurrence:
                    known_pairs += 1
                    evidence_parts.append(
                        f"{pair[0]}+{pair[1]} co-occur in {count} proteins"
                    )
                else:
                    evidence_parts.append(
                        f"{pair[0]}+{pair[1]} not commonly documented as co-occurring"
                    )

    # Check GO term relatedness
    if len(go_features) >= 2:
        total_pairs += 1
        if go_terms_related(go_features):
            known_pairs += 1
            evidence_parts.append("GO terms appear related (shared ontology branch)")
        else:
            evidence_parts.append("GO terms from different ontology branches")

    # Check PROSITE cross-references for Pfam domains
    if pfam_features and prosite_entries:
        for entry in prosite_entries.values():
            desc = entry.get("description", "").lower()
            matched = [pf for pf in pfam_features if pf.lower() in desc]
            if matched:
                known_pairs += 1
                total_pairs += 1
                evidence_parts.append(
                    f"PROSITE {entry['ac']} ({entry['id']}) references {', '.join(matched)}"
                )
                break  # One PROSITE hit is enough

    # Flag contradictory structural + annotation combos
    has_low_confidence = any("low" in f for f in plddt_features)
    has_specific_domain = len(pfam_features) > 0
    if has_low_confidence and has_specific_domain:
        evidence_parts.append(
            "Low-confidence structure with specific domain annotation — may indicate model uncertainty"
        )

    # Classify
    if total_pairs == 0:
        # Single-category or single-feature — classify based on feature count
        if len(features) == 1:
            return "KNOWN_MATCH", "Single feature — trivially known"
        if all(f.startswith("plddt_") or f.startswith("ss_") for f in features):
            return "KNOWN_MATCH", "Structural features only — expected co-occurrence"
        return "NOVEL", "; ".join(evidence_parts) if evidence_parts else "No known associations found"

    ratio = known_pairs / total_pairs
    evidence = "; ".join(evidence_parts)

    if ratio >= 0.8:
        return "KNOWN_MATCH", evidence
    elif ratio >= 0.3:
        return "PARTIALLY_KNOWN", evidence
    else:
        return "NOVEL", evidence


# ---------------------------------------------------------------------------
# Category breakdown
# ---------------------------------------------------------------------------

def categorize_features(features: list[str]) -> dict[str, int]:
    """Count features by category.

    Feature names from extract_features.py:
      plddt: plddt_mean_low, plddt_mean_med, plddt_mean_high,
             plddt_frac_high_low, plddt_frac_high_med, plddt_frac_high_high
      ss: mostly_helix, mostly_sheet, mostly_coil, mixed_ss,
          has_significant_helix, has_significant_sheet
      pfam: PF00069, PF00433, etc.
      go: GO:0005524, etc.
    """
    SS_NAMES = {
        "mostly_helix", "mostly_sheet", "mostly_coil", "mixed_ss",
        "has_significant_helix", "has_significant_sheet",
    }
    cats: dict[str, int] = defaultdict(int)
    for f in features:
        if f.startswith("plddt_"):
            cats["plddt"] += 1
        elif f in SS_NAMES:
            cats["ss"] += 1
        elif f.startswith("PF"):
            cats["pfam"] += 1
        elif f.startswith("GO:"):
            cats["go"] += 1
        else:
            cats["other"] += 1
    return dict(cats)


def itemset_type(cats: dict[str, int]) -> str:
    """Determine if an itemset is pfam_only, go_only, structural_only, or mixed."""
    annotation_cats = {k for k in cats if k in ("pfam", "go")}
    structural_cats = {k for k in cats if k in ("plddt", "ss")}

    if annotation_cats and not structural_cats:
        if annotation_cats == {"pfam"}:
            return "pfam_only"
        if annotation_cats == {"go"}:
            return "go_only"
        return "annotation_mixed"
    if structural_cats and not annotation_cats:
        return "structural_only"
    return "mixed"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate discovered motifs against known biology"
    )
    parser.add_argument(
        "--motifs", type=Path, default=Path("results/motifs_0.01.json"),
        help="Motifs JSON from run_mining.py",
    )
    parser.add_argument(
        "--item-mapping", type=Path, default=Path("data/processed/item_mapping.parquet"),
        help="Item mapping parquet",
    )
    parser.add_argument(
        "--annotations", type=Path, default=Path("data/annotations/uniprot_annotations.parquet"),
        help="UniProt annotations parquet",
    )
    parser.add_argument(
        "--prosite-dat", type=Path, default=Path("data/annotations/prosite.dat"),
        help="PROSITE .dat file",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/validation_report.json"),
        help="Output validation report",
    )
    parser.add_argument(
        "--min-cooccurrence", type=int, default=5,
        help="Minimum co-occurrence count to consider a Pfam pair as known",
    )
    args = parser.parse_args()

    # Load motifs
    if not args.motifs.exists():
        logger.error(f"Motifs file not found: {args.motifs}")
        return 1

    with open(args.motifs) as f:
        motifs_data = json.load(f)

    itemsets = motifs_data.get("itemsets", [])
    logger.info(f"Loaded {len(itemsets)} itemsets from {args.motifs}")

    if not itemsets:
        logger.warning("No itemsets to validate")
        return 0

    # Load item mapping (optional — features may already be decoded)
    item_map = {}
    if args.item_mapping.exists():
        mapping_df = pl.read_parquet(args.item_mapping)
        for row in mapping_df.iter_rows(named=True):
            item_map[row["item_id"]] = {
                "name": row["feature_name"],
                "category": row["feature_category"],
            }
        logger.info(f"Loaded item mapping: {len(item_map)} items")

    # Build Pfam co-occurrence matrix
    pfam_cooccurrence = build_pfam_cooccurrence(args.annotations)

    # Parse PROSITE
    prosite_entries = parse_prosite(args.prosite_dat)

    # Classify each itemset
    results = []
    classification_counts = Counter()
    category_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "known": 0, "novel": 0})

    for item in itemsets:
        features = item.get("features", [])
        # Fallback: decode from item_ids if features not present
        if not features and "item_ids" in item:
            features = [
                item_map[iid]["name"] for iid in item["item_ids"]
                if iid in item_map
            ]

        cats = categorize_features(features)
        itype = itemset_type(cats)
        classification, evidence = classify_itemset(
            features, pfam_cooccurrence, prosite_entries, args.min_cooccurrence
        )

        classification_counts[classification] += 1
        category_stats[itype]["count"] += 1
        if classification == "KNOWN_MATCH":
            category_stats[itype]["known"] += 1
        elif classification == "NOVEL":
            category_stats[itype]["novel"] += 1

        entry = {
            "features": features,
            "support": item.get("support", 0.0),
            "size": item.get("size", len(features)),
            "classification": classification,
            "category_breakdown": cats,
            "evidence": evidence,
            "interpretation": generate_interpretation(features, cats),
        }
        results.append(entry)

    # Separate by classification
    known = [r for r in results if r["classification"] == "KNOWN_MATCH"]
    partial = [r for r in results if r["classification"] == "PARTIALLY_KNOWN"]
    novel = [r for r in results if r["classification"] == "NOVEL"]

    # Rank novel candidates by support (higher = more interesting)
    novel_sorted = sorted(novel, key=lambda x: x["support"], reverse=True)
    top_novel = novel_sorted[:10]

    total = len(results)
    pct = lambda n: round(100 * n / total, 1) if total else 0.0

    summary = {
        "total_itemsets": total,
        "known_match": len(known),
        "partially_known": len(partial),
        "novel": len(novel),
        "pct_known": pct(len(known)),
        "pct_partially_known": pct(len(partial)),
        "pct_novel": pct(len(novel)),
    }

    report = {
        "summary": summary,
        "known_matches": known,
        "partially_known": partial,
        "novel_candidates": novel,
        "top_novel": top_novel,
        "category_stats": dict(category_stats),
    }

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Wrote validation report to {args.output}")

    # Print human-readable summary
    logger.info("")
    logger.info("=== Motif Validation Report ===")
    logger.info("")
    logger.info(f"Total itemsets validated: {total:,}")
    logger.info("")
    logger.info("Classification:")
    logger.info(f"  KNOWN_MATCH:       {len(known):>5,} ({pct(len(known)):>5.1f}%)  - matches known biology")
    logger.info(f"  PARTIALLY_KNOWN:   {len(partial):>5,} ({pct(len(partial)):>5.1f}%)  - some pairs known")
    logger.info(f"  NOVEL:             {len(novel):>5,} ({pct(len(novel)):>5.1f}%)  - new co-occurrence patterns")
    logger.info("")

    if top_novel:
        logger.info(f"Top-{len(top_novel)} Novel Candidates:")
        for i, cand in enumerate(top_novel, 1):
            feats = ", ".join(cand["features"])
            logger.info(f"  {i}. {{{feats}}} (support={cand['support']:.4f})")
            logger.info(f"     → {cand['interpretation']}")
        logger.info("")

    if category_stats:
        logger.info("Category breakdown:")
        for cat, stats in sorted(category_stats.items()):
            logger.info(f"  {cat:20s}: {stats['count']:>5,} total, "
                  f"{stats['known']:>5,} known, {stats['novel']:>5,} novel")
        logger.info("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
