#!/usr/bin/env python3
"""Extract structural and functional features from AlphaFold predictions.

Processes AlphaFold CIF files + UniProt annotations into ET-miner compatible
parquet transaction files for frequent itemset mining.

Features extracted per protein:
  - pLDDT confidence scores (mean, fraction high-confidence)
  - Secondary structure composition via DSSP
  - Pfam domain annotations (top-50 most frequent)
  - GO slim terms (top-30 most frequent)

Output: parquet files with integer-encoded feature vectors.
"""

import argparse
import gzip
import json
import time
from collections import Counter
from pathlib import Path

import polars as pl

try:
    from Bio.PDB import MMCIFParser, DSSP

    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

from loguru import logger

# ---------------------------------------------------------------------------
# Fixed item ID ranges
# ---------------------------------------------------------------------------
# pLDDT mean: 0=low(<50), 1=med(50-90), 2=high(>90)
PLDDT_MEAN_BASE = 0
# pLDDT fraction high: 3=<25%, 4=25-75%, 5=>75%
PLDDT_FRAC_BASE = 3
# Dominant SS: 6=mostly-helix, 7=mostly-sheet, 8=mostly-coil, 9=mixed
SS_DOMINANT_BASE = 6
# SS composition flags: 10=has_significant_helix, 11=has_significant_sheet
SS_COMP_BASE = 10
# Pfam domains: 12-61 (top 50)
PFAM_BASE = 12
PFAM_COUNT = 50
# GO slim terms: 62-91 (top 30)
GO_BASE = PFAM_BASE + PFAM_COUNT  # 62
GO_COUNT = 30

TOTAL_FIXED_ITEMS = GO_BASE + GO_COUNT  # 92


# ---------------------------------------------------------------------------
# Global encoding
# ---------------------------------------------------------------------------
def build_global_encoding(annotations_path: Path) -> tuple[dict[str, int], dict[str, int], pl.DataFrame]:
    """Build item ID mappings from annotation frequencies.

    Returns:
        (pfam_to_id, go_to_id, annotations_df)
    """
    logger.info(f"Loading annotations from {annotations_path}")
    df = pl.read_parquet(annotations_path)

    # Count Pfam domain frequencies
    pfam_counter: Counter[str] = Counter()
    go_counter: Counter[str] = Counter()

    pfam_col = "pfam_domains" if "pfam_domains" in df.columns else None
    go_col = "go_terms" if "go_terms" in df.columns else None

    if pfam_col:
        for domains in df[pfam_col].to_list():
            if domains is not None:
                pfam_counter.update(domains)

    if go_col:
        for terms in df[go_col].to_list():
            if terms is not None:
                go_counter.update(terms)

    # Top-N most frequent
    top_pfam = [domain for domain, _ in pfam_counter.most_common(PFAM_COUNT)]
    top_go = [term for term, _ in go_counter.most_common(GO_COUNT)]

    pfam_to_id = {domain: PFAM_BASE + i for i, domain in enumerate(top_pfam)}
    go_to_id = {term: GO_BASE + i for i, term in enumerate(top_go)}

    logger.info(
        "Global encoding: %d Pfam domains, %d GO terms (max item ID: %d)",
        len(pfam_to_id),
        len(go_to_id),
        TOTAL_FIXED_ITEMS - 1,
    )

    return pfam_to_id, go_to_id, df


def build_item_mapping(pfam_to_id: dict[str, int], go_to_id: dict[str, int]) -> pl.DataFrame:
    """Build the item_mapping dataframe for all feature item IDs."""
    rows = []

    # pLDDT mean bins
    for i, label in enumerate(["plddt_mean_low", "plddt_mean_med", "plddt_mean_high"]):
        rows.append({"item_id": PLDDT_MEAN_BASE + i, "feature_name": label, "feature_category": "plddt_mean"})

    # pLDDT fraction high bins
    for i, label in enumerate(["plddt_frac_high_low", "plddt_frac_high_med", "plddt_frac_high_high"]):
        rows.append({"item_id": PLDDT_FRAC_BASE + i, "feature_name": label, "feature_category": "plddt_fraction"})

    # Dominant SS
    for i, label in enumerate(["mostly_helix", "mostly_sheet", "mostly_coil", "mixed_ss"]):
        rows.append({"item_id": SS_DOMINANT_BASE + i, "feature_name": label, "feature_category": "ss_dominant"})

    # SS composition flags
    rows.append({"item_id": SS_COMP_BASE, "feature_name": "has_significant_helix", "feature_category": "ss_composition"})
    rows.append({"item_id": SS_COMP_BASE + 1, "feature_name": "has_significant_sheet", "feature_category": "ss_composition"})

    # Pfam domains
    for domain, item_id in sorted(pfam_to_id.items(), key=lambda x: x[1]):
        rows.append({"item_id": item_id, "feature_name": domain, "feature_category": "pfam"})

    # GO terms
    for term, item_id in sorted(go_to_id.items(), key=lambda x: x[1]):
        rows.append({"item_id": item_id, "feature_name": term, "feature_category": "go_term"})

    return pl.DataFrame(rows, schema={"item_id": pl.Int64, "feature_name": pl.Utf8, "feature_category": pl.Utf8})


# ---------------------------------------------------------------------------
# pLDDT extraction
# ---------------------------------------------------------------------------
def extract_plddt_features(cif_path: Path) -> tuple[list[int], float] | None:
    """Extract pLDDT features from AlphaFold mmCIF file.

    Parses _ma_qa_metric_local block for per-residue pLDDT scores.
    Also checks _ma_qa_metric_global for the global pLDDT value.

    Returns:
        (item_ids, mean_plddt) or None if data missing/invalid.
    """
    scores = []
    global_plddt = None
    in_local_block = False
    headers = []

    opener = gzip.open if cif_path.suffix == ".gz" else open
    with opener(cif_path, "rt", encoding="utf-8") as f:
        for line in f:
            if "_ma_qa_metric_local." in line:
                in_local_block = True
                headers.append(line.strip().split(".")[-1])
            elif "_ma_qa_metric_global.metric_value" in line:
                # _ma_qa_metric_global.metric_value 83.84
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        global_plddt = float(parts[-1])
                    except ValueError:
                        pass
            elif in_local_block:
                if line.startswith("#") or line.startswith("_"):
                    in_local_block = False
                    continue
                stripped = line.strip()
                if not stripped:
                    continue
                # Parse data row: "A MET 1   2 52.38 1 1"
                parts = stripped.split()
                if "metric_value" in headers:
                    val_idx = headers.index("metric_value")
                    if val_idx < len(parts):
                        try:
                            scores.append(float(parts[val_idx]))
                        except ValueError:
                            pass

    # Use per-residue scores if available, fall back to global
    if scores:
        mean_plddt = sum(scores) / len(scores)
        frac_high = sum(1 for s in scores if s > 90) / len(scores)
    elif global_plddt is not None:
        mean_plddt = global_plddt
        # Can't compute fraction from single value; estimate from mean
        frac_high = 1.0 if global_plddt > 90 else (0.5 if global_plddt > 70 else 0.1)
    else:
        return None

    items = []

    # Mean pLDDT bin
    if mean_plddt < 50:
        items.append(PLDDT_MEAN_BASE + 0)
    elif mean_plddt <= 90:
        items.append(PLDDT_MEAN_BASE + 1)
    else:
        items.append(PLDDT_MEAN_BASE + 2)

    # Fraction high-confidence bin
    if frac_high < 0.25:
        items.append(PLDDT_FRAC_BASE + 0)
    elif frac_high <= 0.75:
        items.append(PLDDT_FRAC_BASE + 1)
    else:
        items.append(PLDDT_FRAC_BASE + 2)

    return items, mean_plddt


# ---------------------------------------------------------------------------
# Secondary structure extraction
# ---------------------------------------------------------------------------
def extract_ss_features(cif_path: Path) -> list[int] | None:
    """Extract secondary structure features via DSSP from a CIF file.

    Returns item IDs or None if DSSP fails.
    """
    if not HAS_BIOPYTHON:
        logger.warning("BioPython not installed, skipping SS extraction")
        return None

    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", str(cif_path))
    except Exception as e:
        logger.debug(f"Failed to parse CIF {cif_path.name}: {e}")
        return None

    model = structure[0]

    # Try DSSP — use label_asym_id for mmCIF chain compatibility
    try:
        dssp = DSSP(model, str(cif_path), dssp="mkdssp")
    except Exception as e:
        logger.debug(f"DSSP failed for {cif_path.name}: {e}")
        return None

    if len(dssp) == 0:
        return None

    # Count SS types: H=helix, E=sheet, everything else=coil
    helix = 0
    sheet = 0
    coil = 0
    for _, (_, ss, *_rest) in dssp.property_dict.items():
        if ss in ("H", "G", "I"):
            helix += 1
        elif ss in ("E", "B"):
            sheet += 1
        else:
            coil += 1

    total = helix + sheet + coil
    if total == 0:
        return None

    frac_h = helix / total
    frac_e = sheet / total
    frac_c = coil / total

    items = []

    # Dominant SS
    if frac_h > 0.5:
        items.append(SS_DOMINANT_BASE + 0)  # mostly-helix
    elif frac_e > 0.5:
        items.append(SS_DOMINANT_BASE + 1)  # mostly-sheet
    elif frac_c > 0.5:
        items.append(SS_DOMINANT_BASE + 2)  # mostly-coil
    else:
        items.append(SS_DOMINANT_BASE + 3)  # mixed

    # Composition flags (>20% threshold)
    if frac_h > 0.2:
        items.append(SS_COMP_BASE + 0)  # has_significant_helix
    if frac_e > 0.2:
        items.append(SS_COMP_BASE + 1)  # has_significant_sheet

    return items


# ---------------------------------------------------------------------------
# Annotation features
# ---------------------------------------------------------------------------
def extract_annotation_features(
    uniprot_id: str,
    annotations: dict[str, dict],
    pfam_to_id: dict[str, int],
    go_to_id: dict[str, int],
) -> list[int]:
    """Map UniProt annotations to item IDs."""
    items = []
    annot = annotations.get(uniprot_id, {})

    for domain in annot.get("pfam_domains", []):
        if domain in pfam_to_id:
            items.append(pfam_to_id[domain])

    for term in annot.get("go_terms", []):
        if term in go_to_id:
            items.append(go_to_id[term])

    return items


# ---------------------------------------------------------------------------
# Per-protein worker
# ---------------------------------------------------------------------------
def find_cif_file(uniprot_id: str, data_dir: Path) -> Path | None:
    """Locate CIF file for a protein across raw subdirectories.

    Searches for both v4 and v6 naming conventions.
    """
    raw_dir = data_dir / "raw"
    for version in ("v6", "v4"):
        for ext in (".cif", ".cif.gz"):
            cif_name = f"AF-{uniprot_id}-F1-model_{version}{ext}"
            for subdir in raw_dir.iterdir():
                if not subdir.is_dir():
                    continue
                cif = subdir / cif_name
                if cif.exists():
                    return cif
    return None


def process_protein(
    uniprot_id: str,
    data_dir: Path,
    annotations: dict[str, dict],
    pfam_to_id: dict[str, int],
    go_to_id: dict[str, int],
    min_plddt: float,
    skip_dssp: bool = False,
) -> tuple[str, list[int]] | None:
    """Process a single protein: extract all features and encode as item IDs.

    Returns (uniprot_id, item_list) or None on skip/failure.
    """
    cif_path = find_cif_file(uniprot_id, data_dir)

    if cif_path is None:
        logger.debug(f"No CIF file found for {uniprot_id}, skipping")
        return None

    # pLDDT features from CIF (required — also used for filtering)
    plddt_result = extract_plddt_features(cif_path)
    if plddt_result is None:
        logger.debug(f"No pLDDT data in CIF for {uniprot_id}, skipping")
        return None

    plddt_items, mean_plddt = plddt_result

    if mean_plddt < min_plddt:
        logger.debug(f"pLDDT {mean_plddt:.1f} < {min_plddt:.1f} for {uniprot_id}, skipping")
        return None

    items = list(plddt_items)

    # Secondary structure features (optional — skip on DSSP failure or --no-dssp)
    if not skip_dssp:
        ss_items = extract_ss_features(cif_path)
        if ss_items is not None:
            items.extend(ss_items)

    # Annotation features
    annot_items = extract_annotation_features(uniprot_id, annotations, pfam_to_id, go_to_id)
    items.extend(annot_items)

    return (uniprot_id, sorted(set(items)))


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------
class Checkpoint:
    """Track processed protein IDs for resumable extraction."""

    def __init__(self, path: Path):
        self.path = path
        self.processed: set[str] = set()
        if path.exists():
            data = json.loads(path.read_text())
            self.processed = set(data.get("processed", []))
            logger.info(f"Resumed checkpoint: {len(self.processed)} already processed")

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"processed": sorted(self.processed)}, indent=2))

    def add(self, uniprot_id: str):
        self.processed.add(uniprot_id)

    def is_done(self, uniprot_id: str) -> bool:
        return uniprot_id in self.processed


# ---------------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------------
def load_annotations_dict(df: pl.DataFrame) -> dict[str, dict]:
    """Convert annotations DataFrame to lookup dict keyed by uniprot_id."""
    annotations: dict[str, dict] = {}
    id_col = "uniprot_id" if "uniprot_id" in df.columns else "accession"

    pfam_col = "pfam_domains" if "pfam_domains" in df.columns else None
    go_col = "go_terms" if "go_terms" in df.columns else None

    for row in df.iter_rows(named=True):
        uid = row[id_col]
        entry: dict[str, list] = {}
        if pfam_col and row.get(pfam_col) is not None:
            entry["pfam_domains"] = row[pfam_col]
        if go_col and row.get(go_col) is not None:
            entry["go_terms"] = row[go_col]
        annotations[uid] = entry

    return annotations


# ---------------------------------------------------------------------------
# Discovery: find all UniProt IDs with CIF files
# ---------------------------------------------------------------------------
def discover_proteins(data_dir: Path) -> list[str]:
    """Find all UniProt IDs that have CIF structure files.

    Searches data/raw/ recursively (download creates subdirs per proteome).
    """
    raw_dir = data_dir / "raw"
    if not raw_dir.exists():
        logger.error(f"Raw data directory not found: {raw_dir}")
        return []

    seen = set()
    ids = []
    for cif in raw_dir.rglob("AF-*-F1-model_v*.cif*"):
        # Extract UniProt ID from filename: AF-{id}-F1-model_v{4,6}.cif[.gz]
        name = cif.name.replace(".gz", "")
        parts = name.replace(".cif", "").split("-")
        if len(parts) >= 3:
            uniprot_id = parts[1]
            if uniprot_id not in seen:
                seen.add(uniprot_id)
                ids.append(uniprot_id)

    logger.info(f"Discovered {len(ids)} protein structures in {raw_dir}")
    return sorted(ids)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_extraction(args: argparse.Namespace):
    """Run the full feature extraction pipeline."""
    data_dir = Path(args.data_dir)
    output_dir = data_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = data_dir / "annotations" / "uniprot_annotations.parquet"
    if not annotations_path.exists():
        logger.error(f"Annotations not found: {annotations_path}")
        logger.error("Run the download step first to fetch UniProt annotations.")
        raise SystemExit(1)

    t0 = time.time()

    # Build global encoding
    pfam_to_id, go_to_id, annotations_df = build_global_encoding(annotations_path)
    annotations = load_annotations_dict(annotations_df)

    # Save item mapping
    item_mapping = build_item_mapping(pfam_to_id, go_to_id)
    item_mapping.write_parquet(output_dir / "item_mapping.parquet")
    logger.info(f"Saved item mapping ({len(item_mapping)} items)")

    # Discover proteins
    all_ids = discover_proteins(data_dir)
    if not all_ids:
        logger.error(f"No protein structures found in {data_dir}/structures/")
        raise SystemExit(1)

    # Checkpoint for resume
    checkpoint = Checkpoint(output_dir / ".checkpoint.json")
    if args.resume:
        remaining = [uid for uid in all_ids if not checkpoint.is_done(uid)]
        logger.info(f"Resume mode: {len(remaining)} remaining out of {len(all_ids)} total")
    else:
        remaining = all_ids
        checkpoint.processed.clear()

    # Process proteins (single-process to avoid pickle issues with large dicts)
    # Multiprocessing with shared dicts is fragile; for I/O-bound DSSP work,
    # the overhead of pickling 500K+ annotation entries negates parallelism gains.
    # Use workers>1 only when testing confirms it helps.
    results: list[tuple[str, list[int]]] = []
    errors = 0
    skipped = 0

    logger.info(f"Processing {len(remaining)} proteins...")

    for i, uid in enumerate(remaining):
        try:
            result = process_protein(
                uid, data_dir, annotations, pfam_to_id, go_to_id, args.min_plddt,
                skip_dssp=getattr(args, 'no_dssp', False),
            )
            if result is not None:
                results.append(result)
                checkpoint.add(result[0])
            else:
                skipped += 1
        except Exception as e:
            logger.warning(f"Error processing {uid}: {e}")
            errors += 1

        if (i + 1) % 1000 == 0:
            checkpoint.save()
            logger.info(
                "Progress: %d/%d processed, %d results, %d skipped, %d errors",
                i + 1, len(remaining), len(results), skipped, errors,
            )

    checkpoint.save()

    elapsed = time.time() - t0

    # Build transactions parquet (include protein_id for clustering step)
    if results:
        tx_df = pl.DataFrame(
            {
                "protein_id": [uid for uid, _ in results],
                "items": [items for _, items in results],
            },
            schema={"protein_id": pl.Utf8, "items": pl.List(pl.Int64)},
        )
        tx_path = output_dir / "alphafold_transactions.parquet"
        tx_df.write_parquet(tx_path)
        logger.info(f"Saved {len(tx_df)} transactions to {tx_path}")

        # Item count stats
        item_counts = [len(items) for _, items in results]
        avg_items = sum(item_counts) / len(item_counts)
    else:
        avg_items = 0.0
        logger.warning("No valid transactions produced")

    # Save extraction stats
    stats = {
        "total_structures": len(all_ids),
        "processed": len(results),
        "skipped": skipped,
        "errors": errors,
        "previously_done": len(all_ids) - len(remaining),
        "avg_items_per_protein": round(avg_items, 2),
        "pfam_domains_mapped": len(pfam_to_id),
        "go_terms_mapped": len(go_to_id),
        "elapsed_seconds": round(elapsed, 1),
        "workers": args.workers,
        "min_plddt": args.min_plddt,
    }
    stats_path = output_dir / "extraction_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    logger.info(
        f"Done in {elapsed:.1f}s: {len(results)} transactions, {skipped} skipped, {errors} errors, {avg_items:.1f} avg items/protein"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract structural features from AlphaFold predictions for ET-miner.",
        epilog="""Examples:
  %(prog)s --data-dir ./data
  %(prog)s --data-dir ./data --workers 16 --resume
  %(prog)s --data-dir ./data --min-plddt 70
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Root data directory (default: ./data)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (currently single-process only, reserved for future use)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint, skipping already-processed proteins",
    )
    parser.add_argument(
        "--min-plddt",
        type=float,
        default=50.0,
        help="Minimum mean pLDDT to include a protein (default: 50.0)",
    )
    parser.add_argument(
        "--no-dssp",
        action="store_true",
        help="Skip DSSP secondary structure extraction (faster, uses pLDDT + annotations only)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if not HAS_BIOPYTHON:
        logger.warning("BioPython not installed — secondary structure features will be skipped")
        logger.warning("Install with: pip install biopython")

    if args.no_dssp:
        logger.info("DSSP disabled via --no-dssp, using pLDDT + annotations only")

    run_extraction(args)


if __name__ == "__main__":
    main()
