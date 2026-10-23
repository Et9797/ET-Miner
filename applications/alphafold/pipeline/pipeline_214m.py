#!/usr/bin/env python3
"""
Full-scale AlphaFold 214M protein pipeline.

Orchestrates the Rust af-extract binary for high-performance feature extraction
from the complete AlphaFold database (214M proteins, ~23 TiB uncompressed).

Key insight: We stream only confidence_v4.json files (~200-500 bytes each)
from proteome tars, avoiding the full 23 TiB CIF download. Combined with
UniProt bulk annotations (TSV, ~20GB), this gives us pLDDT + Pfam + GO
features for all 214M proteins.

Two-pass design:
  1. count-frequencies: Stream all tars → count Pfam/GO occurrences
  2. build-transactions: Stream again → select top-N → write parquet

Usage:
    # Full pipeline on a 48-core CPU instance
    python pipeline_214m.py --data-dir /data/alphafold --all

    # Individual steps
    python pipeline_214m.py --data-dir /data/alphafold --step download-annotations
    python pipeline_214m.py --data-dir /data/alphafold --step download-tars
    python pipeline_214m.py --data-dir /data/alphafold --step extract
    python pipeline_214m.py --data-dir /data/alphafold --step cluster
    python pipeline_214m.py --data-dir /data/alphafold --step verify

    # Batch mode for disk-limited instances (process N tars at a time)
    python pipeline_214m.py --data-dir /data/alphafold --all --batch-size 50
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# AlphaFold DB accession list (proteome → tar mapping)
ALPHAFOLD_FTP_BASE = "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest"
ALPHAFOLD_ACCESSION_LIST = f"{ALPHAFOLD_FTP_BASE}/accession_ids.csv"

# UniProt TrEMBL annotations (bulk TSV via REST)
UNIPROT_REST_BASE = "https://rest.uniprot.org/uniprotkb/stream"
UNIPROT_TREMBL_FIELDS = "accession,xref_interpro,go_id,ec,keyword,lineage,length"
UNIPROT_TREMBL_QUERY = "database:alphafolddb AND reviewed:false"
UNIPROT_SPROT_QUERY = "reviewed:true"

# UniProt DAT files (alternative to REST)
UNIPROT_TREMBL_DAT = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.dat.gz"
UNIPROT_SPROT_DAT = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz"

# AlphaFold proteome tar listing
ALPHAFOLD_PROTEOMES_URL = f"{ALPHAFOLD_FTP_BASE}/download_metadata.json"

# AF-extract binary (built from Rust)
AF_EXTRACT_BIN = "af-extract"


def find_af_extract() -> str:
    """Find the af-extract binary."""
    # Check in common locations
    candidates = [
        Path(__file__).parent.parent / "af-extract" / "target" / "release" / "af-extract",
        Path(__file__).parent.parent / "af-extract" / "target" / "debug" / "af-extract",
        shutil.which(AF_EXTRACT_BIN),
    ]

    for path in candidates:
        if path and Path(str(path)).exists():
            return str(path)

    logger.error(
        "af-extract binary not found.\n"
        "Build it with: cd applications/alphafold/af-extract && cargo build --release"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Step 1: Download annotations
# ---------------------------------------------------------------------------
def download_annotations(data_dir: Path, use_dat: bool = False):
    """Download UniProt annotations (Pfam domains, GO terms).

    Two strategies:
    1. REST API TSV (~20GB, faster, specific fields)
    2. DAT files (~100GB uncompressed, complete but slower)
    """
    annotations_dir = data_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    if use_dat:
        _download_dat_annotations(annotations_dir)
    else:
        _download_rest_annotations(annotations_dir)


def _download_rest_annotations(annotations_dir: Path):
    """Download annotations via UniProt REST API as TSV."""
    # Swiss-Prot annotations
    sprot_tsv = annotations_dir / "uniprot_sprot.tsv"
    if not sprot_tsv.exists():
        logger.info("Downloading Swiss-Prot annotations via REST API...")
        _download_uniprot_tsv(
            query="reviewed:true",
            fields=UNIPROT_TREMBL_FIELDS,
            output=sprot_tsv,
        )
    else:
        logger.info(f"Cached: {sprot_tsv.name}")

    # TrEMBL annotations (the big one — 200M+ entries)
    trembl_tsv = annotations_dir / "uniprot_trembl.tsv"
    if not trembl_tsv.exists():
        logger.info("Downloading TrEMBL annotations via REST API (~20GB)...")
        logger.info("This may take 30-60 minutes depending on connection speed.")
        _download_uniprot_tsv(
            query=UNIPROT_TREMBL_QUERY,
            fields=UNIPROT_TREMBL_FIELDS,
            output=trembl_tsv,
        )
    else:
        logger.info(f"Cached: {trembl_tsv.name}")


def _download_uniprot_tsv(query: str, fields: str, output: Path):
    """Download UniProt data as TSV via REST streaming endpoint."""
    import urllib.request

    url = (
        f"{UNIPROT_REST_BASE}?"
        f"query={urllib.parse.quote(query)}&"
        f"fields={urllib.parse.quote(fields)}&"
        f"format=tsv"
    )

    tmp = output.with_suffix(".tmp")
    try:
        logger.info(f"  URL: {url[:120]}...")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(output)
        size_gb = output.stat().st_size / (1024**3)
        logger.info(f"  Saved: {output.name} ({size_gb:.1f} GB)")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _download_dat_annotations(annotations_dir: Path):
    """Download Swiss-Prot + TrEMBL DAT files (alternative to REST)."""
    import urllib.request

    for name, url in [
        ("uniprot_sprot.dat.gz", UNIPROT_SPROT_DAT),
        ("uniprot_trembl.dat.gz", UNIPROT_TREMBL_DAT),
    ]:
        dest = annotations_dir / name
        if dest.exists():
            logger.info(f"Cached: {name}")
            continue

        logger.info(f"Downloading {name}...")
        tmp = dest.with_suffix(".tmp")
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)
            size_gb = dest.stat().st_size / (1024**3)
            logger.info(f"  Saved: {name} ({size_gb:.1f} GB)")
        except Exception:
            tmp.unlink(missing_ok=True)
            raise


# ---------------------------------------------------------------------------
# Step 2: Download AlphaFold proteome tars
# ---------------------------------------------------------------------------
def download_tars(data_dir: Path, batch_size: int = 0):
    """Download AlphaFold proteome tar files from EBI FTP.

    Uses gsutil for fast parallel downloads if available,
    falls back to wget/curl otherwise.
    """
    tar_dir = data_dir / "tars"
    tar_dir.mkdir(parents=True, exist_ok=True)

    # Get proteome list
    proteomes = _get_proteome_list(data_dir)
    logger.info(f"Total proteomes: {len(proteomes)}")

    # Check which are already downloaded
    existing = {p.stem for p in tar_dir.glob("*.tar")}
    remaining = [p for p in proteomes if p["name"] not in existing]
    logger.info(f"Already downloaded: {len(existing)}, remaining: {len(remaining)}")

    if not remaining:
        logger.info("All proteome tars already downloaded.")
        return

    # Download
    for i, proteome in enumerate(remaining):
        url = f"{ALPHAFOLD_FTP_BASE}/{proteome['file']}"
        dest = tar_dir / proteome["file"]

        logger.info(f"[{i+1}/{len(remaining)}] Downloading {proteome['file']}...")
        _download_file(url, dest)

        if batch_size > 0 and (i + 1) % batch_size == 0:
            logger.info(f"Batch of {batch_size} complete. Pausing for extraction...")
            break


def _get_proteome_list(data_dir: Path) -> list[dict]:
    """Get list of AlphaFold proteome tars from EBI metadata."""
    cache = data_dir / "proteome_list.json"
    if cache.exists():
        with open(cache) as f:
            return json.load(f)

    logger.info("Fetching AlphaFold proteome list...")
    import urllib.request

    # Download accession list which maps proteome → tar file
    acc_list_path = data_dir / "accession_ids.csv"
    if not acc_list_path.exists():
        urllib.request.urlretrieve(ALPHAFOLD_ACCESSION_LIST, acc_list_path)

    # Parse unique proteome tar names from accession list
    # Format: AF-{accession}-F1,{proteome_tar_name},{version}
    proteomes = {}
    with open(acc_list_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                tar_name = parts[1].strip()
                if tar_name and tar_name.endswith(".tar") and tar_name not in proteomes:
                    proteomes[tar_name] = {
                        "file": tar_name,
                        "name": tar_name.replace(".tar", ""),
                    }

    result = list(proteomes.values())
    with open(cache, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Found {len(result)} proteome tars")
    return result


def _download_file(url: str, dest: Path):
    """Download a file with resume support via curl."""
    if dest.exists():
        logger.info(f"  Cached: {dest.name}")
        return

    tmp = dest.with_suffix(".tmp")

    # Prefer curl for resume support
    if shutil.which("curl"):
        cmd = ["curl", "-L", "-C", "-", "-o", str(tmp), url]
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode == 0:
            tmp.rename(dest)
            return

    # Fallback to wget
    if shutil.which("wget"):
        cmd = ["wget", "-c", "-O", str(tmp), url]
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode == 0:
            tmp.rename(dest)
            return

    # Fallback to Python urllib
    import urllib.request
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


# ---------------------------------------------------------------------------
# Step 3: Run af-extract (Rust binary)
# ---------------------------------------------------------------------------
def run_extract(data_dir: Path, top_pfam: int = 200, top_go: int = 200,
                top_interpro: int = 0, top_ec: int = 0, top_taxonomy: int = 0,
                min_plddt: float = 50.0):
    """Run the Rust af-extract binary (two-pass)."""
    af_extract = find_af_extract()
    tar_dir = data_dir / "tars"
    output_dir = data_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir = data_dir / "annotations"

    # Find annotation file (prefer TSV, fall back to DAT.gz)
    annot_file = None
    for candidate in [
        annotations_dir / "uniprot_trembl.tsv",
        annotations_dir / "uniprot_sprot.tsv",
        annotations_dir / "uniprot_trembl.dat.gz",
        annotations_dir / "uniprot_sprot.dat.gz",
    ]:
        if candidate.exists():
            annot_file = candidate
            break

    if annot_file is None:
        logger.error("No annotation file found. Run --step download-annotations first.")
        sys.exit(1)

    logger.info(f"Using annotations: {annot_file.name}")

    freq_file = output_dir / "frequencies.json"
    transactions_file = output_dir / "alphafold_transactions_214m.parquet"
    mapping_file = output_dir / "item_mapping_214m.parquet"

    # Pass 1: Count frequencies
    if not freq_file.exists():
        logger.info("=" * 60)
        logger.info("Pass 1: Count frequencies")
        logger.info("=" * 60)
        t0 = time.time()

        cmd = [
            af_extract, "count-frequencies",
            "--annotations", str(annot_file),
            "--tar-dir", str(tar_dir),
            "--output", str(freq_file),
            "--min-plddt", str(min_plddt),
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            logger.error("af-extract count-frequencies failed")
            sys.exit(1)

        logger.info(f"Pass 1 complete in {time.time() - t0:.1f}s")
    else:
        logger.info(f"Cached: {freq_file.name}")

    # Pass 2: Build transactions
    if not transactions_file.exists():
        logger.info("=" * 60)
        logger.info("Pass 2: Build transactions")
        logger.info("=" * 60)
        t0 = time.time()

        cmd = [
            af_extract, "build-transactions",
            "--annotations", str(annot_file),
            "--frequencies", str(freq_file),
            "--tar-dir", str(tar_dir),
            "--output", str(transactions_file),
            "--item-mapping", str(mapping_file),
            "--top-pfam", str(top_pfam),
            "--top-go", str(top_go),
            "--top-interpro", str(top_interpro),
            "--top-ec", str(top_ec),
            "--top-taxonomy", str(top_taxonomy),
            "--min-plddt", str(min_plddt),
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            logger.error("af-extract build-transactions failed")
            sys.exit(1)

        logger.info(f"Pass 2 complete in {time.time() - t0:.1f}s")
    else:
        logger.info(f"Cached: {transactions_file.name}")


# ---------------------------------------------------------------------------
# Step 3b: Build from metadata (no tars — BigQuery pLDDT + TrEMBL annotations)
# ---------------------------------------------------------------------------
def run_extract_from_metadata(
    data_dir: Path,
    top_pfam: int = 200,
    top_go: int = 200,
    top_interpro: int = 0,
    top_ec: int = 0,
    top_taxonomy: int = 0,
    min_plddt: float = 50.0,
):
    """Build transactions from BigQuery pLDDT CSV + UniProt annotations (no tar files).

    This avoids downloading 23TB of AlphaFold CIF structures entirely.
    pLDDT comes from BigQuery metadata, Pfam/GO from TrEMBL DAT file.
    """
    af_extract = find_af_extract()
    output_dir = data_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir = data_dir / "annotations"

    # Find annotation file
    annot_file = None
    for candidate in [
        annotations_dir / "uniprot_trembl.dat.gz",
        annotations_dir / "uniprot_sprot.dat.gz",
        annotations_dir / "uniprot_trembl.tsv",
        annotations_dir / "uniprot_sprot.tsv",
    ]:
        if candidate.exists():
            annot_file = candidate
            break

    if annot_file is None:
        logger.error("No annotation file found. Run --step download-annotations first.")
        sys.exit(1)

    # Find pLDDT CSV (from BigQuery export)
    plddt_csv = None
    for candidate in [
        data_dir / "plddt_metadata.csv",
        data_dir / "plddt_metadata.csv.gz",
        data_dir / "alphafold_metadata.csv",
        data_dir / "alphafold_metadata.csv.gz",
    ]:
        if candidate.exists():
            plddt_csv = candidate
            break

    if plddt_csv is None:
        logger.error(
            "No pLDDT metadata CSV found.\n"
            "Export from BigQuery:\n"
            "  bq query --format=csv --max_rows=300000000 \\\n"
            "    'SELECT uniprotAccession, globalMetricValue AS mean_plddt\n"
            "     FROM `bigquery-public-data.deepmind_alphafold.metadata`' \\\n"
            "    > plddt_metadata.csv"
        )
        sys.exit(1)

    logger.info(f"Using annotations: {annot_file.name}")
    logger.info(f"Using pLDDT CSV:   {plddt_csv.name}")

    transactions_file = output_dir / "alphafold_transactions_214m.parquet"
    mapping_file = output_dir / "item_mapping_214m.parquet"

    if not transactions_file.exists():
        logger.info("=" * 60)
        logger.info("Build from metadata (no tars)")
        logger.info("=" * 60)
        t0 = time.time()

        cmd = [
            af_extract, "build-from-metadata",
            "--annotations", str(annot_file),
            "--plddt-csv", str(plddt_csv),
            "--output", str(transactions_file),
            "--item-mapping", str(mapping_file),
            "--top-pfam", str(top_pfam),
            "--top-go", str(top_go),
            "--top-interpro", str(top_interpro),
            "--top-ec", str(top_ec),
            "--top-taxonomy", str(top_taxonomy),
            "--min-plddt", str(min_plddt),
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            logger.error("af-extract build-from-metadata failed")
            sys.exit(1)

        logger.info(f"Build complete in {time.time() - t0:.1f}s")
    else:
        logger.info(f"Cached: {transactions_file.name}")


# ---------------------------------------------------------------------------
# Step 4: Cluster (optional)
# ---------------------------------------------------------------------------
def run_cluster(data_dir: Path, identity: float = 0.3, workers: int = 16):
    """Run MMseqs2 clustering on the extracted sequences (optional)."""
    logger.info("=" * 60)
    logger.info(f"Clustering at {identity*100:.0f}% identity")
    logger.info("=" * 60)

    # Delegate to existing cluster_sequences.py
    cluster_script = Path(__file__).parent / "cluster_sequences.py"
    if not cluster_script.exists():
        logger.error(f"Clustering script not found: {cluster_script}")
        sys.exit(1)

    cmd = [
        sys.executable, str(cluster_script),
        "--data-dir", str(data_dir),
        "--identity", str(identity),
        "--workers", str(workers),
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.error("Clustering failed")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Step 5: Verify output
# ---------------------------------------------------------------------------
def verify_output(data_dir: Path):
    """Verify the output parquet is compatible with existing GPU mining pipeline."""
    output_dir = data_dir / "processed"

    try:
        import polars as pl
    except ImportError:
        logger.error("polars not installed. Install with: uv pip install polars")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Verification")
    logger.info("=" * 60)

    # Check transactions parquet
    tx_path = output_dir / "alphafold_transactions_214m.parquet"
    if not tx_path.exists():
        logger.error(f"Transactions not found: {tx_path}")
        sys.exit(1)

    tx_df = pl.read_parquet(tx_path)
    logger.info(f"Transactions: {tx_df.height:,} proteins")
    logger.info(f"Schema: {tx_df.schema}")

    # Verify schema
    assert "protein_id" in tx_df.columns, "Missing protein_id column"
    assert "items" in tx_df.columns, "Missing items column"
    assert tx_df["protein_id"].dtype == pl.Utf8, f"protein_id should be Utf8, got {tx_df['protein_id'].dtype}"
    assert tx_df["items"].dtype == pl.List(pl.Int64), f"items should be List(Int64), got {tx_df['items'].dtype}"

    # Item statistics
    item_counts = tx_df["items"].list.len()
    logger.info(f"Items/protein: mean={item_counts.mean():.1f}, min={item_counts.min()}, max={item_counts.max()}")

    # Check item mapping
    mapping_path = output_dir / "item_mapping_214m.parquet"
    if mapping_path.exists():
        mapping_df = pl.read_parquet(mapping_path)
        logger.info(f"Item mapping: {mapping_df.height} items")

        # Category breakdown
        for cat in mapping_df["feature_category"].unique().sort():
            n = mapping_df.filter(pl.col("feature_category") == cat).height
            logger.info(f"  {cat}: {n} items")

    # Verify GPU compatibility — load without protein_id (as mining pipeline does)
    tx_mining = tx_df.drop("protein_id")
    logger.info(f"GPU-ready transactions: {tx_mining.height:,} rows, schema: {tx_mining.schema}")

    logger.info("")
    logger.info("Verification PASSED")
    logger.info(f"Ready for GPU mining: python run_mining.py --transactions {tx_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Full-scale AlphaFold 214M protein feature extraction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  download-annotations  Download UniProt Pfam/GO annotations (~20GB)
  download-tars         Download AlphaFold proteome tar archives
  extract               Run af-extract (Rust, two-pass)
  cluster               Optional MMseqs2 clustering
  verify                Verify output compatibility

Examples:
  %(prog)s --data-dir /data/alphafold --all
  %(prog)s --data-dir /data/alphafold --step download-annotations
  %(prog)s --data-dir /data/alphafold --step extract --top-pfam 200 --top-go 200 --top-interpro 300

Build Rust binary first:
  cd applications/alphafold/af-extract && cargo build --release
""",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root data directory for all pipeline files",
    )
    parser.add_argument(
        "--step",
        choices=[
            "download-annotations", "download-tars", "extract",
            "extract-from-metadata", "cluster", "verify",
        ],
        help="Run a single step (default: all steps)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all steps sequentially",
    )
    parser.add_argument(
        "--use-dat",
        action="store_true",
        help="Download DAT files instead of REST TSV for annotations",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Process N proteome tars at a time (0 = all at once)",
    )
    parser.add_argument(
        "--top-pfam",
        type=int,
        default=200,
        help="Number of top Pfam domains to include (default: 200)",
    )
    parser.add_argument(
        "--top-go",
        type=int,
        default=200,
        help="Number of top GO terms to include (default: 200)",
    )
    parser.add_argument(
        "--top-interpro",
        type=int,
        default=0,
        help="Number of top InterPro families to include (default: 0 = skip)",
    )
    parser.add_argument(
        "--top-ec",
        type=int,
        default=0,
        help="Number of top EC numbers to include (default: 0 = skip)",
    )
    parser.add_argument(
        "--top-taxonomy",
        type=int,
        default=0,
        help="Number of top taxonomy bins to include (default: 0 = skip)",
    )
    parser.add_argument(
        "--min-plddt",
        type=float,
        default=50.0,
        help="Minimum mean pLDDT threshold (default: 50.0)",
    )
    parser.add_argument(
        "--identity",
        type=float,
        default=0.3,
        help="MMseqs2 clustering identity threshold (default: 0.3)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel workers for clustering (default: 16)",
    )

    args = parser.parse_args()

    if not args.step and not args.all:
        parser.print_help()
        logger.info("\nSpecify --step <name> or --all to run the pipeline.")
        sys.exit(1)

    args.data_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    steps = []
    if args.all:
        steps = ["download-annotations", "download-tars", "extract", "verify"]
    elif args.step:
        steps = [args.step]

    for step in steps:
        logger.info("")
        logger.info(f"{'='*60}")
        logger.info(f"  STEP: {step}")
        logger.info(f"{'='*60}")
        logger.info("")

        if step == "download-annotations":
            download_annotations(args.data_dir, use_dat=args.use_dat)
        elif step == "download-tars":
            download_tars(args.data_dir, batch_size=args.batch_size)
        elif step == "extract":
            run_extract(
                args.data_dir,
                top_pfam=args.top_pfam,
                top_go=args.top_go,
                top_interpro=args.top_interpro,
                top_ec=args.top_ec,
                top_taxonomy=args.top_taxonomy,
                min_plddt=args.min_plddt,
            )
        elif step == "extract-from-metadata":
            run_extract_from_metadata(
                args.data_dir,
                top_pfam=args.top_pfam,
                top_go=args.top_go,
                top_interpro=args.top_interpro,
                top_ec=args.top_ec,
                top_taxonomy=args.top_taxonomy,
                min_plddt=args.min_plddt,
            )
        elif step == "cluster":
            run_cluster(args.data_dir, identity=args.identity, workers=args.workers)
        elif step == "verify":
            verify_output(args.data_dir)

    elapsed = time.time() - t0
    logger.info("")
    logger.info(f"Pipeline complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
