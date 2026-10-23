#!/usr/bin/env python3
"""
Cluster proteins at 30% sequence identity using MMseqs2.

Removes homologous redundancy so frequent itemsets reflect true structural
motifs rather than evolutionary relatedness. Filters the transactions
parquet to keep only cluster representatives.

Usage:
    python cluster_sequences.py
    python cluster_sequences.py --identity 0.3 --coverage 0.8
    python cluster_sequences.py --data-dir /mnt/data --identity 0.5
"""

import argparse
import gzip
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import polars as pl

from loguru import logger


def get_processed_protein_ids(data_dir: Path) -> set[str]:
    """Get UniProt IDs of proteins we have transactions for.

    Scans CIF filenames in data/raw/ subdirectories.
    AlphaFold naming: AF-{UNIPROT_ID}-F1-model_v4.cif
    """
    raw_dir = data_dir / "raw"
    if not raw_dir.exists():
        logger.error(f"Raw data directory not found: {raw_dir}")
        sys.exit(1)

    ids = set()
    for cif_path in raw_dir.rglob("AF-*-F1-model_v*.cif*"):
        name = cif_path.name.replace(".gz", "").replace(".cif", "")
        parts = name.split("-")
        if len(parts) >= 3:
            ids.add(parts[1])

    logger.info(f"Found {len(ids):,} proteins with CIF structures")
    return ids


def prepare_fasta(data_dir: Path, protein_ids: set[str]) -> Path:
    """Prepare a filtered FASTA containing only proteins we processed.

    Prefers Swiss-Prot FASTA if available, otherwise extracts from CIFs.
    """
    output = data_dir / "processed" / "filtered_sequences.fasta"
    if output.exists():
        # Count sequences in cached file
        n = sum(1 for line in open(output) if line.startswith(">"))
        logger.info(f"Cached: {output.name} ({n:,} sequences)")
        return output

    sprot_fasta = data_dir / "annotations" / "uniprot_sprot.fasta.gz"
    if not sprot_fasta.exists():
        logger.error(
            f"Swiss-Prot FASTA not found: {sprot_fasta}\n"
            "Run download_alphafold.py first."
        )
        sys.exit(1)

    logger.info(f"Filtering Swiss-Prot FASTA to {len(protein_ids):,} proteins...")

    output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    writing = False

    with gzip.open(sprot_fasta, "rt", encoding="utf-8") as f_in, open(output, "w") as f_out:
        for line in f_in:
            if line.startswith(">"):
                # >sp|Q9Y6K9|MLEC_HUMAN Malectin OS=Homo sapiens ...
                parts = line.split("|")
                uniprot_id = parts[1] if len(parts) >= 3 else None
                writing = uniprot_id in protein_ids
                if writing:
                    f_out.write(line)
                    n_written += 1
            elif writing:
                f_out.write(line)

    logger.info(f"Wrote {n_written:,} / {len(protein_ids):,} sequences to FASTA")
    if n_written == 0:
        logger.error("No sequences matched. Check UniProt ID format.")
        output.unlink()
        sys.exit(1)

    return output


def run_mmseqs2(
    fasta_path: Path, output_dir: Path, identity: float, coverage: float, workers: int
) -> Path:
    """Run MMseqs2 easy-cluster and return path to representative FASTA."""
    if shutil.which("mmseqs") is None:
        logger.error(
            "mmseqs2 not found in PATH.\n"
            "Install: conda install -c conda-forge -c bioconda mmseqs2"
        )
        sys.exit(1)

    prefix = output_dir / "clusterRes"
    rep_fasta = Path(f"{prefix}_rep_seq.fasta")

    if rep_fasta.exists():
        logger.info(f"Cached: {rep_fasta.name}")
        return rep_fasta

    with tempfile.TemporaryDirectory(dir=output_dir) as tmp:
        cmd = [
            "mmseqs", "easy-cluster",
            str(fasta_path), str(prefix), tmp,
            "--min-seq-id", str(identity),
            "-c", str(coverage),
            "--cov-mode", "1",
            "--threads", str(workers),
        ]
        logger.info(f"Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"MMseqs2 failed:\n{result.stderr}")
            sys.exit(1)

    logger.info("MMseqs2 clustering complete")
    return rep_fasta


def parse_representative_ids(rep_fasta: Path) -> set[str]:
    """Extract UniProt IDs from MMseqs2 representative FASTA."""
    ids = set()
    with open(rep_fasta) as f:
        for line in f:
            if line.startswith(">"):
                # >sp|Q9Y6K9|MLEC_HUMAN ...
                parts = line.split("|")
                if len(parts) >= 3:
                    ids.add(parts[1])
    logger.info(f"Cluster representatives: {len(ids):,}")
    return ids


def filter_transactions(
    data_dir: Path, representative_ids: set[str]
) -> None:
    """Filter transactions parquet to keep only cluster representatives."""
    input_path = data_dir / "processed" / "alphafold_transactions.parquet"
    output_path = data_dir / "processed" / "alphafold_transactions_nr30.parquet"

    if not input_path.exists():
        logger.error(f"Transactions file not found: {input_path}")
        sys.exit(1)

    df = pl.read_parquet(input_path)
    n_before = df.height

    if "protein_id" not in df.columns:
        logger.error(
            "Transactions parquet missing 'protein_id' column.\n"
            "Re-run extract_features.py (must include protein_id)."
        )
        sys.exit(1)

    df_filtered = df.filter(pl.col("protein_id").is_in(list(representative_ids)))
    n_after = df_filtered.height
    reduction = 100 * (1 - n_after / n_before) if n_before > 0 else 0

    df_filtered.write_parquet(output_path)

    logger.info(
        f"{n_before:,} proteins -> {n_after:,} clusters "
        f"({reduction:.0f}% reduction)"
    )
    logger.info(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cluster proteins by sequence identity and filter transactions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Default: 30%% identity, 80%% coverage
  %(prog)s --identity 0.5               # 50%% identity threshold
  %(prog)s --data-dir /mnt/data         # Custom data directory
  %(prog)s --workers 16                 # Use 16 threads for MMseqs2
""",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Data directory (default: ../data)",
    )
    parser.add_argument(
        "--identity",
        type=float,
        default=0.3,
        help="Sequence identity threshold (default: 0.3)",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.8,
        help="Alignment coverage threshold (default: 0.8)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of threads for MMseqs2 (default: 4)",
    )
    args = parser.parse_args()

    processed_dir = args.data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Identify proteins we have
    logger.info("=" * 60)
    logger.info("Step 1: Identify processed proteins")
    logger.info("=" * 60)
    protein_ids = get_processed_protein_ids(args.data_dir)

    # Step 2: Prepare filtered FASTA
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 2: Prepare FASTA for clustering")
    logger.info("=" * 60)
    fasta_path = prepare_fasta(args.data_dir, protein_ids)

    # Step 3: Run MMseqs2
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Step 3: MMseqs2 clustering (identity={args.identity}, coverage={args.coverage})")
    logger.info("=" * 60)
    rep_fasta = run_mmseqs2(
        fasta_path, processed_dir, args.identity, args.coverage, args.workers
    )

    # Step 4: Filter transactions
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 4: Filter transactions to representatives")
    logger.info("=" * 60)
    representative_ids = parse_representative_ids(rep_fasta)
    filter_transactions(args.data_dir, representative_ids)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Clustering complete")
    logger.info("=" * 60)
    logger.info(f"\nNext step: python run_mining.py --data-dir {args.data_dir}")


if __name__ == "__main__":
    main()
