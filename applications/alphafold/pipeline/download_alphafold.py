#!/usr/bin/env python3
"""
Download AlphaFold CIF structures and UniProt annotations.

Downloads proteome-level tar archives from Google Cloud Storage,
Swiss-Prot annotations (Pfam domains, GO terms), and PROSITE patterns
for later validation.

Usage:
    python download_alphafold.py --proteomes human,ecoli,yeast
    python download_alphafold.py --all-swissprot
    python download_alphafold.py --proteomes human --data-dir /mnt/data
"""

import argparse
import gzip
import shutil
import sys
import tarfile
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import download_file

from loguru import logger

# AlphaFold proteome downloads via EBI FTP (HTTPS, no auth required)
# Format: {name: (tar_filename, approx_size_mb)}
ALPHAFOLD_EBI_BASE = "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest"
PROTEOME_TARS = {
    "human": ("UP000005640_9606_HUMAN_v6.tar", 4938),
    "ecoli": ("UP000000625_83333_ECOLI_v6.tar", 153),
    "yeast": ("UP000002311_559292_YEAST_v6.tar", 226),
    "mouse": ("UP000000589_10090_MOUSE_v6.tar", 4565),
    "arabidopsis": ("UP000006548_3702_ARATH_v6.tar", 1181),
    "drosophila": ("UP000000803_7227_DROME_v6.tar", 576),
    "celegans": ("UP000001940_6239_CAEEL_v6.tar", 699),
    "zebrafish": ("UP000000437_7955_DANRE_v6.tar", 1553),
}
SWISSPROT_CIF_TAR = "swissprot_cif_v6.tar"  # 550K structures, 38 GB
UNIPROT_SPROT_DAT = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz"
UNIPROT_SPROT_FASTA = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
PROSITE_DAT = "https://ftp.expasy.org/databases/prosite/prosite.dat"


def parse_uniprot_dat(dat_gz_path: Path, output_path: Path) -> pl.DataFrame:
    """Parse Swiss-Prot DAT file → extract Pfam domains and GO terms per protein.

    Returns DataFrame with columns: [uniprot_id, pfam_domains, go_terms]
    """
    if output_path.exists():
        logger.info(f"Cached: {output_path.name}")
        return pl.read_parquet(output_path)

    logger.info(f"Parsing UniProt DAT file ({dat_gz_path.name})...")

    records = []
    current_id = None
    pfam_domains = []
    go_terms = []

    with gzip.open(dat_gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("AC   ") and current_id is None:
                # First accession number
                current_id = line[5:].strip().rstrip(";").split(";")[0].strip()
            elif line.startswith("DR   Pfam;"):
                # DR   Pfam; PF00089; Trypsin; 1.
                parts = line.strip().split(";")
                if len(parts) >= 2:
                    pfam_id = parts[1].strip()
                    pfam_domains.append(pfam_id)
            elif line.startswith("DR   GO;"):
                # DR   GO; GO:0006915; P:apoptotic process; IEA:UniProtKB-KW.
                parts = line.strip().split(";")
                if len(parts) >= 2:
                    go_id = parts[1].strip()
                    go_terms.append(go_id)
            elif line.startswith("//"):
                # End of record
                if current_id:
                    records.append({
                        "uniprot_id": current_id,
                        "pfam_domains": list(set(pfam_domains)),
                        "go_terms": list(set(go_terms)),
                    })
                current_id = None
                pfam_domains = []
                go_terms = []

                if len(records) % 100_000 == 0:
                    logger.info(f"  Parsed {len(records):,} proteins...")

    logger.info(f"Parsed {len(records):,} Swiss-Prot proteins total")

    df = pl.DataFrame(records)
    df.write_parquet(output_path)
    logger.info(f"Saved: {output_path}")

    # Stats
    has_pfam = df.filter(pl.col("pfam_domains").list.len() > 0).height
    has_go = df.filter(pl.col("go_terms").list.len() > 0).height
    logger.info(f"  With Pfam: {has_pfam:,} ({100*has_pfam/len(records):.1f}%)")
    logger.info(f"  With GO:   {has_go:,} ({100*has_go/len(records):.1f}%)")

    return df


def download_alphafold_proteome(name: str, raw_dir: Path) -> list[Path]:
    """Download AlphaFold proteome tar via EBI HTTPS and extract CIF + confidence JSON."""
    proteome_dir = raw_dir / name
    marker = proteome_dir / ".download_complete"

    if marker.exists():
        cif_files = list(proteome_dir.glob("*.cif"))
        logger.info(f"Cached: {name} ({len(cif_files):,} CIF files)")
        return cif_files

    proteome_dir.mkdir(parents=True, exist_ok=True)

    tar_filename, approx_mb = PROTEOME_TARS[name]
    tar_url = f"{ALPHAFOLD_EBI_BASE}/{tar_filename}"
    tar_path = proteome_dir / tar_filename

    # Download tar
    download_file(tar_url, tar_path, f"{name} proteome (~{approx_mb} MB)")

    # Extract CIF and confidence JSON from tar
    logger.info(f"  Extracting {name} tar...")
    n_cif = 0
    n_conf = 0
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            basename = Path(member.name).name
            # We want .cif.gz and confidence JSON files
            if basename.endswith(".cif.gz") or basename.endswith("-confidence_v4.json"):
                member.name = basename  # Flatten directory structure
                tf.extract(member, proteome_dir)
                if basename.endswith(".cif.gz"):
                    n_cif += 1
                else:
                    n_conf += 1

    logger.info(f"  Extracted {n_cif:,} CIF + {n_conf:,} confidence files")

    # Delete tar to save disk
    tar_path.unlink()
    logger.info(f"  Removed tar to save disk")

    # Decompress .cif.gz files
    gz_files = list(proteome_dir.glob("*.cif.gz"))
    logger.info(f"  Decompressing {len(gz_files):,} CIF files...")
    for gz_path in gz_files:
        cif_path = gz_path.with_suffix("")  # Remove .gz
        with gzip.open(gz_path, "rb") as f_in:
            with open(cif_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        gz_path.unlink()

    marker.touch()
    cif_files = list(proteome_dir.glob("*.cif"))
    logger.info(f"  Done: {len(cif_files):,} CIF files for {name}")
    return cif_files


def download_prosite(annotations_dir: Path) -> Path:
    """Download PROSITE database for validation step."""
    dest = annotations_dir / "prosite.dat"
    return download_file(PROSITE_DAT, dest, "PROSITE database")


def main():
    parser = argparse.ArgumentParser(
        description="Download AlphaFold structures and UniProt annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --proteomes human,ecoli,yeast    # ~34K structures (fast start)
  %(prog)s --proteomes human                # ~23K structures (minimal)
  %(prog)s --all-swissprot                  # ~550K structures (full PoC)

Downloads via EBI HTTPS (no GCS auth required).
Available proteomes: """ + ", ".join(sorted(PROTEOME_TARS.keys())),
    )
    parser.add_argument(
        "--proteomes",
        type=str,
        default="human,ecoli,yeast",
        help="Comma-separated proteome names (default: human,ecoli,yeast)",
    )
    parser.add_argument(
        "--all-swissprot",
        action="store_true",
        help="Download all Swiss-Prot mapped proteomes (~570K structures)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Data directory (default: ../data)",
    )
    parser.add_argument(
        "--skip-structures",
        action="store_true",
        help="Skip structure download (only download annotations)",
    )
    args = parser.parse_args()

    raw_dir = args.data_dir / "raw"
    annotations_dir = args.data_dir / "annotations"
    raw_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download and parse UniProt Swiss-Prot annotations
    logger.info("=" * 60)
    logger.info("Step 1: UniProt Swiss-Prot annotations")
    logger.info("=" * 60)

    dat_path = download_file(
        UNIPROT_SPROT_DAT,
        annotations_dir / "uniprot_sprot.dat.gz",
        "UniProt Swiss-Prot DAT (~660 MB)",
    )
    annotations_df = parse_uniprot_dat(
        dat_path, annotations_dir / "uniprot_annotations.parquet"
    )

    # Also download FASTA for clustering step
    download_file(
        UNIPROT_SPROT_FASTA,
        annotations_dir / "uniprot_sprot.fasta.gz",
        "UniProt Swiss-Prot FASTA (~89 MB)",
    )

    # Step 2: Download AlphaFold structures
    if not args.skip_structures:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Step 2: AlphaFold structures")
        logger.info("=" * 60)

        if args.all_swissprot:
            # Download the combined Swiss-Prot CIF tar (38 GB, 550K structures)
            swissprot_dir = raw_dir / "swissprot"
            marker = swissprot_dir / ".download_complete"
            if marker.exists():
                cif_files = list(swissprot_dir.glob("*.cif"))
                logger.info(f"Cached: swissprot ({len(cif_files):,} CIF files)")
                total_cif = len(cif_files)
            else:
                swissprot_dir.mkdir(parents=True, exist_ok=True)
                tar_url = f"{ALPHAFOLD_EBI_BASE}/{SWISSPROT_CIF_TAR}"
                tar_path = swissprot_dir / SWISSPROT_CIF_TAR
                download_file(tar_url, tar_path, "Swiss-Prot CIF (~38 GB)")
                logger.info("Extracting Swiss-Prot CIF tar...")
                with tarfile.open(tar_path, "r") as tf:
                    tf.extractall(swissprot_dir)
                tar_path.unlink()
                marker.touch()
                cif_files = list(swissprot_dir.rglob("*.cif"))
                total_cif = len(cif_files)
                logger.info(f"Extracted {total_cif:,} CIF files")
        else:
            names = [n.strip() for n in args.proteomes.split(",")]
            for name in names:
                if name not in PROTEOME_TARS:
                    logger.error(
                        f"Unknown proteome: {name}. "
                        f"Available: {', '.join(sorted(PROTEOME_TARS.keys()))}"
                    )
                    sys.exit(1)

            total_cif = 0
            for name in names:
                cif_files = download_alphafold_proteome(name, raw_dir)
                total_cif += len(cif_files)

        logger.info(f"\nTotal CIF files downloaded: {total_cif:,}")

    # Step 3: Download PROSITE for validation
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 3: PROSITE database")
    logger.info("=" * 60)
    download_prosite(annotations_dir)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Download complete")
    logger.info("=" * 60)
    logger.info(f"Annotations: {annotations_dir}")
    logger.info(f"  Swiss-Prot proteins: {annotations_df.height:,}")
    if not args.skip_structures:
        logger.info(f"Structures: {raw_dir}")
    logger.info(f"\nNext step: python extract_features.py --data-dir {args.data_dir}")


if __name__ == "__main__":
    main()
