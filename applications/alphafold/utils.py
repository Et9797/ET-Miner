"""Shared utilities for AlphaFold pipeline, drug-discovery, and experiments.

Deduplicated from patterns found across 28+ scripts by Council of Copii
Schizo-swarm audit (2026-03-29). Single source of truth for:
- Transaction loading, item mapping, K-distribution
- Accession column detection, file downloads
- DRUGGABLE_PFAM_FAMILIES canonical dict
- GO term relatedness heuristic
"""

import time
from pathlib import Path
from urllib.request import urlretrieve

import polars as pl
from loguru import logger


# ---------------------------------------------------------------------------
# Transaction loading
# ---------------------------------------------------------------------------

def load_transactions(
    path: str | Path,
    *,
    min_items: int = 0,
) -> pl.DataFrame:
    """Load transaction parquet, drop protein_id, keep only 'items'.

    Args:
        path: Path to parquet with 'items' column.
        min_items: Filter to transactions with >= this many items (0 = no filter).
        verbose: Print loading stats.
    """
    logger.info(f"Loading {path}...")
    t0 = time.perf_counter()
    df = (
        pl.scan_parquet(str(path))
        .select("items")
        .collect(engine="streaming")
    )
    logger.info(f"  Loaded {len(df):,} proteins in {time.perf_counter() - t0:.1f}s")

    if min_items > 0:
        n_before = len(df)
        df = df.filter(pl.col("items").list.len() >= min_items)
        logger.info(f"  Multi-feature filter: {n_before:,} → {len(df):,} "
                     f"(removed {n_before - len(df):,} single-feature proteins)")
    return df


# ---------------------------------------------------------------------------
# Item mapping
# ---------------------------------------------------------------------------

def load_item_mapping(path: str | Path) -> dict[int, dict]:
    """Load item_id → {name, category} mapping from parquet.

    For simple name-only lookup: ``{k: v['name'] for k, v in mapping.items()}``
    """
    df = pl.read_parquet(str(path))
    mapping = {}
    for row in df.iter_rows(named=True):
        mapping[row["item_id"]] = {
            "name": row["feature_name"],
            "category": row.get("feature_category", ""),
        }
    return mapping


def load_item_names(path: str | Path) -> dict[int, str]:
    """Load item_id → feature_name mapping (simple variant)."""
    df = pl.read_parquet(str(path))
    return dict(zip(df["item_id"].to_list(), df["feature_name"].to_list()))


# ---------------------------------------------------------------------------
# K-distribution
# ---------------------------------------------------------------------------

def k_distribution(result: pl.DataFrame) -> dict[int, int]:
    """Count itemsets per K level (vectorized, handles millions of rows)."""
    counts = (
        result
        .with_columns(pl.col("itemset").list.len().alias("k"))
        .group_by("k")
        .agg(pl.len().alias("count"))
        .sort("k")
    )
    return {int(row["k"]): int(row["count"]) for row in counts.iter_rows(named=True)}


# ---------------------------------------------------------------------------
# Accession column detection
# ---------------------------------------------------------------------------

ACCESSION_CANDIDATES = ("uniprot_accession", "accession", "protein_id", "uniprot_id")


def detect_accession_col(df: pl.DataFrame) -> str:
    """Find accession column name in a DataFrame."""
    for col in ACCESSION_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"No accession column found. Columns: {df.columns}")


# ---------------------------------------------------------------------------
# File download with caching
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, description: str = "") -> Path:
    """Download a file with caching (skip if exists)."""
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        logger.info(f"Cached: {dest.name} ({size_mb:.1f} MB)")
        return dest

    logger.info(f"Downloading {description or dest.name}...")
    logger.info(f"  URL: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    try:
        urlretrieve(url, tmp)
        tmp.rename(dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        logger.info(f"  Saved: {dest} ({size_mb:.1f} MB)")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    return dest


# ---------------------------------------------------------------------------
# GO term relatedness heuristic
# ---------------------------------------------------------------------------

def go_terms_related(terms: list[str], max_prefix_groups: int = 2) -> bool:
    """Check if GO terms are plausibly related using 4-digit prefix heuristic.

    GO terms sharing a 4-digit prefix are considered from the same branch.
    """
    if len(terms) < 2:
        return True
    prefixes = set()
    for t in terms:
        numeric = t.replace("GO:", "")
        if len(numeric) >= 5:
            prefixes.add(numeric[:4])
    return len(prefixes) <= max_prefix_groups


# ---------------------------------------------------------------------------
# Curated druggable Pfam families (CANONICAL — single source of truth)
# ---------------------------------------------------------------------------

DRUGGABLE_PFAM_FAMILIES: dict[str, str] = {
    # Kinases
    "PF00069": "Protein kinase domain",
    "PF07714": "Protein tyrosine kinase",
    "PF00794": "PI3-PI4 kinase",
    # GPCRs
    "PF00001": "7-transmembrane receptor (rhodopsin family)",
    "PF00002": "7-transmembrane receptor (secretin family)",
    "PF00003": "7-transmembrane receptor (metabotropic glutamate family)",
    # Proteases
    "PF00089": "Trypsin",
    "PF00082": "Subtilase family",
    "PF01435": "Peptidase family M48",
    "PF00326": "Prolyl oligopeptidase family",
    # Nuclear receptors
    "PF00104": "Ligand-binding domain of nuclear hormone receptor",
    "PF00105": "Zinc finger, C4 type",
    # Ion channels
    "PF00520": "Ion transport protein",
    "PF07885": "Ion channel (voltage-gated potassium)",
    "PF00060": "Ligand-gated ion channel",
    # Enzymes
    "PF00227": "Proteasome subunit",
    "PF00106": "Short-chain dehydrogenase/reductase",
    "PF00067": "Cytochrome P450",
    "PF00296": "Luciferase-like monooxygenase",
    # Transporters
    "PF00005": "ABC transporter",
    "PF07690": "Major Facilitator Superfamily",
    # Phosphatases
    "PF00102": "Protein-tyrosine phosphatase",
    "PF00782": "DSPc (dual specificity phosphatase)",
    # Epigenetic targets
    "PF00855": "PWWP domain",
    "PF00439": "Bromodomain",
    "PF02373": "JmjC domain",
    "PF00856": "SET domain",
    # Ubiquitin system
    "PF00179": "Ubiquitin-conjugating enzyme",
    "PF00632": "HECT domain",
}
