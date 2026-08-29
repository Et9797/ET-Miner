#!/usr/bin/env python3
"""
Download and prepare the Online Retail II dataset for benchmarking.

Dataset: https://archive.ics.uci.edu/dataset/502/online+retail+ii
Contains ~1M transactions from a UK online retailer (2009-2011).
"""

import io
import zipfile
from pathlib import Path

import polars as pl
import requests
from loguru import logger


def download_dataset(output_dir: Path) -> Path:
    """Download Online Retail II dataset from UCI ML Repository."""
    url = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
    zip_path = output_dir / "online_retail_ii.zip"

    if not zip_path.exists():
        logger.info(f"Downloading dataset from {url}...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        zip_path.write_bytes(response.content)
        logger.info(f"Downloaded to {zip_path}")
    else:
        logger.info(f"Using cached {zip_path}")

    return zip_path


def extract_and_load(zip_path: Path) -> pl.DataFrame:
    """Extract and load the Excel file from the zip archive."""
    logger.info("Extracting and loading Excel file...")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Find the Excel file
        excel_files = [f for f in zf.namelist() if f.endswith('.xlsx')]
        if not excel_files:
            raise ValueError("No Excel file found in archive")

        excel_name = excel_files[0]
        logger.info(f"Reading {excel_name}...")

        with zf.open(excel_name) as f:
            # Read Excel with pandas (polars doesn't directly support Excel in zip)
            import pandas as pd
            # sheet_name=None reads every sheet; the workbook holds one sheet per year
            sheets = pd.read_excel(io.BytesIO(f.read()), engine='openpyxl', sheet_name=None)
            for sheet_name, sheet in sheets.items():
                logger.info(f"  {sheet_name}: {len(sheet):,} rows")
            pdf = pd.concat(sheets.values(), ignore_index=True)

            # Convert all non-numeric columns to string to avoid Arrow mixed-type issues.
            # Selecting by exclusion covers both pandas 2 ('object') and pandas 3 ('str'),
            # where string columns are no longer dtype 'object'.
            numeric_cols = pdf.select_dtypes(include=['number', 'datetime', 'bool']).columns
            for col in pdf.columns.difference(numeric_cols):
                pdf[col] = pdf[col].fillna('').astype(str)

            df = pl.from_pandas(pdf)

    logger.info(f"Loaded {df.height:,} rows")
    return df


def prepare_transactions(df: pl.DataFrame, item_col: str = "StockCode") -> pl.DataFrame:
    """
    Prepare transaction data for Apriori mining.

    Args:
        df: Raw Online Retail II data
        item_col: Column to use as item identifier ('StockCode' or 'Description')

    Returns:
        DataFrame with columns: InvoiceNo, items (List[Int64])
    """
    logger.info(f"Preparing transactions using {item_col} as item identifier...")

    # Clean data
    cleaned = df.filter(
        # Remove cancellations (InvoiceNo starts with 'C')
        ~pl.col("Invoice").cast(pl.Utf8).str.starts_with("C")
    ).filter(
        # Remove null items
        pl.col(item_col).is_not_null()
    ).filter(
        # Remove negative quantities
        pl.col("Quantity") > 0
    )

    logger.info(f"After cleaning: {cleaned.height:,} rows ({df.height - cleaned.height:,} removed)")

    # Create item encoding (item -> integer ID)
    unique_items = (
        cleaned
        .select(item_col)
        .unique()
        .sort(item_col)
        .with_row_index("item_id")
    )

    n_items = unique_items.height
    logger.info(f"Unique items: {n_items:,}")

    # Join to get item IDs
    with_ids = cleaned.join(
        unique_items,
        on=item_col,
        how="left"
    )

    # Group by Invoice to create transactions
    transactions = (
        with_ids
        .group_by("Invoice")
        .agg(
            pl.col("item_id").unique().sort().alias("items")
        )
        .filter(
            # Filter out single-item transactions (not interesting for association mining)
            pl.col("items").list.len() > 1
        )
        .sort("Invoice")
    )

    logger.info(f"Transactions: {transactions.height:,}")
    logger.info(f"Avg items per transaction: {transactions.select(pl.col('items').list.len().mean()).item():.1f}")

    return transactions, unique_items


def save_datasets(
    transactions: pl.DataFrame,
    item_mapping: pl.DataFrame,
    output_dir: Path,
) -> None:
    """Save prepared datasets in multiple formats."""

    # Save parquet for polars-apriori (primary format)
    parquet_path = output_dir / "transactions.parquet"
    transactions.write_parquet(parquet_path)
    logger.info(f"Saved: {parquet_path}")

    # Save item mapping for reference
    mapping_path = output_dir / "item_mapping.parquet"
    item_mapping.write_parquet(mapping_path)
    logger.info(f"Saved: {mapping_path}")

    logger.info("=" * 50)
    logger.info("Dataset Summary")
    logger.info("=" * 50)
    logger.info(f"Transactions: {transactions.height:,}")
    logger.info(f"Unique items: {item_mapping.height:,}")

    item_counts = transactions.select(pl.col("items").list.len())
    logger.info("Items per transaction:")
    logger.debug(f"  Min: {item_counts.min().item()}")
    logger.debug(f"  Max: {item_counts.max().item()}")
    logger.debug(f"  Mean: {item_counts.mean().item():.1f}")
    logger.debug(f"  Median: {item_counts.median().item():.0f}")


def main():
    """Download, prepare, and save the Online Retail II dataset."""
    output_dir = Path(__file__).parent / "online_retail_ii"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already prepared
    parquet_path = output_dir / "transactions.parquet"
    if parquet_path.exists():
        logger.info(f"Dataset already prepared at {parquet_path}")
        df = pl.read_parquet(parquet_path)
        logger.info(f"Transactions: {df.height:,}")
        return

    # Download
    zip_path = download_dataset(output_dir)

    # Load
    df = extract_and_load(zip_path)

    # Prepare using StockCode (faster hashing than Description strings)
    transactions, item_mapping = prepare_transactions(df, item_col="StockCode")

    # Save
    save_datasets(transactions, item_mapping, output_dir)

    logger.info("Done! Dataset ready for benchmarking.")


if __name__ == "__main__":
    main()
