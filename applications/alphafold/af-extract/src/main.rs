//! af-extract: High-performance AlphaFold feature extractor for ET-miner.
//!
//! Two-pass design for 214M protein feature extraction:
//!
//! ```bash
//! # Pass 1: Count feature frequencies
//! af-extract count-frequencies \
//!     --annotations uniprot_trembl.tsv \
//!     --tar-dir /data/tars/ \
//!     --output frequencies.json
//!
//! # Pass 2: Build transactions with top-N features
//! af-extract build-transactions \
//!     --annotations uniprot_trembl.tsv \
//!     --frequencies frequencies.json \
//!     --tar-dir /data/tars/ \
//!     --output alphafold_transactions_214m.parquet \
//!     --item-mapping item_mapping_214m.parquet \
//!     --top-pfam 200 --top-go 200 \
//!     --top-interpro 300 --top-ec 200 --top-taxonomy 50
//! ```

mod annotations;
mod confidence;
mod tar_stream;
mod transaction;

use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use flate2::read::GzDecoder;
use log::info;

use crate::annotations::AnnotationDb;
use crate::tar_stream::{process_tar_dir, ExtractedProtein};
use crate::transaction::{FrequencyCounts, ItemEncoder, write_transactions_parquet};

#[derive(Parser)]
#[command(
    name = "af-extract",
    about = "High-performance AlphaFold feature extractor for ET-miner",
    version,
    author = "Et & C. claudya"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Pass 1: Count Pfam/GO frequencies across all proteins.
    CountFrequencies {
        /// Path to UniProt annotations file (TSV or DAT.gz)
        #[arg(long)]
        annotations: PathBuf,

        /// Directory containing AlphaFold proteome tar files
        #[arg(long)]
        tar_dir: PathBuf,

        /// Output frequencies JSON file
        #[arg(long, default_value = "frequencies.json")]
        output: PathBuf,

        /// Minimum mean pLDDT to include a protein
        #[arg(long, default_value_t = 50.0)]
        min_plddt: f64,
    },

    /// Pass 2: Build transaction parquet with top-N features.
    BuildTransactions {
        /// Path to UniProt annotations file (TSV or DAT.gz)
        #[arg(long)]
        annotations: PathBuf,

        /// Frequencies JSON from count-frequencies pass
        #[arg(long)]
        frequencies: PathBuf,

        /// Directory containing AlphaFold proteome tar files
        #[arg(long)]
        tar_dir: PathBuf,

        /// Output transactions parquet file
        #[arg(long, default_value = "alphafold_transactions_214m.parquet")]
        output: PathBuf,

        /// Output item mapping parquet file
        #[arg(long, default_value = "item_mapping_214m.parquet")]
        item_mapping: PathBuf,

        /// Number of top Pfam domains to include
        #[arg(long, default_value_t = 200)]
        top_pfam: usize,

        /// Number of top GO terms to include
        #[arg(long, default_value_t = 200)]
        top_go: usize,

        /// Number of top InterPro families to include (0 = skip)
        #[arg(long, default_value_t = 0)]
        top_interpro: usize,

        /// Number of top EC numbers to include (0 = skip)
        #[arg(long, default_value_t = 0)]
        top_ec: usize,

        /// Number of top taxonomy bins to include (0 = skip)
        #[arg(long, default_value_t = 0)]
        top_taxonomy: usize,

        /// Minimum mean pLDDT to include a protein
        #[arg(long, default_value_t = 50.0)]
        min_plddt: f64,

        /// Parquet row group size (rows per batch)
        #[arg(long, default_value_t = 100_000)]
        batch_size: usize,
    },

    /// Build transactions from annotation file + pLDDT CSV (no tar files needed).
    ///
    /// Use when pLDDT comes from BigQuery metadata instead of CIF/JSON files.
    /// This avoids downloading 23TB of AlphaFold structures entirely.
    BuildFromMetadata {
        /// Path to UniProt annotations file (TSV or DAT.gz)
        #[arg(long)]
        annotations: PathBuf,

        /// Path to pLDDT metadata CSV (optional — omit for Pfam/GO-only mining)
        #[arg(long)]
        plddt_csv: Option<PathBuf>,

        /// Output transactions parquet file
        #[arg(long, default_value = "alphafold_transactions_214m.parquet")]
        output: PathBuf,

        /// Output item mapping parquet file
        #[arg(long, default_value = "item_mapping_214m.parquet")]
        item_mapping: PathBuf,

        /// Number of top Pfam domains to include
        #[arg(long, default_value_t = 200)]
        top_pfam: usize,

        /// Number of top GO terms to include
        #[arg(long, default_value_t = 200)]
        top_go: usize,

        /// Number of top InterPro families to include (0 = skip)
        #[arg(long, default_value_t = 0)]
        top_interpro: usize,

        /// Number of top EC numbers to include (0 = skip)
        #[arg(long, default_value_t = 0)]
        top_ec: usize,

        /// Number of top taxonomy bins to include (0 = skip)
        #[arg(long, default_value_t = 0)]
        top_taxonomy: usize,

        /// Minimum mean pLDDT to include a protein
        #[arg(long, default_value_t = 50.0)]
        min_plddt: f64,

        /// Parquet row group size (rows per batch)
        #[arg(long, default_value_t = 100_000)]
        batch_size: usize,
    },
}

fn load_annotations(path: &PathBuf) -> Result<AnnotationDb> {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_string())
        .unwrap_or_default();

    if ext == "gz" || path.to_string_lossy().contains(".dat") {
        AnnotationDb::from_dat_gz(path)
    } else {
        AnnotationDb::from_tsv(path)
    }
}

fn run_count_frequencies(
    annotations_path: PathBuf,
    tar_dir: PathBuf,
    output: PathBuf,
    min_plddt: f64,
) -> Result<()> {
    let t0 = Instant::now();

    info!("=== Pass 1: Count Frequencies ===");

    // Load annotations
    let annot_db = load_annotations(&annotations_path)?;

    // Stream through tars
    let proteins = process_tar_dir(&tar_dir, min_plddt)?;

    // Count frequencies
    let mut freq = FrequencyCounts::new();
    for protein in &proteins {
        freq.count_protein(&annot_db, &protein.accession);
    }

    // Save
    freq.save(&output)?;

    let elapsed = t0.elapsed();
    info!(
        "Pass 1 complete: {} proteins, {} Pfam domains, {} GO terms in {:.1}s",
        freq.total_proteins,
        freq.pfam.len(),
        freq.go.len(),
        elapsed.as_secs_f64()
    );

    Ok(())
}

fn run_build_transactions(
    annotations_path: PathBuf,
    frequencies_path: PathBuf,
    tar_dir: PathBuf,
    output: PathBuf,
    item_mapping_path: PathBuf,
    top_pfam: usize,
    top_go: usize,
    top_interpro: usize,
    top_ec: usize,
    top_taxonomy: usize,
    min_plddt: f64,
    batch_size: usize,
) -> Result<()> {
    let t0 = Instant::now();

    info!("=== Pass 2: Build Transactions ===");

    // Load annotations
    let annot_db = load_annotations(&annotations_path)?;

    // Load frequencies
    let freq = FrequencyCounts::load(&frequencies_path)?;

    // Build item encoder (tar-based path always has pLDDT)
    let encoder = ItemEncoder::from_frequencies(
        &freq, top_pfam, top_go, top_interpro, top_ec, top_taxonomy, true,
    );

    // Write item mapping
    encoder.write_item_mapping(&item_mapping_path)?;

    // Stream through tars (second pass)
    let proteins = process_tar_dir(&tar_dir, min_plddt)?;

    // Write transactions parquet
    let n_transactions = write_transactions_parquet(
        &proteins,
        &annot_db,
        &encoder,
        &output,
        batch_size,
    )?;

    let elapsed = t0.elapsed();
    let rate = n_transactions as f64 / elapsed.as_secs_f64();

    info!("=== Results ===");
    info!("Transactions: {}", n_transactions);
    info!("Total items:  {}", encoder.total_items);
    info!("Output:       {}", output.display());
    info!("Item mapping: {}", item_mapping_path.display());
    info!("Time:         {:.1}s ({:.0} proteins/sec)", elapsed.as_secs_f64(), rate);

    Ok(())
}

/// Read pLDDT metadata from a CSV file (BigQuery export or custom format).
///
/// Supports column names: uniprotAccession/accession/accessionId for accession,
/// globalMetricValue/mean_plddt/plddt for pLDDT score.
fn read_plddt_csv(path: &PathBuf, min_plddt: f64) -> Result<Vec<ExtractedProtein>> {
    info!("Reading pLDDT metadata from: {}", path.display());

    let file = File::open(path).context("failed to open pLDDT CSV")?;
    let reader: Box<dyn Read> = if path.extension().map_or(false, |e| e == "gz") {
        Box::new(GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let mut csv_reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(reader);

    // Find column indices from header
    let headers = csv_reader.headers()?.clone();
    let acc_idx = headers
        .iter()
        .position(|h| {
            let lower = h.to_lowercase();
            lower == "uniprotaccession"
                || lower == "accession"
                || lower == "accessionid"
                || lower == "entry"
        })
        .context(
            "CSV must have an accession column (uniprotAccession, accession, accessionId, or Entry)",
        )?;
    let plddt_idx = headers
        .iter()
        .position(|h| {
            let lower = h.to_lowercase();
            lower == "globalmetricvalue"
                || lower == "mean_plddt"
                || lower == "plddt"
                || lower == "avg_plddt"
        })
        .context(
            "CSV must have a pLDDT column (globalMetricValue, mean_plddt, plddt, or avg_plddt)",
        )?;

    info!(
        "  Accession column: '{}' (index {})",
        &headers[acc_idx], acc_idx
    );
    info!(
        "  pLDDT column: '{}' (index {})",
        &headers[plddt_idx], plddt_idx
    );

    let mut proteins = Vec::new();
    let mut total = 0u64;
    let mut skipped = 0u64;

    for result in csv_reader.records() {
        let record = result.context("failed to read CSV record")?;
        total += 1;

        let accession = record.get(acc_idx).unwrap_or("").trim();
        if accession.is_empty() {
            skipped += 1;
            continue;
        }

        let plddt_str = record.get(plddt_idx).unwrap_or("").trim();
        let mean_plddt: f64 = match plddt_str.parse() {
            Ok(v) => v,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };

        if let Some(plddt) = confidence::from_mean_only(mean_plddt) {
            if plddt.mean_plddt >= min_plddt {
                proteins.push(ExtractedProtein {
                    accession: accession.to_string(),
                    plddt,
                });
            } else {
                skipped += 1;
            }
        } else {
            skipped += 1;
        }

        if total % 10_000_000 == 0 {
            info!(
                "  Read {}M rows ({} passed filter)...",
                total / 1_000_000,
                proteins.len()
            );
        }
    }

    info!(
        "Read {} rows: {} passed pLDDT filter (>= {}), {} skipped",
        total,
        proteins.len(),
        min_plddt,
        skipped
    );

    Ok(proteins)
}

fn run_build_from_metadata(
    annotations_path: PathBuf,
    plddt_csv_path: Option<PathBuf>,
    output: PathBuf,
    item_mapping_path: PathBuf,
    top_pfam: usize,
    top_go: usize,
    top_interpro: usize,
    top_ec: usize,
    top_taxonomy: usize,
    min_plddt: f64,
    batch_size: usize,
) -> Result<()> {
    let t0 = Instant::now();

    let include_plddt = plddt_csv_path.is_some();
    if include_plddt {
        info!("=== Build Transactions from Metadata (pLDDT + annotations) ===");
    } else {
        info!("=== Build Transactions from Annotations Only (no pLDDT) ===");
    }

    // Step 1: Load annotations
    let annot_db = load_annotations(&annotations_path)?;

    // Step 2: Build protein list
    let proteins = if let Some(csv_path) = plddt_csv_path {
        read_plddt_csv(&csv_path, min_plddt)?
    } else {
        info!("No pLDDT CSV — using all {} annotated proteins", annot_db.entries.len());
        annot_db
            .entries
            .iter()
            .map(|entry| ExtractedProtein {
                accession: entry.key().clone(),
                plddt: confidence::PlddtResult {
                    items: vec![],
                    mean_plddt: 0.0,
                },
            })
            .collect()
    };

    // Step 3: Count frequencies
    info!("Counting feature frequencies across {} proteins...", proteins.len());
    let mut freq = FrequencyCounts::new();
    for protein in &proteins {
        freq.count_protein(&annot_db, &protein.accession);
    }
    info!(
        "Frequencies: {} Pfam, {} GO, {} InterPro, {} EC, {} taxonomy from {} proteins",
        freq.pfam.len(), freq.go.len(), freq.interpro.len(),
        freq.ec.len(), freq.taxonomy.len(), freq.total_proteins,
    );

    // Step 4: Build item encoder (top-N feature selection)
    let encoder = ItemEncoder::from_frequencies(
        &freq, top_pfam, top_go, top_interpro, top_ec, top_taxonomy, include_plddt,
    );

    // Step 5: Write item mapping
    encoder.write_item_mapping(&item_mapping_path)?;

    // Step 6: Write transactions parquet
    let n_transactions = write_transactions_parquet(
        &proteins, &annot_db, &encoder, &output, batch_size,
    )?;

    let elapsed = t0.elapsed();
    let rate = n_transactions as f64 / elapsed.as_secs_f64();

    info!("=== Results ===");
    info!("Transactions: {}", n_transactions);
    info!("Total items:  {}", encoder.total_items);
    info!("Output:       {}", output.display());
    info!("Item mapping: {}", item_mapping_path.display());
    info!(
        "Time:         {:.1}s ({:.0} proteins/sec)",
        elapsed.as_secs_f64(),
        rate
    );

    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::CountFrequencies {
            annotations,
            tar_dir,
            output,
            min_plddt,
        } => run_count_frequencies(annotations, tar_dir, output, min_plddt),

        Commands::BuildTransactions {
            annotations,
            frequencies,
            tar_dir,
            output,
            item_mapping,
            top_pfam,
            top_go,
            top_interpro,
            top_ec,
            top_taxonomy,
            min_plddt,
            batch_size,
        } => run_build_transactions(
            annotations,
            frequencies,
            tar_dir,
            output,
            item_mapping,
            top_pfam,
            top_go,
            top_interpro,
            top_ec,
            top_taxonomy,
            min_plddt,
            batch_size,
        ),

        Commands::BuildFromMetadata {
            annotations,
            plddt_csv,
            output,
            item_mapping,
            top_pfam,
            top_go,
            top_interpro,
            top_ec,
            top_taxonomy,
            min_plddt,
            batch_size,
        } => run_build_from_metadata(
            annotations,
            plddt_csv,
            output,
            item_mapping,
            top_pfam,
            top_go,
            top_interpro,
            top_ec,
            top_taxonomy,
            min_plddt,
            batch_size,
        ),
    }
}
