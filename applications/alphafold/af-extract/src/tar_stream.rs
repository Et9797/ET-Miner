//! Stream through AlphaFold proteome tar archives, extracting pLDDT scores.
//!
//! Supports two tar formats:
//!
//! **v4 tars** (older, include confidence JSONs):
//! ```text
//! ├── AF-Q9Y6K9-F1-confidence_v4.json        ← parsed
//! ├── AF-Q9Y6K9-F1-model_v4.cif.gz           ← skipped
//! ```
//!
//! **v6 tars** (current, CIF + PDB only):
//! ```text
//! ├── AF-Q9Y6K9-F1-model_v6.cif.gz           ← parsed (pLDDT in _ma_qa_metric_local)
//! ├── AF-Q9Y6K9-F1-model_v6.pdb.gz           ← skipped
//! ```
//!
//! We extract pLDDT from whichever format is available.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use log::{debug, info, warn};
use rayon::prelude::*;

use crate::confidence::{self, PlddtResult};

/// A single protein extracted from a tar archive.
#[derive(Debug)]
pub struct ExtractedProtein {
    /// UniProt accession (e.g., "Q9Y6K9")
    pub accession: String,
    /// pLDDT analysis result
    pub plddt: PlddtResult,
}

/// Type of file found in the tar.
enum TarEntry {
    /// confidence_v4.json — small, fast to parse
    ConfidenceJson { accession: String },
    /// model_v{4,6}.cif.gz — larger, needs gzip decompression + CIF parsing
    CifGz { accession: String },
}

/// Classify a tar entry by its filename.
fn classify_entry(filename: &str) -> Option<TarEntry> {
    let basename = Path::new(filename)
        .file_name()?
        .to_str()?;

    if !basename.starts_with("AF-") {
        return None;
    }

    // Extract accession: AF-{accession}-F1-...
    let rest = &basename[3..]; // skip "AF-"
    let accession = rest.split('-').next()?.to_string();

    if basename.contains("-confidence_v4.json") {
        Some(TarEntry::ConfidenceJson { accession })
    } else if basename.ends_with(".cif.gz") {
        Some(TarEntry::CifGz { accession })
    } else {
        None // skip .pdb.gz, PAE, etc.
    }
}

/// Process a single tar file, extracting pLDDT from all proteins.
///
/// Strategy: prefer confidence JSON if present (v4 tars), otherwise parse CIF.gz (v6 tars).
/// In v6 tars there are no confidence JSONs, so we always parse CIF.gz.
pub fn process_tar(tar_path: &Path, min_plddt: f64) -> Result<Vec<ExtractedProtein>> {
    let file = File::open(tar_path)
        .with_context(|| format!("failed to open tar: {}", tar_path.display()))?;

    let mut archive = tar::Archive::new(file);
    let mut results = Vec::new();
    let mut _processed = 0u64;
    let mut skipped_plddt = 0u64;
    let mut errors = 0u64;

    for entry_result in archive.entries().context("failed to read tar entries")? {
        let mut entry = match entry_result {
            Ok(e) => e,
            Err(e) => {
                warn!("Skipping corrupt tar entry: {}", e);
                errors += 1;
                continue;
            }
        };

        let path = match entry.path() {
            Ok(p) => p.to_path_buf(),
            Err(_) => continue,
        };

        let path_str = path.to_string_lossy().to_string();

        let entry_type = match classify_entry(&path_str) {
            Some(t) => t,
            None => continue,
        };

        // Read entry data into memory
        let mut data = Vec::new();
        if let Err(e) = entry.read_to_end(&mut data) {
            warn!("Failed to read {}: {}", path_str, e);
            errors += 1;
            continue;
        }

        // Parse pLDDT based on file type
        let plddt_result = match &entry_type {
            TarEntry::ConfidenceJson { .. } => confidence::parse_confidence_json(&data),
            TarEntry::CifGz { .. } => confidence::parse_cif_gz(&data),
        };

        let accession = match entry_type {
            TarEntry::ConfidenceJson { accession } | TarEntry::CifGz { accession } => accession,
        };

        match plddt_result {
            Ok(Some(plddt)) => {
                if plddt.mean_plddt >= min_plddt {
                    results.push(ExtractedProtein { accession, plddt });
                } else {
                    skipped_plddt += 1;
                }
                _processed += 1;
            }
            Ok(None) => {
                debug!("No pLDDT data in {}", path_str);
                skipped_plddt += 1;
            }
            Err(e) => {
                debug!("Failed to parse {}: {}", path_str, e);
                errors += 1;
            }
        }
    }

    info!(
        "{}: {} proteins extracted, {} skipped (pLDDT filter), {} errors",
        tar_path.file_name().unwrap_or_default().to_string_lossy(),
        results.len(),
        skipped_plddt,
        errors
    );

    Ok(results)
}

/// Process multiple tar files in parallel using rayon.
pub fn process_tar_dir(
    tar_dir: &Path,
    min_plddt: f64,
) -> Result<Vec<ExtractedProtein>> {
    let mut tar_files: Vec<PathBuf> = Vec::new();

    for entry in std::fs::read_dir(tar_dir)
        .with_context(|| format!("failed to read tar directory: {}", tar_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "tar") {
            tar_files.push(path);
        }
    }

    tar_files.sort();
    info!("Found {} tar files in {}", tar_files.len(), tar_dir.display());

    if tar_files.is_empty() {
        anyhow::bail!("No .tar files found in {}", tar_dir.display());
    }

    // Process tars in parallel
    let results: Vec<Result<Vec<ExtractedProtein>>> = tar_files
        .par_iter()
        .map(|tar_path| process_tar(tar_path, min_plddt))
        .collect();

    // Flatten results, dedup by accession (prefer first occurrence)
    let mut all_proteins = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut total_errors = 0;

    for result in results {
        match result {
            Ok(proteins) => {
                for p in proteins {
                    if seen.insert(p.accession.clone()) {
                        all_proteins.push(p);
                    }
                }
            }
            Err(e) => {
                warn!("Error processing tar: {}", e);
                total_errors += 1;
            }
        }
    }

    info!(
        "Extracted {} unique proteins from {} tars ({} errors)",
        all_proteins.len(),
        tar_files.len(),
        total_errors
    );

    Ok(all_proteins)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_confidence_json() {
        match classify_entry("AF-Q9Y6K9-F1-confidence_v4.json") {
            Some(TarEntry::ConfidenceJson { accession }) => assert_eq!(accession, "Q9Y6K9"),
            _ => panic!("Expected ConfidenceJson"),
        }
    }

    #[test]
    fn test_classify_cif_gz() {
        match classify_entry("AF-A0A385XJ53-F1-model_v6.cif.gz") {
            Some(TarEntry::CifGz { accession }) => assert_eq!(accession, "A0A385XJ53"),
            _ => panic!("Expected CifGz"),
        }
    }

    #[test]
    fn test_classify_pdb_gz_skipped() {
        assert!(classify_entry("AF-Q9Y6K9-F1-model_v6.pdb.gz").is_none());
    }

    #[test]
    fn test_classify_pae_skipped() {
        assert!(classify_entry("AF-Q9Y6K9-F1-predicted_aligned_error_v4.json.gz").is_none());
    }

    #[test]
    fn test_classify_subdir() {
        match classify_entry("subdir/AF-P12345-F1-model_v4.cif.gz") {
            Some(TarEntry::CifGz { accession }) => assert_eq!(accession, "P12345"),
            _ => panic!("Expected CifGz"),
        }
    }
}
