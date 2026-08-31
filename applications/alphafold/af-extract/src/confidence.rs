//! Parse AlphaFold pLDDT scores from confidence JSON or CIF files.
//!
//! Supports two formats:
//!
//! 1. **confidence_v4.json** (v4 tars, ~200-500 bytes):
//! ```json
//! {"confidenceScore": [82.31, 91.44, ...]}
//! ```
//!
//! 2. **mmCIF .cif.gz** (v6 tars, gzipped):
//! ```
//! _ma_qa_metric_local.label_asym_id
//! _ma_qa_metric_local.label_comp_id
//! _ma_qa_metric_local.label_seq_id
//! _ma_qa_metric_local.metric_id
//! _ma_qa_metric_local.metric_value    ← pLDDT scores here
//! _ma_qa_metric_local.model_id
//! _ma_qa_metric_local.ordinal_id
//! A MET 1   2 62.53 1 1
//! A GLY 2   2 83.00 1 2
//! ```
//!
//! pLDDT binning matches Python `extract_features.py` exactly:
//! - Mean: <50 = low(0), 50-90 = med(1), >90 = high(2)
//! - Frac high (>90): <0.25 = low(3), 0.25-0.75 = med(4), >0.75 = high(5)

use std::io::{BufRead, BufReader, Read};

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use serde::Deserialize;

/// Item ID bases — must match Python extract_features.py
pub const PLDDT_MEAN_BASE: u32 = 0;
pub const PLDDT_FRAC_BASE: u32 = 3;

/// Parsed pLDDT result with item IDs and mean score.
#[derive(Debug, Clone)]
pub struct PlddtResult {
    /// Item IDs assigned (always 2: one mean bin + one frac bin)
    pub items: Vec<u32>,
    /// Raw mean pLDDT value for filtering
    pub mean_plddt: f64,
}

/// Raw confidence JSON structure from AlphaFold.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ConfidenceJson {
    #[serde(default)]
    #[serde(rename = "confidenceScore")]
    confidence_score: Option<Vec<f64>>,
    #[serde(default)]
    #[serde(rename = "pLDDT")]
    plddt: Option<Vec<f64>>,
}

/// Bin pLDDT scores into item IDs.
fn bin_plddt_scores(scores: &[f64]) -> Option<PlddtResult> {
    if scores.is_empty() {
        return None;
    }

    let n = scores.len() as f64;
    let sum: f64 = scores.iter().sum();
    let mean_plddt = sum / n;
    let frac_high = scores.iter().filter(|&&s| s > 90.0).count() as f64 / n;

    let mut items = Vec::with_capacity(2);

    // Mean pLDDT bin (matches Python exactly)
    if mean_plddt < 50.0 {
        items.push(PLDDT_MEAN_BASE); // low
    } else if mean_plddt <= 90.0 {
        items.push(PLDDT_MEAN_BASE + 1); // med
    } else {
        items.push(PLDDT_MEAN_BASE + 2); // high
    }

    // Fraction high-confidence bin (matches Python exactly)
    if frac_high < 0.25 {
        items.push(PLDDT_FRAC_BASE); // low
    } else if frac_high <= 0.75 {
        items.push(PLDDT_FRAC_BASE + 1); // med
    } else {
        items.push(PLDDT_FRAC_BASE + 2); // high
    }

    Some(PlddtResult { items, mean_plddt })
}

/// Parse confidence JSON bytes and return pLDDT item IDs.
pub fn parse_confidence_json(data: &[u8]) -> Result<Option<PlddtResult>> {
    let conf: ConfidenceJson =
        serde_json::from_slice(data).context("failed to parse confidence JSON")?;

    let scores = conf.confidence_score.or(conf.plddt);

    match scores {
        Some(s) => Ok(bin_plddt_scores(&s)),
        None => Ok(None),
    }
}

/// Parse pLDDT scores from gzipped mmCIF data (in-memory).
///
/// Reads the `_ma_qa_metric_local` block and extracts `metric_value` column.
/// Matches the exact parsing logic from Python `extract_features.py`.
pub fn parse_cif_gz(data: &[u8]) -> Result<Option<PlddtResult>> {
    let decoder = GzDecoder::new(data);
    let reader = BufReader::with_capacity(32 * 1024, decoder);
    parse_cif_reader(reader)
}

/// Parse pLDDT scores from uncompressed mmCIF data (in-memory).
pub fn parse_cif(data: &[u8]) -> Result<Option<PlddtResult>> {
    let reader = BufReader::new(data);
    parse_cif_reader(reader)
}

/// Internal CIF parser that works with any BufRead source.
fn parse_cif_reader<R: Read>(reader: BufReader<R>) -> Result<Option<PlddtResult>> {
    let mut scores: Vec<f64> = Vec::new();
    let mut in_local_block = false;
    let mut headers: Vec<String> = Vec::new();
    let mut metric_value_idx: Option<usize> = None;

    for line_result in reader.lines() {
        let line = line_result.context("failed to read CIF line")?;

        if line.contains("_ma_qa_metric_local.") {
            in_local_block = true;
            // Extract field name after the dot
            if let Some(field) = line.trim().split('.').last() {
                if field == "metric_value" {
                    metric_value_idx = Some(headers.len());
                }
                headers.push(field.to_string());
            }
        } else if in_local_block {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            // Block ends at '#' or new '_' definition
            if trimmed.starts_with('#') || trimmed.starts_with('_') {
                in_local_block = false;
                // If we hit another block, done with local metrics
                if !scores.is_empty() {
                    break;
                }
                continue;
            }

            // Parse data row: "A MET 1   2 62.53 1 1"
            if let Some(val_idx) = metric_value_idx {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if val_idx < parts.len() {
                    if let Ok(val) = parts[val_idx].parse::<f64>() {
                        scores.push(val);
                    }
                }
            }
        }
    }

    Ok(bin_plddt_scores(&scores))
}

/// Create PlddtResult from just mean pLDDT (no per-residue data).
///
/// Used with BigQuery metadata where only the global metric is available.
/// Only assigns the mean bin (items 0-2), not fraction bins (3-5).
pub fn from_mean_only(mean_plddt: f64) -> Option<PlddtResult> {
    if mean_plddt.is_nan() || mean_plddt < 0.0 {
        return None;
    }
    let mut items = Vec::with_capacity(1);
    if mean_plddt < 50.0 {
        items.push(PLDDT_MEAN_BASE);
    } else if mean_plddt <= 90.0 {
        items.push(PLDDT_MEAN_BASE + 1);
    } else {
        items.push(PLDDT_MEAN_BASE + 2);
    }
    Some(PlddtResult { items, mean_plddt })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_mean_only_bins() {
        let low = from_mean_only(30.0).unwrap();
        assert_eq!(low.items, vec![0]);
        let med = from_mean_only(70.0).unwrap();
        assert_eq!(med.items, vec![1]);
        let high = from_mean_only(95.0).unwrap();
        assert_eq!(high.items, vec![2]);
        assert!(from_mean_only(-1.0).is_none());
        assert!(from_mean_only(f64::NAN).is_none());
    }

    #[test]
    fn test_high_confidence_json() {
        let json = br#"{"confidenceScore": [95.0, 92.0, 98.0, 91.0]}"#;
        let result = parse_confidence_json(json).unwrap().unwrap();
        assert_eq!(result.items, vec![2, 5]); // high mean, high frac
        assert!((result.mean_plddt - 94.0).abs() < 0.01);
    }

    #[test]
    fn test_medium_confidence_json() {
        let json = br#"{"confidenceScore": [70.0, 75.0, 80.0, 65.0]}"#;
        let result = parse_confidence_json(json).unwrap().unwrap();
        assert_eq!(result.items, vec![1, 3]); // med mean, low frac
    }

    #[test]
    fn test_low_confidence_json() {
        let json = br#"{"confidenceScore": [30.0, 40.0, 20.0, 45.0]}"#;
        let result = parse_confidence_json(json).unwrap().unwrap();
        assert_eq!(result.items, vec![0, 3]); // low mean, low frac
    }

    #[test]
    fn test_mixed_frac_medium_json() {
        let json = br#"{"confidenceScore": [95.0, 95.0, 50.0, 50.0]}"#;
        let result = parse_confidence_json(json).unwrap().unwrap();
        assert_eq!(result.items[1], 4); // frac medium
    }

    #[test]
    fn test_plddt_field_fallback() {
        let json = br#"{"pLDDT": [85.0, 88.0]}"#;
        let result = parse_confidence_json(json).unwrap().unwrap();
        assert_eq!(result.items[0], 1); // med mean
    }

    #[test]
    fn test_empty_scores() {
        let json = br#"{"confidenceScore": []}"#;
        assert!(parse_confidence_json(json).unwrap().is_none());
    }

    #[test]
    fn test_boundary_50() {
        let json = br#"{"confidenceScore": [50.0]}"#;
        let result = parse_confidence_json(json).unwrap().unwrap();
        assert_eq!(result.items[0], 1); // med (not low)
    }

    #[test]
    fn test_boundary_90() {
        let json = br#"{"confidenceScore": [90.0]}"#;
        let result = parse_confidence_json(json).unwrap().unwrap();
        assert_eq!(result.items[0], 1); // med (not high)
        assert_eq!(result.items[1], 3); // low frac
    }

    #[test]
    fn test_cif_parsing() {
        let cif = b"data_AF-TEST\n\
_ma_qa_metric_local.label_asym_id\n\
_ma_qa_metric_local.label_comp_id\n\
_ma_qa_metric_local.label_seq_id\n\
_ma_qa_metric_local.metric_id\n\
_ma_qa_metric_local.metric_value\n\
_ma_qa_metric_local.model_id\n\
_ma_qa_metric_local.ordinal_id\n\
A MET 1   2 62.53 1 1\n\
A GLY 2   2 83.00 1 2\n\
A LYS 3   2 93.81 1 3\n\
A ILE 4   2 98.12 1 4\n\
#\n";
        let result = parse_cif(cif).unwrap().unwrap();
        // mean = (62.53 + 83.00 + 93.81 + 98.12) / 4 = 84.365
        assert!((result.mean_plddt - 84.365).abs() < 0.01);
        assert_eq!(result.items[0], 1); // med mean
        // frac_high = 2/4 = 0.5 → medium bin
        assert_eq!(result.items[1], 4); // med frac
    }
}
