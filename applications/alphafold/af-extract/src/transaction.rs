//! Build ET-miner compatible transaction parquet files.
//!
//! Two-pass design:
//! 1. `count-frequencies`: Count Pfam/GO occurrences across all proteins
//! 2. `build-transactions`: Select top-N features, assign item IDs, write parquet
//!
//! Output schema matches Python `extract_features.py`:
//! - `protein_id`: Utf8 (UniProt accession)
//! - `items`: List<Int64> (sorted item IDs)

use std::collections::HashMap;
use std::fs::File;
use std::io::Write as IoWrite;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{
    ArrayRef, Int64Builder, ListBuilder, StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use log::info;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use serde::{Deserialize, Serialize};

use crate::annotations::AnnotationDb;
use crate::tar_stream::ExtractedProtein;

/// Frequency counts for the two-pass design.
#[derive(Debug, Serialize, Deserialize)]
pub struct FrequencyCounts {
    /// Pfam domain → occurrence count
    pub pfam: HashMap<String, u64>,
    /// GO term → occurrence count
    pub go: HashMap<String, u64>,
    /// InterPro ID → occurrence count
    #[serde(default)]
    pub interpro: HashMap<String, u64>,
    /// EC number → occurrence count
    #[serde(default)]
    pub ec: HashMap<String, u64>,
    /// Taxonomy bin → occurrence count
    #[serde(default)]
    pub taxonomy: HashMap<String, u64>,
    /// Total proteins processed
    pub total_proteins: u64,
}

impl FrequencyCounts {
    pub fn new() -> Self {
        Self {
            pfam: HashMap::new(),
            go: HashMap::new(),
            interpro: HashMap::new(),
            ec: HashMap::new(),
            taxonomy: HashMap::new(),
            total_proteins: 0,
        }
    }

    /// Count annotations for a protein.
    pub fn count_protein(&mut self, annot_db: &AnnotationDb, accession: &str) {
        self.total_proteins += 1;

        if let Some(annot) = annot_db.get(accession) {
            for &pfam_id in &annot.pfam_ids {
                let pfam_str = annot_db.pfam_interner.resolve(pfam_id);
                *self.pfam.entry(pfam_str.to_string()).or_insert(0) += 1;
            }
            for &go_id in &annot.go_ids {
                let go_str = annot_db.go_interner.resolve(go_id);
                *self.go.entry(go_str.to_string()).or_insert(0) += 1;
            }
            for &ipr_id in &annot.interpro_ids {
                let ipr_str = annot_db.interpro_interner.resolve(ipr_id);
                *self.interpro.entry(ipr_str.to_string()).or_insert(0) += 1;
            }
            for &ec_id in &annot.ec_numbers {
                let ec_str = annot_db.ec_interner.resolve(ec_id);
                *self.ec.entry(ec_str.to_string()).or_insert(0) += 1;
            }
            for &tax_id in &annot.taxonomy_lineage {
                let tax_str = annot_db.taxonomy_interner.resolve(tax_id);
                *self.taxonomy.entry(tax_str.to_string()).or_insert(0) += 1;
            }
        }
    }

    /// Save frequencies to JSON.
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        info!(
            "Saved frequencies: {} Pfam, {} GO, {} InterPro, {} EC, {} taxonomy",
            self.pfam.len(), self.go.len(), self.interpro.len(),
            self.ec.len(), self.taxonomy.len(),
        );
        Ok(())
    }

    /// Load frequencies from JSON.
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let freq: Self = serde_json::from_str(&data)?;
        info!(
            "Loaded frequencies: {} Pfam, {} GO, {} InterPro, {} EC, {} taxonomy, {} total proteins",
            freq.pfam.len(), freq.go.len(), freq.interpro.len(),
            freq.ec.len(), freq.taxonomy.len(), freq.total_proteins,
        );
        Ok(freq)
    }
}

/// Dynamic item encoding based on top-N feature selection.
pub struct ItemEncoder {
    /// Whether pLDDT items are included (items 0-5)
    include_plddt: bool,
    pfam_to_item: HashMap<String, u32>,
    go_to_item: HashMap<String, u32>,
    interpro_to_item: HashMap<String, u32>,
    ec_to_item: HashMap<String, u32>,
    taxonomy_to_item: HashMap<String, u32>,
    pub total_items: u32,
}

/// Helper: sort by frequency descending, take top-N, assign contiguous item IDs starting at `base`.
fn build_top_n_map(
    freq_map: &HashMap<String, u64>,
    top_n: usize,
    base: u32,
) -> HashMap<String, u32> {
    let mut sorted: Vec<(&String, &u64)> = freq_map.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    sorted
        .iter()
        .take(top_n)
        .enumerate()
        .map(|(i, (name, _))| ((*name).clone(), base + i as u32))
        .collect()
}

impl ItemEncoder {
    /// Create encoder from frequency counts, selecting top-N features.
    ///
    /// Layout: `[pLDDT(0-5) | Pfam | GO | InterPro | EC | Taxonomy]`
    pub fn from_frequencies(
        freq: &FrequencyCounts,
        top_pfam: usize,
        top_go: usize,
        top_interpro: usize,
        top_ec: usize,
        top_taxonomy: usize,
        include_plddt: bool,
    ) -> Self {
        let pfam_base: u32 = if include_plddt { 6 } else { 0 };

        let pfam_to_item = build_top_n_map(&freq.pfam, top_pfam, pfam_base);
        let go_base = pfam_base + pfam_to_item.len() as u32;

        let go_to_item = build_top_n_map(&freq.go, top_go, go_base);
        let interpro_base = go_base + go_to_item.len() as u32;

        let interpro_to_item = build_top_n_map(&freq.interpro, top_interpro, interpro_base);
        let ec_base = interpro_base + interpro_to_item.len() as u32;

        let ec_to_item = build_top_n_map(&freq.ec, top_ec, ec_base);
        let taxonomy_base = ec_base + ec_to_item.len() as u32;

        let taxonomy_to_item = build_top_n_map(&freq.taxonomy, top_taxonomy, taxonomy_base);
        let total_items = taxonomy_base + taxonomy_to_item.len() as u32;

        if include_plddt {
            info!(
                "Item encoding: 6 pLDDT + {} Pfam + {} GO + {} InterPro + {} EC + {} taxonomy = {} total items",
                pfam_to_item.len(), go_to_item.len(),
                interpro_to_item.len(), ec_to_item.len(), taxonomy_to_item.len(),
                total_items,
            );
        } else {
            info!(
                "Item encoding: {} Pfam + {} GO + {} InterPro + {} EC + {} taxonomy = {} total items (no pLDDT)",
                pfam_to_item.len(), go_to_item.len(),
                interpro_to_item.len(), ec_to_item.len(), taxonomy_to_item.len(),
                total_items,
            );
        }

        Self {
            include_plddt,
            pfam_to_item,
            go_to_item,
            interpro_to_item,
            ec_to_item,
            taxonomy_to_item,
            total_items,
        }
    }

    /// Encode a protein into item IDs.
    pub fn encode(
        &self,
        protein: &ExtractedProtein,
        annot_db: &AnnotationDb,
    ) -> Vec<i64> {
        let mut items: Vec<u32> = if self.include_plddt {
            protein.plddt.items.clone()
        } else {
            Vec::new()
        };

        // Add annotation items
        if let Some(annot) = annot_db.get(&protein.accession) {
            for &pfam_id in &annot.pfam_ids {
                let pfam_str = annot_db.pfam_interner.resolve(pfam_id);
                if let Some(&item_id) = self.pfam_to_item.get(pfam_str) {
                    items.push(item_id);
                }
            }
            for &go_id in &annot.go_ids {
                let go_str = annot_db.go_interner.resolve(go_id);
                if let Some(&item_id) = self.go_to_item.get(go_str) {
                    items.push(item_id);
                }
            }
            for &ipr_id in &annot.interpro_ids {
                let ipr_str = annot_db.interpro_interner.resolve(ipr_id);
                if let Some(&item_id) = self.interpro_to_item.get(ipr_str) {
                    items.push(item_id);
                }
            }
            for &ec_id in &annot.ec_numbers {
                let ec_str = annot_db.ec_interner.resolve(ec_id);
                if let Some(&item_id) = self.ec_to_item.get(ec_str) {
                    items.push(item_id);
                }
            }
            for &tax_id in &annot.taxonomy_lineage {
                let tax_str = annot_db.taxonomy_interner.resolve(tax_id);
                if let Some(&item_id) = self.taxonomy_to_item.get(tax_str) {
                    items.push(item_id);
                }
            }
        }

        // Deduplicate and sort
        items.sort_unstable();
        items.dedup();

        items.into_iter().map(|x| x as i64).collect()
    }

    /// Write item mapping parquet (item_id → feature_name, feature_category).
    pub fn write_item_mapping(&self, path: &Path) -> Result<()> {
        let mut ids = Vec::new();
        let mut names = Vec::new();
        let mut categories = Vec::new();

        // pLDDT items (only when included)
        if self.include_plddt {
            let plddt_labels = [
                (0, "plddt_mean_low", "plddt_mean"),
                (1, "plddt_mean_med", "plddt_mean"),
                (2, "plddt_mean_high", "plddt_mean"),
                (3, "plddt_frac_high_low", "plddt_fraction"),
                (4, "plddt_frac_high_med", "plddt_fraction"),
                (5, "plddt_frac_high_high", "plddt_fraction"),
            ];
            for (id, name, cat) in &plddt_labels {
                ids.push(*id as i64);
                names.push(*name);
                categories.push(*cat);
            }
        }

        // Collect all feature categories sorted by item ID
        let category_maps: &[(&HashMap<String, u32>, &str)] = &[
            (&self.pfam_to_item, "pfam"),
            (&self.go_to_item, "go_term"),
            (&self.interpro_to_item, "interpro"),
            (&self.ec_to_item, "ec_number"),
            (&self.taxonomy_to_item, "taxonomy"),
        ];
        for (map, category) in category_maps {
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by_key(|(_, &id)| id);
            for (name, &id) in &entries {
                ids.push(id as i64);
                names.push(name.as_str());
                categories.push(category);
            }
        }

        // Build Arrow arrays
        let id_array: ArrayRef = Arc::new(arrow::array::Int64Array::from(ids));
        let name_array: ArrayRef = Arc::new(StringArray::from(
            names.iter().map(|s| *s).collect::<Vec<&str>>(),
        ));
        let cat_array: ArrayRef = Arc::new(StringArray::from(
            categories.iter().map(|s| *s).collect::<Vec<&str>>(),
        ));

        let schema = Arc::new(Schema::new(vec![
            Field::new("item_id", DataType::Int64, false),
            Field::new("feature_name", DataType::Utf8, false),
            Field::new("feature_category", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(schema.clone(), vec![id_array, name_array, cat_array])?;

        let file = File::create(path)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;

        info!("Saved item mapping: {} items to {}", batch.num_rows(), path.display());
        Ok(())
    }
}

/// Write proteins as a transaction parquet file.
///
/// Output schema:
/// - `protein_id`: Utf8
/// - `items`: List<Int64>
pub fn write_transactions_parquet(
    proteins: &[ExtractedProtein],
    annot_db: &AnnotationDb,
    encoder: &ItemEncoder,
    output_path: &Path,
    batch_size: usize,
) -> Result<u64> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("protein_id", DataType::Utf8, false),
        Field::new(
            "items",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            false,
        ),
    ]));

    let file = File::create(output_path)
        .with_context(|| format!("failed to create parquet: {}", output_path.display()))?;

    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_max_row_group_size(batch_size)
        .build();

    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

    let mut total_rows = 0u64;
    let mut batch_proteins = Vec::new();
    let mut batch_items = Vec::new();

    for protein in proteins {
        let items = encoder.encode(protein, annot_db);
        batch_proteins.push(protein.accession.as_str());
        batch_items.push(items);
        total_rows += 1;

        if batch_proteins.len() >= batch_size {
            write_batch(&mut writer, &schema, &batch_proteins, &batch_items)?;
            batch_proteins.clear();
            batch_items.clear();

            if total_rows % 1_000_000 == 0 {
                info!("  Written {}M transactions...", total_rows / 1_000_000);
            }
        }
    }

    // Write remaining
    if !batch_proteins.is_empty() {
        write_batch(&mut writer, &schema, &batch_proteins, &batch_items)?;
    }

    writer.close()?;

    info!(
        "Written {} transactions to {}",
        total_rows,
        output_path.display()
    );

    Ok(total_rows)
}

/// Write a batch of rows to the parquet writer.
fn write_batch(
    writer: &mut ArrowWriter<File>,
    schema: &Arc<Schema>,
    proteins: &[&str],
    items: &[Vec<i64>],
) -> Result<()> {
    let protein_array: ArrayRef = Arc::new(StringArray::from(proteins.to_vec()));

    let mut list_builder = ListBuilder::new(Int64Builder::new());
    for item_list in items {
        let values = list_builder.values();
        for &item in item_list {
            values.append_value(item);
        }
        list_builder.append(true);
    }
    let items_array: ArrayRef = Arc::new(list_builder.finish());

    let batch = RecordBatch::try_new(schema.clone(), vec![protein_array, items_array])?;
    writer.write(&batch)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_counts() {
        let mut freq = FrequencyCounts::new();
        freq.pfam.insert("PF00089".to_string(), 100);
        freq.pfam.insert("PF00090".to_string(), 50);
        freq.go.insert("GO:0006915".to_string(), 200);

        let encoder = ItemEncoder::from_frequencies(&freq, 10, 10, 0, 0, 0, true);
        assert_eq!(encoder.total_items, 6 + 2 + 1); // 6 pLDDT + 2 Pfam + 1 GO

        // PF00089 (most frequent) should get item 6
        assert_eq!(encoder.pfam_to_item["PF00089"], 6);
        // PF00090 should get item 7
        assert_eq!(encoder.pfam_to_item["PF00090"], 7);
        // GO should get item 8
        assert_eq!(encoder.go_to_item["GO:0006915"], 8);
    }

    #[test]
    fn test_top_n_selection() {
        let mut freq = FrequencyCounts::new();
        for i in 0..300 {
            freq.pfam.insert(format!("PF{:05}", i), (300 - i) as u64);
        }

        let encoder = ItemEncoder::from_frequencies(&freq, 200, 0, 0, 0, 0, true);
        assert_eq!(encoder.pfam_to_item.len(), 200);
        assert_eq!(encoder.total_items, 6 + 200);

        // PF00000 (highest freq=300) should be included
        assert!(encoder.pfam_to_item.contains_key("PF00000"));
        // PF00299 (lowest freq=1) should NOT be included
        assert!(!encoder.pfam_to_item.contains_key("PF00299"));
    }

    #[test]
    fn test_expanded_encoding_layout() {
        let mut freq = FrequencyCounts::new();
        freq.pfam.insert("PF00001".to_string(), 100);
        freq.go.insert("GO:0000001".to_string(), 80);
        freq.interpro.insert("IPR000001".to_string(), 60);
        freq.ec.insert("2.7.11.1".to_string(), 40);
        freq.taxonomy.insert("Eukaryota".to_string(), 200);
        freq.taxonomy.insert("Metazoa".to_string(), 150);

        let encoder = ItemEncoder::from_frequencies(&freq, 10, 10, 10, 10, 10, true);
        // Layout: 6 pLDDT + 1 Pfam + 1 GO + 1 InterPro + 1 EC + 2 taxonomy = 12
        assert_eq!(encoder.total_items, 12);
        assert_eq!(encoder.pfam_to_item["PF00001"], 6);
        assert_eq!(encoder.go_to_item["GO:0000001"], 7);
        assert_eq!(encoder.interpro_to_item["IPR000001"], 8);
        assert_eq!(encoder.ec_to_item["2.7.11.1"], 9);
        // Taxonomy: Eukaryota (freq=200) → 10, Metazoa (freq=150) → 11
        assert_eq!(encoder.taxonomy_to_item["Eukaryota"], 10);
        assert_eq!(encoder.taxonomy_to_item["Metazoa"], 11);
    }

    #[test]
    fn test_no_plddt_layout() {
        let mut freq = FrequencyCounts::new();
        freq.pfam.insert("PF00001".to_string(), 100);
        freq.interpro.insert("IPR000001".to_string(), 50);

        let encoder = ItemEncoder::from_frequencies(&freq, 10, 0, 10, 0, 0, false);
        // No pLDDT: 1 Pfam + 0 GO + 1 InterPro = 2
        assert_eq!(encoder.total_items, 2);
        assert_eq!(encoder.pfam_to_item["PF00001"], 0);
        assert_eq!(encoder.interpro_to_item["IPR000001"], 1);
    }
}
