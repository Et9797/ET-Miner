//! Load UniProt annotations (TSV or DAT format) into a fast lookup.
//!
//! Supports two formats:
//! 1. UniProt TSV (tab-separated, from REST API bulk download)
//! 2. UniProt DAT (Swiss-Prot flat file, gzipped)
//!
//! String interning reduces memory from ~17GB to ~8GB for 214M entries.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use anyhow::{Context, Result};
use dashmap::DashMap;
use flate2::read::MultiGzDecoder;
use log::info;

/// Annotation data for a single protein.
#[derive(Debug, Clone, Default)]
pub struct ProteinAnnotation {
    /// Interned Pfam domain IDs (e.g., index into string table)
    pub pfam_ids: Vec<u32>,
    /// Interned GO term IDs
    pub go_ids: Vec<u32>,
    /// Interned InterPro superfamily/family IDs
    pub interpro_ids: Vec<u32>,
    /// Interned Enzyme Commission numbers
    pub ec_numbers: Vec<u32>,
    /// Interned taxonomy lineage bins (top 2 levels)
    pub taxonomy_lineage: Vec<u32>,
}

/// String interner for memory-efficient annotation storage.
#[derive(Debug)]
pub struct StringInterner {
    map: HashMap<String, u32>,
    strings: Vec<String>,
}

impl StringInterner {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            strings: Vec::new(),
        }
    }

    /// Intern a string, returning its stable ID.
    pub fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        let id = self.strings.len() as u32;
        self.strings.push(s.to_string());
        self.map.insert(s.to_string(), id);
        id
    }

    /// Get the string for an interned ID.
    pub fn resolve(&self, id: u32) -> &str {
        &self.strings[id as usize]
    }

    /// Total number of unique strings.
    pub fn len(&self) -> usize {
        self.strings.len()
    }
}

/// Loaded annotation database.
pub struct AnnotationDb {
    /// Accession → annotation data
    pub entries: DashMap<String, ProteinAnnotation>,
    /// Pfam string interner
    pub pfam_interner: StringInterner,
    /// GO string interner
    pub go_interner: StringInterner,
    /// InterPro string interner
    pub interpro_interner: StringInterner,
    /// EC number string interner
    pub ec_interner: StringInterner,
    /// Taxonomy lineage string interner
    pub taxonomy_interner: StringInterner,
}

impl AnnotationDb {
    /// Load annotations from a UniProt TSV file.
    ///
    /// Expected columns (tab-separated, with header):
    /// - Entry (accession)
    /// - Cross-reference (Pfam) — semicolon-separated Pfam IDs
    /// - Gene Ontology IDs — semicolon-separated GO IDs
    pub fn from_tsv(path: &Path) -> Result<Self> {
        info!("Loading annotations from TSV: {}", path.display());

        let file = File::open(path).context("failed to open TSV")?;
        let reader: Box<dyn Read> = if path.extension().map_or(false, |e| e == "gz") {
            Box::new(MultiGzDecoder::new(file))
        } else {
            Box::new(file)
        };
        let buf = BufReader::with_capacity(1 << 20, reader); // 1MB buffer

        let entries = DashMap::new();
        let mut pfam_interner = StringInterner::new();
        let mut go_interner = StringInterner::new();
        let mut interpro_interner = StringInterner::new();
        let mut ec_interner = StringInterner::new();
        let mut taxonomy_interner = StringInterner::new();

        let mut lines = buf.lines();

        // Parse header to find column indices
        let header = lines
            .next()
            .context("empty TSV file")?
            .context("failed to read header")?;
        let cols: Vec<&str> = header.split('\t').collect();

        let acc_idx = cols
            .iter()
            .position(|&c| c == "Entry")
            .context("missing 'Entry' column")?;
        let pfam_idx = cols
            .iter()
            .position(|&c| c.contains("Pfam"));
        let go_idx = cols
            .iter()
            .position(|&c| c.contains("Gene Ontology") || c.contains("GO"));
        let interpro_idx = cols
            .iter()
            .position(|&c| c.contains("InterPro"));
        let ec_idx = cols
            .iter()
            .position(|&c| c.contains("EC number") || c == "EC");
        let taxonomy_idx = cols
            .iter()
            .position(|&c| c.contains("Taxonomic lineage") || c.contains("Organism") || c == "OC");

        let mut count = 0u64;

        for line_result in lines {
            let line = line_result.context("failed to read TSV line")?;
            let fields: Vec<&str> = line.split('\t').collect();

            if fields.len() <= acc_idx {
                continue;
            }

            let accession = fields[acc_idx].trim();
            if accession.is_empty() {
                continue;
            }

            let mut annot = ProteinAnnotation::default();

            // Parse Pfam domains
            if let Some(idx) = pfam_idx {
                if let Some(field) = fields.get(idx) {
                    for domain in field.split(';') {
                        let d = domain.trim();
                        if !d.is_empty() {
                            annot.pfam_ids.push(pfam_interner.intern(d));
                        }
                    }
                }
            }

            // Parse GO terms
            if let Some(idx) = go_idx {
                if let Some(field) = fields.get(idx) {
                    for term in field.split(';') {
                        let t = term.trim();
                        if !t.is_empty() {
                            annot.go_ids.push(go_interner.intern(t));
                        }
                    }
                }
            }

            // Parse InterPro IDs
            if let Some(idx) = interpro_idx {
                if let Some(field) = fields.get(idx) {
                    for ipr in field.split(';') {
                        let i = ipr.trim();
                        if !i.is_empty() {
                            annot.interpro_ids.push(interpro_interner.intern(i));
                        }
                    }
                }
            }

            // Parse EC numbers
            if let Some(idx) = ec_idx {
                if let Some(field) = fields.get(idx) {
                    for ec in field.split(';') {
                        let e = ec.trim();
                        if !e.is_empty() {
                            annot.ec_numbers.push(ec_interner.intern(e));
                        }
                    }
                }
            }

            // Parse taxonomy lineage (take top 2 levels)
            if let Some(idx) = taxonomy_idx {
                if let Some(field) = fields.get(idx) {
                    for (level, taxon) in field.split(';').enumerate() {
                        if level >= 2 { break; }
                        let t = taxon.trim();
                        if !t.is_empty() {
                            annot.taxonomy_lineage.push(taxonomy_interner.intern(t));
                        }
                    }
                }
            }

            entries.insert(accession.to_string(), annot);
            count += 1;

            if count % 5_000_000 == 0 {
                info!("  Loaded {}M annotations...", count / 1_000_000);
            }
        }

        info!(
            "Loaded {} annotations ({} Pfam, {} GO, {} InterPro, {} EC, {} taxonomy)",
            count,
            pfam_interner.len(),
            go_interner.len(),
            interpro_interner.len(),
            ec_interner.len(),
            taxonomy_interner.len(),
        );

        Ok(Self {
            entries,
            pfam_interner,
            go_interner,
            interpro_interner,
            ec_interner,
            taxonomy_interner,
        })
    }

    /// Load annotations from a Swiss-Prot DAT file (gzipped).
    ///
    /// Parses the same format as Python `download_alphafold.py:parse_uniprot_dat()`.
    pub fn from_dat_gz(path: &Path) -> Result<Self> {
        info!("Loading annotations from DAT: {}", path.display());

        let file = File::open(path).context("failed to open DAT file")?;
        let reader: Box<dyn Read> = if path.extension().map_or(false, |e| e == "gz") {
            Box::new(MultiGzDecoder::new(file))
        } else {
            Box::new(file)
        };
        let buf = BufReader::with_capacity(1 << 20, reader);

        let entries = DashMap::new();
        let mut pfam_interner = StringInterner::new();
        let mut go_interner = StringInterner::new();
        let mut interpro_interner = StringInterner::new();
        let mut ec_interner = StringInterner::new();
        let mut taxonomy_interner = StringInterner::new();

        let mut current_id: Option<String> = None;
        let mut pfam_ids: Vec<u32> = Vec::new();
        let mut go_ids: Vec<u32> = Vec::new();
        let mut interpro_ids: Vec<u32> = Vec::new();
        let mut ec_numbers: Vec<u32> = Vec::new();
        let mut taxonomy_lineage: Vec<u32> = Vec::new();
        let mut count = 0u64;

        for line_result in buf.lines() {
            let line = line_result.context("failed to read DAT line")?;

            if line.starts_with("AC   ") && current_id.is_none() {
                // First accession: "AC   Q9Y6K9; Q9Y6K8;"
                let acc_part = &line[5..];
                if let Some(acc) = acc_part.trim().trim_end_matches(';').split(';').next() {
                    current_id = Some(acc.trim().to_string());
                }
            } else if line.starts_with("DR   Pfam;") {
                // "DR   Pfam; PF00089; Trypsin; 1."
                let parts: Vec<&str> = line.split(';').collect();
                if parts.len() >= 2 {
                    let pfam_id = parts[1].trim();
                    pfam_ids.push(pfam_interner.intern(pfam_id));
                }
            } else if line.starts_with("DR   GO;") {
                // "DR   GO; GO:0006915; P:apoptotic process; IEA:UniProtKB-KW."
                let parts: Vec<&str> = line.split(';').collect();
                if parts.len() >= 2 {
                    let go_id = parts[1].trim();
                    go_ids.push(go_interner.intern(go_id));
                }
            } else if line.starts_with("DR   InterPro;") {
                // "DR   InterPro; IPR000001; Kringle."
                let parts: Vec<&str> = line.split(';').collect();
                if parts.len() >= 2 {
                    let ipr_id = parts[1].trim();
                    interpro_ids.push(interpro_interner.intern(ipr_id));
                }
            } else if line.starts_with("DE   ") && line.contains("EC=") {
                // "DE            EC=3.4.21.4 {evidence};" or "DE   EC=3.4.21.4, EC=1.2.3.4;"
                // Handle multiple EC numbers on a single DE line
                let mut search_from = 0usize;
                while let Some(pos) = line[search_from..].find("EC=") {
                    let abs_pos = search_from + pos + 3;
                    let after_ec = &line[abs_pos..];
                    // EC number ends at space, semicolon, brace, or comma
                    let ec_end = after_ec.find(|c: char| c == ' ' || c == ';' || c == '{' || c == ',')
                        .unwrap_or(after_ec.len());
                    let ec_num = after_ec[..ec_end].trim();
                    if !ec_num.is_empty() && ec_num.contains('.') {
                        ec_numbers.push(ec_interner.intern(ec_num));
                    }
                    search_from = abs_pos + ec_end;
                }
            } else if line.starts_with("OC   ") {
                // "OC   Eukaryota; Metazoa; Chordata; ..."
                // Accumulate taxonomy — may span multiple OC lines.
                // We only keep top 2 levels, so skip if already have 2.
                if taxonomy_lineage.len() < 2 {
                    let lineage_part = &line[5..];
                    for taxon in lineage_part.split(';') {
                        if taxonomy_lineage.len() >= 2 { break; }
                        let t = taxon.trim().trim_end_matches('.');
                        if !t.is_empty() {
                            taxonomy_lineage.push(taxonomy_interner.intern(t));
                        }
                    }
                }
            } else if line.starts_with("//") {
                // End of record
                if let Some(acc) = current_id.take() {
                    // Deduplicate within protein
                    pfam_ids.sort_unstable();
                    pfam_ids.dedup();
                    go_ids.sort_unstable();
                    go_ids.dedup();
                    interpro_ids.sort_unstable();
                    interpro_ids.dedup();
                    ec_numbers.sort_unstable();
                    ec_numbers.dedup();
                    taxonomy_lineage.sort_unstable();
                    taxonomy_lineage.dedup();

                    entries.insert(
                        acc,
                        ProteinAnnotation {
                            pfam_ids: std::mem::take(&mut pfam_ids),
                            go_ids: std::mem::take(&mut go_ids),
                            interpro_ids: std::mem::take(&mut interpro_ids),
                            ec_numbers: std::mem::take(&mut ec_numbers),
                            taxonomy_lineage: std::mem::take(&mut taxonomy_lineage),
                        },
                    );
                    count += 1;

                    if count % 100_000 == 0 {
                        info!("  Parsed {}K DAT records...", count / 1000);
                    }
                }
                pfam_ids.clear();
                go_ids.clear();
                interpro_ids.clear();
                ec_numbers.clear();
                taxonomy_lineage.clear();
            }
        }

        info!(
            "Loaded {} annotations ({} Pfam, {} GO, {} InterPro, {} EC, {} taxonomy)",
            count,
            pfam_interner.len(),
            go_interner.len(),
            interpro_interner.len(),
            ec_interner.len(),
            taxonomy_interner.len(),
        );

        Ok(Self {
            entries,
            pfam_interner,
            go_interner,
            interpro_interner,
            ec_interner,
            taxonomy_interner,
        })
    }

    /// Look up annotations for an accession.
    pub fn get(&self, accession: &str) -> Option<dashmap::mapref::one::Ref<'_, String, ProteinAnnotation>> {
        self.entries.get(accession)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_string_interner() {
        let mut interner = StringInterner::new();
        let a = interner.intern("PF00089");
        let b = interner.intern("PF00090");
        let c = interner.intern("PF00089"); // duplicate

        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(c, 0); // same as first
        assert_eq!(interner.resolve(a), "PF00089");
        assert_eq!(interner.len(), 2);
    }

    fn make_dat_file(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::with_suffix(".dat").unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn test_dat_interpro_parsing() {
        let dat = make_dat_file(
            "AC   Q9Y6K9;\n\
             DR   InterPro; IPR000001; Kringle.\n\
             DR   InterPro; IPR000742; EGF-like domain.\n\
             DR   Pfam; PF00089; Trypsin; 1.\n\
             //\n"
        );
        let db = AnnotationDb::from_dat_gz(dat.path()).unwrap();
        let annot = db.entries.get("Q9Y6K9").unwrap();
        assert_eq!(annot.interpro_ids.len(), 2);
        let names: Vec<&str> = annot.interpro_ids.iter()
            .map(|&id| db.interpro_interner.resolve(id))
            .collect();
        assert!(names.contains(&"IPR000001"));
        assert!(names.contains(&"IPR000742"));
    }

    #[test]
    fn test_dat_ec_parsing() {
        let dat = make_dat_file(
            "AC   P12345;\n\
             DE   RecName: Full=Trypsin;\n\
             DE            EC=3.4.21.4;\n\
             DE   AltName: Full=Something EC=1.2.3.4 {ECO:0000};\n\
             //\n"
        );
        let db = AnnotationDb::from_dat_gz(dat.path()).unwrap();
        let annot = db.entries.get("P12345").unwrap();
        assert_eq!(annot.ec_numbers.len(), 2);
        let nums: Vec<&str> = annot.ec_numbers.iter()
            .map(|&id| db.ec_interner.resolve(id))
            .collect();
        assert!(nums.contains(&"3.4.21.4"));
        assert!(nums.contains(&"1.2.3.4"));
    }

    #[test]
    fn test_dat_taxonomy_parsing() {
        let dat = make_dat_file(
            "AC   P00001;\n\
             OC   Eukaryota; Metazoa; Chordata; Mammalia.\n\
             //\n"
        );
        let db = AnnotationDb::from_dat_gz(dat.path()).unwrap();
        let annot = db.entries.get("P00001").unwrap();
        // Top 2 levels only
        assert_eq!(annot.taxonomy_lineage.len(), 2);
        let taxa: Vec<&str> = annot.taxonomy_lineage.iter()
            .map(|&id| db.taxonomy_interner.resolve(id))
            .collect();
        assert!(taxa.contains(&"Eukaryota"));
        assert!(taxa.contains(&"Metazoa"));
    }

    #[test]
    fn test_dat_combined_record() {
        let dat = make_dat_file(
            "AC   Q99999;\n\
             DE   RecName: Full=Kinase;\n\
             DE            EC=2.7.11.1;\n\
             OC   Eukaryota; Fungi; Ascomycota.\n\
             DR   Pfam; PF00069; Pkinase; 1.\n\
             DR   GO; GO:0004672; F:protein kinase activity; IEA:UniProtKB-KW.\n\
             DR   InterPro; IPR000719; Prot_kinase_dom.\n\
             //\n"
        );
        let db = AnnotationDb::from_dat_gz(dat.path()).unwrap();
        let annot = db.entries.get("Q99999").unwrap();
        assert_eq!(annot.pfam_ids.len(), 1);
        assert_eq!(annot.go_ids.len(), 1);
        assert_eq!(annot.interpro_ids.len(), 1);
        assert_eq!(annot.ec_numbers.len(), 1);
        assert_eq!(annot.taxonomy_lineage.len(), 2);
        assert_eq!(db.ec_interner.resolve(annot.ec_numbers[0]), "2.7.11.1");
    }
}
