// build_tx.rs — Rust TSV→transactions builder (zero deps, two-pass)
// Replaces the Python Pass 1 + Pass 2 in build_expanded_transactions.py
//
// Usage:
//   rustc -O build_tx.rs -o build_tx
//   ./build_tx <proteins.txt> <expanded.tsv> [min_count=8]
//
// Outputs:
//   stdout: transactions TSV (protein_id\titem1,item2,...)
//   stderr: progress + feature_mapping.tsv path
//   feature_mapping.tsv: written next to the input TSV

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Write, BufWriter};
use std::fs::File;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <proteins.txt> <expanded.tsv> [min_count=8]", args[0]);
        std::process::exit(1);
    }
    let proteins_file = &args[1];
    let tsv_file = &args[2];
    let min_count: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);

    // --- Load valid proteins ---
    let t0 = Instant::now();
    let proteins = load_proteins(proteins_file);
    eprintln!("  Loaded {} proteins in {:.1}s", proteins.len(), t0.elapsed().as_secs_f64());

    // --- Pass 1: count feature frequencies ---
    eprintln!("============================================================");
    eprintln!("PASS 1: Count feature frequencies");
    eprintln!("============================================================");
    let (counts, col_indices) = pass1(tsv_file, &proteins);
    eprintln!("  {} unique features found", counts.len());

    // --- Filter + assign IDs ---
    let mut surviving: Vec<(String, usize)> = counts.into_iter()
        .filter(|(_, c)| *c >= min_count)
        .collect();
    surviving.sort_by(|a, b| a.0.cmp(&b.0)); // deterministic order
    let feature_to_id: HashMap<String, u32> = surviving.iter()
        .enumerate()
        .map(|(i, (feat, _))| (feat.clone(), i as u32))
        .collect();
    eprintln!("  {} features with count >= {} (assigned IDs 0..{})",
        feature_to_id.len(), min_count, feature_to_id.len().saturating_sub(1));

    // Category breakdown
    let mut cat_counts: HashMap<&str, usize> = HashMap::new();
    for (feat, _) in &surviving {
        let cat = if feat.starts_with("IPR:") { "IPR:" }
            else if feat.starts_with("GO:") { "GO:" }
            else if feat.starts_with("EC:") { "EC:" }
            else if feat.starts_with("KW:") { "KW:" }
            else if feat.starts_with("TAX:") { "TAX:" }
            else if feat.starts_with("LEN:") { "LEN:" }
            else { "other" };
        *cat_counts.entry(cat).or_default() += 1;
    }
    let mut cats: Vec<_> = cat_counts.iter().collect();
    cats.sort_by_key(|(k, _)| k.to_string());
    for (cat, n) in &cats {
        eprintln!("  {:<8} {:>6} features", cat, n);
    }

    // Write feature mapping
    let mapping_path = tsv_file.replace(".tsv", "_feature_mapping.tsv");
    {
        let f = File::create(&mapping_path).expect("cannot create mapping file");
        let mut w = BufWriter::new(f);
        writeln!(w, "item_id\tfeature_name\tfeature_category\tcount").unwrap();
        for (i, (feat, count)) in surviving.iter().enumerate() {
            let cat = if feat.starts_with("IPR:") { "interpro" }
                else if feat.starts_with("GO:") { "go_term" }
                else if feat.starts_with("EC:") { "ec_number" }
                else if feat.starts_with("KW:") { "keyword" }
                else if feat.starts_with("TAX:") { "taxonomy" }
                else if feat.starts_with("LEN:") { "length_bin" }
                else { "other" };
            writeln!(w, "{}\t{}\t{}\t{}", i, feat, cat, count).unwrap();
        }
    }
    eprintln!("  Wrote feature mapping to {}", mapping_path);

    // --- Pass 2: build transactions ---
    eprintln!("============================================================");
    eprintln!("PASS 2: Build transactions");
    eprintln!("============================================================");
    pass2(tsv_file, &proteins, &feature_to_id, &col_indices);
}

fn load_proteins(path: &str) -> HashSet<String> {
    let f = File::open(path).expect("cannot open proteins file");
    let reader = BufReader::with_capacity(8 * 1024 * 1024, f);
    let mut set = HashSet::new();
    for line in reader.lines() {
        let line = line.unwrap();
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            set.insert(trimmed.to_string());
        }
    }
    set
}

#[derive(Clone)]
struct ColIndices {
    accession: Option<usize>,
    xref_interpro: Option<usize>,
    go_id: Option<usize>,
    ec: Option<usize>,
    keyword: Option<usize>,
    lineage: Option<usize>,
    length: Option<usize>,
}

fn detect_columns(header: &str) -> ColIndices {
    let cols: Vec<String> = header.split('\t').map(|s| s.trim().to_lowercase()).collect();
    let mut idx = ColIndices {
        accession: None, xref_interpro: None, go_id: None,
        ec: None, keyword: None, lineage: None, length: None,
    };
    for (i, col) in cols.iter().enumerate() {
        if col.contains("entry") || col.contains("accession") { idx.accession = Some(i); }
        else if col.contains("interpro") { idx.xref_interpro = Some(i); }
        else if col.contains("gene ontology") || col.contains("go id") || col.contains("go_id") { idx.go_id = Some(i); }
        else if col.contains("ec number") || col.contains("ec_number") || col == "ec" { idx.ec = Some(i); }
        else if col.contains("keyword") { idx.keyword = Some(i); }
        else if col.contains("taxonomic lineage") || col.contains("lineage") { idx.lineage = Some(i); }
        else if col.contains("length") { idx.length = Some(i); }
    }
    eprintln!("  Detected columns: acc={:?} ipr={:?} go={:?} ec={:?} kw={:?} tax={:?} len={:?}",
        idx.accession, idx.xref_interpro, idx.go_id, idx.ec, idx.keyword, idx.lineage, idx.length);
    idx
}

fn parse_features(fields: &[&str], idx: &ColIndices, features: &mut Vec<String>) {
    features.clear();

    // InterPro
    if let Some(i) = idx.xref_interpro {
        if let Some(val) = fields.get(i) {
            for entry in val.split(';') {
                let entry = entry.trim();
                if !entry.is_empty() {
                    features.push(format!("IPR:{}", entry));
                }
            }
        }
    }

    // GO
    if let Some(i) = idx.go_id {
        if let Some(val) = fields.get(i) {
            for entry in val.split(';') {
                let entry = entry.trim();
                if !entry.is_empty() {
                    if entry.starts_with("GO:") {
                        features.push(entry.to_string());
                    } else {
                        features.push(format!("GO:{}", entry));
                    }
                }
            }
        }
    }

    // EC (leaf only)
    if let Some(i) = idx.ec {
        if let Some(val) = fields.get(i) {
            for entry in val.split(';') {
                let entry = entry.trim();
                if !entry.is_empty() {
                    features.push(format!("EC:{}", entry));
                }
            }
        }
    }

    // Keywords
    if let Some(i) = idx.keyword {
        if let Some(val) = fields.get(i) {
            for entry in val.split(';') {
                let entry = entry.trim();
                if !entry.is_empty() {
                    features.push(format!("KW:{}", entry));
                }
            }
        }
    }

    // Taxonomy — class level (index 2)
    if let Some(i) = idx.lineage {
        if let Some(val) = fields.get(i) {
            let val = val.trim();
            if !val.is_empty() {
                let levels: Vec<&str> = val.split(';').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
                if !levels.is_empty() {
                    let tax_level = if levels.len() > 2 { levels[2] } else { levels[levels.len() - 1] };
                    if !tax_level.is_empty() {
                        features.push(format!("TAX:{}", tax_level));
                    }
                }
            }
        }
    }

    // Length bins (50-AA)
    if let Some(i) = idx.length {
        if let Some(val) = fields.get(i) {
            let val = val.trim().replace(",", "");
            if let Ok(length) = val.parse::<usize>() {
                let lo = (length / 50) * 50;
                features.push(format!("LEN:{}-{}", lo, lo + 50));
            }
        }
    }
}

fn pass1(tsv_path: &str, proteins: &HashSet<String>) -> (HashMap<String, usize>, ColIndices) {
    let t0 = Instant::now();
    let f = File::open(tsv_path).expect("cannot open TSV");
    let reader = BufReader::with_capacity(16 * 1024 * 1024, f);
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut n_lines: usize = 0;
    let mut n_valid: usize = 0;
    let mut col_indices: Option<ColIndices> = None;
    let mut features_buf: Vec<String> = Vec::new();

    for line in reader.lines() {
        let line = line.unwrap();
        if col_indices.is_none() {
            col_indices = Some(detect_columns(&line));
            continue;
        }
        let idx = col_indices.as_ref().unwrap();
        n_lines += 1;

        let fields: Vec<&str> = line.split('\t').collect();
        let acc_i = match idx.accession {
            Some(i) => i,
            None => continue,
        };
        let accession = match fields.get(acc_i) {
            Some(s) => s.trim(),
            None => continue,
        };
        if !proteins.contains(accession) {
            continue;
        }
        n_valid += 1;

        parse_features(&fields, idx, &mut features_buf);
        for feat in &features_buf {
            *counts.entry(feat.clone()).or_default() += 1;
        }

        if n_valid % 1_000_000 == 0 {
            let elapsed = t0.elapsed().as_secs_f64();
            let rate = n_lines as f64 / elapsed;
            eprintln!("  Pass 1: {:>12} lines | {:>10} valid | {:>8} unique features | {:.0} lines/s",
                format_num(n_lines), format_num(n_valid), format_num(counts.len()), rate);
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!("  Pass 1 complete: {} lines, {} valid proteins, {} unique features in {:.1}s",
        format_num(n_lines), format_num(n_valid), format_num(counts.len()), elapsed);

    (counts, col_indices.unwrap())
}

fn pass2(tsv_path: &str, proteins: &HashSet<String>, feature_to_id: &HashMap<String, u32>, col_indices: &ColIndices) {
    let t0 = Instant::now();
    let f = File::open(tsv_path).expect("cannot open TSV");
    let reader = BufReader::with_capacity(16 * 1024 * 1024, f);
    let stdout = std::io::stdout();
    let mut out = BufWriter::with_capacity(16 * 1024 * 1024, stdout.lock());
    let mut n_lines: usize = 0;
    let mut n_valid: usize = 0;
    let mut n_kept: usize = 0;
    let mut is_header = true;
    let mut features_buf: Vec<String> = Vec::new();
    let mut ids_buf: Vec<u32> = Vec::new();

    for line in reader.lines() {
        let line = line.unwrap();
        if is_header {
            is_header = false;
            continue;
        }
        n_lines += 1;

        let fields: Vec<&str> = line.split('\t').collect();
        let acc_i = match col_indices.accession {
            Some(i) => i,
            None => continue,
        };
        let accession = match fields.get(acc_i) {
            Some(s) => s.trim(),
            None => continue,
        };
        if !proteins.contains(accession) {
            continue;
        }
        n_valid += 1;

        parse_features(&fields, col_indices, &mut features_buf);
        ids_buf.clear();
        for feat in &features_buf {
            if let Some(&id) = feature_to_id.get(feat) {
                ids_buf.push(id);
            }
        }
        ids_buf.sort_unstable();
        ids_buf.dedup();

        if ids_buf.len() > 1 {
            // Write: protein_id\tid1,id2,id3,...
            write!(out, "{}", accession).unwrap();
            for (j, id) in ids_buf.iter().enumerate() {
                if j == 0 { write!(out, "\t").unwrap(); }
                else { write!(out, ",").unwrap(); }
                write!(out, "{}", id).unwrap();
            }
            writeln!(out).unwrap();
            n_kept += 1;
        }

        if n_valid % 1_000_000 == 0 {
            let elapsed = t0.elapsed().as_secs_f64();
            let rate = n_lines as f64 / elapsed;
            eprintln!("  Pass 2: {:>12} lines | {:>10} valid | {:>10} kept | {:.0} lines/s",
                format_num(n_lines), format_num(n_valid), format_num(n_kept), rate);
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!("  Pass 2 complete: {} lines, {} valid, {} kept (>1 feature) in {:.1}s",
        format_num(n_lines), format_num(n_valid), format_num(n_kept), elapsed);
}

fn format_num(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 { result.push(','); }
        result.push(c);
    }
    result.chars().rev().collect()
}
