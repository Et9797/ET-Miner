// remap_tx.rs — Zero-dep Rust TSV remapper
// Reads feature_mapping.tsv + raw transactions TSV, filters by min_count,
// remaps old IDs → new contiguous IDs, outputs remapped TSV.
//
// Usage: ./remap_tx <feature_mapping.tsv> <transactions_raw.tsv> <min_count> [output.tsv]
//        stdout = remapped transactions if no output file
//        stderr = progress
//
// Compile: rustc -O remap_tx.rs -o remap_tx

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <feature_mapping.tsv> <transactions_raw.tsv> <min_count> [output.tsv]", args[0]);
        std::process::exit(1);
    }

    let mapping_path = &args[1];
    let tx_path = &args[2];
    let min_count: u64 = args[3].parse().expect("min_count must be integer");
    let output_path = args.get(4);

    // --- Step 1: Load feature mapping, filter by min_count, build remap ---
    let t0 = Instant::now();
    let mapping_file = File::open(mapping_path).expect("Cannot open feature mapping");
    let reader = BufReader::with_capacity(1 << 20, mapping_file);

    // Collect surviving features sorted by name for deterministic ID assignment
    let mut surviving: Vec<(u32, String, String, u64)> = Vec::new(); // (old_id, name, category, count)
    let mut total = 0u32;

    for (i, line) in reader.lines().enumerate() {
        let line = line.expect("read error");
        if i == 0 { continue; } // skip header
        total += 1;

        let mut parts = line.splitn(4, '\t');
        let old_id: u32 = parts.next().unwrap().parse().unwrap();
        let name = parts.next().unwrap().to_string();
        let category = parts.next().unwrap().to_string();
        let count: u64 = parts.next().unwrap().parse().unwrap();

        if count >= min_count {
            surviving.push((old_id, name, category, count));
        }
    }

    // Sort by feature_name for deterministic new IDs
    surviving.sort_by(|a, b| a.1.cmp(&b.1));

    // Build old_id → new_id remap
    let mut remap: HashMap<u32, u32> = HashMap::with_capacity(surviving.len());
    for (new_id, (old_id, _, _, _)) in surviving.iter().enumerate() {
        remap.insert(*old_id, new_id as u32);
    }

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!("Feature mapping: {} total → {} with count >= {} ({:.1}s)",
              total, surviving.len(), min_count, elapsed);

    // Category breakdown
    let mut cat_counts: HashMap<&str, usize> = HashMap::new();
    for (_, _, cat, _) in &surviving {
        *cat_counts.entry(cat.as_str()).or_insert(0) += 1;
    }
    let mut cats: Vec<_> = cat_counts.iter().collect();
    cats.sort_by_key(|&(k, _)| *k);
    for (cat, n) in cats {
        eprintln!("  {:<15} {:>6} features", cat, n);
    }

    // Write new feature mapping (for downstream parquet conversion)
    if let Some(out) = output_path {
        let map_out = out.replace(".tsv", "_mapping.tsv");
        let f = File::create(&map_out).expect("Cannot create mapping output");
        let mut w = BufWriter::new(f);
        writeln!(w, "item_id\tfeature_name\tfeature_category\tcount").unwrap();
        for (new_id, (_, name, category, count)) in surviving.iter().enumerate() {
            writeln!(w, "{}\t{}\t{}\t{}", new_id, name, category, count).unwrap();
        }
        eprintln!("Wrote mapping: {} → {}", surviving.len(), map_out);
    }

    // --- Step 2: Stream transactions, remap IDs ---
    let t1 = Instant::now();
    let tx_file = File::open(tx_path).expect("Cannot open transactions TSV");
    let reader = BufReader::with_capacity(1 << 22, tx_file); // 4 MB buffer

    let writer: Box<dyn Write> = if let Some(out) = output_path {
        Box::new(BufWriter::with_capacity(1 << 22, File::create(out).expect("Cannot create output")))
    } else {
        Box::new(BufWriter::with_capacity(1 << 22, io::stdout()))
    };
    let mut writer = writer;

    let mut n_lines = 0u64;
    let mut n_kept = 0u64;
    let mut n_dropped = 0u64;
    let mut id_buf: Vec<u32> = Vec::with_capacity(64);

    for line in reader.lines() {
        let line = line.expect("read error");
        n_lines += 1;

        if line.is_empty() { continue; }

        let tab_pos = match line.find('\t') {
            Some(p) => p,
            None => continue,
        };

        let protein_id = &line[..tab_pos];
        let ids_str = &line[tab_pos + 1..];

        // Remap IDs
        id_buf.clear();
        for id_str in ids_str.split(',') {
            if let Ok(old_id) = id_str.parse::<u32>() {
                if let Some(&new_id) = remap.get(&old_id) {
                    id_buf.push(new_id);
                }
            }
        }

        if id_buf.len() > 1 {
            id_buf.sort_unstable();
            // Write: protein_id\tnew_id1,new_id2,...
            write!(writer, "{}\t{}", protein_id, id_buf[0]).unwrap();
            for &id in &id_buf[1..] {
                write!(writer, ",{}", id).unwrap();
            }
            writeln!(writer).unwrap();
            n_kept += 1;
        } else {
            n_dropped += 1;
        }

        if n_lines % 5_000_000 == 0 {
            let elapsed = t1.elapsed().as_secs_f64();
            let rate = n_lines as f64 / elapsed;
            eprintln!("  {:>12} lines | {:>10} kept | {:>10} dropped | {:.0} lines/s",
                      n_lines, n_kept, n_dropped, rate);
        }
    }

    let elapsed = t1.elapsed().as_secs_f64();
    let rate = n_lines as f64 / elapsed;
    eprintln!("\nDone: {} lines → {} kept, {} dropped ({:.1}s, {:.0} lines/s)",
              n_lines, n_kept, n_dropped, elapsed, rate);
    eprintln!("Total features: {}", surviving.len());
}
