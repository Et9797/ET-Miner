// UniProt .dat flat file → TSV parser (standalone, zero dependencies)
//
// Produces the same 7-column TSV as the UniProt REST stream endpoint.
// Usage: pigz -d -c uniprot_trembl.dat.gz | ./dat_to_tsv > trembl.tsv
//
// Compile: rustc -O dat_to_tsv.rs
// Authors: Et & claudya

use std::io::{self, BufRead, BufWriter, Write};
use std::time::Instant;

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(1 << 20, stdout.lock()); // 1 MB buffer

    writeln!(
        out,
        "Entry\tInterPro\tGene Ontology IDs\tEC number\tKeywords\tTaxonomic lineage\tLength"
    )
    .unwrap();

    let mut accession = String::with_capacity(16);
    let mut interpro: Vec<String> = Vec::with_capacity(16);
    let mut go_ids: Vec<String> = Vec::with_capacity(32);
    let mut ec_numbers: Vec<String> = Vec::with_capacity(4);
    let mut keywords: Vec<String> = Vec::with_capacity(16);
    let mut lineage: Vec<String> = Vec::with_capacity(16);
    let mut length: u64 = 0;
    let mut count: u64 = 0;
    let t0 = Instant::now();

    let mut line_buf = String::with_capacity(512);
    let mut reader = stdin.lock();

    loop {
        line_buf.clear();
        match reader.read_line(&mut line_buf) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(_) => continue,
        }

        let line = line_buf.trim_end_matches('\n').trim_end_matches('\r');

        if line.len() < 2 {
            if line.starts_with("//") {
                goto_emit(
                    &mut out,
                    &accession,
                    &interpro,
                    &go_ids,
                    &ec_numbers,
                    &keywords,
                    &lineage,
                    length,
                    &mut count,
                    t0,
                );
                accession.clear();
                interpro.clear();
                go_ids.clear();
                ec_numbers.clear();
                keywords.clear();
                lineage.clear();
                length = 0;
            }
            continue;
        }

        // Fast tag dispatch on first 2 bytes
        let tag = &line[..2];
        match tag {
            "AC" if line.starts_with("AC   ") && accession.is_empty() => {
                if let Some(acc) = line[5..].trim().trim_end_matches(';').split(';').next() {
                    let acc = acc.trim();
                    if !acc.is_empty() {
                        accession.push_str(acc);
                    }
                }
            }
            "DE" if line.contains("EC=") => {
                let mut rest = &line[..];
                while let Some(pos) = rest.find("EC=") {
                    rest = &rest[pos + 3..];
                    let ec: String = rest
                        .chars()
                        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                        .collect();
                    if !ec.is_empty() {
                        ec_numbers.push(ec);
                    }
                }
            }
            "KW" if line.starts_with("KW   ") => {
                for kw in line[5..].trim().trim_end_matches('.').split(';') {
                    let kw = kw.trim();
                    if !kw.is_empty() {
                        keywords.push(kw.to_string());
                    }
                }
            }
            "OC" if line.starts_with("OC   ") => {
                for oc in line[5..].trim().trim_end_matches('.').split(';') {
                    let oc = oc.trim();
                    if !oc.is_empty() {
                        lineage.push(oc.to_string());
                    }
                }
            }
            "DR" if line.starts_with("DR   ") => {
                let rest = line[5..].trim().trim_end_matches('.');
                if rest.starts_with("InterPro;") {
                    if let Some(ipr) = rest.split(';').nth(1) {
                        let ipr = ipr.trim();
                        if !ipr.is_empty() {
                            interpro.push(ipr.to_string());
                        }
                    }
                } else if rest.starts_with("GO;") {
                    if let Some(go) = rest.split(';').nth(1) {
                        let go = go.trim();
                        if !go.is_empty() {
                            go_ids.push(go.to_string());
                        }
                    }
                }
            }
            "SQ" if line.starts_with("SQ   ") => {
                // SQ   SEQUENCE   256 AA;  28951 MW;  ...
                let mut parts = line[5..].trim().split_whitespace();
                parts.next(); // skip "SEQUENCE"
                if let Some(len_str) = parts.next() {
                    if let Ok(l) = len_str.parse::<u64>() {
                        length = l;
                    }
                }
            }
            "//" => {
                goto_emit(
                    &mut out,
                    &accession,
                    &interpro,
                    &go_ids,
                    &ec_numbers,
                    &keywords,
                    &lineage,
                    length,
                    &mut count,
                    t0,
                );
                accession.clear();
                interpro.clear();
                go_ids.clear();
                ec_numbers.clear();
                keywords.clear();
                lineage.clear();
                length = 0;
            }
            _ => {}
        }
    }

    // Flush any remaining record
    if !accession.is_empty() {
        goto_emit(
            &mut out,
            &accession,
            &interpro,
            &go_ids,
            &ec_numbers,
            &keywords,
            &lineage,
            length,
            &mut count,
            t0,
        );
    }

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "  Done: {} records in {:.0}s ({:.0} rec/s)",
        count,
        elapsed,
        count as f64 / elapsed
    );
}

#[inline]
fn goto_emit(
    out: &mut BufWriter<io::StdoutLock>,
    accession: &str,
    interpro: &[String],
    go_ids: &[String],
    ec_numbers: &[String],
    keywords: &[String],
    lineage: &[String],
    length: u64,
    count: &mut u64,
    t0: Instant,
) {
    if accession.is_empty() {
        return;
    }

    // Entry
    write!(out, "{}\t", accession).unwrap();

    // InterPro: "IPR000157;IPR035897;" (semicolons, no spaces, trailing ;)
    for ipr in interpro {
        write!(out, "{};", ipr).unwrap();
    }
    write!(out, "\t").unwrap();

    // GO IDs: "GO:0003953; GO:0007165" ("; " separated)
    let mut first = true;
    for go in go_ids {
        if !first {
            write!(out, "; ").unwrap();
        }
        write!(out, "{}", go).unwrap();
        first = false;
    }
    write!(out, "\t").unwrap();

    // EC: "3.2.2.-; 3.2.2.6" ("; " separated)
    first = true;
    for ec in ec_numbers {
        if !first {
            write!(out, "; ").unwrap();
        }
        write!(out, "{}", ec).unwrap();
        first = false;
    }
    write!(out, "\t").unwrap();

    // Keywords: "3D-structure;Coiled coil" (";" separated, no spaces)
    first = true;
    for kw in keywords {
        if !first {
            write!(out, ";").unwrap();
        }
        write!(out, "{}", kw).unwrap();
        first = false;
    }
    write!(out, "\t").unwrap();

    // Lineage: "; " separated
    first = true;
    for oc in lineage {
        if !first {
            write!(out, "; ").unwrap();
        }
        write!(out, "{}", oc).unwrap();
        first = false;
    }
    write!(out, "\t").unwrap();

    // Length
    writeln!(out, "{}", length).unwrap();

    *count += 1;
    if *count % 5_000_000 == 0 {
        let elapsed = t0.elapsed().as_secs_f64();
        eprintln!(
            "  {:>12} records  ({:>6.0}s, {:>8.0} rec/s)",
            count,
            elapsed,
            *count as f64 / elapsed
        );
    }
}
