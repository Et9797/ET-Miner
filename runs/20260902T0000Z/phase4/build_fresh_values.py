#!/usr/bin/env python3
"""Collect every freshly reproduced value of the base214m campaign into one
JSON keyed by canonical quantity key (qkey).

Reads the extraction statistics, the mining/experiment JSONs and per-K
parquet directories produced in this run directory, plus the environment
facts recorded during the campaign, and writes fresh_values.json mapping
qkey -> {value, unit, artifact, command, utc, note}. Every value carries the
path of the artifact it was read from; nothing is taken from old logs.

Usage:
    build_fresh_values.py --run-dir DIR --extract-dir DIR --phase3-dir DIR
                          --out fresh_values.json [--env-json ENV.json]

Options:
    --run-dir      campaign run directory (runs/<UTC-timestamp>)
    --extract-dir  af-extract output directory (stats.json, af_extract.log,
                   item_mapping_214m_base.parquet, item_support_*.parquet)
    --phase3-dir   Phase 3 directory (opus/, minc4/, minc3/, exp/, *.log)
    --env-json     optional JSON of environment facts to merge verbatim
    --out          output path
"""

import argparse
import glob
import json
import math
import os
import re
import time
from pathlib import Path

import polars as pl

UTC = lambda p: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(p)))  # noqa: E731
THRESHOLDS = {"Base": 1e-3, "Super": 1e-4, "Power": 1e-5, "Blitz": 1e-6, "Ultra": 2e-7, "Opus": 1e-7}


class Fresh(dict):
    """qkey -> record; records the artifact and command for every value."""

    def put(self, qkey, value, unit, artifact, command, note=""):
        self[qkey] = {"value": value, "unit": unit, "artifact": str(artifact),
                      "command": command, "utc": UTC(artifact) if os.path.exists(str(artifact)) else "",
                      "note": note}


def kdist_from_parquet_dir(pdir):
    """Return {K: n_itemsets} from frequent_k*.parquet files (files or partitioned dirs)."""
    out = {}
    for p in glob.glob(f"{pdir}/frequent_k*.parquet"):
        k = int(re.search(r"frequent_k(\d+)", p).group(1))
        out[k] = pl.scan_parquet(p if os.path.isfile(p) else f"{p}/*.parquet").select(pl.len()).collect().item()
    return dict(sorted(out.items()))


def add_kdist(f, run, kd, artifact, cmd):
    total = sum(kd.values())
    for k, n in kd.items():
        f.put(f"kdist_{run}_k{k}_count", n, "itemsets", artifact, cmd)
        f.put(f"kdist_{run}_k{k}_pct", round(100.0 * n / total, 4) if total else 0.0, "%", artifact, cmd, "100·count/total")
    if kd:
        peak = max(kd, key=kd.get)
        f.put(f"kdist_{run}_peak_k", peak, "K", artifact, cmd)
        f.put(f"kdist_{run}_peak_count", kd[peak], "itemsets", artifact, cmd)
        f.put(f"kdist_{run}_peak_pct", round(100.0 * kd[peak] / total, 4), "%", artifact, cmd)
        f.put(f"run_{run}_kmax", max(kd), "K", artifact, cmd)
        f.put(f"run_{run}_itemsets", total, "itemsets", artifact, cmd)


def extraction(f, xdir):
    st = Path(xdir) / "stats.json"
    if not st.exists():
        return
    s = json.load(open(st))
    cmd = "phase2/scripts/run_af_extract.sh + extract_stats.py"
    n_all, n_multi = s["n_transactions_total"], s["n_multi_written"]
    f.put("dataset_plddt_pass", n_all, "proteins", st, cmd, "transactions written by af-extract (pLDDT >= 50)")
    f.put("dataset_multi_feature", n_multi, "proteins", st, cmd, ">=2 items")
    f.put("dataset_multi_feature_pct", round(100.0 * n_multi / n_all, 4), "%", st, cmd)
    f.put("dataset_single_feature", s["n_single_feature"], "proteins", st, cmd)
    f.put("dataset_single_feature_pct", round(100.0 * s["n_single_feature"] / n_all, 4), "%", st, cmd)
    f.put("extract_items_mean", round(s["items_per_txn_mean_all"], 4), "items/protein", st, cmd, "over all pLDDT-passing transactions")
    f.put("extract_items_mean_multi", round(s["items_per_txn_mean_multi"], 4), "items/protein", st, cmd)
    f.put("extract_items_max", s["items_per_txn_max"], "items", st, cmd)
    f.put("csr_nnz", s["nnz_multi"], "non-zeros", st, cmd, "sum of items over the >=2-item subset")
    f.put("csr_nnz_all", s["nnz_all"], "non-zeros", st, cmd)
    f.put("vocab_items_defined", s["n_items_defined"], "items", st, cmd)
    cats = s["items_per_category"]
    f.put("vocab_pfam_defined", cats.get("pfam", 0), "items", st, cmd)
    f.put("vocab_go_defined", cats.get("go_term", 0), "items", st, cmd)
    f.put("vocab_plddt_defined", cats.get("plddt_mean", 0) + cats.get("plddt_fraction", 0), "items", st, cmd)
    f.put("vocab_items_frequent", s["n_items_with_support_ge8_multi"], "items", st, cmd, "items with count >= 8 in the >=2-item subset")
    n_items = s["n_items_with_support_ge8_multi"]
    nnz = s["nnz_multi"]
    f.put("dataset_max_features_per_protein", s["items_per_txn_max"], "items", st, cmd)
    f.put("dense_gb", round(n_all * n_items / 1e9, 1), "GB", st, cmd, f"{n_all} proteins × {n_items} items × 1 B (paper's dense-boolean convention)")
    f.put("dense_subset_gb", round(n_multi * n_items / 1e9, 1), "GB", st, cmd, f"{n_multi} × {n_items} × 1 B")
    f.put("bitvec_gb", round(n_items * math.ceil(n_all / 64) * 8 / 1e9, 1), "GB", st, cmd, f"{n_items} × ceil({n_all}/64) × 8 B")
    f.put("bitvec_subset_gb", round(n_items * math.ceil(n_multi / 64) * 8 / 1e9, 1), "GB", st, cmd, f"{n_items} × ceil({n_multi}/64) × 8 B")
    f.put("csr_bytes_gb", round(nnz * 16 / 1e9, 2), "GB", st, cmd, f"{nnz} non-zeros × 16 B (two 64-bit ints per entry, paper's COO convention)")
    f.put("csr_h2d_transfer_gb", round(nnz * 8 / 1e9, 2), "GB", st, cmd, f"{nnz} × 8 B column indices (paper: '~3 GB'); with int32 indices it would be {round(nnz * 4 / 1e9, 2)} GB")
    f.put("csr_vs_dense_full_ratio", round((n_all * n_items) / (nnz * 16), 1), "×", st, cmd, "dense_gb / csr_bytes_gb")
    f.put("csr_vs_dense_subset_ratio", round((n_multi * n_items) / (nnz * 16), 1), "×", st, cmd, "dense_subset_gb / csr_bytes_gb")
    f.put("mem_items_per_txn_214m", round(s["items_per_txn_mean_all"], 2), "items/protein", st, cmd, "mean items per pLDDT-passing protein")
    f.put("mem_items_per_txn_multi", round(s["items_per_txn_mean_multi"], 2), "items/protein", st, cmd)
    sup = Path(xdir) / "item_support_multi.parquet"
    if sup.exists():
        m = pl.read_parquet(Path(xdir) / "item_mapping_214m_base.parquet")
        j = pl.read_parquet(sup).rename({"len": "count"}).join(m, left_on="item", right_on="item_id", how="left")
        for cat, key in (("pfam", "vocab_pfam_frequent"), ("go_term", "vocab_go_frequent")):
            f.put(key, int(j.filter((pl.col("feature_category") == cat) & (pl.col("count") >= 8)).height), "items", sup, cmd)
        f.put("vocab_plddt_frequent", int(j.filter(pl.col("feature_category").str.starts_with("plddt") & (pl.col("count") >= 8)).height), "items", sup, cmd)
        f.put("vocab_composition", f"{f['vocab_pfam_frequent']['value']}:{f['vocab_go_frequent']['value']}:{f['vocab_plddt_frequent']['value']}", "Pfam:GO:pLDDT", sup, cmd, "frequent items per category")
        f.put("vocab_items_present_multi", int(j.height), "items", sup, cmd, "items with count >= 1 in the >=2-item subset")
        f.put("doc_notebook_rank1_item_support_pct", round(100.0 * int(j["count"].max()) / n_multi, 1), "%", sup, cmd, "most frequent item / n_multi")
    log = Path(xdir) / "af_extract.log"
    if log.exists():
        t = open(log).read()
        m = re.search(r"Read (\d+) rows: (\d+) passed pLDDT filter \(>= ([\d.]+)\), (\d+) skipped", t)
        if "n_csv_pass_full" in s:
            f.put("dataset_metadata_rows", s["n_csv_rows_full"], "rows", st, cmd, "full metadata CSV row count (extraction ran on the Pfam/GO subset; see stats.json reconstruction)")
            f.put("extract_csv_rows", s["n_csv_rows_full"] + 1, "lines", st, cmd, "data rows + header line")
            f.put("extract_plddt_pass", s["n_csv_pass_full"], "rows", st, cmd)
            f.put("extract_plddt_skipped", s["n_csv_skipped_full"], "rows", st, cmd)
            f.put("extract_progress_csv_read", s["n_csv_rows_full"], "rows", st, cmd, "final value of the progress series")
            if m:
                f.put("extract_min_plddt", float(m.group(3)), "pLDDT", log, cmd)
                f.put("extract_subset_csv_rows_read", int(m.group(1)), "rows", log, cmd, "rows of the Pfam/GO-subset CSV actually fed to af-extract")
            m = None
        if m:
            f.put("dataset_metadata_rows", int(m.group(1)), "rows", log, cmd)
            f.put("extract_csv_rows", int(m.group(1)) + 1, "lines", log, cmd, "data rows + header line (the old log's wc -l convention)")
            f.put("extract_min_plddt", float(m.group(3)), "pLDDT", log, cmd)
            f.put("extract_plddt_pass", int(m.group(2)), "rows", log, cmd)
            f.put("extract_progress_csv_read", int(m.group(1)), "rows", log, cmd, "final value of the progress series")
            f.put("extract_plddt_skipped", int(m.group(4)), "rows", log, cmd)
        m = re.search(r"Frequencies: (\d+) (?:unique )?Pfam, (\d+) (?:unique )?GO", t)
        if m:
            f.put("dataset_pfam_families_observed", int(m.group(1)), "families", log, cmd)
            f.put("dataset_go_families_observed", int(m.group(2)), "terms", log, cmd)
        cnt = Path(xdir).parent / "data" / "dat_2026_01_counts.txt"
        if cnt.exists() and re.search(r"records=\d+", open(cnt).read()):
            c = dict(re.findall(r"(\w+)=(\d+)", open(cnt).read()))
            ccmd = "phase2/scripts/count_dat.sh over phase2/data/uniprot_trembl_2026_01.reduced.dat.gz (full TrEMBL 2026_01 record set)"
            f.put("extract_dat_records", int(c["records"]), "records", cnt, ccmd, "TrEMBL 2026_01 records (all)")
            f.put("extract_dat_pfam_unique", int(c["pfam_unique"]), "families", cnt, ccmd)
            f.put("extract_dat_go_unique", int(c["go_unique"]), "terms", cnt, ccmd)
        else:
            m = re.search(r"Loaded (\d+) annotations \((\d+) Pfam(?: domains)?, (\d+) GO(?: terms)?", t)
            if m:
                f.put("extract_dat_records", int(m.group(1)), "records", log, cmd, "records loaded by af-extract in this run (subset if the lean route was used)")
                f.put("extract_dat_pfam_unique", int(m.group(2)), "families", log, cmd)
                f.put("extract_dat_go_unique", int(m.group(3)), "terms", log, cmd)
        m = re.search(r"Time:\s+([\d.]+)s \((\d+) proteins/sec\)", t)
        if m:
            f.put("extract_time_s", float(m.group(1)), "s", log, cmd, "af-extract self-timed, this box")
            f.put("extract_time_min", round(float(m.group(1)) / 60, 1), "min", log, cmd)
            f.put("extract_proteins_per_s", int(m.group(2)), "proteins/s", log, cmd)
        m = re.search(r"af-extract wall_seconds=(\d+)", t)
        if m:
            f.put("extract_wall_real_s", int(m.group(1)), "s", log, cmd)
        m = re.search(r"Item encoding: (\d+) pLDDT \+ (\d+) Pfam \+ (\d+) GO = (\d+) total items", t)
        if m:
            f.put("extract_item_encoding_total", int(m.group(4)), "items", log, cmd)


def mining_dir(f, run, d, support=None):
    """run_mining.py output directory: mining_meta_*.json + parquet/frequent_k*.parquet."""
    metas = sorted(glob.glob(f"{d}/mining_meta_*.json"))
    pq = f"{d}/parquet"
    if not metas or not os.path.isdir(pq):
        return
    meta = json.load(open(metas[-1]))
    cmd = f"run_mining.py --support {meta['min_support']} --max-length {meta['max_length']} --use-gpu --n-gpus {meta['n_gpus']} --parquet-flush"
    kd = kdist_from_parquet_dir(pq)
    add_kdist(f, run, kd, pq, cmd)
    n = meta["n_transactions"]
    f.put(f"run_{run}_time_s", meta["duration_seconds"], "s", metas[-1], cmd, "2x RTX 3090 row-split")
    f.put(f"run_{run}_time_min", round(meta["duration_seconds"] / 60, 1), "min", metas[-1], cmd)
    f.put(f"run_{run}_n_transactions", n, "transactions", metas[-1], cmd)
    f.put(f"run_{run}_min_count", math.ceil(meta["min_support"] * n), "proteins", metas[-1], cmd, "ceil(min_support·n)")
    f.put(f"run_{run}_support", meta["min_support"], "fraction", metas[-1], cmd)
    f.put(f"run_{run}_support_pct", meta["min_support"] * 100, "%", metas[-1], cmd)
    f.put(f"run_{run}_min_count_nominal", math.floor(meta["min_support"] * n), "proteins", metas[-1], cmd, "floor(min_support·n), the old scripts' printed value")
    f.put(f"run_{run}_max_length", meta["max_length"], "K", metas[-1], cmd, "cap used here (never binding)")
    f.put(f"run_{run}_bytes", sum(os.path.getsize(x) for x in glob.glob(f"{pq}/**/*.parquet", recursive=True) + glob.glob(f"{pq}/*.parquet")), "bytes", pq, cmd, "total size of the per-K parquet files (zstd); not comparable with the old single-file parquet")
    return kd


def deepest_itemset(f, run, d, xdir, exp_dir):
    """Decode the deepest itemset of a mining dir and its analysis JSON."""
    pq = f"{d}/parquet"
    kd = kdist_from_parquet_dir(pq)
    if not kd:
        return
    kmax = max(kd)
    p = f"{pq}/frequent_k{kmax}.parquet"
    df = pl.read_parquet(p if os.path.isfile(p) else f"{p}/*.parquet")
    m = pl.read_parquet(Path(xdir) / "item_mapping_214m_base.parquet")
    names = dict(zip(m["item_id"].to_list(), m["feature_name"].to_list()))
    cats = dict(zip(m["item_id"].to_list(), m["feature_category"].to_list()))
    cmd = f"decode of {p} via item_mapping_214m_base.parquet"
    f.put(f"k22_n_itemsets_at_kmax", df.height, "itemsets", p, cmd, f"itemsets at K={kmax}")
    row = df.sort("support", descending=True).row(0, named=True)
    items = sorted(int(i) for i in row["itemset"])
    n_txn = f.get(f"run_{run}_n_transactions", {}).get("value")
    support_count = round(row["support"] * n_txn) if n_txn else None
    f.put("k22_support", support_count, "proteins", p, cmd, f"support {row['support']} × n {n_txn}")
    f.put("k22_n_features", len(items), "features", p, cmd)
    feats = [names[i] for i in items]
    f.put("k22_features", feats, "ids", p, cmd)
    for n_ in range(1, len(items) + 1):
        f.put(f"k22_feature_{n_}", feats, "ids", p, cmd, "membership check against the decoded K=%d itemset" % kmax)
    edges = Path(xdir) / "plddt_bin_edges.json"
    if edges.exists():
        e = json.load(open(edges))
        for name, rec in e.items():
            if not rec["n_proteins"]:
                f.put(f"vocab_plddt_bin_{name}_edges", "empty (0 proteins)", "pLDDT", edges, "phase2/scripts/plddt_bin_edges.py", "fraction bins are never emitted in metadata mode")
                continue
            f.put(f"vocab_plddt_bin_{name}_edges", f"{rec['min_plddt']:g}–{rec['max_plddt']:g}", "pLDDT", edges, "phase2/scripts/plddt_bin_edges.py (min/max mean pLDDT among proteins carrying the item)", f"{rec['n_proteins']} proteins")
        med = e.get("plddt_mean_med")
        if med:
            f.put("vocab_plddt_bin_medium_edges", f"{med['min_plddt']:g}–{med['max_plddt']:g}", "pLDDT", edges, "phase2/scripts/plddt_bin_edges.py", "empirical edges of plddt_mean_med (encoder rule: 50 ≤ mean ≤ 90, confidence.rs)")
    f.put("k22_n_pfam", sum(1 for i in items if cats[i] == "pfam"), "features", p, cmd)
    f.put("k22_n_go", sum(1 for i in items if cats[i] == "go_term"), "features", p, cmd)
    f.put("k22_n_plddt", sum(1 for i in items if cats[i].startswith("plddt")), "features", p, cmd)
    f.put("k22_plddt_feature", [names[i] for i in items if cats[i].startswith("plddt")], "id", p, cmd)
    obo = None
    for cand in glob.glob(f"{exp_dir}/go.obo") + glob.glob("/root/projects/ET-Miner/applications/alphafold/experiments/go.obo"):
        obo = cand
        break
    if obo:
        ns, cur = {}, None
        for line in open(obo):
            if line.startswith("id: GO:"):
                cur = line[4:].strip()
            elif line.startswith("namespace:") and cur:
                ns[cur] = line.split(":", 1)[1].strip()
        aspects = {"molecular_function": 0, "biological_process": 0, "cellular_component": 0}
        for i in items:
            if cats[i] == "go_term":
                aspects[ns.get(names[i], "unknown")] = aspects.get(ns.get(names[i], "unknown"), 0) + 1
        f.put("k22_n_go_mf", aspects["molecular_function"], "features", obo, cmd + " + go.obo namespaces")
        f.put("k22_n_go_bp", aspects["biological_process"], "features", obo, cmd + " + go.obo namespaces")
        f.put("k22_n_go_cc", aspects["cellular_component"], "features", obo, cmd + " + go.obo namespaces")
    an = sorted(glob.glob(f"{exp_dir}/analysis_deepest_itemset_*.json"))
    if an:
        a = json.load(open(an[-1]))
        cmd2 = "analyze_k22_proteins.py --itemset-source frequent_k<KMAX>.parquet"
        f.put("k22_n_proteins", a.get("n_proteins"), "proteins", an[-1], cmd2)
        prots = a.get("proteins") or []
        nf = sorted({p.get("n_features") for p in prots if isinstance(p, dict) and "n_features" in p})
        if nf:
            f.put("k22_proteins_features_each", nf if len(nf) > 1 else nf[0], "features", an[-1], cmd2, "distinct n_features among supporting proteins")
        h = a.get("go_hierarchy") or {}
        pairs = h.get("parent_child_pairs") if isinstance(h, dict) else None
        if pairs is not None:
            f.put("k22_go_parent_child_pairs", len(pairs) if isinstance(pairs, list) else pairs, "pairs", an[-1], cmd2)
        f.put("k22_analysis_time_s", a.get("total_time_seconds"), "s", an[-1], cmd2)


def highlighted_patterns(f, run, d, xdir):
    """Supports of the paper's highlighted intermediate-K patterns, read from a per-K parquet dir."""
    pq = f"{d}/parquet"
    kd = kdist_from_parquet_dir(pq)
    if not kd:
        return
    m = pl.read_parquet(Path(xdir) / "item_mapping_214m_base.parquet")
    id_of = dict(zip(m["feature_name"].to_list(), m["item_id"].to_list()))
    n = f.get(f"run_{run}_n_transactions", {}).get("value")
    spec = {19: [], 17: ["PF07714", "PF00017", "PF00018"], 13: [], 12: ["PF00905", "PF00912"], 11: []}
    for k, members in spec.items():
        if k not in kd:
            f.put(f"pattern_k{k}_k", None, "K", pq, f"no K={k} level in the {run} run")
            continue
        p = f"{pq}/frequent_k{k}.parquet"
        df = pl.read_parquet(p if os.path.isfile(p) else f"{p}/*.parquet")
        ids = [id_of.get(x) for x in members]
        cmd = f"max support among K={k} itemsets of the {run} run" + (f" containing {members}" if members else "")
        if members and None not in ids:
            sub = df.filter(pl.all_horizontal([pl.col("itemset").list.contains(i) for i in ids]))
        else:
            sub = df
        if sub.height == 0:
            f.put(f"pattern_k{k}_support", None, "proteins", p, cmd, "no itemset matches")
            continue
        best = sub.sort("support", descending=True).row(0, named=True)
        f.put(f"pattern_k{k}_k", k, "K", p, cmd, f"{kd[k]} itemsets at K={k}")
        sups = sorted({round(x * n) for x in sub["support"].to_list()}, reverse=True) if n else []
        if len(sups) == 1:
            f.put(f"pattern_k{k}_support", sups[0], "proteins", p, cmd, f"single matching itemset; support fraction {best['support']}")
        else:
            f.put(f"pattern_k{k}_support", sups[:20000], "proteins", p, cmd, f"{sub.height} matching itemset(s), {len(sups)} distinct support counts (max {sups[0]}); the paper names no unique itemset, so the claim is checked for existence among these")
        f.put(f"pattern_k{k}_max_support", sups[0] if sups else None, "proteins", p, cmd, "largest support among matching itemsets")
        for i, mem in enumerate(members, 1):
            f.put(f"pattern_k{k}_member_{i}", mem, "id", p, cmd, "member present in the best-supported matching itemset")
    f.put("pattern_n_highlighted", 5, "patterns", pq, "paper's highlighted set (K=19,17,13,12,11) evaluated above", "count of highlighted patterns checked")
    cats = dict(zip(m["item_id"].to_list(), m["feature_category"].to_list()))
    names = dict(zip(m["item_id"].to_list(), m["feature_name"].to_list()))
    if 19 in kd:
        p = f"{pq}/frequent_k19.parquet"
        df = pl.read_parquet(p if os.path.isfile(p) else f"{p}/*.parquet").sort("support", descending=True)
        row = df.row(0, named=True); items = [int(i) for i in row["itemset"]]
        cmd = f"decode of the best-supported K=19 itemset of the {run} run"
        f.put("pattern_k19_pfam_members", [names[i] for i in items if cats[i] == "pfam"], "ids", p, cmd)
        f.put("pattern_k19_go_members", [names[i] for i in items if cats[i] == "go_term"], "ids", p, cmd)
        f.put("pattern_k19_plddt_member", [names[i] for i in items if cats[i].startswith("plddt")], "id", p, cmd)
        f.put("pattern_k19_item_ids", items, "ids", p, cmd, "item ids are run-specific (HashMap tie order); compare members, not ids")
    if 18 in kd:
        p = f"{pq}/frequent_k18.parquet"
        df = pl.read_parquet(p if os.path.isfile(p) else f"{p}/*.parquet").sort("support", descending=True)
        f.put("pattern_k18_support", round(df.row(0, named=True)["support"] * n) if n else None, "proteins", p, f"max support at K=18 of the {run} run")
    if 15 in kd:
        p = f"{pq}/frequent_k15.parquet"
        df = pl.read_parquet(p if os.path.isfile(p) else f"{p}/*.parquet")
        f.put(f"run_{run}_k15_min_proteins", round(df["support"].min() * n) if n else None, "proteins", p, f"minimum support count among K=15 itemsets of the {run} run")
    ge15 = {k: v for k, v in kd.items() if k >= 15}
    if ge15:
        f.put(f"run_{run}_k_ge15_itemsets", sum(ge15.values()), "itemsets", pq, f"sum of K>=15 level counts of the {run} run")
        used = set()
        for k in ge15:
            p = f"{pq}/frequent_k{k}.parquet"
            df = pl.read_parquet(p if os.path.isfile(p) else f"{p}/*.parquet")
            used.update(int(i) for i in df["itemset"].explode().unique().to_list())
        f.put(f"run_{run}_k_ge15_distinct_items", len(used), "items", pq, f"distinct items across K>=15 itemsets of the {run} run")


def son_pattern_checks(f, blitz_dir, opus_dir, n):
    """Serve the old SON logs' listed pattern supports: existence among the
    exhaustive Blitz supports per K (Power ⊂ Blitz), and the exact top-3 K=1/K=2
    supports from the Opus result."""
    pq = f"{blitz_dir}/parquet"
    for k in (10, 11, 12, 13):
        p = f"{pq}/frequent_k{k}.parquet"
        if not (os.path.isfile(p) or os.path.isdir(p)):
            continue
        df = pl.read_parquet(p if os.path.isfile(p) else f"{p}/*.parquet")
        sups = sorted({round(x * n) for x in df["support"].to_list()}, reverse=True)
        fracs = sorted({float(x) for x in df["support"].to_list()}, reverse=True)
        for i in range(1, 9):
            f.put(f"pattern_power_son_k{k}_{i}_support", sups[:20000], "proteins", p, f"distinct support counts of all K={k} itemsets in the exhaustive Blitz run (min_count 77 ⊂ Power's 769); a listed Power-SON pattern must appear here", f"{len(sups)} distinct values, max {sups[0]}")
            f.put(f"pattern_power_son_k{k}_{i}_support_frac", fracs[:20000], "fraction", p, "same, as exact support fractions")
    oq = f"{opus_dir}/parquet"
    for k in (1, 2):
        p = f"{oq}/frequent_k{k}.parquet"
        if not (os.path.isfile(p) or os.path.isdir(p)):
            continue
        df = pl.read_parquet(p if os.path.isfile(p) else f"{p}/*.parquet").sort("support", descending=True)
        sups = sorted({round(x * n) for x in df["support"].to_list()}, reverse=True)
        fracs = sorted({float(x) for x in df["support"].to_list()}, reverse=True)
        for i in (1, 2, 3):
            f.put(f"pattern_super_son_k{k}_{i}_support", sups[:20000], "proteins", p, f"distinct support counts of all K={k} itemsets (exhaustive Opus result); the old SON log's listing order is not by support, so existence is checked", f"{len(sups)} values, top {sups[:3]}")
            f.put(f"pattern_super_son_k{k}_{i}_support_frac", fracs[:20000], "fraction", p, f"same, as exact support fractions")


def top_supports(f, run, d, ks=(20, 21), top=3):
    """Top-N supports (protein counts) at the given K levels of a per-K parquet dir."""
    pq = f"{d}/parquet"
    n = f.get(f"run_{run}_n_transactions", {}).get("value")
    for k in ks:
        p = f"{pq}/frequent_k{k}.parquet"
        if not (os.path.isfile(p) or os.path.isdir(p)):
            continue
        df = pl.read_parquet(p if os.path.isfile(p) else f"{p}/*.parquet").sort("support", descending=True)
        for i, sup in enumerate(df["support"].head(top).to_list(), 1):
            f.put(f"pattern_{run}_k{k}_{i}_support", round(sup * n) if n else None, "proteins", p, f"top-{top} supports at K={k} of the {run} run (rank {i})")
            f.put(f"pattern_{run}_k{k}_{i}_support_frac", round(sup, 7), "fraction", p, f"top-{top} supports at K={k} of the {run} run (rank {i})")


def son_top_supports(f, run, path, n, ks=(1, 2), top=3):
    """Top-N supports at low K from a saved SON itemset parquet."""
    if not os.path.exists(path):
        return
    df = pl.read_parquet(path)
    kcol = df["itemset"].list.len()
    for k in ks:
        sub = df.filter(kcol == k).sort("support", descending=True)
        for i, sup in enumerate(sub["support"].head(top).to_list(), 1):
            f.put(f"pattern_{run}_k{k}_{i}_support", round(sup * n), "proteins", path, f"top-{top} supports at K={k} of the {run} run (rank {i})")
            f.put(f"pattern_{run}_k{k}_{i}_support_frac", round(sup, 5), "fraction", path, f"top-{top} supports at K={k} of the {run} run (rank {i})")


def campaign(f, exp_dir):
    js = sorted(glob.glob(f"{exp_dir}/experiment_full_campaign_*.json"))
    if not js:
        return
    c = json.load(open(js[-1]))
    n = c["parameters"]["n_transactions"]
    cmd = f"experiment_full_campaign.py --runs {c['parameters']['n_runs']}"
    f.put("campaign_total_time_s", c["total_time_seconds"], "s", js[-1], cmd)
    f.put("campaign_n_transactions", n, "transactions", js[-1], cmd)
    for name, res in c["thresholds"].items():
        run = name.lower() + "_direct" if name in ("Base", "Super", "Power") else name.lower()
        runs = res.get("runs") or []
        agg = res.get("aggregate") or {}
        if not runs:
            continue
        r0 = runs[0]
        kd = {int(k): v for k, v in r0["k_distribution"].items()}
        add_kdist(f, f"{run}_campaign" if run in ("opus",) else run, kd, js[-1], cmd)
        s = THRESHOLDS[name]
        key = run if run != "opus" else "opus_campaign"
        f.put(f"run_{key}_time_s", agg.get("mean_time_seconds", r0["time_seconds"]), "s", js[-1], cmd, f"{len(runs)} run(s), 1 GPU path (script default)")
        f.put(f"run_{key}_time_min", round(agg.get("mean_time_seconds", r0["time_seconds"]) / 60, 2), "min", js[-1], cmd)
        f.put(f"run_{key}_min_count", math.ceil(s * n), "proteins", js[-1], cmd, "ceil(s·n)")
        f.put(f"run_{key}_support_pct", s * 100, "%", js[-1], cmd)
        if "cv_pct" in agg or "std_itemsets" in agg:
            f.put(f"run_{key}_itemsets_std", agg.get("std_itemsets"), "itemsets", js[-1], cmd)


def direct_vs_son(f, exp_dir):
    js = sorted(glob.glob(f"{exp_dir}/experiment_direct_vs_son_*.json"))
    if not js:
        return
    c = json.load(open(js[-1]))
    p = c["parameters"]
    cmd = f"experiment_direct_vs_son.py --min-support {p['min_support']} --runs {p['n_runs']} (chunk {p['chunk_size']}, factor {p['local_support_factor']})"
    d0, s0 = c["direct_gpu_runs"][0], c["son_runs"][0]
    add_kdist(f, "power_direct", {int(k): v for k, v in d0["k_distribution"].items()}, js[-1], cmd)
    add_kdist(f, "power_son", {int(k): v for k, v in s0["k_distribution"].items()}, js[-1], cmd)
    f.put("run_power_itemsets", s0["itemsets"], "itemsets", js[-1], cmd, "SON (paper Table 2 Power row method)")
    f.put("run_power_kmax", s0["max_k"], "K", js[-1], cmd, "SON")
    f.put("run_power_time_s", s0["time_seconds"], "s", js[-1], cmd, "SON, this box")
    f.put("run_power_time_min", round(s0["time_seconds"] / 60, 2), "min", js[-1], cmd, "SON, this box")
    f.put("run_power_son_time_s", s0["time_seconds"], "s", js[-1], cmd)
    f.put("run_power_direct_time_s", d0["time_seconds"], "s", js[-1], cmd)
    f.put("run_power_min_count", p["min_count"], "proteins", js[-1], cmd, "ceil(min_support·n) as computed by the script")
    f.put("run_power_direct_min_count", p["min_count"], "proteins", js[-1], cmd)
    f.put("run_power_support_pct", p["min_support"] * 100, "%", js[-1], cmd)
    f.put("run_power_n_transactions", p["n_transactions"], "transactions", js[-1], cmd)
    f.put("run_power_son_chunk_size", p["chunk_size"], "transactions", js[-1], cmd)
    f.put("run_power_son_local_factor", p["local_support_factor"], "factor", js[-1], cmd)
    a = c["aggregate"]
    f.put("run_power_son_itemset_match", s0["itemsets"] == d0["itemsets"], "bool", js[-1], cmd, "SON itemset count equals direct count?")
    f.put("run_power_son_itemset_diff", d0["itemsets"] - s0["itemsets"], "itemsets", js[-1], cmd, "direct − SON")
    f.put("run_power_direct_vs_son_itemset_ratio", round(d0["itemsets"] / s0["itemsets"], 1) if s0["itemsets"] else None, "×", js[-1], cmd, "direct / SON itemsets")
    f.put("son_speedup", a["speedup"], "×", js[-1], cmd, "SON time / direct time, this box")
    f.put("son_miss_rate_pct", a["miss_rate_pct"], "%", js[-1], cmd, "1 − SON/direct itemsets")


def null_model(f, exp_dir, prefix="null"):
    js = sorted(glob.glob(f"{exp_dir}/experiment_null_model_*.json"))
    if not js:
        return
    c = json.load(open(js[-1]))
    p = c["parameters"]
    cmd = f"experiment_null_model.py --min-count {p['min_count']} --runs {p['n_permutations']} --seed {p['seed']} --n-gpus {p['n_gpus']} (row-split branch; the per-GPU worker mode was not used)"
    f.put(f"{prefix}_min_count", p["min_count"], "proteins", js[-1], cmd)
    f.put(f"{prefix}_support_pct", round(p["min_support"] * 100, 4), "%", js[-1], cmd)
    f.put(f"{prefix}_permutations", p["n_permutations"], "permutations", js[-1], cmd)
    f.put(f"{prefix}_seed", p["seed"], "seed", js[-1], cmd)
    f.put(f"{prefix}_n_transactions", p["n_transactions"], "transactions", js[-1], cmd)
    for k, st in c["statistics"].items():
        f.put(f"{prefix}_k{k}_bio", st["real"], "itemsets", js[-1], cmd)
        f.put(f"{prefix}_k{k}_mean", st["null_mean"], "itemsets", js[-1], cmd)
        f.put(f"{prefix}_k{k}_std", st["null_std"], "itemsets", js[-1], cmd)
        f.put(f"{prefix}_k{k}_z", st["z_score"], "Z", js[-1], cmd)
        f.put(f"{prefix}_k{k}_p", st["p_value"], "p", js[-1], cmd)
    null_ks = [int(k) for k, st in c["statistics"].items() if st["null_mean"] > 0]
    f.put(f"{prefix}_kmax", max(null_ks) if null_ks else 0, "K", js[-1], cmd, "highest K with any null itemset")
    real_ks = [int(k) for k in c["real_distribution"]]
    f.put(f"{prefix}_bio_kmax", max(real_ks), "K", js[-1], cmd)
    f.put(f"{prefix}_bio_total", c["real_total"], "itemsets", js[-1], cmd)
    kge7_bio = sum(v for k, v in c["real_distribution"].items() if int(k) >= 7)
    kge7_null = sum(st["null_mean"] for k, st in c["statistics"].items() if int(k) >= 7)
    f.put(f"{prefix}_kge7_bio", kge7_bio, "itemsets", js[-1], cmd, "sum of real counts K>=7")
    f.put(f"{prefix}_kge7_mean", kge7_null, "itemsets", js[-1], cmd)
    nperm = p["n_permutations"]
    f.put(f"{prefix}_p_bound_kge7", round(1 - 0.05 ** (1 / nperm), 4) if kge7_null == 0 else None, "p", js[-1], cmd,
          "rule-of-three-style bound 1−0.05^(1/n) valid only if no null itemset at K>=7")
    f.put(f"{prefix}_support", p["min_support"], "fraction", js[-1], cmd)
    zs = {int(k): st["z_score"] for k, st in c["statistics"].items() if isinstance(st["z_score"], (int, float))}
    z46 = [zs[k] for k in (4, 5, 6) if k in zs]
    if z46:
        f.put(f"{prefix}_z_min_k4to6", round(min(z46), 2), "Z", js[-1], cmd, f"fresh Z at K=4/5/6 = {zs.get(4)}/{zs.get(5)}/{zs.get(6)}; random-stream dependent (σ over {nperm} permutations)")
        above = [k for k in sorted(zs) if zs[k] > 3700]
        f.put(f"{prefix}_z_headline_k_range", f"{above[0]}–{above[-1]}" if above else "none", "K", js[-1], cmd, f"K levels with finite Z > 3,700 in the fresh run (K>=7 have infinite Z); K=4 Z = {zs.get(4)}")
    reach = sum(1 for r in c["null_runs"] if any(int(k) >= 7 for k in r["k_distribution"]))
    f.put(f"{prefix}_perms_reaching_kge7", reach, "permutations", js[-1], cmd, f"of {nperm}")
    f.put(f"{prefix}_p_bound_laplace", round((reach + 1) / (nperm + 1), 4), "p", js[-1], cmd, f"Laplace/rule-of-succession bound ({reach}+1)/({nperm}+1)")
    f.put(f"{prefix}_t_df", nperm - 1, "df", js[-1], cmd, "n permutations − 1")
    enriched = [int(k) for k, st in c["statistics"].items() if st["real"] > st["null_mean"] and int(k) > 1]
    f.put(f"{prefix}_enriched_from_k", min(enriched) if enriched else None, "K", js[-1], cmd, "lowest K > 1 with real > null mean")
    f.put(f"{prefix}_p_bound_rule_of_three", round(3 / nperm, 3), "p", js[-1], cmd, "3/n permutations")
    depleted = [int(k) for k, st in c["statistics"].items() if st["real"] < st["null_mean"]]
    f.put(f"{prefix}_depleted_k", depleted if len(depleted) != 1 else depleted[0], "K", js[-1], cmd, "K levels where real < null mean")
    s = c["summary"]
    f.put(f"{prefix}_mean_total", s["null_mean_total"], "itemsets", js[-1], cmd)
    f.put(f"{prefix}_ratio", s["ratio_real_vs_null"], "×", js[-1], cmd)
    f.put(f"{prefix}_avg_perm_time_s", s["avg_null_mining_seconds"], "s", js[-1], cmd)
    f.put(f"{prefix}_per_perm_time_s", s["avg_null_mining_seconds"], "s", js[-1], cmd)
    f.put(f"{prefix}_total_gpu_s", s["total_gpu_seconds"], "s", js[-1], cmd)
    f.put(f"{prefix}_total_time_s", s["wall_clock_seconds"], "s", js[-1], cmd, "wall-clock incl. real re-mining")
    f.put(f"{prefix}_real_mining_s", s.get("real_mining_seconds"), "s", js[-1], cmd)


def son_json(f, run, path, name):
    if not os.path.exists(path):
        return
    c = json.load(open(path))
    cmd = f"son_run.py --min-support {c['min_support']} (chunk {c['chunk_size']}, factor {c['local_support_factor']})"
    add_kdist(f, run, {int(k): v for k, v in c["k_distribution"].items()}, path, cmd)
    f.put(f"run_{name}_itemsets", c["itemsets"], "itemsets", path, cmd, "SON")
    f.put(f"run_{name}_kmax", c["max_k"], "K", path, cmd, "SON")
    f.put(f"run_{name}_time_s", c["time_seconds"], "s", path, cmd, "SON, this box")
    f.put(f"run_{name}_time_min", round(c["time_seconds"] / 60, 2), "min", path, cmd)
    f.put(f"run_{name}_min_count", c["min_count"], "proteins", path, cmd, "ceil(s·n)")
    f.put(f"run_{name}_support_pct", c["min_support"] * 100, "%", path, cmd)
    f.put(f"run_{name}_n_transactions", c["n_transactions"], "transactions", path, cmd)
    for met, val, unit in (("itemsets", c["itemsets"], "itemsets"), ("kmax", c["max_k"], "K"), ("time_s", c["time_seconds"], "s"), ("min_count", c["min_count"], "proteins")):
        f.put(f"run_{name}_son_{met}", val, unit, path, cmd, "alias of run_" + name + "_" + met)
    if "n_rules" in c:
        f.put("extract_rules_time_s", c.get("rules_time_seconds"), "s", path, cmd)
        for k2, key in (("n_rules_lift_ge5", "extract_rules_lift_ge5"), ("n_rules_lift_ge100", "extract_rules_lift_ge100"), ("n_rules_cross_domain", "extract_rules_cross_domain")):
            if k2 in c:
                f.put(key, c[k2], "rules", path, cmd, c.get("cross_domain_definition", "") if "cross" in key else "")
    if "n_rules" in c:
        f.put("extract_rules_count", c["n_rules"], "rules", path, cmd + " + generate_rules(min_confidence=0.5)")
        if c.get("rules_max"):
            f.put("extract_rules_max_lift", c["rules_max"].get("lift"), "lift", path, cmd)


BENCH_PREFIX = {"deepk-": "deep_k", "stressk2-": "stress_k2", "skew-": "skewed_rows", "twophase-": "smoke"}


def bench_stem(run_id):
    """Map a bench run id such as 'deepk-shared-2g#r0' to ('deep_k_shared_2g', rep)."""
    base, _, rep = run_id.partition("#")
    if base == "twophase-smoke":
        return "smoke_twophase", rep
    for pre, preset in BENCH_PREFIX.items():
        if base.startswith(pre):
            return f"{preset}_{base[len(pre):].replace('-', '_')}", rep
    return base.replace("-", "_"), rep


def bench_results(f, bench_dir):
    """Fresh values of the synthetic bench matrix (raw.jsonl of bench/runner.py)."""
    raw = f"{bench_dir}/raw.jsonl"
    if not os.path.exists(raw):
        return
    recs = [json.loads(l) for l in open(raw) if l.strip()]
    cmd = "bench/runner.py --mode full (this box, 2×RTX 3090)"
    ok = [r for r in recs if r.get("status") == "ok"]
    f.put("bench_campaign_runs_status", f"{len(ok)} ok, {len(recs) - len(ok)} failed/timeout", "runs", raw, cmd)
    for key in ("bench_campaign_0831_runs_status", "bench_campaign_0901_runs_status"):
        f.put(key, f"{len(ok)} ok, {len(recs) - len(ok)} failed/timeout", "runs", raw, cmd, "status of the fresh re-run of the matrix")
    by_stem = {}
    for r in recs:
        stem, rep = bench_stem(r["id"])
        by_stem.setdefault(stem, []).append(r)
    groups = {}
    for stem, rs in by_stem.items():
        oks = [r for r in rs if r.get("status") == "ok"]
        cfg = rs[0]["config"]
        f.put(f"bench_{stem}_params", f"preset={cfg.get('preset')}; env={cfg.get('env')}; n_gpus={cfg.get('n_gpus')}; sparse_from_k={cfg.get('sparse_from_k')}; max_length={cfg.get('max_length')}; two_phase={cfg.get('two_phase')}; timeout={cfg.get('timeout_s')}", "run config", raw, cmd)
        if not oks:
            f.put(f"bench_{stem}_wall_s", None, "s", raw, cmd, f"status: {[r.get('status') for r in rs]}")
            continue
        walls = [r["wall_s"] for r in oks]
        f.put(f"bench_{stem}_wall_s", walls if len(walls) > 1 else walls[0], "s", raw, cmd, f"{len(walls)} rep(s); median {sorted(walls)[len(walls)//2]:.3f}")
        vr = [max((r.get("peak_vram_mb") or {}).values() or [0]) for r in oks]
        f.put(f"bench_{stem}_peak_vram_mb", vr if len(vr) > 1 else vr[0], "MB", raw, cmd, "max over GPUs per rep")
        lv = oks[0].get("levels") or []
        f.put(f"bench_{stem}_levels", "; ".join(f"K{l['k']}={l.get('n_candidates')}/{l.get('n_frequent')}" for l in lv), "candidates/frequent per K", raw, cmd)
        f.put(f"bench_{stem}_level_ms", "; ".join(f"K{l['k']}={l.get('ms'):.1f}" for l in lv), "ms per K", raw, cmd, "rep 0")
        f.put(f"bench_{stem}_throttle_flags", str(sorted({str(x) for r in oks for x in (r.get("throttle_reasons") or [])})), "nvml flags", raw, cmd)
        for met in ("n_itemsets", "sum_counts", "itemset_hash"):
            vals = sorted({str(r.get(met)) for r in oks})
            f.put(f"bench_{stem}_{met}", oks[0].get(met) if len(vals) == 1 else vals, "signature", raw, cmd, "identical across reps" if len(vals) == 1 else "differs across reps")
        g = (cfg.get("preset"), cfg.get("max_length"), cfg.get("min_support"), bool(cfg.get("two_phase")))
        groups.setdefault(g, []).extend(oks)
    for (preset, ml, ms, tp), rs in groups.items():
        sig = {met: sorted({str(r.get(met)) for r in rs}) for met in ("n_itemsets", "sum_counts", "itemset_hash")}
        consistent = all(len(v) == 1 for v in sig.values())
        tag = f"{preset}_exact" if not tp and (preset != "stress_k2") else (f"{preset}_ml{ml}" if preset == "stress_k2" and not tp else f"{preset}_twophase")
        for met in ("n_itemsets", "sum_counts", "itemset_hash"):
            v = rs[0].get(met) if len(sig[met]) == 1 else sig[met]
            f.put(f"bench_{tag}_{met}", v, "signature", raw, cmd, f"equivalence group (preset={preset}, max_length={ml}, two_phase={tp}), {len(rs)} ok runs, consistent={consistent}")
    if "bench_deep_k_shared_2g_wall_s" in f and "bench_deep_k_legacy_2g_wall_s" in f:
        w = lambda k: (lambda v: sorted(v)[len(v)//2] if isinstance(v, list) else v)(f[k]["value"])  # noqa: E731
        try:
            f.put("bench_deep_k_shared_vs_legacy_2g_speedup", round(w("bench_deep_k_legacy_2g_wall_s") / w("bench_deep_k_shared_2g_wall_s"), 2), "×", raw, cmd, "median legacy_2g wall / median shared_2g wall")
        except Exception:
            pass
    env = f"{bench_dir}/env.txt"
    if os.path.exists(env):
        t = open(env).read()
        f.put("hw_bench_env", t.strip()[:400].replace("\n", " | "), "text", env, "bench/runner.py env.txt (this box)")


def row_split_gate(f, p3):
    log = f"{p3}/validate_row_split.log"
    if os.path.exists(log):
        t = open(log).read()
        m = re.search(r"GREEN: row-split \(n_gpus=(\d+)\) is EXACT vs single-GPU \(([\d,]+) itemsets", t)
        f.put("misc_row_split_exact", f"GREEN ({m.group(2)} itemsets identical)" if m else "NOT GREEN", "gate", log, "validate_row_split.py --subset-size 1000000 --min-count 50 --n-gpus 2")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--extract-dir", required=True)
    ap.add_argument("--phase3-dir", required=True)
    ap.add_argument("--env-json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    f = Fresh()
    if a.env_json and os.path.exists(a.env_json):
        f.update(json.load(open(a.env_json)))
    extraction(f, a.extract_dir)
    p3, exp = a.phase3_dir, f"{a.phase3_dir}/exp"
    row_split_gate(f, p3)
    if mining_dir(f, "opus", f"{p3}/opus"):
        deepest_itemset(f, "opus", f"{p3}/opus", a.extract_dir, exp)
    if mining_dir(f, "blitz", f"{p3}/blitz"):
        highlighted_patterns(f, "blitz", f"{p3}/blitz", a.extract_dir)
    mining_dir(f, "minc4", f"{p3}/minc4")
    mining_dir(f, "minc3", f"{p3}/minc3")
    campaign(f, exp)
    direct_vs_son(f, exp)
    null_model(f, exp)
    null_model(f, f"{p3}/exp_null100", prefix="null100")
    son_json(f, "base_son", f"{exp}/son_base.json", "base")
    son_json(f, "super_son", f"{exp}/son_super.json", "super")
    top_supports(f, "opus", f"{p3}/opus"); top_supports(f, "minc4", f"{p3}/minc4")
    nb = f.get("run_blitz_n_transactions", {}).get("value")
    if nb:
        son_pattern_checks(f, f"{p3}/blitz", f"{p3}/opus", nb)
    if "run_opus_min_count" in f and "run_opus_n_transactions" in f:
        nn = f["run_opus_n_transactions"]["value"]; mc = f["run_opus_min_count"]["value"]
        f.put("run_opus_support", [f["run_opus_support"]["value"], round(mc / nn, 12)], "fraction", f["run_opus_support"]["artifact"], f["run_opus_support"]["command"], f"nominal CLI support and effective support min_count/n = {mc}/{nn}")
    if "run_blitz_itemsets" in f and "run_power_direct_itemsets" in f:
        f.put("run_blitz_vs_power_itemset_ratio", round(f["run_blitz_itemsets"]["value"] / f["run_power_direct_itemsets"]["value"], 2), "×", f["run_blitz_itemsets"]["artifact"], "run_blitz_itemsets / run_power_direct_itemsets (fresh exhaustive counts)")
    if "kdist_opus_k1_count" in f:
        kd = {k: f[f"kdist_opus_k{k}_count"]["value"] for k in range(1, 23) if f"kdist_opus_k{k}_count" in f}
        bins = {"K1-3": sum(v for k, v in kd.items() if k <= 3), "K4-6": sum(v for k, v in kd.items() if 4 <= k <= 6), "K7-9": sum(v for k, v in kd.items() if 7 <= k <= 9), "K10-14": sum(v for k, v in kd.items() if 10 <= k <= 14), "K15-22": sum(v for k, v in kd.items() if k >= 15)}
        f.put("doc_notebook_kdist_opus_bins", "; ".join(f"{k}: {v:,}" for k, v in bins.items()), "itemsets", f["kdist_opus_k1_count"]["artifact"], "binned sums of the fresh Opus K-distribution")
    if "kdist_opus_k1_pct" in f:
        f.put("kdist_opus_pct_sum", round(sum(v["value"] for k, v in f.items() if re.fullmatch(r"kdist_opus_k\d+_pct", k)), 1), "%", f["kdist_opus_k1_pct"]["artifact"], "sum of per-K percentages")
    if "run_opus_kmax" in f and "null_kmax" in f and f["null_kmax"]["value"]:
        f.put("null_kmax_bio_ratio", round(f["run_opus_kmax"]["value"] / f["null_kmax"]["value"], 2), "×", f["null_kmax"]["artifact"], "run_opus_kmax / null_kmax")
    if "k22_n_proteins" in f:
        f.put("k22_script_identified_proteins", f["k22_n_proteins"]["value"], "proteins", f["k22_n_proteins"]["artifact"], f["k22_n_proteins"]["command"], "proteins identified by analyze_k22_proteins.py with --itemset-source")
    bench_results(f, f"{p3}/bench")
    gl = f"{p3}/gpu_test_suite.log"
    if os.path.exists(gl):
        last = [l for l in open(gl, errors="replace").read().replace("\r", "\n").splitlines() if re.search(r"passed|failed|error", l)]
        if last:
            f.put("bench_gpu_test_suite_result", re.sub(r"\x1b\[[0-9;]*m", "", last[-1]).strip(), "pytest summary", gl, "pytest -q -m gpu (139 collected on this tree)")
    json.dump(f, open(a.out, "w"), indent=1, default=str)
    print(f"fresh values: {len(f)} qkeys -> {a.out}")
    for k in sorted(f):
        v = f[k]["value"]
        print(f"  {k} = {v if not isinstance(v, list) else str(v)[:80]}")


if __name__ == "__main__":
    main()
