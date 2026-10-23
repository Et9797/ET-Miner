#!/usr/bin/env python3
"""Derive the ≥2-item transaction subset and dataset statistics from an
af-extract output directory.

Reads transactions_214m_base.parquet and item_mapping_214m_base.parquet,
writes transactions_214m_base_multi.parquet (rows with at least two items,
the set the miners consume) and stats.json with the counts the paper
reports: total transactions, multi-feature transactions and their share,
single-feature count, items per transaction, non-zero entries (CSR nnz) of
the multi-feature subset, and per-category item counts of the vocabulary.

When the extraction was run on the Pfam/GO-bearing subset of the metadata
(memory-reduced route), pass --full-csv: the proteins outside that subset
are single-item (pLDDT-bin only) transactions by construction, so the
full-set totals are reconstructed exactly from the full CSV's pLDDT >= 50
row count.

Usage:
    extract_stats.py EXTRACT_DIR [--full-csv PLDDT_CSV] [--min-plddt 50]
"""

import json
import sys
import time

import polars as pl


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("extract_dir")
    ap.add_argument("--full-csv", default=None)
    ap.add_argument("--min-plddt", type=float, default=50.0)
    a = ap.parse_args()
    d = a.extract_dir.rstrip("/")
    t0 = time.time()
    lf = pl.scan_parquet(f"{d}/transactions_214m_base.parquet")
    n_run = lf.select(pl.len()).collect(engine="streaming").item()
    lens = lf.select(pl.col("items").list.len().alias("n")).collect(engine="streaming")["n"]
    stats = {"n_run_rows": n_run}
    outside = 0
    if a.full_csv:
        c = pl.scan_csv(a.full_csv, schema={"uniprotAccession": pl.Utf8, "globalMetricValue": pl.Float64}).select([
            pl.len().alias("rows"), (pl.col("globalMetricValue") >= a.min_plddt).sum().alias("pass"),
            (pl.col("globalMetricValue") < a.min_plddt).sum().alias("skip")]).collect(engine="streaming").row(0, named=True)
        stats.update({"n_csv_rows_full": int(c["rows"]), "n_csv_pass_full": int(c["pass"]), "n_csv_skipped_full": int(c["skip"]),
                      "full_csv": a.full_csv, "reconstruction": "proteins with pLDDT >= min but without any Pfam/GO DAT record are single-item transactions by construction; totals below include them"})
        outside = int(c["pass"]) - n_run
    n_total = n_run + outside
    stats.update({
        "n_transactions_total": n_total,
        "n_outside_run_single_item": outside,
        "n_multi_feature": int((lens >= 2).sum()),
        "n_single_feature": int((lens == 1).sum()) + outside,
        "n_zero_feature": int((lens == 0).sum()),
        "items_per_txn_mean_all": (float(lens.sum()) + outside) / n_total,
        "items_per_txn_max": int(lens.max()),
        "nnz_all": int(lens.sum()) + outside,
        "nnz_multi": int(lens.filter(lens >= 2).sum()),
        "items_per_txn_mean_multi": float(lens.filter(lens >= 2).mean()),
    })
    stats["pct_multi_feature"] = 100.0 * stats["n_multi_feature"] / n_total
    stats["pct_single_feature"] = 100.0 * stats["n_single_feature"] / n_total
    hist = lens.value_counts().sort("n")
    stats["items_per_txn_histogram"] = {int(r["n"]): int(r["count"]) for r in hist.to_dicts()}
    if outside:
        stats["items_per_txn_histogram"][1] = stats["items_per_txn_histogram"].get(1, 0) + outside
    lf.filter(pl.col("items").list.len() >= 2).sink_parquet(f"{d}/transactions_214m_base_multi.parquet")
    n_multi_written = pl.scan_parquet(f"{d}/transactions_214m_base_multi.parquet").select(pl.len()).collect().item()
    stats["n_multi_written"] = n_multi_written
    m = pl.read_parquet(f"{d}/item_mapping_214m_base.parquet")
    stats["n_items_defined"] = m.height
    stats["items_per_category"] = {r["feature_category"]: int(r["count"]) for r in
                                   m.group_by("feature_category").len().rename({"len": "count"}).to_dicts()}
    # per-item support over the multi-feature subset (K=1 counts)
    sup = (pl.scan_parquet(f"{d}/transactions_214m_base_multi.parquet")
           .select(pl.col("items").explode().alias("item")).group_by("item").len()
           .collect(engine="streaming").sort("item"))
    sup.write_parquet(f"{d}/item_support_multi.parquet")
    stats["n_items_with_support_ge8_multi"] = int((sup["len"] >= 8).sum())
    stats["n_items_present_multi"] = sup.height
    sup_all = (lf.select(pl.col("items").explode().alias("item")).group_by("item").len()
               .collect(engine="streaming").sort("item"))
    sup_all.write_parquet(f"{d}/item_support_all.parquet")
    stats["n_items_present_all"] = sup_all.height
    stats["elapsed_s"] = time.time() - t0
    json.dump(stats, open(f"{d}/stats.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "items_per_txn_histogram"}, indent=2))


if __name__ == "__main__":
    main()
