#!/usr/bin/env python3
"""Derive the empirical pLDDT bin edges of the extracted vocabulary.

Joins the transactions parquet (protein_id, items) with the metadata CSV
(uniprotAccession, globalMetricValue) and reports, for every pLDDT item in
the item mapping, the number of proteins carrying it and the minimum and
maximum mean pLDDT among them. Writes plddt_bin_edges.json next to the
transactions.

Usage:
    plddt_bin_edges.py EXTRACT_DIR CSV
"""

import json
import sys

import polars as pl


def main():
    d, csv = sys.argv[1].rstrip("/"), sys.argv[2]
    m = pl.read_parquet(f"{d}/item_mapping_214m_base.parquet")
    plddt = {int(r["item_id"]): r["feature_name"] for r in m.filter(pl.col("feature_category").str.starts_with("plddt")).to_dicts()}
    tx = pl.scan_parquet(f"{d}/transactions_214m_base.parquet").select(["protein_id", "items"])
    meta = pl.scan_csv(csv, schema={"uniprotAccession": pl.Utf8, "globalMetricValue": pl.Float64})
    joined = tx.join(meta, left_on="protein_id", right_on="uniprotAccession", how="inner")
    out = {}
    for item, name in plddt.items():
        r = (joined.filter(pl.col("items").list.contains(item))
             .select([pl.len().alias("n"), pl.col("globalMetricValue").min().alias("min"), pl.col("globalMetricValue").max().alias("max")])
             .collect(engine="streaming").row(0, named=True))
        out[name] = {"item_id": item, "n_proteins": int(r["n"]), "min_plddt": r["min"], "max_plddt": r["max"]}
    json.dump(out, open(f"{d}/plddt_bin_edges.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
