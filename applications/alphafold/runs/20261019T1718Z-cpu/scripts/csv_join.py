"""Join the pLDDT metadata CSV to the accessions that carry a Pfam/GO DAT record.

Keeps metadata rows with pLDDT >= 50 whose accession is in the accession
list and writes them as CSV (the input of af-extract's lean extraction).
The polars code is the join step of run_af_extract_lean.sh.

Usage:
    run_py.sh csv_join.py PLDDT_CSV ACCESSIONS.txt OUT.csv

Options: none.
"""

import sys
import time

import polars as pl

csv, acc, out = sys.argv[1:4]
t0 = time.time()
accs = pl.scan_csv(acc, has_header=False, new_columns=["uniprotAccession"],
                   schema={"uniprotAccession": pl.Utf8}).unique()
lf = pl.scan_csv(csv, schema={"uniprotAccession": pl.Utf8, "globalMetricValue": pl.Float64}).filter(
    pl.col("globalMetricValue") >= 50)
joined = lf.join(accs, on="uniprotAccession", how="inner")
joined.sink_csv(out)
n = pl.scan_csv(out, schema={"uniprotAccession": pl.Utf8, "globalMetricValue": pl.Float64}).select(
    pl.len()).collect().item()
print(f"csv join: {n} rows with pLDDT>=50 and a Pfam/GO DAT record -> {out} ({time.time()-t0:.0f}s)")
