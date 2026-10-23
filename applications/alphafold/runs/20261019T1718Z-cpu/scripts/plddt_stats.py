"""Counts of the pLDDT metadata CSV, for RESULTS.md rows M-001..M-009.

Rows, distinct accessions, rows with pLDDT >= 50 and < 50, nulls, min and
max, rows > 90 and 50 <= pLDDT <= 90, the header, and the file's sha256.

Usage:
    run_py.sh plddt_stats.py CSV OUT.json

Options: none.
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as h  # noqa: E402

h.check_env()


def main():
    import polars as pl

    csv, out = sys.argv[1], sys.argv[2]
    header = Path(csv).open().readline().rstrip("\n")
    v = pl.col("globalMetricValue")
    lf = pl.scan_csv(csv, schema={"uniprotAccession": pl.Utf8, "globalMetricValue": pl.Float64})
    r = lf.select(
        rows=pl.len(), distinct_accessions=pl.col("uniprotAccession").n_unique(),
        null_accessions=pl.col("uniprotAccession").null_count(), nulls=v.null_count(),
        ge50=(v >= 50).sum(), lt50=(v < 50).sum(), gt90=(v > 90).sum(),
        between50_90=((v >= 50) & (v <= 90)).sum(), min=v.min(), max=v.max(),
    ).collect().row(0, named=True)
    sha = hashlib.sha256()
    with open(csv, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            sha.update(block)
    r.update(header=header, bytes=Path(csv).stat().st_size, sha256=sha.hexdigest(),
             min_2dp=round(r["min"], 2), max_2dp=round(r["max"], 2))
    h.dump_json(r, out)
    print(json.dumps(r))


if __name__ == "__main__":
    main()
