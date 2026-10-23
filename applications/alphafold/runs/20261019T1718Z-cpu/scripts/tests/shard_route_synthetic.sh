#!/bin/bash
# End-to-end check of the shard route with the real af-extract binary on synthetic data.
#
# Builds a 60,000-record DAT and pLDDT CSV, runs af-extract once directly (top 30/30)
# and through split_csv_shards.sh + filter_dat_by_accessions.sh + af_extract_shard.sh
# (4 shards, full vocabulary) + merge_shards.py (top 30/30), and requires
# verify_shard_route.py to pass; then plants one changed row and requires it to fail.
#
# Usage: shard_route_synthetic.sh WORK_DIR
set -euo pipefail
W=$1; S=$(cd "$(dirname "$0")/.." && pwd)
AFX=$(cd "$S/../../../applications/alphafold/af-extract" && pwd)/target/release/af-extract
rm -rf "$W"; mkdir -p "$W/direct"
python3 - "$W" <<'PY'
import gzip, random, sys
d = sys.argv[1]; rnd = random.Random(11)
pf = [f"PF{i:05d}" for i in range(400)]; go = [f"GO:{i:07d}" for i in range(400)]
w = [1.0 / (i + 1) ** 0.8 for i in range(400)]
with gzip.open(f"{d}/all.dat.gz", "wt") as f, open(f"{d}/meta.csv", "w") as c:
    c.write("uniprotAccession,globalMetricValue\n")
    for i in range(60000):
        acc = f"A{i:07d}"
        f.write(f"ID   {acc}_X   Unreviewed;  100 AA.\nAC   {acc}; B{i:07d};\n")
        for p in set(rnd.choices(pf, w, k=rnd.randint(0, 3))):
            f.write(f"DR   Pfam; {p}; Name; 1.\n")
        for g in set(rnd.choices(go, w, k=rnd.randint(0, 4))):
            f.write(f"DR   GO; {g}; F:thing; IEA:InterPro.\n")
        f.write("//\n")
        c.write(f"{acc},{round(rnd.uniform(50, 99), 2)}\n")
PY
"$S/af_extract_shard.sh" "$AFX" "$W/all.dat.gz" "$W/meta.csv" "$W/direct" 30 30 > /dev/null
"$S/split_csv_shards.sh" "$W/meta.csv" "$W" 4 > /dev/null
for s in 0 1 2 3; do
  "$S/filter_dat_by_accessions.sh" "$W/all.dat.gz" "$W/s$s/plddt.csv" "$W/s$s/shard.dat.gz" > "$W/s$s/filter.log" 2>&1
  "$S/af_extract_shard.sh" "$AFX" "$W/s$s/shard.dat.gz" "$W/s$s/plddt.csv" "$W/s$s" 100000 100000 > /dev/null
done
"$S/run_py.sh" merge_shards.py "$W/merged" "$W/s0" "$W/s1" "$W/s2" "$W/s3" --top-pfam 30 --top-go 30 > /dev/null
"$S/run_py.sh" verify_shard_route.py "$W/direct" "$W/merged" "$W/verify.json" > /dev/null && echo "shard route: pass"
"$S/run_py.sh" -c "
import polars as pl
p = '$W/merged/transactions_214m_base.parquet'; t = pl.read_parquet(p)
t = t.with_columns(pl.when(pl.int_range(pl.len()) == 7).then(pl.lit([1], dtype=pl.List(pl.Int64))).otherwise(pl.col('items')).alias('items'))
t.write_parquet(p)"
if "$S/run_py.sh" verify_shard_route.py "$W/direct" "$W/merged" "$W/verify_planted.json" > /dev/null; then
  echo "planted difference NOT caught"; exit 1
fi
echo "planted difference: caught"
