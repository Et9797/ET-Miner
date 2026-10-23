#!/bin/bash
# Checks compare.py's level-at-a-time mode against the reports written before it existed.
#
# Runs the harness tests, re-runs three earlier compares (C1 vs the GPU Opus
# tables filtered at 76,891; S2 vs the GPU SON Super table, a single file;
# G0 SON vs efficient-apriori) into RUN_DIR/data/checks/compare_levels/ and
# requires per-level counts, signatures and verdicts equal to the earlier
# reports, then plants a missing row and a +1 count in a copy of G0's K=3
# table and requires both to be caught and adjudicated against the CPU side.
#
# Usage: check_compare_levels.sh RUN_DIR
set -euo pipefail
R=$(cd "$1" && pwd)
cd "$R/scripts"
G=../../20260902T0000Z
D=../data/extract_2026_01/transactions_214m_base_multi.parquet
TR="--translate $G/phase2/extract_2026_01/item_mapping_214m_base.parquet ../data/extract_2026_01/item_mapping_214m_base.parquet"
O=../data/checks/compare_levels
mkdir -p "$O"
echo "== harness tests"
./run_py.sh -m pytest tests -q -p no:cacheprovider
echo "== regression against earlier reports"
./run_py.sh compare.py ../data/mined/C1/parquet $G/phase3/2026_01/opus/parquet --n 76890945 --ref-min-count 76891 \
  --threshold 76891 --data $D $TR --out $O/C1_vs_opus.json --label C1_recheck
./run_py.sh compare.py ../data/mined/S2/parquet $G/phase3/2026_01/exp/son_super_itemsets.parquet --n 76890945 \
  --threshold 7690 --data $D $TR --out $O/S2_vs_son_super.json --label S2_recheck
./run_py.sh compare.py ../data/mined/G0/son/parquet ../data/mined/G0/ea/parquet --n 1000000 --threshold 50 \
  --data $D --head 1000000 --out $O/G0_son_vs_ea.json --label G0_recheck
./run_py.sh - "$O" <<'EOF'
import json, sys
o = sys.argv[1]
for new, old in [("C1_vs_opus", "../data/mined/C1/cmp_gpu_opus.json"), ("S2_vs_son_super", "../data/mined/S2/cmp_gpu_son.json"),
                 ("G0_son_vs_ea", "../data/mined/G0/cmp_son_ea.json")]:
    a, b = json.load(open(f"{o}/{new}.json")), json.load(open(old))
    same = {k: a[k] == b[k] for k in ("per_k", "signature_cpu", "signature_ref", "verdict", "n_differing")}
    print(new, "equal to", old, same)
    assert all(same.values())
EOF
echo "== planted differences (G0 SON K=3: row 7 count +1, row 11 removed)"
rm -rf "$O/mut_g0" && mkdir -p "$O/mut_g0" && cp ../data/mined/G0/son/parquet/*.parquet "$O/mut_g0/"
./run_py.sh - "$O/mut_g0/k3.parquet" <<'EOF'
import sys
import polars as pl
f = sys.argv[1]
d = pl.read_parquet(f)
d = d.with_columns(pl.when(pl.int_range(pl.len()) == 7).then(pl.col("count") + 1).otherwise(pl.col("count")).alias("count"))
d.filter(pl.int_range(pl.len()) != 11).write_parquet(f)
EOF
./run_py.sh compare.py "$O/mut_g0" ../data/mined/G0/ea/parquet --n 1000000 --threshold 50 --data $D --head 1000000 \
  --out $O/mutated_vs_ea.json --label mutated || true
./run_py.sh - "$O/mutated_vs_ea.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
got = sorted((x["kind"], x["wrong"]) for x in d["differing"])
print("mutated:", d["verdict"], got)
assert d["verdict"] == "mismatch" and got == [("count_mismatch", ["cpu"]), ("ref_only", ["cpu"])]
EOF
rm -rf "$O/mut_g0"
echo "result: pass"
