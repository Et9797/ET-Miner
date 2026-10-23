"""Decode the highlighted intermediate-K itemsets of the Blitz run and report level and pattern maxima.

For each highlighted K-level it prints the number of itemsets, the level maximum support (proteins),
the members of the maximum itemset, and the maximum support among the itemsets that contain a named
feature set (all named Pfam accessions, or one of each named alternative group).

Usage:
    python decode_intermediate_k.py [--json OUT]

Options:
    --json OUT   also write the summary as JSON.
"""
import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
PARQ = ROOT / "runs/20260902T0000Z/phase3/2026_01/blitz/parquet"
MAP = ROOT / "runs/20260902T0000Z/phase2/extract_2026_01/item_mapping_214m_base.parquet"
N_TXN = 76_890_945  # mining subset (RESULTS X-011)

PATTERNS = {
    11: {"AAA+Clp": [["PF00004", "PF07724", "PF17871"], ["PF10431", "PF00574", "PF02861"]]},
    12: {"PF00905+PF00912": [["PF00905"], ["PF00912"]]},
    13: {"PF00270+SOS": [["PF00270"], ["GO:0009432"]], "PF00270+repair+recombination+replication+SOS": [["PF00270"], ["GO:0006281"], ["GO:0006310"], ["GO:0006260"], ["GO:0009432"]]},
    17: {"PF07714+PF00017+PF00018": [["PF07714"], ["PF00017"], ["PF00018"]]},
    19: {},
}


def main():
    names = dict(pl.read_parquet(MAP).select("item_id", "feature_name").iter_rows())
    ids = {v: k for k, v in names.items()}
    out = {}
    for k, pats in PATTERNS.items():
        df = pl.read_parquet(PARQ / f"frequent_k{k}.parquet").with_columns((pl.col("support") * N_TXN).round().cast(pl.Int64).alias("count"))
        top = df.sort("count", descending=True).row(0, named=True)
        rec = {"itemsets": df.height, "level_max": top["count"], "level_max_members": [names[i] for i in top["itemset"]], "patterns": {}}
        for label, groups in pats.items():
            missing = [alt for g in groups for alt in g if alt not in ids]
            expr = pl.lit(True)
            for g in groups:
                gid = [ids[a] for a in g if a in ids]
                expr = expr & pl.col("itemset").list.eval(pl.element().is_in(gid)).list.any()
            sub = df.filter(expr).sort("count", descending=True)
            rec["patterns"][label] = {"matching_itemsets": sub.height, "max": sub.row(0, named=True)["count"] if sub.height else None,
                                      "max_members": [names[i] for i in sub.row(0, named=True)["itemset"]] if sub.height else None,
                                      "not_in_vocabulary": missing}
        out[k] = rec
        print(f"K={k}: {rec['itemsets']} itemsets; level max {rec['level_max']:,} = {rec['level_max_members']}")
        for label, r in rec["patterns"].items():
            print(f"   {label}: {r['matching_itemsets']} itemsets, max {r['max']:,} = {r['max_members']}; not in vocabulary: {r['not_in_vocabulary']}")
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
