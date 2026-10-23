"""Merge af-extract shard outputs into the full-set transactions and item mapping.

Each shard is an unmodified af-extract `build-from-metadata` run over a
contiguous row range of the joined pLDDT CSV, with a vocabulary large enough
to keep every Pfam/GO feature. A feature's count is the number of proteins
carrying it, so summing shard counts gives the full-set counts; the top-N
Pfam and GO features are then ranked exactly as af-extract ranks them (count
descending; ties, which af-extract orders at random, by feature_name), ids
are assigned in af-extract's layout (pLDDT 0-5, Pfam from 6, GO after Pfam),
and every shard row is re-encoded (out-of-vocabulary features dropped, items
unique and ascending) and written in shard order, which is CSV order.

Usage:
    run_py.sh merge_shards.py OUT_DIR SHARD_DIR [SHARD_DIR ...] [options]

Options:
    --top-pfam N   Pfam vocabulary size (default 500)
    --top-go N     GO vocabulary size (default 500)
Each SHARD_DIR holds transactions.parquet, item_mapping.parquet and af_extract.log.
Writes OUT_DIR/transactions_214m_base.parquet, item_mapping_214m_base.parquet and
merge_shards.json (per-shard checks, counts, ties at and inside the cut).
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as h  # noqa: E402

h.check_env()

PLDDT_ROWS = 6
CATS = ("pfam", "go_term")


def shard_counts(d):
    """Return (mapping, per-feature protein counts) of one shard, with consistency checks."""
    import polars as pl

    m = pl.read_parquet(d / "item_mapping.parquet")
    assert m["item_id"].to_list() == list(range(m.height)), f"{d}: mapping ids not 0..n-1 in order"
    sup = (pl.scan_parquet(d / "transactions.parquet").select(pl.col("items").explode().alias("item_id"))
           .group_by("item_id").len().collect(engine="streaming"))
    c = m.join(sup, on="item_id", how="left").with_columns(pl.col("len").fill_null(0).alias("count"))
    feat = c.filter(pl.col("feature_category").is_in(CATS))
    assert feat["count"].min() >= 1, f"{d}: a mapped feature has no protein (vocabulary was cut)"
    return m, c


def log_facts(d):
    """Return the af-extract log lines that state loaded records, CSV rows and frequencies."""
    txt = (d / "af_extract.log").read_text()
    keep = [ln for ln in txt.splitlines() if re.search(r"Loaded \d+ annotations|Frequencies:|rows|Wrote|wrote", ln)]
    return keep[-12:]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--top-pfam", type=int, default=500)
    ap.add_argument("--top-go", type=int, default=500)
    a = ap.parse_args()

    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shards = [Path(s) for s in a.shards]
    maps, counts = zip(*(shard_counts(d) for d in shards), strict=True)
    fixed = maps[0].head(PLDDT_ROWS)
    for m in maps:
        assert m.head(PLDDT_ROWS).equals(fixed), "pLDDT rows differ between shards"
        assert set(m["feature_category"].unique()) <= set(fixed["feature_category"].unique()) | set(CATS)
    allc = pl.concat([c.filter(pl.col("feature_category").is_in(CATS)).select("feature_category", "feature_name",
                                                                          "count") for c in counts])
    glob = allc.group_by("feature_category", "feature_name").agg(pl.col("count").sum())
    rows, info, base = [], {}, PLDDT_ROWS
    for cat, top in (("pfam", a.top_pfam), ("go_term", a.top_go)):
        r = glob.filter(pl.col("feature_category") == cat).sort(["count", "feature_name"], descending=[True, False])
        cnt = r["count"].to_list()
        kept = r.head(top).with_columns(item_id=pl.int_range(base, base + min(top, r.height), dtype=pl.Int64))
        tied_groups = (kept.group_by("count").agg(pl.col("feature_name").sort(), pl.col("item_id").sort())
                       .filter(pl.col("feature_name").list.len() > 1).sort("count", descending=True))
        info[cat] = {"distinct_features": r.height, "kept": kept.height,
                     "count_at_cut": cnt[top - 1] if r.height >= top else None,
                     "count_after_cut": cnt[top] if r.height > top else None,
                     "tie_at_cut": r.height > top and cnt[top - 1] == cnt[top],
                     "tied_groups_inside": tied_groups.to_dicts()}
        rows.append(kept.select("item_id", "feature_name", "feature_category"))
        base += kept.height
    mapping = pl.concat([fixed.select("item_id", "feature_name", "feature_category")] + rows)
    mapping.write_parquet(out / "item_mapping_214m_base.parquet")
    gid = {(c, f): i for i, f, c in mapping.iter_rows()}
    schema = pa.schema([pa.field("protein_id", pa.string(), nullable=False),
                        pa.field("items", pa.list_(pa.field("item", pa.int64())), nullable=False)])
    writer = pq.ParquetWriter(out / "transactions_214m_base.parquet", schema, compression="snappy")
    per_shard = []
    for d, m in zip(shards, maps, strict=True):
        old = m["item_id"].to_list()
        new = [gid.get((c, f), -1) if c in CATS else i
               for i, f, c in m.select("item_id", "feature_name", "feature_category").iter_rows()]
        lut = dict(zip(old, new, strict=True))
        tx = pl.read_parquet(d / "transactions.parquet")
        mapped = pl.element().replace_strict(lut, return_dtype=pl.Int64)
        enc = tx.select("protein_id", items=pl.col("items").list.eval(mapped.filter(mapped >= 0).unique().sort()))
        assert enc["protein_id"].null_count() == 0 and (enc["items"].list.len() >= 1).all()
        assert enc["items"].list.min().min() >= 0, "an out-of-vocabulary id survived the re-encoding"
        for start in range(0, enc.height, 100_000):
            writer.write_table(enc.slice(start, 100_000).to_arrow().cast(schema), row_group_size=100_000)
        per_shard.append({"shard": str(d), "rows": tx.height, "nnz_shard": int(tx["items"].list.len().sum()),
                          "nnz_merged": int(enc["items"].list.len().sum()),
                          "mapped_features": {c: int((m["feature_category"] == c).sum()) for c in CATS},
                          "log": log_facts(d)})
        del tx, enc
    writer.close()
    summary = {"shards": per_shard, "rows_total": sum(s["rows"] for s in per_shard), "vocabulary": info,
               "mapping_rows": mapping.height, "tie_break": "count descending, then feature_name ascending"}
    h.dump_json(summary, out / "merge_shards.json")
    print(json.dumps({"rows_total": summary["rows_total"], "mapping_rows": mapping.height,
                      **{c: {k: v for k, v in info[c].items() if k != "tied_groups_inside"} for c in CATS},
                      "tied_groups_inside": {c: len(info[c]["tied_groups_inside"]) for c in CATS}}))


if __name__ == "__main__":
    main()
