"""Check that the shard route reproduces one unmodified af-extract run.

Compares a direct af-extract output with the merge_shards.py output for the
same CSV: the protein_id sequence, every row's items decoded to
(category, feature_name), the mapping's feature set per category, and ids
(equal except inside groups of equal count, which af-extract orders at random).

Usage:
    run_py.sh verify_shard_route.py DIRECT_DIR MERGED_DIR OUT.json

Options: none. DIRECT_DIR holds transactions.parquet + item_mapping.parquet;
MERGED_DIR holds transactions_214m_base.parquet + item_mapping_214m_base.parquet.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as h  # noqa: E402

h.check_env()


def decoded(tx, mapping):
    """Return rows as sorted lists of 'category:feature_name' strings."""
    import polars as pl

    name = dict(zip(mapping["item_id"].to_list(),
                    (mapping["feature_category"] + ":" + mapping["feature_name"]).to_list(), strict=True))
    return tx.select("protein_id", items=pl.col("items").list.eval(
        pl.element().replace_strict(name, return_dtype=pl.Utf8)).list.sort())


def main():
    import polars as pl

    d, m, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    dmap, mmap = pl.read_parquet(d / "item_mapping.parquet"), pl.read_parquet(m / "item_mapping_214m_base.parquet")
    dtx, mtx = pl.read_parquet(d / "transactions.parquet"), pl.read_parquet(m / "transactions_214m_base.parquet")
    a, b = decoded(dtx, dmap), decoded(mtx, mmap)
    key = ["feature_category", "feature_name"]
    ids = dmap.join(mmap, on=key, how="full", suffix="_merged", coalesce=True)
    moved = ids.filter(pl.col("item_id") != pl.col("item_id_merged"))
    res = {
        "rows_direct": dtx.height, "rows_merged": mtx.height,
        "protein_id_sequence_equal": dtx["protein_id"].equals(mtx["protein_id"]),
        "decoded_rows_equal": a.equals(b),
        "rows_differing": int((a["items"] != b["items"]).sum()) if a.height == b.height else None,
        "mapping_feature_sets_equal": ids["item_id"].null_count() == 0 and ids["item_id_merged"].null_count() == 0,
        "mapping_rows": [dmap.height, mmap.height], "ids_moved": moved.height,
        "moved": moved.to_dicts()[:50],
        "within_row_ascending": bool((mtx["items"].list.eval(pl.element().diff().drop_nulls() > 0).list.all()).all()),
        "dtypes_merged": {c: str(t) for c, t in mtx.schema.items()},
        "dtypes_direct": {c: str(t) for c, t in dtx.schema.items()},
    }
    res["pass"] = (res["protein_id_sequence_equal"] and res["decoded_rows_equal"]
                   and res["mapping_feature_sets_equal"] and res["within_row_ascending"])
    h.dump_json(res, out)
    print(json.dumps({k: v for k, v in res.items() if k != "moved"}))
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
