"""Support count of one itemset, named by feature, in a CPU per-K itemset table (RESULTS P-026).

Usage:
    run_py.sh triple_support.py ITEMSET_DIR ITEM_MAPPING FEATURE [FEATURE ...] --out OUT.json

Options:
    --out OUT.json   result: features, item ids, count, support and the source table
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as h  # noqa: E402

h.check_env()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("itemsets")
    ap.add_argument("mapping")
    ap.add_argument("features", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import polars as pl

    ids = dict(pl.read_parquet(a.mapping).select("feature_name", "item_id").iter_rows())
    target = sorted(ids[f] for f in a.features)
    src = Path(a.itemsets) / f"k{len(target)}.parquet"
    hit = pl.read_parquet(src).filter(pl.col("itemset") == pl.lit(target, dtype=pl.List(pl.Int64)))
    assert hit.height <= 1
    out = {"features": a.features, "item_ids": target, "source": str(src.resolve()),
           "count": int(hit["count"][0]) if hit.height else None,
           "support": float(hit["support"][0]) if hit.height else None}
    h.dump_json(out, a.out)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
