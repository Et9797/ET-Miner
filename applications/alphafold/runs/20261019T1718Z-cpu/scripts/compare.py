"""Row-by-row comparison of a CPU itemset table against a reference.

Both sides are loaded as canonical per-K tables (ascending Int64 itemsets,
absolute counts; GPU supports are turned into counts with a residual check),
the reference optionally filtered at a min_count, and fully outer-joined per K
on the itemset. When both sides are directories of per-level files
(`k{K}.parquet`, `frequent_k{K}.parquet`) one level is loaded at a time, so
tables of tens of millions of rows fit in memory. Every differing row is
settled by a direct count over the transaction set; the side that disagrees
with the direct count is wrong.

Usage:
    run_py.sh compare.py CPU_DIR REF_PATH --n N --out REPORT.json [options]

Options:
    --ref-min-count M     keep reference rows with count >= M (e.g. Opus filtered to Base)
    --cpu-min-count M     keep CPU rows with count >= M
    --threshold M         the run's min_count, used to adjudicate presence (default: ref-min-count)
    --data PARQUET        transaction set for direct counts of differing rows
    --head N              direct counts over the first N rows only (G0)
    --translate GPU_MAP CPU_MAP   map reference item ids to CPU ids by feature_name
    --label TEXT          id recorded in the report
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as h  # noqa: E402

h.check_env()


def keyed(part):
    """Add a string key column for the itemset."""
    import polars as pl

    return part.with_columns(key=pl.col("itemset").list.eval(pl.element().cast(pl.Utf8)).list.join(","))


def compare(cpu, ref):
    """Return (per-K summary, differing rows) of two canonical per-K table dicts."""
    import polars as pl

    per_k, diffs = {}, []
    for k in sorted(set(cpu) | set(ref)):
        a = keyed(cpu[k]) if k in cpu else None
        b = keyed(ref[k]) if k in ref else None
        empty = pl.DataFrame(schema={"itemset": pl.List(pl.Int64), "count": pl.Int64, "key": pl.Utf8})
        a = empty if a is None else a
        b = empty if b is None else b
        j = a.join(b.rename({"itemset": "itemset_ref", "count": "count_ref"}), on="key", how="full", coalesce=True)
        cpu_only = j.filter(pl.col("count_ref").is_null())
        ref_only = j.filter(pl.col("count").is_null())
        both = j.filter(pl.col("count").is_not_null() & pl.col("count_ref").is_not_null())
        mism = both.filter(pl.col("count") != pl.col("count_ref"))
        per_k[k] = {"cpu": a.height, "ref": b.height, "both": both.height, "cpu_only": cpu_only.height,
                    "ref_only": ref_only.height, "count_mismatch": mism.height}
        for kind, part in (("cpu_only", cpu_only), ("ref_only", ref_only), ("count_mismatch", mism)):
            for row in part.iter_rows(named=True):
                items = row["itemset"] if row["itemset"] is not None else row["itemset_ref"]
                diffs.append({"k": k, "kind": kind, "itemset": list(items), "cpu": row["count"],
                              "ref": row["count_ref"]})
    return per_k, diffs


def level_files(path):
    """Return {K: file} for a directory of per-level parquet files, else None."""
    path = Path(path)
    if not path.is_dir():
        return None
    out = {}
    for f in path.glob("*.parquet"):
        m = re.search(r"(?:^|_)k(\d+)\.parquet$", f.name)
        if not m:
            return None
        assert int(m.group(1)) not in out, f"two files for K={m.group(1)} under {path}: {out.get(int(m.group(1)))}, {f}"
        out[int(m.group(1))] = f
    return out or None


def compare_levels(cpu_files, ref_files, n, cpu_min, ref_min, lut):
    """Level-at-a-time `compare` with incremental signatures; returns (per_k, diffs, sig_cpu, sig_ref)."""
    per_k, diffs = {}, []
    sig_c, sig_r = h.Signature(), h.Signature()
    for k in sorted(set(cpu_files) | set(ref_files)):
        a = h.load_tables(cpu_files[k], n, cpu_min) if k in cpu_files else {}
        b = h.load_tables(ref_files[k], n, ref_min, translate=lut) if k in ref_files else {}
        assert set(a) <= {k} and set(b) <= {k}, f"level file for K={k} holds other levels"
        pk, d = compare(a, b)
        per_k.update(pk)
        diffs += d
        if k in a:
            sig_c.update(k, a[k])
        if k in b:
            sig_r.update(k, b[k])
    return per_k, diffs, sig_c.result(), sig_r.result()


def adjudicate(diffs, data, head, threshold):
    """Direct-count every differing row and name the side that disagrees with it."""
    counter = h.DirectCounter(h.load_items(data, head), {i for d in diffs for i in d["itemset"]})
    for d in diffs:
        c = counter.count(d["itemset"])
        d["direct"] = c
        present = threshold is None or c >= threshold
        cpu_ok = (d["cpu"] == c) if d["cpu"] is not None else not present
        ref_ok = (d["ref"] == c) if d["ref"] is not None else not present
        d["wrong"] = [side for side, ok in (("cpu", cpu_ok), ("ref", ref_ok)) if not ok]
    return diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cpu")
    ap.add_argument("ref")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-min-count", type=int)
    ap.add_argument("--cpu-min-count", type=int)
    ap.add_argument("--threshold", type=int)
    ap.add_argument("--data")
    ap.add_argument("--head", type=int)
    ap.add_argument("--translate", nargs=2, metavar=("GPU_MAP", "CPU_MAP"))
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    import polars as pl

    lut = None
    if a.translate:
        g, c = (pl.read_parquet(p).select("item_id", "feature_name") for p in a.translate)
        m = g.join(c, on="feature_name", how="full", suffix="_cpu")
        assert m["item_id"].null_count() == 0 and m["item_id_cpu"].null_count() == 0, "feature sets differ"
        lut = dict(zip(m["item_id"].to_list(), m["item_id_cpu"].to_list(), strict=True))
    cf, rf = level_files(a.cpu), level_files(a.ref)
    if cf and rf:
        per_k, diffs, sig_cpu, sig_ref = compare_levels(cf, rf, a.n, a.cpu_min_count, a.ref_min_count, lut)
    else:
        cpu = h.load_tables(a.cpu, a.n, a.cpu_min_count)
        ref = h.load_tables(a.ref, a.n, a.ref_min_count, translate=lut)
        per_k, diffs = compare(cpu, ref)
        sig_cpu, sig_ref = h.signature(cpu), h.signature(ref)
    assert sig_cpu["itemsets"] and sig_ref["itemsets"], "an empty side would pass trivially"
    threshold = a.threshold if a.threshold is not None else a.ref_min_count
    if diffs:
        assert a.data, "differing rows need --data for adjudication"
        diffs = adjudicate(diffs, a.data, a.head, threshold)
    identical = not diffs
    assert identical == (sig_cpu["sha256"] == sig_ref["sha256"]), "join and signature disagree"
    report = {
        "label": a.label, "cpu": str(Path(a.cpu).resolve()), "ref": str(Path(a.ref).resolve()), "n_rows": a.n,
        "ref_min_count": a.ref_min_count, "cpu_min_count": a.cpu_min_count, "translated": bool(lut),
        "verdict": "identical" if identical else "mismatch", "per_k": per_k,
        "signature_cpu": sig_cpu, "signature_ref": sig_ref, "n_differing": len(diffs), "differing": diffs,
    }
    h.dump_json(report, a.out)
    print(json.dumps({"label": a.label, "verdict": report["verdict"], "cpu_itemsets": sig_cpu["itemsets"],
                      "ref_itemsets": sig_ref["itemsets"], "n_differing": len(diffs),
                      "wrong": sorted({w for d in diffs for w in d.get("wrong", [])})}))


if __name__ == "__main__":
    main()
