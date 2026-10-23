"""Check the rebuilt data against RESULTS.md rows M-001..M-009, U-007, U-009, X-001..X-016
and the GPU run's committed extraction files.

Expected values are parsed from the RESULTS.md table cells (never retyped);
fresh values are read from this run's artifacts. Timings, sha256 of
compressed files and the CSV's sha256 are reported but not required to match.
Exits 1 on any required mismatch (data drift).

Usage:
    run_py.sh verify_phase1.py RUN_DIR GPU_RUN_DIR RESULTS.md

Options: none. Writes RUN_DIR/data/phase1_verification.json.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as h  # noqa: E402

h.check_env()


def rows_of(results_md):
    """Return {row id: (value cell, what cell)} of the RESULTS.md tables."""
    out = {}
    for line in Path(results_md).read_text().splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 4 and re.fullmatch(r"[A-Z]-\d{3}", cells[1]):
            out[cells[1]] = (cells[2], cells[4])
    return out


def nums(text):
    """Return the numbers in a cell, in order (thousands separators removed)."""
    return [float(x.replace(",", "")) if "." in x else int(x.replace(",", ""))
            for x in re.findall(r"\d[\d,]*(?:\.\d+)?", text)]


def kv(text):
    """Return key=value integers of a log or counts line set."""
    return {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", text)}


def sha256(path):
    """Return the sha256 of a file."""
    s = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            s.update(b)
    return s.hexdigest()


def main():
    import polars as pl

    run, gpu, res_md = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    D, X, G = run / "data", run / "data/extract_2026_01", gpu / "phase2/extract_2026_01"
    R = rows_of(res_md)
    checks = []

    def check(rid, fresh, expected, required=True, note=""):
        checks.append({"id": rid, "fresh": fresh, "expected": expected, "match": fresh == expected,
                       "required": required, "note": note})

    m = json.loads((D / "plddt_stats.json").read_text())
    check("M-001", m["rows"], nums(R["M-001"][0])[0])
    check("M-002", m["distinct_accessions"], nums(R["M-002"][0])[0])
    check("M-003", m["ge50"], nums(R["M-003"][0])[0])
    check("M-004", m["nulls"], nums(R["M-004"][0])[0])
    check("M-005", [m["min_2dp"], m["max_2dp"]], nums(R["M-005"][0])[:2])
    check("M-006", [m["rows"], m["header"]], [nums(R["M-006"][0])[0], "uniprotAccession,globalMetricValue"])
    check("M-006 sha256", m["sha256"][:8], re.search(r"sha256 (\w+)…", R["M-006"][1]).group(1)[:8], required=False,
          note="row order of 16 parallel streams varies; not expected to match")
    check("M-007", [m["ge50"], m["lt50"]], nums(R["M-007"][0])[:2])
    check("M-008", m["gt90"], nums(R["M-008"][0])[0])
    check("M-009", m["between50_90"], nums(R["M-009"][0])[0])

    prog = (run / "logs/stream_trembl_2026_01_progress.log").read_text()
    size = re.search(r"member uniprot_trembl\.dat\.gz size=(\d+)", prog)
    end = kv(re.findall(r"^end .*$", prog, re.M)[-1])
    check("U-007", int(size.group(1)) if size else None, nums(R["U-007"][0])[0])
    check("U-009", [end.get("copied"), end.get("consumed")],
          [nums(R["U-009"][0])[0], nums(R["U-009"][1].split("consumed")[1])[0]])

    cnt = kv((D / "dat_2026_01_counts.txt").read_text())
    x16 = nums(R["X-016"][0])
    lines = nums(R["X-016"][1].split(";")[1])
    check("X-016", [cnt["records"], cnt["pfam_unique"], cnt["go_unique"], cnt["pfam_lines"], cnt["go_lines"]],
          x16[:3] + lines[:2])
    check("X-001", cnt["records"], nums(R["X-001"][0])[0], note="from count_dat.sh (X-016)")
    check("X-002", [cnt["pfam_unique"], cnt["go_unique"]], nums(R["X-002"][0])[:2], note="from count_dat.sh (X-016)")

    f5 = kv((run / "logs/filter_dat_pfamgo.log").read_text())
    n_acc = sum(1 for _ in open(D / "dat_pfamgo_accessions_2026_01.txt"))
    check("X-005", [f5["records_kept"], n_acc, f5["records_total"]],
          [nums(R["X-005"][0])[0], nums(R["X-005"][0])[0], nums(R["X-016"][0])[0]])
    jn = re.search(r"csv join: (\d+) rows", (run / "logs/csv_join.log").read_text())
    check("X-006", int(jn.group(1)), nums(R["X-006"][0])[0])
    f7 = kv((run / "logs/filter_dat_by_accessions.log").read_text())
    check("X-007", f7["records_kept"], nums(R["X-007"][0])[0])

    ms = json.loads((X / "merge_shards.json").read_text())
    check("X-008", [ms["vocabulary"]["pfam"]["distinct_features"], ms["vocabulary"]["go_term"]["distinct_features"],
                    ms["mapping_rows"]], nums(R["X-008"][0])[:2] + [nums(R["X-008"][1].split("=")[-1])[0]])
    check("X-009", ms["rows_total"], nums(R["X-009"][0])[0])
    shard_loaded = []
    for s in ms["shards"]:
        log = " ".join(s["log"])
        loaded = re.search(r"Loaded (\d+) annotations", log)
        read = re.search(r"Read (\d+) rows: (\d+) passed .*?(\d+) skipped", log)
        shard_loaded.append([int(loaded.group(1)), s["rows"], int(read.group(1)), int(read.group(3))])
    check("shards", [all(a == b == c and z == 0 for a, b, c, z in shard_loaded)], [True],
          note="per shard: DAT records loaded == shard rows == CSV rows read, 0 skipped")

    fresh, ref = json.loads((X / "stats.json").read_text()), json.loads((G / "stats.json").read_text())
    skip = {"elapsed_s", "full_csv"}
    for k in sorted(set(ref) | set(fresh)):
        if k not in skip:
            check(f"stats.{k}", fresh.get(k), ref.get(k), note="X-010..X-015")
    check("X-011", fresh["n_multi_feature"], nums(R["X-011"][0])[0])
    check("X-014", [fresh["nnz_multi"], fresh["nnz_all"]], nums(R["X-014"][0])[:2])
    check("X-015", [fresh["n_items_defined"], fresh["n_items_present_multi"]], nums(R["X-015"][0])[:2])

    fm = pl.read_parquet(X / "item_mapping_214m_base.parquet")
    gm = pl.read_parquet(G / "item_mapping_214m_base.parquet")
    key = ["feature_category", "feature_name"]
    j = gm.join(fm, on=key, how="full", coalesce=True, suffix="_cpu")
    check("mapping feature set", j["item_id"].null_count() + j["item_id_cpu"].null_count(), 0,
          note="a different feature set is data drift")
    moved = j.filter(pl.col("item_id") != pl.col("item_id_cpu")).sort("item_id")
    check("mapping ids", moved.height, 0, required=False,
          note="ids of equal-count features are randomly ordered by af-extract; GPU tables are translated if > 0")
    for name in ("item_support_all", "item_support_multi"):
        a = pl.read_parquet(X / f"{name}.parquet").join(fm, left_on="item", right_on="item_id")
        b = pl.read_parquet(G / f"{name}.parquet").join(gm, left_on="item", right_on="item_id")
        jj = a.join(b, on=key, how="full", coalesce=True, suffix="_gpu")
        bad = jj.filter(pl.col("len").is_null() | pl.col("len_gpu").is_null() | (pl.col("len") != pl.col("len_gpu")))
        check(name, [jj.height, bad.height], [b.height, 0], note="per-feature protein counts by feature_name")
    check("plddt_bin_edges.json", json.loads((X / "plddt_bin_edges.json").read_text()),
          json.loads((G / "plddt_bin_edges.json").read_text()))
    exl = (gpu / "EXCLUDED_LARGE_FILES.md").read_text()
    for f in ("transactions_214m_base.parquet", "transactions_214m_base_multi.parquet"):
        gsha = re.search(rf"{re.escape(f)}`\s*\|(?:[^|]*\|){{2}}\s*`([0-9a-f]{{64}})`", exl)
        check(f"sha256 {f}", sha256(X / f), gsha.group(1) if gsha else None, required=False,
              note="row order follows the joined CSV, whose order is not reproducible; bytes also depend on the writer")
    drift = [c for c in checks if c["required"] and not c["match"]]
    out = {"checks": checks, "required_failed": [c["id"] for c in drift], "mapping_moved": moved.to_dicts(),
           "data_drift": bool(drift)}
    h.dump_json(out, D / "phase1_verification.json")
    for c in checks:
        print(f"{'ok ' if c['match'] else ('DRIFT' if c['required'] else 'info')} {c['id']}: "
              f"{str(c['fresh'])[:90]} | expected {str(c['expected'])[:90]}")
    sys.exit(1 if drift else 0)


if __name__ == "__main__":
    main()
