"""Gate G0: CPU in-core == CPU SON (4 chunks) == efficient-apriori on the first rows.

Reads the three runs' meta JSONs and the two compare reports (in-core vs
oracle, SON vs oracle), requires both `identical`, equal signatures and a
4-chunk SON run, and states whether the GPU run's per-level counts
(validate_row_split.log) are a comparable cross-check: only when the rebuilt
row order equals the GPU run's.

Usage:
    run_py.sh g0_check.py G0_DIR GPU_LOG PHASE1_VERIFICATION_JSON --out G0.json

Options: none. G0_DIR holds incore/, son/, ea/ (meta.json each) and
cmp_incore_ea.json, cmp_son_ea.json.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as h  # noqa: E402

h.check_env()


def gpu_levels(log):
    """Return {K: count} of the single-GPU column in validate_row_split.log's K-distribution."""
    out = {}
    for m in re.finditer(r"K=\s*(\d+):\s+([\d,]+) vs\s+([\d,]+)", Path(log).read_text()):
        out[int(m.group(1))] = int(m.group(2).replace(",", ""))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("g0")
    ap.add_argument("gpu_log")
    ap.add_argument("phase1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    g = Path(a.g0)
    meta = {k: json.loads((g / k / "meta.json").read_text()) for k in ("incore", "son", "ea")}
    cmp_ = {k: json.loads((g / f"cmp_{k}_ea.json").read_text()) for k in ("incore", "son")}
    sigs = {k: m["signature"]["sha256"] for k, m in meta.items()}
    p1 = json.loads(Path(a.phase1).read_text())
    order = [c for c in p1["checks"] if c["id"] == "sha256 transactions_214m_base_multi.parquet"][0]
    same_order = bool(order["match"])
    gpu = gpu_levels(a.gpu_log)
    cpu = {int(k): v for k, v in meta["incore"]["signature"]["k_distribution"].items()}
    res = {
        "n_rows": meta["incore"]["n_rows"], "min_count": meta["incore"]["min_count"],
        "itemsets": {k: m["signature"]["itemsets"] for k, m in meta.items()},
        "k_max": {k: m["k_max"] for k, m in meta.items()}, "signatures": sigs,
        "son_chunks": meta["son"]["chunks"],
        "son_pass1_events": sum(e["phase"] == "pass1" for e in meta["son"]["progress"]),
        "ea_max_length": meta["ea"]["max_length"],
        "compare": {k: c["verdict"] for k, c in cmp_.items()},
        "gpu_total": sum(gpu.values()), "gpu_k_distribution": gpu, "cpu_k_distribution": cpu,
        "row_order_equals_gpu": same_order,
    }
    res["exact_three_way"] = (len(set(sigs.values())) == 1 and all(v == "identical" for v in res["compare"].values())
                              and res["son_chunks"] == 4 and res["son_pass1_events"] == 4)
    if same_order:
        res["gpu_crosscheck"] = "identical" if cpu == gpu else "mismatch"
    else:
        res["gpu_crosscheck"] = ("not comparable: the first 1,000,000 rows of the rebuilt mined set are not the GPU "
                                 "run's rows (row order follows the joined pLDDT CSV, whose order comes from 16 "
                                 "parallel Storage Read streams and an unordered polars join; sha256 of the "
                                 "rebuilt multi parquet != EXCLUDED_LARGE_FILES.md; ORIGINAL_DATA_DIR not set)")
    res["pass"] = res["exact_three_way"] and res["gpu_crosscheck"] != "mismatch"
    h.dump_json(res, a.out)
    print(json.dumps({k: res[k] for k in ("itemsets", "k_max", "compare", "son_chunks", "exact_three_way",
                                          "row_order_equals_gpu", "pass")}))
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
