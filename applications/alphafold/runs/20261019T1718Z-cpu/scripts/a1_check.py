"""Compare the CPU deepest-itemset analysis (A1) with the GPU run's.

Compares the decoded features (as a set of names), K, the supporting proteins (as a set of
accessions) and the GO true-path result (parent-child pairs, independent K).

Usage:
    python3 a1_check.py CPU_ANALYSIS_JSON GPU_ANALYSIS_JSON --out OUT.json

Options: none beyond --out.
"""

import argparse
import json
from pathlib import Path


def key(d):
    return {"k": d["k"], "features": sorted(f["id"] for f in d["features"]),
            "proteins": sorted(p["protein_id"] for p in d["proteins"]), "n_proteins": d["n_proteins"],
            "n_parent_child_pairs": d["go_hierarchy"].get("n_parent_child_pairs"),
            "k_independent": d["go_hierarchy"].get("k_independent")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cpu")
    ap.add_argument("gpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c, g = key(json.loads(Path(a.cpu).read_text())), key(json.loads(Path(a.gpu).read_text()))
    out = {"cpu": c, "gpu": g, "equal": {k: c[k] == g[k] for k in c}, "cpu_json": str(Path(a.cpu).resolve()),
           "gpu_json": str(Path(a.gpu).resolve())}
    out["verdict"] = "identical" if all(out["equal"].values()) else "mismatch"
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"verdict": out["verdict"], "equal": out["equal"]}))


if __name__ == "__main__":
    main()
