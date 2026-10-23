"""Wall-time estimates for the PLAN.md runs from the measured sizing runs and the GPU per-level counts.

Model (the CPU route's cost per candidate depends on the level K, through the row space it
counts over, and hardly on the threshold; its candidate counts are far below the GPU's):
  * in-core run T: load + fixed overhead + sum over K of F_T(K) * r(K) * c(K), with F_T(K) the
    GPU log's frequent count at K for T's threshold (the result T must reproduce), r(K) the
    measured CPU candidates / frequent at K and c(K) the measured time per candidate at K, both
    from the reference run (C3). Levels deeper than the reference reached use r = 1 and c of
    the reference's deepest level with >= 1,000 candidates. The fixed overhead is the
    reference's mine wall minus its level times;
  * check: C2 predicted from C3's r and c, and C3 from C2's; with another reference run (--ref),
    also every other measured count-inference run (C3G, C4, C5, C6) predicted from the reference;
    the largest measured/predicted ratio, at least 1.5, is the safety factor on every estimate of a
    run not yet measured;
  * SON S_T: pass 1 = 2 chunks x (one 40M-row chunk in-core at the local threshold, measured as
    P1T at Base, scaled by the in-core wall ratio C_T / C1); pass 2 = C_T's wall;
  * X1 (4 chunks at Power): S3's estimate;
  * one null permutation at min_count 769: the in-core model on the GPU null's mean K
    distribution (100 permutations) plus 2 x C3's load for the shuffle;
  * C3G (C3 with use_generator_pruning): at most C3's wall;
  * R1 and A1: 0.1 h each;
  * repetitions: a run estimated under 15 minutes counts 3 times.

Usage:
    run_py.sh sizing.py RUN_DIR GPU_LEVELS_JSON GPU_NULL100_JSON --out sizing.json [--ref ID]

Options:
    --ref ID   measured in-core run whose per-level costs model C4-C6 and M4 (default C3;
               C3G for the count-inference fallback)

GPU_LEVELS_JSON is `gpu_levels.py` output for full_campaign_1gpu.log and minc4_mining.log.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as h  # noqa: E402

h.check_env()

GPU_SECTION = {"C1": "Base", "C2": "Super", "C3": "Power", "C4": "Blitz", "C5": "Ultra", "C6": "Opus"}


class Model:
    """Per-level candidate ratio and per-candidate cost of one measured in-core run."""

    def __init__(self, meta):
        lv = {x["k"]: x for x in meta["levels"]}
        self.r = {k: x["candidates"] / max(x["frequent"], 1) for k, x in lv.items()}
        self.c = {k: x["ms"] / 1e3 / max(x["candidates"], 1) for k, x in lv.items()}
        self.deep_c = self.c[max(k for k, x in lv.items() if x["candidates"] >= 1000)]
        self.fixed = max(meta["mine_wall_s"] - sum(x["ms"] for x in lv.values()) / 1e3, 0.0)
        self.load = meta["load_s"]

    def levels_s(self, freq):
        """Seconds of the levels for {K: frequent count}."""
        return sum(f * self.r.get(k, 1.0) * self.c.get(k, self.deep_c) for k, f in freq.items())

    def predict(self, freq):
        return self.load + self.fixed + self.levels_s(freq)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("gpu_levels")
    ap.add_argument("gpu_null")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="C3")
    a = ap.parse_args()

    run = Path(a.run)
    ci = [i for i in ("C3G", "C4", "C5", "C6") if (run / "data/mined" / i / "meta.json").exists()]
    meta = {i: json.loads((run / "data/mined" / i / "meta.json").read_text())
            for i in dict.fromkeys(("C1", "C2", "C3", "P1T", a.ref, *ci))}
    gl = json.loads(Path(a.gpu_levels).read_text())
    camp = next(v for f, v in gl.items() if f.endswith("full_campaign_1gpu.log"))
    minc4 = next(v for f, v in gl.items() if f.endswith("minc4_mining.log"))["all"]
    gpu = {i: {x["k"]: x["frequent"] for x in camp[s]} for i, s in GPU_SECTION.items()}
    gpu["M4"] = {x["k"]: x["frequent"] for x in minc4}
    gpu["C3G"] = gpu["C3"]
    null_runs = json.loads(Path(a.gpu_null).read_text())["null_runs"]
    ks = sorted({int(k) for r in null_runs for k in r["k_distribution"]})
    gpu["null"] = {k: statistics.mean(r["k_distribution"].get(str(k), 0) for r in null_runs) for k in ks}
    wall = {i: m["load_s"] + m["mine_wall_s"] for i, m in meta.items()}

    m2, m3 = Model(meta["C2"]), Model(meta["C3"])
    check = {
        "C2_from_C3": {"predicted_s": m3.predict(gpu["C2"]), "measured_s": wall["C2"]},
        "C3_from_C2": {"predicted_s": m2.predict(gpu["C3"]), "measured_s": wall["C3"]},
    }
    if a.ref != "C3":
        mr = Model(meta[a.ref])
        for i in ci:
            if i != a.ref:
                check[f"{i}_from_{a.ref}"] = {"predicted_s": mr.predict(gpu[i]), "measured_s": wall[i]}
    for c in check.values():
        c["ratio"] = c["measured_s"] / c["predicted_s"]
    safety = max(1.5, *(c["ratio"] for c in check.values()))

    est = {i: wall[i] for i in ("C1", "C2", "C3")}
    mref = Model(meta[a.ref])
    for i in ("C4", "C5", "C6", "M4"):
        est[i] = safety * mref.predict(gpu[i])
    for i, c in (("S1", "C1"), ("S2", "C2"), ("S3", "C3")):
        est[i] = safety * (2 * wall["P1T"] * wall[c] / wall["C1"] + wall[c])
    est["X1"] = est["S3"]
    perm = safety * (m3.predict(gpu["null"]) + 2 * m3.load)
    est["N5"] = 5 * perm
    est["N100"] = 95 * perm
    est["C3G"] = wall["C3"]
    est["R1"] = est["A1"] = 360.0
    counted = {i: (3 * s if s < 900 else s) for i, s in est.items()}

    out = {
        "model": __doc__.split("Usage:")[0].strip(), "ref": a.ref, "safety_factor": safety, "model_check": check,
        "measured_wall_s": wall, "null_perm_estimate_s": perm,
        "c3_candidates_per_frequent": m3.r, "c3_s_per_candidate": m3.c, "c3_deep_s_per_candidate": m3.deep_c,
        "c3_fixed_s": m3.fixed, "gpu_frequent_per_k": gpu,
        "estimate_s": est, "estimate_with_repetitions_s": counted,
        "estimate_h": {i: round(s / 3600, 2) for i, s in counted.items()},
        "peak_rss_mb": {i: m["process"]["peak_rss_mb"] for i, m in meta.items()},
    }
    h.dump_json(out, a.out)
    print(json.dumps({"safety": round(safety, 2), "check": {k: round(v["ratio"], 2) for k, v in check.items()},
                      "perm_min": round(perm / 60, 1), "estimate_h": out["estimate_h"]}))


if __name__ == "__main__":
    main()
