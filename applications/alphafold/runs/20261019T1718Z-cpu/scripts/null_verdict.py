"""Null-model verdict by the criterion pre-registered in CPU_REPRO_GOAL.md, Phase 4 step 2.

The null is `consistent` when all of these hold, else `inconsistent`:
  (a) every CPU permutation has K_max 6;
  (b) N5 only: the mean CPU total lies within mu +/- 3 sigma / sqrt(5), mu and sigma
      (sample standard deviation, ddof=1) of the GPU run's 100 permutation totals;
  (c) N100 only: a two-sided Mann-Whitney U test of CPU vs GPU per-permutation totals
      gives p >= 0.01;
  (d) no permutation reaches the real (CPU C3) count at K = 4, 5 or 6, and none has an
      itemset with K >= 7.

Usage:
    run_py.sh null_verdict.py NULL_DIR REAL_META GPU_NULL100_JSON --mode n5|n100 --out OUT.json

Options:
    --mode n5|n100   which run: N5 uses permutations 0-4, N100 uses 0-99 (all required present)
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as h  # noqa: E402

h.check_env()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("null_dir")
    ap.add_argument("real_meta")
    ap.add_argument("gpu_json")
    ap.add_argument("--mode", choices=("n5", "n100"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from scipy.stats import mannwhitneyu

    n = 5 if a.mode == "n5" else 100
    perms = []
    for i in range(n):
        meta = json.loads((Path(a.null_dir) / f"perm_{i:03d}" / "meta.json").read_text())
        perms.append({"perm": i, "total": meta["total_itemsets"], "k_max": meta["k_max"],
                      "k_distribution": {int(k): v for k, v in meta["k_distribution"].items()}})
    real = {int(k): v for k, v in json.loads(Path(a.real_meta).read_text())["signature"]["k_distribution"].items()}
    gpu = json.loads(Path(a.gpu_json).read_text())
    gtot = [r["total_itemsets"] for r in gpu["null_runs"]]
    assert len(gtot) == 100
    mu, sd = statistics.mean(gtot), statistics.stdev(gtot)
    ctot = [p["total"] for p in perms]
    cmean = statistics.mean(ctot)
    crit = {"a_kmax_all_6": all(p["k_max"] == 6 for p in perms)}
    half = 3 * sd / math.sqrt(5)
    if a.mode == "n5":
        crit["b_mean_within_3sigma_sqrt5"] = abs(cmean - mu) <= half
    else:
        crit["c_mannwhitney_p_ge_0.01"] = bool(mannwhitneyu(ctot, gtot, alternative="two-sided").pvalue >= 0.01)
    reach = {k: [p["perm"] for p in perms if p["k_distribution"].get(k, 0) >= real.get(k, 0)] for k in (4, 5, 6)}
    deep = [p["perm"] for p in perms if p["k_max"] >= 7]
    crit["d_no_perm_reaches_real_at_k456_and_none_k_ge_7"] = not any(reach.values()) and not deep
    out = {
        "mode": a.mode, "n_cpu": n, "cpu_totals": ctot, "cpu_mean": cmean,
        "cpu_sd": statistics.stdev(ctot) if n > 1 else None, "cpu_min": min(ctot), "cpu_max": max(ctot),
        "gpu_mu": mu, "gpu_sigma_ddof1": sd, "gpu_sigma_ddof0": statistics.pstdev(gtot),
        "gpu_min": min(gtot), "gpu_max": max(gtot), "band_3sigma_sqrt5": [mu - half, mu + half],
        "mannwhitney_p": float(mannwhitneyu(ctot, gtot, alternative="two-sided").pvalue),
        "real_k_distribution": real, "perms_reaching_real": reach, "perms_with_k_ge_7": deep,
        "k_max_per_perm": [p["k_max"] for p in perms], "criteria": crit,
        "verdict": "consistent" if all(crit.values()) else "inconsistent",
        "gpu_json": str(Path(a.gpu_json).resolve()),
    }
    h.dump_json(out, a.out)
    print(json.dumps({k: out[k] for k in ("mode", "verdict", "cpu_mean", "gpu_mu", "gpu_sigma_ddof1",
                                          "mannwhitney_p", "criteria")}))


if __name__ == "__main__":
    main()
