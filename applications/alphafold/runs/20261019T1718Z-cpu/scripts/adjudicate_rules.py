"""Adjudicate the R1 rule-count difference between the CPU run and the GPU run.

Counts the rules at a min_confidence four ways and compares the rule sets:
  * exact: integer enumeration over the CPU Base itemsets (rule A -> X \\ A when
    2 * count(X) >= count(A) for min_confidence 0.5, i.e. count(X) / count(A) >= c;
    lift >= L when count(X) * N >= L * count(A) * count(X \\ A));
  * main: the pinned engine's public `generate_rules` on the CPU itemsets (= R1) and on
    the GPU SON Base itemsets as stored (items not ascending);
  * branch: this branch's `src/et_miner/core/rules.py::generate_rules` (the generator the
    GPU run used), loaded from its file, on the same two inputs.
The GPU table's item ids are translated to CPU ids by feature_name before rule sets are
compared. Each variant also reports its rules with lift 0 (consequent lookup missed) and its
cross-domain rules (lhs + rhs span more than one feature_category, son_run.py's definition).

Usage:
    run_py.sh adjudicate_rules.py CPU_ITEMSETS GPU_SON_PARQUET BRANCH_RULES_PY --n N
        --translate GPU_MAP CPU_MAP --out OUT.json [--min-confidence C]

Options:
    --min-confidence C   default 0.5
"""

import argparse
import importlib.util
import json
import sys
from fractions import Fraction
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as h  # noqa: E402

h.check_env()


def summary(rules, cat):
    """Return counts and the (lhs, rhs) set of a list of Rule objects; `cat` maps item id -> category."""
    lifts = [r.lift for r in rules]
    cross = sum(len({cat[i] for i in (*r.lhs, *r.rhs)}) > 1 for r in rules)
    return {"n_rules": len(rules), "lift_ge5": sum(x >= 5 for x in lifts), "lift_ge100": sum(x >= 100 for x in lifts),
            "lift_zero": sum(x == 0 for x in lifts), "cross_domain": cross,
            "max_lift": max(lifts) if lifts else None}, {(tuple(r.lhs), tuple(r.rhs)) for r in rules}


def exact(tables, n, conf):
    """Integer enumeration of every rule of every itemset (K >= 2) of canonical per-K tables."""
    count = {}
    for t in tables.values():
        for items, c in zip(t["itemset"].to_list(), t["count"].to_list(), strict=True):
            count[tuple(items)] = c
    conf = Fraction(str(conf))
    rules, ge5, ge100, ties = set(), 0, 0, 0
    for x, cx in count.items():
        for r in range(1, len(x)):
            for a in combinations(x, r):
                b = tuple(i for i in x if i not in a)
                ca, cb = count[a], count[b]
                if Fraction(cx, ca) < conf:
                    continue
                ties += Fraction(cx, ca) == conf
                rules.add((a, b))
                ge5 += cx * n >= 5 * ca * cb
                ge100 += cx * n >= 100 * ca * cb
    return {"n_rules": len(rules), "lift_ge5": ge5, "lift_ge100": ge100, "confidence_ties": ties}, rules


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cpu")
    ap.add_argument("gpu_son")
    ap.add_argument("branch_rules")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--translate", nargs=2, required=True)
    ap.add_argument("--min-confidence", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import polars as pl
    from et_miner import generate_rules as main_rules

    spec = importlib.util.spec_from_file_location("branch_rules", a.branch_rules)
    branch = importlib.util.module_from_spec(spec)
    sys.modules["branch_rules"] = branch
    spec.loader.exec_module(branch)

    g, c = (pl.read_parquet(p).select("item_id", "feature_name", "feature_category") for p in a.translate)
    m = g.join(c.drop("feature_category"), on="feature_name", how="full", suffix="_cpu")
    lut = dict(zip(m["item_id"].to_list(), m["item_id_cpu"].to_list(), strict=True))
    gcat = dict(zip(g["item_id"].to_list(), g["feature_category"].to_list(), strict=True))
    ccat = dict(zip(c["item_id"].to_list(), c["feature_category"].to_list(), strict=True))

    def to_cpu(rule_set):
        return {(tuple(sorted(lut[i] for i in lhs)), tuple(sorted(lut[i] for i in rhs))) for lhs, rhs in rule_set}

    tables = h.load_tables(a.cpu, a.n)
    cpu_df = pl.concat([t.with_columns(support=pl.col("count") / a.n).select("itemset", "support")
                        for t in tables.values()])
    gpu_df = pl.read_parquet(a.gpu_son).select("itemset", "support")
    n_unsorted = gpu_df.filter(pl.col("itemset") != pl.col("itemset").list.sort()).height

    res, sets = {}, {}
    res["exact_cpu"], sets["exact_cpu"] = exact(tables, a.n, a.min_confidence)
    res["exact_cpu"]["cross_domain"] = sum(len({ccat[i] for i in (*x, *y)}) > 1 for x, y in sets["exact_cpu"])
    conf = a.min_confidence
    res["main_cpu"], sets["main_cpu"] = summary(main_rules(cpu_df, min_confidence=conf), ccat)
    res["main_gpu_stored"], s = summary(main_rules(gpu_df, min_confidence=conf), gcat)
    sets["main_gpu_stored"] = to_cpu(s)
    res["branch_gpu_stored"], s = summary(branch.generate_rules(gpu_df, min_confidence=conf), gcat)
    sets["branch_gpu_stored"] = to_cpu(s)
    res["branch_cpu"], sets["branch_cpu"] = summary(branch.generate_rules(cpu_df, min_confidence=conf), ccat)
    out = {
        "n": a.n, "min_confidence": a.min_confidence, "gpu_itemsets": gpu_df.height,
        "gpu_itemsets_not_ascending": n_unsorted, "counts": res,
        "rule_set_equal_to_exact": {k: v == sets["exact_cpu"] for k, v in sets.items()},
        "branch_gpu_stored_subset_of_exact": sets["branch_gpu_stored"] <= sets["exact_cpu"],
    }
    h.dump_json(out, a.out)
    print(json.dumps({k: out[k] for k in ("gpu_itemsets_not_ascending", "rule_set_equal_to_exact",
                                          "branch_gpu_stored_subset_of_exact")}))
    print(json.dumps({k: {x: v.get(x) for x in ("n_rules", "lift_ge5", "lift_ge100", "lift_zero", "cross_domain")}
                      for k, v in res.items()}))


if __name__ == "__main__":
    main()
