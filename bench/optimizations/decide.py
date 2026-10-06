"""Apply PROTOCOL.md's decision rule to a phase A campaign's runner rows, and check the final run.

Prints non-ok rows, the signatures per equivalence group, and per decision
(DP-O1 rows, DP-O2 compact, DP-O3 reuse against base) one table row per regime:
both arms' wall time and the decision's evidence from the level split as
median [min, max], and the rule-2 outcome. With a second file, the final
check's rows (`--mode optimizations-final`): each regime's wall time, which
K=2 kernel and which K>=3 reduce ran, and whether its signature matches the
campaign's.

Usage:
    uv run python bench/optimizations/decide.py CAMPAIGN_RAW_JSONL [FINAL_RAW_JSONL]
"""

import json
import math
import statistics
import sys
from collections import defaultdict


def load(path):
    return [json.loads(line) for line in open(path)]


def regime_arm(r):
    workload, name = r["config"]["base_id"].split("-", 1)
    parts = name.split("-")
    return (workload, "-".join(parts[:-1])), parts[-1]


def group(r):
    c = r["config"]
    return c["preset"], c.get("max_length"), c.get("min_support"), bool(c.get("prune_equal_support"))


def signature(r):
    return r["n_itemsets"], r["sum_counts"], r["itemset_hash"]


def fmt(v):
    if not v:
        return "—"
    if len(v) == 1:
        return f"{v[0]:.2f}"
    return f"{statistics.median(v):.2f} [{min(v):.2f}, {max(v):.2f}]"


def wins(a, b):
    """Rule 2: a beats b."""
    if not a or not b:
        return False
    ma, mb = statistics.median(a), statistics.median(b)
    return ma <= 0.9 * mb and max(a) < min(b) and mb - ma >= 1.0


def evidence(r):
    split = r.get("timings", {}).get("level_split", {})
    return {
        "K=2 level": split.get("2", {}).get("level", 0.0),
        "K>=3 reduce + compact": sum(v.get("reduce", 0) + v.get("compact", 0) for k, v in split.items() if int(k) >= 3),
        "materialize": sum(v.get("materialize", 0) for v in split.values()),
    }


rows = load(sys.argv[1])
cells = defaultdict(list)
ev = defaultdict(lambda: defaultdict(list))
sigs = defaultdict(set)
bad = []
for r in rows:
    regime, arm = regime_arm(r)
    if arm == "cal":
        print(f"calibration {r['id']}: {r.get('status')}")
        continue
    if r.get("status") != "ok":
        bad.append((r["id"], r.get("status")))
        continue
    if r.get("fallbacks"):
        bad.append((r["id"], "fallback " + r["fallbacks"][0]))
    cells[(regime, arm)].append(r["wall_s"])
    for name, value in evidence(r).items():
        ev[(regime, arm)][name].append(value)
    sigs[group(r)].add(signature(r))

print("non-ok / fallback rows:", bad or "none")
print("signature groups:", {k: len(v) for k, v in sigs.items()})
for arm, (dp, ev_name) in {"rows": ("DP-O1", "K=2 level"), "compact": ("DP-O2", "K>=3 reduce + compact"),
                           "reuse": ("DP-O3", "materialize")}.items():
    print(f"\n## {dp}: {arm} against base (evidence: {ev_name}, s)\n")
    print(f"| regime | base | {arm} | {ev_name}, base | {ev_name}, {arm} | outcome |")
    print("|---|---|---|---|---|---|")
    geo = defaultdict(list)
    for rg in sorted({rg for rg, a in cells if a == arm}):
        b, x = cells.get((rg, "base"), []), cells[(rg, arm)]
        out = f"{arm} wins" if wins(x, b) else ("base wins" if wins(b, x) else "tie")
        geo["base"].append(statistics.median(b))
        geo[arm].append(statistics.median(x))
        print(f"| {rg[0]} {rg[1]} | {fmt(b)} | {fmt(x)} | {fmt(ev[(rg, 'base')][ev_name])} "
              f"| {fmt(ev[(rg, arm)][ev_name])} | {out} ({statistics.median(x) - statistics.median(b):+.2f} s) |")
    for a, v in geo.items():
        print(f"{a} geomean over {len(v)} regimes: {math.exp(sum(map(math.log, v)) / len(v)):.3f}")

if len(sys.argv) > 2:
    print("\n## Final check (knobs unset, one rep)\n")
    print("| regime | wall | K=2 kernel | K>=3 reduce | signature |")
    print("|---|---|---|---|---|")
    for r in load(sys.argv[2]):
        regime, _ = regime_arm(r)
        if r.get("status") != "ok":
            print(f"| {regime[0]} {regime[1]} | {r.get('status')} | | | |")
            continue
        split = r.get("timings", {}).get("level_split", {})
        k2 = "rows" if split.get("2", {}).get("count_rows", 0) > 0 else "dense"
        compacted = any(v.get("compact", 0) > 0 for k, v in split.items() if int(k) >= 3)
        reduce = "compact" if compacted else ("dense" if r["config"]["n_gpus"] > 1 else "—")
        campaign = sigs.get(group(r), set())
        sig = "matches" if campaign == {signature(r)} else f"MISMATCH (campaign {campaign})"
        fb = f", fallback {r['fallbacks'][0]}" if r.get("fallbacks") else ""
        print(f"| {regime[0]} {regime[1]} | {r['wall_s']:.2f} | {k2} | {reduce} | {sig}{fb} |")
