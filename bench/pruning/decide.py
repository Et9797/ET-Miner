"""Apply PROTOCOL.md's decision rule to a pruning campaign's runner rows.

Prints non-ok rows, the signatures per equivalence group, and one table row per
regime (workload and GPU count / layout / free-sets): each arm's wall time and
Σ K>=3 level time as median [min, max], and the rule-2 outcome of DP-P (prune
against off) and DP-I (infer against prune).

Usage:
    uv run python bench/pruning/decide.py bench/results/2026-10-05-pruning-final/raw.jsonl

Options:
    RAW_JSONL   the runner's raw.jsonl of a `--mode pruning` campaign
"""

import json
import math
import statistics
import sys
from collections import defaultdict

raw = sys.argv[1]
rows = [json.loads(line) for line in open(raw)]
cells = defaultdict(list)
k3 = defaultdict(list)
sigs = defaultdict(set)
bad = []
for r in rows:
    c = r["config"]
    base = c["base_id"]
    workload, name = base.split("-", 1)
    parts = name.split("-")
    arm = parts[-1]
    variant = "-".join(parts[:-1])  # C1, C1-free, C1-esco, C2
    regime = (workload, variant)
    if r.get("status") != "ok":
        bad.append((r["id"], r.get("status")))
        continue
    if r.get("fallbacks"):
        bad.append((r["id"], "fallback " + r["fallbacks"][0]))
    cells[(regime, arm)].append(r["wall_s"])
    split = r.get("timings", {}).get("level_split", {})
    k3[(regime, arm)].append(sum(v["level"] for kk, v in split.items() if int(kk) >= 3))
    sigs[(c["preset"], c.get("max_length"), c.get("min_support"), bool(c.get("prune_equal_support")))].add(
        (r["n_itemsets"], r["sum_counts"], r["itemset_hash"]))


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


print("non-ok / fallback rows:", bad or "none")
print("signature groups:", {k: len(v) for k, v in sigs.items()})
regimes = sorted({rg for rg, _ in cells})
print("\n| regime | off | prune | infer | Σ K≥3 off | Σ K≥3 prune | Σ K≥3 infer | DP-P | DP-I |")
print("|---|---|---|---|---|---|---|---|---|")
geo = defaultdict(list)
for rg in regimes:
    off, pr, inf = cells.get((rg, "off"), []), cells.get((rg, "prune"), []), cells.get((rg, "infer"), [])
    dpp = "prune wins" if wins(pr, off) else ("off wins" if wins(off, pr) else ("tie" if off and pr else "—"))
    dpi = "infer wins" if wins(inf, pr) else ("prune wins" if wins(pr, inf) else ("tie" if inf and pr else "—"))
    for a, v in (("off", off), ("prune", pr), ("infer", inf)):
        if v:
            geo[a].append(statistics.median(v))
    print(f"| {rg[0]} {rg[1]} | {fmt(off)} | {fmt(pr)} | {fmt(inf)} | {fmt(k3.get((rg, 'off'), []))} "
          f"| {fmt(k3.get((rg, 'prune'), []))} | {fmt(k3.get((rg, 'infer'), []))} | {dpp} | {dpi} |")
for a, v in geo.items():
    print(a, "geomean over", len(v), "regimes:", round(math.exp(sum(math.log(x) for x in v) / len(v)), 3))
