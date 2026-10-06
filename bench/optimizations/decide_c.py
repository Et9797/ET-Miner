"""Apply PROTOCOL-C.md's decision rule (DP-O6) to the phase C campaign rows, and check the final run.

Usage:
    uv run python bench/optimizations/decide_c.py CAMPAIGN_RAW_JSONL [FINAL_RAW_JSONL]
"""

import json
import math
import statistics
import sys
from collections import defaultdict

PHASES = ("count_percand", "count_group", "count_tiled", "count_fused")


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
    if not a or not b:
        return False
    ma, mb = statistics.median(a), statistics.median(b)
    return ma <= 0.9 * mb and max(a) < min(b) and mb - ma >= 1.0


def k3(r):
    split = r.get("timings", {}).get("level_split", {})
    return {p: sum(v.get(p, 0.0) for k, v in split.items() if int(k) >= 3) for p in PHASES}


rows = load(sys.argv[1])
cells = defaultdict(list)
ev = defaultdict(lambda: defaultdict(list))
vram = defaultdict(list)
sigs = defaultdict(set)
revs = set()
bad = []
for r in rows:
    regime, arm = regime_arm(r)
    revs.add(r.get("rev"))
    if r.get("status") != "ok":
        bad.append((r["id"], r.get("status")))
        continue
    if r.get("fallbacks"):
        bad.append((r["id"], "fallback " + r["fallbacks"][0]))
    cells[(regime, arm)].append(r["wall_s"])
    pv = r.get("peak_vram_mb") or 0
    vram[(regime, arm)].append(max(pv.values()) if isinstance(pv, dict) else pv)
    for p, v in k3(r).items():
        ev[(regime, arm)][p].append(v)
    sigs[group(r)].add(signature(r))

print("revisions:", revs)
print("non-ok / fallback rows:", bad or "none")
print("signature groups:", {k: len(v) for k, v in sigs.items()})
print("\n## DP-O6: group against base (K>=3 split: per-candidate / group / tiled+fused, s)\n")
print("| regime | base | group | split, base | split, group | outcome |")
print("|---|---|---|---|---|---|")
geo = defaultdict(list)
for rg in sorted({rg for rg, a in cells}, key=lambda x: (x[0], x[1])):
    b, g = cells.get((rg, "base"), []), cells.get((rg, "group"), [])
    out = "group wins" if wins(g, b) else ("base wins" if wins(b, g) else "tie")

    def split(arm):
        e = ev[(rg, arm)]
        med = {p: statistics.median(e[p]) if e[p] else 0.0 for p in PHASES}
        return f"{med['count_percand']:.2f} / {med['count_group']:.2f} / {med['count_tiled'] + med['count_fused']:.2f}"

    if b and g:
        geo["base"].append(statistics.median(b))
        geo["group"].append(statistics.median(g))
        delta = f" ({statistics.median(g) - statistics.median(b):+.2f} s)"
    else:
        delta = ""
    print(f"| {rg[0]} {rg[1]} | {fmt(b)} | {fmt(g)} | {split('base')} | {split('group')} | {out}{delta} |")
for a, v in geo.items():
    print(f"{a} geomean over {len(v)} regimes: {math.exp(sum(map(math.log, v)) / len(v)):.3f}")
print("peak VRAM MB per device, median of the max:")
for (rg, a), v in sorted(vram.items()):
    print(f"  {rg[0]} {rg[1]} {a}: {statistics.median(v):.0f}")

if len(sys.argv) > 2:
    print("\n## Final check (knobs unset, one rep)\n")
    print("| regime | wall | K>=3 split per-candidate / group / tiled | signature |")
    print("|---|---|---|---|")
    for r in load(sys.argv[2]):
        regime, _ = regime_arm(r)
        if r.get("status") != "ok":
            print(f"| {regime[0]} {regime[1]} | {r.get('status')} | | |")
            continue
        e = k3(r)
        campaign = sigs.get(group(r), set())
        sig = "matches" if campaign == {signature(r)} else f"MISMATCH (campaign {campaign})"
        fb = f", fallback {r['fallbacks'][0]}" if r.get("fallbacks") else ""
        print(
            f"| {regime[0]} {regime[1]} | {r['wall_s']:.2f} | {e['count_percand']:.2f} / {e['count_group']:.2f} / "
            f"{e['count_tiled'] + e['count_fused']:.2f} | {sig}{fb} |"
        )
