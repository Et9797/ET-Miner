"""Per-level counts of the GPU run's mining logs (candidates, frequent, seconds per K).

Parses "Threshold: NAME" section headers (if any) and the per-level lines
"K=k: candidates=C → frequent=F (Ts)" (single-GPU miner) or, for the row-split
miner, "K=k: C candidates," / "K=2: C total pairs" with "K=k: F frequent in Ts",
into one JSON record per section.

Usage:
    python3 gpu_levels.py LOG [LOG ...] > levels.json

Options: none.
"""

import json
import re
import sys

LEVEL = re.compile(r"K=(\d+): candidates=([\d,]+) → frequent=([\d,]+) \(([\d.]+)s\)")
HEAD = re.compile(r"Threshold: (\w+) \(")
RS_CAND = re.compile(r" K=(\d+): ([\d,]+) (?:candidates,|total pairs)")
RS_FREQ = re.compile(r" K=(\d+): ([\d,]+) frequent(?: items)? in ([\d.]+)s")


def parse(path):
    """Return {section: [{k, candidates, frequent, s}]} of one log."""
    out, cur, pending = {}, "all", {}
    for line in open(path, errors="replace"):
        m = HEAD.search(line)
        if m:
            cur = m.group(1)
            continue
        m = LEVEL.search(line)
        if m:
            k, c, f, t = m.groups()
            out.setdefault(cur, []).append({"k": int(k), "candidates": int(c.replace(",", "")),
                                            "frequent": int(f.replace(",", "")), "s": float(t)})
            continue
        m = RS_CAND.search(line)
        if m:
            pending[int(m.group(1))] = int(m.group(2).replace(",", ""))
            continue
        m = RS_FREQ.search(line)
        if m:
            k, f, t = int(m.group(1)), int(m.group(2).replace(",", "")), float(m.group(3))
            out.setdefault(cur, []).append({"k": k, "candidates": pending.pop(k, f), "frequent": f, "s": t})
    return out


if __name__ == "__main__":
    print(json.dumps({p: parse(p) for p in sys.argv[1:]}, indent=1))
