"""Mechanical decision of a moonshot idea's campaign: rules 1 and 2 per regime, and the facts for 4 and 5.

Reads the runner's raw.jsonl. A config id is ``<workload>-C<n>-<label><arm>#r<rep>``;
``--arms`` names the arms, the first being today's path. A regime is one
(workload, GPUs, label). Per regime and arm: the median wall over the ok reps
with [min, max], and the input span's median. Rule 1: the arms' signatures
(itemsets, summed counts, hash) agree within the regime and within every
equivalence group (dataset, support, max_length, free sets), and no row logged
a fallback it was not allowed. Rule 2 against the first arm: a median at least
10 % lower, disjoint [min, max] ranges and at least 1 s saved.

Usage: uv run python bench/moonshots/decide.py RAW_JSONL --arms host device [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict

_ID = re.compile(r"^(?P<workload>[^-]+)-C(?P<n>\d+)-(?P<rest>.+)#r(?P<rep>\d+)$")


def _signature(row: dict) -> tuple:
    return row.get("n_itemsets"), row.get("sum_counts"), row.get("itemset_hash")


def _group(row: dict) -> tuple:
    c = row["config"]
    return c["dataset"], c["min_support"], c["max_length"], bool(c.get("prune_equal_support"))


def _parse(row: dict, arms: list[str]) -> tuple[str, int, str, str] | None:
    m = _ID.match(row["id"])
    if not m:
        return None
    rest = m["rest"]
    arm = next((a for a in sorted(arms, key=len, reverse=True) if rest.endswith(a)), None)
    if arm is None:
        return None
    return m["workload"], int(m["n"]), rest[: -len(arm)], arm


def decide(rows: list[dict], arms: list[str]) -> dict:
    """{"regimes": [...], "must_fix": [...]} for the rows of the named arms (the latest row per id)."""
    latest = {r["id"]: r for r in rows}
    walls: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    inputs: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    sigs: dict[tuple, set] = defaultdict(set)
    must_fix: list[str] = []
    for r in latest.values():
        parsed = _parse(r, arms)
        if parsed is None:
            continue
        workload, n, label, arm = parsed
        regime = (workload, n, label)
        if r.get("status") != "ok":
            must_fix.append(f"{r['id']}: status {r.get('status')}")
            continue
        allowed = set(r["config"].get("allow_fallback", []))
        bad = [f for f in r.get("fallbacks", []) if not any(a in f for a in allowed)]
        if bad:
            must_fix.append(f"{r['id']}: logged fallback {bad}")
        walls[regime][arm].append(r["wall_s"])
        span = r.get("timings", {}).get("span", {})
        if "input" in span:
            inputs[regime][arm].append(span["input"])
        sigs[_group(r)].add(_signature(r))
    for group, s in sigs.items():
        if len(s) > 1:
            must_fix.append(f"signature mismatch in {group}: {sorted(map(str, s))}")

    out = []
    base = arms[0]
    for regime in sorted(walls):
        entry = {"workload": regime[0], "n_gpus": regime[1], "label": regime[2], "arms": {}}
        for arm in arms:
            w = walls[regime].get(arm, [])
            if not w:
                continue
            entry["arms"][arm] = {
                "reps": len(w),
                "median": statistics.median(w),
                "min": min(w),
                "max": max(w),
                "input": statistics.median(inputs[regime][arm]) if inputs[regime].get(arm) else None,
            }
        b = entry["arms"].get(base)
        for arm in arms[1:]:
            a = entry["arms"].get(arm)
            if a is None or b is None:
                continue
            saved = b["median"] - a["median"]
            a["vs_" + base] = {
                "ratio": a["median"] / b["median"] if b["median"] else None,
                "saved_s": saved,
                "ten_pct": a["median"] <= 0.9 * b["median"],
                "disjoint": a["max"] < b["min"] or b["max"] < a["min"],
                "one_s": saved >= 1.0,
            }
            v = a["vs_" + base]
            v["wins"] = v["ten_pct"] and v["disjoint"] and v["one_s"]
            v["loses"] = b["median"] <= 0.9 * a["median"] and v["disjoint"] and -saved >= 1.0
        out.append(entry)
    return {"regimes": out, "must_fix": must_fix}


def _table(result: dict, arms: list[str]) -> str:
    base = arms[0]
    head = ["regime", "GPUs"] + [f"{a} wall [min, max]" for a in arms] + [f"{a} input" for a in arms]
    head += [f"{a}/{base}" for a in arms[1:]] + ["rule 2"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for e in result["regimes"]:
        cells = [f"{e['workload']}{'-' + e['label'].rstrip('-') if e['label'] else ''}", str(e["n_gpus"])]
        for a in arms:
            s = e["arms"].get(a)
            cells.append(f"{s['median']:.2f} [{s['min']:.2f}, {s['max']:.2f}]" if s else "—")
        for a in arms:
            s = e["arms"].get(a)
            cells.append(f"{s['input']:.2f}" if s and s["input"] is not None else "—")
        verdicts = []
        for a in arms[1:]:
            v = e["arms"].get(a, {}).get("vs_" + base)
            cells.append(f"{v['ratio']:.3f}" if v else "—")
            if v:
                verdicts.append(f"{a} wins" if v["wins"] else f"{a} loses" if v["loses"] else "tie")
        cells.append(", ".join(verdicts) or "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.raw) if line.strip()]
    result = decide(rows, args.arms)
    if args.json:
        print(json.dumps(result, indent=1))
        return 0
    print(_table(result, args.arms))
    print()
    print("must-fix:" if result["must_fix"] else "must-fix: none")
    for m in result["must_fix"]:
        print(f"- {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
