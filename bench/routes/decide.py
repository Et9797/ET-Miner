"""The route consolidation's decision rule (bench/routes/PROTOCOL.md), applied mechanically to the runner's rows.

Reads one or more raw.jsonl files (the latest row per id). A config id is ``<regime>-<arm>#r<rep>`` with
``arm`` = ``<route><n>-<name>`` (``bench/routes_matrix.py``). Per regime and arm: the median wall over the ok
reps with [min, max], peak VRAM (largest device) and peak RSS. Per decision point (``DPS``):

- rule 1: every row of a (dataset, min_support, max_length, free sets) group has one signature, and no ok row
  logged a fallback; a mismatch disqualifies the arm (must-fix) and the decision point removes nothing;
- rule 2 per regime: an arm wins when its median is >= 10 % below the best alternative's, the [min, max]
  ranges do not overlap and it saves >= 1 s; a config that errors or times out cannot win;
- rules 3-7 and 11 as the protocol states them, per member (leave-one-out) or per path (head-to-head).

Usage: uv run python bench/routes/decide.py RAW.jsonl [RAW.jsonl ...] [--json OUT] [--md OUT]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

_ID = re.compile(r"^(?P<regime>.+)-(?P<arm>[A-Z]\d-[a-z0-9-]+)#r(?P<rep>\d+)$")

#: Leave-one-out decision points: DP -> {member: (full arm name, arm without the member)}; the route and GPU
#: count come from the regime's rows.
LEAVE_ONE_OUT = {
    "G3": {
        "rows": ("default", "no-rows"),
        "percand": ("default", "no-k2percand"),
        "tiled": ("default", "no-k2tiled"),
        "fused": ("default", "no-k2fused"),
        "shared": ("rows", "rows-global"),
    },
    "G5": {
        "percand": ("default", "no-percand"),
        "group": ("default", "no-group"),
        "tiled": ("default", "no-tiled"),
        "fused": ("default", "no-fused"),
    },
    "C1": {"gram": ("default", "bitvec"), "bitvec": ("default", "gram")},
    "C2": {"popcount": ("default", "proj"), "proj": ("default", "popcount"),
           "son-percand": ("default", "groups"), "son-groups": ("default", "percand")},
}
#: A path whose peak VRAM stays below this used no GPU memory of its own (rule 7 compares RSS only).
GPU_FLOOR_MB = 64
#: Rule 7's memory guard counts a rise only above this absolute amount as well as 1.25x (owner, 2026-10-28,
#: after Phase 2b: 46-58 MB of VRAM or 0.2 GB of RSS does not block a removal).
RISE_FLOOR_MB = 1024
#: Paths rule 9 keeps whatever their head-to-head says: SON on the CPU is the CPU route's streaming strategy.
RULE_9 = {("S1", "cpu")}
#: Head-to-head decision points: DP -> {path: [arm labels that run it]} (route tag + name, no GPU count).
HEAD_TO_HEAD = {
    "G1": {"device": ["C-default", "C-coded"], "host": ["C-host"], "rebuilt": ["C-rebuilt"]},
    "G2": {"lex": ["C-default"], "keep": ["C-keep"]},
    "G6": {"dense": ["C-default"], "esco-auto": ["C-esco-auto"], "esco3": ["C-esco3"]},
    "C3": {"cpu": ["F-default"], "rust": ["R-default"]},
    "S1": {"cpu": ["D-default"], "boolean": ["D-boolean"], "ported": ["D-ported"]},
    "S2": {"boolean": ["E-boolean"], "ported": ["E-ported"]},
}


def _load(paths: list[str]) -> list[dict]:
    rows: dict[str, dict] = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[r["id"]] = r
    return list(rows.values())


def _sig(r: dict) -> tuple:
    return r.get("n_itemsets"), r.get("sum_counts"), r.get("itemset_hash")


def _group(r: dict) -> tuple:
    c = r["config"]
    return c["dataset"], c["min_support"], c.get("max_length"), bool(c.get("prune_equal_support"))


def _peak_vram(r: dict) -> float:
    return max((r.get("peak_vram_mb") or {}).values(), default=0)


class Cell:
    """One (regime, arm): its reps."""

    def __init__(self):
        self.walls: list[float] = []
        self.failed: list[str] = []
        self.vram: list[float] = []
        self.rss: list[float] = []

    @property
    def ok(self) -> bool:
        return bool(self.walls) and not self.failed

    @property
    def median(self) -> float:
        return statistics.median(self.walls)

    def lo(self) -> float:
        return min(self.walls)

    def hi(self) -> float:
        return max(self.walls)

    def fmt(self) -> str:
        if self.failed and not self.walls:
            return f"FAIL ({self.failed[0][:60]})"
        s = f"{self.median:.2f} [{self.lo():.2f}, {self.hi():.2f}]"
        return s + (f" ({len(self.walls)} rep)" if len(self.walls) == 1 else "") + (" +FAIL" if self.failed else "")


def wins(a: Cell, b: Cell) -> bool:
    """Rule 2: ``a`` wins against ``b``."""
    if not a.ok:
        return False
    if not b.ok:
        return bool(b.failed)
    return a.median <= 0.9 * b.median and a.hi() < b.lo() and b.median - a.median >= 1.0


def cells(rows: list[dict]) -> tuple[dict, list[str]]:
    """({regime: {arm: Cell}}, must-fix) with rule 1 applied: a group with two signatures or an ok row with a
    logged fallback is listed."""
    out: dict[str, dict[str, Cell]] = defaultdict(lambda: defaultdict(Cell))
    must_fix: list[str] = []
    sigs: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        m = _ID.match(r["id"])
        if not m:
            continue
        cell = out[m["regime"]][m["arm"]]
        status = r.get("status", "")
        if status == "ok":
            cell.walls.append(float(r["wall_s"]))
            cell.vram.append(_peak_vram(r))
            cell.rss.append(float(r.get("peak_rss_mb") or 0))
            sigs[_group(r)].setdefault(_sig(r), []).append(r["id"])
        else:
            cell.failed.append(status)
            if status.startswith("fallback"):
                must_fix.append(f"{r['id']}: {status[:200]}")
    for group, by_sig in sigs.items():
        if len(by_sig) > 1:
            must_fix.append(f"signature mismatch in {group}: " + "; ".join(f"{s}: {ids[:3]}" for s, ids in by_sig.items()))
    return out, must_fix


def _arms(regime_cells: dict[str, Cell], name: str) -> list[tuple[str, Cell]]:
    return [(arm, c) for arm, c in regime_cells.items() if arm.split("-", 1)[1] == name]


def leave_one_out(table: dict, dp: str) -> list[dict]:
    """Per member: the regimes where both arms ran, who wins, the decision (rules 3, 7, 11)."""
    out = []
    for member, (full, without) in LEAVE_ONE_OUT[dp].items():
        regimes = []
        for regime, rc in sorted(table.items()):
            for arm_full, c_full in _arms(rc, full):
                route_n = arm_full.split("-", 1)[0]
                c_without = rc.get(f"{route_n}-{without}")
                if c_without is None:
                    continue
                regimes.append({
                    "regime": f"{regime} ({route_n})",
                    "full": c_full.fmt(), "without": c_without.fmt(),
                    "full_wins": wins(c_full, c_without), "without_wins": wins(c_without, c_full),
                    "slowdown": (c_without.median / c_full.median) if c_full.ok and c_without.ok else None,
                    "capability": (not c_without.ok and c_full.ok) or (
                        c_full.ok and c_without.ok and any(
                            max(w) > 1.25 * max(f) + 1 and max(w) - max(f) >= RISE_FLOOR_MB
                            for w, f in ((c_without.vram, c_full.vram), (c_without.rss, c_full.rss)))),
                })
        if not regimes:
            out.append({"member": member, "decision": "no regime ran", "regimes": []})
            continue
        full_wins = any(r["full_wins"] for r in regimes)
        without_wins = any(r["without_wins"] for r in regimes)
        if full_wins and without_wins:
            decision = "rule 5: partial winner (both win somewhere)"
        elif full_wins:
            decision = "stays (rule 3: the full dispatch wins a regime)"
        else:
            decision = "goes (rule 3)" + (": without it wins somewhere and loses nowhere" if without_wins else
                                          ": neither wins anywhere")
            if any(r["capability"] for r in regimes):
                decision += "; BLOCKED by rule 7 (capability guard)"
        rule11 = [r["regime"] for r in regimes if r["slowdown"] and r["slowdown"] >= 1.25]
        out.append({"member": member, "decision": decision, "rule11": rule11, "regimes": regimes})
    return out


def _guard(path: str, c: Cell, replacement: Cell | None) -> str | None:
    """Rule 7 for removing ``path`` from one regime, where ``replacement`` serves it instead."""
    if not c.ok:
        return None
    if replacement is None or not replacement.ok:
        return f"{path} ran where no other path did"
    for what, mine, theirs in (("VRAM", c.vram, replacement.vram), ("RSS", c.rss, replacement.rss)):
        if what == "VRAM" and max(mine) < GPU_FLOOR_MB:
            continue  # a CPU path: its "VRAM" is the context of other processes, no capability to lose
        if max(theirs) > 1.25 * max(mine) + 1 and max(theirs) - max(mine) >= RISE_FLOOR_MB:
            return f"peak {what} {max(theirs) / max(max(mine), 1):.2f}x (+{(max(theirs) - max(mine)) / 1024:.1f} GB) without {path}"
    return None


def head_to_head(table: dict, dp: str) -> list[dict]:
    """Per path: the regimes it wins (rule 2 against the best alternative, at one GPU count), the decision (rules 4
    and 7: a path that wins nothing goes unless its removal fails a regime or raises its peak memory > 1.25x)."""
    paths = HEAD_TO_HEAD[dp]
    regimes = []
    for regime, rc in sorted(table.items()):
        for n in sorted({arm.split("-", 1)[0][1:] for arm in rc}):
            present = {}
            for path, labels in paths.items():
                for label in labels:
                    route, name = label.split("-", 1)
                    for arm, c in rc.items():
                        if arm.split("-", 1)[0] == f"{route}{n}" and arm.split("-", 1)[1] == name:
                            present[path] = (arm, c)
            if len(present) < 2:
                continue
            row = {"regime": regime if n == "1" else f"{regime} ({n} GPUs)",
                   "arms": {p: f"{a}: {c.fmt()}" for p, (a, c) in present.items()}, "winner": None, "guard": {}}
            for path, (_, c) in present.items():
                others = [oc for p, (_, oc) in present.items() if p != path]
                ok_others = [o for o in others if o.ok]
                best = min(ok_others, key=lambda o: o.median) if ok_others else (others[0] if others else None)
                if best is not None and wins(c, best) and all(wins(c, o) or not o.ok for o in others):
                    row["winner"] = path
                if (why := _guard(path, c, min(ok_others, key=lambda o: o.median) if ok_others else None)):
                    row["guard"][path] = why
            regimes.append(row)
    won = {r["winner"] for r in regimes if r["winner"]}
    decisions = {}
    for p in paths:
        if p in won:
            decisions[p] = "wins " + ", ".join(r["regime"] for r in regimes if r["winner"] == p)
            continue
        if (dp, p) in RULE_9:
            decisions[p] = "wins no regime; stays (rule 9: SON is never removed as a strategy)"
            continue
        blocked = [f"{r['regime']}: {r['guard'][p]}" for r in regimes if p in r["guard"]]
        decisions[p] = "wins no regime (rule 4: goes, subject to rule 9)" + (
            f"; BLOCKED by rule 7 ({'; '.join(blocked)})" if blocked else "")
    return [{"decisions": decisions, "regimes": regimes}]


def g7(table: dict) -> list[dict]:
    """G7, head-to-head: each regime's default config against its ``-noprune`` twin (prune_apriori=False)."""
    regimes = []
    for regime, rc in sorted(table.items()):
        if not regime.endswith("-noprune") or regime[: -len("-noprune")] not in table:
            continue
        base = table[regime[: -len("-noprune")]]
        for arm, c in rc.items():
            if arm.endswith("-default") and arm in base:
                p = base[arm]
                winner = "prune" if wins(p, c) else "noprune" if wins(c, p) else None
                regimes.append({"regime": f"{regime[: -len('-noprune')]} ({arm[:2]})",
                                "arms": {"prune": p.fmt(), "noprune": c.fmt()}, "winner": winner})
    won = {r["winner"] for r in regimes if r["winner"]}
    decisions = {p: ("wins " + ", ".join(r["regime"] for r in regimes if r["winner"] == p) if p in won else
                     "wins no regime (rule 4)") for p in ("prune", "noprune")}
    return [{"decisions": decisions, "regimes": regimes}]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="+")
    ap.add_argument("--json", default=None)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    table, must_fix = cells(_load(args.raw))
    result = {"must_fix": must_fix, "dps": {}}
    for dp in LEAVE_ONE_OUT:
        result["dps"][dp] = leave_one_out(table, dp)
    for dp in HEAD_TO_HEAD:
        result["dps"][dp] = head_to_head(table, dp)
    result["dps"]["G7"] = g7(table)
    lines = ["# Decisions (bench/routes/decide.py)", ""]
    lines += ["## Rule 1", ""] + ([f"- (must-fix) {m}" for m in must_fix] or ["- none"]) + [""]
    for dp, entries in result["dps"].items():
        lines += [f"## {dp}", ""]
        for e in entries:
            if "member" in e:
                lines += [f"**{e['member']}**: {e['decision']}" + (f"; rule 11: {', '.join(e['rule11'])}" if e.get("rule11") else ""), ""]
                if e["regimes"]:
                    lines += ["| regime | full | without | full wins | without wins |", "|---|---|---|---|---|"]
                    lines += [f"| {r['regime']} | {r['full']} | {r['without']} | {r['full_wins']} | {r['without_wins']} |"
                              for r in e["regimes"]]
                    lines += [""]
            else:
                lines += [f"- **{p}**: {d}" for p, d in e["decisions"].items()] + [""]
                if e["regimes"]:
                    paths = list(HEAD_TO_HEAD.get(dp, {"prune": 0, "noprune": 0}))
                    lines += ["| regime | " + " | ".join(paths) + " | winner |", "|---" * (len(paths) + 2) + "|"]
                    lines += [f"| {r['regime']} | " + " | ".join(r["arms"].get(p, "—") for p in paths)
                              + f" | {r['winner'] or 'none'} |" for r in e["regimes"]]
                    lines += [""]
    text = "\n".join(lines)
    if args.md:
        Path(args.md).write_text(text + "\n")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(text)
    return 1 if must_fix else 0


if __name__ == "__main__":
    raise SystemExit(main())
