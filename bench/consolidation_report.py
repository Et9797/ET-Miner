"""Render a consolidation campaign and apply the decision rule mechanically.

Reads <out>/raw.jsonl (bench/runner.py --mode consolidation) and, when present,
<out>/microbench.jsonl (bench/microbench_rust.py time), and writes
<out>/report.md: signature consistency per regime, a timing table per regime,
and one table per decision point with the winner of each regime under rule 2
of bench/consolidation/PROTOCOL.md.

Usage: python bench/consolidation_report.py --out bench/results/<date>-consolidation
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

GPU_WORKLOADS = ("smoke", "deepk", "skew", "oom2", "sk2ml2", "sk2ml3", "dsl", "wide", "or005", "or003", "or002")


def _load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _k_ms(row: dict, pred) -> float | None:
    levels = [lv["ms"] for lv in row.get("levels", []) if pred(lv["k"])]
    return sum(levels) / 1000 if levels else None


METRICS = {
    "wall": lambda r: r.get("wall_s"),
    "k2": lambda r: _k_ms(r, lambda k: k == 2),
    "k3plus": lambda r: _k_ms(r, lambda k: k >= 3),
    "levels": lambda r: (r.get("levels_ms_k2plus") or 0) / 1000 or None,
}


class Cell:
    """The reps of one config under one metric."""

    def __init__(self, rows: list[dict], metric: str):
        self.rows = rows
        self.statuses = sorted({r.get("status", "?") for r in rows})
        self.ok = bool(rows) and all(r.get("status") == "ok" for r in rows)
        vals = [METRICS[metric](r) for r in rows if r.get("status") == "ok"]
        self.values = [v for v in vals if v is not None]
        timeouts = [r for r in rows if r.get("status") == "timeout"]
        self.lower_bound = None
        if timeouts:
            self.lower_bound = max(r["config"]["timeout_s"] for r in timeouts)

    @property
    def usable(self) -> bool:
        return self.ok and bool(self.values)

    @property
    def med(self) -> float:
        if self.usable:
            return statistics.median(self.values)
        return self.lower_bound if self.lower_bound is not None else math.inf

    @property
    def lo(self) -> float:
        return min(self.values) if self.usable else self.med

    @property
    def hi(self) -> float:
        return max(self.values) if self.usable else self.med

    def fmt(self) -> str:
        if self.usable:
            reps = f" ({len(self.values)})" if len(self.values) != 3 else ""
            return f"{self.med:.2f} [{self.lo:.2f}, {self.hi:.2f}]{reps}"
        if self.lower_bound is not None:
            return f">{self.lower_bound} (timeout)"
        return "—" if not self.rows else "FAIL: " + "; ".join(s[:60] for s in self.statuses)


def winner(cells: dict[str, Cell]) -> str | None:
    """Rule 2: >= 10% below the best alternative, disjoint [min, max], >= 1 s saved."""
    for name, c in cells.items():
        if not c.usable:
            continue
        others = [o for n, o in cells.items() if n != name and o.rows]
        if not others:
            continue
        best = min(others, key=lambda o: o.med)
        if c.med <= 0.9 * best.med and c.hi < best.lo and best.med - c.med >= 1.0:
            return name
    return None


def _table(title: str, regimes: list[str], variants: list[str], cells_for, note: str = "") -> list[str]:
    lines = [f"### {title}", ""]
    if note:
        lines += [note, ""]
    lines.append("| regime | " + " | ".join(variants) + " | winner (rule 2) |")
    lines.append("|---|" + "---|" * (len(variants) + 1))
    for reg in regimes:
        cells = cells_for(reg)
        if not any(c.rows for c in cells.values()):
            continue
        w = winner(cells)
        lines.append(f"| {reg} | " + " | ".join(cells[v].fmt() for v in variants) + f" | {w or 'none'} |")
    return lines + [""]


def render(out: Path) -> str:
    rows = [r for r in _load(out / "raw.jsonl") if r.get("config", {}).get("mode") == "consolidation"]
    latest = {}
    for r in rows:
        latest[r["id"]] = r
    by_base: dict[str, list[dict]] = defaultdict(list)
    for r in latest.values():
        by_base[r["config"]["base_id"]].append(r)

    def cell(base: str, metric: str = "wall") -> Cell:
        return Cell(by_base.get(base, []), metric)

    lines = ["# Consolidation campaign report", ""]
    ok = sum(1 for r in latest.values() if r.get("status") == "ok")
    gpu_h = sum((r.get("proc_s") or r.get("wall_s") or 0) * max(1, len(_devices(r["config"]))) for r in latest.values())
    box_h = sum((r.get("proc_s") or r.get("wall_s") or 0) for r in latest.values())
    lines += [
        f"{len(latest)} config-reps, {ok} ok. Box time {box_h / 3600:.2f} h; GPU-hours "
        f"(process time × devices used) {gpu_h / 3600:.2f}.",
        "",
        "Cells: median [min, max] in seconds over the ok reps (a rep count in parentheses when it is not 3);",
        "`>N (timeout)` is a lower bound; `FAIL` gives the status. Rule 2 needs the winner's median ≥ 10% below",
        "the best alternative, disjoint [min, max] ranges, and ≥ 1 s saved.",
        "",
    ]

    # Signature consistency per regime.
    groups: dict[tuple, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for r in latest.values():
        if r.get("status") != "ok":
            continue
        c = r["config"]
        key = (c["dataset"], c["min_support"], c.get("max_length"), bool(c.get("prune_equal_support")))
        groups[key][f"{r['n_itemsets']}:{r['itemset_hash'][:12]}"].append(r["id"])
    lines += ["## Signatures", "", "| regime | signatures | configs |", "|---|---|---|"]
    divergent = []
    for key, sigs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        lines.append(f"| {key} | {len(sigs)} | {sum(len(v) for v in sigs.values())} |")
        if len(sigs) > 1:
            divergent.append((key, sigs))
    lines.append("")
    for key, sigs in divergent:
        lines.append(f"**SIGNATURE DIVERGENCE in {key}:**")
        for sig, ids in sigs.items():
            lines.append(f"- `{sig}`: {', '.join(sorted(ids)[:8])}")
        lines.append("")

    fails = sorted((r["id"], r.get("status")) for r in latest.values() if r.get("status") != "ok")
    if fails:
        lines += ["## Non-ok config-reps", ""]
        lines += [f"- `{i}`: {str(s)[:200]}" for i, s in fails]
        lines.append("")

    lines += ["## Decision points", ""]

    def miners(reg: str) -> dict[str, Cell]:
        best = {}
        for miner, bases in (("A", ["A1-shared", "A1-legacy"]), ("B", ["B1"]),
                             ("C", ["C1-shared", "C1-legacy", "C1-tiny0"])):
            cands = [cell(f"{reg}-{b}") for b in bases]
            usable = [c for c in cands if c.usable]
            best[miner] = min(usable, key=lambda c: c.med) if usable else next((c for c in cands if c.rows), Cell([], "wall"))
        return best

    lines += _table("DP1 — in-core miner (fastest variant per miner, wall s)", list(GPU_WORKLOADS), ["A", "B", "C"],
                    miners)
    for miner, bases in (("A1", ["A1-shared", "A1-legacy"]), ("C1", ["C1-shared", "C1-legacy"]),
                         ("C2", ["C2-shared", "C2-legacy"])):
        lines += _table(f"DP2 — K=2 kernel within {miner} (K=2 level s)", list(GPU_WORKLOADS), bases,
                        lambda reg, bases=bases: {b: cell(f"{reg}-{b}", "k2") for b in bases})
    k3 = ["A1-shared", "A1-legacy", "B1", "C1-shared", "C1-legacy", "C1-tiny0"]
    lines += _table("DP3 — K≥3 counting (Σ K≥3 level s; B = count_k3plus_gpu_resident)", list(GPU_WORKLOADS), k3,
                    lambda reg: {b: cell(f"{reg}-{b}", "k3plus") for b in k3})
    for miner, bases in (("A1", ["A1-shared", "A1-legacy"]), ("C1", ["C1-shared", "C1-legacy", "C1-tiny0"])):
        lines += _table(f"DP3 — K≥3 counting within {miner}", list(GPU_WORKLOADS), bases,
                        lambda reg, bases=bases: {b: cell(f"{reg}-{b}", "k3plus") for b in bases})
    dp4 = ["C1-shared", "C1-tiny0", "C1-legacy"]
    lines += _table("DP4 — what tiled cannot serve (wall s)", list(GPU_WORKLOADS), dp4,
                    lambda reg: {b: cell(f"{reg}-{b}") for b in dp4},
                    "C1-shared: tiny groups, mega-groups and multi-chunk K=2 on the per-candidate kernel; "
                    "C1-tiny0: tiny groups tiled too; C1-legacy: everything per-candidate.")
    for v in ("shared", "legacy"):
        bases = [f"C1-{v}", f"C1-{v}-auto", f"C1-{v}-k3"]
        lines += _table(f"DP5 — layout on C1-{v} (wall s)", list(GPU_WORKLOADS), bases,
                        lambda reg, bases=bases: {b: cell(f"{reg}-{b}") for b in bases})
        bases = [f"A1-{v}", f"A1-{v}-auto"]
        lines += _table(f"DP5 — layout on A1-{v} (wall s)", list(GPU_WORKLOADS), bases,
                        lambda reg, bases=bases: {b: cell(f"{reg}-{b}") for b in bases})
    dp6 = ["C2-shared", "C2-legacy", "Asplit2-shared", "Bsplit2"]
    lines += _table("DP6 — multi-GPU (Σ K≥2 level s)", list(GPU_WORKLOADS), dp6,
                    lambda reg: {b: cell(f"{reg}-{b}", "levels") for b in dp6})
    scale = ["C1-shared", "C2-shared", "C1-legacy", "C2-legacy"]
    lines += ["C1 vs C2 (wall s), for scaling:", ""]
    lines += _table("DP6 — row-split scaling", list(GPU_WORKLOADS), scale,
                    lambda reg: {b: cell(f"{reg}-{b}") for b in scale})

    lines += ["### DP7 — SON (wall s; per-pass s from the rep with the median wall)", "",
              "| regime | config | wall | pass 1 | pass 2 | chunks |", "|---|---|---|---|---|---|"]
    for reg in ("dsl", "deepk"):
        for b in ("D1-resident", "D1-gpu", "E2", "D1-cpu"):
            c = cell(f"{reg}-{b}")
            if not c.rows:
                continue
            t = {}
            if c.usable:
                mid = sorted(c.rows, key=lambda r: r.get("wall_s", 0))[len(c.rows) // 2]
                t = mid.get("timings", {})
            lines.append(f"| {reg} | {b} | {c.fmt()} | {t.get('pass1_s', '—')} | {t.get('pass2_s', '—')} | "
                         f"{t.get('n_chunks', '—')} |")
    lines.append("")
    dp8 = ["F-polars", "F-sparse", "F-sparse-norust", "F-auto"]
    lines += _table("DP8 — CPU tier (wall s)", list(GPU_WORKLOADS), dp8,
                    lambda reg: {b: cell(f"{reg}-{b}") for b in dp8})
    lines += _table("DP8 — R1: CPU K>2 counting (Σ K≥3 level s)", list(GPU_WORKLOADS), ["F-sparse", "F-sparse-norust"],
                    lambda reg: {b: cell(f"{reg}-{b}", "k3plus") for b in ("F-sparse", "F-sparse-norust")})
    for v in ("shared", "legacy"):
        bases = [f"C1-{v}", f"C1-{v}-norust", f"C1-{v}-noprune"]
        lines += _table(f"DP9 — host roles on C1-{v} (wall s)", list(GPU_WORKLOADS), bases,
                        lambda reg, bases=bases: {b: cell(f"{reg}-{b}") for b in bases})
    bases = ["C1-legacy-free", "C1-legacy-free-norust"]
    lines += _table("DP9 — free-set runs (R4) (wall s)", list(GPU_WORKLOADS), bases,
                    lambda reg: {b: cell(f"{reg}-{b}") for b in bases})
    for v in ("shared", "legacy"):
        bases = [f"C2-{v}", f"C2-{v}-filter-cupy", f"C2-{v}-filter-cpu"]
        lines += _table(f"DP10 — survivor filter on C2-{v} (wall s)", list(GPU_WORKLOADS), bases,
                        lambda reg, bases=bases: {b: cell(f"{reg}-{b}") for b in bases})
    bases = ["C2-legacy", "C2-legacy-balance-nnz"]
    lines += _table("DP10 — row balance (wall s)", list(GPU_WORKLOADS), bases,
                    lambda reg: {b: cell(f"{reg}-{b}") for b in bases})

    mb = out / "microbench.jsonl"
    if mb.exists():
        lines += ["### DP9 — per-call microbench, Rust vs fallback (s, median of reps)", "",
                  "| call | size | Rust | fallback | fallback / Rust |", "|---|---|---|---|---|"]
        for r in _load(mb):
            rust = statistics.median(r["rust_s"])
            fb = statistics.median(r["fallback_s"])
            size = ", ".join(f"{k}={r[k]:,}" for k in ("rows", "candidates", "groups", "prev_rows", "k") if k in r)
            bound = ">" if r.get("fallback_lower_bound") else ""
            ratio = f"{bound}{fb / rust:.1f}×" if rust > 0 else "—"
            lines.append(f"| {r['file']} | {size} | {rust:.4f} | {bound}{fb:.4f} | {ratio} |")
        lines.append("")

    lines += ["## All configs", "", "| config | wall s | K=2 s | K≥3 s | peak VRAM MB | peak RSS MB | throttled | status |",
              "|---|---|---|---|---|---|---|---|"]
    for base in sorted(by_base):
        c = cell(base)
        rs = by_base[base]
        vram = max((max(r.get("peak_vram_mb", {}).values(), default=0) for r in rs), default=0)
        rss = max((r.get("peak_rss_mb") or 0 for r in rs), default=0)
        thr = sorted({d for r in rs for d in r.get("throttled_devices", [])})
        lines.append(f"| {base} | {c.fmt()} | {cell(base, 'k2').fmt()} | {cell(base, 'k3plus').fmt()} | {vram} | "
                     f"{rss:.0f} | {thr or ''} | {', '.join(s[:40] for s in c.statuses)} |")
    lines.append("")
    return "\n".join(lines)


def _devices(cfg: dict) -> list[int]:
    return [] if cfg.get("route") == "F" else list(range(int(cfg.get("n_gpus", 1))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "report.md").write_text(render(out))
    print(f"wrote {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
