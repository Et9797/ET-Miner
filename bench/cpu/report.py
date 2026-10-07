"""Tables for the CPU-tier baseline: per-regime times, where the time goes, and each lever's projection.

Reads the campaign's raw.jsonl (bench/runner.py --mode cpu-baseline) and the
offline stakes' stakes.jsonl (bench/cpu/stakes.py), and writes report.md next
to them.

Usage:
    uv run python bench/cpu/report.py bench/results/2026-10-07-cpu-baseline

Options:
    DIR   results directory holding raw.jsonl and (optionally) stakes.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpu.matrix import F_ARMS, THREADS, WORKLOADS  # noqa: E402

#: K=2 / K>=3 sub-phases of "count" by route.
SUB = ("len_filter", "density", "to_csr", "k2_sparse", "matmul", "kgt2", "rust_call", "polars_count", "polars_collect")


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _cell(vals: list[float]) -> str:
    if not vals:
        return "—"
    med = statistics.median(vals)
    if len(vals) == 1:
        return f"{med:.2f} (1 rep)"
    return f"{med:.2f} [{min(vals):.2f}, {max(vals):.2f}]"


def collect(raw: list[dict]) -> dict:
    """{base_id: {"ok": [rows], "bad": [rows]}} keeping the last row per id."""
    latest = {r["id"]: r for r in raw}
    by: dict[str, dict] = defaultdict(lambda: {"ok": [], "bad": []})
    for r in latest.values():
        c = r.get("config", {})
        if c.get("mode") != "cpu":
            continue
        if str(r.get("status", "")).startswith("skipped"):
            continue
        by[c["base_id"]]["ok" if r.get("status") == "ok" else "bad"].append(r)
    return by


def median_row(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r["wall_s"])
    return rows[(len(rows) - 1) // 2]


def split_summary(r: dict) -> dict:
    """Phase seconds summed per band: K=1 (build), K=2, K>=3, tail."""
    t = r.get("timings", {})
    sp = t.get("split") or {}
    out: dict[str, float] = defaultdict(float)
    if r["config"]["route"] == "EA":
        out["index"] = t.get("ea_index_s") or 0.0
        for k, v in sp.items():
            band = "k1" if int(k) == 1 else ("k2" if int(k) == 2 else "k3+")
            out[f"{band}.cand_gen"] += v["cand_gen"]
            out[f"{band}.count"] += v["count"]
        return dict(out)
    for k, ph in sp.items():
        k = int(k)
        if k == 1:
            out["build"] += ph.get("matrix_build", 0.0)
            out["k1"] += ph.get("level", 0.0)
            out["k1.pre"] += ph.get("span", 0.0) - ph.get("matrix_build", 0.0) - ph.get("level", 0.0)
            continue
        band = "k2" if k == 2 else "k3+"
        for p in ("cand_gen", "count", "other", *SUB):
            out[f"{band}.{p}"] += ph.get(p, 0.0)
        out[f"{band}.gap"] += ph.get("span", 0.0) - ph.get("level", 0.0)
    out["tail"] = sum((t.get("tail") or {}).values())
    return dict(out)


def baseline_table(by: dict) -> list[str]:
    lines = ["| regime | " + " | ".join(f"{a}-{t}" for a in F_ARMS for t in THREADS) + " | EA | EA convert |",
             "|---|" + "---|" * (len(F_ARMS) * len(THREADS) + 2)]
    for w in WORKLOADS:
        cells = []
        for a in F_ARMS:
            for t in THREADS:
                g = by.get(f"{w}-{a}-{t}", {"ok": [], "bad": []})
                cell = _cell([r["wall_s"] for r in g["ok"]])
                if not g["ok"] and g["bad"]:
                    b = g["bad"][0]
                    cell = f"> {b.get('proc_s')} ({b.get('status')[:10]})"
                cells.append(cell)
        ea = by.get(f"{w}-EA", {"ok": [], "bad": []})
        cells.append(_cell([r["wall_s"] for r in ea["ok"]]))
        cells.append(_cell([r["timings"].get("convert_s", 0.0) for r in ea["ok"]]))
        lines.append(f"| {w} | " + " | ".join(cells) + " |")
    return lines


def ratio_table(by: dict) -> list[str]:
    lines = ["| regime | " + " | ".join(f"{a}-{t}" for a in F_ARMS for t in THREADS) + " |",
             "|---|" + "---|" * (len(F_ARMS) * len(THREADS))]
    for w in WORKLOADS:
        ea = [r["wall_s"] for r in by.get(f"{w}-EA", {"ok": []})["ok"]]
        if not ea:
            continue
        e = statistics.median(ea)
        cells = []
        for a in F_ARMS:
            for t in THREADS:
                v = [r["wall_s"] for r in by.get(f"{w}-{a}-{t}", {"ok": []})["ok"]]
                cells.append(f"{statistics.median(v) / e:.2f}×" if v else "—")
        lines.append(f"| {w} | " + " | ".join(cells) + " |")
    return lines


def memory_table(by: dict) -> list[str]:
    lines = ["| regime | " + " | ".join(f"{a}-{t}" for a in F_ARMS for t in THREADS) + " | EA |",
             "|---|" + "---|" * (len(F_ARMS) * len(THREADS) + 1)]
    for w in WORKLOADS:
        cells = []
        for b in [f"{w}-{a}-{t}" for a in F_ARMS for t in THREADS] + [f"{w}-EA"]:
            v = [r["ru_maxrss_mb"] for r in by.get(b, {"ok": []})["ok"]]
            cells.append(f"{statistics.median(v):.0f}" if v else "—")
        lines.append(f"| {w} | " + " | ".join(cells) + " |")
    return lines


def split_table(by: dict, configs: list[str]) -> list[str]:
    keys = ["build", "k1", "k1.pre", "k2.cand_gen", "k2.count", "k2.other", "k3+.cand_gen", "k3+.count",
            "k3+.other", "tail"]
    sub = ["k2.len_filter", "k2.density", "k2.to_csr", "k2.k2_sparse", "k2.matmul", "k2.polars_count",
           "k2.polars_collect", "k3+.len_filter", "k3+.density", "k3+.to_csr", "k3+.kgt2", "k3+.rust_call",
           "k3+.polars_count", "k3+.polars_collect"]
    lines = ["| config | wall | " + " | ".join(keys) + " | count detail |", "|---|---|" + "---|" * (len(keys) + 1)]
    for w in WORKLOADS:
        for c in configs:
            r = median_row(by.get(f"{w}-{c}", {"ok": []})["ok"])
            if r is None:
                continue
            s = split_summary(r)
            if c == "EA":
                detail = (f"index {s.get('index', 0):.2f}; K=2 gen {s.get('k2.cand_gen', 0):.2f} count "
                          f"{s.get('k2.count', 0):.2f}; K≥3 gen {s.get('k3+.cand_gen', 0):.2f} count "
                          f"{s.get('k3+.count', 0):.2f}")
                lines.append(f"| {w}-EA | {r['wall_s']:.2f} | " + " | ".join("" for _ in keys) + f" | {detail} |")
                continue
            detail = ", ".join(f"{k} {s[k]:.2f}" for k in sub if s.get(k, 0) >= 0.01)
            lines.append(f"| {w}-{c} | {r['wall_s']:.2f} | " + " | ".join(f"{s.get(k, 0):.2f}" for k in keys)
                         + f" | {detail} |")
    return lines


def levels_table(by: dict, config: str) -> list[str]:
    lines = ["| regime | K | candidates | frequent | level s | cand_gen | count | other |", "|---|---|---|---|---|---|---|---|"]
    for w in WORKLOADS:
        r = median_row(by.get(f"{w}-{config}", {"ok": []})["ok"])
        if r is None:
            continue
        sp = r["timings"].get("split") or {}
        for lv in r["levels"]:
            ph = sp.get(str(lv["k"]), {})
            lines.append(f"| {w} | {lv['k']} | {lv['n_candidates']:,} | {lv['n_frequent']:,} | {lv['ms'] / 1000:.2f} | "
                         f"{ph.get('cand_gen', 0):.2f} | {ph.get('count', 0):.2f} | {ph.get('other', 0):.2f} |")
    return lines


def stakes_tables(stakes: list[dict]) -> list[str]:
    if not stakes:
        return ["(no stakes yet)"]
    by: dict[tuple, list[dict]] = defaultdict(list)
    for s in stakes:
        by[(s["workload"], s["threads"])].append(s)
    lines = ["| workload | T | L2 numpy_map | L2 explode_join | L1 gram_sparse | L1 gram_dense | L1 polars_pairs | "
             "L4 array Σ | L4 pyref Σ | L3 rust_simd Σ | L3 bitvec Σ | L3 bitvec_lut Σ | L3 proj Σ | L3 gbitvec Σ | "
             "rust_full | lattice = campaign |",
             "|---|---|" + "---|" * 14]
    for (w, t), rows in by.items():
        def tot(lever, variant):
            v = [r["s"] for r in rows if r["lever"] == lever and r["variant"] == variant and "s" in r]
            return f"{sum(v):.3f}" if v else "—"
        summ = next((r for r in rows if r["lever"] == "summary"), {})
        lines.append(f"| {w} | {t} | {tot('L2', 'numpy_map')} | {tot('L2', 'explode_join')} | {tot('L1', 'gram_sparse')} | "
                     f"{tot('L1', 'gram_dense')} | {tot('L1', 'polars_pairs')} | {tot('L4', 'array')} | "
                     f"{tot('L4', 'pyref')} | {tot('L3', 'rust_simd')} | {tot('L3', 'bitvec')} | "
                     f"{tot('L3', 'bitvec_lut')} | {tot('L3', 'proj')} | {tot('L3', 'gbitvec')} | "
                     f"{tot('ref', 'rust_full')} | {summ.get('campaign_matches')} |")
    return lines


def main() -> int:
    d = Path(sys.argv[1])
    by = collect(_rows(d / "raw.jsonl"))
    stakes = _rows(d / "stakes.jsonl")
    out = ["# CPU-tier baseline: generated tables", "",
           "Generated by `bench/cpu/report.py` from `raw.jsonl` and `stakes.jsonl`. Seconds; median [min, max].", "",
           "## Wall time per regime", "", *baseline_table(by), "",
           "## Ratio to efficient-apriori (median / EA median)", "", *ratio_table(by), "",
           "## Peak RSS (ru_maxrss, MB, median)", "", *memory_table(by), "",
           "## Where the time goes (median rep; phase seconds summed per band)", "",
           *split_table(by, ["F-polars-T1", "F-polars-T4", "F-sparse-T1", "F-sparse-T4", "EA"]), "",
           "## Levels (F-sparse-T1, median rep)", "", *levels_table(by, "F-sparse-T1"), "",
           "## Levels (F-polars-T1, median rep)", "", *levels_table(by, "F-polars-T1"), "",
           "## Stakes (seconds summed over levels)", "", *stakes_tables(stakes), ""]
    (d / "report.md").write_text("\n".join(out) + "\n")
    print(f"wrote {d / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
