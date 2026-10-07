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


def _stake_index(stakes: list[dict]) -> dict:
    """{(workload, threads): {(lever, variant): {k: seconds}}} over the timed stake rows."""
    idx: dict = defaultdict(lambda: defaultdict(dict))
    for r in stakes:
        if "s" in r:
            idx[(r["workload"], r["threads"])][(r["lever"], r["variant"])][r.get("k")] = r["s"]
    return idx


def _tot(d: dict, key: tuple) -> float | None:
    v = d.get(key)
    return sum(v.values()) if v else None


def stake_costs(idx: dict, w: str, t: int) -> dict | None:
    """New-pipeline seconds per lever at thread setting ``t`` (numpy/scipy variants come from the T1 run).

    L2 = CSR build (faster of numpy_map, explode_join) + CSC + K=1 bincount; L1 = faster Gram;
    L4 = array generation; L3 = per level the fastest counter, and gbitvec alone; L5 = the
    threshold masks + array emission; bitvec_build counted once when a K>=3 level exists.
    """
    s1, st = idx.get((w, 1)), idx.get((w, t))
    if not s1 or not st:
        return None
    def pick(lever: str, names: list[str]) -> float:
        vals = (_tot(st, (lever, n)) or _tot(s1, (lever, n)) for n in names)
        return min((x for x in vals if x is not None), default=0.0)

    out = {"L2": pick("L2", ["numpy_map", "explode_join"]) + (_tot(st, ("L2", "tocsc")) or 0)
           + (_tot(st, ("K1", "bincount")) or 0),
           "L1": pick("L1", ["gram_sparse", "gram_dense"]),
           "L4": _tot(st, ("L4", "array")) or 0.0}
    ks = sorted(set().union(*[set(v) for (lv, _), v in s1.items() if lv == "L3"]) - {None})
    best, gb = 0.0, 0.0
    for k in ks:
        cands = [st.get(("L3", "rust_simd"), {}).get(k)] + [s1.get(("L3", n), {}).get(k)
                                                           for n in ("bitvec", "bitvec_lut", "proj", "gbitvec")]
        cands = [c for c in cands if c is not None]
        best += min(cands) if cands else 0.0
        gb += s1.get(("L3", "gbitvec"), {}).get(k) or 0.0
    bv = _tot(s1, ("L3", "bitvec_build")) or 0.0
    out["L3"] = best + (bv if ks else 0.0)
    out["L3_gbitvec"] = gb + (bv if ks else 0.0)
    out["L5"] = (_tot(st, ("L5", "threshold_mask")) or 0.0) + (_tot(s1, ("L5", "emit_arrays")) or 0.0)
    out["rust_full"] = _tot(st, ("ref", "rust_full"))
    out["total"] = out["L2"] + out["L1"] + out["L4"] + out["L3"] + out["L5"]
    return out


def baseline_parts(r: dict) -> dict:
    """The baseline phases each lever replaces, from one row's split."""
    s = split_summary(r)

    def g(k: str) -> float:
        return s.get(k, 0.0)

    return {
        "L2": g("build") + g("k1") + g("k2.to_csr") + g("k3+.to_csr"),
        "L1": g("k2.cand_gen") + g("k2.count") - g("k2.len_filter") - g("k2.density") - g("k2.to_csr"),
        "L4": g("k3+.cand_gen"),
        "L3": g("k3+.count") - g("k3+.len_filter") - g("k3+.density") - g("k3+.to_csr"),
        "L5": g("k2.len_filter") + g("k3+.len_filter") + g("k2.density") + g("k3+.density") + g("k2.other")
        + g("k3+.other") + g("tail"),
    }


def projection_tables(by: dict, stakes: list[dict]) -> list[str]:
    """Per lever: baseline phase seconds vs stake seconds; per regime: composed projection vs EA."""
    idx = _stake_index(stakes)
    lines = ["### Per lever: replaced baseline phases → stake (seconds; F-auto median rep)", "",
             "| regime | L2 | L1 | L4 | L3 (best / gbitvec only) | L5 |", "|---|---|---|---|---|---|"]
    comp = ["### Composed projection (all five levers) vs baseline and EA", "",
            "| regime | F-polars now | F-auto now | projected | EA | projected / EA | rust_full (ref) |",
            "|---|---|---|---|---|---|---|"]
    for w in WORKLOADS:
        ea = [r["wall_s"] for r in by.get(f"{w}-EA", {"ok": []})["ok"]]
        for t_name, t in THREADS.items():
            c = stake_costs(idx, w, t)
            r = median_row(by.get(f"{w}-F-auto-{t_name}", {"ok": []})["ok"])
            if c is None or r is None:
                continue
            b = baseline_parts(r)
            lines.append(f"| {w} {t_name} | {b['L2']:.2f} → {c['L2']:.2f} | {b['L1']:.2f} → {c['L1']:.2f} | "
                         f"{b['L4']:.2f} → {c['L4']:.2f} | {b['L3']:.2f} → {c['L3']:.2f} / {c['L3_gbitvec']:.2f} | "
                         f"{b['L5']:.2f} → {c['L5']:.2f} |")
            pol = [x["wall_s"] for x in by.get(f"{w}-F-polars-{t_name}", {"ok": []})["ok"]]
            auto = [x["wall_s"] for x in by.get(f"{w}-F-auto-{t_name}", {"ok": []})["ok"]]
            e = statistics.median(ea) if ea else None
            rf = f"{c['rust_full']:.2f}" if c["rust_full"] is not None else "—"
            comp.append(f"| {w} {t_name} | {_cell(pol)} | {_cell(auto)} | {c['total']:.2f} | "
                        f"{_cell(ea)} | {c['total'] / e:.3f} | {rf} |" if e else
                        f"| {w} {t_name} | {_cell(pol)} | {_cell(auto)} | {c['total']:.2f} | — | — | {rf} |")
    return lines + [""] + comp


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
           "## Stakes (seconds summed over levels)", "", *stakes_tables(stakes), "",
           "## Projections", "", *projection_tables(by, stakes), ""]
    (d / "report.md").write_text("\n".join(out) + "\n")
    print(f"wrote {d / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
