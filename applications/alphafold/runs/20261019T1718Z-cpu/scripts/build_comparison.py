"""Build CPU_COMPARISON.md and CPU_COMPARISON_rows.json from the run's artifacts.

One row per compared value: the GPU row id (repo-root RESULTS.md), the GPU value read from its
artifact, the CPU value read from this run's artifact, the artifact paths and the verdict
(`identical`, `mismatch (...)`, `hardware-comparison`, `consistent` / `inconsistent`,
`not comparable (...)`, `not-run (...)`). Runs without artifacts are skipped; the runs PLAN.md
drops are listed from its "## Not run" section (lines "- ID: reason"). A second table puts the
CPU's per-level counts and times next to the GPU logs' (P-028's single-GPU campaign log, the minc4
log, and P-029's SON pass times). CPU walls are parquet load + engine call (RESULTS rows quote the
engine call alone).

Usage:
    run_py.sh build_comparison.py RUN_DIR GPU_RUN_DIR

Options: none. Writes RUN_DIR/CPU_COMPARISON.md and RUN_DIR/CPU_COMPARISON_rows.json.
"""

import json
import re
import statistics
import sys
from pathlib import Path

N = 76_890_945
CAMPAIGN = "phase3/2026_01/single_gpu/exp/experiment_full_campaign_20260902_200731.json"
CAMPAIGN_2GPU = "phase3/2026_01/exp/experiment_full_campaign_20260902_091611.json"
G0_LOG = "phase3/2026_01/validate_row_split.log"
DIRECT_SON = "phase3/2026_01/single_gpu/exp/experiment_direct_vs_son_20260902_215108.json"
SON = {"S1": "phase3/2026_01/exp/son_base.json", "S2": "phase3/2026_01/exp/son_super.json"}
LEVELS_GPU = "data/sizing/gpu_levels.json"
SON_LOG_1GPU = "phase3/2026_01/single_gpu/direct_vs_son_1gpu.log"
MINC4_META = "phase3/2026_01/minc4/mining_meta_4.5519013974922534e-08_20260902_110856.json"
A1_GPU = "phase3/2026_01/exp/analysis_deepest_itemset_20260902_062222.json"
DECODE_GPU = "phase4/citation_audit/intermediate_k_decode.json"
NULL = {"n5": "phase3/2026_01/exp/experiment_null_model_20260902_070216.json",
        "n100": "phase3/2026_01/exp_null100/experiment_null_model_20260902_115339.json"}
C_SECTION = {"C1": "Base", "C2": "Super", "C3": "Power", "C4": "Blitz", "C5": "Ultra", "C6": "Opus"}
S_OF = {"S1": "C1", "S2": "C2", "S3": "C3"}


def load(p):
    return json.loads(Path(p).read_text())


def kdist(d):
    return {int(k): int(v) for k, v in d.items()}


def walls(run, ids):
    """Return (median, min, max, n) of load + mine wall over the existing repetition dirs."""
    w = [load(p)["load_s"] + load(p)["mine_wall_s"] for i in ids if (p := run / "data/mined" / i / "meta.json").exists()]
    return (statistics.median(w), min(w), max(w), len(w)) if w else None


def peak_mb(m):
    """Peak RSS of the whole process (engine call and table normalization)."""
    return max(m["process"]["peak_rss_mb"], m.get("process_end", {}).get("peak_rss_mb", 0))


def fmt_wall(t):
    med, lo, hi, n = t
    return f"{med:,.1f} s" + (f" (median of {n}, [{lo:,.1f}, {hi:,.1f}])" if n > 1 else " (1 run)")


def main():
    run, gpu = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    rows, runs = [], []

    def run_line(rid, cpu, gpu_val, verdict, cpu_wall, gpu_wall, rss, art):
        runs.append({"run": rid, "cpu": cpu, "gpu": gpu_val, "verdict": verdict, "cpu_wall": cpu_wall,
                     "gpu_wall": gpu_wall, "peak_rss_mb": rss, "artifact": art})

    def add(what, gid, gval, cval, art, verdict):
        rows.append({"id": f"CC-{len(rows) + 1:03d}", "what": what, "gpu_id": gid, "gpu_value": gval,
                     "cpu_value": cval, "artifact": art, "verdict": verdict})

    env = load(run / "env.json")
    add("pinned engine", "—", "—", env["engine"]["sha"], "env.json", "recorded")

    for c in load(run / "data/phase1_verification.json")["checks"]:
        if c["required"]:
            gid = c["id"] if re.match(r"^[MUX]-\d{3}$", c["id"]) else "—"
            add(f"data {c['id']}", gid, c["expected"], c["fresh"], "data/phase1_verification.json",
                "identical" if c["match"] else "mismatch")

    g0 = load(run / "data/mined/G0/G0.json")
    add("G0 itemsets (first 1M rows, min_count 50): in-core / SON / efficient-apriori", "P-001",
        g0["gpu_total"], f"{g0['itemsets']['incore']:,} / {g0['itemsets']['son']:,} / {g0['itemsets']['ea']:,}",
        "data/mined/G0/G0.json",
        "identical (three-way); vs GPU: " + ("identical" if g0["gpu_crosscheck"] == "identical" else g0["gpu_crosscheck"]))
    gm = {k: load(run / "data/mined/G0" / k / "meta.json") for k in ("incore", "son")}
    g0_call = re.findall(r"Mining single-GPU \(n_gpus=1\).*?__main__:main:\d+ - +[\d,]+ itemsets in ([\d.]+)s",
                         (gpu / G0_LOG).read_text(), re.S)
    assert len(g0_call) == 1, g0_call
    g0_k = max(int(k) for k in g0["gpu_k_distribution"])
    run_line("G0", f"{g0['itemsets']['incore']:,} / K{g0['k_max']['incore']}", f"{g0['gpu_total']:,} / K{g0_k} (P-001)",
             "in-core = SON (4 chunks) = efficient-apriori; GPU total not comparable (other rows, CI-002)",
             "in-core " + fmt_wall(walls(run, ["G0/incore", "G0/incore_r2", "G0/incore_r3"]))
             + "; SON " + fmt_wall(walls(run, ["G0/son", "G0/son_r2", "G0/son_r3"])),
             f"{g0_call[0]} s (single-GPU reference call, `{G0_LOG}`)",
             max(peak_mb(m) for m in gm.values()), "data/mined/G0/G0.json")

    camp = load(gpu / CAMPAIGN)["thresholds"]
    for cid, sec in C_SECTION.items():
        meta_p = run / "data/mined" / cid / "meta.json"
        cmp_p = run / "data/mined" / cid / "cmp_gpu_opus.json"
        if not meta_p.exists():
            continue
        m, g = load(meta_p), camp[sec]["runs"][0]
        cmp_ = load(cmp_p)
        ci = " (CPU with count inference)" if m.get("use_generator_pruning") else ""
        art = f"data/mined/{cid}/meta.json, data/mined/{cid}/cmp_gpu_opus.json"
        add(f"{cid} {sec} itemsets / K_max{ci}", "P-028", f"{g['itemsets']:,} / K{g['max_k']}",
            f"{m['signature']['itemsets']:,} / K{m['k_max']}", art,
            cmp_["verdict"] + f" (full outer join vs Opus tables at {camp[sec]['min_count']:,}, {cmp_['n_differing']} differing)")
        g2 = load(gpu / CAMPAIGN_2GPU)["thresholds"][sec]["runs"][0]
        add(f"{cid} {sec} K distribution vs the 2-GPU campaign", "P-014", kdist(g2["k_distribution"]),
            kdist(m["signature"]["k_distribution"]), art + f", {CAMPAIGN_2GPU}",
            "identical" if kdist(g2["k_distribution"]) == kdist(m["signature"]["k_distribution"]) else "mismatch")
        add(f"{cid} {sec} K distribution", "P-028", kdist(g["k_distribution"]), kdist(m["signature"]["k_distribution"]),
            art, "identical" if kdist(g["k_distribution"]) == kdist(m["signature"]["k_distribution"]) else "mismatch")
        add(f"{cid} {sec} wall (CPU: parquet load + apriori call){ci}", "P-028", f"{g['time_seconds']:,.1f} s (one RTX 3090)",
            fmt_wall(walls(run, [cid, f"{cid}_r2", f"{cid}_r3"])) + f"; peak RSS {peak_mb(m):,.0f} MB",
            art, "hardware-comparison")
        run_line(cid + (" (count inference)" if ci else ""), f"{m['signature']['itemsets']:,} / K{m['k_max']}",
                 f"{g['itemsets']:,} / K{g['max_k']} (P-028)",
                 f"{cmp_['verdict']}, {cmp_['n_differing']} differing (Opus tables at {camp[sec]['min_count']:,})",
                 fmt_wall(walls(run, [cid, f"{cid}_r2", f"{cid}_r3"])), f"{g['time_seconds']:,.1f} s (one RTX 3090)",
                 peak_mb(m), art)

    if (run / "data/mined/C4/cmp_gpu_blitz.json").exists():
        c = load(run / "data/mined/C4/cmp_gpu_blitz.json")
        add("C4 Blitz vs the GPU Blitz tables (`blitz/parquet`)", "P-011", f"{c['signature_ref']['itemsets']:,}",
            f"{c['signature_cpu']['itemsets']:,}", "data/mined/C4/cmp_gpu_blitz.json",
            f"{c['verdict']} ({c['n_differing']} differing)")

    if (run / "data/mined/M4/cmp_gpu_minc4.json").exists():
        m, c = load(run / "data/mined/M4/meta.json"), load(run / "data/mined/M4/cmp_gpu_minc4.json")
        g = c["signature_ref"]
        art = "data/mined/M4/meta.json, data/mined/M4/cmp_gpu_minc4.json"
        add("M4 (min_count 4) itemsets / K_max", "P-017", f"{g['itemsets']:,} / K{max(int(k) for k in g['k_distribution'])}",
            f"{m['signature']['itemsets']:,} / K{m['k_max']}", art,
            f"{c['verdict']} (full outer join vs `minc4/parquet`, {c['n_differing']} differing)")
        add("M4 wall (CPU: parquet load + apriori call)", "P-017",
            f"{load(gpu / MINC4_META)['duration_seconds']:,.1f} s (2 GPUs)",
            fmt_wall(walls(run, ["M4"])) + f"; peak RSS {peak_mb(m):,.0f} MB", art, "hardware-comparison")
        lv4 = {x["k"]: x["frequent"] for x in next(v for f, v in load(run / LEVELS_GPU).items()
                                                   if f.endswith("minc4_mining.log"))["all"]}
        add("M4 K distribution vs the GPU log", "P-017", lv4, kdist(m["signature"]["k_distribution"]),
            "data/mined/M4/meta.json, data/sizing/gpu_levels.json (minc4_mining.log)",
            "identical" if lv4 == kdist(m["signature"]["k_distribution"]) else "mismatch")
        run_line("M4 (count inference)", f"{m['signature']['itemsets']:,} / K{m['k_max']}",
                 f"{g['itemsets']:,} / K{max(int(k) for k in g['k_distribution'])} (P-017)",
                 f"{c['verdict']}, {c['n_differing']} differing (`minc4/parquet`)", fmt_wall(walls(run, ["M4"])),
                 f"{load(gpu / MINC4_META)['duration_seconds']:,.1f} s (2 GPUs)", peak_mb(m), art)

    dvs = load(gpu / DIRECT_SON)
    for sid, cid in S_OF.items():
        meta_p = run / "data/mined" / sid / "meta.json"
        if not meta_p.exists():
            continue
        m = load(meta_p)
        if sid in SON:
            g = load(gpu / SON[sid])
            gid, gtot, gk, gt = {"S1": "P-015", "S2": "P-016"}[sid], g["itemsets"], g["max_k"], g["time_seconds"]
            box = "2× RTX 3090 box, `use_gpu=True`; GPUs used not recorded"
            gpu_cmp = run / "data/mined" / sid / "cmp_gpu_son.json"
        else:
            g = dvs["son_runs"][0]
            gid, gtot, gk, gt = "P-029", g["itemsets"], g["max_k"], g["time_seconds"]
            box = "one RTX 3090"
            gpu_cmp = run / "data/mined" / sid / "cmp_gpu_opus.json"
        c_cmp = load(run / "data/mined" / sid / f"cmp_{cid}.json")
        art = f"data/mined/{sid}/meta.json, {gpu_cmp.relative_to(run)}, data/mined/{sid}/cmp_{cid}.json"
        add(f"{sid} SON itemsets / K_max", gid, f"{gtot:,} / K{gk}", f"{m['signature']['itemsets']:,} / K{m['k_max']}", art,
            f"{load(gpu_cmp)['verdict']} vs GPU; {c_cmp['verdict']} vs {cid} (set and counts, P-027)")
        add(f"{sid} SON wall (CPU: parquet load + apriori_streaming call)", gid, f"{gt:,.1f} s ({box})", fmt_wall(walls(run, [sid, f"{sid}_r2", f"{sid}_r3"]))
            + f"; pass 1 {m['pass1_s']:,.1f} s, pass 2 {m['pass2_s']:,.1f} s", art, "hardware-comparison")
        run_line(sid, f"{m['signature']['itemsets']:,} / K{m['k_max']}", f"{gtot:,} / K{gk} ({gid})",
                 f"{load(gpu_cmp)['verdict']} vs GPU, {c_cmp['verdict']} vs {cid}",
                 fmt_wall(walls(run, [sid, f"{sid}_r2", f"{sid}_r3"])), f"{gt:,.1f} s ({box})",
                 peak_mb(m), art)
    if (run / "data/mined/S3/meta.json").exists():
        s3, c3 = walls(run, ["S3"])[0], walls(run, ["C3"])[0]
        add("SON / in-core wall ratio at Power", "P-029", f"{dvs['aggregate']['speedup']}×",
            f"{s3 / c3:.2f}× ({s3:,.1f} / {c3:,.1f} s)", "data/mined/S3/meta.json, data/mined/C3/meta.json",
            "hardware-comparison")

    if (run / "data/mined/X1/cmp_C3.json").exists():
        m, c = load(run / "data/mined/X1/meta.json"), load(run / "data/mined/X1/cmp_C3.json")
        add(f"X1 SON at Power, {m['chunks']} chunks of {m['chunk_size']:,}", "—", "—",
            f"{m['signature']['itemsets']:,} / K{m['k_max']}; wall {fmt_wall(walls(run, ['X1']))} vs S3 "
            f"{walls(run, ['S3'])[0]:,.1f} s", "data/mined/X1/meta.json, data/mined/X1/cmp_C3.json",
            f"{c['verdict']} vs C3 (set and counts)")
        run_line(f"X1 ({m['chunks']} chunks)", f"{m['signature']['itemsets']:,} / K{m['k_max']}", "— (no GPU run)",
                 f"{c['verdict']} vs C3", fmt_wall(walls(run, ["X1"])), "—", peak_mb(m),
                 "data/mined/X1/meta.json, data/mined/X1/cmp_C3.json")

    a1 = run / "data/mined/A1"
    if (a1 / "k22_check.json").exists():
        k = load(a1 / "k22_check.json")
        add("A1 K=22 itemset: features, supporting proteins, GO parent-child pairs", "P-007..P-009",
            f"K{k['gpu']['k']}; {k['gpu']['n_proteins']} proteins; {k['gpu']['n_parent_child_pairs']} pairs",
            f"K{k['cpu']['k']}; {k['cpu']['n_proteins']} proteins; {k['cpu']['n_parent_child_pairs']} pairs",
            "data/mined/A1/k22_check.json", k["verdict"] + " (feature and protein sets compared)")
    if (a1 / "intermediate_k_decode.json").exists():
        def norm(d):
            return {k: (r["itemsets"], r["level_max"], sorted(r["level_max_members"]),
                        {lab: (q["matching_itemsets"], q["max"], sorted(q["max_members"] or []))
                         for lab, q in r["patterns"].items()}) for k, r in d.items()}
        gd, cd = load(gpu / DECODE_GPU), load(a1 / "intermediate_k_decode.json")
        add("A1 Blitz decode: per-level counts, maxima, member sets, patterns (K=11, 12, 13, 17, 19)", "P-030/P-031",
            ", ".join(f"K{k} {r['level_max']:,}" for k, r in gd.items()),
            ", ".join(f"K{k} {r['level_max']:,}" for k, r in cd.items()), "data/mined/A1/intermediate_k_decode.json",
            "identical" if norm(gd) == norm(cd) else "mismatch")
    p026 = a1 / "p026_c6.json" if (a1 / "p026_c6.json").exists() else a1 / "p026.json"
    if p026.exists():
        import polars as pl
        names = dict(pl.read_parquet(gpu / "phase2/extract_2026_01/item_mapping_214m_base.parquet")
                     .select("feature_name", "item_id").iter_rows())
        tgt = sorted(names[f] for f in ("PF00270", "GO:0003724", "GO:0045087"))
        hit = pl.read_parquet(gpu / "phase3/2026_01/opus/parquet/frequent_k3.parquet").filter(
            pl.col("itemset").cast(pl.List(pl.Int64)).list.sort() == pl.lit(tgt, dtype=pl.List(pl.Int64)))
        gcount = round(hit["support"][0] * N) if hit.height else None
        c = load(p026)
        add("P-026 support of {PF00270, GO:0003724, GO:0045087}", "P-026", f"{gcount:,}", f"{c['count']:,}",
            str(p026.relative_to(run)), "identical" if gcount == c["count"] else "mismatch")
    if (a1 / "k22_check.json").exists() and (a1 / "intermediate_k_decode.json").exists() and p026.exists():
        k = load(a1 / "k22_check.json")
        ok = k["verdict"] == "identical" and norm(gd) == norm(cd) and gcount == c["count"]
        run_line("A1", f"K=22 itemset with {k['cpu']['n_proteins']} proteins; Blitz decode; P-026 {c['count']:,}",
                 f"K=22, {k['gpu']['n_proteins']} proteins; decode; P-026 {gcount:,} (P-007..P-009, P-026, P-030, P-031)",
                 "identical" if ok else "mismatch",
                 f"{load(k['cpu_json'])['total_time_seconds']:,.1f} s (K=22 analysis)",
                 f"{load(k['gpu_json'])['total_time_seconds']:,.1f} s (K=22 analysis)", 0,
                 "data/mined/A1/k22_check.json, data/mined/A1/intermediate_k_decode.json, " + str(p026.relative_to(run)))

    if (run / "data/mined/R1/meta.json").exists():
        r, s = load(run / "data/mined/R1/meta.json"), load(gpu / SON["S1"])
        adj = load(run / "data/mined/R1/adjudication.json")["counts"]
        add("R1 rules (min_confidence 0.5) all / lift ≥ 5 / lift ≥ 100", "P-015",
            f"{s['n_rules']:,} / {s['n_rules_lift_ge5']:,} / {s['n_rules_lift_ge100']:,}",
            f"{r['n_rules']:,} / {r['n_rules_lift_ge5']:,} / {r['n_rules_lift_ge100']:,}",
            "data/mined/R1/meta.json, data/mined/R1/adjudication.json",
            "mismatch (adjudicated for the CPU: exact enumeration "
            f"{adj['exact_cpu']['n_rules']:,}; the branch generator on the GPU's non-ascending SON table gives "
            f"{adj['branch_gpu_stored']['n_rules']:,}; INCONSISTENCIES CI-001)")
        add("R1 max lift", "P-015", f"{s['rules_max']['lift']:.4f}", f"{r['max_lift']:.4f}", "data/mined/R1/meta.json",
            "identical" if round(s["rules_max"]["lift"], 6) == round(r["max_lift"], 6) else "mismatch")
        add("R1 cross-domain rules (lhs ∪ rhs spans > 1 feature category)", "P-015", f"{s['n_rules_cross_domain']:,}",
            f"{adj['exact_cpu']['cross_domain']:,}", "data/mined/R1/adjudication.json",
            f"mismatch (adjudicated for the CPU; the branch generator gives {adj['branch_gpu_stored']['cross_domain']:,})")
        rw = [load(run / "data/mined" / i / "meta.json")["generate_s"] for i in ("R1", "R1_r2", "R1_r3")
              if (run / "data/mined" / i / "meta.json").exists()]
        run_line("R1", f"{r['n_rules']:,} rules ({r['n_rules_lift_ge5']:,} lift ≥ 5, {r['n_rules_lift_ge100']:,} ≥ 100)",
                 f"{s['n_rules']:,} ({s['n_rules_lift_ge5']:,}, {s['n_rules_lift_ge100']:,}) (P-015)",
                 "mismatch, adjudicated for the CPU by exact enumeration (CI-001)",
                 f"{statistics.median(rw):.3f} s (median of {len(rw)})", f"{s['rules_time_seconds']} s",
                 r["process"]["peak_rss_mb"], "data/mined/R1/meta.json, data/mined/R1/adjudication.json")

    for mode in ("n5", "n100"):
        vp = run / "data/mined/N" / f"verdict_{mode}.json"
        if not vp.exists():
            continue
        v, g = load(vp), load(gpu / NULL[mode])
        gt = [x["total_itemsets"] for x in g["null_runs"]]
        gkmax = sorted({max(int(k) for k in x["k_distribution"]) for x in g["null_runs"]})
        add(f"null {mode.upper()} mean total (K_max per permutation)", {"n5": "P-012", "n100": "P-018"}[mode],
            f"{statistics.mean(gt):,.1f} (K_max {gkmax} over {len(gt)})",
            f"{v['cpu_mean']:,.1f} (K_max {sorted(set(v['k_max_per_perm']))})", str(vp.relative_to(run)),
            f"{v['verdict']} (criteria: {', '.join(k for k, ok in v['criteria'].items() if ok)}"
            + (f"; failed: {', '.join(k for k, ok in v['criteria'].items() if not ok)}" if v["verdict"] != "consistent" else "")
            + ")")
        ptimes = [load(run / "data/mined/N" / f"perm_{i:03d}" / "meta.json") for i in range(v["n_cpu"])]
        km = sorted(set(v["k_max_per_perm"]))
        run_line(mode.upper(), f"mean {v['cpu_mean']:,.1f} over {v['n_cpu']} permutations, K_max {km} in all",
                 f"mean {statistics.mean(gt):,.1f} over {len(gt)} ({ {'n5': 'P-012', 'n100': 'P-018'}[mode]})",
                 v["verdict"], f"{statistics.median(m['build_s'] + m['mine_wall_s'] for m in ptimes):,.1f} s per permutation (median)",
                 f"{statistics.mean(x['time_seconds'] for x in g['null_runs']):,.2f} s per permutation (2 GPUs)",
                 max(peak_mb(m) for m in ptimes), str(vp.relative_to(run)))

    if (run / "data/mined/P1T/meta.json").exists():
        m = load(run / "data/mined/P1T/meta.json")
        run_line("P1T (sizing: SON pass-1 chunk at Base)", f"{m['signature']['itemsets']:,} / K{m['k_max']}",
                 "— (sizing only)", "not compared (first 40,000,000 rows at local min_count 36,000)",
                 fmt_wall(walls(run, ["P1T", "P1T_r2", "P1T_r3"])), "—", peak_mb(m), "data/mined/P1T/meta.json")

    if (run / "data/mined/C3G/cmp_C3.json").exists():
        c3g = load(run / "data/mined/C3G/cmp_C3.json")
        add("C3G (C3 with use_generator_pruning) vs C3", "—", "—",
            f"{load(run / 'data/mined/C3G/meta.json')['signature']['itemsets']:,} itemsets; wall "
            f"{walls(run, ['C3G'])[0]:,.1f} s vs {walls(run, ['C3'])[0]:,.1f} s", "data/mined/C3G/cmp_C3.json",
            c3g["verdict"])
        m = load(run / "data/mined/C3G/meta.json")
        run_line("C3G (C3 with count inference)", f"{m['signature']['itemsets']:,} / K{m['k_max']}", "— (fallback check)",
                 f"{c3g['verdict']} vs C3", fmt_wall(walls(run, ["C3G"])), "—", peak_mb(m),
                 "data/mined/C3G/meta.json, data/mined/C3G/cmp_C3.json")

    plan = (run / "PLAN.md").read_text()
    sec = plan.split("## Not run", 1)[1] if "## Not run" in plan else ""
    for mm in re.finditer(r"^- (\w+): (.+)$", sec, re.M):
        add(f"{mm.group(1)}", "—", "—", "—", "PLAN.md", f"not-run ({mm.group(2).strip()})")

    levels = []
    gl = load(run / LEVELS_GPU)
    camp_lv = next(v for f, v in gl.items() if f.endswith("full_campaign_1gpu.log"))
    minc4_lv = next(v for f, v in gl.items() if f.endswith("minc4_mining.log"))["all"]
    for cid, glv in [(c, camp_lv[sec]) for c, sec in C_SECTION.items()] + [("M4", minc4_lv)]:
        mp = run / "data/mined" / cid / "meta.json"
        if not mp.exists():
            continue
        g_by_k = {x["k"]: x for x in glv}
        for x in load(mp)["levels"]:
            gx = g_by_k.get(x["k"], {})
            levels.append({"run": cid, "k": x["k"], "cpu_candidates": x["candidates"], "cpu_frequent": x["frequent"],
                           "cpu_s": round(x["ms"] / 1e3, 2), "gpu_candidates": gx.get("candidates"),
                           "gpu_frequent": gx.get("frequent"), "gpu_s": gx.get("s"),
                           "gpu_id": "P-017 (2 GPUs, minc4_mining.log)" if cid == "M4" else "P-028 (single-GPU log)"})
    son_pass = {}
    if (run / "data/mined/S3/meta.json").exists():
        import datetime as dt
        ts = {}
        for line in (gpu / SON_LOG_1GPU).read_text().splitlines():
            for key, pat in (("start", "SON streaming:"), ("p1", "Pass 1 complete"), ("p2", "Pass 2 complete")):
                if pat in line:
                    ts[key] = dt.datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S.%f")
        m3 = load(run / "data/mined/S3/meta.json")
        son_pass = {"gpu_pass1_s": (ts["p1"] - ts["start"]).total_seconds(), "gpu_pass2_s": (ts["p2"] - ts["p1"]).total_seconds(),
                    "cpu_pass1_s": m3["pass1_s"], "cpu_pass2_s": m3["pass2_s"], "gpu_log": SON_LOG_1GPU}

    (run / "CPU_COMPARISON_rows.json").write_text(json.dumps({"rows": rows, "runs": runs, "levels": levels, "son_pass_power": son_pass},
                                                            indent=1, default=str) + "\n")
    md = ["# CPU_COMPARISON — CPU in-core / SON vs the GPU run", "",
          "Built by `scripts/build_comparison.py` from the artifacts named in each row (GPU values read from the "
          f"GPU run's artifacts under `{gpu.relative_to(run.parents[1])}/`, ids from the repo-root RESULTS.md).", "",
          "| row | what | GPU id | GPU value | CPU value | artifact | verdict |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        cell = [str(r[k]).replace("|", "/") for k in ("id", "what", "gpu_id", "gpu_value", "cpu_value", "artifact", "verdict")]
        md.append("| " + " | ".join(cell) + " |")
    if son_pass:
        md += ["", "## SON pass times at Power (hardware-comparison)", "",
               f"GPU P-029 (one RTX 3090, `{SON_LOG_1GPU}` timestamps): pass 1 {son_pass['gpu_pass1_s']:,.1f} s, "
               f"pass 2 {son_pass['gpu_pass2_s']:,.1f} s. CPU S3 (`data/mined/S3/meta.json`): pass 1 "
               f"{son_pass['cpu_pass1_s']:,.1f} s, pass 2 {son_pass['cpu_pass2_s']:,.1f} s."]
    md += ["", "## Per-level counts and times (hardware-comparison for the times)", "",
           "CPU from `data/mined/<run>/meta.json` (`levels`); GPU from `data/sizing/gpu_levels.json` "
           "(`scripts/gpu_levels.py` over the GPU logs). Candidate counts differ by design (the routes generate "
           "candidates differently); frequent counts must agree.", "",
           "| run | K | CPU candidates | CPU frequent | CPU s | GPU candidates | GPU frequent | GPU s | GPU id |",
           "|---|---|---|---|---|---|---|---|---|"]
    for x in levels:
        md.append(f"| {x['run']} | {x['k']} | {x['cpu_candidates']:,} | {x['cpu_frequent']:,} | {x['cpu_s']:,.2f} | "
                  + (f"{x['gpu_candidates']:,} | {x['gpu_frequent']:,} | {x['gpu_s']} | " if x["gpu_frequent"] is not None
                     else "— | — | — | ") + f"{x['gpu_id']} |")
    (run / "CPU_COMPARISON.md").write_text("\n".join(md) + "\n")
    print(f"{len(rows)} rows; verdicts: " + json.dumps({v: sum(r['verdict'].split(' ')[0] == v for r in rows)
                                                        for v in sorted({r['verdict'].split(' ')[0] for r in rows})}))


if __name__ == "__main__":
    main()
