"""Fill the generated sections of CPU_REPORT.md from CPU_COMPARISON_rows.json and the run's artifacts.

Replaces the text between `<!-- BEGIN:<name> -->` and `<!-- END:<name> -->` for the sections
runs (one line per PLAN.md run), son (SON / in-core wall ratios), null (N5 and N100 verdicts with
their numbers) and notrun (PLAN.md "## Not run"). Run build_comparison.py first.

Usage:
    python3 build_report.py RUN_DIR

Options: none.
"""

import json
import re
import sys
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text())


def wall(run, i):
    m = load(run / "data/mined" / i / "meta.json")
    return m["load_s"] + m["mine_wall_s"]


def runs_table(rows):
    out = ["| run | CPU result | GPU result | verdict | CPU wall | GPU wall | CPU peak RSS | artifact |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        rss = f"{r['peak_rss_mb'] / 1024:,.1f} GiB" if r["peak_rss_mb"] else "—"
        cells = [r["run"], r["cpu"], r["gpu"], r["verdict"], r["cpu_wall"], r["gpu_wall"], rss, f"`{r['artifact']}`"]
        out.append("| " + " | ".join(str(c).replace("|", "/") for c in cells) + " |")
    return "\n".join(out)


def son_section(run, data, gpu):
    dvs = load(gpu / "phase3/2026_01/single_gpu/exp/experiment_direct_vs_son_20260902_215108.json")
    lines = ["| threshold | CPU SON (2 chunks, first run) | CPU in-core (first run) | CPU ratio | GPU ratio |",
             "|---|---|---|---|---|"]
    for s, c, name in (("S1", "C1", "Base"), ("S2", "C2", "Super"), ("S3", "C3", "Power")):
        ws, wc = wall(run, s), wall(run, c)
        g = (f"{dvs['aggregate']['speedup']}× (P-029: {dvs['son_runs'][0]['time_seconds']:,.2f} s vs "
             f"{dvs['direct_gpu_runs'][0]['time_seconds']:,.2f} s, one RTX 3090)") if s == "S3" else "—"
        lines.append(f"| {name} | {ws:,.1f} s | {wc:,.1f} s | {ws / wc:.2f}× | {g} |")
    x1 = wall(run, "X1")
    sp = data.get("son_pass_power") or {}
    lines += ["", f"X1 (Power, 4 chunks of 19,222,737): {x1:,.1f} s, {x1 / wall(run, 'C3'):.2f}× C3 and "
              f"{x1 / wall(run, 'S3'):.2f}× S3 (X1 shared the CPU for a few minutes with low-priority checks, D-015). "
              "On the CPU route SON's pass 1 already returns exact counts for almost "
              "every candidate (\"within the partition bound … exact\" in the engine log), so pass 2 has little or "
              f"nothing to count: at Power pass 1 {sp.get('cpu_pass1_s', 0):,.1f} s and pass 2 "
              f"{sp.get('cpu_pass2_s', 0):,.1f} s, against the GPU's {sp.get('gpu_pass1_s', 0):,.1f} s and "
              f"{sp.get('gpu_pass2_s', 0):,.1f} s (P-029). All timings are hardware-comparison."]
    return "\n".join(lines)


def null_section(run):
    out = []
    for mode, gid in (("n5", "the 100-permutation run P-018; the GPU's own N5 is P-012"), ("n100", "P-018, P-022")):
        v = load(run / "data/mined/N" / f"verdict_{mode}.json")
        crit = ", ".join(f"{k} {'yes' if ok else 'NO'}" for k, ok in v["criteria"].items())
        out.append(
            f"- **{mode.upper()}: {v['verdict']}.** {v['n_cpu']} CPU permutations, totals mean {v['cpu_mean']:,.1f} "
            f"(range {v['cpu_min']:,}–{v['cpu_max']:,}), K_max {sorted(set(v['k_max_per_perm']))} in every one; GPU "
            f"({gid}) μ {v['gpu_mu']:,.2f}, σ {v['gpu_sigma_ddof1']:.2f} (ddof 1), range "
            f"{v['gpu_min']:,}–{v['gpu_max']:,}; 3σ/√5 band [{v['band_3sigma_sqrt5'][0]:,.2f}, "
            f"{v['band_3sigma_sqrt5'][1]:,.2f}]; Mann–Whitney p {v['mannwhitney_p']:.3f}; permutations reaching the "
            f"real count at K = 4/5/6: {sum(len(x) for x in v['perms_reaching_real'].values())}; with K ≥ 7: "
            f"{len(v['perms_with_k_ge_7'])}. Criteria: {crit} (`data/mined/N/verdict_{mode}.json`).")
    out.append("- The CPU shuffles with NumPy and the GPU run with CuPy from the same seeds, so single permutations "
               "differ; the criterion compares them as distributions (brief Phase 4 step 2, fixed before N5 ran).")
    return "\n".join(out)


def notrun_section(run):
    plan = (run / "PLAN.md").read_text()
    sec = plan.split("## Not run", 1)[1].strip() if "## Not run" in plan else ""
    items = re.findall(r"^- (\w+): (.+)$", sec, re.M)
    if not items:
        return ("No run of PLAN.md was dropped: C6 (with A1), M4 and X1 were first dropped on early estimates and "
                "re-admitted after measured runs lowered them (PLAN.md Re-estimates; deviations D-011, approved by Et "
                "in D-014).")
    return "\n".join(f"- {i}: {r}" for i, r in items)


def main():
    run = Path(sys.argv[1]).resolve()
    gpu = run.parent / "20260902T0000Z"
    data = load(run / "CPU_COMPARISON_rows.json")
    parts = {"runs": runs_table(data["runs"]), "son": son_section(run, data, gpu), "null": null_section(run),
             "notrun": notrun_section(run)}
    p = run / "CPU_REPORT.md"
    s = p.read_text()
    for name, body in parts.items():
        pat = re.compile(rf"(<!-- BEGIN:{name} -->\n)(?:.*?\n)?(<!-- END:{name} -->)", re.S)
        assert pat.search(s), f"marker {name} missing"
        s = pat.sub(lambda m, b=body: m.group(1) + b + "\n" + m.group(2), s)
    p.write_text(s)
    print(f"filled {sorted(parts)}")


if __name__ == "__main__":
    main()
