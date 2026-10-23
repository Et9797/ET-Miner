"""Check every CPU value in the paper against this run's artifacts and write PAPER_VALUES.md.

Recomputes each CPU-side value that paper/et_miner_proteome.tex reports (Section "Reproduction on a
CPU", Table tab:cpu, and the sentences in the abstract, introduction, null model, discussion and
conclusion that cite it) from env.json, data/phase1_verification.json, the run metas, the compare
reports, the null verdict and permutation metas, the A1 checks and the SON logs; reads the GPU times
it sets them against from the GPU run's own JSONs and null-model log; formats each value as the tex
prints it and requires that exact string in the tex. It also requires the scope and hardware wording
(full-data tables, no discrete GPU) and rejects the phrasings that overstate them. Writes
RUN_DIR/PAPER_VALUES.md (one row per value: tex string, source) and exits 1 when a string is missing,
a rejected phrasing occurs, or a verdict is not identical.

Usage:
    python3 paper_values.py RUN_DIR

Options: none.
"""

import json
import re
import statistics
import sys
from pathlib import Path

GPU_CAMPAIGN = "phase3/2026_01/single_gpu/exp/experiment_full_campaign_20260902_200731.json"
GPU_MINC4 = "phase3/2026_01/minc4/mining_meta_4.5519013974922534e-08_20260902_110856.json"
GPU_SON = {"S1": "phase3/2026_01/exp/son_base.json", "S2": "phase3/2026_01/exp/son_super.json"}
GPU_DIRECT_SON = "phase3/2026_01/single_gpu/exp/experiment_direct_vs_son_20260902_215108.json"
GPU_NULL_JSON = "phase3/2026_01/exp_null100/experiment_null_model_20260902_115339.json"
GPU_NULL_LOG = "phase3/2026_01/null_model_769_100.log"

TABLE = [
    ("Base", "C1", ("campaign", "Base"), "s"),
    ("Super", "C2", ("campaign", "Super"), "s"),
    ("Power", "C3", ("campaign", "Power"), "s"),
    ("Power, CI", "C3G", ("campaign", "Power"), "s"),
    ("Blitz, CI", "C4", ("campaign", "Blitz"), "min"),
    ("Ultra, CI", "C5", ("campaign", "Ultra"), "min"),
    ("Opus, CI", "C6", ("campaign", "Opus"), "min"),
    ("4 proteins, CI", "M4", ("minc4", None), "min"),
    ("Base", "S1", ("son", "S1"), "s"),
    ("Super", "S2", ("son", "S2"), "s"),
    ("Power", "S3", ("direct_son", None), "s"),
    ("Power, 4 chunks", "X1", None, None),
]

REQUIRED_WORDING = [
    (
        "abstract scope",
        "reproduced every full-data itemset table the GPU run saved, itemset for itemset and count "
        "for count, and the itemset counts and $K$-distributions of its other full-data mining runs",
    ),
    (
        "introduction scope",
        "reproduced every saved full-data itemset table exactly and the itemset counts of every "
        "other mining run on the full data, and matched the null model as a distribution",
    ),
    ("conclusion scope", "reproduced every full-data itemset table the GPU run saved"),
    (
        "data availability scope",
        "The complete GPU run that produced every value in this paper outside Section~\\ref{sec:cpu-repro}",
    ),
]
REJECTED_WORDING = [
    "without a GPU,",
    "without a GPU (",
    "without a GPU:",
    "no GPU)",
    "of its other runs",
    "every other run (",
    "The complete run that produced every value",
    "reproduced every mining run",
    "Independent Reproduction",
]


def load(p):
    return json.loads(Path(p).read_text())


def reps(run, i):
    """Metas of run i and its repetitions (i_r2, i_r3)."""
    return [
        load(d / "meta.json")
        for d in sorted((run / "data/mined").glob(f"{i}*"))
        if re.fullmatch(rf"{i}(_r\d)?", d.name)
    ]


def gpu_time(gpu, spec):
    """GPU seconds of the run the paper sets against a CPU row, read from the GPU run's JSONs."""
    kind, key = spec
    if kind == "campaign":
        return load(gpu / GPU_CAMPAIGN)["thresholds"][key]["runs"][0]["time_seconds"]
    if kind == "minc4":
        return load(gpu / GPU_MINC4)["duration_seconds"]
    if kind == "son":
        return load(gpu / GPU_SON[key])["time_seconds"]
    return load(gpu / GPU_DIRECT_SON)["son_runs"][0]["time_seconds"]


def t_cpu(s):
    return f"{s:,.1f}\\,s" if s < 900 else f"{s / 3600:.2f}\\,h"


def t_gpu(s, unit):
    return f"{s / 60:,.1f}\\,min" if unit == "min" else f"{s:,.1f}\\,s"


def ratio(r):
    return f"{r:.0f}$\\times$" if r >= 100 else f"{r:.1f}$\\times$" if r >= 1 else f"{r:.2f}$\\times$"


def gib(mb):
    return f"{mb / 1024:.1f}\\,GiB"


class Ledger:
    """Expected tex strings with their sources; collects what the tex lacks or wrongly contains."""

    def __init__(self, tex):
        self.tex, self.rows, self.failed = tex, [], []

    def add(self, what, s, source):
        ok = s in self.tex
        self.rows.append((what, s, source, ok))
        if not ok:
            self.failed.append(f"missing in tex: {what}: {s!r}")

    def reject(self, s):
        if s in self.tex:
            self.failed.append(f"rejected phrasing in tex: {s!r}")

    def require(self, cond, msg):
        if not cond:
            self.failed.append(msg)


def main():
    run = Path(sys.argv[1]).resolve()
    gpu = run.parent / "20260902T0000Z"
    tex_path = run.parents[1] / "paper/et_miner_proteome.tex"
    L = Ledger(tex_path.read_text())
    for what, s in REQUIRED_WORDING:
        L.add(what, s, "scope of the comparison (CI-002: G0 not comparable; null compared as a distribution)")
    for s in REJECTED_WORDING:
        L.reject(s)

    env = load(run / "env.json")
    eng = env["engine"]
    L.add("CPU model", env["cpu_model"].split(" with ")[0], "env.json cpu_model")
    L.require(eng["CUDA_VISIBLE_DEVICES"] == "" and not eng["dev_nvidia"], "a CUDA device was visible")
    if "Graphics" in env["cpu_model"]:
        L.add("no discrete GPU", "without a discrete GPU", "env.json cpu_model (APU), gpu")
        L.add(
            "integrated graphics, caption",
            "no CUDA device; the processor's integrated graphics unused",
            "env.json cpu_model, engine.CUDA_VISIBLE_DEVICES, engine.dev_nvidia",
        )
        L.add(
            "integrated graphics, text", "its integrated graphics were not used and no CUDA device was present", "same"
        )
    L.add("cores, threads", f"{env['physical_cores']} cores, {env['nproc']} threads", "env.json")
    L.add("memory", f"{env['mem_total_gib']}\\,GiB of memory", "env.json mem_total_gib")
    L.add("OS", "Ubuntu~" + env["os"].split()[1], "env.json os")
    L.add("Python", f"Python~{eng['python']}", "env.json engine.python")
    L.add("Polars", f"Polars~{eng['polars']}", "env.json engine.polars")
    L.add("NumPy", f"NumPy~{eng['numpy']}", "env.json engine.numpy")
    L.add("engine commit", f"commit \\texttt{{{eng['sha'][:7]}}}", "env.json engine.sha (CPU-001)")
    L.require(env["THREADS"] == 6 and set(eng["pins"].values()) == {"6"}, "threads not pinned to 6")
    L.add("threads", "pinned to six threads", "env.json THREADS, engine.pins")

    p1 = load(run / "data/phase1_verification.json")
    checks = {c["id"]: c for c in p1["checks"]}
    n_ok = sum(1 for c in p1["checks"] if c["match"])
    L.require(not p1["required_failed"] and not p1["data_drift"], "phase 1 verification has failures")
    other = sorted(c["id"] for c in p1["checks"] if not c["match"])
    expected = [
        "M-006 sha256",
        "mapping ids",
        "sha256 transactions_214m_base.parquet",
        "sha256 transactions_214m_base_multi.parquet",
    ]
    L.require(other == expected, f"unexpected non-matching checks {other}")
    L.add(
        "data checks",
        f"Of the {len(p1['checks'])} checks of the rebuilt data against the September run, {n_ok} agreed",
        "data/phase1_verification.json (CPU-021)",
    )
    g0 = load(run / "data/mined/G0/G0.json")
    L.require(not g0["row_order_equals_gpu"], "G0: row order")
    L.add(
        "row order",
        f"yield {g0['itemsets']['incore']:,} itemsets at a threshold of {g0['min_count']} proteins, "
        f"the September file's {g0['gpu_total']:,}",
        "data/mined/G0/G0.json (CPU-023, CI-002)",
    )
    L.add("metadata rows", f"({checks['M-001']['fresh']:,} entries)", "phase1 M-001")
    L.add("pLDDT-passing", f"{checks['M-003']['fresh']:,} pLDDT-passing", "phase1 M-003")
    L.add("multi-feature", f"{checks['X-011']['fresh']:,} multi-feature", "phase1 X-011")
    L.add("non-zero", f"{checks['X-014']['fresh'][0]:,} non-zero", "phase1 X-014")
    L.add("frequent items", f"{checks['X-015']['fresh'][1]:,} frequent items", "phase1 X-015")
    L.require(checks["stats.items_per_txn_histogram"]["match"], "items-per-transaction histogram differs")
    L.require(len(p1["mapping_moved"]) == 2, "item mapping: expected one swapped pair")
    log = (run / "logs/stream_trembl_2026_01_progress.log").read_text()
    end = re.findall(r"^end .*copied=(\d+) consumed=\d+ full=True md5=([0-9a-f]{32})", log, re.M)[-1]
    L.require(
        end[1] in (run / "data/uniprot_meta/knowledgebase_RELEASE.metalink").read_text(),
        "archive md5 not in UniProt's RELEASE.metalink",
    )
    L.add(
        "flat file size",
        f"{int(end[0]) / 2**30:.1f}\\,GiB compressed UniProt TrEMBL",
        "logs/stream_trembl_2026_01_progress.log (copied = the TrEMBL member, CPU-010)",
    )
    L.add(
        "archive md5",
        "the archive's MD5 checksum equals the one UniProt publishes",
        "same log + data/uniprot_meta/knowledgebase_RELEASE.metalink",
    )

    rows = {r["run"].split()[0]: r for r in load(run / "CPU_COMPARISON_rows.json")["runs"]}
    sec, gsec = {}, {}
    for label, i, spec, unit in TABLE:
        ms = reps(run, i)
        m = ms[0]
        cmps = sorted((run / "data/mined" / i).glob("cmp_*.json"))
        L.require(bool(cmps), f"{i}: no compare report")
        for f in cmps:
            c = load(f)
            L.require(c["verdict"] == "identical" and c["n_differing"] == 0, f"{f.name} of {i} is not identical")
        cpu = sec[i] = statistics.median(x["load_s"] + x["mine_wall_s"] for x in ms)
        L.require(f"{cpu:,.1f} s" in rows[i]["cpu_wall"], f"{i}: wall differs from CPU_COMPARISON")
        if spec:
            g = gsec[i] = gpu_time(gpu, spec)
            gpu_cell, rat = t_gpu(g, unit), ratio(cpu / g)
        else:
            gpu_cell = rat = "---"
        peak = max(x["process"]["peak_rss_mb"] for x in ms)
        cells = [
            label,
            f"{m['min_count']:,}",
            f"{m['signature']['itemsets']:,}",
            str(m["k_max"]),
            t_cpu(cpu),
            gpu_cell,
            rat,
            gib(peak),
        ]
        L.add(
            f"table row {i}",
            " & ".join(cells) + " \\\\",
            f"data/mined/{i}*/meta.json, cmp_*.json; GPU: {spec[0] if spec else '-'} JSON",
        )
    s3, x1 = load(run / "data/mined/S3/meta.json"), load(run / "data/mined/X1/meta.json")
    L.add(
        "SON chunking",
        f"two chunks of at most {s3['chunk_size'] / 1e6:.0f} million transactions; four of at most "
        f"{x1['chunk_size'] / 1e6:.1f} million",
        "data/mined/{S3,X1}/meta.json",
    )
    L.require((s3["chunks"], x1["chunks"]) == (2, 4), "chunk counts")

    L.add(
        "count inference at Power",
        f"in {sec['C3G']:,.1f}\\,s instead of {sec['C3']:,.1f}\\,s",
        "C3G, C3 meta (CPU-042)",
    )
    no_ci = [sec[i] / gsec[i] for i in ("C1", "C2", "C3")]
    ci = [sec[i] / gsec[i] for i in ("C3G", "C4", "C5", "C6", "M4")]
    L.add(
        "ratio ranges",
        f"it took {ratio(min(no_ci)).split('$')[0]}--{ratio(max(no_ci))} the GPU's time without "
        f"count inference and {ratio(min(ci)).split('$')[0]}--{ratio(max(ci))} with it",
        "table rows",
    )
    L.add("discussion, Power gap", f"the gap reached {ratio(sec['C3'] / gsec['C3'])} at Power", "table row C3")
    call = {
        i: max(x["process"]["peak_rss_mb"] for x in reps(run, i)) for i in ("C1", "C2", "C3", "C4", "C5", "C6", "M4")
    }
    L.add(
        "memory during mining",
        f"at most {gib(max(call[i] for i in call if i != 'M4'))} of memory for the six "
        f"campaign thresholds and {gib(call['M4'])} at 4 proteins",
        "data/mined/C1-C6,M4/meta.json process.peak_rss_mb",
    )
    L.add("memory, discussion", f"at most {gib(max(call.values()))} of memory", "same")
    m4 = load(run / "data/mined/M4/meta.json")
    L.add(
        "memory, 4 proteins after the call",
        f"reached {gib(m4['process_end']['peak_rss_mb'])} afterwards",
        "data/mined/M4/meta.json process_end.peak_rss_mb (CPU-060)",
    )
    L.add(
        "count inference saving", f"alone shortened the CPU run {sec['C3'] / sec['C3G']:.1f}$\\times$", "C3/C3G walls"
    )
    dvs = load(gpu / GPU_DIRECT_SON)
    sp = load(run / "CPU_COMPARISON_rows.json")["son_pass_power"]
    L.add(
        "GPU SON chunk pass",
        f"chunk pass alone taking {sp['gpu_pass1_s'] / dvs['direct_gpu_runs'][0]['time_seconds']:.0f}"
        "$\\times$ the direct run",
        "P-029: son_pass_power.gpu_pass1_s (direct_vs_son_1gpu.log) / direct_vs_son JSON",
    )
    son = [sec[s] / sec[c] for s, c in (("S1", "C1"), ("S2", "C2"), ("S3", "C3"))]
    L.add(
        "SON / in-core",
        ", ".join(f"{r:.2f}$\\times$" for r in son[:2]) + f" and {son[2]:.2f}$\\times$ the in-core time",
        "S1/C1, S2/C2, S3/C3 walls of the table",
    )
    L.add("SON / in-core range", f"{min(son):.2f}--{max(son):.2f}$\\times$", "same")
    L.add("SON 4 chunks", f"{sec['X1'] / sec['C3']:.2f}$\\times$ with four chunks", "X1/C3 walls of the table")
    pass1 = re.search(
        r"Pass 1 complete: (\d+) candidate itemsets from \d+ items; (\d+) within the partition bound, "
        r"(\d+) of them exact",
        next((run / "logs").glob("S3_2*.log")).read_text(),
    )
    n_cand, bound, exact = map(int, pass1.groups())
    L.add(
        "SON pass 2",
        f"count {bound - exact} of the {n_cand:,} candidates and took {s3['pass2_s']:.1f}\\,s",
        "logs/S3_*.log pass-1 line, data/mined/S3/meta.json pass2_s (CPU-039)",
    )

    v = load(run / "data/mined/N/verdict_n100.json")
    L.require(
        v["verdict"] == "consistent"
        and v["n_cpu"] == 100
        and set(v["k_max_per_perm"]) == {6}
        and not any(v["perms_reaching_real"].values())
        and not v["perms_with_k_ge_7"],
        "N100 verdict",
    )
    L.add(
        "null totals CPU",
        f"mean {v['cpu_mean']:,.1f} itemsets, range {v['cpu_min']:,}--{v['cpu_max']:,}",
        "data/mined/N/verdict_n100.json (CPU-057)",
    )
    L.add(
        "null totals GPU",
        f"mean {v['gpu_mu']:,.1f}, range {v['gpu_min']:,}--{v['gpu_max']:,}",
        "verdict_n100.json (GPU P-018)",
    )
    L.add("null Mann-Whitney", f"$p{{=}}{v['mannwhitney_p']:.3f}$", "verdict_n100.json")
    perms = [load(p) for p in sorted((run / "data/mined/N").glob("perm_*/meta.json"))]
    L.require(len(perms) == 100, f"{len(perms)} permutation metas")
    gl = re.findall(r"mine=([\d.]+)s, total=([\d.]+)s", (gpu / GPU_NULL_LOG).read_text())
    L.require(len(gl) == 100, f"{len(gl)} GPU permutation lines")
    whole = statistics.median(p["build_s"] + p["mine_wall_s"] for p in perms)
    mine = statistics.median(p["mine_wall_s"] for p in perms)
    L.add(
        "null time",
        f"took {whole / 60:.1f}\\,min on the CPU against {statistics.median(float(t) for _, t in gl):.1f}"
        f"\\,s on the two GPUs, and its mining alone {mine / 60:.1f}\\,min against "
        f"{statistics.median(float(x) for x, _ in gl):.1f}\\,s (medians)",
        f"data/mined/N/perm_*/meta.json (build_s, mine_wall_s); GPU {GPU_NULL_LOG} mine=/total= lines",
    )
    gpu_null = statistics.mean(r["time_seconds"] for r in load(gpu / GPU_NULL_JSON)["null_runs"])
    L.add(
        "GPU null mining time (Section 3.5)",
        f"({gpu_null:.1f}\\,s of mining per permutation)",
        f"{GPU_NULL_JSON} null_runs.time_seconds (P-022; mining only in the row-split branch)",
    )
    L.add(
        "null memory",
        f"peaked at {gib(max(p['process']['peak_rss_mb'] for p in perms))}",
        "data/mined/N/perm_*/meta.json process.peak_rss_mb",
    )
    L.require(load(run / "data/mined/N/verdict_n5.json")["verdict"] == "consistent", "N5 verdict")

    k22 = load(run / "data/mined/A1/k22_check.json")
    L.require(k22["verdict"] == "identical" and all(k22["equal"].values()), "A1 K=22 check")
    cc = {r["id"]: r for r in load(run / "CPU_COMPARISON_rows.json")["rows"]}
    L.require(cc["CC-091"]["verdict"] == "identical", "A1 decode verdict")
    L.add("K=22 itemset", "the same 22 features and the same 8 proteins", "data/mined/A1/k22_check.json (CPU-050)")
    L.add(
        "P-026",
        f"the {load(run / 'data/mined/A1/p026_c6.json')['count']:,} proteins of the DEAD-box triple",
        "data/mined/A1/p026_c6.json (CPU-051)",
    )
    L.add("abstract, Opus", f"mining the Opus threshold took it {sec['C6'] / 3600:.1f}\\,hours", "C6 wall")
    L.add("conclusion, Opus", f"at the Opus threshold in {sec['C6'] / 3600:.1f} hours with count inference", "C6 wall")
    L.add(
        "discussion, Opus",
        f"Opus threshold in {t_cpu(sec['C6'])}, {ratio(sec['C6'] / gsec['C6'])} the GPU's {t_gpu(gsec['C6'], 'min')}",
        "C6 wall, GPU campaign JSON",
    )

    out = [
        "# PAPER_VALUES — CPU values in `paper/et_miner_proteome.tex` and their sources",
        "",
        "Generated by `scripts/paper_values.py`; each tex string is recomputed from the source and must occur "
        "verbatim in the tex. GPU times are read from the GPU run's JSONs and null-model log. Rejected "
        "phrasings (must not occur): " + "; ".join(f"`{s}`" for s in REJECTED_WORDING) + ".",
        "",
        "| value | tex string | source | in tex |",
        "|---|---|---|---|",
    ]
    out += [f"| {w} | `{s.replace('|', '/')}` | {src} | {'yes' if ok else 'NO'} |" for w, s, src, ok in L.rows]
    out += [
        "",
        "The SON / in-core ratios use the walls of the paper's table (medians of three for C1 and S1); "
        "CPU_REPORT.md's SON section uses first runs, hence 1.06 there and 1.09 here at Base.",
        "",
        f"Result: {'pass' if not L.failed else 'FAIL'} ({len(L.rows)} values, {len(REJECTED_WORDING)} rejected "
        "phrasings absent)."
        if not L.failed
        else f"Result: FAIL ({len(L.failed)} problems).",
    ]
    out += [f"- {f}" for f in L.failed]
    (run / "PAPER_VALUES.md").write_text("\n".join(out) + "\n")
    print(
        "\n".join(L.failed)
        or f"pass: {len(L.rows)} values found in {tex_path.name}, {len(REJECTED_WORDING)} rejected phrasings absent"
    )
    sys.exit(1 if L.failed else 0)


if __name__ == "__main__":
    main()
