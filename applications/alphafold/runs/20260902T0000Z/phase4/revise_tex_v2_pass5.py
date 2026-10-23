"""V2 pass 5: put the pinned single-GPU timings into the tex.

Reads the campaign JSON of the pinned re-run (and, when present, the pinned direct-vs-SON
JSON), formats the values exactly as the paper prints them, and applies exact-string
replacements (each anchor must occur once). Prints the values it used so they can be
copied into RESULTS.md.

Usage:
    python revise_tex_v2_pass5.py <tex> <campaign.json> [direct_vs_son.json]
"""
import json, sys

path, camp_path = sys.argv[1], sys.argv[2]
dvs_path = sys.argv[3] if len(sys.argv) > 3 else None
tex = open(path, encoding="utf-8").read()
orig = tex


def rep(old, new):
    global tex
    n = tex.count(old)
    assert n == 1, f"expected 1 occurrence, found {n}: {old[:90]!r}"
    tex = tex.replace(old, new)


camp = json.load(open(camp_path))
th = camp["thresholds"]
t = {name: th[name]["runs"][0]["time_seconds"] for name in ("Base", "Super", "Power", "Blitz", "Ultra", "Opus")}
it = {name: th[name]["runs"][0]["itemsets"] for name in t}
kmax = {name: th[name]["runs"][0]["max_k"] for name in t}
assert [it[n] for n in t] == [5305, 113405, 475865, 2841280, 14558875, 26849505], it
assert [kmax[n] for n in t] == [9, 14, 14, 19, 20, 22], kmax


def fmt_s(x):
    return f"{x:.1f}\\,s"


def fmt_min(x):
    return f"{x / 60:.1f}\\,min"


opus_min = f"{t['Opus'] / 60:.1f}"
print("campaign times (s):", {k: round(v, 1) for k, v in t.items()}, "total", camp["total_time_seconds"], "| Opus min:", opus_min)

# Table 2 rows (previous values from the mixed-configuration campaign)
rep(r"Base & 0.1\% & 76,891 & 5,305 & 9 & 15.6\,s \\", f"Base & 0.1\\% & 76,891 & 5,305 & 9 & {fmt_s(t['Base'])} \\\\")
rep(r"Super & 0.01\% & 7,690 & 113,405 & 14 & 43.9\,s \\", f"Super & 0.01\\% & 7,690 & 113,405 & 14 & {fmt_s(t['Super'])} \\\\")
rep(r"Power & 0.001\% & 769 & 475,865 & 14 & 83.1\,s \\", f"Power & 0.001\\% & 769 & 475,865 & 14 & {fmt_s(t['Power'])} \\\\")
rep(r"Blitz & 0.0001\% & 77 & 2,841,280 & 19 & 4.6\,min \\", f"Blitz & 0.0001\\% & 77 & 2,841,280 & 19 & {fmt_min(t['Blitz'])} \\\\")
rep(r"Ultra & 0.00002\% & 16 & 14,558,875 & 20 & 11.5\,min \\", f"Ultra & 0.00002\\% & 16 & 14,558,875 & 20 & {fmt_min(t['Ultra'])} \\\\")
rep(r"Opus & 0.00001\% & 8 & 26,849,505 & 22 & 19.1\,min \\", f"Opus & 0.00001\\% & 8 & 26,849,505 & 22 & {fmt_min(t['Opus'])} \\\\")

# headline occurrences of the Opus time
rep(r"in 19.1\,minutes (excluding feature extraction) on a single NVIDIA GeForce RTX 3090, revealing",
    f"in {opus_min}\\,minutes (excluding feature extraction) on a single NVIDIA GeForce RTX 3090, revealing")
rep(r"in 19.1\,minutes of mining on a single NVIDIA GeForce RTX 3090, enabling",
    f"in {opus_min}\\,minutes of mining on a single NVIDIA GeForce RTX 3090, enabling")
rep(r"this preprocessing is separate from the 19.1\,minute mining runtime reported below.",
    f"this preprocessing is separate from the {opus_min}\\,minute mining runtime reported below.")
rep(r"ET-miner & \textbf{76.9M} & 1,002 & \textbf{22} & 1$\times$ RTX 3090 & 19.1\,min \\",
    f"ET-miner & \\textbf{{76.9M}} & 1,002 & \\textbf{{22}} & 1$\\times$ RTX 3090 & {opus_min}\\,min \\\\")
rep(r"mining completes in 19.1 minutes (excluding the one-time feature extraction, Section~\ref{sec:methods}) and reaches $K{=}22$.",
    f"mining completes in {opus_min} minutes (excluding the one-time feature extraction, Section~\\ref{{sec:methods}}) and reaches $K{{=}}22$.")

# setup paragraph: how the single-GPU configuration was enforced and verified
rep(r"The six-run mining campaign and the direct-versus-streaming comparison ran on a single GPU; the permutation null model and the per-level itemset exports used both GPUs with the row-split path.",
    r"The six-run mining campaign and the direct-versus-streaming comparison ran on a single GPU (the second device hidden with \texttt{CUDA\_VISIBLE\_DEVICES=0}; \texttt{nvidia-smi} samples taken every 5\,s show it idle throughout); the permutation null model and the per-level itemset exports used both GPUs with the row-split path.")

if dvs_path:
    dvs = json.load(open(dvs_path))
    d = dvs["direct_gpu_runs"][0]
    s = dvs["son_runs"][0]
    assert d["itemsets"] == 475865 and s["itemsets"] == 475865 and d["max_k"] == 14 and s["max_k"] == 14, (d, s)
    speed = s["time_seconds"] / d["time_seconds"]
    print("direct vs SON (s):", d["time_seconds"], s["time_seconds"], "speedup", round(speed, 2))
    ds, ss = f"{d['time_seconds']:.1f}", f"{s['time_seconds']:,.1f}"
    sp = f"{speed:.0f}"
    rep(r"identical count and $K$-distribution at Power; in 779.6\,s, 1,426.0\,s and 5,673.7\,s)",
        f"identical count and $K$-distribution at Power; in 779.6\\,s, 1,426.0\\,s and {ss}\\,s)")
    rep(r"in the controlled comparison at the Power threshold the direct path needed 91.7\,s against 5,673.7\,s for SON, a $62\times$ difference (the campaign run of Table~\ref{tab:campaign} is a separate execution of the same direct path, 83.1\,s).",
        f"in the controlled comparison at the Power threshold the direct path needed {ds}\\,s against {ss}\\,s for SON, a ${sp}\\times$ difference (the campaign run of Table~\\ref{{tab:campaign}} is a separate execution of the same direct path, {fmt_s(t['Power'])}).")
    rep(r"but $62\times$ more slowly (5,673.7\,s vs.\ 91.7\,s on the RTX 3090)",
        f"but ${sp}\\times$ more slowly ({ss}\\,s vs.\\ {ds}\\,s on the RTX 3090)")

assert tex != orig
open(path, "w", encoding="utf-8").write(tex)
print("ok pass5:", len(orig), "->", len(tex), "chars")
