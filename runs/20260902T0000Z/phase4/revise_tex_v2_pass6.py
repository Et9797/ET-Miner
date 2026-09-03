"""V2 pass 6: SON comparison values from the pinned single-GPU direct-vs-SON run.

Usage:
    python revise_tex_v2_pass6.py <tex> <direct_vs_son.json>
"""
import json, sys

path, dvs_path = sys.argv[1], sys.argv[2]
tex = open(path, encoding="utf-8").read()
orig = tex


def rep(old, new):
    global tex
    n = tex.count(old)
    assert n == 1, f"expected 1 occurrence, found {n}: {old[:90]!r}"
    tex = tex.replace(old, new)


dvs = json.load(open(dvs_path))
d = dvs["direct_gpu_runs"][0]
s = dvs["son_runs"][0]
assert d["itemsets"] == 475865 and s["itemsets"] == 475865 and d["max_k"] == 14 and s["max_k"] == 14
speed = s["time_seconds"] / d["time_seconds"]
ds, ss, sp = f"{d['time_seconds']:.1f}", f"{s['time_seconds']:,.1f}", f"{speed:.0f}"
print("direct", ds, "SON", ss, "speedup", sp)

rep(r"identical count and $K$-distribution at Power; in 779.6\,s, 1,426.0\,s and 5,673.7\,s)",
    f"identical count and $K$-distribution at Power; in 779.6\\,s, 1,426.0\\,s and {ss}\\,s)")
rep(r"in the controlled comparison at the Power threshold the direct path needed 91.7\,s against 5,673.7\,s for SON, a $62\times$ difference (the campaign run of Table~\ref{tab:campaign} is a separate execution of the same direct path, 83.1\,s).",
    f"in the controlled comparison at the Power threshold the direct path needed {ds}\\,s against {ss}\\,s for SON, a ${sp}\\times$ difference (the campaign run of Table~\\ref{{tab:campaign}} is a separate execution of the same direct path, 74.7\\,s).")
rep(r"but $62\times$ more slowly (5,673.7\,s vs.\ 91.7\,s on the RTX 3090), because its global counting pass must re-count every locally frequent candidate against every chunk.",
    f"but ${sp}\\times$ more slowly ({ss}\\,s vs.\\ {ds}\\,s on the RTX 3090): its chunk-local mining pass took 88\\,minutes for the two chunks and the global re-counting pass a further 14\\,minutes.")
assert tex != orig
open(path, "w", encoding="utf-8").write(tex)
print("ok pass6")
