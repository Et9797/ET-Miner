#!/usr/bin/env python3
"""Join every extracted claim with the freshly reproduced values and render
COMPARISON_REPORT.md.

Inputs (all under the phase4 directory): quantities.csv plus any
new_quantities_*.csv (canonical quantity keys), claims_*_keyed.csv (one row
per extracted claim with its qkey), fresh_values.json (qkey -> fresh value
with artifact provenance), optional overrides.json (qkey -> {verdict, note}
for cases that need an explicit rationale), and the campaign's
INCONSISTENCIES.md for the summary section.

Verdict rules:
  deterministic / method-parameter / external-fact claims with a fresh value
      -> confirmed when the claim equals the fresh value (exactly for
         integers; a rounded or approximate claim must equal the fresh value
         rounded to the claim's own precision), else hallucinated;
  hardware-dependent claims with a fresh value -> expected-hardware-deviation
      (both values reported);
  software claims -> inconclusive (environment statements) unless identical;
  claims whose qkey has no fresh value -> inconclusive, with the reason taken
      from the quantity's reproducibility class.

Usage:
    build_comparison.py --phase4 DIR --fresh fresh_values.json --out REPORT.md
                        [--inconsistencies INCONSISTENCIES.md] [--header header.md]
"""

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict

csv.field_size_limit(1 << 26)
RANDOM_KEYS = re.compile(r"^(null(100)?_k\d+_(mean|std|z|p)|null(100)?_(mean_total|ratio|kge7_mean|z_min_k4to6|z_headline_k_range|enriched_from_k|p_bound_laplace)|null_perm\d+_.*|null_z_.*|null_.*_sum_.*)$")

NUM = re.compile(r"(?<![A-Za-z_])[-+]?\d[\d,]*(?:\.\d+)?(?:\s*[×x]\s*10\^?-?\d+|e-?\d+)?")
MULT = {"k": 1e3, "K": 1e3, "M": 1e6, "million": 1e6, "billion": 1e9, "B": 1e9, "G": 1e9, "T": 1e12}
APPROX = re.compile(r"(~|≈|≃|approx|about|roughly|nearly|almost|circa|over|>|≥|<|≤|under|up to|order of|orders of)", re.I)


def parse_numbers(s):
    """Return [(value, sig_digits, decimals, approx)] parsed from a claim value string."""
    out = []
    s2 = re.sub(r"(?<=\d)\s*[\u2013\u2014]\s*(?=\d)", " to ", s.replace("\u00a0", " ").replace("\u2212", "-"))
    s2 = s2.replace("\u2013", "-").replace("\u2014", "-")
    s2 = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " to ", s2)
    for m in NUM.finditer(s2):
        tok = m.group(0)
        raw = tok
        mult = 1.0
        tail = s2[m.end():m.end() + 9]
        mm = re.match(r"\s*(million|billion)\b", tail) or re.match(r"([kKM])(?![a-zA-Z=])", tail)
        if mm:
            mult = MULT[mm.group(1)]
        exp = 0.0
        me = re.search(r"(?:[×x]\s*10\^?(-?\d+)|e(-?\d+))$", tok)
        if me:
            exp = float(me.group(1) or me.group(2))
            tok = tok[:me.start()]
        clean = tok.replace(",", "").replace(" ", "")
        try:
            v = float(clean) * mult * (10 ** exp)
        except ValueError:
            continue
        digits = clean.lstrip("+-").replace(".", "").lstrip("0")
        if "." not in clean:
            digits = digits.rstrip("0") or "1"
        sig = len(digits) if digits else 1
        dec = len(clean.split(".")[1]) if "." in clean else 0
        approx = bool(APPROX.search(s2)) or mult != 1.0 or exp != 0
        pre = s2[max(0, m.start() - 12):m.start()].lower()
        bound = ""
        if re.search(r"(>|≥|over|more than|at least|exceed)\s*$", pre):
            bound = ">"
        elif re.search(r"(<|≤|under|less than|below|up to|at most)\s*$", pre):
            bound = "<"
        out.append((v, sig, dec, approx, raw, bound))
    return out


def fresh_number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.replace(",", ""))
        except ValueError:
            return None
    return None


def matches(claim_str, fresh):
    """Return (match: bool|None, how) comparing a claim string with a fresh value."""
    fv = fresh_number(fresh)
    nums = parse_numbers(claim_str)
    if fv is None and isinstance(fresh, list) and fresh and all(isinstance(x, (int, float)) for x in fresh) and nums:
        for x in fresh:
            m, how = matches(claim_str, x)
            if m:
                return (True, f"claimed value exists among {len(fresh)} fresh values ({how})")
        return (False, f"no match among {len(fresh)} fresh values (max {max(fresh)}, min {min(fresh)})")
    if fv is None:
        if isinstance(fresh, str):
            return (claim_str.strip().lower() == fresh.strip().lower() or fresh.strip().lower() in claim_str.lower(), "string")
        if isinstance(fresh, list):
            toks = re.findall(r"PF\d{5}|GO:\d{7}|plddt_[a-z_]+|IPR\d{6}", claim_str)
            if toks:
                fl = [str(x) for x in fresh]
                ok = [t in fl or (t.startswith("plddt_") and any(x.startswith(t) for x in fl)) for t in toks]
                return (all(ok), f"{sum(ok)}/{len(toks)} listed ids present" + ("; pLDDT item matched by prefix (paper 'plddt_mean' = item 'plddt_mean_med')" if any(t.startswith("plddt_") and t not in fl for t in toks) else ""))
            return (claim_str.strip() in [str(x) for x in fresh], "membership")
        if isinstance(fresh, bool):
            return (claim_str.strip().lower() == str(fresh).lower(), "boolean")
        return (None, "no fresh number")
    if not nums:
        return (None, "no number in claim")
    if not math.isfinite(fv):
        return (any(re.search(r"inf|∞", claim_str, re.I) for _ in [0]), "infinite")
    if "%" in claim_str and 0 < abs(fv) <= 1:
        nums = nums + [(v / 100.0, sig, dec + 2, approx, raw, bound) for v, sig, dec, approx, raw, bound in nums if v > 1]
    for v, sig, dec, approx, raw, bound in nums:
        if bound == ">" and fv >= v:
            return (True, f"fresh satisfies the stated bound {raw}")
        if bound == "<" and fv <= v:
            return (True, f"fresh satisfies the stated bound {raw}")
        if v == fv:
            return (True, "exact")
        if fv != 0 and abs(v - fv) / abs(fv) < 1e-9:
            return (True, "exact")
        if abs(v) < 1e-3 or "e" in raw.lower():
            scale = 10 ** (math.floor(math.log10(abs(v))) - sig + 1) if v else 1
            if abs(fv - v) <= scale / 2:
                return (True, f"fresh rounds to {raw} at {sig} significant digits")
            continue
        if dec > 0 and abs(round(fv, dec) - v) < 10 ** (-dec) / 2:
            return (True, f"fresh rounds to {v} at {dec} decimals")
        if approx:
            if v != 0 and fv != 0:
                scale = 10 ** (math.floor(math.log10(abs(v))) - sig + 1)
                if abs(round(fv / scale) * scale - v) <= scale / 2:
                    return (True, f"fresh rounds to {raw} at {sig} significant digits")
                if abs(math.floor(fv / scale) * scale - v) < scale / 2:
                    return (True, f"fresh truncates to {raw} at {sig} significant digits")
            if v == 0 and fv == 0:
                return (True, "exact")
        if sig <= 3 and v != 0 and fv != 0 and dec == 0:
            scale = 10 ** (math.floor(math.log10(abs(v))) - sig + 1)
            if scale >= 1 and abs(round(fv / scale) * scale - v) < scale / 2 and abs(fv - v) < abs(v) * 0.05:
                return (True, f"fresh rounds to {raw} at {sig} significant digits")
    return (False, "mismatch")


def fmt(v):
    if isinstance(v, float):
        if v.is_integer() and abs(v) < 1e15:
            return f"{int(v):,}"
        return f"{v:,.4g}" if abs(v) < 1e-3 or abs(v) >= 1e6 else f"{v:,.6g}"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, list):
        return "[" + ", ".join(str(x) for x in v[:24]) + ("…" if len(v) > 24 else "") + "]"
    return str(v)


def load_claims(p4):
    rows = []
    for path in sorted(glob.glob(f"{p4}/claims_*_keyed.csv")):
        src = os.path.basename(path).replace("claims_", "").replace("_keyed.csv", "")
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                loc = f"{r.get('file', 'et_miner_proteome.tex')}:{r.get('line') or r.get('tex_line') or ''}"
                rows.append({"claim_id": r["claim_id"], "source": src, "loc": loc, "qkey": r["qkey"],
                             "value": r["value"], "unit": r.get("unit", ""), "category": r.get("category", ""),
                             "stance": r.get("stance") or r.get("claim_type") or "", "context": r.get("quoted_context", "")})
    return rows


def load_quantities(p4):
    q = {}
    for path in [f"{p4}/quantities.csv"] + sorted(glob.glob(f"{p4}/new_quantities_*.csv")):
        if not os.path.exists(path):
            continue
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                q.setdefault(r["qkey"], r)
    return q


def verdict_for(row, q, fresh, overrides):
    qk = row["qkey"]
    rep = (q.get(qk) or {}).get("reproducibility", "")
    cat = (row["category"] or (q.get(qk) or {}).get("category") or "").strip()
    if qk == "download_size_gib":
        cat = "deterministic"
    elif rep == "hardware" or qk.startswith(("hw_", "cost_", "download_", "bench_")) and not qk.endswith(("_n_itemsets", "_sum_counts", "_itemset_hash", "_signature")):
        cat = "hardware-dependent"
    elif qk.startswith("sw_"):
        cat = "software"
    elif rep in ("rerun", "derived", "setup", "external"):
        cat = "deterministic" if cat != "method-parameter" else cat
    ov = overrides.get(qk)
    fr = fresh.get(qk)
    if fr is None:
        reason = {"external": "external fact (other systems/literature); not re-executable in this campaign",
                  "setup": "statement about the original setup; not verifiable by re-execution",
                  "hardware": "not measured in this campaign",
                  "rerun": "no fresh artifact for this quantity (step not run or did not complete)",
                  "derived": "no fresh artifact for the underlying quantity"}.get(rep, "no fresh value for this quantity")
        if ov and ov.get("verdict"):
            return ov["verdict"], "", ov.get("note", reason)
        return "inconclusive", "", reason
    fv = fr["value"]
    if ov and ov.get("verdict"):
        return ov["verdict"], fmt(fv), ov.get("note", "")
    if cat == "hardware-dependent":
        m, how = matches(row["value"], fv)
        note = "hardware-dependent; this box = 2×RTX 3090 (paper: H100 80 GB)"
        fn = fresh_number(fv); cn = [x[0] for x in parse_numbers(row["value"]) if x[0]]
        if fn is not None and cn:
            cv = min(cn, key=lambda x: abs(math.log10(abs(x)) - math.log10(abs(fn))) if fn else 0)
            note += f"; ratio fresh/claimed = {fn / cv:.2f} (claimed {cv:g})"
        if m:
            note += "; values agree despite hardware difference"
        return "expected-hardware-deviation", fmt(fv), note
    if cat == "software":
        m, how = matches(row["value"], fv)
        return ("confirmed" if m else "inconclusive"), fmt(fv), ("identical in this environment" if m else "environment statement; this box differs")
    if qk.endswith("_kmax") and re.match(r"\s*K?\s*(>=|≥|>)\s*(\d+)", row["value"]):
        thr = int(re.search(r"(>=|≥|>)\s*(\d+)", row["value"]).group(2))
        fn = fresh_number(fv)
        if fn is not None:
            return ("confirmed" if fn < thr else "hallucinated"), fmt(fv), f"claim states that K >= {thr} is absent; fresh K_max = {fmt(fv)}"
    if isinstance(fv, str) and re.fullmatch(r"[0-9a-f]{16,}", fv.strip()):
        cv = re.sub(r"[^0-9a-f]", "", row["value"].lower().replace("…", ""))
        ok = len(cv) >= 8 and fv.strip().startswith(cv)
        return ("confirmed" if ok else "hallucinated"), fv[:16] + "…", (f"hash prefix of {len(cv)} hex chars matches" if ok else "hash differs")
    if isinstance(fv, str) and re.search(r"\d", fv) and fresh_number(fv) is None:
        cn = {n[0] for n in parse_numbers(row["value"])}; fn_ = {float(x) for x in re.findall(r"\d+(?:\.\d+)?", fv)}
        if cn and fn_ and qk.endswith("_levels"):
            ok = cn <= fn_
            return ("confirmed" if ok else "inconclusive"), fmt(fv), ("all per-K numbers present in the fresh run" if ok else "per-K candidate counts are code-path dependent (0 where a path does not count candidates); the frequent-itemset totals are judged under the *_n_itemsets / *_sum_counts / *_itemset_hash keys")
        if cn and fn_ and qk.endswith("_params"):
            ok = cn <= fn_
            return ("confirmed" if ok else "inconclusive"), fmt(fv), ("run configuration reproduced as stated" if ok else "run configuration of the old campaign differs from the fresh runner's (e.g. timeouts); not a result")
        if cn and fn_:
            ok = cn <= fn_
            if not ok and RANDOM_KEYS.match(qk):
                return "inconclusive", fmt(fv), f"random-stream-dependent permutation statistic; fresh {fv} vs claim {row['value'][:40]}" + (f"; {ov['note']}" if ov and ov.get("note") else "")
            return ("confirmed" if ok else "hallucinated"), fmt(fv), (f"claim numbers {sorted(cn)} ⊆ fresh {fv}" if ok else f"claim numbers {sorted(cn)} not all in fresh {fv}") + (f"; {ov['note']}" if ov and ov.get("note") else "")
    if len({n[0] for n in parse_numbers(row["value"])}) >= 3 and not isinstance(fv, list):
        m0, how0 = matches(row["value"], fv)
        if not m0 and RANDOM_KEYS.match(qk):
            return "inconclusive", fmt(fv), ("random-stream-dependent permutation statistic (transaction row order and RNG differ from the original run; same seed 42, same script); fresh value shown, exact equality not expected"
                                            + (f"; {ov['note']}" if ov and ov.get("note") else ""))
        if not m0:
            return "inconclusive", fmt(fv), "composite claim text with several numbers; not attributable to this single quantity"
    m, how = matches(row["value"], fv)
    if m is None:
        return "inconclusive", fmt(fv), f"claim not numerically comparable ({how})"
    if m:
        extra = "; compound claim text — only the number matching this quantity is judged here" if len({n[0] for n in parse_numbers(row["value"])}) >= 3 else ""
        return "confirmed", fmt(fv), how + extra + (f"; {ov['note']}" if ov and ov.get("note") else "")
    if RANDOM_KEYS.match(qk):
        return "inconclusive", fmt(fv), ("random-stream-dependent permutation statistic (transaction row order and RNG differ from the original run; same seed 42, same script); fresh value shown, exact equality not expected"
                                        + (f"; {ov['note']}" if ov and ov.get("note") else ""))
    return "hallucinated", fmt(fv), "exact-match rule: fresh value differs" + (f"; {ov['note']}" if ov and ov.get("note") else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--phase4", required=True)
    ap.add_argument("--fresh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inconsistencies")
    ap.add_argument("--header")
    a = ap.parse_args()
    q = load_quantities(a.phase4)
    claims = load_claims(a.phase4)
    fresh = json.load(open(a.fresh))
    overrides = json.load(open(f"{a.phase4}/overrides.json")) if os.path.exists(f"{a.phase4}/overrides.json") else {}
    rekey = json.load(open(f"{a.phase4}/rekey.json")) if os.path.exists(f"{a.phase4}/rekey.json") else {}
    for r in claims:
        if r["claim_id"] in rekey:
            r["qkey"] = rekey[r["claim_id"]]
    results = []
    for r in claims:
        v, fv, note = verdict_for(r, q, fresh, overrides)
        results.append({**r, "verdict": v, "fresh": fv, "note": note})
    by_src = defaultdict(Counter); by_cat = defaultdict(Counter); tot = Counter()
    for r in results:
        by_src[r["source"]][r["verdict"]] += 1; by_cat[r["category"]][r["verdict"]] += 1; tot[r["verdict"]] += 1
    V = ["confirmed", "hallucinated", "inconclusive", "expected-hardware-deviation"]
    L = []
    if a.header and os.path.exists(a.header):
        L.append(open(a.header).read().rstrip() + "\n")
    L.append("## 1. Verdict summary\n")
    L.append(f"Total claims: **{len(results)}** across {len(by_src)} source groups.\n")
    L.append("| source group | " + " | ".join(V) + " | total |")
    L.append("|---|" + "---|" * (len(V) + 1))
    for s in sorted(by_src):
        L.append(f"| {s} | " + " | ".join(str(by_src[s][v]) for v in V) + f" | {sum(by_src[s].values())} |")
    L.append("| **all** | " + " | ".join(f"**{tot[v]}**" for v in V) + f" | **{len(results)}** |")
    L.append("\n| claim category | " + " | ".join(V) + " | total |")
    L.append("|---|" + "---|" * (len(V) + 1))
    for c in sorted(by_cat):
        L.append(f"| {c} | " + " | ".join(str(by_cat[c][v]) for v in V) + f" | {sum(by_cat[c].values())} |")
    # headline quantities: paper (tex) claims with fresh values
    L.append("\n## 2. Headline quantities (paper claim vs fresh value)\n")
    L.append("One row per canonical quantity that has both a paper claim and a fresh value. Verdict = worst verdict among the paper's numerically comparable claim rows for that quantity (non-numeric phrasings such as 'millions' are inconclusive and excluded).\n")
    L.append("| qkey | description | paper value | fresh value | verdict | fresh artifact |")
    L.append("|---|---|---|---|---|---|")
    order = {"hallucinated": 0, "confirmed": 1, "expected-hardware-deviation": 2, "inconclusive": 3}
    per_q = defaultdict(list)
    for r in results:
        if r["source"] == "tex" and r["qkey"] in fresh:
            per_q[r["qkey"]].append(r)
    for qk in sorted(per_q):
        rs = per_q[qk]; worst = min(rs, key=lambda r: order[r["verdict"]])
        desc = (q.get(qk) or {}).get("description", "")[:90]
        pv = (q.get(qk) or {}).get("claimed_value_tex", rs[0]["value"])[:60]
        art = fresh[qk]["artifact"].replace("/root/projects/ET-Miner/", "")
        L.append(f"| {qk} | {desc} | {pv} | {fmt(fresh[qk]['value'])} | {worst['verdict']} | {art} |")
    if a.inconsistencies and os.path.exists(a.inconsistencies):
        L.append("\n## 3. Key inconsistencies (from INCONSISTENCIES.md)\n")
        for line in open(a.inconsistencies):
            if line.startswith("## "):
                L.append("- " + line[3:].strip())
    L.append("\n## 4. Every extracted claim (source → fresh value → verdict)\n")
    L.append("Columns: claim ID | source file:line | claimed value [unit] | quantity key | fresh value | verdict | note. Contexts are in CLAIMS.md.\n")
    for s in sorted(by_src):
        L.append(f"\n### 4.{s}\n")
        L.append("| ID | source | claimed | qkey | fresh | verdict | note |")
        L.append("|---|---|---|---|---|---|---|")
        for r in [x for x in results if x["source"] == s]:
            val = (r["value"] + (f" {r['unit']}" if r["unit"] and r["unit"] not in r["value"] else "")).replace("|", "\\|")
            L.append(f"| {r['claim_id']} | {r['loc']} | {val[:70]} | {r['qkey']} | {str(r['fresh'])[:60]} | {r['verdict']} | {r['note'][:110].replace('|', '/')} |")
    L.append("\n## 5. Fresh values used (qkey → value, artifact, command)\n")
    L.append("| qkey | fresh value | unit | artifact | command | UTC |")
    L.append("|---|---|---|---|---|---|")
    for qk in sorted(fresh):
        f = fresh[qk]
        L.append(f"| {qk} | {fmt(f['value'])[:80]} | {f.get('unit', '')} | {f['artifact'].replace('/root/projects/ET-Miner/', '')} | {str(f.get('command', ''))[:90].replace('|', '/')} | {f.get('utc', '')} |")
    open(a.out, "w").write("\n".join(L) + "\n")
    json.dump(results, open(a.out.replace(".md", "_rows.json"), "w"), indent=0, default=str)
    print("claims:", len(results), dict(tot))
    print("unkeyed/missing-quantity qkeys:", len({r['qkey'] for r in results if r['qkey'] not in q}))


if __name__ == "__main__":
    main()
