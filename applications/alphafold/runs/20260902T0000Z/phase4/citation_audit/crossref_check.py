"""Check every hand-written \\bibitem of a LaTeX file against CrossRef and Semantic Scholar.

Usage: python crossref_check.py TEX OUT_JSON OUT_MD

Options: none. Queries are rate-limited to one request per 1.3 s per service.
"""
import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request

UA = "ET-Miner-citation-audit/1.0"


def get(url):
    """Return the JSON document at url, or None on any HTTP/network error."""
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())
    except Exception as exc:  # noqa: BLE001
        return {"_error": str(exc)}


def clean(s):
    """Strip LaTeX markup from a bibliography fragment."""
    s = re.sub(r"\\[\'\"`^~=.]\s*", "", s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)
    s = s.replace("~", " ").replace("{", "").replace("}", "").replace("--", "-").replace("$", "")
    return re.sub(r"\s+", " ", s).strip().rstrip(".")


def parse_bib(tex):
    """Return [{key, authors, title, venue}] for each \\bibitem in the thebibliography block."""
    block = re.search(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", tex, re.S).group(0)
    out = []
    for it in re.split(r"\\bibitem\{", block)[1:]:
        key, rest = it.split("}", 1)
        rest = rest.split("\\end{thebibliography}")[0]
        parts = [clean(p) for p in re.split(r"\\newblock", rest)]
        out.append({"key": key, "authors": parts[0], "title": parts[1] if len(parts) > 1 else "",
                    "venue": parts[2] if len(parts) > 2 else ""})
    return out


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower())


def first_family(authors):
    a = re.sub(r"\bet al\b\.?", "", authors).split(",")[0].split(" and ")[0].strip()
    a = re.sub(r"^(The )?", "", a)
    return a.split()[-1] if a else ""


def crossref(title, family, use_author=True):
    params = {"query.bibliographic": title, "rows": 3}
    if use_author and family:
        params["query.author"] = family
    d = get("https://api.crossref.org/works?" + urllib.parse.urlencode(params))
    items = (d or {}).get("message", {}).get("items", []) if "_error" not in (d or {}) else []
    best, score = None, 0.0
    for it in items:
        t = " ".join(it.get("title", []))
        r = difflib.SequenceMatcher(None, norm(title), norm(t)).ratio()
        if r > score:
            best, score = it, r
    if not best:
        return {"error": (d or {}).get("_error", "no items"), "score": 0}
    date = best.get("issued", {}).get("date-parts", [[None]])[0][0]
    return {
        "score": round(score, 3),
        "doi": best.get("DOI"),
        "title": " ".join(best.get("title", [])),
        "container": " ".join(best.get("container-title", [])),
        "volume": best.get("volume"), "issue": best.get("issue"), "page": best.get("page"),
        "year": date, "type": best.get("type"),
        "authors": [a.get("family", "") for a in best.get("author", [])][:8],
        "abstract": re.sub(r"<[^>]+>", "", best.get("abstract", ""))[:1500],
    }


def s2(title):
    q = urllib.parse.urlencode({"query": title, "limit": 3,
                                "fields": "title,year,venue,authors,externalIds,openAccessPdf,abstract"})
    d = get("https://api.semanticscholar.org/graph/v1/paper/search?" + q)
    if not d or "_error" in d:
        return {"error": (d or {}).get("_error", "none")}
    best, score = None, 0.0
    for it in d.get("data", []):
        r = difflib.SequenceMatcher(None, norm(title), norm(it.get("title", ""))).ratio()
        if r > score:
            best, score = it, r
    if not best:
        return {"score": 0}
    return {"score": round(score, 3), "title": best.get("title"), "year": best.get("year"),
            "venue": best.get("venue"), "ids": best.get("externalIds"),
            "authors": [a.get("name") for a in best.get("authors", [])][:8],
            "oa_pdf": (best.get("openAccessPdf") or {}).get("url"),
            "abstract": (best.get("abstract") or "")[:1500]}


def compare(bib, cr):
    """Return a list of mismatch notes between the bib entry and the CrossRef record."""
    notes = []
    if cr.get("score", 0) < 0.85:
        return ["no confident CrossRef match (score %s)" % cr.get("score")]
    venue = bib["venue"]
    year = re.findall(r"(19|20)\d\d", venue)
    bib_year = [m.start() for m in re.finditer(r"(19|20)\d\d", venue)]
    years = re.findall(r"\b((?:19|20)\d\d)\b", venue)
    if cr.get("year") and years and str(cr["year"]) not in years:
        notes.append(f"year: bib {years} vs CrossRef {cr['year']}")
    if cr.get("volume") and cr["volume"] not in venue:
        notes.append(f"volume: CrossRef {cr['volume']} not in bib venue")
    if cr.get("page"):
        first = cr["page"].split("-")[0]
        if first not in venue:
            notes.append(f"pages: CrossRef {cr['page']} vs bib venue")
    fam = first_family(bib["authors"])
    if cr.get("authors") and fam and norm(fam) not in [norm(a) for a in cr["authors"]]:
        notes.append(f"first author: bib '{fam}' not among CrossRef {cr['authors'][:3]}")
    return notes


def main():
    tex, out_json, out_md = sys.argv[1:4]
    bib = parse_bib(open(tex).read())
    rows = []
    for b in bib:
        cr = crossref(b["title"], first_family(b["authors"]))
        time.sleep(1.3)
        if cr.get("score", 0) < 0.85:
            cr2 = crossref(b["title"], "", use_author=False)
            time.sleep(1.3)
            if cr2.get("score", 0) > cr.get("score", 0):
                cr = cr2
        s = s2(b["title"])
        time.sleep(1.3)
        b["crossref"], b["s2"], b["mismatch"] = cr, s, compare(b, cr)
        rows.append(b)
        print(b["key"], cr.get("doi"), cr.get("score"), b["mismatch"], flush=True)
    json.dump(rows, open(out_json, "w"), indent=1)
    with open(out_md, "w") as f:
        f.write("| key | bib venue | CrossRef DOI | score | CR container/vol(issue):page year | mismatch | S2 OA pdf |\n|---|---|---|---|---|---|---|\n")
        for b in rows:
            cr, s = b["crossref"], b["s2"]
            f.write(f"| {b['key']} | {b['venue']} | {cr.get('doi')} | {cr.get('score')} | {cr.get('container')} {cr.get('volume')}({cr.get('issue')}):{cr.get('page')} {cr.get('year')} | {'; '.join(b['mismatch']) or 'ok'} | {s.get('oa_pdf')} |\n")


if __name__ == "__main__":
    main()
