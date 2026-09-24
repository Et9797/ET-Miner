"""Static checks over docs/: links, anchors, source references, forbidden words, style."""
import json, re, sys, importlib, ast
from pathlib import Path
ROOT = Path("/home/user/ET-Miner"); DOCS = ROOT / "docs"
problems = []

def texts():
    for p in sorted(DOCS.rglob("*")):
        if p.suffix == ".md":
            yield p, p.read_text(), p.read_text()
        elif p.suffix == ".ipynb":
            nb = json.loads(p.read_text())
            md = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
            allsrc = "\n".join("".join(c["source"]) for c in nb["cells"])
            yield p, md, allsrc

def slug(h):
    h = h.strip().lower()
    h = re.sub(r"[^\w\- ]", "", h)
    return h.replace(" ", "-")

def anchors(p):
    if p.suffix == ".md":
        src = p.read_text()
    else:
        nb = json.loads(p.read_text()); src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    return {slug(m) for m in re.findall(r"^#+\s+(.*)$", src, re.M)}

for p, md, allsrc in texts():
    for target in re.findall(r"\]\(([^)]+)\)", md):
        if target.startswith("http"): continue
        path, _, anchor = target.partition("#")
        dest = (p.parent / path).resolve() if path else p
        if not dest.exists():
            problems.append(f"{p.relative_to(ROOT)}: broken link {target}"); continue
        if anchor and dest.is_file() and anchor not in anchors(dest):
            problems.append(f"{p.relative_to(ROOT)}: missing anchor #{anchor} in {dest.relative_to(ROOT)}")
    # source references path:name
    for path, name in re.findall(r"`((?:src|rust_ext|bench|tests)/[\w/\.\-]+\.(?:py|rs|cu|toml|md|sh))(?::([\w\.]+))?`", allsrc):
        f = ROOT / path
        if not f.exists():
            problems.append(f"{p.relative_to(ROOT)}: missing file {path}"); continue
        if name:
            last = name.split(".")[-1]
            if not re.search(rf"\b(def|fn|class|struct)\s+{re.escape(last)}\b|\b{re.escape(last)}\s*[:=(]|@property\s+def {last}", f.read_text()):
                problems.append(f"{p.relative_to(ROOT)}: {path}:{name} not found")
    low = md.lower()
    for word in ["claude", "anthropic", "ai assist", "agent", "preprint", "retract", "alphafold", "bugs_found", "vast.ai", "ahmic", "copilot", "chatgpt", "gpt-"]:
        if word in low:
            problems.append(f"{p.relative_to(ROOT)}: forbidden word '{word}'")
    for pat, label in [("—", "em dash"), ("!", "exclamation"), (r"\b(blazing|powerful|seamless|cutting-edge|lightning|revolutionary|incredibl)", "marketing")]:
        for m in re.finditer(pat, md):
            ctx = md[max(0, m.start()-20):m.end()+20].replace("\n", " ")
            if label == "exclamation" and ("!=" in md[m.start():m.start()+2] or "`" in ctx): continue
            problems.append(f"{p.relative_to(ROOT)}: {label}: ...{ctx}...")
    for brit in ["honour", "favour", "behaviour", "colour", "summaris", "optimis", "materialis", "synchronis", "recognis", "analyse", "normalis", "initialis", "serialis"]:
        if brit in low:
            problems.append(f"{p.relative_to(ROOT)}: British spelling '{brit}'")
    if p.suffix == ".ipynb":
        nb = json.loads(p.read_text())
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code":
                if not c.get("outputs") and c.get("execution_count") is None:
                    problems.append(f"{p.relative_to(ROOT)}: cell {i} not executed")
                for o in c.get("outputs", []):
                    if o["output_type"] == "error": problems.append(f"{p.relative_to(ROOT)}: cell {i} error")
                    t = "".join(o.get("text", "")) + "".join(o.get("data", {}).get("text/plain", ""))
                    if "/home/user" in t or "/tmp/" in t: problems.append(f"{p.relative_to(ROOT)}: cell {i} leaks a local path")
                    if "Warning" in t: problems.append(f"{p.relative_to(ROOT)}: cell {i} warning in output")
        # heading numbering
        heads = re.findall(r"^## (.*)$", md, re.M)
        for h in heads:
            if not re.match(r"(\d+\. |Summary and next step|Summary$)", h): problems.append(f"{p.relative_to(ROOT)}: unnumbered heading '{h}'")
print("\n".join(problems) or "no problems")
