"""Reorder a hand-written thebibliography block into first-citation order and drop uncited entries.

Usage: python reorder_bibliography.py TEX [--check]

Options:
  --check   report the current order, the first-citation order and the orphans without writing.

Citation order is taken over the whole document with the bibliography block removed (body first,
then the appendix that follows the bibliography), which is the order BibTeX's unsrt style would
produce. The width argument of \\begin{thebibliography}{N} is set to the new entry count.
"""
import re
import sys


def split(tex):
    m = re.search(r"\\begin\{thebibliography\}\{\d+\}\n(.*?)\\end\{thebibliography\}", tex, re.S)
    if not m:
        raise SystemExit("no thebibliography block")
    return tex[: m.start()], m.group(1), tex[m.end():]


def parse_items(block):
    items = {}
    order = []
    for chunk in re.split(r"(?=\\bibitem\{)", block):
        if not chunk.startswith("\\bibitem{"):
            continue
        key = re.match(r"\\bibitem\{([^}]+)\}", chunk).group(1)
        items[key] = chunk.strip("\n") + "\n"
        order.append(key)
    return items, order


def cited_keys(text):
    seen = []
    for m in re.finditer(r"\\cite\{([^}]*)\}", text):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and k not in seen:
                seen.append(k)
    return seen


def main():
    path = sys.argv[1]
    check = "--check" in sys.argv
    tex = open(path, encoding="utf-8").read()
    head, block, tail = split(tex)
    items, current = parse_items(block)
    cites = cited_keys(head + tail)
    missing = [k for k in cites if k not in items]
    orphans = [k for k in current if k not in cites]
    new_order = [k for k in cites if k in items]
    print("bibitems:", len(current), "cited keys:", len(cites), "orphans:", orphans, "missing:", missing)
    first_mismatch = next((i for i, (a, b) in enumerate(zip(current, new_order)) if a != b), None)
    print("first position where current order differs from citation order:", first_mismatch)
    if check:
        print("citation order:", new_order)
        return
    if missing:
        raise SystemExit(f"cited keys without bibitem: {missing}")
    new_block = "\\begin{thebibliography}{%d}\n\n" % len(new_order) + "\n".join(items[k] for k in new_order) + "\n"
    out = head + new_block + "\\end{thebibliography}" + tail
    open(path, "w", encoding="utf-8").write(out)
    print("wrote", path, "with", len(new_order), "entries; removed", orphans)


if __name__ == "__main__":
    main()
