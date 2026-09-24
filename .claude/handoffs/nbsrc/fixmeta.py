import json, sys
for p in sys.argv[1:]:
    nb = json.load(open(p))
    nb["metadata"].pop("jupytext", None)
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    json.dump(nb, open(p, "w"), indent=1, ensure_ascii=False); open(p, "a").write("\n")
