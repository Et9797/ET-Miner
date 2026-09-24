import json, sys
nb = json.load(open(sys.argv[1]))
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code" or i < start: continue
    for o in c.get("outputs", []):
        if o["output_type"] == "stream": txt = "".join(o["text"])
        elif o["output_type"] in ("execute_result", "display_data"): txt = "".join(o["data"].get("text/plain", ""))
        elif o["output_type"] == "error": txt = "ERROR " + o["ename"] + ": " + o["evalue"]
        else: continue
        print(f"--- cell {i} [{o['output_type']}]"); print(txt[:3000])
