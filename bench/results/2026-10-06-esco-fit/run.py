"""B: where "auto" transitions on dsl after the fit check, and the peak VRAM. One rep, no timing claims."""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "bench"))
from consolidation_matrix import _cfg  # noqa: E402
from runner import capture_environment, run_config  # noqa: E402

OUT = REPO / "bench/results/2026-10-06-esco-fit"
BUDGET_GPU_S = 0.1 * 3600
cfgs = []
for n in (1, 2):
    for infer in (False, True):
        name = f"C{n}-esco-auto{'-infer' if infer else ''}"
        c = _cfg(name, "dsl", "C", n_gpus=n, sparse_from_k="auto", timeout_s=900, use_generator_pruning=infer,
                 env={"NCCL_P2P_DISABLE": "1"})
        cfgs.append({**c, "id": f"{c['base_id']}#r0", "rep": 0})

capture_environment(OUT)
used = 0.0
with (OUT / "raw.jsonl").open("a") as f:
    for c in cfgs:
        if used + 120 * c["n_gpus"] > BUDGET_GPU_S:
            print(f"budget stop before {c['id']}: {used / 3600:.3f} GPU-h used", flush=True)
            break
        r = run_config(c, OUT)
        used += r.get("proc_s", 0) * c["n_gpus"]
        f.write(json.dumps(r) + "\n")
        f.flush()
        print(f"{c['id']}: {r['status']} wall={r.get('wall_s')} peak={r.get('peak_vram_mb')} "
              f"trans={r.get('timings', {}).get('density_transitions')}", flush=True)
print(f"GPU-h used: {used / 3600:.3f}", flush=True)
