# Phase 3: the reduced matrix, baseline against the final tree (2026-10-18)

Protocol: `../../moonshots/PROTOCOL-PHASE3.md`; summary:
`../../moonshots/REPORT.md`. Box: 2× RTX 3060, EPYC 7282, thread pools at
16, `NCCL_P2P_DISABLE=1`, every `ET_MINER_*` knob unset.

- `final/raw.jsonl`: the stacked tree at `36a3e21` (`perf/moonshot-polish`),
  60 rows.
- `baseline/raw.jsonl`: `src/` of `c0f0fd2` (`main` + `deep_k20`) over the
  same harness, 60 rows. They are stamped `36a3e21-dirty.1b7971f81b54`, the
  digest of exactly that `src/` swap.
- `baseline.jsonl` / `final.jsonl` / `compare.jsonl`: the rows relabelled for
  `decide.py`; `decision.md`, `split.md`: the comparison and the wall split.
- The per-row `*.result.json` and `*.log` files are not committed.

Every regime carries one signature across both trees (must-fix: none). The
baseline used 0.405 GPU-hours and the final tree 0.178.

```bash
OUT=bench/results/2026-10-18-moonshots-phase3
uv run python bench/runner.py --mode moonshots-phase3 --out $OUT/final --max-gpu-hours 0.25
git checkout c0f0fd2 -- src
uv run python bench/runner.py --mode moonshots-phase3 --out $OUT/baseline --max-gpu-hours 0.55
git checkout HEAD -- src
# relabel and compare: see PROTOCOL-PHASE3.md
```
