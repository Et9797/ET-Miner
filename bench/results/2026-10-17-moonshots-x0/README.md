# X0: count inference off and on (2026-10-17)

`perf/moonshot-x0` at `68c7d27` (M1 + M2); every row of `raw.jsonl` is stamped
`68c7d27`. Protocol: `../../moonshots/PROTOCOL-X0.md`; report:
`../../moonshots/REPORT-X0.md`. Box: as M1 and M2 (2× RTX 3060, EPYC 7282,
thread pools at 16, `NCCL_P2P_DISABLE=1`). No source change, so the gates of
M2 (`../2026-10-17-moonshots-m2/README.md`) cover this tree.

- `raw.jsonl`: 96 rows (8 regimes × 2 arms × 1 and 2 GPUs × 3 reps). The
  per-row `*.result.json` and `*.log` files are not committed.
- `decision.md`: `decide.py --arms plain infer`.
- `split.md`: the wall split per config.

```bash
OUT=bench/results/2026-10-17-moonshots-x0
uv run python bench/runner.py --mode moonshots-x0 --out $OUT --max-gpu-hours 0.4
uv run python bench/moonshots/decide.py $OUT/raw.jsonl --arms plain infer > $OUT/decision.md
uv run python bench/moonshots/split_table.py $OUT/raw.jsonl > $OUT/split.md
```
