# M2 campaign: dense against row-wise K=3 (2026-10-17)

`perf/moonshot-m2`, stacked on M1. The sweep ran at `313dfc9`; the campaign
and the final check at `902bbad`, where every row of `raw.jsonl` is stamped.
Protocol: `../../moonshots/PROTOCOL-M2.md`; report and decision:
`../../moonshots/REPORT-M2.md`.

## Box

The M1 box (`../2026-10-17-moonshots-m1/README.md`): 2× RTX 3060 12 GB, EPYC
7282, thread pools pinned to 16, `NCCL_P2P_DISABLE=1`, a shared host.

## Gates (before the campaign, at `313dfc9`)

| check | result |
|---|---|
| `uv run pytest tests/test_tier_equivalence.py -q` | 73 passed |
| `uv run pytest -q -m "gpu and not slow"` | 609 passed |
| `uv run pytest tests/test_k3_rows.py -q` | 27 passed |
| `uv run python bench/selfcheck.py` | READY: 15 kernels on 2 devices |
| `uv run ruff check src tests bench` | clean |

## Files

- `k3_crossover.jsonl` / `.txt`: the crossover sweep (18 points) and its
  printed analysis.
- `raw.jsonl`: 120 campaign rows (3 reps, rep-major) and 20 final-check rows
  (`*-final#r0`, the K=3 kernel unset). The per-row `*.result.json` and `*.log`
  files are not committed (`.gitignore`).
- `decision.md`: `decide.py --arms dense rows`.
- `split.md`: the wall split per config.
- `env.txt`: the environment capture.

## Commands

```bash
OUT=bench/results/2026-10-17-moonshots-m2
uv run python bench/moonshots/k3_crossover.py --out $OUT/k3_crossover.jsonl --max-gpu-hours 0.35
uv run python bench/runner.py --mode moonshots-m2 --out $OUT --max-gpu-hours 0.55
uv run python bench/runner.py --mode moonshots-m2-final --out $OUT --max-gpu-hours 0.1
uv run python bench/moonshots/decide.py $OUT/raw.jsonl --arms dense rows > $OUT/decision.md
uv run python bench/moonshots/split_table.py $OUT/raw.jsonl > $OUT/split.md
```
