# K=3 dispatch statistic: the sweep (2026-10-18)

`perf/m2-dispatch-stat` at `76c0aac` (`main` at `37819e2` plus `f7acff3`, which
logs s beside r3, the protocol and the handoff). Every row of
`k3_crossover.jsonl` is stamped `76c0aac`. Protocol:
`../../moonshots/PROTOCOL-K3-DISPATCH.md`.

## Box

The M1 box: 2× RTX 3060 12 GB, EPYC 7282, driver 580.173.02, CuPy 14.1.1, a
shared host. The sweep ran on device 0 alone, NCCL off, thread pools at 16.

## Gates (before the sweep)

| check | result |
|---|---|
| `uv run pytest -q -m "not slow" -k "not test_support_00"` (before the level split's fix) | 1628 passed, 1 failed (`test_level_split_times_the_row_wise_k3`: `k2_dispatch` untimed at K=3) |
| `uv run pytest -q tests/test_moonshots_bench.py` (after the fix, in `f7acff3`) | 19 passed |
| `uv run pytest -q tests/test_k3_rows.py tests/test_k2_rows.py tests/test_input_layer.py tests/test_tier_equivalence.py` | 253 passed |
| `uv run ruff check src tests bench` | clean |

## Run

- 42 of the 43 points ran, all valid: both kernels gave the same itemsets and
  counts, the row-wise K=3 was logged exactly on the rows runs, and no run
  logged a fallback.
- The sweep stopped at its cap (`--max-gpu-hours 0.7`) after 0.724 GPU-hours,
  before the last point, `zipf-N1M-F2400-L32`, ran. None errored or timed out.

## Files

- `k3_crossover.jsonl`: one row per point (statistics, the K=3 times of every
  rep, signatures).
- `k3_crossover.txt`: `--analyze` at `76c0aac` (the points by s, r3's
  crossings, and the regret table).
- `sweep.log`: the sweep's progress lines.
- `env.txt`: the environment capture.

## Commands

```bash
OUT=bench/results/2026-10-18-k3-dispatch
uv run python bench/moonshots/k3_crossover.py --grid dispatch --out $OUT/k3_crossover.jsonl --max-gpu-hours 0.7
uv run python bench/moonshots/k3_crossover.py --grid dispatch --out $OUT/k3_crossover.jsonl --analyze > $OUT/k3_crossover.txt
```
