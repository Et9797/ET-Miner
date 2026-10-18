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

## Held-out points (Amendment 1)

- At `16d9d7b` (`2bd575f`, the amendment, plus the environment capture in
  `env.txt`); every row of `k3_heldout.jsonl` is stamped `16d9d7b`.
- 4 of the 8 points ran, all valid, all four from the disagreeing six. The run
  stopped at its cap (`--max-gpu-hours 0.15`) after 0.183 GPU-hours.
  - Not run: `zipf-N4M-F600-L16`, `zipf-N2M-F1000-L32` (disagreeing),
    `zipfrev-N1M-F1600-L32`, `zipf-N4M-F800-L16` (agreeing).
- A first start at `2bd575f` was stopped within its first point, about 20 s
  of GPU time, without a row. The environment capture had appended to the
  tracked `env.txt`, which would have stamped every row dirty.

## Files

- `k3_crossover.jsonl`: one row per point (statistics, the K=3 times of every
  rep, signatures).
- `k3_crossover.txt`: `--analyze` at `76c0aac` (the points by s, r3's
  crossings, and the regret table).
- `sweep.log` (not committed; `*.log` is ignored): the sweep's progress lines.
- `k3_heldout.jsonl` / `k3_heldout.txt`: the held-out points and their
  `--analyze` at the frozen thresholds.
- `heldout.log` (not committed): the held-out run's progress lines.
- `env.txt`: the environment captures (before the sweep and before the
  held-out points).

## Commands

```bash
OUT=bench/results/2026-10-18-k3-dispatch
uv run python bench/moonshots/k3_crossover.py --grid dispatch --out $OUT/k3_crossover.jsonl --max-gpu-hours 0.7
uv run python bench/moonshots/k3_crossover.py --grid dispatch --out $OUT/k3_crossover.jsonl --analyze > $OUT/k3_crossover.txt
uv run python bench/moonshots/k3_crossover.py --grid heldout --out $OUT/k3_heldout.jsonl --max-gpu-hours 0.15
uv run python bench/moonshots/k3_crossover.py --grid heldout --out $OUT/k3_heldout.jsonl --analyze > $OUT/k3_heldout.txt
```
