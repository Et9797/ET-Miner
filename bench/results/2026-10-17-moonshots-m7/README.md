# M7: count inference with and without the deduction bounds (2026-10-17)

`perf/moonshot-m7` at `9abcd99` (stacked on X0, M2, M1); every row of
`raw.jsonl` is stamped `9abcd99`. Protocol: `../../moonshots/PROTOCOL-M7.md`;
report: `../../moonshots/REPORT-M7.md`. Box: as M1–X0.

## Gates (at `9abcd99`, before the campaign)

| check | result |
|---|---|
| `uv run pytest tests/test_tier_equivalence.py -q` | 79 passed |
| `uv run pytest -q -m "gpu and not slow"` | 645 passed |
| `uv run pytest tests/test_deduction.py -q` | 31 passed |
| `uv run python bench/selfcheck.py` | READY (the bounds launched too) |
| `uv run ruff check src tests bench` | clean |

## Files

- `raw.jsonl`: 96 campaign rows (8 regimes × `pascal` / `bounds` × 1 and 2
  GPUs × 3 reps) and 16 final-check rows (`*-final#r0`, the knob unset).
- `decision.md`: `decide.py --arms pascal bounds`; `split.md`: the wall split.

```bash
OUT=bench/results/2026-10-17-moonshots-m7
uv run python bench/runner.py --mode moonshots-m7 --out $OUT --max-gpu-hours 0.45
uv run python bench/moonshots/decide.py $OUT/raw.jsonl --arms pascal bounds > $OUT/decision.md
uv run python bench/moonshots/split_table.py $OUT/raw.jsonl > $OUT/split.md
```
