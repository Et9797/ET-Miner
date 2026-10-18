# M5b: keep against lex row order (2026-10-17)

`perf/moonshot-m5b` at `797b54c`; every row of `raw.jsonl` is stamped `797b54c`.
Protocol: `../../moonshots/PROTOCOL-M5b.md`; report:
`../../moonshots/REPORT-M5b.md`. Box: as M1–M7.

## Gates (at `797b54c`, before the campaign)

| check | result |
|---|---|
| `uv run pytest tests/test_tier_equivalence.py -q` | 91 passed |
| `uv run pytest -q -m "gpu and not slow"` | 692 passed |
| `uv run pytest tests/test_row_order.py -q` | 36 passed |
| `uv run python bench/selfcheck.py` | READY |
| `uv run ruff check src tests bench` | clean |

## Files

- `raw.jsonl`: 120 campaign rows (10 regimes × keep / lex × 1 and 2 GPUs × 3
  reps) and 20 final-check rows (`*-final#r0`, the order unset).
- `decision.md`: `decide.py --arms keep lex`; `split.md`: the wall split.

```bash
OUT=bench/results/2026-10-17-moonshots-m5b
uv run python bench/runner.py --mode moonshots-m5b --out $OUT --max-gpu-hours 0.6
uv run python bench/moonshots/decide.py $OUT/raw.jsonl --arms keep lex > $OUT/decision.md
uv run python bench/moonshots/split_table.py $OUT/raw.jsonl > $OUT/split.md
```
