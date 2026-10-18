# K=3 dispatch statistic: the final check (2026-10-18)

`perf/m2-dispatch-stat` at `b520037`, where the dispatch counts K=3 row-wise
when s = (R2 + ρ² T3) / (slots × words) < 1.0e-2. Every row of `raw.jsonl` is
stamped `b520037`. Protocol: `../../moonshots/PROTOCOL-K3-DISPATCH.md`
(the matrix guard); the sweep and the held-out points are in
`../2026-10-18-k3-dispatch/`.

## Box

The M1 box: 2× RTX 3060 12 GB, EPYC 7282, thread pools at 16,
`NCCL_P2P_DISABLE=1`, a shared host.

## Gates (at `b520037`, before the check)

| check | result |
|---|---|
| `uv run pytest tests/test_tier_equivalence.py -q` | 91 passed |
| `uv run pytest -q -m "not slow" -k "not test_support_00"` | 1629 passed |
| `uv run pytest -q -m "gpu and slow" -k "not test_support_00"` | 17 passed |
| the not-slow suite with CuPy hidden (`sys.modules["cupy"] = None`), as on the CPU-only CI runner | 898 passed, 563 skipped |
| `uv run ruff check src tests bench` | clean |

No kernel source changed, so `bench/selfcheck.py` was not re-run.

## Result

`--mode moonshots-m2-final`: one rep of every moonshot regime on one GPU and
on two, every knob unset. It took 0.037 GPU-hours.
- All 20 rows are ok, with no logged fallback.
- Every signature (itemsets, summed counts, itemset hash) equals the
  Phase 3 final tree's (`../2026-10-18-moonshots-phase3/final.jsonl`).
- The level split's K=3 phases show the kernel each regime ran. sk2ml3 and
  oom2ml3 ran row-wise (`count_rows` > 0) on one GPU and on two. The other
  16 ran dense, as before the switch.

## Commands

```bash
OUT=bench/results/2026-10-18-k3-dispatch-final
uv run python bench/runner.py --mode moonshots-m2-final --out $OUT --max-gpu-hours 0.085
```
