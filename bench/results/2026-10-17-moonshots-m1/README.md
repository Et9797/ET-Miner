# M1 campaign: host against device input layer (2026-10-17)

`perf/moonshot-m1` at `6884f78`; every row of `raw.jsonl` is stamped
`6884f78`. Protocol: `../../moonshots/PROTOCOL-M1.md`; report and decision:
`../../moonshots/REPORT-M1.md`.

## Box

- 2× RTX 3060 12 GB (sm_86, topology NODE), driver 580.173.02 (CUDA 13.0),
  CuPy 14.1.1 / NVRTC 12.9, `NCCL_P2P_DISABLE=1` pinned.
- AMD EPYC 7282 (16 cores, 32 threads, AVX2), 125 GB RAM; thread pools pinned
  to 16. A shared host (load average about 6.5 from other tenants), and the
  owner's `watch -n 0.1 nvidia-smi` running throughout.
- Not the Phase 0 box; nothing here is compared with Phase 0.

## Gates (on this tree, before the run)

| check | result |
|---|---|
| `uv run pytest tests/test_tier_equivalence.py -q` | 66 passed (2-GPU legs included) |
| `uv run pytest -q -m "gpu and not slow"` | 576 passed |
| `uv run pytest -q -m "not slow and not gpu" -k "not test_support_00"` | 886 passed |
| `uv run pytest tests/test_input_layer.py -q -m slow` | 12 passed (every large preset, 1 and 2 GPUs) |
| `uv run python bench/selfcheck.py` | READY: 14 kernels on 2 devices |
| `uv run ruff check src tests bench` | clean |

## Files

- `raw.jsonl`: 120 campaign rows (3 reps, rep-major) and 20 final-check rows
  (`*-final#r0`, the layer unset); `*.result.json` per row.
- `decision.md`: `decide.py --arms host device` (rule 2 per regime; must-fix: none).
- `split.md`: the wall split per config, the device phases included.
- `env.txt`: the environment capture of both invocations.

## Commands

```bash
OUT=bench/results/2026-10-17-moonshots-m1
uv run python bench/runner.py --mode moonshots-m1 --out $OUT --max-gpu-hours 0.95
uv run python bench/runner.py --mode moonshots-m1-final --out $OUT --max-gpu-hours 0.3
uv run python bench/moonshots/decide.py $OUT/raw.jsonl --arms host device > $OUT/decision.md
uv run python bench/moonshots/split_table.py $OUT/raw.jsonl > $OUT/split.md
```
