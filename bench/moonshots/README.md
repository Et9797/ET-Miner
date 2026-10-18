# Moonshots for the GPU route

Ideas the incremental SPEC (`../optimizations/SPEC.md`, O1–O8) could not reach,
each exact or out. Phase 0 measures the baseline and the instruction rates,
Phase 1 puts a stake on every idea (`STAKES.md`, `SPEC.md`), Phase 2 builds the
ones the owner picks, one pre-registered protocol per idea
(`PROTOCOL-<id>.md`).

## Phase 0: baseline, wall split, instruction rates, dumps

On the GPU box, from the repository root, with the datasets generated
(`python -m et_miner.synthetic --preset all --out datasets/synth`,
`datasets/prepare_online_retail.py`):

```bash
OUT=bench/results/$(date +%F)-moonshots-baseline
uv run python bench/runner.py --mode moonshots-baseline --out $OUT --max-gpu-hours 0.8
uv run python bench/moonshots/split_table.py $OUT/raw.jsonl > $OUT/split.md
uv run python bench/moonshots/microbench.py --out $OUT/microbench.jsonl
```

- `--mode moonshots-baseline` runs every regime at the defaults (knobs unset),
  3 reps rep-major, on one GPU and, where the box has two, on two. Each config
  pins the thread pools to the box's logical CPU count and `NCCL_P2P_DISABLE=1`,
  and records the level split (`../level_split.py`) with the span before K=1
  and after the last level (`timings["span"]`: the Polars collects, COO to CSR,
  the int64 copies, the host-to-device copies, the bitvector kernel, the
  result frame) and the dataset read outside the wall (`timings["read_s"]`).
- Rep 0 on one GPU dumps each regime's lattice (`<config>.lattice.parquet`),
  the input of every Phase 1 stake script.
- `microbench.py` measures the instruction rates the kernels sit on (see its
  docstring) on every device.

## Phase 1: stakes (CPU only)

Each script reads a lattice dump of Phase 0 (and the dataset where it says
`--dataset`); see each docstring.

| script | ideas |
|---|---|
| `split_table.py` | M1, X0, the wall split of every regime |
| `k3_rows_stake.py` | M2 at K=3 (exact row-wise increments), M9 (per-shard counts) |
| `replay_stakes.py` | M2 bound per level, M5a (kept rows), M7 (deduction rules), M8 (look-ahead), kernel work |
| `k3_census.py` | M10 on the K=3 explosions (item orders), M2's bound at K=3 |
| `component_stakes.py` | M4 (item components per starting level) |
| `tile_stakes.py` | M5b and M12 (all-zero word tiles per row order) |
| `dedup_stakes.py` | M11 (distinct projected rows) |
| `input_floor.py` | M1's host floor (explode, row lengths) |
| `relabel_lattice.py` | M10's relabelled lattice for the replays |
| `preset_check.py` | M4's proposed `deep_k20` preset, mined on the CPU |
| `stakes.py` | every idea's upper bound from the outputs above |

The outcome is `SPEC.md` (one section per idea) and `STAKES.md` (the ranked
table); the outputs and their commands are in
`../results/2026-10-17-moonshots-stakes/README.md`.

## Follow-ups after the merge

One branch off `main` per item of `REPORT.md`, with its own protocol before
any timed run.

| protocol | question | summary script |
|---|---|---|
| `PROTOCOL-TEARDOWN.md` | where the 2-GPU tail goes (Findings 2) | `tail_table.py` |
