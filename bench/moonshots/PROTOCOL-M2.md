# M2 — Row-wise K=3: measurement protocol

The design is `SPEC.md` § M2 and its amendment; the code is
`src/et_miner/gpu/kernels/k3_rows.py` and `_count_k3_rows` in
`src/et_miner/gpu/row_split.py`, at `313dfc9` on `perf/moonshot-m2`. The
branch stacks on `perf/moonshot-m1` (the owner's choice), so both arms read
their input through M1's device input layer.

**Order note.** The crossover sweep ran first. Its procedure was fixed in
`k3_crossover.py`'s docstring, committed at `313dfc9` before it ran. This
file was written while it ran and before the campaign's first timed run.

## Box and budget

- The M1 box: 2× RTX 3060 12 GB, EPYC 7282, driver 580.173.02, CuPy 14.1.1.
  It is a shared host; `PROTOCOL-M1.md` records the details.
- Budget: ≤ 1 GPU-hour for M2 and M10 together (the owner's, per idea):
  - the sweep runs under `--max-gpu-hours 0.35`;
  - the campaign under `--max-gpu-hours 0.55`;
  - the final check under 0.1.

## M10 is not run

M10 (relabel items by ascending support) was the approved fallback arm. Its
stake was the dense K=3 generation on sk2ml3 (≤ 7.8 s) and oom2ml3 (≤ 0.51 s).
Row-wise K=3 counts both levels without generating the candidates. The
indicative runs at `313dfc9`: sk2ml3's K=3 21.7 → 0.62 s, oom2ml3's
1.33 → 0.10 s. On dsl neither order helps (`SPEC.md` § M10). M10 has no stake
left in the matrix, so it is not built.

## Crossover sweep (`k3_crossover.py`)

- Uniform rows on one GPU: N = 1M, F ∈ {50, …, 1600} items, L ∈ {8, 16, 32}.
- `min_count` is a third of the mean pair count, so nearly every pair is
  frequent and the slots approach the generated candidates.
- Each point runs dense and rows, a warm-up and then 3 interleaved reps; the
  K=3 level's median decides.
- `r3*` is the geometric mean at the lowest rows-faster → rows-slower
  crossing.
- `K3_ROWS_MAX_R` takes `r3*` before the campaign. If the points cross more
  than once, the report records it, and the constant takes the lowest
  crossing.

*Result* (`k3_crossover.jsonl`, 18 points, 0.141 GPU-hours): the points
cross more than once, so the constant takes the lowest crossing,
`K3_ROWS_MAX_R = 2.5e-4`. That crossing lies between F400-L8 (r3 = 1.69e-4,
rows 0.11× dense) and F800-L32 (3.73e-4, 1.05×). Every point below it is a
rows win (0.007–0.19×).

## Campaign arms

| arm | pin | K=3 |
|---|---|---|
| `dense` | `ET_MINER_K3_KERNEL=dense` | today's: prefix groups on the bitvecs (per-candidate / group / tiled by the measured dispatch) |
| `rows` | `ET_MINER_K3_KERNEL=rows` | row-wise into the slot layout; a level it does not serve (rows over 64 active items, `prune_apriori=False`, ESCO) falls back to the bitvecs and logs it |

Every config also pins:
- the thread pools to 16;
- `NCCL_P2P_DISABLE=1`;
- on one GPU, `CUDA_VISIBLE_DEVICES=0` and `ET_MINER_DISABLE_NCCL=1`.

Every other knob is unset, the input layer included.

## Regimes, repetitions, recording

- The moonshot regimes on one GPU and on two (as M1's).
- Fresh process per config, warm-up on smoke's 20,000-row slice with the same
  pins, 3 reps rep-major, medians with [min, max].
- Recorded per row:
  - the wall;
  - the level split, with `k3_graph` and `count_rows` at K=3;
  - peak VRAM and RSS;
  - throttle reasons;
  - logged fallbacks;
  - the signature.
- No tracked-file edits during a run.

## Decision rule (consolidation rules 1, 2, 4, 5; `decide.py --arms dense rows`)

1. A signature mismatch or a logged fallback the config does not allow is a
   must-fix. The rows arm's own fallback to the bitvecs (long rows) is a
   dispatch, not a fallback, and is logged at INFO.
2. Rows win a regime with a median wall ≥ 10 % below dense, disjoint
   [min, max] ranges and ≥ 1 s saved.
4. Rows that win some regimes and lose others stay behind the r3 dispatch
   (facts known before K=3: the active rows' pairs, the generated candidates,
   the words, the slots against one chunk). The dispatch must pick rows in
   every regime rows win, and dense in every regime rows lose; the final
   check confirms it.
5. If rows win nowhere, they go.

## Final check

One rep of every regime on one and two GPUs, with `ET_MINER_K3_KERNEL` unset
(the r3 dispatch). Each row logs which K=3 ran, and the signatures must match
the campaign's.

## Commands

```bash
OUT=bench/results/2026-10-17-moonshots-m2
uv run python bench/moonshots/k3_crossover.py --out $OUT/k3_crossover.jsonl --max-gpu-hours 0.35
uv run python bench/runner.py --mode moonshots-m2 --out $OUT --max-gpu-hours 0.55
uv run python bench/runner.py --mode moonshots-m2-final --out $OUT --max-gpu-hours 0.1
uv run python bench/moonshots/decide.py $OUT/raw.jsonl --arms dense rows > $OUT/decision.md
uv run python bench/moonshots/split_table.py $OUT/raw.jsonl > $OUT/split.md
```
