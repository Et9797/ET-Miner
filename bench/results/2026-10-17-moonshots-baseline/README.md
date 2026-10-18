# Moonshots Phase 0: baseline, wall split, instruction rates (2026-10-17)

`main` at `e4089a9` plus the harness of `ec3de12` (bench only; `src/` is
`main`'s). Every row of `raw.jsonl` is stamped `ec3de12`.

## Box

- 1× RTX A4000 16 GB (sm_86, GA104, 48 SMs), driver 535.146.02 (CUDA 12.2),
  CuPy 14.1.1 with NVRTC 12.9, `NCCL_P2P_DISABLE=1` pinned (one GPU, so no
  collective runs).
- Intel Pentium Gold G5420: 2 cores, 4 threads, 3.8 GHz, **no AVX, AVX2, FMA
  or BMI**. 31 GB RAM.
- Polars 2.0.0 runs on `polars-runtime-compat` (installed into the venv next to
  the locked `polars-runtime-32`, which needs AVX2 and dies with an illegal
  instruction on this CPU; Polars prefers the compat runtime when both are
  present). `pyproject.toml` and `uv.lock` are unchanged.
- Thread pools pinned to 4 (the logical CPU count) in every config.

This is a much weaker host than the 2× A4000 / Ryzen 5 5600X box of the
earlier campaigns, and the host phases (Polars, scipy) are slower here: dsl's
input phases take 18.6 s against about 10.5 s outside the levels there. The
GPU phases are the same card and agree with the earlier numbers (dsl K≥2 6.56
against 6.28 s, sk2ml3's tiled K=3 16.62 against 16.33 s). Nothing here is
compared with the other box.

**Not run here: every 2-GPU row** (the box has one GPU) and the 2-GPU legs of
the gates (86 tests skip with "needs 2 GPUs" / "needs 2 CUDA devices"). On a
2-GPU box, from the repository root after `bash bench/setup_box.sh`:

```bash
uv run pytest tests/test_tier_equivalence.py -q && uv run pytest -q -m "gpu and not slow"
OUT=bench/results/$(date +%F)-moonshots-baseline-2gpu
uv run python bench/runner.py --mode moonshots-baseline --out $OUT --only C2- --max-gpu-hours 0.6
uv run python bench/moonshots/split_table.py $OUT/raw.jsonl > $OUT/split.md
uv run python bench/moonshots/microbench.py --out $OUT/microbench.jsonl
```

## Gates (on this tree, before the run)

| check | result |
|---|---|
| `uv run pytest tests/test_tier_equivalence.py -q` | 32 passed, 22 skipped (all "needs 2 CUDA devices") |
| `uv run pytest -q -m "gpu and not slow"` | 390 passed, 87 skipped (86 two-GPU, 1 Online Retail not yet prepared at that point) |
| `uv run pytest -q -m "not slow" -k "not test_support_00"` | 1264 passed, 86 skipped (all two-GPU), 14 deselected |
| `uv run python bench/selfcheck.py` | READY: 13 kernels compiled and launched with the expected answer |
| runner signature check | 30 configs in 9 equivalence groups, consistent |

`-k "not test_support_00"` deselects the four Online Retail support tests,
as in every earlier gate (they do not finish with Online Retail present).

## Wall split (1 GPU, median of 3, seconds; `split.md`)

| regime | wall [min, max] | input before K=1 | K=1 | K≥2 levels | result frame + tail |
|---|---|---|---|---|---|
| smoke | 0.07 [0.07, 0.08] | 0.04 | 0.01 | 0.02 | 0.00 |
| deepk | 0.88 [0.86, 0.89] | 0.79 | 0.01 | 0.07 | 0.00 |
| skew | 1.10 [1.08, 1.12] | 1.02 | 0.01 | 0.07 | 0.00 |
| or003 | 0.16 [0.15, 0.18] | 0.10 | 0.01 | 0.05 | 0.00 |
| or002 | 0.31 [0.29, 0.37] | 0.10 | 0.01 | 0.20 | 0.01 |
| dsl | 25.21 [25.02, 25.52] | 18.57 | 0.01 | 6.56 | 0.03 |
| dsl, count inference | 21.54 [21.13, 22.17] | 18.57 | 0.01 | 2.92 | 0.02 |
| dsl, free sets | 21.09 [21.06, 21.10] | 18.42 | 0.01 | 2.64 | 0.01 |
| oom2ml3 | 2.48 [2.33, 2.54] | 1.05 | 0.02 | 1.39 | 0.01 |
| sk2ml3 | 23.22 [23.18, 23.33] | 5.08 | 0.08 | 18.03 | 0.04 |

The input phases (dsl / sk2ml3): host K=1 2.28 / 1.46, explode and join
6.53 / 1.53, COO to CSR 8.25 / 1.76, int64 copies 0.26 / 0.05,
host-to-device 1.02 / 0.21, bitvector kernel 0.15 / 0.08. The dataset read,
outside the wall, is 8.6 s on dsl.

The K≥2 counting (dsl / dsl inference / dsl free / sk2ml3): per-candidate
3.21 / 0.62 / 0.69 / 0.00, group 2.75 / 1.56 / 1.20 / 0.01, tiled 0.13 / 0.13
/ 0.13 / 16.62; sk2ml3's row-wise K=2 0.28 and filter 0.39.

## Instruction rates (`microbench.jsonl`, device 0, median of 5)

| probe | rate | per SM per clock (1,740 MHz) |
|---|---|---|
| `__popc` (32-bit) | 1.17e12 /s | 14.0 |
| `__popcll` (64-bit) | 5.82e11 /s | 7.0 |
| LOP3 (32-bit) | 4.63e12 /s | 55.4 |
| tiled kernel's inner loop, dense shared tiles | 5.93e11 slot-words/s | 7.1 |
| `mma.sync.m16n8k256 .b1 .and.popc` | 9.51e12 slot-words/s | 113.9 |
| `mma.sync.m8n8k128 .b1 .and.popc` | 9.48e12 slot-words/s | 113.5 |
| device copy | 357 GB/s | |
| host to device, pageable / pinned | 10.2 / 13.1 GB/s | |

A slot-word is one pair over 64 rows. The b1 tensor-core rate is 16.3× the
64-bit popcount rate, and the tiled kernel's inner loop already runs at the
popcount rate. `bmma_exact`: the m16n8k256 tile product equals `__popcll` on
all 36 cases (random, all-ones and sparse tiles; 1 to 1,001 words, padded to a
multiple of 4 with zeros).

The registered kernels on synthetic data (ns per unit per word):

| kernel (shape) | zero data (staging only) | sparse | dense |
|---|---|---|---|
| tiled, K=3, 64 suffixes, 31,250 words (per tile-pair) | 0.756 | 0.777 | 2.324 |
| group, K=8, 16 suffixes, 312,500 words (per pair) | 0.0036 | 0.0036 | 0.0063 |
| per-candidate, K=8, 4 suffixes, 312,500 words (per pair) | 0.0076 | 0.0078 | 0.0584 |

sk2ml3's tiled K=3 level runs at 16.62 s / (376,436 counted tile-pairs ×
31,250 words) = 1.41 ns per tile-pair per word: between staging alone (0.76)
and a dense count (2.32).

## Dumps

Rep 0 of every one-GPU config wrote `<config>.lattice.parquet` and its
`.lattice.json` sidecar: the input of the Phase 1 stake scripts. They are
committed (15 MB): `!bench/results/**` in `.gitignore` overrides `*.parquet`,
which earlier campaigns kept out by hand. The per-shard counts M9 needs are computed exactly
on the CPU from the dataset and the miner's row cuts
(`bench/moonshots/k3_rows_stake.py`), so no 2-GPU dump is needed for them.

## Commands

```bash
OUT=bench/results/2026-10-17-moonshots-baseline
uv run python bench/runner.py --mode moonshots-baseline --out $OUT --max-gpu-hours 0.8
uv run python bench/moonshots/microbench.py --reps 5 --out $OUT/microbench.jsonl
uv run python bench/moonshots/split_table.py $OUT/raw.jsonl > $OUT/split.md
```

GPU time: 0.14 GPU-hours for the campaign (Σ proc_s) plus about 0.01 for the
probes.
