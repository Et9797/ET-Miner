# Route optimizations, phase C (O6) — report

Phase C of `SPEC.md` on `perf/route-optimizations-c`. Protocol, fixed before the
first timed run: `PROTOCOL-C.md`. O7 was dropped before it (SPEC, O7
amendment). Evidence: `bench/results/2026-10-06-group-kernel/`:

- `group_crossover.jsonl`: the kernel sweep, at `910ec52`.
- `raw.jsonl`, `final.jsonl` and `env.txt`: the campaign and the final check,
  at `238e95a` (the sweep's table, nothing else changed).

The offline stakes are in `bench/results/2026-10-06-phase-c-stake/`.
`uv run python bench/optimizations/decide_c.py
bench/results/2026-10-06-group-kernel/raw.jsonl
bench/results/2026-10-06-group-kernel/final.jsonl` prints the tables below.

Box: 2× RTX A4000 16 GB (sm_86, `NCCL_P2P_DISABLE=1`), Ryzen 5 5600X, 46 GB
RAM, idle throughout.

## What changed

- **Group kernel** (`gpu/kernels/_src/group_pairs.cu`, wrapper
  `gpu/kernels/group_pairs.py`):
  - One block per prefix group of at most 64 suffixes. The block classifies its
    pairs with the subset test and lists the pairs to count. It stages the
    prefix AND and the used suffix rows one 32-word tile at a time, then counts
    only the listed pairs.
  - Up to 128 listed pairs, lanes are words, with one register accumulator per
    pair; above that, threads are pairs.
  - It writes the per-candidate kernel's entries, so the compacted reduce and
    the chunk loop apply unchanged.
- **Three-way dispatch** (`row_split_chunks.group_kernels`), per prefix group of
  a dense K≥3 level:
  - per-candidate below `GROUP_MIN_PAIRS[k]`;
  - the group kernel up to `GROUP_TILED_MIN_PAIRS[k]`;
  - tiled above that, and for every group of more than 64 suffixes.
  - `ET_MINER_SMALL_GROUP_KERNEL=percand` pins today's two-way dispatch;
    `=group` pins the group kernel below the tiled crossover.
- **Tier chain** (`CLAUDE.md`, `tests/test_tier_equivalence.py`):
  - the group kernel pinned on 1 GPU (plain, forced chunks, count inference,
    no subset test) and on 2 GPUs (plain, forced chunks, count inference);
  - the default leg asserts that the group kernel runs on smoke.
- **Bench**:
  - `group_crossover.py`;
  - `runner.py --mode o6 | o6-final`;
  - the level-split phase `count_group`;
  - `decide_c.py`;
  - `--mode waste` pins `percand`, the dispatch `candidate_waste.py` models.

## Step 1: kernel sweep

Synthetic groups, every pair counted, 1,000 groups per point, median of 3
interleaved reps per kernel. Times are per launch at 312,500 words.

| K | group / tiled, m = 2 … 64 | per-candidate faster up to | table: per-candidate below / tiled from |
|---|---|---|---|
| 3 | 0.12 – 0.77 (every m) | m = 6 (15 pairs) | 20 / 2080 |
| 4 | 0.16 – 0.77 (every m) | m = 4 (6 pairs) | 9 / 2080 |
| 5 | 0.19 – 0.83 (every m) | m = 4 | 9 / 2080 |
| 6 | 0.23 – 0.95 (every m) | m = 4 | 9 / 2080 |
| 7 | — | — | 9 / 877 (geometric mean of K=6 and K=8) |
| 8 | 0.41 – 0.86 up to m = 24; 1.16 at 32, 0.80 at 48, 1.18 at 64 | m = 4 | 9 / 370 |

2080 = C(65, 2): at K=3–6 the 64-suffix cap decides. The 31,250-word lines give
the same table (reading 1 holds).

On dsl the new dispatch sends the generated candidates as follows:

| dispatch | per-candidate | group | tiled |
|---|---|---|---|
| new | 224,745 | 667,258 | 3,081 |
| today | 460,153 | — | 434,931 |

## Step 2: campaign (DP-O6, wall s, median [min, max] over 3 reps)

All 84 configs returned ok, with no fallback. The signatures form 9
equivalence groups, every row at `238e95a`.

| regime | base | group | rule 2 |
|---|---|---|---|
| dsl, 1 GPU | 20.19 [19.87, 20.21] | 16.76 [16.73, 16.82] | **group wins** (−3.43 s) |
| dsl, count inference, 1 GPU | 15.16 [15.12, 15.20] | 13.16 [12.99, 13.19] | **group wins** (−2.00 s) |
| dsl, 2 GPUs | 15.33 [15.32, 15.39] | 13.45 [13.37, 13.83] | **group wins** (−1.88 s) |
| dsl, free sets, 1 GPU | 14.05 [13.97, 14.14] | 12.76 [12.66, 12.87] | tie (−1.29 s, 9.2 %) |
| dsl, count inference, 2 GPUs | 12.93 [12.92, 12.94] | 11.87 [11.85, 11.91] | tie (−1.06 s, 8.2 %) |
| sk2ml3, 1 GPU | 20.52 [20.51, 20.53] | 20.73 [20.72, 20.78] | tie (+0.21 s) |
| sk2ml3, 2 GPUs | 16.09 [16.08, 16.10] | 16.08 [16.05, 16.17] | tie (−0.00 s) |
| oom2ml3, 1 GPU | 1.81 [1.81, 1.82] | 1.86 [1.86, 1.86] | tie (+0.05 s) |
| oom2ml3, 2 GPUs | 2.62 [2.62, 2.63] | 2.61 [2.60, 2.65] | tie (−0.01 s) |
| smoke, deepk, skew, or003, or002, 1 GPU | 0.07 – 0.71 | 0.08 – 0.72 | tie (−0.03 to +0.02 s) |

**Outcome.** `group` wins three regimes and loses none. The sweep's dispatch is
the default; `ET_MINER_SMALL_GROUP_KERNEL=percand` stays as the pin. Peak VRAM
is the same in every regime.

Where dsl's K≥3 counting went (1 GPU, median, s):

| arm | per-candidate | group | tiled | total |
|---|---|---|---|---|
| base | 5.28 | — | 4.17 | 9.45 |
| group | 3.23 | 2.74 | 0.06 | 6.03 |

## Final check (knobs unset, one rep, `238e95a`)

All 14 regimes were ok, and every signature matches the campaign's. Selected
walls:

| regime | wall |
|---|---|
| dsl, 1 GPU | 16.75 |
| dsl, 2 GPUs | 13.40 |
| dsl, count inference, 1 GPU | 13.18 |
| dsl, count inference, 2 GPUs | 11.92 |
| dsl, free sets, 1 GPU | 12.62 |
| sk2ml3, 1 GPU | 20.34 |
| sk2ml3, 2 GPUs | 16.06 |

## Findings

1. **(should) The per-candidate kernel still takes 3.23 s on dsl with one
   GPU.** It counts the smallest groups (below 9 pairs from K=4, that is ≤ 4
   suffixes). The sweep put the per-candidate kernel ahead there, but it
   counted every pair. On dsl the subset test skips 35–45 % of the big
   levels' candidates, so whether the group kernel counts those groups faster
   on real levels is not measured. A pinned run settles it cheaply:
   `ET_MINER_SMALL_GROUP_KERNEL=group` on dsl, about 17 s per GPU count.
2. **(should) The measured gain is about half the offline model's.** The stake
   model gave −6.1 s on dsl with one GPU, assuming today's per-row and per-slot
   rates (`../results/2026-10-06-phase-c-stake/README.md`); the campaign
   measured −3.43 s. The model itself flagged that a kernel with fewer
   redundant reads gets fewer cache hits.
3. **(nice) At K=8 the group and tiled kernels cross more than once.** The
   tiled kernel's cost steps at 33 suffixes (from 1 to 3 tile-pairs), so the
   group kernel wins again at m = 48 (0.80) and loses at m = 64 (1.18). The
   table uses the first crossing (370 pairs). On dsl only 3,081 candidates go
   tiled in total.
4. **(nice) sk2ml3 on one GPU is 0.21 s slower on `group` (+1.0 %, ranges
   apart).**
   - The difference is all in the tiled part (16.48 → 16.70 s); the group
     kernel itself takes 0.006 s there.
   - HW slowdown and thermal (reason 0x48) hit all three `group` reps of this
     regime and none of the `base` reps. Within a rep `base` always runs just
     before `group`.
   - The final check, on the `group` default, took 20.34 s.
5. **(nice) oom2ml3 on one GPU is 0.05 s slower on `group`.** The difference
   is in K=1 (0.022 → 0.064 s), which the group kernel does not touch.
6. **(nice) Two dsl regimes miss rule 2's 10 % bar:** free sets (−1.29 s,
   9.2 %) and count inference on two GPUs (−1.06 s, 8.2 %). Neither loses.

## Budget

| step | GPU-hours |
|---|---|
| sweep | 0.155 |
| campaign | 0.384 |
| final check | 0.062 |
| total | 0.601 of the 1.5 agreed |

The offline stakes used no GPU time.
