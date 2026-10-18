# The 2-GPU tail: report

Protocol: `PROTOCOL-TEARDOWN.md`. Run: `../results/2026-10-18-moonshots-teardown/`
(`raw.jsonl`, `tail.md`, `split.md`), 30 rows at `8d7d317`, every row ok, no
fallback, every signature equal to Phase 3's 2-GPU rows, 0.126 GPU-hours.

**The tail is the NCCL communicators' destroy.** On two GPUs the miner drops
its two communicators at its return. NCCL 2.31 then waits for its proxy
thread, which wakes every 500 ms when idle. So the call ends 0–0.5 s later
than its work, by chance: the wait is 500 ms minus the time since the last
NCCL call, modulo 500 ms.

| config | wall s | tail s | tail_nccl s | tail_other s | destroy / wall |
|---|---|---|---|---|---|
| smoke | 0.567 [0.554, 0.568] | 0.364 [0.334, 0.372] | 0.363 [0.333, 0.371] | 0.001 [0.001, 0.001] | 64% |
| deepk | 0.632 [0.626, 0.648] | 0.245 [0.234, 0.277] | 0.244 [0.233, 0.276] | 0.001 [0.001, 0.001] | 39% |
| skew | 0.634 [0.630, 0.649] | 0.278 [0.278, 0.317] | 0.277 [0.277, 0.316] | 0.000 [0.000, 0.001] | 44% |
| or003 | 0.582 [0.574, 0.584] | 0.256 [0.252, 0.260] | 0.253 [0.248, 0.257] | 0.001 [0.001, 0.001] | 43% |
| or002 | 1.071 [0.565, 1.085] | 0.505 [0.041, 0.528] | 0.490 [0.028, 0.513] | 0.004 [0.004, 0.004] | 46% |
| dsl | 3.234 [3.227, 3.243] | 0.260 [0.212, 0.288] | 0.248 [0.200, 0.276] | 0.001 [0.001, 0.001] | 8% |
| dsl-infer | 2.720 [2.719, 2.724] | 0.400 [0.380, 0.423] | 0.391 [0.371, 0.414] | 0.001 [0.001, 0.001] | 14% |
| dsl-free | 2.718 [2.718, 2.751] | 0.366 [0.326, 0.367] | 0.362 [0.323, 0.363] | 0.001 [0.001, 0.001] | 13% |
| oom2ml3 | 1.121 [1.120, 1.127] | 0.451 [0.445, 0.473] | 0.442 [0.438, 0.466] | 0.002 [0.002, 0.003] | 39% |
| sk2ml3 | 1.822 [1.814, 2.322] | 0.107 [0.096, 0.553] | 0.075 [0.063, 0.520] | 0.007 [0.007, 0.007] | 4% |
| all 30 rows | 1.103 [0.554, 3.243] | 0.322 [0.041, 0.553] | 0.319 [0.028, 0.520] | 0.001 [0.000, 0.007] | |

Medians with [min, max] over 3 reps, two GPUs, every knob unset.

## Findings

1. **(should)** *The tail is named.* Every config's median `tail_other` is at
   most 0.007 s, under the one-GPU maximum of 0.027 s (protocol rule 2). What
   remains of the tail is the result frame, up to 0.03 s on sk2ml3. The
   destroy takes 0.319 s at the median of all rows: 64 % of smoke's wall,
   39–46 % of the other small regimes', 8–14 % of the dsl regimes'.
2. **(should)** *Moving the destroy off the call's path would let two GPUs pay
   on the dsl regimes.* Subtracting `tail_nccl` from each row (an estimate,
   not a measurement) against Phase 3's one-GPU rows on this box:

   | regime | 1 GPU | 2 GPUs | 2 GPUs − destroy |
   |---|---|---|---|
   | dsl | 3.86 [3.85, 3.90] | 3.23 [3.23, 3.24] | 2.99 [2.97, 3.03] |
   | dsl-free | 2.76 [2.73, 2.77] | 2.72 [2.72, 2.75] | 2.39 [2.36, 2.40] |
   | dsl-infer | 2.40 [2.35, 2.50] | 2.72 [2.72, 2.72] | 2.33 [2.31, 2.35] |
   | sk2ml3 | 1.77 [1.74, 1.78] | 1.82 [1.81, 2.32] | 1.76 [1.74, 1.80] |
   | oom2ml3 | 0.38 [0.37, 0.39] | 1.12 [1.12, 1.13] | 0.68 [0.66, 0.68] |
   | smoke | 0.06 [0.06, 0.07] | 0.57 [0.55, 0.57] | 0.20 [0.20, 0.22] |

   The small regimes stay slower on two GPUs. Candidates, from scratch
   scripts (NCCL 2.31.2, `NCCL_P2P_DISABLE=1`, 4 reps each):
   - destroy on a background thread: the call returns at once; Python
     work beside it runs at 0.25 s against 0.23 s alone, and the destroy
     ends 0.39–0.49 s later;
   - `abort()` in place of `destroy()`: slower, 0.54–0.56 s;
   - keeping the communicators for the next call: the destroy moves to the
     process's exit, and a second call skips the init. It holds their
     memory between calls, and a failed collective would have to drop them.

   Each needs its own protocol and the owner's word.
3. **(nice)** *The next fixed cost on two GPUs is the reduce.* On smoke, deepk
   and skew the per-level reduce takes 0.07–0.09 s of 0.14–0.24 s of levels.
   In the scratch script the first all-reduce on new communicators took
   0.06–0.08 s for 4 MB. NCCL init itself is 0.03 s here, because the warm-up
   call pays NCCL's first init (0.6 s) in the same process.
4. **(nice)** *or002's 2-GPU median moved with the destroy, not with its
   work.* It was 0.58 s in Phase 3 and is 1.07 s here: one rep of three
   waited 0.03 s and two waited 0.49–0.51 s. Its levels are the same.
5. **(nice)** The dsl rows ran under the SW power cap (0x4) on all three
   reps, as in M1. It is recorded, not controlled.

## Amendment 1: the destroy off the call's path

Change: the miner destroys its communicators on a thread of its own once its
levels are done (`09d08fd`, `PROTOCOL-TEARDOWN.md` Amendment 1). Run:
`../results/2026-10-18-moonshots-teardown-after/` (`decision.md`, `tail.md`),
30 rows at `09d08fd`, every row ok, no fallback, every signature equal to
Phase 3's, 0.118 GPU-hours.

**Every 2-GPU regime got faster, by about the destroy it no longer waits
for.** The decision rules hold:
- signatures match and no row logged a fallback;
- `tail_nccl` is 0 in every row, and `tail_other` is at most 0.007 s;
- no regime loses.

Rule 2 calls every regime a tie, because it asks for 1 s saved.

| regime | before s | after s | after / before | before `tail_nccl` s | 1 GPU (Phase 3) s |
|---|---|---|---|---|---|
| smoke | 0.57 [0.55, 0.57] | 0.20 [0.20, 0.22] | 0.36 | 0.36 | 0.06 [0.06, 0.07] |
| deepk | 0.63 [0.63, 0.65] | 0.37 [0.35, 0.39] | 0.58 | 0.24 | 0.15 [0.15, 0.17] |
| skew | 0.63 [0.63, 0.65] | 0.37 [0.35, 0.39] | 0.58 | 0.28 | 0.17 [0.16, 0.18] |
| or003 | 0.58 [0.57, 0.58] | 0.32 [0.32, 0.35] | 0.55 | 0.25 | 0.11 [0.10, 0.12] |
| or002 | 1.07 [0.56, 1.08] | 0.55 [0.51, 0.59] | 0.52 | 0.49 | 0.27 [0.27, 0.27] |
| dsl | 3.23 [3.23, 3.24] | 3.02 [2.98, 3.09] | 0.93 | 0.25 | 3.86 [3.85, 3.90] |
| dsl-infer | 2.72 [2.72, 2.72] | 2.33 [2.29, 2.39] | 0.86 | 0.39 | 2.40 [2.35, 2.50] |
| dsl-free | 2.72 [2.72, 2.75] | 2.40 [2.38, 2.43] | 0.88 | 0.36 | 2.76 [2.73, 2.77] |
| oom2ml3 | 1.12 [1.12, 1.13] | 0.69 [0.69, 0.70] | 0.62 | 0.44 | 0.38 [0.37, 0.39] |
| sk2ml3 | 1.82 [1.81, 2.32] | 1.79 [1.78, 1.82] | 0.98 | 0.08 | 1.77 [1.74, 1.78] |

### Findings

1. **(should)** *The change stays.* Two GPUs now beat one on dsl (3.02
   against 3.86 s) and dsl-free (2.40 against 2.76 s) with disjoint ranges.
   They tie on dsl-infer (2.33 against 2.40 s, overlapping ranges) and sk2ml3
   (1.79 against 1.77 s). The small regimes stay slower on two GPUs (0.20–0.69
   against 0.06–0.38 s): the reduce of finding 3 above and the levels' fixed
   costs remain.
2. **(nice)** *The 2-GPU walls no longer depend on chance.* or002's range was
   0.56–1.08 s; it is 0.51–0.59 s now. sk2ml3's 2.32 s outlier is gone.
3. **(nice)** The two runs are about two hours apart on a shared host. Each
   wall moved by about the before rows' `tail_nccl`, between 0.21 and 0.52 s.
   Two of the three dsl rows ran under the SW power cap (0x4), as before.
