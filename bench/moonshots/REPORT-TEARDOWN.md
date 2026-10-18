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
