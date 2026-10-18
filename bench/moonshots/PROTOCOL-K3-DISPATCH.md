# K=3 dispatch statistic: measurement protocol

`REPORT.md` Findings 4 (`REPORT-M2.md` finding 1): the row-wise K=3 dispatch
reads r3 = R2 / (generated × words), the row pairs over the dense level's
work. It leaves out the row-wise kernel's second cost, the increments: one
per row triple whose three pairs are frequent. This protocol asks whether

  s = (R2 + ρ² T3) / (generated × words)

picks the faster K=3 kernel more often, and by how many milliseconds. Fixed
before the first timed run, on `perf/m2-dispatch-stat`.

- R2 and T3 are Σ_rows C(len, 2) and C(len, 3) over the active columns (the
  columns of some frequent pair).
- ρ = P2 / R2, with P2 the K=2 level's summed counts: the share of the row
  pairs that are frequent pairs. ρ² T3 estimates the increments.
- words = ceil(N / 64).

## The tree

`main` at `37819e2`, plus commit `f7acff3` on this branch:
- `row_split_chunks.k3_rows_statistic` computes s;
- the row sources give T3 beside R2 (`pairs_triples`);
- the miner logs r3, s and ρ at K=3, whichever kernel runs.

**The dispatch is unchanged:** rows when r3 < `K3_ROWS_MAX_R = 2.5e-4` and the
slots fit one chunk. The sweep pins the kernel on every timed run, so the
statistic only labels the points.

**The denominator is the generated candidates, not the slots.** The report
wrote slots. The dense level iterates the generated candidates
(`build_k3plus_groups_from_flat`'s prefix groups, Σ_u C(deg⁺(u), 2)), and
r3 divides by them too. On M2's sweep every pair is frequent, so slots and
generated candidates are equal there. On Zipf rows they differ by 4–5×. The
slots' variant (`s_slots`) is reported but has no say in the decision.

## What is known before the run

- **M2's sweep at `313dfc9`** (`../results/2026-10-17-moonshots-m2/k3_crossover.jsonl`,
  18 uniform points, ρ = 1). Sorted by s, rows win every point up to
  s = 5.1e-4 (0.007–0.19×). The three points at s ≈ 4.1e-3 are mixed (0.27,
  0.79 and 1.05×). From s ≈ 0.033 on, all but one point lose; the exception
  is a 21 ms tie at 0.94×.
- **On that grid, s ≈ 64 (L / F)³**, so a row length and an item count with the
  same ratio land at the same s. The grid cannot separate them, and its only
  ρ is 1.
- **What r3 implies for long rows.** With ρ = 1, s / r3 = (L + 1) / 3. At r3's
  threshold, s is 7.5e-4 for L = 8 but 5.4e-3 for L = 64.
  - For short rows, r3 sends points to dense where s says rows.
  - For long rows, r3 admits points where s says dense: uniform F1600-L64 has
    r3 = 1.9e-4 (rows today) but s = 4.1e-3.
  - At F1600, a rough cost model puts that point at about 34 s row-wise
    against 22 s dense (0.8 ns per increment, from F1600-L32's 4.06 s for
    4.96e9 increments).
- **Planning on the CPU** (pair counts of the generated rows; no GPU, not
  campaign data). On Zipf rows, ρ is 0.91–0.92 at `min_count` = a third of the
  mean pair count.
  - On uniform rows with `min_count` at the mean pair count, ρ is 0.52–0.54.
  - A dry run of the child on a 200k-row Zipf point (ρ = 0.80) estimated
    71.4M increments by ρ² T3 against 65.2M counted.

## Box and budget

- The M1 box: 2× RTX 3060 12 GB, EPYC 7282, driver 580.173.02, CuPy 14.1.1.
  It is a shared host.
- One GPU (device 0), NCCL off, thread pools at 16, as in M2's sweep.
- Budget ≤ 1 GPU-hour for the item:
  - the sweep runs under `--max-gpu-hours 0.7` (estimated 0.45–0.55);
  - the final check under 0.1.

## The sweep (`k3_crossover.py --grid dispatch`, 43 points)

Each point is N rows of L distinct items, drawn from F items. `min_count` is a
third of the mean pair count unless the id ends in `-c1` (the mean pair
count itself). `max_length = 3`.

| group | points | why |
|---|---|---|
| M2's grid, re-measured | uniform N1M, F 50–1600, L 8 / 16 / 32 (18) | the old points at this tree's speed |
| long rows | uniform N1M: F800-L24, F1200-L32 / 40 / 48, F1600-L40 / 48 / 56 / 64 | r3 admits all but F1200-L48; s ranges 1.0e-3–4.1e-3 |
| short rows, 4M rows | uniform N4M: F200 / 250 / 300 / 400-L8, F450 / 600-L12 | r3 declines four of them; s ranges 5.1e-4–4.1e-3 |
| ρ about 1/2 | uniform N1M, `-c1`: F800-L32, F1600-L32 / 48 / 64 | the ρ² term away from 1 |
| Zipf | N1M: F800-L16 / 32, F1600-L16 / 24 / 32 / 48, F2400-L32 | correlated pairs, generated ≫ slots |

Per point, in a fresh process:
- an untimed warm-up of each kernel on the point's first 20,000 rows. M2
  warmed up on the whole point; every run releases the pool first, so only
  the kernels' compilation carries over.
- Then 3 interleaved reps (dense, rows, dense, …), `ET_MINER_K3_KERNEL`
  pinned. The K=3 level's time (`level_callback`) is recorded; the point's
  value is its median per kernel.
- When ρ < 1, one more untimed rows run sums the kernel's slot counts: the
  increments. When ρ is 1, every row triple is a candidate, so the increments
  are T3.

Recorded per point:
- the generated candidates, the slots, R2, T3, ρ, r3 and s, from the miner's
  log;
- the increments;
- `s_slots` and `s_true` = (R2 + increments) / (generated × words);
- the K=3 times of every rep;
- the signature.

No tracked-file edits or commits while it runs.

## Validity

- Both kernels must mine the same itemsets with the same counts, the miner
  must log the row-wise K=3 exactly on the rows runs, and no run may log a
  fallback. A point that fails these is a must-fix: the item stops there.
- A point that errors or times out (900 s) is excluded and listed. If more
  than 3 do, the analysis is not run, and the report says why.

## Analysis: regret

For a statistic X and a threshold θ, the dispatch picks rows at a point when
X < θ. Every point of the sweep has its slots in one chunk.
- **Regret** is Σ over the points of (the chosen kernel's median − the
  faster one's), in ms.
- **The fitted threshold** of X is the θ of least regret among the
  geometric means of adjacent points sorted by X, plus 0 and ∞. On a tie it
  is the lowest.
- The analysis (`k3_crossover.py --analyze`) reports, from the same 43 points:
  - B0, r3 at today's 2.5e-4;
  - B1, r3 at its fitted threshold;
  - P, s at its fitted threshold;
  - `s_slots` and `s_true` at theirs, descriptive only;
  - for each, the interval of thresholds within 5 % of the least regret.

## Decision rule

1. **s replaces r3** if regret(P) ≤ 0.9 × regret(B1) *and*
   regret(B0) − regret(P) ≥ 1,000 ms. The constant `K3_ROWS_MAX_S` is then
   P's threshold, rounded to two significant digits inside its 5 % interval.
   The one-chunk condition and the pins stay.
2. Otherwise r3 stays at 2.5e-4. Refitting r3 alone is not this item.
3. **Matrix guard.** With s adopted, the final check runs one rep of every
   moonshot regime on one and two GPUs, `ET_MINER_K3_KERNEL` unset
   (`--mode moonshots-m2-final`). The dispatch must pick rows in sk2ml3 and
   oom2ml3 and dense in the other 16, as now, and every signature must equal
   Phase 3's. If a regime switches kernels, the switch needs a campaign of its
   own. Then nothing is adopted on this evidence, and the item stops for the
   owner.

## Commands

```bash
OUT=bench/results/2026-10-18-k3-dispatch
uv run python bench/moonshots/k3_crossover.py --grid dispatch --out $OUT/k3_crossover.jsonl --max-gpu-hours 0.7
uv run python bench/moonshots/k3_crossover.py --grid dispatch --out $OUT/k3_crossover.jsonl --analyze > $OUT/k3_crossover.txt
# only with s adopted, on the tree that switches the dispatch:
uv run python bench/runner.py --mode moonshots-m2-final --out $OUT --max-gpu-hours 0.1
```
