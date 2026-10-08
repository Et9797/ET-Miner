# SON on the array miner: Phase S0 (stakes)

Protocol: `bench/cpu/PROTOCOL.md`, Amendment 3. Harness: `bench/cpu/son_stakes.py`
at `6dabc9e`; every row carries that rev with a clean tree. One rep, cap 600 s
per config, 4 chunks, `local_support_factor` 0.9. Box: AMD Ryzen 5 4600G, 12
logical CPUs, 30 GB (`env.txt`). Rows: `raw.jsonl` (48: 6 workloads × T1/T4 ×
4 arms).

## Correctness

Every workload has one signature across all ok rows (`son_stakes.py --check`):
smoke, deepk, skew, wide, or005, or0001k2. The pass-1 candidate counts of
`current` and the array arms agree wherever `current` finished.

## Wall time and peak RSS

`cap` = the 600 s cap was hit (a lower bound; the cap includes load and warm-up).

| regime | current | array | array-pc | incore |
|---|---|---|---|---|
| smoke T1 | 1.47 s / 177 MB | 0.17 s / 131 MB | 0.16 s / 130 MB | 0.09 s / 146 MB |
| smoke T4 | 0.96 s / 198 MB | 0.17 s / 135 MB | 0.14 s / 137 MB | 0.12 s / 162 MB |
| deepk T1 | 31.17 s / 374 MB | 4.93 s / 300 MB | 5.18 s / 296 MB | 2.13 s / 363 MB |
| deepk T4 | 15.14 s / 386 MB | 4.52 s / 338 MB | 4.66 s / 339 MB | 1.70 s / 399 MB |
| skew T1 | 157.01 s / 550 MB | 31.87 s / 372 MB | 42.96 s / 382 MB | 2.76 s / 410 MB |
| skew T4 | 78.97 s / 552 MB | 15.37 s / 449 MB | 35.19 s / 453 MB | 1.75 s / 461 MB |
| wide T1 | 267.89 s / 1,361 MB | 1.21 s / 234 MB | 1.24 s / 251 MB | 0.87 s / 267 MB |
| wide T4 | 235.92 s / 1,732 MB | 0.66 s / 232 MB | 0.65 s / 229 MB | 0.38 s / 266 MB |
| or005 T1 | cap | 21.83 s / 257 MB | 13.47 s / 254 MB | 0.43 s / 192 MB |
| or005 T4 | cap | 21.18 s / 282 MB | 13.20 s / 280 MB | 0.28 s / 200 MB |
| or0001k2 T1 | cap | 7.13 s / 774 MB | 7.12 s / 775 MB | 0.82 s / 332 MB |
| or0001k2 T4 | cap | 7.34 s / 867 MB | 7.39 s / 865 MB | 0.56 s / 384 MB |

## Rule 2: go for the port

**Go.** Both array arms meet rule 2 against `current` in every regime but one:
3.2× (deepk T4, `array-pc`) to 360× (wide T4) faster where `current` finished,
and below half the cap where it did not. The exception is smoke T4, 5.6× faster
but 0.79 s saved, under the 1 s floor. Peak RSS is below `current`'s in every regime where `current`
finished (wide T4: 232 vs 1,732 MB). Where `current` hit the cap (or005,
or0001k2) no memory comparison exists; or0001k2's 774–867 MB is the 5.4M
candidate pairs held for pass 2, against 332–384 MB in-core.

SON stays far above the in-core route (or005: 13–22 s vs 0.3–0.4 s). Most of
that is SON's own work, not the engine's: four chunks at 0.9 × the threshold
give pass 1 a union of 1.13M local candidates on or005 against 10,488 frequent
itemsets, and pass 2 counts all of them in every chunk.

## Rule 3: the pass-2 K≥3 counter

Per level, prefix-group counting (`count_candidates`, arm `array`) against
per-candidate AND (arm `array-pc`), with the level's mean candidates per prefix
group and its row-space words (computed offline, as the harness chose its
spaces; no timing):

- No single mean group size separates the levels each counter wins. At means
  2.5–6.6, prefix groups win on deepk and skew while per-candidate wins on
  or005 and smoke; at 172 and 463 (skew K=4 and K=3) per-candidate wins at T1
  and prefix groups at T4.
- By the rule's fallback, the better geometric mean of pass-2 K≥3 counting time
  over the ten regimes with K≥3: per-candidate 0.233 s, prefix groups 0.587 s.
  **Per-candidate is the rule's outcome.**
- What it loses: skew T1 28.2 vs 16.7 s (+11.4 s), skew T4 28.2 vs 8.3 s
  (+19.9 s), deepk +0.1–0.2 s. It wins or005 (4.8 vs 13.1 s, −8.3 s), smoke
  and wide (≈ −0.01 to −0.02 s). The geometric mean is carried by regimes whose
  K≥3 counting takes under 25 ms.
- Per-candidate is single-threaded in the prototype; prefix groups use the pool
  for heavy groups, which is why skew T4 separates further than skew T1.

Observed, not part of a registered rule: every level of smoke, wide and or005
(row spaces of 134–391 words) is won by per-candidate. On deepk and skew (906–
3,879 words) per-candidate wins every level with a mean group size ≤ 2.04 and
prefix groups every level with 2.56–87 (levels with a mean of 1.00 tie), except K=3/K=4 of skew at T1 and K=3 of
deepk, which per-candidate wins.

## Decision (owner)

Rule 3 as registered picks per-candidate. The owner approved instead a dispatch
on two facts known before the level, read from these rows after the fact:
per-candidate at most 512 row-space words or a mean group size below 2.5,
prefix groups otherwise (`PROTOCOL.md` Amendment 4). S1's three reps confirm
the built tree.
