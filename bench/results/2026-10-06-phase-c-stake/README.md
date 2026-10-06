# Phase C stakes, offline (2026-10-06)

What O6 and O7 of `bench/optimizations/SPEC.md` could gain, computed on the CPU
from the lattice dumps of the candidate-waste run
(`bench/results/2026-10-05-candidate-waste/*-C1-split*.lattice.*`, not
committed, regenerable with `bench/runner.py --mode waste`) and the measured
level splits of phase A's final check (`../2026-10-05-optimizations/final.jsonl`,
the current defaults) and the pruning run (`../2026-10-05-pruning-final/raw.jsonl`).
No GPU time.

Run from this directory:

```
W=../2026-10-05-candidate-waste
{ uv run python o6_stake.py $W/dsl-C1-split.lattice ../2026-10-05-optimizations/final.jsonl dsl-C1-final
  uv run python o6_stake.py $W/dsl-C1-split.lattice ../2026-10-05-optimizations/final.jsonl dsl-C2-final
  uv run python o6_stake.py $W/dsl-C1-split.lattice ../2026-10-05-optimizations/final.jsonl dsl-C2-infer-final --infer
  uv run python o6_stake.py $W/dsl-C1-split-free.lattice ../2026-10-05-pruning-final/raw.jsonl dsl-C1-free-prune
} > o6_stake.txt
uv run python k3_tiles.py $W/oom2ml3/oom2ml3-C1-split.lattice $W/sk2ml3-C1-split.lattice > k3_tiles.txt
```

On two GPUs each device counts half the words, so the per-word rates printed
for the `C2` configs are twice the per-device rates.

## O6: where the K≥3 counting goes (`o6_stake.txt`)

Only deep_sparse_large (dsl) has K≥3 counting that rule 2 can resolve. Elsewhere
it is ≤ 0.04 s (smoke, deepk, skew, or003, or002), or a K=3 explosion where
both kernels' work is already set by the subset test (below).

| dsl, defaults | wall | per-candidate kernel | tiled kernel |
|---|---|---|---|
| 1 GPU | 20.17 | 5.17 | 4.11 |
| 2 GPUs | 15.42 | 2.59 | 2.09 |
| 2 GPUs, count inference | 12.88 | 0.75 | 1.38 |
| 1 GPU, free sets (pruning run) | 17.83 | 1.44 | 1.76 |

Both kernels follow a simple law, steady across the levels that carry the time
(K=5–10, one GPU):

- **Tiled kernel: 1.97–2.19 ns per counted 32×32 tile-pair per word**, whatever
  the share of its 1,024 slots that hold a counted pair. Its time is set by
  the slots. On dsl the counted tile-pairs hold 212,754 counted pairs in
  6,245,376 slots: **3.4 % use** (2.7 % with count inference, 4.1 % for free
  sets). The groups have ≤ 85 suffixes, so most of a 32×32 tile-pair is empty
  or skipped.
- **Per-candidate kernel: 1.38–1.50 TB/s** of k-row reads per counted
  candidate, about 3× the card's DRAM bandwidth, so most reads hit L1/L2. dsl
  has 371,563 counted candidates in 123,345 groups: **3.0 per group**.

What a smaller work unit changes, at the same per-slot or per-row rate
(optimistic: smaller tiles lose register blocking, and fewer redundant reads
mean fewer cache hits):

| part (1 GPU, defaults) | today | work under O6 | at today's rate |
|---|---|---|---|
| tiled groups, 16×16 tiles | 4.11 s | 1,811,968 slots (3.45× fewer) | 1.19 s (−2.9 s) |
| tiled groups, 8×8 tiles | 4.11 s | 1,069,824 slots (5.8× fewer) | 0.70 s (−3.4 s) |
| per-candidate groups, 8×8 tiles | 5.17 s | 7,951,552 slots | ≈ 5.2 s (no gain) |
| per-candidate groups, one block per group | 5.17 s | 1,186,799 row reads (39 %) | 2.02 s (−3.2 s) |

- Small tiles cannot help the per-candidate groups: with 3.0 counted pairs per
  group, an 8×8 tile is still 95 % empty. Only a kernel whose work follows the
  counted pairs (one block per group, the prefix AND staged once per word, a
  list of the counted pairs) reaches them.
- Rule 2 needs ≥ 2.02 s on one GPU and ≥ 1.54 s on two. At the rates above
  either part alone clears one GPU; on two GPUs each part is −1.5 to −1.6 s,
  at the threshold, and both together clear it.

## O6 on the K=3 explosions (`k3_tiles.txt`)

| workload | tile | counted tile-pairs | their slots | slot use |
|---|---|---|---|---|
| sk2ml3 | 32 | 376,436 of 11,571,649 (3.25 %) | 385,470,464 | 37.6 % |
| sk2ml3 | 16 | 1,321,105 | 338,202,880 | 42.9 % |
| sk2ml3 | 8 | 4,588,701 | 293,676,864 | 49.4 % |
| oom2ml3 | 32 | 62,368 of 2,045,954 (3.05 %) | 63,864,832 | 24.8 % |
| oom2ml3 | 16 | 205,714 | 52,662,784 | 30.0 % |
| oom2ml3 | 8 | 671,794 | 42,994,816 | 36.8 % |

Smaller tiles cut the counted slots by 12 % (16) and 24 % (8) on sk2ml3: the
surviving pairs cluster in the counted tiles. The per-candidate groups hold 54
and 46 candidates. O6 has no stake here.

## O7 (pruning report's findings 2–4; `../../pruning/REPORT.md`)

| item | stake | regime | rule 2 needs |
|---|---|---|---|
| (a) free-set host filter | 0.22–0.24 s ("other", free-prune vs free-off) | dsl free 17.83 s; or002 free 0.50 s | ≥ 1 s |
| (b) subset test where levels cost ms | +0.05 s (or002), +0.14 s (or002 ESCO), +0.18 s (or002 free) | or002 0.27–0.50 s | ≥ 1 s |
| (c) K=3 frequent-pair bitmap | ≤ 1.37 s | sk2ml3 1 GPU, 20.68 s | ≥ 2.07 s |

- (c) is now bounded exactly. The pruning run without the subset test counted
  all 11,571,649 tile-pairs of sk2ml3 in 462.16 s, 39.9 µs each (one rep). The
  376,436 counted ones (the census above; the earlier 3.6 % came from a sample)
  then take 15.03 s of today's 16.40 s tiled time. That leaves ≤ 1.37 s for the
  classification and the block overhead of the 11.2 M skipped tile-pairs, and a
  bitmap removes only part of the classification. The bound is an upper bound:
  counted tile-pairs sit on the densest items and cost more than the average.
  Every GPU classifies every candidate, so two GPUs have about the same
  ≤ 1.4 s of 16.11 s, below 1.61 s.
- (a) and (b) are below 1 s in every regime. Ties go to rule 5, which keeps
  the code without the item.
