# Route consolidation round 2: findings (Phases 2a and 2b)

Rows: `bench/results/2026-10-27-routes-cpu/raw.jsonl` (Phase 2a, 226 configs, CPU) and
`bench/results/2026-10-27-routes-gpu/raw.jsonl` (Phase 2b, 528 configs, 5.95 of 6 GPU-hours), C4 in
`bench/results/2026-10-27-routes-cpu/c4.jsonl`. The mechanical rule on both: `decide.md` here
(`bench/routes/decide.py`). Protocol: `bench/routes/PROTOCOL.md`. Box: 2x RTX 3060 12 GB, 17 threads.

Rule 1 (signature mismatch, logged fallback): none, in either phase.

## Decisions per decision point

| DP | outcome by the rule | rule |
|---|---|---|
| G3 K=2 | `rows`, `tiled`, `fused` stay; **`percand` (K=2 per-candidate) goes**, **`shared` (row-wise K=2 in shared memory) goes** | 3 |
| G5 K>=3 | `percand`, `group`, `tiled`, `fused` all stay | 3 |
| C1 CPU K=2 | `gram` stays; **`bitvec` goes** | 3 |
| C2 CPU K>=3 | `popcount` stays; **the `PROJ_MIN_SUFFIXES` dispatch to projection goes**; projection stays by role (the only K>=3 path when bitvectors exceed `BITVEC_BUDGET_BYTES`, e.g. dslxl: 637 MB > 512 MB); SON pass 2 `percand` and `groups` both stay | 3, 7 |
| C3 Rust miner | partial winner: Rust wins deepk, skew, dsl; the CPU route wins deepk20, or0001k2, or002, or003; geomean CPU 2.87 s vs Rust 4.84 s (timeouts at their bound) | 5, 7: owner |
| C4 counting API | `polars` (`count_support_vectorized`) wins all 15 dsl batches; **`sparse` (`count_support_sparse`) goes** (wins nothing, OOM on dsl K=2); `arrays` wins nothing | 4 |
| G1 input layer | device wins dsl, dslxl, sk2ml3, dsl-str, dsl-wide; host and rebuilt win nothing, **blocked by rule 7** (small absolute rises, below) | 4, 7: owner |
| G2 row order | lex wins dsl, dsl-free, dslxl; keep wins nothing, **blocked by rule 7** (lex peaks 1.74x / 1.79x VRAM on dsl / dslxl) | 4, 7: owner |
| G6 ESCO | dense wins dsl, oom2ml3, sk2ml3; **esco-auto and esco3 win nothing, not blocked: ESCO goes** | 4 |
| G7 `prune_apriori=False` | prune wins oom2ml3 (76x), sk2ml3 (365x); **noprune wins nothing: goes** | 4 |
| S1 SON, 1 device | ported wins deepk, dsl, dslxl (dsl 8.1 s vs boolean 176 s vs CPU 1091 s); boolean **blocked by rule 7** (ported peaks 1.9x / 2.0x VRAM on dsl / dslxl); SON on the CPU stays (rule 9) | 4, 7, 9: owner |
| S2 SON, 2 GPUs | ported wins deepk, dsl, dslxl (dslxl 12.7 s vs 306 s); boolean **blocked by rule 7** on deepk only (281 vs 153 MB) | 4, 7: owner |
| G4 | not re-measured: G5 changed no kernel K=3 falls back to | — |
| H Rust host roles | **not run**: its dumps and R5 timing count against the GPU budget, 0.05 GPU-hours left | owner |

## Rule 7 blocks, absolute numbers (largest device, max over reps)

| DP / removal | regime | kept path | replacement | rise |
|---|---|---|---|---|
| G1 host | deepk | host 133 MB VRAM | device 191 MB | 1.44x |
| G1 host | skew | host 149 MB | device 195 MB | 1.31x |
| G1 rebuilt | deepk, skew | 147 / 149 MB | 191 / 195 MB | 1.30x / 1.31x |
| G1 rebuilt | dsl-wide | rebuilt 2,001 MB | coded 2,555 MB | 1.28x |
| G1 rebuilt | or002, or002-str | rebuilt 699 / 727 MB RSS | host 903 / 933 MB RSS | 1.29x / 1.28x |
| G2 keep | dsl, dslxl | keep 1,465 / 4,125 MB VRAM | lex 2,555 / 7,375 MB | 1.74x / 1.79x |
| S1 boolean | dsl-son, dslxl-son | boolean 655 / 1,697 MB VRAM | ported 1,237 / 3,435 MB | 1.89x / 2.02x |
| S2 boolean | deepk-son (2 GPUs) | boolean 153 MB VRAM | ported 281 MB | 1.84x |

The other side of the same rows: host peaks 31.4 GB RSS on dslxl (device 8.5 GB, the box has 34.8 GiB);
boolean SON peaks 18.3 / 25.6 GB RSS on dslxl-son 1 / 2 GPUs (ported 7.4 / 7.8 GB).

## Findings

1. (must-fix) `n_gpus=2` with a `bitvecs=` input larger than about 2/3 of the input device fails with a
   CuPy `OutOfMemoryError`: `shard_prebuilt_bitvecs` (`gpu/row_split.py:213-219`) makes each shard
   contiguous on the input device while the caller's array stays resident (sk2ml3-bv: 8.75 GB + 4.375 GB on
   a 12 GB card). The same input mines in 72.8 s on one GPU. `n_gpus` is exact, so no fallback; fix: shard
   without the temporary (copy each row range straight to its device) where it fits, and an upfront
   `MemoryError` naming the sizes where it cannot.
2. (should) The CPU route cannot mine dsl (20M rows) in 900 s (every F1 arm timed out), while
   `apriori_from_csr` does it in 227 s and `use_generator_pruning` / `prune_equal_support` in 281 / 273 s;
   the CPU route on dslxl ran past 1,800 s. C1/C2 decide nothing on dsl.
3. (should) `apriori_from_csr` slows steeply with depth: deepk20 at `max_length` 7 / 9 / 11 takes 16 / 197 /
   878 s (the CPU route: 65 s for all 20 levels); it is also 14x and 24x slower than the CPU route on or002
   and or0001k2.
4. (should) `count_support_batched(sparse=True)` was OOM-killed (34.8 GiB cgroup) on dsl K=2: 3,570 pairs
   over 20M rows x 85 columns; `sparse=False` counts the same batch in 0.8 s. Moot if C4's removal goes in.
5. (should) Rule 11 (a removal slows a regime by >= 25%): removing `prune_apriori=False` (G7) slows or002
   from 0.21 to 0.28 s (+33%); removing Rust (C3, if it goes) slows deepk 3.0x, skew 2.5x and smoke 3x. No
   other removal above slows a regime by 25% (largest: G2 keep on deepk20, +21%; G3 per-candidate K=2 on
   deepk15, +18%).
6. (nice) `routes_matrix._reps` keys `SUBSECOND` on the workload, not the regime, so oom2-bv (8.5 s),
   oom2ml3-bv (10.4 s) and oom2ml3-noprune (27 s) ran their reps in the shed-first group. No rep was shed
   this time.
7. (nice) dsl `esco3` raises `TidsetFitError` on a 12 GB card (11.27 GB of tidsets, 10.95 GB free); by design,
   before any allocation, with advice. ESCO goes by G6 either way.
8. (nice) `cargo clippy -- -D warnings`: 2 errors on main (div_ceil, complex type), unchanged.

## Waiting for the owner

1. Rule 7 on small absolute rises: apply as written (host, rebuilt, boolean SON stay as named fallbacks),
   or add a floor so 58 MB of VRAM or 0.2 GB of RSS does not block a removal.
2. G2 keep: lex costs 3.2 GB more VRAM on dslxl and is 2.7x faster there. Keep `keep` as the fallback
   for inputs whose lex order does not fit, port a bounded lex (<= ~150 lines), or drop it.
3. C3: Rust wins 3 regimes and loses 4; the geomean favours the CPU route, but without Rust the CPU route
   fails dsl (timeout) where Rust mines it in 227 s (rule 7).
4. H: extra GPU budget for the Rust host-role microbenchmarks (estimate 0.3-0.5 GPU-hours), or decide
   H without them.
5. G6 and G7 remove public parameters (`sparse_from_k` and the sparse CSR path; `prune_apriori`). The rule
   says both go; confirm the API removal.
