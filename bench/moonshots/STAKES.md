# Moonshot stakes (Phase 1)

Upper bounds on the seconds each idea could save, from the Phase 0 run and the
Phase 1 CPU replays. Box: 1× RTX A4000 with a Pentium G5420 (no AVX), so the
host phases are slow (see `../results/2026-10-17-moonshots-baseline/README.md`).
Medians of 3; one GPU unless marked. Evidence and formulas are in `SPEC.md`;
`stakes.py` recomputes every row:

```bash
uv run python bench/moonshots/stakes.py bench/results/2026-10-17-moonshots-baseline \
    bench/results/2026-10-17-moonshots-stakes
```

- "Rule 2 reachable" means the bound is ≥ 1 s and ≥ 10 % of the regime's wall
  on this box. Rule 2 also needs non-overlapping ranges, which only the
  campaign can show.
- Bounds overlap where ideas touch the same work: see Interactions.

## Ranked by upper-bound seconds saved

| rank | idea | regimes it touches | bound, s (share of wall) | rule 2 reachable | est. lines | risk |
|---|---|---|---|---|---|---|
| 1 | M1 GPU-native input layer | dsl, dsl-infer, dsl-free, sk2ml3 (all regimes slightly) | dsl 16.8 (66 %), dsl-infer 16.8 (78 %), dsl-free 16.6 (79 %), sk2ml3 4.8 (21 %); deepk / skew / oom2ml3 0.70 / 0.91 / 0.97 | yes: dsl ×3, sk2ml3 | 350–450 (Python + 1 kernel) | medium: dtypes, nulls and the K=1 filter must stay byte-identical (testable) |
| 2 | M2 row-wise K=3 | sk2ml3, oom2ml3 | sk2ml3 16.2 (70 %); oom2ml3 0.83 (33 %) | yes: sk2ml3 | 450–600 (kernel, device triangle index, chunking, dispatch) | medium-high: index build, atomics contention, chunking by first pair |
| 3 | M10 item order | sk2ml3, oom2ml3 | sk2ml3 7.8 (34 %); oom2ml3 0.51; dsl ≈ 0 | yes: sk2ml3 | 120–200 | low-medium: emission order on every option |
| 4 | M3 binary tensor cores | sk2ml3, oom2ml3, dsl | sk2ml3 7.3 (31 %); dsl 1.2 (4.8 %); dsl-infer 0.74; oom2ml3 0.49 | yes: sk2ml3 | 250–350 (PTX in two kernels, per-device check) | high: `CLAUDE.md` rule change, sm_90 b1 untested |
| 5 | M5b row reordering (+ tile skip) | dsl ×3, sk2ml3, oom2ml3 | dsl 5.6 (22 %), dsl-infer 2.0 (9.3 %), dsl-free 1.7 (8.3 %); sk2ml3 1.85 (8 %); oom2ml3 0.14 | yes: dsl | 250–350 (permutation, occupancy bits, skip in 3 kernels) | medium: the skip must find zero tiles cheaply (see M5 in SPEC) |
| 6 | M8 two-level look-ahead | dsl (no inference) | dsl 4.0 (16 %); dsl-infer 0.38 | yes: dsl | 300–400 | high: per-group triple storage, next-level bookkeeping |
| 7 | X0 count inference by default | dsl (+ ties elsewhere) | dsl 3.66 (14.5 %), measured | yes: dsl | ~5 (a default) | low; owner's decision |
| 8 | M6 replicate and split (2 GPUs) | sk2ml3, oom2ml3 on 2 GPUs | other box: ≈ 4.0 of 16.06 on sk2ml3 (25 %), ≈ 1.0 of 2.64 on oom2ml3 (38 %) | likely (2-GPU box needed) | 200–300 | medium; re-opens DP6 |
| 9 | M5a row trimming | dsl ×3 | dsl 2.34 (9.3 %) with free rebuilds every level, 0.90 at the half rule; dsl-infer 0.65 | **dropped** on today's walls (2.34 < 2.52); re-open after M1 (≈ 28 % of dsl's wall then) | 150–250 | medium |
| 10 | M9 pigeonhole reduce (2 GPUs) | sk2ml3, oom2ml3 on 2 GPUs | flags 1.19 % / 0.88 % of written entries; other box: ≤ ≈ 1.9 of 16.06 on sk2ml3 | maybe (2-GPU box needed) | 120–180 | low-medium |
| 11 | M7 deduction rules (depth ≤ 3) | dsl-infer | dsl-infer 1.65 (7.7 % now; ≈ 34 % after M1) | **dropped** on today's walls (1.65 < 2.15); re-open after M1 | 200–300 (third classification outcome, two more indexes) | medium |
| 12 | M4 lattice finisher | none of the matrix (needs the `deep_k20` preset) | 0 | **dropped** for the current matrix; re-open with `deep_k20` or AlphaFold | 350–450 | medium |
| 13 | M12 occupancy skip, no reorder (added) | sk2ml3, oom2ml3 | sk2ml3 0.32; oom2ml3 0.01; dsl 0 | **dropped** (≤ 0.32 s) | — | — |
| 14 | M11 weighted dedup | — | Online Retail ratio / planes 0.20–0.25 | **not built** (brief's condition fails) | — | — |

The bounds assume free implementations:
- M1: only the explode and the copies remain.
- M2: each row pair and each increment at the measured row-wise K=2 rate.
- M3: the popcount part at 1/16.3.
- M5b: a skipped tile costs nothing.
- M8: the next level costs nothing beyond the triples' ALU.
- M10: today's per-unit kernel rates.

## Interactions

- **sk2ml3's K=3 level (16.6 s)** is the target of M2, M10, M3 and M5b at once.
  If M2 wins, the tiled K=3 level goes away, and M10, M3 and M5b keep nothing
  there. M10 is the cheap fallback: about 120 lines, no kernel.
- **dsl's K≥3 levels** (6.6 s; 2.9 with inference) are shared by M5b, M8, M7,
  X0 and M3. With inference (X0) and M7, at most about 1.3 s of counting remains.
- **M1 changes what rule 2 sees.** The input phases are 74–87 % of the dsl
  walls here. After M1, dsl-infer's wall falls from 21.5 s to about 4.8 s, and
  M7's 1.65 s becomes about 34 % of it. Ideas measured after M1 clear rule 2
  more easily, so the order matters for the decision rule, not only for the
  sum.
- **M6 and M9** are both 2-GPU only, and M9 serves only data M6 cannot
  replicate.

## Suggested order (the owner decides)

1. **M1.** The largest stake, independent of every other idea, and it moves
   the walls the later protocols judge.
2. **M2.** The largest kernel stake, on sk2ml3. **M10** goes with it as the
   cheap fallback, judged in the same protocol (K=3 dispatch: rows, or tiled
   in the better order).
3. **X0** (a table for the owner), then **M7** on top of count inference.
4. **M5b** on dsl, judged after X0 and M7 shrink its target.
5. **M6** (then M9) on a 2-GPU box.
6. **M3** only if the tiled kernels still matter after M2 (rule change).
7. **M4** only with the `deep_k20` preset (or AlphaFold) in the matrix.
8. **M8** last, if dsl without inference stays a regime that matters.

## Facts from the brief, checked

| fact | verdict | where |
|---|---|---|
| The GPU route's CSR is built by explode + hash join + COO→CSR, with int64 indices uploaded from pageable memory | true | `core/matrix.py::_build_csr_from_transactions` (three Polars collects, `coo_matrix(...).tocsr()`); `gpu/csr_bitvec.py` (`astype(np.int64)`, `cp.asarray` without the pinned pool) |
| The CPU route's integer path took dsl's CSR build from 9.2 to 5.0 s at one thread (`b328cc1`) | true | the commit message (probe, base 9.2 s) |
| The tiled kernel costs 1.97–2.19 ns per counted tile-pair per word (phase C) | true there; here sk2ml3 runs at 1.41 ns (its tiles are sparse) | `../results/2026-10-06-phase-c-stake/README.md`; Phase 0 |
| 16 32-bit popcounts per clock per SM, so a ~6e11/s 64-bit peak; the tiled kernel sits near the roof | true: 14.0 popc32 and 7.0 popc64 per SM per clock (5.82e11/s); the inner loop runs at 5.93e11 slot-words/s on dense tiles. On sk2ml3 the kernel spends ≥ 54 % on staging, not popcounts | `microbench.jsonl` |
| sk2ml3 K=3: 11.8 B generated, 145 M pass the subset test, 1.31 M frequent; 376,436 counted tile-pairs | true: 11,796,796,856 / 144,951,409 / 1,310,438; 376,298 counted tile-pairs under today's three-way dispatch (376,436 under the phase-C two-way dispatch) | Phase 0 levels; `census-sk2ml3.jsonl` |
| Row-wise K=2 cut sk2ml3's K=2 from 29.16 to 0.65 s | true there; here K=2 takes 0.66 s (0.28 s counting) | `../optimizations/REPORT.md`; Phase 0 |
| `run_chunked_dense_level` is synchronous per chunk and frees every pool after each chunk; every GPU classifies every candidate | true | `gpu/row_split_chunks.py` (`free_all_blocks` per chunk per device; every shard launches every chunk) |
| DP6's split arms ran the legacy fused wrappers through host RAM, before the subset test and row-wise K=2 | true | `e4bb3ae:src/et_miner/gpu/dispatch.py` (`count_pairs_fused_k2_multi_gpu`, `count_k3plus_fully_fused_multi_gpu` whatever the variant; `bitvecs_gpu.get()`) |
| `use_generator_pruning` defaults to `False`; on dsl it cuts the wall with identical output | true: 25.21 → 21.54 s here (16.75 → 13.18 on the other box) | `core/apriori.py:412`; Phase 0 |
| The increments of row-wise K=3 on sk2ml3 are about 1e9 | **false**: 375,181,499 (exact) | `k3rows-sk2ml3.json` |
| ~10.5 s of dsl's wall falls outside the levels | true on the other box; here 18.6 s, almost all of it the input before K=1 (the result frame is 0.01 s) | Phase 0 span |
