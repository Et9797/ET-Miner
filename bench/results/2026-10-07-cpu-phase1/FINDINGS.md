# CPU tier, Phase 1: findings

Campaign: `bench/cpu/PROTOCOL.md` amendment 2, at commit 70fb816, on a 4 vCPU
Intel Xeon @ 2.10 GHz with 15 GB (numpy 2.5.2, scipy 1.18.0, polars 2.0.0,
efficient-apriori 2.0.6). Raw rows: `raw.jsonl`; tables: `compare.md`
(`bench/cpu/compare.py`).

## Outcome

- **Goal test**: the F-polars and F-auto arms beat efficient-apriori in 32 of
  32 regime cells (8 workloads × 1 and 4 threads) under rule 2 with the 10 %
  floor below 10 s. F-sparse wins its 16 of 16 too. Median ratios at 1 thread
  run from 4.7× (smoke, 0.56 vs 0.12 s) to 47× (wide_vocab, 60.05 vs 1.27 s);
  at 4 threads from 4.7× (smoke) to 109× (wide_vocab, 0.55 s).
- **Before / after**: every arm wins rule 2 against Phase 0 except smoke, where
  the 1 s floor exceeds the whole Phase 0 run (0.33–0.83 s → 0.12 s).
- **Memory**: no arm's median `ru_maxrss` exceeds 1.25× of Phase 0. The largest
  ratios are deep_k and skewed_rows at 1 thread (1.17–1.18×: 299 → 350 MB,
  331 → 389 MB); wide_vocab and Online Retail II fall to 0.18–0.36×.
- **Signatures**: one itemset hash per workload across every arm of both
  campaigns; the runner's equivalence check passed on all 168 configs.

## Where the time goes now (F-auto, `compare.md`)

- deep_k and skewed_rows: the CSR build through Polars takes 0.8–1.2 s (30 % of
  the wall), K≥3 counting 1.2–2.0 s. Four threads gain nothing on deep_k
  (2.74 → 2.69 s); the threaded bitvector build is slower there (0.40 → 0.62 s).
- wide_vocab, or005 and or0001k2: the K=2 Gram dominates (0.36–1.73 s at
  1 thread). Four threads speed the Gram up 2.1–5.5× on wide_vocab and every
  Online Retail II workload.
- or002: K≥3 counting (5.2 s) and candidate generation with its subset test
  (1.3 s) at 1 thread.

## Dispatch rules (measured, replacing the old thresholds)

| Rule | Value | Evidence |
|---|---|---|
| K=2 on bitvectors instead of the scipy Gram | candidate pairs × words ≤ 3 × pair occurrences | bitvector K=2 incl. its build at 0.61× / 0.69× of the Gram at work ratios 1.7 / 2.7 (skewed_rows, deep_k); 6.3–38.6× slower at 7.1–175 (smoke, Online Retail II, wide_vocab) |
| K≥3 group by projection instead of bitvector pairs | ≥ 40 suffixes up to 2,048 words, ≥ 80 above | `l3_crossover.jsonl`: projection faster from about 30–40 suffixes at 570–1,563 words and 66–100 at 3,907–15,625 words |
| prefix group to the thread pool | ≥ 2M pairs × words | tuning measurement: every group on the pool made or002 2.5× slower (GIL hand-offs on small groups) |
| K=2 Gram split for the thread pool | ≥ 5M pair occurrences | tuning measurement, not in the campaign rows: smaller Grams lost time to the split |

## Deviations

- The campaign was interrupted once by a container restart after 70 of 168
  rows and resumed with the same command at the same commit. `env.txt` holds
  both captures: same CPU model, memory and package versions.
- Found after the run and fixed in 51a7bfe (no effect on the timed code path):
  SON's single-chunk fallback forwarded `sparse=` to the CPU route and so raised
  the new deprecation warning under `streaming=True`.

## Open items (owner's decision)

1. `tests/test_tier_equivalence.py::test_tier2_rust_matches_oracle` calls
   `apriori(sparse=True)`, which now runs the same array miner as Tier 1. The
   chain still passes but no longer covers the Rust miner. Proposal: point the
   Tier 2 leg at `et_miner.apriori_from_csr` (the Rust entry point, still
   exported) and rename it in `CLAUDE.md`.
2. SON (`streaming=True`) still generates candidates with the quadratic
   `_generate_candidates` and counts with the Polars or sparse counters
   (`_choose_counting_strategy`, now used by SON only). Porting SON's passes
   onto `cpu_miner` is the follow-up.
3. `tests/test_smoke_correctness.py::TestCPUvsEfficientApriori::test_support_001`
   cannot finish on this machine with any engine: at support 0.001 Online
   Retail II has 71M frequent itemsets at K=6 alone and the lattice keeps
   growing (11.6 GB RSS when stopped). Its oracle call also skips the CLAUDE.md
   convention (no −0.5, `max_length=100`) and compares counts only.
4. The GPU route's `_build_csr_from_transactions` could move to
   `build_transaction_csr` (int32 indptr, chunked mapping) once tested on a GPU.
