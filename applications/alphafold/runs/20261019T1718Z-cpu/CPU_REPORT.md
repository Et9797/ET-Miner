# CPU_REPORT — AlphaFold base214m on ET-Miner's CPU and CPU-SON routes

Run directory `runs/20261019T1718Z-cpu/` (branch `alphafold-cpu-reproduction`). Every number below cites a row of
this run's `RESULTS.md` (`CPU-…`) or `CPU_COMPARISON.md` (`CC-…`); GPU numbers cite the repo-root `RESULTS.md`
(`P-…`, `M-…`, `X-…`) and are read from the GPU run's artifacts (JSONs, logs, parquet tables) by
`scripts/build_comparison.py` and `scripts/build_report.py`, which generate the tables in the Runs, SON and Null
sections.

## In plain words

The GPU run of 2026-09-02 searched 76.9 million AlphaFold proteins for groups of annotations that occur together
(an "itemset", like a shopping basket that keeps coming back: these protein families and these functions, on the
same proteins). This run did the same search again on an ordinary 6-core desktop CPU, without any GPU, after
rebuilding the protein data from the original public sources.

- **Every full-data itemset table is identical to the GPU's** (C1–C6 and M4 in-core, S1–S3 SON; X1 equals C3),
  row by row: the same itemsets and the same counts, from 5,305 itemsets at the strictest threshold up to
  48.0 million at the loosest (min_count 4). Nothing had to be adjudicated. The small G0 check agrees three ways on
  the CPU but covers other proteins than the GPU's G0, and the shuffled null tables are compared as distributions.
- **The GPU's association-rule numbers were wrong** (rules such as "proteins with A usually also have B"). The GPU run
  reported 29,540 rules; the correct count, verified by an exact enumeration, is 53,447. Its counts of strong rules
  (lift ≥ 5 and ≥ 100) and of cross-domain rules were wrong for the same reason: the GPU run's rule generator
  silently skipped rules, or set their lift to 0, when itemsets were stored out of order (CI-001).
- **The CPU is slower, as expected:** on the engine's default route it took 9× (Base) to 343× (Power) as long as
  one RTX 3090. With the engine's count inference switched on (it derives counts it can prove instead of counting
  them; results stay exact, CPU-042), the deeper runs took 36× (Blitz) down to 7× (Opus) as long, and the
  min_count 4 run 4.6× as long as the GPU run's two GPUs. The GPU run did not use count inference. The streaming
  (SON) route costs the CPU only 1.06–1.15× its in-core time, against 98.8× on the GPU.
- **The random-shuffle control** ("null model", the data with annotations shuffled between proteins, to show what
  chance alone produces) behaves on the CPU as on the GPU: both null runs (5 and 100 shuffles) are `consistent` by
  the criterion fixed before they ran.
- **Every run of the plan completed**; none was dropped.

## Box, engine, threads

- AMD Ryzen 5 4600G (6 cores, 12 threads), 30.65 GiB RAM (below the brief's 64 GiB; waived by Et, deviations D-001),
  2 GiB swap, Ubuntu 22.04.5, no NVIDIA device (`env.json`, CPU-005).
- ET-Miner `main` at 37819e2f89c69c1e5bf0c3eb06f3142bfbcc7f68 in `../ET-Miner-engine` (CPU-001), Python 3.14.7,
  polars 2.0.0, Rust extension built (CPU-004).
- Every pool pinned to 6 threads (`n_jobs=6`; POLARS/RAYON/OMP/MKL/OPENBLAS = 6) and `CUDA_VISIBLE_DEVICES=` in
  every process (`scripts/run_py.sh`, `_harness.check_env`).

## Data

- pLDDT: `bigquery-public-data.deepmind_alphafold.metadata` read through the BigQuery Storage Read API (read
  session only, 16 streams): 214,683,829 rows, 3,495,993,272 B, equal in size to the GPU run's CSV (CPU-007).
  No export, load, destination table or created dataset (CPU-054).
- UniProtKB TrEMBL 2026_01: the 161.9 GB archive streamed and reduced on the fly (the third attempt; two earlier
  ones were stopped within their first ~1.5 GB, D-004, D-007); whole-archive md5 e41d313b61d595a05d4f80032df44c33
  equals UniProt's RELEASE.metalink (CPU-010). The 160.8 GB member was never stored.
- Rebuilt transactions: 76,890,945 multi-feature proteins, 1,002 items present, 316,421,093 entries, at most 46
  items per protein. M-001..M-009, U-007, U-009, X-001, X-002 and X-005..X-016 all reproduced exactly (51 checks,
  CPU-021; CC-002..CC-052). Sources, sizes and checksums: `data/MANIFEST.md`; uncommitted files with sha256:
  `EXCLUDED_LARGE_FILES.md`.
- Two differences, neither a data drift: one equal-count Pfam pair has swapped item ids (GPU tables are translated
  by feature name before every comparison, D-008, CPU-019), and the first million rows are other proteins than the
  GPU run's (the row order follows the parallel BigQuery streams; CPU-020, CI-002).

## Gates

- Engine: `pytest -q -m "not slow"` 897 passed / 564 skipped (every skip a CUDA test), tier-equivalence gate 5 passed
  / 86 GPU legs skipped (CPU-002, CPU-003).
- G0 (first 1,000,000 rows, min_count 50): in-core = SON with 4 chunks = efficient-apriori, 167,042 itemsets, K ≤ 14,
  identical tables (CPU-023). The GPU's 168,674 is not comparable: the first million rows are different proteins
  here (CI-002).

## Runs

One line per run of the frozen PLAN.md (`CPU_COMPARISON_rows.json` → `runs`; the value-level comparison is
`CPU_COMPARISON.md`). Verdicts on itemsets come from a full outer join of the whole table against the GPU reference
(`scripts/compare.py`). CPU walls are parquet load + engine call; median [min, max] of 3 for runs under 15 minutes.
Timings are hardware-comparison: AMD Ryzen 5 4600G, 6 threads, against the GPU box named in each cell.

<!-- BEGIN:runs -->
| run | CPU result | GPU result | verdict | CPU wall | GPU wall | CPU peak RSS | artifact |
|---|---|---|---|---|---|---|---|
| G0 | 167,042 / K14 | 168,674 / K14 (P-001) | in-core = SON (4 chunks) = efficient-apriori; GPU total not comparable (other rows, CI-002) | in-core 3.1 s (median of 3, [3.1, 3.9]); SON 9.2 s (median of 3, [9.2, 9.4]) | 3.5 s (single-GPU reference call, `phase3/2026_01/validate_row_split.log`) | 0.5 GiB | `data/mined/G0/G0.json` |
| C1 | 5,305 / K9 | 5,305 / K9 (P-028) | identical, 0 differing (Opus tables at 76,891) | 268.1 s (median of 3, [267.8, 274.9]) | 31.1 s (one RTX 3090) | 11.5 GiB | `data/mined/C1/meta.json, data/mined/C1/cmp_gpu_opus.json` |
| C2 | 113,405 / K14 | 113,405 / K14 (P-028) | identical, 0 differing (Opus tables at 7,690) | 6,381.5 s (1 run) | 52.6 s (one RTX 3090) | 11.8 GiB | `data/mined/C2/meta.json, data/mined/C2/cmp_gpu_opus.json` |
| C3 | 475,865 / K14 | 475,865 / K14 (P-028) | identical, 0 differing (Opus tables at 769) | 25,604.1 s (1 run) | 74.7 s (one RTX 3090) | 12.1 GiB | `data/mined/C3/meta.json, data/mined/C3/cmp_gpu_opus.json` |
| C4 (count inference) | 2,841,280 / K19 | 2,841,280 / K19 (P-028) | identical, 0 differing (Opus tables at 77) | 10,443.8 s (1 run) | 292.9 s (one RTX 3090) | 12.3 GiB | `data/mined/C4/meta.json, data/mined/C4/cmp_gpu_opus.json` |
| C5 (count inference) | 14,558,875 / K20 | 14,558,875 / K20 (P-028) | identical, 0 differing (Opus tables at 16) | 15,199.8 s (1 run) | 1,996.4 s (one RTX 3090) | 12.4 GiB | `data/mined/C5/meta.json, data/mined/C5/cmp_gpu_opus.json` |
| C6 (count inference) | 26,849,505 / K22 | 26,849,505 / K22 (P-028) | identical, 0 differing (Opus tables at 8) | 17,670.1 s (1 run) | 2,631.2 s (one RTX 3090) | 12.5 GiB | `data/mined/C6/meta.json, data/mined/C6/cmp_gpu_opus.json` |
| M4 (count inference) | 48,007,493 / K22 | 48,007,493 / K22 (P-017) | identical, 0 differing (`minc4/parquet`) | 20,653.4 s (1 run) | 4,536.9 s (2 GPUs) | 18.2 GiB | `data/mined/M4/meta.json, data/mined/M4/cmp_gpu_minc4.json` |
| S1 | 5,305 / K9 | 5,305 / K9 (P-015) | identical vs GPU, identical vs C1 | 291.1 s (median of 3, [290.8, 291.2]) | 779.6 s (2× RTX 3090 box, `use_gpu=True`; GPUs used not recorded) | 7.7 GiB | `data/mined/S1/meta.json, data/mined/S1/cmp_gpu_son.json, data/mined/S1/cmp_C1.json` |
| S2 | 113,405 / K14 | 113,405 / K14 (P-016) | identical vs GPU, identical vs C2 | 7,106.3 s (1 run) | 1,426.0 s (2× RTX 3090 box, `use_gpu=True`; GPUs used not recorded) | 8.0 GiB | `data/mined/S2/meta.json, data/mined/S2/cmp_gpu_son.json, data/mined/S2/cmp_C2.json` |
| S3 | 475,865 / K14 | 475,865 / K14 (P-029) | identical vs GPU, identical vs C3 | 29,451.7 s (1 run) | 6,151.1 s (one RTX 3090) | 8.0 GiB | `data/mined/S3/meta.json, data/mined/S3/cmp_gpu_opus.json, data/mined/S3/cmp_C3.json` |
| X1 (4 chunks) | 475,865 / K14 | — (no GPU run) | identical vs C3 | 16,322.1 s (1 run) | — | 5.8 GiB | `data/mined/X1/meta.json, data/mined/X1/cmp_C3.json` |
| A1 | K=22 itemset with 8 proteins; Blitz decode; P-026 1,234 | K=22, 8 proteins; decode; P-026 1,234 (P-007..P-009, P-026, P-030, P-031) | identical | 12.7 s (K=22 analysis) | 21.5 s (K=22 analysis) | — | `data/mined/A1/k22_check.json, data/mined/A1/intermediate_k_decode.json, data/mined/A1/p026_c6.json` |
| R1 | 53,447 rules (51,126 lift ≥ 5, 40,428 ≥ 100) | 29,540 (17,499, 9,283) (P-015) | mismatch, adjudicated for the CPU by exact enumeration (CI-001) | 0.161 s (median of 3) | 0.19 s | 0.2 GiB | `data/mined/R1/meta.json, data/mined/R1/adjudication.json` |
| N5 | mean 171,290.4 over 5 permutations, K_max [6] in all | mean 171,249.6 over 5 (P-012) | consistent | 882.0 s per permutation (median) | 18.84 s per permutation (2 GPUs) | 24.0 GiB | `data/mined/N/verdict_n5.json` |
| N100 | mean 171,300.5 over 100 permutations, K_max [6] in all | mean 171,306.3 over 100 (P-018) | consistent | 883.5 s per permutation (median) | 18.59 s per permutation (2 GPUs) | 24.0 GiB | `data/mined/N/verdict_n100.json` |
| P1T (sizing: SON pass-1 chunk at Base) | 6,125 / K9 | — (sizing only) | not compared (first 40,000,000 rows at local min_count 36,000) | 153.6 s (median of 3, [153.0, 154.5]) | — | 6.2 GiB | `data/mined/P1T/meta.json` |
| C3G (C3 with count inference) | 475,865 / K14 | — (fallback check) | identical vs C3 | 5,477.5 s (1 run) | — | 12.3 GiB | `data/mined/C3G/meta.json, data/mined/C3G/cmp_C3.json` |
<!-- END:runs -->

## SON against in-core

<!-- BEGIN:son -->
| threshold | CPU SON (2 chunks, first run) | CPU in-core (first run) | CPU ratio | GPU ratio |
|---|---|---|---|---|
| Base | 291.2 s | 274.9 s | 1.06× | — |
| Super | 7,106.3 s | 6,381.5 s | 1.11× | — |
| Power | 29,451.7 s | 25,604.1 s | 1.15× | 98.81× (P-029: 6,151.06 s vs 62.25 s, one RTX 3090) |

X1 (Power, 4 chunks of 19,222,737): 16,322.1 s, 0.64× C3 and 0.55× S3 (X1 shared the CPU for a few minutes with low-priority checks, D-015). On the CPU route SON's pass 1 already returns exact counts for almost every candidate ("within the partition bound … exact" in the engine log), so pass 2 has little or nothing to count: at Power pass 1 29,422.2 s and pass 2 28.1 s, against the GPU's 5,300.4 s and 846.8 s (P-029). All timings are hardware-comparison.
<!-- END:son -->

## Null model

<!-- BEGIN:null -->
- **N5: consistent.** 5 CPU permutations, totals mean 171,290.4 (range 171,263–171,324), K_max [6] in every one; GPU (the 100-permutation run P-018; the GPU's own N5 is P-012) μ 171,306.29, σ 65.68 (ddof 1), range 171,148–171,463; 3σ/√5 band [171,218.17, 171,394.41]; Mann–Whitney p 0.609; permutations reaching the real count at K = 4/5/6: 0; with K ≥ 7: 0. Criteria: a_kmax_all_6 yes, b_mean_within_3sigma_sqrt5 yes, d_no_perm_reaches_real_at_k456_and_none_k_ge_7 yes (`data/mined/N/verdict_n5.json`).
- **N100: consistent.** 100 CPU permutations, totals mean 171,300.5 (range 171,128–171,433), K_max [6] in every one; GPU (P-018, P-022) μ 171,306.29, σ 65.68 (ddof 1), range 171,148–171,463; 3σ/√5 band [171,218.17, 171,394.41]; Mann–Whitney p 0.622; permutations reaching the real count at K = 4/5/6: 0; with K ≥ 7: 0. Criteria: a_kmax_all_6 yes, c_mannwhitney_p_ge_0.01 yes, d_no_perm_reaches_real_at_k456_and_none_k_ge_7 yes (`data/mined/N/verdict_n100.json`).
- The CPU shuffles with NumPy and the GPU run with CuPy from the same seeds, so single permutations differ; the criterion compares them as distributions (brief Phase 4 step 2, fixed before N5 ran).
<!-- END:null -->

## Not run and open items

<!-- BEGIN:notrun -->
No run of PLAN.md was dropped: C6 (with A1), M4 and X1 were first dropped on early estimates and re-admitted after measured runs lowered them (PLAN.md Re-estimates; deviations D-011, approved by Et in D-014).
<!-- END:notrun -->

Open items:
1. The selection reading (D-011) and the re-admissions after re-estimation go beyond the brief's literal text; Et
   approved them (D-014).
2. The AlphaFold DB licence (CC BY 4.0) could not be verified from an allowed inbound source (`data/MANIFEST.md`).
3. Per-protein identity of the rebuilt transactions with the GPU run's file cannot be checked: that file is not on
   this box (CI-002). Every aggregate statistic and every full-table mining result agrees.
4. Repo-root I-012's old cross-domain rule count (45,286) is not reproduced; its definition is not in the repository
   (CI-001).

## Review findings

Two independent adversarial reviews (fresh subagents that built none of this): an interim one on 2026-10-21 over
everything finished by then, and a final one on 2026-10-23 over the whole campaign. Both re-counted the itemset tables
with their own code and confirmed every "identical" verdict. Every finding was fixed; the full lists with the fixes
are in `REVIEW_interim.md` and `REVIEW_final.md`. Sorted by severity:

1. (must-fix, interim) `data/MANIFEST.md` was ignored by a repo-wide `.gitignore` rule and never committed. Fixed.
2. (must-fix, interim) M4 and X1, re-admitted after re-estimation, would not have started (the running queue had read
   PLAN.md earlier). Fixed by a chained queue restart.
3. (should, interim) The sizing safety factor was not re-derived when the reference run changed. Fixed (PLAN.md
   2026-10-21T18:45Z; M4 estimated at 20.4 h).
4. (should, interim) The selection reading (D-011) and the re-admissions go beyond the brief's text. Recorded;
   approved by Et (D-014).
5. (should, interim) CI-001 described the rule-generator bug incompletely (lift-0 rules, cross-domain). Fixed.
6. (should, interim) CI-002 and CPU-020 overstated the row-order evidence. Reworded.
7. (should, interim) The report draft overstated two statements. Fixed.
8. (should, interim) Three values had no artifact. Fixed (P-026 on C3, provenance scan log, GPU null K_max).
9. (should, interim) CPU_COMPARISON lacked per-level times and had wrong labels. Fixed.
10. (should, final) `compare.py`'s level mode accepted two files for one level, and no test covered the level mode.
    Fixed (guard plus 7 tests).
11. (should, final) D-015's checks had no artifact. Fixed (`logs/check_compare_levels_20261023T0906Z.log`).
12. (should, final) Some report values were typed into the build scripts. Now read from their artifacts.
13. (should, final) M4's peak memory was understated. Peak RSS is now the whole process's.
14. (nice, interim) Small inconsistencies (test count, wall definitions, CPU-024's GPU time, CPU-022's gzip cause).
    Fixed.
15. (nice, interim) The branch tracked an upstream; removed. (Corrected later: a bare push would still work under
    `push.default=current`; no push happened.)
16. (nice, interim) D-012 missed two by-hand overlaps. Fixed.
17. (nice, final) Missing K-distribution rows for P-014 and P-017. Added (identical).
18. (nice, final) Wrong labels (N5 GPU id, check names as GPU ids, P-029 for X1). Fixed.
19. (nice, final) CPU-058's median mixed two sets. Corrected.
20. (nice, final) The bundle was one commit behind; X1's wall overlapped the D-015 checks; P1T ran once. Bundle
    rewritten, caveat stated, P1T repeated.
