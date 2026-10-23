# Claims extraction — review documents, part 2 (B2 results review, B3 revision notes, senior reviews Mar 23 / Jun 01)

Run: `runs/20260902T0000Z/phase1` · Extractor pass over four files, read in full (325 + 659 + 160 + 94 lines).

File labels used in the tables:

| label | file | lines |
|---|---|---|
| F1 | `/root/projects/ET-Miner/paper/review_b2_results.md` | 325 |
| F2 | `/root/projects/ET-Miner/paper/revision_notes_b3.tex` | 659 |
| F3 | `/root/projects/ET-Miner/paper/senior_review_jun01.md` | 160 |
| F4 | `/root/projects/ET-Miner/paper/senior_review_mar23.md` | 94 |

Conventions:

- IDs run R2-001… in file order F1→F4, then first-occurrence line order. The `line` column lists the first occurrence first, then every other line in the same file where the identical value with the identical meaning and stance recurs (exact repeats are consolidated into one row; a different rounding, meaning or stance gets its own row).
- Category: `deterministic` = should reproduce exactly from the same inputs/parameters (counts, Z-scores, fractions, memory footprints derived from data dimensions, arithmetic); `hardware-dependent` = timings, throughput, speedups, costs, GPU model/VRAM/count, host RAM; `method-parameter` = thresholds, min_count, permutation counts, bin definitions, vocabulary cut-offs; `external-fact` = literature/database facts (citations, other systems' scale, InterPro2GO mappings, UniProt sizes, prices); `software` = versions, commits, integer widths.
- Stance: `quotes-paper` (the text repeats a value it attributes to the paper, or to another document it is checking), `asserts-own` (the author's own recomputation, measurement, new result, or prediction), `disputes` (the text says a paper value is wrong, unsupported or implausible), `requests` (asks for a value to be added/changed/run).
- Nothing here is judged for correctness; every value may be wrong in the source.
- Not tabulated (locators, not claims): document dates (given in Section C), paper/bibliography/code line-number references (`line 396`, `r660`, `gpu_dispatch.py:128-131`, `apriori.py:235-268`), table/figure numbers, timestamps embedded in filenames, subagent IDs, LaTeX layout numbers (`\hskip 6pt`), review-process meta counts (3 axes, 3 subagents, 4 earlier review docs), and the purely hypothetical example patterns in F2 §7 (`EC:2.7.11.1`, `LEN:800--1000aa`, `pLDDT:<70` inside example braces — the bin edges themselves are tabulated from §2). `TBD` cells in F2 are listed in Section C, not as rows.

---

## SECTION A — CLAIMS TABLE

### F1 — `review_b2_results.md`

| ID | file | line | value | unit | category | stance | quoted context (≤ 25 words, verbatim) |
|---|---|---|---|---|---|---|---|
| R2-001 | F1 | 7 | 26.8M | itemsets in godmode parquet | deterministic | asserts-own | `archived/alphafold/results_214m/itemsets_214m_godmode.parquet` (26.8M itemsets) |
| R2-002 | F1 | 18, 42, 191, 195, 196, 304, 308 | 8 | min_count, godmode/main run (paper) | method-parameter | quotes-paper | Godmode (main result): min_count=8 … The godmode run used min_count=8 as claimed. |
| R2-003 | F1 | 22, 30, 38, 57, 61, 88, 130, 288, 317 | 76,890,945 | n_transactions (log / JSON) | deterministic | asserts-own | n_transactions \| 76,890,945 \| godmode log line 2 |
| R2-004 | F1 | 23 | 8 | min_count (godmode log) | method-parameter | asserts-own | min_count \| 8 \| godmode log line 2 |
| R2-005 | F1 | 24, 87 | 1.04e-07 (~0.00001%) | min_support (8 / 76,890,945) | method-parameter | asserts-own | min_support \| 1.04e-07 (~0.00001%) \| 8 / 76,890,945 |
| R2-006 | F1 | 26, 31, 42 | 768 | min_count, direct-vs-SON experiment (JSON) | method-parameter | asserts-own | Direct vs SON experiment: min_count=768 |
| R2-007 | F1 | 32, 40, 131 | 1e-05 (0.001%) | min_support, direct-vs-SON & null (JSON) | method-parameter | asserts-own | min_support \| 1e-05 (0.001%) \| JSON `parameters.min_support` |
| R2-008 | F1 | 34, 39, 42, 194 | 769 | min_count, null model (JSON) | method-parameter | asserts-own | Null model: min_count=769 |
| R2-009 | F1 | 44 | 96× | threshold ratio null/godmode | deterministic | asserts-own | operate at a 96x stricter threshold than the godmode run |
| R2-010 | F1 | 50, 288 | 76.9M | multi-feature proteins (paper) | deterministic | quotes-paper | Claim: 76.9M multi-feature proteins (after dedup), from 205.6M total. |
| R2-011 | F1 | 50 | 205.6M | total proteins (paper) | deterministic | quotes-paper | Claim: 76.9M multi-feature proteins (after dedup), from 205.6M total. |
| R2-012 | F1 | 56, 61, 318 | 205,620,298 | total proteins (paper) | deterministic | quotes-paper | Total proteins \| 205,620,298 \| 205,620,298 \| VERIFIED |
| R2-013 | F1 | 56 | 205,620,298 | total proteins (transactions parquet) | deterministic | asserts-own | Total proteins \| 205,620,298 \| 205,620,298 \| VERIFIED |
| R2-014 | F1 | 57, 61 | 76,890,945 | multi-feature (>1 item) proteins (paper) | deterministic | quotes-paper | Multi-feature (>1 item) \| 76,890,945 \| 76,890,945 \| VERIFIED |
| R2-015 | F1 | 58, 61 | 37.4% | multi-feature share (paper) | deterministic | quotes-paper | Percentage multi-feature \| 37.4% \| 37.39% \| VERIFIED |
| R2-016 | F1 | 58, 73 | 37.39% | multi-feature share (parquet) | deterministic | asserts-own | 37.39% have >1 feature (used in mining) |
| R2-017 | F1 | 59 | 128.7M | single-feature proteins (paper) | deterministic | quotes-paper | Single-feature \| 128.7M (62.6%) \| 128,729,353 (62.61%) \| VERIFIED |
| R2-018 | F1 | 59 | 62.6% | single-feature share (paper) | deterministic | quotes-paper | Single-feature \| 128.7M (62.6%) \| 128,729,353 (62.61%) \| VERIFIED |
| R2-019 | F1 | 59 | 128,729,353 | single-feature proteins (parquet) | deterministic | asserts-own | Single-feature \| 128.7M (62.6%) \| 128,729,353 (62.61%) \| VERIFIED |
| R2-020 | F1 | 59, 72 | 62.61% | single-feature share (parquet) | deterministic | asserts-own | 62.61% of proteins have only 1 feature (excluded from mining) |
| R2-021 | F1 | 67, 75, 289 | 20.45% | "dedup" rate (PROJECT_STATE.md, not the paper) | deterministic | quotes-paper | Claim (PROJECT_STATE.md only): "Dedup affected 20.45%" |
| R2-022 | F1 | 69 | ~20.45% | proteins with duplicate items after null shuffle | deterministic | asserts-own | after shuffling items across proteins, ~20.45% of proteins have duplicate items that must be removed |
| R2-023 | F1 | 79, 103, 189, 217, 290 | K=22 | max itemset depth, main run (paper) | deterministic | quotes-paper | K=22 Validation … the Opus K-distribution (K_max=22) |
| R2-024 | F1 | 85, 90, 111, 290, 323 | 1 | itemsets at K=22 in godmode parquet | deterministic | asserts-own | K=22 itemsets in godmode parquet \| **Exactly 1** |
| R2-025 | F1 | 86 | [1, 13, 23, 507, 509, 510, 513, 519, 529, 533, 569, 632, 651, 677, 766, 784, 795, 850, 887, 917, 965, 966] | item IDs of the K=22 itemset (22 IDs) | deterministic | asserts-own | Items in the itemset \| 22 IDs: [1, 13, 23, 507, 509, 510, 513, …] |
| R2-026 | F1 | 87 | 1.04e-07 | support of the K=22 itemset | deterministic | asserts-own | Support value \| 1.04e-07 |
| R2-027 | F1 | 88, 90, 290, 323 | 8 | proteins supporting the K=22 itemset | deterministic | asserts-own | Estimated protein count \| support * 76,890,945 = **8 proteins** |
| R2-028 | F1 | 96 | 1 (plddt_mean_med) | pLDDT features in K=22 itemset | deterministic | asserts-own | pLDDT \| 1 \| plddt_mean_med |
| R2-029 | F1 | 97 | 2 (PF00270, PF00271) | Pfam features in K=22 itemset | deterministic | asserts-own | Pfam \| 2 \| PF00270 (DEAD/DEAH N-term), PF00271 (Helicase C-term) |
| R2-030 | F1 | 98 | 8 (GO:0005524, GO:0016787, GO:0000287, GO:0003697, GO:0003724, GO:0003725, GO:0003678, GO:0000978) | GO MF features in K=22 itemset | deterministic | asserts-own | GO (MF) \| 8 \| GO:0005524, GO:0016787, GO:0000287, … |
| R2-031 | F1 | 99 | 4 (GO:0030154, GO:0045087, GO:0051607, GO:0034605) | GO BP features in K=22 itemset | deterministic | asserts-own | GO (BP) \| 4 \| GO:0030154, GO:0045087, GO:0051607, GO:0034605 |
| R2-032 | F1 | 100 | 7 (GO:0005737, GO:0005829, GO:0005634, GO:0005739, GO:0030424, GO:0030425, GO:0016607) | GO CC features in K=22 itemset | deterministic | asserts-own | GO (CC) \| 7 \| GO:0005737, GO:0005829, GO:0005634, … |
| R2-033 | F1 | 101, 103 | 22 | decoded features total (= paper Table 5) | deterministic | asserts-own | Feature decode matches Table 5 in the paper exactly. All 22 features confirmed. |
| R2-034 | F1 | 109, 156, 272, 293 | 1,002 | frequent K=1 items (parquet = log) | deterministic | asserts-own | 1 \| 1,002 \| 1,002 \| Yes |
| R2-035 | F1 | 110 | 3,529,257 | K=9 itemsets (parquet = log) | deterministic | asserts-own | 9 \| 3,529,257 \| 3,529,257 \| Yes |
| R2-036 | F1 | 112, 114, 316 | 26,849,505 | total itemsets (parquet = log) | deterministic | asserts-own | Parquet row count matches log exactly: 26,849,505 itemsets. |
| R2-037 | F1 | 124, 134 | 475,865 | Direct GPU itemsets @0.001% | deterministic | asserts-own | Direct GPU \| 475,865 \| 50.72s \| 14 |
| R2-038 | F1 | 124, 211 | 50.72 s | Direct GPU time (JSON) | hardware-dependent | asserts-own | Direct GPU \| 475,865 \| 50.72s \| 14 |
| R2-039 | F1 | 124, 198 | 14 | Direct GPU max K @0.001% | deterministic | asserts-own | Direct GPU \| 475,865 \| 50.72s \| 14 … real extends to K=14 vs null K=6 |
| R2-040 | F1 | 125, 134 | 22,846 | SON itemsets @0.001% | deterministic | asserts-own | SON \| 22,846 \| 1085.6s \| 13 |
| R2-041 | F1 | 125, 127, 210 | 1085.6 s | SON time (JSON) | hardware-dependent | asserts-own | SON \| 22,846 \| 1085.6s \| 13 |
| R2-042 | F1 | 125 | 13 | SON max K | deterministic | asserts-own | SON \| 22,846 \| 1085.6s \| 13 |
| R2-043 | F1 | 127, 136, 291, 319 | 21.4× | speedup 1085.6 / 50.72 | hardware-dependent | asserts-own | **Speedup:** 1085.6 / 50.72 = **21.4x** |
| R2-044 | F1 | 127 | 21× | speedup as rounded in paper caption | hardware-dependent | quotes-paper | paper rounds to "21x" in caption text, reports "21.4x" in controlled comparison |
| R2-045 | F1 | 132 | H100 | GPU model for both direct and SON runs | hardware-dependent | asserts-own | Same GPU: yes (both run on H100) -- **SAME** |
| R2-046 | F1 | 134 | 453,019 | itemsets missed by SON | deterministic | asserts-own | **SON itemset loss:** 475,865 - 22,846 = 453,019 (95.2% lost) |
| R2-047 | F1 | 134 | 95.2% | SON miss rate | deterministic | asserts-own | 453,019 (95.2% lost) -- **VERIFIED**, matches JSON `comparison.itemset_diff` |
| R2-048 | F1 | 146–150, 194 | 6 | null max K, each of 5 runs (JSON) | deterministic | asserts-own | 1 \| 6 \| 20 … 5 \| 6 \| 22 |
| R2-049 | F1 | 146 | 20 | null run 1, K=6 itemset count | deterministic | asserts-own | 1 \| 6 \| 20 |
| R2-050 | F1 | 147 | 22 | null run 2, K=6 count | deterministic | asserts-own | 2 \| 6 \| 22 |
| R2-051 | F1 | 148 | 23 | null run 3, K=6 count | deterministic | asserts-own | 3 \| 6 \| 23 |
| R2-052 | F1 | 149 | 22 | null run 4, K=6 count | deterministic | asserts-own | 4 \| 6 \| 22 |
| R2-053 | F1 | 150 | 22 | null run 5, K=6 count | deterministic | asserts-own | 5 \| 6 \| 22 |
| R2-054 | F1 | 152, 156, 160, 173, 177, 179, 181, 183, 209, 292, 294 | 5 | permutations in the null model | method-parameter | quotes-paper | Max K = 6 across all 5 permutations. Paper claim matches exactly. |
| R2-055 | F1 | 152, 292 | 6 | null max K (paper claim) | deterministic | quotes-paper | Max K = 6 across all 5 permutations. Paper claim matches exactly. |
| R2-056 | F1 | 156, 293 | 1,002 | K=1 count in every null run (marginals preserved) | deterministic | asserts-own | All 5 null runs produce K=1 = 1,002 (identical to real data). |
| R2-057 | F1 | 160 | 0 | null itemsets at K>=7, all 5 runs (JSON) | deterministic | asserts-own | All 5 runs: 0 itemsets at K>=7. |
| R2-058 | F1 | 160 | 0 at K>=7 | paper claim "no permutation produced any pattern at K>=7" | deterministic | quotes-paper | Paper claim "no permutation produced any pattern at K>=7" is correct. |
| R2-059 | F1 | 166 | 22,019 | K=2 biological itemsets (paper) | deterministic | quotes-paper | 2 \| 22,019 \| 22,019 \| 63,702 \| 63,702.4 \| -987 \| -987.08 |
| R2-060 | F1 | 166 | 22,019 | K=2 biological itemsets (JSON) | deterministic | asserts-own | 2 \| 22,019 \| 22,019 \| 63,702 \| 63,702.4 \| -987 \| -987.08 |
| R2-061 | F1 | 166 | 63,702 | K=2 null μ (paper) | deterministic | quotes-paper | 2 \| 22,019 \| 22,019 \| 63,702 \| 63,702.4 \| -987 \| -987.08 |
| R2-062 | F1 | 166 | 63,702.4 | K=2 null μ (JSON) | deterministic | asserts-own | 2 \| 22,019 \| 22,019 \| 63,702 \| 63,702.4 \| -987 \| -987.08 |
| R2-063 | F1 | 166 | −987 | K=2 Z (paper) | deterministic | quotes-paper | 2 \| 22,019 \| 22,019 \| 63,702 \| 63,702.4 \| -987 \| -987.08 |
| R2-064 | F1 | 166 | −987.08 | K=2 Z (JSON) | deterministic | asserts-own | 2 \| 22,019 \| 22,019 \| 63,702 \| 63,702.4 \| -987 \| -987.08 |
| R2-065 | F1 | 167 | 108,059 | K=4 bio (paper) | deterministic | quotes-paper | 4 \| 108,059 \| 108,059 \| 25,468 \| 25,467.8 \| +3,791 \| 3790.74 |
| R2-066 | F1 | 167 | 108,059 | K=4 bio (JSON) | deterministic | asserts-own | 4 \| 108,059 \| 108,059 \| 25,468 \| 25,467.8 \| +3,791 \| 3790.74 |
| R2-067 | F1 | 167 | 25,468 | K=4 null μ (paper) | deterministic | quotes-paper | 4 \| 108,059 \| 108,059 \| 25,468 \| 25,467.8 \| +3,791 \| 3790.74 |
| R2-068 | F1 | 167 | 25,467.8 | K=4 null μ (JSON) | deterministic | asserts-own | 4 \| 108,059 \| 108,059 \| 25,468 \| 25,467.8 \| +3,791 \| 3790.74 |
| R2-069 | F1 | 167 | +3,791 | K=4 Z (paper) | deterministic | quotes-paper | 4 \| 108,059 \| 108,059 \| 25,468 \| 25,467.8 \| +3,791 \| 3790.74 |
| R2-070 | F1 | 167 | 3790.74 | K=4 Z (JSON) | deterministic | asserts-own | 4 \| 108,059 \| 108,059 \| 25,468 \| 25,467.8 \| +3,791 \| 3790.74 |
| R2-071 | F1 | 168 | 78,596 | K=6 bio (paper) | deterministic | quotes-paper | 6 \| 78,596 \| 78,596 \| 22 \| 21.8 \| +71,728 \| 71728.1 |
| R2-072 | F1 | 168 | 78,596 | K=6 bio (JSON) | deterministic | asserts-own | 6 \| 78,596 \| 78,596 \| 22 \| 21.8 \| +71,728 \| 71728.1 |
| R2-073 | F1 | 168 | 22 | K=6 null μ (paper) | deterministic | quotes-paper | 6 \| 78,596 \| 78,596 \| 22 \| 21.8 \| +71,728 \| 71728.1 |
| R2-074 | F1 | 168 | 21.8 | K=6 null μ (JSON) | deterministic | asserts-own | 6 \| 78,596 \| 78,596 \| 22 \| 21.8 \| +71,728 \| 71728.1 |
| R2-075 | F1 | 168 | +71,728 | K=6 Z (paper) | deterministic | quotes-paper | 6 \| 78,596 \| 78,596 \| 22 \| 21.8 \| +71,728 \| 71728.1 |
| R2-076 | F1 | 168 | 71728.1 | K=6 Z (JSON) | deterministic | asserts-own | 6 \| 78,596 \| 78,596 \| 22 \| 21.8 \| +71,728 \| 71728.1 |
| R2-077 | F1 | 169, 322 | 88,745 | K=7–14 bio (paper) | deterministic | quotes-paper | 7-14 \| 88,745 \| 88,745 \| 0 \| 0 \| inf \| inf |
| R2-078 | F1 | 169 | 88,745 | K=7–14 bio (JSON) | deterministic | asserts-own | 7-14 \| 88,745 \| 88,745 \| 0 \| 0 \| inf \| inf |
| R2-079 | F1 | 169 | 0 | K=7–14 null μ (paper) | deterministic | quotes-paper | 7-14 \| 88,745 \| 88,745 \| 0 \| 0 \| inf \| inf |
| R2-080 | F1 | 169 | 0 | K=7–14 null μ (JSON) | deterministic | asserts-own | 7-14 \| 88,745 \| 88,745 \| 0 \| 0 \| inf \| inf |
| R2-081 | F1 | 169, 179 | inf | K=7–14 Z (paper) | deterministic | quotes-paper | 7-14 \| 88,745 \| 88,745 \| 0 \| 0 \| inf \| inf |
| R2-082 | F1 | 169 | inf | K=7–14 Z (JSON) | deterministic | asserts-own | 7-14 \| 88,745 \| 88,745 \| 0 \| 0 \| inf \| inf |
| R2-083 | F1 | 177 | < 5% | CV of null counts across runs, K=2..6 | deterministic | asserts-own | The null model is remarkably stable across runs (CV < 5% for all K levels). |
| R2-084 | F1 | 179 | 0 events in 5 trials | null K>=7 events | deterministic | asserts-own | With 0 events in 5 trials, the Clopper-Pearson 95% upper confidence bound on P(K>=7 \| null) |
| R2-085 | F1 | 179, 294 | 0.451 (45.1%) | Clopper-Pearson 95% upper bound on P(K>=7 given null) | deterministic | asserts-own | upper confidence bound on P(K>=7 \| null) is **0.451** (45.1%) |
| R2-086 | F1 | 179 | 45% | share of null runs not excludable from producing K>=7 | deterministic | asserts-own | we CANNOT statistically rule out that up to 45% of null runs might produce K>=7 patterns |
| R2-087 | F1 | 179 | p=0 | p-value for K>=7 (paper) | deterministic | disputes | The Z=infinity and p=0 claims for K>=7 are mathematical artifacts (division by zero in standard deviation) |
| R2-088 | F1 | 181, 310 | 100 | permutations needed (minimum) | method-parameter | requests | At least 100 permutations (upper bound: 0.030) |
| R2-089 | F1 | 181 | 0.030 | upper bound on P(K>=7 given null) with 100 permutations | deterministic | asserts-own | At least 100 permutations (upper bound: 0.030) |
| R2-090 | F1 | 181 | 1000 | permutations (ideal) | method-parameter | requests | ideally 1000 permutations (upper bound: 0.003) |
| R2-091 | F1 | 181 | 0.003 | upper bound with 1000 permutations | deterministic | asserts-own | ideally 1000 permutations (upper bound: 0.003) |
| R2-092 | F1 | 183 | 20–23 | K=6 null count range over 5 runs | deterministic | asserts-own | The extreme stability across the 5 runs (K=6 count: 20-23, near-zero variance) |
| R2-093 | F1 | 189, 304 | 769 vs 8 | min_count null vs main (paper line 396 quote) | method-parameter | quotes-paper | run at a higher threshold than the main analysis (min_count=769 vs. 8) for computational efficiency |
| R2-094 | F1 | 194, 198, 200, 304, 322 | 0.001% | null-model support | method-parameter | asserts-own | Null model at 0.001% (min_count=769): max K=6 |
| R2-095 | F1 | 195, 198, 308 | 0.00001% | main-run support | method-parameter | asserts-own | Real data at 0.00001% (min_count=8): max K=22 |
| R2-096 | F1 | 195, 198 | 22 | max K of real data at min_count=8 | deterministic | asserts-own | Real data at 0.00001% (min_count=8): max K=22 |
| R2-097 | F1 | 196 | UNKNOWN (could be K=7, 8, or higher) | null max K if run at min_count=8 | deterministic | disputes | If null model were run at 0.00001% (min_count=8): max K = **UNKNOWN** (could be K=7, 8, or higher) |
| R2-098 | F1 | 198, 304, 308 | K=15–22 | depth range not covered by any null test | deterministic | disputes | It does NOT validate the additional K=15 through K=22 patterns found at the lower threshold. |
| R2-099 | F1 | 200 | K>=7 | depth from which patterns are biological at 0.001% | deterministic | asserts-own | The core finding (K>=7 is biological at 0.001%) is likely sound. |
| R2-100 | F1 | 208, 296, 320 | 7.3 min | godmode run time (paper) | hardware-dependent | quotes-paper | Godmode: 7.3 min \| Log: "440.5s" \| 440.5 / 60 = 7.34 min -- **VERIFIED** |
| R2-101 | F1 | 208, 296, 320 | 440.5 s | godmode run time (log) | hardware-dependent | asserts-own | Godmode: 7.3 min \| Log: "440.5s" |
| R2-102 | F1 | 208 | 7.34 min | 440.5 / 60 | hardware-dependent | asserts-own | 440.5 / 60 = 7.34 min -- **VERIFIED** |
| R2-103 | F1 | 209, 321 | 662 s | null model total time (paper) | hardware-dependent | quotes-paper | Null total: 662s \| JSON: 662.17s |
| R2-104 | F1 | 209 | 662.17 s | null total time (JSON, sum of 5 runs) | hardware-dependent | asserts-own | Sum of 5 runs: 99.22+135.62+141.97+143.06+142.30 = 662.17s |
| R2-105 | F1 | 209 | 99.22 s | null run 1 time | hardware-dependent | asserts-own | Sum of 5 runs: 99.22+135.62+141.97+143.06+142.30 = 662.17s |
| R2-106 | F1 | 209 | 135.62 s | null run 2 time | hardware-dependent | asserts-own | Sum of 5 runs: 99.22+135.62+141.97+143.06+142.30 = 662.17s |
| R2-107 | F1 | 209 | 141.97 s | null run 3 time | hardware-dependent | asserts-own | Sum of 5 runs: 99.22+135.62+141.97+143.06+142.30 = 662.17s |
| R2-108 | F1 | 209 | 143.06 s | null run 4 time | hardware-dependent | asserts-own | Sum of 5 runs: 99.22+135.62+141.97+143.06+142.30 = 662.17s |
| R2-109 | F1 | 209 | 142.30 s | null run 5 time | hardware-dependent | asserts-own | Sum of 5 runs: 99.22+135.62+141.97+143.06+142.30 = 662.17s |
| R2-110 | F1 | 210 | 18.1 min | SON time (paper) | hardware-dependent | quotes-paper | SON: 18.1 min \| JSON: 1085.6s \| 1085.6 / 60 = 18.09 min -- **VERIFIED** |
| R2-111 | F1 | 210 | 18.09 min | 1085.6 / 60 | hardware-dependent | asserts-own | 1085.6 / 60 = 18.09 min -- **VERIFIED** |
| R2-112 | F1 | 211 | 50.7 s | Direct GPU time (paper) | hardware-dependent | quotes-paper | Direct GPU: 50.7s \| JSON: 50.72s \| **VERIFIED** |
| R2-113 | F1 | 222–224 | 2 GO terms (GO:0003676, GO:0005524) | InterPro2GO mapping of IPR011545 (PF00270) | external-fact | asserts-own | **IPR011545** (integrates PF00270 DEAD/DEAH helicase) maps to: GO:0003676 … **GO:0005524 (ATP binding)** |
| R2-114 | F1 | 225–226 | 0 GO terms | InterPro2GO mapping of IPR001650 (PF00271) | external-fact | asserts-own | **IPR001650** (integrates PF00271 Helicase C-terminal) maps to: No GO terms |
| R2-115 | F1 | 228, 249, 297 | 1 of 19 | GO terms in K=22 confirmed auto-derived (GO:0005524) | deterministic | asserts-own | Only **1 of 19 GO terms** (GO:0005524, ATP binding) is confirmed auto-derived from Pfam via InterPro2GO. |
| R2-116 | F1 | 228, 237, 241, 249, 297 | 19 | GO terms in the K=22 itemset | deterministic | asserts-own | we cannot determine how many of the 19 GO terms are experimentally validated |
| R2-117 | F1 | 243, 250 | 0 | parent-child GO pairs in the K=22 set | deterministic | asserts-own | **No parent-child pairs exist in the K=22 set.** |
| R2-118 | F1 | 243 | 22 | independent K after GO-hierarchy check (PROJECT_STATE.md) | deterministic | quotes-paper | PROJECT_STATE.md: "GO hierarchy: 0 parent-child pairs -> Independent K = 22" |
| R2-119 | F1 | 254 | 22 → 21 | effective independent feature count | deterministic | asserts-own | This reduces the effective independent feature count from 22 to 21. |
| R2-120 | F1 | 266, 279, 298, 302 | 247 | Pfam domains (paper Table 1) | deterministic | disputes | Pfam domains \| 247 \| **500** |
| R2-121 | F1 | 266, 275, 276, 279, 298, 302 | 500 | Pfam domains (item_mapping parquet) | deterministic | asserts-own | Actually 500 Pfam + 500 GO + 2 pLDDT = 1,002 |
| R2-122 | F1 | 267 | 302 | GO MF terms (paper Table 1) | deterministic | disputes | GO terms (MF) \| 302 \| **cannot split** |
| R2-123 | F1 | 268 | 289 | GO BP terms (paper Table 1) | deterministic | disputes | GO terms (BP) \| 289 \| **cannot split** |
| R2-124 | F1 | 269 | 161 | GO CC terms (paper Table 1) | deterministic | disputes | GO terms (CC) \| 161 \| **cannot split** |
| R2-125 | F1 | 270, 279, 298, 302 | 752 | GO terms total (paper Table 1) | deterministic | disputes | GO terms (total) \| 752 \| **500** |
| R2-126 | F1 | 270, 275, 276, 279, 298, 302 | 500 | GO terms (item_mapping parquet) | deterministic | asserts-own | GO terms (total) \| 752 \| **500** |
| R2-127 | F1 | 271, 276, 277, 302 | 3 | pLDDT bins (paper Table 1) | deterministic | disputes | pLDDT bins \| 3 \| **2** (frequent) |
| R2-128 | F1 | 271, 275, 277, 279, 298, 302 | 2 | pLDDT bins frequent at min_count=8 (plddt_mean_med, plddt_mean_high) | deterministic | asserts-own | only 2 are frequent (plddt_mean_med and plddt_mean_high; plddt_mean_low has < 8 proteins) |
| R2-129 | F1 | 272, 274, 302 | 1,002 | total features (paper Table 1 = parquet) | deterministic | quotes-paper | The total matches (1,002 frequent features), but the breakdown is wrong |
| R2-130 | F1 | 276 | 1,006 | entries in item_mapping_214m.parquet | deterministic | asserts-own | The item_mapping_214m.parquet contains 1,006 entries (500 pfam + 500 go_term + 6 plddt) |
| R2-131 | F1 | 276 | 6 | plddt entries in item mapping | deterministic | asserts-own | 1,006 entries (500 pfam + 500 go_term + 6 plddt) |
| R2-132 | F1 | 276 | 4 | mapping entries not frequent at min_count=8 | deterministic | asserts-own | of which 4 are not frequent at min_count=8 (plddt_mean_low + 3 plddt_fraction items) |
| R2-133 | F1 | 276 | 3 | plddt_fraction items (not frequent) | deterministic | asserts-own | (plddt_mean_low + 3 plddt_fraction items) |
| R2-134 | F1 | 277 | < 8 | proteins carrying plddt_mean_low | deterministic | asserts-own | plddt_mean_low has < 8 proteins |
| R2-135 | F1 | 279, 302 | 500:500:2 (not 247:752:3) | feature-type ratio | deterministic | asserts-own | The actual ratio is 500:500:2, not 247:752:3. |
| R2-136 | F1 | 308 | min_count=8 (0.00001%) | requested null-model threshold | method-parameter | requests | **Run null model at 0.00001% threshold (min_count=8)** to properly validate the K=22 result. |
| R2-137 | F1 | 312 | 21 | independent feature count to be noted in paper | deterministic | requests | "GO:0005524 (ATP binding) is automatically mapped from PF00270 via InterPro2GO, reducing the independent feature count to 21." |

### F2 — `revision_notes_b3.tex`

| ID | file | line | value | unit | category | stance | quoted context (≤ 25 words, verbatim) |
|---|---|---|---|---|---|---|---|
| R2-138 | F2 | 1, 14, 25, 54, 192, 210, 246, 251, 290, 312, 318, 322, 329, 336, 341, 355, 356, 363, 373, 377, 380, 413, 415, 422, 429, 437, 474, 534, 586, 600, 624, 639 | 35K | features, expanded vocabulary (rounded) | deterministic | asserts-own | LaTeX revision blocks for 35K-feature results |
| R2-139 | F2 | 14, 23, 39, 54, 60, 336, 387, 555, 565, 567, 574, 637 | 0.001% | support of the completed 35K run / controlled comparison | method-parameter | asserts-own | 35K MINING CAMPAIGN RESULTS (0.001% COMPLETE) |
| R2-140 | F2 | 17, 100, 117, 127, 178, 264, 319, 470 | 1,002 | features, initial vocabulary | deterministic | quotes-paper | after the existing Table 2 (1,002-feature campaign) |
| R2-141 | F2 | 21, 60, 99, 117, 127, 201, 215, 240, 264, 320, 388, 470, 628 | 35,012 | features, expanded vocabulary | deterministic | asserts-own | Mining campaign results with the expanded 35,012-feature vocabulary |
| R2-142 | F2 | 22, 177, 185 | 109.2 million | proteins, expanded dataset | deterministic | asserts-own | across 109.2 million proteins on 4$\times$NVIDIA H200 143\,GB (row-split architecture) |
| R2-143 | F2 | 22, 186, 230, 239, 383, 438, 602, 616, 624, 651 | 4 | GPU count (H200) for 35K experiments | hardware-dependent | asserts-own | on 4$\times$NVIDIA H200 143\,GB (row-split architecture) |
| R2-144 | F2 | 22, 186, 220, 230, 383, 438, 602, 616, 624, 651 | H200 (SXM5 at l.624) | GPU model | hardware-dependent | asserts-own | 4$\times$NVIDIA H200 143\,GB … 4$\times$NVIDIA H200 SXM5 141\,GB GPUs |
| R2-145 | F2 | 22, 187, 220, 641 | 143 GB | H200 VRAM | hardware-dependent | asserts-own | This exceeds the 143\,GB VRAM of the NVIDIA H200 |
| R2-146 | F2 | 23, 61 | 3 | replicates of the 0.001% run | method-parameter | asserts-own | The 0.001\% run reports mean $\pm$ std over 3 replicates [ACTUAL] |
| R2-147 | F2 | 37, 47, 585, 600, 605, 654 | 0.1% | Base run support | method-parameter | asserts-own | Base & 0.1\% & 109,225 & \textbf{TBD} & ${\geq}10$ & \textbf{TBD} |
| R2-148 | F2 | 37, 585, 605 | 109,225 | Base min proteins (min_count) | method-parameter | asserts-own | Base & 0.1\% & 109,225 … The Base run (0.1\%, min\_count=109,225) |
| R2-149 | F2 | 37 | ≥10 | Base max K (partial, run ongoing) | deterministic | asserts-own | ${\geq}10$\textsuperscript{$\ast$} … Base run (0.1\%) partial |
| R2-150 | F2 | 38, 655 | 0.01% | Super run support | method-parameter | asserts-own | Super & 0.01\% & 10,923 & \textbf{TBD} |
| R2-151 | F2 | 38 | 10,923 | Super min proteins | method-parameter | asserts-own | Super & 0.01\% & 10,923 |
| R2-152 | F2 | 39, 61, 637 | 606,292 ± 28 | Power itemsets, mean ± std over 3 replicates | deterministic | asserts-own | Power & 0.001\% & 1,093 & 606,292 $\pm$ 28 & 17 & 654.3 $\pm$ 6.7 |
| R2-153 | F2 | 39, 320, 336, 340 | 1,093 | Power min_count (0.001% of 109.2M) | method-parameter | asserts-own | Power & 0.001\% & 1,093 … min\_count$=$1,093 (0.001\% of 109M) |
| R2-154 | F2 | 39, 320, 322, 356, 431, 643, 658 | 17 | Power max K | deterministic | asserts-own | 606,292 $\pm$ 28 & 17 & 654.3 $\pm$ 6.7 … $K{=}17$ (35,012 features, min\_count$=$1,093) |
| R2-155 | F2 | 39 | 654.3 ± 6.7 s | Power time, mean ± std | hardware-dependent | asserts-own | 606,292 $\pm$ 28 & 17 & 654.3 $\pm$ 6.7 & Direct row-split |
| R2-156 | F2 | 40, 566, 655 | 0.0001% | Blitz run support | method-parameter | asserts-own | Blitz & 0.0001\% & 109 & \textbf{TBD} |
| R2-157 | F2 | 40 | 109 | Blitz min proteins | method-parameter | asserts-own | Blitz & 0.0001\% & 109 |
| R2-158 | F2 | 41 | 0.00002% | Ultra run support | method-parameter | asserts-own | Ultra & 0.00002\% & 22 & \textbf{TBD} |
| R2-159 | F2 | 41 | 22 | Ultra min proteins | method-parameter | asserts-own | Ultra & 0.00002\% & 22 |
| R2-160 | F2 | 42, 337 | 0.00001% | Opus run support | method-parameter | asserts-own | Opus & 0.00001\% & 11 & \textbf{TBD} |
| R2-161 | F2 | 42 | 11 | Opus min proteins (35K) | method-parameter | asserts-own | Opus & 0.00001\% & 11 |
| R2-162 | F2 | 48, 596, 606, 610 | 10,041,611 | Base K=9 frequent itemsets (partial) | deterministic | asserts-own | $K{=}9$ reached 10,041,611 frequent itemsets and $K{=}10$ started with 12,491,079 locally frequent candidates before timeout |
| R2-163 | F2 | 49, 597, 607 | 12,491,079 | Base K=10 locally frequent candidates (started, not completed) | deterministic | asserts-own | $K{=}10$ started with 12,491,079 locally frequent candidates before timeout. Run is ongoing. |
| R2-164 | F2 | 61, 79, 362 | K=7 | peak of the 35K 0.001% K-distribution | deterministic | asserts-own | Peak at $K{=}7$ (10.62\%). |
| R2-165 | F2 | 61, 79 | 10.62% | share of itemsets at K=7 | deterministic | asserts-own | Peak at $K{=}7$ (10.62\%). |
| R2-166 | F2 | 62–64 | deterministic at K≤3 and K≥11; variance at K=4..10 | replicate-variance profile | deterministic | asserts-own | Values deterministic at $K{\leq}3$ and $K{\geq}11$; minor variance at intermediate levels from GPU floating-point non-determinism |
| R2-167 | F2 | 73, 400, 448 | 34,920 | K=1 itemsets (35K, 0.001%) | deterministic | asserts-own | 1 & 34,920 & 5.76 & 10 & 43,869 & 7.24 |
| R2-168 | F2 | 73 | 5.76% | K=1 share | deterministic | asserts-own | 1 & 34,920 & 5.76 |
| R2-169 | F2 | 73 | 43,869 | K=10 itemsets | deterministic | asserts-own | 10 & 43,869 & 7.24 |
| R2-170 | F2 | 73 | 7.24% | K=10 share | deterministic | asserts-own | 10 & 43,869 & 7.24 |
| R2-171 | F2 | 74 | 46,853 | K=2 itemsets (kdist table) | deterministic | asserts-own | 2 & 46,853 & 7.73 & 11 & 30,116 & 4.97 |
| R2-172 | F2 | 74 | 7.73% | K=2 share | deterministic | asserts-own | 2 & 46,853 & 7.73 |
| R2-173 | F2 | 74 | 30,116 | K=11 itemsets | deterministic | asserts-own | 11 & 30,116 & 4.97 |
| R2-174 | F2 | 74 | 4.97% | K=11 share | deterministic | asserts-own | 11 & 30,116 & 4.97 |
| R2-175 | F2 | 75 | 53,238 | K=3 itemsets (kdist table) | deterministic | asserts-own | 3 & 53,238 & 8.78 & 12 & 17,355 & 2.86 |
| R2-176 | F2 | 75 | 8.78% | K=3 share | deterministic | asserts-own | 3 & 53,238 & 8.78 |
| R2-177 | F2 | 75 | 17,355 | K=12 itemsets | deterministic | asserts-own | 12 & 17,355 & 2.86 |
| R2-178 | F2 | 75 | 2.86% | K=12 share | deterministic | asserts-own | 12 & 17,355 & 2.86 |
| R2-179 | F2 | 76 | 59,086 | K=4 itemsets (kdist table) | deterministic | asserts-own | 4 & 59,086 & 9.75 & 13 & 8,096 & 1.34 |
| R2-180 | F2 | 76 | 9.75% | K=4 share | deterministic | asserts-own | 4 & 59,086 & 9.75 |
| R2-181 | F2 | 76 | 8,096 | K=13 itemsets | deterministic | asserts-own | 13 & 8,096 & 1.34 |
| R2-182 | F2 | 76 | 1.34% | K=13 share | deterministic | asserts-own | 13 & 8,096 & 1.34 |
| R2-183 | F2 | 77 | 62,226 | K=5 itemsets (kdist table) | deterministic | asserts-own | 5 & 62,226 & 10.26 & 14 & 2,946 & 0.49 |
| R2-184 | F2 | 77 | 10.26% | K=5 share | deterministic | asserts-own | 5 & 62,226 & 10.26 |
| R2-185 | F2 | 77 | 2,946 | K=14 itemsets | deterministic | asserts-own | 14 & 2,946 & 0.49 |
| R2-186 | F2 | 77 | 0.49% | K=14 share | deterministic | asserts-own | 14 & 2,946 & 0.49 |
| R2-187 | F2 | 78 | 63,708 | K=6 itemsets | deterministic | asserts-own | 6 & 63,708 & 10.51 & 15 & 800 & 0.13 |
| R2-188 | F2 | 78 | 10.51% | K=6 share | deterministic | asserts-own | 6 & 63,708 & 10.51 |
| R2-189 | F2 | 78 | 800 | K=15 itemsets | deterministic | asserts-own | 15 & 800 & 0.13 |
| R2-190 | F2 | 78 | 0.13% | K=15 share | deterministic | asserts-own | 15 & 800 & 0.13 |
| R2-191 | F2 | 79 | 64,396 | K=7 itemsets (peak) | deterministic | asserts-own | \textbf{7} & \textbf{64,396} & \textbf{10.62} & 16 & 152 & 0.03 |
| R2-192 | F2 | 79 | 152 | K=16 itemsets | deterministic | asserts-own | 16 & 152 & 0.03 |
| R2-193 | F2 | 79 | 0.03% | K=16 share | deterministic | asserts-own | 16 & 152 & 0.03 |
| R2-194 | F2 | 80 | 62,948 | K=8 itemsets | deterministic | asserts-own | 8 & 62,948 & 10.38 & 17 & 1 & $<\!$0.01 |
| R2-195 | F2 | 80 | 10.38% | K=8 share | deterministic | asserts-own | 8 & 62,948 & 10.38 |
| R2-196 | F2 | 80 | 1 | K=17 itemsets | deterministic | asserts-own | 17 & 1 & $<\!$0.01 |
| R2-197 | F2 | 80 | <0.01% | K=17 share | deterministic | asserts-own | 17 & 1 & $<\!$0.01 |
| R2-198 | F2 | 81 | 55,582 | K=9 itemsets | deterministic | asserts-own | 9 & 55,582 & 9.17 |
| R2-199 | F2 | 81 | 9.17% | K=9 share | deterministic | asserts-own | 9 & 55,582 & 9.17 |
| R2-200 | F2 | 99, 129, 355 | 7 | annotation sources in expanded vocabulary | method-parameter | asserts-own | The expanded vocabulary comprises 35,012 features across seven annotation sources |
| R2-201 | F2 | 100 | 35× | vocabulary size increase (35,012 / 1,002) | deterministic | asserts-own | a $35\times$ increase over the initial 1,002-feature set |
| R2-202 | F2 | 101, 185 | 3,423 | min feature count for vocabulary retention | method-parameter | asserts-own | Features retained at min\_count${\geq}$3,423. |
| R2-203 | F2 | 109, 131, 135, 329 | 12,773 | InterPro domain/family features | deterministic | asserts-own | Protein domains \& families & InterPro & 12,773 & (247) |
| R2-204 | F2 | 109, 134, 326, 329 | 247 | Pfam domains in the 1,002 set | deterministic | quotes-paper | Protein domains \& families & InterPro & 12,773 & (247) |
| R2-205 | F2 | 110, 137, 140 | 5,763 | GO terms (expanded) | deterministic | asserts-own | GO terms (all three ontologies) & Gene Ontology & 5,763 & (752) |
| R2-206 | F2 | 110, 139, 326 | 752 | GO terms in the 1,002 set | deterministic | quotes-paper | GO terms (all three ontologies) & Gene Ontology & 5,763 & (752) |
| R2-207 | F2 | 111, 144, 330 | 1,107 | EC number features (leaf-only) | deterministic | asserts-own | Enzyme Commission numbers & IUBMB (leaf-only) & 1,107 & --- |
| R2-208 | F2 | 112, 151, 330 | 15,162 | UniProt Keyword features | deterministic | asserts-own | UniProt Keywords & UniProtKB & 15,162 & --- |
| R2-209 | F2 | 113, 158 | 175 | taxonomic-class features (leaf-only) | deterministic | asserts-own | Taxonomic lineage & UniProt (class-level, leaf-only) & 175 & --- |
| R2-210 | F2 | 114, 166, 168 | 26 | sequence-length bins (log-scale) | method-parameter | asserts-own | Sequence length bins & Derived (log-scale) & 26 & --- |
| R2-211 | F2 | 115, 172, 173 | 6 | pLDDT confidence bins (expanded) | method-parameter | asserts-own | pLDDT confidence bins & AlphaFold & 6 & (3) |
| R2-212 | F2 | 115, 175 | 3 | pLDDT bins in the 1,002 set | method-parameter | quotes-paper | pLDDT confidence bins & AlphaFold & 6 & (3) … expanded from the 3-bin encoding in the initial vocabulary |
| R2-213 | F2 | 127, 264, 470 | 1,002 → 35,012 | vocabulary expansion (old → new) | deterministic | asserts-own | We expanded the feature vocabulary from 1,002 to 35,012 features |
| R2-214 | F2 | 134–135 | 247 → 12,773 | domain coverage, Pfam → InterPro (old → new) | deterministic | asserts-own | increasing domain coverage from 247 to 12,773 entries while eliminating deduplication concerns |
| R2-215 | F2 | 139–140 | 752 → 5,763 | GO coverage (old → new) | deterministic | asserts-own | expand coverage from 752 to 5,763 terms by lowering the frequency threshold |
| R2-216 | F2 | 145 | 4 | levels in the EC hierarchy | external-fact | asserts-own | EC numbers classify enzyme-catalyzed reactions in a four-level hierarchy (class.subclass.sub-subclass.serial) |
| R2-217 | F2 | 167–168 | 100–150, 150–200, …, >5,000 aa | length-bin edges (log-scale, 26 bins) | method-parameter | asserts-own | log-scale bins (e.g., 100--150~aa, 150--200~aa, $\ldots$, $>$5{,}000~aa), providing 26 discrete size categories |
| R2-218 | F2 | 173–174 | 3 mean-pLDDT bins (<70, 70–90, >90) + 3 fraction bins | pLDDT binning scheme (6 bins) | method-parameter | asserts-own | three for mean pLDDT ($<$70, 70--90, $>$90) and three for the fraction of high-confidence residues |
| R2-219 | F2 | 175 | 3 → 6 | pLDDT bins (old → new) | method-parameter | asserts-own | expanded from the 3-bin encoding in the initial vocabulary |
| R2-220 | F2 | 177–178 | 76.9M → 109.2M | proteins with ≥1 feature (old → new) | deterministic | asserts-own | 109.2~million proteins with at least one annotation feature (up from 76.9~million in the 1,002-feature set) |
| R2-221 | F2 | 178, 337 | 76.9 million (77M at l.337) | proteins in the 1,002-feature set | deterministic | quotes-paper | up from 76.9~million in the 1,002-feature set … (0.00001\% of 77M) |
| R2-222 | F2 | 180 | 150 GB | UniProt .dat.gz input processed | external-fact | asserts-own | Feature extraction processed 150\,GB of UniProt \texttt{.dat.gz} files using a custom Rust parser |
| R2-223 | F2 | 182 | 95.6K records/s | dat_to_tsv.rs parser throughput | hardware-dependent | asserts-own | (\texttt{dat\_to\_tsv.rs}, 95.6K records/s) |
| R2-224 | F2 | 183 | 495K lines/s | build_tx.rs throughput | hardware-dependent | asserts-own | two-pass Rust transaction builder (\texttt{build\_tx.rs}, 495K lines/s) |
| R2-225 | F2 | 184 | 858 s | transaction build time (frequency counting + remapping) | hardware-dependent | asserts-own | performed frequency counting and feature remapping in 858\,s |
| R2-226 | F2 | 185 | ~0.003% | 3,423 as fraction of 109.2M | method-parameter | asserts-own | 3,423 (corresponding to ${\sim}0.003\%$ of 109.2M proteins) |
| R2-227 | F2 | 186, 629 | 477 GB | total bitvector matrix (35K × 109.2M) | deterministic | asserts-own | constrain the bitvector matrix to 477\,GB total across four NVIDIA H200 143\,GB GPUs |
| R2-228 | F2 | 187, 241, 628 | ~119 GB | bitvector shard per device | deterministic | asserts-own | (${\sim}119$\,GB per device with 12\,GB headroom for workspace allocations) |
| R2-229 | F2 | 187 | 12 GB | per-device headroom for workspace | hardware-dependent | asserts-own | ${\sim}119$\,GB per device with 12\,GB headroom for workspace allocations |
| R2-230 | F2 | 207, 215–216, 271, 642 | \|F\| × ⌈N/64⌉ × 8 B ; T_bitvec = \|C_K\|·K·⌈N/64⌉ | bitvector memory / cost model | deterministic | asserts-own | the bitvector matrix for each chunk has dimensions $\|F\| \times \lceil N_{\text{chunk}} / 64 \rceil$ |
| R2-231 | F2 | 212, 215 | 40M | SON chunk size (example) | method-parameter | asserts-own | A single chunk of $N_{\text{chunk}} = 40$M transactions requires: |
| R2-232 | F2 | 216, 641 | 175 GB | chunk bitvector at 40M transactions, 35,012 features | deterministic | asserts-own | 35{,}012 \times \left\lceil \frac{40 \times 10^6}{64} \right\rceil \times 8 \;\text{bytes} = 175\;\text{GB} |
| R2-233 | F2 | 220–221 | H200 = highest-capacity datacenter GPU currently available | hardware ranking claim | external-fact | asserts-own | the 143\,GB VRAM of the NVIDIA H200---the highest-capacity datacenter GPU currently available |
| R2-234 | F2 | 222 | 30M | reduced SON chunk size | method-parameter | asserts-own | (e.g., $N_{\text{chunk}} = 30$M $\rightarrow$ 131\,GB) |
| R2-235 | F2 | 222 | 131 GB | chunk bitvector at 30M | deterministic | asserts-own | (e.g., $N_{\text{chunk}} = 30$M $\rightarrow$ 131\,GB) |
| R2-236 | F2 | 223 | 4 | sequential SON passes ⌈109M/30M⌉ | deterministic | asserts-own | would require at least $\lceil 109\text{M} / 30\text{M} \rceil = 4$ sequential passes |
| R2-237 | F2 | 223, 292, 295, 336 | 109M / 109 million | proteins (rounded) | deterministic | asserts-own | Across 109~million transactions, this produces $8.6 \times 10^{10}$ subset enumerations |
| R2-238 | F2 | 231 | > 22 hours | SON streaming run hung at Pass 1 (4×H200) before termination | hardware-dependent | asserts-own | a SON streaming run on 4$\times$H200 GPUs hung at Pass~1 for over 22 hours before being terminated |
| R2-239 | F2 | 239 | ~27M | proteins per GPU (4 GPUs) | deterministic | asserts-own | Each GPU holds a fraction of the proteins (${\sim}27$M each on 4 GPUs) |
| R2-240 | F2 | 242, 630 | 1 | allreduce per K-level | method-parameter | asserts-own | Local support counts are reduced across GPUs with a single \texttt{allreduce} operation per $K$-level |
| R2-241 | F2 | 250, 542, 553, 567, 575, 649 | 21.4× | SON slowdown / Direct GPU speedup (1K features, 0.001%) | hardware-dependent | asserts-own | ``SON is $21.4\times$ slower'' (Section~\ref{sec:results}, 1K features) |
| R2-242 | F2 | 286 | T_sparse = N · C(\|t\|, K) | hash-tree Apriori cost model | deterministic | asserts-own | T_{\text{sparse}} = N \cdot \binom{\bar{\|t\|}}{K} |
| R2-243 | F2 | 291, 355 | ~12 | average annotated features per protein (35K) | deterministic | asserts-own | With the 35K-feature dataset ($\bar{\|t\|} \approx 12$) at $K{=}5$ |
| R2-244 | F2 | 292 | 792 | C(12,5) subsets per transaction at K=5 | deterministic | asserts-own | each transaction generates $\binom{12}{5} = 792$ subsets |
| R2-245 | F2 | 293 | 8.6 × 10^10 | subset enumerations at K=5 across 109M transactions | deterministic | asserts-own | Across 109~million transactions, this produces $8.6 \times 10^{10}$ subset enumerations at $K{=}5$ alone |
| R2-246 | F2 | 295–296, 300 | ~1.7 × 10^6 | 64-bit ops per candidate, ⌈109×10^6/64⌉ | deterministic | asserts-own | $\lceil 109{\times}10^6 / 64 \rceil \approx 1.7 \times 10^6$ 64-bit operations per candidate at \emph{any} $K$ |
| R2-247 | F2 | 297 | 100 | hypothetical average transaction length | method-parameter | asserts-own | At $\bar{\|t\|}{=}100$ and $K{=}5$: |
| R2-248 | F2 | 298 | 75,287,520 | C(100,5) subsets per transaction | deterministic | asserts-own | sparse traversal generates $\binom{100}{5} = 75{,}287{,}520$ subsets per transaction |
| R2-249 | F2 | 319, 323, 339, 353, 358, 643 | K=22 | 1K max depth (min_count=8) | deterministic | quotes-paper | maximum itemset depth from $K{=}22$ (1,002 features, min\_count$=$8) to $K{=}17$ |
| R2-250 | F2 | 319, 337, 338 | 8 | 1K Opus min_count | method-parameter | quotes-paper | $K{=}22$ (1,002 features, min\_count$=$8) … min\_count$=$8 (0.00001\% of 77M) for the 1K Opus run |
| R2-251 | F2 | 319–320, 643 | K=22 → K=17 | max itemset depth, 1K → 35K (old → new) | deterministic | asserts-own | the \emph{decrease} in maximum itemset depth from $K{=}22$ … to $K{=}17$ (35,012 features, min\_count$=$1,093) |
| R2-252 | F2 | 337 | 137× | absolute threshold ratio 1,093 / 8 | deterministic | asserts-own | a $137\times$ higher absolute threshold |
| R2-253 | F2 | 339, 353 | 8 | proteins sharing the K=22 itemset | deterministic | quotes-paper | (the $K{=}22$ itemset was shared by exactly 8 proteins) |
| R2-254 | F2 | 340 | > 1,000 | proteins a pattern must reach at min_count=1,093 | method-parameter | asserts-own | At min\_count$=$1,093, only patterns present in $>$1,000 proteins qualify. |
| R2-255 | F2 | 346 | 1 (not 4) | features per EC annotation under leaf-only encoding | method-parameter | asserts-own | EC~3.4.21.4 contributes one feature, not four (the leaf plus three ancestors) |
| R2-256 | F2 | 347 | 7–8 | "free" GO co-occurrences per protein from true-path rule (1K vocabulary) | deterministic | asserts-own | GO's true-path rule contributed up to 7--8 ``free'' co-occurrences per protein from hierarchical nesting alone |
| R2-257 | F2 | 353–354 | 8 proteins with exactly 22 features each | K=22 ceiling determined by annotation depth | deterministic | asserts-own | The $K{=}22$ ceiling in the 1K dataset was determined by 8 proteins with exactly 22 features each |
| R2-258 | F2 | 359 | 1 | pLDDT bin in the 1K K=22 pattern | deterministic | asserts-own | combined only Pfam and GO terms with one pLDDT bin, all from two closely related annotation sources |
| R2-259 | F2 | 362 | K=9 | 1K K-distribution peak | deterministic | quotes-paper | The $K$-distribution peak also shifted: from $K{=}9$ (1K features) to $K{=}7$ (35K features) |
| R2-260 | F2 | 362 | K=9 → K=7 | K-distribution peak, 1K → 35K (old → new) | deterministic | asserts-own | from $K{=}9$ (1K features) to $K{=}7$ (35K features) |
| R2-261 | F2 | 370, 373, 382, 388, 644 | 2 | permutations completed (35K null model) | method-parameter | asserts-own | Two permutations were completed on 4$\times$H200 GPUs |
| R2-262 | F2 | 373 | 5 | permutations in the existing (1K) null model | method-parameter | quotes-paper | Replace existing 5-permutation null model with 35K 2-perm data. |
| R2-263 | F2 | 374, 434, 437, 647, 656 | 100 | permutations planned (TBD) | method-parameter | requests | Prepare for 100-permutation update. … 35K null model: 100-permutation results (\textbf{TBD}) |
| R2-264 | F2 | 383 | 1,310 s | total GPU time, 2 permutations | hardware-dependent | asserts-own | (total GPU time: 1,310\,s, wall-clock: 1,456\,s) |
| R2-265 | F2 | 383 | 1,456 s | wall-clock, 2 permutations | hardware-dependent | asserts-own | (total GPU time: 1,310\,s, wall-clock: 1,456\,s) |
| R2-266 | F2 | 388–389, 415, 646 | K=5 | null model max K (35K) | deterministic | asserts-own | The null model collapses at $K{=}5$ (1 itemset per permutation) |
| R2-267 | F2 | 389, 404, 415 | 1 | null itemsets at K=5, per permutation | deterministic | asserts-own | it barely reaches $K{=}5$ (exactly 1 itemset per permutation) |
| R2-268 | F2 | 390, 402, 423, 645 | +722.9 (≈723) | K=3 Z (35K) | deterministic | asserts-own | ($Z{=}723$ at $K{=}3$, $Z{>}20{,}000$ at $K{=}4$, $Z{=}\infty$ for $K{\geq}5$) |
| R2-269 | F2 | 390, 403, 645 | +20,585 (>20,000) | K=4 Z (35K) | deterministic | asserts-own | 4 & 59,095 & 872 & 2.8 & $+20{,}585$ & ${\approx}\,0$ & enriched |
| R2-270 | F2 | 390–391, 404, 405 | +∞ | Z for K≥5 (35K) | deterministic | asserts-own | $Z{=}\infty$ for $K{\geq}5$ |
| R2-271 | F2 | 400 | 34,920 | K=1 null μ (marginals preserved) | deterministic | asserts-own | 1 & 34,920 & 34,920 & 0.0 & 0.0 & 1.0 & preserved |
| R2-272 | F2 | 400 | 0.0 | K=1 null σ | deterministic | asserts-own | 1 & 34,920 & 34,920 & 0.0 & 0.0 & 1.0 & preserved |
| R2-273 | F2 | 400 | 0.0 | K=1 Z | deterministic | asserts-own | 1 & 34,920 & 34,920 & 0.0 & 0.0 & 1.0 & preserved |
| R2-274 | F2 | 400 | 1.0 | K=1 p | deterministic | asserts-own | 1 & 34,920 & 34,920 & 0.0 & 0.0 & 1.0 & preserved |
| R2-275 | F2 | 401, 449 | 46,847 | K=2 bio (null-model table) | deterministic | asserts-own | 2 & 46,847 & 59,415 & 123.7 & $-101.6$ & 1.0 & depleted |
| R2-276 | F2 | 401 | 59,415 | K=2 null μ | deterministic | asserts-own | 2 & 46,847 & 59,415 & 123.7 & $-101.6$ & 1.0 & depleted |
| R2-277 | F2 | 401 | 123.7 | K=2 null σ | deterministic | asserts-own | 2 & 46,847 & 59,415 & 123.7 & $-101.6$ & 1.0 & depleted |
| R2-278 | F2 | 401 | −101.6 | K=2 Z | deterministic | asserts-own | 2 & 46,847 & 59,415 & 123.7 & $-101.6$ & 1.0 & depleted |
| R2-279 | F2 | 401 | 1.0 | K=2 p | deterministic | asserts-own | 2 & 46,847 & 59,415 & 123.7 & $-101.6$ & 1.0 & depleted |
| R2-280 | F2 | 402, 450 | 53,235 | K=3 bio (null-model table) | deterministic | asserts-own | 3 & 53,235 & 15,920 & 51.6 & $+722.9$ & ${\approx}\,0$ & enriched |
| R2-281 | F2 | 402 | 15,920 | K=3 null μ | deterministic | asserts-own | 3 & 53,235 & 15,920 & 51.6 & $+722.9$ |
| R2-282 | F2 | 402 | 51.6 | K=3 null σ | deterministic | asserts-own | 3 & 53,235 & 15,920 & 51.6 & $+722.9$ |
| R2-283 | F2 | 402 | ≈0 | K=3 p | deterministic | asserts-own | 3 & 53,235 & 15,920 & 51.6 & $+722.9$ & ${\approx}\,0$ & enriched |
| R2-284 | F2 | 403, 451 | 59,095 | K=4 bio (null-model table) | deterministic | asserts-own | 4 & 59,095 & 872 & 2.8 & $+20{,}585$ & ${\approx}\,0$ & enriched |
| R2-285 | F2 | 403 | 872 | K=4 null μ | deterministic | asserts-own | 4 & 59,095 & 872 & 2.8 & $+20{,}585$ |
| R2-286 | F2 | 403 | 2.8 | K=4 null σ | deterministic | asserts-own | 4 & 59,095 & 872 & 2.8 & $+20{,}585$ |
| R2-287 | F2 | 403 | ≈0 | K=4 p | deterministic | asserts-own | 4 & 59,095 & 872 & 2.8 & $+20{,}585$ & ${\approx}\,0$ & enriched |
| R2-288 | F2 | 404, 452 | 62,241 | K=5 bio (null-model table) | deterministic | asserts-own | 5 & 62,241 & 1.0 & 0.0 & $+\infty$ & ${\approx}\,0$ & enriched |
| R2-289 | F2 | 404 | 1.0 | K=5 null μ | deterministic | asserts-own | 5 & 62,241 & 1.0 & 0.0 & $+\infty$ |
| R2-290 | F2 | 404 | 0.0 | K=5 null σ | deterministic | asserts-own | 5 & 62,241 & 1.0 & 0.0 & $+\infty$ |
| R2-291 | F2 | 404 | ≈0 | K=5 p | deterministic | asserts-own | 5 & 62,241 & 1.0 & 0.0 & $+\infty$ & ${\approx}\,0$ & enriched |
| R2-292 | F2 | 405, 432, 453 | 348,853 | K=6–17 bio itemsets | deterministic | asserts-own | 6--17 & 348,853 & 0 & 0.0 & $+\infty$ & ${\approx}\,0$ & enriched |
| R2-293 | F2 | 405, 416 | 0 | K=6–17 null μ (zero itemsets at K≥6) | deterministic | asserts-own | 6--17 & 348,853 & 0 & 0.0 … produces zero itemsets at $K{\geq}6$ |
| R2-294 | F2 | 405 | 0.0 | K=6–17 null σ | deterministic | asserts-own | 6--17 & 348,853 & 0 & 0.0 & $+\infty$ |
| R2-295 | F2 | 405 | ≈0 | K=6–17 p | deterministic | asserts-own | 6--17 & 348,853 & 0 & 0.0 & $+\infty$ & ${\approx}\,0$ & enriched |
| R2-296 | F2 | 414, 646 | K=6 | 1K null-model max K | deterministic | quotes-paper | With 1K features, the null model reached $K{=}6$ (22 itemsets mean) |
| R2-297 | F2 | 414 | 22 | 1K null K=6 mean itemsets | deterministic | quotes-paper | With 1K features, the null model reached $K{=}6$ (22 itemsets mean) |
| R2-298 | F2 | 421 | Z=−143 | 1K K=3 Z (depleted) | deterministic | quotes-paper | With 1K features, $K{=}3$ was depleted ($Z{=}{-}143$) |
| R2-299 | F2 | 423 | 3.3× | bio/null triples ratio at K=3 (35K) | deterministic | asserts-own | biology produces $3.3\times$ more triples than chance |
| R2-300 | F2 | 429 | 5.5× | total bio/null itemset ratio (35K) | deterministic | asserts-own | The 35K dataset produces $5.5\times$ more total itemsets than the null (606K vs.\ 111K) |
| R2-301 | F2 | 430, 645 | 606K | bio total itemsets (rounded) | deterministic | asserts-own | (606K vs.\ 111K) |
| R2-302 | F2 | 430, 645 | 111K | null total itemsets (35K, 2 permutations) | deterministic | asserts-own | (606K vs.\ 111K) |
| R2-303 | F2 | 431 | 12 | K-levels beyond null reach (K=6..17) | deterministic | asserts-own | With 12 levels beyond the null's reach ($K{=}6$ through $K{=}17$) containing 348,853 itemsets |
| R2-304 | F2 | 438 | ~655 s | time per permutation (GPU Fisher–Yates, 4×H200) | hardware-dependent | asserts-own | GPU-accelerated Fisher--Yates shuffle on 4$\times$H200, ${\sim}655$\,s per permutation. |
| R2-305 | F2 | 474–476 | 4 | additional annotation sources vs 1K vocabulary | method-parameter | asserts-own | The 35K vocabulary introduces four additional annotation sources (EC numbers, UniProt Keywords, taxonomic lineage, and sequence length) |
| R2-306 | F2 | 476 | 6 | categories of cross-source patterns | deterministic | asserts-own | creating six categories of cross-source patterns that could not be discovered with any single-source analysis |
| R2-307 | F2 | 548, 552, 558, 576 | 475,865 | Direct GPU itemsets (1K, 0.001%) — retained in replacement text | deterministic | quotes-paper | Direct GPU discovers 475,865 itemsets in 50.7\,s versus SON's 22,846 in 1,085.6\,s |
| R2-308 | F2 | 548, 552, 571, 575 | 50.7 s | Direct GPU time (1K) | hardware-dependent | quotes-paper | Direct GPU discovers 475,865 itemsets in 50.7\,s |
| R2-309 | F2 | 549, 552, 558, 576 | 22,846 | SON itemsets (1K) | deterministic | quotes-paper | versus SON's 22,846 in 1,085.6\,s |
| R2-310 | F2 | 549, 553, 571, 575 | 1,085.6 s | SON time (1K) | hardware-dependent | quotes-paper | versus SON's 22,846 in 1,085.6\,s |
| R2-311 | F2 | 549, 562, 571 | 21× | speedup as currently written in paper (three places) | hardware-dependent | quotes-paper | REPLACE: "… 1,085.6\,s---a $21\times$ speedup." |
| R2-312 | F2 | 548–553 | 21× → 21.4× | Section 3.1 speedup wording (old → new) | hardware-dependent | requests | REPLACE: "…a $21\times$ speedup." WITH: … a $\mathbf{21.4\times}$ speedup (Table~\ref{tab:campaign}, footnote) |
| R2-313 | F2 | 555 | 769 | min_count of the controlled comparison (0.001%) | method-parameter | asserts-own | isolates the method effect at identical support (0.001\%, $\text{min\_count}{=}769$) |
| R2-314 | F2 | 558, 567, 576 | 95.2% | SON miss rate (22,846 of 475,865) | deterministic | asserts-own | SON misses 95.2\% of patterns (22,846 of 475,865) through two compounding mechanisms. |
| R2-315 | F2 | 561–568 | 21× → 21.4× (+ 95.2% miss rate added) | Table 2 footnote (old → new) | hardware-dependent | requests | Replace: "Controlled same-support comparison: $21\times$" With: … $21.4\times$ speedup, 95.2\% SON miss rate |
| R2-316 | F2 | 565–566 | Power (0.001%) vs Blitz (0.0001%) | what the Table 2 † footnote compares (method and threshold both vary) | method-parameter | asserts-own | Compares Power (0.001\%) to Blitz (0.0001\%), varying both method and threshold. |
| R2-317 | F2 | 570–576 | 21× → 21.4× | Discussion 4.1 speedup wording (old → new) | hardware-dependent | requests | Replace: "a $21\times$ speedup (50.7\,s vs.\ 1,085.6\,s)" With: … a $21.4\times$ speedup (50.7\,s vs.\ 1,085.6\,s) |
| R2-318 | F2 | 588, 608 | 1,164 | Base (0.1%) K=1 frequent items | deterministic | asserts-own | K=1:   1,164 frequent items (0.3s) |
| R2-319 | F2 | 588 | 0.3 s | Base K=1 time | hardware-dependent | asserts-own | K=1:   1,164 frequent items (0.3s) |
| R2-320 | F2 | 589, 609 | 8,554 | Base K=2 frequent pairs | deterministic | asserts-own | K=2:   8,554 frequent pairs (1.2s) |
| R2-321 | F2 | 589 | 1.2 s | Base K=2 time | hardware-dependent | asserts-own | K=2:   8,554 frequent pairs (1.2s) |
| R2-322 | F2 | 590, 609 | 28,804 | Base K=3 frequent | deterministic | asserts-own | K=3:  28,804 frequent (2.0s) |
| R2-323 | F2 | 590 | 2.0 s | Base K=3 time | hardware-dependent | asserts-own | K=3:  28,804 frequent (2.0s) |
| R2-324 | F2 | 591, 609 | 88,561 | Base K=4 frequent | deterministic | asserts-own | K=4:  88,561 frequent (5.3s) |
| R2-325 | F2 | 591 | 5.3 s | Base K=4 time | hardware-dependent | asserts-own | K=4:  88,561 frequent (5.3s) |
| R2-326 | F2 | 592, 609 | 280,784 | Base K=5 frequent | deterministic | asserts-own | K=5: 280,784 frequent (17.5s) |
| R2-327 | F2 | 592 | 17.5 s | Base K=5 time | hardware-dependent | asserts-own | K=5: 280,784 frequent (17.5s) |
| R2-328 | F2 | 593, 609 | 835,466 | Base K=6 frequent | deterministic | asserts-own | K=6: 835,466 frequent (52.1s) |
| R2-329 | F2 | 593 | 52.1 s | Base K=6 time | hardware-dependent | asserts-own | K=6: 835,466 frequent (52.1s) |
| R2-330 | F2 | 594, 610 | 2,207,022 | Base K=7 frequent | deterministic | asserts-own | K=7: 2,207,022 frequent (145.3s) |
| R2-331 | F2 | 594 | 145.3 s | Base K=7 time | hardware-dependent | asserts-own | K=7: 2,207,022 frequent (145.3s) |
| R2-332 | F2 | 595, 610 | 5,063,845 | Base K=8 frequent | deterministic | asserts-own | K=8: 5,063,845 frequent (348.5s) |
| R2-333 | F2 | 595 | 348.5 s | Base K=8 time | hardware-dependent | asserts-own | K=8: 5,063,845 frequent (348.5s) |
| R2-334 | F2 | 596 | 743.4 s | Base K=9 time | hardware-dependent | asserts-own | K=9: 10,041,611 frequent (743.4s) |
| R2-335 | F2 | 600–601 | > 15 | expected Base K-max (prediction) | deterministic | asserts-own | At 0.1% support with 35K features, we may see K-max > 15 and total itemsets in the tens of millions. |
| R2-336 | F2 | 601, 611–612 | tens of millions | expected Base total itemsets (prediction) | deterministic | asserts-own | the complete Base run will yield tens of millions of itemsets at high $K$ values |
| R2-337 | F2 | 625 | 141 GB | H200 SXM5 VRAM (hardware section) | hardware-dependent | asserts-own | performed on 4$\times$NVIDIA H200 SXM5 141\,GB GPUs with 1.5\,TB host RAM |
| R2-338 | F2 | 625 | 1.5 TB | host RAM | hardware-dependent | asserts-own | 141\,GB GPUs with 1.5\,TB host RAM |
| R2-339 | F2 | 625 | Python 3.12 | software version | software | asserts-own | running Python~3.12, CuPy~13.4, NumPy~2.0, CUDA~12.6, and Ubuntu~22.04 |
| R2-340 | F2 | 625 | CuPy 13.4 | software version | software | asserts-own | running Python~3.12, CuPy~13.4, NumPy~2.0, CUDA~12.6, and Ubuntu~22.04 |
| R2-341 | F2 | 626 | NumPy 2.0 | software version | software | asserts-own | NumPy~2.0, CUDA~12.6, and Ubuntu~22.04 |
| R2-342 | F2 | 626 | CUDA 12.6 | software version | software | asserts-own | NumPy~2.0, CUDA~12.6, and Ubuntu~22.04 |
| R2-343 | F2 | 626 | Ubuntu 22.04 | OS version | software | asserts-own | NumPy~2.0, CUDA~12.6, and Ubuntu~22.04 |
| R2-344 | F2 | 627 | ~27.3M | proteins per GPU (row-split) | deterministic | asserts-own | The row-split architecture distributes ${\sim}27.3$M proteins per GPU |
| R2-345 | F2 | 630 | ~12 bytes | nccl.allReduce payload per K-level | deterministic | asserts-own | Local support counts are reduced across GPUs via \texttt{nccl.allReduce} with ${\sim}12$\,bytes per $K$-level. |

### F3 — `senior_review_jun01.md`

| ID | file | line | value | unit | category | stance | quoted context (≤ 25 words, verbatim) |
|---|---|---|---|---|---|---|---|
| R2-346 | F3 | 4 | 968 | lines in `et_miner_proteome.tex` (reviewed version) | deterministic | asserts-own | `/home/et/personal-projects/et-miner/papers/et_miner_proteome.tex` (968 regels) |
| R2-347 | F3 | 18, 94, 98 | 26,849,505 | kdist sum K=1..22 (recounted by hand) | deterministic | asserts-own | kdist K=1..22 = 26,849,505 exact … kdist-som = 26,849,505 (handmatig nageteld ✓) |
| R2-348 | F3 | 18, 98 | 16,812,646,639 | expanded-run cumulative itemsets (sum verified) | deterministic | asserts-own | expanded cumulatief = 16,812,646,639 exact |
| R2-349 | F3 | 19, 100 | 9× | speedup/ratio in paper (verified) | hardware-dependent | quotes-paper | speedup-ratio's (9×, 124×, 21×, 95.2%, 5.1×) kloppen binnen afronding |
| R2-350 | F3 | 19, 100 | 124× | speedup/ratio in paper (verified) | hardware-dependent | quotes-paper | speedup-ratio's (9×, 124×, 21×, 95.2%, 5.1×) kloppen binnen afronding |
| R2-351 | F3 | 19, 100 | 21× | Direct-vs-SON speedup in paper (verified) | hardware-dependent | quotes-paper | speedup-ratio's (9×, 124×, 21×, 95.2%, 5.1×) kloppen binnen afronding |
| R2-352 | F3 | 19, 101 | 95.2% | SON miss rate in paper (verified) | deterministic | quotes-paper | speedup-ratio's (9×, 124×, 21×, 95.2%, 5.1×) kloppen binnen afronding |
| R2-353 | F3 | 19, 55, 101 | 5.1× | scale ratio vs GMiner in paper (verified) | deterministic | quotes-paper | "15M" (scale-tabel, basis van de 5.1×-claim) |
| R2-354 | F3 | 31 | 37 | `\bibitem` entries (r574–757) | deterministic | asserts-own | Bibliografie: 37 `\bibitem` (r574–757). 32 distincte `\cite`-keys, allemaal resolvend |
| R2-355 | F3 | 31 | 32 | distinct `\cite` keys, all resolving | deterministic | asserts-own | 32 distincte `\cite`-keys, allemaal resolvend (geen broken refs) |
| R2-356 | F3 | 39 | 3077 → 3077–3083 | coin2009 (actually Terrapon et al.) page range (old → new) | external-fact | disputes | Pagina's ook incompleet (3077 → 3077–3083) |
| R2-357 | F3 | 40, 149 | 50(D1):D419–D427 → D439–D444 | varadi2022 page range (old → new) | external-fact | disputes | `50(D1):D419--D427` → moet **D439–D444** zijn |
| R2-358 | F3 | 40 | D412–D419 | mistry2021 page range (r712), source of the copy error | external-fact | quotes-paper | collideert exact met de echte range van `mistry2021` (`D412--D419`, r712) → klassieke copy-fout |
| R2-359 | F3 | 41 | 100M | BIGMiner transactions as stated in paper table | external-fact | disputes | Tabel zegt BIGMiner = "100M transactions / 30× servers" |
| R2-360 | F3 | 41 | 30 | BIGMiner servers/nodes as stated in paper (unverifiable) | external-fact | disputes | Het "30 nodes"-getal is niet verifieerbaar uit open bronnen. |
| R2-361 | F3 | 41 | 6.5 billion | BIGMiner transactions per the BIGMiner paper | external-fact | asserts-own | De BIGMiner-paper rapporteert schaling tot **6,5 miljard** transacties |
| R2-362 | F3 | 41 | ≈65× | 6.5B / 100M | external-fact | asserts-own | (≈65× meer dan de geclaimde 100M) |
| R2-363 | F3 | 41 | three orders of magnitude | paper's scale claim vs prior work | deterministic | disputes | verzwak je juist de claim "three orders of magnitude beyond any prior result" |
| R2-364 | F3 | 47 | ~100K | Meysman2015 structures as stated in paper (r499) | external-fact | disputes | Claim-support: "limited to ${\sim}100$K structures" — werkelijk **~32.142** structuren (overschat ~3×) |
| R2-365 | F3 | 47 | ~32,142 | Meysman2015 structures (actual) | external-fact | asserts-own | werkelijk **~32.142** structuren (overschat ~3×) |
| R2-366 | F3 | 47 | ~3× | overestimate factor | external-fact | asserts-own | (overschat ~3×) |
| R2-367 | F3 | 47 | ~32K | suggested replacement value | external-fact | requests | `${\sim}32$K structures` |
| R2-368 | F3 | 48 | 54(7):1–36 → 54(9), Article 179 (179:1–179:35) | acmsurvey2021 issue/pages (old → new) | external-fact | disputes | Issue + pagina's fout: `54(7):1--36` → **54(9), Article 179 (179:1–179:35)** |
| R2-369 | F3 | 49 | 2008 tech-report title vs DaMoN 2009 paper, pp 34–42 | fang2009 title/venue/pages | external-fact | disputes | de titel van het **2008 HKUST tech-report**; de DaMoN-2009 workshop-paper heet "**Frequent itemset mining on graphics processors**" (pp 34–42) |
| R2-370 | F3 | 50 | vol 250, Art. 123928 | chon2024 ESWA volume/article (title wrong) | external-fact | asserts-own | ESWA vol 250, Art. 123928. |
| R2-371 | F3 | 51 | 52(D1):D368 → D368–D375 | varadi2024 page range (old → new) | external-fact | disputes | Ontbrekende eind-pagina: `52(D1):D368` → **D368–D375** |
| R2-372 | F3 | 52 | 21:1507–1521 → 21(3):1507–1520 | chon2018b pages (old → new) | external-fact | disputes | Eind-pagina off-by-one: `21:1507--1521` → **21(3):1507–1520** |
| R2-373 | F3 | 53 | pp. 432–444 | savasere1995 pages (missing in paper) | external-fact | requests | Pagina's ontbreken volledig. \| `pp.\ 432--444` |
| R2-374 | F3 | 54 | 50–350× | GPU speedup range claimed in paper (r106; range verified) | external-fact | quotes-paper | voor "GPU achieved 50–350× speedups" … De range zelf klopt |
| R2-375 | F3 | 54 | 350× | djenouri2019 speedup (confirmed) | external-fact | asserts-own | (350× = djenouri2019, 100× = zhang2011, beide bevestigd) |
| R2-376 | F3 | 54 | 100× | zhang2011 speedup (confirmed) | external-fact | asserts-own | (350× = djenouri2019, 100× = zhang2011, beide bevestigd) |
| R2-377 | F3 | 54 | 2 | CPU algorithms (han2000 FP-Growth, zaki2000 Eclat) wrongly inside the GPU-speedup cite cluster | external-fact | disputes | bundelt twee **CPU-algoritmes** (`han2000` FP-Growth, `zaki2000` Eclat) onder een GPU-speedup-claim |
| R2-378 | F3 | 55 | 15M | GMiner transactions, scale table (r466) | external-fact | disputes | Zelfde systeem krijgt "15M" (scale-tabel, basis van de 5.1×-claim) én "1.7M (real)" |
| R2-379 | F3 | 55 | 1.7M | GMiner transactions, gpu-arch table (r877) | external-fact | quotes-paper | én "1.7M (real)" (gpu-arch-tabel). 1.7M = webdocs FIMI (bevestigd 1,692,082) |
| R2-380 | F3 | 55 | 1,692,082 | webdocs FIMI transaction count | external-fact | asserts-own | 1.7M = webdocs FIMI (bevestigd 1,692,082) |
| R2-381 | F3 | 55 | 4× GTX 1080 | GMiner hardware as stated in paper (unverifiable) | hardware-dependent | quotes-paper | (GPU "4× GTX 1080" niet verifieerbaar — paywall.) |
| R2-382 | F3 | 59–60 | 5 (cited 0×) | orphan bibitems: webb2007, webb2014, abramson2024, zaki1997, miettinen2020 | deterministic | asserts-own | `webb2007` (r724), `webb2014` (r729), `abramson2024` (r734), `zaki1997` (r749), `miettinen2020` (r754) — … **nergens ge-`\cite`d** |
| R2-383 | F3 | 76 | 316M | CSR nnz (r175) | deterministic | quotes-paper | Body: "316M nnz, ~5.1 GB, **two** 64-bit ints/entry" (16 B/entry, COO-paar) |
| R2-384 | F3 | 76 | ~5.1 GB | CSR/COO size (r175) | deterministic | quotes-paper | Body: "316M nnz, ~5.1 GB, **two** 64-bit ints/entry" |
| R2-385 | F3 | 76 | 2 × 64-bit ints = 16 B/entry | COO entry size (r175) | deterministic | quotes-paper | "**two** 64-bit ints/entry" (16 B/entry, COO-paar) |
| R2-386 | F3 | 76 | ~3 GB | H2D transfer size (r179/r485) | deterministic | quotes-paper | Elders: H2D-transfer "~3 GB" (alleen kolom-index, 8 B → 316M×8=2.53 GB) |
| R2-387 | F3 | 76 | 8 B/entry | CSR column-index entry size (appendix r920) | deterministic | quotes-paper | Appendix-tabel: "CSR (**8** bytes/entry)" → 19 GB voor 214M |
| R2-388 | F3 | 76 | 2.53 GB | 316M × 8 B | deterministic | asserts-own | (alleen kolom-index, 8 B → 316M×8=2.53 GB) |
| R2-389 | F3 | 76, 152 | 19 GB | CSR size for 214M proteins (appendix r920) | deterministic | quotes-paper | Appendix-tabel: "CSR (**8** bytes/entry)" → 19 GB voor 214M |
| R2-390 | F3 | 76, 152 | 3 (3 / 5.1 / 19 GB) | distinct CSR sizes given without labels | deterministic | disputes | Drie cijfers (3 / 5.1 / 19 GB) voor "CSR" zonder dat de lezer ze kan rijmen. |
| R2-391 | F3 | 76 | nnz of the 214M set | value missing from paper | deterministic | requests | geef de nnz van de 214M-set (nu nergens vermeld) |
| R2-392 | F3 | 77 | 40× | memory reduction (r175) | deterministic | quotes-paper | 40× = naïeve **byte**-matrix (206 GB, 1 B/boolean) ÷ CSR-COO (5.1 GB) |
| R2-393 | F3 | 77 | 1.4× | memory reduction (r908/r920) | deterministic | quotes-paper | 1.4× = **bitpacked** dense (27 GB) ÷ CSR (19 GB) |
| R2-394 | F3 | 77, 152 | 206 GB | naïve byte matrix (1 B/boolean) | deterministic | quotes-paper | naïeve **byte**-matrix (206 GB, 1 B/boolean) |
| R2-395 | F3 | 77 | 1 B/boolean | byte-matrix convention | deterministic | asserts-own | 206 GB = 1 byte/boolean (niet bitpacked) |
| R2-396 | F3 | 77 | 27 GB | bitpacked dense (appendix) | deterministic | quotes-paper | **bitpacked** dense (27 GB) ÷ CSR (19 GB) |
| R2-397 | F3 | 77, 152 | 26 GB | actual on-GPU bitvector | deterministic | asserts-own | de echte on-GPU bitvector is 26 GB = de "27 GB" appendix-dense, factor ~8 kleiner |
| R2-398 | F3 | 77 | ~8× | 206 GB / 26 GB | deterministic | asserts-own | factor ~8 kleiner |
| R2-399 | F3 | 78 | ~445 GB | expanded bitvector size (r258) | deterministic | disputes | **`~445 GB` expanded bitvector: GB/GiB-mix** |
| R2-400 | F3 | 78 | ~56 GB/device | expanded per-device bitvector (paper) | deterministic | quotes-paper | "~56 GB/device × 8" = 448, niet 445 |
| R2-401 | F3 | 78, 92 | 8 | devices in the expanded run (r258 "× 8"; r965 "× 8") | hardware-dependent | quotes-paper | "~56 GB/device × 8" = 448 … "$332 ≈ $3.50/GPU/uur × 8 × ~12h" |
| R2-402 | F3 | 78 | 448 GB | 56 × 8 | deterministic | asserts-own | "~56 GB/device × 8" = 448, niet 445 |
| R2-403 | F3 | 78 | 478 GB | 109.2M × 35012 / 8 (decimal GB) | deterministic | asserts-own | Decimaal 109.2M×35012/8 = **478 GB** |
| R2-404 | F3 | 78 | 109.2M | expanded protein count | deterministic | quotes-paper | Decimaal 109.2M×35012/8 = **478 GB** |
| R2-405 | F3 | 78 | 35012 | expanded feature count | deterministic | quotes-paper | Decimaal 109.2M×35012/8 = **478 GB** |
| R2-406 | F3 | 78 | 445.19 GiB | same size in GiB | deterministic | asserts-own | 445 klopt **alleen** als **GiB** (445.19 GiB) |
| R2-407 | F3 | 78 | 55.6 GiB | 56 GB per device in GiB | deterministic | asserts-own | als GiB, dan is "56 GB/device" ook fout (→ 55.6 GiB) |
| R2-408 | F3 | 84, 142, 153 | p < 0.17 | one-sided binomial p for 0/5 permutations > K=6 (r414, r429, r435) | deterministic | disputes | "$p < 0.17$ by one-sided binomial test" … **Niet onderbouwd door de standaard one-sided binomiale upper bound.** |
| R2-409 | F3 | 84 | 0/5 | permutations exceeding K=6 | deterministic | quotes-paper | (0/5 permutaties > K=6) |
| R2-410 | F3 | 84 | K=6 | 1K null-model ceiling | deterministic | quotes-paper | (0/5 permutaties > K=6) |
| R2-411 | F3 | 84, 153 | ≈0.45 | 95% upper bound for 0 successes in 5 trials | deterministic | asserts-own | Voor 0 successen in 5 trials is de 95%-upper-bound ≈ **0.45** |
| R2-412 | F3 | 84 | 3/5 = 0.6 | rule-of-three bound | deterministic | asserts-own | (rule-of-three 3/5 = 0.6) |
| R2-413 | F3 | 84 | 1/6 | 0.17 ≈ 1/6, origin unclear | deterministic | asserts-own | 0.17 ≈ 1/6 — herkomst onduidelijk en oogt te optimistisch. |
| R2-414 | F3 | 90 | 1,092 | expanded-run min_count (r260) | method-parameter | disputes | min_count expanded = 1,092 \| 260 \| ⌈0.001%×109,224,173⌉ = ⌈1092.24⌉ = **1093** |
| R2-415 | F3 | 90 | 109,224,173 | expanded-run n (exact) | deterministic | quotes-paper | ⌈0.001%×109,224,173⌉ = ⌈1092.24⌉ = **1093** |
| R2-416 | F3 | 90 | 1092.24 | 0.001% × 109,224,173 | deterministic | asserts-own | ⌈0.001%×109,224,173⌉ = ⌈1092.24⌉ = **1093** |
| R2-417 | F3 | 90 | 1,092 → 1093 | min_count fix (old → new) if ceil is used | method-parameter | requests | 1092→1093, óf vermeld dat round() gebruikt is en pas r800 aan. |
| R2-418 | F3 | 90 | 769 | Power min_count (ceil applied) | method-parameter | quotes-paper | Power gebruikt wél ceil (769). Richting inconsistent. |
| R2-419 | F3 | 91 | $243 | saving stated in paper (r965) | hardware-dependent | disputes | "$243 saving" \| 965 \| 332 − 90 = **242**, niet 243. |
| R2-420 | F3 | 91, 92 | $332 | run cost stated in paper | hardware-dependent | quotes-paper | 332 − 90 = **242** … "$332 ≈ $3.50/GPU/uur × 8 × ~12h" |
| R2-421 | F3 | 91 | $90 | comparison cost stated in paper | hardware-dependent | quotes-paper | 332 − 90 = **242**, niet 243. |
| R2-422 | F3 | 91 | $243 → $242 | saving (old → new) | hardware-dependent | requests | `\$242` |
| R2-423 | F3 | 92 | $3.50/GPU/hour | GPU rental price used in paper (r940) | external-fact | quotes-paper | "$332 ≈ $3.50/GPU/uur × 8 × ~12h" |
| R2-424 | F3 | 92 | ~12 h | run duration used in paper cost formula | hardware-dependent | quotes-paper | "$332 ≈ $3.50/GPU/uur × 8 × ~12h" \| 965 vs 940 |
| R2-425 | F3 | 92 | $336 | 3.50 × 8 × 12 | hardware-dependent | asserts-own | 3.50×8×12 = **$336**; $332 impliceert 11.86 h (≠ "~12 hours") |
| R2-426 | F3 | 92 | 11.86 h | hours implied by $332 | hardware-dependent | asserts-own | $332 impliceert 11.86 h (≠ "~12 hours") |
| R2-427 | F3 | 93, 99 | 62.6% | excluded single-feature proteins (Limitation 4, r525; base run) | deterministic | quotes-paper | "excludes ... (62.6%)" (Limitation 4) \| 525 \| Getal correct voor de **base**-run (37.4+62.6=100) |
| R2-428 | F3 | 93, 99 | 37.4% | multi-feature share (base run) | deterministic | quotes-paper | Getal correct voor de **base**-run (37.4+62.6=100) |
| R2-429 | F3 | 93, 99 | 53.1% | expanded-run coverage claim | deterministic | quotes-paper | spanning met de 53.1%-expanded-claim |
| R2-430 | F3 | 94 | 26.8M | total itemsets as written (r301, r306) | deterministic | disputes | "26.8M total" \| 301, 306 \| 26,849,505 → 26.85M; "26.8M" is **truncatie** i.p.v. afronding |
| R2-431 | F3 | 94 | 26.85M | correct 2-decimal rounding | deterministic | asserts-own | 26,849,505 → 26.85M |
| R2-432 | F3 | 94 | 26.9M | correct 1-decimal rounding | deterministic | asserts-own | "26.8M" is **truncatie** i.p.v. afronding (zou 26.9M zijn) |
| R2-433 | F3 | 99 | 7 | growth factors in paper (all verified) | deterministic | quotes-paper | alle 7 groeifactoren (25.8×…3.4×) ✓ |
| R2-434 | F3 | 99 | 25.8× | largest growth factor | deterministic | quotes-paper | alle 7 groeifactoren (25.8×…3.4×) ✓ |
| R2-435 | F3 | 99 | 3.4× | smallest growth factor | deterministic | quotes-paper | alle 7 groeifactoren (25.8×…3.4×) ✓ |
| R2-436 | F3 | 99 | 76,891 | largest min_count (base run) | method-parameter | quotes-paper | alle min_counts (76,891 … 8) ✓ |
| R2-437 | F3 | 99 | 8 | smallest min_count | method-parameter | quotes-paper | alle min_counts (76,891 … 8) ✓ |
| R2-438 | F3 | 100 | 13.14% | percentage in paper (quantity unspecified; verified) | deterministic | quotes-paper | 37.4% / 53.1% / 13.14% ✓ |
| R2-439 | F3 | 100 | Z > 3,700 for K=4–6 | Z-scores, abstract↔table↔conclusion | deterministic | quotes-paper | Z>3,700 voor K=4–6 consistent abstract↔tabel↔conclusie ✓ |
| R2-440 | F3 | 101 | 475,865 | null-table Bio column total = Direct-GPU total | deterministic | quotes-paper | null Bio-kolom = 475,865 = Direct-GPU-totaal ✓ |
| R2-441 | F3 | 101 | 0.32 | pruning factor, technique 1 | hardware-dependent | quotes-paper | pruning 1−(0.32)(0.88)(0.96) = 73% ✓ |
| R2-442 | F3 | 101 | 0.88 | pruning factor, technique 2 | hardware-dependent | quotes-paper | pruning 1−(0.32)(0.88)(0.96) = 73% ✓ |
| R2-443 | F3 | 101 | 0.96 | pruning factor, technique 3 | hardware-dependent | quotes-paper | pruning 1−(0.32)(0.88)(0.96) = 73% ✓ |
| R2-444 | F3 | 102, 115 | 73% | combined projected K=9 runtime reduction | hardware-dependent | quotes-paper | pruning 1−(0.32)(0.88)(0.96) = 73% ✓ … "are estimated to reduce $K{=}9$ runtime by approximately 73%" |
| R2-445 | F3 | 102 | ~264 B | total allreduce transfer (22 × 12) | deterministic | quotes-paper | ~264 B totaal transfer (22×12) ✓ |
| R2-446 | F3 | 102 | 22 | K-levels in the transfer calculation | deterministic | quotes-paper | ~264 B totaal transfer (22×12) ✓ |
| R2-447 | F3 | 102, 127 | 12 bytes | per-K-level allreduce transfer (r842/r873) | deterministic | quotes-paper | `$\sim$12 bytes` (873) vs `${\sim}12$~bytes` (842) |
| R2-448 | F3 | 109 | 70–90 / 583–589 / K=4–8 | en-dash range examples (pLDDT bin, citation pages, K range) | external-fact | quotes-paper | en-dash-ranges (`70--90`, `583--589`, `$K{=}4$--$8$`) zijn consistent en correct toegepast |
| R2-449 | F3 | 115 | K=9 | level of the projected runtime reduction (r92, r549) | deterministic | quotes-paper | "we estimate would reduce** $K{=}9$ runtime …" |
| R2-450 | F3 | 121 | 40+ | occurrences of "ET-miner" (vs one "ET-Miner", r795) | deterministic | asserts-own | enige uitschieter tussen 40+ `ET-miner` |
| R2-451 | F3 | 123 | 7.3 minutes | run time (r121, r92) | hardware-dependent | quotes-paper | r121 `7.3~minutes` (beschermd), r92/r130/r549 plain spatie ("7.3 minutes", "63 minutes", "4.3 hours") |
| R2-452 | F3 | 123 | 63 minutes | run time (r130) | hardware-dependent | quotes-paper | ("7.3 minutes", "63 minutes", "4.3 hours") |
| R2-453 | F3 | 123, 143 | 4.3 hours | run time (r549; expanded 35K run per l.143) | hardware-dependent | quotes-paper | ("7.3 minutes", "63 minutes", "4.3 hours") … **16,8 miljard itemsets t/m K=8, 4.3 uur** |
| R2-454 | F3 | 123 | 80 GB | house-style unit example (`80\,GB`) | hardware-dependent | quotes-paper | Huis-stijl is `\,` (80\,GB). |
| R2-455 | F3 | 125 | Opus 4.6 | model name in author info (r559 "Opus4.6") | software | quotes-paper | **`Opus4.6`** mist spatie (author-info). \| `Opus 4.6` |
| R2-456 | F3 | 138, 149 | 247/752/3 → 500/500/2 | Table 1 breakdown fixed between versions (r143–150) | deterministic | asserts-own | Table 1 feature-breakdown "fabricated" (247/752/3) \| ✅ **GEFIXT** — toont nu 500/500/2 (r143–150). |
| R2-457 | F3 | 140 | 2 | pLDDT bins, Run 1 (current version, r150) | deterministic | quotes-paper | pLDDT-bins 3 vs 2 \| ✅ **GEFIXT** — Run 1 = 2 bins (r150). |
| R2-458 | F3 | 143 | 606,292 | 35K @0.001% itemsets per revision_notes_b3 | deterministic | quotes-paper | `revision_notes_b3.tex` noteerde voor de 35K-run @ 0.001%: **606,292 itemsets, K_max=17, 654 s** |
| R2-459 | F3 | 143 | K_max=17 | 35K @0.001% max K per revision_notes_b3 | deterministic | quotes-paper | **606,292 itemsets, K_max=17, 654 s** |
| R2-460 | F3 | 143 | 654 s | 35K @0.001% time per revision_notes_b3 | hardware-dependent | quotes-paper | **606,292 itemsets, K_max=17, 654 s** |
| R2-461 | F3 | 143 | 16.8 billion | 35K @0.001% itemsets through K=8 (current paper) | deterministic | quotes-paper | De huidige paper claimt voor dezelfde run **16,8 miljard itemsets t/m K=8, 4.3 uur** |
| R2-462 | F3 | 143 | K=8 | depth reached in current paper's 35K run | deterministic | quotes-paper | **16,8 miljard itemsets t/m K=8, 4.3 uur** |
| R2-463 | F3 | 143 | ~27,000× | ratio between the two 35K results | deterministic | asserts-own | Dat is ~27.000× verschil — vrijwel zeker het effect van de eerder genoemde **NCCL int32-overflow bugfix** |
| R2-464 | F3 | 143 | int32 | NCCL overflow bug (integer width) | software | asserts-own | het effect van de eerder genoemde **NCCL int32-overflow bugfix** (pre-fix undercount) |

### F4 — `senior_review_mar23.md`

| ID | file | line | value | unit | category | stance | quoted context (≤ 25 words, verbatim) |
|---|---|---|---|---|---|---|---|
| R2-465 | F4 | 15, 20, 22 | 247 | Pfam domains, Run 1 (paper Table 1, lines 143–145) | deterministic | disputes | Pfam domains & InterPro/Pfam & 247 & --- \\ … The numbers 247 and 752 do not correspond to any configuration in the codebase. |
| R2-466 | F4 | 16, 20, 22 | 752 | GO terms, Run 1 (paper Table 1) | deterministic | disputes | GO terms & Gene Ontology & 752 & 5,763 \\ |
| R2-467 | F4 | 16 | 5,763 | GO terms, Run 2 (paper Table 1) | deterministic | quotes-paper | GO terms & Gene Ontology & 752 & 5,763 \\ |
| R2-468 | F4 | 17, 20, 70 | 3 | pLDDT bins, Run 1 (paper Table 1) | deterministic | disputes | pLDDT confidence bins & AlphaFold & 3 & 6 \\ … Table 1 claims 3 pLDDT bins for Run 1. |
| R2-469 | F4 | 17, 70 | 6 | pLDDT bins, Run 2 / defined in item mapping | deterministic | quotes-paper | pLDDT confidence bins & AlphaFold & 3 & 6 \\ … The item mapping defines 6 bins but 4 have < 8 proteins. |
| R2-470 | F4 | 20, 22 | 1,002 | Run 1 feature total | deterministic | quotes-paper | The paper claims Run 1 consists of 247 Pfam + 752 GO + 3 pLDDT = 1,002. |
| R2-471 | F4 | 22, 89 | 500 | Pfam domains, actual (item_mapping_214m.parquet) | deterministic | asserts-own | **The actual composition is 500 Pfam + 500 GO + 2 pLDDT = 1,002** |
| R2-472 | F4 | 22, 89 | 500 | GO terms, actual | deterministic | asserts-own | **The actual composition is 500 Pfam + 500 GO + 2 pLDDT = 1,002** |
| R2-473 | F4 | 22, 89 | 2 | pLDDT bins, actual | deterministic | asserts-own | **The actual composition is 500 Pfam + 500 GO + 2 pLDDT = 1,002** |
| R2-474 | F4 | 22 | top_pfam=500, top_go=500 | pipeline parameters (non-default) | method-parameter | asserts-own | Pipeline was run with `top_pfam=500, top_go=500` (non-default parameters). |
| R2-475 | F4 | 27, 29 | 769 vs 8 | min_count null vs main (paper line 438 quote) | method-parameter | quotes-paper | run at a higher threshold than the main analysis (min_count=769 vs. 8) for computational efficiency |
| R2-476 | F4 | 29 | 8 | min_count requested for null-model rerun | method-parameter | requests | Fix: either run the null model at min_count=8, or honestly state that K=15-22 significance is extrapolated. |
| R2-477 | F4 | 29 | K=15–22 | depth range whose significance is extrapolated | deterministic | disputes | or honestly state that K=15-22 significance is extrapolated |
| R2-478 | F4 | 42 | 68% | pruning saving, technique 1 (Table 8, K=9) — projection | hardware-dependent | disputes | Table 8 presents pruning savings of 68%, 12%, 4%, 2% (combined 73%) for K=9. PROJECT_STATE confirms these are projections |
| R2-479 | F4 | 42 | 12% | pruning saving, technique 2 — projection | hardware-dependent | disputes | pruning savings of 68%, 12%, 4%, 2% (combined 73%) for K=9 |
| R2-480 | F4 | 42 | 4% | pruning saving, technique 3 — projection | hardware-dependent | disputes | pruning savings of 68%, 12%, 4%, 2% (combined 73%) for K=9 |
| R2-481 | F4 | 42 | 2% | pruning saving, technique 4 — projection | hardware-dependent | disputes | pruning savings of 68%, 12%, 4%, 2% (combined 73%) for K=9 |
| R2-482 | F4 | 42 | 73% | combined projected K=9 runtime reduction (abstract, unqualified) | hardware-dependent | disputes | The abstract says "reduce projected K=9 runtime by 73%" without qualifying this as an estimate. |
| R2-483 | F4 | 42 | K=9 | level of the pruning projection | deterministic | quotes-paper | (combined 73%) for K=9 |
| R2-484 | F4 | 50 | 12,773 | InterPro entries, Run 2 (replacing Pfam) | deterministic | quotes-paper | Table 1 shows Run 2 has "---" for Pfam domains, replaced by 12,773 InterPro entries. |
| R2-485 | F4 | 52, 54, 56 | 769× | more transactions than next largest GPU system (paper line 456) | deterministic | disputes | "ET-miner processes 769x more transactions than the next largest GPU system." … The 769× appears to compare against CPU systems at 100K transactions. |
| R2-486 | F4 | 56 | 100K | transactions of the CPU systems the 769× apparently compares to | external-fact | quotes-paper | The 769× appears to compare against CPU systems at 100K transactions. |
| R2-487 | F4 | 56 | 100M | BIGMiner transactions (paper table) | external-fact | quotes-paper | BIGMiner processes 100M transactions — MORE than ET-miner's 76.9M. |
| R2-488 | F4 | 56 | 76.9M | ET-miner transactions | deterministic | quotes-paper | BIGMiner processes 100M transactions — MORE than ET-miner's 76.9M. |
| R2-489 | F4 | 56 | 15M | GMiner transactions (paper table) | external-fact | quotes-paper | GMiner is listed at 15M (76.9M/15M = 5.1×, not 769×). |
| R2-490 | F4 | 56 | 5.1× | 76.9M / 15M | deterministic | asserts-own | GMiner is listed at 15M (76.9M/15M = 5.1×, not 769×). |
| R2-491 | F4 | 66 | 5 | permutations in null model (paper line 410) | method-parameter | quotes-paper | The null model uses only 5 permutations (line 410). |
| R2-492 | F4 | 66 | 0.17 | lowest achievable bound on P(K≥7 given null) with 5 trials | deterministic | asserts-own | Cannot bound P(K≥7\|null) below 0.17 with only 5 trials. |
| R2-493 | F4 | 66 | +∞ | Z-scores for K≥7 as reported (paper) | deterministic | disputes | Z-scores for K≥7 reported as "+∞" — misleading with zero observations. |
| R2-494 | F4 | 66 | 0 out of 5 | trials producing K≥7 | deterministic | asserts-own | Can only say "0 out of 5 trials produced K≥7." |
| R2-495 | F4 | 70 | 2 | pLDDT bins frequent at min_count=8 (plddt_mean_med, plddt_mean_high) | deterministic | asserts-own | Only 2 are frequent at min_count=8: `plddt_mean_med` and `plddt_mean_high`. |
| R2-496 | F4 | 70 | 4 | defined bins with < 8 proteins | deterministic | asserts-own | The item mapping defines 6 bins but 4 have < 8 proteins. |
| R2-497 | F4 | 70 | < 8 | proteins in each of the 4 infrequent bins | deterministic | asserts-own | The item mapping defines 6 bins but 4 have < 8 proteins. |
| R2-498 | F4 | 74 | K=1 through K=8 | Run 2 levels reported from the pre-fix (NCCL int32 bug) run | deterministic | disputes | K=1 through K=8 numbers for Run 2 come from a run with the NCCL int32 bug present |
| R2-499 | F4 | 74 | f8c56cb5 | commit fixing the NCCL int32 bug | software | asserts-own | (fixed later in commit f8c56cb5) |
| R2-500 | F4 | 74 | < 2^31 | per-GPU counts (PROJECT_STATE's correctness argument, not re-verified) | software | disputes | PROJECT_STATE argues correctness because per-GPU counts < 2^31, but this was not independently verified by re-running. |
| R2-501 | F4 | 80 | 6 (0 hallucinated) | citations spot-checked: Jumper 2021, Agrawal 1994, Han 2000, Chon 2018, Abramson 2024, Naulaerts 2015 | external-fact | asserts-own | Spot-checked sample: Jumper 2021, Agrawal 1994, Han 2000, Chon 2018, Abramson 2024, Naulaerts 2015 — all verified correct. |
| R2-502 | F4 | 86 | 16.8 billion | itemsets (Run 2 / expanded) | deterministic | quotes-paper | 16.8 billion itemsets across 109M proteins is a real achievement. |
| R2-503 | F4 | 86 | 109M | proteins (Run 2 / expanded) | deterministic | quotes-paper | 16.8 billion itemsets across 109M proteins is a real achievement. |
| R2-504 | F4 | 86 | K=22 | the K=22 finding | deterministic | quotes-paper | The K=22 finding is compelling. |
| R2-505 | F4 | 89 | 500/500/2 | required Table 1 fix | deterministic | requests | 1. Fix Table 1: 500/500/2 |

---

## SECTION B — FLAGGED INCONSISTENCIES

### B.1 — Stated by the texts themselves

| ID | file | line | what is claimed inconsistent | the values/locations involved |
|---|---|---|---|---|
| B-01 | F1 | 44, 185–200, 295, 304 | "A fortiori" argument is logically backwards: a null model run at a stricter threshold cannot validate the more permissive main run | null min_count=769 (0.001%) vs main min_count=8 (0.00001%); null K_max=6; real K_max=14 @0.001% vs 22 @0.00001%; paper line 396 |
| B-02 | F1 | 173–183, 294, 310 | 5 permutations cannot bound P(K≥7 given null); Z=inf and p=0 are division-by-zero artifacts | 0 events / 5 trials → Clopper-Pearson 95% upper bound 0.451; would need 100 (0.030) or 1000 (0.003) permutations |
| B-03 | F1 | 198, 304, 308 | K=15–22 patterns are covered by no null test; should be labelled exploratory or tested at min_count=8 | K=14 (0.001%) vs K=22 (0.00001%) |
| B-04 | F1 | 127 | Two different speedup figures for the same comparison in the paper | "21x" (caption) vs "21.4x" (controlled comparison text) |
| B-05 | F1 | 164–171 | Paper null-table values are rounded relative to the JSON ("minor rounding in paper table") | 63,702 vs 63,702.4; 25,468 vs 25,467.8; 22 vs 21.8; −987 vs −987.08; +3,791 vs 3790.74; +71,728 vs 71728.1 |
| B-06 | F1 | 228–237, 254, 312 | GO:0005524 is auto-derived from PF00270 via InterPro2GO but the paper does not note it; GO evidence codes absent from item mapping (method detail missing) | independent features 22 → 21; 1 of 19 GO terms |
| B-07 | F1 | 262–279, 302 | Table 1 feature breakdown contradicts the item-mapping parquet ("fabricated or refers to a different version of the data") | paper 247 Pfam / 302 MF / 289 BP / 161 CC / 752 GO / 3 pLDDT vs parquet 500 / – / – / – / 500 / 2; total 1,002 in both |
| B-08 | F1 | 276–277 | pLDDT bin count: paper says 3, only 2 are frequent; mapping has 1,006 entries but 1,002 frequent | 3 vs 2; 1,006 vs 1,002 (4 infrequent: plddt_mean_low + 3 plddt_fraction; plddt_mean_low < 8 proteins) |
| B-09 | F1 | 67–75 | "Dedup affected 20.45%" (PROJECT_STATE.md) is not in the paper and describes null-shuffle duplicates, not the source-data split | 20.45% vs 62.61% single-feature / 37.39% multi-feature |
| B-10 | F1 | 26–42 | Two different min_count values for the same nominal 0.001% support in the two experiment JSONs (stated, not flagged) | 768 (direct-vs-SON) vs 769 (null model) |
| B-11 | F2 | 21–23, 39, 60–64 | Replicates of the same 35K run disagree in itemset count and time, attributed to "GPU floating-point non-determinism in row-split reduction" | 606,292 ± 28 itemsets; 654.3 ± 6.7 s; deterministic only at K≤3 and K≥11 |
| B-12 | F2 | 309–366, 643 | Maximum depth decreased between vocabularies ("counterintuitive"); three mechanisms proposed | K=22 (1,002 features, min_count 8) → K=17 (35,012 features, min_count 1,093); 137× threshold ratio |
| B-13 | F2 | 362 | K-distribution peak shifted between vocabularies | K=9 (1K) → K=7 (35K) |
| B-14 | F2 | 413–416, 646 | Null-model ceiling changed between vocabularies | K=6 (22 itemsets mean, 1K) → K=5 (1 itemset per permutation, 35K); zero at K≥6 |
| B-15 | F2 | 420–426 | K=3 flips from depleted to enriched between analyses | Z=−143 (1K) → Z=+723 (35K); 3.3× more triples than chance |
| B-16 | F2 | 429–430 | Comparison sentence omits the 1K ratio it compares against (number missing) | "5.5× … (606K vs. 111K), compared to the 1K ratio at the same support threshold" — 1K ratio never given |
| B-17 | F2 | 547–576 | Speedup wording to be changed in three paper locations (value changed between versions) | 21× → 21.4× in Sec 3.1, Table 2 footnote (adds 95.2% miss rate) and Discussion 4.1 |
| B-18 | F2 | 565–566 | Table 2 † footnote compares runs differing in both method and threshold | Power (0.001%) vs Blitz (0.0001%) |
| B-19 | F2 | 127–178 | Vocabulary/dataset values changed between paper versions | 1,002 → 35,012 features; 247 → 12,773 domains; 752 → 5,763 GO; 3 → 6 pLDDT bins; 76.9M → 109.2M proteins |
| B-20 | F2 | 370–374, 434–457 | Null model to be replaced: 5-permutation (1K) by 2-permutation (35K), 100-permutation still TBD | 5 → 2 → 100 permutations |
| B-21 | F2 | 37–42, 47–49, 653–658 | Five of six campaign rows are TBD; Base run stopped at timeout | Base: ≥10 K, 10,041,611 @K=9, 12,491,079 @K=10 started; Super/Blitz/Ultra/Opus all TBD |
| B-22 | F2 | 383 | Two different totals for the same 2-permutation null run | 1,310 s GPU time vs 1,456 s wall-clock |
| B-23 | F2 | 210–235 | SON declared physically infeasible at 35K: chunk bitvector exceeds VRAM; empirical run hung | 175 GB (40M chunk) > 143 GB; 131 GB (30M chunk) → 4 passes; hung > 22 h at Pass 1 |
| B-24 | F3 | 39 | coin2009: wrong first author (Terrapon), incomplete pages | 3077 → 3077–3083 |
| B-25 | F3 | 40 | varadi2022 page range wrong; collides with mistry2021 (copy error) | D419–D427 → D439–D444; mistry2021 = D412–D419 |
| B-26 | F3 | 41 | BIGMiner scale misreported; "30 nodes" unverifiable; undermines "three orders of magnitude" framing | 100M (paper) vs 6.5 billion (BIGMiner paper), ≈65× |
| B-27 | F3 | 47 | meysman2015 structure count overstated ~3× | ~100K (paper) vs ~32,142 |
| B-28 | F3 | 48 | acmsurvey2021 issue/pages wrong | 54(7):1–36 → 54(9), Article 179 (179:1–179:35) |
| B-29 | F3 | 49 | fang2009 bib title is the 2008 tech-report title, not the DaMoN 2009 paper | pp 34–42 |
| B-30 | F3 | 50 | chon2024 title paraphrased wrongly | ESWA vol 250, Art. 123928 |
| B-31 | F3 | 51 | varadi2024 end page missing | D368 → D368–D375 |
| B-32 | F3 | 52 | chon2018b end page off by one; issue missing | 21:1507–1521 → 21(3):1507–1520 |
| B-33 | F3 | 53 | savasere1995 pages missing | → pp. 432–444 |
| B-34 | F3 | 54 | GPU-speedup citation cluster bundles two CPU algorithms (han2000, zaki2000) | 50–350× (350× djenouri2019, 100× zhang2011 confirmed) |
| B-35 | F3 | 55 | GMiner transaction count differs between two paper tables; hardware unverifiable | 15M (r466, basis of 5.1×) vs 1.7M (r877; webdocs FIMI = 1,692,082); "4× GTX 1080" |
| B-36 | F3 | 57–63 | Five bibitems never cited | webb2007, webb2014, abramson2024, zaki1997, miettinen2020 (0×) |
| B-37 | F3 | 76 | CSR given three sizes under three unlabelled byte conventions; nnz of the 214M set never stated | 316M nnz: ~5.1 GB (16 B/entry, r175) / ~3 GB (r179, r485; 316M × 8 B = 2.53 GB) / 19 GB for 214M (8 B/entry, r920) |
| B-38 | F3 | 77 | "40×" and "1.4×" memory reductions silently compare different representations | 40× = 206 GB (1 B/boolean) ÷ 5.1 GB; 1.4× = 27 GB bitpacked ÷ 19 GB; real on-GPU bitvector 26 GB (~8× below 206) |
| B-39 | F3 | 78 | Expanded bitvector size inconsistent through a GB/GiB mix | ~445 GB (r258) vs 56 GB × 8 = 448 vs 478 GB decimal; 445.19 GiB; 56 GB/device → 55.6 GiB |
| B-40 | F3 | 84, 142, 153 | "p < 0.17 by one-sided binomial test" not supported for 0/5; origin unclear | 0.17 (≈1/6) vs ≈0.45 (rule of three 3/5 = 0.6) |
| B-41 | F3 | 90 | Expanded min_count uses round() while pseudocode (r800, ⌈σ·n⌉) and the Power run (769) use ceil | 1,092 vs ⌈1092.24⌉ = 1093 (n = 109,224,173) |
| B-42 | F3 | 91 | Cost-saving arithmetic off by one dollar | $243 vs 332 − 90 = $242 |
| B-43 | F3 | 92 | Cost formula does not reproduce the stated cost | $332 vs 3.50 × 8 × 12 = $336 (implies 11.86 h, not ~12 h) |
| B-44 | F3 | 93 | 62.6% (Limitation 4) is the base-run figure, unlabelled; tension with expanded coverage claim | 37.4% + 62.6% = 100% vs 53.1% (expanded) |
| B-45 | F3 | 94 | "26.8M" is a truncation, not a rounding | 26,849,505 → 26.85M / 26.9M |
| B-46 | F3 | 115, 150 | Abstract sentence grammatically broken while conclusion (r549) states the same claim correctly | "we estimate would reduce K=9 runtime" vs "are estimated to reduce K=9 runtime by approximately 73%" |
| B-47 | F3 | 123 | Unit spacing for written-out times inconsistent with house style | 7.3 minutes / 63 minutes / 4.3 hours vs `80\,GB` |
| B-48 | F3 | 126 | Same pruning technique named differently in abstract and appendix | "warp-cooperative" (r92) vs "Warp-level" (r951) |
| B-49 | F3 | 127 | "~12 bytes" typeset two ways | `$\sim$12 bytes` (r873) vs `${\sim}12$~bytes` (r842) |
| B-50 | F3 | 136–141 | Earlier-flagged issues confirmed fixed in the current version (values changed between versions) | Table 1 247/752/3 → 500/500/2 (r143–150); "a fortiori" removed (r437); pLDDT 3 → 2 (r150); "GPU-resident" → "GPU-accelerated" |
| B-51 | F3 | 142 | 5-permutation weakness only partly resolved | p<0.17 claim still unsupported (see B-40) |
| B-52 | F3 | 143 | 35K @0.001% result differs ~27,000× between revision_notes_b3 and the current paper; attributed to the NCCL int32-overflow fix; needs confirmation that 16.8B is canonical | 606,292 itemsets / K_max=17 / 654 s (B3) vs 16.8 billion through K=8 / 4.3 h (paper; cumulative 16,812,646,639 internally consistent) |
| B-53 | F4 | 11–22, 86, 89 | Table 1 Run-1 breakdown "fabricated"; flagged 2026-02-20 and never corrected | 247/752/3 vs 500/500/2 (top_pfam=500, top_go=500); total 1,002 |
| B-54 | F4 | 24–29, 91 | "A fortiori" argument logically inverted | min_count 769 (null) vs 8 (main); K=15–22 significance extrapolated |
| B-55 | F4 | 31–38, 90 | "GPU-resident" / "all operations entirely on-GPU" contradicts Council of Copii finding: candidate generation runs on CPU (`gpu_dispatch.py:128-131`; `build_prefix_groups_gpu` not connected) | paper lines 162, 189; should read "GPU-accelerated" |
| B-56 | F4 | 40–42, 92 | Table 8 pruning savings are projections, never measured; abstract does not qualify them | 68% / 12% / 4% / 2% (73% combined, K=9) |
| B-57 | F4 | 48–50, 93 | Run 1 vs Run 2 feature semantics differ (Pfam vs InterPro) yet K-distributions compared side-by-side (Figure 3) | 247 Pfam (Run 1) vs 12,773 InterPro (Run 2, "---" for Pfam) |
| B-58 | F4 | 52–56 | 769× scale claim arithmetic unclear; BIGMiner is larger than ET-miner | 769× (only vs 100K CPU systems); BIGMiner 100M > 76.9M; GMiner 15M → 5.1× |
| B-59 | F4 | 58–62 | "closed itemset pruning" is post-processing (`apriori.py:235-268`, `prune_equal_support`), not in-mining pruning | paper line 528 |
| B-60 | F4 | 64–66, 94 | 5 permutations statistically weak; "+∞" Z-scores misleading with zero observations | cannot bound P(K≥7 given null) below 0.17; 0 of 5 trials |
| B-61 | F4 | 68–70 | pLDDT bin count discrepancy | 3 claimed vs 2 frequent at min_count=8 (6 defined, 4 with < 8 proteins) |
| B-62 | F4 | 72–74 | Run 2 K=1–8 numbers come from a run with the NCCL int32 bug (fixed in f8c56cb5), never re-verified by re-running | "per-GPU counts < 2^31" argument only |

### B.2 — Extractor-observed cross-file / intra-file disagreements (NOT asserted by any of the four texts; listed for the audit, not judged)

| ID | files / lines | observation | values |
|---|---|---|---|
| BX-01 | F1 179 vs F4 66 vs F3 84 | Three different statements of the bound on P(K≥7 given null) for 0/5 permutations | 0.451 (F1, Clopper-Pearson 95%) vs 0.17 (F4, "cannot bound below") vs "0.17 unsupported, ≈0.45" (F3) |
| BX-02 | F2 109–115, 134, 139, 175, 326 vs F1 262–279, F3 138, F4 11–22 | F2 (dated 2026-02-21) still uses the 1K breakdown that F1 (2026-02-20) had disputed and F3/F4 confirm as wrong | 247 / 752 / 3 (F2) vs 500 / 500 / 2 |
| BX-03 | F2 39, 320, 336 vs F3 90 | min_count for 35K @0.001% | 1,093 (F2) vs 1,092 in the current paper (F3 N5) |
| BX-04 | F2 22, 187, 220, 641 vs F2 625 | H200 VRAM stated two ways in the same document | 143 GB vs 141 GB ("H200 SXM5 141 GB") |
| BX-05 | F2 22, 186–187, 239, 627–629 vs F3 78, 92 | Expanded-run GPU count and bitvector size differ between B3 notes and the paper version F3 reviewed | 4 × H200, 477 GB, ~119 GB/device (F2) vs "~56 GB/device × 8" ≈ 445 GB and "$3.50 × 8 GPUs × ~12 h" (paper per F3) |
| BX-06 | F2 401–405 vs F2 73–81 | Bio column of the 2-permutation null table does not equal the kdist table in the same document | 46,847 / 53,235 / 59,095 / 62,241 / K6–17 = 348,853 (sum 605,191) vs 46,853 / 53,238 / 59,086 / 62,226 / K6–17 = 349,969 (sum 606,292) |
| BX-07 | F2 37–42 | Blitz "min proteins" does not follow ceil like the other rows | 109 vs ⌈0.0001% × 109,224,173⌉ = 110 (109,225; 10,923; 1,093; 22; 11 all match ceil) |
| BX-08 | F2 187 | Per-device shard plus stated headroom does not sum to either stated VRAM figure | ~119 GB + 12 GB = 131 GB vs 143 GB / 141 GB |
| BX-09 | F1 57 vs F2 177–178 | The 76.9M protein set is described as "multi-feature (>1 item)" in F1 but the 76.9M → 109.2M comparison in F2 is phrased as proteins "with at least one annotation feature" | 76,890,945 (>1 item) vs 109.2M (≥1 feature) |
| BX-10 | F1 31 vs F2 555 | min_count of the controlled Direct-vs-SON comparison | 768 (experiment JSON, F1) vs 769 (B3 replacement text) |
| BX-11 | F2 547–576 vs F3 19, 100 | F2 asks to replace 21× with 21.4× in three places; F3 (June) still verifies "21×" in the paper | 21× vs 21.4× |
| BX-12 | F2 39 vs F3 143 / F4 74, 86 | 35K @0.001% result (F3 states it, B-52); F4 independently attributes Run 2 K=1–8 to the pre-fix run | 606,292 / K_max 17 / 654 s vs 16.8B through K=8 / 4.3 h |
| BX-13 | F1 132 vs F2 22, 624 | Hardware differs between the 1K experiments and the 35K experiments (not a conflict, but relevant for timing reproduction) | H100 (1K direct-vs-SON, null) vs 4 × H200 (35K) |

---

## SECTION C — PER-FILE SUMMARIES

### F1 — `review_b2_results.md` (325 lines)

- What: "Agent B2: Critical Experimental Result Verification Report" — an internal audit of every quantitative claim in `papers/et_miner_proteome.tex` against the raw artefacts (godmode itemsets parquet, item mapping, direct-vs-SON JSON, null-model JSON, godmode log, `transactions_214m.parquet`).
- Version addressed: the pre-March paper (no version label given); the results are the 1,002-feature ("214m"/godmode) campaign with min_count=8, plus the 0.001% direct-vs-SON and 5-permutation null experiments (JSONs dated 2026-02-19).
- Date: 2026-02-20.
- Verdict: 9 checks; 7 VERIFIED (protein counts, K=22 itemset and its 22-feature decode, 26,849,505 itemsets, 21.4× speedup, null K-distributions, all timings), 2 DISCREPANCY (Table 1 breakdown 247/752/3 vs actual 500/500/2; "a fortiori" argument logically inverted), 3 RISK FLAGs (5 permutations insufficient — Clopper-Pearson bound 0.451; GO:0005524 auto-derived via InterPro2GO → 21 independent features; null threshold 96× stricter than main run).
- Must-fix before submission: Table 1 breakdown; rewrite of the "a fortiori" sentence. Recommended: null model at min_count=8, 100+ permutations, note the GO:0005524 derivation.
- Notable: it establishes the canonical 1K numbers reused everywhere else (76,890,945 / 205,620,298 / 26,849,505 / 475,865 vs 22,846 / 440.5 s / 662.17 s), and records that the two "0.001%" experiments used different min_counts (768 vs 769).

### F2 — `revision_notes_b3.tex` (659 lines)

- What: LaTeX revision blocks ("Agent B3") to be pasted into the paper to add the expanded 35,012-feature / 109.2M-protein results run on 4 × H200; each block carries placement instructions. Values tagged [ACTUAL] come from `results_35k/experiment_direct_vs_son_35k_20260221.json`, `results_35k/experiment_null_model_20260221_003203.json`, `results_35k/wave3_base_partial.log`; [TBD] awaits runs.
- Version addressed: the same pre-March paper as F1 (it proposes a new Table `tab:campaign-35k`, replaces Table 1 with `tab:features-35k`, replaces the 5-permutation null section, and rewrites the 21× speedup text in three places to 21.4×).
- Date: "Updated: 2026-02-21".
- Content: 35K campaign table (only Power 0.001% complete: 606,292 ± 28 itemsets, K_max 17, 654.3 ± 6.7 s over 3 replicates; Base 0.1% partial to K=9 = 10,041,611); full 17-level K-distribution; new vocabulary table (12,773 InterPro / 5,763 GO / 1,107 EC / 15,162 Keywords / 175 taxa / 26 length bins / 6 pLDDT bins = 35,012); SON infeasibility argument (175 GB > 143 GB; run hung > 22 h); bitvector-vs-sparse complexity argument; K_max 22 → 17 explanation; 2-permutation null model (null collapses at K=5; Z=+723 at K=3, +20,585 at K=4); cross-source discovery framework; Wave-3 Base partial log; hardware/software description (Python 3.12, CuPy 13.4, NumPy 2.0, CUDA 12.6, Ubuntu 22.04, 1.5 TB RAM).
- TBD (never filled here): Base/Super/Blitz/Ultra/Opus rows; 100-permutation null table; cross-source pattern examples; K=17 itemset characterisation.
- Notable: still carries the 247/752/3 breakdown for the 1K set that F1 disputed one day earlier; states H200 VRAM as both 143 GB and 141 GB; its 606,292-itemset result is later (F3) reported as superseded by 16.8 billion after an NCCL int32-overflow fix.

### F3 — `senior_review_jun01.md` (160 lines, Dutch)

- What: "Senior-Reviewer Pass" over `et_miner_proteome.tex` (968 lines) along three axes — mis-cited references (web-verified), internal numerical consistency, spelling/grammar — produced by three parallel subagents plus manual verification. Report only; no edits.
- Version addressed: the current (post-March) paper, which already contains the expanded run (16,812,646,639 cumulative itemsets), the fixed Table 1 (500/500/2), the removed "a fortiori" argument, and "GPU-accelerated" wording. It cross-checks against the four earlier review documents (Feb–May 2026).
- Date: 2026-06-01.
- Verdict: the arithmetic core is "opvallend gezond" — both large table sums check exactly (26,849,505; 16,812,646,639), all min_counts, growth factors, Z-scores and ratios (9×, 124×, 21×, 95.2%, 5.1×) hold within rounding. Real problems are elsewhere: 4 MAJOR citation errors (wrong first authors barrio2024→Lau, coin2009→Terrapon; varadi2022 pages; BIGMiner scale 100M vs 6.5B), 9 MINOR bibliographic fixes, 5 orphan references; memory figures using three unlabelled byte conventions (CSR 3 / 5.1 / 19 GB; 206 vs 26 GB dense; 445 GB GB/GiB mix); the "p < 0.17" statistic unsupported (should be ≈0.45); small arithmetic slips ($243 vs $242; $332 vs $336; 1,092 vs 1093; 26.8M truncation); one broken abstract sentence.
- Flags as NEW / TO CONFIRM: the ~27,000× gap between B3's 606,292 itemsets (K_max 17, 654 s) and the current paper's 16.8 billion through K=8 (4.3 h), attributed to the NCCL int32-overflow bugfix.
- Fix priority: R1–R3 citations, G1 abstract, R4 BIGMiner, N1–N2 memory labels, N4 p-value, then the rest.

### F4 — `senior_review_mar23.md` (94 lines)

- What: "Senior Review" by an Opus 4.6 reviewer agent of "Protein Structural Motif Discovery at AlphaFold Scale". Verdict: Major Revisions Required.
- Version addressed: the paper version that already contains Run 2 (16.8 billion itemsets across 109M proteins, 12,773 InterPro entries, 5,763 GO, 6 pLDDT bins) but still has the original Run-1 Table 1 (247/752/3) and the "a fortiori" sentence (line 438).
- Date: 2026-03-23.
- Critical (C1–C4): Table 1 Run-1 breakdown fabricated (actual 500/500/2 from `top_pfam=500, top_go=500`; flagged by B2 on 2026-02-20 and never fixed); "a fortiori" inverted (769 vs 8); "GPU-resident" contradicts the Council of Copii finding that candidate generation is CPU-side; Table 8 pruning savings (68/12/4/2%, 73% combined at K=9) are projections presented as results.
- Important (I1–I6): Run 1 vs Run 2 feature semantics not comparable; 769× scale claim arithmetic unclear (BIGMiner 100M > 76.9M; GMiner 15M → 5.1×); closed-itemset "pruning" is post-processing; 5 permutations weak (cannot bound P(K≥7 given null) below 0.17; "+∞" Z misleading); pLDDT 3 vs 2 bins (6 defined, 4 with < 8 proteins); Run 2 K=1–8 numbers from the NCCL int32-bug run (fixed in f8c56cb5), not re-verified.
- Citations spot-checked (6) all correct.
- Required actions: fix Table 1 (500/500/2), "GPU-resident" → "GPU-accelerated", fix/remove "a fortiori", qualify pruning as projection, discuss Run 1/Run 2 semantics, more permutations or qualified p-values.

---

## SECTION D — TOTALS

Section A rows: **505** (R2-001 … R2-505). Section B: 62 text-stated rows (B-01…B-62) + 13 extractor-observed rows (BX-01…BX-13).

### D.1 — Rows per file

| file | rows |
|---|---|
| F1 | 137 |
| F2 | 208 |
| F3 | 119 |
| F4 | 41 |
| **total** | **505** |

### D.2 — Rows per category

| category | rows |
|---|---|
| deterministic | 330 |
| hardware-dependent | 74 |
| method-parameter | 57 |
| external-fact | 35 |
| software | 9 |
| **total** | **505** |

### D.3 — Rows per stance

| stance | rows |
|---|---|
| quotes-paper | 127 |
| asserts-own | 323 |
| disputes | 40 |
| requests | 15 |
| **total** | **505** |

### D.4 — Category × stance

| category | quotes-paper | asserts-own | disputes | requests | total |
|---|---|---|---|---|---|
| deterministic | 82 | 224 | 21 | 3 | 330 |
| hardware-dependent | 25 | 39 | 6 | 4 | 74 |
| method-parameter | 11 | 39 | 1 | 6 | 57 |
| external-fact | 8 | 14 | 11 | 2 | 35 |
| software | 1 | 7 | 1 | 0 | 9 |
| **total** | **127** | **323** | **40** | **15** | **505** |

### D.5 — File × stance

| file | quotes-paper | asserts-own | disputes | requests | total |
|---|---|---|---|---|---|
| F1 | 33 | 91 | 9 | 4 | 137 |
| F2 | 18 | 186 | 0 | 4 | 208 |
| F3 | 62 | 34 | 18 | 5 | 119 |
| F4 | 14 | 12 | 13 | 2 | 41 |

### D.6 — File × category

| file | deterministic | hardware-dependent | method-parameter | external-fact | software | total |
|---|---|---|---|---|---|---|
| F1 | 104 | 18 | 13 | 2 | 0 | 137 |
| F2 | 135 | 30 | 35 | 3 | 5 | 208 |
| F3 | 65 | 21 | 5 | 26 | 2 | 119 |
| F4 | 26 | 5 | 4 | 4 | 2 | 41 |

