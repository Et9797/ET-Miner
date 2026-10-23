# Phase 1 — Claims extracted from review files (part 1: three review files)

Run: `runs/20260902T0000Z` · Extraction date: 2026-09-02 · Scope: exhaustive, faithful extraction of every numeric value / quantitative claim in three review documents. **No correctness judgement is made here; every number below may be hallucinated in its source file.**

Source files (read in full, every line):

| key | file | lines |
|---|---|---|
| F1 | `/root/projects/ET-Miner/paper/PAPER_V2_REVIEW.md` | 160 |
| F2 | `/root/projects/ET-Miner/paper/peer_review_jul12.md` | 179 |
| F3 | `/root/projects/ET-Miner/paper/review_b1_hostile.md` | 350 |

Conventions:

- One row per **occurrence** of a numeric value / quantitative claim, in file order (F1, F2, F3) then line order. Repeated values on different lines get separate rows so that later phases can map line → value.
- `category`: deterministic = a function of data + parameters (counts, K, percentages, Z/t statistics, sizes derived arithmetically, protein/item counts); hardware-dependent = timings, throughput, speedups, GPU model/count/VRAM, memory-fit feasibility, compute cost; method-parameter = thresholds, support %, min_count, permutation counts, seeds, vocabulary caps, bin bounds, inclusion criteria, confidence levels; external-fact = facts external to the experiment (UniProt release, TrEMBL size, competitor numbers, literature, GO prevalence) **and** review-meta / bibliographic / repository-inventory counts (e.g. '73 claims adjudicated', severity scores, citation counts); software = version numbers.
- `stance`: quotes-paper = the review repeats a value it attributes to the paper; asserts-own = the reviewer states a value of its own (recomputation, value read from a log/JSON artifact, hypothetical, estimate, review-meta count, severity score); disputes = the reviewer says a paper value (or artifact-derived value) is wrong / implausible / invalid; requests = the reviewer asks for a number to be added, changed, listed, or a run to be performed with that parameter.
- Values read by a reviewer from logs/JSON artifacts (not from the paper) are labelled asserts-own; the quoted context names the artifact where the review does.
- **Excluded on purpose** (locators, not claims): paper `.tex` line numbers (e.g. 'line 129'), log/JSON/script line references (e.g. `pipeline_214m.log:2063`), section/table/figure/appendix numbers, review item numbering (Major 1, D1, M3.3, Stage 1–7), timestamps embedded in file names, commit hashes, citation-key years (`miettinen2020`), and the bare shorthand labels '1K'/'35K' when they merely name a vocabulary (the underlying 1,002 / 34,920 are extracted wherever stated; '35K' is kept where it enters arithmetic). Review dates are reported in Section C rather than as claim rows.
- Symbols: `x` = ×, `>=` = ≥, `<=` = ≤, `->` = →, `--` = en dash; commas in numbers are preserved as written.

---

## SECTION A — CLAIMS TABLE

Total rows: **835** (F1: 327, F2: 163, F3: 345).

| ID | file | line | value | unit | category | stance | quoted context (≤ 25 words, verbatim) |
|---|---|---|---|---|---|---|---|
| R1-001 | F1 | 7 | 9 | verification buckets | external-fact | asserts-own | "Across 9 verification buckets, 73 claims were adjudicated" |
| R1-002 | F1 | 7 | 73 | claims adjudicated | external-fact | asserts-own | "Across 9 verification buckets, 73 claims were adjudicated" |
| R1-003 | F1 | 11 | 62 | claims CONFIRMED | external-fact | asserts-own | "CONFIRMED / 62" (verdict table) |
| R1-004 | F1 | 12 | 4 | claims DISCREPANT | external-fact | asserts-own | "DISCREPANT / 4" (verdict table) |
| R1-005 | F1 | 13 | 7 | claims UNVERIFIABLE | external-fact | asserts-own | "UNVERIFIABLE / 7" (verdict table) |
| R1-006 | F1 | 15 | 6/6 | campaign-table rows verified | external-fact | asserts-own | "the campaign table (6/6)" |
| R1-007 | F1 | 15 | 22 | K-distribution rows | deterministic | quotes-paper | "the full 22-row K-distribution (8/8, line-for-line)" |
| R1-008 | F1 | 15 | 8/8 | K-distribution checks passed | external-fact | asserts-own | "the full 22-row K-distribution (8/8, line-for-line)" |
| R1-009 | F1 | 15 | 6/6 | Direct-vs-SON checks passed | external-fact | asserts-own | "the Direct-vs-SON comparison (6/6)" |
| R1-010 | F1 | 15 | 247 | Pfam items (v1 Table 1) | deterministic | disputes | "The old fabricated v1 Table 1 (247/752/3) has been corrected" |
| R1-011 | F1 | 15 | 752 | GO items (v1 Table 1) | deterministic | disputes | "The old fabricated v1 Table 1 (247/752/3) has been corrected" |
| R1-012 | F1 | 15 | 3 | pLDDT items (v1 Table 1) | deterministic | disputes | "The old fabricated v1 Table 1 (247/752/3) has been corrected" |
| R1-013 | F1 | 15 | 500 | Pfam items (verified) | method-parameter | asserts-own | "corrected to the verified 500/500/6 = 1,006 ground truth" |
| R1-014 | F1 | 15 | 500 | GO items (verified) | method-parameter | asserts-own | "corrected to the verified 500/500/6 = 1,006 ground truth" |
| R1-015 | F1 | 15 | 6 | pLDDT bins (verified) | method-parameter | asserts-own | "corrected to the verified 500/500/6 = 1,006 ground truth" |
| R1-016 | F1 | 15 | 1,006 | total items | deterministic | asserts-own | "corrected to the verified 500/500/6 = 1,006 ground truth" |
| R1-017 | F1 | 19 | >=8 | proteins (feature-retention cutoff, paper wording) | method-parameter | disputes | "'retained all features occurring in at least 8 proteins' does not describe how the vocabulary was built" |
| R1-018 | F1 | 19 | 500 | top-N Pfam frequency cap | method-parameter | asserts-own | "the vocabulary is a top-500-Pfam + top-500-GO frequency cap" |
| R1-019 | F1 | 19 | 500 | top-N GO frequency cap | method-parameter | asserts-own | "the vocabulary is a top-500-Pfam + top-500-GO frequency cap" |
| R1-020 | F1 | 19 | 24,291 | unique Pfam terms | deterministic | asserts-own | "out of 24,291 / 25,993 unique terms" |
| R1-021 | F1 | 19 | 25,993 | unique GO terms | deterministic | asserts-own | "out of 24,291 / 25,993 unique terms" |
| R1-022 | F1 | 19 | >=8 | mining min-support (proteins) | method-parameter | asserts-own | "the '>=8' is the mining min-support, a different quantity" |
| R1-023 | F1 | 20 | 40x | memory reduction (dense to CSR) | deterministic | disputes | "Inflated 40x memory-reduction claim (lines 167, 402)" |
| R1-024 | F1 | 20 | 5.1 | GB (CSR) | deterministic | quotes-paper | "the 5.1 GB CSR is the 76.9M mined subset" |
| R1-025 | F1 | 20 | 76.9M | proteins (mined subset) | deterministic | asserts-own | "the 5.1 GB CSR is the 76.9M mined subset" |
| R1-026 | F1 | 20 | 206 | GB (dense) | deterministic | quotes-paper | "the 206 GB dense is the 205.6M full set" |
| R1-027 | F1 | 20 | 205.6M | proteins (full set) | deterministic | asserts-own | "the 206 GB dense is the 205.6M full set" |
| R1-028 | F1 | 20 | ~15x | memory reduction (same dataset) | deterministic | asserts-own | "Within one dataset the reduction is ~15x, not 40x" |
| R1-029 | F1 | 21 | 8 | proteins (K=22 accessions) | deterministic | quotes-paper | "the paper says the 8 K=22 accessions 'can be recovered by querying the source transaction data'" |
| R1-030 | F1 | 21 | 22 | K (deepest itemset) | deterministic | quotes-paper | "the paper says the 8 K=22 accessions 'can be recovered'" |
| R1-031 | F1 | 22 | <0.45 | p (upper bound) | deterministic | quotes-paper | "the p<0.45 bound is the exact one-sided binomial 95% bound" |
| R1-032 | F1 | 22 | 95% | one-sided confidence level | method-parameter | asserts-own | "the exact one-sided binomial 95% bound (1-0.05^(1/5)=0.451)" |
| R1-033 | F1 | 22 | 1-0.05^(1/5)=0.451 | exact binomial bound | deterministic | asserts-own | "the exact one-sided binomial 95% bound (1-0.05^(1/5)=0.451)" |
| R1-034 | F1 | 22 | 3/5=0.60 | rule-of-three bound | deterministic | asserts-own | "'the rule of three,' which actually gives 3/5=0.60" |
| R1-035 | F1 | 24 | 4 | min_count (verification run) | method-parameter | disputes | "unbacked min_count=4 claim" |
| R1-036 | F1 | 30 | >=8 | support cutoff (heading) | method-parameter | disputes | "D1 - Vocabulary construction misdescribed as an >=8-support cutoff" |
| R1-037 | F1 | 31 | >=8 | proteins (feature retention) | method-parameter | quotes-paper | "We retained all features occurring in at least 8 proteins (the lowest support threshold used)." |
| R1-038 | F1 | 32 | 24291 | unique Pfam | deterministic | asserts-own | "24291 unique Pfam, 25993 unique GO from 205620298 proteins" |
| R1-039 | F1 | 32 | 25993 | unique GO | deterministic | asserts-own | "24291 unique Pfam, 25993 unique GO from 205620298 proteins" |
| R1-040 | F1 | 32 | 205620298 | proteins | deterministic | asserts-own | "24291 unique Pfam, 25993 unique GO from 205620298 proteins" |
| R1-041 | F1 | 32 | 6 | pLDDT items | method-parameter | asserts-own | "Item encoding: 6 pLDDT + 500 Pfam + 500 GO = 1006 total items" |
| R1-042 | F1 | 32 | 500 | Pfam items | method-parameter | asserts-own | "Item encoding: 6 pLDDT + 500 Pfam + 500 GO = 1006 total items" |
| R1-043 | F1 | 32 | 500 | GO items | method-parameter | asserts-own | "Item encoding: 6 pLDDT + 500 Pfam + 500 GO = 1006 total items" |
| R1-044 | F1 | 32 | 1006 | total items | deterministic | asserts-own | "Item encoding: 6 pLDDT + 500 Pfam + 500 GO = 1006 total items" |
| R1-045 | F1 | 32 | top-500 | per-type frequency cap | method-parameter | asserts-own | "The vocabulary is a top-500-per-type frequency cap, not a >=8-support retention" |
| R1-046 | F1 | 32 | >=8 | mining min_count | method-parameter | asserts-own | "The '>=8' is the mining min_count (godmode_mining.log:2), a separate quantity" |
| R1-047 | F1 | 33 | 205.6M | proteins | deterministic | asserts-own | "At 205.6M proteins, vastly more than 500 Pfam/GO families occur in >=8 proteins" |
| R1-048 | F1 | 33 | >500 | Pfam/GO families occurring in >=8 proteins | deterministic | asserts-own | "vastly more than 500 Pfam/GO families occur in >=8 proteins" |
| R1-049 | F1 | 33 | >=8 | proteins | method-parameter | asserts-own | "vastly more than 500 Pfam/GO families occur in >=8 proteins" |
| R1-050 | F1 | 36 | 40x | reduction (heading) | deterministic | disputes | "D2 - '40x reduction' conflates two different datasets" |
| R1-051 | F1 | 37 | 316 million | non-zero entries | deterministic | quotes-paper | "316 million non-zero entries occupying ~5.1 GB ... a 40x reduction from the ~206 GB naive dense" |
| R1-052 | F1 | 37 | ~5.1 | GB (coordinate format) | deterministic | quotes-paper | "316 million non-zero entries occupying ~5.1 GB" |
| R1-053 | F1 | 37 | 40x | reduction | deterministic | disputes | "a 40x reduction from the ~206 GB naive dense representation" |
| R1-054 | F1 | 37 | ~206 | GB (naive dense) | deterministic | quotes-paper | "a 40x reduction from the ~206 GB naive dense representation" |
| R1-055 | F1 | 38 | 316,421,093 | non-zeros | deterministic | asserts-own | "ties the 316,421,093 non-zeros / 5.1 GB CSR to the 76,890,945-transaction subset" |
| R1-056 | F1 | 38 | 5.1 | GB (CSR) | deterministic | asserts-own | "ties the 316,421,093 non-zeros / 5.1 GB CSR to the 76,890,945-transaction subset" |
| R1-057 | F1 | 38 | 76,890,945 | transactions (subset) | deterministic | asserts-own | "ties the 316,421,093 non-zeros / 5.1 GB CSR to the 76,890,945-transaction subset" |
| R1-058 | F1 | 38 | 206 | GB dense (full set) | deterministic | asserts-own | "the 206 GB dense figure is for the 205.6M full set" |
| R1-059 | F1 | 38 | 205.6M | proteins (full set) | deterministic | asserts-own | "the 206 GB dense figure is for the 205.6M full set" |
| R1-060 | F1 | 38 | 205,620,298 | transactions | deterministic | asserts-own | "205,620,298 x 1002 x 1 B = 206 GB" |
| R1-061 | F1 | 38 | 1002 | items (dense columns) | deterministic | asserts-own | "205,620,298 x 1002 x 1 B = 206 GB" |
| R1-062 | F1 | 38 | 1 | byte per dense entry | method-parameter | asserts-own | "205,620,298 x 1002 x 1 B = 206 GB" |
| R1-063 | F1 | 38 | 206 | GB (computed) | deterministic | asserts-own | "205,620,298 x 1002 x 1 B = 206 GB" |
| R1-064 | F1 | 38 | 76.9M | transactions (subset) | deterministic | asserts-own | "Within the 76.9M subset: dense = 76.9M x 1002 x 1 B = 77 GB" |
| R1-065 | F1 | 38 | 77 | GB dense (subset) | deterministic | asserts-own | "dense = 76.9M x 1002 x 1 B = 77 GB" |
| R1-066 | F1 | 38 | 5.06 | GB (CSR) | deterministic | asserts-own | "so 77/5.06 = ~15x, not 40x" |
| R1-067 | F1 | 38 | ~15x | reduction (same subset) | deterministic | asserts-own | "so 77/5.06 = ~15x, not 40x" |
| R1-068 | F1 | 38 | 40x | reduction | deterministic | disputes | "so 77/5.06 = ~15x, not 40x" |
| R1-069 | F1 | 39 | 40x | reduction | deterministic | disputes | "The 40x figure is manufactured by pairing a smaller-dataset CSR against a larger-dataset dense" |
| R1-070 | F1 | 39 | 1.4x | reduction (bit-packed, appendix) | deterministic | quotes-paper | "already reports the honest same-dataset ratio (1.4x bit-packed, 7.8x at 0.01% density)" |
| R1-071 | F1 | 39 | 7.8x | reduction (at 0.01% density, appendix) | deterministic | quotes-paper | "already reports the honest same-dataset ratio (1.4x bit-packed, 7.8x at 0.01% density)" |
| R1-072 | F1 | 39 | 0.01% | density | deterministic | quotes-paper | "7.8x at 0.01% density" |
| R1-073 | F1 | 40 | 316,421,093 | nz | deterministic | asserts-own | "direct_mining.log (316,421,093 nz on 76,890,945 txns)" |
| R1-074 | F1 | 40 | 76,890,945 | txns | deterministic | asserts-own | "direct_mining.log (316,421,093 nz on 76,890,945 txns)" |
| R1-075 | F1 | 40 | 205,620,298 | total transactions | deterministic | asserts-own | "beyond_mining.log (Total transactions 205,620,298)" |
| R1-076 | F1 | 42 | 8 | K=22 accessions (heading) | deterministic | quotes-paper | "D3 - 8 K=22 accessions 'recoverable' contradicts 'matrix not deposited'" |
| R1-077 | F1 | 42 | 22 | K (heading) | deterministic | quotes-paper | "D3 - 8 K=22 accessions 'recoverable' contradicts 'matrix not deposited'" |
| R1-078 | F1 | 43 | 8 (eight) | matching proteins (K=22) | deterministic | quotes-paper | "The eight matching proteins can be recovered by querying the source transaction data" |
| R1-079 | F1 | 45 | 8 | IDs (K=22 proteins) | deterministic | asserts-own | "The 8 IDs are neither listed in the paper nor regenerable from this repo" |
| R1-080 | F1 | 49 | <0.45 | p (heading) | deterministic | disputes | "D4 - p<0.45 mis-attributed to the 'rule of three'" |
| R1-081 | F1 | 50 | <0.45 | p | deterministic | quotes-paper | "p < 0.45, the 95% one-sided upper bound for 0 of 5 by the rule of three" |
| R1-082 | F1 | 50 | 95% | one-sided upper bound | method-parameter | quotes-paper | "the 95% one-sided upper bound for 0 of 5 by the rule of three" |
| R1-083 | F1 | 50 | 0 of 5 | null runs reaching K>=7 | deterministic | quotes-paper | "the 95% one-sided upper bound for 0 of 5 by the rule of three" |
| R1-084 | F1 | 51 | 0/5 | null runs reaching K>=7 | deterministic | asserts-own | "0/5 null runs reach K>=7 is confirmed" |
| R1-085 | F1 | 51 | >=7 | K (deep-pattern cutoff) | method-parameter | asserts-own | "0/5 null runs reach K>=7 is confirmed" |
| R1-086 | F1 | 51 | 1-0.05^(1/5)=0.4507 | exact binomial 95% bound | deterministic | asserts-own | "The exact one-sided binomial 95% bound is 1-0.05^(1/5)=0.4507" |
| R1-087 | F1 | 51 | 95% | confidence level | method-parameter | asserts-own | "The exact one-sided binomial 95% bound is 1-0.05^(1/5)=0.4507" |
| R1-088 | F1 | 51 | 3/5=0.60 | rule-of-three bound | deterministic | asserts-own | "The rule of three gives 3/n = 3/5 = 0.60, not 0.45" |
| R1-089 | F1 | 51 | 0.45 | p bound (paper value contrasted with rule of three) | deterministic | quotes-paper | "The rule of three gives 3/n = 3/5 = 0.60, not 0.45." |
| R1-090 | F1 | 52 | 0.45 | p bound | deterministic | asserts-own | "The number (0.45) is correct and defensible as the exact binomial bound" |
| R1-091 | F1 | 61 | 7 | confirmed dataset/vocabulary claims | external-fact | asserts-own | "Dataset & vocabulary (7):" |
| R1-092 | F1 | 62 | 214M | proteins | deterministic | quotes-paper | "214M proteins (line 129) = 214,683,829 rows read" |
| R1-093 | F1 | 62 | 214,683,829 | rows read | deterministic | asserts-own | "214M proteins (line 129) = 214,683,829 rows read - pipeline_214m.log:8,2061" |
| R1-094 | F1 | 63 | 205,620,298 | proteins processed | deterministic | quotes-paper | "205,620,298 processed (lines 129,152) = exact 'passed pLDDT filter' / 'Total transactions'" |
| R1-095 | F1 | 64 | 150 | GB (input data) | external-fact | quotes-paper | "150 GB in 63 min (line 129)" |
| R1-096 | F1 | 64 | 63 | min (feature extraction) | hardware-dependent | quotes-paper | "150 GB in 63 min (line 129)" |
| R1-097 | F1 | 64 | 150G | TrEMBL size (log) | external-fact | asserts-own | "'TrEMBL: 150G'; af_extract 3777.0s = 62.95 min" |
| R1-098 | F1 | 64 | 3777.0 | s (af_extract) | hardware-dependent | asserts-own | "af_extract 3777.0s = 62.95 min" |
| R1-099 | F1 | 64 | 62.95 | min (af_extract) | hardware-dependent | asserts-own | "af_extract 3777.0s = 62.95 min" |
| R1-100 | F1 | 65 | 8 | min-support | method-parameter | quotes-paper | "min-support 8 -> 1,002 frequent single items (Table 1 caption)" |
| R1-101 | F1 | 65 | 1,002 | frequent single items | deterministic | quotes-paper | "min-support 8 -> 1,002 frequent single items (Table 1 caption)" |
| R1-102 | F1 | 66 | 1,006 | items | deterministic | quotes-paper | "1,006 items = 6 pLDDT + 500 Pfam + 500 GO (Table 1)" |
| R1-103 | F1 | 66 | 6 | pLDDT items | method-parameter | quotes-paper | "1,006 items = 6 pLDDT + 500 Pfam + 500 GO (Table 1)" |
| R1-104 | F1 | 66 | 500 | Pfam items | method-parameter | quotes-paper | "1,006 items = 6 pLDDT + 500 Pfam + 500 GO (Table 1)" |
| R1-105 | F1 | 66 | 500 | GO items | method-parameter | quotes-paper | "1,006 items = 6 pLDDT + 500 Pfam + 500 GO (Table 1)" |
| R1-106 | F1 | 67 | 1,002 | items passing min-support | deterministic | asserts-own | "1,002 pass (1,000 Pfam/GO + 2 pLDDT bins)" |
| R1-107 | F1 | 67 | 1,000 | Pfam/GO items passing | deterministic | asserts-own | "1,002 pass (1,000 Pfam/GO + 2 pLDDT bins)" |
| R1-108 | F1 | 67 | 2 | pLDDT bins passing | deterministic | asserts-own | "1,002 pass (1,000 Pfam/GO + 2 pLDDT bins)" |
| R1-109 | F1 | 67 | 2 (two) | pLDDT labels in decoded patterns | deterministic | asserts-own | "corroborated by only two pLDDT labels in decoded_top_k_patterns.txt" |
| R1-110 | F1 | 68 | 76,890,945 | multi-feature proteins | deterministic | quotes-paper | "76,890,945 (37.4%) multi-feature proteins (lines 91,120,152)" |
| R1-111 | F1 | 68 | 37.4% | fraction multi-feature | deterministic | quotes-paper | "76,890,945 (37.4%) multi-feature proteins (lines 91,120,152)" |
| R1-112 | F1 | 70 | 6 | confirmed memory claims | external-fact | asserts-own | "Memory (6):" |
| R1-113 | F1 | 70 | 206 | GB dense | deterministic | quotes-paper | "206 GB dense (line 105)" |
| R1-114 | F1 | 70 | 316M | nz | deterministic | quotes-paper | "316M nz / 5.1 GB coordinate (line 167)" |
| R1-115 | F1 | 70 | 5.1 | GB coordinate | deterministic | quotes-paper | "316M nz / 5.1 GB coordinate (line 167)" |
| R1-116 | F1 | 70 | ~26 | GB bit-packed | deterministic | quotes-paper | "~26 GB bit-packed (lines 167,171)" |
| R1-117 | F1 | 70 | ~3 | GB H2D transfer | deterministic | quotes-paper | "~3 GB H2D transfer (line 171)" |
| R1-118 | F1 | 70 | ~264 | B (metadata over 22 K-levels) | deterministic | quotes-paper | "~264 B over 22 K-levels (line 814)" |
| R1-119 | F1 | 70 | 22 | K-levels | deterministic | quotes-paper | "~264 B over 22 K-levels (line 814)" |
| R1-120 | F1 | 70 | 214M | row label (appendix memory table) | deterministic | quotes-paper | "appendix 214M row 27 GB/19 GB/1.4x (line 887)" |
| R1-121 | F1 | 70 | 27 | GB (appendix 214M row) | deterministic | quotes-paper | "appendix 214M row 27 GB/19 GB/1.4x (line 887)" |
| R1-122 | F1 | 70 | 19 | GB (appendix 214M row) | deterministic | quotes-paper | "appendix 214M row 27 GB/19 GB/1.4x (line 887)" |
| R1-123 | F1 | 70 | 1.4x | ratio (appendix 214M row) | deterministic | quotes-paper | "appendix 214M row 27 GB/19 GB/1.4x (line 887)" |
| R1-124 | F1 | 72 | 6/6 | campaign rows verified | external-fact | asserts-own | "Campaign table (6/6):" |
| R1-125 | F1 | 72 | 0.1% | support (Base) | method-parameter | quotes-paper | "Base (0.1%/76,891/5,305/K9/1.9min/SON)" |
| R1-126 | F1 | 72 | 76,891 | min_count (Base) | method-parameter | quotes-paper | "Base (0.1%/76,891/5,305/K9/1.9min/SON)" |
| R1-127 | F1 | 72 | 5,305 | itemsets (Base) | deterministic | quotes-paper | "Base (0.1%/76,891/5,305/K9/1.9min/SON)" |
| R1-128 | F1 | 72 | 9 | K max (Base) | deterministic | quotes-paper | "Base (0.1%/76,891/5,305/K9/1.9min/SON)" |
| R1-129 | F1 | 72 | 1.9 | min (Base) | hardware-dependent | quotes-paper | "Base (0.1%/76,891/5,305/K9/1.9min/SON)" |
| R1-130 | F1 | 72 | 0.01% | support (Super) | method-parameter | quotes-paper | "Super (0.01%/7,689/51,124/K13/4.3min/SON)" |
| R1-131 | F1 | 72 | 7,689 | min_count (Super) | method-parameter | quotes-paper | "Super (0.01%/7,689/51,124/K13/4.3min/SON)" |
| R1-132 | F1 | 72 | 51,124 | itemsets (Super) | deterministic | quotes-paper | "Super (0.01%/7,689/51,124/K13/4.3min/SON)" |
| R1-133 | F1 | 72 | 13 | K max (Super) | deterministic | quotes-paper | "Super (0.01%/7,689/51,124/K13/4.3min/SON)" |
| R1-134 | F1 | 72 | 4.3 | min (Super) | hardware-dependent | quotes-paper | "Super (0.01%/7,689/51,124/K13/4.3min/SON)" |
| R1-135 | F1 | 72 | 0.001% | support (Power) | method-parameter | quotes-paper | "Power (0.001%/768/22,846/K13/18.1min/SON)" |
| R1-136 | F1 | 72 | 768 | min_count (Power) | method-parameter | quotes-paper | "Power (0.001%/768/22,846/K13/18.1min/SON)" |
| R1-137 | F1 | 72 | 22,846 | itemsets (Power) | deterministic | quotes-paper | "Power (0.001%/768/22,846/K13/18.1min/SON)" |
| R1-138 | F1 | 72 | 13 | K max (Power) | deterministic | quotes-paper | "Power (0.001%/768/22,846/K13/18.1min/SON)" |
| R1-139 | F1 | 72 | 18.1 | min (Power) | hardware-dependent | quotes-paper | "Power (0.001%/768/22,846/K13/18.1min/SON)" |
| R1-140 | F1 | 72 | 0.0001% | support (Blitz) | method-parameter | quotes-paper | "Blitz (0.0001%/77/2,841,280/K19/2.0min/Direct)" |
| R1-141 | F1 | 72 | 77 | min_count (Blitz) | method-parameter | quotes-paper | "Blitz (0.0001%/77/2,841,280/K19/2.0min/Direct)" |
| R1-142 | F1 | 72 | 2,841,280 | itemsets (Blitz) | deterministic | quotes-paper | "Blitz (0.0001%/77/2,841,280/K19/2.0min/Direct)" |
| R1-143 | F1 | 72 | 19 | K max (Blitz) | deterministic | quotes-paper | "Blitz (0.0001%/77/2,841,280/K19/2.0min/Direct)" |
| R1-144 | F1 | 72 | 2.0 | min (Blitz) | hardware-dependent | quotes-paper | "Blitz (0.0001%/77/2,841,280/K19/2.0min/Direct)" |
| R1-145 | F1 | 72 | 0.00002% | support (Ultra) | method-parameter | quotes-paper | "Ultra (0.00002%/16/14,558,875/K20/4.7min/Direct)" |
| R1-146 | F1 | 72 | 16 | min_count (Ultra) | method-parameter | quotes-paper | "Ultra (0.00002%/16/14,558,875/K20/4.7min/Direct)" |
| R1-147 | F1 | 72 | 14,558,875 | itemsets (Ultra) | deterministic | quotes-paper | "Ultra (0.00002%/16/14,558,875/K20/4.7min/Direct)" |
| R1-148 | F1 | 72 | 20 | K max (Ultra) | deterministic | quotes-paper | "Ultra (0.00002%/16/14,558,875/K20/4.7min/Direct)" |
| R1-149 | F1 | 72 | 4.7 | min (Ultra) | hardware-dependent | quotes-paper | "Ultra (0.00002%/16/14,558,875/K20/4.7min/Direct)" |
| R1-150 | F1 | 72 | 0.00001% | support (Opus) | method-parameter | quotes-paper | "Opus (0.00001%/8/26,849,505/K22/7.3min/Direct)" |
| R1-151 | F1 | 72 | 8 | min_count (Opus) | method-parameter | quotes-paper | "Opus (0.00001%/8/26,849,505/K22/7.3min/Direct)" |
| R1-152 | F1 | 72 | 26,849,505 | itemsets (Opus) | deterministic | quotes-paper | "Opus (0.00001%/8/26,849,505/K22/7.3min/Direct)" |
| R1-153 | F1 | 72 | 22 | K max (Opus) | deterministic | quotes-paper | "Opus (0.00001%/8/26,849,505/K22/7.3min/Direct)" |
| R1-154 | F1 | 72 | 7.3 | min (Opus) | hardware-dependent | quotes-paper | "Opus (0.00001%/8/26,849,505/K22/7.3min/Direct)" |
| R1-155 | F1 | 74 | 8/8 | K-distribution checks passed | external-fact | asserts-own | "K-distribution (8/8):" |
| R1-156 | F1 | 74 | 26,849,505 | total itemsets | deterministic | quotes-paper | "total 26,849,505; peak K=9 = 3,529,257 (13.14%)" |
| R1-157 | F1 | 74 | 9 | K (distribution peak) | deterministic | quotes-paper | "peak K=9 = 3,529,257 (13.14%)" |
| R1-158 | F1 | 74 | 3,529,257 | itemsets at K=9 | deterministic | quotes-paper | "peak K=9 = 3,529,257 (13.14%)" |
| R1-159 | F1 | 74 | 13.14% | share of itemsets at K=9 | deterministic | quotes-paper | "peak K=9 = 3,529,257 (13.14%)" |
| R1-160 | F1 | 74 | 1,002 | itemsets at K=1 | deterministic | quotes-paper | "K=1=1,002; K=2=73,786; K=22=1 (8 proteins)" |
| R1-161 | F1 | 74 | 73,786 | itemsets at K=2 | deterministic | quotes-paper | "K=1=1,002; K=2=73,786; K=22=1 (8 proteins)" |
| R1-162 | F1 | 74 | 1 | itemsets at K=22 | deterministic | quotes-paper | "K=22=1 (8 proteins)" |
| R1-163 | F1 | 74 | 8 | proteins supporting K=22 | deterministic | quotes-paper | "K=22=1 (8 proteins)" |
| R1-164 | F1 | 74 | 3..21 | K rows verified | deterministic | quotes-paper | "all K=3..21 rows" |
| R1-165 | F1 | 74 | 100.0% | sum of K-distribution percentages | deterministic | asserts-own | "percentages recomputed and sum to 100.0%" |
| R1-166 | F1 | 76 | 2 | K=22 composition checks | external-fact | asserts-own | "K=22 composition (2):" |
| R1-167 | F1 | 76 | 22 | features (K=22 itemset) | deterministic | quotes-paper | "22 features = 2 Pfam + 8 MF + 4 BP + 7 CC + 1 pLDDT" |
| R1-168 | F1 | 76 | 2 | Pfam features in K=22 | deterministic | quotes-paper | "22 features = 2 Pfam + 8 MF + 4 BP + 7 CC + 1 pLDDT" |
| R1-169 | F1 | 76 | 8 | GO:MF features in K=22 | deterministic | quotes-paper | "22 features = 2 Pfam + 8 MF + 4 BP + 7 CC + 1 pLDDT" |
| R1-170 | F1 | 76 | 4 | GO:BP features in K=22 | deterministic | quotes-paper | "22 features = 2 Pfam + 8 MF + 4 BP + 7 CC + 1 pLDDT" |
| R1-171 | F1 | 76 | 7 | GO:CC features in K=22 | deterministic | quotes-paper | "22 features = 2 Pfam + 8 MF + 4 BP + 7 CC + 1 pLDDT" |
| R1-172 | F1 | 76 | 1 | pLDDT feature in K=22 | deterministic | quotes-paper | "22 features = 2 Pfam + 8 MF + 4 BP + 7 CC + 1 pLDDT" |
| R1-173 | F1 | 78 | 17 | confirmed null-model claims | external-fact | asserts-own | "Null model (17):" |
| R1-174 | F1 | 78 | 5 | permutations | method-parameter | quotes-paper | "5 permutations, seed 42, 662 s total, min_count=769" |
| R1-175 | F1 | 78 | 42 | seed | method-parameter | quotes-paper | "5 permutations, seed 42, 662 s total, min_count=769" |
| R1-176 | F1 | 78 | 662 | s (null model total) | hardware-dependent | quotes-paper | "5 permutations, seed 42, 662 s total, min_count=769" |
| R1-177 | F1 | 78 | 769 | min_count (null model) | method-parameter | quotes-paper | "5 permutations, seed 42, 662 s total, min_count=769" |
| R1-178 | F1 | 78 | -987 | Z at K=2 | deterministic | quotes-paper | "Z at K=2 (-987), K=3 (-143), K=4 (+3,791), K=5 (+7,402), K=6 (+71,728)" |
| R1-179 | F1 | 78 | -143 | Z at K=3 | deterministic | quotes-paper | "Z at K=2 (-987), K=3 (-143), K=4 (+3,791), K=5 (+7,402), K=6 (+71,728)" |
| R1-180 | F1 | 78 | +3,791 | Z at K=4 | deterministic | quotes-paper | "Z at K=2 (-987), K=3 (-143), K=4 (+3,791), K=5 (+7,402), K=6 (+71,728)" |
| R1-181 | F1 | 78 | +7,402 | Z at K=5 | deterministic | quotes-paper | "Z at K=2 (-987), K=3 (-143), K=4 (+3,791), K=5 (+7,402), K=6 (+71,728)" |
| R1-182 | F1 | 78 | +71,728 | Z at K=6 | deterministic | quotes-paper | "Z at K=2 (-987), K=3 (-143), K=4 (+3,791), K=5 (+7,402), K=6 (+71,728)" |
| R1-183 | F1 | 78 | 3 | K (null-model peak) | deterministic | quotes-paper | "null peaks K=3 (46.2%)" |
| R1-184 | F1 | 78 | 46.2% | null share at K=3 | deterministic | quotes-paper | "null peaks K=3 (46.2%)" |
| R1-185 | F1 | 78 | 6 | K (null max depth) | deterministic | quotes-paper | "null max depth K=6 (mean 22, std 1.1)" |
| R1-186 | F1 | 78 | 22 | mean null itemsets at K=6 | deterministic | quotes-paper | "null max depth K=6 (mean 22, std 1.1)" |
| R1-187 | F1 | 78 | 1.1 | std null itemsets at K=6 | deterministic | quotes-paper | "null max depth K=6 (mean 22, std 1.1)" |
| R1-188 | F1 | 78 | 88,745 | biological itemsets at K>=7 | deterministic | quotes-paper | "bio K>=7 = 88,745; real_total 475,865 to K=14" |
| R1-189 | F1 | 78 | 475,865 | real_total itemsets (0.001%) | deterministic | quotes-paper | "bio K>=7 = 88,745; real_total 475,865 to K=14" |
| R1-190 | F1 | 78 | 14 | K max (real, 0.001%) | deterministic | quotes-paper | "real_total 475,865 to K=14" |
| R1-191 | F1 | 78 | 1,002 | K=1 itemsets preserved under permutation | deterministic | quotes-paper | "K=1 preserves 1,002; ~130 s/permutation" |
| R1-192 | F1 | 78 | ~130 | s per permutation | hardware-dependent | quotes-paper | "K=1 preserves 1,002; ~130 s/permutation" |
| R1-193 | F1 | 78 | n=5 | permutations (t-statistic basis) | method-parameter | quotes-paper | "The n=5 Z-scores are appropriately recast as t-statistics (4 df)" |
| R1-194 | F1 | 78 | 4 | df (t-statistics) | deterministic | quotes-paper | "The n=5 Z-scores are appropriately recast as t-statistics (4 df)" |
| R1-195 | F1 | 80 | 6/6 | Direct-vs-SON checks passed | external-fact | asserts-own | "Direct vs SON (6/6):" |
| R1-196 | F1 | 80 | 475,865 | itemsets (Direct) | deterministic | quotes-paper | "Direct 475,865 in 50.7 s; SON 22,846 in 1,085.6 s; 21x speedup; SON misses 95.2%" |
| R1-197 | F1 | 80 | 50.7 | s (Direct) | hardware-dependent | quotes-paper | "Direct 475,865 in 50.7 s" |
| R1-198 | F1 | 80 | 22,846 | itemsets (SON) | deterministic | quotes-paper | "SON 22,846 in 1,085.6 s" |
| R1-199 | F1 | 80 | 1,085.6 | s (SON) | hardware-dependent | quotes-paper | "SON 22,846 in 1,085.6 s" |
| R1-200 | F1 | 80 | 21x | speedup (Direct vs SON) | hardware-dependent | quotes-paper | "21x speedup; SON misses 95.2%" |
| R1-201 | F1 | 80 | 95.2% | SON miss rate | deterministic | quotes-paper | "21x speedup; SON misses 95.2%" |
| R1-202 | F1 | 82 | 3 | confirmed scale-comparison claims | external-fact | asserts-own | "Scale comparison (3):" |
| R1-203 | F1 | 82 | 5.1x | transactions ratio vs GMiner | deterministic | quotes-paper | "5.1x (76.9M/15M); 62.6% = 128.7M excluded" |
| R1-204 | F1 | 82 | 76.9M | transactions (ET-miner) | deterministic | quotes-paper | "5.1x (76.9M/15M)" |
| R1-205 | F1 | 82 | 15M | transactions (GMiner) | external-fact | quotes-paper | "5.1x (76.9M/15M)" |
| R1-206 | F1 | 82 | 62.6% | proteins excluded (single-feature) | deterministic | quotes-paper | "62.6% = 128.7M excluded" |
| R1-207 | F1 | 82 | 128.7M | proteins excluded | deterministic | quotes-paper | "62.6% = 128.7M excluded" |
| R1-208 | F1 | 82 | 76.9M | transactions (ET-miner row) | deterministic | quotes-paper | "ET-miner row 76.9M/1,002/K22/1xH100/7.3min" |
| R1-209 | F1 | 82 | 1,002 | items (ET-miner row) | deterministic | quotes-paper | "ET-miner row 76.9M/1,002/K22/1xH100/7.3min" |
| R1-210 | F1 | 82 | 22 | K (ET-miner row) | deterministic | quotes-paper | "ET-miner row 76.9M/1,002/K22/1xH100/7.3min" |
| R1-211 | F1 | 82 | 1xH100 | GPU count/model | hardware-dependent | quotes-paper | "ET-miner row 76.9M/1,002/K22/1xH100/7.3min" |
| R1-212 | F1 | 82 | 7.3 | min (ET-miner row) | hardware-dependent | quotes-paper | "ET-miner row 76.9M/1,002/K22/1xH100/7.3min" |
| R1-213 | F1 | 84 | 7 | confirmed citation claims | external-fact | asserts-own | "Citations (7):" |
| R1-214 | F1 | 84 | 47 | \cite commands | external-fact | asserts-own | "47 \cite -> 37 unique keys, all with matching \bibitem and none unused" |
| R1-215 | F1 | 84 | 37 | unique citation keys | external-fact | asserts-own | "47 \cite -> 37 unique keys, all with matching \bibitem and none unused" |
| R1-216 | F1 | 84 | 37 | bibliography entries | external-fact | asserts-own | "\begin{thebibliography}{37} = 37 entries" |
| R1-217 | F1 | 84 | 0 (zero) | cross-paper tool contamination | external-fact | asserts-own | "zero cross-paper (opus_pocket: fpocket, efficient-apriori, scikit-learn, PDBbind) contamination" |
| R1-218 | F1 | 94 | 2025_01 | UniProt release | external-fact | quotes-paper | "UniProt release 2025_01 / line 129, 478 / Log records only input filename" |
| R1-219 | F1 | 94 | Feb 2026 | run date (log) | external-fact | asserts-own | "Log records only input filename uniprot_trembl.dat.gz and Feb 2026 run date; no release string" |
| R1-220 | F1 | 95 | 22 | K | deterministic | quotes-paper | "K=22 shared by exactly 8 proteins / lines 254, 286" |
| R1-221 | F1 | 95 | 8 | proteins (K=22) | deterministic | quotes-paper | "K=22 shared by exactly 8 proteins / lines 254, 286" |
| R1-222 | F1 | 95 | 19 | K max in decoded_top_k_patterns.txt | deterministic | asserts-own | "decoded_top_k_patterns.txt caps at K=19 (187 proteins) and is a different run" |
| R1-223 | F1 | 95 | 187 | proteins (K=19 pattern in decoded file) | deterministic | asserts-own | "caps at K=19 (187 proteins) and is a different run" |
| R1-224 | F1 | 95 | >=20 | K (no artifact contains any) | deterministic | asserts-own | "no artifact contains any K>=20 itemset" |
| R1-225 | F1 | 96 | 22->21 | independent features after GO check | deterministic | quotes-paper | "22->21 independent (0 GO parent-child pairs; InterPro2GO link removed)" |
| R1-226 | F1 | 96 | 0 | GO parent-child pairs | deterministic | quotes-paper | "22->21 independent (0 GO parent-child pairs; InterPro2GO link removed)" |
| R1-227 | F1 | 97 | 4 | min_count (verification run) | method-parameter | quotes-paper | "min_count=4 -> 48M itemsets, same K max / line 335" |
| R1-228 | F1 | 97 | 48M | itemsets (min_count=4 run) | deterministic | disputes | "No min_count=4 run exists. The only '48M' is 'Written 48M transactions' during extraction" |
| R1-229 | F1 | 97 | 48M | transactions written (extraction log) | deterministic | asserts-own | "The only '48M' is 'Written 48M transactions' during extraction - a transaction count, not itemsets" |
| R1-230 | F1 | 97 | 769 | min_count (null-model JSON) | method-parameter | asserts-own | "experiment_null_model_...json (min_count 769)" |
| R1-231 | F1 | 98 | 8 | K=22 UniProt accessions | deterministic | quotes-paper | "8 K=22 UniProt accessions / line 333 / See D3 - not recoverable" |
| R1-232 | F1 | 98 | 22 | K | deterministic | quotes-paper | "8 K=22 UniProt accessions" |
| R1-233 | F1 | 99 | 15M | transactions (GMiner basis) | external-fact | quotes-paper | "Internally used consistently (GMiner 15M basis for 5.1x)" |
| R1-234 | F1 | 99 | 5.1x | transactions ratio | deterministic | quotes-paper | "Internally used consistently (GMiner 15M basis for 5.1x)" |
| R1-235 | F1 | 100 | 37 | references | external-fact | quotes-paper | "External correctness of 37 references / lines 541-724" |
| R1-236 | F1 | 108 | 231 | lines (manuscript rewrite in commit diff) | external-fact | asserts-own | "its diff is a 231-line manuscript rewrite that deletes expanded-run figures" |
| R1-237 | F1 | 110 | 606K | itemsets (expanded run, revision_notes_b3.tex) | deterministic | disputes | "606K itemsets @ 4xH200 in revision_notes_b3.tex vs 16.8B @ 8xH200 in the plan" |
| R1-238 | F1 | 110 | 4xH200 | GPUs (expanded run, revision_notes_b3.tex) | hardware-dependent | disputes | "606K itemsets @ 4xH200 in revision_notes_b3.tex vs 16.8B @ 8xH200 in the plan" |
| R1-239 | F1 | 110 | 16.8B | itemsets (expanded run, plan) | deterministic | disputes | "606K itemsets @ 4xH200 in revision_notes_b3.tex vs 16.8B @ 8xH200 in the plan" |
| R1-240 | F1 | 110 | 8xH200 | GPUs (expanded run, plan) | hardware-dependent | disputes | "606K itemsets @ 4xH200 in revision_notes_b3.tex vs 16.8B @ 8xH200 in the plan" |
| R1-241 | F1 | 111 | n=5 | permutations (caveat) | method-parameter | quotes-paper | "Abstract/conclusion omit the n=5 caveat on Z>3,700 (lines 91, 498)" |
| R1-242 | F1 | 111 | >3,700 | Z (headline) | deterministic | quotes-paper | "Abstract/conclusion omit the n=5 caveat on Z>3,700 (lines 91, 498)" |
| R1-243 | F1 | 111 | 4 | df | deterministic | quotes-paper | "Body (table caption + s4.4) correctly recasts these as t-statistics (4 df)" |
| R1-244 | F1 | 112 | Feb 2026 | date (stale Dutch translation) | external-fact | asserts-own | "Stale Dutch translation et_miner_proteome_nl.tex (old title, Feb 2026 date)" |
| R1-245 | F1 | 113 | 8 | K=22 accessions to list | deterministic | requests | "(a) list the 8 K=22 accessions - see D3" |
| R1-246 | F1 | 113 | ~11,000 | proteins (approximate count) | deterministic | quotes-paper | "replace the '~11,000 / ~10,500 / ~16,000' approximate protein counts (lines 353, 355, 357)" |
| R1-247 | F1 | 113 | ~10,500 | proteins (approximate count) | deterministic | quotes-paper | "replace the '~11,000 / ~10,500 / ~16,000' approximate protein counts (lines 353, 355, 357)" |
| R1-248 | F1 | 113 | ~16,000 | proteins (approximate count) | deterministic | quotes-paper | "replace the '~11,000 / ~10,500 / ~16,000' approximate protein counts (lines 353, 355, 357)" |
| R1-249 | F1 | 113 | 7.3 | min (mining-only) | hardware-dependent | requests | "(d) state explicitly that 7.3 min is mining-only wherever the figure appears standalone" |
| R1-250 | F1 | 122 | >=8 | proteins (feature retention, OLD text) | method-parameter | quotes-paper | "OLD: We retained all features occurring in at least 8 proteins (the lowest support threshold used)." |
| R1-251 | F1 | 123 | 500 | most frequent Pfam domains (proposed text) | method-parameter | requests | "The feature vocabulary comprises the 500 most frequent Pfam domains and the 500 most frequent GO terms" |
| R1-252 | F1 | 123 | 500 | most frequent GO terms (proposed text) | method-parameter | requests | "The feature vocabulary comprises the 500 most frequent Pfam domains and the 500 most frequent GO terms" |
| R1-253 | F1 | 123 | 24,291 | Pfam families observed (proposed text) | deterministic | requests | "(drawn from 24,291 Pfam and 25,993 GO families observed across the corpus)" |
| R1-254 | F1 | 123 | 25,993 | GO families observed (proposed text) | deterministic | requests | "(drawn from 24,291 Pfam and 25,993 GO families observed across the corpus)" |
| R1-255 | F1 | 123 | 6 | pLDDT confidence bins (proposed text) | method-parameter | requests | "together with 6 pLDDT confidence bins, for 1,006 defined items" |
| R1-256 | F1 | 123 | 1,006 | defined items (proposed text) | deterministic | requests | "together with 6 pLDDT confidence bins, for 1,006 defined items" |
| R1-257 | F1 | 123 | 8 | proteins (minimum support, proposed text) | method-parameter | requests | "The minimum support threshold of 8 proteins is applied during mining, not during vocabulary construction" |
| R1-258 | F1 | 125 | 40x | reduction (cross-dataset claim to fix) | deterministic | disputes | "[D2] Line 167 - fix the cross-dataset 40x claim. Make the reduction ratio same-dataset." |
| R1-259 | F1 | 126 | 316 million | non-zero entries (OLD text) | deterministic | quotes-paper | "the resulting matrix contains 316 million non-zero entries occupying ~5.1 GB in coordinate format" |
| R1-260 | F1 | 126 | ~5.1 | GB (OLD text) | deterministic | quotes-paper | "316 million non-zero entries occupying ~5.1 GB in coordinate format" |
| R1-261 | F1 | 126 | 2 x 64-bit | integers per COO entry (OLD text) | method-parameter | quotes-paper | "(two 64-bit integers per entry)" |
| R1-262 | F1 | 126 | 40x | reduction (OLD text) | deterministic | disputes | "a 40x reduction from the ~206 GB naive dense representation (one byte per boolean entry)" |
| R1-263 | F1 | 126 | ~206 | GB (OLD text) | deterministic | quotes-paper | "a 40x reduction from the ~206 GB naive dense representation" |
| R1-264 | F1 | 126 | 1 | byte per boolean entry (OLD text) | method-parameter | quotes-paper | "(one byte per boolean entry)" |
| R1-265 | F1 | 127 | 76.9M | proteins (mining subset, NEW text) | deterministic | requests | "for the 76.9M-protein mining subset the resulting matrix contains 316 million non-zero entries" |
| R1-266 | F1 | 127 | 316 million | non-zero entries (NEW text) | deterministic | requests | "the resulting matrix contains 316 million non-zero entries occupying ~5.1 GB" |
| R1-267 | F1 | 127 | ~5.1 | GB (NEW text) | deterministic | requests | "316 million non-zero entries occupying ~5.1 GB in coordinate format" |
| R1-268 | F1 | 127 | ~15x | reduction (NEW text) | deterministic | requests | "a ~15x reduction from the ~77 GB dense representation of that same subset" |
| R1-269 | F1 | 127 | ~77 | GB dense of subset (NEW text) | deterministic | requests | "a ~15x reduction from the ~77 GB dense representation of that same subset" |
| R1-270 | F1 | 127 | ~206 | GB dense of full set (NEW text) | deterministic | requests | "relative to the ~206 GB dense matrix of the full 205.6M-protein set it is ~40x smaller" |
| R1-271 | F1 | 127 | 205.6M | proteins (full set, NEW text) | deterministic | requests | "relative to the ~206 GB dense matrix of the full 205.6M-protein set it is ~40x smaller" |
| R1-272 | F1 | 127 | ~40x | smaller (cross-dataset, NEW text) | deterministic | requests | "relative to the ~206 GB dense matrix of the full 205.6M-protein set it is ~40x smaller" |
| R1-273 | F1 | 127 | 2 x 64-bit | integers per COO entry (NEW text) | method-parameter | requests | "occupying ~5.1 GB in coordinate format (two 64-bit integers per entry)" |
| R1-274 | F1 | 128 | 40x | reduction (no longer single-dataset) | deterministic | disputes | "stops presenting 40x as a single-dataset reduction" |
| R1-275 | F1 | 131 | 206 | GB (OLD memory path) | deterministic | quotes-paper | "The 206 GB -> 5.1 GB -> 26 GB memory reduction path is applicable to any sparse" |
| R1-276 | F1 | 131 | 5.1 | GB (OLD memory path) | deterministic | quotes-paper | "The 206 GB -> 5.1 GB -> 26 GB memory reduction path" |
| R1-277 | F1 | 131 | 26 | GB (OLD memory path) | deterministic | quotes-paper | "The 206 GB -> 5.1 GB -> 26 GB memory reduction path" |
| R1-278 | F1 | 132 | 206 | GB full-set dense (NEW text) | deterministic | requests | "The full-set 206 GB dense matrix, the 5.1 GB CSR of the mined subset, and the 26 GB GPU bitvector matrix" |
| R1-279 | F1 | 132 | 5.1 | GB CSR of mined subset (NEW text) | deterministic | requests | "the 5.1 GB CSR of the mined subset, and the 26 GB GPU bitvector matrix" |
| R1-280 | F1 | 132 | 26 | GB GPU bitvector matrix (NEW text) | deterministic | requests | "the 26 GB GPU bitvector matrix illustrate a compression path" |
| R1-281 | F1 | 132 | ~15x | same-subset dense-to-CSR reduction (NEW text) | deterministic | requests | "(the same-subset dense-to-CSR reduction is ~15x; see Appendix)" |
| R1-282 | F1 | 135 | 8 | UniProt accessions (to obtain and list) | deterministic | requests | "obtain the 8 UniProt accessions from the authors and list them" |
| R1-283 | F1 | 135 | 8 (eight) | matching proteins (proposed text) | deterministic | requests | "The eight matching proteins (accessions listed in Table) were identified with the released analysis code" |
| R1-284 | F1 | 136 | 2025_01 | UniProt release (fallback text) | external-fact | requests | "they are regenerable only from UniProt release 2025_01 via the released feature-extraction pipeline" |
| R1-285 | F1 | 138 | 0.45 | p bound (correct source to be stated) | deterministic | requests | "replace the phrase by the rule of three with the correct source of 0.45" |
| R1-286 | F1 | 139 | <0.45 | p (OLD caption) | deterministic | quotes-paper | "(p < 0.45, the 95% one-sided upper bound for 0 of 5 by the rule of three)." |
| R1-287 | F1 | 139 | 95% | one-sided upper bound (OLD caption) | method-parameter | quotes-paper | "(p < 0.45, the 95% one-sided upper bound for 0 of 5 by the rule of three)." |
| R1-288 | F1 | 139 | 0 of 5 | null runs (OLD caption) | deterministic | quotes-paper | "(p < 0.45, the 95% one-sided upper bound for 0 of 5 by the rule of three)." |
| R1-289 | F1 | 140 | <0.45 | p (NEW caption) | deterministic | requests | "(p < 0.45, the exact one-sided binomial 95% upper bound 1-0.05^{1/5} for 0 of 5)." |
| R1-290 | F1 | 140 | 95% | binomial upper bound (NEW caption) | method-parameter | requests | "the exact one-sided binomial 95% upper bound 1-0.05^{1/5} for 0 of 5" |
| R1-291 | F1 | 140 | 1-0.05^{1/5} | bound formula (NEW caption) | deterministic | requests | "the exact one-sided binomial 95% upper bound 1-0.05^{1/5} for 0 of 5" |
| R1-292 | F1 | 140 | 0 of 5 | null runs (NEW caption) | deterministic | requests | "the exact one-sided binomial 95% upper bound 1-0.05^{1/5} for 0 of 5" |
| R1-293 | F1 | 141 | <0.45 | p (paper line 391) | deterministic | quotes-paper | "yielding p < 0.45 (the 95% one-sided upper bound for 0 of 5 by the rule of three)" |
| R1-294 | F1 | 141 | 95% | one-sided upper bound (paper line 391) | method-parameter | quotes-paper | "(the 95% one-sided upper bound for 0 of 5 by the rule of three)" |
| R1-295 | F1 | 141 | 0 of 5 | null runs (paper line 391) | deterministic | quotes-paper | "(the 95% one-sided upper bound for 0 of 5 by the rule of three)" |
| R1-296 | F1 | 141 | 1-0.05^{1/5} | bound formula (replacement) | deterministic | requests | "... the exact one-sided binomial 95% upper bound 1-0.05^{1/5} for 0 of 5" |
| R1-297 | F1 | 141 | <0.45 | p (table cell, paper line 382) | deterministic | quotes-paper | "Line 382 table cell shows only <0.45 and needs no change once the caption is fixed" |
| R1-298 | F1 | 143 | 4 | min_count (unverifiable run) | method-parameter | disputes | "No artifact records a min_count=4 run or a 48M-itemset count" |
| R1-299 | F1 | 143 | 48M | itemsets (unverifiable) | deterministic | disputes | "No artifact records a min_count=4 run or a 48M-itemset count" |
| R1-300 | F1 | 143 | 48M | transactions written during extraction | deterministic | asserts-own | "(the only '48M' in logs is transactions written during extraction)" |
| R1-301 | F1 | 144 | 4 | min_count (OLD text) | method-parameter | quotes-paper | "A verification run at min_count=4 discovered 48 million itemsets with identical maximum" |
| R1-302 | F1 | 144 | 48 million | itemsets (OLD text) | deterministic | disputes | "A verification run at min_count=4 discovered 48 million itemsets with identical maximum" |
| R1-303 | F1 | 144 | 8 | proteins (NEW text) | deterministic | requests | "Because each of the 8 proteins carries exactly 22 vocabulary features, K=23 is impossible at any support threshold" |
| R1-304 | F1 | 144 | 22 | vocabulary features per K=22 protein (NEW text) | deterministic | requests | "each of the 8 proteins carries exactly 22 vocabulary features" |
| R1-305 | F1 | 144 | 23 | K (impossible, NEW text) | deterministic | requests | "K=23 is impossible at any support threshold; the ceiling is therefore structural" |
| R1-306 | F1 | 146 | 13 | K (highlighted pattern) | deterministic | quotes-paper | "the highlighted K=13 / K=12 / K=11 patterns from decoded_top_k_patterns.txt" |
| R1-307 | F1 | 146 | 12 | K (highlighted pattern) | deterministic | quotes-paper | "the highlighted K=13 / K=12 / K=11 patterns from decoded_top_k_patterns.txt" |
| R1-308 | F1 | 146 | 11 | K (highlighted pattern) | deterministic | quotes-paper | "the highlighted K=13 / K=12 / K=11 patterns from decoded_top_k_patterns.txt" |
| R1-309 | F1 | 146 | ~11,000 | proteins (K=13 pattern) | deterministic | quotes-paper | "replace (~11,000 proteins), (~10,500 proteins), (~16,000 proteins) with the exact integers" |
| R1-310 | F1 | 146 | ~10,500 | proteins (K=12 pattern) | deterministic | quotes-paper | "replace (~11,000 proteins), (~10,500 proteins), (~16,000 proteins) with the exact integers" |
| R1-311 | F1 | 146 | ~16,000 | proteins (K=11 pattern) | deterministic | quotes-paper | "replace (~11,000 proteins), (~10,500 proteins), (~16,000 proteins) with the exact integers" |
| R1-312 | F1 | 148 | 2025_01 | UniProt/Swiss-Prot release (paper line 478) | external-fact | disputes | "line 478 says UniProt/Swiss-Prot release 2025_01" (vs TrEMBL at line 129) |
| R1-313 | F1 | 149 | 2025_01 | UniProt/Swiss-Prot release (OLD) | external-fact | quotes-paper | "our results reflect UniProt/Swiss-Prot release 2025_01" |
| R1-314 | F1 | 150 | 2025_01 | UniProt TrEMBL release (NEW) | external-fact | requests | "our results reflect UniProt TrEMBL release 2025_01" |
| R1-315 | F1 | 152 | 2025_01 | release string (not in any artifact) | external-fact | disputes | "The '2025_01' release is not recorded in any artifact. Confirm the exact release with the authors" |
| R1-316 | F1 | 152 | February 2026 | access date (softened wording) | external-fact | requests | "soften to 'the UniProt TrEMBL release accessed February 2026'" |
| R1-317 | F1 | 154 | n=5 | permutations (scope to add to headline Z) | method-parameter | requests | "[Integrity 4] Conclusion line 498 - add the n=5 scope to the headline Z." |
| R1-318 | F1 | 155 | >3,700 | Z (OLD conclusion) | deterministic | quotes-paper | "(Z>3,700 for K=4--6; no null run reached K>=7)." |
| R1-319 | F1 | 155 | 4-6 | K range (OLD conclusion) | deterministic | quotes-paper | "(Z>3,700 for K=4--6; no null run reached K>=7)." |
| R1-320 | F1 | 155 | >=7 | K (no null run reached, OLD) | deterministic | quotes-paper | "(Z>3,700 for K=4--6; no null run reached K>=7)." |
| R1-321 | F1 | 156 | >3,700 | Z (NEW conclusion) | deterministic | requests | "(Z>3,700 for K=4--6, as t-statistics from 5 permutations; no null run reached K>=7)" |
| R1-322 | F1 | 156 | 4-6 | K range (NEW conclusion) | deterministic | requests | "(Z>3,700 for K=4--6, as t-statistics from 5 permutations" |
| R1-323 | F1 | 156 | 5 | permutations (NEW conclusion) | method-parameter | requests | "as t-statistics from 5 permutations; no null run reached K>=7" |
| R1-324 | F1 | 156 | >=7 | K (NEW conclusion) | deterministic | requests | "no null run reached K>=7; see Section" |
| R1-325 | F1 | 158 | 7.3 | minutes (paper line 120) | hardware-dependent | quotes-paper | "discovers the complete feature co-occurrence landscape in 7.3 minutes -> ... in 7.3 minutes of mining" |
| R1-326 | F1 | 158 | 7.3 | minutes (paper line 496) | hardware-dependent | quotes-paper | "mining completes in 7.3 minutes -> mining completes in 7.3 minutes (excluding the 63-minute one-time feature extraction" |
| R1-327 | F1 | 158 | 63 | minute feature extraction (NEW text) | hardware-dependent | requests | "(excluding the 63-minute one-time feature extraction, Section)" |
| R1-328 | F2 | 12 | 76.9 million | multi-feature proteins | deterministic | quotes-paper | "across 76.9 million multi-feature proteins (1,002 features) on a single NVIDIA H100 in 7.3 minutes" |
| R1-329 | F2 | 12 | 1,002 | features | deterministic | quotes-paper | "across 76.9 million multi-feature proteins (1,002 features)" |
| R1-330 | F2 | 12 | 1 (single) | NVIDIA H100 GPU | hardware-dependent | quotes-paper | "on a single NVIDIA H100 in 7.3 minutes" |
| R1-331 | F2 | 12 | 7.3 | minutes | hardware-dependent | quotes-paper | "on a single NVIDIA H100 in 7.3 minutes" |
| R1-332 | F2 | 12 | 26.8 million | co-occurrence patterns | deterministic | quotes-paper | "discovering 26.8 million co-occurrence patterns up to K = 22" |
| R1-333 | F2 | 12 | 22 | K (max) | deterministic | quotes-paper | "discovering 26.8 million co-occurrence patterns up to K = 22" |
| R1-334 | F2 | 12 | ~3 orders of magnitude | transaction count vs prior single-machine GPU FIM | external-fact | quotes-paper | "roughly three orders of magnitude larger than any previously reported single-machine GPU FIM system[SOURCE?]" |
| R1-335 | F2 | 12 | 0.001% | null-model threshold | method-parameter | quotes-paper | "A permutation null model (at the 0.001% threshold)" |
| R1-336 | F2 | 14 | 3 (three) | ways claims outrun support | external-fact | asserts-own | "outrun its statistical and reproducibility support in three specific ways (Major 1-3 below)" |
| R1-337 | F2 | 14 | 22 | K (flagship biological result) | deterministic | quotes-paper | "the flagship biological result (K = 22) rests on the thinnest possible evidence base" |
| R1-338 | F2 | 18 | 21x | speedup (Direct vs SON) | hardware-dependent | quotes-paper | "a convincing demonstration (21x / 95.2% miss-rate) of why the SON approximation is inadequate" |
| R1-339 | F2 | 18 | 95.2% | SON miss rate | deterministic | quotes-paper | "a convincing demonstration (21x / 95.2% miss-rate)" |
| R1-340 | F2 | 22 | 0.001% | threshold validated by null model | method-parameter | quotes-paper | "The null model validates only the 0.001% threshold; the headline K = 22 result lives at 0.00001%" |
| R1-341 | F2 | 22 | 22 | K (headline) | deterministic | quotes-paper | "the headline K = 22 result lives at 0.00001%, where no null model was run" |
| R1-342 | F2 | 22 | 0.00001% | threshold (headline result) | method-parameter | quotes-paper | "the headline K = 22 result lives at 0.00001%, where no null model was run" |
| R1-343 | F2 | 23 | n = 5 | permutations | method-parameter | disputes | "Statistical support rests on n = 5 permutations; the reported 'Z' magnitudes and the p < 0.45 bound cannot carry the weight" |
| R1-344 | F2 | 23 | <0.45 | p bound | deterministic | disputes | "the reported 'Z' magnitudes and the p < 0.45 bound cannot carry the weight the narrative places on them" |
| R1-345 | F2 | 24 | 22 | K (flagship pattern) | deterministic | quotes-paper | "The flagship K = 22 'neuronal antiviral' pattern is a single itemset in 8 proteins" |
| R1-346 | F2 | 24 | 8 | proteins (K=22) | deterministic | quotes-paper | "a single itemset in 8 proteins whose accessions are not listed" |
| R1-347 | F2 | 31 | 0.001% | support (Power threshold, null model) | method-parameter | quotes-paper | "run only at the Power threshold (0.001% support, min_count = 769)" |
| R1-348 | F2 | 31 | 769 | min_count (null model) | method-parameter | quotes-paper | "run only at the Power threshold (0.001% support, min_count = 769)" |
| R1-349 | F2 | 31 | 9 | K (distribution peak) | deterministic | quotes-paper | "the K = 9 distribution peak and the K = 22 ceiling - are obtained at the Opus threshold" |
| R1-350 | F2 | 31 | 22 | K (ceiling) | deterministic | quotes-paper | "the K = 9 distribution peak and the K = 22 ceiling - are obtained at the Opus threshold" |
| R1-351 | F2 | 31 | 0.00001% | support (Opus threshold) | method-parameter | quotes-paper | "obtained at the Opus threshold (0.00001%, min_count = 8)" |
| R1-352 | F2 | 31 | 8 | min_count (Opus) | method-parameter | quotes-paper | "obtained at the Opus threshold (0.00001%, min_count = 8)" |
| R1-353 | F2 | 33 | 15-22 | K range (deep patterns lacking null validation) | deterministic | asserts-own | "whether the K = 15-22 patterns are enriched over a count-preserving null" |
| R1-354 | F2 | 35 | 5 | permutations (suggested Opus-threshold null) | method-parameter | requests | "even 5 permutations at min_count = 8 would establish whether the null produces any deep (K >= 7) patterns" |
| R1-355 | F2 | 35 | 8 | min_count (suggested null run) | method-parameter | requests | "even 5 permutations at min_count = 8 would establish" |
| R1-356 | F2 | 35 | >=7 | K (deep-pattern cutoff) | method-parameter | requests | "whether the null produces any deep (K >= 7) patterns there" |
| R1-357 | F2 | 35 | >=15 | K (concentration analysis) | method-parameter | requests | "how many distinct proteins contribute to all K >= 15 itemsets" |
| R1-358 | F2 | 37 | n = 5 | permutations (heading) | method-parameter | disputes | "Major 2 - Statistical inference from n = 5 permutations is over-leveraged" |
| R1-359 | F2 | 38 | 4 | df (t-statistics) | deterministic | quotes-paper | "correctly labels the enrichment statistics as t-statistics (4 df) rather than large-sample Z-scores" |
| R1-360 | F2 | 38 | <0.45 | p (absence bound) | deterministic | quotes-paper | "corrects the absence bound to p < 0.45 (rule of three, 0/5)" |
| R1-361 | F2 | 38 | 0/5 | null runs with deep patterns | deterministic | quotes-paper | "corrects the absence bound to p < 0.45 (rule of three, 0/5)" |
| R1-362 | F2 | 38 | 5 (five) | permutations | method-parameter | disputes | "five permutations is too few to support the inferential language still used around them" |
| R1-363 | F2 | 41 | <0.45 | p | deterministic | disputes | "A p < 0.45 bound does not reject the null at any conventional level" |
| R1-364 | F2 | 41 | >=7 | K | deterministic | quotes-paper | "'K>=7 patterns are absent from all 5 null runs'" |
| R1-365 | F2 | 41 | 5 | null runs | method-parameter | quotes-paper | "'K>=7 patterns are absent from all 5 null runs'" |
| R1-366 | F2 | 42 | +71,728 | effect size (Z at K=6) | deterministic | quotes-paper | "Reporting a '+71,728' effect size from 5 permutations invites the objection" |
| R1-367 | F2 | 42 | 5 | permutations | method-parameter | quotes-paper | "Reporting a '+71,728' effect size from 5 permutations" |
| R1-368 | F2 | 42 | n = 5 | permutations (sigma estimate) | method-parameter | disputes | "sigma estimated from n = 5 is itself extremely noisy" |
| R1-369 | F2 | 42 | 6 | K (value not reproducing from rounded mu/sigma) | deterministic | disputes | "the K = 6 value does not reproduce cleanly from the rounded mu/sigma shown" |
| R1-370 | F2 | 42 | 4-6 | K range (raw null counts requested) | deterministic | requests | "reporting the raw null counts per permutation for K = 4-6" |
| R1-371 | F2 | 43 | 100+ | permutations needed | method-parameter | quotes-paper | "'100+ permutations would be needed to establish p < 0.01 bounds.'" |
| R1-372 | F2 | 43 | <0.01 | p (target bound) | method-parameter | quotes-paper | "'100+ permutations would be needed to establish p < 0.01 bounds.'" |
| R1-373 | F2 | 43 | ~130 | s per permutation | hardware-dependent | quotes-paper | "Since each permutation is ~130 s, 100 permutations is ~3.6 GPU-hours" |
| R1-374 | F2 | 43 | 100 | permutations | method-parameter | asserts-own | "100 permutations is ~3.6 GPU-hours - entirely feasible on the same single H100" |
| R1-375 | F2 | 43 | ~3.6 | GPU-hours | hardware-dependent | asserts-own | "100 permutations is ~3.6 GPU-hours - entirely feasible on the same single H100" |
| R1-376 | F2 | 43 | 1 (single) | H100 | hardware-dependent | quotes-paper | "entirely feasible on the same single H100" |
| R1-377 | F2 | 45 | 22 | K (heading, flagship result) | deterministic | quotes-paper | "Major 3 - The flagship K = 22 biological result is under-supported and not fully reproducible" |
| R1-378 | F2 | 46 | 22 | K (flagship pattern) | deterministic | quotes-paper | "The K = 22 'neuronal antiviral RNA-helicase' pattern is given prominence" |
| R1-379 | F2 | 46 | 1 (single) | itemset | deterministic | quotes-paper | "As currently supported it is a single itemset shared by 8 proteins" |
| R1-380 | F2 | 46 | 8 | proteins | deterministic | quotes-paper | "As currently supported it is a single itemset shared by 8 proteins" |
| R1-381 | F2 | 46 | 0 | GO parent-child pairs | deterministic | quotes-paper | "the GO true-path check (0 parent-child pairs; one InterPro2GO-derived link, giving 21 independent features)" |
| R1-382 | F2 | 46 | 1 (one) | InterPro2GO-derived link | deterministic | quotes-paper | "(0 parent-child pairs; one InterPro2GO-derived link, giving 21 independent features)" |
| R1-383 | F2 | 46 | 21 | independent features | deterministic | quotes-paper | "(0 parent-child pairs; one InterPro2GO-derived link, giving 21 independent features)" |
| R1-384 | F2 | 47 | 8 | UniProt accessions (not listed) | deterministic | requests | "The 8 UniProt accessions are not listed." |
| R1-385 | F2 | 47 | 8 (eight) | identifiers (supplementary table) | deterministic | requests | "a reviewer will expect the eight identifiers in a supplementary table" |
| R1-386 | F2 | 48 | n = 8 | proteins | deterministic | quotes-paper | "With n = 8 proteins, a single densely (electronically) annotated protein family can generate the entire signature" |
| R1-387 | F2 | 48 | 1 (single) | protein family (could generate signature) | deterministic | asserts-own | "a single densely (electronically) annotated protein family can generate the entire signature" |
| R1-388 | F2 | 50 | 8 | accessions (supplementary table) | deterministic | requests | "add a supplementary table of the 8 accessions with per-term evidence codes" |
| R1-389 | F2 | 50 | 22 | K (pattern to demote) | deterministic | quotes-paper | "demote the K = 22 pattern from 'highlighted discovery' to 'illustrative maximum-depth example'" |
| R1-390 | F2 | 53 | 5.1x | more transactions than prior systems | deterministic | disputes | "the '5.1x more transactions' and 'deeper than any previously reported GPU FIM result' claims are therefore about scale reached" |
| R1-391 | F2 | 53 | 21x | speedup (Direct-GPU vs SON) | hardware-dependent | quotes-paper | "Direct-GPU vs SON at identical support (21x, 95.2% miss rate) - is excellent" |
| R1-392 | F2 | 53 | 95.2% | SON miss rate | deterministic | quotes-paper | "Direct-GPU vs SON at identical support (21x, 95.2% miss rate)" |
| R1-393 | F2 | 53 | 76.9M | transactions | deterministic | asserts-own | "Prior tools plausibly cannot ingest 76.9M transactions" |
| R1-394 | F2 | 53 | 1-15M | transaction slice (suggested subset comparison) | method-parameter | requests | "even a subset comparison (e.g., ET-miner vs GMiner on a 1-15M-transaction slice) would substantiate the efficiency claim" |
| R1-395 | F2 | 56 | 26.8 million | patterns (headline) | deterministic | disputes | "The headline '26.8 million patterns' is the complete frequent-itemset space, which by construction is dominated by subset/superset redundancy" |
| R1-396 | F2 | 56 | 2^K - 2 | frequent sub-itemsets per frequent K-itemset | deterministic | asserts-own | "(every frequent K-itemset implies 2^K - 2 frequent sub-itemsets)" |
| R1-397 | F2 | 62 | 7.3 | minutes (mining only) | hardware-dependent | quotes-paper | "the '7.3 minutes (mining only)' qualifier is good" |
| R1-398 | F2 | 63 | 3 (three) | distinct memory quantities coexisting | external-fact | asserts-own | "Three legitimate but different quantities coexist - ~206 GB ... ~26 GB ... ~5.1 GB ... ~10 GB" (four listed) |
| R1-399 | F2 | 63 | ~206 | GB (naive 1-byte dense) | deterministic | quotes-paper | "~206 GB (naive 1-byte dense)" |
| R1-400 | F2 | 63 | 1 | byte per entry (naive dense) | method-parameter | quotes-paper | "~206 GB (naive 1-byte dense)" |
| R1-401 | F2 | 63 | ~26 | GB (bit-packed dense / on-GPU bitvector, 214M set) | deterministic | quotes-paper | "~26 GB (bit-packed dense / on-GPU bitvector, 214M set)" |
| R1-402 | F2 | 63 | 214M | set (proteins) | deterministic | quotes-paper | "~26 GB (bit-packed dense / on-GPU bitvector, 214M set)" |
| R1-403 | F2 | 63 | ~5.1 | GB (CSR-COO) | deterministic | quotes-paper | "~5.1 GB (CSR-COO)" |
| R1-404 | F2 | 63 | ~10 | GB (bit-packed dense of the 76.9M subset) | deterministic | quotes-paper | "~10 GB (bit-packed dense of the 76.9M subset)" |
| R1-405 | F2 | 63 | 76.9M | subset (proteins) | deterministic | quotes-paper | "~10 GB (bit-packed dense of the 76.9M subset)" |
| R1-406 | F2 | 64 | 2 of 6 | pLDDT bins passing support threshold | deterministic | quotes-paper | "Only 2 of 6 pLDDT bins pass the support threshold" |
| R1-407 | F2 | 64 | 70-90 | pLDDT range (medium confidence bin) | method-parameter | quotes-paper | "the recurring one is 'medium confidence (70-90).'" |
| R1-408 | F2 | 65 | ~11,000 | proteins (intermediate-K count) | deterministic | quotes-paper | "Intermediate-K protein counts ('~11,000', '~10,500', '~16,000') are approximate where the artefacts support near-exact values" |
| R1-409 | F2 | 65 | ~10,500 | proteins (intermediate-K count) | deterministic | quotes-paper | "Intermediate-K protein counts ('~11,000', '~10,500', '~16,000') are approximate" |
| R1-410 | F2 | 65 | ~16,000 | proteins (intermediate-K count) | deterministic | quotes-paper | "Intermediate-K protein counts ('~11,000', '~10,500', '~16,000') are approximate" |
| R1-411 | F2 | 66 | 1 (single) | H100 (all experiments) | hardware-dependent | quotes-paper | "Now honestly framed as a capability ('all experiments used a single H100')" |
| R1-412 | F2 | 68 | 0 | GO parent-child pairs (K=22 set) | deterministic | quotes-paper | "The added quantification (0 parent-child pairs in the K = 22 set) is excellent" |
| R1-413 | F2 | 68 | 22 | K | deterministic | quotes-paper | "(0 parent-child pairs in the K = 22 set)" |
| R1-414 | F2 | 69 | 22 | K (accessions not deposited) | deterministic | quotes-paper | "some source artefacts (full transaction parquet, K = 22 accessions) are not currently deposited" |
| R1-415 | F2 | 75 | <0.45 | p (correction present) | deterministic | quotes-paper | "t-statistic footnote and p < 0.45 correction now present and correct" |
| R1-416 | F2 | 76 | 22 | K (Table 5 reference) | deterministic | quotes-paper | "s3.5 + Table 5 (K = 22): interpretation softened correctly; see Major 3 for accessions + evidence codes." |
| R1-417 | F2 | 77 | 768 | min_count (Power) | method-parameter | quotes-paper | "min_count values (Power = 768, Ultra = 16) reconcile with the mining logs" |
| R1-418 | F2 | 77 | 16 | min_count (Ultra) | method-parameter | quotes-paper | "min_count values (Power = 768, Ultra = 16) reconcile with the mining logs" |
| R1-419 | F2 | 78 | 15M | transactions (GMiner synthetic) | external-fact | quotes-paper | "the GMiner '15M synthetic / 1.7M real' labelling, which is now correct" |
| R1-420 | F2 | 78 | 1.7M | transactions (GMiner real) | external-fact | quotes-paper | "the GMiner '15M synthetic / 1.7M real' labelling, which is now correct" |
| R1-421 | F2 | 79 | 26,849,505 | K-distribution column sum | deterministic | asserts-own | "column sums to 26,849,505 exactly; matches the godmode log" |
| R1-422 | F2 | 85 | 8 | min_count (Opus, requested null) | method-parameter | requests | "Can the permutation null model be run at the Opus threshold (min_count = 8), even at n = 5" |
| R1-423 | F2 | 85 | n = 5 | permutations (requested) | method-parameter | requests | "even at n = 5, to test whether deep (K >= 7) patterns exceed a count-preserving null" |
| R1-424 | F2 | 85 | >=7 | K (deep) | method-parameter | requests | "to test whether deep (K >= 7) patterns exceed a count-preserving null there" |
| R1-425 | F2 | 85 | 0.001% | threshold (null model) | method-parameter | quotes-paper | "why is the count-preserving null at 0.001% argued to generalise to 0.00001%?" |
| R1-426 | F2 | 85 | 0.00001% | threshold (headline) | method-parameter | quotes-paper | "why is the count-preserving null at 0.001% argued to generalise to 0.00001%?" |
| R1-427 | F2 | 86 | ~130 | s per permutation | hardware-dependent | quotes-paper | "Each permutation is ~130 s." |
| R1-428 | F2 | 86 | 100 | permutations (requested) | method-parameter | requests | "Is there an obstacle to running the 100 permutations the manuscript itself identifies as necessary for p < 0.01?" |
| R1-429 | F2 | 86 | <0.01 | p | method-parameter | quotes-paper | "the 100 permutations the manuscript itself identifies as necessary for p < 0.01" |
| R1-430 | F2 | 87 | 8 | K=22 UniProt accessions (requested) | deterministic | requests | "Can the 8 K = 22 UniProt accessions and their per-GO-term evidence codes be provided" |
| R1-431 | F2 | 87 | 22 | K | deterministic | quotes-paper | "Can the 8 K = 22 UniProt accessions and their per-GO-term evidence codes be provided" |
| R1-432 | F2 | 88 | 26.8M | total itemsets | deterministic | quotes-paper | "What are the closed and maximal frequent-itemset counts corresponding to the 26.8M total?" |
| R1-433 | F2 | 90 | 9 | K (distribution peak) | deterministic | quotes-paper | "The K = 9 distribution peak: how sensitive is its position to the support threshold" |
| R1-434 | F2 | 90 | >=2 | features (multi-feature inclusion criterion) | method-parameter | quotes-paper | "the multi-feature-protein inclusion criterion (>= 2 features)" |
| R1-435 | F2 | 96 | n = 5 | permutations | method-parameter | disputes | "n = 5 cannot support the inferential weight placed on it (Major 2)" |
| R1-436 | F2 | 96 | 8 | proteins (anecdote) | deterministic | quotes-paper | "the single most-highlighted pattern is an 8-protein anecdote without listed accessions" |
| R1-437 | F2 | 96 | <=4 | GPU-hours (to fix Majors 1-2) | hardware-dependent | asserts-own | "Two of these are fixable with <= 4 GPU-hours on the machine already used (100 permutations; an Opus-threshold null)" |
| R1-438 | F2 | 96 | 100 | permutations | method-parameter | requests | "fixable with <= 4 GPU-hours on the machine already used (100 permutations; an Opus-threshold null)" |
| R1-439 | F2 | 105 | hundreds of millions | proteins | deterministic | quotes-paper | "'Which combinations of protein features co-occur more than expected across hundreds of millions of proteins?'" |
| R1-440 | F2 | 106 | 22 | K | deterministic | quotes-paper | "Exhaustive FIM to K=22 on 76.9M proteins in 7.3 min; unimodal K-peak at 9; null-model enrichment for K>=4 at 0.001%" |
| R1-441 | F2 | 106 | 76.9M | proteins | deterministic | quotes-paper | "Exhaustive FIM to K=22 on 76.9M proteins in 7.3 min" |
| R1-442 | F2 | 106 | 7.3 | min | hardware-dependent | quotes-paper | "Exhaustive FIM to K=22 on 76.9M proteins in 7.3 min" |
| R1-443 | F2 | 106 | 9 | K (peak) | deterministic | quotes-paper | "unimodal K-peak at 9" |
| R1-444 | F2 | 106 | >=4 | K (null-model enrichment) | deterministic | quotes-paper | "null-model enrichment for K>=4 at 0.001%" |
| R1-445 | F2 | 106 | 0.001% | threshold | method-parameter | quotes-paper | "null-model enrichment for K>=4 at 0.001%" |
| R1-446 | F2 | 114 | 22 | K (accessions not deposited) | deterministic | quotes-paper | "but source transaction data + K=22 accessions not deposited; Major 3, Minor 8" |
| R1-447 | F2 | 114 | n=5 | permutations (null model, checklist) | method-parameter | quotes-paper | "Statistics (null-model n=5; Major 2)" |
| R1-448 | F2 | 115 | n=5 | permutations (no power analysis) | method-parameter | quotes-paper | "sample size / power (no power analysis; n=5 permutations)" |
| R1-449 | F2 | 115 | 42 | seed (Fisher-Yates shuffle) | method-parameter | quotes-paper | "randomization (Fisher-Yates shuffle, seed 42)" |
| R1-450 | F2 | 115 | >=2 | features (inclusion) | method-parameter | quotes-paper | "inclusion/exclusion (>=2 features, min_count thresholds)" |
| R1-451 | F2 | 115 | 2025_01 | UniProt release | external-fact | quotes-paper | "data-collection protocol (UniProt release 2025_01)" |
| R1-452 | F2 | 115 | 3.10 | Python version | software | quotes-paper | "software versions (Python 3.10/CuPy 13.0/NumPy 1.26/CUDA 12.4/Ubuntu 22.04, H100 80GB SXM5)" |
| R1-453 | F2 | 115 | 13.0 | CuPy version | software | quotes-paper | "software versions (Python 3.10/CuPy 13.0/NumPy 1.26/CUDA 12.4/Ubuntu 22.04, H100 80GB SXM5)" |
| R1-454 | F2 | 115 | 1.26 | NumPy version | software | quotes-paper | "software versions (Python 3.10/CuPy 13.0/NumPy 1.26/CUDA 12.4/Ubuntu 22.04, H100 80GB SXM5)" |
| R1-455 | F2 | 115 | 12.4 | CUDA version | software | quotes-paper | "software versions (Python 3.10/CuPy 13.0/NumPy 1.26/CUDA 12.4/Ubuntu 22.04, H100 80GB SXM5)" |
| R1-456 | F2 | 115 | 22.04 | Ubuntu version | software | quotes-paper | "software versions (Python 3.10/CuPy 13.0/NumPy 1.26/CUDA 12.4/Ubuntu 22.04, H100 80GB SXM5)" |
| R1-457 | F2 | 115 | 80 | GB VRAM (H100 SXM5) | hardware-dependent | quotes-paper | "H100 80GB SXM5" |
| R1-458 | F2 | 115 | millions | itemsets tested without FDR | deterministic | asserts-own | "multiple-comparison correction (millions of itemsets tested; no FDR - acknowledged as Limitation 1 but not applied)" |
| R1-459 | F2 | 116 | 4 | figures present on disk | external-fact | asserts-own | "Figures/Tables (4 figures present on disk; tables well-formed)" |
| R1-460 | F2 | 116 | 22 | K (interpretation still prominent) | deterministic | quotes-paper | "Objectivity (K=22 interpretation softened but still prominent; Major 3)" |
| R1-461 | F2 | 117 | 5 | highlighted patterns | deterministic | quotes-paper | "selective reporting (5 highlighted patterns chosen for known validation - disclosed, acceptable)" |
| R1-462 | F2 | 117 | n=5 | permutations ('Z' inappropriate) | method-parameter | quotes-paper | "inappropriate tests (n=5 'Z')" |
| R1-463 | F2 | 118 | 5 | explicit limitations | external-fact | quotes-paper | "Limitations (5 explicit limitations, now including 0-parent-child note)" |
| R1-464 | F2 | 118 | 0 | parent-child pairs | deterministic | quotes-paper | "(5 explicit limitations, now including 0-parent-child note)" |
| R1-465 | F2 | 118 | 22 | K (interpretation exception) | deterministic | quotes-paper | "Interpretation (mostly data-supported; K=22 the exception)" |
| R1-466 | F2 | 119 | 26.8M | 'patterns' (redundant space) | deterministic | disputes | "overstated conclusions (26.8M 'patterns' counts redundant space; Major 5)" |
| R1-467 | F2 | 120 | 2024 | reference currency (through year) | external-fact | asserts-own | "Currency (through 2024)" |
| R1-468 | F2 | 120 | 13 | citation errors corrected this revision | external-fact | asserts-own | "Accuracy (13 citation errors corrected this revision; independently re-verified)" |
| R1-469 | F2 | 120 | 1 (one) | uncited bibitem | external-fact | asserts-own | "one uncited bibitem (miettinen2020; Minor 6)" |
| R1-470 | F2 | 123 | n=5 | permutations (normality assumption) | method-parameter | quotes-paper | "'Z'-framing assumes normality of null counts from n=5; not tested" |
| R1-471 | F2 | 124 | n=5 | permutations (no power analysis) | method-parameter | quotes-paper | "Sample-size/power justification (n=5, no power analysis; Major 2)" |
| R1-472 | F2 | 125 | 5 | permutations (replication) | method-parameter | quotes-paper | "replication (5 permutations)" |
| R1-473 | F2 | 129 | 8 | K=22 accessions not listed | deterministic | quotes-paper | "the 8 K=22 accessions not listed - Major 3" |
| R1-474 | F2 | 129 | 22 | K | deterministic | quotes-paper | "the 8 K=22 accessions not listed - Major 3" |
| R1-475 | F2 | 135 | 5 | permutation spread (could be shown) | method-parameter | requests | "the K-distribution/null comparison could show the 5-permutation spread" |
| R1-476 | F2 | 138 | 2 (two) | orphaned figure files on disk | external-fact | asserts-own | "two orphaned figure files (figures/kdist_shift.pdf, figures/pruning_savings.pdf) remain on disk from the removed expanded run" |
| R1-477 | F2 | 143 | 4.6 | Claude Code Opus version (listed co-author) | software | quotes-paper | "(C. claudya = 'Anthropic, Claude Code Opus 4.6' as a listed author)" |
| R1-478 | F2 | 143 | 1 (one) | author affiliated with AI vendor | external-fact | asserts-own | "one author is affiliated with the AI vendor whose tool co-produced the work - disclosure expected" |
| R1-479 | F2 | 143 | v1 | Zenodo preprint version | software | asserts-own | "a Zenodo v1 exists - the relationship to v1 should be disclosed as a versioned preprint update" |
| R1-480 | F2 | 151 | 5 | major concerns | external-fact | asserts-own | "Major concerns identified & justified (5)" |
| R1-481 | F2 | 151 | 8 | minor issues | external-fact | asserts-own | "Minor issues categorized (8)" |
| R1-482 | F2 | 151 | 3 | missing statements (COI, funding, subjects) | external-fact | asserts-own | "Ethics verified (Stage 6 - surfaced 3 missing statements: COI, funding, subjects)" |
| R1-483 | F2 | 156 | 4.6 | Claude Code Opus version (AI co-author) | software | quotes-paper | "AI-tool listed as a co-author ('Claude Code Opus 4.6')" |
| R1-484 | F2 | 158 | millions | itemsets (no multiple-testing correction) | deterministic | asserts-own | "No multiple-testing correction applied to the millions of itemsets" |
| R1-485 | F2 | 159 | 2 (two) | orphaned figure PDFs | external-fact | asserts-own | "Two orphaned figure PDFs (kdist_shift.pdf, pruning_savings.pdf) from the removed expanded run remain in figures/" |
| R1-486 | F2 | 160 | v1 | Zenodo version (relationship to disclose) | software | asserts-own | "Relationship to the published Zenodo v1 should be disclosed as a versioned-preprint update" |
| R1-487 | F2 | 171 | v1 | Zenodo version (disclosed post-review) | software | asserts-own | "discloses the Zenodo-v1 versioned-preprint relationship" |
| R1-488 | F2 | 173 | 0 (zero) | uncited bibitems remaining | external-fact | asserts-own | "miettinen2020 - now cited (Boolean-matrix representation, s1); zero uncited bibitems remain" |
| R1-489 | F2 | 179 | 100 | permutations (still open) | method-parameter | requests | "Major 2 (100 permutations)" |
| R1-490 | F2 | 179 | 22 | K (accessions + evidence codes still open) | deterministic | requests | "Major 3 (K=22 accessions + evidence codes)" |
| R1-491 | F3 | 9 | 606K | itemsets (35K expansion) | deterministic | asserts-own | "v2 (2026-02-21): Updated with 35K-feature expansion (606K itemsets, K_max=17, null model, SON infeasibility)" |
| R1-492 | F3 | 9 | 17 | K_max (35K expansion) | deterministic | asserts-own | "35K-feature expansion (606K itemsets, K_max=17, null model, SON infeasibility)" |
| R1-493 | F3 | 15 | 1,002 | features (original vocabulary) | deterministic | quotes-paper | "The authors have expanded from 1,002 features to 34,920 features (35x vocabulary expansion)" |
| R1-494 | F3 | 15 | 34,920 | features (expanded vocabulary) | deterministic | asserts-own | "expanded from 1,002 features to 34,920 features (35x vocabulary expansion)" |
| R1-495 | F3 | 15 | 35x | vocabulary expansion factor | deterministic | asserts-own | "expanded from 1,002 features to 34,920 features (35x vocabulary expansion)" |
| R1-496 | F3 | 15 | 3 (three) | new result files | external-fact | asserts-own | "and produced three new result files:" |
| R1-497 | F3 | 17 | 3 (triplicate) | Direct GPU runs | method-parameter | asserts-own | "Direct GPU triplicate (experiment_direct_vs_son_35k_20260221.json)" |
| R1-498 | F3 | 17 | 606,292 | mean itemsets (triplicate) | deterministic | asserts-own | "606,292 mean itemsets (std=28), K_max=17, 654s mean on 4xH200" |
| R1-499 | F3 | 17 | 28 | std itemsets (triplicate) | deterministic | asserts-own | "606,292 mean itemsets (std=28), K_max=17, 654s mean on 4xH200" |
| R1-500 | F3 | 17 | 17 | K_max | deterministic | asserts-own | "606,292 mean itemsets (std=28), K_max=17, 654s mean on 4xH200" |
| R1-501 | F3 | 17 | 654 | s mean runtime | hardware-dependent | asserts-own | "606,292 mean itemsets (std=28), K_max=17, 654s mean on 4xH200" |
| R1-502 | F3 | 17 | 4xH200 | GPUs | hardware-dependent | asserts-own | "654s mean on 4xH200. SON is INFEASIBLE (175 GB bitvec > 143 GB VRAM)" |
| R1-503 | F3 | 17 | 175 | GB (SON bitvector) | deterministic | asserts-own | "SON is INFEASIBLE (175 GB bitvec > 143 GB VRAM)" |
| R1-504 | F3 | 17 | 143 | GB VRAM (H200) | hardware-dependent | asserts-own | "SON is INFEASIBLE (175 GB bitvec > 143 GB VRAM)" |
| R1-505 | F3 | 18 | 2 | permutations (35K null model) | method-parameter | asserts-own | "Null model (experiment_null_model_20260221_003203.json): 2 permutations" |
| R1-506 | F3 | 18 | 2 | K (null peak) | deterministic | asserts-own | "Null peaks at K=2 (59K itemsets), collapses by K=5 (1 itemset)" |
| R1-507 | F3 | 18 | 59K | null itemsets at K=2 | deterministic | asserts-own | "Null peaks at K=2 (59K itemsets), collapses by K=5 (1 itemset)" |
| R1-508 | F3 | 18 | 5 | K (null collapse) | deterministic | asserts-own | "Null peaks at K=2 (59K itemsets), collapses by K=5 (1 itemset)" |
| R1-509 | F3 | 18 | 1 | null itemsets at K=5 | deterministic | asserts-own | "collapses by K=5 (1 itemset)" |
| R1-510 | F3 | 18 | 17 | K (biological max, 35K) | deterministic | asserts-own | "Biological data extends to K=17." |
| R1-511 | F3 | 18 | 5.45x | total enrichment ratio | deterministic | asserts-own | "5.45x total enrichment ratio. Z=723 at K=3, Z=20,585 at K=4." |
| R1-512 | F3 | 18 | 723 | Z at K=3 (35K) | deterministic | asserts-own | "Z=723 at K=3, Z=20,585 at K=4." |
| R1-513 | F3 | 18 | 20,585 | Z at K=4 (35K) | deterministic | asserts-own | "Z=723 at K=3, Z=20,585 at K=4." |
| R1-514 | F3 | 19 | 0.1% | support (wave 3) | method-parameter | asserts-own | "Mining at 0.1% support reaches K=10+ with millions of itemsets per level" |
| R1-515 | F3 | 19 | 10+ | K reached (wave 3) | deterministic | asserts-own | "Mining at 0.1% support reaches K=10+ with millions of itemsets per level" |
| R1-516 | F3 | 19 | millions | itemsets per level (wave 3) | deterministic | asserts-own | "reaches K=10+ with millions of itemsets per level" |
| R1-517 | F3 | 19 | 743 | s elapsed at K=9 (wave 3) | hardware-dependent | asserts-own | "still running after 743s at K=9 (10M itemsets at that level alone)" |
| R1-518 | F3 | 19 | 9 | K (wave 3 progress) | deterministic | asserts-own | "still running after 743s at K=9 (10M itemsets at that level alone)" |
| R1-519 | F3 | 19 | 10M | itemsets at K=9 (wave 3) | deterministic | asserts-own | "still running after 743s at K=9 (10M itemsets at that level alone)" |
| R1-520 | F3 | 27 | 9/10 | severity (M1) | external-fact | asserts-own | "M1. Two permutations is worse than five -- the null model has regressed [Severity: 9/10]" |
| R1-521 | F3 | 29 | 5 | permutations (1K null model) | method-parameter | quotes-paper | "The original 1K-feature null model used 5 permutations." |
| R1-522 | F3 | 29 | 2 | permutations (35K null model) | method-parameter | asserts-own | "The 35K-feature null model uses only 2 permutations" |
| R1-523 | F3 | 29 | 7138484576005690180 | seed (permutation 1) | method-parameter | asserts-own | "(seeds 7138484576005690180 and 4047939128787533792, experiment_null_model_20260221_003203.json line 8)" |
| R1-524 | F3 | 29 | 4047939128787533792 | seed (permutation 2) | method-parameter | asserts-own | "(seeds 7138484576005690180 and 4047939128787533792, experiment_null_model_20260221_003203.json line 8)" |
| R1-525 | F3 | 33 | n=2 | data points (null) | method-parameter | disputes | "Variance from n=2 is meaningless. With 2 data points, you have exactly 1 degree of freedom." |
| R1-526 | F3 | 33 | 1 | degree of freedom | deterministic | asserts-own | "With 2 data points, you have exactly 1 degree of freedom." |
| R1-527 | F3 | 33 | 123.7 | std of null at K=2 | deterministic | asserts-own | "The reported standard deviations (e.g., K=2: std=123.7; K=3: std=51.6; K=4: std=2.8)" |
| R1-528 | F3 | 33 | 51.6 | std of null at K=3 | deterministic | asserts-own | "(e.g., K=2: std=123.7; K=3: std=51.6; K=4: std=2.8)" |
| R1-529 | F3 | 33 | 2.8 | std of null at K=4 | deterministic | asserts-own | "(e.g., K=2: std=123.7; K=3: std=51.6; K=4: std=2.8)" |
| R1-530 | F3 | 33 | 722.91 | Z at K=3 (35K) | deterministic | disputes | "The 'Z-score' at K=3 of 722.91 is computed as (53235 - 15919.5) / 51.6" |
| R1-531 | F3 | 33 | 53235 | biological itemsets at K=3 (35K) | deterministic | asserts-own | "computed as (53235 - 15919.5) / 51.6 -- but the denominator is estimated from 2 observations" |
| R1-532 | F3 | 33 | 15919.5 | null mean itemsets at K=3 (35K) | deterministic | asserts-own | "computed as (53235 - 15919.5) / 51.6" |
| R1-533 | F3 | 33 | 51.6 | null std at K=3 (denominator) | deterministic | asserts-own | "computed as (53235 - 15919.5) / 51.6" |
| R1-534 | F3 | 33 | 2 | observations | method-parameter | asserts-own | "the denominator is estimated from 2 observations" |
| R1-535 | F3 | 33 | 1 | df (t-statistic equivalent) | deterministic | asserts-own | "A Z-statistic from n=2 is a t-statistic with 1 degree of freedom, which has no finite moments" |
| R1-536 | F3 | 35 | 5 | K (both null runs produce 1 itemset) | deterministic | asserts-own | "At K=5, both null runs produced exactly 1 itemset." |
| R1-537 | F3 | 35 | 1 | null itemsets at K=5 (each run) | deterministic | asserts-own | "At K=5, both null runs produced exactly 1 itemset." |
| R1-538 | F3 | 35 | 0.0 | std at K=5 | deterministic | asserts-own | "The standard deviation is 0.0 from n=2 identical values." |
| R1-539 | F3 | 35 | inf | z_score at K=5 (string) | deterministic | disputes | "The z_score is reported as 'inf' (a string, not a float -- line 104). This is a division-by-zero artifact" |
| R1-540 | F3 | 35 | 10 or 100 | permutations (hypothetical) | method-parameter | asserts-own | "With 10 or 100 permutations, the K=5 null count might range from 0 to 5" |
| R1-541 | F3 | 35 | 0 to 5 | null count range at K=5 (hypothetical) | deterministic | asserts-own | "the K=5 null count might range from 0 to 5, producing a finite and meaningful Z-score" |
| R1-542 | F3 | 35 | n=2 | permutations (identical values) | method-parameter | asserts-own | "The standard deviation is 0.0 from n=2 identical values." |
| R1-543 | F3 | 37 | 1,090 | min_count (35K null model) | method-parameter | asserts-own | "The 35K null uses min_count=1,090 (approximately 0.001% of 109M transactions, line 6-7)" |
| R1-544 | F3 | 37 | 0.001% | support (35K null) | method-parameter | asserts-own | "min_count=1,090 (approximately 0.001% of 109M transactions" |
| R1-545 | F3 | 37 | 109M | transactions (35K dataset) | deterministic | asserts-own | "(approximately 0.001% of 109M transactions, line 6-7)" |
| R1-546 | F3 | 37 | 1,093 | min_count (35K mining campaign) | method-parameter | asserts-own | "The 35K mining campaign uses min_count=1,093 (0.001%, line 9 of the direct_vs_son file)" |
| R1-547 | F3 | 37 | 0.001% | support (35K mining) | method-parameter | asserts-own | "The 35K mining campaign uses min_count=1,093 (0.001%" |
| R1-548 | F3 | 37 | 8 | min_count (paper Opus threshold) | method-parameter | quotes-paper | "neither matches the paper's Opus threshold (min_count=8)" |
| R1-549 | F3 | 39 | 5 | perms (1K null) | method-parameter | quotes-paper | "In the 1K-feature null model (5 perms), K=2 was depleted (Z=-987) and K=3 was depleted (Z=-143)" |
| R1-550 | F3 | 39 | -987 | Z at K=2 (1K) | deterministic | quotes-paper | "K=2 was depleted (Z=-987) and K=3 was depleted (Z=-143)" |
| R1-551 | F3 | 39 | -143 | Z at K=3 (1K) | deterministic | quotes-paper | "K=2 was depleted (Z=-987) and K=3 was depleted (Z=-143)" |
| R1-552 | F3 | 39 | 2 | perms (35K null) | method-parameter | asserts-own | "In the 35K-feature null model (2 perms), K=2 is again depleted (Z=-102)" |
| R1-553 | F3 | 39 | -102 | Z at K=2 (35K) | deterministic | asserts-own | "K=2 is again depleted (Z=-102) but K=3 is now massively enriched (Z=+723)" |
| R1-554 | F3 | 39 | +723 | Z at K=3 (35K) | deterministic | asserts-own | "K=3 is now massively enriched (Z=+723)" |
| R1-555 | F3 | 43 | 100+ | permutations (requested, 35K) | method-parameter | requests | "Run 100+ permutations at min_count=1,093 on the 35K dataset." |
| R1-556 | F3 | 43 | 1,093 | min_count (requested run) | method-parameter | requests | "Run 100+ permutations at min_count=1,093 on the 35K dataset." |
| R1-557 | F3 | 43 | ~655 | s per permutation (35K, 4xH200) | hardware-dependent | asserts-own | "At ~655s per permutation on 4xH200, 100 permutations = ~18 hours, 200 = ~36 hours." |
| R1-558 | F3 | 43 | 4xH200 | GPUs | hardware-dependent | asserts-own | "At ~655s per permutation on 4xH200" |
| R1-559 | F3 | 43 | 100 | permutations | method-parameter | asserts-own | "100 permutations = ~18 hours, 200 = ~36 hours" |
| R1-560 | F3 | 43 | ~18 | hours (100 perms, 35K) | hardware-dependent | asserts-own | "100 permutations = ~18 hours, 200 = ~36 hours" |
| R1-561 | F3 | 43 | 200 | permutations | method-parameter | asserts-own | "100 permutations = ~18 hours, 200 = ~36 hours" |
| R1-562 | F3 | 43 | ~36 | hours (200 perms, 35K) | hardware-dependent | asserts-own | "100 permutations = ~18 hours, 200 = ~36 hours" |
| R1-563 | F3 | 43 | 5 | perms (1K null) | method-parameter | quotes-paper | "The 1K-feature null model (5 perms, ~86s/perm) should also be expanded to 100+ perms." |
| R1-564 | F3 | 43 | ~86 | s per permutation (1K) | hardware-dependent | asserts-own | "The 1K-feature null model (5 perms, ~86s/perm) should also be expanded to 100+ perms." |
| R1-565 | F3 | 43 | 100+ | perms (requested, 1K) | method-parameter | requests | "should also be expanded to 100+ perms" |
| R1-566 | F3 | 45 | 7/10 | severity (M2) | external-fact | asserts-own | "M2. GO true-path inflation: partially addressed by the K-drop but still unquantified [Severity: 7/10]" |
| R1-567 | F3 | 47 | 22 | K_max (1K features) | deterministic | quotes-paper | "The K_max drop from 22 (1K features) to 17 (35K features) is striking." |
| R1-568 | F3 | 47 | 17 | K_max (35K features) | deterministic | asserts-own | "The K_max drop from 22 (1K features) to 17 (35K features) is striking." |
| R1-569 | F3 | 51 | 30% | redundant GO pairs (hypothetical, 1K) | deterministic | asserts-own | "If the 1K vocabulary has 30% redundant pairs and the 35K has 5%, this explains the K-drop" |
| R1-570 | F3 | 51 | 5% | redundant GO pairs (hypothetical, 35K) | deterministic | asserts-own | "If the 1K vocabulary has 30% redundant pairs and the 35K has 5%" |
| R1-571 | F3 | 53 | 22 | K (itemset with GO inflation) | deterministic | quotes-paper | "The K=22 itemset from the 1K-feature analysis contains verified GO hierarchy inflation" |
| R1-572 | F3 | 53 | 19-20 | corrected independent K | deterministic | disputes | "The corrected independent K is approximately 19-20." |
| R1-573 | F3 | 55 | 17 | K (35K itemset needing scrutiny) | deterministic | asserts-own | "The 35K K=17 itemset needs the same scrutiny." |
| R1-574 | F3 | 55 | 17 | co-occurring features (requested listing) | deterministic | requests | "What are the 17 co-occurring features in the K=17 pattern?" |
| R1-575 | F3 | 57 | 5/10 | severity (M3, current) | external-fact | asserts-own | "M3. SON comparison ... [Severity: 5/10, improved from 8/10]" |
| R1-576 | F3 | 57 | 8/10 | severity (M3, v1) | external-fact | asserts-own | "[Severity: 5/10, improved from 8/10]" |
| R1-577 | F3 | 61 | 40M | chunk size (SON, transactions) | method-parameter | asserts-own | "'40M chunk x 35K features = 175 GB bitvec exceeds 143 GB H200 VRAM'" |
| R1-578 | F3 | 61 | 35K | features (bitvec arithmetic) | deterministic | asserts-own | "'40M chunk x 35K features = 175 GB bitvec exceeds 143 GB H200 VRAM'" |
| R1-579 | F3 | 61 | 175 | GB (SON bitvec) | deterministic | asserts-own | "'40M chunk x 35K features = 175 GB bitvec exceeds 143 GB H200 VRAM'" |
| R1-580 | F3 | 61 | 143 | GB (H200 VRAM) | hardware-dependent | asserts-own | "'40M chunk x 35K features = 175 GB bitvec exceeds 143 GB H200 VRAM' (son_status: FAILED_OOM_HUNG)" |
| R1-581 | F3 | 63 | 143 | GB (H200 VRAM) | hardware-dependent | asserts-own | "the highest-end GPU available (NVIDIA H200 at 143 GB)" |
| R1-582 | F3 | 63 | <33M | chunk_size required to fit | deterministic | asserts-own | "you would need chunk_size < 33M, which at 109M transactions means 4+ chunks" |
| R1-583 | F3 | 63 | 109M | transactions | deterministic | asserts-own | "which at 109M transactions means 4+ chunks with proportionally worse miss rates" |
| R1-584 | F3 | 63 | 4+ | chunks | deterministic | asserts-own | "at 109M transactions means 4+ chunks with proportionally worse miss rates" |
| R1-585 | F3 | 67 | 20M | chunk_size (feasible SON) | method-parameter | asserts-own | "chunk_size=20M x 35K features = ~87 GB, fitting on an H200. This would create 6 chunks" |
| R1-586 | F3 | 67 | ~87 | GB (bitvec at 20M chunk) | deterministic | asserts-own | "chunk_size=20M x 35K features = ~87 GB, fitting on an H200" |
| R1-587 | F3 | 67 | 6 | chunks | deterministic | asserts-own | "This would create 6 chunks with terrible miss rates" |
| R1-588 | F3 | 69 | 95.2% | SON miss rate (1K) | deterministic | quotes-paper | "The SON infeasibility at 35K is a STRONGER argument than the 1K 95.2% miss rate." |
| R1-589 | F3 | 73 | 3 (Three) | Direct GPU runs | method-parameter | asserts-own | "Three Direct GPU runs with mean=606,292, std=28, CV=0.0046% demonstrates excellent reproducibility" |
| R1-590 | F3 | 73 | 606,292 | mean itemsets | deterministic | asserts-own | "Three Direct GPU runs with mean=606,292, std=28, CV=0.0046%" |
| R1-591 | F3 | 73 | 28 | std itemsets | deterministic | asserts-own | "Three Direct GPU runs with mean=606,292, std=28, CV=0.0046%" |
| R1-592 | F3 | 73 | 0.0046% | CV of itemset count | deterministic | asserts-own | "Three Direct GPU runs with mean=606,292, std=28, CV=0.0046%" |
| R1-593 | F3 | 73 | 17, 16, 17 | K_max across runs | deterministic | asserts-own | "The slight variation in K_max (17, 16, 17 across runs) at K=17 (with 2, 0, 1 itemsets respectively)" |
| R1-594 | F3 | 73 | 2, 0, 1 | itemsets at K=17 across runs | deterministic | asserts-own | "at K=17 (with 2, 0, 1 itemsets respectively) suggests the boundary itemsets are at the noise floor" |
| R1-595 | F3 | 75 | 8/10 | severity (M4) | external-fact | asserts-own | "M4. The K-max drop (K=22 -> K=17) demands explanation [Severity: 8/10]" |
| R1-596 | F3 | 75 | 22 -> 17 | K-max drop | deterministic | asserts-own | "M4. The K-max drop (K=22 -> K=17) demands explanation" |
| R1-597 | F3 | 77 | 1,002 | features (1K) | deterministic | quotes-paper | "When expanding from 1,002 to 34,920 features:" |
| R1-598 | F3 | 77 | 34,920 | features (35K) | deterministic | asserts-own | "When expanding from 1,002 to 34,920 features:" |
| R1-599 | F3 | 78 | 22 | K_max (1K) | deterministic | quotes-paper | "K_max dropped from 22 to 17" |
| R1-600 | F3 | 78 | 17 | K_max (35K) | deterministic | asserts-own | "K_max dropped from 22 to 17" |
| R1-601 | F3 | 79 | 9 | K (peak, 1K) | deterministic | quotes-paper | "The K-distribution peak shifted from K=9 (1K) to K=7 (35K)" |
| R1-602 | F3 | 79 | 7 | K (peak, 35K) | deterministic | asserts-own | "The K-distribution peak shifted from K=9 (1K) to K=7 (35K)" |
| R1-603 | F3 | 80 | 26.8M | itemsets (1K, min_count=8) | deterministic | quotes-paper | "Total itemsets dropped from 26.8M (at min_count=8) to 606K (at min_count=1,093)" |
| R1-604 | F3 | 80 | 8 | min_count (1K Opus) | method-parameter | quotes-paper | "Total itemsets dropped from 26.8M (at min_count=8) to 606K (at min_count=1,093)" |
| R1-605 | F3 | 80 | 606K | itemsets (35K, min_count=1,093) | deterministic | asserts-own | "Total itemsets dropped from 26.8M (at min_count=8) to 606K (at min_count=1,093)" |
| R1-606 | F3 | 80 | 1,093 | min_count (35K) | method-parameter | asserts-own | "to 606K (at min_count=1,093)" |
| R1-607 | F3 | 82 | 1,093 vs 8 | min_count (support threshold difference) | method-parameter | asserts-own | "The third point is explained by the higher support threshold (1,093 vs 8)" |
| R1-608 | F3 | 84 | 1,093 | min_count (35K experiment) | method-parameter | asserts-own | "The 35K experiment uses min_count=1,093 while the 1K Opus run uses min_count=8." |
| R1-609 | F3 | 84 | 8 | min_count (1K Opus run) | method-parameter | quotes-paper | "The 35K experiment uses min_count=1,093 while the 1K Opus run uses min_count=8." |
| R1-610 | F3 | 84 | 22 | K (pattern supported by 8 proteins) | deterministic | quotes-paper | "The K=22 pattern in the 1K run was supported by exactly 8 proteins -- it would be invisible at min_count=1,093" |
| R1-611 | F3 | 84 | 8 | proteins supporting K=22 | deterministic | quotes-paper | "The K=22 pattern in the 1K run was supported by exactly 8 proteins" |
| R1-612 | F3 | 86 | 109M | transactions (assumed same) | deterministic | asserts-own | "each feature is present in fewer proteins on average (assuming the same 109M transactions)" |
| R1-613 | F3 | 88 | millions | proteins carrying 'cytoplasm' term | deterministic | asserts-own | "removes a term that appeared in millions of proteins, reducing deep co-occurrence potential" |
| R1-614 | F3 | 90 | 4 | GPUs (row-splitting) | hardware-dependent | asserts-own | "The 35K runs use 4-GPU row-splitting with local_min_count=274 (line 2 of power_test_v1_full.log)" |
| R1-615 | F3 | 90 | 274 | local_min_count (power test) | method-parameter | asserts-own | "4-GPU row-splitting with local_min_count=274 (line 2 of power_test_v1_full.log)" |
| R1-616 | F3 | 90 | 14,800 | locally frequent K=2 pairs (union across GPUs) | deterministic | asserts-own | "'K=2: 14,800 locally frequent (union across GPUs), 8,554 frequent'" |
| R1-617 | F3 | 90 | 8,554 | globally frequent K=2 pairs | deterministic | asserts-own | "'K=2: 14,800 locally frequent (union across GPUs), 8,554 frequent'" |
| R1-618 | F3 | 90 | 42% | locally-frequent K=2 pairs failing global recount | deterministic | asserts-own | "shows that 42% of locally-frequent K=2 pairs fail the global recount" |
| R1-619 | F3 | 92 | 8 | min_count (controlled experiment requested) | method-parameter | requests | "A controlled experiment would run the 35K features at min_count=8 on a single GPU (if VRAM allows)" |
| R1-620 | F3 | 92 | 1 (single) | GPU (controlled experiment) | hardware-dependent | requests | "run the 35K features at min_count=8 on a single GPU (if VRAM allows)" |
| R1-621 | F3 | 94 | 7/10 | severity (M5) | external-fact | asserts-own | "M5. 8 vs 3 K=22 proteins -- still unresolved [Severity: 7/10]" |
| R1-622 | F3 | 94 | 8 vs 3 | K=22 proteins (support count vs identified) | deterministic | disputes | "M5. 8 vs 3 K=22 proteins -- still unresolved" |
| R1-623 | F3 | 96 | 8 | proteins (mining result support count) | deterministic | quotes-paper | "The discrepancy between 8 proteins (mining result support count) and 3 proteins identified (analysis script fallback heuristic)" |
| R1-624 | F3 | 96 | 3 | proteins identified (analysis script fallback heuristic) | deterministic | disputes | "and 3 proteins identified (analysis script fallback heuristic) remains unresolved" |
| R1-625 | F3 | 96 | 8 | UniProt accessions (must be named) | deterministic | requests | "All 8 UniProt accessions must be named and verified." |
| R1-626 | F3 | 98 | 6/10 | severity (M6) | external-fact | asserts-own | "M6. Wave 3 partial results hint at massive scale but are incomplete [Severity: 6/10]" |
| R1-627 | F3 | 100 | 0.1% | support (wave 3) | method-parameter | asserts-own | "mining at 0.1% support (min_count=109,225) reaching K=10 with the following per-level counts" |
| R1-628 | F3 | 100 | 109,225 | min_count (wave 3, 0.1%) | method-parameter | asserts-own | "mining at 0.1% support (min_count=109,225) reaching K=10" |
| R1-629 | F3 | 100 | 10 | K reached (wave 3) | deterministic | asserts-own | "mining at 0.1% support (min_count=109,225) reaching K=10" |
| R1-630 | F3 | 104 | 1,164 | itemsets at K=1 (wave 3) | deterministic | asserts-own | table row "1 / 1,164 / 0.3s" |
| R1-631 | F3 | 104 | 0.3 | s cumulative at K=1 (wave 3) | hardware-dependent | asserts-own | table row "1 / 1,164 / 0.3s" |
| R1-632 | F3 | 105 | 8,554 | itemsets at K=2 (wave 3) | deterministic | asserts-own | table row "2 / 8,554 / 1.2s" |
| R1-633 | F3 | 105 | 1.2 | s cumulative at K=2 (wave 3) | hardware-dependent | asserts-own | table row "2 / 8,554 / 1.2s" |
| R1-634 | F3 | 106 | 28,804 | itemsets at K=3 (wave 3) | deterministic | asserts-own | table row "3 / 28,804 / 2.0s" |
| R1-635 | F3 | 106 | 2.0 | s cumulative at K=3 (wave 3) | hardware-dependent | asserts-own | table row "3 / 28,804 / 2.0s" |
| R1-636 | F3 | 107 | 88,561 | itemsets at K=4 (wave 3) | deterministic | asserts-own | table row "4 / 88,561 / 5.3s" |
| R1-637 | F3 | 107 | 5.3 | s cumulative at K=4 (wave 3) | hardware-dependent | asserts-own | table row "4 / 88,561 / 5.3s" |
| R1-638 | F3 | 108 | 280,784 | itemsets at K=5 (wave 3) | deterministic | asserts-own | table row "5 / 280,784 / 17.5s" |
| R1-639 | F3 | 108 | 17.5 | s cumulative at K=5 (wave 3) | hardware-dependent | asserts-own | table row "5 / 280,784 / 17.5s" |
| R1-640 | F3 | 109 | 835,466 | itemsets at K=6 (wave 3) | deterministic | asserts-own | table row "6 / 835,466 / 52.1s" |
| R1-641 | F3 | 109 | 52.1 | s cumulative at K=6 (wave 3) | hardware-dependent | asserts-own | table row "6 / 835,466 / 52.1s" |
| R1-642 | F3 | 110 | 2,207,022 | itemsets at K=7 (wave 3) | deterministic | asserts-own | table row "7 / 2,207,022 / 145.3s" |
| R1-643 | F3 | 110 | 145.3 | s cumulative at K=7 (wave 3) | hardware-dependent | asserts-own | table row "7 / 2,207,022 / 145.3s" |
| R1-644 | F3 | 111 | 5,063,845 | itemsets at K=8 (wave 3) | deterministic | asserts-own | table row "8 / 5,063,845 / 348.5s" |
| R1-645 | F3 | 111 | 348.5 | s cumulative at K=8 (wave 3) | hardware-dependent | asserts-own | table row "8 / 5,063,845 / 348.5s" |
| R1-646 | F3 | 112 | 10,041,611 | itemsets at K=9 (wave 3) | deterministic | asserts-own | table row "9 / 10,041,611 / 743.4s" |
| R1-647 | F3 | 112 | 743.4 | s cumulative at K=9 (wave 3) | hardware-dependent | asserts-own | table row "9 / 10,041,611 / 743.4s" |
| R1-648 | F3 | 113 | 12.5M | locally frequent candidates at K=10 (wave 3) | deterministic | asserts-own | table row "10 / ??? (12.5M locally frequent) / >743s, still running" |
| R1-649 | F3 | 113 | >743 | s at K=10 (still running) | hardware-dependent | asserts-own | table row "10 / ??? (12.5M locally frequent) / >743s, still running" |
| R1-650 | F3 | 115 | 30x | more itemsets per K-level (0.1% vs 0.001% run) | deterministic | asserts-own | "This is 30x more itemsets per K-level than the 0.001% run (e.g., K=7: 2.2M vs 64K)" |
| R1-651 | F3 | 115 | 0.001% | support (comparison run) | method-parameter | asserts-own | "30x more itemsets per K-level than the 0.001% run" |
| R1-652 | F3 | 115 | 2.2M | itemsets at K=7 (0.1% run) | deterministic | asserts-own | "(e.g., K=7: 2.2M vs 64K)" |
| R1-653 | F3 | 115 | 64K | itemsets at K=7 (0.001% run) | deterministic | asserts-own | "(e.g., K=7: 2.2M vs 64K)" |
| R1-654 | F3 | 115 | 10 | K (distribution still growing) | deterministic | asserts-own | "the distribution is still growing at K=10" |
| R1-655 | F3 | 117 | hundreds of millions | itemsets (projected if wave 3 completes) | deterministic | asserts-own | "If this run completes, it could produce hundreds of millions of itemsets at 35K features." |
| R1-656 | F3 | 117 | 12+ | minutes at K=9 | hardware-dependent | asserts-own | "the runtime (already 12+ minutes at K=9) suggests the full campaign may take hours per threshold" |
| R1-657 | F3 | 119 | 1,093 | min_count (power test) | method-parameter | asserts-own | "show mining at min_count=1,093 with K=2 taking 586s alone (vs 1.2s at min_count=109,225)" |
| R1-658 | F3 | 119 | 586 | s (K=2 level at min_count=1,093) | hardware-dependent | asserts-own | "K=2 taking 586s alone (vs 1.2s at min_count=109,225)" |
| R1-659 | F3 | 119 | 1.2 | s (K=2 level at min_count=109,225) | hardware-dependent | asserts-own | "K=2 taking 586s alone (vs 1.2s at min_count=109,225)" |
| R1-660 | F3 | 119 | 109,225 | min_count | method-parameter | asserts-own | "(vs 1.2s at min_count=109,225)" |
| R1-661 | F3 | 119 | 902K | frequent pairs (min_count=1,093) | deterministic | asserts-own | "the lower threshold admits 902K frequent pairs vs 8.5K, creating a vastly larger candidate space at K>=3" |
| R1-662 | F3 | 119 | 8.5K | frequent pairs (min_count=109,225) | deterministic | asserts-own | "admits 902K frequent pairs vs 8.5K" |
| R1-663 | F3 | 119 | >=3 | K (candidate space) | deterministic | asserts-own | "creating a vastly larger candidate space at K>=3" |
| R1-664 | F3 | 119 | 4 | K reached (power test) | deterministic | asserts-own | "The power test reached K=4 with 17.4M itemsets before apparently stalling." |
| R1-665 | F3 | 119 | 17.4M | itemsets at K=4 (power test) | deterministic | asserts-own | "The power test reached K=4 with 17.4M itemsets before apparently stalling." |
| R1-666 | F3 | 121 | 606K | itemsets at 0.001% support | deterministic | asserts-own | "Presenting 606K itemsets at 0.001% support when the 0.1% run alone produces 10M+ itemsets at K=9 undersells" |
| R1-667 | F3 | 121 | 0.001% | support | method-parameter | asserts-own | "Presenting 606K itemsets at 0.001% support" |
| R1-668 | F3 | 121 | 0.1% | support | method-parameter | asserts-own | "when the 0.1% run alone produces 10M+ itemsets at K=9" |
| R1-669 | F3 | 121 | 10M+ | itemsets at K=9 (0.1% run) | deterministic | asserts-own | "the 0.1% run alone produces 10M+ itemsets at K=9" |
| R1-670 | F3 | 121 | 9 | K | deterministic | asserts-own | "the 0.1% run alone produces 10M+ itemsets at K=9" |
| R1-671 | F3 | 129 | 2 | permutations (35K) | method-parameter | asserts-own | "The 35K null model uses 2 permutations instead of the 1K model's 5." |
| R1-672 | F3 | 129 | 5 | permutations (1K) | method-parameter | quotes-paper | "The 35K null model uses 2 permutations instead of the 1K model's 5." |
| R1-673 | F3 | 133 | 22 -> 17 | K drop | deterministic | asserts-own | "The K=22 -> K=17 drop suggests less GO inflation at 35K features, but this is inferential" |
| R1-674 | F3 | 139 | 26.8M | itemsets (provenance gap) | deterministic | quotes-paper | "M4-v1. 26.8M itemsets data provenance gap [RETAINED, unchanged]" |
| R1-675 | F3 | 141 | 7.3 | minute claim (mining vs total pipeline) | hardware-dependent | disputes | "The 7.3-minute claim (mining time vs total pipeline time) is still ambiguous." |
| R1-676 | F3 | 143 | 8 vs 3 | K=22 proteins (heading, retained from v1) | deterministic | disputes | "M5-v1. 8 vs 3 K=22 proteins [RETAINED as M5 above]" |
| R1-677 | F3 | 143 | 22 | K (heading) | deterministic | quotes-paper | "M5-v1. 8 vs 3 K=22 proteins [RETAINED as M5 above]" |
| R1-678 | F3 | 151 | 3/10 | severity (m1) | external-fact | asserts-own | "m1. Triplicate reproducibility reveals boundary instability [Severity: 3/10]" |
| R1-679 | F3 | 153 | 606,319 | itemsets (run 1) | deterministic | asserts-own | "minor variation in total itemsets (606,319 / 606,293 / 606,263) and K_max (17 / 16 / 17)" |
| R1-680 | F3 | 153 | 606,293 | itemsets (run 2) | deterministic | asserts-own | "minor variation in total itemsets (606,319 / 606,293 / 606,263)" |
| R1-681 | F3 | 153 | 606,263 | itemsets (run 3) | deterministic | asserts-own | "minor variation in total itemsets (606,319 / 606,293 / 606,263)" |
| R1-682 | F3 | 153 | 17 / 16 / 17 | K_max per run | deterministic | asserts-own | "and K_max (17 / 16 / 17)" |
| R1-683 | F3 | 153 | 2, 0, 1 | itemsets at K=17 per run | deterministic | asserts-own | "The K=17 level had 2, 0, and 1 itemsets across runs." |
| R1-684 | F3 | 153 | 152 | itemsets at K=16 (stable across runs) | deterministic | asserts-own | "The K=16 level is stable at 152 across all runs." |
| R1-685 | F3 | 155 | 16-17 | K_max (recommended reporting) | deterministic | requests | "Report K_max as '16-17' with a note that K=17 patterns are at the support boundary." |
| R1-686 | F3 | 157 | 5/10 | severity (m2) | external-fact | asserts-own | "m2. The 35K feature vocabulary composition is undocumented [Severity: 5/10]" |
| R1-687 | F3 | 159 | 247 | Pfam items (Table 1) | deterministic | quotes-paper | "The paper documents the 1K feature vocabulary in Table 1 (247 Pfam, 302 GO:MF, 289 GO:BP, 161 GO:CC, 3 pLDDT)" |
| R1-688 | F3 | 159 | 302 | GO:MF items (Table 1) | deterministic | quotes-paper | "Table 1 (247 Pfam, 302 GO:MF, 289 GO:BP, 161 GO:CC, 3 pLDDT)" |
| R1-689 | F3 | 159 | 289 | GO:BP items (Table 1) | deterministic | quotes-paper | "Table 1 (247 Pfam, 302 GO:MF, 289 GO:BP, 161 GO:CC, 3 pLDDT)" |
| R1-690 | F3 | 159 | 161 | GO:CC items (Table 1) | deterministic | quotes-paper | "Table 1 (247 Pfam, 302 GO:MF, 289 GO:BP, 161 GO:CC, 3 pLDDT)" |
| R1-691 | F3 | 159 | 3 | pLDDT items (Table 1) | deterministic | quotes-paper | "Table 1 (247 Pfam, 302 GO:MF, 289 GO:BP, 161 GO:CC, 3 pLDDT)" |
| R1-692 | F3 | 159 | 34,920 | features (35K vocabulary) | deterministic | asserts-own | "What are the 34,920 features?" |
| R1-693 | F3 | 159 | 34,920 | frequent items at K=1 (35K logs) | deterministic | asserts-own | "The logs show K=1: 34,920 frequent items, suggesting all 34,920 are above the min_count threshold." |
| R1-694 | F3 | 162 | 1,093-2,000 | proteins per feature (hypothetical) | deterministic | asserts-own | "If most new features appear in only 1,093-2,000 proteins, they contribute to K=2 pairs but not deep patterns." |
| R1-695 | F3 | 163 | 1,002 | original features | deterministic | quotes-paper | "Does the 35K vocabulary include all 1,002 original features?" |
| R1-696 | F3 | 165 | 5/10 | severity (m3) | external-fact | asserts-own | "m3. Multi-GPU row-splitting introduces a new approximation [Severity: 5/10]" |
| R1-697 | F3 | 167 | 4 | GPUs (row-splitting) | hardware-dependent | asserts-own | "The 35K results use 4-GPU row-splitting (wave3_base_partial.log line 18: local_min_count=27,307 for global min_count=109,225)" |
| R1-698 | F3 | 167 | 27,307 | local_min_count (wave 3) | method-parameter | asserts-own | "local_min_count=27,307 for global min_count=109,225" |
| R1-699 | F3 | 167 | 109,225 | global min_count (wave 3) | method-parameter | asserts-own | "local_min_count=27,307 for global min_count=109,225" |
| R1-700 | F3 | 167 | 14,800 | locally frequent K=2 (union) | deterministic | asserts-own | "(e.g., line 25: '14,800 locally frequent (union across GPUs), 8,554 frequent')" |
| R1-701 | F3 | 167 | 8,554 | globally frequent K=2 | deterministic | asserts-own | "'14,800 locally frequent (union across GPUs), 8,554 frequent'" |
| R1-702 | F3 | 167 | 42% | candidates pruned by global recount | deterministic | asserts-own | "requiring a global recount that prunes 42% of candidates" |
| R1-703 | F3 | 169 | 1/4 | fraction of transactions per GPU | method-parameter | asserts-own | "each GPU sees 1/4 of the transactions and applies a reduced local threshold" |
| R1-704 | F3 | 169 | 95.2% | SON miss rate (1K analysis) | deterministic | quotes-paper | "The same cascade effect that caused SON to miss 95.2% of patterns in the 1K analysis could be active here" |
| R1-705 | F3 | 169 | global/4 | local_min_count formula | method-parameter | asserts-own | "(local_min_count = global/4 = exact proportional split)" |
| R1-706 | F3 | 173 | 4/10 | severity (m4) | external-fact | asserts-own | "m4. Feature vocabulary expansion justification missing [Severity: 4/10]" |
| R1-707 | F3 | 175 | 1,002 | features | deterministic | quotes-paper | "The jump from 1,002 to 34,920 is not a round number and is not motivated in any documentation." |
| R1-708 | F3 | 175 | 34,920 | features | deterministic | asserts-own | "The jump from 1,002 to 34,920 is not a round number" |
| R1-709 | F3 | 177 | ~60% | of all proteins annotated GO:0005515 protein binding | external-fact | asserts-own | "GO:0005515 protein binding which annotates ~60% of all proteins and creates millions of trivial co-occurrences" |
| R1-710 | F3 | 177 | millions | trivial co-occurrences | deterministic | asserts-own | "annotates ~60% of all proteins and creates millions of trivial co-occurrences" |
| R1-711 | F3 | 179 | <2x min_count | median feature frequency (hypothetical) | deterministic | asserts-own | "If median frequency < 2x min_count, most features are barely above the noise floor." |
| R1-712 | F3 | 181 | 21.4x | speedup (1K SON comparison) | hardware-dependent | quotes-paper | "m5. The '21.4x speedup' from v1 is now secondary [Severity: 2/10, reduced from 4/10]" |
| R1-713 | F3 | 181 | 2/10 | severity (m5, current) | external-fact | asserts-own | "[Severity: 2/10, reduced from 4/10]" |
| R1-714 | F3 | 181 | 4/10 | severity (m5, v1) | external-fact | asserts-own | "[Severity: 2/10, reduced from 4/10]" |
| R1-715 | F3 | 183 | 21.4x | speedup | hardware-dependent | quotes-paper | "The 1K SON comparison (21.4x speedup, 95.2% miss rate) is now secondary to the 35K result" |
| R1-716 | F3 | 183 | 95.2% | SON miss rate | deterministic | quotes-paper | "The 1K SON comparison (21.4x speedup, 95.2% miss rate)" |
| R1-717 | F3 | 185 | 3/10 | severity (m6) | external-fact | asserts-own | "m6. The 'bell curve' terminology [RETAINED from v1, severity 3/10]" |
| R1-718 | F3 | 187 | 7 | K (35K distribution peak) | deterministic | asserts-own | "The 35K K-distribution peaks at K=7 (64,400 itemsets) and has a different shape than the 1K distribution (peak at K=9)" |
| R1-719 | F3 | 187 | 64,400 | itemsets at K=7 (35K) | deterministic | asserts-own | "The 35K K-distribution peaks at K=7 (64,400 itemsets)" |
| R1-720 | F3 | 187 | 9 | K (1K distribution peak) | deterministic | quotes-paper | "a different shape than the 1K distribution (peak at K=9)" |
| R1-721 | F3 | 189 | 3/10 | severity (m7) | external-fact | asserts-own | "m7. Intermediate biological discoveries lack quantitative validation [RETAINED from v1, severity 3/10]" |
| R1-722 | F3 | 191 | ~611 | proteins (approximate count) | deterministic | quotes-paper | "The approximate protein counts ('~611 proteins,' '~11,000 proteins') should be exact." |
| R1-723 | F3 | 191 | ~11,000 | proteins (approximate count) | deterministic | quotes-paper | "The approximate protein counts ('~611 proteins,' '~11,000 proteins') should be exact." |
| R1-724 | F3 | 193 | 4/10 | severity (m8) | external-fact | asserts-own | "m8. No closed/maximal itemset analysis [RETAINED from v1, severity 4/10]" |
| R1-725 | F3 | 195 | 606K | itemsets (35K, likely redundant) | deterministic | asserts-own | "The 606K itemsets at 35K features likely include massive redundancy." |
| R1-726 | F3 | 197 | 3/10 | severity (m9) | external-fact | asserts-own | "m9. Title claims 'AlphaFold Scale' but mines annotations [RETAINED from v1, severity 3/10]" |
| R1-727 | F3 | 201 | 2/10 | severity (m10) | external-fact | asserts-own | "m10. No comparison with cuML/RAPIDS [RETAINED from v1, severity 2/10]" |
| R1-728 | F3 | 205 | 3/10 | severity (m11, current) | external-fact | asserts-own | "m11. Multi-GPU claims are now substantiated but incompletely [Severity: 3/10, improved from 5/10]" |
| R1-729 | F3 | 205 | 5/10 | severity (m11, v1) | external-fact | asserts-own | "[Severity: 3/10, improved from 5/10]" |
| R1-730 | F3 | 207 | 4xH200 | GPUs (35K results) | hardware-dependent | asserts-own | "The 35K results use 4xH200 GPUs, providing the first multi-GPU evidence." |
| R1-731 | F3 | 207 | 8.9-12.9 | s (bitvector build time across runs) | hardware-dependent | asserts-own | "The bitvector build time (8.9-12.9s across runs) and per-level times are reported, but speedup vs 1 GPU is not." |
| R1-732 | F3 | 207 | 1 | GPU (baseline missing) | hardware-dependent | requests | "but speedup vs 1 GPU is not" |
| R1-733 | F3 | 209 | 4/10 | severity (m12) | external-fact | asserts-own | "m12. Runtime comparison across scales is inconsistent [Severity: 4/10]" |
| R1-734 | F3 | 213 | 1,002 | features (1K Opus row) | deterministic | quotes-paper | table row "1K Opus / 1,002 / 76.9M / 0.00001% (8) / 1xH100 / 7.3 min / 26.8M" |
| R1-735 | F3 | 213 | 76.9M | transactions (1K Opus row) | deterministic | quotes-paper | table row "1K Opus / 1,002 / 76.9M / 0.00001% (8) / 1xH100 / 7.3 min / 26.8M" |
| R1-736 | F3 | 213 | 0.00001% | threshold (1K Opus row) | method-parameter | quotes-paper | table row "1K Opus / 1,002 / 76.9M / 0.00001% (8) / 1xH100 / 7.3 min / 26.8M" |
| R1-737 | F3 | 213 | 8 | min_count (1K Opus row) | method-parameter | quotes-paper | table row "1K Opus / 1,002 / 76.9M / 0.00001% (8) / 1xH100 / 7.3 min / 26.8M" |
| R1-738 | F3 | 213 | 1xH100 | GPUs (1K Opus row) | hardware-dependent | quotes-paper | table row "1K Opus / 1,002 / 76.9M / 0.00001% (8) / 1xH100 / 7.3 min / 26.8M" |
| R1-739 | F3 | 213 | 7.3 | min (1K Opus row) | hardware-dependent | quotes-paper | table row "1K Opus / 1,002 / 76.9M / 0.00001% (8) / 1xH100 / 7.3 min / 26.8M" |
| R1-740 | F3 | 213 | 26.8M | itemsets (1K Opus row) | deterministic | quotes-paper | table row "1K Opus / 1,002 / 76.9M / 0.00001% (8) / 1xH100 / 7.3 min / 26.8M" |
| R1-741 | F3 | 214 | 34,920 | features (35K Base row) | deterministic | asserts-own | table row "35K Base / 34,920 / 109.2M / 0.001% (1,093) / 4xH200 / 10.9 min / 606K" |
| R1-742 | F3 | 214 | 109.2M | transactions (35K Base row) | deterministic | asserts-own | table row "35K Base / 34,920 / 109.2M / 0.001% (1,093) / 4xH200 / 10.9 min / 606K" |
| R1-743 | F3 | 214 | 0.001% | threshold (35K Base row) | method-parameter | asserts-own | table row "35K Base / 34,920 / 109.2M / 0.001% (1,093) / 4xH200 / 10.9 min / 606K" |
| R1-744 | F3 | 214 | 1,093 | min_count (35K Base row) | method-parameter | asserts-own | table row "35K Base / 34,920 / 109.2M / 0.001% (1,093) / 4xH200 / 10.9 min / 606K" |
| R1-745 | F3 | 214 | 4xH200 | GPUs (35K Base row) | hardware-dependent | asserts-own | table row "35K Base / 34,920 / 109.2M / 0.001% (1,093) / 4xH200 / 10.9 min / 606K" |
| R1-746 | F3 | 214 | 10.9 | min (35K Base row) | hardware-dependent | asserts-own | table row "35K Base / 34,920 / 109.2M / 0.001% (1,093) / 4xH200 / 10.9 min / 606K" |
| R1-747 | F3 | 214 | 606K | itemsets (35K Base row) | deterministic | asserts-own | table row "35K Base / 34,920 / 109.2M / 0.001% (1,093) / 4xH200 / 10.9 min / 606K" |
| R1-748 | F3 | 216 | 4x | more GPUs (35K vs 1K) | hardware-dependent | asserts-own | "The 35K run uses 4x more GPUs, 100x higher threshold, and still takes 50% longer." |
| R1-749 | F3 | 216 | 100x | higher threshold (35K vs 1K) | method-parameter | asserts-own | "The 35K run uses 4x more GPUs, 100x higher threshold, and still takes 50% longer." |
| R1-750 | F3 | 216 | 50% | longer runtime (35K vs 1K) | hardware-dependent | asserts-own | "and still takes 50% longer" |
| R1-751 | F3 | 216 | 35x | more features / larger bitvectors | deterministic | asserts-own | "This is expected (35x more features = 35x larger bitvectors = memory-bandwidth bound)" |
| R1-752 | F3 | 224 | ~119 | GB bitvector matrix per GPU (35K) | deterministic | asserts-own | "The 35K bitvector matrix is ~119 GB per GPU (line 3 of power_test logs), requiring 4-GPU row-splitting." |
| R1-753 | F3 | 224 | 4 | GPUs (row-splitting required) | hardware-dependent | asserts-own | "requiring 4-GPU row-splitting" |
| R1-754 | F3 | 228 | 9 | K (peak, 1K) | deterministic | quotes-paper | "The peak shifting from K=9 (1K) to K=7 (35K) with a characteristic unimodal shape in both cases" |
| R1-755 | F3 | 228 | 7 | K (peak, 35K) | deterministic | asserts-own | "The peak shifting from K=9 (1K) to K=7 (35K)" |
| R1-756 | F3 | 236 | 2 (two) | vocabulary scales demonstrated | external-fact | asserts-own | "demonstrated at two vocabulary scales (1K and 35K) and two hardware configurations (1xH100 and 4xH200)" |
| R1-757 | F3 | 236 | 2 (two) | hardware configurations (1xH100, 4xH200) | hardware-dependent | asserts-own | "two hardware configurations (1xH100 and 4xH200)" |
| R1-758 | F3 | 236 | 0.1% | support (wave 3) | method-parameter | asserts-own | "mining at 0.1% support can discover tens of millions of itemsets at 35K features" |
| R1-759 | F3 | 236 | tens of millions | itemsets (wave 3 projection) | deterministic | asserts-own | "mining at 0.1% support can discover tens of millions of itemsets at 35K features, though this run is incomplete" |
| R1-760 | F3 | 240 | 175 | GB (SON bitvec) | deterministic | asserts-own | "The 175 GB bitvec calculation (40M chunk x 35K features) is straightforward and verifiable." |
| R1-761 | F3 | 240 | 40M | chunk (SON) | method-parameter | asserts-own | "The 175 GB bitvec calculation (40M chunk x 35K features)" |
| R1-762 | F3 | 244 | 0.0046% | CV (triplicate) | deterministic | asserts-own | "The 35K direct GPU runs demonstrate excellent reproducibility (CV=0.0046%)." |
| R1-763 | F3 | 248 | 5 | perms (1K null) | method-parameter | quotes-paper | "Both the 1K (5 perms) and 35K (2 perms) null models show the same qualitative pattern" |
| R1-764 | F3 | 248 | 2 | perms (35K null) | method-parameter | asserts-own | "Both the 1K (5 perms) and 35K (2 perms) null models show the same qualitative pattern" |
| R1-765 | F3 | 248 | 2 | K (depleted in both nulls) | deterministic | asserts-own | "depleted K=2, enriched K>=3 (35K) or K>=4 (1K), null collapse well before the biological K_max" |
| R1-766 | F3 | 248 | >=3 | K (enriched, 35K) | deterministic | asserts-own | "enriched K>=3 (35K) or K>=4 (1K)" |
| R1-767 | F3 | 248 | >=4 | K (enriched, 1K) | deterministic | quotes-paper | "enriched K>=3 (35K) or K>=4 (1K)" |
| R1-768 | F3 | 258 | 100+ | null-model permutations (both scales) | method-parameter | requests | "R1. Run 100+ null model permutations at BOTH scales [CRITICAL]" |
| R1-769 | F3 | 262 | 100 | perms (35K) | method-parameter | requests | "35K features: 100 perms x ~655s / 4 GPUs = ~4.5 hours. 200 perms = ~9 hours." |
| R1-770 | F3 | 262 | ~655 | s per perm (35K) | hardware-dependent | asserts-own | "35K features: 100 perms x ~655s / 4 GPUs = ~4.5 hours." |
| R1-771 | F3 | 262 | 4 | GPUs | hardware-dependent | asserts-own | "100 perms x ~655s / 4 GPUs = ~4.5 hours" |
| R1-772 | F3 | 262 | ~4.5 | hours (100 perms, 35K) | hardware-dependent | asserts-own | "100 perms x ~655s / 4 GPUs = ~4.5 hours" |
| R1-773 | F3 | 262 | 200 | perms (35K) | method-parameter | requests | "200 perms = ~9 hours" |
| R1-774 | F3 | 262 | ~9 | hours (200 perms, 35K) | hardware-dependent | asserts-own | "200 perms = ~9 hours" |
| R1-775 | F3 | 263 | 100 | perms (1K) | method-parameter | requests | "1K features: 100 perms x ~86s / 1 GPU = ~2.4 hours." |
| R1-776 | F3 | 263 | ~86 | s per perm (1K) | hardware-dependent | asserts-own | "1K features: 100 perms x ~86s / 1 GPU = ~2.4 hours." |
| R1-777 | F3 | 263 | 1 | GPU (1K null) | hardware-dependent | asserts-own | "100 perms x ~86s / 1 GPU = ~2.4 hours" |
| R1-778 | F3 | 263 | ~2.4 | hours (100 perms, 1K) | hardware-dependent | asserts-own | "100 perms x ~86s / 1 GPU = ~2.4 hours" |
| R1-779 | F3 | 267 | ~$50-100 | USD (cloud H200 cost for permutations) | hardware-dependent | asserts-own | "Estimated cost: ~$50-100 on cloud H200 instances. Trivial for a Nature Methods submission." |
| R1-780 | F3 | 271 | 8 | min_count (controlled 35K run) | method-parameter | requests | "Run 35K features at min_count=8 on a single GPU (if ~119 GB fits on H200) or the smallest feasible threshold." |
| R1-781 | F3 | 271 | 1 (single) | GPU | hardware-dependent | requests | "Run 35K features at min_count=8 on a single GPU" |
| R1-782 | F3 | 271 | ~119 | GB (bitvector to fit on H200) | deterministic | asserts-own | "(if ~119 GB fits on H200)" |
| R1-783 | F3 | 271 | ~17 | K_max (if unchanged at low threshold) | deterministic | asserts-own | "If K_max stays at ~17 even at low threshold, the drop is due to feature dilution or GO hierarchy effects" |
| R1-784 | F3 | 271 | 22 | K_max (if it rises toward) | deterministic | asserts-own | "If K_max rises toward 22, the drop is purely a threshold effect" |
| R1-785 | F3 | 279 | 0.1% | support (wave 3) | method-parameter | asserts-own | "The wave 3 partial shows the 35K feature set at 0.1% support producing millions of itemsets per K-level." |
| R1-786 | F3 | 279 | millions | itemsets per K-level | deterministic | asserts-own | "producing millions of itemsets per K-level" |
| R1-787 | F3 | 279 | 0.001% | support (power test) | method-parameter | asserts-own | "lower thresholds (0.001%) produce 900K+ frequent pairs and 17M+ K=4 itemsets" |
| R1-788 | F3 | 279 | 900K+ | frequent pairs (0.001%) | deterministic | asserts-own | "lower thresholds (0.001%) produce 900K+ frequent pairs and 17M+ K=4 itemsets" |
| R1-789 | F3 | 279 | 17M+ | K=4 itemsets (0.001%) | deterministic | asserts-own | "produce 900K+ frequent pairs and 17M+ K=4 itemsets" |
| R1-790 | F3 | 279 | 4xH200 | GPUs (feasibility) | hardware-dependent | asserts-own | "suggesting the full campaign at low support may be computationally infeasible on 4xH200" |
| R1-791 | F3 | 281 | 8 | K=22 proteins to name (heading) | deterministic | requests | "R5. Name the 8 K=22 proteins [HIGH PRIORITY, from v1]" |
| R1-792 | F3 | 281 | 22 | K (heading) | deterministic | quotes-paper | "R5. Name the 8 K=22 proteins [HIGH PRIORITY, from v1]" |
| R1-793 | F3 | 283 | 8 | K=22 proteins (to name) | deterministic | requests | "R5. Name the 8 K=22 proteins [HIGH PRIORITY, from v1] ... Resolve the 8-vs-3 discrepancy." |
| R1-794 | F3 | 283 | 22 | K | deterministic | quotes-paper | "R5. Name the 8 K=22 proteins" |
| R1-795 | F3 | 283 | 8-vs-3 | K=22 protein count discrepancy | deterministic | disputes | "Cross-reference with UniProt. Resolve the 8-vs-3 discrepancy." |
| R1-796 | F3 | 287 | 606K | itemsets (raw, 35K) | deterministic | asserts-own | "The 606K number may collapse to 50K-100K closed itemsets." |
| R1-797 | F3 | 287 | 50K-100K | closed itemsets (projected) | deterministic | asserts-own | "The 606K number may collapse to 50K-100K closed itemsets." |
| R1-798 | F3 | 291 | 42% | K=2 candidates pruned by locally-frequent union | deterministic | asserts-own | "The locally-frequent union step prunes 42% of K=2 candidates -- what is the analogous pruning at K>=5?" |
| R1-799 | F3 | 291 | >=5 | K (pruning analysis requested) | deterministic | requests | "what is the analogous pruning at K>=5?" |
| R1-800 | F3 | 299 | 109M | transactions (35K) | deterministic | asserts-own | "The 109M transactions (35K features) vs 76.9M (1K features) difference suggests the 35K vocabulary includes more organisms." |
| R1-801 | F3 | 299 | 76.9M | transactions (1K) | deterministic | quotes-paper | "The 109M transactions (35K features) vs 76.9M (1K features) difference" |
| R1-802 | F3 | 309 | 1,002 | original features | deterministic | quotes-paper | "Are the 1,002 original features a subset of the 34,920?" |
| R1-803 | F3 | 309 | 34,920 | features (35K) | deterministic | asserts-own | "Are the 1,002 original features a subset of the 34,920?" |
| R1-804 | F3 | 313 | 2 (Two) | data points (vocabulary scales) | external-fact | asserts-own | "Two data points (1K: peak=9, K_max=22; 35K: peak=7, K_max=17) suggest an inverse relationship" |
| R1-805 | F3 | 313 | 9 | K peak (1K) | deterministic | quotes-paper | "(1K: peak=9, K_max=22; 35K: peak=7, K_max=17)" |
| R1-806 | F3 | 313 | 22 | K_max (1K) | deterministic | quotes-paper | "(1K: peak=9, K_max=22; 35K: peak=7, K_max=17)" |
| R1-807 | F3 | 313 | 7 | K peak (35K) | deterministic | asserts-own | "(1K: peak=9, K_max=22; 35K: peak=7, K_max=17)" |
| R1-808 | F3 | 313 | 17 | K_max (35K) | deterministic | asserts-own | "(1K: peak=9, K_max=22; 35K: peak=7, K_max=17)" |
| R1-809 | F3 | 317 | 6 | null K_max (1K, 5 perms) | deterministic | quotes-paper | "The null K_max is 6 (1K, 5 perms) vs 5 (35K, 2 perms)." |
| R1-810 | F3 | 317 | 5 | perms (1K) | method-parameter | quotes-paper | "The null K_max is 6 (1K, 5 perms) vs 5 (35K, 2 perms)." |
| R1-811 | F3 | 317 | 5 | null K_max (35K, 2 perms) | deterministic | asserts-own | "The null K_max is 6 (1K, 5 perms) vs 5 (35K, 2 perms)." |
| R1-812 | F3 | 317 | 2 | perms (35K) | method-parameter | asserts-own | "The null K_max is 6 (1K, 5 perms) vs 5 (35K, 2 perms)." |
| R1-813 | F3 | 317 | ~3.5x | biological/null K_max ratio (1K: 6->22) | deterministic | asserts-own | "If the null-to-biological K_max ratio is approximately constant (~3.5x for 1K: 6->22; ~3.4x for 35K: 5->17)" |
| R1-814 | F3 | 317 | ~3.4x | biological/null K_max ratio (35K: 5->17) | deterministic | asserts-own | "(~3.5x for 1K: 6->22; ~3.4x for 35K: 5->17)" |
| R1-815 | F3 | 321 | 76.9M | transactions (1K run) | deterministic | quotes-paper | "The 1K run processes 76.9M transactions x 1,002 features in 7.3 min on 1xH100." |
| R1-816 | F3 | 321 | 1,002 | features (1K run) | deterministic | quotes-paper | "The 1K run processes 76.9M transactions x 1,002 features in 7.3 min on 1xH100." |
| R1-817 | F3 | 321 | 7.3 | min (1K run) | hardware-dependent | quotes-paper | "The 1K run processes 76.9M transactions x 1,002 features in 7.3 min on 1xH100." |
| R1-818 | F3 | 321 | 1xH100 | GPUs (1K run) | hardware-dependent | quotes-paper | "in 7.3 min on 1xH100" |
| R1-819 | F3 | 321 | 109M | transactions (35K run) | deterministic | asserts-own | "The 35K run processes 109M transactions x 34,920 features in 10.9 min on 4xH200." |
| R1-820 | F3 | 321 | 34,920 | features (35K run) | deterministic | asserts-own | "The 35K run processes 109M transactions x 34,920 features in 10.9 min on 4xH200." |
| R1-821 | F3 | 321 | 10.9 | min (35K run) | hardware-dependent | asserts-own | "The 35K run processes 109M transactions x 34,920 features in 10.9 min on 4xH200." |
| R1-822 | F3 | 321 | 4xH200 | GPUs (35K run) | hardware-dependent | asserts-own | "in 10.9 min on 4xH200" |
| R1-823 | F3 | 321 | ~50x | more feature-transaction-products per GPU-second (35K vs 1K) | hardware-dependent | asserts-own | "Normalized: the 35K run achieves ~50x more feature-transaction-products per GPU-second" |
| R1-824 | F3 | 327 | 2 | permutations (35K null, 'new concern') | method-parameter | asserts-own | "while creating new ones (K-max drop explanation, 2-perm null model, incomplete campaigns, multi-GPU approximation error)" |
| R1-825 | F3 | 329 | 5 | perms (1K null, 'weak') | method-parameter | quotes-paper | "the null model has regressed from weak (5 perms) to indefensible (2 perms)" |
| R1-826 | F3 | 329 | 2 | perms (35K null, 'indefensible') | method-parameter | asserts-own | "the null model has regressed from weak (5 perms) to indefensible (2 perms)" |
| R1-827 | F3 | 333 | 100+ | permutations (condition for minor revision) | method-parameter | requests | "If the authors run 100+ permutations and explain the K-max drop, the paper moves from 'major revision' to 'minor revision.'" |
| R1-828 | F3 | 341 | 9/10 | severity (M1, summary table) | external-fact | asserts-own | severity table "M1: Null model permutation count (now n=2) / 9/10 / ESCALATED" |
| R1-829 | F3 | 341 | n=2 | permutations (35K null) | method-parameter | asserts-own | severity table "M1: Null model permutation count (now n=2)" |
| R1-830 | F3 | 342 | 7/10 | severity (M2, summary table) | external-fact | asserts-own | severity table "M2: GO hierarchy inflation unquantified / 7/10" |
| R1-831 | F3 | 343 | 5/10 | severity (M3, summary table) | external-fact | asserts-own | severity table "M3: SON comparison framing / 5/10" |
| R1-832 | F3 | 344 | 8/10 | severity (M4, summary table) | external-fact | asserts-own | severity table "M4: K-max drop unexplained (NEW) / 8/10" |
| R1-833 | F3 | 345 | 7/10 | severity (M5, summary table) | external-fact | asserts-own | severity table "M5: 8 vs 3 K=22 proteins / 7/10 / Unchanged" |
| R1-834 | F3 | 345 | 8 vs 3 | K=22 proteins (summary table) | deterministic | disputes | severity table "M5: 8 vs 3 K=22 proteins / 7/10 / Unchanged" |
| R1-835 | F3 | 346 | 6/10 | severity (M6, summary table) | external-fact | asserts-own | severity table "M6: Wave 3 incomplete (NEW) / 6/10" |

---

## SECTION B — REVIEW-FLAGGED INCONSISTENCIES

Every place a reviewer says two numbers disagree, a value is implausible, a method detail is missing or contradicts another section, or a value changed between paper versions. Wording in column 4 paraphrases the reviewer; column 5 lists the values/locations the reviewer puts in tension.

| ID | file | line(s) | what the reviewer claims is inconsistent | the two (or more) values/locations involved |
|---|---|---|---|---|
| B-01 | F1 | 19, 30-34, 121-123 | Vocabulary construction misdescribed: paper says features retained if in ≥8 proteins; log shows a top-500-per-type frequency cap; "≥8" is the mining min_count, a different quantity | paper line 129 "at least 8 proteins" vs `pipeline_214m.log:2063-2064` "24291 unique Pfam, 25993 unique GO" → "6 pLDDT + 500 Pfam + 500 GO = 1006"; `godmode_mining.log:2` min_count 8 |
| B-02 | F1 | 20, 36-40, 125-132 | "40× reduction" pairs the CSR of the 76.9M mined subset against the dense matrix of the 205.6M full set; same-dataset ratio is ~15×; body also inconsistent with the paper's own appendix ratios | 5.1 GB CSR (76,890,945 txns, `direct_mining.log`) vs 206 GB dense (205,620,298 × 1002 × 1 B, `beyond_mining.log`); 77 GB / 5.06 GB = ~15×; appendix `tab:memory-comparison` 1.4× bit-packed / 7.8× at 0.01% density vs body 40× (paper lines 167, 402) |
| B-03 | F1 | 21, 42-47, 134-136 | Paper says the 8 K=22 proteins "can be recovered by querying the source transaction data" but also says the transaction matrix is not deposited; script needs an absent parquet; no accession list saved | paper line 333 (recoverable via `analyze_k22_proteins.py`) vs paper line 530 (matrix "not deposited"); `analyze_k22_proteins.py:162-190` requires `transactions_214m_base.parquet` (absent); `item_mapping` absent; no `accessions_deepest_itemset_*.tsv` |
| B-04 | F1 | 22, 49-53, 138-141 | p<0.45 is attributed to the "rule of three", but the rule of three gives 3/5 = 0.60; 0.45 is actually the exact one-sided binomial 95% bound 1−0.05^(1/5) | 0.45 (paper lines 367, 382, 391) vs 0.60 (rule of three) vs 0.4507 (exact binomial); 0/5 null runs reach K≥7 (`experiment_null_model_20260219_061046.json:153,163`) |
| B-05 | F1 | 15 | Value changed between paper versions: v1 Table 1 vocabulary composition was "fabricated" and has been replaced by the verified composition | v1 Table 1 247/752/3 vs verified 500/500/6 = 1,006 |
| B-06 | F1 | 24, 97, 143-144 | Paper's "verification run at min_count=4 discovered 48 million itemsets with identical maximum" has no artifact; the only "48M" in logs is a transaction count during extraction (conflation risk) | min_count=4 / 48M itemsets (paper line 335) vs "Written 48M transactions" (`pipeline_214m.log:2113`); null-model JSON min_count 769 |
| B-07 | F1 | 95 | "K=22 shared by exactly 8 proteins" cannot be verified: the named cross-check file caps at K=19 and is a different run; no artifact contains any K≥20 itemset | K=22 / 8 proteins (paper lines 254, 286) vs `decoded_top_k_patterns.txt` max K=19 (187 proteins) |
| B-08 | F1 | 94, 152 | UniProt release string "2025_01" is not recorded in any artifact; log shows only the input filename and a Feb 2026 run date | "2025_01" (paper lines 129, 478, 530) vs `pipeline_214m.log:3,10` (`uniprot_trembl.dat.gz`, Feb 2026) |
| B-09 | F1 | 148-150 | Data source named inconsistently within the paper: TrEMBL in Methods vs Swiss-Prot in Discussion | paper line 129 "UniProt TrEMBL" (matches log input `uniprot_trembl.dat.gz`) vs paper line 478 "UniProt/Swiss-Prot release 2025_01" |
| B-10 | F1 | 96 | 22→21 independent features (0 GO parent-child pairs; InterPro2GO link removed) is computed at runtime and cannot be reproduced read-only (missing cached `go.obo`, no saved output) | paper line 333 (22→21, 0 pairs) vs `analyze_k22_proteins.py:277-420` (runtime pronto + go.obo, no artifact) |
| B-11 | F1 | 99-100 | Unverifiable from local artifacts: competitor rows (Borgelt/Fang/GMiner/BIGMiner) and the external accuracy of the 37 references — literature-sourced only | paper lines 420-423 (GMiner 15M basis for 5.1×); paper lines 541-724 (37 refs) |
| B-12 | F1 | 72 | Log filenames do not match the paper's run names (mapping had to be verified) | `pipeline/ultra/extreme/direct/beyond/godmode_mining.log` vs paper run names Base/Super/Power/Blitz/Ultra/Opus |
| B-13 | F1 | 108 | Commit subject contradicts its content: "validation produced identical results" vs a 231-line manuscript rewrite that deletes expanded-run figures | commit `b318df5` subject vs its diff (231 lines) |
| B-14 | F1 | 109 | Stray expanded-run artifacts exist although the plan says no artifact of the expanded run was preserved | `results_35k/*.json` present vs `plans/leg-het-vast-in-crispy-glade.md` "geen bewaard artefact" |
| B-15 | F1 | 110 | Expanded/"v2" run reported with two mutually incompatible figures across documents; current .tex contains neither | 606K itemsets @ 4×H200 (`revision_notes_b3.tex`) vs 16.8B itemsets @ 8×H200 (plan) |
| B-16 | F1 | 111, 154-156 | Headline Z>3,700 in abstract/conclusion lacks the n=5 caveat that the body applies (t-statistics, 4 df) | Z>3,700 (paper lines 91, 498) vs table caption / §4.4 t-statistics (4 df), 5 permutations |
| B-17 | F1 | 112 | Dutch translation is stale and diverges from the English paper | `et_miner_proteome_nl.tex` (old title, Feb 2026 date) vs current English .tex |
| B-18 | F1 | 113, 146 | Approximate supporting-protein counts are given where exact values exist in an artifact | ~11,000 / ~10,500 / ~16,000 (paper lines 353, 355, 357; K=13/12/11 patterns) vs exact integers in `decoded_top_k_patterns.txt` |
| B-19 | F1 | 113, 158 | 7.3 min is presented standalone without stating it is mining-only, while extraction alone took 63 min | 7.3 min (paper lines 120, 496) vs 63-minute feature extraction (paper line 129; af_extract 3777.0 s) |
| B-20 | F2 | 22, 31-35, 85 | Null model run only at the Power threshold, but the headline K=9 peak and K=22 ceiling come from the Opus threshold where no null model was run; generalisation across thresholds unjustified | 0.001% / min_count 769 (null model, §3.6, Table 4) vs 0.00001% / min_count 8 (Opus, K=22, K=9 peak) |
| B-21 | F2 | 23, 38-41, 96 | n=5 permutations cannot carry the inferential weight; p<0.45 does not reject the null at any conventional level although the sentence implies significance | n=5; p<0.45 (0/5 null runs with K≥7); "Z" magnitudes relabelled t (4 df) |
| B-22 | F2 | 42 | The K=6 effect size does not reproduce cleanly from the rounded μ/σ shown in the table (σ from n=5 unstable) | +71,728 (K=6) vs rounded μ/σ in Table 4 |
| B-23 | F2 | 43, 86 | Manuscript itself says 100+ permutations are needed for p<0.01, at ~130 s each (≈3.6 GPU-hours) — yet only 5 were run | 100+ permutations / p<0.01 / ~130 s per permutation vs n=5 actually run |
| B-24 | F2 | 24, 47, 69, 129 | The 8 K=22 accessions are not listed; the paper says they are recoverable from source transaction data, but that data (transaction parquet) is not deposited / was lost with the compute environment | 8 accessions (Table 5, §3.5) vs `transactions_214m.parquet` not deposited (Stage 4) |
| B-25 | F2 | 48, 87 | Missing method detail: GO evidence codes (EXP/IDA vs IEA) not distinguished for the K=22 signature, decisive with n=8 proteins | n=8 proteins; per-term evidence codes absent |
| B-26 | F2 | 53 | Scale comparison is apples-to-oranges: every competitor ran on a different dataset and hardware generation, so "5.1× more transactions" is scale reached, not a controlled comparison; no same-data baseline | Table 3 (5.1×, GMiner 15M) vs Direct-vs-SON (21×, 95.2%) the only controlled comparison |
| B-27 | F2 | 56, 88, 119 | "26.8 million patterns" is the full redundant itemset space; closed/maximal counts never reported although filtering is implemented | 26.8M total vs (unreported) closed / maximal counts; 2^K − 2 sub-itemsets per K-itemset |
| B-28 | F2 | 63 | Multiple different memory quantities coexist across sections and risk reader confusion (reviewer says "three" but lists four) | ~206 GB (naive dense), ~26 GB (bit-packed, 214M set), ~5.1 GB (CSR-COO), ~10 GB (bit-packed, 76.9M subset) — Sections 1, 2.3, Appendix D |
| B-29 | F2 | 64 | "Structural property" wording oversells: only 2 of 6 pLDDT bins pass and the recurring one is a single medium-confidence bin | 2 of 6 pLDDT bins; "medium confidence (70–90)" vs Table 5 "structural property" |
| B-30 | F2 | 65 | Approximate intermediate-K protein counts where artefacts support near-exact values | ~11,000 / ~10,500 / ~16,000 vs exact supports in artefacts |
| B-31 | F2 | 66 | Multi-GPU scaling listed as a headline design choice but exercised by no reported result (all experiments single H100) | "automatic multi-GPU scaling" vs "all experiments used a single H100" |
| B-32 | F2 | 67, 120, 173 | Bibliography inconsistency: `miettinen2020` in bibliography but uncited (fixed post-review; zero uncited remain) | 1 uncited bibitem → 0 |
| B-33 | F2 | 78 | Value changed between versions: GMiner row labelling "15M synthetic / 1.7M real" is "now correct" (implying earlier mislabelling) | GMiner 15M synthetic vs 1.7M real (Table 3) |
| B-34 | F2 | 115, 158 | Missing method detail: no multiple-comparison correction applied to millions of itemsets (acknowledged as Limitation 1 only) | millions of itemsets tested; no FDR |
| B-35 | F2 | 120 | Value changed between versions: 13 citation errors corrected in this revision | 13 citation errors (previous version) vs re-verified references (this version) |
| B-36 | F2 | 138, 159, 172 | Repository inconsistent with manuscript: two figure PDFs from the removed expanded run remain on disk (removed post-review) | `figures/kdist_shift.pdf`, `figures/pruning_savings.pdf` vs no `\includegraphics` in .tex |
| B-37 | F2 | 143, 160, 171 | Relationship to the published Zenodo v1 not disclosed (versioned-preprint update); previously fabricated Table 1 corrected in this revision | Zenodo v1 vs current manuscript; "previously fabricated Table 1" |
| B-38 | F2 | 129, 131 | Missing data-availability statement; raw source data not in repository | `transactions_214m.parquet` not deposited; statement ❌ missing (added post-review, line 171) |
| B-39 | F3 | 27-29, 127-129, 329, 341 | Null model regressed: 35K-feature null uses 2 permutations vs 5 in the 1K-feature null | 5 permutations (1K) vs 2 permutations (35K; seeds 7138484576005690180, 4047939128787533792) |
| B-40 | F3 | 33 | Z-scores from n=2 are statistically invalid (t with 1 df, no finite moments); std values from 2 observations have CIs spanning zero to infinity | Z=722.91 at K=3 = (53235 − 15919.5)/51.6; std K=2 123.7, K=3 51.6, K=4 2.8 |
| B-41 | F3 | 35, 43, 265 | Division-by-zero artifact presented as evidence: K=5 null std=0.0 (both runs = 1 itemset) yields z_score "inf" stored as a string | K=5: 1 itemset in both runs, std 0.0, z_score "inf" (JSON line 104) |
| B-42 | F3 | 37 | Thresholds do not match: 35K null model min_count differs from the 35K mining campaign min_count, and neither matches the paper's Opus threshold | min_count 1,090 (null, ≈0.001% of 109M) vs 1,093 (mining, 0.001%) vs 8 (Opus) |
| B-43 | F3 | 39 | Enrichment sign reversal at K=3 between vocabularies is unexplained and unmentioned in the manuscript | 1K (5 perms): K=2 Z=−987, K=3 Z=−143 vs 35K (2 perms): K=2 Z=−102, K=3 Z=+723 |
| B-44 | F3 | 41 | Missing method detail: Fisher–Yates shuffle creates duplicate features within proteins which are then removed; duplication rate undocumented/uncontrolled | permutation dedup (1K and 35K null models) |
| B-45 | F3 | 47-55, 75-92, 133, 271, 313 | K_max drop and peak shift when the vocabulary grew are counter-intuitive and unexplained; four candidate causes (threshold, dilution, GO collapse, multi-GPU local threshold) not disentangled | K_max 22 → 17; peak K=9 → K=7; 26.8M (min_count 8) → 606K (min_count 1,093); 1,002 → 34,920 features |
| B-46 | F3 | 53 | K=22 itemset contains GO hierarchy inflation (cytoplasm/cytosol, cytoplasm/mitochondrion, nucleus/nuclear speckle); corrected independent K is lower than reported | K=22 vs corrected independent K ≈ 19–20 |
| B-47 | F3 | 57-67, 240 | SON "physically impossible" framing needs nuance: 40M-chunk bitvec exceeds VRAM, but a smaller chunk would fit (with more chunks / worse miss rate) | 40M × 35K = 175 GB > 143 GB H200 vs chunk 20M × 35K = ~87 GB (6 chunks); chunk_size < 33M → 4+ chunks at 109M transactions |
| B-48 | F3 | 73, 151-155 | Triplicate runs disagree at the boundary: K_max and K=17 counts vary across runs (support-boundary noise or multi-GPU non-determinism); recommend reporting K_max as 16–17 | totals 606,319 / 606,293 / 606,263; K_max 17 / 16 / 17; K=17 itemsets 2 / 0 / 1; K=16 stable at 152 |
| B-49 | F3 | 90, 165-171, 291 | Multi-GPU row-splitting reintroduces a SON-like approximation: locally-frequent union over-estimates and the global recount prunes 42% at K=2; error at higher K unquantified; no single-GPU baseline | local_min_count 274 (power test) / 27,307 for global 109,225 (wave 3); 14,800 locally frequent vs 8,554 globally frequent K=2 pairs |
| B-50 | F3 | 94-96, 143-145, 281-283, 345 | K=22 support count disagrees with number of proteins the analysis script identified; accessions never named | 8 proteins (mining support) vs 3 proteins (analysis-script fallback heuristic) |
| B-51 | F3 | 98-121, 277-279 | Wave 3 (0.1%) campaign incomplete: log cuts off at K=9 after 743 s with K=10 unknown; per-level counts ~30× the 0.001% run; feasibility of the full 35K campaign at low support unclear | K=9 10,041,611 itemsets at 743.4 s; K=10 "???" (12.5M locally frequent); K=7 2.2M (0.1%) vs 64K (0.001%) |
| B-52 | F3 | 119, 279 | Power test at min_count=1,093 stalls: K=2 alone takes 586 s vs 1.2 s at min_count=109,225 because 902K vs 8.5K frequent pairs; stalled at K=4 with 17.4M itemsets | 586 s vs 1.2 s; 902K vs 8.5K pairs; K=4 17.4M |
| B-53 | F3 | 139-141 | Naming inconsistency and ambiguous timing: 1K result files named `godmode` vs paper run name `Opus`; 7.3-minute claim ambiguous (mining vs total pipeline) | `godmode` vs `Opus`; 7.3 min mining vs total pipeline |
| B-54 | F3 | 157-163, 173-179, 307-309 | 35K vocabulary composition, selection, and overlap with the 1K vocabulary undocumented; 34,920 is not a round number and unmotivated; whether all 1,002 original features are included is unverified | Table 1 (247 Pfam, 302 GO:MF, 289 GO:BP, 161 GO:CC, 3 pLDDT) documented vs 34,920 features undocumented; K=1: 34,920 frequent items |
| B-55 | F3 | 181-183 | 1K SON comparison reported from a single run; triplicate runs and per-K miss rates missing | 21.4× speedup, 95.2% miss rate (single run) |
| B-56 | F3 | 185-187 | "Bell curve" terminology not supported by any formal fit; the two distributions differ in shape and peak | 35K peak K=7 (64,400 itemsets) vs 1K peak K=9 |
| B-57 | F3 | 189-191 | Approximate protein counts for intermediate discoveries should be exact | ~611 proteins; ~11,000 proteins |
| B-58 | F3 | 193-195, 285-287 | Closed/maximal itemset counts not reported; raw 606K likely inflated by GO redundancy | 606K raw vs projected 50K–100K closed |
| B-59 | F3 | 205-207 | Multi-GPU speedup unknown: no single-GPU baseline for 35K; only build times and per-level times reported | bitvector build 8.9–12.9 s across runs; speedup vs 1 GPU not reported |
| B-60 | F3 | 209-216, 319-321 | Runtime comparison across scales is inconsistent / un-normalised: 35K run uses 4× the GPUs and a 100× higher threshold yet takes 50% longer; needs itemsets-per-GPU-second | 1K Opus: 1,002 feat / 76.9M txn / 0.00001% (8) / 1×H100 / 7.3 min / 26.8M vs 35K Base: 34,920 / 109.2M / 0.001% (1,093) / 4×H200 / 10.9 min / 606K |
| B-61 | F3 | 297-299 | Transaction counts differ between the two vocabularies without explanation (suggests different organism coverage) | 109M transactions (35K) vs 76.9M (1K) |
| B-62 | F3 | 315-317 | Null K_max differs across scales; ratio to biological K_max noted as approximately constant but unexplored | null K_max 6 (1K, 5 perms) vs 5 (35K, 2 perms); ratios ~3.5× (6→22) vs ~3.4× (5→17) |
| B-63 | F3 | 43, 258-267 | Compute for adequate permutations is affordable yet not done: 100 perms ≈ 18 h (35K) / ≈ 2.4 h (1K); ~$50–100 cloud cost | ~655 s/perm (35K, 4×H200); ~86 s/perm (1K); 100 / 200 perms |

### B-addendum — extractor-noted cross-file observations (NOT reviewer-flagged; recorded only because they may matter for the audit; no correctness judgement)

| ID | files / lines | observation | values involved |
|---|---|---|---|
| X-01 | F1 L78, F2 L43/L86 vs F3 L43/L263 | Per-permutation time for the 1K null model is stated differently across reviews | ~130 s/permutation (F1, F2) vs ~86 s/perm (F3) |
| X-02 | F1 L80, F2 L18/L53 vs F3 L181/L183 | Direct-vs-SON speedup stated with different precision | 21× (F1, F2) vs 21.4× (F3) |
| X-03 | F3 L159 vs F1 L15 | F3 quotes the paper's Table 1 vocabulary as 247 Pfam / 302 GO:MF / 289 GO:BP / 161 GO:CC / 3 pLDDT (sums to 247/752/3); F1 calls exactly that v1 table "fabricated" and corrected to 500/500/6 = 1,006 | 247/752/3 vs 500/500/6 |
| X-04 | F1 L96, F2 L46 vs F3 L53 | Independent-feature count of the K=22 itemset after GO true-path checking differs | 21 independent (0 parent-child pairs + 1 InterPro2GO link; F1/F2) vs ≈19–20 corrected (F3, parent-child pairs asserted present) |
| X-05 | F1 L72, F2 L77 vs F1 L78, F2 L31 | Power-campaign min_count vs null-model min_count both described as the 0.001% threshold | 768 (Power campaign) vs 769 (null model) |
| X-06 | F1 L110 vs F3 L17/L214 | The 606K @ 4×H200 expanded-run figure that F1 says is contradicted by a 16.8B @ 8×H200 figure in the plan is the same 35K result F3 analyses in detail (606,292 mean, 4×H200) | 606K @ 4×H200 vs 16.8B @ 8×H200 |
| X-07 | F3 L43 vs F3 L262 | Within F3, the wall-clock estimate for 100 permutations at 35K is given two ways (second divides the already-4-GPU per-perm time by 4 again) | 100 perms = ~18 hours (L43) vs ~4.5 hours (L262); 200 perms = ~36 h vs ~9 h |
| X-08 | F2 L63 | Reviewer says "Three legitimate but different quantities coexist" but enumerates four | ~206 GB, ~26 GB, ~5.1 GB, ~10 GB |
| X-09 | F1 L70 / F2 L63 vs F3 L224 | Bit-packed / bitvector matrix sizes quoted for different datasets and vocabularies — not directly comparable across reviews | ~26 GB (1,002 items, 214M set), ~10 GB (76.9M subset), 27/19 GB appendix row vs ~119 GB per GPU (34,920 items, 4-way row split) |
| X-10 | F3 L18 vs F3 L33 | 35K null-model K=3 Z quoted rounded and unrounded in the same file | Z=723 (L18, L39) vs 722.91 (L33) |
| X-11 | F3 L37 vs F3 L214 vs F3 L100 | 35K transaction count given as 109M, 109.2M; 0.1% → min_count 109,225 implies ≈109.2M; 0.001% → 1,093 (mining) but 1,090 (null) | 109M / 109.2M / 109,225 / 1,093 / 1,090 |
| X-12 | F2 L38 vs F1 L22/L49-53 | F2 accepts "p < 0.45 (rule of three, 0/5)" as a correct fix, whereas F1 disputes the "rule of three" attribution (gives 0.60) and credits 0.45 to the exact binomial bound | 0.45 "rule of three" (F2) vs 0.4507 exact binomial / 0.60 rule of three (F1) |
| X-13 | F1 L74/L95 vs F3 L94-96 | F1 confirms K=22 = 1 itemset / 8 proteins against the godmode log but marks the 8 proteins unverifiable (no K≥20 artifact); F3 reports the analysis script found only 3 proteins | 8 (log support) vs 3 (script) vs no artifact with K≥20 |

---

## SECTION C — PER-FILE SUMMARIES

### C.1 F1 — `paper/PAPER_V2_REVIEW.md` ("Adversarial Review — papers/et_miner_proteome.tex")

- **Type / author:** automated adversarial cross-check of every quantitative claim in the .tex against the mining logs, experiment JSONs and decoded-pattern artifacts under `applications/alphafold/results_214m/`, plus "repository integrity seeds from prior analysis". No named human reviewer.
- **Date:** 2026-08-01 (line 3).
- **Paper version reviewed:** the restored base-run manuscript ("V2" per the file name); it explicitly refers to "the old fabricated v1 Table 1 (247/752/3)" having been corrected to 500/500/6 = 1,006, and states the current .tex contains no expanded-run (35K) claims. Paper line numbers cited run to ~887.
- **Scope:** 9 verification buckets, 73 claims adjudicated: 62 CONFIRMED, 4 DISCREPANT, 7 UNVERIFIABLE. Confirmed buckets: dataset/vocabulary (7), memory (6), campaign table (6/6), K-distribution (8/8), K=22 composition (2), null model (17), Direct-vs-SON (6/6), scale comparison (3), citations (7).
- **Discrepancies:** D1 vocabulary described as ≥8-support retention instead of top-500 caps; D2 40× memory reduction conflates 76.9M-subset CSR with 205.6M-set dense (~15× same-dataset); D3 "8 K=22 accessions recoverable" contradicts "matrix not deposited"; D4 p<0.45 mis-attributed to the rule of three (exact binomial 1−0.05^(1/5)=0.4507; rule of three = 0.60).
- **Unverifiable:** UniProt release 2025_01; K=22 in exactly 8 proteins (cross-check file caps at K=19/187 proteins); 22→21 independence check; min_count=4 → 48M itemsets (no such run; "48M" is a transaction count); the 8 accessions; competitor rows; external accuracy of 37 references.
- **Integrity items:** misleading commit `b318df5` (231-line rewrite); stray `results_35k/*.json` vs plan text; self-contradictory expanded run (606K @ 4×H200 vs 16.8B @ 8×H200); missing n=5 caveat on Z>3,700 in abstract/conclusion; stale Dutch translation; approximate counts ~11,000/~10,500/~16,000; 7.3 min not marked mining-only.
- **Verdict:** "The paper's headline numbers are overwhelmingly faithful to the artifacts"; provides an ordered 12-item fix list with exact OLD→NEW text (TrEMBL vs Swiss-Prot at line 478 also flagged). No accept/reject recommendation — it is an audit + fix list; explicitly instructs "Do NOT reintroduce any expanded-run figure."

### C.2 F2 — `paper/peer_review_jul12.md` ("Peer Review — Higher-Order Protein Feature Co-occurrence at AlphaFold Scale")

- **Type / author:** simulated journal-style peer review (methods / applied-computation preprint), unnamed reviewer, with a 7-stage checklist appendix and a post-review fix log. Manuscript `papers/et_miner_proteome.tex` by E. Ahmic and C. claudya ("Anthropic, Claude Code Opus 4.6").
- **Date:** 2026-07-12 (review date and Appendix B fix date).
- **Paper version reviewed:** the "restored base-run manuscript" — the revision in which numbers were re-locked to preserved artefacts after a previous version with a fabricated Table 1 and 13 citation errors; the expanded (35K) run has been removed (two orphaned figure PDFs remain). Title now "Higher-Order Protein Feature Co-occurrence…" (no motif overclaim). A Zenodo v1 exists.
- **Headline numbers as summarised by the reviewer:** 76.9M multi-feature proteins, 1,002 features, single H100, 7.3 min (mining only), 26.8M patterns, K=22, null model at 0.001% (min_count 769, n=5, seed 42, ~130 s/perm), 21× / 95.2% Direct-vs-SON, software Python 3.10 / CuPy 13.0 / NumPy 1.26 / CUDA 12.4 / Ubuntu 22.04 / H100 80GB SXM5.
- **Majors (5):** (1) null model does not cover the Opus threshold where K=22 / K=9 peak live; (2) n=5 permutations over-leveraged (p<0.45 non-significant; +71,728 from unstable σ; 100 permutations ≈ 3.6 GPU-hours feasible); (3) K=22 result is one itemset in 8 unlisted proteins with no evidence codes; (4) no same-dataset baseline (Table 3 apples-to-oranges; 5.1× is scale not speed); (5) 26.8M counts the redundant space — closed/maximal counts missing.
- **Minors (8):** abstract scope caveat; memory-figure table (206 / 26 / 5.1 / 10 GB); pLDDT "structural property" oversell (2 of 6 bins, 70–90); approximate counts; multi-GPU framing; uncited `miettinen2020`; true-path check for intermediate-K; data-availability paragraph. Checklist surfaced: no COI, no funding, no ethics statement, AI co-author policy, no FDR, orphaned figures, Zenodo-v1 relationship.
- **Verdict:** **Major revision** — "publishable as a methods contribution"; engineering sound, biological framing outruns statistics/reproducibility. Appendix B: funding, COI, ethics, data-availability statements added, orphaned figures removed, `miettinen2020` cited; AI co-authorship retained by author decision; Majors 1–5 and the FDR note still open.

### C.3 F3 — `paper/review_b1_hostile.md` ("Hostile Peer Review: Protein Structural Motif Discovery at AlphaFold Scale")

- **Type / author:** "Agent B1 (Senior Paper Reviewer, Nature Methods simulation)" — deliberately hostile review.
- **Date:** 2026-02-21 (review v2, "updated with 35K-feature results"); review v1 dated 2026-02-20 covered only the 1K-feature results. Result files referenced live in `applications/alphafold/results_35k/` and `results_214m/`.
- **Paper version reviewed:** the early (Feb 2026) manuscript still titled "Protein Structural Motif Discovery at AlphaFold Scale", containing the 1K (1,002-feature) results (Table 1 quoted as 247 Pfam / 302 GO:MF / 289 GO:BP / 161 GO:CC / 3 pLDDT; 26.8M itemsets; K=22; 7.3 min on 1×H100; 21.4× / 95.2% SON comparison; 5-perm null), plus three **new, not-yet-in-manuscript** 35K-feature result files: Direct-GPU triplicate (606,292 ± 28 itemsets, K_max 17, 654 s on 4×H200, SON OOM at 175 GB > 143 GB), a 2-permutation null model (min_count 1,090; Z=723 at K=3, 20,585 at K=4, "inf" at K=5), and an incomplete wave-3 0.1% run (10,041,611 itemsets at K=9 after 743 s).
- **Majors (6, severity /10):** M1 null model regressed to n=2 (9); M2 GO true-path inflation unquantified, K=22 corrected to ≈19–20 (7); M3 SON framing — infeasibility stronger than strawman but needs chunk-size nuance (5, from 8); M4 K_max drop 22→17 and peak 9→7 unexplained (8, new); M5 8 vs 3 K=22 proteins unresolved (7); M6 wave 3 incomplete (6, new).
- **Minors (12):** triplicate boundary instability (K_max 17/16/17); undocumented 35K vocabulary; multi-GPU row-splitting approximation (42% of K=2 candidates pruned at recount); unmotivated 34,920; 21.4× now secondary; "bell curve" wording; approximate counts (~611, ~11,000); no closed/maximal counts; title vs annotations; no cuML/RAPIDS comparison; multi-GPU speedup unknown; un-normalised cross-scale runtime comparison (7.3 min 1×H100 vs 10.9 min 4×H200).
- **Strengths (8):** CSR→bitvector pipeline scales to ~119 GB/GPU; multi-scale K-distribution; biology recovery; genuine scale; SON infeasibility as a novel finding; triplicate reproducibility (CV 0.0046%); consistent null enrichment pattern across scales (conditional); honest limitations.
- **Recommended experiments (9):** 100+ permutations at both scales (~18 h or ~4.5 h at 35K, ~2.4 h at 1K; ~$50–100), controlled K_max experiment at min_count=8, GO-hierarchy-aware K, complete the 35K campaign, name the 8 proteins, closed/maximal counts, quantify multi-GPU error, per-K SON miss rate, taxonomic stratification.
- **Verdict:** **Major revision required.** "The paper is stronger with the 35K results, but the null model has regressed from weak (5 perms) to indefensible (2 perms)"; would move to minor revision if 100+ permutations are run and the K-max drop is explained.

---

## SECTION D — TOTALS BY CATEGORY AND STANCE

### D.1 Rows per file

| file | rows |
|---|---|
| F1 `PAPER_V2_REVIEW.md` | 327 |
| F2 `peer_review_jul12.md` | 163 |
| F3 `review_b1_hostile.md` | 345 |
| **all** | **835** |

### D.2 Totals by category (all files, then per file)

| category | all | F1 | F2 | F3 |
|---|---|---|---|---|
| deterministic | 474 | 210 | 76 | 188 |
| hardware-dependent | 91 | 22 | 13 | 56 |
| method-parameter | 175 | 59 | 46 | 70 |
| external-fact | 85 | 36 | 18 | 31 |
| software | 10 | 0 | 10 | 0 |
| **total** | **835** | 327 | 163 | 345 |

### D.3 Totals by stance (all files, then per file)

| stance | all | F1 | F2 | F3 |
|---|---|---|---|---|
| quotes-paper | 338 | 171 | 108 | 59 |
| asserts-own | 372 | 89 | 26 | 257 |
| disputes | 46 | 25 | 11 | 10 |
| requests | 79 | 42 | 18 | 19 |
| **total** | **835** | 327 | 163 | 345 |

### D.4 Category × stance cross-tabulation (all files)

| category \ stance | quotes-paper | asserts-own | disputes | requests | total |
|---|---|---|---|---|---|
| deterministic | 217 | 185 | 31 | 41 | 474 |
| hardware-dependent | 33 | 50 | 3 | 5 | 91 |
| method-parameter | 70 | 65 | 10 | 30 | 175 |
| external-fact | 11 | 69 | 2 | 3 | 85 |
| software | 7 | 3 | 0 | 0 | 10 |
| **total** | 338 | 372 | 46 | 79 | **835** |

Section B contains **63** reviewer-flagged inconsistency rows plus **13** extractor-noted cross-file observations (clearly separated; not reviewer-flagged).

