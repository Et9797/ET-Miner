# Agent B2: Critical Experimental Result Verification Report

**Reviewer:** Agent B2 (Critical Experimental Result Assessor)
**Date:** 2026-02-20
**Scope:** All quantitative claims in the ET-miner paper (`papers/et_miner_proteome.tex`)
**Data sources audited:**
- `archived/alphafold/results_214m/itemsets_214m_godmode.parquet` (26.8M itemsets)
- `archived/alphafold/results_214m/item_mapping_214m.parquet` (feature mapping)
- `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json`
- `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json`
- `applications/alphafold/results_214m/logs/godmode_mining.log`
- `/mnt/d/alphafold-data/transactions_214m.parquet` (source transactions)

---

## 1. Support Threshold Consistency

### Godmode (main result): min_count=8

| Parameter | Value | Source |
|-----------|-------|--------|
| n_transactions | 76,890,945 | godmode log line 2 |
| min_count | 8 | godmode log line 2 |
| min_support | 1.04e-07 (~0.00001%) | 8 / 76,890,945 |

### Direct vs SON experiment: min_count=768

| Parameter | Value | Source |
|-----------|-------|--------|
| n_transactions | 76,890,945 | JSON `parameters.n_transactions` |
| min_count | 768 | JSON `parameters.min_count` |
| min_support | 1e-05 (0.001%) | JSON `parameters.min_support` |

### Null model: min_count=769

| Parameter | Value | Source |
|-----------|-------|--------|
| n_transactions | 76,890,945 | JSON `parameters.n_transactions` |
| min_count | 769 | JSON `parameters.min_count` |
| min_support | ~1e-05 (0.001%) | JSON `parameters.min_support` |

**VERIFIED** -- The godmode run used min_count=8 as claimed. The experiments used min_count=768-769 (0.001%). The paper correctly states these are different thresholds in different table rows and footnotes.

**RISK FLAG** -- The null model and direct-vs-SON experiments operate at a 96x stricter threshold than the godmode run. The "a fortiori" argument connecting them is flawed (see item 6).

---

## 2. Protein Count Accuracy

**Claim:** 76.9M multi-feature proteins (after dedup), from 205.6M total.

**Verification from** `/mnt/d/alphafold-data/transactions_214m.parquet`:

| Metric | Paper | Actual | Match |
|--------|-------|--------|-------|
| Total proteins | 205,620,298 | 205,620,298 | VERIFIED |
| Multi-feature (>1 item) | 76,890,945 | 76,890,945 | VERIFIED |
| Percentage multi-feature | 37.4% | 37.39% | VERIFIED |
| Single-feature | 128.7M (62.6%) | 128,729,353 (62.61%) | VERIFIED |

**VERIFIED** -- All protein counts match exactly. The paper correctly reports 76,890,945 multi-feature proteins (37.4%) from 205,620,298 total.

---

## 3. Dedup Rate

**Claim (PROJECT_STATE.md only):** "Dedup affected 20.45%"

This claim is NOT in the paper. It refers to the null model permutation test, where after shuffling items across proteins, ~20.45% of proteins have duplicate items that must be removed. This is a property of the null model construction, not of the original data.

The actual "dedup" statistics for the source data:
- 62.61% of proteins have only 1 feature (excluded from mining)
- 37.39% have >1 feature (used in mining)

**VERIFIED** -- The 20.45% claim is not in the paper and correctly refers to the null model shuffle artifact, not the original data split.

---

## 4. K=22 Validation

### 4a. Existence and uniqueness

| Check | Result |
|-------|--------|
| K=22 itemsets in godmode parquet | **Exactly 1** |
| Items in the itemset | 22 IDs: [1, 13, 23, 507, 509, 510, 513, 519, 529, 533, 569, 632, 651, 677, 766, 784, 795, 850, 887, 917, 965, 966] |
| Support value | 1.04e-07 |
| Estimated protein count | support * 76,890,945 = **8 proteins** |

**VERIFIED** -- Exactly 1 itemset at K=22, with support count >= 8.

### 4b. Feature decode (via item_mapping_214m.parquet)

| Category | Count | Features |
|----------|-------|----------|
| pLDDT | 1 | plddt_mean_med |
| Pfam | 2 | PF00270 (DEAD/DEAH N-term), PF00271 (Helicase C-term) |
| GO (MF) | 8 | GO:0005524, GO:0016787, GO:0000287, GO:0003697, GO:0003724, GO:0003725, GO:0003678, GO:0000978 |
| GO (BP) | 4 | GO:0030154, GO:0045087, GO:0051607, GO:0034605 |
| GO (CC) | 7 | GO:0005737, GO:0005829, GO:0005634, GO:0005739, GO:0030424, GO:0030425, GO:0016607 |
| **Total** | **22** | |

**VERIFIED** -- Feature decode matches Table 5 in the paper exactly. All 22 features confirmed.

### 4c. K distribution from parquet vs log

| K | Parquet | Log | Match |
|---|---------|-----|-------|
| 1 | 1,002 | 1,002 | Yes |
| 9 | 3,529,257 | 3,529,257 | Yes |
| 22 | 1 | 1 | Yes |
| **Total** | **26,849,505** | **26,849,505** | **Yes** |

**VERIFIED** -- Parquet row count matches log exactly: 26,849,505 itemsets.

---

## 5. Speedup Calculation

### Direct vs SON at identical 0.001% support

| Method | Itemsets | Time | Max K |
|--------|----------|------|-------|
| Direct GPU | 475,865 | 50.72s | 14 |
| SON | 22,846 | 1085.6s | 13 |

**Speedup:** 1085.6 / 50.72 = **21.4x** (paper rounds to "21x" in caption text, reports "21.4x" in controlled comparison)

**Fairness check:**
- Same dataset: 76,890,945 transactions -- **SAME**
- Same min_support: 1e-05 (0.001%) -- **SAME**
- Same GPU: yes (both run on H100) -- **SAME**

**SON itemset loss:** 475,865 - 22,846 = 453,019 (95.2% lost) -- **VERIFIED**, matches JSON `comparison.itemset_diff`

**VERIFIED** -- The 21.4x speedup calculation is correct and the comparison is fair.

---

## 6. Null Model Integrity

### 6a. Max K in null

| Run | Max K | K=6 count |
|-----|-------|-----------|
| 1 | 6 | 20 |
| 2 | 6 | 22 |
| 3 | 6 | 23 |
| 4 | 6 | 22 |
| 5 | 6 | 22 |

**VERIFIED** -- Max K = 6 across all 5 permutations. Paper claim matches exactly.

### 6b. K=1 sanity check (marginal preservation)

All 5 null runs produce K=1 = 1,002 (identical to real data). **VERIFIED** -- Marginals perfectly preserved.

### 6c. K>=7 in null

All 5 runs: 0 itemsets at K>=7. **VERIFIED** -- Paper claim "no permutation produced any pattern at K>=7" is correct.

### 6d. Aggregate statistics match

| K | Paper Bio | JSON Bio | Paper Null mu | JSON Null mu | Paper Z | JSON Z |
|---|-----------|----------|---------------|--------------|---------|--------|
| 2 | 22,019 | 22,019 | 63,702 | 63,702.4 | -987 | -987.08 |
| 4 | 108,059 | 108,059 | 25,468 | 25,467.8 | +3,791 | 3790.74 |
| 6 | 78,596 | 78,596 | 22 | 21.8 | +71,728 | 71728.1 |
| 7-14 | 88,745 | 88,745 | 0 | 0 | inf | inf |

**VERIFIED** -- All numbers match between paper and JSON (minor rounding in paper table).

### 6e. Are 5 permutations enough?

**RISK FLAG** -- This is the most significant statistical concern.

**For K=2 through K=6:** The null model is remarkably stable across runs (CV < 5% for all K levels). The Z-scores for K=4 through K=6 are in the thousands to tens of thousands. 5 permutations is adequate to establish that K>=4 patterns are massively enriched.

**For K>=7:** With 0 events in 5 trials, the Clopper-Pearson 95% upper confidence bound on P(K>=7 | null) is **0.451** (45.1%). This means we CANNOT statistically rule out that up to 45% of null runs might produce K>=7 patterns. The Z=infinity and p=0 claims for K>=7 are mathematical artifacts (division by zero in standard deviation), not validated statistical results.

**For robust statistical claims about K>=7:** At least 100 permutations (upper bound: 0.030) or ideally 1000 permutations (upper bound: 0.003) are needed. The paper's wording ("absent from all null runs") is technically accurate but creates a stronger impression than 5 permutations warrants.

**Mitigating factor:** The extreme stability across the 5 runs (K=6 count: 20-23, near-zero variance) strongly suggests the null model's K-distribution is deterministic at this scale, making the K=6 boundary likely real. But this is an inductive argument, not a statistical proof.

### 6f. The "a fortiori" argument

**DISCREPANCY** -- The paper states (line 396):

> "This null model was run at a higher threshold than the main analysis (min_count=769 vs. 8) for computational efficiency; since higher thresholds are strictly more conservative, the biological significance of the Opus K-distribution (K_max=22) follows a fortiori."

This argument is **logically backwards**. At a LOWER threshold (min_count=8), the null model would be more permissive and could potentially produce DEEPER patterns than K=6. The "a fortiori" reasoning requires that the null model be run at a MORE permissive threshold than the experiment, not a STRICTER one.

Specifically:
- Null model at 0.001% (min_count=769): max K=6
- Real data at 0.00001% (min_count=8): max K=22
- If null model were run at 0.00001% (min_count=8): max K = **UNKNOWN** (could be K=7, 8, or higher)

The null model validates the K-distribution at 0.001% (where real extends to K=14 vs null K=6). It does NOT validate the additional K=15 through K=22 patterns found at the lower threshold. The gap between K=14 (at 0.001%) and K=22 (at 0.00001%) is not covered by any null model test.

**Severity:** The core finding (K>=7 is biological at 0.001%) is likely sound. But the paper's specific claim that K=22 significance "follows a fortiori" from the K=6 null ceiling is an invalid logical step.

---

## 7. Timing Claims

| Claim | Evidence | Verification |
|-------|----------|--------------|
| Godmode: 7.3 min | Log: "440.5s" | 440.5 / 60 = 7.34 min -- **VERIFIED** |
| Null total: 662s | JSON: 662.17s | Sum of 5 runs: 99.22+135.62+141.97+143.06+142.30 = 662.17s -- **VERIFIED** |
| SON: 18.1 min | JSON: 1085.6s | 1085.6 / 60 = 18.09 min -- **VERIFIED** |
| Direct GPU: 50.7s | JSON: 50.72s | **VERIFIED** |

All timing claims match their evidence.

---

## 8. Annotation Artifact Risk in K=22

### 8a. InterPro2GO automatic mapping

Checked via InterPro REST API:
- **IPR011545** (integrates PF00270 DEAD/DEAH helicase) maps to:
  - GO:0003676 (nucleic acid binding) -- NOT in K=22 set
  - **GO:0005524 (ATP binding)** -- IS in K=22 set
- **IPR001650** (integrates PF00271 Helicase C-terminal) maps to:
  - No GO terms

**Result:** Only **1 of 19 GO terms** (GO:0005524, ATP binding) is confirmed auto-derived from Pfam via InterPro2GO.

### 8b. Other electronic annotation sources

Beyond InterPro2GO, UniProt assigns GO terms via:
- UniRule/HAMAP/PIRNR (IEA evidence code)
- Ortholog transfer (IBA evidence code)
- Literature curation (EXP/IDA/IPI etc.)

The item_mapping_214m.parquet does not distinguish GO evidence codes, so we cannot determine how many of the 19 GO terms are experimentally validated vs. electronically inferred. For DEAD/DEAH box helicases, several MF terms (helicase activity GO:0003724, DNA helicase GO:0003678, dsRNA binding GO:0003725) are biologically plausible as independently curated annotations, not merely auto-propagated from the domain.

### 8c. GO hierarchy (true-path rule)

The paper acknowledges (Limitation 2) that the GO true-path rule creates definitional co-occurrence. However, examining the 19 GO terms:

**No parent-child pairs exist in the K=22 set.** This was verified in a prior analysis (PROJECT_STATE.md: "GO hierarchy: 0 parent-child pairs -> Independent K = 22"). The GO terms span different sub-ontologies (MF, BP, CC) and unrelated branches within each.

### 8d. Summary assessment

| Risk factor | Severity | Notes |
|-------------|----------|-------|
| InterPro2GO auto-mapping | LOW | Only 1/19 GO terms confirmed auto-derived |
| GO hierarchy inflation | LOW | 0 parent-child pairs in K=22 |
| IEA evidence mixing | MEDIUM | Cannot verify evidence codes from available data |
| Pfam-GO functional coupling | MEDIUM | Helicases naturally associate with helicase-related GO terms, but this IS the biology being discovered |

**RISK FLAG** -- The GO:0005524 (ATP binding) is almost certainly auto-derived from PF00270 via InterPro2GO, creating 1 definitional (non-independent) co-occurrence. This reduces the effective independent feature count from 22 to 21. The paper should note this but it does not materially change the finding.

The broader risk of IEA annotation circularity is acknowledged in the paper (line 448: "electronically propagated GO annotations may create artificial co-occurrence") and limitations (item 2: true-path rule). The paper handles this appropriately.

---

## 9. Feature Vocabulary Breakdown (Table 1)

**DISCREPANCY** -- The paper's Table 1 claims:

| Feature Type | Paper claims | Actual (from parquet) |
|-------------|-------------|----------------------|
| Pfam domains | 247 | **500** |
| GO terms (MF) | 302 | **cannot split** |
| GO terms (BP) | 289 | **cannot split** |
| GO terms (CC) | 161 | **cannot split** |
| GO terms (total) | 752 | **500** |
| pLDDT bins | 3 | **2** (frequent) |
| **Total** | **1,002** | **1,002** |

The total matches (1,002 frequent features), but the breakdown is wrong:
- Actually 500 Pfam + 500 GO + 2 pLDDT = 1,002
- The item_mapping_214m.parquet contains 1,006 entries (500 pfam + 500 go_term + 6 plddt), of which 4 are not frequent at min_count=8 (plddt_mean_low + 3 plddt_fraction items)
- The paper says 3 pLDDT bins but only 2 are frequent (plddt_mean_med and plddt_mean_high; plddt_mean_low has < 8 proteins)

**Severity:** This is an error in Table 1. The total is correct but the per-category breakdown is fabricated or refers to a different version of the data. The actual ratio is 500:500:2, not 247:752:3.

---

## Summary

| # | Check | Verdict | Notes |
|---|-------|---------|-------|
| 1 | Support threshold consistency | VERIFIED | Different thresholds correctly labeled |
| 2 | 76.9M protein count | VERIFIED | Exact match: 76,890,945 |
| 3 | 20.45% dedup rate | VERIFIED | Not in paper; correctly refers to null model |
| 4 | K=22 existence & features | VERIFIED | 1 itemset, 8 proteins, 22 features decoded |
| 5 | 21.4x speedup | VERIFIED | Fair comparison, correct calculation |
| 6a | Null model max K=6 | VERIFIED | All 5 runs confirm |
| 6b | Null K=1 sanity | VERIFIED | All 1,002 features preserved |
| 6c | 5 permutations sufficiency | **RISK FLAG** | Cannot bound P(K>=7|null) below 0.45 with only 5 trials |
| 6d | "A fortiori" argument | **DISCREPANCY** | Logically backwards: lower threshold is more permissive, not more conservative |
| 7 | 440.5s / 7.3 min timing | VERIFIED | Exact match |
| 8 | InterPro2GO artifact risk | **RISK FLAG** | 1/19 GO terms confirmed auto-derived (GO:0005524) |
| 9 | Table 1 feature breakdown | **DISCREPANCY** | 500:500:2, not 247:752:3 |

### Critical Issues (must fix before submission)

1. **Table 1 feature breakdown is wrong.** The actual composition is 500 Pfam + 500 GO + 2 pLDDT = 1,002, not 247 Pfam + 752 GO + 3 pLDDT = 1,002. Either regenerate the correct breakdown (splitting GO into MF/BP/CC sub-categories) or update Table 1 to match reality.

2. **The "a fortiori" argument (line 396) is logically invalid.** The null model was run at a STRICTER threshold (min_count=769) than the godmode run (min_count=8). A stricter null test does not validate a more permissive real-data run. The sentence should be revised to accurately state: the null model validates K-distribution significance at 0.001% support (K_max=14), and the K=15-22 patterns found at lower support require separate validation or should be presented as exploratory.

### Recommended Fixes (improve rigor)

3. **Run null model at 0.00001% threshold (min_count=8)** to properly validate the K=22 result. If this is computationally too expensive, acknowledge that the K=15-22 significance is extrapolated rather than tested.

4. **Increase null model to 100+ permutations** for the 0.001% threshold to enable proper confidence interval on P(K>=7|null).

5. **Note GO:0005524 auto-derivation** in the K=22 discussion. One sentence: "GO:0005524 (ATP binding) is automatically mapped from PF00270 via InterPro2GO, reducing the independent feature count to 21."

### Numbers That Check Out

- 26,849,505 total itemsets
- 76,890,945 multi-feature proteins
- 205,620,298 total proteins
- 21.4x speedup (Direct GPU vs SON)
- 440.5s = 7.3 min godmode timing
- 662s null model total
- 88,745 itemsets at K>=7 (at 0.001%)
- K=22: exactly 1 itemset, 8 proteins, 22 features
- All Z-scores for K=2 through K=6
- All null model per-run K-distributions
