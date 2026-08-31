# Senior Review: "Protein Structural Motif Discovery at AlphaFold Scale"

**Reviewer:** Opus 4.6 Senior Reviewer Agent
**Date:** 2026-03-23
**Verdict:** Major Revisions Required

---

## CRITICAL ISSUES (Must Fix — Paper Will Be Rejected If Not Addressed)

### C1. Table 1 Feature Breakdown Is Fabricated

**Lines 143-145:**
```
Pfam domains & InterPro/Pfam & 247 & --- \\
GO terms & Gene Ontology & 752 & 5,763 \\
pLDDT confidence bins & AlphaFold & 3 & 6 \\
```

The paper claims Run 1 consists of 247 Pfam + 752 GO + 3 pLDDT = 1,002.

**The actual composition is 500 Pfam + 500 GO + 2 pLDDT = 1,002** (verified from direct parquet inspection of `item_mapping_214m.parquet`). Pipeline was run with `top_pfam=500, top_go=500` (non-default parameters). The numbers 247 and 752 do not correspond to any configuration in the codebase. Agent B2's review flagged this on February 20, 2026 — over a month ago, never corrected.

### C2. The "A Fortiori" Argument Is Logically Inverted

**Line 438:**
> "This null model was run at a higher threshold than the main analysis (min_count=769 vs. 8) for computational efficiency; since higher thresholds are strictly more conservative, the biological significance of the Opus K-distribution follows a fortiori."

A stricter null test does NOT validate a more permissive real-data run. The null model at min_count=769 had an easier job (fewer patterns needed to be significant). The real data at min_count=8 can find patterns the null model was never tested against. Fix: either run the null model at min_count=8, or honestly state that K=15-22 significance is extrapolated.

### C3. "GPU-Resident" Claim Contradicts Own Council Findings

The paper claims (line 162, 189, throughout): "all operations entirely on-GPU" and "GPU-resident path."

**PROJECT_STATE.md records the Council of Copii finding:**
> Paper claim: "GPU-accelerated" is CORRECT, "GPU-resident" is INCORRECT (candidate gen still CPU)

Candidate generation via prefix grouping happens on CPU (`gpu_dispatch.py:128-131`). The `build_prefix_groups_gpu` function exists but is NOT connected to the multi-GPU row-split path. The paper should say "GPU-accelerated" — the dominant cost (support counting) is GPU, but candidate generation is CPU-side.

### C4. Pruning Techniques Are Projected, Not Measured

Table 8 presents pruning savings of 68%, 12%, 4%, 2% (combined 73%) for K=9. PROJECT_STATE confirms these are projections — the production run with pruning has never completed. The abstract says "reduce projected K=9 runtime by 73%" without qualifying this as an estimate.

---

## IMPORTANT ISSUES (Should Fix — Weakens Paper Significantly)

### I1. Run 1 vs Run 2 Feature Semantics Change Not Discussed

Table 1 shows Run 2 has "---" for Pfam domains, replaced by 12,773 InterPro entries. Pfam was the backbone of Run 1's biological interpretations. InterPro includes Pfam as a member database but also PROSITE, PRINTS, HAMAP, etc. Run 1 and Run 2 results are NOT directly comparable, yet presented side-by-side in K-distribution comparisons (Figure 3).

### I2. Scale Comparison 769× Claim Arithmetic Unclear

**Line 456:** "ET-miner processes 769x more transactions than the next largest GPU system."

The 769× appears to compare against CPU systems at 100K transactions. BIGMiner processes 100M transactions — MORE than ET-miner's 76.9M. GMiner is listed at 15M (76.9M/15M = 5.1×, not 769×). The caption needs to specify exactly what comparison produces this ratio.

### I3. Closed Itemset "Pruning" Is Post-Processing

**Line 528:** "ET-miner implements closed itemset pruning"

Looking at `apriori.py:235-268`, `prune_equal_support` is post-processing on already-mined itemsets, not pruning during mining (which would reduce computational cost). The phrasing implies an integrated mining optimization.

### I4. 5 Permutations Is Statistically Weak

The null model uses only 5 permutations (line 410). Cannot bound P(K≥7|null) below 0.17 with only 5 trials. Z-scores for K≥7 reported as "+∞" — misleading with zero observations. Can only say "0 out of 5 trials produced K≥7."

### I5. pLDDT Bin Count Discrepancy

Table 1 claims 3 pLDDT bins for Run 1. Only 2 are frequent at min_count=8: `plddt_mean_med` and `plddt_mean_high`. The item mapping defines 6 bins but 4 have < 8 proteins.

### I6. Run 2 Numbers Not Re-Verified Post Bug Fix

K=1 through K=8 numbers for Run 2 come from a run with the NCCL int32 bug present (fixed later in commit f8c56cb5). PROJECT_STATE argues correctness because per-GPU counts < 2^31, but this was not independently verified by re-running.

---

## CITATIONS

Spot-checked sample: Jumper 2021, Agrawal 1994, Han 2000, Chon 2018, Abramson 2024, Naulaerts 2015 — all verified correct. No hallucinated references found.

---

## BOTTOM LINE

The technical work is genuinely impressive. 16.8 billion itemsets across 109M proteins is a real achievement. The K=22 finding is compelling. But the paper contains known errors (C1, C2, C3) that were flagged internally over a month ago and never corrected. Table 1's fabricated feature counts will destroy credibility if a reviewer checks the open-sourced data.

**Required actions before submission:**
1. Fix Table 1: 500/500/2
2. Fix "GPU-resident" → "GPU-accelerated"
3. Fix or remove "a fortiori" argument
4. Qualify pruning savings as projections in abstract
5. Discuss Run 1 vs Run 2 feature semantics change
6. Increase null model permutations or qualify p-value claims
