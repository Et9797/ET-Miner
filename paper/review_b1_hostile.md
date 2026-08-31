# Hostile Peer Review: "Protein Structural Motif Discovery at AlphaFold Scale"

**Reviewer:** Agent B1 (Senior Paper Reviewer, Nature Methods simulation)
**Date:** 2026-02-21 (updated with 35K-feature results)
**Recommendation:** Major revision required

**Review revision history:**
- v1 (2026-02-20): Initial review based on 1K-feature results
- v2 (2026-02-21): Updated with 35K-feature expansion (606K itemsets, K_max=17, null model, SON infeasibility)

---

## EXECUTIVE SUMMARY OF 35K RESULTS

The authors have expanded from 1,002 features to 34,920 features (35x vocabulary expansion) and produced three new result files:

1. **Direct GPU triplicate** (`experiment_direct_vs_son_35k_20260221.json`): 606,292 mean itemsets (std=28), K_max=17, 654s mean on 4xH200. SON is INFEASIBLE (175 GB bitvec > 143 GB VRAM).
2. **Null model** (`experiment_null_model_20260221_003203.json`): 2 permutations. Null peaks at K=2 (59K itemsets), collapses by K=5 (1 itemset). Biological data extends to K=17. 5.45x total enrichment ratio. Z=723 at K=3, Z=20,585 at K=4.
3. **Wave 3 partial** (`wave3_base_partial.log`): Mining at 0.1% support reaches K=10+ with millions of itemsets per level, still running after 743s at K=9 (10M itemsets at that level alone).

These results both strengthen and weaken the paper's claims in specific ways analyzed below.

---

## MAJOR CONCERNS (any one of these would justify rejection)

### M1. Two permutations is worse than five -- the null model has regressed [Severity: 9/10]

The original 1K-feature null model used 5 permutations. The 35K-feature null model uses **only 2 permutations** (seeds `7138484576005690180` and `4047939128787533792`, `experiment_null_model_20260221_003203.json` line 8). This is not a step forward; it is a step backward.

**Specific problems:**

1. **Variance from n=2 is meaningless.** With 2 data points, you have exactly 1 degree of freedom. The reported standard deviations (e.g., K=2: std=123.7; K=3: std=51.6; K=4: std=2.8) have confidence intervals spanning effectively zero to infinity. The "Z-score" at K=3 of 722.91 is computed as `(53235 - 15919.5) / 51.6` -- but the denominator is estimated from 2 observations. A Z-statistic from n=2 is a t-statistic with 1 degree of freedom, which has no finite moments and catastrophically heavy tails. Reporting it as a Z-score is statistically invalid.

2. **At K=5, both null runs produced exactly 1 itemset.** The standard deviation is 0.0 from n=2 identical values. The z_score is reported as `"inf"` (a string, not a float -- line 104). This is a division-by-zero artifact presented as statistical evidence. With 10 or 100 permutations, the K=5 null count might range from 0 to 5, producing a finite and meaningful Z-score. With n=2 you cannot know.

3. **The null model was again run at a different threshold than the main results.** The 35K null uses min_count=1,090 (approximately 0.001% of 109M transactions, line 6-7). The 35K mining campaign uses min_count=1,093 (0.001%, line 9 of the direct_vs_son file). These are close but not identical, suggesting separate computation rather than exact parameter matching. More critically, neither matches the paper's Opus threshold (min_count=8). The same *a fortiori* hand-wave applies.

4. **The enrichment shift from 1K to 35K is interesting but unexplored.** In the 1K-feature null model (5 perms), K=2 was depleted (Z=-987) and K=3 was depleted (Z=-143). In the 35K-feature null model (2 perms), K=2 is again depleted (Z=-102) but K=3 is now **massively enriched** (Z=+723). This reversal is biologically significant: with 35K features, the combinatorial space is vastly larger, and the null model can no longer produce even K=3 patterns at biological rates. Yet the paper makes no mention of this shift, because the 35K results are not yet in the manuscript. If presented, this cross-scale comparison would strengthen the biological argument substantially -- but it requires adequate permutation counts to be credible.

5. **The deduplication problem from v1 of this review remains unaddressed.** The Fisher-Yates shuffle on the feature column creates duplicate features within proteins, which are then removed. With 35K features, the duplication rate should be lower (larger feature pool), but this is not documented or controlled.

**What would fix this:** Run 100+ permutations at min_count=1,093 on the 35K dataset. At ~655s per permutation on 4xH200, 100 permutations = ~18 hours, 200 = ~36 hours. This is expensive but not prohibitive for a Nature Methods submission. The 1K-feature null model (5 perms, ~86s/perm) should also be expanded to 100+ perms. Report proper confidence intervals using the t-distribution, not Gaussian Z-scores. Fix the `"inf"` string to a proper sentinel value.

### M2. GO true-path inflation: partially addressed by the K-drop but still unquantified [Severity: 7/10]

The K_max drop from 22 (1K features) to 17 (35K features) is striking. One might expect that expanding the feature vocabulary from 1K to 35K would enable *deeper* co-occurrence patterns, not shallower ones. The most likely explanation: the 35K vocabulary includes more granular GO terms that are NOT redundant ancestors of each other, while the 1K vocabulary concentrated on high-level terms where GO hierarchy inflation was maximal.

**This is actually good news for the paper** -- it suggests that the 35K K-distribution is less inflated by GO hierarchy. But the paper must:

1. **Quantify the GO hierarchy contribution in both vocabularies.** What fraction of the 1K features are GO ancestor-descendant pairs? What fraction of the 35K features? If the 1K vocabulary has 30% redundant pairs and the 35K has 5%, this explains the K-drop and validates the 35K results as more biologically meaningful.

2. **The original K=22 concerns remain.** The K=22 itemset from the 1K-feature analysis contains verified GO hierarchy inflation (cytoplasm/cytosol, cytoplasm/mitochondrion, nucleus/nuclear speckle -- see v1 review M2). The corrected independent K is approximately 19-20. If the paper retains the 1K-feature results (which it should, as the Opus run), this must be addressed.

3. **The 35K K=17 itemset needs the same scrutiny.** What are the 17 co-occurring features in the K=17 pattern? Are any GO ancestor-descendant pairs? What is the corrected independent K? Without this analysis, the 35K ceiling is as suspect as the 1K ceiling.

### M3. SON comparison: from "unfair strawman" to "computationally fundamental" -- but needs proper framing [Severity: 5/10, improved from 8/10]

**The 35K results transform the SON narrative.** At 35K features, SON does not merely underperform -- it is **physically impossible** to execute:

> "40M chunk x 35K features = 175 GB bitvec exceeds 143 GB H200 VRAM" (experiment JSON, son_status: `FAILED_OOM_HUNG`)

This is no longer a strawman comparison; it is a fundamental scaling limit. SON's bitvector size scales as O(chunk_size x n_features), and with 35K features, even a single chunk exceeds the memory of the highest-end GPU available (NVIDIA H200 at 143 GB). No parameter tuning can fix this -- you would need chunk_size < 33M, which at 109M transactions means 4+ chunks with proportionally worse miss rates.

**However, this needs proper framing:**

1. **SON with smaller chunks could still run.** chunk_size=20M x 35K features = ~87 GB, fitting on an H200. This would create 6 chunks with terrible miss rates, but it would be technically feasible. The paper should show this is not just OOM but fundamentally broken by miss rate analysis at realistic chunk sizes.

2. **The SON infeasibility at 35K is a STRONGER argument than the 1K 95.2% miss rate.** The paper currently presents the 1K comparison (Section 3.1). If the 35K results are incorporated, the argument becomes: "SON is not merely slow and lossy -- at biologically relevant feature scales, it is physically impossible." This is a much stronger claim.

3. **The per-K miss rate analysis from v1 (concern M3.3) can now only be done for the 1K comparison**, since 35K SON never completed. The 1K per-K analysis should still be presented.

4. **The triplicate runs at 35K are well-done.** Three Direct GPU runs with mean=606,292, std=28, CV=0.0046% demonstrates excellent reproducibility. The slight variation in K_max (17, 16, 17 across runs) at K=17 (with 2, 0, 1 itemsets respectively) suggests the boundary itemsets are at the noise floor of the support threshold. This is expected but should be noted.

### M4. The K-max drop (K=22 -> K=17) demands explanation [Severity: 8/10]

This is a **new major concern** not present in v1. When expanding from 1,002 to 34,920 features:
- K_max dropped from 22 to 17
- The K-distribution peak shifted from K=9 (1K) to K=7 (35K)
- Total itemsets dropped from 26.8M (at min_count=8) to 606K (at min_count=1,093)

The third point is explained by the higher support threshold (1,093 vs 8), but the first two are counterintuitive and require explanation. Consider:

**Hypothesis A: Support threshold effect.** The 35K experiment uses min_count=1,093 while the 1K Opus run uses min_count=8. Higher thresholds prune rare deep patterns. The K=22 pattern in the 1K run was supported by exactly 8 proteins -- it would be invisible at min_count=1,093. This is likely the dominant explanation, but the paper must present it explicitly.

**Hypothesis B: Feature dilution.** With 35K features, each feature is present in fewer proteins on average (assuming the same 109M transactions). Sparser features produce sparser bitvectors, reducing the probability of deep co-occurrence. This would explain why the K-distribution shifts left even after controlling for support threshold.

**Hypothesis C: GO hierarchy collapse.** The 35K vocabulary may include more specific GO terms that replace the broad ancestor terms in the 1K vocabulary. Replacing `cytoplasm` with `cytoplasmic vesicle membrane` removes a term that appeared in millions of proteins, reducing deep co-occurrence potential.

**Hypothesis D: Multi-GPU row-splitting artifact.** The 35K runs use 4-GPU row-splitting with `local_min_count=274` (line 2 of power_test_v1_full.log). If a global pattern is distributed across GPUs such that no single GPU sees 274 supporting proteins, it will be missed in the locally-frequent union step. This is a subtle form of the SON problem reintroduced at the GPU level. The wave3 log (line 26: "K=2: 14,800 locally frequent (union across GPUs), 8,554 frequent") shows that 42% of locally-frequent K=2 pairs fail the global recount. At higher K, this filter could be more aggressive.

**The paper MUST address which of these hypotheses explains the K-max drop.** A controlled experiment would run the 35K features at min_count=8 on a single GPU (if VRAM allows) to isolate the threshold effect from the multi-GPU and feature-dilution effects.

### M5. 8 vs 3 K=22 proteins -- still unresolved [Severity: 7/10]

This concern from v1 is unchanged. The 35K expansion does not address it because the K=22 pattern comes from the 1K-feature analysis. The discrepancy between 8 proteins (mining result support count) and 3 proteins identified (analysis script fallback heuristic) remains unresolved. All 8 UniProt accessions must be named and verified.

### M6. Wave 3 partial results hint at massive scale but are incomplete [Severity: 6/10]

The `wave3_base_partial.log` shows mining at 0.1% support (min_count=109,225) reaching K=10 with the following per-level counts:

| K | Itemsets | Time (cumulative) |
|---|---------|-------------------|
| 1 | 1,164 | 0.3s |
| 2 | 8,554 | 1.2s |
| 3 | 28,804 | 2.0s |
| 4 | 88,561 | 5.3s |
| 5 | 280,784 | 17.5s |
| 6 | 835,466 | 52.1s |
| 7 | 2,207,022 | 145.3s |
| 8 | 5,063,845 | 348.5s |
| 9 | 10,041,611 | 743.4s |
| 10 | ??? (12.5M locally frequent) | >743s, still running |

This is **30x more itemsets per K-level than the 0.001% run** (e.g., K=7: 2.2M vs 64K) and the distribution is still growing at K=10. The log cuts off, suggesting the run was terminated or is ongoing. Two concerns:

1. **If this run completes, it could produce hundreds of millions of itemsets at 35K features.** This would demonstrate that ET-miner's architecture scales to feature-rich vocabularies, but the runtime (already 12+ minutes at K=9) suggests the full campaign may take hours per threshold.

2. **The power test logs** (`power_test_v1_full.log`, `power_test_v2.log`) show mining at min_count=1,093 with K=2 taking 586s alone (vs 1.2s at min_count=109,225). This is because the lower threshold admits 902K frequent pairs vs 8.5K, creating a vastly larger candidate space at K>=3. The power test reached K=4 with 17.4M itemsets before apparently stalling. This raises questions about whether the full 35K campaign at low support thresholds is computationally feasible.

**The paper should either:** (a) complete the wave 3 campaign and report results, or (b) explicitly state which threshold-feature combinations are computationally feasible and which are not. Presenting 606K itemsets at 0.001% support when the 0.1% run alone produces 10M+ itemsets at K=9 undersells the system's capabilities.

---

## MAJOR CONCERNS RETAINED FROM V1 (status updated)

### M1-v1. Five permutations was statistically indefensible; two is worse [ESCALATED to M1 above]

Status: **Worsened.** The 35K null model uses 2 permutations instead of the 1K model's 5. See M1 above.

### M2-v1. GO true-path inflation [RETAINED as M2 above, severity reduced]

Status: **Partially mitigated by the K-drop.** The K=22 -> K=17 drop suggests less GO inflation at 35K features, but this is inferential, not demonstrated. See M2 above.

### M3-v1. The SON comparison is a strawman [TRANSFORMED, severity reduced]

Status: **Transformed from strawman to fundamental impossibility.** SON is not merely slow at 35K -- it cannot physically execute. This actually STRENGTHENS the paper if properly framed. See M3 above.

### M4-v1. 26.8M itemsets data provenance gap [RETAINED, unchanged]

Status: **Unchanged.** The 35K results have proper triplicate runs with JSON output and consistent naming. The 1K results still have the naming inconsistency (`godmode` vs `Opus`). The 7.3-minute claim (mining time vs total pipeline time) is still ambiguous.

### M5-v1. 8 vs 3 K=22 proteins [RETAINED as M5 above]

Status: **Unchanged.** The 35K expansion does not address this. See M5 above.

---

## MINOR CONCERNS

### m1. Triplicate reproducibility reveals boundary instability [Severity: 3/10]

The three 35K runs show minor variation in total itemsets (606,319 / 606,293 / 606,263) and K_max (17 / 16 / 17). The K=17 level had 2, 0, and 1 itemsets across runs. The K=16 level is stable at 152 across all runs. This means the K=17 boundary is at the noise floor -- patterns supported by exactly the threshold count that appear or disappear with multi-GPU row-splitting randomness or floating-point non-determinism.

**Recommendation:** Report K_max as "16-17" with a note that K=17 patterns are at the support boundary. Alternatively, investigate whether the variation comes from multi-GPU non-determinism (GPU row assignment, atomic operation ordering) or genuine threshold boundary effects.

### m2. The 35K feature vocabulary composition is undocumented [Severity: 5/10]

The paper documents the 1K feature vocabulary in Table 1 (247 Pfam, 302 GO:MF, 289 GO:BP, 161 GO:CC, 3 pLDDT). What are the 34,920 features? The logs show K=1: 34,920 frequent items, suggesting all 34,920 are above the min_count threshold. Questions:

1. What feature types were added? More granular GO terms? InterPro domains beyond Pfam? EC numbers? Taxonomic features?
2. What is the frequency distribution of the 35K features? If most new features appear in only 1,093-2,000 proteins, they contribute to K=2 pairs but not deep patterns.
3. Does the 35K vocabulary include all 1,002 original features? If so, the 1K results should be a strict subset of the 35K results at the same threshold. Is this verified?

### m3. Multi-GPU row-splitting introduces a new approximation [Severity: 5/10]

The 35K results use 4-GPU row-splitting (`wave3_base_partial.log` line 18: `local_min_count=27,307` for global `min_count=109,225`). This is a different architecture than the single-GPU direct mining used for the 1K results. The "locally frequent union" step (e.g., line 25: "14,800 locally frequent (union across GPUs), 8,554 frequent") shows that the local union over-estimates frequency, requiring a global recount that prunes 42% of candidates.

This is essentially SON at the GPU level -- each GPU sees 1/4 of the transactions and applies a reduced local threshold. The same cascade effect that caused SON to miss 95.2% of patterns in the 1K analysis could be active here, though mitigated by the lower local factor (local_min_count = global/4 = exact proportional split).

**Recommendation:** Quantify the multi-GPU approximation error. Run a single-GPU baseline at the same threshold (if memory permits) and compare K-distributions.

### m4. Feature vocabulary expansion justification missing [Severity: 4/10]

Why 35K features? The jump from 1,002 to 34,920 is not a round number and is not motivated in any documentation. Presumably the 1K vocabulary used a frequency cutoff that admitted only common features, and the 35K vocabulary uses a lower cutoff. But:

1. Are there redundant features in the 35K set? (E.g., `GO:0005515 protein binding` which annotates ~60% of all proteins and creates millions of trivial co-occurrences)
2. Was any feature selection or redundancy filtering applied?
3. What is the median feature frequency? If median frequency < 2x min_count, most features are barely above the noise floor.

### m5. The "21.4x speedup" from v1 is now secondary [Severity: 2/10, reduced from 4/10]

The 1K SON comparison (21.4x speedup, 95.2% miss rate) is now secondary to the 35K result where SON is physically impossible. If the paper presents both, the narrative arc is clear: SON is merely bad at 1K features and completely infeasible at 35K. But the 1K comparison should still report triplicate runs and per-K miss rates.

### m6. The "bell curve" terminology [RETAINED from v1, severity 3/10]

The 35K K-distribution peaks at K=7 (64,400 itemsets) and has a different shape than the 1K distribution (peak at K=9). Neither is formally fitted to a parametric distribution. The term "bell curve" should be replaced with "unimodal distribution" or simply described as "peaking at K=N."

### m7. Intermediate biological discoveries lack quantitative validation [RETAINED from v1, severity 3/10]

The approximate protein counts ("~611 proteins," "~11,000 proteins") should be exact.

### m8. No closed/maximal itemset analysis [RETAINED from v1, severity 4/10]

The 606K itemsets at 35K features likely include massive redundancy. How many closed frequent itemsets? How many maximal? This is especially important at 35K where feature redundancy (multiple GO terms for the same process) could inflate counts.

### m9. Title claims "AlphaFold Scale" but mines annotations [RETAINED from v1, severity 3/10]

Unchanged. The 35K expansion adds more annotations, not structural features.

### m10. No comparison with cuML/RAPIDS [RETAINED from v1, severity 2/10]

Unchanged.

### m11. Multi-GPU claims are now substantiated but incompletely [Severity: 3/10, improved from 5/10]

The 35K results use 4xH200 GPUs, providing the first multi-GPU evidence. However, no single-GPU baseline exists for the 35K feature set, so the multi-GPU scaling factor is unknown. The bitvector build time (8.9-12.9s across runs) and per-level times are reported, but speedup vs 1 GPU is not.

### m12. Runtime comparison across scales is inconsistent [Severity: 4/10]

| Config | Features | Transactions | Threshold | GPUs | Time | Itemsets |
|--------|----------|-------------|-----------|------|------|---------|
| 1K Opus | 1,002 | 76.9M | 0.00001% (8) | 1xH100 | 7.3 min | 26.8M |
| 35K Base | 34,920 | 109.2M | 0.001% (1,093) | 4xH200 | 10.9 min | 606K |

The 35K run uses 4x more GPUs, 100x higher threshold, and still takes 50% longer. This is expected (35x more features = 35x larger bitvectors = memory-bandwidth bound), but the paper should present a normalized comparison. Itemsets-per-second, or itemsets-per-GPU-second, would enable fair cross-scale comparison.

---

## STRENGTHS (updated with 35K evidence)

### S1. The CSR-to-bitvector pipeline is elegant engineering [RETAINED, strengthened]

The architecture scales from 1K to 35K features without modification. The 35K bitvector matrix is ~119 GB per GPU (line 3 of power_test logs), requiring 4-GPU row-splitting. The system automatically handles this, demonstrating genuine architectural flexibility.

### S2. The K-distribution is now a multi-scale result [ENHANCED]

With both 1K and 35K K-distributions, the paper can make a much stronger claim: the bell-shaped K-distribution is a *robust structural property* of protein annotation data, not an artifact of a specific vocabulary size. The peak shifting from K=9 (1K) to K=7 (35K) with a characteristic unimodal shape in both cases is genuinely interesting.

### S3. Biological pattern recovery [RETAINED]

Unchanged. The 1K results recover known drug targets and molecular machines.

### S4. Scale is genuine and now multi-dimensional [ENHANCED]

The system has been demonstrated at two vocabulary scales (1K and 35K) and two hardware configurations (1xH100 and 4xH200). The wave 3 partial results suggest mining at 0.1% support can discover tens of millions of itemsets at 35K features, though this run is incomplete.

### S5. SON infeasibility is a genuinely novel finding [NEW]

No prior work has demonstrated that SON streaming is physically impossible at biologically relevant feature scales. The 175 GB bitvec calculation (40M chunk x 35K features) is straightforward and verifiable. This transforms the SON comparison from "we beat the baseline" to "the baseline is architecturally impossible at this scale."

### S6. Triplicate reproducibility [NEW]

The 35K direct GPU runs demonstrate excellent reproducibility (CV=0.0046%). This addresses v1 concern M3.4 about single-run comparisons.

### S7. Null model enrichment pattern is consistent across scales [NEW, conditional]

Both the 1K (5 perms) and 35K (2 perms) null models show the same qualitative pattern: depleted K=2, enriched K>=3 (35K) or K>=4 (1K), null collapse well before the biological K_max. This cross-scale consistency is powerful evidence for genuine biological structure -- *if* adequate permutation counts are provided.

### S8. The paper is honest about limitations [RETAINED]

Section 4.4 remains commendable. The addition of 35K results addresses some limitations (vocabulary size) while creating new ones (incomplete campaigns, inadequate null model power).

---

## RECOMMENDED EXPERIMENTS (updated priority order)

### R1. Run 100+ null model permutations at BOTH scales [CRITICAL]

**Priority: Highest.** This is the single experiment that would most strengthen the paper.

- **35K features:** 100 perms x ~655s / 4 GPUs = ~4.5 hours. 200 perms = ~9 hours.
- **1K features:** 100 perms x ~86s / 1 GPU = ~2.4 hours.

Report proper t-distribution confidence intervals. Fix the `"inf"` z-score strings. Compare null K-distributions across the two vocabulary scales to demonstrate that the enrichment pattern is robust.

**Estimated cost:** ~$50-100 on cloud H200 instances. Trivial for a Nature Methods submission.

### R2. Explain the K-max drop with a controlled experiment [HIGH PRIORITY]

Run 35K features at min_count=8 on a single GPU (if ~119 GB fits on H200) or the smallest feasible threshold. If K_max stays at ~17 even at low threshold, the drop is due to feature dilution or GO hierarchy effects, not threshold. If K_max rises toward 22, the drop is purely a threshold effect and does not require additional explanation beyond what the 1K Opus run already demonstrates.

### R3. GO hierarchy-aware K-distribution at both scales [HIGH PRIORITY]

Compute the fraction of GO ancestor-descendant pairs in both vocabularies. Report raw and corrected K_max. If 35K has substantially less hierarchy inflation than 1K, present this as evidence that the expanded vocabulary produces more biologically meaningful patterns.

### R4. Complete the 35K mining campaign [MEDIUM PRIORITY]

The wave 3 partial shows the 35K feature set at 0.1% support producing millions of itemsets per K-level. Complete this campaign (or at least one threshold) and report. The power test logs show that lower thresholds (0.001%) produce 900K+ frequent pairs and 17M+ K=4 itemsets, suggesting the full campaign at low support may be computationally infeasible on 4xH200. If so, state this explicitly as a future hardware requirement.

### R5. Name the 8 K=22 proteins [HIGH PRIORITY, from v1]

Unchanged. Cross-reference with UniProt. Resolve the 8-vs-3 discrepancy.

### R6. Closed and maximal frequent itemset counts at both scales [MEDIUM PRIORITY]

Especially important for the 35K results where feature redundancy (multiple GO terms for the same process) likely inflates raw counts. The 606K number may collapse to 50K-100K closed itemsets.

### R7. Quantify multi-GPU approximation error [MEDIUM PRIORITY]

Run a single-GPU baseline for the 35K feature set at one threshold (if VRAM permits, potentially with a smaller transaction subset) and compare K-distributions. The locally-frequent union step prunes 42% of K=2 candidates -- what is the analogous pruning at K>=5?

### R8. Per-K SON miss rate for the 1K comparison [LOW PRIORITY, from v1]

Still useful but less critical now that SON infeasibility at 35K makes the stronger argument. Extract per-K recovery rates from `experiment_direct_vs_son_20260219_050326.json`.

### R9. Taxonomic stratification [LOW PRIORITY, from v1]

Break down K-distributions by kingdom. The 109M transactions (35K features) vs 76.9M (1K features) difference suggests the 35K vocabulary includes more organisms. Stratification would reveal whether deep patterns are model-organism-specific.

---

## NEW CROSS-SCALE ANALYSIS OPPORTUNITIES

The existence of both 1K and 35K results creates analysis opportunities not available in v1:

### C1. Feature vocabulary overlap

Are the 1,002 original features a subset of the 34,920? If yes, every 1K-feature itemset should appear in the 35K results (at the same threshold), providing an internal consistency check.

### C2. K-distribution shape as a function of vocabulary size

Two data points (1K: peak=9, K_max=22; 35K: peak=7, K_max=17) suggest an inverse relationship between vocabulary size and K_max. Is this an artifact of the different support thresholds, or a genuine property of protein annotation data? A controlled experiment (same threshold, varying vocabulary size) would answer this.

### C3. Null model comparison across scales

The null K_max is 6 (1K, 5 perms) vs 5 (35K, 2 perms). Does the null model also produce shallower patterns with more features? If the null-to-biological K_max ratio is approximately constant (~3.5x for 1K: 6->22; ~3.4x for 35K: 5->17), this would suggest a universal scaling law for biological vs random co-occurrence depth.

### C4. Computational scaling analysis

The 1K run processes 76.9M transactions x 1,002 features in 7.3 min on 1xH100. The 35K run processes 109M transactions x 34,920 features in 10.9 min on 4xH200. Normalized: the 35K run achieves ~50x more feature-transaction-products per GPU-second (accounting for H200 vs H100 memory bandwidth differences). This scaling analysis would position ET-miner's architecture for future, even larger vocabularies.

---

## VERDICT (updated)

The 35K-feature expansion addresses several v1 concerns (multi-GPU evidence, SON infeasibility as a fundamental rather than parameter-dependent limitation, triplicate reproducibility) while creating new ones (K-max drop explanation, 2-perm null model, incomplete campaigns, multi-GPU approximation error).

**Net assessment:** The paper is **stronger** with the 35K results, but the null model has **regressed** from weak (5 perms) to indefensible (2 perms). The K-max drop is unexplained and will be the first question from any reviewer.

**Recommended disposition:** Major revision. The engineering contribution (GPU-resident FIM scaling from 1K to 35K features, SON infeasibility demonstration, multi-GPU architecture) is solid. The statistical validation is insufficient. The biological interpretation needs cross-scale analysis.

**If the authors run 100+ permutations and explain the K-max drop, the paper moves from "major revision" to "minor revision."** The core architecture will survive peer review intact. The claims about biological significance need proper statistical backing.

---

## SEVERITY SUMMARY

| Concern | Severity | Status vs v1 |
|---------|----------|-------------|
| M1: Null model permutation count (now n=2) | 9/10 | ESCALATED |
| M2: GO hierarchy inflation unquantified | 7/10 | Reduced (K-drop is evidence) |
| M3: SON comparison framing | 5/10 | Reduced (infeasibility is stronger) |
| M4: K-max drop unexplained (NEW) | 8/10 | NEW |
| M5: 8 vs 3 K=22 proteins | 7/10 | Unchanged |
| M6: Wave 3 incomplete (NEW) | 6/10 | NEW |

---

*Review updated 2026-02-21. All references to result files correspond to JSON/log files in `applications/alphafold/results_35k/` and `applications/alphafold/results_214m/`. Previous review (v1) dated 2026-02-20 covered only the 1K-feature results.*
