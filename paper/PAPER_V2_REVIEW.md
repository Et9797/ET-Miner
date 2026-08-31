# Adversarial Review — `papers/et_miner_proteome.tex`

Reviewer: automated adversarial cross-check of every quantitative claim against the mining logs, experiment JSONs, and decoded-pattern artifacts under `applications/alphafold/results_214m/`, plus repository integrity seeds from prior analysis. Date: 2026-08-01.

## Summary

Across 9 verification buckets, **73 claims** were adjudicated:

| Verdict | Count |
|---|---|
| CONFIRMED | 62 |
| DISCREPANT | 4 |
| UNVERIFIABLE | 7 |

The paper's headline numbers are overwhelmingly faithful to the artifacts: the campaign table (6/6), the full 22-row K-distribution (8/8, line-for-line), the null-model Z-scores (all magnitudes reproduce the JSON), the Direct-vs-SON comparison (6/6), and the citation graph (no orphan cites, no cross-paper tool contamination) all check out exactly. The old fabricated v1 Table 1 (247/752/3) has been corrected to the verified 500/500/6 = 1,006 ground truth.

**Top risks (most severe first):**

1. **Methods misdescription (line 129):** "retained all features occurring in at least 8 proteins" does *not* describe how the vocabulary was built. The log shows the vocabulary is a top-500-Pfam + top-500-GO frequency cap out of 24,291 / 25,993 unique terms; the "≥8" is the *mining* min-support, a different quantity. A methods reviewer will catch this conflation.
2. **Inflated 40× memory-reduction claim (lines 167, 402):** the 5.1 GB CSR is the 76.9M *mined subset*; the 206 GB dense is the 205.6M *full set*. Within one dataset the reduction is ~15×, not 40×. Cross-dataset pairing inflates the figure.
3. **Broken reproducibility promise (line 333 vs line 530):** the paper says the 8 K=22 accessions "can be recovered by querying the source transaction data," but line 530 states the transaction matrix is *not deposited*, and no accession list survives in the repo. The claim is internally contradictory and not satisfiable with released artifacts.
4. **Statistical mislabeling (lines 367, 382, 391):** the p<0.45 bound is the *exact one-sided binomial* 95% bound (1−0.05^(1/5)=0.451), but the paper attributes it to "the rule of three," which actually gives 3/5=0.60. The number is defensible; the attribution is wrong.

Plus several integrity items inherited from prior analysis (misleading commit subject, stale Dutch translation, unbacked min_count=4 claim) detailed at the end.

---

## Discrepancies (most severe first)

### D1 — Vocabulary construction misdescribed as an ≥8-support cutoff
- **Paper (line 129):** "We retained all features occurring in at least 8 proteins (the lowest support threshold used)."
- **Artifact:** `pipeline_214m.log:2063-2064` — "24291 unique Pfam, 25993 unique GO from 205620298 proteins", then "Item encoding: 6 pLDDT + 500 Pfam + 500 GO = 1006 total items". The vocabulary is a **top-500-per-type frequency cap**, not a ≥8-support retention. The "≥8" is the mining `min_count` (`godmode_mining.log:2`), a separate quantity.
- **Why it matters:** At 205.6M proteins, vastly more than 500 Pfam/GO families occur in ≥8 proteins, so the sentence does not describe the actual vocabulary and understates the selectivity of the cap.
- **Source:** `applications/alphafold/results_214m/logs/pipeline_214m.log:2063,2064`; `godmode_mining.log:2`

### D2 — "40× reduction" conflates two different datasets
- **Paper (line 167; repeated line 402):** "316 million non-zero entries occupying ~5.1 GB ... a 40× reduction from the ~206 GB naive dense representation."
- **Artifact:** `direct_mining.log` ties the 316,421,093 non-zeros / 5.1 GB CSR to the **76,890,945-transaction subset**, while the 206 GB dense figure is for the **205.6M full set** (`beyond_mining.log`, 205,620,298 × 1002 × 1 B = 206 GB). Within the 76.9M subset: dense = 76.9M × 1002 × 1 B = 77 GB, so 77/5.06 = **~15×**, not 40×.
- **Why it matters:** The 40× figure is manufactured by pairing a smaller-dataset CSR against a larger-dataset dense. The appendix (`tab:memory-comparison`) already reports the honest same-dataset ratio (1.4× bit-packed, 7.8× at 0.01% density), so the body's 40× is inconsistent with the paper's own appendix.
- **Source:** `direct_mining.log` (316,421,093 nz on 76,890,945 txns); `beyond_mining.log` (Total transactions 205,620,298); paper-internal arithmetic

### D3 — 8 K=22 accessions "recoverable" contradicts "matrix not deposited"
- **Paper (line 333):** "The eight matching proteins can be recovered by querying the source transaction data for this signature with the released analysis code (`analyze_k22_proteins.py`)."
- **Paper (line 530):** "The full intermediate transaction matrix is not deposited owing to its size."
- **Artifact:** `analyze_k22_proteins.py:162-190` requires `--data transactions_214m_base.parquet` (default `/workspace/data/…`), which is **absent**; `item_mapping` is git-ignored/absent; no `accessions_deepest_itemset_*.tsv` was ever saved. The 8 IDs are neither listed in the paper nor regenerable from this repo.
- **Why it matters:** Internal contradiction plus an unmet reproducibility promise. Matches the plan contingency (`plans/leg-het-vast-in-crispy-glade.md:127-129`: "data is grotendeels weg").
- **Source:** `applications/alphafold/experiments/analyze_k22_proteins.py:162-190,476`; `papers/et_miner_proteome.tex:333,530`; filesystem (no transactions parquet, no accessions TSV)

### D4 — p<0.45 mis-attributed to the "rule of three"
- **Paper (lines 367, 382, 391):** "p < 0.45, the 95% one-sided upper bound for 0 of 5 by the rule of three."
- **Artifact/computation:** 0/5 null runs reach K≥7 is confirmed (`experiment_null_model_…json:153,163`). The exact one-sided binomial 95% bound is 1−0.05^(1/5)=**0.4507**. The rule of three gives 3/n = 3/5 = **0.60**, not 0.45.
- **Why it matters:** The number (0.45) is correct and defensible as the exact binomial bound; only the name "rule of three" is wrong. Low severity but a statistician will flag it.
- **Source:** `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json:153,163`; paper-internal

---

## Confirmed claims

Grouped by bucket; all values matched the artifacts to the stated rounding.

**Dataset & vocabulary (7):**
- 214M proteins (line 129) = 214,683,829 rows read — `pipeline_214m.log:8,2061`.
- 205,620,298 processed (lines 129,152) = exact "passed pLDDT filter" / "Total transactions" — `pipeline_214m.log:2061,2273,2286`.
- 150 GB in 63 min (line 129): "TrEMBL: 150G"; af_extract 3777.0s = 62.95 min — `pipeline_214m.log:8,2277`.
- min-support 8 → 1,002 frequent single items (Table 1 caption) — `godmode_mining.log:2,3`.
- 1,006 items = 6 pLDDT + 500 Pfam + 500 GO (Table 1) — `pipeline_214m.log:2064,2274,2287`.
- 1,002 pass (1,000 Pfam/GO + 2 pLDDT bins) — `godmode_mining.log:3,5`; corroborated by only two pLDDT labels in `decoded_top_k_patterns.txt`.
- 76,890,945 (37.4%) multi-feature proteins (lines 91,120,152) — `pipeline_214m.log:2291,2294`; `decoded_top_k_patterns.txt:3`.

**Memory (6):** 206 GB dense (line 105), 316M nz / 5.1 GB coordinate (line 167), ~26 GB bit-packed (lines 167,171), ~3 GB H2D transfer (line 171), ~264 B over 22 K-levels (line 814), appendix 214M row 27 GB/19 GB/1.4× (line 887) — all arithmetically consistent with `direct_mining.log` / `beyond_mining.log`.

**Campaign table (6/6):** Base (0.1%/76,891/5,305/K9/1.9min/SON), Super (0.01%/7,689/51,124/K13/4.3min/SON), Power (0.001%/768/22,846/K13/18.1min/SON), Blitz (0.0001%/77/2,841,280/K19/2.0min/Direct), Ultra (0.00002%/16/14,558,875/K20/4.7min/Direct), Opus (0.00001%/8/26,849,505/K22/7.3min/Direct) — logs: `pipeline_214m.log`, `ultra_mining.log`, `extreme_mining.log`, `direct_mining.log`, `beyond_mining.log`, `godmode_mining.log`. (Filenames differ from paper run names; mapping verified.)

**K-distribution (8/8):** total 26,849,505; peak K=9 = 3,529,257 (13.14%); K=1=1,002; K=2=73,786; K=22=1 (8 proteins); all K=3..21 rows; percentages recomputed and sum to 100.0% — `godmode_mining.log:3-27`.

**K=22 composition (2):** 22 features = 2 Pfam + 8 MF + 4 BP + 7 CC + 1 pLDDT matches `analyze_k22_proteins.py:35-63` feature-for-feature (paper↔code self-consistency). Neuronal-antiviral narrative is now hedged to "consistent with … a co-occurrence, not a demonstrated mechanism" (line 333), complying with the plan.

**Null model (17):** 5 permutations, seed 42, 662 s total, min_count=769; Z at K=2 (−987), K=3 (−143), K=4 (+3,791), K=5 (+7,402), K=6 (+71,728); null peaks K=3 (46.2%); null max depth K=6 (mean 22, std 1.1); bio K≥7 = 88,745; real_total 475,865 to K=14; K=1 preserves 1,002; ~130 s/permutation — `experiment_null_model_20260219_061046.json`. The n=5 Z-scores are appropriately recast as t-statistics (4 df) and flagged uncalibrated in the table caption and §4.4.

**Direct vs SON (6/6):** Direct 475,865 in 50.7 s; SON 22,846 in 1,085.6 s; 21× speedup; SON misses 95.2% — `experiment_direct_vs_son_20260219_050326.json`.

**Scale comparison (3):** 5.1× (76.9M/15M); 62.6% = 128.7M excluded; ET-miner row 76.9M/1,002/K22/1×H100/7.3min — all internally consistent.

**Citations (7):** 47 \cite → 37 unique keys, all with matching \bibitem and none unused; `\begin{thebibliography}{37}` = 37 entries; tool stack (CuPy/NumPy/CUDA) matches method with **zero** cross-paper (opus_pocket: fpocket, efficient-apriori, scikit-learn, PDBbind) contamination; Zenodo DOI and GitHub URL well-formed and consistent.

---

## Unverifiable claims

These could not be checked against any repository artifact; flagged for author attestation, not correction.

| Claim | Paper | Why unverifiable | Source |
|---|---|---|---|
| UniProt release 2025_01 | line 129, 478 | Log records only input filename `uniprot_trembl.dat.gz` and Feb 2026 run date; no release string. Access-date consistent. | `pipeline_214m.log:3,10` |
| K=22 shared by exactly 8 proteins | lines 254, 286 | Named cross-check file `decoded_top_k_patterns.txt` caps at K=19 (187 proteins) and is a *different* run; no artifact contains any K≥20 itemset. | `decoded_top_k_patterns.txt` (max K=19) |
| 22→21 independent (0 GO parent-child pairs; InterPro2GO link removed) | line 333 | Computed at runtime via pronto + go.obo; no go.obo cached, no saved JSON output, not reproducible read-only. | `analyze_k22_proteins.py:277-420` |
| min_count=4 → 48M itemsets, same K max | line 335 | **No** min_count=4 run exists. The only "48M" is "Written 48M transactions" during extraction — a transaction count, not itemsets. High conflation risk. | `experiment_null_model_…json` (min_count 769); `pipeline_214m.log:2113` |
| 8 K=22 UniProt accessions | line 333 | See D3 — not recoverable from surviving artifacts. | filesystem |
| Competitor rows (Borgelt/Fang/GMiner/BIGMiner) | lines 420-423 | Literature-sourced; no local artifact. Internally used consistently (GMiner 15M basis for 5.1×). | paper-internal |
| External correctness of 37 references | lines 541-724 | Only internal cite↔bibitem consistency checkable; bibliographic accuracy is external. | paper-internal |

---

## Integrity items to apply

Carried from prior analysis; corroborated by these findings where noted.

1. **Misleading commit history.** Commit `b318df5` has subject "Alphafold experiment validation produced identical results" but its diff is a 231-line manuscript rewrite that *deletes* expanded-run figures. Amend/annotate the commit record for audit honesty. (Repo history, not the .tex.)
2. **Stray `results_35k/*.json` artifacts exist** even though `plans/leg-het-vast-in-crispy-glade.md` claims the expanded run has "geen bewaard artefact." Either reconcile the plan text or remove the orphaned artifacts so the record is internally consistent.
3. **Expanded/"v2" run is self-contradictory across docs** (606K itemsets @ 4×H200 in `revision_notes_b3.tex` vs 16.8B @ 8×H200 in the plan). The current `.tex` correctly contains *no* expanded-run claims — keep it that way; do not reintroduce either figure until one is backed by a saved artifact.
4. **Abstract/conclusion omit the n=5 caveat on Z>3,700** (lines 91, 498). Body (table caption + §4.4) correctly recasts these as t-statistics (4 df); the headline should at minimum cross-reference that scope (partially done in the abstract via "see Section 4.4 for scope").
5. **Stale Dutch translation** `et_miner_proteome_nl.tex` (old title, Feb 2026 date). Out of fix scope for the English paper, but flag for the authors so the two versions do not diverge publicly.
6. **Plan outstanding items not yet applied to the `.tex`:** (a) list the 8 K=22 accessions — see D3; (b) replace the "~11,000 / ~10,500 / ~16,000" approximate protein counts (lines 353, 355, 357) with exact values from `decoded_top_k_patterns.txt`; (c) make the t-statistic framing consistent everywhere Z appears; (d) state explicitly that 7.3 min is mining-only wherever the figure appears standalone.

---

## Fix list for et_miner_proteome.tex

Ordered; each item gives the exact change. Do NOT reintroduce any expanded-run figure. (Author-input needed items are marked ⚑.)

1. **[D1] Line 129 — rewrite the vocabulary sentence.** The current text misdescribes vocabulary construction.
   - OLD: `We retained all features occurring in at least 8 proteins (the lowest support threshold used).`
   - NEW: `The feature vocabulary comprises the 500 most frequent Pfam domains and the 500 most frequent GO terms (drawn from 24{,}291 Pfam and 25{,}993 GO families observed across the corpus), together with 6 pLDDT confidence bins, for 1{,}006 defined items. The minimum support threshold of 8 proteins is applied during \emph{mining}, not during vocabulary construction.`

2. **[D2] Line 167 — fix the cross-dataset 40× claim.** Make the reduction ratio same-dataset.
   - OLD: `the resulting matrix contains 316~million non-zero entries occupying ${\sim}5.1$\,GB in coordinate format (two 64-bit integers per entry), a $40\times$ reduction from the ${\sim}206$\,GB naive dense representation (one byte per boolean entry).`
   - NEW: `for the 76.9M-protein mining subset the resulting matrix contains 316~million non-zero entries occupying ${\sim}5.1$\,GB in coordinate format (two 64-bit integers per entry), a ${\sim}15\times$ reduction from the ${\sim}77$\,GB dense representation of that same subset (one byte per boolean entry); relative to the ${\sim}206$\,GB dense matrix of the full 205.6M-protein set it is ${\sim}40\times$ smaller.`
   (This keeps both figures but attaches each to its correct dataset, and stops presenting 40× as a single-dataset reduction.)

3. **[D2] Line 402 — align the Discussion memory-path wording with fix 2.**
   - OLD: `The $206\text{\,GB} \to 5.1\text{\,GB} \to 26\text{\,GB}$ memory reduction path is applicable to any sparse, high-dimensional transaction dataset.`
   - NEW: `The full-set $206\text{\,GB}$ dense matrix, the $5.1\text{\,GB}$ CSR of the mined subset, and the $26\text{\,GB}$ GPU bitvector matrix illustrate a compression path applicable to any sparse, high-dimensional transaction dataset (the same-subset dense-to-CSR reduction is ${\sim}15\times$; see Appendix~\ref{app:gpu-comparison}).`

4. **[D3] Lines 333 + 530 — resolve the recoverability contradiction.** Pick ONE:
   - **Preferred ⚑:** obtain the 8 UniProt accessions from the authors and list them (add a one-line footnote to Table~\ref{tab:k22} or an inline list at line 333), and change line 333 to: `The eight matching proteins (accessions listed in Table~\ref{tab:k22}) were identified with the released analysis code (\texttt{analyze\_k22\_proteins.py}).`
   - **Fallback (if data is unrecoverable):** at line 333 replace `can be recovered by querying the source transaction data for this signature with the released analysis code (\texttt{analyze\_k22\_proteins.py}).` with `were identified during the original run; because the full transaction matrix is not deposited (Section~\ref{sec:conclusion}), they are regenerable only from UniProt release 2025\_01 via the released feature-extraction pipeline, not from the deposited artifacts.` — this removes the contradiction with line 530.

5. **[D4] Lines 367, 382, 391 — fix the "rule of three" attribution.** In all three occurrences replace the phrase `by the rule of three` with the correct source of 0.45:
   - OLD (caption line 367): `($p < 0.45$, the 95\% one-sided upper bound for 0 of 5 by the rule of three).`
   - NEW: `($p < 0.45$, the exact one-sided binomial 95\% upper bound $1-0.05^{1/5}$ for 0 of 5).`
   - Apply the identical phrase swap at line 391 (`yielding $p < 0.45$ (the 95\% one-sided upper bound for 0 of 5 by the rule of three)` → `… the exact one-sided binomial 95\% upper bound $1-0.05^{1/5}$ for 0 of 5`). Line 382 table cell shows only `$<0.45$` and needs no change once the caption is fixed.

6. **[UNVERIFIABLE, high risk] Line 335 — back or soften the min_count=4 claim.** No artifact records a min_count=4 run or a 48M-*itemset* count (the only "48M" in logs is transactions written during extraction).
   - ⚑ If an artifact exists, cite it. Otherwise change `A verification run at $\text{min\_count}{=}4$ discovered 48~million itemsets with identical maximum, confirming the ceiling is invariant across all biologically meaningful thresholds.` to a claim that follows from data on hand, e.g.: `Because each of the 8 proteins carries exactly 22 vocabulary features, $K{=}23$ is impossible at any support threshold; the ceiling is therefore structural, not a threshold artifact.` (drops the unverifiable 48M figure while keeping the sound argument).

7. **[Integrity 6b] Lines 353, 355, 357 — replace approximate protein counts with exact values ⚑.** Pull the exact supporting-protein counts for the highlighted K=13 / K=12 / K=11 patterns from `decoded_top_k_patterns.txt` and replace `(${\sim}11{,}000$ proteins)`, `(${\sim}10{,}500$ proteins)`, `(${\sim}16{,}000$ proteins)` with the exact integers. If the exact values are not in a preserved artifact, keep the `~` and add a footnote noting they are order-of-magnitude estimates.

8. **[Internal consistency] Line 478 — TrEMBL vs Swiss-Prot.** Line 129 states processing was from UniProt **TrEMBL** (consistent with the log input `uniprot_trembl.dat.gz`), but line 478 says `UniProt/Swiss-Prot release 2025_01`.
   - OLD (line 478): `our results reflect UniProt/Swiss-Prot release 2025\_01`
   - NEW: `our results reflect UniProt TrEMBL release 2025\_01`

9. **[UNVERIFIABLE] Lines 129, 478, 530 — release string ⚑.** The "2025_01" release is not recorded in any artifact. Confirm the exact release with the authors; if it cannot be pinned, soften to `the UniProt TrEMBL release accessed February 2026`.

10. **[Integrity 4] Conclusion line 498 — add the n=5 scope to the headline Z.**
    - OLD: `($Z{>}3{,}700$ for $K{=}4$--$6$; no null run reached $K{\geq}7$).`
    - NEW: `($Z{>}3{,}700$ for $K{=}4$--$6$, as $t$-statistics from 5 permutations; no null run reached $K{\geq}7$; see Section~\ref{sec:null-model}).`

11. **[Integrity 6d] Lines 120, 496 — mark 7.3 min as mining-only.** At line 120 change `discovers the complete feature co-occurrence landscape in 7.3\,minutes` → `… in 7.3\,minutes of mining`; at line 496 change `mining completes in 7.3 minutes` → `mining completes in 7.3 minutes (excluding the 63-minute one-time feature extraction, Section~\ref{sec:methods})`.

12. **[Out-of-file, track separately]** Do not edit here, but record for the authors: amend/annotate misleading commit `b318df5`; reconcile the stray `results_35k/*.json` artifacts against the plan; refresh the stale `et_miner_proteome_nl.tex`. These are repository-integrity actions, not manuscript edits.
