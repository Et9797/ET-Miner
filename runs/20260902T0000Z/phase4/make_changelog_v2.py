"""Generate CHANGELOG_V1_V2.md for the V2 preprint from the comparison rows and a curated traceability map.

Usage:
    python make_changelog.py <repo-root>

Sections: value changes (from COMPARISON_REPORT_rows.json tex rows whose verdict is not
'confirmed', with the V2 value and artifact), removed claims, sentences whose meaning changed,
flagged items awaiting a decision, and a traceability appendix mapping every numeric token of
paper/et_miner_proteome.tex to a RESULTS.md row or COMPARISON_REPORT.md key (asserted complete).
"""
import json, re, sys
from collections import OrderedDict
from pathlib import Path

root = Path(sys.argv[1])
rows = json.load(open(root / "COMPARISON_REPORT_rows.json"))
fv = json.load(open(root / "runs/20260902T0000Z/phase4/fresh_values.json"))
tex = open(root / "paper/et_miner_proteome.tex", encoding="utf-8").read()

def art(qkey):
    v = fv.get(qkey)
    if isinstance(v, dict) and v.get("artifact"):
        return str(v["artifact"]).replace("/root/projects/ET-Miner/", "")
    return ""

# V2 value overrides where the paper uses a different fresh quantity than the report's default key
V2 = {
    "run_opus_time_min": ("43.9 min (2,631.2 s; single RTX 3090 pinned with CUDA_VISIBLE_DEVICES=0)", "RESULTS P-028"),
    "run_base_time_min": ("31.1 s direct (Table 2); SON 779.6 s (text)", "RESULTS P-028 / P-015"),
    "run_super_time_min": ("52.6 s direct (Table 2); SON 1,426.0 s (text)", "RESULTS P-028 / P-016"),
    "run_power_time_min": ("74.7 s direct (Table 2); SON time from the pinned comparison (text)", "RESULTS P-028 / P-029"),
    "run_blitz_time_min": ("4.9 min (292.9 s)", "RESULTS P-028"),
    "run_ultra_time_min": ("33.3 min (1,996.4 s)", "RESULTS P-028"),
    "extract_time_min": ("48 min stream+reduce, 29 min filtering, 9.3 min extraction (162,865 proteins/s)", "RESULTS U-011, X-005, X-007, X-009"),
    "hw_gpu_model": ("NVIDIA GeForce RTX 3090 (24 GB, cc 8.6)", "RESULTS E-001"),
    "hw_gpu_count": ("2 installed; 1 (pinned, second device verified idle) for Table 2 and the SON comparison, 2 (row split) for the null model and per-K exports", "RESULTS E-001, P-028, P-029; null_model_769_100.log, opus/mining_meta"),
    "hw_gpu_vram_gb": ("24", "RESULTS E-001"),
    "hw_host_ram_gb": ("69.6 GiB (container cgroup)", "RESULTS E-004"),
    "son_speedup": ("from the pinned comparison (text)", "RESULTS P-029"),
    "run_power_direct_time_s": ("from the pinned comparison (text)", "RESULTS P-029"),
    "run_power_son_time_s": ("from the pinned comparison (text)", "RESULTS P-029"),
    "null_total_time_s": ("2,007 s for 100 permutations", "RESULTS P-018"),
    "null_per_perm_time_s": ("18.6 s", "RESULTS P-018"),
    "dataset_uniprot_release": ("2026_01", "RESULTS U-005, X-001"),
    "run_super_min_count": ("7,690", "RESULTS P-014"),
    "run_super_itemsets": ("113,405", "RESULTS P-014 / P-016"),
    "run_super_kmax": ("14", "RESULTS P-014"),
    "run_power_min_count": ("769", "RESULTS P-014"),
    "run_power_itemsets": ("475,865", "RESULTS P-014 / P-010"),
    "run_power_kmax": ("14", "RESULTS P-014"),
    "run_blitz_vs_power_itemset_ratio": ("6.0x (5.97)", "RESULTS P-014"),
    "run_power_son_itemsets": ("475,865 (SON identical to direct)", "RESULTS P-013"),
    "son_miss_rate_pct": ("0 (claim removed; SON is exact)", "RESULTS P-013"),
    "vocab_plddt_bin_medium_edges": ("50--90", "phase2/extract_2026_01/plddt_bin_edges.json; RESULTS M-009"),
    "dataset_max_features_per_protein": ("46 (argument rewritten as empirical ceiling)", "RESULTS X-013, X-017, P-017"),
    "mem_items_per_txn_214m": ("2.16 (full set) / 4.12 (mining subset)", "RESULTS X-013, X-018"),
    "null_z_min_k4to6": ("removed; Table 5 reports +2,728 / +10,632 / +68,730 (100 permutations)", "RESULTS P-018"),
    "null_z_headline_k_range": ("removed (see above)", "RESULTS P-018"),
    "misc_manuscript_date": ("September 2026 (version 2)", "metadata"),
    "dataset_access_date": ("retrieved 2 September 2026", "RESULTS U-009 (stream log)"),
    "sw_python": ("3.10.13", "COMPARISON_REPORT §2 sw_python"),
    "sw_cupy": ("14.1.1", "COMPARISON_REPORT §2 sw_cupy"),
    "sw_numpy": ("2.2.6", "COMPARISON_REPORT §2 sw_numpy"),
    "sw_cuda": ("driver 595.71.05 with CUDA 13.2, toolkit 12.1", "COMPARISON_REPORT §2 sw_cuda; RESULTS E-001"),
    "sw_os": ("Ubuntu 22.04.3", "COMPARISON_REPORT §2 sw_os"),
    "null_k6_max": ("25 (range 19--25)", "RESULTS P-022"),
    "null_peak_pct": ("46.2", "RESULTS P-022"),
    "null_p_bound_formula": ("1-0.05^(1/100) = 0.0295", "RESULTS P-023"),
    "null_perms_for_p001": ("removed (100 permutations were run)", "RESULTS P-018"),
    "null_p_target": ("removed", "RESULTS P-018"),
    "k22_impossible_k": ("removed; ceiling stated as empirical", "RESULTS X-017, P-017"),
    "k22_independent_features": ("21 (InterPro2GO link verified)", "RESULTS X-022"),
    "k22_definitional_links": ("1 (verified)", "RESULTS X-022"),
    "misc_example_triple_support": ("1,234", "RESULTS P-026"),
    "null_k4_enrichment_ratio": ("4.2-fold (108,059 vs 25,442)", "RESULTS P-018"),
    "mem_ratio_214m": ("4.9x (full set) / 3.1x (subset)", "RESULTS X-018"),
    "mem_dense_214m_gb": ("25.8 GB (205.6M rows)", "RESULTS X-018"),
    "mem_csr_214m_gb": ("5.2 GB (full set) / 3.1 GB (subset)", "RESULTS X-018"),
    "mem_proteome_density_pct": ("0.22 % (full set) / 0.41 % (subset)", "RESULTS X-018"),
    "mem_breakeven_density_pct": ("~1.5 %", "RESULTS X-020"),
    "mem_density_0p01pct": ("0.1 % (label corrected)", "RESULTS X-020"),
    "mem_density_0p1pct": ("1 % (label corrected)", "RESULTS X-020"),
    "mem_density_1pct": ("10 % (label corrected)", "RESULTS X-020"),
    "run_base_method": ("Direct CSR->GPU (exhaustive); SON additionally run, identical", "RESULTS P-014, P-015"),
    "run_super_method": ("Direct CSR->GPU; SON additionally run, identical", "RESULTS P-014, P-016"),
    "run_power_method": ("Direct CSR->GPU; SON additionally run, identical", "RESULTS P-014, P-013"),
    "run_blitz_vs_power_speedup": ("removed", "RESULTS P-014"),
    "son_k2_recovery": ("removed", "RESULTS P-013"),
}
V2.update({
    "misc_example_triple_k": ("3 (unchanged; the triple's support is now measured, P-026)", "RESULTS P-026"),
    "vocab_pfam_go_frequent": ("1,000 = 500 Pfam + 500 GO", "RESULTS X-015"),
    "campaign_n_runs": ("6", "RESULTS P-014"),
    "campaign_support_span_orders": ("4 (0.1 % to 0.00001 %)", "RESULTS P-014"),
    "run_blitz_method": ("Direct CSR->GPU (unchanged)", "RESULTS P-014"),
    "run_ultra_method": ("Direct CSR->GPU (unchanged)", "RESULTS P-014"),
    "run_opus_method": ("Direct CSR->GPU (unchanged)", "RESULTS P-014"),
    "run_blitz_vs_power_support_ratio": ("stated as 'from 769 to 77 proteins'", "RESULTS P-014"),
    "null_peak_k": ("3", "RESULTS P-022"),
    "null_p_bound_confidence_pct": ("95", "RESULTS P-023"),
    "mem_ratio_0p01pct": ("7.8x (unchanged; density label corrected to 0.1 %)", "RESULTS X-020"),
    "mem_csr_bytes_per_entry": ("8 B per non-zero plus 8 B per row pointer (caption clarified)", "RESULTS X-018"),
    "mem_theoretical_items": ("1,000 (unchanged)", "RESULTS X-020"),
    "mem_items_per_txn_0p01pct": ("1 (unchanged; label 0.1 %)", "RESULTS X-020"),
    "mem_items_per_txn_0p1pct": ("10 (unchanged; label 1 %)", "RESULTS X-020"),
    "mem_items_per_txn_1pct": ("100 (unchanged; label 10 %)", "RESULTS X-020"),
    "mem_dense_76p9m_gb": ("9.6 (unchanged)", "RESULTS X-018"),
    "mem_csr_0p01pct_gb": ("1.2 (unchanged)", "RESULTS X-020"),
    "mem_csr_0p1pct_gb": ("6.8 (unchanged)", "RESULTS X-020"),
    "mem_csr_1pct_gb": ("62.1 (unchanged)", "RESULTS X-020"),
    "mem_ratio_0p1pct": ("1.4x (unchanged; label 1 %)", "RESULTS X-020"),
    "mem_ratio_1pct": ("0.15x (unchanged; label 10 %)", "RESULTS X-020"),
})
EDITORIAL = {"misc_concept_figure_k", "misc_concept_figure_support", "pattern_support_precision",
             "misc_transaction_scale_claim", "k22_functional_roles", "pattern_n_highlighted",
             "dataset_metadata_rows", "run_opus_itemsets", "ext_zenodo_doi"}

null_keys = {f"null_k{k}_{s}" for k in range(1, 7) for s in ("mean", "std", "z")}

def v2_value(qkey, fresh):
    if qkey in V2:
        return V2[qkey]
    if qkey in null_keys or qkey in ("null_kge7_std", "null_kge7_z"):
        return (f"Table 5 (100 permutations): {fv.get('null100_' + qkey[5:], {}).get('value', '')}", "RESULTS P-018")
    return (str(fresh), art(qkey) or "COMPARISON_REPORT §2 " + qkey)

tex_rows = [r for r in rows if r["claim_id"].startswith("T-")]
changed = [r for r in tex_rows if r["verdict"] in ("hallucinated", "expected-hardware-deviation")]
inconc = [r for r in tex_rows if r["verdict"] == "inconclusive"]

EXT_PREFIX = ("ext_",)
literature = [r for r in inconc if r["qkey"].startswith(EXT_PREFIX) or r["qkey"] in ("hw_popcll_cycles", "run_opus_vs_gminer_transaction_ratio", "alg_pcie_reduction_vs_conventional")]
design = [r for r in inconc if r["qkey"].startswith(("alg_", "pipe_", "arch_", "sw_execution")) and r not in literature]
literature = [r for r in literature if r["qkey"] not in EDITORIAL]
resolved = [r for r in inconc if r not in literature and r not in design and r["qkey"] not in EDITORIAL]

out = []
w = out.append
w("# CHANGELOG V1 → V2 — paper/et_miner_proteome.tex\n")
w("V1 = the preprint source as committed at 8eea0be (`git show 8eea0be:paper/et_miner_proteome.tex`; line numbers below refer to it). V2 = branch `v2`, reporting only the 2026-09-02 run on this machine (2 × NVIDIA GeForce RTX 3090). Every V2 value cites a RESULTS.md row or a COMPARISON_REPORT.md §2 key; the artifacts live under `runs/20260902T0000Z/`. Nothing was uploaded.\n")
w("## 1. Changed values (V1 → V2)\n")
w("One row per V1 claim whose verdict in COMPARISON_REPORT.md was *hallucinated* or *expected-hardware-deviation*, plus every *inconclusive* claim that V2 replaces with a fresh value. Column 'V2 value' is what the V2 tex now says; 'artifact' is the fresh source.\n")
w("| V1 claim id | V1 line | quantity | V1 value | V2 value | artifact / RESULTS row |")
w("|---|---|---|---|---|---|")
seen = set()
for r in changed + resolved:
    key = (r["loc"], r["qkey"], str(r["value"]))
    if key in seen:
        continue
    seen.add(key)
    val, src = v2_value(r["qkey"], r["fresh"])
    w(f'| {r["claim_id"]} | {r["loc"].split(":")[-1]} | `{r["qkey"]}` | {str(r["value"]).replace("|", "/")} | {val} | {src} |')
w("")
w("## 2. Removed claims\n")
for line in [
    "- Hardware: 'single NVIDIA H100 80 GB SXM5 with 128 GB host RAM', 'CuPy 13.0, NumPy 1.26, CUDA 12.4' (V1 l.213) and every other H100 mention (abstract l.91, intro l.120, Table 2 caption l.216, Table 6 row l.425, l.446, conclusion l.496). Replaced by the RTX 3090 setup (RESULTS E-001, E-004, Phase 0 audit in PROGRESS.md, COMPARISON_REPORT §2 sw_* rows).",
    "- All H100 timings: 7.3 / 2.0 / 4.7 / 1.9 / 4.3 / 18.1 min, 50.7 s, 1,085.6 s, 63 min extraction, 662 s and ~130 s per permutation (V1 l.91, 120, 129, 224–230, 238, 362, 393, 404, 496). Replaced by the pinned single-RTX-3090 campaign times (P-028), the pinned direct-vs-SON times (P-029), the SON Base/Super times (P-015, P-016), the extraction stages (U-011, X-005, X-007, X-009) and the 100-permutation timing (P-018).",
    "- Table 2 'Method' and 'Key transition' columns, the dagger footnote ('9× faster, 124× more itemsets', 'controlled same-support comparison: 21×') and the Streaming-SON rows 51,124 / 22,846 itemsets with K max 13 (V1 l.216–236).",
    "- The paragraph explaining why SON 'misses 95.2 %' through two compounding mechanisms (V1 l.238) and its Discussion counterpart (V1 l.404): the released SON is exact (P-029, P-015, P-016, P-027).",
    "- Figure 3 caption text about the SON→Direct transition ('9× speedup despite 10× lower support', '21×') (V1 l.243).",
    "- 'no protein carries more than 22 vocabulary features', 'K=23 impossible regardless of the support threshold' (V1 l.335): false premise (X-013: maximum 46). Replaced by the empirical ceiling paragraph (X-017, P-017).",
    "- Five-permutation statistics: 'Z > 3,700 for K=4–6' (V1 l.91, 498), Table 5 values −987 / −143 / +3,791 / +7,402 / +71,728 with μ 63,702 / 79,134 / 25,468 / 1,992 / 22 and σ 42.2 / 41.5 / 21.8 / 13.8 / 1.1, 'p < 0.45', 't-statistics with 4 df' (V1 l.362–393), and the sentence 'A full validation with 100+ permutations would be needed … limited us to 5 trials' (V1 l.393). The preprint's five draws were not recorded and cannot be regenerated; V2 reports the 100-permutation run only (P-018, P-022, P-023).",
    "- 'Medium confidence (70–90)' for the pLDDT bin of the K=22 itemset (V1 l.327): the bin is 50–90 (plddt_bin_edges.json, M-009).",
    "- 'UniProt release 2025_01' (V1 l.129, 478, 529) and 'accessed February 2026' (l.129): the data are TrEMBL 2026_01 retrieved on 2 September 2026 (U-005, X-001, U-009).",
    "- Appendix memory table row 'Actual: ~10 items/txn, 27 GB dense, 19 GB CSR, 1.4×' and the sentence 'For our proteome dataset (~1 % density), CSR provides a modest 1.4× reduction' (V1 l.874, 886); theoretical-row density labels 0.01 % / 0.1 % / 1 % (V1 l.889–891) were a factor 10 too small for 1 / 10 / 100 items over 1,000 (X-018, X-020).",
    "- '~26 GB of GPU memory' as the resident bitvector matrix (V1 l.171, 402, Algorithm 1 comment l.769): the matrix mined in every run is the 76.9M-row subset, 9.6 GB (X-021); 26 GB is kept only as the size the full set would have. '~3 GB' host-to-device transfer (l.171, 440, 859) → 2.5 GB (X-021).",
    "- 'If this 3-feature combination appears in thousands of proteins' (V1 l.103): the triple occurs in 1,234 mined proteins (P-026).",
    "- 'exceeds the null expectation by many orders of magnitude' at K=4 (V1 l.462): the ratio is 4.2 (108,059 / 25,442, P-018).",
    "- 'current-generation datacenter GPUs' (V1 l.496) and 'a single high-end datacenter GPU' (V1 l.402): V2 says 'a single consumer GPU'.",
    "- Appendix C.3 per-level transfer figures ('~12 bytes' ×3, '~264 bytes'), Table 7 'GPU (lexsort)', 'Yes (skip when AND = 0)', '~12 bytes', 'Automatic (memory-aware)', Table 8 'GPU (cp.where)', 'GPU (binary search + triangular inverse)', 'GPU (atomicAdd + lexsort)', '~4 bytes', 'GPU → CPU (bulk, once)', Algorithm 1 'BUILD_PREFIX_GROUPS_GPU' and 'BULK_TRANSFER (Single transfer)', Discussion 4.2 'The only per-iteration communication is a single integer … (~12 bytes)' (V1 l.443, 775, 780, 806-813, 834-840, 850-867): they describe the unused gpu_resident variant (§5; RESULTS X-024).",
    "- '~3 GB' host-to-device transfer (V1 l.171, 440, 859): 3.15 GB, written '~3.1 GB' (RESULTS X-023; an intermediate draft said 2.5 GB, which counted only the column indices).",
    "- Table 4 item label 'plddt_mean' (V1 l.327): the member is `plddt_mean_med` (RESULTS P-008).",
    "- 'far exceeding even the 80 GB capacity of the highest-end GPUs' (V1 l.105): unsourced hardware figure; now 'far exceeding the 24 GB of the GPU used here' (E-001).",
    "- Figures: `mining_campaign.pdf` (encoded the SON counts and the H100 timings), `k_distribution.pdf` (printed the peak share as 13.15 %; exact 13.14 %) and `architecture.pdf` (embedded 'GPU (H100 80 GB)', '150 GB', '26 GB', '264 B', '41×', '4.4×') were regenerated by `paper/figures/make_figures.py` from the campaign JSON and RESULTS rows (F-001, F-002). The '4.4×' label of the old architecture figure had no traceable derivation and was not carried over.",
]:
    w(line)
w("")
w("## 3. Sentences whose meaning changed (ripple effects)\n")
for line in [
    "1. Abstract, timing sentence: '7.3 minutes (mining only) on a single NVIDIA H100' → '43.9 minutes (excluding feature extraction) on a single NVIDIA GeForce RTX 3090' (P-028).",
    "2. Abstract, significance sentence: 'Z > 3,700 for K=4–6' → '100 permutations … never produces an itemset beyond K=6, whereas the biological data contain 88,745 itemsets with K≥7 (empirical p≈0.01)' (P-018, P-023).",
    "3. Abstract, closing sentence (new): all results come from one complete execution on freshly downloaded data (September 2026) with deposited artifacts.",
    "4. Introduction, example triple: hypothetical 'appears in thousands of proteins' → factual 'appears in 1,234 of the proteins mined here; when such a combination recurs …' (P-026).",
    "5. Introduction, closing sentence: '7.3 minutes of mining on a single NVIDIA H100' → '43.9 minutes of mining on a single NVIDIA GeForce RTX 3090' (P-028).",
    "6. Methods 2.1, release sentence: 2025_01 / accessed February 2026 → 2026_01, 202,556,314 entries, retrieved 2 September 2026 (U-005, X-001, U-009).",
    "7. Methods 2.1, pLDDT bins: added that only the two mean-pLDDT bins (50–90, >90) are populated under the pLDDT ≥ 50 filter (plddt_bin_edges.json); Table 1 caption gained 'the other 4 pLDDT bins are empty'.",
    "8. Methods 2.1, extraction sentence: '150 GB in 63 minutes' → the three-stage route actually run (48 min stream+reduce, 29 min filtering, 9.3 min extractor time / 10.4 min wall at 162,865 proteins/s) and 'separate from the 43.9-minute mining runtime' (U-011, X-005, X-007, X-009, X-019, P-028).",
    "9. Methods 2.2: 'was used for all results' now specifies one GPU for the campaign and the row-split two-GPU form for the null model and per-level exports (logs: full_campaign_r1.log, null_model_769_100.log, opus/blitz/minc4 mining_meta n_gpus=2).",
    "10. Methods 2.5, multi-GPU sentence: 'all experiments use a single GPU' → mentions the row-split option and that the campaign uses one GPU; SON sentence now states the global counting pass makes the result exact.",
    "11. Results 3.1, setup paragraph: rewritten for the RTX 3090 machine; states which runs used one versus two GPUs and that the single-GPU runs were pinned with CUDA_VISIBLE_DEVICES=0 with the second device verified idle by nvidia-smi sampling (P-028, P-029).",
    "12. Table 2: exhaustive counts for all six rows, one time column (single RTX 3090, pinned, second device verified idle), Super/Power rows changed (7,690 / 113,405 / K 14; 769 / 475,865 / K 14), caption rewritten (P-028).",
    "13. Results 3.1, SON paragraph: the 'key engineering breakthrough / SON misses 95.2 %' narrative → SON is exact and 99× slower on one GPU (62.2 s vs 6,151.1 s); the direct path matters for time (P-029, P-015, P-016, P-027).",
    "14. Figure 3 caption: one time series, single RTX 3090.",
    "15. Table 4, structural property row: '(70–90)' → '(mean pLDDT 50–90)'.",
    "16. Results 3.3, InterPro2GO clause: now names IPR011545 / IPR001650 and states that neither entry maps to any other member (X-022).",
    "17. Results 3.3, accessions sentence: the eight proteins were identified with analyze_k22_proteins.py and their accessions are deposited (P-009).",
    "18. Results 3.3, ceiling paragraph: 'structural, K=23 impossible' → empirical ceiling (max 46 features; 32 proteins ≥22, 19 ≥23; min_count-4 run still ends at K=22) (X-013, X-017, P-017).",
    "19. Results 3.4, footnote: pattern counts are read from the per-K itemset tables of the Blitz run (P-011).",
    "20. Results 3.5, method sentence: five permutations / 662 s H100 → one hundred permutations, 2,007 s, two RTX 3090s, row-split path, 18.6 s per permutation (P-018); the dedup clarification was added.",
    "21. Results 3.5, results sentence: 'at most 23 itemsets at K=6' → 'between 19 and 25 (mean 21.7)' (P-022).",
    "22. Table 5: 100-permutation μ/σ/Z and p = 0.0099 rows; caption rewritten (P-018, P-023).",
    "23. Results 3.5, 'Three findings' paragraph: new Z values, p at the resolution limit 1/101, binomial bound 0.0295 (P-018, P-023).",
    "24. Results 3.5, K=1 sentence: 'marginal frequencies are preserved' → 'every item stays frequent under the shuffle'.",
    "25. Results 3.5, scope paragraph: the biological reference is now named as the Power row of Table 2; the '100+ permutations would be needed' sentence removed.",
    "26. Discussion 4.1, first paragraph: 'single high-end datacenter GPU' → 'single consumer GPU'; matrix sizes now '5.1 GB coordinate-format sparse matrix … its 9.6 GB GPU bitvector matrix (26 GB for the full set)' (X-018, X-021).",
    "27. Discussion 4.1, SON paragraph: 'cost of the SON approximation, 21×, misses 95.2 %' → price is time, not completeness; 99×, with the time split between the chunk-local mining pass (88 min) and the global re-count (14 min) as measured (P-029); exactness argument.",
    "28. Table 6, ET-miner row: '1× H100, 7.3 min' → '1× RTX 3090, 43.9 min' (E-001, P-028).",
    "29. Discussion 4.2, multi-GPU sentence: single H100 → single RTX 3090 for Table 2, row-split path for exports and null model.",
    "30. Discussion 4.4: 'many orders of magnitude' → '4.2-fold (108,059 vs. 25,442; …)' (P-018).",
    "31. Limitations 3: release 2025_01 → 2026_01.",
    "32. Conclusion: 'current-generation datacenter GPUs … single H100 … 7.3 minutes … 63-minute extraction' → 'a single consumer GPU … one NVIDIA GeForce RTX 3090 … 43.9 minutes' (P-028); null-model sentence → 100 permutations, no K≥7, p≈0.01.",
    "33. Funding: added the compute statement (rented vast.ai instance, two RTX 3090s).",
    "34. Data and Code Availability: release 2026_01 (archived under previous_releases) and the deposited run directory runs/20260902T0000Z/.",
    "35. Appendix B, Algorithm 1 comment: '~26 GB VRAM' → '9.6 GB VRAM for 76.9M rows' (X-021, X-023).",
    "36. Appendix C/D: '~3 GB' transfer → '~3.1 GB' (X-023); Table 7 multi-GPU strategy → 'Candidate-index or row splitting'.",
    "37. Appendix D, memory paragraph and Table 9: measured rows (2.16 / 4.12 items per transaction; 4.9× / 3.1×), corrected density labels and break-even (~1.5 %), caption notes the 16-byte coordinate form (X-018, X-020).",
    "38. Front matter: date line → 'September 2026 (version 2)'; bibliography gained Phipson & Smyth 2010 (permutation p-values).",
    "39. Methods 2.1: '205,620,298 proteins from UniProt TrEMBL' → 'the 205,620,298 AlphaFold DB entries (of 214,683,829) whose mean pLDDT is at least 50' (M-001, M-003); the empty-bin explanation now names the actual cause (the low bin is filtered out; the fraction bins are never assigned by the metadata-based extraction, af-extract confidence.rs:173); extraction time stated as 9.3 min extractor time / 10.4 min wall (X-019).",
    "40. Methods 2.2: 'three execution paths' → CPU path plus one GPU path with single-GPU, candidate-split/row-split and sparse-CSR variants; the single-GPU variant ran the campaign (§5).",
    "41. Methods 2.3/2.4: '(~26 GB)' → '(~26 GB for the full set, 9.6 GB for the mining subset)'; 'single host-to-device transfer (~2.5 GB)' → 'one-time host-to-device copy (~3.1 GB of 64-bit row pointers and column indices)' (X-023).",
    "42. Results 3.1: 'returned exactly the same itemsets' → set identity stated only where verified (Base, Super: P-027) and 'identical count and K-distribution' at Power.",
    "43. Discussion 4.2: 'the entire Apriori algorithm … executes without returning data to the CPU … single integer (~12 bytes)' → per-iteration traffic is the uploaded prefix groups and the downloaded 12-byte survivor records, proportional to the number of frequent itemsets (X-024).",
    "44. Data and Code Availability: added the pointer to COMPARISON_REPORT.md as the itemized list of superseded preprint values.",
    "45. Appendix B (Algorithm 1), C.2, C.3, Table 7, Table 8: rewritten for the executed single-GPU path (§5; X-024): prefix groups built on the host and uploaded, 12-byte survivor records downloaded per level (0.32 GB for the Opus run), warp-level early exit, sorting on the host, 8-byte loop counter, dispatch by candidate count.",
    "46. Appendix C.1: hard-coded 'Section 2.4' → cross-reference to the labelled subsection.",
    "47. Table 2 and Discussion 4.1: 'returned identical itemsets' / 'returns exactly the exhaustive itemset table' → the same itemset counts, with set identity stated only where it was verified (Base and Super, P-027); at Power the artifacts compare counts and K-distributions (P-029).",
    "48. Results 3.1 setup: added that the two streaming runs at Base and Super were not pinned to a device and that the streaming path counts on one device only (P-015, P-016).",
    "49. Appendix C.2 and Table 7: the early-exit description matched the legacy ballot kernel; the runs used the shared-memory kernel, whose exit skips a whole 32-word tile when the staged prefix AND is zero (src/et_miner/gpu/kernels/_src/shared_tiled.cu:143-152).",
    "50. Results 3.4: the intermediate-K supports ~11,000 / ~10,500 / ~16,000 came from decoded summaries this run did not regenerate. K=12 now gives the measured 16,185 with 111 matching itemsets; K=13 and K=11, which name no domain identifiers, give the level maxima 11,521 and 28,913; the footnote says which is which (P-030).",
    "51. Discussion 4.2: 'multi-GPU with memory-aware chunk sizing' → candidate-index or row split, the latter with chunk sizing measured from the installed VRAM (src/et_miner/gpu/row_split_chunks.py; the candidate split has no memory model).",
    "52. Introduction and Discussion 4.2: 'a few million transactions, three orders of magnitude smaller' → 'a few million real transactions, one to two orders of magnitude smaller', which is what Table 6 shows (GMiner 1.7M real, 15M synthetic, vs 76.9M).",
    "53. Discussion 4.2: '~10 GB' for the bit-packed dense bitmap of the mining subset → 9.6 GB, matching every other mention (X-018).",
]:
    w(line)
w("")
w("## 4. Flagged for your decision (inconclusive claims kept in V2)\n")
w("These are not results of this paper and cannot be verified by re-execution. They are still in the V2 tex; say the word and I remove or reword any of them.\n")
w("### 4a. Literature values about other systems (kept as attributed citations)\n")
w("| V1 claim id | line | quantity | value | where |")
w("|---|---|---|---|---|")
for r in literature:
    w(f'| {r["claim_id"]} | {r["loc"].split(":")[-1]} | `{r["qkey"]}` | {str(r["value"]).replace("|", "/")} | {str(r["context"])[:70].replace("|", "/")} |')
w("")
w("### 4b. Design statements derived from the implementation, not measured\n")
w("Checked against the source by a fresh-context code inspection (see §5); each row says whether the code supports the statement.\n")
w("| V1 claim id | line | quantity | value | note |")
w("|---|---|---|---|---|")
for r in design:
    w(f'| {r["claim_id"]} | {r["loc"].split(":")[-1]} | `{r["qkey"]}` | {str(r["value"]).replace("|", "/")} | {str(r["note"])[:80].replace("|", "/")} |')
w("")
w("### 4c. Editorial counts and non-numeric phrasings (kept)\n")
w("`pattern_n_highlighted` (five highlighted patterns), `pattern_support_precision` (order-of-magnitude footnote, now sourced to the Blitz tables), `k22_functional_roles` (four roles), `misc_concept_figure_k/support` (toy example in Figure 1), `campaign_n_runs` (six), `campaign_support_span_orders` (four), `misc_transaction_scale_claim` ('100-million-transaction scale' for 76.9M), `dataset_metadata_rows`/`run_opus_itemsets` word forms ('hundreds of millions', 'millions'), `vocab_pfam_go_frequent` (1,000 = 500 + 500), `alg_transaction_min_features` (>1), `alg_dense_bytes_per_bool`, `alg_coo_bytes_per_entry`, `mem_csr_bytes_per_entry`, `mem_theoretical_items`, `ext_zenodo_doi`.\n")
w("## 5. Code-inspection findings (fresh-context subagent, read-only, `src/et_miner/gpu/`)\n")
for line in [
    "- **Dispatch (RESULTS X-025).** `src/et_miner/gpu/dispatch.py:128-129` routes every K≥3 level with ≥ 500,000 candidates to all visible GPUs (`CANDIDATE_COUNT_THRESHOLD_K3 = 500_000`, `get_gpu_count() > 1`), replicating the 9.6 GB bitvector matrix through host memory at each such level (`kernels/k3plus.py:382,403`). No launch script pinned a device, and `full_campaign_r1.log` shows the throughput step at that threshold (Opus K=14: 680,128 candidates in 22.5 s; K=15: 310,661 in 56.8 s). The Table 2 campaign (P-014) and the direct-vs-SON comparison (P-010/P-013) were therefore not single-GPU runs. V2 reports instead the pinned re-run (`CUDA_VISIBLE_DEVICES=0`, both GPUs sampled by nvidia-smi every 5 s): `runs/20260902T0000Z/phase3/2026_01/single_gpu/` (RESULTS P-028, P-029).",
    "- **GPU residency (accurate for one device).** The matrix is allocated once (`csr_bitvec.py:276`) and reused at every level (`mining.py:682,726`); with a single visible device nothing re-uploads it. Kept.",
    "- **Per-level PCIe traffic (V1 appendix inaccurate; RESULTS X-024).** The executed path `_apriori_from_bitvecs` builds prefix groups on the CPU and uploads them (`kernels/k3plus.py:539-606`), downloads survivors as 12-byte (int64 index, int32 count) records (`kernels/shared_tiled.py:181,193-194`), sorts and decodes on the host, and appends results per level (`mining.py:548,694,736`). The '~12 bytes per level', '~264 bytes total', 'prefix groups by boundary detection on GPU', 'lexsort on GPU', 'bulk transfer once' and 'cp.where' statements describe `_apriori_from_bitvecs_gpu_resident` (`mining.py:824-977`, `gpu_resident=True`), which no reported run used. Appendix C.3, Algorithm 1, Table 7 and Table 8 were rewritten accordingly; the main-text statement 'prefix group construction and result collection occur on the CPU with lightweight PCIe transfers' was already correct.",
    "- **64-bit words / __popcll (accurate).** uint64 matrix (`csr_bitvec.py:276`), `__popcll` in every counting kernel (`kernels/_src/pairs_k2.cu:36-39`, `k3plus_fullyfused.cu:52-64`, `shared_tiled.cu:150-156`). Kept.",
    "- **Early termination (partially accurate).** The K≥3 kernels skip the remaining features of a 32-word window once the running AND is zero in all lanes (`k3plus_fullyfused.cu:52-63`, `shared_tiled.cu:143-152`); no candidate is abandoned and the K=2 kernel has no early exit. Reworded in Appendix C.2 and Table 7.",
    "- **Multi-GPU (accurate existence; dispatch wording fixed).** Row split partitions transactions and sums partial counts (`csr_bitvec.py:408-517`, `row_split.py:56-90`, `nccl.py:173-189`); candidate split exists (`k2.py:97-236`, `k3plus.py:333-495`); dispatch is by candidate count and device count, not memory-aware (`dispatch.py:34-36,57-58,128-129`). Table 7 row reworded.",
    "- **'Three execution paths' (inaccurate).** `apriori()` dispatches over CPU dense/sparse tiers and one GPU tier with single-GPU, candidate-split, row-split, gpu_resident and sparse-CSR variants (`core/apriori.py:344-485`; `tests/test_tier_equivalence.py` enumerates seven tiers). Methods 2.2 reworded.",
    "- **9.6 GB (accurate) and the H2D copy (undercounted; RESULTS X-023).** `csr_bitvec.py:245,276` allocate 1,002 × 1,201,422 × 8 B; the host-to-device copy is two int64 arrays, indices 2.53 GB plus row pointers 0.62 GB = 3.15 GB (`csr_bitvec.py:257-272`). '~2.5 GB' → '~3.1 GB' in Methods 2.4, Discussion 4.2, Table 8 and the architecture figure.",
]:
    w(line)
w("")
w("## 6. Traceability appendix — every numeric token in the V2 tex\n")
w("Tokens were extracted programmatically from the document body (preamble and bibliography excluded, identifiers such as GO/Pfam accessions and the DOI included). 'Source' names the RESULTS.md row, the COMPARISON_REPORT.md §2 key, or the non-claim role of the token.\n")

TRACE = {
"200": "M-001 ('over 200 million')", "1,002": "X-015", "76.9": "X-011", "43.9": "P-028 (2,631.2 s)", "3090": "E-001",
"26.8": "P-002", "22": "P-003 (K max); K index; K=22 itemset P-006..P-009", "0.001": "P-028/P-018 (0.001 % support); Table 3 % (P-004)",
"100": "P-018 (100 permutations); X-020 (100 items/txn row); 'BIGMiner 100M/100K' literature (§4a)",
"6": "P-018 (null K max); X-015 (6 bins); P-028 (six runs); K index", "88,745": "P-018", "7": "P-018 (K≥7); P-007 (7 CC terms); K index",
"0.01": "P-023 (p≈0.01); P-028 (0.01 % support); Table 3 %", "2026": "U-005 (2026_01); U-009 (retrieved 2026-09-02); version date",
"1,234": "P-026", "50": "M-003 (pLDDT≥50); M-009 (bin 50–90); '50–350×' literature (§4a)", "350": "literature (§4a)",
"206": "X-018 (dense full set)", "205.6": "X-010", "80": "literature: 80 GB datacenter GPUs (§4a)",
"4": "P-018 (enriched from K=4); X-015 (4 empty bins); K index; Figure 1 toy example", "5.1": "COMPARISON csr_bytes_gb (5.06)",
"2": "E-001 (two GPUs); X-015 (2 bins); P-007 (2 Pfam); P-013 (2 chunks); K index", "214": "M-001", "205,620,298": "X-010 / M-003",
"01": "U-005 (release 2026_01)", "202,556,314": "X-001 / U-005", "500": "X-008 / X-015", "24": "E-001 (24 GB); Phase 0 audit (24 cores); X-008 (24,291)",
"291": "X-008", "25": "X-008 (25,993); P-022 (K=6 max 25)", "993": "X-008", "90": "M-008/M-009 (bin edges)", "006": "X-015 (1,006)",
"8": "P-002/P-028 (min_count 8); P-006 (8 proteins); P-007 (8 MF terms); X-024 (8-byte counts/counter); K index", "149.8": "U-007", "48": "U-011 (2,879 s)", "29": "X-005 + X-007 (765 s + 956 s)",
"90.5": "X-006", "9.3": "X-009", "162,865": "X-009", "1,006": "X-015", "1,000": "X-015 (500+500); X-020 (theoretical |F1|)",
"76,890,945": "X-011", "37.4": "X-011", "316": "X-014", "64": "code inspection: uint64 matrix, __popcll (§5)", "15": "COMPARISON csr_vs_dense_subset_ratio (15.2); K index",
"77": "COMPARISON dense_subset_gb (77); P-028 (min_count 77)", "40": "COMPARISON csr_vs_dense_full_ratio (40.7); P-013 (40 M chunk)", "26": "COMPARISON bitvec_gb (25.8)",
"9.6": "X-021; P-018 (K=5 σ)", "2.5": "X-021 (2.53 GB)", "8.6": "E-001", "595.71": "E-001", "05": "E-001 (driver 595.71.05)", "13.2": "E-001 / Phase 0 audit",
"12.1": "Phase 0 audit (nvcc 12.1)", "7402": "Phase 0 audit (EPYC 7402P)", "69.6": "E-004", "3.10": "COMPARISON sw_python (3.10.13)", "13": "COMPARISON sw_python; K index",
"14.1": "COMPARISON sw_cupy", "2.2": "COMPARISON sw_numpy (2.2.6)", "22.04": "COMPARISON sw_os", "16": "P-028 (Ultra min_count); K index", "0.1": "P-028 (0.1 % support); X-020 (0.1 % density row)",
"76,891": "P-028", "5,305": "P-028 / P-015", "9": "P-028 (Base K max); K index", "31.1": "P-028", "7,690": "P-028", "113,405": "P-028 / P-016", "14": "P-028; K index; P-029 (pass 2, 14.1 min)",
"52.6": "P-028", "769": "P-028 / P-010 / P-018", "475,865": "P-028 / P-010", "74.7": "P-028", "0.0001": "P-028", "2,841,280": "P-028 / P-011",
"19": "P-028 (Blitz K max); X-017 (19 proteins ≥23); K index", "4.9": "P-028 (292.9 s); X-018 (4.9× full-set ratio)", "0.00002": "P-028 (nominal)", "14,558,875": "P-028", "20": "P-028 (Ultra K max); K index",
"33.3": "P-028 (1,996.4 s)", "0.00001": "P-028 (nominal)", "26,849,505": "P-002 / P-028", "0.9": "P-013 (local factor)", "779.6": "P-015", "1,426.0": "P-016",
 "62.2": "P-029 (62.25 s)", "6,151.1": "P-029", "99": "P-029 (98.8×)", "88": "P-029 (pass 1, 88.3 min)", "6.0": "COMPARISON run_blitz_vs_power_itemset_ratio (5.97)", "3,529,257": "P-004", "13.14": "P-004", "3.53": "P-004",
"4.2": "P-018 (108,059/25,442); LaTeX column width p{4.2cm}", "46": "X-013 / X-017", "32": "X-017; '32-bit __popc' literature (§4a); X-024 (32-word window)", "48,007,493": "P-017", "75.6": "P-017", "342": "P-017", "27": "P-017",
"187": "P-011", "611": "P-011", "7714": "P-011 (PF07714)", "0017": "P-011 (PF00017)", "0018": "P-011 (PF00018)",
"000": "COMPARISON pattern_k13_support / pattern_k12_support / pattern_k11_support (~11,000 / ~10,500 / ~16,000)", "0905": "COMPARISON pattern_k12_member_1", "0912": "COMPARISON pattern_k12_member_2",
"42": "P-018 (seed)", "2,007": "P-018", "18.6": "P-018", "46.2": "P-022", "21.7": "P-022", "101": "P-023", "0.0099": "P-023", "95": "P-023 (95 % bound)", "0.05": "P-023 (formula constant)", "0.0295": "P-023",
"0.0": "P-018 (Table 5)", "1.0": "P-018 (Table 5)", "22,019": "P-018 / P-010", "63,703": "P-018", "33.6": "P-018", "239": "P-018 (−1,239)", "73,205": "P-018 / P-010", "79,144": "P-018", "46.5": "P-018",
"128": "P-018 (−128)", "108,059": "P-018 / P-010", "25,442": "P-018", "30.3": "P-018", "728": "P-018 (+2,728)", "104,239": "P-018 / P-010", "1,993": "P-018", "632": "P-018 (+10,632)", "78,596": "P-018 / P-010",
"1.1": "P-018", "68": "P-018 (+68,730)", "730": "P-018 (+68,730)", "0": "P-018 (null K≥7); P-009 (0 parent–child pairs)",
"30": "literature: BIGMiner 30 nodes, GMiner ~30 (§4a)", "1.7": "literature: GMiner 1.7M (§4a)", "280": "literature: GTX 280 (§4a)", "1080": "literature: GTX 1080 (§4a)", "150": "literature: GMiner 20–150 s (§4a)",
"128.7": "X-012", "62.6": "X-012", "10.5281": "metadata (Zenodo DOI)", "18674353": "metadata (Zenodo DOI)", "997": "metadata (author e-mail)", "20260902": "metadata (run directory)",
"0.5": "LaTeX layout (vspace)", "1.5": "LaTeX layout (vspace); X-020 (~1.5 % break-even)", "0.4": "LaTeX layout (addlinespace)", "2.4": "section number reference",
"264": "removed in V2 (X-024)", "2018": "citation year (table header)", "2024": "citation year (table header)",
"7.8": "X-018/X-020 (theoretical row)", "1.2": "X-018/X-020 (theoretical row)", "6.8": "X-018/X-020", "62.1": "X-018/X-020", "1.4": "X-018/X-020 (theoretical row)", "0.15": "X-018/X-020",
"0.22": "X-018", "0.41": "X-018", "3.1": "X-018", "2.16": "X-018 / X-013", "25.8": "X-018", "5.2": "X-018", "4.12": "X-018 / X-013", "002": "X-015 (|F1| = 1,002)",
"1": "K index; counts of one (1 GPU, 1 itemset at K=22 = P-003)", "3": "K index; structural counts (three innovations); P-026 (3-feature)", "5": "K index; five highlighted patterns (editorial)",
"10": "K index; X-021 (~10 GB bitmap = 9.6); X-020 (10 items/txn row)", "12": "K index; X-024 (12-byte survivor records)", "21": "K index; X-022 (21 independent features)", "23": "P-004 (K=21 count); X-017 (≥23 features)",
"11": "K index", "17": "K index", "18": "K index", "0.00": "P-004 (K=1 share)", "1,996,772": "P-004", "7.44": "P-004", "73,786": "P-004", "0.27": "P-004", "1,259,045": "P-004", "4.69": "P-004",
"452,777": "P-004", "1.69": "P-004", "679,471": "P-004", "2.53": "P-004 (K=14 share of Table 3)", "1,184,461": "P-004", "4.41": "P-004", "310,527": "P-004", "1.16": "P-004", "1,974,126": "P-004", "7.35": "P-004",
"118,659": "P-004", "0.44": "P-004", "2,626,332": "P-004", "9.78": "P-004", "37,261": "P-004", "0.14": "P-004", "3,118,459": "P-004", "11.61": "P-004", "9,375": "P-004", "0.03": "P-004",
"3,442,954": "P-004", "12.82": "P-004", "1,818": "P-004", "255": "P-004", "3,293,612": "P-004", "12.27": "P-004", "2,739,532": "P-004", "10.20": "P-004",
"0270": "P-008 (PF00270); X-022", "0271": "P-008 (PF00271); X-022", "005524": "P-008; X-022", "016787": "P-008", "000287": "P-008", "003697": "P-008", "003724": "P-008; P-026", "003725": "P-008",
"003678": "P-008", "000978": "P-008", "030154": "P-008", "045087": "P-008; P-026", "051607": "P-008", "034605": "P-008", "005737": "P-008", "005829": "P-008", "005634": "P-008", "005739": "P-008",
"030424": "P-008", "030425": "P-008", "016607": "P-008", "11545": "X-022 (IPR011545)", "01650": "X-022 (IPR001650)", "11,521": "P-030", "16,185": "P-030", "28,913": "P-030", "111": "P-030",
"214,683,829": "M-001", "10.4": "X-019 (622 s wall)", "0.32": "X-024 (12 B × 26,849,505)", "2,048": "X-024 / code inspection (32 words × 64 proteins)",
"3.1": "X-023 (3.15 GB); X-018 (3.1× subset ratio)",
}

body = tex[tex.index(r"\begin{document}"): tex.index(r"\begin{thebibliography}")] + tex[tex.index(r"\end{thebibliography}"):]
body = re.sub(r"(?<!\\)%.*", "", body)
body = re.sub(r"\\(label|ref|cite|includegraphics|url|href|hypersetup)\{[^}]*\}", " ", body)
body = re.sub(r"\\(begin|end)\{[^}]*\}", " ", body)
tokens = OrderedDict()
for m in re.finditer(r"(?<![A-Za-z:\\])(\d[\d,]*(?:\.\d+)?)(?![\d])", body):
    tokens.setdefault(m.group(1).rstrip(","), 0)
    tokens[m.group(1).rstrip(",")] += 1
missing = [t for t in tokens if t not in TRACE]
assert not missing, f"untraced tokens: {missing}"
w(f"{len(tokens)} distinct tokens, all mapped.\n")
w("| token | occurrences | source |")
w("|---|---|---|")
for t, n in tokens.items():
    w(f"| {t} | {n} | {TRACE[t]} |")
w("")
(root / "CHANGELOG_V1_V2.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("wrote CHANGELOG_V1_V2.md:", len(out), "lines;", len(changed), "changed rows,", len(resolved), "resolved inconclusive,", len(literature), "literature,", len(design), "design")
