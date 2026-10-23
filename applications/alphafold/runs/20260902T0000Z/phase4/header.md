# COMPARISON_REPORT.md — base214m reproduction of the ET-Miner AlphaFold preprint

Campaign run directory: `runs/20260902T0000Z/` (all artifacts on this machine; nothing uploaded).
Companion files: `CLAIMS.md` (all extracted claims with context), `RESULTS.md` (fresh values with
artifact paths and UTC timestamps), `INCONSISTENCIES.md`, `PROGRESS.md`.

## 0. What was reproduced, on what, and how

**Hardware of this reproduction (fresh, RESULTS.md E-001..E-015).** vast.ai container with
2 × NVIDIA GeForce RTX 3090 (24 GB, sm_86; no GPU P2P), AMD EPYC 7402P (24 cores),
69.6 GiB cgroup RAM limit, 200 GB overlay disk, driver 595.71.05 (CUDA 13.2), nvcc 12.1,
CuPy 14.1.1, Python 3.10.13, Ubuntu 22.04.3. The paper reports a single H100 80 GB SXM5 with
128 GB host RAM; every timing/throughput claim is therefore judged as
*expected-hardware-deviation*, never as hallucinated.

**Data actually required (Phase 1 finding).** The base214m pipeline
(`af-extract build-from-metadata`) reads NO AlphaFold structure files. Its inputs are
(1) the UniProt TrEMBL flat file (Pfam and GO cross-references) and (2) the two-column
`uniprotAccession, globalMetricValue` export of the BigQuery table
`bigquery-public-data.deepmind_alphafold.metadata` (mean pLDDT). The 23 TiB structure
download was therefore not needed and no ABORT applies.

**Fresh data acquisition (Phase 2).**
- Metadata: 214,683,829 rows exported through the BigQuery Storage Read API
  (`phase2/data/plddt_metadata_storageapi.csv`, sha256 in RESULTS.md M-006); the runbook's
  `bq query --format=csv` route never produced output (INCONSISTENCIES I-009).
- UniProt release: the paper states release 2025_01, but the old pipeline log parsed
  202,556,314 TrEMBL records, which equals the 2026_01 release exactly (2025_01 has
  252,633,201; RESULTS.md U-001..U-006), and the "150 GB" annotation file is the 149.8 GiB
  2026_01 member. The reproduction therefore uses **UniProt 2026_01**
  (`previous_releases/release-2026_01/knowledgebase/knowledgebase2026_01.tar.gz`, streamed;
  only the DAT record lines the extractor parses were kept — ID/AC/DE/OC/DR Pfam|GO|InterPro/`//`
  — a lossless reduction for this parser, `annotations.rs:279-336`). A 2025_01 run was started as
  the paper-as-stated check and stopped on Et's instruction (disk); it is not part of the verdicts
  (INCONSISTENCIES I-006).
- Extraction: `af-extract build-from-metadata --top-pfam 500 --top-go 500 --min-plddt 50`.
  The unmodified extractor holds every DAT record and every CSV row in RAM and was OOM-killed by
  the 69.6 GiB cgroup (the paper's host had 128 GB; RESULTS.md X-004). The re-run
  (`phase2/scripts/run_af_extract_lean.sh`) feeds it the same inputs restricted, losslessly, to
  what can influence the mined set: the 90,506,840 pLDDT-passing proteins that have a Pfam/GO
  record in TrEMBL 2026_01, and only those DAT records (the other 115,113,458 passing proteins
  carry a single pLDDT-bin item by construction and never enter the ≥2-item set nor the top-500
  ranking). Full-set totals are reconstructed from the full CSV (`stats.json`, X-010..X-015); the
  full-DAT record counts come from a separate pass over the complete reduced DAT (X-016). Every
  experiment script then mines the ≥2-item subset (`utils.load_transactions(min_items=2)`).

**Mining (Phase 3, `phase2/scripts/run_phase3.sh`, all on 2 × RTX 3090).** Row-split
2-GPU exactness gate (`validate_row_split.py`), Opus run with per-K parquet flush
(`run_mining.py --support 1e-7 --max-length 50 --n-gpus 2 --parquet-flush`), deepest-itemset
analysis (`analyze_k22_proteins.py --itemset-source frequent_k<KMAX>.parquet`), Direct-vs-SON at
0.001 % (`experiment_direct_vs_son.py`, chunk 40,000,000, local factor 0.9), the permutation null
model at min_count 769 with 5 permutations and seed 42 (`experiment_null_model.py`), the
six-threshold campaign (`experiment_full_campaign.py --runs 1`), SON at 0.1 % and 0.01 %
(`phase2/scripts/son_run.py`, same defaults as the Direct-vs-SON script), plus the log-only
thresholds min_count 4 and 3. Deviations from the committed scripts: three stale import paths
were corrected (I-007); `--runs 1` instead of 3 for the campaign and Direct-vs-SON (single values
are what the paper reports); the SON chunking parameters of the original Base/Super/Power runs
are not recorded anywhere (claims_logs_new.md §C-19), so the current script defaults were used.

**Verdict rules (from the briefing).** Deterministic values (counts, supports, K, statistics,
vocabulary sizes) must match the fresh value EXACTLY → *confirmed*, else *hallucinated*; a claim
that is explicitly rounded ("26.8M", "37.4 %", "~130 s", "214 million") is confirmed only if the
fresh value rounds (or truncates, for "214 million"-style figures) to the claim at the claim's own
precision. Hardware-dependent values (timings, throughput, memory bandwidth) →
*expected-hardware-deviation* with both values shown. Values that cannot be re-executed here
(facts about other systems, citation details, statements about the original setup, quantities
whose run did not complete) → *inconclusive* with the reason. Permutation-null statistics
(per-K null mean/σ/Z/p, null totals) depend on the random stream: the same seed and script were
used, but the transaction row order (a different metadata export) and the CuPy version differ, so
exact equality is not expected; such a claim is *confirmed* only when it matches at its stated
precision and otherwise *inconclusive* with the fresh value shown — never hallucinated on that
basis alone. The biological ("real") counts and the K at which the null vanishes are exact
quantities and are judged strictly. Every fresh value cites its artifact
path (Section 5); no fresh value is taken from an old log. The one place an old log is used at all
is as a fingerprint: its parsed-record count identified the UniProt release, and that count was then
re-derived from the freshly downloaded release (RESULTS.md X-001, X-016).

**Headline results in one paragraph.** The dataset statistics, the vocabulary, all six exhaustive
mining counts, the full Opus K-distribution, the K=22 itemset (members, support 8, eight proteins
with exactly 22 features), the highlighted intermediate patterns, and the structure of the null
model (null vanishes above K=6; real data hold 88,745 itemsets at K ≥ 7) reproduce exactly. Three
things do not: (1) the paper's *streaming-SON* figures (22,846 / 51,124 itemsets, K_max 13, 95.2 %
miss rate, 21× speedup) — the released streaming code is exact and returns the exhaustive counts
(I-011); (2) the stated UniProt release (2025_01 vs the 2026_01 actually used, I-006) and a few
method details (the medium pLDDT bin is 50–90, not 70–90; the largest protein carries 46 vocabulary
items, not 22); (3) the headline "Z > 3,700 for K = 4–6": with the same seed but a different random
stream the fresh Z at K = 4 is 3,048 (K = 5: 9,917; K = 6: 175,697), so that specific bound is not
met at K = 4, while the qualitative conclusion (enormous enrichment, zero null itemsets at K ≥ 7)
holds. All timings are 2–8× slower on the RTX 3090s than the paper's H100 figures, as expected.
