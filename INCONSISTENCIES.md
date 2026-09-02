# INCONSISTENCIES.md — discrepancies between scripts, logs, reviews, and the .tex

Each entry: what disagrees, where (file:line or artifact), and the evidence
command used to confirm it on this machine. "Reported-by" marks items first
surfaced by an extraction subagent; "verified" means re-checked directly.

## I-001 — min_count 768 vs 769 for the same support and N (verified)

- `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json`:
  `parameters.min_support = 1e-05`, `parameters.n_transactions = 76890945`,
  `parameters.min_count = 768`.
- `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json`:
  `parameters.min_support = 1.0001177641918693e-05` (= 769/76890945),
  `parameters.min_count = 769`, `parameters.recomputed_real = False`, and
  `real_distribution` is byte-identical to the direct-GPU `k_distribution`
  of the first file (copied, not re-mined at 769).
- ceil(1e-5 × 76,890,945) = ceil(768.909) = 769; 768 is the floor. The two
  artifacts therefore describe the same "real" itemset table under two
  different thresholds. Evidence: `python3` walk of both JSONs
  (2026-09-02T00:25Z).

## I-002 — "89,566 itemsets at K ≥ 7" vs JSON sum 88,745 (verified)

- `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` prose:
  "89,566 real itemsets at K >= 7 ..." (two occurrences).
- `experiment_null_model_20260219_061046.json` `real_distribution` K=7..14 =
  48699+25011+10508+3488+869+152+17+1 = 88,745. Evidence: `python3` sum
  (2026-09-02T00:25Z).

## I-003 — three different mining runs conflated as "the" base214m result (reported-by logs subagent; partially verified)

- 475,865 itemsets @ min_support 1e-5 (0.001 %), K_max 14, 50.72 s
  (`experiment_direct_vs_son_...json` direct_gpu_result — verified).
- 2,841,280 itemsets @ 1e-6 (0.0001 %), K_max 19, 119.3 s
  (`blitz_reference` in the same JSON — time/K verified; itemset count as
  reported by the subagent, to re-check).
- "26.8M itemsets, K ≤ 22, 7.3 min on H100" (README / GLOSSARY / RUNBOOK
  prose; no artifact survives — reported-by subagent, to re-check against
  the .tex claims).

## I-004 — protein count: 76,890,945 vs "214M" vs "76.9M of 205.6M" (reported-by logs subagent)

- All surviving artifacts use n_transactions = 76,890,945.
- README / GLOSSARY speak of the 214M-protein set; `deploy/RUNBOOK_base214m.md`
  speaks of 76.9M out of 205.6M (Dutch: "76,9M van 205,6M"). To reconcile
  against the extraction filters in the pipeline code (Phase 1) and the
  .tex method section.

## I-005 — surviving timings are all 2×RTX 3090 bench runs; AlphaFold timings exist only as H100/H200 prose (reported-by logs subagent)

- `bench/results/2026-08-31-3090x2/*` and `bench/results/2026-09-01-3090x2-sparse/*`
  are synthetic-preset benchmarks on 2×3090 (driver 580.159.03).
- Every AlphaFold/base214m timing in README/GLOSSARY/RUNBOOK refers to an
  H100 or H200 and has no surviving log. These can only be judged as
  expected-hardware-deviation against fresh 3090 timings.

## Operational hazard (not a discrepancy)

- `deploy/RUNBOOK_base214m.md` line 65 sets `ET_UPLOAD_GCS=1 ET_UPLOAD_TAG=...`
  to push results to GCS. The campaign rules forbid any upload; every run in
  this campaign must leave `ET_UPLOAD_GCS` unset/0 and the deploy step that
  pushes must not be executed.

## I-006 — the paper's stated UniProt release (2025_01) does not match the release actually used (2026_01) (verified)

- `paper/et_miner_proteome.tex` l.129: "UniProt ... (release 2025\_01, accessed
  February 2026)"; l.478 and l.529 repeat "release 2025\_01".
- `applications/alphafold/results_214m/logs/pipeline_214m.log` l.2036
  (run of 2026-02-09): "Loaded 202556314 annotations (25475 Pfam domains,
  26536 GO terms)" from `/root/uniprot_trembl.dat.gz`; the runbook downloads
  `current_release/.../uniprot_trembl.dat.gz` (RUNBOOK_base214m.md l.66-67).
- Official release notes fetched fresh on 2026-09-02 (see RESULTS.md U-001..U-006):
  TrEMBL entry counts — 2025_01: 252,633,201; 2025_02: 252,188,522;
  2025_03: 253,061,697; 2025_04: 199,006,240; **2026_01: 202,556,314**;
  2026_02: 149,234,636. Only 2026_01 equals the log's record count exactly.
- Consequence: the deterministic values in the paper (if genuine) derive from
  release 2026_01, and the method section misstates the release. The
  campaign therefore reproduces on BOTH releases: 2026_01 (log-faithful,
  primary) and 2025_01 (paper-as-stated, secondary). Verdicts for
  deterministic claims are judged against the 2026_01 run, with the 2025_01
  value shown alongside.
- Related: the current release (2026_02) TrEMBL dat.gz is 118,072,276,938 B,
  so "150 GB of compressed annotation data" (tex l.129) cannot refer to
  today's file either; the fresh member sizes of the 2025_01 and 2026_01
  dat.gz files are recorded in RESULTS.md when the streams report them.

## I-007 — experiment scripts import modules that no longer exist (verified; fixed for this campaign)

- `experiments/experiment_null_model.py` l.200/460/542 imported
  `et_miner.cuda_csr_bitvec` and l.461 `et_miner.apriori`;
  `experiments/compute_maximal.py` l.52 and `pipeline/filter_self_sufficient.py`
  l.45 imported `et_miner.rules`. None of these modules exist on this tree
  (`src/et_miner/{cuda_csr_bitvec,apriori,rules}.py` absent); the symbols
  live in `et_miner.gpu.csr_bitvec`, `et_miner.gpu.row_split`, and
  `et_miner.core.rules`. The suite could not have run unmodified on the
  committed code. Fix applied 2026-09-02T00:50Z (import paths only, no logic
  change): `git diff applications/alphafold` shows the three files.

## I-008 — the surviving mining logs were not produced by any script in the repo (reported-by pipeline subagent; spot-checked)

- Strings such as `>>> Total time:`, `K=1: 1,002 candidates → 1,002 frequent (289ms)`
  and `>>> Saved to /root/alphafold-data/full_214m/itemsets_214m_*.parquet`
  in `results_214m/logs/*.log` occur nowhere in `src/` or `applications/`
  (`grep -rn` 2026-09-02T00:49Z); only `Direct CSR path: {} transactions, min_count={}`
  matches (`src/et_miner/core/matrix.py:502`). The exact driver scripts of
  the February 2026 runs are not in the repository; the reproduction uses
  the committed experiment scripts and library entry points instead.

## I-009 — the runbook's `bq query --format=csv --max_rows=300000000` export route does not work in practice (verified)

- `deploy/RUNBOOK_base214m.md` l.70-73 and `pipeline_214m.py` l.429-436 prescribe
  exporting the 214.7M-row metadata table with the `bq` CLI. Run here
  (2026-09-02T00:33Z, `phase2/scripts/export_plddt_bq_cli.sh`): the query job
  finished in 5 s but the CLI wrote 0 bytes in 30 min while its RSS grew to
  3.9 GB (paging all rows into memory). Killed at 01:04Z.
- The campaign used the BigQuery Storage Read API on the same table and the
  same two columns instead (`phase2/scripts/export_plddt_storage_api.py`,
  60 s, 16 streams); the resulting CSV reproduces the row/pLDDT counts of the
  old log exactly (RESULTS.md M-006..M-009).
- The old `watcher.log`/`pipeline_214m.log` show the CSV had "214683830 rows"
  (header included) — consistent with any export route.

## I-010 — `experiment_null_model.py` cannot run on 24 GB GPUs without releasing cached GPU memory (verified; one-line fix applied)

- Both GPU modes assume an 80 GB H100 (`--perm-per-gpu` docstring l.402-405). On this box the
  per-GPU worker mode failed with a CuPy OutOfMemory (2.5 GB request with 8.2 GB held) and the
  sequential mode failed allocating the 9.7 GB permutation bit-vector while the CuPy pool still
  cached 22.4 GB from the real-distribution mining (`phase3/2026_01/null_model_769_5.permgpu_failed.log`,
  `null_model_769_5.log` first attempt).
- The sequential branch still fails after inserting `cp.get_default_memory_pool().free_all_blocks()`
  before `_prepare_gpu_resident_data` (peak ≈ 25 GB: two item buffers + two 9.7 GB bit-vector
  copies); that memory-hygiene line is left in place (`git diff applications/alphafold/experiments/experiment_null_model.py`).
- The unmodified 2-GPU row-split branch (`--n-gpus 2` without `--perm-per-gpu`; bit-vectors split
  4.8 GB per GPU) runs: 5 permutations in 107.6 s wall (`phase3/2026_01/null_model_769_5.log`,
  `exp/experiment_null_model_20260902_070216.json`). Row-split exactness was verified by the
  pre-flight gate (RESULTS P-001).

## I-011 — the committed SON implementation is exact at 0.001 %, contradicting the paper's lossy-SON numbers (verified)

- Paper Table 2 / §Results (tex l.224-238, 404): Streaming SON at 0.001 % found 22,846 itemsets
  (K_max 13) in 1,085.6 s and "misses 95.2 %" of the 475,865 exhaustive itemsets; the same values
  are in `results_214m/logs/extreme_mining.log` ("MAX LENGTH 20") and in the surviving JSON's
  `son_reference`.
- Fresh run of the committed `experiment_direct_vs_son.py` (SON defaults: chunk_size 40,000,000,
  local_support_factor 0.9, GPU): SON returns exactly the Direct result — 475,865 itemsets, K=14,
  miss rate 0.0 % — in 5,673.7 s (CPU-bound pass 2), i.e. the current `apriori_streaming` is
  lossless at this threshold (`phase3/2026_01/exp/experiment_direct_vs_son_20260902_083831.json`).
- The SON configuration of the February-2026 runs (chunking, local factor, length cap, code
  version) is not recorded in any log (claims_logs_new.md §C-19); the paper's SON counts,
  K_max values and miss rate are therefore not reproducible from the released code and are
  judged by the exact-match rule against the fresh SON result.

## I-012 — association-rule counts of the old pipeline step do not follow from the released rule generator (verified)

- `results_214m/logs/pipeline_214m.log` Step 4: 53,447 rules at min_confidence 0.5 from the Base
  SON itemsets (51,126 with lift ≥ 5; 40,428 with lift ≥ 100; 45,286 "cross-domain"; max lift 969).
- Fresh: the same 5,305 Base itemsets (identical count and K_max) through
  `et_miner.core.rules.generate_rules(min_confidence=0.5)` give 29,540 rules (17,499 with lift ≥ 5;
  9,283 with lift ≥ 100; 27,072 spanning more than one feature category); the maximum lift, 968.996,
  matches the old "969" (`phase3/2026_01/exp/son_base.json`). The rule generator (and the
  cross-domain definition) used in February 2026 are not in the repository (`pipeline_214m.py`
  has no rules step today), so only the max-lift value is reproducible.
