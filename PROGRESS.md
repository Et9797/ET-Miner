# PROGRESS.md — base214m reproduction campaign

Durable state for the multi-hour run. Re-read this and RESULTS.md after any
context compaction. Operating rules live in CLAUDE.md.

## Goal (set 2026-09-02T00:02Z by Et via /goal)

Reproduce every experimental value from the ET-Miner AlphaFold preprint and
produce COMPARISON_REPORT.md mapping EVERY value in paper/et_miner_proteome.tex
(plus every value in the paper reviews and the log files) to
source → freshly reproduced value → verdict {confirmed | hallucinated |
inconclusive | expected-hardware-deviation}. Goal met ONLY when
COMPARISON_REPORT.md exists, is printed to the transcript, covers every
extracted claim with no TODO/pending rows, and every reproduced value cites a
fresh artifact path or exact command (never an old-log value); all artifacts
stay on this machine. If the box cannot hold the data or the run is otherwise
impossible → ABORT_REPORT.md with disk/bandwidth evidence counts as completion.
Stop and hand back to Et on unrecoverable error or when a human decision is
needed.

## Run directory

- `runs/20260902T0000Z/` (created 2026-09-02T00:04Z; subdirs logs/, phase1/,
  phase2/, phase3/, phase4/). `runs/` is git-ignored.
- Status loop: session cron job `feb62618`, every 5 min, prompt
  "inform me of the run status" (session-only, expires after 7 days).

## Status

- Current phase: **COMPLETE (2026-09-02) — COMPARISON_REPORT.md final and printed to the transcript in 33 chunks; all goal conditions met. Status cron job feb62618 may still be armed (cancel with CronDelete).**
- Last completed step: Phase 0 audit; run dir created; 5 Phase-1 subagents
  launched 2026-09-02T00:04Z (pipeline reconstruction, .tex claims,
  reviews claims ×2, log claims) writing into `runs/20260902T0000Z/phase1/`
- Next step: merge subagent outputs → CLAIMS.md + INCONSISTENCIES.md;
  decide Phase 2 data route from the raw-data requirement found in the code
  (full 214M vs. subset vs. metadata-only) and the measured GCS bandwidth

## Phase 0 — hardware / disk audit (2026-09-01T23:58Z)

- [x] GPU: **2 × NVIDIA GeForce RTX 3090, 24576 MiB each, compute cap 8.6**
      (`nvidia-smi --query-gpu=...`). Driver 595.71.05, CUDA runtime 13.2,
      `nvcc` 12.1. **This box is NOT an H100** — all timing/throughput
      claims must be judged as expected-hardware-deviation.
- [x] CPU: AMD EPYC 7402P, 24 cores / 24 threads, 1 socket, 1 NUMA node.
- [x] RAM: host shows 125 GiB total; container cgroup limit
      `/sys/fs/cgroup/memory.max` = 74,782,343,168 B (~69.6 GiB);
      container usage at snapshot ~1.3 GiB. Swap 8 GiB (host, fully used
      by other tenants).
- [x] Disk (container root, overlay): **200 GB total, 200 GB available**
      (939 MB used). Inodes: 195,351,424 total, 191,245,709 free.
      The host NVMe (/dev/nvme0n1, 1.9 TB, 850 GB free) is NOT mounted
      for data use — only bind-mounted for /etc/hosts etc.
- [x] Disk verdict vs. full AlphaFold v4 requirement (~23 TiB, ~644M files
      after untar): **the full dataset is plainly infeasible here**
      (200 GB / 191M inodes). A subset route (BigQuery manifest +
      `gcloud storage cp --read-paths-from-stdin`, or the proteome tar
      shards) is the only viable path; decision pending the goal message.
- [x] gcloud: installed at `/root/projects/downloads/google-cloud-sdk/bin/gcloud`
      (SDK 583.0.0, bq 2.1.38), authenticated as etje975@gmail.com.
- [x] Tooling: python3 `/opt/conda/bin/python3`, rsync, curl, wget, tar,
      tmux present. **Missing: uv, aria2c, pigz.**
- [x] Existing AlphaFold data on disk: none found
      (`find / -xdev -iname '*alphafold*' ...` only hits repo files).
- [x] Other tenants of this container: a jupyter-notebook server (PID 407)
      and the vast.ai ssh forwarder; nothing else consuming GPU/RAM.

## Phase 1 — checklist

- [x] pipeline_reconstruction.md (subagent) → phase1/ (710 lines; VERDICT: base214m reads NO structure files — inputs are uniprot_trembl.dat.gz + BigQuery metadata CSV; paper pins UniProt 2025_01)
- [x] claims_tex.md (subagent) → phase1/ (543 rows T-001..T-543: deterministic 343, method-parameter 90, external-fact 55, hardware-dependent 41, software 14)
- [x] claims_reviews_part1.md (835 rows R1-001..R1-835) + claims_reviews_part2.md (505 rows R2-001..R2-505) → phase1/
- [x] claims_logs_new.md (690 rows L2-001..L2-690; the 11 logs pushed in e10801c) → phase1/
- [x] phase4/quantities.csv (336 qkeys) + claims_tex_keyed.csv (543 rows keyed) → registry for the comparison join
- [x] claims_logs.md (subagent) → phase1/ (948 rows; NO AlphaFold mining log survives; only results_214m/*.json + decoded_top_k_patterns.txt)
- [ ] CLAIMS.md assembled at repo root (every claim with ID + source)
- [x] INCONSISTENCIES.md started (I-001..I-005 + upload hazard) (scripts vs .tex vs reviews vs logs)
- [x] GCS bucket layout + inbound bandwidth measured → phase2/ (RESULTS.md E-005..E-013: ~157 MB/s wall, 317 MiB/s in-transfer; shard = 10k proteins, ~71 KB/protein)
- [ ] Phase 2 route decided (full / subset / metadata-only / abort)

## Phase 0 — inventory of surviving files

- `paper/et_miner_proteome.tex` (65,595 B) — V1 paper source (CLAIMS source)
- `paper/PAPER_V2_REVIEW.md` (19,720 B) — review
- `paper/peer_review_jul12.md` (27,049 B) — review
- `paper/review_b1_hostile.md` (29,488 B) — review
- `paper/review_b2_results.md` (16,203 B) — review
- `paper/revision_notes_b3.tex` (32,374 B) — revision notes
- `paper/senior_review_jun01.md` (15,269 B) — review
- `paper/senior_review_mar23.md` (5,493 B) — review
- `applications/alphafold/` — AlphaFold application: `af-extract/` (Rust
  extractor: tar_stream/confidence/annotations/transaction), `deploy/`
  (RUNBOOK_base214m.md, deploy_base214m.sh, run_mining.sh,
  run_all_experiments.sh, run_null_model_35k.sh, deploy_project_milky_way.sh),
  `pipeline/` (download_alphafold.py, extract_features.py, pipeline_214m.py,
  run_mining.py, postprocess_tx.py, cluster_sequences.py,
  filter_self_sufficient.py, resolve_feature_names.py, validate_motifs.py),
  `experiments/` (experiment_full_campaign.py, experiment_null_model.py,
  experiment_direct_vs_son.py, compute_maximal.py, analyze_k22_proteins.py,
  systematic_bio_analysis.py, validate_row_split.py,
  analysis_alpha_centauri.ipynb), `results_214m/` (GLOSSARY.md,
  decoded_top_k_patterns.txt, experiment_direct_vs_son_20260219_050326.json,
  experiment_null_model_20260219_061046.json), `rust/` (build_tx, remap_tx,
  dat_to_tsv), utils.py, requirements.txt
- Log-like files: `bench/results/2026-08-31-3090x2/{FINDINGS.md,env.txt,
  raw.jsonl,report.md}` and `bench/results/2026-09-01-3090x2-sparse/{...}`
  (bench campaign on 2×3090, NOT AlphaFold); the two `results_214m/*.json`
  files above are the only surviving AlphaFold result artifacts
- `bench/` — GPU campaign harness (runner.py, child_run.py, selfcheck.py,
  run_smoke.sh, run_full.sh, setup_box.sh, results/)
- `src/et_miner/` — miner library; `rust_ext/` — Rust extension;
  `tests/` — test suite; `datasets/` — non-AlphaFold dataset prep
- Log files: 11 old mining/pipeline logs pushed by Et in commit e10801c
  (2026-09-02T00:40Z) under `applications/alphafold/results_214m/logs/`
  (pipeline_214m.log 2,377 lines, watcher.log 2,386 lines, plus 9 mining
  logs). CLAIMS only — never adopt their values.

## Long-running jobs (PID / log / resume command)

- CHAIN (started 2026-09-02T01:12Z, PID 16531): waits for the 2026_01 stream → runs
  `phase2/scripts/run_af_extract.sh 2026_01` (log logs/af_extract_2026_01.log,
  outputs phase2/extract_2026_01/) → runs `phase2/scripts/run_phase3.sh 2026_01`
  (log logs/phase3_2026_01.log, per-step logs + .done markers in phase3/2026_01/).
  Chain log: logs/chain_2026_01.log. RESUME after interruption: if extraction
  finished (phase2/extract_2026_01/stats.json exists) run
  `setsid phase2/scripts/run_phase3.sh 2026_01 > logs/phase3_2026_01.log 2>&1 &`
  (skips steps with .done markers); else re-run the chain script.

- (STOPPED 01:04Z, user decision) TrEMBL 2025_01 stream: PID 8860,
  log runs/20260902T0000Z/logs/stream_trembl_2025_01.log + phase2/data/stream_trembl.log;
  output phase2/data/uniprot_trembl_2025_01.reduced.dat.gz(.partial).
  Resume: re-run `runs/20260902T0000Z/phase2/scripts/stream_trembl_2025_01.sh` (restarts from scratch; no mid-stream resume).
- (KILLED 01:04Z, see I-009) bq CLI CSV export (runbook route): PID 8730, log logs/export_plddt_bq_cli.log,
  output phase2/data/plddt_metadata.csv(.partial). Resume: re-run phase2/scripts/export_plddt_bq_cli.sh.
- Storage Read API CSV export (cross-check): PID 8981 (started 00:37Z), log logs/export_plddt_storage_api.log,
  output phase2/data/plddt_metadata_storageapi.csv. Resume: `TOKEN=$(gcloud auth print-access-token); python3 phase2/scripts/export_plddt_storage_api.py <out> --token $TOKEN`.
- uv sync: PID 5490 (started 00:20Z), log logs/uv_sync.log.
- Subagent: claims extraction from the 11 newly pushed logs (commit e10801c) → phase1/claims_logs_new.md

## Environment setup jobs (started 2026-09-02T00:20Z)

- [x] uv installed → /root/.local/bin/uv (log: runs/20260902T0000Z/logs/install_uv.log)
- [x] pigz installed via apt (log: runs/20260902T0000Z/logs/apt_pigz.log)
- [x] rustup DONE (cargo 1.98.0)
- [x] uv sync + uv sync --extra gpu DONE (cupy 14.1.1, 2 devices visible) — logs uv_sync.log, uv_sync_gpu.log
- [x] rust_ext built + installed (et_miner_rust 0.2.0; log logs/build_rust_ext.log)
- [x] af-extract built (target/release/af-extract, 1m18s; log logs/build_af_extract.log)

## Phase 2 — data route decision (2026-09-02T00:30Z)

- Code verdict (phase1/pipeline_reconstruction.md §2): base214m reads NO
  structure files. Inputs = `uniprot_trembl.dat.gz` (Pfam + GO cross-refs)
  + a 2-column CSV (`uniprotAccession,globalMetricValue`) from BigQuery.
  The 23 TiB structure download is NOT required → no ABORT.
- Paper pins UniProt release **2025_01** (tex l.129, 478, 529). The runbook
  URL uses `current_release` (now 2026_02, 118.07 GB dat.gz) — mismatch
  logged. EBI archive: `previous_releases/release-2025_01/knowledgebase/knowledgebase2025_01.tar.gz`
  = 194,669,985,833 B (181 GiB, combined sprot+trembl tarball; no trembl-only
  tarball exists for 2025_01). Plan: stream the tarball, extract the
  `uniprot_trembl.dat.gz` member, keep only the record lines af-extract
  parses, recompress → reduced dat.gz (fits disk); never store 181 GiB.
- BigQuery access verified (project et-research-491307): SQL used →
  `SELECT COUNT(*) AS n_rows, COUNT(DISTINCT uniprotAccession) AS n_accessions, COUNTIF(globalMetricValue >= 50) AS n_plddt_ge50, COUNTIF(globalMetricValue IS NULL) AS n_null_plddt, MIN(globalMetricValue) AS min_plddt, MAX(globalMetricValue) AS max_plddt FROM \`bigquery-public-data.deepmind_alphafold.metadata\``
  → RESULTS.md M-001..M-005 (214,683,829 rows; 205,620,298 with pLDDT ≥ 50).
- [x] plddt_metadata_storageapi.csv DONE + validated (RESULTS M-006..M-009);
  bq CLI route (PID 8739) still buffering in RAM after 6 min with 0 bytes written — kept as cross-check only, kill if RSS > 15 GB
- [x] 2025_01 stream STOPPED at 01:04Z on Et's instruction (disk; release identity settled by the
  fingerprint) — partial output deleted; 2025_01 will NOT be mined (paper-as-stated check not run).
- [x] reduced uniprot_trembl_2026_01.dat.gz DONE 01:28Z (27.1 GB; RESULTS U-009..U-011); extraction started 01:28:44Z via the chain
  2026_01 stream started 2026-09-02T00:44Z (log-faithful PRIMARY: TrEMBL 2026_01 has
  202,556,314 entries = old log's count; see INCONSISTENCIES I-006). Resume either:
  `phase2/scripts/stream_trembl_release.sh <REL> <tarball>` (restarts from scratch)
- [x] af-extract build-from-metadata DONE (lean route attempt 3, 05:14→05:23Z) → phase2/extract_2026_01/{transactions_214m_base.parquet, item_mapping_214m_base.parquet}
- [x] ≥2-item filter → transactions_214m_base_multi.parquet (n = 76,890,945)
- [x] bench/selfcheck.py READY (19 kernels, NCCL OK, no P2P) and tests/test_tier_equivalence.py 9 passed (RESULTS E-014, E-015)

## Log ↔ paper-run mapping (from phase1/claims_logs_new.md §B/§C; CLAIMS, to reproduce)

| paper row | threshold | min_count (log) | log file | logged result (claim) |
|---|---|---|---|---|
| Base | 1e-3 SON | 76,891 (floor 76,890?) | pipeline_214m.log Step 3 (old pipeline_214m.py; no SON step in current code) | 5,305 itemsets, K≤9, 113.9 s |
| Super | 1e-4 SON | "7,689" | ultra_mining.log | 51,124, K13, 257.6 s |
| Power | 1e-5 SON | "768" | extreme_mining.log (MAX LENGTH 20) | 22,846, K13, 1,085.6 s |
| Power (direct, Table 3/null) | 1e-5 direct | 768 (JSON) | no log; results_214m/experiment_direct_vs_son_*.json | 475,865, K14, 50.72 s |
| Blitz | 1e-6 direct | 77 | direct_mining.log | 2,841,280, K19, 119.3 s |
| Ultra | 2e-7 direct | 16 | beyond_mining.log | 14,558,875, K20, 281.0 s |
| Opus | 1e-7 direct | 8 | godmode_mining.log | 26,849,505, K22, 440.5 s |
| (no paper row) | min_count 4 | 4 | holdmybeer_real.log | 48,007,493, K22, 1,228.5 s |
| (no paper row) | 5.2e-8 | ambiguous | holdmybeer.log | 18,935,899, K21, 573.1 s |
| (no paper row) | min_count 3 | 3 | yolo.log | no result |
| (no paper row) | 1e-6 SON | "76" | madman_mining.log | aborted |
| null model | 769, 5 perms, seed 42 | 769 | results_214m/experiment_null_model_*.json | null mean total 171,320; K≤6 |

## Phase 3 plan (2026_01 data)

- [x] P3.0 extraction DONE 05:24Z (lean route, attempt 3): 24,291 Pfam / 25,993 GO / 76,890,945 multi (37.39 %) / nnz 316,421,093 — RESULTS X-008..X-015
- [x] P3.1 validate_row_split GREEN (168,674 itemsets identical) — RESULTS P-001
- [x] P3.2 Opus DONE 06:09Z: 26,849,505 itemsets, K=22, kdist exact vs paper, 2,647 s — RESULTS P-002..P-008; Blitz DONE 06:19Z (2,841,280, K19) — P-011
- [x] P3.3 full_campaign_r1 DONE 09:16Z — RESULTS P-014 (all exhaustive counts match the paper; Super direct 113,405 vs paper SON 51,124)
- [x] P3.4 direct_vs_son DONE 08:38Z: direct 475,865/K14/91.7 s; SON 475,865/K14/5,673.7 s — SON is EXACT (miss 0 %) vs paper's 22,846/K13/95.2 % → I-011; RESULTS P-013
- [x] P3.5 SON Base 5,305/K9 (P-015) and Super 113,405/K14 (P-016) DONE 09:53Z — both exact (I-011)
- [x] P3.6 null model 5 perms DONE 07:02Z (row-split branch; RESULTS P-012); null100 queued last
- [x] P3.7 analyze_k22 DONE 06:22Z (re-run after go.obo 403; curl-fetched go.obo into phase3/2026_01/exp/): 8 proteins × 22 features, 0 parent-child pairs — P-009
- [~] P3.8 minc4 queued; minc3 SKIPPED (marker pre-set; yolo.log has no result); rules via son_base
- [ ] P3.9 compute_maximal.py — smoke test did not finish in 10 min on 128k itemsets; only if a claim needs it

## Phase 4 — keying subagents (launched 2026-09-02T01:14Z)

- [x] claims_reviews_part1_keyed.csv (835 rows; 195 new keys)
- [x] claims_reviews_part2_keyed.csv (505 rows; 251 new keys)
- [x] claims_logs_keyed.csv (948 rows; 334 new keys)
- [x] claims_logs_new_keyed.csv (690 rows; 347 new keys)
- [x] merge not needed: build_comparison.py globs claims_*_keyed.csv + quantities.csv + new_quantities_*.csv (3,521 claims, 1,463 keys, 0 unkeyed — verified 05:30Z)
- [x] phase4/build_fresh_values.py written; interim run on extraction outputs = 63 keys (fresh_values_interim.json)
- [~] phase4/build_comparison.py written + rules fixed (multipliers, precision, mode by reproducibility class); interim report phase4/COMPARISON_interim.md
- [x] adversarial review subagent → fix gaps → print COMPARISON_REPORT.md
- [x] phase4/build_fresh_values.py written + tested on smoke outputs (273 keys)
- [x] phase4/build_comparison.py written + parse rules tested (21/21)
- [x] phase4/make_env_json.py → phase4/env.json (17 keys); phase4/header.md drafted
- FINAL ASSEMBLY COMMANDS (after Phase 3):
  `.venv/bin/python runs/20260902T0000Z/phase4/build_fresh_values.py --run-dir runs/20260902T0000Z --extract-dir runs/20260902T0000Z/phase2/extract_2026_01 --phase3-dir runs/20260902T0000Z/phase3/2026_01 --env-json runs/20260902T0000Z/phase4/env.json --out runs/20260902T0000Z/phase4/fresh_values.json`
  `.venv/bin/python runs/20260902T0000Z/phase4/build_comparison.py --phase4 runs/20260902T0000Z/phase4 --fresh runs/20260902T0000Z/phase4/fresh_values.json --out COMPARISON_REPORT.md --inconsistencies INCONSISTENCIES.md --header runs/20260902T0000Z/phase4/header.md`

## Extraction attempt 1 FAILED (OOM, rc=137, 2026-09-02T01:46:52Z) → attempt 2 plan (lossless memory reduction)

- Cause: af-extract holds the whole annotation map (202.6M records incl. InterPro/EC/taxonomy
  vectors) plus every CSV row in RAM; the 69.6 GiB cgroup killed it during the CSV read.
- Fix (no code change, provably identical output for the mined set):
  1. awk pass over the reduced DAT → keep only records with ≥1 `DR Pfam`/`DR GO` line and only
     their ID/AC/DR Pfam|GO/`//` lines → `phase2/data/uniprot_trembl_2026_01.pfamgo.dat.gz`
     + accession list `phase2/data/dat_pfamgo_accessions_2026_01.txt`.
  2. polars: CSV rows (pLDDT ≥ 50 ∩ accession in that list) → `phase2/data/plddt_metadata_pfamgo.csv`.
     Proteins outside this set can only be single-item (pLDDT-bin) transactions; they never
     enter the ≥2-item mined set and cannot change Pfam/GO frequency ranks.
  3. af-extract on (1)+(2) with identical parameters; extract_stats.py reconstructs the
     full-set totals arithmetically from the full CSV counts (n_total = 205,620,298 pass rows).
- Session note: the three keying subagents were killed by an API session limit (reset 04:30Z); relaunched 04:32Z.
- CHAIN2 (started 2026-09-02T04:36Z): waits for phase2/data/uniprot_trembl_2026_01.pfamgo.dat.gz
  (filter_dat_pfamgo.sh, log logs/filter_dat_pfamgo.log) → run_af_extract_lean.sh 2026_01
  (log logs/af_extract_lean_2026_01.log) → run_phase3.sh 2026_01 (log logs/phase3_2026_01.log).
  Chain log: logs/chain2_2026_01.log. Resume: re-run `phase2/scripts/chain2.sh 2026_01` (or
  run_phase3.sh directly once phase2/extract_2026_01/stats.json exists).
- Attempt 2 (pfamgo DAT, 158.7M records) killed pre-emptively at 04:58Z with RSS 63 GB while still
  parsing (≈350 B/record → OOM certain with the CSV structures). CHAIN3 (04:59Z): filter the DAT to
  the 90,506,840 CSV accessions (`filter_dat_by_accessions.sh`, log logs/filter_dat_by_accessions.log)
  → `DAT_FILE=…csvacc.dat.gz run_af_extract_lean.sh 2026_01` (log logs/af_extract_lean_2026_01.log)
  → run_phase3.sh. Chain log: logs/chain3_2026_01.log. Resume: re-run `phase2/scripts/chain3.sh 2026_01`.
- 06:52Z INCIDENT: direct_vs_son SON leg OOM-killed (rc=137, cgroup oom_kill=2) when the null model was
  started manually alongside it (SON 36 GB + null 43 GB > 69.6 GiB). Sequencer (chain3/run_phase3) and the
  concurrently started campaign step were stopped at 06:53Z. RULE: never co-schedule two mining jobs.
  RESUME: after phase3/2026_01/null_model_769_5.log shows "manual null rc=0", run
  `setsid runs/20260902T0000Z/phase2/scripts/run_phase3.sh 2026_01 >> runs/20260902T0000Z/logs/phase3_2026_01.log 2>&1 &`
  (resumes at direct_vs_son; null_model_769_5.done already exists — delete it first if the manual run failed).
- 07:0xZ null model (5 perms, seed 42) DONE via the 2-GPU row-split branch (`--n-gpus 2`, no --perm-per-gpu;
  per-GPU worker mode and single-GPU sequential mode both OOM on 24 GB cards — INCONSISTENCIES I-010).
  Sequencer resumed (`run_phase3.sh 2026_01`, resumes at direct_vs_son; SON leg is CPU-bound ~1 h).
- 11:20Z run_phase3.sh finished; null100 step failed (per-GPU worker OOM) and bench step failed (wrong cwd).
  CHAIN_TAIL (phase2/scripts/chain_tail.sh, log logs/chain_tail.log): null100 via row-split (log
  phase3/2026_01/null_model_769_100.log, --checkpoint --resume so it can be re-run) → bench matrix from the
  repo root (phase3/2026_01/bench/, log bench_full.log). Resume: re-run chain_tail.sh.
- [x] P3.8 minc4 DONE 11:09Z: 48,007,493 / K22 / 4,536.9 s — RESULTS P-017 (matches holdmybeer_real.log)

## 11:56Z status
- [x] null100 DONE (row-split branch; RESULTS P-018). [x] Adversarial review #1 done on the checkpoint report;
  all defects addressed in phase4 (staging report). [~] bench matrix relaunched 11:55Z after generating
  datasets/synth presets (`python -m et_miner.synthetic --preset all --out datasets/synth`); log
  phase3/2026_01/bench_full.log, output phase3/2026_01/bench/ (first attempt kept in bench_failed_attempt1/).
- FINAL ASSEMBLY after bench_full.done: builder (with bench parser) → build_comparison → COMPARISON_REPORT.md →
  sanity checks → light re-review → print to transcript.
- 12:20Z Re-verification #2 passed (13/14; the one row fixed: full-precision fractions). Staging report:
  3,521 claims → confirmed 1,285 / hallucinated 119 / expected-hardware-deviation 341 / inconclusive 1,776
  (bench rows still partly inconclusive until the matrix finishes). Bench: 11/27 configs done; stress_k2
  legacy/shared configs (long) running. after_bench.sh watcher will run `pytest -q -m gpu` once
  phase3/2026_01/bench_full.done appears, then FINAL ASSEMBLY (see 11:56Z entry).

## 14:0xZ FINAL
- [x] bench matrix DONE (26 ok / 1 timeout; RESULTS P-019); GPU test suite DONE (P-020)
- [x] COMPARISON_REPORT.md rebuilt with all artifacts (3,521 claims; sanity checks pass); adversarial review #1 + re-verification #2 addressed
- [x] print COMPARISON_REPORT.md to the transcript (in chunks) — done 2026-09-02, 33 chunks (runs/20260902T0000Z/phase4/print_chunks/part_000..part_032)

## Archival (2026-09-02, requested by Et)

- [x] status cron job feb62618 cancelled.
- [x] `runs/` removed from .gitignore; all campaign artifacts committed on
  branch `alphafold-experimental-results-reproduction` and pushed to origin.
  Only the raw inputs above GitHub's 100 MB limit stay out of git; they are
  listed with size and sha256 in `runs/20260902T0000Z/EXCLUDED_LARGE_FILES.md`
  and can be regenerated with `runs/20260902T0000Z/phase2/scripts/`.
- [x] `runs/20260902T0000Z/README.md` explains the directory layout; CLAUDE.md
  carries a campaign-status note for future sessions.
