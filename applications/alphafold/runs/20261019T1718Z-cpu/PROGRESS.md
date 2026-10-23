# PROGRESS — CPU reproduction (`CPU_REPRO_GOAL.md`)

Goal set: 2026-10-19T17:23:32Z (first set 17:18:56Z, stopped under Space, resumed by Et:
"work with the current memory", deviations.md D-001). Run directory: `runs/20261019T1718Z-cpu/`
(`<RUN>`). Engine: `../ET-Miner-engine` @ 37819e2 (D-002). THREADS = 6.

All harness Python runs through `scripts/run_py.sh` (engine venv, `CUDA_VISIBLE_DEVICES=`,
all pools pinned to THREADS). Mining runs go through `scripts/queue.sh <RUN>` (PLAN.md queue).

## Phase 0 — Setup and engine gate
- [x] 1. Box audit -> `env.json`. RAM 30.65 GiB (< 64 GiB, waived D-001); disk 511.9 GB free.
- [x] 2. Engine checkout @ 37819e2, `uv sync`, Rust extension, `pytest -m "not slow"` 897 passed /
      564 skipped (all CUDA/cupy), tier gate 5 passed / 86 skipped (GPU). RESULTS CPU-001..005.
- [x] 3. Harness in `scripts/`: `_harness.py`, `cpu_mine.py`, `cpu_son.py`, `compare.py`, `ea_oracle.py`,
      `rules.py`, `null_model.py`, `queue.sh`, `run_py.sh`, `after_pid.sh`; tests 9 passed (CPU-006).
      CLI smoke on a 20k-row fixture: in-core = SON (4 chunks) = efficient-apriori (1,758 itemsets),
      filtered compare, rules, null resume; queue skip/fail/gate checked.

## Phase 1 — Rebuild the data (started early: the inbound link is ~2.2 MB/s, D-004)
- [x] 1. pLDDT Storage Read: done 2026-10-19T18:02Z, 214,683,829 rows / 3,495,993,272 B in 1,800 s
      (`logs/export_plddt_20261019T1732Z.log`); M-001..M-009 reproduced (RESULTS CPU-007..009).
- [x] 2. TrEMBL 2026_01 stream + reduce with whole-archive md5/sha256 (D-007): 2026-10-19T18:13:23Z ->
      2026-10-20T05:37:17Z (`logs/stream_trembl_2026_01_20261019T1813Z.log`, progress log); full=True, md5 =
      RELEASE.metalink (CPU-010).
- [x] 3. `scripts/phase1_chain.sh <RUN>` (PID 402565, log `logs/phase1_chain_20261019T1753Z.log`) finished
      2026-10-20T06:38:47Z: count_dat, filter_dat_pfamgo, csv_join, filter_dat_by_accessions, shard sample test
      (0 differing rows), 4-shard extraction + merge, extract_stats, plddt_bin_edges. `verify_phase1.py`:
      51 ok, no data drift (CPU-011..CPU-021; `logs/verify_phase1_20261020T0639Z.log`). Item ids: one tie pair
      swapped -> GPU tables translated by feature_name (D-008). Row order differs from the GPU run (CPU-020).
- [x] 4. Original copy: `ORIGINAL_DATA_DIR` not set — skipped.
- [x] 5. `data/MANIFEST.md` (sources, times, sizes, checksums, terms; AlphaFold licence not verifiable from an
      allowed source). `EXCLUDED_LARGE_FILES.md`: 44 files, 63.4 GB (`scripts/large_files.py`).

## Phase 2 — G0, sizing, frozen PLAN.md
- [x] 1. G0 passed 2026-10-20T06:50Z: 167,042 / K14, in-core = SON (4 chunks) = efficient-apriori (CPU-023).
- [x] 2. Sizing: C1 / C2 / C3 / P1T done 2026-10-20T15:51Z (queue PID 490545, `logs/queue_20261020T0646Z.log`);
      C1-C3 identical to the GPU Opus tables filtered (CPU-025..027). Model `data/sizing/sizing.json` (CPU-030).
- [x] 3. PLAN.md frozen 2026-10-20T16:00Z: core + N5 + C3G queued; C4..X1 decided after C3G.

## Phase 3 — Mining per PLAN.md
- [ ] Queue started 20261020T1557Z (UTC), PID 540636, log `logs/queue_20261020T1557Z.log`; per-run logs
      `logs/<ID>_<UTC>.log`, markers `state/<ID>.done|failed`. Resume: `setsid scripts/queue.sh <RUN> >
      logs/queue_<UTC>.log 2>&1 &` (done runs skipped; a killed run restarts from K=1).
- [x] Minimum core done 2026-10-21T02:31Z: G0 (+reps), C1 (+reps), C2, C3, R1 (+reps, CI-001), S1 (+reps), S2, S3;
      all itemset tables identical to the GPU references (RESULTS CPU-025..039).
- [x] N5 consistent (CPU-040); C3G identical to C3, 78.6 % faster -> C4-C6/M4 use count inference (CPU-042).
- [x] C4 identical to the GPU Blitz tables (CPU-044); A1 Blitz decode and P-026 identical (CPU-046, CPU-047).
- [x] C6 identical (CPU-048), A1 K=22 identical (CPU-050), P-026 1,234 (CPU-051); C5 identical (CPU-052).
- [x] N100 consistent (CPU-057), 2026-10-22T16:55Z.
- [x] M4 identical to the GPU minc4 tables (CPU-059; first compare OOM, level-wise re-run, D-015); X1 identical to C3
      (CPU-061). Phase 3 done 2026-10-23T03:18Z; mining used 68.32 h of 96 h; no run dropped.

## Phase 4 — Compare, review, report
- [x] Interim independent review (adversarial-verifier, 2026-10-21T18:35Z): 2 must-fix (MANIFEST.md ignored by the
      repo-wide rule; M4/X1 not re-queued), 7 should, 3 nice — all addressed (see CPU_REPORT.md "Review findings").
- [x] CPU_COMPARISON.md (+ rows JSON) and CPU_REPORT.md generated (`scripts/build_comparison.py`, `build_report.py`).
- [x] Final independent review (2026-10-23): no must-fix, 4 should, 6 nice — all fixed (`REVIEW_final.md`).
- [x] Final commit, bundle, GOAL-EVIDENCE (2026-10-23).

## Paper revision (2026-10-23, requested by Et)
- [x] `paper/et_miner_proteome.tex`: Section "Reproduction on a CPU" with Table `tab:cpu`, sentences in the abstract,
      introduction, Methods 2.2, Sections 3.1/3.3/3.5, Discussion 4.1, Conclusion, Funding, Data availability; reference
      `bastide2000`; date October 2026 (version 2 kept). PDF rebuilt (14 pages).
- [x] Values checked against the artifacts (RESULTS CPU-065, `PAPER_VALUES.md`); fresh review `REVIEW_paper.md`: 1
      must-fix, 7 should, 5 nice — fixed except the go.obo note (D-009) and the underfull boxes.
- [x] Council review (4 reviewers, 2 rounds): unanimous block on scope, hardware and null-timing wording; all three
      fixed plus most flags (RESULTS CPU-066, `REVIEW_paper.md` second part).
- [ ] Commit (awaiting Et's permission); `paper/blog_post.md` not updated.
