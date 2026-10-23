# Deviations from `CPU_REPRO_GOAL.md`

- **D-001 — RAM below the Phase 0 need.** The box has 30.65 GiB RAM (brief: >= 64 GiB).
  Et's decision 2026-10-19: work with the current memory. A run that cannot fit falls under
  the per-run Space rule. Evidence: `env.json`, `BLOCKED_space_resolved.md`.
- **D-002 — engine checkout is a local clone, not `git worktree add`.** Et's global rules
  forbid git worktrees. `../ET-Miner-engine` is `git clone --no-hardlinks` of this repo,
  detached at main `37819e2`, with its own `.venv`. Nothing is committed there.
- **D-003 — Online Retail II test dataset.** `tests/test_smoke_correctness.py` (CPU) skips
  without `datasets/online_retail_ii/transactions.parquet`. It was rebuilt in the engine
  checkout with its own `datasets/prepare_online_retail.py` from the zip already on this box
  (`../ET-Miner/datasets/online_retail_ii/online_retail_ii.zip`, sha256 572e3627…, copied;
  log `logs/engine_prepare_online_retail.log` says "Using cached", no download).
- **D-004 — slow inbound link (~2.2 MB/s total).** The first pLDDT read and TrEMBL stream
  ran concurrently and shared the link: the read reached ~7k rows/s, too slow for one
  access token (1 h) and the read session (6 h). Both were stopped after ~6 min (TrEMBL
  ~0.7 GB of range files, removed; partial CSV removed; their logs removed). The read was
  restarted alone with a refreshing token handler; the TrEMBL stream is chained to start
  when it exits (`scripts/after_pid.sh`). Stream settings: 8 parallel 128 MiB ranges,
  window 8 (the GPU run: 16 x 256 MiB, window 8); output bytes do not depend on them.
- **D-005 — extraction by shards (af-extract unmodified).** af-extract `build-from-metadata` on the
  full csvacc DAT + joined CSV holds a DashMap of all 90.5M records and the whole CSV at once
  (~36-40 GiB by the code; the GPU run's 158.7M-record attempt reached ~63 GB RSS), which does not fit
  in 30.65 GiB. The joined CSV is split into 4 contiguous row ranges; each range's DAT is cut with the
  unchanged `filter_dat_by_accessions.sh`; the unmodified binary runs per shard with
  `--top-pfam 100000 --top-go 100000` (every feature kept); `scripts/merge_shards.py` sums the
  per-feature protein counts, ranks the top 500 Pfam / 500 GO exactly as af-extract does (count
  descending) and re-encodes the rows in CSV order. af-extract orders equal-count features at random
  (std HashMap); the merge breaks such ties by feature_name. Proven on synthetic data
  (`scripts/tests/shard_route_synthetic.sh`) and on a 2M-row real sample in the chain (direct vs shards).
  The parquet bytes differ from af-extract's writer; rows, items and order do not.
- **D-006 — earlier untracked attempt excluded locally.** `runs/20261012T1559Z-cpu/` (an earlier,
  uncommitted attempt from 2026-10-12, with its own reduced TrEMBL file) is left untouched as the brief
  requires and not used. It is listed in `.git/info/exclude` (local, untracked) so `git status` can be clean;
  the directory itself is unchanged.
- **D-007 — whole-archive checksum while streaming.** UniProt publishes the tarball's md5 in
  `release-2026_01/knowledgebase/RELEASE.metalink` (saved as `data/uniprot_meta/knowledgebase_RELEASE.metalink`:
  size 161,890,570,993, md5 e41d313b61d595a05d4f80032df44c33). The copy `scripts/stream_tar_member.py` adds
  md5/sha256 over every consumed archive byte and reads the archive to its end after the member is copied
  (diff: `scripts/stream_tar_member.py.diff`; tested on a local tar.gz: member bytes identical, md5 equal to
  md5sum). The member bytes written are unchanged. The first stream (started 18:02:58Z, without hashing, ~1.5 GB
  in) was stopped and restarted at 18:13Z; its logs are kept as `logs/*_aborted_nohash.log`.
- **D-008 — GPU tables translated by feature_name.** The rebuilt item mapping has the same 1,006 features as
  the GPU file, but one equal-count Pfam pair has swapped ids (PF08544: GPU 458, CPU 459; PF01841: GPU 459,
  CPU 458; `data/phase1_verification.json`). af-extract orders equal-count features at random and
  `merge_shards.py` breaks such ties by name. Every GPU table is mapped GPU id -> feature_name -> CPU id and
  re-sorted before the join (`compare.py --translate`); the other three tied pairs kept their ids.
- **D-009 — A1's K=22 script runs on the engine venv with pronto layered on.** `analyze_k22_proteins.py` imports
  pronto, which the engine venv lacks. It runs unchanged on the engine venv's interpreter with `uv run --no-project
  --python <engine venv> --with 'pronto>=2.5'` (pronto 2.7.4 from PyPI; polars 2.0.0 from the engine venv), with the
  GPU run's committed `go.obo` copied into the output dir so it downloads nothing (`scripts/a1_k22.sh`, log
  `logs/A1_k22_20261021T1313Z.log`: "Using cached …/go.obo").
- **D-010 — `decode_intermediate_k.py` copied with paths changed.** Its paths are hard-coded to the GPU run's Blitz
  tables and mapping; the copy reads C4's tables and this run's mapping, and the file names `k{K}.parquet`
  (`scripts/decode_intermediate_k.py.diff`: three lines). P-026 (no script in the GPU run: "polars filter") is read
  with `scripts/triple_support.py`.
- **D-011 — reading of "add runs in this order while the summed estimates fit".** Each run in the brief's order is
  added when its estimate fits the remaining budget; one that does not fit is dropped and the next considered; the
  order is never changed (PLAN.md Selection). Estimates are revised after each measured run (brief Phase 2 step 4),
  which re-admitted C6, M4 and X1 (PLAN.md Re-estimates 2026-10-21T08:20Z and 13:20Z).
- **D-012 — compares run in the queue.** Each run's full-table compares run right after it, in the same queue, so no
  two processes share the CPU during a timed run. Exceptions: the C1–C3 compares were first run by hand while the
  queue was idle (and again in the queue); `adjudicate_rules.py` ran by hand for about a minute at the start of S2
  (S2's wall, 7,104.9 s, includes that overlap); the P-026 check on C3 ran by hand at 05:18Z, seconds before C4
  started; the interim independent review (2026-10-21, light single-threaded probes, nice 19) and the re-runs of the
  adjudication, harness tests and provenance scan overlapped N100 permutations 7–9, whose timings may be slightly
  inflated (null timings are hardware-comparison only).
- **D-013 — `load_tables` pre-filters support-only references.** Before normalizing, a GPU table is cut at
  support · N ≥ min_count − 0.75 (a superset of the exact count filter, which still follows); harness tests pass
  (11 passed). This keeps the Opus references (26.8M rows) small for the higher thresholds.
- **D-014 — Et's decisions of 2026-10-21T18:50Z.** Asked in the session (AskUserQuestion), Et approved: (1) restarting the
  Claude Code wait command that the host had stopped under memory pressure; (2) the D-011 reading of the selection rule
  and the re-admission of C6, M4 and X1 after re-estimation; (3) the safety factor 2.74 for M4 (20.4 h; with 6.00 M4
  would have been dropped).
  2026-10-22T22:50Z: Et gave standing approval to restart the wait command whenever the host stops it (reported each time).
- **D-015 — `compare.py` compares one level at a time.** The first M4 compare (48,007,493 rows per side, every level
  loaded at once) was killed by the kernel for memory (rc 137, `logs/M4_cmp_20261022T2240Z.log`, marker kept as
  `state/M4_cmp.failed_oom_20261022T2243Z`). When both sides are directories of per-level files, `compare.py` now loads
  one level at a time and builds the signatures incrementally (`_harness.Signature`; `signature()` uses it); two files
  for one level are rejected. Checks, first by hand during X1 (nice 19, 2 threads; D-012) and again, logged, after the
  final review: `scripts/check_compare_levels.sh` → `logs/check_compare_levels_20261023T0906Z.log` (18 tests passed,
  7 of them on the level mode in `scripts/tests/test_compare_levels.py`; C1 vs Opus, S2 vs the GPU SON table and G0
  SON vs efficient-apriori re-run into `data/checks/compare_levels/` with per-level counts, signatures and verdicts
  equal to the earlier reports; a missing row and a +1 count planted in G0's K=3 table both caught and adjudicated
  against the CPU side; RESULTS CPU-064).
