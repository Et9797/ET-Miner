# Reproduce the AlphaFold base214m experiment on ET-Miner's CPU and CPU-SON routes

Goal: re-run the AlphaFold base214m experiment of branch `alphafold-experimental-results-reproduction` (campaign `runs/20260902T0000Z/`, 2× RTX 3090) on ET-Miner's in-core CPU route and its CPU SON streaming route, and compare every result with the GPU run.
- The data is rebuilt the way the GPU run built it. Reading the AlphaFold metadata from BigQuery is allowed; exporting anything is not. Every raw, intermediate and result file stays on local disk for later analysis.
- Itemsets and counts must match the GPU run exactly. Timings are reported side by side, without a verdict.

Read first. On this branch: `CLAUDE.md` (its operating rules apply), `runs/20260902T0000Z/README.md`, `RESULTS.md`, `PROGRESS.md`, `INCONSISTENCIES.md` (at least I-001, I-006, I-009, I-011, I-012), `runs/20260902T0000Z/EXCLUDED_LARGE_FILES.md`, `runs/20260902T0000Z/phase2/scripts/`, `applications/alphafold/experiments/` and `applications/alphafold/af-extract/`. On `main`: `CLAUDE.md`, the module docstrings of `src/et_miner/core/cpu_miner.py` and `src/et_miner/streaming/son.py`, and `bench/cpu/PROTOCOL.md` (thread settings, Amendments 3, 5 and 8).

Everything below is my reading of `alphafold-experimental-results-reproduction` at `8c1594c` and `main` at `37819e2`. Verify every claim against the code and the artifacts before relying on it.

## PARAMETERS

- `BUDGET_HOURS`: wall-clock budget for the mining runs of Phases 2–3 [96]
- `THREADS`: threads for every pool [the box's physical core count]
- `GCP_BILLING_PROJECT`: billing project for the BigQuery Storage Read session [et-research-491307]
- `ORIGINAL_DATA_DIR`: a local copy of `runs/20260902T0000Z/` that includes the excluded large files, if Et puts one on the box [none]

## What the GPU run did (verify)

Paths are relative to `runs/20260902T0000Z/phase3/2026_01/` unless they start with `phase4/`.

| id | run | GPU setting | GPU result (RESULTS.md) | reference artifact |
|---|---|---|---|---|
| G0 | first 1,000,000 rows of the mined set, min_count 50 | `validate_row_split.py` | 168,674 itemsets, K ≤ 14 (P-001) | `validate_row_split.log` (per-level counts) |
| C1–C6 | campaign Base / Super / Power / Blitz / Ultra / Opus: min_support 1e-3 / 1e-4 / 1e-5 / 1e-6 / 2e-7 / 1e-7, so min_count 76,891 / 7,690 / 769 / 77 / 16 / 8 | `experiment_full_campaign.py --runs 1`, one pinned RTX 3090 | 5,305 / K9; 113,405 / K14; 475,865 / K14; 2,841,280 / K19; 14,558,875 / K20; 26,849,505 / K22. Walls 31.1 / 52.6 / 74.7 / 292.9 / 1,996.4 / 2,631.2 s (P-028) | `single_gpu/exp/experiment_full_campaign_20260902_200731.json`; per-K tables in `opus/parquet/` and `blitz/parquet/` |
| M4 | min_count 4, support (4 − 0.5)/N | `run_mining.py`, 2 GPUs | 48,007,493 / K22, 4,536.9 s (P-017) | `minc4/parquet/` |
| S1–S3 | SON at Base / Super / Power, chunk 40,000,000, local factor 0.9 | `son_run.py`, `experiment_direct_vs_son.py` | 5,305 / 113,405 / 475,865, set-identical to exhaustive (P-015, P-016, P-027). Power on one 3090: SON 6,151.06 s vs direct 62.25 s (P-029) | `exp/son_base_itemsets.parquet`, `exp/son_super_itemsets.parquet`, `single_gpu/exp/experiment_direct_vs_son_20260902_215108.json` |
| R1 | rules from Base, min_confidence 0.5 | `son_run.py --rules-min-confidence 0.5` | 29,540 rules; 17,499 with lift ≥ 5; 9,283 with lift ≥ 100 (P-015) | `exp/son_base.json` |
| N5, N100 | permutation null at min_count 769; per-permutation seeds from `np.random.default_rng(42)` | `experiment_null_model.py` | real 475,865; null mean 171,249.6 (5) and 171,306.3 (100, σ 65.7, range 171,148–171,463); K_max 6 in every permutation (P-012, P-018, P-022, P-023) | `exp/experiment_null_model_20260902_070216.json`, `exp_null100/experiment_null_model_20260902_115339.json` |
| A1 | deepest itemset and decodes | `analyze_k22_proteins.py`, `phase4/decode_intermediate_k.py` | the K=22 itemset (support 8, members of P-007/P-008), P-026, P-030, P-031 | `exp/analysis_deepest_itemset_*.json`, `phase4/citation_audit/intermediate_k_decode.json` |

Facts the plan leans on (verify each):

- **GPU tables.** Per-K parquets hold `itemset` (List(Int32) of `item_id`s from `phase2/extract_2026_01/item_mapping_214m_base.parquet`) and `support` (Float64 = count / 76,890,945).
  - Recover counts with `round(support × N)` and assert a residual below 1e-6.
  - The Opus tables hold every itemset with count ≥ 8. Filtering them at a higher min_count gives the exact reference for C1–C5 and S1–S3.
  - `son_*_itemsets.parquet` don't keep items ascending. Sort before comparing.
- **The mined set** is `transactions_214m_base_multi.parquet`: 76,890,945 rows with ≥ 2 items, 1,002 items present, 316,421,093 list entries, at most 46 items per row (X-011..X-015). It comes from TrEMBL 2026_01 (not 2025_01, I-006) and the AlphaFold v4 metadata, with pLDDT ≥ 50 (X-001..X-010).
- **pLDDT from BigQuery.** The GPU run read `uniprotAccession, globalMetricValue` from `bigquery-public-data.deepmind_alphafold.metadata` with `phase2/scripts/export_plddt_storage_api.py`: the Storage Read API, 16 parallel streams, about 60 s, a gcloud access token and a billing project (M-006).
  - The `bq query --format=csv` route failed (I-009). The workaround named in `phase1/pipeline_reconstruction.md` (`--destination_table`, then `bq extract` to GCS) writes a table and exports it, so it is not allowed here.
  - The script takes the token as a command-line argument. Its adapted copy must get the token without putting it on a command line or in a log.
- **The CPU route on `main`** (`core/cpu_miner.py`) builds one CSR of the frequent items and mines every level from it on `n_jobs` threads. SON's CPU passes run on the same array miner.
  - On the CPU route `output_dir` and `resume_from_k` raise (they belong to the row-split GPU miner). The harness writes per-K tables itself, and a killed run restarts from K=1.
  - `level_callback` reports per level on the CPU route; SON reports through `progress_callback`.
  - `apriori_streaming` is public and takes `local_support_factor` (default 0.95 on `main`; the GPU run used 0.9). Every factor in (0, 1] gives the same result.
  - `use_generator_pruning=True` (count inference) is exact and runs on the CPU route; the GPU run didn't use it.
- **The branch scripts are GPU-only.** The experiment scripts hard-code `use_gpu=True`, `experiment_null_model.py` imports GPU internals, and the phase2 scripts hard-code `/root/projects/ET-Miner/...` paths. `applications/alphafold/deploy/run_mining.sh` loads results into BigQuery when `bq` is authenticated. None of them run unmodified here.
- **The null can't be replayed bit for bit.** The GPU run shuffled with `cupy.random.RandomState(seed)`, which NumPy can't reproduce. The CPU null uses the same seed sequence with NumPy, and Phase 4 compares it as a distribution.

## Ground rules

1. **Exactness.**
   - efficient-apriori is the oracle (`CLAUDE.md`, including its call convention). Itemsets and absolute counts must match exactly. Never compare counts only, and never relax a tolerance.
   - Before every mine, assert that `ceil(Fraction(str(s)) * N)` equals the intended min_count. Where no round support hits the target, use `(min_count − 0.5) / N`.
   - Every CPU table is compared row by row with its GPU reference: a full outer join on the ascending itemset tuple, comparing counts.
   - A differing row is settled by a direct count over the full transaction set (rows that contain every item of the itemset). The route that disagrees with the direct count is wrong. Record it in `INCONSISTENCIES.md`.
   - If the CPU engine is wrong: write a minimal repro to `<RUN>/UPSTREAM.md`, mark that run failed and continue with the others. Never patch the engine.
2. **Pinned engine.**
   - Use ET-Miner `main` at the newest commit on which `uv run pytest -q -m "not slow"` passes on this box. Check it out as a detached worktree `../ET-Miner-engine` with its own venv, and build the Rust extension there (`CLAUDE.md` on `main`) so the Rust tier leg runs. Never commit in that worktree.
   - Every CPU run uses that venv. Record the SHA, `et_miner.__file__` and `et_miner.has_rust_extension()` in `<RUN>/env.json`. The branch's own `src/et_miner` mines nothing here.
   - Public API only: `apriori`, `apriori_streaming`, `generate_rules`. No `ET_MINER_*` knobs, no internals.
   - CPU only: every process runs with `CUDA_VISIBLE_DEVICES=` (empty) and `use_gpu=False`. Record that no device was visible.
3. **BigQuery reads only; all data stays local.**
   - Allowed: reading `bigquery-public-data.deepmind_alphafold.metadata` through the Storage Read API into a local file, as the GPU run did.
   - Not allowed: any BigQuery export or write. No `bq extract`, no `bq load`, no query with a destination table, no dataset or table creation, and no loading of results (`run_mining.sh` does that).
   - Nothing leaves the box: no GCS or other uploads (`ET_UPLOAD_GCS` unset), no paste services, no `git push`.
   - Inbound data comes only from GitHub (this repo), package indexes and installers (PyPI, crates.io, uv, rustup), that BigQuery read, and the UniProt FTP archive of release 2026_01.
   - Every downloaded and derived file stays under `<RUN>/data/`: the pLDDT CSV, the reduced and filtered TrEMBL files, the accession list, the joined CSV and both transaction parquets. So does every result. Don't delete pipeline outputs; temporary stream chunks may go.
   - The access token never lands in a log, a file under the repo, a commit or the transcript.
   - The 161 GB TrEMBL member is streamed and reduced on the fly, as in the GPU run. It is not stored.
4. **Run directory.** `runs/<UTC-timestamp>-cpu/`, called `<RUN>` here. `runs/20260902T0000Z/` and everything else already under `runs/` stays untouched; copy a script before adapting it.
   - Durable state in `<RUN>/`, per the operating rules: `PROGRESS.md`, `RESULTS.md` (row ids `CPU-…`, each with artifact path or command and UTC time), `INCONSISTENCIES.md`, `PLAN.md`, `deviations.md`.
5. **Long jobs.**
   - Run downloads, extraction and mining detached (`setsid … > <RUN>/logs/<step>_<UTC>.log 2>&1 &`), with PID and log path in `PROGRESS.md`.
   - Each finished run writes a `.done` marker. A queue script runs `PLAN.md` in order and skips done runs.
   - To wait, arm a Claude Code background command that exits when the PID does (e.g. `while kill -0 <PID>; do sleep 60; done`) with the longest timeout the tool allows, and re-arm it when it returns. Goal evaluation waits while it runs, and its exit starts your next turn. Don't poll in a tight loop of turns.
   - Never paste large output into the transcript.
6. **Threads.** Pin every pool in every process, before polars or numpy is imported:
   - explicit `n_jobs=THREADS`, never -1 (it ignores affinity and cgroups);
   - `POLARS_MAX_THREADS`, `RAYON_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` = `THREADS`.

   Record `nproc`, `os.cpu_count()`, the CPU model and the cgroup CPU and memory limits. Record peak RSS and CPU seconds per run.
7. **Verdicts** (the operating rules, adapted):
   - deterministic values (itemsets, counts, K, rule counts, set identity): `identical`, or `mismatch` with the adjudication;
   - timings: `hardware-comparison`, the CPU value next to the GPU value and both boxes, no verdict;
   - null statistics: `consistent` or `inconsistent` per the Phase 4 criterion;
   - a run the plan dropped: `not-run`, with the reason.

   Every value cites a fresh artifact or the command that produced it. A GPU value is quoted from its RESULTS.md row and artifact, never retyped from the paper.
8. **Git.**
   - Commit per step on `alphafold-cpu-reproduction`.
   - Never push and never open a PR: everything stays local, and Et rsyncs the repo and its bundle.
   - Files over 100 MB are gitignored and listed with size and sha256 in `<RUN>/EXCLUDED_LARGE_FILES.md`.
   - At the end of every phase, write `git bundle create ~/et-miner-cpu-repro.bundle --all`.
9. **Documentation style.** Module docstrings are short and self-contained (what it does, Usage, Options). Implementation details go in function docstrings. No conversation or origin history in code.
10. **Language.** Chat in Dutch, concisely. Files, code, comments and commits in English. The `GOAL-EVIDENCE` and `GOAL-STOP` blocks are in English.
11. **Report format.** Reports and review findings are numbered lists. Prefix each item with `(must-fix)`, `(should)` or `(nice)`, sorted in that order.
12. **Out of scope:** engine changes, GPU runs, experiments not in the table, the V2 paper text, and new analyses of the results (Et analyses them later).
13. **Tooling.** Use explorer-opus for cross-file tracing and adversarial-verifier for the review. Run `/handoff` early if context gets tight.

## Stop conditions

When one of these holds:
1. write `<RUN>/BLOCKED.md`: the cause, the evidence, what Et must do and how to resume;
2. commit it and write the bundle;
3. print a block that starts with `GOAL-STOP:` and names the condition, then stop working.

The conditions:
- **Engine.** None of the last 20 `main` commits passes `pytest -m "not slow"` on this box, or gate G0 fails.
- **Access.** The BigQuery read fails for want of a gcloud login, a billing project or permission to create read sessions, or the UniProt 2026_01 archive can't be reached.
- **Space.** `df` or the cgroup memory limit shows the data or a run can't fit.
- **Data drift.** The rebuilt data misses any count of M-001..M-009, X-001, X-002 or X-005..X-016 exactly (timings excluded). Nothing downstream would be comparable.
- **Budget.** Phase 2's estimate for the minimum core exceeds `BUDGET_HOURS`, or the elapsed-time guard of the `/goal` line trips.

Otherwise run straight through.

## Phase 0 — Setup and engine gate

1. Box audit into `<RUN>/env.json` and `PROGRESS.md`: CPU model and cores, RAM and cgroup limit, free disk, OS, Python, uv, rustc, and the active gcloud account and `GCP_BILLING_PROJECT` (never the token). Compare with the needs (≥ 200 GB disk, ≥ 64 GiB RAM) and stop under **Space** if they can't be met.
2. Engine: the worktree at the pinned SHA, `uv sync`, the Rust extension, then `uv run pytest -q -m "not slow"` and `uv run pytest tests/test_tier_equivalence.py -q`. GPU-marked tests skip on a CPU box, which is expected here; every CPU leg must pass. Keep both logs.
3. Harness in `<RUN>/scripts/`, never in `applications/`:
   - `cpu_mine.py`: in-core run. Loads the parquet, asserts the min_count, calls `apriori(df, min_support=s, max_length=None, use_gpu=False, n_jobs=THREADS, level_callback=…)`. Writes per-K parquet with `itemset` (List(Int64), ascending), `count` (Int64) and `support`, and a meta JSON: per-level ms, wall, CPU seconds, peak RSS, K distribution and a signature (itemsets, summed counts, sha256 over the sorted rows).
   - `cpu_son.py`: SON run. `apriori_streaming(df, min_support=s, chunk_size=…, local_support_factor=…, use_gpu=False, n_jobs=THREADS, show_progress=False, progress_callback=…)`. Asserts the chunk count, records per-pass times, writes the same tables and meta.
   - `compare.py`: CPU against a GPU reference, including references filtered from the Opus tables, with adjudication by direct count.
   - `null_model.py`: seeds as in `experiment_null_model.py`; a NumPy shuffle of the flat item vector within the original row offsets; duplicates removed within each row; then an in-core mine at min_count 769. A fixture test shows that this construction equals explode → shuffle → `group_by(...).agg(unique)` for the same shuffled vector.
   - `queue.sh`: runs `PLAN.md` in order, writes `.done` markers, skips done runs.
   - Tests on small fixtures: a planted difference must be caught, counts recovered from supports, item order normalized.

Commit: `cpu-repro: setup`.

## Phase 1 — Rebuild the data locally (BigQuery reads only)

Adapt copies of the phase2 scripts in `<RUN>/scripts/` with parametrized paths.

1. **pLDDT.** Run the adapted copy of `export_plddt_storage_api.py` with `--project GCP_BILLING_PROJECT` and 16 streams, writing `plddt_metadata_storageapi.csv` (header `uniprotAccession,globalMetricValue`, the af-extract layout).
   - Compute M-001..M-009 from that CSV (row count, distinct accessions, pLDDT ≥ 50, nulls, min and max, > 90, 50–90). They must match exactly.
   - M-006's sha256 is not expected to match: parallel streams return rows in a varying order.
2. **TrEMBL 2026_01.** Stream the member of `knowledgebase2026_01.tar.gz` and reduce it (`stream_trembl_release.sh`, `stream_tar_member.py`), then run `filter_dat_pfamgo.sh` and `filter_dat_by_accessions.sh`. They must reproduce U-007, U-009 and the record counts of X-005, X-007 and X-016 (X-016 carries the X-001 and X-002 counts). Compressed sizes and sha256 of `.gz` files may differ with the compressor version.
3. **Extraction.** Build af-extract (`cargo build --release`) and run the steps of `run_af_extract_lean.sh`, `extract_stats.py` and `plddt_bin_edges.py`. Outputs: both transaction parquets, the item mapping, `stats.json`, `plddt_bin_edges.json`.
   - They must reproduce X-006 and X-008..X-015.
   - The bin edges must equal the committed file, and `stats.json` the committed one apart from timings.
   - The item mapping must equal the committed one by `item_id` and `feature_name`. If only the ids are permuted, translate GPU tables by `feature_name` before every comparison and record it in `deviations.md`. A different feature set is data drift.
   - Record whether the rebuilt row order equals the GPU run's (it does if the parquet's sha256 matches `EXCLUDED_LARGE_FILES.md`, or by the `ORIGINAL_DATA_DIR` comparison). G0's expected total depends on it.
4. **Original copy.** If `ORIGINAL_DATA_DIR` is set, check its files against the sha256 values in `EXCLUDED_LARGE_FILES.md`. Then compare the rebuilt transaction parquets with the originals by `protein_id` (identical item lists).
5. **Manifest.** `<RUN>/data/MANIFEST.md`: every URL, download time, size, sha256, and each source's terms of use (verify them).

Commit: `cpu-repro: data`.

## Phase 2 — Real-data gate, sizing, frozen plan

1. **Gate G0.** First 1,000,000 rows of the mined set (`head`, as in `validate_row_split.py`), min_count 50. CPU in-core == CPU SON with `chunk_size=250_000` (assert 4 chunks) == efficient-apriori (explicit `max_length` = the longest row), exact on itemsets and counts. If the row order equals the GPU run's, the total must also be 168,674 with the per-level counts of the GPU log. Otherwise record that cross-check as not comparable, with the reason.
2. **Sizing.** Run C1, C2 and C3 in-core (every plan has them), with per-level times.
   - Estimate every other run from the measured per-level cost and the GPU per-level counts (candidates and frequent itemsets per K in the GPU logs). Estimate S1–S3 from a pass-1 timing on one 40M-row chunk at Base.
   - Write the model, its inputs and every estimate in `PLAN.md`.
3. **Freeze `PLAN.md`.**
   - The minimum core is G0, C1–C3, S1–S3 and R1. If it doesn't fit `BUDGET_HOURS`, stop under **Budget**.
   - Then add runs in this order while the summed estimates fit: N5, C4 (Blitz), C6 (Opus) with A1, C5 (Ultra), N100, M4, and X1 (S3 again with `chunk_size=19_222_737`, asserting 4 chunks).
   - Count-inference fallback, decided before C4 starts: if C6 doesn't fit with defaults, run C3 again with `use_generator_pruning=True`. If its output is identical and its wall is ≥ 10% lower, C4–C6 and M4 use it. Record that in `PLAN.md` and re-estimate.
   - Repetitions: 3 (median, [min, max]) for runs under 15 minutes, otherwise 1, flagged.
4. **Re-estimate after every run.** A run past 2× its estimate continues if the new estimate still fits the remaining budget. Otherwise kill it, keep its per-level log, mark it `not-run (budget)` and move on.

Commit: `cpu-repro: plan`.

## Phase 3 — Mine, per `PLAN.md`

- **In-core C1–C6, M4:** `cpu_mine.py` on `transactions_214m_base_multi.parquet`.
- **SON S1–S3:** `cpu_son.py` with `chunk_size=40_000_000` and `local_support_factor=0.9`, asserting 2 chunks. Direct vs SON at Power is C3 against S3.
- **R1:** `generate_rules` on C1's itemsets at min_confidence 0.5. Count all rules, lift ≥ 5 and lift ≥ 100. A rule difference on identical itemsets is a rule-generator issue (see I-012), not a miner failure.
- **N5, N100:** the real run is C3. Each permutation stores its seed, full itemset table, K distribution and timings, with a checkpoint per permutation. N100 reuses N5's five permutations, since the first five seeds are the same (P-022).
- **A1:** run `analyze_k22_proteins.py` and `phase4/decode_intermediate_k.py` unchanged on the CPU tables: Opus for K=22 and P-026, Blitz for P-030 and P-031. If a script can't run in this venv, copy it to `<RUN>/scripts/` and record the minimal diff.

Commit per finished run: `cpu-repro: <id>`.

## Phase 4 — Compare, review, report

1. **Compare.** Run `compare.py` for every completed run. Also check set identity of S against C at each threshold (as P-027), and K distributions against P-014, P-028 and P-017.
2. **Null criterion,** fixed now, before N5 runs. The null is `consistent` when all of these hold:
   - every CPU permutation has K_max 6;
   - N5: the CPU mean total lies within μ ± 3σ/√5 of the GPU 100-permutation totals (μ and σ computed from the `exp_null100` JSON);
   - N100: a two-sided Mann–Whitney U test on per-permutation totals gives p ≥ 0.01;
   - the real-vs-null conclusion matches: no permutation reaches the real count at K = 4, 5 or 6, and none has an itemset with K ≥ 7.

   Otherwise it is `inconsistent`, reported as (should) with the numbers.
3. **Timings.** CPU walls and per-level times next to P-028 and P-029 (one RTX 3090), and the CPU SON/in-core ratio next to the GPU's 98.8×. Label them `hardware-comparison`.
4. **`<RUN>/CPU_COMPARISON.md`** (+ `CPU_COMPARISON_rows.json`): one row per value with the GPU id, GPU value, CPU value, artifact and verdict.
5. **Independent review** by the adversarial-verifier skill, or a fresh subagent that built none of this. Brief: "try to break it". Targets:
   - comparisons that pass trivially: empty or wrongly filtered tables, Int32/Int64 or item-order bugs, count recovery;
   - a GPU number retyped instead of read from its artifact, or a value without an artifact;
   - a BigQuery export, load, destination table or created dataset, an upload or a push, anywhere in scripts, logs or shell history; an access token in a log, file or commit; data missing from `<RUN>/data/`;
   - a GPU that was visible, unpinned threads, min_count conversion errors;
   - a null verdict that doesn't follow the pre-registered criterion;
   - budget cuts not recorded in `PLAN.md`.

   Fix the must-fix findings, commit, write the bundle.
6. **`<RUN>/CPU_REPORT.md`**, then the evidence block in chat.

## Final report (`<RUN>/CPU_REPORT.md`)

- Box, engine SHA and thread settings.
- Data: sources, sizes, sha256, and the match with the M- and X- rows.
- Gates: engine tests and G0.
- Per run: CPU result, GPU result, verdict, CPU wall against GPU wall, peak RSS.
- SON against in-core on the CPU, next to the GPU ratio.
- The null verdict with its numbers.
- Not-run runs with their reasons, and open items.
- Review findings: numbered, severity-tagged, sorted.

## `GOAL-EVIDENCE` block (print in your last turn)

```
GOAL-EVIDENCE
engine: <sha> | not-slow: <passed>/<skipped> | tier gate: <result> | <log paths> | GPU visible: no
G0: in-core = SON (4 chunks) = efficient-apriori = <n> itemsets (168,674 expected if row order matches; else reason) | <artifact>
data: M-001..M-009, X-… reproduced exactly | BigQuery: Storage Read only, exports/loads/tables: none | du -sh <RUN>/data: <size> | <MANIFEST path>
run <id>: <itemsets> / K<max> | vs GPU: identical or mismatch (<adjudication>) | CPU <s> s vs GPU <s> s | <artifact>
   (one line per run in PLAN.md)
null: N5 <consistent|inconsistent> (<numbers>); N100 <…>
not-run: <id>: <reason>; …
review: <n> findings, must-fix fixed in <commit>
git: <branch> @ <sha> | status clean | pushed: no | uploads: none | bundle <path> (<size>)
```
