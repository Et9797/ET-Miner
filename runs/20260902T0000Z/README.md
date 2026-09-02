# runs/20260902T0000Z — base214m reproduction campaign (2026-09-02)

Fresh re-execution of the ET-Miner AlphaFold co-occurrence preprint on a vast.ai
box with 2 × RTX 3090 (24 GB), 69.6 GiB cgroup RAM, 200 GB disk. The durable
narrative lives at the repo root: `PROGRESS.md` (phases, commands, PIDs, how
each step was resumed), `RESULTS.md` (every fresh value with artifact path and
UTC time), `INCONSISTENCIES.md` (I-001..I-012), `CLAIMS.md` (3,521 extracted
claims) and the deliverable `COMPARISON_REPORT.md` (+ `_rows.json`).

## Layout

- `logs/` — stdout/stderr of every detached job (extraction, mining, null model,
  bench), timestamped.
- `phase1/` — codebase-exploration and claim-extraction reports produced by
  subagents (source of `CLAIMS.md`).
- `phase2/` — data acquisition and feature extraction.
  `scripts/` holds every script written for the campaign (tar-member range
  streamer, TrEMBL release streamer, Pfam/GO and accession filters, BigQuery
  Storage-API pLDDT export, lean `af-extract` driver, stats, pLDDT bin edges,
  smoke extraction). `extract_2026_01/` holds `stats.json`, the item mapping,
  item-support tables, `af_extract.log` and `plddt_bin_edges.json`.
  `data/` keeps the small derived files (record counts, stream logs); the
  multi-GB raw inputs are excluded from git, see `EXCLUDED_LARGE_FILES.md`.
  `smoke_2026_01/` is the end-to-end smoke subset; `bq/` the BigQuery count
  checks; `bwtest/`, `streamtest/`, `metadata_sample/` are download probes.
- `phase3/2026_01/` — mining. `opus/`, `blitz/`, `minc4/` hold
  `frequent_k*.parquet` per K plus `mining_meta_*.json`; `exp/` the experiment
  JSONs (full campaign, direct-vs-SON, null model n=5, SON base/super, K=22
  analysis, `go.obo`); `exp_null100/` the 100-permutation null model;
  `bench/` the GPU bench (`raw.jsonl`, `report.md`, `env.txt`);
  `*.log`, `validate_row_split.log`, `gpu_test_suite.log`, `.done` markers,
  and the sequencer scripts `run_phase3.sh`, `chain3.sh`, `chain_tail.sh`,
  `after_bench.sh`.
- `phase4/` — verdict machinery: `quantities.csv` (canonical quantity keys),
  `claims_*_keyed.csv`, `build_fresh_values.py` → `fresh_values.json`,
  `build_comparison.py` → `COMPARISON_REPORT.md`, `env.json`, `overrides.json`,
  `rekey.json`, `header.md`, `adversarial_review_prompt.md`, and
  `print_chunks/` (the report split into ≤ 28 KB pieces for the transcript).

## Regenerating the excluded raw inputs

`phase2/scripts/stream_trembl_release.sh` streams `uniprot_trembl.dat.gz` for
release 2026_01 from the UniProt FTP archive; `filter_dat_pfamgo.sh` and
`filter_dat_by_accessions.sh` produce the reduced `.dat.gz` files;
`export_plddt_storage_api.py` exports `uniprotAccession, globalMetricValue`
from `bigquery-public-data.deepmind_alphafold.metadata`;
`run_af_extract_lean.sh` rebuilds the transactions parquet. Expected sizes and
sha256 of every excluded file are in `EXCLUDED_LARGE_FILES.md`.

## Directory map (depth 3, generated)

```
.
./logs
./phase1
./phase2
./phase2/bq
./phase2/bwtest
./phase2/data
./phase2/data/stream_work_2026_01
./phase2/extract_2026_01
./phase2/metadata_sample
./phase2/scripts
./phase2/scripts/__pycache__
./phase2/smoke_2026_01
./phase2/smoke_2026_01/exp
./phase2/smoke_2026_01/mine_test
./phase2/smoke_2026_01/p3test
./phase2/streamtest
./phase2/streamtest/docs
./phase2/streamtest/work
./phase3
./phase3/2026_01
./phase3/2026_01/bench
./phase3/2026_01/bench_failed_attempt1
./phase3/2026_01/blitz
./phase3/2026_01/exp
./phase3/2026_01/exp_null100
./phase3/2026_01/minc4
./phase3/2026_01/opus
./phase4
./phase4/__pycache__
./phase4/print_chunks
```
