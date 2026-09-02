# base214m — pipeline reconstruction from code (read-only)

Repo: `/root/projects/ET-Miner`, branch `alphafold-experimental-results-reproduction`
(HEAD `65d9098` "add alphafold experiment scripts", 2026-08-31, on top of `main` `25d0957`).
`applications/`, `paper/` exist ONLY on this branch (`git ls-tree main applications` is empty).
Written 2026-09-02 for `runs/20260902T0000Z`. Nothing below is a measurement; every number
is either read from code (cited `file:line`) or quoted from a doc/log-derived statement and
labelled as such. No repo file was modified; no GPU job, download or long process was run.

Legend used below: **[code]** = what the code does; **[doc]** = statement in a README/runbook/
paper/review, not verified here; **[artifact]** = value read from a surviving JSON/TXT.

---

## 0. One-paragraph verdict

`base214m` is the "base vocabulary" AlphaFold run: UniProt TrEMBL annotations (Pfam + GO) joined to
per-protein **global mean pLDDT** from the BigQuery table `bigquery-public-data.deepmind_alphafold.metadata`,
encoded as 1,006 items (6 pLDDT bins + top-500 Pfam + top-500 GO), written as one Parquet of
~205.6 M transactions, filtered at load time to the ~76.9 M proteins with ≥2 items, and mined
exhaustively (no `max_length`) with the GPU Apriori at six support thresholds down to
min_count = 8 ("Opus", K_max = 22 [doc]). **The canonical extraction route reads NO structure
files at all** — no `.cif`, no `.pdb.gz`, no PAE, no proteome tars — only the TrEMBL `.dat.gz`
and the BigQuery CSV export (evidence in §2). The complete experiment suite for the run is
`applications/alphafold/deploy/run_all_experiments.sh` (campaign 6×3, direct-vs-SON, null@769,
null@8, deepest-itemset analysis) plus `experiments/compute_maximal.py`. Several of those
scripts import module paths that no longer exist after the package restructure (§9), so the
suite cannot run unmodified on this tree.

---

## 1. Pipeline stages in execution order

### Stage 0 — environment / builds
| What | Where |
|---|---|
| Python package + GPU extra (`cupy-cuda12x[ctk]>=13.0`, `nvidia-nccl-cu12`) | `pyproject.toml:41-63` |
| dev group (pytest, ruff, maturin, efficient-apriori, matplotlib, psutil, tqdm, sparse-dot-mkl, mkl, mlxtend) | `pyproject.toml:65-77` |
| Rust extension `et_miner_rust` (maturin, from inside `rust_ext/`) | `bench/setup_box.sh:20-27`, `README.md:42-61` |
| `af-extract` binary (cargo, arrow/parquet 53, rayon, dashmap, csv) | `applications/alphafold/af-extract/Cargo.toml:8-30`, `deploy/deploy_base214m.sh:198-200`, `pipeline/pipeline_214m.py:579-580` |
| application-only deps (biopython, pronto, loguru …) | `applications/alphafold/requirements.txt:3-13` |
| kernel compile self-check, NCCL/P2P/shm matrix | `bench/selfcheck.py:19-148` |
| CLAUDE.md correctness gate (tier chain vs efficient-apriori) | `tests/test_tier_equivalence.py:1-215`, `bench/run_smoke.sh:12`, `bench/run_full.sh:13` |

### Stage 1 — data acquisition (canonical "metadata" route)
Entry point: shell commands in `deploy/RUNBOOK_base214m.md:63-73` and echoed by
`deploy/deploy_base214m.sh:274-282` (there is no Python downloader for this route;
`pipeline_214m.py --step download-annotations` downloads DAT/TSV only, `pipeline_214m.py:90-179`).

```
aria2c -x16 -s16 -d /workspace/data \
  'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.dat.gz'
aria2c -x16 -s16 -d /workspace/data \
  'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz'
bq query --use_legacy_sql=false --format=csv --max_rows=300000000 \
  'SELECT uniprotAccession, globalMetricValue FROM `bigquery-public-data.deepmind_alphafold.metadata`' \
  > /workspace/data/plddt_metadata.csv
```
Inputs: none. Outputs: `uniprot_trembl.dat.gz` (TrEMBL flat file), `uniprot_sprot.dat.gz`
(downloaded by the runbook but **not passed to af-extract**, see §9-13), `plddt_metadata.csv`
(two columns `uniprotAccession,globalMetricValue`). `pipeline_214m.py:416-437` looks for the CSV
under the names `plddt_metadata.csv[.gz]` / `alphafold_metadata.csv[.gz]` and prints the same
`bq` hint (`pipeline_214m.py:429-436`).

### Stage 2 — extraction + vocabulary construction + transaction encoding (one binary, one pass)
Entry point: `af-extract build-from-metadata` (`af-extract/src/main.rs:163-174` CLI,
`main.rs:393-476` implementation). Exact base214m command (`RUNBOOK_base214m.md:78-84`,
`deploy_base214m.sh:285-290`):

```
AFX=applications/alphafold/af-extract/target/release/af-extract
$AFX build-from-metadata \
  --annotations /workspace/data/uniprot_trembl.dat.gz \
  --plddt-csv   /workspace/data/plddt_metadata.csv \
  --top-pfam 500 --top-go 500 \
  --output       /workspace/data/transactions_214m_base.parquet \
  --item-mapping /workspace/data/item_mapping_214m_base.parquet
```
Defaults not overridden: `--min-plddt 50.0` (`main.rs:167-169`), `--batch-size 100000`
row-group size (`main.rs:171-173`), `--top-interpro/--top-ec/--top-taxonomy 0` (`main.rs:155-165`).
Python wrapper equivalent: `pipeline_214m.py --data-dir D --step extract-from-metadata --top-pfam 500 --top-go 500`
(`pipeline_214m.py:381-472`; NB its own defaults are 200/200, `pipeline_214m.py:614-624`, and it
writes `D/processed/alphafold_transactions_214m.parquet` + `item_mapping_214m.parquet`,
`pipeline_214m.py:442-443`, i.e. different file names than the runbook).

Sub-steps [code]:
1. `load_annotations` picks the DAT parser when the path contains `.dat` or ends in `.gz`
   (`main.rs:177-188`); `AnnotationDb::from_dat_gz` streams the gzip once (`annotations.rs:250-393`).
2. `read_plddt_csv` builds the protein list from the CSV, keeping rows with parseable pLDDT
   `>= min_plddt` (`main.rs:288-391`, filter at `main.rs:360-368`).
3. Frequency counting over that protein list (`main.rs:437-446`, `transaction.rs:65-90`).
4. Top-N encoder (`main.rs:449-451`, `transaction.rs:146-201`) → item mapping parquet
   (`main.rs:454`, `transaction.rs:257-324`).
5. Transactions parquet (`main.rs:457-459`, `transaction.rs:332-393`).

Output schemas [code]:
* `transactions_*.parquet`: `protein_id: Utf8` (UniProt accession), `items: List<Int64>` (sorted,
  de-duplicated item ids) — `transaction.rs:339-346`; SNAPPY, row groups of `--batch-size`
  (`transaction.rs:351-354`). **Every** pLDDT-passing protein is written, including ones whose
  only item is the pLDDT bin (`transaction.rs:362-366`; no min-items filter).
* `item_mapping_*.parquet`: `item_id: Int64`, `feature_name: Utf8`, `feature_category: Utf8`
  (`transaction.rs:306-310`); categories `plddt_mean`, `plddt_fraction`, `pfam`, `go_term`,
  `interpro`, `ec_number`, `taxonomy` (`transaction.rs:264-286`).
* `frequencies.json` only exists on the tar route (`count-frequencies`, `main.rs:190-225`).

Checkpointing: none inside af-extract (single shot). The Python wrapper skips the step if the
output parquet already exists (`pipeline_214m.py:445-472`).

Optional verify: `pipeline_214m.py --step verify` asserts the schema and prints items/protein and
mapping category counts (`pipeline_214m.py:505-556`).

### Stage 2b — alternative routes present in the tree (NOT the base214m route)
* Tar route: `af-extract count-frequencies` + `build-transactions --tar-dir` over the EBI proteome
  tars (`main.rs:190-282`; `pipeline_214m.py --all` runs `download-annotations, download-tars,
  extract, verify`, `pipeline_214m.py:674-675`). It parses `AF-*-confidence_v4.json` or
  `AF-*-model_v{4,6}.cif.gz` inside the tars for per-residue pLDDT (`tar_stream.rs:3-17,47-67`) and
  explicitly skips `.pdb.gz` and PAE files (`tar_stream.rs:62-66`, tests `tar_stream.rs:235-243`).
  Proteome list comes from `accession_ids.csv` on the EBI FTP (`pipeline_214m.py:47-48,220-254`).
* PoC route: `pipeline/download_alphafold.py` (a few proteomes / Swiss-Prot CIF tar,
  `download_alphafold.py:29-45`) + `pipeline/extract_features.py` (CIF pLDDT + DSSP secondary
  structure + top-50 Pfam/top-30 GO, fixed 92-item vocabulary, `extract_features.py:35-52`) +
  `cluster_sequences.py` (MMseqs2 30 %). Different vocabulary; irrelevant to base214m.
* 35K-vocab route ("Project Milky Way"/"Alpha Centauri"): `rust/dat_to_tsv.rs`, `rust/build_tx.rs`,
  `rust/remap_tx.rs`, `pipeline/postprocess_tx.py`, `resolve_feature_names.py`,
  `filter_self_sufficient.py`, `deploy/deploy_project_milky_way.sh`, `run_mining.sh`,
  `run_null_model_35k.sh`. Not base214m (35,012 InterPro/GO/EC/KW/TAX/LEN/pLDDT items).

### Stage 3 — pre-flight: row-split equivalence gate
`experiments/validate_row_split.py --data T.parquet --subset-size 1000000 --min-count 50 --n-gpus 2 -v`
(`RUNBOOK_base214m.md:90-96`). Mines the first N rows single-GPU and row-split multi-GPU and
requires byte-identical `(itemset, support)` sets; also requires the Rust extension
(`validate_row_split.py:94-102`) and a reference reaching K≥3 (`:159-167`). Exit 0/1.

### Stage 4 — mining / experiment suite
Entry point: `bash applications/alphafold/deploy/run_all_experiments.sh <transactions.parquet>`
with `N_GPUS` env (default 4; asserts that many devices, `run_all_experiments.sh:28,44-55`).
It **must be run with cwd = `applications/alphafold`** — all script paths are relative
(`experiments/...`, `run_all_experiments.sh:61,72,84,100,114`) and output goes to `./results_214m`
(`:27`), log `results_214m/experiment_log_<ts>.txt` (`:30`). Steps, in order:

| # | Command (`run_all_experiments.sh` lines) | Library call | Output |
|---|---|---|---|
| 1 | `experiment_full_campaign.py --data D --runs 3 --output-dir results_214m -v` (`:61-65`) | `apriori(df, min_support=s, use_gpu=True, max_length=None, n_gpus=args.n_gpus)` per threshold × run (`experiment_full_campaign.py:58-61`) | `results_214m/experiment_full_campaign_<ts>.json` (`:186-205`) |
| 2 | `experiment_direct_vs_son.py --data D --runs 3 --output-dir results_214m -v` (`:72-76`) | direct: `apriori(... max_length=None, n_gpus)` (`:93-94`); SON: `apriori_streaming(df, min_support, chunk_size=40_000_000, local_support_factor=0.9, use_gpu=True)` (`:118-125`) | `experiment_direct_vs_son_<ts>.json` (`:183-224`) |
| 3 | `experiment_null_model.py --data D --min-count 769 --runs 100 --n-gpus $N_GPUS --perm-per-gpu --output-dir results_214m -v` (`:84-91`) | real: `apriori(df, min_support, use_gpu=True, max_length=None, n_gpus=1)` (`:371-372`); null: GPU shuffle + `build_bitvecs_from_gpu_arrays` + `apriori(bitvecs=...)` per worker (`:187-268`) | `experiment_null_model_<ts>.json` (`:695-740`), checkpoint `null_model_checkpoint.jsonl` only with `--checkpoint/--resume` (`:325-345,441-450`) |
| 4 | same with `--min-count 8` (`:100-107`) | idem | idem |
| 5 | `analyze_k22_proteins.py --data D --output-dir results_214m -v` (`:114-117`) — no `--itemset-source`, so it falls back to the hard-coded v1 K=22 feature list (`analyze_k22_proteins.py:510-513`) | Polars filters on `items` | `analysis_deepest_itemset_<ts>.json`, `accessions_deepest_itemset_<ts>.tsv` (`:527-558`) |

Note: steps 1 and 2 do **not** receive `N_GPUS` (`run_all_experiments.sh:61-76`) → they run with the
scripts' default `--n-gpus 1` (`experiment_full_campaign.py:92-95`, `experiment_direct_vs_son.py:60-63`).

Library routing [code] (`src/et_miner/core/apriori.py`):
* `use_gpu=True` → `_build_csr_from_transactions` (frequent-item filter + explode/join → scipy CSR,
  `core/matrix.py:470-562`) → if `n_gpus > 1` or `anchor_items`: `_apriori_row_split_multi_gpu`
  (`apriori.py:436-453`, `gpu/row_split.py:56-744`), else single-GPU `_build_gpu_bitvec_matrix`
  + `_apriori_from_bitvecs` (`apriori.py:455-485`, `gpu/mining.py:401-780`).
* `output_dir` / `resume_from_k` are honored **only** by the row-split path (`apriori.py:447-448`
  vs `:472-485`); the single-GPU path keeps every itemset in host RAM and returns one DataFrame.
* Per-K Parquet flush: `gpu/io/flush.py:63-265` → `frequent_k{K}.parquet` (single table when
  `n < ET_MINER_FLUSH_PARALLEL_THRESHOLD`=1e8 rows, `flush.py:94-96,103-137`) or partitioned
  `frequent_k{K}/part_*.parquet`; schema `itemset: large_list<int32>`, `support: float64`
  (`row_split.py:147-153`, `flush.py:146-151`); zstd level 2 (`flush.py:98-101`).
* The only application entry points that produce per-K parquets are
  `pipeline/run_mining.py --use-gpu --n-gpus N>1 --parquet-flush [--resume-from-k K]`
  (`run_mining.py:231-241`, writes `<output-dir>/parquet/frequent_k*.parquet` +
  `mining_meta_<support>_<ts>.json`, `:246-268`) and `deploy/run_mining.sh` (35K data path hard-coded,
  `run_mining.sh:27`). **None of the five suite steps writes per-K parquets** (see §9-2).

### Stage 5 — post-processing / statistics
* `experiments/compute_maximal.py --results-dir results_214m/parquet --output-json results_214m/closed_maximal.json -v`
  (`RUNBOOK_base214m.md:120-125`): per-K total / closed / maximal via a K↔K+1 drop-1 join
  (`compute_maximal.py:83-172`); `--capped-top-level` flag when the run was length-capped (`:248-250`).
* `experiments/analyze_k22_proteins.py --data D --item-mapping M --itemset-source results_214m/parquet/frequent_k<KMAX>.parquet [--annotations goa_uniprot.gaf] --output-dir results_214m -v`
  (`RUNBOOK_base214m.md:110-116`): picks the deepest itemset (`:77-127`), lists supporting
  accessions (`:130-274`), GO true-path pairs via `pronto` + `go.obo` (`:277-420`), optional GAF
  evidence codes (`:423-443`).
* (not base214m) `filter_self_sufficient.py`, `systematic_bio_analysis.py`, `validate_motifs.py`.

### Stage 6 — figures / tables
No figure script exists in the tree. The paper's figures are `\includegraphics{figures/*.pdf}`
(`paper/et_miner_proteome.tex:111,160,242,253`) and the `figures/` directory is absent.
`experiments/analysis_alpha_centauri.ipynb` (29 cells) produces Plotly figures (K-distribution,
Zipf fit, feature frequency, co-occurrence heatmap, Jaccard diversity, UMAP, SON comparison, null
overlay, rules) but loads `archived/alphafold/results_214m/itemsets_214m_godmode.parquet` and
`item_mapping_214m.parquet` (notebook cell 1) which are absent; the notebook holds no rendered
outputs except one 2-line stream (cell 1). The paper tables (`tab:features`, `tab:campaign`,
`tab:kdist`, `tab:k22`, `tab:null-model`) are hand-typed in `paper/et_miner_proteome.tex`.

---

## 2. RAW DATA REQUIREMENT (most important item)

### 2.1 Verdict
**The base214m extraction runs from metadata alone. No AlphaFold structure files are read.**

Evidence [code]:
* The subcommand doc: "Build transactions from annotation file + pLDDT CSV (no tar files needed).
  Use when pLDDT comes from BigQuery metadata instead of CIF/JSON files. This avoids downloading
  23TB of AlphaFold structures entirely." — `af-extract/src/main.rs:126-130`.
* `run_build_from_metadata` never touches `tar_stream` (`main.rs:393-476`); the protein list is
  built by `read_plddt_csv` (`main.rs:419-421`) or, without a CSV, from the annotation keys
  (`main.rs:422-434`, then no pLDDT items at all).
* pLDDT in this mode is the single global value: `confidence::from_mean_only` assigns only the
  mean bin (items 0–2) and never the fraction bins (3–5) — `confidence.rs:170-187`.
* The runbook's correction #1 states the same: "De pLDDT-via-BigQuery-route (23TB CIF vermijden)
  matcht `af-extract build-from-metadata --plddt-csv ...`" — `RUNBOOK_base214m.md:37-40` [doc].
* Reviews describing the original run corroborate: the lost `pipeline_214m.log` recorded
  "214,683,829 rows read … 205,620,298 passed pLDDT filter" and input `uniprot_trembl.dat.gz`
  (`paper/PAPER_V2_REVIEW.md:62-64,94` [doc]).
* PAE JSON, `.pdb.gz` and per-residue pLDDT are consumed **nowhere** on this route; even the tar
  route skips PAE/PDB (`tar_stream.rs:62-66`).

### 2.2 Inputs, readers and fields

| Input | Reader | Fields used | Full set needed? |
|---|---|---|---|
| **UniProt TrEMBL `uniprot_trembl.dat.gz`** (Swiss-Prot flat-file format, gzip; `MultiGzDecoder`, `annotations.rs:254-259`) | `AnnotationDb::from_dat_gz` `annotations.rs:250-393` | `AC` first accession (`:279-284`); `DR   Pfam;` id (`:285-291`); `DR   GO;` id (`:292-298`); `DR   InterPro;` (`:299-305`, unused for base vocab); `DE … EC=` (`:306-321`, unused); `OC` top-2 lineage (`:322-335`, unused); record end `//` with per-protein sort+dedup (`:336-372`) | **Yes, whole file**: (a) the top-500 ranking is computed over all pLDDT-passing accessions (`main.rs:437-446`), so a subset changes the vocabulary; (b) every accession in the CSV that lacks a DAT record silently gets no annotation items (`transaction.rs:216`). |
| **BigQuery export `plddt_metadata.csv`** from `bigquery-public-data.deepmind_alphafold.metadata` | `read_plddt_csv` `main.rs:288-391` (csv crate, `.gz` accepted `:292-296`) | header-detected accession column `uniprotAccession|accession|accessionId|Entry` (`:303-314`) and pLDDT column `globalMetricValue|mean_plddt|plddt|avg_plddt` (`:315-326`); rows with unparsable pLDDT or `< --min-plddt 50` are skipped (`:351-368`); progress every 10 M rows (`:373-379`) | **Yes** — it *defines* the transaction set (one transaction per CSV row that passes the filter, `main.rs:419-421`, `transaction.rs:362-366`). The 214 M → 205.6 M numbers come from this filter [doc, `PAPER_V2_REVIEW.md:62-63`]. |
| `uniprot_sprot.dat.gz` | (downloaded by runbook, `RUNBOOK_base214m.md:68-69`) | — | **Not consumed** by the runbook's af-extract command (`RUNBOOK_base214m.md:80` passes only trembl). |
| Proteome tars / `.cif.gz` / `confidence_v4.json` / `accession_ids.csv` | `tar_stream.rs`, `pipeline_214m.py:185-254` | per-residue pLDDT only | **Not needed** (tar route only). |
| PAE json, `.pdb.gz` | nobody | — | never read (`tar_stream.rs:62-66`). |
| `go.obo` (OBO Foundry), `goa_uniprot.gaf` | `analyze_k22_proteins.py:306-330,423-443`; `resolve_feature_names.py:39-63` | GO names/ancestors; evidence codes | post-processing only, optional |
| PROSITE `prosite.dat` | `validate_motifs.py:36-71` | — | PoC route only |

### 2.3 Size statements found (all [doc]/[code comment], none measured here)
* TrEMBL `.dat.gz`: "TrEMBL: 150G" (lost log quoted in `paper/PAPER_V2_REVIEW.md:64`); paper:
  "Feature extraction processed 150 GB of compressed annotation data in 63 minutes"
  (`paper/et_miner_proteome.tex:129`). `pipeline_214m.py:95` says "DAT files (~100GB uncompressed)"
  (inconsistent with 150 GB compressed, §9-16). Note the current release will be larger than the
  2025_01 release the paper cites.
* REST TSV alternative: "~20GB" (`pipeline_214m.py:94,123,568`); Swiss-Prot DAT "~660 MB"
  (`download_alphafold.py:224`); Swiss-Prot FASTA "~89 MB" (`:234`).
* Structures (not needed): "214M proteins, ~23 TiB uncompressed" (`pipeline_214m.py:6`,
  `main.rs:129`); human proteome tar 4,938 MB etc. (`download_alphafold.py:33-40`);
  Swiss-Prot CIF tar 38 GB (`:42`).
* BigQuery CSV: no size stated anywhere. Arithmetic (mine, not a measurement): ~214.7 M rows ×
  ~18 B ≈ 4 GB.
* Output transactions parquet for 214 M: no size stated; the 35K-vocab parquet is "~2.3 GB"
  (`deploy_project_milky_way.sh:128`) for 109 M rows [doc].
* Host RAM during extraction: "String interning reduces memory from ~17GB to ~8GB for 214M entries"
  (`annotations.rs:7`) — plus the `Vec<ExtractedProtein>` for every CSV row (`main.rs:337,362-365`),
  which the code keeps entirely in memory (unquantified in code).
* Mining-side sizes [doc]: CSR of the 76.9 M subset "316 million non-zero entries … ~5.1 GB"
  (`tex:167`); bitvec matrix "~26 GB" for 205.6 M × 1,002 (`tex:171`); dense 206 GB (`tex:105`).
  [code arithmetic] bitvec bytes = `n_cols × ceil(n_rows/64) × 8` (`csr_bitvec.py:245,276`):
  1,002 × 1,201,421 × 8 ≈ 9.6 GB for 76,890,945 rows; ≈ 4.8 GB per GPU when row-split over 2.

### 2.4 What "full 214M" means for a reproduction
The transaction *set* is every BigQuery metadata row with global pLDDT ≥ 50 (≈205.6 M [doc]);
the *mined* set is the ≥2-item subset (76,890,945 [artifact/doc]). Both the vocabulary ranking
and the multi-feature count depend on the full corpus, so a subset run reproduces neither the
1,006-item mapping nor the K-distribution. A subset is fine for smoke tests (the same binary
with `--top-pfam/--top-go` and a truncated CSV), not for base214m.

---

## 3. The "base vocab"

Definition [code], `af-extract/src/transaction.rs:146-201` + `confidence.rs`:

* Item id layout: `[pLDDT (0–5) | Pfam | GO | InterPro | EC | Taxonomy]`, contiguous, each block
  ordered by **descending frequency** with ids assigned `base + rank` (`transaction.rs:130-144,
  149-174`; pfam_base = 6 when pLDDT is included, `:159`).
* pLDDT bins (`confidence.rs:23-25, 33-35`):
  * item 0 `plddt_mean_low` mean < 50; item 1 `plddt_mean_med` 50 ≤ mean ≤ 90; item 2
    `plddt_mean_high` mean > 90 (`confidence.rs:71-78`, `from_mean_only` `:179-185`; labels
    `transaction.rs:264-271`).
  * items 3/4/5 `plddt_frac_high_{low,med,high}`: fraction of residues > 90 is < 0.25 / ≤ 0.75 /
    > 0.75 (`confidence.rs:81-87`) — **only computable from per-residue data (tar route)**.
* Pfam items: raw Pfam accessions from `DR   Pfam;` lines (e.g. `PF00069`), top `--top-pfam`
  by protein count; GO items: raw `GO:xxxxxxx` ids from `DR   GO;` lines, top `--top-go`
  (`annotations.rs:285-298`, `transaction.rs:161-165`). Counting is per protein after per-protein
  dedup (`annotations.rs:340-343`).
* base214m parameters: `--top-pfam 500 --top-go 500` → 6 + 500 + 500 = **1,006 defined items**
  (`RUNBOOK_base214m.md:76,84`; `deploy_base214m.sh:284,288`; `tex:129,134`). Code defaults are
  200/200 → 406 items (`main.rs:148-153`, `pipeline_214m.py:614-624`, `main.rs:19` example).
* Filters: `--min-plddt 50` on the CSV rows (`main.rs:360-368`); no min-items filter at
  extraction; the ≥2-item filter is applied at load time (`utils.py:23-49`, `min_items=2` in
  `experiment_full_campaign.py:101`, `experiment_direct_vs_son.py:70`, `experiment_null_model.py:202,348`).
* Expected frequent items at min_count 8: 1,002 = 1,000 Pfam/GO + 2 pLDDT bins (`tex:134`,
  `review_b2_results.md:274-277`, `senior_review_mar23.md:70` [doc]). [code] this is
  **structural**, not a support effect: in metadata mode items 3–5 are never emitted
  (`confidence.rs:170-187`) and item 0 is impossible because rows with mean < 50 are dropped
  before encoding (`main.rs:360-368`) — so exactly items 1 and 2 can be non-empty.
* Determinism caveat [code]: `build_top_n_map` sorts a `HashMap` iterator by count
  (`transaction.rs:136-137`); Rust `HashMap` iteration order is randomised per process, so ties
  at the 500th rank (and the id order among equal counts) are not reproducible across runs —
  item ids must always be interpreted through the mapping parquet written by the *same* run
  (`analyze_k22_proteins.py:111-117` warns about exactly this).

---

## 4. Exact mining parameters for base214m

All from code; n = number of ≥2-item transactions (76,890,945 in the original run [artifact
`experiment_direct_vs_son_20260219_050326.json` `parameters.n_transactions`]).

| Parameter | Value | Source |
|---|---|---|
| Support thresholds (campaign) | `("Base",1e-3) ("Super",1e-4) ("Power",1e-5) ("Blitz",1e-6) ("Ultra",2e-7) ("Opus",1e-7)` | `experiment_full_campaign.py:40-47` |
| min_count rule | `ceil(min_support × n)` | `core/result.py:15-17`; `experiment_full_campaign.py:115` |
| → min_counts for n=76,890,945 | 76,891 / 7,690 / 769 / 77 / 16 / 8 | arithmetic (paper prints 7,689 and 768 for Super/Power, §9-9) |
| Direct-vs-SON threshold | `--min-support 0.00001` (Power) | `experiment_direct_vs_son.py:44-47` |
| Null-model thresholds | `--min-count 769` and `--min-count 8`; `min_support = min_count / n` | `run_all_experiments.sh:86,102`; `experiment_null_model.py:351-352` |
| Null permutations / seed | `--runs 100` (suite) / 5 (artifact); `--seed 42`; per-permutation seeds drawn from `np.random.default_rng(seed)` (`integers(0, 2**63)`), GPU shuffle via `cp.random.RandomState(gpu_seed)` | `run_all_experiments.sh:87,103`; `experiment_null_model.py:292-294,319-323,223-228` |
| Campaign repeats | `--runs 3` per threshold; direct-vs-SON `--runs 3` | `run_all_experiments.sh:64,74` |
| max_length (K max) | `None` (unbounded; loop stops when a level is empty or `len(prev_frequent) < k`) | `experiment_full_campaign.py:59`, `experiment_direct_vs_son.py:93`, `experiment_null_model.py:243,371,579`; loop condition `gpu/mining.py:591`, `row_split.py:334` |
| sparse_from_k | `None` (never transitions) in every experiment; `"auto"` exists but is only used by `mine_two_phase` and bench | `apriori.py:239`; `row_split.py:754`; `bench/runner.py:94-95` |
| prune_equal_support / closed pruning | `False` (default) → full non-closed pattern space | `apriori.py:215`, `:449-450` |
| GPU mode | `use_gpu=True`; `n_gpus` from `--n-gpus` (default 1) → single-GPU fused kernels; `n_gpus>1` → row-split with NCCL reduce | `apriori.py:422-485`; `row_split.py:166-172` |
| Auto multi-GPU inside the "single-GPU" path | K=2 splits pairs across all visible GPUs when pairs ≥ 15 M; K≥3 when candidates ≥ 500 K — each level copies the whole bitvec matrix D2H+H2D to the other GPUs | `gpu/dispatch.py:31-36,69-73,128-129`; `kernels/k3plus.py:382,403`; `kernels/k2.py:130,150` |
| Fused-kernel output cap | 10,000,000 survivors per level (per GPU); >5 % overflow raises, ≤5 % silently truncates with a warning | `kernels/k3plus.py:222,269-272,305`; `kernels/k2.py:40,76`; `kernels/shared_tiled.py:199,215`; `kernels/loader.py:32-40` |
| Kernel variant / filter / balance | `ET_MINER_KERNEL_VARIANT=auto`→`shared`; `ET_MINER_FILTER_IMPL=compact`; `ET_MINER_ROW_BALANCE=rows` | `dispatch.py:14-23`; `_env.py:100-131` |
| Row-split chunking | candidates per dense chunk = `(avail − group_bytes − margin) / 6 B`, margin = max(1 GiB, 4 % VRAM) capped at avail/4; `ET_MINER_MAX_CHUNK_CANDS` caps it | `row_split_chunks.py:36-42,55-97,120-147` |
| batch_size | `apriori(batch_size=10_000)` default (CPU path only) | `apriori.py:211` |
| Memory guards (single-GPU path only) | `max_ram_gb=800`, `max_vram_gb=70` | `apriori.py:236-237`, `mining.py:499-516` |
| SON (direct-vs-SON only) | `chunk_size=40_000_000`, `local_support_factor=0.9`, `use_gpu=True`, `gpu_resident=False` | `experiment_direct_vs_son.py:48-55,118-125`; `streaming/son.py:84-86` |
| Per-K flush | `ET_FLUSH_COMPRESSION=zstd` (level 2), `ET_FLUSH_CHUNK_SIZE=5e7`, `ET_FLUSH_THREADS=4`, parallel above 1e8 rows | `_env.py:52-69`; `flush.py:94-101,140-144` |
| Upload | `ET_UPLOAD_GCS=1`, `ET_UPLOAD_TAG=base214m_<date>`, bucket `gs://et-miner-results` | `RUNBOOK_base214m.md:18,65`; `_env.py:76-89`; `row_split.py:186-195` |
| Config/preset names | "Base/Super/Power/Blitz/Ultra/Opus" (campaign), "God Mode"/"godmode" = Opus (notebook cell 0; reviews), run tag `base214m_YYYYMMDD`. No TOML/preset encodes them; `et_miner.config` defaults (`min_support=0.01`, `config.py:51`) are unrelated. | as cited |

Quoted threshold table (`experiment_full_campaign.py:40-47`):
```python
THRESHOLDS = [
    ("Base",  0.001),       # 0.1%
    ("Super", 0.0001),      # 0.01%
    ("Power", 0.00001),     # 0.001%
    ("Blitz", 0.000001),    # 0.0001%
    ("Ultra", 0.0000002),   # 0.00002%
    ("Opus",  0.0000001),   # 0.00001%
]
```
Quoted null-model call (`experiment_null_model.py:240-244`):
```python
null_result = apriori(bitvecs=(bitvecs_gpu, col_to_item, n_transactions),
                      min_support=min_support, max_length=None)
```

---

## 5. Statistics / post-processing computed by the scripts

| Script → function | Statistic | Output file |
|---|---|---|
| `experiment_full_campaign.py::run_single` (`:50-73`) | per run: `itemsets`, `time_seconds`, `max_k`, `k_distribution` | `experiment_full_campaign_<ts>.json` (`:186-205`) |
| same, `main` (`:130-161`) | per threshold: mean/std/CV % of itemset count, mean/std time, max/min K; summary table | idem, keys `thresholds.<name>.aggregate` |
| `experiment_direct_vs_son.py` (`:141-181`) | direct vs SON: mean±std itemsets & time, `speedup = son_mean_time/direct_mean_time`, `miss_rate = 1 − son/direct`, CV of itemset counts (determinism check `direct_cv < 0.01`) | `experiment_direct_vs_son_<ts>.json` (`:189-224`) |
| `experiment_null_model.py::compute_statistics` (`:146-184`) | per K: real count, null mean, null std (ddof=1), z = (real−μ)/σ, one-sided p = 1−Φ(z) (`scipy.stats.norm`), direction, `significant` (p<0.05); infinite z when σ=0 | `experiment_null_model_<ts>.json` `statistics` (`:701-735`) |
| same, `main` (`:647-693`) | ratio real/null totals, avg null mining s, total GPU-s, wall-clock, parallel speedup, peak-K analysis (Z>2 verdict), K=1 marginal-preservation sanity check (±5 % or ±50) | idem `summary`; log |
| `experiment_null_model.py::shuffle_transactions` (`:77-143`) | dedup-affected protein % after shuffle | log only |
| `analyze_k22_proteins.py` (`:77-127,130-274,277-420`) | deepest itemset (K, support, ids), supporting protein accessions (+n_features), GO parent–child pairs, redundant terms, "independent K" | `analysis_deepest_itemset_<ts>.json`, `accessions_deepest_itemset_<ts>.tsv` (`:527-558`) |
| `compute_maximal.py::compute_level` (`:118-172`) | per K: total, closed (no K+1 superset with equal support, rel tol 1e-9), maximal (no frequent K+1 superset); totals | `closed_maximal.json` (`:211-218,265-268`) |
| `validate_row_split.py` (`:141-183`) | K-distribution single vs row-split, first divergences | exit code + log |
| `run_mining.py::print_dataset_stats` (`:31-62`) | n_transactions, unique items, avg/min/max items per txn, top-10 items | log + `motifs_*.json` metadata |
| `run_mining.py::run_baseline` (`:82-121`) | random-shuffle baseline: mean/std itemsets, ratio, z-score | `motifs_<support>_<ts>.json` `baseline` |
| `run_mining.py::run_sweep` (`:124-151`) | itemsets per size per threshold, time | log |
| `run_mining.py::decode_itemsets/print_summary` (`:65-79,154-175`) | decoded feature names, counts by size, top-20 by support | `motifs_*.json` |
| `utils.k_distribution` (`:81-90`) | itemsets per K | used by all experiments |
| `core/rules.py::generate_rules` (`:46-108`) | rule support/confidence/lift | notebook Part 3 only |
| `validate_motifs.py` (`:156-244,413-431`) | KNOWN/PARTIALLY_KNOWN/NOVEL classification | `validation_report.json` (PoC route) |
| `filter_self_sufficient.py` (`:54-243`) | support ratio vs K−1 subsets | `.stats.json` (35K route) |
| `analysis_alpha_centauri.ipynb` | K-dist plots, Zipf fit (log-log slope, R²), feature frequency/depth, top-60 Jaccard heatmap + Ward clustering, per-K Jaccard diversity, UMAP/t-SNE of K≥8, SON waterfall, null overlay, rule network | notebook only (needs absent parquet) |

No enrichment/lift statistics are computed on the itemsets themselves outside `generate_rules`.

---

## 6. Performance instrumentation

| Where | What is logged |
|---|---|
| `af-extract` | `Time: {:.1}s ({:.0} proteins/sec)` at the end (`main.rs:279,469-473`); `Read {}M rows ({} passed filter)` every 10 M CSV rows (`main.rs:373-379`); `Parsed {}K DAT records` every 100 K (`annotations.rs:363-365`); `Loaded {}M annotations` every 5 M TSV lines (`annotations.rs:222-224`); `Written {}M transactions` every 1 M (`transaction.rs:373-375`); final `Transactions / Total items` (`main.rs:464-468`). Uses `env_logger` at info with ms timestamps (`main.rs:479-481`). |
| `pipeline_214m.py` | per-step elapsed (`:342,373,470`), pipeline total (`:715-717`) |
| `gpu/mining.py` (single GPU) | header `APRIORI BITVEC: n cols, n txns, min_support, min_count, max_length` (`:520-524`); per K `K={k}: candidates=… → frequent=… ({elapsed}s) | cumulative=… | RAM=…GB VRAM=…GB` (`:490-497`); exhaustion / max_length messages with total time (`:749-753,768-774`) |
| `gpu/row_split.py` (multi GPU) | `Bitvec build: Xs across N GPUs` (`:137`); NCCL status (`:170-172`); `K=1: n frequent items in Xs` (`:318`); `K={k}: {n_freq} frequent in {k_time}s` (`:599`); resume timing (`:274`); K=2 chunk plan (`:453-456`); debug K≥3 chunk plan/budget (`:535-539`) |
| `gpu/row_split_chunks.py` | per chunk `candidates → frequent (% pass rate, min_count)` (`:285-289`) |
| `io/flush.py` | flushed rows, MB, codec, seconds, threads (`:126,244-248`) |
| `utils.load_transactions` | load seconds, multi-feature filter counts (`:42,47-48`) |
| experiments | `time_seconds` per run; null: `shuffle+bitvec` s, `mine_seconds` vs `time_seconds` per permutation, `Wall-clock time`, `Total GPU-seconds`, `Parallel speedup` (`experiment_null_model.py:235-236,253-266,622-623,655-663`); direct-vs-SON mean±std s and speedup (`:147-152,173`); campaign mean±std s and total campaign time (`:150-151,183`) |
| `level_callback(k, n_candidates, n_frequent, duration_ms)` API | `apriori.py:225,276-277`; used by `bench/child_run.py:108-109` |
| bench harness | wall_s, per-level ms, peak VRAM per GPU via `nvidia-smi` polling every 0.5 s, throttle reasons, result signatures (`bench/child_run.py:21-68,70-85,173-185`); `bench/report.py` medians/A-B tables |
| Not instrumented | proteins/min for mining (only proteins/sec for extraction); kernel-level timings (no CUDA events anywhere); PCIe transfer volumes (only doc claims, `tex:808-813`). |

---

## 7. Hardware / environment assumptions

* Original run: "single NVIDIA H100 80 GB SXM5 with 128 GB host RAM, Python 3.10, CuPy 13.0,
  NumPy 1.26, CUDA 12.4, Ubuntu 22.04" (`paper/et_miner_proteome.tex:213` [doc]); notebook says
  "H100/H200" (cell 0). Runbook targets **4×H200** (or 8×) on vast.ai (`RUNBOOK_base214m.md:9`,
  `run_all_experiments.sh:2,16-17`), timing "~5-6 h wall-clock, ~$45-90" (`RUNBOOK:152-154` [doc]).
* Code-level constants tuned on RTX 3090: multi-GPU dispatch thresholds "Calibrated from RTX 3090 4x
  benchmarks" (`gpu/dispatch.py:3,25-31,66-69`); chunk margin comment "old hardcoded 6 GiB, which
  was 25% of an RTX 3090" (`row_split_chunks.py:39-42`); CUDA sources must build on sm_86 and
  sm_90 (`CLAUDE.md` CUDA section); bench campaign designed for 2×3090 24 GB (`bench/README.md:3`),
  measured on driver 580.159.03 / CuPy 14.1.1 / CUDA 13.0 (`bench/results/2026-08-31-3090x2/env.txt:8-9`,
  `FINDINGS.md:4-5`).
* `bench/setup_box.sh` installs: curl/build-essential if missing (`:8-9`), `uv` (`:12-15`),
  `uv sync --locked --extra gpu` (`:18`), rustup minimal + `maturin develop --release` inside
  `rust_ext` (`:23-27`), sets `CUDA_PATH=/usr/local/cuda` if present (`:34-37`), probes a RawKernel
  compile and self-heals with `cupy-cuda12x[ctk]` (`:39-50`), runs `bench/selfcheck.py` (`:56`) and
  pre-generates synthetic presets (`:59`). `deploy_base214m.sh` additionally installs aria2, gcloud
  SDK, pushes ADC credentials and builds `af-extract` (`:121-201`).
* CuPy/NVRTC needs CUDA headers: `CUDA_PATH` (`run_all_experiments.sh:21-24`, `bench/run_full.sh:10`,
  `bench/README.md:9-14`). NCCL over SHM needs `/dev/shm ≥ 2 GB` (`bench/README.md:15-19`,
  `selfcheck.py:48-56`); `NCCL_SHM_DISABLE=1` fallback; `ET_MINER_DISABLE_NCCL=1` forces the
  staged D2D reduce (`gpu/nccl.py:32-34`).
* `ET_*` knobs relevant to a large run (all in `src/et_miner/_env.py:8-43`): `ET_MINER_KERNEL_VARIANT`,
  `ET_MINER_FILTER_IMPL`, `ET_MINER_ROW_BALANCE`, `ET_MINER_DISABLE_NCCL`, `ET_MINER_MAX_CHUNK_CANDS`,
  `ET_MINER_TILED_MIN_GROUP_PAIRS`, `ET_MINER_FLUSH_PARALLEL_THRESHOLD`, `ET_MINER_LEGACY_WRITE`,
  `ET_FLUSH_COMPRESSION/CHUNK_SIZE/THREADS`, `ET_PARQUET_TMPDIR`, `ET_PARQUET_BACKUP_DIR`,
  `ET_UPLOAD_GCS`, `ET_UPLOAD_TAG`, `ET_MINER_GCS_BUCKET`, `ET_MINER_GCS_CREDENTIALS`, `GCS_TOKEN`,
  `ET_MINER_LOG_DIR`. Plus `CUDA_VISIBLE_DEVICES` (bench pins `=0` for 1-GPU configs,
  `bench/runner.py:47-51`) and `NCCL_DEBUG=INFO` (`bench/run_*.sh:8`).
* Library limits: `n_transactions < 2^31` (`mining.py:445-449`, `row_split.py:114-118`); item ids
  `< 2^31` (`row_split.py:148-150`).
* Dependencies: polars ≥1.39, pyarrow ≥25, numpy, scipy, scikit-learn, pandas, loguru, joblib
  (`pyproject.toml:41-52`); GCS extra `google-cloud-storage>=2.14` (`:61-63`); ADC credential
  lookup order (`io/gcs.py:16-19,115-127`).

### 7.1 Fit on THIS box (2×RTX 3090 24 GB, 24-core EPYC, ~69.6 GiB cgroup, 200 GB disk, nvcc 12.1, no uv)
* VRAM [code arithmetic]: 76.9 M-row bitvec ≈ 9.6 GB (1,002 cols) — fits one 3090 with room for the
  10 M-entry fused buffers (~0.2 GB) and the CSR H2D staging (~3.1 GB, freed after build); row-split
  ≈ 4.8 GB per GPU plus dense count chunks sized from measured headroom. If the un-filtered
  205.6 M-row parquet is mined by mistake, the bitvec is ≈ 25.8 GB → does **not** fit one 3090,
  only row-split over two (≈ 12.9 GB each).
* Host RAM: CSR build materialises ~316 M (row,col) int64 pairs ≈ 5 GB plus scipy conversion
  (`core/matrix.py:534-559`); the single-GPU path keeps 26.8 M itemsets as Python lists before
  `_build_result_df` (`mining.py:533,649,693,735,777`) — several GB, unquantified; the SON pass in
  direct-vs-SON builds a Polars boolean matrix per 40 M-row chunk (`son.py:250-256`) — with
  ~1,000 columns at 1 B/bool that is ~40 GB per chunk, which is marginal under a 69.6 GiB limit
  (paper's host had 128 GB). Extraction RAM: 8–17 GB [doc] for the annotation map plus the full
  in-memory protein vector — unquantified, plausibly 20–40 GB.
* Disk: ~150 GB `.dat.gz` [doc] + ~4 GB CSV + output parquet (a few GB) + per-K parquets (Opus:
  26.8 M rows × ~44 B ≈ 1.2 GB before compression) ≈ 160 GB of the 200 GB free — feasible only if
  nothing else is stored; delete the `.dat.gz` after extraction or stream it (§10).
* Speed: unknown for 3090; the fused kernels are bandwidth-bound (3090 ≈ 936 GB/s vs H100 ≈ 3.35 TB/s),
  so expect a multiple of the paper's 7.3 min for Opus; the null@8 100-permutation set is the
  long pole ("~4 min/perm on H200", `run_all_experiments.sh:98` [doc]).
* Toolchain: `uv` absent → `pip`/venv from `/opt/conda/bin/python3` works (`pyproject` is plain
  hatchling); `cupy-cuda12x` wheels run on the CUDA 13.2 driver [PROGRESS.md]; NVRTC headers come
  from `cupy-cuda12x[ctk]` or `CUDA_PATH=/usr/local/cuda` (nvcc 12.1 present).

---

## 8. Surviving artifacts (AlphaFold / base214m related)

| Path | Content (1 line) |
|---|---|
| `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | [artifact] old single-run format: Direct GPU 475,865 itemsets / 50.72 s / K=14 at min_count 768 on 76,890,945 txns; "son_reference" 22,846 / 1,085.6 s / K=13; "blitz_reference" 2,841,280 / 119.3 s / K=19; speedup 21.4×. Schema ≠ what the current script writes (§9-11). |
| `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | [artifact] 5 permutations, seed 42, min_count 769, `recomputed_real: false` (real dist copied), real K-dist to K=14 (475,865), null max K=6 (20–23 itemsets), per-K z/p, 662.17 s total. Old schema (`total_experiment_seconds`). |
| `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | [artifact] 41,665 lines: 5,351 decoded itemsets for K=15..19 ("Total proteins in dataset: 76,890,945"); this is the K=19 (Blitz, min_count 77) run, **not** the K=22 Opus run; item ids reference an absent mapping. |
| `applications/alphafold/results_214m/GLOSSARY.md` | hand-written Pfam/GO glossary "from the 2.84M direct GPU mining results (K=1-19)". |
| `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | code + markdown only; one 2-line stream output; loads absent `archived/alphafold/results_214m/itemsets_214m_godmode.parquet`. |
| `applications/alphafold/rust/build_tx`, `rust/remap_tx` | prebuilt Linux ELF binaries (4.0 MB each) for the 35K route. |
| `paper/et_miner_proteome.tex` | v1 manuscript; the claims source (tables of campaign, K-dist, K=22, null). |
| `paper/PAPER_V2_REVIEW.md`, `peer_review_jul12.md`, `review_b1_hostile.md`, `review_b2_results.md`, `revision_notes_b3.tex`, `senior_review_jun01.md`, `senior_review_mar23.md` | reviews; `PAPER_V2_REVIEW.md:62-80` and `review_b2_results.md:6-12` name the **lost** artifacts: `results_214m/logs/{pipeline_214m,godmode_mining,direct_mining,beyond_mining,ultra_mining,extreme_mining}.log`, `archived/alphafold/results_214m/itemsets_214m_godmode.parquet`, `item_mapping_214m.parquet`, `/mnt/d/alphafold-data/transactions_214m.parquet`. None exist in the tree (`find` for `*.log`, `*.parquet`, `logs/`, `archived/` returns nothing under `applications/`). |
| `bench/results/2026-08-31-3090x2/{FINDINGS.md,report.md,env.txt,raw.jsonl}` | synthetic-preset GPU campaign on a 2×3090 box (28 configs; stress_k2 peak VRAM 17.3 GB; tier chain 7/7). Not AlphaFold data, but hardware-relevant. |
| `bench/results/2026-09-01-3090x2-sparse/{...}` | sparse-CSR follow-up (13 configs, deep_k). |
| `PROGRESS.md`, `runs/20260902T0000Z/RUN_DIR.txt` | this campaign's state; the run dir also now holds `logs/bandwidth_test.log`, `phase2/bwtest/…tar`, `phase2/metadata_sample/gcd_metadata-00000-of-10000.json` created by the caller during this session. |

Git history (read-only): `git log --all --stat --diff-filter=D -- applications bench` shows only
bench deletions (`b1f147e`: `bench/results/campaign/{raw.jsonl,report.md}`, `bench/results/smoke_run.log`;
`6f789d8`: 12 per-config `.log/.result.json` under `bench/results/campaign/`). No AlphaFold file was
ever deleted from git — the whole `applications/` + `paper/` tree entered git in one commit
(`65d9098`, 50 files, 2026-08-31) already without logs/parquets, and `*.log`, `*.parquet`,
`results/`, `logs/` are gitignored (`.gitignore:2-3,27,30`). Other deletions in history are library
refactors (`k3plus_indirect.cu`, `k3plus_sampled.cu` in `e84c9d3`; `csr_intersect.cu` in `b8a5032`;
`gpu/auto_chunk.py`, `gpu/kernels.py`, `dispatch.py`, `stats.py`, `datasets/online_retail_ii/online_retail_ii.zip`).
Branches: `alphafold-experimental-results-reproduction` (this), `main`, and `claude/*` feature branches.

---

## 9. Internal inconsistencies

1. **Stale imports — the suite cannot run on this tree.** `experiment_null_model.py:200,460,542`
   import `et_miner.cuda_csr_bitvec` and `:461` `et_miner.apriori`; `compute_maximal.py:52` and
   `filter_self_sufficient.py:45` import `et_miner.rules`; `deploy_base214m.sh:228`,
   `deploy_project_milky_way.sh:308,311`, `run_mining.sh:100,138` import `et_miner.apriori` /
   `et_miner.cuda_csr_bitvec`; notebook cell 25 imports `et_miner.rules`. None of these modules exist
   (`src/et_miner/{rules,apriori,cuda_csr_bitvec}.py` missing); the functions live in
   `et_miner.gpu.csr_bitvec` (`build_bitvecs_from_gpu_arrays:314`, `build_bitvecs_row_split_from_arrays:441`),
   `et_miner.gpu.row_split` (`_apriori_row_split_multi_gpu:56`), `et_miner.core.rules`
   (`_iter_row_groups:110`, `_list_to_scalar_cols:157`, `_explode_drop1:162`, `_detect_k:220`). Effect:
   null-model `--perm-per-gpu` workers and the row-split branch crash; the sequential branch silently
   falls back to the CPU explode/shuffle path (`:540-545`); `compute_maximal.py` fails at import.
2. **No per-K parquets are produced by the suite** yet the runbook post-processes them:
   `experiment_full_campaign.py:58-61` calls `apriori` without `output_dir`, and `run_all_experiments.sh`
   never calls `run_mining.py`; but `RUNBOOK_base214m.md:113,120-122` feed `results_214m/parquet/frequent_k*.parquet`
   to `analyze_k22_proteins.py` and `compute_maximal.py`. Only `run_mining.py --parquet-flush` (requires
   `--n-gpus > 1`, `run_mining.py:233`) writes them.
3. `run_all_experiments.sh` uses relative paths (`experiments/…`, `results_214m`; `:27,61,72,84,100,114`)
   while `RUNBOOK_base214m.md:100-102` and `deploy_base214m.sh:299-302` invoke it from the repo root
   (`cd /workspace/ET-miner`) → "No such file". `N_GPUS` defaults to 4 with a hard assert (`:28,53`);
   steps 1–2 ignore `N_GPUS` (`:61-76`).
4. Vocabulary size: code/CLI defaults 200 Pfam + 200 GO (`main.rs:19-20,98-103,148-153`;
   `pipeline_214m.py:614-624`) vs canonical 500/500 (`RUNBOOK:82`, `tex:129`); `pipeline_214m.py --all`
   runs the tar route, not `extract-from-metadata` (`:674-675`).
5. `pipeline_214m.py`'s REST field list `accession,xref_interpro,go_id,ec,keyword,lineage,length`
   (`:52`) contains no Pfam → the TSV route cannot build the base vocab (`annotations.rs:125-127`
   finds no "Pfam" column). Acknowledged in `RUNBOOK:41-43`.
6. `run_mining.py` defaults: `--max-length 4` (`:195`), `--support 0.01` (`:193`), input
   `alphafold_transactions_nr30.parquet` (`:26`), and **no multi-feature filter**
   (`load_transactions(args.input)` `:218` → `min_items=0`) — mining the raw 205.6 M-row parquet with
   `--support 1e-7` gives min_count 21, not 8, and a 25.8 GB bitvec. The experiments filter with
   `min_items=2` (`experiment_full_campaign.py:101`).
7. Method drift for Base/Super/Power: paper Table 2 reports those three as "Streaming SON"
   (`tex:224-226`, 5,305 / 51,124 / 22,846 itemsets) while `experiment_full_campaign.py:7-9,39` runs all
   six thresholds as exact Direct GPU (SON "infeasible" comment refers to the 35K vocab). A re-run
   will produce exact (larger) counts; e.g. the artifact already shows Power = 475,865 exact vs 22,846 SON.
8. Paper Table 1 "pLDDT bins … Defined 6, Frequent 2" attributes the 4 missing bins to the ≥8
   support threshold (`tex:134,144`); in code they are structurally empty in metadata mode
   (`confidence.rs:170-187`, `main.rs:360-368`). `senior_review_mar23.md:70` and
   `review_b2_results.md:277` say "plddt_mean_low has < 8 proteins" — it has 0 by construction.
9. min_count rounding: code uses `ceil` (`core/result.py:16`) → Super 7,690, Power 769; paper Table 2
   prints 7,689 and 768 (`tex:225-226`) and the direct-vs-SON artifact records 768 while the null
   artifact records 769 for the same 1e-5. A re-run gives 769 for both.
10. K=22 pLDDT feature: the hard-coded fallback names it `plddt_mean_medium` "pLDDT 70-90"
    (`analyze_k22_proteins.py:62`) and the paper says "Medium confidence (70–90)" (`tex:327`), but the
    encoder's item is `plddt_mean_med` = 50 ≤ mean ≤ 90 (`transaction.rs:266`, `confidence.rs:74-75,181-183`).
    The name mismatch also defeats the fallback's substring match (`analyze_k22_proteins.py:219`),
    pushing the script into its "≥ n_feat features" heuristic (`:253-274`) — the reviews' "3 proteins
    identified (fallback heuristic)" (`review_b1_hostile.md:96`).
11. Artifact schemas vs current scripts: the surviving JSONs use `direct_gpu_result/son_reference/
    blitz_reference/comparison` and `total_experiment_seconds`, `recomputed_real: false`; the current
    scripts write `direct_gpu_runs/son_runs/aggregate` (`experiment_direct_vs_son.py:189-221`) and
    `permutation_seeds`, `wall_clock_seconds`, `recomputed_real: True`, `real_mining_seconds`
    (`experiment_null_model.py:701-737`). The notebook (cell 11) reads the old keys.
12. `experiment_null_model.py:364-368` comments that "the multi-GPU row-split diverges from
    single-GPU at K>=8 (the pre-flight caught over-counting)" and forces `n_gpus=1` for the real
    distribution; the bench campaigns on the current tree report bit-identical signatures for 1 vs 2 GPUs
    up to K=11 (`bench/results/2026-08-31-3090x2/FINDINGS.md:17-20`, `…-sparse/FINDINGS.md:13-16`).
    Which statement holds for this tree is exactly what `validate_row_split.py` must decide.
13. Runbook downloads `uniprot_sprot.dat.gz` (`RUNBOOK:68-69`) but only passes `uniprot_trembl.dat.gz`
    to af-extract (`:80`); Swiss-Prot accessions in the pLDDT CSV therefore get no annotations. Paper
    says "UniProt TrEMBL" (`tex:129`) but Limitation 3 once said "UniProt/Swiss-Prot"
    (`PAPER_V2_REVIEW.md:148-150`); notebook says "TrEMBL + SwissProt" (cell 0).
14. Referenced-but-missing files: `scripts/warmup_kernels.py` (`deploy_base214m.sh:256`, non-fatal;
    `deploy_project_milky_way.sh:351`, fatal), `tests/test_compute_maximal.py` (`RUNBOOK:28,33`),
    `applications/alphafold/.env` (`RUNBOOK:31`, gitignored), `.claude/rules/et-miner.md` (`RUNBOOK:74`),
    `papers/peer_review_jul12.md` (actual path `paper/…`, `RUNBOOK:4`), `~/.claude/plans/…` (`RUNBOOK:5`),
    `figures/*.pdf` (`tex:111,160,242,253`), `archived/alphafold/…` (notebook), `results_214m/logs/*.log` (reviews).
15. `README.md:209,219-224` says "26.8 million co-occurrence patterns across ~76M … 214M total, 76.9M
    with multiple annotations, vocabulary 1,002 items" and `paper:129` says 205,620,298 processed; the
    "214M" is the CSV row count before the pLDDT ≥ 50 filter (`PAPER_V2_REVIEW.md:62-63`).
16. Size statements disagree: "DAT files (~100GB uncompressed)" (`pipeline_214m.py:95`) vs "150 GB of
    compressed annotation data" (`tex:129`) / "TrEMBL: 150G" (`PAPER_V2_REVIEW.md:64`).
17. Output dtype: docs say `itemset: List[Int64]` (`README.md:286`, `core/result.py:22`) but the row-split
    flush writes `large_list<int32>` (`row_split.py:147-153`, `flush.py:148-151`); `pipeline_214m.py:533`
    asserts `List(Int64)` on the *transaction* parquet only.
18. Determinism: top-N tie-breaking via `HashMap` iteration (`transaction.rs:136-137`) → item ids and the
    500th-rank membership may differ between extraction runs (see §3).
19. The `--perm-per-gpu` docstring assumes an 80 GB H100 per permutation (`experiment_null_model.py:402-405`);
    `apriori`'s `max_vram_gb=70` guard (`apriori.py:237`) is meaningless on 24 GB cards.
20. `pipeline_214m.py` output names (`processed/alphafold_transactions_214m.parquet`, `:442-443`) differ
    from the runbook's (`transactions_214m_base.parquet`), and `--step verify` only knows the former.

---

## 10. Runbook for THIS box (2×RTX 3090 24 GB, 24 cores, ~69.6 GiB RAM, 200 GB disk, nvcc 12.1, conda python, no uv)

Everything below is derived from the code paths above; nothing has been executed. Items marked
⚠ would not run unmodified or do not fit; items marked ✎ require a one-line local change that the
caller must decide on (this reconstruction did not edit the repo).

### 10.0 Environment
```bash
cd /root/projects/ET-Miner
/opt/conda/bin/python3 -m venv /root/venvs/etminer && source /root/venvs/etminer/bin/activate
pip install -U pip maturin
pip install -e ".[gpu]"                      # cupy-cuda12x[ctk], nvidia-nccl-cu12, polars, pyarrow …  (pyproject.toml:41-60)
pip install pytest efficient-apriori mlxtend psutil tqdm matplotlib   # dev group (pyproject.toml:65-77) — or `pip install --group dev .` with pip ≥ 25.1
pip install -r applications/alphafold/requirements.txt              # pronto, biopython (only pronto is needed)
export CUDA_PATH=/usr/local/cuda                                   # NVRTC headers (bench/run_full.sh:10)
(cd rust_ext && maturin develop --release)                          # Tier 2 + prune_closed_flat_compact (deploy_base214m.sh:194-195)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal; source ~/.cargo/env
(cd applications/alphafold/af-extract && cargo build --release)     # af-extract binary
python bench/selfcheck.py                                           # every kernel compiles on sm_86; NCCL/P2P/shm matrix
pytest tests/test_tier_equivalence.py -q                            # CLAUDE.md hard gate (7 CPU + GPU legs)
```
(Alternative: install `uv` per `bench/setup_box.sh:12-18` and use `uv sync --locked --extra gpu`.)

### 10.1 Data acquisition (≈ 155 GB; the disk is the constraint)
```bash
mkdir -p /root/data/af && cd /root/data/af
# pLDDT: BigQuery → CSV with header uniprotAccession,globalMetricValue (main.rs:303-326 accepts exactly these)
bq query --use_legacy_sql=false --format=csv --max_rows=300000000 \
  'SELECT uniprotAccession, globalMetricValue FROM `bigquery-public-data.deepmind_alphafold.metadata`' \
  > plddt_metadata.csv
```
⚠ A 214 M-row result through `bq query --format=csv` paging is untested here; the robust variant is
`bq query --destination_table <proj>:<ds>.plddt --replace …`, `bq extract --destination_format CSV … gs://<bucket>/plddt_*.csv`,
`gcloud storage cp`, then concatenate with one header (af-extract reads one file; `.gz` accepted,
`main.rs:292-296`). Needs a billable GCP project (gcloud is authenticated per PROGRESS.md).
The caller's `runs/…/phase2/metadata_sample/gcd_metadata-00000-of-10000.json` suggests the GCS
`metadata/` JSON shards as an alternative source — **no code reads those**; they would need a
JSON→CSV conversion producing the two columns above.
```bash
# TrEMBL flat file (~150 GB compressed [doc]); resumable:
curl -L -C - -o uniprot_trembl.dat.gz \
  https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.dat.gz
```
Disk-saving alternative (single pass, no resume): af-extract reads the DAT sequentially once
(`annotations.rs:250-393`) and selects the gzip reader by file name (`main.rs:183`, `annotations.rs:254`),
so a FIFO named `uniprot_trembl.dat.gz` fed by `curl … | cat > fifo` avoids storing 150 GB; a network
error restarts the whole pass. Swiss-Prot is not needed for the runbook command (§9-13).

### 10.2 Extraction (checkpoint: none; re-run from scratch on failure)
```bash
cd /root/projects/ET-Miner
AFX=applications/alphafold/af-extract/target/release/af-extract
RUST_LOG=info $AFX build-from-metadata \
  --annotations /root/data/af/uniprot_trembl.dat.gz \
  --plddt-csv   /root/data/af/plddt_metadata.csv \
  --top-pfam 500 --top-go 500 \
  --output       /root/data/af/transactions_214m_base.parquet \
  --item-mapping /root/data/af/item_mapping_214m_base.parquet \
  2>&1 | tee /root/data/af/af_extract.log
```
GATE (`RUNBOOK:86-87`): record from the log `Read N rows: M passed pLDDT filter`, `Frequencies: … Pfam,
… GO`, `Item encoding: 6 pLDDT + 500 Pfam + 500 GO = 1006 total items`, `Transactions: …`, `Time: … proteins/sec`.
Then compute the multi-feature count and pre-filter once (the miners need the ≥2-item subset and
`run_mining.py` has no filter, §9-6):
```bash
python - <<'PY'
import polars as pl
lf = pl.scan_parquet("/root/data/af/transactions_214m_base.parquet")
print("total:", lf.select(pl.len()).collect(engine="streaming").item())
multi = lf.filter(pl.col("items").list.len() >= 2)
multi.sink_parquet("/root/data/af/transactions_214m_base_multi.parquet")
print("multi-feature:", pl.scan_parquet("/root/data/af/transactions_214m_base_multi.parquet").select(pl.len()).collect().item())
PY
rm /root/data/af/uniprot_trembl.dat.gz    # reclaim ~150 GB once the parquet is verified
```
⚠ Host RAM for af-extract is unquantified in code (annotation map 8–17 GB [doc] + every CSV row in
memory); watch RSS — a 69.6 GiB cgroup OOM-kills silently. Extraction is single-threaded per file.

### 10.3 Pre-flight gate (RUNBOOK Fase 0.3b)
```bash
CUDA_PATH=/usr/local/cuda python applications/alphafold/experiments/validate_row_split.py \
  --data /root/data/af/transactions_214m_base_multi.parquet --subset-size 1000000 --min-count 50 --n-gpus 2 -v
```
Exit 0 → row-split (n_gpus=2) is exact and may be used for everything; exit 1 → single-GPU only.
Note the "single-GPU" reference inside this script still auto-dispatches K≥3 levels with ≥500 K
candidates across both visible GPUs (`dispatch.py:128-129`); results are meant to be identical.

### 10.4 Canonical Opus mine with per-K checkpoints (what §5 post-processing needs)
```bash
cd applications/alphafold/pipeline
CUDA_PATH=/usr/local/cuda ET_FLUSH_COMPRESSION=zstd \
python run_mining.py --input /root/data/af/transactions_214m_base_multi.parquet \
  --item-mapping /root/data/af/item_mapping_214m_base.parquet \
  --support 0.0000001 --max-length 50 --use-gpu --n-gpus 2 --parquet-flush \
  --output-dir /root/data/af/results_214m/opus -v 2>&1 | tee /root/data/af/results_214m/opus_mining.log
```
* `--support 1e-7` → `min_count = ceil(1e-7 × n)` = 8 for n = 76,890,945 (`core/result.py:16`); verify the
  logged `min_count` against the new n (`row_split.py:127-129`).
* `--max-length 50` stands in for "unbounded" (argparse int, `run_mining.py:195`; 22 was the observed
  ceiling [doc]); the loop terminates naturally when a level is empty (`row_split.py:601-602`).
* Outputs `…/opus/parquet/frequent_k{K}.parquet` (int32 lists) + `mining_meta_1e-07_<ts>.json`
  (`run_mining.py:246-268`). Resume after a crash: add `--resume-from-k <last complete K>`
  (`row_split.py:224-278`; it re-derives counts from the `support` column).
* VRAM: ~4.8 GB bitvec per GPU + candidate chunks from measured headroom; RAM: current level
  flat arrays only (K=9: 3.5 M × 9 × 4 B). Fits.
* Optional archive: `ET_UPLOAD_GCS=1 ET_UPLOAD_TAG=base214m_$(date +%Y%m%d)` uploads each flushed
  file to `gs://et-miner-results/<tag>/` (`row_split.py:186-195`, `io/gcs.py:231-264`; needs ADC).
Repeat with `--support 0.000001` (Blitz) etc. if per-K parquets are wanted at other thresholds.

### 10.5 Experiment suite (RUNBOOK Fase 2) — needs the import fixes first
✎ Prerequisite edits (§9-1): in `experiment_null_model.py` change `et_miner.cuda_csr_bitvec` →
`et_miner.gpu.csr_bitvec` and `et_miner.apriori` → `et_miner.gpu.row_split`; in `compute_maximal.py`
and `filter_self_sufficient.py` change `et_miner.rules` → `et_miner.core.rules`. Then:
```bash
cd /root/projects/ET-Miner/applications/alphafold          # run_all_experiments.sh paths are relative to here (§9-3)
export CUDA_PATH=/usr/local/cuda
N_GPUS=2 bash deploy/run_all_experiments.sh /root/data/af/transactions_214m_base_multi.parquet
```
What it does on this box: (1) 6 thresholds × 3 runs, single-GPU path (n_gpus=1, in-RAM results; no
checkpoint — a crash loses the whole step); (2) direct-vs-SON, single-GPU + SON with 40 M-row chunks
(⚠ the SON boolean matrix per chunk is ~40 GB host RAM at ~1,000 items — consider running
`experiment_direct_vs_son.py --chunk-size 10000000` separately, which deviates from the paper's
protocol); (3) null@769 × 100 and (4) null@8 × 100 with `--perm-per-gpu` over 2 GPUs — add
`--checkpoint` and re-run with `--resume` after preemption (`experiment_null_model.py:306-313`);
(5) `analyze_k22_proteins.py` with the hard-coded v1 list (⚠ name mismatch §9-10) — instead run:
```bash
python experiments/analyze_k22_proteins.py --data /root/data/af/transactions_214m_base.parquet \
  --item-mapping /root/data/af/item_mapping_214m_base.parquet \
  --itemset-source /root/data/af/results_214m/opus/parquet/frequent_k<KMAX>.parquet \
  --output-dir results_214m -v      # `--data` must include protein_id → use the unfiltered or multi parquet (both keep it)
```
Pin `CUDA_VISIBLE_DEVICES=0` for any run you want to be truly single-GPU (otherwise every K≥3 level
with ≥500 K candidates does a ~9.6 GB D2H + H2D bitvec copy to GPU 1, `k3plus.py:382,403`).

### 10.6 Closed + maximal (RUNBOOK Fase 3)
```bash
python experiments/compute_maximal.py --results-dir /root/data/af/results_214m/opus/parquet \
  --output-json results_214m/closed_maximal.json -v       # add --capped-top-level if K hit --max-length
```
Requires contiguous `frequent_k1…kmax` files (`compute_maximal.py:186-197`); streams row groups of 1 M
(`:242`).

### 10.7 Things that do not fit / are not reproducible as written
* The 150 GB TrEMBL file plus the CSV and outputs consume ~80 % of the 200 GB disk; keep nothing else
  there or use the FIFO stream. The 23 TiB structure set is not needed (§2).
* Wall-clock is unknown on 3090s; the H200 estimates (`RUNBOOK:152-154`, ~5–6 h) do not transfer.
  Null@8 × 100 permutations is the long pole; consider `--runs 5` first (the artifact's n).
* The paper's Base/Super/Power SON numbers (5,305 / 51,124 / 22,846) will not be reproduced by the
  current exact-only campaign script (§9-7); only direct-vs-SON re-measures SON, and only at Power.
* Exact item ids / vocabulary membership at rank-500 ties may differ from the 2026-02 run (§3), and the
  current UniProt release differs from 2025_01 (`tex:129,478` [doc]).
