# Data manifest — CPU reproduction (`runs/20261019T1718Z-cpu/data/`)

Every inbound transfer and every derived file of Phase 1. Sizes in bytes. Files that are not
committed (gitignored: above 100 MB, or re-derivable shard intermediates) are listed with size and
sha256 in `../EXCLUDED_LARGE_FILES.md`; everything else under `data/` is committed.

## Inbound sources

| # | source | URL / object | transfer (UTC) | bytes | checksum | terms of use |
|---|---|---|---|---|---|---|
| 1 | BigQuery Storage Read API (read session only; billing project `et-research-491307`) | table `bigquery-public-data.deepmind_alphafold.metadata`, columns `uniprotAccession, globalMetricValue`, 16 streams | 2026-10-19T17:32Z → 18:02Z (`logs/export_plddt_20261019T1732Z.log`) | 3,495,993,272 (CSV written) | sha256 `6068f714550a0d6c13a90d215b68486b22893b489d8203d1d6db823fe375919e` (`plddt_stats.json`) | see "Terms" (1) |
| 2 | BigQuery API, metadata reads (`bq show`, datasets.get / tables.get; no job, no write) | dataset `bigquery-public-data:deepmind_alphafold`, table `…metadata` | 2026-10-20T06:4xZ | 892 / 5,999 | `bq_meta/*.json` (committed) | — |
| 3 | UniProt FTP | `https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2026_01/knowledgebase/RELEASE.metalink` | 2026-10-19T18:11Z | 3,436 | sha256 `09367a2685fc5348fde293e94010efc2672c19165a8638b914d614f987125466` (`uniprot_meta/knowledgebase_RELEASE.metalink`, committed) | CC BY 4.0 (its `<license>` element) |
| 4 | UniProt FTP (8 parallel HTTP range requests, streamed, not stored) | `https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2026_01/knowledgebase/knowledgebase2026_01.tar.gz` | 2026-10-19T18:13:23Z → 2026-10-20T05:37:17Z (`logs/stream_trembl_2026_01_20261019T1813Z.log`, `logs/stream_trembl_2026_01_progress.log`) | 161,890,570,993 consumed (whole archive); member `uniprot_trembl.dat.gz` 160,834,306,675 copied into the reducer | md5 `e41d313b61d595a05d4f80032df44c33` (= RELEASE.metalink), sha256 `a796b7b13f653706e403f616504c9ddc4432dee39e1f6b3fabcb29e848950733`, both over every archive byte | CC BY 4.0 (RELEASE.metalink) |

An earlier stream (2026-10-19T17:33Z / 18:02Z, without hashing, about 1.5 GB consumed) was stopped and its
partial output deleted (`deviations.md` D-004, D-007; logs `logs/*_aborted_nohash.log`).
The TrEMBL member itself is never written to disk: it is reduced on the fly (as the GPU run did), and the
temporary stream chunks in `stream_work_2026_01/` are removed as they are consumed.

Terms:
1. AlphaFold DB data (pLDDT). Neither the BigQuery dataset nor the table metadata carries a licence
   field (`bq_meta/deepmind_alphafold.json`, `bq_meta/deepmind_alphafold.metadata.json`: description
   "Metadata for the AlphaFold Protein Structure Database", labels `dg_data_source: public`). The
   AlphaFold DB states its licence (CC BY 4.0) on its own web pages, which are outside the brief's
   allowed inbound sources, so it is recorded here as **not verified from an allowed source**.
2. UniProtKB 2026_01: CC BY 4.0, verified in `uniprot_meta/knowledgebase_RELEASE.metalink`
   (`<license><name>Creative Commons Attribution 4.0 International (CC BY 4.0)</name>`).

Table metadata at read time: 214,683,829 rows, 123,137,084,440 logical bytes, last modified
2024-02-15T21:02:02Z, created 2022-07-20T02:17:29Z (`bq_meta/deepmind_alphafold.metadata.json`).

## Derived files (pipeline order)

| file | produced by | rows / records | bytes | committed |
|---|---|---|---|---|
| `plddt_metadata_storageapi.csv` | `scripts/export_plddt_storage_api.py` | 214,683,829 | 3,495,993,272 | no |
| `plddt_stats.json` | `scripts/plddt_stats.py` | — | small | yes |
| `uniprot_trembl_2026_01.reduced.dat.gz` | `scripts/stream_trembl_release.sh` + `stream_tar_member.py` | 202,556,314 | 27,116,638,807 | no |
| `dat_2026_01_counts.txt` | `scripts/count_dat.sh` | — | small | yes |
| `uniprot_trembl_2026_01.pfamgo.dat.gz`, `dat_pfamgo_accessions_2026_01.txt` | `scripts/filter_dat_pfamgo.sh` | 158,700,122 | 10,580,479,054 / 1,678,722,670 (both = GPU X-005) | no |
| `extract_2026_01/plddt_metadata_pfamgo.csv` | `scripts/csv_join.py` | 90,506,840 | 1,470,718,515 | no |
| `uniprot_trembl_2026_01.csvacc.dat.gz` | `scripts/filter_dat_by_accessions.sh` | 90,506,840 | 6,447,078,382 (= GPU X-007) | no |
| `extract_2026_01/shardtest/` | `scripts/phase1_chain.sh` step `sample` (direct af-extract vs 4 shards on 2M rows) | 2,000,000 | — | logs, mappings, `verify_shard_route.json` |
| `extract_2026_01/shards/s0..s3/` | `scripts/split_csv_shards.sh`, `filter_dat_by_accessions.sh`, `af_extract_shard.sh` | 4 × 22,626,710 | — | logs, mappings |
| `extract_2026_01/transactions_214m_base.parquet`, `item_mapping_214m_base.parquet`, `merge_shards.json` | `scripts/merge_shards.py --top-pfam 500 --top-go 500` | 90,506,840 / 1,006 | 1,306,842,460 / small | parquet no, rest yes |
| `extract_2026_01/transactions_214m_base_multi.parquet`, `stats.json`, `item_support_*.parquet` | `scripts/extract_stats.py --full-csv` | 76,890,945 | 888,932,921 / small | parquet no, rest yes |
| `extract_2026_01/plddt_bin_edges.json` | `scripts/plddt_bin_edges.py` | — | small | yes |
| `phase1_verification.json` | `scripts/verify_phase1.py` | 51 checks ok, 0 required failed | small | yes |

Exact sizes and sha256 of every uncommitted file: `../EXCLUDED_LARGE_FILES.md`.
