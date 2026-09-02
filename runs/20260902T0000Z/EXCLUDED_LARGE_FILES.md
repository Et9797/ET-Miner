# Excluded large files (not in git)

These files of the 2026-09-02 base214m reproduction exceed GitHub's 100 MB
per-file limit and are therefore excluded by `.gitignore`. Every one of them is
a raw input or an intermediate that is re-derivable from public sources with the
scripts in `phase2/scripts/` (see `README.md`, "Regenerating the excluded raw
inputs"). Sizes and sha256 were computed on the campaign box on 2026-09-02 so a
regenerated copy can be verified byte-for-byte.

| file | bytes | size | sha256 |
|---|---|---|---|
| `phase2/bwtest/proteome-tax_id-9606-12_v4.tar` | 730406912 | 0.0 MB | `fef8fbcfb26b4a34d8cce31c932e50ceb557290697b96688f13bc71052f49c95` |
| `phase2/data/dat_pfamgo_accessions_2026_01.txt` | 1678722670 | 0.0 MB | `5a411b17a79cadbe6e12f3e1eb649071324c0dccf93c5502056d0871ea5d3b42` |
| `phase2/data/plddt_metadata_storageapi.csv` | 3495993272 | 0.0 MB | `e089f052a6fe500aeca89ac10b4ccbdaaa9096c7175244a6143395edbedb2b32` |
| `phase2/data/uniprot_trembl_2026_01.csvacc.dat.gz` | 6447078382 | 0.0 MB | `cbebc9a25144775a18ad923698b7f17cd97e7526aa1e694f01db927726b6bebc` |
| `phase2/data/uniprot_trembl_2026_01.pfamgo.dat.gz` | 10580479054 | 0.0 MB | `30c409830f6b05423f56ebdb1b8091f8e4869297475924290ba9de31d27e55c1` |
| `phase2/data/uniprot_trembl_2026_01.reduced.dat.gz` | 27116638807 | 0.0 MB | `35dc29b82d44f953aee014502fdab281bf701f42b81ed89589e7893128e53afc` |
| `phase2/extract_2026_01/plddt_metadata_pfamgo.csv` | 1470718515 | 0.0 MB | `f905e3b42237d5a2be12285b007a1913e0876dc1676ae604149f8a3d0d17f198` |
| `phase2/extract_2026_01/transactions_214m_base.parquet` | 1317592610 | 0.0 MB | `3cbf518a6d391be188a8f446f8fe8450d616cab024f4fecb2b83790e4811fe23` |
| `phase2/extract_2026_01/transactions_214m_base_multi.parquet` | 890682214 | 0.0 MB | `17a51ca4f2892efdfc415f1367bf1f9711ffd400be44d1a05dda3420e9eb8912` |
| `phase2/streamtest/docs.tar.gz` | 345581195 | 0.0 MB | `c6912e16ef53f4b71a19aeaf0405f2dd297ca3616917d79eb0285a4d933b8f1d` |

Origin of each file:

- `phase2/data/uniprot_trembl_2026_01.reduced.dat.gz` — TrEMBL 2026_01 flat file reduced to the record lines the extractor reads (`stream_trembl_release.sh`).
- `phase2/data/uniprot_trembl_2026_01.pfamgo.dat.gz` — records carrying Pfam or GO cross-references (`filter_dat_pfamgo.sh`).
- `phase2/data/uniprot_trembl_2026_01.csvacc.dat.gz` — records whose accession occurs in the AlphaFold metadata export (`filter_dat_by_accessions.sh`).
- `phase2/data/dat_pfamgo_accessions_2026_01.txt` — accession list of the Pfam/GO records.
- `phase2/data/plddt_metadata_storageapi.csv` — `uniprotAccession, globalMetricValue` export of `bigquery-public-data.deepmind_alphafold.metadata` (`export_plddt_storage_api.py`; sha256 also in RESULTS.md M-006).
- `phase2/extract_2026_01/plddt_metadata_pfamgo.csv` — the metadata export restricted to Pfam/GO accessions (input of `run_af_extract_lean.sh`).
- `phase2/extract_2026_01/transactions_214m_base.parquet`, `..._multi.parquet` — the extracted transaction sets (all pLDDT-passing proteins; proteins with more than one feature) produced by `af-extract build-from-metadata`.
- `phase2/bwtest/proteome-tax_id-9606-12_v4.tar` — human-proteome tar used only as a bandwidth probe.
- `phase2/streamtest/docs.tar.gz` — download probe for the tar-member range streamer (its `work/` chunks are excluded too).
