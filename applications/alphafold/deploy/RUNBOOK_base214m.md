# RUNBOOK — base-214m definitieve run (artefact-onderbouwd)

Volledige, uitvoerbare draaiboek voor de base-vocab run over de complete AlphaFold
DB (214M), die Majors 1/2/3/5 uit `papers/peer_review_jul12.md` sluit. Gebaseerd op
`~/.claude/plans/leg-het-vast-in-crispy-glade-md-zou-je-snoopy-turing.md`.

## Wat Et moet leveren (ik kan dit niet zelf)

1. **GPU-box**: 4×H200 (of 8×) op Vast.ai. `<ssh-port> <ssh-host>`. Provisioning is
   handmatig — er is geen vast-API in de repo.
2. **GCS-auth (keyless, ADC — geen SA-key nodig).** `gcs.py` authenticeert met ADC
   authorized_user JSON (OAuth refresh-token, auto-refresh). Org-policies die SA-keys
   blokkeren raken dit niet. Doe één van:
   - lokaal `gcloud auth application-default login` → `deploy_base214m.sh` stap 4 pusht
     die ADC automatisch naar de box; OF
   - op de box `gcloud auth application-default login --no-launch-browser`.
   `GCP_SA_KEY_B64` mag leeg blijven. Verifieer met `gcloud storage ls gs://et-miner-results`.
   `ET_UPLOAD_TAG` staat op `base214m_20260712` — pas aan indien gewenst.

Zonder box + creds staan Fase 0-5 stil; alle **code** (hieronder) is af, getest en
klaar.

## Roll-ready status (lokaal af + geverifieerd)

| Onderdeel | Bestand | Status |
|---|---|---|
| Row-split pre-flight (Fase 0.3b) | `experiments/validate_row_split.py` | compileert, lint schoon |
| Maximal + closed (Fase 3, Major 5) | `experiments/compute_maximal.py` | 7/7 unit-tests groen |
| Deepest-itemset accessions (Major 3) | `experiments/analyze_k22_proteins.py` | smoke-test groen |
| Deploy base-vocab | `deploy/deploy_base214m.sh` | `bash -n` OK |
| GCS-creds scaffold | `applications/alphafold/.env` | placeholders, gitignored |

Tests: `uv run --with pyarrow pytest tests/test_compute_maximal.py -q`

## Correcties op het plan (gevonden tijdens uitwerking)

1. **`build-from-metadata`, niet `build-transactions`.** De pLDDT-via-BigQuery-route
   (23TB CIF vermijden) matcht `af-extract build-from-metadata --plddt-csv ...`.
   `build-transactions` vereist `--tar-dir` (de tars zelf). Beide geven de 6 pLDDT-bins
   bij `include_plddt=true`.
2. **Annotatie-veldset = `xref_pfam` (niet `xref_interpro`).** De base-vocab is
   Pfam+GO+pLDDT. `pipeline_214m.py` default op interpro; voor de DAT.gz-route maakt dit
   niet uit (af-extract leest Pfam-DR-regels rechtstreeks uit de .dat.gz).
3. **Row-split NCCL wordt 100× geraakt door de null@8** → `validate_row_split.py` is de
   verplichte gate. `apriori(n_gpus=2)` routeert intern naar `_apriori_row_split_multi_gpu`,
   dus het script vergelijkt de echte productie-entry vs single-GPU.
4. **Maximal berekend als Polars K↔K+1-pass** (niet Rust): de Rust compact-encoding werkt
   op in-memory column-indices, de per-K parquets bevatten item-ids/float-support. De
   pass leidt closed EN maximal in één keer af (anti-monotonie), en cross-checkt zo de
   aparte `prune_equal_support`-closed-run.

## Draaiboek (op de box)

### Fase 0 — Deploy + pre-flight
```bash
# lokaal, vanaf repo root:
applications/alphafold/deploy/deploy_base214m.sh <ssh-port> <ssh-host>
```
Dit synct code (geen data), bouwt `et_miner_rust` (maturin) + `af-extract` (cargo),
pusht GCS-creds, warmt kernels. Aan het eind print het de exacte A-E commando's.

### Fase 0.4 — Data-acquisitie (DIRECT op de box, aria2c)
```bash
cd /workspace/ET-miner && source .venv/bin/activate
export ET_UPLOAD_GCS=1 ET_UPLOAD_TAG=base214m_$(date +%Y%m%d)
aria2c -x16 -s16 -d /workspace/data \
  'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.dat.gz'
aria2c -x16 -s16 -d /workspace/data \
  'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz'
bq query --use_legacy_sql=false --format=csv --max_rows=300000000 \
  'SELECT uniprotAccession, globalMetricValue FROM `bigquery-public-data.deepmind_alphafold.metadata`' \
  > /workspace/data/plddt_metadata.csv
```
NB: aria2c NOOIT via de lokale tunnel — direct op de box (`.claude/rules/et-miner.md`).

### Fase 1 — Extractie (base vocab 1006 defined / 1002 frequent)
```bash
AFX=applications/alphafold/af-extract/target/release/af-extract
$AFX build-from-metadata \
  --annotations /workspace/data/uniprot_trembl.dat.gz \
  --plddt-csv /workspace/data/plddt_metadata.csv \
  --top-pfam 500 --top-go 500 \
  --output /workspace/data/transactions_214m_base.parquet \
  --item-mapping /workspace/data/item_mapping_214m_base.parquet
```
**GATE:** noteer uit de log het multi-feature-eiwit-aantal + total-items. Dit zijn de
nieuwe Table 1 / abstract-getallen (v1 was 76,9M van 205,6M).

### Fase 0.3b — Row-split validatie (VÓÓR de null@8)
```bash
python3 applications/alphafold/experiments/validate_row_split.py \
  --data /workspace/data/transactions_214m_base.parquet \
  --subset-size 1000000 --min-count 50 --n-gpus 2 -v
```
GREEN (exit 0) → draai de suite met `N_GPUS=4`. RED (exit 1) → `N_GPUS=1` fallback voor
de null-runs (geen NCCL, serieel/trager).

### Fase 2 — Experiment-suite (5 artefact-groepen → GCS)
```bash
N_GPUS=4 ET_UPLOAD_GCS=1 \
  bash applications/alphafold/deploy/run_all_experiments.sh \
  /workspace/data/transactions_214m_base.parquet
```
Draait: (1) mining-campagne 6×3, (2) Direct-vs-SON, (3) null@769 100-perm,
(4) **null@8 100-perm** (de kritische Major 1+2), (5) deepest-itemset accessions.
Draai in `tmux` (preemption-bestendig; null@8 heeft `--checkpoint/--resume`).

Voor Major 3 met evidence-codes, de accessions-stap parametrisch:
```bash
python3 applications/alphafold/experiments/analyze_k22_proteins.py \
  --data /workspace/data/transactions_214m_base.parquet \
  --item-mapping /workspace/data/item_mapping_214m_base.parquet \
  --itemset-source results_214m/parquet/frequent_k<KMAX>.parquet \
  --annotations /workspace/data/goa_uniprot.gaf \  # optioneel, voor evidence-codes
  --output-dir results_214m -v
```

### Fase 3 — Closed + maximal (Major 5)
```bash
python3 applications/alphafold/experiments/compute_maximal.py \
  --results-dir results_214m/parquet \
  --output-json results_214m/closed_maximal.json -v
# Voeg --capped-top-level toe ALS de campagne length-capped was (max K), anders is
# de top-level closed/maximal een bovengrens.
```

## Orchestratie — /goal + /loop

`/loop 25m` met deze poll-prompt (elke beurt de inventaris + kern-metrics PRINTEN — de
/goal-evaluator leest alleen wat in de conversatie verschijnt):

> Poll de base214m-run. Print: (a) `gsutil ls -r gs://et-miner-results/base214m_*/`
> inventaris; (b) pull nieuwe parquets + print kern-metrics: multi-feature-eiwitaantal
> (Fase 1 gate), null@8 K-distributie, closed+maximal counts; (c) remote `tmux ls` +
> laatste 20 regels van het experiment-log. Als een fase klaar is, vink hem af.

`/goal` completion-conditie eromheen:

> Stop wanneer alle 5 artefact-groepen op GCS staan (inputs, campaign, direct-vs-son,
> null@769, null@8 + accessions + closed_maximal) EN ik in de conversatie heb
> gerapporteerd: de null@8 K-distributie (null vs bio op de Opus-drempel), de
> closed+maximal counts, en het multi-feature-eiwitaantal.

## Fase 4-5 — Her-lock paper + Zenodo v2 (na de run)

`papers/et_miner_proteome.tex`: vervang feb-artefact-getallen door de nieuwe canonieke
run (Table 1, `tab:campaign/kdist/null-model`, Majors 1/2/3/5, FDR-noot,
data-availability → "gedeponeerd op GCS + Zenodo"). Compile 2×, cite-audit, council-pass,
peer-review-checklist afvinken, Zenodo v2 met diff vs v1. Elk gewijzigd getal opnieuw
tegen het nieuwe artefact verifiëren.

## Compute & timing (4×H200, ~85M multi-feature)
Data ~0,5-1u · extractie ~0,5-1u · campagne+Direct/SON ~1u · null@769 ~25min ·
**null@8 ~1,5-2u** · closed/maximal+accessions ~0,5u → **~5-6u wall-clock**, ~$45-90.
