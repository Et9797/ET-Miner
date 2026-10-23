# AlphaFold application

ET-Miner applied to the AlphaFold Protein Structure Database: 214,683,829 entries, 205,620,298 with mean pLDDT >= 50,
annotated from UniProtKB TrEMBL 2026_01; the 76,890,945 proteins with at least two of the 1,002 frequent features
(500 Pfam domains, 500 GO terms, 2 pLDDT bins) are mined with Apriori.

## Contents

| path | what |
|---|---|
| `af-extract/`, `rust/` | Rust feature extractor and transaction tools (sources; build with cargo/rustc) |
| `pipeline/`, `experiments/`, `deploy/`, `utils.py` | extraction pipeline, mining experiments, deployment scripts |
| `results_214m/` | outputs of the V1 preprint; the claims that the reproduction checked (`CLAIMS.md`) |
| `paper/` | the V2 preprint (`et_miner_proteome.tex`/`.pdf`), its figures (`figures/make_figures.py`) and a blog draft |
| `runs/20260902T0000Z/` | full GPU re-execution of 2026-09-02 (2 x RTX 3090); start at its `README.md` |
| `runs/20261019T1718Z-cpu/` | CPU-only re-execution of 2026-10-19..23 (Ryzen 5 4600G, 6 threads); start at `CPU_REPORT.md` |
| `COMPARISON_REPORT.md` (+ `_rows.json`) | per-claim verdicts of the V1 preprint against the GPU re-execution |
| `CLAIMS.md`, `RESULTS.md`, `INCONSISTENCIES.md` | extracted claims, fresh GPU values, discrepancies |
| `CHANGELOG_V1_V2.md` | V1 to V2 preprint, value by value |
| `PROGRESS.md` | journal of the GPU campaign (cited as a source in `CLAIMS.md` and the report) |
| `CPU_REPRO_GOAL.md` | brief of the CPU campaign, including the null-model criterion fixed before the runs |
| `EXCLUDED_FROM_MAIN.md` | files of the reproduction branches that are not on main, with size and sha256 |

The reports were written at the root of the reproduction branches and moved here unchanged, so their paths follow two
conventions. `runs/...`, `paper/...`, report names such as `RESULTS.md`, and "repository root" in the campaign reports
mean this directory, `applications/alphafold/`. Paths that start with `applications/alphafold/` or `src/` mean the
repository root.

## Results in brief

- GPU (RTX 3090; one for the six-threshold campaign, two for the 4-protein run and the null model): exhaustive mining of the 76.9M proteins down to 8 proteins (Opus) gives 26,849,505 itemsets up to
  K=22 in 43.9 min; at 4 proteins 48,007,493 itemsets, still K=22. A 100-permutation null model never exceeds K=6.
- CPU: every full-data itemset table the GPU run saved is reproduced itemset for itemset and count for count, from data
  rebuilt from the same sources; the null model is consistent as a distribution. Opus took 4.9 h on six threads with
  count inference. Details and caveats: `runs/20261019T1718Z-cpu/CPU_REPORT.md`.
- The association-rule counts of the GPU run (29,540 rules) came from a rule generator that dropped rules on
  non-ascending itemsets; the exact count is 53,447 (`runs/20261019T1718Z-cpu/INCONSISTENCIES.md`, CI-001). The paper
  does not report rules.

## Engine versions and re-running

The GPU run used ET-Miner at `25d0957` (the base of the reproduction branches), the CPU run main at `37819e2`
(`runs/20261019T1718Z-cpu/env.json`). The run scripts are kept byte for byte as they were run; against a later engine
they may need adjusting. Four of them locate the repository from the branch layout and need one change each on main:

| script | as run | on main |
|---|---|---|
| `runs/20261019T1718Z-cpu/scripts/phase1_chain.sh:15` | `$R/../../applications/alphafold/af-extract` | `$R/../../af-extract` |
| `runs/20261019T1718Z-cpu/scripts/tests/shard_route_synthetic.sh:12` | `$S/../../../applications/alphafold/af-extract` | `$S/../../../af-extract` |
| `runs/20261019T1718Z-cpu/scripts/a1_k22.sh:15` | `$REPO/runs/...` | `$REPO/applications/alphafold/runs/...`, with the go.obo listed in `EXCLUDED_FROM_MAIN.md` |
| `runs/20260902T0000Z/phase2/scripts/son_run.py:39` | `parents[4] / "applications" / "alphafold"` | `parents[4]` |

Large inputs and tables stay local (see `.gitignore` here and `EXCLUDED_FROM_MAIN.md`). Versions of the third-party
files that are not in the repository: Gene Ontology `go.obo` release `releases/2026-07-26` (the K=22 parent-child check
in both runs) and InterPro2GO version `2026/07/06 18:13:03` (the citation audit, `runs/20260902T0000Z/phase4/v2_checks/`).

The gcloud account that ran the BigQuery reads is replaced by `<redacted>` in the records copied here, and the address
of the rented GPU machine in the usage examples of `deploy/` by placeholders. Data derived
from UniProt, the AlphaFold Database and the Gene Ontology, and quoted third-party text such as the publisher abstracts
in `runs/20260902T0000Z/phase4/citation_audit/`, remain under their sources' terms; the repository licence does not
change them.
