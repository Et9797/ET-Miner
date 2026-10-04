# Gene-neighborhood co-occurrence mining with ET-Miner: Technical Spec
Status: final. Source: /interview, 2026-10-04. Depth: full.

## Overview

A pipeline that mines conserved multi-gene sets from prokaryotic genome
neighborhoods with ET-Miner. Each transaction is a window of consecutive genes
on one contig; each item is the protein family of one gene. The pipeline runs on
quality-filtered GTDB species representatives and on consumer hardware
(2x RTX 3090). It has two deliverables: (1) evidence that the pipeline recovers
known multi-gene systems (housekeeping operons as a hard control, anti-phage
defense systems as a reported recall curve), and (2) a ranked list of
candidate novel defense systems, scored by enrichment in defense islands
(the co-localisation approach of Doron et al. 2018), with protein sequences
for follow-up. It is a showcase of ET-Miner on biological data; there is no
wet-lab validation in scope.

One framing point shapes the whole design: ET-Miner is exact, so every
itemset with `count >= min_count` in the transactions is found. Recall of
known systems therefore measures **what the pipeline loses before mining**
(windowing, item definition, family splitting, filters), not whether the
miner finds what is in its input. The oracle gate (MH-7) covers the miner;
the recall analysis covers the representation.

## Context (recon)

- No pipeline for a biological dataset lives in this repo. The AlphaFold
  application in `README.md` has no code here. The only dataset prep is
  `datasets/prepare_online_retail.py` (polars + requests + loguru, parquet out).
- Entry point: `et_miner.apriori()` (`src/et_miner/core/apriori.py:370`).
  With `use_gpu=True, streaming=False` it builds CSR from transactions and
  dispatches to the row-split miner (`gpu/row_split.py::_apriori_row_split_multi_gpu`),
  which honours `n_gpus`, `max_length`, `output_dir` (per-K parquet flush),
  `prune_equal_support`, `max_ram_gb`/`max_vram_gb` and `level_callback`.
- `prune_equal_support=True` returns exactly the frequent **free-sets**
  (generators), not the complete lattice and not closed sets
  (`apriori.py:423`). The SON streaming route refuses it (`apriori.py:276`).
- `anchor_items` is a post-filter on the row-split route; it does not shrink
  the candidate space (`gpu/row_split.py:1088`). This spec does not use it.
- The row-split miner refuses `n_transactions > 2^31 - 1` (`gpu/row_split.py:203`).
  There are no transaction weights anywhere in the engine.
- `level_callback(k, n_candidates, n_frequent, ms)` fires per level
  (`gpu/row_split.py:515,793`). `bench/child_run.py` runs one config per fresh
  process and records per-device peak VRAM and peak RSS. That is the pattern
  to copy for the sweep.
- `min_count` derivation is `count >= ceil(min_support * N)`, implemented
  exactly in `et_miner.core.result._min_count` (`tests/test_min_count.py`).
- Correctness policy (`CLAUDE.md`): efficient-apriori is the oracle, called with
  `min_support = (min_count - 0.5) / N` and an explicit `max_length`.

## Requirements

### Must Have

- MH-1 [accepted] **Genome set.** GTDB species representatives (one genome per
  species) from one pinned GTDB release, filtered to CheckM2 completeness >= 90%
  and contamination <= 5% (GTDB-supplied values). The pipeline reports the
  number of genomes kept and dropped.
- MH-2 [accepted] **One item per protein.** The item is the protein's dominant
  Pfam domain (highest bit score among hits passing the Pfam gathering
  threshold). Proteins without a Pfam hit get their MMseqs2 cluster at ~30%
  identity. Items are integer ids with a vocabulary table mapping id ->
  (source, Pfam accession or cluster id, label).
- MH-3 [accepted] **Annotation source.** Reuse precomputed per-protein Pfam
  hits for the pinned release where they exist. Annotate only the missing
  genomes with pyhmmer, using the same Pfam version and threshold mode.
- MH-4 [accepted] **Two tilings.** Per contig, in gene order, non-overlapping
  windows of W=20 genes, once at offset 0 (tiling A) and once at offset 10
  (tiling B). Windows never cross contig boundaries. Each tiling is a separate
  transaction set and is mined separately.
- MH-5 [accepted] **Core filter.** Families present in > 50% of kept genomes
  (genome-level prevalence) are removed from the discovery transactions. The
  control transactions keep every family.
- MH-6 [accepted] **Control run (hard go/no-go).** Mine tiling A and tiling B
  with all families at a high threshold (see D-6). Every operon in a
  checked-in control set must be recovered (definition in D-8) in at least one
  tiling.
- MH-7 [accepted] **Oracle gate.** Before any sweep run, a seeded subsample of
  ~1e5 discovery windows is mined by ET-Miner (the same route and flags as
  the sweep) and by efficient-apriori. Any difference in itemsets or absolute
  counts hard-fails the pipeline (details in D-7).
- MH-8 [stated] **Threshold sweep.** Discovery runs at min_count in
  {10, 20, 50, 100} for each tiling (8 runs), `max_length=11`,
  `prune_equal_support=True` (see T-3), `use_gpu=True`, `n_gpus=2`.
- MH-9 [accepted] **Failure is data.** A sweep point that OOMs, trips the
  engine's memory guard or exceeds 4 h wall-clock time is recorded with its
  reason, the last completed level K, and peak VRAM/RSS. The sweep continues.
  There is no SON fallback and no vocabulary shrinking for a failed point.
- MH-10 [accepted] **Ground truth.** DefenseFinder (pinned software and model
  versions) runs on every kept genome. Its detected loci are the ground truth
  for recall and the defense-gene positions for enrichment.
- MH-11 [accepted]/[assumed] **Recall report.** For each completed sweep point,
  report locus-level and type-level recall (D-9). Recall is computed only over
  eligible types/signatures, i.e. those with >= min_count DefenseFinder loci.
  There is no recall target [stated]. Every eligible miss gets a cause from a
  fixed taxonomy (D-9).
- MH-12 [accepted] **Candidate ranking.** Free-sets of size >= 2 that are not
  known systems (D-10) are scored for defense-island enrichment against a
  global background, with a one-sided Fisher exact test and Benjamini-Hochberg
  correction (D-11).
- MH-13 [accepted] **Outputs.** Per-K parquet per run (`output_dir`), a
  candidate table, and a protein FASTA per top candidate (D-12).
- MH-14 [accepted] **Code location.** A separate repository that depends on
  et-miner and on `et_miner_rust`, both pinned to one et-miner revision (as
  `README.md` describes). No bio dependencies enter this repo.

### Should Have

- SH-1 [assumed] **Pre-flight memory estimate.** Before launching a run,
  estimate bitvector bytes as `n_frequent_items x ceil(N/64) x 8`. If the
  estimate exceeds 80% of total VRAM, record the point as "does not fit
  (pre-flight)" without launching it. The estimate is logged next to the
  measured peak for every run that launches.
- SH-2 [assumed] **Determinism.** One config file pins the GTDB release, Pfam
  version, MMseqs2 version and parameters, DefenseFinder versions, the et-miner
  revision and all seeds. Every output carries the config hash.
- SH-3 [assumed] **Closure expansion.** For each reported candidate, compute
  its closure: the items present in every occurrence window. Report it next to
  the free-set, because a free-set is the minimal generator and can be smaller
  than the biological unit.
- SH-4 [assumed] **Candidate deduplication.** After ranking, collapse candidates
  whose closures have Jaccard >= 0.5 and keep the highest-ranked one. The others
  are listed as members of its group.

### Nice to Have

- NH-1 [assumed] A phylum-stratified enrichment (Cochran-Mantel-Haenszel) as a
  sensitivity check next to the global test.
- NH-2 [assumed] Phase 2 on AllTheBacteria. It is out of scope for this spec
  (see Out of Scope), but nothing in the design should block it.

## Acceptance Criteria

- AC-1 (MH-1): `genomes.parquet` holds only genomes from the pinned release
  with completeness >= 90 and contamination <= 5. The run log states the
  kept and dropped counts, and they add up to the release's representative
  count.
- AC-2 (MH-2): every protein in `proteins.parquet` has exactly one item. A
  protein with a Pfam hit above GA has the item of its highest-bit-score
  domain, and a protein without one has an MMseqs2 cluster item. A unit test
  with a hand-built three-protein fixture (multi-domain, single-domain,
  no-hit) passes.
- AC-3 (MH-3): for a sample of 100 genomes that have precomputed hits, a
  pyhmmer re-annotation with the pinned Pfam version gives the same dominant
  domain for >= 99% of proteins. Otherwise the run stops and reports the
  mismatch rate. (This checks that reused and computed annotations are
  interchangeable.)
- AC-4 (MH-4): for a synthetic contig of 45 genes, tiling A gives windows
  [0,20), [20,40), [40,45) and tiling B gives [0,10), [10,30), [30,45). A
  property test confirms that every run of <= 11 consecutive genes lies
  wholly inside at least one window of A or B. No window spans two contigs.
- AC-5 (MH-5): no discovery transaction contains an item whose genome-level
  prevalence is > 50%. Control transactions contain all items.
- AC-6 (MH-6): the control run recovers 100% of the checked-in control set.
  Otherwise the pipeline stops before the sweep and lists the missing operons
  with their direct counts.
- AC-7 (MH-7): on the seeded subsample, ET-Miner's complete lattice
  (`prune_equal_support=False`) equals efficient-apriori's output exactly
  (itemsets and absolute counts). ET-Miner's free-set output equals the
  free-sets derived from the oracle lattice. The command exits non-zero on any
  difference.
- AC-8 (MH-8): the sweep produces one result record per (tiling, min_count)
  pair, 8 in total, each with status in {ok, oom, memory_guard, timeout,
  preflight_no_fit}.
- AC-9 (MH-9): a forced failure (a run started with `max_vram_gb` set below
  the level-1 footprint) produces a `memory_guard` record with the last
  completed K, and the next sweep point still runs.
- AC-10 (MH-10): `defense_loci.parquet` has one row per DefenseFinder-detected
  system with genome, contig, gene index range, system type and the
  mandatory/accessory role of each protein.
- AC-11 (MH-11): for every completed sweep point, the recall table lists each
  eligible signature as recovered or missed. Every miss has exactly one cause
  from the D-9 taxonomy. A test checks the derived-support recovery rule
  (D-8) against direct counts on a fixture where a mandatory set is non-free.
- AC-12 (MH-12): every candidate row has n, k, the background rate, the
  odds ratio, a p-value and a BH q-value. No candidate overlaps a DefenseFinder
  system in >= 50% of its occurrences. The q-values are monotone in p within a
  run.
- AC-13 (MH-13): for every completed run, the per-K parquet files exist under
  that run's `output_dir`. The candidate table and one FASTA per top-N
  candidate exist. Each FASTA holds the proteins of up to 5 example loci,
  with headers that give genome, contig and gene index.
- AC-14 (MH-14): the pipeline repo's lockfile pins `et-miner` and
  `et_miner_rust` to the same git revision. `uv sync` in a clean clone
  succeeds.

## Technical Architecture

- [accepted] Separate repo, Python + Polars + parquet, matching the engine's
  stack. Rationale: the bio tooling (pyhmmer, MMseqs2, DefenseFinder) and its
  dependencies stay out of the engine and its CI.
- [assumed] External tools are called as subprocesses (MMseqs2,
  DefenseFinder) or through Python bindings (pyhmmer). Each stage reads and
  writes parquet and is idempotent: it skips work when its output exists and
  carries the current config hash.
- [accepted] Mining runs one sweep point per child process (the
  `bench/child_run.py` pattern): a fresh CUDA context per run, a hard
  wall-clock kill at 4 h, and a single JSON result line on stdout.
- [assumed] NCCL settings follow `bench/README.md` for the box in use.

Data flow:

```
GTDB metadata ──► S1 genome QC ──► genomes.parquet
GTDB proteins ──► S2 gene table ─► genes.parquet (genome, contig, idx, start, end, strand, protein_id)
                  S3 annotation ─► proteins.parquet (protein_id, item, item_source)
                  S4 DefenseFinder ► defense_loci.parquet
                  S5 vocabulary ─► vocab.parquet (item, prevalence, is_core, label)
                  S6 windows ────► windows_{discovery,control}_{A,B}.parquet (window_id, genome, contig, first_idx, items)
                  S7 oracle gate     (hard-fail)
                  S8 control run     (hard-fail)
                  S9 sweep ──────► runs/{tiling}_{min_count}/k*.parquet + result.json
                  S10 recall ────► recall.parquet
                  S11 candidates ► candidates.parquet + fasta/
```

## Detailed Design

- **D-1 Gene order (S2)** [assumed]. Gene order and coordinates come from
  the GTDB protein FASTA headers (Prodigal format: contig-prefixed ids plus
  start/end/strand). Genes are sorted by (contig, start). If the pinned
  release's headers lack coordinates, S2 stops with a clear error; see Open
  Questions.
- **D-2 Dominant domain (S3)** [accepted]. Among Pfam hits passing the
  gathering threshold, pick the hit with the highest bit score per protein.
  Break ties by lowest Pfam accession.
- **D-3 Clustering (S3)** [accepted]/[assumed]. Only proteins without a Pfam
  hit are clustered: MMseqs2 at 30% identity [accepted] and 80% coverage
  [assumed]. The cluster representative's id is the item key.
- **D-4 Vocabulary (S5)** [accepted]. Prevalence is the fraction of kept
  genomes with >= 1 protein carrying the item. `is_core = prevalence > 0.5`.
- **D-5 Windows (S6)** [accepted]/[assumed]. Items are deduplicated within a
  window (set semantics; paralogs collapse) [assumed]. Windows with < 2
  distinct items after the core filter are dropped [assumed]. N is reported
  after dropping. The pipeline asserts `N < 2^31 - 1` per tiling.
- **D-6 Control threshold (S8)** [assumed]. The control run uses min_count =
  25% of the kept genome count. *This corrects the "1% of N" wording proposed
  during the interview (Q12): N counts windows (~10^7), so 1% of N would
  exceed the number of genomes and nothing would be frequent.*
- **D-7 Oracle gate (S7)** [accepted]. Use a seeded random sample of 1e5
  discovery windows from tiling A at min_count = 10 and `max_length=11`.
  Call ET-Miner with `min_support = (min_count - 0.5) / N_sub` and assert
  `_min_count(min_support, N_sub) == min_count` first. Call efficient-apriori
  with the same `min_support` and `max_length`. Run ET-Miner twice: complete
  lattice (exact equality with the oracle) and free-set mode (equality with
  the free-sets derived from the oracle lattice).
- **D-8 Recovery rule** [assumed]. Sweep outputs are free-sets, so a set M
  need not appear literally. Define `derived(M) = min{support(S) : S ⊆ M, S
  non-empty, S in output}`. M is **recovered** in a run iff
  `direct_count(M) >= min_count` and `derived(M) == direct_count(M)`.
  `direct_count` is an independent scan of the same windows. Soundness: every
  frequent M has a free subset with equal support, and that subset is in the
  output. This rule serves both the control set (MH-6) and recall (MH-11).
- **D-9 Recall (S10)** [assumed]. The **signature** of a DefenseFinder locus is
  the set of items of its mandatory proteins [accepted, Q7]. Locus-level
  recall is the fraction of eligible loci whose signature is recovered in >= 1
  tiling. Type-level recall is the fraction of eligible system types with >= 1
  recovered signature. Eligible means >= min_count loci share that signature.
  Miss-cause taxonomy, checked in this order:
  `span_gt_11` (the mandatory genes span > 11 genes),
  `core_filtered` (a mandatory item is core),
  `signature_split` (the type's loci spread over several signatures, none
  eligible alone),
  `contig_break` (the locus touches a contig end),
  `other`.
- **D-10 Known-system filter (S11)** [accepted]/[assumed]. A free-set is
  "known" if >= 50% of its occurrence windows overlap a DefenseFinder-detected
  locus [assumed threshold]. Known free-sets go to recall, not to candidates.
- **D-11 Enrichment (S11)** [accepted]. A window is **defense-adjacent** if
  it, or the 10 genes on either side of it on the same contig, contains a
  DefenseFinder-hit gene. The background rate p0 is the fraction of
  defense-adjacent windows in the tiling. For a candidate, n is its number of
  occurrence windows and k the defense-adjacent ones among them. Test with a
  one-sided Fisher exact test on [[k, n-k], [K0, N0-K0]], then apply BH within
  each run. Occurrences come from an item -> window inverted index (sorted
  window-id arrays intersected per candidate), not from the miner. The score
  uses the tiling where the candidate has the higher support [assumed].
- **D-12 Outputs (S11)** [accepted]/[assumed]. The final candidate table comes
  from the lowest min_count that completed in both tilings [assumed]. Columns:
  free-set items and labels, closure (SH-3), support A, support B, n, k,
  odds ratio, p, q, group id (SH-4), and up to 5 example loci. A FASTA is
  written for the top 100 by q, then odds ratio [assumed top-N].

## Edge Cases & Error Handling

| Case | Expected behavior |
|---|---|
| Contig shorter than a window | Becomes one short window; dropped if it has < 2 distinct items (D-5) |
| Locus at a contig end | Mined as-is; a resulting recall miss is tagged `contig_break` |
| Same family twice in a window (paralogs) | Counted once (set semantics) |
| N per tiling >= 2^31 - 1 | S6 refuses; it cannot happen for GTDB representatives at W=20 but is asserted |
| Sweep point OOM / guard / timeout | Recorded per MH-9; sweep continues |
| Pre-flight estimate > 80% VRAM | Recorded as `preflight_no_fit`, not launched (SH-1) |
| Oracle mismatch | Hard-fail before any sweep run (MH-7) |
| Control operon missing | Hard-fail before the sweep; report its direct counts (MH-6) |
| Reused vs computed Pfam disagree | Stop if agreement < 99% (AC-3) |
| Mandatory genes span > 11 genes | Not guaranteed by the tilings; tagged `span_gt_11`, not a pipeline bug |
| Free-set smaller than the system | Reported with its closure (SH-3) |
| DefenseFinder finds no system of a type | The type is not eligible; it is not counted as a miss |

## Tradeoffs & Decisions

| Decision | Reason | Rejected alternative |
|---|---|---|
| Two offset tilings, mined separately [accepted] | Spans <= 11 are always captured whole, and each locus counts at most once per tiling | Sliding stride W/2 (support inflated up to 2x); gene-centered (N ~10x, each locus counted up to W times) |
| One item per protein, dominant domain [accepted] | No trivial "systems" from multi-domain proteins; 1:1 mapping to DefenseFinder genes | Every domain an item; domain architecture as item (splits support) |
| Pfam + MMseqs2 for the rest [accepted] | New systems are often made of genes without known domains | Pfam only; clusters only |
| Core families out of discovery [accepted] | Prevents 2^20-sized lattices from housekeeping windows at low thresholds | Keep everything and rely only on free-set pruning |
| `prune_equal_support=True` in the sweep [assumed] | 3-50x fewer candidates on dense data (engine docstring) on 48 GB VRAM; recall still works via D-8 | Complete lattice (literal lookups, larger memory footprint) |
| Failure recorded, no fallback [accepted] | Keeps the sweep points comparable; the hardware limit is itself a result | SON fallback (no free-set pruning); vocabulary shrink (changes the question) |
| Global background + Fisher + BH [accepted] | Simple and defensible | Phylum-stratified (NH-1); raw fraction without a test |
| No `anchor_items` [assumed] | It is a post-filter with no compute saving, and the enrichment step needs all free-sets anyway | Anchor on known defense genes |
| No fixed recall target [stated] | Recall measures representation loss, reported per threshold | >= 80% / >= 95% target |

## Open Questions

- OQ-1 (builder, before S3): do GTDB or AnnoTree publish per-protein Pfam
  hits for the pinned release, with the protein ids used in the GTDB protein
  FASTA? If not, MH-3 falls back to a full pyhmmer run. That is a CPU cost
  estimated in the thousands of core-hours (uncertain, measure on 100 genomes
  first).
- OQ-2 (builder, before S2): do the pinned release's protein FASTA headers
  carry contig ids and coordinates? If not, run Prodigal on the genome FASTA.
- OQ-3 (user): which operons form the control set? Proposal: bacterial
  F-type ATP synthase, the ribosomal S10-spc-alpha cluster restricted to <= 11
  consecutive genes, and NADH dehydrogenase (nuo) restricted to <= 11
  consecutive genes. Each entry is a Pfam-accession set in a checked-in file
  with a source citation.
- OQ-4 (user): licence terms for redistributing the outputs (GTDB data,
  Pfam, DefenseFinder models, ET-Miner's PolyForm Noncommercial licence) need
  checking before anything is published.
- OQ-5 (user): the 50% thresholds in D-10 and SH-4 are defaults with no
  data behind them. Revisit them after the first sweep.

## Out of Scope

- Wet-lab validation of candidates.
- AllTheBacteria or any genome set beyond GTDB representatives (phase 2).
- Multi-GPU nodes or rented data-centre GPUs.
- SON streaming, `anchor_items`, transaction weighting.
- Gene order and strand within a candidate (itemsets are sets; synteny is
  checked by hand on the example loci).
- Biosynthetic gene clusters and secretion systems as targets (they may
  show up among candidates but are not scored or validated as such).
- Any change to the et-miner engine.

## Appendix: Interview Log

- Q1 Goal: showcase + ranked candidate list; recall of known systems is the measurable outcome, no wet lab [accepted]
- Q2 Hardware: own consumer GPUs [stated]
- Q3 Source (phase 1): GTDB species representatives, one genome per species [accepted]
- Q4 Items: Pfam HMM hits; MMseqs2 clusters (~30% identity) for proteins without a Pfam hit [accepted]
- Q5 Transaction: non-overlapping windows of W=20 genes, two tilings (offset 0 and 10), mined separately; any set spanning <=11 consecutive genes lies whole in a window of at least one tiling; within a tiling each locus counts at most once [accepted]
- Q6 Code location: separate repo depending on et-miner (+ et_miner_rust pinned via subdirectory); bio deps stay out of the engine [accepted]
- Q7 Recall definition: ground truth = DefenseFinder on the same genomes; a system type is recovered when one mined itemset holds all its mandatory families; recall only over types with >= min_count loci (non-vacuous) [accepted]
- Q8 Threshold: sweep min_count in {10, 20, 50, 100} per tiling (8 discovery runs) [stated]
- Q9 Ranking: defense-island enrichment (fraction of loci within +-10 genes of a known defense gene), after filtering out itemsets with a DefenseFinder hit (Doron et al. 2018 approach) [accepted]
- Q10 Recall target: none fixed; recall reported per threshold [stated]
- Q11 Sweep point that OOMs or times out: recorded as a data point (peak memory, failing level K); sweep continues; no SON fallback, no vocabulary shrink [accepted]
- Q12 Core families (present in >50% of genomes): excluded from discovery vocabulary; separate control run at high support with all families must recover known operons [accepted]; threshold corrected from "1% of N" to 25% of genome count in D-6 [assumed]
- Note: with no fixed recall target, the hard go/no-go is the control run; defense recall is reported only [accepted, proposed after Q12 and not objected]
- Q13 Item per protein: dominant Pfam domain (highest bit score), else MMseqs2 cluster; one item per protein; DefenseFinder genes map 1:1 to items [accepted]
- Q14 Genome QC: GTDB-supplied CheckM2 completeness >= 90% and contamination <= 5% (MIMAG high quality); dropped count reported [accepted]
- Q15 Mining correctness: efficient-apriori oracle on a ~1e5-window subsample, CLAUDE.md convention (min_support=(min_count-0.5)/N, explicit max_length), exact itemsets AND counts, hard-fail before the sweep [accepted]
- Q16 GPUs: 2x RTX 3090 (24 GB each, 48 GB total) [stated]
- Q17 Wall-clock limit: 4 h per sweep point (one tiling, one min_count) [accepted]
- Q18 Pfam annotations: reuse precomputed per-protein Pfam hits (GTDB/AnnoTree) if available for the release; annotate only missing genomes with pyhmmer [accepted]
- Q19 max_length = 11: a k-itemset spans >= k genes, and the tilings only guarantee spans <= 11, so larger sets would carry biased supports [accepted]
- Q20 Enrichment null model: global background, one-sided Fisher exact test, Benjamini-Hochberg [accepted]
- Q21 Outputs: per-K parquet per run, candidate table, protein FASTA per top candidate [accepted]
- Post-interview (recon): `prune_equal_support` emits free-sets only, so recovery uses derived supports (D-8) and candidates carry closures (SH-3) [assumed]
