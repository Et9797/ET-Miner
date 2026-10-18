# Moonshots for the GPU route — spec

Ideas beyond `../optimizations/SPEC.md` (O1–O8), each exact or out. Every
figure cites its source:

- **Phase 0** — `../results/2026-10-17-moonshots-baseline/`: 1× RTX A4000 16 GB
  (sm_86), Pentium Gold G5420 (2 cores, no AVX; Polars on its compat runtime),
  `main`'s `src/`, one GPU, median of 3. `split.md` holds the wall split;
  `microbench.jsonl` the instruction rates.
- **Phase 1** — `../results/2026-10-17-moonshots-stakes/`: CPU replays of the
  Phase 0 lattice dumps and datasets, and `stakes.py`'s bounds from them.
  The README there lists the commands.

"Estimate" marks what is a model, not a measurement. Every number is from this
box. The 2-GPU numbers quoted for M6 and M9 come from the 2× A4000 box of
`../results/2026-10-06-group-kernel/` and are marked as such. They are context
only and decide nothing.

## Ground rules (from the brief; binding for Phase 2)

- **Correctness.** `CLAUDE.md` is binding:
  - every new path gets pinned tier-chain legs on 1 and 2 GPUs, with forced
    chunks wherever chunks exist, and with count inference and free sets where
    the path serves them;
  - efficient-apriori stays the oracle, under the `(min_count - 0.5) / N`
    convention and with an explicit `max_length`.
- **Exactness argument first.** Each idea below carries a proof sketch. Phase 2
  adds a property test against brute force on random small inputs: ties at
  `min_count`, empty rows, duplicate items, `n_rows` not a multiple of 64 or
  256, all-ones words, one GPU and two.
- **Semantics.** Itemsets, counts, item dtypes and emission order are unchanged.
  So are the per-level `output_dir` files, `resume_from_k`, `max_length`,
  `anchor_items`, `bitvecs=`, the memory guards, `profile` and
  `level_callback`.
- **Kernels.** The `CLAUDE.md` constraints apply (M3 needs a rule change).
  Knobs are `ET_MINER_*` pins only, and an unserved option falls back to
  today's path by dispatch.
- **Measurement.** One `PROTOCOL-<id>.md` per idea before its first timed run,
  with the consolidation protocol's rules 1, 2, 4 and 5. Rule 2 needs a ≥ 10 %
  lower median, non-overlapping ranges and ≥ 1 s saved.

## Where the time goes (Phase 0, 1 GPU)

| regime | wall | input before K=1 | K≥2 levels | largest K≥2 parts |
|---|---|---|---|---|
| dsl | 25.21 | 18.57 | 6.56 | per-candidate 3.21, group 2.75 |
| dsl, count inference | 21.54 | 18.57 | 2.92 | group 1.56, per-candidate 0.62 |
| dsl, free sets | 21.09 | 18.42 | 2.64 | group 1.20, per-candidate 0.69 |
| sk2ml3 | 23.22 | 5.08 | 18.03 | tiled K=3 16.62 |
| oom2ml3 | 2.48 | 1.05 | 1.39 | tiled K=3 0.89 |
| skew / deepk | 1.10 / 0.88 | 1.02 / 0.79 | 0.07 / 0.07 | |
| or002 / or003 / smoke | 0.31 / 0.16 / 0.07 | ≤ 0.10 | ≤ 0.20 | |

The result frame and tail take ≤ 0.04 s everywhere. On this host the input
phases outweigh every level loop but sk2ml3's: on dsl they are COO to CSR 8.25 s,
explode and join 6.53 s, the host K=1 2.28 s, the host-to-device copies 1.02 s.

## M1 — GPU-native input layer

**Evidence** (Phase 0 span; `input-floor-*.json`). On this host the input
phases before K=1 are most of every in-core wall but sk2ml3's:

| phase | dsl | sk2ml3 |
|---|---|---|
| host K=1 (explode, group_by) | 2.28 | 1.46 |
| explode and join onto the frequent items | 6.53 | 1.53 |
| COO to CSR (scipy) | 8.25 | 1.76 |
| int64 copies of the CSR | 0.26 | 0.05 |
| host-to-device copies (pageable) | 1.02 | 0.21 |
| bitvector kernel | 0.15 | 0.08 |
| input before K=1 / wall | 18.57 / 25.21 | 5.08 / 23.22 |

The host work M1 keeps, timed on the same box:
- the exploded item ids as int32 NumPy arrays in 5 M-row chunks: 1.57 s on dsl
  (247 M entries), 0.17 s on sk2ml3;
- the row lengths: 0.07 s and 0.01 s.

Their pinned copy at the measured 13.1 GB/s costs 0.08 s on dsl. The floor
(explode + copies + today's bitvector kernel) is 1.81 s on dsl and 0.27 s on
sk2ml3.
- *Bound.* ≤ 16.8 s on dsl (66 % of the wall; 78 % with count inference and
  with free sets) and ≤ 4.8 s on sk2ml3 (21 %).
- *Small regimes.* deepk / skew / oom2ml3 ≤ 0.70 / 0.91 / 0.97 s, under rule
  2's 1 s (floors at dsl's 6.4 ns explode per entry).
- *Read.* The dataset read (8.6 s on dsl) is outside the wall and outside M1.
- *Other box.* Ryzen 5600X, 6 threads: the old box's ~10.5 s outside the levels
  on dsl is the same work. M1's share there would be smaller but still the
  largest single stake on dsl.

**Proposal.**
- *Integer item columns* (Int8–Int64, UInt8–UInt32) whose ids span ≤ 2²²
  (`cpu_miner._int_ids`'s rule):
  1. Per row chunk, Polars explodes the column, drops null items and computes
     the row lengths. The int32 ids (minus the minimum) and lengths go into
     pinned buffers and are copied asynchronously while the next chunk
     explodes.
  2. On the device, K=1 is an occurrence bincount into a 2²²-entry table
     (16 MB). The host keeps every id with ≥ min_count occurrences, today's
     filter by occurrences, so the bitvector columns are today's columns.
  3. A lookup table maps ids to columns. A scan of the lengths gives each
     entry's row, and the bitvector kernel sets the bits from (row, column)
     with `atomicOr`, as `csr_to_bitvec.cu` does, on int32 inputs.
  4. The chunks stay on the device (dsl: 1.1 GB int32) or are re-copied if they
     do not fit.
- *Row-wise K=2's CSR of frequent positions*, and its r, come from the same
  device entries: sort (row, position) keys with CuPy, unique, cut per shard.
- *Wide integer ids* map through their column ids as today (`_as_item_dtype`).
- *Other dtypes* keep today's path by dispatch. By code reading, string items
  cannot pass the GPU route today: `row_split.py` copies `col_to_item` into an
  int64 array after the bitvectors are built. That was not run here; it is a
  separate defect, not M1's to change.
- *Wild variant.* Parquet decode on the GPU (KvikIO / cuDF) would also remove
  the 8.6 s read. That is a new runtime dependency, so it needs the owner's word.

**Exactness.** The bitvectors are the same bits:
- the same rows (n_transactions counts empty and null rows as today);
- the same column set and order (ascending item, the occurrence filter);
- `atomicOr` is idempotent, so duplicate items in a row set one bit, as today.

Phase 2's test compares the device-built bitvectors byte for byte with
today's on every preset, plus null lists, null items, duplicate items, empty
rows, ids at the span limit, negative ids, UInt32 ids above 2³¹, and an
`n_rows` not a multiple of 64.

**Dispatch.** By dtype and span, with transactions input. `bitvecs=` and the
other dtypes keep today's path. `ET_MINER_INPUT_LAYER=host|device` pins it.

**Acceptance.** Identical signatures; the byte-identity test above; peak VRAM
and RSS recorded (the int64 host CSR and its copies go away). Tier legs with the
device path pinned, on 1 and 2 GPUs (the shards cut the device entries by rows),
plus row-wise K=2 on and off.

**Amendment (as built, `gpu/input_layer.py`).** Three departures from the
proposal; the exactness argument above is unchanged:
- *No explode.* Each Polars chunk's Arrow buffers are read in place (row
  offsets and values). Only a chunk with null items, or with a null list that
  holds entries, goes through Polars' explode, its nulls becoming the null id
  `span`, which the lookup table maps to no column.
- *Unsorted rows and repeated items.* `ids_to_bitvec` flags a shard whose row
  columns do not strictly ascend; its K=2 rows are then sorted and
  deduplicated on the device (`_canonicalize`), as the host CSR's
  `sum_duplicates` does.
- *VRAM.* The entries stay on the device for K=2 only when 28 B per entry
  (`KEEP_ROWS_BYTES`) fit beside the bitvectors; otherwise they are dropped
  and K=2 counts from the bitvectors. Today's path never needs that choice;
  it is logged.
  - A shard with unsorted rows or repeated items is sorted in blocks of
    `CANON_BLOCK_ENTRIES` entries, and the decision adds one block at
    `CANON_ENTRY_BYTES` per entry. Sorting every entry at once took about
    42 B per entry (the adversarial review's must-fix, `REPORT.md`).
  - The memory guards drop the kept entries before they compare, so a limit
    the bitvectors fit passes as on the host path.

## M2 — Row-wise counting beyond K=2

**Evidence** (`k3rows-*.json`). The K=3 level is enumerated exactly from each
row's frequent items. A row holds a counted candidate when all three of its
pairs are frequent pairs: that is the K=3 subset test.
- *Check.* The candidates whose summed count reaches min_count are exactly the
  K=3 level of the lattice, with equal counts (`k3_matches_lattice`).

| | sk2ml3 | oom2ml3 |
|---|---|---|
| row pairs Σ C(L, 2) (lookups) | 562,968,354 | 98,735,918 |
| row triples Σ C(L, 3) (a naive test) | 4,447,206,933 | 653,703,515 |
| increments (Σ supports of the counted candidates) | 375,181,499 | 19,074,068 |
| counted candidates / occurring in some row | 144,951,409 / 85,983,192 | 15,817,779 / 7,469,032 |
| tiled kernel today: counted slots × words | 1.20e13 slot-words | 4.98e11 |
| K=3 counting today | 16.63 s | 0.89 s |

- *Rows.* A row of sk2ml3 holds up to 3,372 counted candidates; 12,907 rows
  hold none.
- *Increments.* 375 M increments against 1.2e13 slot-words: the brief's guess
  of about 1e9 was high by 2.7×.
- *Model* (`stakes.py`). Each row pair and each increment at the regime's own
  row-wise K=2 rate: 0.49 ns per row pair on sk2ml3 (0.276 s for 563 M row
  pairs), 0.53 ns on oom2ml3. That gives 0.46 s and 0.06 s.
- *Bound.* ≤ 16.2 s of sk2ml3's 23.22 (70 %), ≤ 0.83 s of oom2ml3's 2.48 (33 %,
  below rule 2's 1 s).
- *K≥4 on dsl, deepk, skew and or002.* The rows hold C(active items, k)
  subsets: 1.5–6.5 × the counted candidates × words at the best level, and up to
  7e13 subsets per level on dsl and 5e12 on or002. `U_k` (the smallest-parent bound) is
  above the dense work at every dsl level. Row-wise counting has no stake there.

**Proposal (K=3).** One warp per row, on the row-wise K=2's frequent-position
CSR (already built for K=2).
- Stage the row's positions (≤ 64) in shared memory.
- Build the row's frequent-pair mask: a 64-bit neighbour mask per position, from
  C(L, 2) lookups into the K=2 level (a binary search in the prefix item's sorted
  successor list, i.e. the K=3 group's suffixes).
- For each frequent pair (i < j) in the row, the triangles are
  `popc(nbr[i] & nbr[j] & above(j))`. Each set bit k is a counted candidate (a, b, c).
- Its slot in a compact int32 array over the counted candidates is
  `offset[pair(a, b)] + rank(c)`: offsets per frequent pair (1.66 M + 1 int64)
  over the sorted c lists (145 M int32, 580 MB on sk2ml3), the triangles of
  the frequent-pair graph built on the device after K=2.
- `atomicAdd(slot, 1)`. Hot candidates (the most popular triples) are
  privatized in shared memory, as `count_pairs_k2_rows_shared` does.
- *Decode.* The survivors decode from (pair, c) directly. The level-end lexsort
  restores today's order, which runs whenever `prune_apriori` is on.
- *Multi-GPU.* Each shard counts its rows into the same compact layout, and the
  compact arrays reduce as today (M9 applies).
- *Larger levels.* A level whose compact array exceeds the chunk budget is
  counted by ranges of the first pair, each row enumerating only the pairs in
  the range (the K=2 row kernel's b-range scheme).

**Exactness.** support(X) = Σ_rows [X ⊆ row]. A row adds 1 to exactly the
counted candidates it contains: each triangle among its unique frequent
positions, once, by the i < j < k enumeration. Infrequent items cannot appear
in a frequent itemset, so dropping them changes no counted support. A candidate
outside the counted set is not counted, exactly as the subset test skips it
today.

**Dispatch.** Rule 4, from facts known before K=3: the row pairs and the
increments bound (U_3, or Σ_rows C(L,3) as a looser one), against the tiled
kernel's counted slots × words from the census of the K=2 survivors. A swept
crossover in the style of `k2_crossover.py` fixes it. `prune_apriori=False`
counts every generated candidate and keeps the dense kernels; so does
`bitvecs=` input (no rows).

**Acceptance.** Identical signatures. Tier legs with row-wise K=3 pinned
(`ET_MINER_K3_KERNEL=rows`), on 1 and 2 GPUs (plain, forced chunks, count
inference, free sets).

**Amendment (as built, `gpu/kernels/k3_rows.py`).**
- *Slots by pair, not by triangle.* Pair p = (u, v) owns one slot per frequent
  partner w > v of v, at `slot_off[p]` + the rank of w among v's partners. That
  rank, like the pair's own index, is one word of a frequent-pair bitmap plus
  a per-word rank table, so no lookup searches a triangle list. On sk2ml3 the
  layout holds 151.1 M slots against 145.0 M counted candidates; slots whose
  (u, w) is not frequent stay 0. The triangle list measured 1.1 s of binary
  searches on sk2ml3; the slot rank removes them (kernel 1.56 → 0.39 s).
- *Work split.* The row's frequent pairs (i, j), not its positions, are
  spread over the warp's lanes.
- *Bitmap.* Built on the device, m × ⌈m/64⌉ words and an int32 rank per word
  (12 B per 64 pairs of active items), up to 512 MB; above that the kernel
  searches the neighbour lists.
- *Dispatch.* Rows when `r3 = Σ_rows C(L, 2) / (generated × words)` is below
  `K3_ROWS_MAX_R` and the slots fit one chunk. The active rows' pairs come from
  the row source (`DeviceRows.pairs` or `_HostRows.pairs`) before the level
  runs, and the crossover is swept by `k3_crossover.py`. In a dense
  frequent-pair graph the slot count approaches the generated candidates, so
  a chunked slot space would re-read every row per chunk.
- Rows of more than 64 active items, `prune_apriori=False`, `bitvecs=` input,
  an ESCO level and `resume_from_k` at K=3 count from the bitvecs. A property test against brute force: duplicate items,
empty rows, ties at min_count, rows of 64 frequent items, one GPU and two.

## M3 — Binary tensor cores for the tiled family

**Evidence.**
- *Rates* (`microbench.jsonl`). `mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc`
  runs at 9.51e12 slot-words/s, against 5.82e11/s for `__popcll`: 16.3× (113.9
  against 7.0 per SM per clock). A slot-word is one pair over 64 rows.
  - The tiled kernel's inner loop, replayed on dense shared tiles, runs at
    5.93e11 slot-words/s: it already sits on the popcount roof.
  - The m16n8k256 product equals `__popcll` on all 36 exactness cases.
- *Where the popcount is.* The registered kernels on zero data (staging only)
  against dense data:
  - tiled at K=3, 64 suffixes, 31,250 words: 0.756 against 2.324 ns per
    tile-pair per word;
  - group at K=8, 16 suffixes, 312,500 words: 0.0036 against 0.0063 ns per pair
    per word.

  sk2ml3's tiled K=3 level runs at 1.41 ns per counted tile-pair per word, so at
  most 46 % of it is above the staging-only time.
- *Bound.* The popcount part at 1/16.3 of its cost (`stakes.py`):
  - sk2ml3: ≤ 7.26 s of 23.22 (16.62 s tiled; 31 %);
  - oom2ml3: ≤ 0.49 s;
  - dsl: ≤ 1.21 s (group 2.75 s at its dense compute share of 42 %; 4.8 %);
  - dsl with count inference: ≤ 0.74 s.

**Proposal.** Inline PTX in the tiled and group kernels, behind a per-device
switch.
- *Tiled kernel.* Each 32-word staged tile is 8 k-steps of 256 bits, and the
  32×32 tile-pair is 2 × 4 m16n8 output tiles: one per warp, 8 MMAs per warp
  per word tile.
  - The prefix AND is applied once per staged suffix row of tile B.
  - Fragments come from the existing shared-memory tiles, with no layout change:
    a0/a1 and b0 read 32-bit word q of the chunk, a2/a3 and b1 word q + 4, rows
    g and g + 8 (the layout `microbench.py` checks).
  - Padding rows and words are zero.
- *Group kernel.* The staged rows of a group (≤ 64) as one 64×64 product: 4 × 8
  MMA tiles. Only the listed pairs are written; the others are computed and
  dropped, which costs nothing measurable at 16× the rate.
- *Hopper.* `mma.sync` b1 is in the PTX ISA for sm_80+, so sm_90 should compile
  it under NVRTC without arch flags. Its b1 throughput on Hopper is not
  measured. `wgmma` b1 needs sm_90a, which NVRTC without arch flags does not
  target, so it stays out.
- *Control arm, no rule change.* A Harley–Seal carry-save accumulator in the
  tiled inner loop, with one popcount per 8 words per pair (LOP3 runs at 55
  per SM per clock against 7 for the 64-bit popcount).

**Exactness.** AND then popcount, summed in s32. A pair's count is
popcount(A_r & B_c) summed over the k positions, and that sum does not depend
on bit order. A and B use the same packing for the same k, and padding is zero.
Counts are ≤ n_rows < 2³¹, so s32 cannot overflow.

**Dispatch.** Per device, at first use:
- compute capability ≥ 8.0, and a known-answer launch (the `bmma_exact` cases)
  equal to the popcount path;
- otherwise the popcount kernels.

`ET_MINER_TILED_MMA=0|1` pins it. No regime-level crossover: the same kernel,
faster per slot.

**Gate.** `CLAUDE.md` allows intrinsics that are "sm_60+ and NVRTC-compilable
with no arch flags". b1 MMA is sm_80+. Both targets (sm_86, sm_90) qualify,
but the rule's text does not. **Stop before the kernel enters `_src`** for the
owner's rule change.

**Acceptance.** Identical signatures. Tier legs with the MMA path pinned on and
off, on 1 and 2 GPUs (plain, forced chunks, count inference). A property test of
the MMA tile product against `__popcll` on ragged random tiles. `selfcheck.py`
launches both paths. sm_90 stays an open risk until a Hopper box runs the
selfcheck.

**Amendment: dropped after M2 (Phase 3).** The tiled family it would speed up
spends at most 0.12 s in any regime of the matrix after M2 (dsl, one GPU; M5b's
`lex` arm): sk2ml3's and oom2ml3's K=3 count row-wise. No stake, so no kernel
enters `_src` and the CLAUDE.md intrinsic rule stays as it is.

## M4 — Lattice finisher: a zeta transform per item component

**Evidence** (`components-*.jsonl`).
- *Components.* No regime of the standard matrix has a starting level K0 at
  which every component is small. Before level K0, the items of each frequent
  (K0−1)-itemset are joined.
  - dsl: one component of 80 items at K0=3, 63 at K0=5–7, 52–53 at K0=11–13,
    and 34 or fewer only from K0=14 (572 itemsets left).
  - deepk: 41 items from K0=7.
  - skew: 31 items at K0=9 (4 itemsets left).
  - Online Retail: one component of hundreds of items until its last levels.
  - The popular background items join the motifs. On dsl the top item is in
    81.9 % of the rows, so its unions with every motif tier stay frequent.
- *The per-family variant.* Group the frequent (K0−1)-itemsets by their
  (K0−2)-prefix P, with E(P) the extending items. The tables fit from dsl's
  K0=7 (largest E(P): 22 items, 16 MB), but the conditional histograms read
  Σ count(P) × |E(P)| = 8.0e10 row-items at K0=7, 5.2e10 at K0=10 and 1.3e10 at
  K0=12. That is more than the level loop reads for those levels, and dsl's
  levels from K0=12 cost 0.31 s.
- *Stake on the standard matrix.* 0 s.

**Amendment: dropped for the current matrix.** No regime of the standard
matrix has a stake. Re-open with the proposed preset below, or with AlphaFold.

**Target regime and proposed preset.** M4's case is the AlphaFold base-vocab
regime (~1,000 items, K up to 22), which no preset reproduces: there the deep
itemsets are long motifs that no hub item joins. Preset `deep_k20`
(added to `PRESETS` on the owner's word, 2026-10-17):

```python
"deep_k20": SynthSpec(
    name="deep_k20", n_rows=1_000_000, vocab_size=4_000, zipf_a=1.1,
    row_len_mean=10, row_len_max=48,
    motif_count=2, motif_size=20, motif_penetration=0.011,
    min_support=0.01, seed=2020,
)
```

- *Lattice.* Two 20-item motifs planted in 11,000 rows each (min_count
  10,000), so 2 × 2²⁰ frequent itemsets reach K=20.
- *No hub.* The most popular background item is in 82.7 % of the rows
  (`estimate_level_sizes`), so a motif subset joined with it holds
  ≈ 11,000 × 0.827 = 9,097 < 10,000 rows. No background item joins a motif.
- *Components.* Background itemsets stop near K=5 (the top items' product
  falls below 1 % at the sixth item). From K0 ≈ 6 the components are the two
  motifs, 20 items each: 4 MB tables, a histogram over ~22,000 rows.

Its `check_preset_purpose` clause:
- `motif_size >= 18`;
- planted count ≥ min_count (the existing vacuity guard);
- `motif_penetration × max_item_frequency < min_support`, with the frequency
  from `estimate_level_sizes`' p_item, so no background item can join a motif's
  deep itemsets;
- `motif_count × 2**motif_size <= 4_000_000`, so the result frame stays bounded.

*Checked on the CPU* (the spec generated in memory, mined with
`apriori(use_gpu=False)` in 119.5 s on this box; script in the Phase 1 README):
- 2,097,793 frequent itemsets up to K=20. Per level from K=6 they are exactly
  2 × C(20, k): no background item joins a motif.
- Components: at K0=5, 20, 20 and 18 items (a background component); from K0=6,
  the two motifs alone (2 × 4 MB tables).
- The finisher from K0=6 would emit 2,053,752 of the 2,097,793 itemsets.

The GPU level loop's time on it is not measured (Phase 1 is CPU only). It is
the first thing a Phase 2 protocol for M4 measures.

**Proposal** (for that regime).
1. Before level K0 ≥ 3, compute the components (union-find over level K0−1).
2. When every component with ≥ K0 items has ≤ C_MAX items (2^C_MAX int32
   within the budget; 26 → 256 MB):
   - for each component, histogram the rows projected onto its items (transpose
     64-row words into row masks; skip rows with fewer than K0 component items);
   - run a superset-sum zeta transform (n_c · 2^n_c adds);
   - emit every subset S with f(S) ≥ min_count and K0 ≤ |S| ≤ max_length.
3. Emit level by level, in K order and today's sort order. Each level goes
   through the flush and `level_callback` as if mined, with `n_candidates` =
   the level's emitted itemsets (documented), and one `profile` phase per
   level.
4. `resume_from_k` ≥ K0 recomputes the components from the loaded level.
5. Options:
   - free sets read freeness from the same table (S is free iff no S−x has
     f(S−x) = f(S));
   - count inference is moot;
   - anchors stay an output selector;
   - multi-GPU sums the histograms.
6. Components too big fall back to the level loop.

**Exactness.** f(S) is the number of rows ⊇ S. Every frequent itemset of size
≥ K0 lies in one component, so the emitted set is exactly Apriori's levels
from K0.

**Dispatch.** Component sizes and the histogram's rows, known before K0.
`ET_MINER_FINISHER=0|1` pins it.

**Acceptance.** Identical signatures on the new preset and every existing one
(where it does not engage). Tier legs with the finisher forced at small K0 on
smoke (its components are small from K0=6: 16 items). A property test against
brute force, including `max_length`, free sets, anchors and resume.

## M5 — Shrink the row space

### (a) Trimming

**Evidence** (`replay-*.jsonl`, `rows`). The share of dsl's rows that hold at
least k of level k−1's items:

| K | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|
| kept | 79.5 % | 77.0 % | 73.4 % | 65.4 % | 53.8 % | 47.6 % | 41.6 % | 35.8 % |

- *Bound.* Each level's counting × the share of rows dropped, with free
  rebuilds at every level:
  - ≤ 2.34 s on dsl (9.3 %);
  - 0.65 s with count inference;
  - 0.54 s with free sets.
- *At the CPU miner's rule* (rebuild once ≤ half the rows remain): 0.90 /
  0.05 / 0.03 s.
- *Elsewhere.* The other regimes keep ≥ 90 % of the rows on the levels that
  cost anything (≤ 0.01 s).

**Amendment (a): dropped on today's walls.** The bound, 2.34 s on dsl, is 9.3 %
of its 25.21 s wall, below rule 2's 10 % (2.52 s), and no other regime comes
close. After M1 the same bound would be about 28 % of dsl's wall (8.4 s).
Re-open then, measured against the post-M1 tree.

### (b) Reordering

**Evidence** (`tiles-dsl.jsonl`, a 1,000-group sample per level). The share
of a group's 2,048-row tiles whose prefix AND is zero, weighted by the
group's pairs:

| row order | K=3 | K=5 | K=7 | K=8 | K=10 | K=13 |
|---|---|---|---|---|---|---|
| dataset order | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| lex (descending-support items) | 56.3 % | 87.1 % | 93.0 % | 93.5 % | 95.2 % | 97.1 % |
| row length, then lex | 50.5 % | 83.5 % | 90.6 % | 91.5 % | 93.6 % | 95.9 % |

In the dataset's order no tile is empty: dsl's deep itemsets hold 1–2 % of
the rows, spread over every 2,048-row tile.

*Bound.* Each level's counting × its zero-tile share, assuming a skipped tile
costs nothing:
- dsl: ≤ 5.58 s (22 %), 2.01 s with count inference, 1.74 s with free sets;
- sk2ml3: the census's staged row tiles drop 20.8 % under lex, ≤ 1.85 s at the
  tiled kernel's 54 % staging share;
- oom2ml3: ≤ 0.14 s.

*Finding zero tiles without computing the prefix AND.* Under lex, the AND
of the prefix items' per-item tile bits (`zero_by_item_occupancy`) finds
almost every zero tile, without computing the prefix AND: 80.6 % at K=4,
86.8 % at K=5, 92.6 % at K=7, 93.2 % at K=8 and 94.9 % at K=10.

**Proposal.**
- *(a)* Before level k, flag the rows with fewer than k of level k−1's items.
  Once the kept rows fall below a swept share, compact the active columns'
  bitvectors over the kept rows, from M1's device CSR (or from the bitvectors
  by a gather of words). `min_count` stays computed from the original N.
- *(b)* Permute the rows once after K=1, before the bitvectors are built:
  1. a lexicographic sort on each row's frequent items in descending-support
     order, as two uint64 keys per row (85 frequent items on dsl);
  2. build the bitvectors in the new order;
  3. give the kernels per-item tile-occupancy bits (columns × tiles bits: 104 KB
     on dsl);
  4. per staged word tile, skip the tile when the AND of the prefix items'
     occupancy bits is zero, before staging any row;
  5. in the per-candidate kernel, skip the word block the same way.

  Applies to transactions input only; `bitvecs=` stays read-only.

**Exactness.**
- *(a)* A dropped row holds fewer than k of the items that can appear in a
  k-candidate, so it holds no k-candidate. The active items only shrink with k,
  so it holds no later candidate either.
- *(b)* Counts are sums over rows and do not depend on the row order. A
  skipped tile has a zero prefix AND, so it contributes zero to every pair of
  its group. Nothing outside the counts sees rows: emission, flush and resume
  are row-free.

**Dispatch.** (a): kept share below a swept threshold. (b): the zero-tile share
of the K=2 survivors' items, estimated before K=3 from the occupancy bits; an
order is applied only where it beats the dataset order by a swept margin.
`ET_MINER_ROW_ORDER=keep|lex` and `ET_MINER_ROW_TRIM=0|1` pin them.

**Acceptance.** Identical signatures. Tier legs with each pinned, on 1 and 2
GPUs (the permutation is global, before the row split), forced chunks, count
inference, free sets, and the row-wise K=2 (its CSR is permuted too). A
property test that permuted, tile-skipping counts equal brute force.

## M6 — Replicate and split instead of row split and reduce

**Evidence.** Two GPUs only, so nothing was measured on this box. Context from
the 2× A4000 box (`../results/2026-10-06-group-kernel/final.jsonl`, final check,
one rep):

| regime (other box) | 1 GPU | 2 GPUs | 2-GPU K≥2 parts |
|---|---|---|---|
| sk2ml3 | 20.34 | 16.06 | tiled 10.22, reduce 1.44, compact 0.53 (1 GPU: tiled 16.33) |
| oom2ml3 | 1.81 | 2.64 | tiled 0.58, reduce 0.76, compact 0.10 (1 GPU: tiled 0.88) |
| dsl | 16.75 | 13.40 | per-candidate 1.55, group 1.35 (1 GPU: 3.08, 2.66) |

- The tiled K=3 level scales 1.6× on two GPUs, not 2×: every GPU classifies all
  11.8 B generated candidates and skips the same 11.2 M tile-pairs.
- Splitting the counted work instead would cost about half the 1-GPU tiled time
  per GPU (8.2 s on sk2ml3, 0.44 s on oom2ml3), with no reduce and no compact.
  Estimated saving on that box: about 4.0 s of 16.06 on sk2ml3 and about 1.0 s
  of 2.64 on oom2ml3; nothing on dsl, whose kernels already split.
- DP6, which chose the row split, measured strategy and kernel together. Its
  split arms ran the legacy fused wrappers:
  - at `e4bb3ae`, `gpu/dispatch.py` sent K=2 to `count_pairs_fused_k2_multi_gpu`
    and K≥3 to `count_k3plus_fully_fused_multi_gpu` whatever the variant;
  - the K=2 wrapper copied the matrix to every GPU through host RAM
    (`bitvecs_gpu.get()`);
  - both predate the subset test and the row-wise K=2.

**Proposal.** When each device fits the frequent columns' bitvectors next to
its level budget (the 1-GPU fit check, per device), build the full bitvectors
on every device from the CSR: one upload per device, no host round trip of the
matrix.
- Each level's candidate space is cut into interleaved group-aligned ranges, or
  handed out from a dynamic queue of chunks.
- Each GPU runs today's one-GPU level on its ranges: classification, counting
  and filter.
- The survivors are concatenated in candidate order on the host.
- Row-wise K=2 reads every row on every GPU and counts that GPU's range of b.
- Data that does not fit keeps the row split.

**Exactness.** Every candidate is counted once, over all rows, by one device.
The ranges partition the candidate space, and survivors keep their global
index, so decode and order are unchanged.

**Dispatch.** Rule 4, from facts known before the level: the bitvector bytes
against each device's free VRAM. Also, from the census, the counted work's
balance per range.

**Acceptance.** Identical signatures. 2-GPU tier legs (plain, forced chunks,
count inference, free sets, row-wise K=2). Measurable on a 2-GPU box only.

**Amendment: dropped after M2 (Phase 3).** M6 split the tiled K=3 level of
sk2ml3 and oom2ml3 across GPUs. After M2 that level counts row-wise (sk2ml3's
K=3 is 0.6 s on one GPU). The stake is now the 2-GPU K≥2 levels minus half
of the 1-GPU ones, from M5b's `lex` arm (`../results/2026-10-17-moonshots-m5b/`):
- sk2ml3 1.40 − 1.39 / 2 = 0.70 s;
- dsl 1.74 − 2.54 / 2 = 0.47 s;
- oom2ml3 0.53 − 0.25 / 2 = 0.40 s;
- under 0.15 s elsewhere.

No regime reaches rule 2's 1 s, and replication would add a second full
bitvector build. Not built.

## M7 — Deduction rules as count inference

**Evidence** (`replay-dsl.jsonl`, every countable candidate of every level).
On dsl, the counted candidates that count inference does not settle (no
non-free subset) are settled by the non-derivable-itemset bounds of depth ≤ 3
as follows. An upper bound below min_count never occurred. Every settled
candidate was frequent, and its l = u equals its true count (checked).

| K | counted with inference | settled by the bounds | share |
|---|---|---|---|
| 5 | 41,583 | 372 | 0.9 % |
| 6 | 53,082 | 38,104 | 71.8 % |
| 7 | 44,635 | 40,635 | 91.0 % |
| 8 | 26,372 | 26,372 | 100 % |
| 9 | 11,800 | 11,800 | 100 % |
| 10 | 2,500 | 2,500 | 100 % |

- *Stake.* Each level's counting under count inference, times its settled
  share: ≤ 1.65 s of dsl-infer's 21.54 s on this box (the bounds' own cost is not
  subtracted).
  - That is 7.7 % of the wall, below rule 2's 10 %: the input phases are 86 %
    of the wall here.
  - After M1 the same 1.65 s is about 34 % of the wall (21.54 − 16.76 = 4.78 s).
- *Other regimes.* or002 / or003 / deepk / skew: ≤ 0.004 s.
- *Brute force.* The bounds held on all 673 countable candidates of smoke K=3–5
  (no bound violated).

**Proposal.** In the count-inference path (`use_generator_pruning=True`), a
counted candidate X gets bounds from its subsets of size k−1, k−2 and k−3:
- `u = min( min_a σ(X−a), min_{a<b<c} [Σ σ(X−a) − Σ σ(X−a−b) + σ(X−a−b−c)] )`;
- `l = max( 0, max_{a<b} [σ(X−a) + σ(X−b) − σ(X−a−b)] )`.

Then:
- if `u < min_count` the candidate is skipped (it is infrequent);
- if `l = u` it takes `u` without counting, written by the first device only
  (as inferred counts are today);
- otherwise it is counted.

The device needs indexes of levels k−1, k−2 and k−3 with counts.
- Today's index holds level k−1. Keeping two more is a few hundred MB at dsl's
  sizes (levels 6–9 hold 88–115 K itemsets).
- `_classify_candidate` grows a third outcome (`CAND_BOUND`).
- `σ(∅) = N` enters only at K=3.

Depth 2 alone (no level k−3) is not enough (`replay-dsl-ndi2.jsonl`): it
settles 118 and 217 candidates at K=5 and K=6 and none above, against 119,783
with depth 3. The depth-3 rule is what settles dsl's levels, so the index of
level k−3 is required.

**Exactness.** Calders & Goethals (2002): for every Y ⊆ X,
σ(X) ≤ Σ_{Y⊆J⊂X} (−1)^{|X∖J|+1} σ(J) when |X∖Y| is odd and ≥ when it is even.
These are inclusion–exclusion inequalities on the rows that hold Y, so
`l ≤ σ(X) ≤ u` holds exactly for integer counts. Equal bounds fix σ(X); an
upper bound below min_count proves the candidate infrequent. Pascal inference
is the case of a non-free subset (depth 2 with σ(X−a−b) = σ(X−a)).

**Dispatch.** Only with count inference, on levels with an index of k−1, k−2
and k−3 within the index memory cap (today's quarter of free VRAM). Below that,
today's Pascal-only test. Free-set runs keep today's path (count inference is
complete-lattice only).

**Amendment: dropped on today's walls.** 1.65 s is 7.7 % of dsl-infer's
21.54 s, below rule 2's 10 %. No other regime has a stake. After M1 it is
about 34 % of the wall. Re-open then; the design above stands.

**Amendment (re-opened after M1, as built).** dsl with count inference is
5.8 s on this tree, so M7 is re-opened (`PROTOCOL-M7.md`).
- `_classify_candidate` keeps the k counts of the (k−1)-subsets. When
  inference leaves a candidate of at most 24 items, it calls `_deduce`
  (`__noinline__`, so the counting kernels' registers stay as they were).
- `_deduce` reads levels k−2 and k−3 from the index (`deep2_*`,
  `deep3_*`) and the row count for the empty set at K=3. It computes the
  lower bound from every pair and the upper bound from every triple, and
  stops early once the bounds meet or the upper bound falls below
  min_count.
- `row_split` keeps the complete levels of the last three K
  (`_complete_levels`) in complete-lattice runs with inference.
- `ET_MINER_DEDUCTION=0|1` pins it; unset, it is on wherever the levels
  exist and the three indexes fit the cap.
- The kernels' parameter lists grow by the deep levels through two macros
  (`SUBSET_DEEP_PARAMS` / `SUBSET_DEEP_ARGS`).

**Acceptance.** Identical signatures. Tier legs with count inference and the
bounds pinned (`ET_MINER_DEDUCTION=0|1`), on 1 and 2 GPUs (plain, forced
chunks, each kernel pinned). A property test against brute force: ties at
min_count, `l = u` below min_count, `u < min_count`, and all-ones and empty
rows.

## M8 — Two-level look-ahead in the group kernel

**Evidence** (`replay-dsl.jsonl`). Every level-(k+1) candidate P∪{a,b,c} has
a, b and c among the suffixes of the level-k group P. So the triples of each
group's suffixes cover the whole next level.
- *Cost.* The triples outnumber the next level's generated candidates 2.5–2.7×
  on dsl's big levels: K=6 groups hold 465,219 triples against K=7's 181,999
  generated; K=7 434,625 against 171,479; K=8 320,767 against 125,178.
- *Bound* (`stakes.py`). The next level's counting, minus the triples'
  popcounts at the group kernel's dense ALU rate (dense minus zero data,
  0.0027 ns per pair-word):
  - ≤ 3.99 s on dsl (15.8 % of 25.21);
  - ≤ 0.38 s with count inference, where the next levels count much less.
- *Not counted in the bound.*
  - The triples' results must be kept until the next level (8 bytes each, up
    to 465 K per level: small).
  - The staged rows must stay staged for C(m, 3) work. That is ALU, while
    today's next level is mostly staging (2.8e-11 s per staged row-word against
    3.0e-12 per pair-word in the dsl cost fit).

**Proposal.** In the group kernel, while a word tile is staged, also accumulate
popcount(prefix AND & r_a & r_b & r_c) for the triples a < b < c of the used
suffixes.
- Store them per group, indexed by (a, b, c) rank.
- After level k, the next level's candidates whose generating pairs survived
  take their count from the stored triples.
- The subset test still decides which of them are counted at all.
- Dispatch on the previous level's pass rate.

**Exactness.** A triple's popcount over all words is the exact support of
P∪{a,b,c}. The next level takes the stored value only for its own candidates,
so every emitted count is a full count.

**Dispatch.** Rule 4: the previous level's pass rate and C(m,3)/C(m,2) per
group, against a swept crossover.

**Acceptance.** Identical signatures. Tier legs with the look-ahead pinned on 1
and 2 GPUs (plain, forced chunks, count inference). A property test that the
stored triple counts equal brute force.

**Note.** M7 and count inference remove most of what M8 would save: on dsl
with inference the bound is 0.38 s. M8 pays only on runs without inference,
which X0 could make rare.

## M9 — Pigeonhole reduce (row split only)

**Evidence** (`k3rows-*.json`, the miner's 2-GPU row cuts, exact per-shard
counts). On two GPUs a frequent candidate reaches ⌈min_count / 2⌉ on some shard.
The written entries (counted candidates) with some shard at or above that bound:

| | sk2ml3 (bound 15) | oom2ml3 (bound 8) |
|---|---|---|
| written entries | 144,951,409 | 15,817,779 |
| flagged | 1,727,429 (1.19 %) | 139,531 (0.88 %) |
| frequent | 1,310,438 | 97,198 |

- *Saving.* The reduce would move 1.2 % of today's compacted entries, plus one
  bit per written entry (18 MB on sk2ml3).
- *Context* (other box, 2 GPUs). The compacted reduce and compact took
  1.44 + 0.53 s on sk2ml3 and 0.76 + 0.10 s on oom2ml3. At most about 1.9 s and
  0.8 s of those would remain to save.

**Proposal.** After counting, each GPU sets one bit per written entry: the
local count ≥ ⌈min_count / n⌉, or the entry is inferred. The bitmasks are
OR-reduced (an NCCL bitwise-or all-reduce, or the staged path). Then only the
flagged entries' counts are gathered and summed, in the compacted order; the
others are infrequent (see Exactness) and never cross the bus.

**Exactness.** If every shard count is ≤ ⌈m/n⌉ − 1, the sum is ≤ n(⌈m/n⌉ − 1) < m,
so the entry is infrequent. Inferred entries are flagged unconditionally. The
filter then runs on exact sums of the flagged entries.

**Dispatch.** Row split only, n ≥ 2, compacted levels. If M6 wins, M9 serves
only data too large to replicate.

**Acceptance.** Identical signatures. 2-GPU tier legs (plain, forced chunks,
count inference). A property test of the pigeonhole filter against the dense
reduce on random shard counts with ties at the bound.

**Amendment: dropped after M2 (Phase 3).** The compacted reduce it shrinks
was sk2ml3's and oom2ml3's dense K=3 level. After M2 the largest 2-GPU reduce
plus compact in the matrix (M5b's `lex` arm) is:
- sk2ml3 0.37 + 0.00 s (the row-wise K=2 and K=3 slot arrays);
- oom2ml3 0.27 s;
- dsl 0.09 + 0.02 s.

That stake is below 1 s everywhere, so M9 is not built.

## M10 — Item order for generation

**Evidence** (`census-*.jsonl`, exact K=3 census per order; `replay-dsl-order-*.jsonl`).
The column order is ascending item id. On the synthetic presets that is
descending support (ids are Zipf ranks), the worst orientation of the
frequent-pair graph: the most popular item's group holds all its frequent
partners.

| sk2ml3 K=3 | orig (ids) | ascending support | ascending degree |
|---|---|---|---|
| generated candidates | 11,796,796,856 | 145,203,574 | 145,202,379 |
| prefix groups / largest | 784 / 34,999 suffixes | 34,998 / 560 | 34,998 / 560 |
| counted tile-pairs (tiled groups) | 376,298 | 167,136 | 167,184 |
| pairs on the group kernel | 5,210 | 9,293,981 | 9,293,981 |
| counted candidates | 144,951,409 | same | same |

- *sk2ml3 model.* Each kernel at today's per-unit rate on this box (tiled 1.41
  ns per tile-pair-word; group kernel fit 2.8e-11 s per staged row-word and
  3.0e-12 per pair-word). The K=3 counting goes from 16.63 to an estimated 8.83
  s: ≤ 7.79 s of 23.22 (33.6 %).
- *oom2ml3.* 2.08 B → 15.9 M generated; ≤ 0.51 s.
- *dsl.* Neither order helps: the cost model gives 5.88 s today, 5.92
  (ascending) and 5.79 (descending).

**Proposal.** Relabel the frequent columns by ascending K=1 support (ties by
item id) before K=2.
- Generation, the subset index and the kernels run on the new labels.
- At each level end, the emitted itemsets map back to the canonical columns and
  are lexsorted there. Today's level-end sort already runs whenever
  `prune_apriori` or K=2.
- `col_to_item`, the flush and resume see the canonical order.
- Row-wise K=2's positions and r are order-invariant.

**Exactness.** The frequent set does not depend on item order. The subset test
admits exactly the candidates whose (k−1)-subsets are all frequent, a set
property. Counts are counts. Emission is re-sorted into today's canonical
order.
- Defect d01 (emission order) gets a test that compares the emitted order with
  today's for every option.
- `prune_apriori=False` emits each level in decode order, not lexsorted. M10
  must reproduce that order or fall back by dispatch there. **Fall back**: that
  path exists for measurement only.

**Dispatch.** Rule 4, from facts known before K=3: Σ C(out-degree, 2) of the
frequent-pair graph under each order (generated candidates) and the counted
tile-pairs, computable on the device from the K=2 survivors.

**Acceptance.** Identical signatures and emission order. Tier legs with the
order pinned (`ET_MINER_ITEM_ORDER=ids|support`), on 1 and 2 GPUs (plain, forced
chunks, count inference, free sets, `resume_from_k`, `output_dir`).

## M11 — Weighted row deduplication (stake only)

**Evidence** (`dedup-*.jsonl`). Rows projected onto the frequent items, or onto
a level's active items, and the largest multiplicity W (bit_length(W) weight
planes):

| workload | rows → distinct (frequent items) | ratio | planes | ratio / planes |
|---|---|---|---|---|
| Online Retail (or002) | 36,306 → 35,646 | 1.02 | 5 | 0.20 |
| Online Retail (or003) | 36,227 → 35,479 | 1.02 | 5 | 0.20 |
| dsl | 19,972,584 → 3,319,730 | 6.0 | 18 | 0.33 |
| dsl, level 9 / 10 / 13 active items | | 71 / 88 / 127 | 17 / 17 / 16 | 4.2 / 5.2 / 7.9 |

**Amendment: not built.** The brief's condition is ratio / planes ≥ 4 on a
real workload, and Online Retail II reaches 0.20–0.25 at every level.
AlphaFold was not available (`ET_BENCH_ALPHAFOLD` unset). The synthetic dsl
passes only from level 9, where count inference already settles most counts.
Re-open with an AlphaFold number.

## M12 — Word-tile occupancy skip without reordering (added)

**Evidence** (`tiles-*.jsonl`, `orig`). In the dataset's order, per-item tile
occupancy finds few zero tiles:
- sk2ml3's counted tile-pairs could skip 3.6 % of their staged row tiles
  (prefix tiles 0.0 %, suffix rows 3.6 %): ≤ 0.32 s;
- oom2ml3 2.9 %: ≤ 0.01 s;
- dsl: none.

**Amendment: dropped.** M12 cannot meet rule 2 in any regime (≤ 0.32 s). Its
mechanism is M5(b)'s step 4, and it is kept only as part of M5(b).

## X0 — Count inference by default (quick)

**Evidence.** dsl 25.21 → 21.54 s with `use_generator_pruning=True`, identical
output (one signature group). The other regimes were not measured with
inference in Phase 0. The pruning campaign found ties there (or002 +0.07 s).

**Proposal.** Measure the whole standard matrix with and without inference
(about 0.1 GPU-hours on this box) and show the table. Flip the default only on
the owner's word.
