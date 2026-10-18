# M5b — Row reordering and the tile-occupancy skip: measurement protocol

The design is `SPEC.md` § M5 (b). M5 (a), trimming, was dropped in Phase 1.
The code:
- `gpu/input_layer.py` `_row_order` and the permuted `ids_to_bitvec`, plus
  `row_order_keys` in `_src/csr_to_bitvec.cu`;
- `gpu/kernels/occupancy.py`;
- the occupancy test in `_src/k3plus_dense.cu` and `_src/group_pairs.cu`;
- `row_split.py`'s `_occ`.

It is on `perf/moonshot-m5b`, which stacks on M7, X0, M2 and M1. Fixed before
the first timed run.

## As built

- *Permutation per shard, not global.* After K=1 each shard's rows are sorted
  by two 64-bit keys: the row's frequent items of rank < 128, ranked by
  descending occurrences, each a bit with rank 0 as the top bit. The
  bitvectors are built in that order. Counts are sums over rows, so a
  permutation within a shard changes none. A global permutation would move
  rows between shards for no exactness gain.
- *Occupancy.* Per column, one bit per tile of 32 words (2,048 rows), built
  from the bitvectors after they exist.
  - The per-candidate kernel skips a warp's tile when one of the candidate's
    items does not occupy it.
  - The group kernel skips a tile when one of the prefix items does not.
  - The tiled kernel and K=2 are unchanged: dsl spends 0.12 s and 0.11 s
    there.
- *Scope.* Only the device input layer reorders. The host CSR and `bitvecs=`
  keep their order and get no occupancy, so their kernels run as today. The
  row-wise K=2 and K=3 rows keep the dataset order: row-wise sums do not
  depend on it.
- `ET_MINER_ROW_ORDER=keep|lex` pins it. Unset, it is lex (provisional until
  this campaign decides).

## Box and budget

The M1 box (2× RTX 3060, EPYC 7282, thread pools at 16, `NCCL_P2P_DISABLE=1`,
a shared host). Budget ≤ 1 GPU-hour; the campaign and its final check run
with `--max-gpu-hours 0.6`.

## Arms, regimes, repetitions

| arm | pin |
|---|---|
| `keep` | `ET_MINER_ROW_ORDER=keep` (the dataset's row order, no occupancy) |
| `lex` | `ET_MINER_ROW_ORDER=lex` |
| `final` | unset (the dispatch), one rep after the campaign |

- Every moonshot regime (`MOONSHOT_REGIMES`, free sets and count inference on
  dsl included), on one GPU and on two.
- 3 reps rep-major, fresh process per config, warm-up, medians with
  [min, max].
- The level split records the order's cost (`input_dev_order`) and the
  occupancy's (`input_occupancy`).
- No tracked-file edits during a run.

## Decision rule (rules 1, 2, 4, 5; `decide.py --arms keep lex`)

1. One signature per regime across arms and final rows; a mismatch or an
   unallowed fallback is a must-fix.
2. `lex` wins a regime with a ≥ 10 % lower median, disjoint ranges and ≥ 1 s
   saved.
4. If `lex` wins some and loses none, it stays the default. If it loses some,
   a dispatch rule from facts known after K=1 must leave those regimes on
   `keep` (the frequent items' count, the row lengths, the rows), or the
   default goes to `keep`.
5. If it wins nowhere, M5b goes.

## Commands

```bash
OUT=bench/results/2026-10-17-moonshots-m5b
uv run python bench/runner.py --mode moonshots-m5b --out $OUT --max-gpu-hours 0.6
uv run python bench/moonshots/decide.py $OUT/raw.jsonl --arms keep lex > $OUT/decision.md
uv run python bench/moonshots/split_table.py $OUT/raw.jsonl > $OUT/split.md
```
