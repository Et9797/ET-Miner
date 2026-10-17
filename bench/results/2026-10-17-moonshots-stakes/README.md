# Moonshots Phase 1: stakes (2026-10-17)

CPU-only replays of the Phase 0 lattice dumps
(`../2026-10-17-moonshots-baseline/*.lattice.parquet`) and datasets. No GPU
time. The numbers feed `bench/moonshots/SPEC.md` and `STAKES.md`; `stakes.py`
turns them into bounds:

```bash
uv run python bench/moonshots/stakes.py bench/results/2026-10-17-moonshots-baseline \
    bench/results/2026-10-17-moonshots-stakes > bench/results/2026-10-17-moonshots-stakes/stakes.jsonl
```

## Files and commands

From the repository root, with `B=bench/results/2026-10-17-moonshots-baseline`
and `S=bench/results/2026-10-17-moonshots-stakes`:

| file | command | for |
|---|---|---|
| `replay-<regime>.jsonl` | `replay_stakes.py $B/<cfg>.lattice.parquet --dataset <dataset>` | M2 bound, M5a, M7 (depth 3), M8, kernel work per level |
| `replay-dsl-ndi2.jsonl` | `replay_stakes.py $B/dsl-C1-base.lattice.parquet --ndi-depth 2` | M7 at depth 2 |
| `replay-dsl-order-{asc,desc}.jsonl` | `relabel_lattice.py $B/dsl-C1-base.lattice.parquet <order> <tmp>` then `replay_stakes.py <tmp>.lattice.parquet --no-ndi` | M10 on dsl |
| `census-{oom2ml3,sk2ml3}.jsonl` | `k3_census.py $B/<cfg>.lattice.parquet` | M10 on the K=3 explosions, U_3 |
| `k3rows-{oom2ml3,sk2ml3}.json` | `k3_rows_stake.py <dataset> $B/<cfg>.lattice.parquet --gpus 2` | M2 at K=3 (exact increments), M9 (per-shard counts) |
| `components-<regime>.jsonl` | `component_stakes.py $B/<cfg>.lattice.parquet --dataset <dataset>` | M4 (components and families) |
| `tiles-dsl.jsonl` | `tile_stakes.py deep $B/dsl-C1-base.lattice.parquet --dataset deep_sparse_large --groups 1000` | M5b on dsl |
| `tiles-{oom2ml3,sk2ml3}.jsonl` | `tile_stakes.py k3 $B/<cfg>.lattice.parquet --dataset <dataset>` | M5b and M12 on the K=3 explosions |
| `dedup-<regime>.jsonl` | `dedup_stakes.py $B/<cfg>.lattice.parquet --dataset <dataset>` | M11 |
| `input-floor-<dataset>.json` | `input_floor.py <dataset>` | M1's host floor |
| `preset-deep_k20.json`, `components-deep_k20.jsonl` | `preset_check.py <tmp>/deep_k20` then `component_stakes.py <tmp>/deep_k20.lattice.parquet` | M4's proposed preset |
| `stakes.jsonl` | `stakes.py $B $S` | every bound |

The regimes `dsl-free` and `dsl-infer` replay `dsl-C1-free-base` and
`dsl-C1-base` (count inference does not change the lattice). The datasets are
`smoke`, `deep_k`, `skewed_rows`, `online_retail`, `deep_sparse_large`,
`oom_regression` and `stress_k2`.

## Checks the scripts make

- `replay_stakes.py`: every level's generated candidates and frequent itemsets
  equal the Phase 0 run's (`levels` in `raw.jsonl`). Every candidate the
  deduction rules settle as frequent has l = u equal to its true count. No
  upper bound below min_count falls on a frequent candidate. On smoke, the
  bounds of all 673 countable K=3–5 candidates contain their brute-force
  counts.
- `k3_rows_stake.py`: the candidates whose summed row-wise count reaches
  min_count are exactly the lattice's K=3 level, counts included
  (`k3_matches_lattice`). The aggregated increments equal the enumerated ones.
- `component_stakes.py`: every frequent itemset of size ≥ K0 lies inside one
  component (asserted).
- `k3_census.py`: the counted candidates (144,951,409 on sk2ml3) are the same
  under every order.

## Box

The Phase 0 box (Pentium Gold G5420, 2 cores, 31 GB), Polars on its compat
runtime. `input-floor-*.json` is timed on it. The other files are counts and
do not depend on the box.
