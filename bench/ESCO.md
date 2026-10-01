# ESCO restoration and comparison

Merge `8548716` removed ESCO in commit `fb0f28a` (DP5). The commit calls it
the sparse CSR layout: at a fixed K, or when the previous frequent level's
mean count drops below N/32, dense item bitvectors are converted into
sorted int32 transaction-id lists. Later levels intersect their two prefix
parents' tidsets, count each GPU's shard and reduce the partial counts.
The switch is one-way. `sparse_from_k=None` remains the default.

The removal reported no useful winning regime, but recorded Retail at
support 0.002 as 0.56 s sparse versus 0.82 s dense. It also recorded sparse
OOM on `deep_sparse_large` while dense finished. That evidence warrants
keeping ESCO available for comparison; it does not establish that the N/32
byte crossover is a speed crossover. Conversion and tidset materialization
are part of the measured wall time.

This restoration uses the surviving row-split miner and its current
chunking, threshold filter and NCCL/staged reduction. It restores the
suffix-slot → previous-row mapping, so sorting and free-set masks also
permute survivor ids before materialization. Caller bitvectors remain
read-only. Dense references from fused launches are dropped before the
conversion can release owned bitvectors. SON rejects the unsupported
`sparse_from_k` setting rather than dropping it.

## Correctness gate

```bash
uv run pytest -q tests/test_tier_equivalence.py tests/test_density_transition.py \
  tests/test_density_auto_gpu.py tests/test_csr_warp.py tests/test_group_src_rows.py \
  tests/test_free_set_semantics.py tests/test_gpu_ownership.py tests/test_esco_benchmark.py
uv run python bench/selfcheck.py
```

The smoke gate compares exact itemsets and absolute counts with
efficient-apriori at an explicit depth and the integer-safe support boundary.
It includes auto and fixed K=3 ESCO, one/two GPUs, and forced chunks.
The one-GPU ESCO legs assert that the sparse counting wrapper actually ran.
`selfcheck` compiles and launches all four restored kernels on every device.

## Online Retail at extremely low support

The prepared Online Retail II dataset has 36,422 baskets over 4,985 item ids.
At 0.0001 the minimum count is 4; at 0.00005 it is 2. On the downloaded
dataset, a 53-item basket repeats four times and a 93-item basket repeats
twice. Every nonempty subset of each basket is frequent, giving full-lattice
lower bounds of 2^53−1 and 2^93−1 itemsets. The comparison therefore uses
explicit `max_length=2`, 3 and 4, preserving the full row set and requested
support. K=2 is a control before ESCO can switch, K=3 exercises the switch,
and K=4 also exercises sparse survivor materialization.

```bash
uv run python datasets/prepare_online_retail.py
uv run python -m et_miner.synthetic --preset smoke --out datasets/synth
# CPU and canonical oracle through K=2; works on a machine without CUDA.
uv run python bench/runner.py --mode esco-retail --cpu-only \
  --out bench/results/retail-low-cpu
# On a CUDA machine, after the correctness gate and selfcheck above:
uv run python bench/runner.py --mode esco-retail --out bench/results/retail-low-gpu
uv run python bench/consolidation_report.py --out bench/results/retail-low-gpu
```

Every support/depth/device count compares dense automatic dispatch, dense
tiled pinned, dense per-candidate pinned, ESCO auto, and ESCO from K=3.
CPU sparse and efficient-apriori controls run through K=2. Three reps are
run per config, in rep-major order, with shallow workloads first. Each run
has a 30-minute timeout and failures/OOM are recorded; failed runs cannot
pass the equivalence gate. Forced K=3 configs through K≥3 also fail if no
transition occurred. Auto configs record their actual transition level,
including an empty list if they stayed dense. Logs retain level progress
when a run times out.

`--only or0001-k3` or `--only or00005-k3` selects a smaller experiment;
the runner then correctly reports incomplete coverage of the entire matrix.
No comparisons at K=2 alone establish ESCO performance.

## Wider comparison

```bash
uv run python -m et_miner.synthetic --preset all --out datasets/synth
uv run python bench/runner.py --mode esco --out bench/results/esco
uv run python bench/consolidation_report.py --out bench/results/esco
```

This covers smoke, deep K, skewed rows, large sparse/deep data, wide
vocabulary and Retail at 0.003/0.002. The same dense controls and ESCO modes
run on one GPU and, when available, two; manageable workloads include CPU
sparse controls. Recorded times, peak RSS/VRAM and result signatures belong
to the revision and hardware captured by the runner.
