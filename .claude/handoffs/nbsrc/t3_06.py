# %% [markdown]
# # Tier 3.6: Multi-GPU mining
#
# With `apriori(..., use_gpu=True, n_gpus=N)` ET-Miner splits the **rows** across N GPUs. Every GPU holds the bit
# vectors of its own rows, counts the same candidates in the same order, and the partial counts are summed before any
# itemset is kept or dropped. This notebook shows each step: the split, the per-GPU counts, the sum, why the keep-or-drop
# decision has to use the sum, how to run on N GPUs, and what happens when there are fewer GPUs than requested.
#
# **What you will learn**
#
# - how `src/et_miner/gpu/csr_bitvec.py:_row_split_cuts` assigns row ranges to GPUs;
# - why per-GPU partial counts add up to exact global counts;
# - why frequency must be decided on the summed counts, not per GPU;
# - how the sum is reduced (NCCL or a staged fallback), and on which device;
# - how to run on N GPUs, and the failure modes and fallbacks.
#
# **Prerequisites**: [05-son-streaming](05-son-streaming.ipynb).
# Cells marked *GPU* print a skip line without a CUDA device, and cells marked *multi-GPU* without two.
# Sections 1 to 3 simulate the split on the CPU with the library's own functions, so they run everywhere.

# %% [markdown]
# ## 0. Setup and device check

@@GPU_SETUP@@

# %% [markdown]
# ## 1. How rows are split
#
# The row-split miner cuts the rows into contiguous ranges, one per GPU. By default every range has the same number
# of rows (`ET_MINER_ROW_BALANCE=rows`); `nnz` instead cuts at equal numbers of (row, item) entries.
# The next cell calls the cut function on the shared sample for 2 and 4 GPUs.

# %%
import numpy as np

from et_miner.gpu.csr_bitvec import _row_split_cuts

smoke = pl.read_parquet(DATA / "smoke.parquet")
N, S = smoke.height, 0.01
rows = smoke["items"].to_list()  # item ids are 0..n-1, sorted and unique within each row
indptr = np.cumsum([0] + [len(r) for r in rows]).astype(np.int64)
indices = np.concatenate([np.asarray(r, dtype=np.int64) for r in rows])
n_items = int(indices.max()) + 1
for n_gpus in (2, 4):
    for balance in ("rows", "nnz"):
        cuts = _row_split_cuts(indptr, N, n_gpus, balance)
        desc = ", ".join(f"[{a:,}, {b:,}) nnz={int(indptr[b] - indptr[a]):,}" for a, b in cuts)
        print(f"{n_gpus} GPUs, balance={balance}: {desc}")

# %% [markdown]
# Row lengths in this sample are similar across the table, so both modes give almost the same cuts.
# Ranges stay contiguous in both modes, which keeps each shard a plain slice of the CSR arrays.

# %% [markdown]
# ## 2. Each GPU counts its own rows; the sum is exact
#
# Every GPU builds bit vectors for its rows only (`n_items x ceil(shard_rows / 64)` words) and counts every candidate
# of the level on them. A candidate's count is a sum over rows, and the shards partition the rows, so adding the
# per-shard counts gives the exact global count. The next cell reproduces that on the CPU for level 2 with two shards:
# it builds each shard's bit vectors with the Rust function `build_column_bitvecs_u64` (the GPU builds the same layout
# with its `csr_to_bitvec` kernel),
# counts all pairs of frequent items per shard with NumPy, and sums.

# %%
import et_miner_rust

from et_miner.core.result import _min_count

min_count = _min_count(S, N)
freq_items = np.flatnonzero(np.bincount(indices, minlength=n_items) >= min_count)
pairs = [(int(a), int(b)) for i, a in enumerate(freq_items) for b in freq_items[i + 1 :]]
left, right = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def shard_pair_counts(start: int, end: int) -> np.ndarray:
    sub_ptr = indptr[start : end + 1] - indptr[start]
    sub_idx = indices[indptr[start] : indptr[end]]
    bits = np.asarray(et_miner_rust.build_column_bitvecs_u64(sub_ptr, sub_idx, end - start, n_items))
    return np.bitwise_count(bits[left] & bits[right]).sum(axis=1).astype(np.int64)


cuts = _row_split_cuts(indptr, N, 2, "rows")
partials = [shard_pair_counts(a, b) for a, b in cuts]
summed = partials[0] + partials[1]
whole = shard_pair_counts(0, N)
assert np.array_equal(summed, whole)
print(f"{len(pairs):,} candidate pairs; shard 0 + shard 1 == whole table for every pair")
print("frequent pairs from the summed counts:", int((summed >= min_count).sum()))

# %% [markdown]
# The two partial count arrays add up to the whole-table counts for every one of the 6,903 candidates.
# On the GPU these partial arrays are int32, one entry per candidate, in the same order on every device.

# %% [markdown]
# ## 3. Why the keep-or-drop decision must be global
#
# A GPU sees only its own rows. If each GPU applied the threshold to its own rows (a count of
# `ceil(s * shard_rows)`), it would keep or drop candidates by local evidence. The next cell counts, for the two-shard
# split above, the pairs where local and global decisions disagree.

# %%
local_mins = [_min_count(S, b - a) for a, b in cuts]
globally = summed >= min_count
in_any = (partials[0] >= local_mins[0]) | (partials[1] >= local_mins[1])
in_all = (partials[0] >= local_mins[0]) & (partials[1] >= local_mins[1])
print("frequent globally:", int(globally.sum()))
print("frequent in at least one shard but not globally:", int((in_any & ~globally).sum()))
print("frequent globally but not in every shard:       ", int((globally & ~in_all).sum()))

# %% [markdown]
# Per-GPU decisions disagree with the global answer in both directions: some locally frequent pairs are not globally
# frequent, and some globally frequent pairs miss the threshold on one shard. Dropping a candidate on one GPU would
# also change which candidates the next level generates, and the GPUs would stop counting the same list.
# So the miner sums first and decides once, on one device, and every GPU generates the next level from that one
# decision. (SON in notebook 5 does use local decisions, but only to propose candidates, and then recounts globally.)

# %% [markdown]
# ## 4. Where the sum happens
#
# The partial arrays are reduced onto GPU 0, not on the host. `src/et_miner/gpu/nccl.py` prefers an NCCL reduce to
# GPU 0; without NCCL, or with `ET_MINER_DISABLE_NCCL=1`, it adds the peers' arrays slice by slice through a fixed
# staging buffer on GPU 0. GPU 0 then filters the survivors on the device, and only survivors (index and count) cross
# to the host. The next cell prints the module's description and the row-split miner's own summary.

# %%
show("src/et_miner/gpu/nccl.py", r"^\"\"\"NCCL collective helpers", 8)
show("src/et_miner/gpu/row_split.py", r"GPU-resident dense counting architecture", 11)

# %% [markdown]
# Candidate lists are never exchanged: every GPU derives the same list from the same frequent itemsets, so element `i`
# of every partial array belongs to the same candidate.

# %% [markdown]
# ## 5. How to run on N GPUs
#
# | Goal | Call |
# |---|---|
# | mine on N GPUs, rows split | `apriori(df, min_support=s, use_gpu=True, n_gpus=N)` |
# | same, with the sparse levels of notebook 4 | add `sparse_from_k=3` or `"auto"` |
# | choose which physical GPUs | set `CUDA_VISIBLE_DEVICES` before Python starts |
# | cut by entries instead of rows | `ET_MINER_ROW_BALANCE=nnz` |
# | force the non-NCCL reduce | `ET_MINER_DISABLE_NCCL=1` |
# | SON streaming over N GPUs | `apriori(df, streaming=True, n_gpus=N, chunk_size=...)` |
#
# The row-split miner is also the route for `prune_equal_support`, `anchor_items` and `output_dir` on the GPU, even
# with one GPU. The next cell (*multi-GPU*) mines the sample on 1 to all visible GPUs (`n_gpus=1` runs the single-GPU
# bit-vector miner and serves as the baseline; 2 and more run the row-split miner), asserts each result equals
# tier 1, and times each run after a warm-up. The sample is small, so the timings mostly show fixed costs; the scaling
# that matters is on inputs with many rows.

# %%
import time


def counted(result):
    return {(tuple(s), round(v * N)) for s, v in zip(result["itemset"].to_list(), result["support"].to_list())}


tier1 = counted(apriori(smoke, min_support=S, sparse=False))
if SKIP_MULTI:
    print("skipped: fewer than two CUDA devices")
else:
    print(hardware())
    for n in range(1, N_GPUS + 1):
        apriori(smoke, min_support=S, use_gpu=True, n_gpus=n)  # warm-up
        t0 = time.perf_counter()
        result = apriori(smoke, min_support=S, use_gpu=True, n_gpus=n)
        seconds = time.perf_counter() - t0
        assert counted(result) == tier1
        print(f"n_gpus={n}: {seconds:.4f} s, result == tier 1")

# %% [markdown]
# Every GPU count must give the tier-1 answer; the repository's equivalence test asserts the same for two GPUs,
# with and without the sparse levels.

# %% [markdown]
# ## 6. Fallbacks and failure modes
#
# **Fewer GPUs than requested.** The row-split builder caps `n_gpus` at the number of visible devices
# (`src/et_miner/gpu/csr_bitvec.py:build_bitvecs_row_split_from_arrays`), and the candidate fan-out of the single-GPU
# miner caps it the same way and logs the decision (`src/et_miner/gpu/dispatch.py:_resolve_gpus`). The next cell asks
# the fan-out rule, which runs on the CPU, how many devices it would use for level 2 of the sample.

# %%
from et_miner.gpu.dispatch import PAIR_COUNT_THRESHOLD, should_use_multi_gpu

print("visible devices:", N_GPUS, "| pair threshold for fan-out:", f"{PAIR_COUNT_THRESHOLD:,}")
for requested in (1, 4):
    print(f"n_gpus={requested}, {len(pairs):,} pairs -> fan out:", should_use_multi_gpu(len(freq_items), n_gpus=requested))

# %% [markdown]
# The fan-out needs more than one visible device and at least 15,000,000 pairs; below that the level runs on one GPU.
#
# **Combinations that are refused.** `apriori` checks the parameters before any GPU work and raises instead of
# ignoring one. The next cell shows the multi-GPU cases, plus multi-GPU streaming on a machine without CUDA.

# %%
from et_miner import apriori_streaming_multi_gpu

for kwargs in (dict(use_gpu=True, n_gpus=2, gpu_resident=True), dict(use_gpu=True, n_gpus=2, profile=True),
               dict(streaming=True, n_gpus=2, profile=True)):
    try:
        apriori(smoke, min_support=S, **kwargs)
    except ValueError as err:
        print(f"{kwargs}\n  -> ValueError: {str(err)[:140]}...")
try:
    multi = apriori_streaming_multi_gpu(smoke, min_support=S, n_gpus=2, chunk_size=15_000, show_progress=False)
    assert counted(multi) == tier1
    print("apriori_streaming_multi_gpu ran, 4 chunks over 2 GPUs, result == tier 1")
except Exception as err:
    print(f"apriori_streaming_multi_gpu -> {type(err).__name__}: {str(err)[:140]}")

# %% [markdown]
# `gpu_resident` and `profile` do not exist on the row-split miner, so the calls stop with an explanation.
# Multi-GPU streaming raises `MiningError` when no CUDA device is present. Other failure modes named in the source:
# row counts must stay below 2^31 (int32 row ids and counts), and on machines without peer-to-peer access NCCL may fall
# back to slower transports; `bench/README.md` covers the second on rented multi-GPU machines.

# %% [markdown]
# ## Summary
#
# - Rows are split into contiguous ranges, one per GPU; each GPU counts the same candidates on its rows.
# - Partial counts add up exactly; they are reduced onto GPU 0, which decides frequency once for all GPUs.
# - `apriori(..., use_gpu=True, n_gpus=N)` runs it; `n_gpus` is capped at the visible devices.
#
# **Checklist when you run this notebook on two or more GPUs**
#
# - [ ] Setup cell lists every GPU.
# - [ ] Section 5 prints one line per GPU count, each ending `result == tier 1`.
# - [ ] Section 6: `apriori_streaming_multi_gpu` prints that it ran with 4 chunks and matched tier 1.
#
# This is the last tutorial. The [docs README](../README.md) lists everything else in the repository.
