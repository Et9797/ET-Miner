# %% [markdown]
# # Tier 3.3: Shared-memory and tiled counting routes
#
# Counting is a grid of candidates times 64-row words. The GPU tier has two kernel families for the dense part of
# that grid: **legacy** kernels, where each candidate reads its own items' words from global memory, and a
# **shared/tiled** kernel, where a block stages words in shared memory once and serves many candidates from there.
# This notebook explains the tiling, how a route is chosen, how to force one, and how to compare them.
#
# **What you will learn**
#
# - how candidates are grouped by prefix, and why that allows reuse;
# - what the tiled kernel keeps in shared memory;
# - which counting routes exist and which switch selects each;
# - how the row-split chunk planner sends small groups to the legacy kernel;
# - how to time the two variants against each other.
#
# **Prerequisites**: [02-gpu-resident-mining](02-gpu-resident-mining.ipynb).
# Cells marked *GPU* print a skip line without a CUDA device; the setup cell prints the hardware.

# %% [markdown]
# ## 0. Setup and device check

@@GPU_SETUP@@

# %% [markdown]
# ## 1. The counting grid and its prefix groups
#
# A level's candidates come from the prefix join: all candidates that share their first K-1 items form a
# **prefix group**, and within a group every pair of suffixes `(i < j)` is one candidate.
# A group of `g` suffixes has `g(g-1)/2` candidates; level 2 is one group with an empty prefix.
# The next cell mines the shared sample on the CPU and builds the groups of levels 2 and 3 from the result.

# %%
import math
from collections import Counter

import numpy as np

smoke = pl.read_parquet(DATA / "smoke.parquet")
N, S = smoke.height, 0.01
base = apriori(smoke, min_support=S, sparse=False)
by_k = {k: sorted(tuple(s) for s in base.filter(pl.col("itemset").list.len() == k)["itemset"].to_list())
        for k in (1, 2)}
n_words = math.ceil(N / 64)
groups = {2: [len(by_k[1])], 3: sorted(Counter(p[:-1] for p in by_k[2]).values(), reverse=True)}
for k, sizes in groups.items():
    pairs = sum(g * (g - 1) // 2 for g in sizes)
    print(f"K={k}: {len(sizes)} prefix group(s), largest {sizes[0]} suffixes, {pairs:,} group candidates x {n_words} words")

# %% [markdown]
# Level 2 is a single group of 118 items. Level 3 splits into groups of very different sizes.
# These are the candidates the group enumeration produces, before any subset prune.

# %% [markdown]
# ## 2. What the tiled kernel keeps in shared memory
#
# The header of `src/et_miner/gpu/kernels/_src/shared_tiled.cu` describes the kernel.
# One block handles one pair of 32-suffix tiles inside one prefix group. For each run of 32 words it stages the
# prefix AND once, plus the words of both tiles, in shared memory; then its 256 threads count up to 32 x 32 pairs
# from there.

# %%
show("src/et_miner/gpu/kernels/_src/shared_tiled.cu", r"^// One block = one", 26)

# %% [markdown]
# Shared memory is on-chip and far faster than global memory, but small (the kernels stay under the 48 KB static
# limit). The gain comes from reuse: a word loaded once serves many candidates instead of one.

# %% [markdown]
# ## 3. A load model of the two kernels
#
# The header gives the global loads per word: the legacy kernels load `k` words per candidate, the tiled kernel
# loads `(k - 2) + 2T` words per tile pair (T = 32) and serves up to `T*T` candidates with them.
# The next cell applies that model to the groups of section 1. It is arithmetic on the header's formula, not a measurement.

# %%
T = 32
rows = []
for k, sizes in groups.items():
    candidates = sum(g * (g - 1) // 2 for g in sizes)
    tile_pairs = sum((nt := math.ceil(g / T)) * (nt + 1) // 2 for g in sizes)
    legacy = candidates * k
    tiled = tile_pairs * ((k - 2) + 2 * T)
    rows.append(dict(k=k, candidates=candidates, tile_pairs=tile_pairs, legacy_loads_per_word=legacy,
                     tiled_loads_per_word=tiled, ratio=round(legacy / tiled, 2)))
print(pl.DataFrame(rows))
for g in (3, 12, 91):
    nt = math.ceil(g / T)
    print(f"one K=3 group of {g:2d} suffixes: legacy {g * (g - 1) // 2 * 3:5d} loads/word, "
          f"tiled {nt * (nt + 1) // 2 * (1 + 2 * T):5d} loads/word")

# %% [markdown]
# Summed over a level, the model gives the tiled kernel far fewer loads, and the advantage is largest for level 2,
# one large group. Per group the picture differs: a group of a few suffixes still makes a tile pair load `2T` words
# per word while serving only a few candidates, so the model favors the legacy kernel for tiny groups.
# That is why the row-split planner routes small groups to the legacy kernel (section 5).

# %% [markdown]
# ## 4. The routes and their switches
#
# | Route | When | Kernel sources | Switch |
# |---|---|---|---|
# | single GPU, default | `use_gpu=True` | `pairs_k2.cu`, `k3plus_fullyfused.cu` (legacy) or `shared_tiled.cu` (fused variant) | `ET_MINER_KERNEL_VARIANT` |
# | single GPU, resident | `gpu_resident=True` | `pairs_k2.cu`, `k3plus_gpu_resident.cu` | none; the variant is not read |
# | row-split (multi-GPU, pruning) | `n_gpus>1`, `prune_equal_support`, `anchor_items` | `pairs_k2_dense.cu`, `k3plus_dense.cu` or `shared_tiled.cu` (dense variant), per chunk | `ET_MINER_KERNEL_VARIANT`, `ET_MINER_TILED_MIN_GROUP_PAIRS` |
# | sparse CSR levels | `sparse_from_k` | `csr_warp.cu` | notebook 4 |
#
# `ET_MINER_KERNEL_VARIANT` accepts `auto`, `legacy` or `shared`, and is read at call time by
# `src/et_miner/gpu/dispatch.py:resolved_kernel_variant`. The next cell shows how each value resolves; it needs no GPU.

# %%
from et_miner.gpu.dispatch import resolved_kernel_variant

show("src/et_miner/gpu/dispatch.py", r"^def resolved_kernel_variant", 10)
for value in ("auto", "legacy", "shared", None):
    if value is None:
        os.environ.pop("ET_MINER_KERNEL_VARIANT", None)
    else:
        os.environ["ET_MINER_KERNEL_VARIANT"] = value
    print(f"ET_MINER_KERNEL_VARIANT={value!s:7} -> {resolved_kernel_variant()}")
os.environ.pop("ET_MINER_KERNEL_VARIANT", None)

# %% [markdown]
# Unset and `auto` both resolve to `shared` in this version. Setting `legacy` is the way to force the legacy kernels.

# %% [markdown]
# ## 5. Which groups the row-split planner sends where
#
# On the row-split route, level K>=3 is counted in chunks that must hold whole prefix groups for the tiled kernel.
# `src/et_miner/gpu/row_split_chunks.py:plan_group_chunks` builds the chunk plan on the CPU and marks two kinds of
# group for the legacy kernel: groups with fewer candidate pairs than `ET_MINER_TILED_MIN_GROUP_PAIRS` (default 64),
# and groups larger than one chunk. It drops the tiny-group routing for a level when that routing would split the
# plan into many more chunks. The next cell plans level 3 of the sample with a large chunk budget and with a
# small one.

# %%
from et_miner.gpu.row_split_chunks import plan_group_chunks

pairs_per_group = [g * (g - 1) // 2 for g in groups[3]]
cumulative = np.concatenate([[0], np.cumsum(pairs_per_group)])
for budget in (1_000_000, 200):
    plan = plan_group_chunks(cumulative, max_cands=budget)
    tiled = sum(c.size for c in plan if not c.use_legacy)
    legacy = sum(c.size for c in plan if c.use_legacy)
    print(f"budget {budget:>9,} candidates/chunk: {len(plan)} chunks | tiled candidates {tiled} | legacy candidates {legacy}")
print("groups with >= 64 pairs:", sum(p >= 64 for p in pairs_per_group), "of", len(pairs_per_group))

# %% [markdown]
# With a large budget the plan sends the few big groups to the tiled kernel and the runs of tiny groups to the
# legacy kernel. A small budget splits the work into more chunks, and groups that no longer fit a chunk go to legacy.
# The plan is a pure function of its inputs, so every GPU in a row-split run computes the same plan.

# %% [markdown]
# ## 6. Timing the two variants
#
# The variant is read at call time, so setting the environment variable between calls is enough.
# The next cell (*GPU*) mines the sample on one GPU with each variant, asserts identical results, and times three
# runs of each after a warm-up. The second part (*multi-GPU*) does the same on the row-split route with two GPUs.

# %%
import time


def counted(result):
    return {(tuple(s), round(v * N)) for s, v in zip(result["itemset"].to_list(), result["support"].to_list())}


def time_variant(variant: str, **kwargs) -> tuple[float, set]:
    os.environ["ET_MINER_KERNEL_VARIANT"] = variant
    try:
        apriori(smoke, min_support=S, use_gpu=True, **kwargs)  # warm-up and kernel compile
        best, result = float("inf"), None
        for _ in range(3):
            t0 = time.perf_counter()
            result = apriori(smoke, min_support=S, use_gpu=True, **kwargs)
            best = min(best, time.perf_counter() - t0)
        return best, counted(result)
    finally:
        os.environ.pop("ET_MINER_KERNEL_VARIANT", None)


tier1 = counted(base)
if SKIP_GPU:
    print("single GPU: skipped: no CUDA device")
else:
    print(hardware())
    for variant in ("legacy", "shared"):
        seconds, got = time_variant(variant)
        assert got == tier1
        print(f"single GPU, {variant:6}: best of 3 = {seconds:.4f} s, result == tier 1")
if SKIP_MULTI:
    print("row-split, 2 GPUs: skipped: fewer than two CUDA devices")
else:
    for variant in ("legacy", "shared"):
        seconds, got = time_variant(variant, n_gpus=2)
        assert got == tier1
        print(f"row-split 2 GPUs, {variant:6}: best of 3 = {seconds:.4f} s, result == tier 1")

# %% [markdown]
# Both variants must give the tier-1 answer; the tiled kernel is designed to be bit-identical to the legacy ones.
# On a sample this small, fixed costs dominate the wall time; compare variants on your own data at your own threshold.

# %% [markdown]
# ## Summary and next step
#
# - Candidates come in prefix groups; the tiled kernel stages a group's prefix AND and two suffix tiles in shared
#   memory and serves up to 32 x 32 candidates per block from them.
# - `ET_MINER_KERNEL_VARIANT=legacy|shared|auto` selects the kernel family (auto means shared at HEAD);
#   the resident route does not read it (auto means shared in this version).
# - On the row-split route, `plan_group_chunks` sends tiny and oversized groups to the legacy kernel.
#
# **Checklist when you run this notebook on a GPU**
#
# - [ ] Section 6 prints two single-GPU lines, each ending `result == tier 1`.
# - [ ] With two or more GPUs, section 6 also prints two row-split lines.
#
# Next: [04-sparse-dense-crossover](04-sparse-dense-crossover.ipynb).
