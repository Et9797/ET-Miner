# %% [markdown]
# # Tier 2.2: Inside the Rust core
#
# The Rust extension `et_miner_rust` counts itemsets on packed bit vectors, one bit per row.
# This notebook follows the data from Python into Rust and back, and shows each step on the hand-made table.
# Source quotes are read from the files at run time, so the line numbers match the checked-out version.
#
# **What you will learn**
#
# - what crosses the PyO3 boundary, and what is borrowed or copied;
# - the data layout on the Rust side: CSR in, CSC and per-item bit vectors inside;
# - how support counting and candidate generation are written;
# - how the work is spread over threads;
# - how the Python API calls into Rust.
#
# **Prerequisites**: a built extension ([01-build-and-install](01-build-and-install.md)) and
# [tier 1, notebook 2](../tier1-polars/02-apriori-step-by-step.ipynb) for the algorithm itself.

# %% [markdown]
# ## 0. Setup
#
# The cell checks that the extension is importable and prints its version and thread count.
# `show()` prints a short, numbered excerpt of a source file.

# %%
import os
import warnings

os.environ.setdefault("LOGURU_LEVEL", "WARNING")
warnings.filterwarnings("ignore", message="IProgress not found")

import datetime
import platform
import re
import time
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

import et_miner

assert et_miner.HAS_RUST, "build the extension first: see 01-build-and-install.md"
import et_miner_rust

REPO = Path("../..").resolve()
DATA = Path("../data")


def show(rel_path: str, pattern: str, n_lines: int) -> None:
    """Print n_lines of a repository file, starting at the first line matching pattern."""
    lines = (REPO / rel_path).read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if re.search(pattern, line))
    print(f"{rel_path}:{start + 1}")
    for i in range(start, min(start + n_lines, len(lines))):
        print(f"{i + 1:5d}  {lines[i]}")


def hardware() -> str:
    cpu = platform.processor() or platform.machine()
    return f"{cpu}, {os.cpu_count()} logical CPUs, Python {platform.python_version()}, {datetime.date.today()}"


print("et_miner_rust", et_miner.get_rust_version(), "| rayon threads:", et_miner_rust.get_num_threads())
print("exported:", [n for n in dir(et_miner_rust) if not n.startswith("_") and n != "et_miner_rust"])

# %% [markdown]
# The module exports counting functions, a full Apriori, bit-vector helpers for the GPU path and pruning helpers.
# This notebook uses `count_itemsets_simd`, `build_column_bitvecs_u64`, `apriori_from_csr` and `get_num_threads`.

# %% [markdown]
# ## 1. The input: a CSR matrix
#
# Rust never sees Polars lists. It takes a sparse matrix in **CSR** form (compressed sparse row):
#
# - `indices`: the item index of every (row, item) pair, row after row;
# - `indptr`: where each row starts in `indices` (length `n_rows + 1`).
#
# The next cell builds the CSR of the hand-made table with items `a`..`f` mapped to 0..5.

# %%
toy = pl.read_parquet(DATA / "toy_8x6.parquet")
names = sorted({i for row in toy["items"] for i in row})
code = {name: idx for idx, name in enumerate(names)}
rows = [sorted(code[i] for i in row) for row in toy["items"].to_list()]
indptr = np.cumsum([0] + [len(r) for r in rows]).astype(np.int64)
indices = np.concatenate(rows).astype(np.int64)
n_rows, n_cols = len(rows), len(names)
print("items  :", code)
print("indptr :", indptr)
print("indices:", indices)

# %% [markdown]
# Row 0 is `indices[0:3] = [0, 1, 2]`, which is `{a, b, c}`; row 1 is `indices[3:5] = [0, 1]`, and so on.
# Within a row the indices must be strictly increasing; section 2 shows what happens otherwise.

# %% [markdown]
# ## 2. The PyO3 boundary: borrowed or copied
#
# Arrays arrive as `PyReadonlyArray1<i64>`, a read-only view of the NumPy buffer.
# The helper below turns that view into a Rust slice.
# If the array is contiguous it **borrows** the NumPy memory; otherwise it **copies** it once.

# %%
show("rust_ext/src/lib.rs", r"^fn contiguous", 10)

# %% [markdown]
# Three rules follow from the signatures, and the next cell demonstrates each:
#
# 1. the dtype must be `int64`; an `int32` array is rejected, it is not converted;
# 2. a non-contiguous view (for example every second element) is accepted and copied;
# 3. the CSR shape is validated before any work (`validate_csr` in the same file), so bad input raises a Python error.
#
# Itemsets are passed as Python lists; PyO3 converts them into a new `Vec<Vec<usize>>`, which is a copy.
# The counts come back as a new NumPy `uint32` array.

# %%
pairs = [[0, 1], [0, 2], [1, 2]]
print("int64, contiguous:", et_miner_rust.count_itemsets_simd(indptr, indices, n_rows, n_cols, pairs))

try:
    et_miner_rust.count_itemsets_simd(indptr.astype(np.int32), indices.astype(np.int32), n_rows, n_cols, pairs)
except TypeError as err:
    print("int32 -> TypeError:", err)

strided = np.repeat(indices, 2)[::2]
print("strided view, contiguous?", strided.flags["C_CONTIGUOUS"], "->",
      et_miner_rust.count_itemsets_simd(indptr, strided, n_rows, n_cols, pairs))

bad = indices.copy()
bad[[0, 1]] = bad[[1, 0]]
try:
    et_miner_rust.count_itemsets_simd(indptr, bad, n_rows, n_cols, pairs)
except ValueError as err:
    print("unsorted row -> ValueError:", err)

# %% [markdown]
# The counts `[5, 4, 4]` for `{a,b}`, `{a,c}`, `{b,c}` match the hand counts of tier 1.
# The strided view gives the same answer through the copy path.
# On the Python side, `src/et_miner/core/sparse.py:_count_support_sparse_k_gt_2_rust` calls `.astype(np.int64)` on
# SciPy's `int32` arrays, and `astype` always makes a new array, so that route pays one copy per call before Rust borrows it.

# %% [markdown]
# ## 3. The layout inside Rust: one bit vector per item
#
# Both `count_itemsets_simd` and `apriori_from_csr` first turn CSR into CSC (column-major),
# then build one bit vector per item: bit *r* is set when row *r* contains the item.
# 64 rows fit in one `u64` word. `build_column_bitvecs_u64` returns this layout to Python. The GPU builds the same
# layout on the device with its own kernel (`csr_to_bitvec.cu`) and uses this Rust function only as a fallback.
# The next cell prints each item's bits for the 8 rows, row 0 on the right.

# %%
bitvecs = np.asarray(et_miner_rust.build_column_bitvecs_u64(indptr, indices, n_rows, n_cols))
print("shape (items, words):", bitvecs.shape, "| dtype:", bitvecs.dtype)
for name, idx in code.items():
    print(f"{name}: {int(bitvecs[idx, 0]):08b}")

# %% [markdown]
# Item `a` is in rows 0, 1, 2, 4, 5 and 7, so bits 0, 1, 2, 4, 5 and 7 are set.
# The layout costs `n_items * ceil(n_rows / 64) * 8` bytes, whatever the density.

# %% [markdown]
# ## 4. Support counting: AND, then popcount
#
# The count of an itemset is the number of set bits in the AND of its items' vectors.
# `rust_ext/src/core/apriori.rs:count_bitvec_intersection_2` does this word by word for pairs:

# %%
show("rust_ext/src/core/apriori.rs", r"^fn count_bitvec_intersection_2", 7)

# %% [markdown]
# `count_ones` counts the set bits of one 64-bit word, so one step covers 64 rows.
# The default build is portable (`rust_ext/.cargo/config.toml` sets no target CPU); built with
# `RUSTFLAGS="-C target-cpu=native"` on a CPU that has it, `count_ones` becomes a single popcount instruction.
# The next cell does the same in NumPy on the vectors from section 3 and compares with Rust.

# %%
ab = bitvecs[code["a"]] & bitvecs[code["b"]]
abc = ab & bitvecs[code["c"]]
print("{a,b}  :", f"{int(ab[0]):08b}", "->", int(np.bitwise_count(ab).sum()))
print("{a,b,c}:", f"{int(abc[0]):08b}", "->", int(np.bitwise_count(abc).sum()))
print("rust   :", et_miner_rust.count_itemsets_simd(indptr, indices, n_rows, n_cols, [[0, 1], [0, 1, 2]]))

# %% [markdown]
# NumPy and Rust agree: `{a,b}` is in 5 rows and `{a,b,c}` in 3.

# %% [markdown]
# ## 5. Candidate generation in Rust
#
# `apriori_from_csr` generates candidates with the same prefix join and subset prune as tier 1.
# The excerpt shows the join over sorted frequent itemsets and the prune call.

# %%
show("rust_ext/src/core/candidates.rs", r"Two itemsets can be joined if they share", 14)

# %% [markdown]
# The prune is `all_subsets_frequent`, a hash-set lookup of every (K-1)-subset, as in
# `src/et_miner/core/candidates.py:_is_valid_candidate`.

# %% [markdown]
# ## 6. The whole loop in Rust: `apriori_from_csr`
#
# `apriori_from_csr(indptr, indices, n_rows, n_cols, min_support, max_length)` runs every level in Rust
# (`max_length=0` means no limit). It returns column indices and integer counts, not a DataFrame.
# The minimum count uses the same exact rule as Python (`exact_min_count` in `rust_ext/src/core/apriori.rs`
# and `src/et_miner/core/result.py:_min_count` read the same test table).
# The next cell mines the toy table and compares with tier 1.

# %%
itemsets, counts = et_miner_rust.apriori_from_csr(indptr, indices, n_rows, n_cols, 0.25, 0)
rust_toy = {tuple(names[i] for i in s): int(c) for s, c in zip(itemsets, counts)}
tier1 = et_miner.apriori(toy, min_support=0.25, sparse=False)
tier1_toy = {tuple(s): round(v * n_rows) for s, v in zip(tier1["itemset"].to_list(), tier1["support"].to_list())}
print(rust_toy)
assert rust_toy == tier1_toy
print("identical to tier 1:", len(rust_toy), "itemsets")

# %% [markdown]
# The Rust loop returns the same 13 itemsets and counts as the Polars tier.

# %% [markdown]
# ## 7. Threads: rayon
#
# The counting loops use rayon's `par_iter`: the candidate list is split across a work-stealing thread pool.
# `count_itemsets_simd` takes an optional `n_threads`; `0` means the global pool (all cores),
# and the Python side passes the caller's `n_jobs` through (`_call_with_budget` in `src/et_miner/core/sparse.py`).

# %%
show("rust_ext/src/core/apriori.rs", r"^fn count_and_filter_k2", 16)

# %% [markdown]
# The next cell counts all pairs of the 200 most common items of the shared sample (19,900 pairs over 60,000 rows)
# with one thread and with the full pool.

# %%
smoke = pl.read_parquet(DATA / "smoke.parquet")
s_rows = [np.asarray(r, dtype=np.int64) for r in smoke["items"].to_list()]
s_indptr = np.cumsum([0] + [len(r) for r in s_rows]).astype(np.int64)
s_indices = np.concatenate(s_rows)
s_ncols = int(s_indices.max()) + 1
top = np.argsort(-np.bincount(s_indices, minlength=s_ncols))[:200]
all_pairs = [[int(a), int(b)] for i, a in enumerate(sorted(top)) for b in sorted(top)[i + 1 :]]
timing = {}
for n_threads in (1, 0):
    t0 = time.perf_counter()
    c = et_miner_rust.count_itemsets_simd(s_indptr, s_indices, len(s_rows), s_ncols, all_pairs, n_threads)
    timing["1 thread" if n_threads == 1 else f"pool ({et_miner_rust.get_num_threads()} threads)"] = round(time.perf_counter() - t0, 3)
print(hardware())
print(len(all_pairs), "pairs:", timing, "seconds")

# %% [markdown]
# The pool is faster than one thread on this machine. Each call also rebuilds the CSC and the bit vectors
# before counting, a fixed cost that no thread count removes; notebook 3 comes back to it.

# %% [markdown]
# ## 8. How the Python API reaches Rust
#
# There are two ways in.
#
# 1. `et_miner.apriori(..., sparse=True)`, or `sparse=None` when the automatic choice picks sparse.
#    The loop stays in Python. `src/et_miner/core/matrix.py:count_support_batched` hands each level to
#    `src/et_miner/core/sparse.py:count_support_sparse`, which converts the boolean matrix to SciPy CSR,
#    counts pairs with a sparse matrix product, and sends itemsets of length 3 and more to `count_itemsets_simd`.
# 2. `et_miner.apriori_from_csr(...)`, the loop of section 6, with no Python in between.
#
# The next cell runs route 1 on the shared sample and captures the library's debug line for each Rust call.

# %%
captured = []
sink = logger.add(lambda m: captured.append(m.record["message"]), level="DEBUG",
                  filter=lambda r: "RUST" in r["message"])
routed = et_miner.apriori(smoke, min_support=0.01, sparse=True)
logger.remove(sink)
print(routed.height, "itemsets")
print("\n".join(captured))

# %% [markdown]
# Levels 3 and up each make one call into Rust; level 2 stays in SciPy.
# With `n_jobs=1`, the default, each call runs with a budget of one thread.

# %% [markdown]
# ## Summary and next step
#
# - Rust takes `int64` CSR arrays, borrows them when contiguous, validates them, and copies itemset lists in.
# - Inside, each item becomes a bit vector; a count is an AND and a popcount per 64 rows.
# - Candidate generation is the same join and prune as tier 1; rayon spreads candidates over threads.
# - `apriori(sparse=True)` calls Rust per level for K >= 3; `apriori_from_csr` runs the whole loop in Rust.
#
# Next: [03-same-results-and-speed](03-same-results-and-speed.ipynb) checks the results and measures both routes.
