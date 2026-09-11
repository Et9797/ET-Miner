# BUGS_FOUND.md

Confirmed defects in ET-Miner at commit `22cb1dc`, found by a four-reviewer
adversarial code review (architecture / numerics / alternatives / correctness),
two rounds each, with every finding cross-validated by at least a second
reviewer before it was recorded here.

**Method.** Each finding below is either measured — a reproduction was run and
its output is pasted — or proved by citation to the specific lines that make it
true. Nothing is listed on suspicion. Suspicions, retractions and
verified-correct negative results are recorded at the end, because knowing what
was ruled out is as useful as knowing what was found.

**Baseline.** At HEAD, `uv run pytest -q -m "not slow"` is **471 passed, 4
skipped**, and `tests/test_tier_equivalence.py` passes **9/9**. Every defect
below is live against a green suite. See *Why the suite doesn't catch any of
this* — that turned out to be the most transferable finding of the review.

**Scope.** Three sequential reviews covering the whole tree: host core + Rust
extension, GPU path + CUDA kernels, and streaming/SON + IO + config/CLI.
**62 confirmed defects**, all live against a green suite.

| Section | HIGH | MEDIUM | LOW | Total |
|---|---|---|---|---|
| 1 — host core + Rust extension | 6 | 7 | 8 | 21 |
| 2 — GPU path + CUDA kernels | 3 | 5 | 3 | 11 |
| 3 — streaming / SON, IO, config, CLI | 8 | 14 | 8 | 30 |
| **Total** | **17** | **26** | **19** | **62** |

### Start here

If you fix five things, fix these:

1. **#33** — `apriori_streaming_multi_gpu(n_gpus>=2)` returns **wrong supports** on
   any multi-GPU machine, on the default path, with no fault injection. Measured
   0.667 where the truth is 1.0. One line: `gpu/bitvec.py:65`.
2. **#22** — `anchor_items` loses **95–99%** of the itemsets it advertises in every
   configuration the public API can produce. The fix is validated end-to-end.
3. **#1/#2/#3** — one root cause (non-canonical itemset element order) reaching
   three consumers: 28% of rules silently dropped, and mixed-tier parquets losing
   38% of drop-1 rules while corrupting 39% of self-sufficiency ratios.
4. **#4** — `_sparse_matmul` casts int32 → float32, so k=2 counts saturate at 2²⁴
   at the exact scale this engine targets.
5. **#23/#24** — GPU result buffers truncate silently at any magnitude, and
   non-deterministically.

### Two things this review *disproved*

- **`docs/specs/et_miner_fix_spec.md` Defect A is fixed.** Refuted at HEAD by two
  reviewers independently, one at the spec's exact conditions. That spec is
  checked in, still presents Defect A as open, and points at a `docs/recon/`
  reproduction that no longer exists — **it should be amended in place**, or it
  will keep sending sessions after a defect that commit `9ec6b8c` fixed and
  documented at the site of the fix.
- **Defect B and the `level_callback` gap from the same spec are also fixed.**

### A structural recommendation

`streaming/async_pipeline.py` and `streaming/ramdisk.py` have **zero importers
anywhere in the repository** (verified with a plain `grep -r` including
`.gitignore`d trees, and independently by AST scan). Between them they carry seven
of the 62 defects. Deleting them closes all seven at zero behavioural risk and
removes two of the four drifted min-count copies — see Section 3 for the
conditions.

---

## Summary — Section 1: host core + Rust extension

| # | Severity | Defect | Where |
|---|---|---|---|
| 1 | **HIGH** | Tier 1 emits itemsets in lexicographic column-*name* order — unstable across tiers *and* across thresholds | `core/apriori.py:734` |
| 2 | **HIGH** | `generate_rules()` keys its support map unsorted but queries it sorted — 28% of rules silently dropped, others get `lift = 0.0` | `core/rules.py:43` vs `:81`,`:89` |
| 3 | **HIGH** | Both parquet consumers join positionally: mixed-tier input drops 38% of drop-1 rules and silently corrupts 39% of self-sufficiency ratios | `core/rules.py:340`, `:536`,`:541`,`:545` |
| 4 | **HIGH** | `_sparse_matmul` casts int32 → float32 for MKL; k=2 counts saturate at 2²⁴ | `core/sparse.py:118-121` |
| 5 | **HIGH** | `panic = "abort"` turns every FFI panic into an uncatchable SIGABRT | `rust_ext/Cargo.toml:38` |
| 6 | **HIGH** | `streaming=True` silently discards `prune_equal_support` / `use_generator_pruning` | `core/apriori.py:436-473` |
| 7 | MEDIUM | `profile=True` returns a bare DataFrame on three routes, and it unpacks silently into two Series | `core/apriori.py:437-453`, `:495-512` |
| 8 | MEDIUM | `anchor_items` silently ignored on the CPU and streaming routes | `core/apriori.py:254`, `:495` |
| 9 | MEDIUM | `output_dir` / `resume_from_k` silently dropped outside the row-split route | `core/apriori.py:387-399` |
| 10 | MEDIUM | `memory_budget_gb` silently dropped on the multi-GPU streaming route | `core/apriori.py:441-453` |
| 11 | MEDIUM-HIGH | `_min_count` overshoots by one, dropping itemsets the mandated oracle keeps — and the whole cone above them | `core/result.py:17` |
| 12 | MEDIUM | `compute_self_sufficiency` divides by `max` where its own semantics require `min` | `core/rules.py:541` |
| 13 | MEDIUM | `count_support_batched`'s length filter is set by `itemsets[0]` — order-dependent undercount | `core/matrix.py:317-321` |
| 14 | MEDIUM | The mandated correctness gate derives its oracle threshold from the expression it is meant to check | `tests/test_tier_equivalence.py:73` |
| 15 | LOW | The "O(1) memory" k=2 streaming path materialises the whole pair list | `core/candidates.py:76-77` |
| 16 | LOW | Importing `core.sparse` overrides the process-global MKL thread count | `core/sparse.py:98` |
| 17 | LOW | MKL path discovery probes one hardcoded name at one hardcoded location, misses, and fails open silently | `core/sparse.py:44-47` |
| 18 | LOW | `n_jobs` is dead for k>2 sparse counting whenever the Rust extension is built | `core/sparse.py:458-460` |
| 19 | LOW | `_prune_groups_apriori`'s docs describe neither implementation — in two files, one citing a verification | `gpu/mining.py:262-268`, `groups.rs:471-472` |
| 20 | LOW | `core/result.py` documents a shared-emission-point invariant that the row-split path does not honour | `core/result.py:3-6` |
| 21 | MEDIUM | `compute_self_sufficiency` crashes with `TypeError` instead of returning its documented empty frame — the guard is unreachable | `core/rules.py:548` |

**Defects 1, 2 and 3 are one root cause reaching three consumers.** Fixing the
emission order closes all three. Two of those consumers never touch
`generate_rules`, which is why the defect took two review rounds to size
correctly.

---

## 1. Tier 1 emits itemsets in lexicographic column-name order

**Severity: HIGH.** Confirmed by all four reviewers; measured by three.

`build_boolean_matrix` names its columns `i_0 … i_N` with no zero padding
(`core/matrix.py:136`), assigning them positionally over the *surviving*
frequent items sorted by item id (`core/matrix.py:121-137`). Every ordering
decision in the CPU path is then a **string** comparison over those names —
`sorted(p[0] for p in prev_frequent)` (`core/candidates.py:71`, also `:117`,
`:157`) and `pl.col("a") < pl.col("b")` (`core/candidates.py:94`, `:224`).
From 11 frequent items on, `"i_10" < "i_2"` while `item_ids[10] > item_ids[2]`,
and `core/apriori.py:734` maps that order straight to item ids with no re-sort.
`core/result.py:33-38` does not sort either.

```
sorted(['i_1','i_10','i_2']) -> ['i_1', 'i_10', 'i_2']
```

Measured on the emitted frames, twice independently:

```
500 transactions / 15 items, min_support=0.1, max_length=3:
  total itemsets: 212   unsorted emitted: 114   e.g. [[10, 2], [10, 3], [10, 4], ...]

300 transactions / 15 items, min_support=0.05:
  cpu unsorted rows: 910 / 1335
  gpu unsorted rows:   0 / 1335
  same as sets: True
```

**This is a defect in its own right, not merely a trigger for #2.** The order is
consistent within a run but *not across runs*, because an item's column index
depends on how many other items cleared the threshold below it — i.e. on
`min_support`. Same data, same itemset:

```
min_support=0.8 (2 frequent items):  {2,10} emitted as [[2, 10]]
min_support=0.3 (15 frequent items): {2,10} emitted as [[10, 2]]
```

So any dict key, parquet join, stored artifact or frozen digest keyed on the
itemset list silently stops matching when the threshold moves. That lands
directly on this repository's frozen-reference workflow.

It also means CLAUDE.md's mandated equivalence chain is **false on the returned
frames** and only true after `tests/test_tier_equivalence.py:44` canonicalises
with `tuple(sorted(int(i) for i in itemset))`. The gate meant to enforce the
invariant is what hides its violation. The GPU path is the correct one:
`gpu/row_split.py:671` maps a monotone `col_to_item_arr` over integer column
indices and emits ascending item ids.

Two docstrings in `core/rules.py` assert the invariant that this breaks:
`:173-174` ("The antecedent list is already sorted because the input itemsets
are sorted") and `:263-264` ("The K and K-1 parquets must be sorted
lexicographically (which is the default output of et-miner's apriori
pipeline)"). That parenthetical is true of the GPU route and false of the CPU
route. `:267-268` is not a third assertion but the *mechanism* that makes the
invariant load-bearing — it describes the positional scalar-column unpack that
#3 exploits.

**What this does NOT break** — verified, so the report does not overclaim: the
mining itself is correct. The string order is a consistent total order *within*
a run, so the prefix join, the apriori subset test and the free-set test are all
order-invariant — checked against brute force over 10 seeds on nested-vocabulary
data through K=5, and the full CPU lattice against `efficient-apriori` across 8
parameter variants. The defect is entirely in what is *emitted*, not in what is
computed. That distinction is why the defect was initially under-rated; see
*Why the suite doesn't catch any of this*.

**Fix — three parts. Parts 2 and 3 are what stop it coming back.**

1. **Root cause, one line.** Pad the column names at `core/matrix.py:136`:
   `f"i_{idx:0{width}d}"` with `width = len(str(len(item_ids)))`. Since `idx` is
   monotone in item id, padding makes lexicographic order *identical* to item-id
   order, so `sorted()` at `core/candidates.py:71` and the
   `pl.col("a") < pl.col("b")` filters at `:94` and `:224` become correct with no
   change to the join logic — the CPU route then agrees with the GPU route by
   construction rather than by a second sort bolted on at emission. Verified
   safe: nothing parses these names back to an index (`col_to_item` is a dict
   lookup, `_polars_to_sparse_csr` derives its mapping positionally at
   `core/matrix.py:430`, `count_support_vectorized` uses them only as alias
   fragments at `:188`).

   **No test breaks — established by full enumeration, not by sampling.** There
   are 13 `build_boolean_matrix` call sites in the suite
   (`tests/test_streaming_bug.py:34`, `:64`, `:88`, `:124`, `:153`;
   `tests/test_sparse.py:334`; `tests/test_apriori.py:210`, `:226`, `:239`,
   `:248`, `:261`, `:273`, `:288`). Ten derive their column names from the
   returned `col_to_item` and are safe at **any** item count; two
   (`tests/test_apriori.py:239`, `:248`) are the no-frequent-items and
   empty-input cases and assert only `len(col_to_item) == 0`, so they never touch
   a name. **Exactly one is coupled to the name format**:
   `tests/test_streaming_bug.py:153` calls the
   function and then hardcodes `i_0`/`i_1`/`i_2` at `:155`, `:161-163` and `:165`.
   It is safe only because its fixture at `:150-151` has three items, so the width
   stays 1 — **growing that fixture past 9 items would break it**, which is the
   one place the "small fixture" reason is load-bearing rather than incidental.
   Every other `i_N` literal in the suite lives in `tests/test_sparse.py`, in
   self-constructed fixtures that never call the function — including the
   100-column frame at `:160-174`, whose `("i_5","i_10")` at `:183` names its own
   columns and is not width-sensitive at all.

   The width depends on how many items were frequent, so it varies run to run.
   That is harmless — ordering only needs the width constant *within* a run — but
   surprising enough to deserve a comment at the site, or someone will simplify
   it back. Suggested wording: *the width depends on `len(item_ids)` and is
   deliberately internal; what must stay true is that padded-name order equals
   item-id order, which requires only that the width be constant within a run.*

   **The fix was applied and the suite run — with a control.** Padding was
   patched into `build_boolean_matrix` at every import site:

   ```
   471 passed, 4 skipped, 10 deselected in 77.56s     <- identical to baseline

   control, patch live:     [padplugin] padding active
                            frequent items: 15 | first/last col names: i_00 i_14  -> 1 passed
   control, patch absent:   frequent items: 15 | first/last col names: i_0  i_14
                            E  AssertionError: padding NOT active: got i_0
   ```

   The control fails without the patch, so the green is real rather than the
   green an inoperative patch would also produce. See *Why the suite doesn't
   catch any of this* for what that identical `471 passed` actually proves.

   Carrying integer column indices instead of stringified names is the durable
   follow-up — padding leaves the representation string-based, so a future naming
   change reintroduces the defect — but it is not required to fix this.

**The single-root-cause claim is confirmed by construction.** Padding was
implemented as a wrapper and measured: it closes #1, #2 and #3 simultaneously,
with the lattice bit-identical and the rule count landing exactly on the
order-insensitive figure measured independently in round 1.

```
  unpadded   itemsets=1335  unsorted=910  rules= 9722
  PADDED     itemsets=1335  unsorted=  0  rules=13580
  lattice unchanged by the fix: True
  mixed-tier drop1 [unpadded] CPU K3 + GPU K2 ->  845/1365
  mixed-tier drop1 [PADDED  ] CPU K3 + GPU K2 -> 1365/1365
  padded sparse==padded polars: True | padded polars==GPU: True
```

**Padding makes the emitted order canonical, not merely consistent — so it closes
#1 in full, including the cross-threshold half.** This was disputed during review
and is worth stating positively, because a reader who believes half of #1 survives
will either bolt a redundant sort onto emission or leave the defect open. Two
facts compose: `core/matrix.py:135` takes `item_ids` from `freq_1.sort("item")`,
so the column index is **monotone in item id** whatever set of items cleared the
threshold; and with the width constant within a run, lexicographic order over the
padded names equals numeric order over the indices. Therefore sorted-by-padded-name
order **is** ascending-item-id order — for any `min_support`, any item count, any
route. Verified directly on the report's own example:

```
min_support=0.8 (2 frequent):  unpadded -> [2, 10]   padded (i_0,i_1)   -> [2, 10]
min_support=0.3 (15 frequent): unpadded -> [10, 2]   padded (i_02,i_10) -> [2, 10]
```

Same list at both thresholds, and the same list the GPU route emits. What still
varies with the threshold is the internal *name* (`i_2` vs `i_02` for one item),
and names never leave the function — `col_to_item` resolves them to item ids at
`core/apriori.py:734` before anything is emitted. The residual instability has no
external surface.
2. **Write the contract down.** `apriori()`'s Returns block
   (`core/apriori.py:318-320`) commits to nothing about element order. The order
   is currently stated *only* as a downstream assumption, at `core/rules.py:173-174`
   and `:263-264` — two consumers asserting an invariant no producer promises and
   no test checks. State the canonical form producer-side.
3. **Enforce in the gate additively.** Add a per-tier assertion alongside the
   existing comparison — `assert all(list(x) == sorted(x) for x in
   result_df["itemset"].to_list())` — and **leave `_result_to_counted_set`
   untouched.**

> ⚠ **Do not "fix" this by removing the `sorted()` at
> `tests/test_tier_equivalence.py:44`.** That was recommended during review and
> then withdrawn, for two reasons. CLAUDE.md says of those assertions "Never
> skip, weaken, tolerance-relax, or replace these assertions with count-only
> checks", and removing a canonicalisation that insulates the gate from a third
> party makes it *more* fragile. It would also only work by accident:
> efficient-apriori happens to emit ascending keys because it builds from sorted
> transactions, which is an undocumented property of a dependency. The first time
> that changes, the gate goes red for reasons unrelated to ET-Miner and someone
> re-adds the `sorted()` to get a run green.

*Note on where to enforce:* `core/result.py::_build_result_df` looks like the
shared emission point and is documented as one, but is not — see #20. Normalising
there is still a workable alternative to padding, but for a narrower reason than
it appears: it covers every route that is *currently broken* (CPU direct,
`gpu/mining`, SON streaming, multi-GPU streaming), and the one route it misses —
row-split, at `gpu/row_split.py:739-744` and `:751-756` — already emits ascending
item ids. **That leaves the reference-correct route as the only one not guarded
in-code, which is precisely the route you most want protected against
regression.** It is why part 3 is load-bearing rather than belt-and-braces: the
gate already exercises all seven routes, so the per-tier assertion is the only
mechanism that covers row-split.

---

## 2. `generate_rules()` silently drops rules and reports `lift = 0.0`

**Severity: HIGH.** Confirmed by all four reviewers; measured by two.

`_build_support_lookup` keys the map on the itemset's **stored** order:

```python
core/rules.py:43   return {tuple(row["itemset"]): row["support"] for row in ...}
```

but both lookups query it in **sorted** order:

```python
core/rules.py:81   lhs_support = support_map.get(tuple(sorted(lhs)), 0.0)
core/rules.py:89   rhs_support = support_map.get(tuple(sorted(rhs)), 0.0)
```

Given #1 those differ, producing two distinct silent failures: an `lhs` miss
returns `0.0` and the rule is discarded by the `continue` at `core/rules.py:82-83`;
an `rhs` miss returns `0.0` and `core/rules.py:90` emits `lift = 0.0` — a
plausible-looking wrong number rather than a dropped row.

```
500 transactions / 15 items, min_support=0.1, max_length=3, min_confidence=0.0:
  rules produced: 628   rules expected: 762   DROPPED: 134
  rules with WRONG lift: 134
  e.g. ([0], [3, 11], reported lift 0.0, true lift 1.0926573426573427)

300 transactions / 15 items, min_support=0.05:
  rules from generate_rules: 9722   expected (order-insensitive): 13580   (28% lost)
```

At `min_confidence=0.0` the only reachable `continue` is the
`lhs_support == 0.0` branch, so the key mismatch is the whole cause.

**Reachable through the shipped CLI in two calls**: `cli.py:188` runs `apriori()`,
`cli.py:200` passes the result straight to `generate_rules()`.

> **Do not quote this rate across producers, and note the real fix site.** The
> masking effect in #35 is asymmetric: SON frames carry duplicate rows that
> accidentally supply the sorted key, so SON is the *less* affected producer. The
> direct path has a single column mapping, cannot emit duplicates, and runs at full
> severity. Measured on direct-path frames at `max_length=3`, comparing
> `generate_rules` as-emitted against the same frame with every tuple sorted:
>
> ```
> deterministic  377 itemsets, 180 k>=2 and NOT ascending
>    as-emitted 1608 rules  vs sorted 1872  -> 264 rules missing outright (14%)
>    plus 264 of the 1608 survivors carry lift == 0.0 (16.4%)
>
> Zipf           9788 itemsets, 4965 k>=2 and NOT ascending
>    as-emitted 7250 rules  vs sorted 9670  -> 2420 rules missing outright (25%)
>    0 survivors with lift == 0.0 on this frame
> ```
>
> **Two distinct failure modes from one defect**, and which dominates is
> frame-dependent: an LHS miss at `:81` drops the rule entirely at `:82-83`; an RHS
> miss at `:89` leaves it with `lift = 0.0` at `:90`.
>
> Sorting at the SON merge (#35) fixes only SON frames. Keying
> `_build_support_lookup` by `tuple(sorted(...))` at `core/rules.py:43` repairs
> every producer at once.
>
> **But that does not displace #1's fix, and the two are not alternatives.**
> `rules.py:43` fixes #2 only. #1's harm is not confined to `generate_rules` — #3
> documents two consumers that never call it, which is why the defect took two
> review rounds to size correctly, and a change inside `rules.py` leaves those
> untouched. Separately, a public API returning `[[10, 2], [10, 3]]` is wrong on
> its own terms whatever any consumer does with it. So: **`core/rules.py:43`
> because a lookup should not depend on its producer's tuple ordering, and
> `core/matrix.py:136` because the documented return value is a set of itemsets.**
> Neither subsumes the other; fixing the root cause does not remove the reason to
> harden the consumer.

#1 and #2 are independently sufficient causes and either fix alone closes the
rule loss. Both should still be fixed — #1 for the cross-run instability, #2
because a lookup helper that disagrees with its own callers is a trap for the
next caller.

---

## 3. Both parquet consumers join positionally — mixed-tier artifacts silently lose rows and corrupt numbers

**Severity: HIGH.** Measured. This is #1's root cause reaching two **further**
consumers, neither of which contains a `sorted()` anywhere — so neither is the
#2 key mismatch.

`_list_to_scalar_cols` (`core/rules.py:157-159`) unpacks a list column into
scalar join keys **by position** (`list.get(j)`), and `_explode_drop1`
(`:209-215`) builds antecedents with `list.gather(...)`, which preserves the
source itemset's element order rather than re-sorting. The joins are then plain
positional equality with `how="inner"`. A (K-1)-subset extracted from a
CPU-produced K-itemset carries the CPU's element order; the same subset stored
in a GPU-produced (K-1) parquet carries ascending item ids. The keys miss, and
an inner join on a missing key is indistinguishable from a genuinely absent
subset.

**Two consumers, two complementary failure modes, one root cause.**

**(a) `generate_rules_drop1` — loses rows, keeps every surviving number correct**
(`core/rules.py:340`, with `:157-159`, `:209-215`, `:305-310`, `:337`):

```
800 transactions / 15 items, min_support 0.05; K=2 and K=3 mined twice,
once via the CPU route and once with use_gpu=True:

K=3 sets identical across tiers: True | K=3 size: 454
  CPU K3 + CPU K2            rules=1362/1362  dropped=  0  wrong antecedent_support=0
  GPU K3 + GPU K2            rules=1362/1362  dropped=  0  wrong antecedent_support=0
  CPU K3 + GPU K2 (MIXED)    rules= 842/1362  dropped=520  wrong antecedent_support=0
  GPU K3 + CPU K2 (MIXED)    rules= 842/1362  dropped=520
distinct CPU antecedent tuples: 105 | present in GPU K2 keyset: 65 | absent: 40
```

520 of 1362 — 38% — lost in both mixing directions. The drops are pure: a
canonical-form mismatch always *misses*, never collides, and there is a
structural reason it must — the join key **is** the ordered tuple of item ids,
so a collision would require two different itemsets to unpack to the same tuple.

**(b) `compute_self_sufficiency` — loses nothing, silently corrupts 39% of the
numbers** (`core/rules.py:494`, `:496-501`, `:533`, `:536`, `:541`, `:545`):

```
  CPU K3 + CPU K2            rows= 454/454  itemsets DROPPED=0
  CPU K3 + GPU K2 (MIXED)    rows= 454/454  itemsets DROPPED=0
  surviving rows with a WRONG max_k_minus1_support under mixing: 175 of 454
   e.g. [((0, 4, 12), 0.20625 -> 0.205), ((8, 9, 11), 0.2025 -> 0.1875)]
```

No itemset is dropped, because the `group_by` at `:541` keeps an itemset as long
as **one** of its subsets survives the inner join at `:536`; the max is then
taken over a truncated survivor list.

**A second failure mode exists, and it is now measured.** The inner join at
`:545` (`chunk.join(grouped, on="_row_idx", how="inner")`) loses *every* subset
of a K-itemset when the stored tuple is strictly reverse-sorted by item id. With
unpadded `i_N` names that needs indices spanning several digit lengths in
decreasing magnitude (`i_200 < i_30 < i_4` lexicographically while 200 > 30 > 4),
which requires roughly **≥201 frequent items** — so the 0/454 above is a fixture
limit (15 items), not evidence of absence. Constructed and confirmed:

```
frequent items: 248 | K=2: 3 | K=3: 1
CPU stored K=3 tuples: [([200, 30, 4], 0.35)]
GPU stored K=3 tuples: [([4, 30, 200], 0.35)]

  self_sufficiency CPU K3 + CPU K2          rows=1/1  DROPPED=0
  self_sufficiency CPU K3 + GPU K2 (MIXED)  RAISED TypeError
  drop1            CPU K3 + GPU K2 (MIXED)  rules=0/3
```

**And it does not manifest as the silent missing row that was predicted — it
manifests as an unhandled crash**, because the empty-result guard it lands in is
unreachable (#21).

**Mechanism, stated correctly: the driver is the frequent-item COUNT, not the
item ids.** The column names are built over indices `0..N-1`
(`core/matrix.py:136`), not over item ids, so what matters is how many items
cleared the threshold. (In the 248-item fixture above the ids happened to equal
the indices, which made an id-based framing look right.) Correct thresholds:

- **K=3** requires the tuple to be *fully* reversed, which needs three
  digit-length classes of column index — i.e. **≥101 frequent items**.
- **K≥4** requires only two classes — i.e. **≥11 frequent items**.

So the K≥4 modes are reachable at essentially **any realistic vocabulary**, not
only above ~201 items. That widens the trigger considerably.

*Closed form that explains the boundary* (a derivation resting on the
measurements below, necessary but not sufficient — offered as an explanation of
the curve, not as an independent result): within one digit-length class all names
share a width, so lexicographic order equals index order and any subset drawn
entirely from one class joins successfully. A row vanishes only if **no class
holds K−1 of the K elements**, i.e. every class holds ≤ K−2, which requires at
least `C ≥ K/(K−2)` digit-length classes. That gives C ≥ 3 at K=3 and C ≥ 2 for
every K ≥ 4 — exactly the two measured thresholds, and it explains why 0% → 27%
is a hard structural boundary rather than gradual scaling. The K=4 case is a 2+2
split over indices `0..14`, e.g. `{8,9,10,11}`, where every cross pair inverts
because `"i_10" < "i_8"`. Tested per-itemset against the measured vanishing:

```
frequent items=15 -> column indices 0..14, digit classes C=2
  K=3: vanished&predicted=0  predicted-not-vanished=0  vanished-not-predicted=0  -> necessary: True,  sufficient: True
  K=4: vanished&predicted=4  predicted-not-vanished=2  vanished-not-predicted=0  -> necessary: True,  sufficient: False
  K=5: vanished&predicted=2  predicted-not-vanished=0  vanished-not-predicted=0  -> necessary: True,  sufficient: True
```

No counterexample to the necessary direction at any K; sufficiency fails at K=4
with two false positives, as predicted in advance. The four vanishing K=4
itemsets — `[10,13,4,7]`, `[11,14,2,5]`, `[11,14,2,8]`, `[11,14,5,8]` — are each
a 2+2 split across the two digit classes, the exact configuration the derivation
calls for. **The practical consequence: from K=4 up the vanishing mode needs only
11 frequent items**, not the ~101 or ~201 figures that circulated earlier — so it
is available at every level and vocabulary a real campaign uses.

**And the damage grows steeply with K.** Measured on a 15-frequent-item fixture
with levels to K=5:

```
 K     n   VANISH (all subsets miss)   WRONG number   intact
 3    30                           0             23        7
 4    15                           4             10        1
 5     3                           2              1        0

     K=3 -> self_sufficiency rows=30/30   drop1 rules=51/90
     K=4 -> self_sufficiency rows=11/15   drop1 rules=14/60
     K=5 -> self_sufficiency rows= 1/3    drop1 rules= 1/15
```

Vanishing fraction 0% → 27% → 67%; drop-1 rule loss 43% → 77% → 93%.
**The 38% and 39% figures quoted above are measured at K=3 and are a floor, not
a typical value.** This engine exists for K=7–8 campaigns, where the vanishing
mode is expected to dominate.

**This is the worse of the two failure modes**, because of detectability: a
missing row is catchable by comparing the output row count against the K
parquet; a wrong `max_k_minus1_support` is catchable by nothing. The output
looks intact.

**Relationship to #12 — settled: these do NOT compound.** With subsets `S`
present in the K-1 parquet and survivors `V ⊆ S`:

```
ratio_clean    = sup_K / max(S)   <- clean single-route output
ratio_mixed    = sup_K / max(V)   <- mixed-tier output
ratio_intended = sup_K / min(S)   <- what core/rules.py:462-468 describes
```

`max(V) ≤ max(S)` gives the measured inflation, but `max(V) ≥ min(V) ≥ min(S)`
bounds it above, so `ratio_clean ≤ ratio_mixed ≤ ratio_intended`. Verified on all
454 rows: 0 violations of either inequality. The corruption moves the number
monotonically *toward* the intended value and provably cannot overshoot it. Nor
do the two compound on the action axis: #12 errs toward keeping (conservative),
mixed-tier inflation errs toward discarding (destructive).

**How far that goes, measured:**

```
rows where the MIXED (corrupted) ratio lands EXACTLY on the intended min-based value: 104/454
   — i.e. 104 of the 175 corrupted rows, 59%
rows where the CLEAN single-route ratio equals the intended min-based value:            0/454
```

Whenever the surviving subset set happens to reduce to the minimum-support one,
`max(V) = min(S)` and the two wrongs land exactly on the right answer. So on 59%
of corrupted rows the corrupted artifact agrees with the metric the docstring
describes, while the clean artifact agrees on none.

**This is not an argument that the corruption is benign.** Those rows are
"correct" only relative to the min-based metric the code does not implement, and
they are closer by a data-dependent amount nobody can bound; on the other 350
rows the error is unbounded in the opposite direction. The honest summary:
*single-route output is systematically wrong in one direction; mixed-route output
is wrong by an unbounded data-dependent amount in the other; neither is
interpretable, and only the second is undetectable.* The practical consequence is
in #12's verification note — a corrupted run can look **more** plausible than a
clean one.

**Why this is reachable, not theoretical.** `output_dir` is dropped on the CPU
route (#9 — measured "files written: []"), so a CPU-mined level has to be
written by hand with `write_parquet`, which is the obvious thing to do and which
`generate_rules_drop1` invites: it documents its input purely by schema
("Schema: itemset (list[i32]), support (f64)", `core/rules.py:276-279`) with no
route qualification. `resume_from_k` is the second door — it reloads a level a
previous, possibly differently-routed run wrote.

The framing that matters: **this is an invariant enforced on one side of a
boundary and assumed on the other.** The in-memory side canonicalises with
`tuple(sorted(...))` at `tests/test_tier_equivalence.py:44` and certifies the
two levels equal; the on-disk side joins them positionally at
`core/rules.py:157-159` and `:340`.

---

## 4. `_sparse_matmul` casts int32 → float32; k=2 counts saturate at 2²⁴

**Severity: HIGH.** Confirmed by all four reviewers; measured independently by three.

```python
core/sparse.py:118-121
    if A.dtype not in (np.float32, np.float64):
        A = A.astype(np.float32)
    if B.dtype not in (np.float32, np.float64):
        B = B.astype(np.float32)
    return dot_product_mkl(A, B)
```

The CSR data is `np.int32` (`core/matrix.py:464`, chosen deliberately — the
comment notes "uint8 overflows at 256"), so the cast always fires. float32 has a
24-bit significand: once the accumulator reaches 2²⁴ = 16,777,216, `acc + 1`
rounds back to `acc` and the sum sticks there. The result is read back with
`int(cooccur[i, j])` at `core/sparse.py:508` and `:548`.

```
20,000,000 rows, two fully-populated columns (true pair count = 20,000,000):
  _count_support_sparse_k2_batch -> {('i_0','i_1'): 16777216}
  scipy reference (int32)        -> 20000000
```

An error of −3,222,784 (−16.1%), and it is an **under**-count: because the level
filter is `count >= min_count`, genuinely frequent pairs are silently dropped and
the loss cascades into every higher K. Between 2²⁴ and 2²⁵ the representable
spacing is 2, so counts in that range are wrong by ±1 even without saturating.

The scipy fallback in the same function (`core/sparse.py:123-124`, `A @ B`) keeps
int32 and is exact — so the two branches of one function disagree at scale.

**Trigger is not exotic.** `_choose_counting_strategy` (`core/sparse.py:765-785`)
auto-selects the sparse path as soon as there are more than ~448 frequent items,
or when the dense estimate exceeds 1 GB — which at 205M rows happens at ~45
items. This is precisely the regime the module's own docstrings target
(205M × 1006).

> **Scope — read this before trying to reproduce.** The defect fires **only when
> `sparse_dot_mkl` imports successfully**. Where MKL is not on the loader path
> the `except ImportError` fallback at `core/sparse.py:123-124` runs instead and
> counts are exact. Both branches, measured in one process on the same 20M-row
> fixture:
>
> ```
> true count: 20000000
> MKL branch (this box):      {('i_0', 'i_1'): 16777216}
> scipy fallback branch:      {('i_0', 'i_1'): 20000000}
> fallback exact: True | MKL branch exact: False
> ```
>
> The same function is exact or catastrophically wrong depending purely on
> whether an optional import resolves. On a clean pip venv where the stale probe
> in #17 prevents that import, you will see correct counts — that is not a
> refutation of this bug. On this box, which is the campaign box, the MKL branch
> is the live one and the counts are wrong.

**Fix, verified rather than asserted** — same 20M-row fixture, three branches:

```
current  (float32): 16777216
proposed (float64): 20000000
scipy int32       : 20000000
true              : 20000000
```

float64 is exact for every integer below 2⁵³. Better still, skip MKL for integer
dtypes and fall through to `A @ B`, which scipy evaluates in int32 — exact to
2,147,483,647, comfortably above any guarded transaction count.

**⚠ Fix ordering: this must land before or with any fix to #17.** Repairing the
MKL path-discovery probe in isolation would *activate* this saturation for
installs that currently return correct answers — the LOW fix makes the HIGH bug
newly reachable.

---

## 5. `panic = "abort"` turns every FFI panic into an uncatchable process kill

**Severity: HIGH.** Confirmed by all four reviewers; reproduced independently by two.

```toml
rust_ext/Cargo.toml:38   panic = "abort"       # Smaller binary, no unwinding overhead
```

That sits under `[profile.release]`, and CLAUDE.md's documented build is
`maturin develop --release`, so the shipped extension always aborts. PyO3's
error model depends on unwinding to produce a `PanicException`; `abort` removes
it. Five ordinary-looking inputs, each in its own subprocess, all exit 134
(SIGABRT) with no traceback and nothing catchable by `try/except`:

```
=== noncontig ===          panicked at src/lib.rs:55:28: called `Result::unwrap()` on an `Err` value: NotContiguousError   exit=134
=== ncols_small ===        panicked at src/core/matrix.rs:36:23: index out of bounds: the len is 1 but the index is 1      exit=134
=== nrows_big ===          panicked at src/core/counting.rs:38:27: index out of bounds: the len is 3 but the index is 3    exit=134
=== prune_len_mismatch === assertion `left == right` failed: current_counts length mismatch                               exit=134
=== bitvec_zero_rows ===   panicked at src/core/bitvec.rs:48:23: index out of bounds: the len is 0 but the index is 0      exit=134
```

Vectors: any non-C-contiguous numpy array (`arr[::2]`, an F-order array, a
strided view) to `count_itemsets_sparse` / `count_itemsets_simd` /
`build_column_bitvecs_u64` / `bitvec_to_tidsets`; `n_rows=0`; an `n_cols`
smaller than the largest column index; an `n_rows` larger than `len(indptr)-1`;
a `current_counts` length mismatch in `prune_non_free_flat`. Further abort
sites: `groups.rs:71-78` and `groups.rs:495` assert on shapes arriving from the
GPU path.

**This is a bug, not an accepted design choice**, because it is not applied
consistently and the codebase already depends on it not being true:

- `lib.rs` implements an explicit non-contiguous copy fallback **five** times —
  `build_k3plus_groups_from_flat` (`:236-243`), `prune_non_free_flat` (`:284-317`),
  `prune_non_free_flat_compact` (`:367-400`), `unique_columns_from_flat` (`:441-447`)
  — and omits it **six** times in the same file (`:54-57`, `:78-79`, `:97-98`,
  `:111`, `:124`, `:191-192`). Half the surface was hardened against exactly this.
- `lib.rs:319-324` raises a proper `PyValueError` for an unsorted `prev_flat`,
  then the very next call aborts the process for a length mismatch.
- `gpu/mining.py:324-329` wraps a Rust call in `except (ImportError, AttributeError)`
  / `except TypeError` with a stale-wheel fallback. That entire recovery path is
  dead under `abort`.

An error-handling contract asserted on the Python side and made unimplementable
on the Rust side is a boundary defect regardless of the performance argument.

**Impact.** A long GPU campaign loses every unflushed level, with no traceback
to say why.

**Reachability caveat, for honesty:** the shipped call site
`_count_support_sparse_k_gt_2_rust` (`core/sparse.py:425-426`) does
`csr.indptr.astype(np.int64)`, which always produces a contiguous copy, so the
non-contiguous vector is not reachable *there*. The exposure is the other entry
points and any direct use of the extension.

---

## 6. `streaming=True` silently discards `prune_equal_support` and `use_generator_pruning`

**Severity: HIGH.**

The `if streaming:` branch at `core/apriori.py:436` is evaluated *before* either
consumer of `_route_for_pruning` (`:384` and `:495`), and neither callee accepts
the parameter — `apriori_streaming` (`streaming/son.py:79-95`) and
`apriori_streaming_multi_gpu` (`streaming/multi_gpu.py:113-126`) have no such
argument. The caller asks for free-sets and receives the complete frequent
lattice, with nothing in the return value or the logs to say so.

This is the exact contract break commit `9ec6b8c` set out to eliminate, still
live on this route. The file already demonstrates the correct pattern: an
explicit `ValueError` at `core/apriori.py:337-342` for the same class of
unsupported combination. With `use_gpu=False` that guard does not fire at all,
so the failure is completely silent; with `use_gpu=True, profile=True` the
caller instead gets an error that blames `profile`, which is not the problem.

**Fix.** Hoist the unsupported-combination check above the `if streaming:`
branch and raise the same explicit `ValueError` the file already raises.

---

## 7. `profile=True` returns a bare DataFrame on three routes — and it unpacks silently

**Severity: MEDIUM.** Measured.

The signature and docstring promise a 2-tuple (`core/apriori.py:255`, `:318-320`:
"If profile=True: tuple of (DataFrame, ProfilingSession)"). Three routes return
a bare frame instead: `streaming` + `n_gpus>1` (`:437-453`, callee declared
`-> pl.DataFrame` at `streaming/multi_gpu.py:126`), and the row-split route with
`n_gpus>1` or with `anchor_items` (`:495-512`, callee declared `-> pl.DataFrame`
at `gpu/row_split.py:70`). The guard that exists for exactly this at `:337-342`
covers only `_route_for_pruning`.

Because a result frame has exactly two columns, the failure is worse than an
exception:

```
profile=True + n_gpus=2 -> type: DataFrame | is tuple: False
unpack succeeded. type(result)= Series  type(session)= Series
session.summary exists? False
result[:3] = [[0], [1], [2]]
```

`result, session = apriori(...)` succeeds and hands back two Series — the
supports silently gone, the "session" a column of floats.

---

## 8. `anchor_items` is silently ignored on the CPU and streaming routes

**Severity: MEDIUM.** Measured.

`anchor_items` appears exactly three times in `core/apriori.py`: the signature
(`:254`), the routing test (`:495`) and the forwarded kwarg (`:511`). It appears
nowhere in the CPU loop (`:546-761`), and the `streaming` branch at `:436`
returns before `:495` is ever reached.

```
anchor_items on CPU: full= 304  anchored= 304  identical: True
anchor_items on GPU: anchored= 83
```

The caller believes it asked for two-phase anchor-restricted mining and gets the
complete lattice — a much larger, differently-shaped result, silently.

---

## 9. `output_dir` and `resume_from_k` are silently dropped outside the row-split route

**Severity: MEDIUM.** Measured.

The `bitvecs=` + pruning route (`core/apriori.py:387-399`) forwards
`bitvecs_list`, `prune_non_free`, `prune_apriori` and `sparse_from_k` — and
neither `output_dir` nor `resume_from_k`. The sibling call at `:498-512` *does*
pass both, proving the callee accepts them. On the CPU path both are ignored
entirely.

```
output_dir + single-GPU bitvec path -> files written: []
output_dir + CPU path               -> files written: []
resume_from_k=3 on single-GPU path  -> rows: 304   (identical to the no-resume run)
```

`output_dir` is documented as the per-K parquet flush that "prevents CPU RAM OOM
at K=7+ scale" (`core/apriori.py:244`). A user who sets it on these routes gets
no flush, the OOM it was meant to prevent, an empty output directory and no
diagnostic. `resume_from_k` silently re-mines from K=1 — on a multi-day
campaign. This is also the door through which #3 becomes reachable: with no
flush on the CPU route, levels get written by hand.

---

## 10. `memory_budget_gb` is silently dropped on the multi-GPU streaming route

**Severity: MEDIUM.**

`core/apriori.py:441-453` enumerates every forwarded kwarg and
`memory_budget_gb` is not among them; `streaming/multi_gpu.py:113-125` has no
such parameter, while `streaming/son.py:85` does. The docstring at
`core/apriori.py:296-297` says it "Overrides chunk_size. Only used when
streaming=True". The run sizes its chunks against a budget the user never chose
— on the one path whose reason to exist is not exceeding memory.

**Related, same shape (#7–#10):** `level_callback(k, n_candidates, n_frequent,
duration_ms)` is documented at `core/apriori.py:300-301` with no route
qualification, but reports a route-dependent quantity. The GPU group path
over-approximates the subset test where the CPU path does not (see negative
results), so the two routes report different candidate counts for the same input
at the same level; and per `docs/specs/et_miner_fix_spec.md:151-159` the
row-split dense branch passes `n_candidates = 0` for every K ≥ 2. A consumer
doing per-level cost accounting receives a real count, a differently-defined
count, or a zero, with no way to tell which.

---

## 11. `_min_count` overshoots by one, dropping itemsets the mandated oracle keeps

**Severity: MEDIUM-HIGH.** Confirmed by all four reviewers; measured end-to-end
against the oracle at two scales.

```python
core/result.py:17                  return math.ceil(min_support * n_transactions)
rust_ext/src/core/apriori.rs:73    let min_count = (min_support * n_rows as f64).ceil() as u32;
src/et_miner/synthetic.py:71-72    return math.ceil(self.min_support * self.n_rows)
```

`0.07` has no exact binary64 representation, so the product can land just above
the exact value and `ceil` returns one too many:

```
0.07 * 100    = 7.000000000000001      math.ceil -> 8    (exact ceiling: 7)
0.07 * 10000  = 700.0000000000001      math.ceil -> 701  (exact ceiling: 700)
0.07 * 100000 = 7000.000000000001      math.ceil -> 7001 (exact ceiling: 7000)
```

The excess exceeds half an ulp near those values, so it rounds up and cannot
fall back. CLAUDE.md documents the rule as `count >= ceil(s·N)`. What is
implemented is `count >= ceil(fl64(s)·N)`, which can be one higher.

**The loss is not one itemset — it is the whole cone above it:**

```
10,000 transactions, item 0 in exactly 700 rows (support exactly 0.07):
  _min_count(0.07,10000) = 701 (exact ceil = 700)
    Tier1 polars       item 0 present: False   n_itemsets=3
    Tier2 rust/sparse  item 0 present: False   n_itemsets=3
    GPU                item 0 present: False   n_itemsets=3
    efficient-apriori  item 0 present: True    n_itemsets=7
  ORACLE ITEMSETS MISSING FROM MINER: [(0,), (0, 1), (0, 1, 2), (0, 2)]
```

Reproduced independently at N=100 with the same signature. Sweeps found 48
divergent (N, s) pairs in one range and 31 in another (N=100..1000, s=0.01..0.49).

All three tiers agree with each other and all three disagree with
`efficient-apriori`, whose criterion is `count / len(transactions) >= min_support`
(`efficient_apriori/itemsets.py:285`) and `700/10000 >= 0.07` is True.
**Under CLAUDE.md's own mandatory correctness policy, which names
efficient-apriori as *the* canonical oracle, that settles the question of
intended contract.**

Latent rather than live on the current presets: `smoke` is unaffected, because
`0.01 * 60000` has excess `1.25e-14` against a half-ulp of `5.68e-14` near 600
and rounds to exactly 600.0.

**⚠ Fix ordering: all three sites must land in one commit.**
`core/result.py:17`, `rust_ext/src/core/apriori.rs:73` and `synthetic.py:71-72`
currently agree **because they are the same wrong expression**. Fixing one alone
breaks the tier-equivalence chain at exactly the (s, N) pairs where the error
lives. Note `SynthSpec.min_count` also feeds `check_preset_purpose`'s vacuity
guard. Fix: `math.ceil(Fraction(str(min_support)) * n_transactions)` — `str()`
on a float is the shortest round-tripping decimal, so `Fraction(str(0.07)) ==
7/100` exactly. Add one oracle case at an exactly-on-threshold count to the
smoke gate (see #14).

---

## 12. `compute_self_sufficiency` divides by `max` where its own semantics require `min`

**Severity: MEDIUM.** Confirmed by all four reviewers; measured.

```python
core/rules.py:459-460   formula, stated as max(...)
core/rules.py:462-468   interpretation
core/rules.py:541       pl.col("km1_support").max().alias("max_k_minus1_support")
core/rules.py:566-568   ratio = support / max_k_minus1_support
```

The documented interpretation is: "A ratio close to 1.0 means the K-th item adds
almost no information beyond what the (K-1)-subset already captures — the
itemset is 'near-closed' and can be filtered out", and "Ratios well below 1.0
indicate genuine combinatorial signal."

Since `support_K <= support(W)` for every (K-1)-subset `W`, requiring
`support_K == max(subset supports)` forces **all** subsets to have identical
support — a degenerate corner, not the "near-closed" family described. The
condition actually described is `support_K == min(subset supports)`.

```
{1,2,3} at support 0.30, subsets {1,2}=0.30, {1,3}=0.90, {2,3}=0.95
(item 3 fully implied by {1,2} — maximally redundant)

  max_k_minus1_support = 0.95   self_sufficiency_ratio = 0.315789
  under min:                    self_sufficiency_ratio = 1.0
```

A perfectly redundant itemset scores 0.32 and reads as "genuine combinatorial
signal" — exactly backwards. Anyone filtering on this ratio keeps the redundancy
and discards the signal.

**The code is wrong, not the docstring**, on two independent grounds:

1. The package already implements the correct predicate twice, both as the min
   test — `_prune_equal_support` (`core/apriori.py:147-151`) prunes when *any*
   (k-1)-subset count equals the itemset's count, and
   `rust_ext/src/core/groups.rs:291-304` does the same. `core/rules.py:541` is a
   third copy of one predicate that drifted to the other aggregate.
2. **"Recalibrate the cutoff" is not available**, because under `max` the ratio
   is not monotone in the property described. Two itemsets that are *equally*
   redundant by the engine's own predicate score `0.30/0.32 = 0.9375` and
   `0.30/0.95 = 0.3157894736842105` — identical redundancy, opposite scores,
   opposite recommended actions. Under `min` both score `1.0`. It is the wrong
   *kind* of aggregation, not a mis-scaled one.

The formula sentence at `:459-460` must change with the code. Public API:
`src/et_miner/__init__.py:31`, `__all__` at `:87`.

**Note:** this function also carries #3's positional-join exposure. On
mixed-route artifacts the max is taken over an arbitrary subset of the
(K-1)-subsets, moving the ratio in the *opposite* direction to this defect
without correcting it (see #3 for the proof that they do not compound), and
K-itemsets with no surviving subset leave the output silently via `:545`.

> **How to verify a fix — this matters, because the obvious check gives a false
> pass.** Verify against a **single-route** artifact and check one row by hand
> against `support_K / min(subset supports)`. On a clean single-route artifact
> this defect is trivially detectable: the implemented ratio differs from the
> intended one on *every* row (0/454 agreement), because `support_K/max` equals
> `support_K/min` only when all subsets share a support, which does not occur in
> practice. On a **mixed-route** artifact it is undetectable by inspection and
> worse than uniformly wrong — 59% of the corrupted rows are exactly right by
> coincidence (see #3), and a correct value does not look anomalous, so
> spot-checking produces false reassurance. That 59% is coincidence, not partial
> self-correction: it depends on which subsets the positional join happened to
> drop, i.e. on an item-id ordering accident, and has nothing to do with the
> metric's semantics.

---

## 13. `count_support_batched`'s length filter is set by `itemsets[0]` — an order-dependent undercount

**Severity: MEDIUM.** Confirmed by three reviewers; measured.

```python
core/matrix.py:317-319   k = min_transaction_length
                         if k is None:
                             k = len(itemsets[0])
core/matrix.py:321       matrix = _filter_transactions_by_length(matrix, k)
```

Whichever itemset happens to be first sets the row filter for **every** itemset
in the call, and `core/matrix.py:73-79` then drops every transaction with fewer
than `k` items. The result changes when you permute the input list:

```
ground truth        : {('i_0','i_1','i_2','i_3'): 1, ('i_0','i_1'): 4}
length_filter=True  : {('i_0','i_1','i_2','i_3'): 1, ('i_0','i_1'): 1}   <- 75% undercount
length_filter=False : {('i_0','i_1','i_2','i_3'): 1, ('i_0','i_1'): 4}
short-first order   : {('i_0','i_1'): 4, ('i_0','i_1','i_2','i_3'): 1}   <- correct
```

A caller building the itemset list from a **set** gets a different answer per
run.

`apriori()` itself always passes a uniform level, so this is unreachable through
the top-level API — but `count_support_batched` is exported public API
(`src/et_miner/__init__.py:93`), its own docstring says the filter is
"Auto-detected from itemsets" (plural, `core/matrix.py:295-297`) when it reads
one element, and its downstream `count_support_sparse` explicitly partitions
k=2 from k>2 (`core/sparse.py:652-659`) — so mixed-length input is a supported
shape one layer down and an unsupported shape one layer up, with no check
between.

**The repository already knows.** `tests/test_streaming_bug.py:134-141` is a
class named `TestMixedLengthFilterRegression` whose docstring states the defect
verbatim, including "a non-deterministic false-negative source (set-iteration
order)" — and then `:164` **pins the buggy behaviour** rather than fixing it:

```python
assert buggy[("i_0",)] < fixed[("i_0",)]
```

Both in-tree callers that hit the mixed-length shape work around the default
instead of fixing the function: `streaming/son.py:433-436` and
`streaming/multi_gpu.py:459-462` both pass `enable_length_filter=False` with an
identical comment naming the hazard.

**Fixing this is a behaviour change with a test to update — state it up front
rather than discovering it during implementation.** `tests/test_streaming_bug.py:143-165`
is exactly this scenario. Take the minimum length across the batch instead of
`itemsets[0]` and `k` becomes 1, nothing is filtered, `buggy == fixed`, and the
assertion at `:165` fails. Two details make this the strongest argument that #13
is a defect rather than intended behaviour.

First, the test contradicts itself in three places. Its summary line reads *"A
mixed-length batch with a long itemset first **must not** undercount"* — a
correctness requirement. Its body then explains the undercount, `:165` asserts it
happens, and the function is named
`test_length_filter_undercounts_mixed_length_batch`. The name and the body side
with the defect; only the first line of the docstring asks for the correct
behaviour.

Second, **it is currently green** — so anyone auditing this and grepping the
suite finds a passing test that appears to sanction the behaviour.

> ⚠ **Warning for whoever fixes this.** `tests/test_streaming_bug.py:165` **will
> fail**, and that failure is the fix working. The assertion is what needs
> deleting — replace it with the correct expectation (`buggy == fixed`). This is
> the only defect in the report whose repair turns the suite red, so it is the
> one most likely to be reverted by someone trying to get a green run.

---

## 14. The mandated correctness gate derives its oracle threshold from the expression it is meant to check

**Severity: MEDIUM** (test-design defect — it is what lets #11 exist undetected).

```python
tests/test_tier_equivalence.py:73   ea_support = (SPEC.min_count - 0.5) / SPEC.n_rows
tests/test_tier_equivalence.py:87   apriori(df, min_support=SPEC.min_support, ...)
src/et_miner/synthetic.py:71-72     def min_count(self): return math.ceil(self.min_support * self.n_rows)
src/et_miner/core/result.py:17      return math.ceil(min_support * n_transactions)
```

The miners are called with `SPEC.min_support`, so each computes
`_min_count(SPEC.min_support, N)`. The oracle is called with
`(SPEC.min_count − 0.5)/N`, where `SPEC.min_count` is the **same expression**,
character-for-character. Any error in that expression therefore moves the
miner's threshold and the oracle's threshold by exactly the same amount, and the
comparison still passes. **The gate cannot detect a min-count error by
construction.**

CLAUDE.md calls the `−0.5` convention load-bearing "so the boundary is exact for
integer counts". It is — but it also guarantees the oracle is never probed *at*
the boundary, so the one thing the convention makes exact is the one thing the
gate can never test.

**Fix.** Keep the `−0.5` convention; it is right for what it does. Add one
independent boundary case: assert directly that an itemset whose count is
exactly `ceil(Fraction(str(s)) * N)` is returned, with the threshold computed
from the decimal rather than from `_min_count`. The gate must not source its own
expected value from the code under test.

---

## 15. The "O(1) memory" k=2 streaming path materialises the whole pair list

**Severity: LOW.** Measured.

```python
core/candidates.py:76-77   return list(_generate_candidates_k2_streaming(items))
```

on the branch whose docstring at `core/candidates.py:45` claims "O(1) memory vs
n*(n-1)/2 cross-join materialization". The `list()` fully drains the generator.

```
threshold=5000  items=6000  pairs=17,997,000  type=list  peak RSS delta=1242 MB
```

At the `stress_k2` preset (35,000 items) that is ~600M pairs.

Scoped honestly: the branch does still avoid the Polars cross-join's intermediate
DataFrame, so it is a real partial saving mislabelled as a total one, and it
cannot be fixed inside `core/candidates.py` — `_generate_candidates` returns
`list[tuple[str, ...]]` (`:19`) and the caller takes `len()`
(`core/apriori.py:652`) then iterates twice (`:666`, `:710`). The lazy path is
unreachable by contract, not by accident. Making the claim true requires
reworking `core/apriori.py:649-710` into a single pass.

---

## 16. Importing `core.sparse` overrides the process-global MKL thread count

**Severity: LOW.** Measured independently by two reviewers.

`core/sparse.py:98` calls `_init_mkl()` at module scope, reaching
`mkl_set_num_threads` at `:76`:

```
user set MKL threads              -> 2
after import et_miner.core.sparse -> 24
```

A host application's own MKL configuration is silently overridden,
oversubscribing every other MKL consumer in the process. Compounding it,
`_restore_mkl_threads` (`:89-90`) re-derives the count from the environment
rather than restoring the value it displaced, so the `finally` blocks at
`:369-371` and `:606-607` do not actually restore. And the module is imported
lazily from inside `count_support_batched` (`core/matrix.py:344`), so the side
effect fires mid-run at first count rather than at `import et_miner`, which is
where a host would look for it.

---

## 17. MKL path discovery probes one hardcoded name at one hardcoded location, misses, and fails open silently

**Severity: LOW.**

`_setup_mkl_library_path` (`core/sparse.py:34-56`) exists, per its own docstring
at `:36-38`, because "pip installs MKL libraries to {venv}/lib/ which is not in
the standard library search path". It tests exactly one filename at exactly one
location — `os.path.join(sys.prefix, "lib", "libmkl_rt.so.2")` at `:44` — and in
this tree it misses on both halves:

```
sparse.py:44 probes: {venv}/lib/libmkl_rt.so.2
find .venv -name "libmkl_rt.so*"  ->  .venv/lib/libmkl_rt.so.3
exists sys.prefix/lib/libmkl_rt.so.2 : False
actually loaded MKL (/proc/self/maps): ['/opt/conda/lib/libmkl_rt.so.2',
    '/opt/conda/lib/libmkl_core.so.2', '/opt/conda/lib/libmkl_intel_lp64.so.2', ...]
```

The venv ships `.so.3`; the loadable MKL is a `.so.2` somewhere else entirely.
The guard at `:46` therefore returns at `:47` **with no log line**, so nothing in
the output distinguishes "the path was already fine" from "the probe missed".

Consequence: the helper's whole purpose is to make the *venv's* MKL findable, and
it fails at that — so in a repo that pins cupy, polars and the Rust extension by
version, the numerics silently run against an unpinned system MKL. That is not
academic, because #4's float32 saturation runs through whichever MKL loaded.

*Not a stale-soname claim:* `.so.2` is a correct soname for a conda MKL, and on a
pip install that ships `libmkl_rt.so.2` into `{venv}/lib` the probe fires exactly
as designed. The defect is the single hardcoded name at a single hardcoded
location, plus failing open without a diagnostic.

**⚠ See #4's fix ordering.** Repairing this probe in isolation would activate
the float32 saturation for installs currently returning correct answers.

---

## 18. `n_jobs` is dead for k>2 sparse counting whenever the Rust extension is built

**Severity: LOW.** Confirmed by three reviewers; measured by instrumentation.

```python
core/sparse.py:458-460   def _should_use_rust(n_itemsets, n_items):
                             return RUST_INSTALLED
```

Both parameters are ignored, and `core/sparse.py:265-269` takes the Rust branch
before `n_workers = _get_effective_workers(n_jobs)` is evaluated at `:272`.
`_RUST_MIN_ITEMSETS = 1` (`:455`) is declared and never read — the fossil of the
threshold that used to gate this.

```
n_jobs= 1: rust=1  python_parallel=0  python_sequential=0
n_jobs= 8: rust=1  python_parallel=0  python_sequential=0
n_jobs=-1: rust=1  python_parallel=0  python_sequential=0
_should_use_rust(1,1) and _should_use_rust(10**9,10**9) both return True
```

Documented as active at `core/apriori.py:287-288` and `core/sparse.py:634-638`.
Still honoured for k=2 (`core/sparse.py:672`), which makes it half-honoured and
harder to notice than not honoured at all.

Results are unaffected, so this is LOW — but the harm is not lost throughput:
the Rust path takes all cores via rayon regardless, so a caller who set
`n_jobs=1` to constrain itself (documented as "sequential execution") is
silently oversubscribed. On a shared box or a long campaign that is the
difference between a bounded job and one that takes the machine.

---

## 19. `_prune_groups_apriori` is documented wrongly in two files, one citing a verification

**Severity: LOW** (documentation only — the code is correct).

```
gpu/mining.py:267-268
  "A suffix is only kept if ALL its (k-1)-subsets are in prev_frequent_set"
  "(Verified by Auditor: 16/16 math checks pass, 2026-03-25)"
```

Both implementations do something materially different: a suffix is kept if it
participates in **at least one** valid pair (`gpu/mining.py:352-375`:
`valid[i] = True; valid[j] = True` per valid pair, never a per-suffix
all-subsets test; `rust_ext/src/core/groups.rs:563-566` is identical in shape),
and the group is then rebuilt from every surviving suffix with `total_candidates`
recomputed over all C(m,2) pairs among them (`groups.rs:571-574`, `:603-604`),
re-admitting pairs just found invalid.

`groups.rs` also contradicts itself: `:471-472` promises pair-level removal while
`:477-478` correctly describes the suffix-level implementation the code has.

The behaviour is sound (see negative results). But a false "only kept if ALL"
claim carrying a verification badge, on the exact function
`docs/specs/et_miner_fix_spec.md:97` names as a Defect-A suspect, is how a
correct component gets ruled in or out for the wrong reason. The Rust doc at
`groups.rs:477-478` is the model to copy.

---

## 20. `core/result.py` documents a shared-emission-point invariant the row-split path does not honour

**Severity: LOW** (documentation / maintenance trap — no runtime fault).

```
core/result.py:3-6
  "the direct, SON-streaming, and multi-GPU mining paths (and their tests) all
   build their output through these three helpers, so their signatures should
   stay put even as the mining modules evolve."
```

The multi-GPU row-split path does not. `gpu/row_split.py:19` imports
`_build_result_df` and calls it in exactly one place — the empty-result early
return at `:368`. Its two real return paths build the frame directly:
`pl.DataFrame({"itemset": pl.Series("itemset", arrow_list), "support":
all_supports})` at `:739-744`, with a list-building fallback at `:751-756`. Only
`core/apriori.py:758` and `gpu/mining.py:775`, `:967` route through the helper.

**What makes it dangerous is that it is two-thirds true.** `row_split` *does*
import and use two of the three helpers — `_min_count` (`:21`, called at `:135`)
and `_empty_result` (`:20`, called at `:721` and `:725`). Only the output
construction sits outside them. So a reader spot-checking "does row_split use the
result helpers?" gets *yes*, and the false part is the narrower, better-hidden
one. **A docstring that is two-thirds accurate defeats the check that would catch
it.**

**Why this earns a place in the list rather than a footnote:** during this review
two reviewers independently proposed enforcing #1's canonical order at
`_build_result_df` *because the module says every path goes through it*, and an
earlier draft of this report carried that fix. It would have missed the row-split
path — the one route that gets the order right today, and therefore exactly the
one you would want a regression guard on. The review reproduced the precise
failure pattern it was documenting.

---

## 21. `compute_self_sufficiency` crashes instead of returning its documented empty frame

**Severity: MEDIUM.** Measured.

The function has an empty-result path at `core/rules.py:552-561` that returns a
correctly-typed empty DataFrame. **It is dead code.** `:548` appends the chunk
result unconditionally:

```python
core/rules.py:548        result_chunks.append(result)          # even when result has 0 rows
core/rules.py:552        if not result_chunks:                 # unreachable once any chunk was read
core/rules.py:554-561        return pl.DataFrame(schema={...}) # dead
```

So `result_chunks` is never empty once any chunk has been read, `combined` is a
0-row frame, `combined['self_sufficiency_ratio'].min()` returns `None`, and the
logging f-string at `:569-573` raises:

```
TypeError: unsupported format string passed to NoneType.__format__
```

**The sibling function in the same file gets this right** — `generate_rules_drop1`
guards with `if n_rules > 0: result_chunks.append(result)` at `:399`. The two
parquet consumers differ on exactly this one line.

**Trigger:** any call where no K-itemset retains a surviving (K-1)-subset after
the inner join at `:536`. Reached by #3's mixed-tier artifacts once the vocabulary
exceeds ~201 frequent items, and independently by any mismatched or empty K / K-1
pair. Measured above in #3.

The crash is arguably better than the alternative — a silent wrong answer — but
it is not the documented behaviour, it arrives as an unhandled `TypeError` from
inside a log statement rather than a diagnostic, and it kills a long campaign at
the reporting step after all the mining work is done.

---

## Why the suite doesn't catch any of this

471 tests pass and every defect above is live. That is not a coincidence.

**The one-line version: in almost every case here, the checking apparatus
inherited the assumption it was supposed to test.** `tests/test_tier_equivalence.py:44`
canonicalises with `sorted()` because whoever wrote it believed the tiers agree
up to ordering — which is precisely the claim in dispute. The oracle gate derives
its threshold from the expression under test. None of these are *weak* checks;
they are checks that **cannot fail for the reason that matters**.

Concretely, the suite is green in **three structurally different ways**, and they
need different fixes — collapsing them into "missing coverage" gets two of the
three wrong:

1. **No coverage.** Most of #6–#10: no test asserts that those parameters do
   anything at all. This is the only one that is ordinary missing coverage.
2. **Normalised away.** `tests/test_tier_equivalence.py:44` canonicalises with
   `tuple(sorted(int(i) for i in itemset))` *before* comparing, so #1's property
   cannot register. The test is not weak — it is **shaped so the defect is
   invisible**, which is exactly why deleting the `sorted()` is the wrong
   instinct (see the warning under #1) and why the fix is an *added* assertion.
   #14 is the same shape: the gate sources its expected value from the code under
   test, so #11 cannot register either.
3. **Pinned as expected.** `tests/test_streaming_bug.py:165` asserts
   `buggy[("i_0",)] < fixed[("i_0",)]` — the suite encodes #13's undercount as
   correct behaviour, in a class named `...Regression` whose docstring asks for
   the opposite. **The suite will go red when someone fixes #13.**

### The strongest form of this, measured

Failure mode 2 is usually argued. Here it was measured. The same
`471 passed, 4 skipped` appears in **three distinct states of the codebase**:

| State | Reality | Suite |
|---|---|---|
| Unpatched | #1 live — 910 of 1335 emitted itemsets unsorted | `471 passed` |
| Patched | #1 fixed — 0 unsorted; #2 and #3 close with it (rules 9,722 → 13,580, lattice bit-identical) | `471 passed` |
| Patched inoperatively | the patch present but not taking effect | `471 passed` |

**The suite is invariant to whether this defect exists, whether it is fixed, and
whether the fix works.** That is a sharper claim than "no test covers this", and
it has a direct consequence for remediation: someone can apply the #1 fix, see
471 green, and have learned nothing. It is why fix part 3 — the per-tier order
assertion in the gate — is load-bearing rather than tidy-up.

It also names what `tests/test_tier_equivalence.py` is missing. The reason it
passes 9/9 against every defect in this report is not that it is weak; it is that
it has **no control for the property it claims to check**, because `:44`
canonicalises before comparing. A test that would pass identically whether or not
the property holds is measuring something else.

There is a matching methodological trap worth recording, and the review walked
into it repeatedly. Three reviewers independently tested `generate_rules_drop1`
**within a single route**, got 1362/1362 and zero metric errors, concluded the
function was sound — and all three had to retract when the cross-tier case was
tried. That is not three individual slips: the mandated gate makes the same scope
error structurally, for the same reason. Every check available, automated and
manual, compared the tiers in a form that erases the difference between them.

The general form, which cost this review more time than anything else:
**a claim confirmed on the case that motivated it has not been tested.** A
structural invariant gave the right answer on the 15-item fixture and the wrong
one on the 248-item fixture; a routing claim was right for the CPU path and wrong
for row-split; two reviewers each made a test-file claim that was right about the
file they checked and wrong about the one they did not. All of them survived a
plausibility check and died on a second, *different* case. That is exactly what
`tests/test_tier_equivalence.py` does when it passes 9/9 against every defect
here: it checks the tiers on the case that motivated it, in the form that erases
the difference.

The emission order in #1 is not written down as a producer contract anywhere in
the tree — `apriori()`'s Returns block (`core/apriori.py:318-320`) commits to
nothing about it, while `core/rules.py:173-174` and `:263-264` both *assume* the
guarantee while consuming it. So there was nothing to check the producer against,
and the one invariant that touches it is enforced by a test that launders the
property. **An unstated invariant guarded by a test that normalises it away is
invisible by construction** — which is why every reviewer's first instinct was to
verify *mining* (the only property with a written specification, and the one that
turned out to be fine).

A fourth instance of the same shape is #20: a module docstring asserting that all
paths share an emission point, which two reviewers believed and acted on before
anyone checked it.

## Remediation ordering — two constraints that will bite if ignored

1. **#11 must land at all three sites in one commit.** `core/result.py:17`,
   `rust_ext/src/core/apriori.rs:73` and `synthetic.py:71-72` currently agree
   *because they are the same wrong expression*. A one-sided fix breaks the
   tier-equivalence chain at exactly the (s, N) pairs where the error lives.
2. **#4 must land before or with #17.** Repairing the MKL path-discovery probe
   in isolation would *activate* the float32 saturation for installs that
   currently return correct answers — the LOW fix makes the HIGH bug newly
   reachable.

3. **#35 must canonicalise to the *sorted* representative — the word is
   load-bearing.** SON's duplicate emission is currently *masking* #2: the
   duplicate row supplies the sorted key `core/rules.py:89` queries for.
   Collapsing to an arbitrary representative silently drops 132 of 1,718 rules'
   lift from 1.0 to 0.0 (measured), i.e. worse than leaving the bug in. Sorting
   needs no sequencing against #2 — it fixes #2 on the SON path too.
4. **Fixing #13 turns the suite red, and that is correct.**
   `tests/test_streaming_bug.py:165` asserts the defect. Delete that assertion as
   part of the fix; do not revert the fix to restore a green run.
5. **#1's fix is three-part**, and part 3 must be *additive*. Pad the column
   names, state the canonical order in the `apriori()` return contract, and add a
   per-tier order assertion to the equivalence gate — **without** removing the
   existing `sorted()` canonicalisation, which CLAUDE.md forbids weakening and
   which insulates the gate from an undocumented property of efficient-apriori.
   Do not place the enforcement at `core/result.py::_build_result_df`; it is not
   the shared emission point it claims to be (#20).

---

## Negative results — what was checked and found correct

Recorded because ruling things out has value, and because two of these correct
the record in `docs/specs/et_miner_fix_spec.md`.

- **Spec Defect B is fixed at HEAD.** `apriori(bitvecs=...)` no longer ignores
  `prune_equal_support` — it routes to row-split (`core/apriori.py:384-399`) and
  produces exactly the free-sets — and no longer mutates the caller's array:
  measured "0 caller columns modified". `_deallocate_dead_bitvecs` is gone, and
  a grep for in-place writes to a caller's bitvec array in `gpu/mining.py` and
  `gpu/row_split.py` finds none.
- **Spec Defect A did not reproduce at HEAD** in an end-to-end test at the
  default chunk size: on a 32-item nested-vocabulary set (40k rows, min_support
  0.01, max_length 6, 237,405-itemset full lattice) the GPU row-split free-sets
  equalled the CPU free-sets exactly — 4,335 == 4,335, zero missing, zero extra,
  zero count mismatches. **This is one configuration, not a clearance**; the
  spec's own repro forces many chunks per level with a small
  `ET_MINER_MAX_CHUNK_CANDS`. The GPU review is re-testing under the spec's
  conditions.
- **`rust_ext/src/core/groups.rs` is not where Defect A lives.** Fuzzed over 300
  random previous levels (widths 2-4, vocabularies 6-11, 2-40 itemsets each)
  against a brute-force enumeration of every k-candidate whose (k-1)-subsets are
  all present: **0/300 trials lost a valid candidate**; 83/300 carried extra
  ones. The error is strictly one-sided.
- **The suffix-granularity apriori prune is a sound over-approximation, not a
  bug.** All four reviewers converged on this after disagreeing initially. The
  pair-validity test at `groups.rs:534-561` is *exact*, not merely conservative:
  for a candidate `prefix + [s_i, s_j]`, the two suffix-drop subsets
  (`prefix + [s_j]`, `prefix + [s_i]`) are rows of the level the group was built
  from and therefore frequent by construction, so only the k−2 prefix-drop
  subsets need testing — which is exactly what `:548-558` tests. The caller was
  checked too: `gpu/row_split.py:427`/`:540` build groups from
  `prev_frequent_flat` while `:437`/`:547` prune against `prev_full_flat`, a
  superset under free-set pruning, so the parents are present a fortiori. The
  re-crossing at `:563-574` re-admits invalid pairs, but by anti-monotonicity
  those cannot reach `min_count`, so the cost is wasted counting and never a
  wrong output. Only the documentation is defective (#19).
  *Precise wording, because an earlier draft of this was wrong:* the GPU group
  path over-approximates the subset test **where the CPU path does not** —
  `_generate_candidates_simple` (`core/candidates.py:126-135`) and
  `_is_valid_candidate` (`:238-248`) apply the full per-candidate subset test, so
  the CPU path never emits a candidate with a *known*-infrequent (k-1)-subset. It
  over-approximates only against counts it does not yet have, which is inherent
  to Apriori; `groups.rs:571-574` over-approximates against counts it already
  has. The surplus is provably one-sided, so the emitted lattice is identical;
  the cost is wasted counting.
- **Candidate generation is complete.** All three branches
  (`core/candidates.py:101-193`), including the >1000-itemset and >=100-group
  Polars paths, match brute force on a 1298-itemset fixture.
- **CPU free-set and Pascal-inference pruning are correct.**
  `core/apriori.py:108-156` and `:161-213` match brute force on
  nested-vocabulary data through K=5 over 10 seeds; the `licensed` gate at
  `:191-213` is the correct fix for the counterexample it documents.
- **The Rust counting kernels and `apriori_from_csr` are correct.**
  `counting.rs:23-130` and `apriori.rs:65-142` match brute force over 8 seeds.
  Bitvector tail words are zero by construction, so no masking is needed
  (`bitvec.rs:34-56`). The Rust join at `candidates.rs:84-101` is complete given
  lexicographically sorted input, which rayon's order-preserving `collect`
  guarantees (`apriori.rs:190-231`). Boundary K is handled at `apriori.rs:98-116`.
- **The synthetic planted-motif oracle is a valid lower bound**
  (`synthetic.py:104-148`).
- **The drop-1 parquet join is sound within a single run**, despite #1: the K and
  K-1 levels of one run share a column-name order, so `_explode_drop1`'s
  antecedent still joins. The docstrings at `core/rules.py:173-174` and
  `:263-264` are wrong about *why* it works; the join itself is fine
  same-route. It breaks only across routes (#3).

## Retractions — claims raised during review and then withdrawn

- `rust_ext/src/core/cooccurrence.rs:29,84` allocating `n_cols²` per row is **not
  a bug today**: `dir(et_miner_rust)` exports 13 names, none a cooccurrence
  symbol, `lib.rs` never registers it (no entry in the `#[pymodule]` block at
  `:541-574`), and no Python caller exists in `src/`, `tests/` or `bench/`. Dead
  code with a latent defect — a note for anyone who later wires it up.
- `apriori(batch_size="auto")` raising `TypeError` is **not a bug and not a
  documentation mismatch**. `count_support_batched(..., batch_size="auto")` works
  and is correctly documented (`core/matrix.py:279`); `apriori()` simply never
  advertises "auto" — its signature is `int | None` (`core/apriori.py:224`) and
  its docstring says only "None = no batching". The only observation worth
  keeping is that `:224` narrows a capability its callee advertises at
  `core/matrix.py:253` (`int | str | None`) — a capability gap, not a defect.
- **`_setup_mkl_library_path` mutating `os.environ["LD_LIBRARY_PATH"]`** was
  filed and then withdrawn: the guard at `core/sparse.py:46` returns first, so
  the mutation never runs on this install. `LD_LIBRARY_PATH` was byte-identical
  before and after import. #17 is the accurate, measured replacement. A companion
  claim — that mutating `os.environ` cannot affect `dlopen` in a running process
  because glibc captures the search path at startup — is correct in general but
  untestable here for the same reason, and is not filed.
- `CscMatrix::from_csr` (`rust_ext/src/core/matrix.rs:35`) counting over all of
  `csr_indices` rather than `csr_indices[..indptr[n_rows]]` is a **hardening
  note, not a bug**: scipy leaves no slack, and the SIMD/bitvec path is immune
  anyway because unfilled CSC slots default to 0 and merely re-OR bit 0.
- Inflated 1-itemset counts from `explode()` on transactions with repeated items
  (`core/matrix.py:121-129`) **do not affect `apriori()` results** — the matrix
  columns are recounted at `core/apriori.py:581`. Performance and memory only.
- **The recommendation to remove the `sorted()` from
  `tests/test_tier_equivalence.py:44`** was made twice during review and
  withdrawn: it violates CLAUDE.md's prohibition on weakening those assertions,
  and would work only by accident. See the boxed warning under #1.
- **"The mixed-tier corruption compounds with #12"** was written into the record
  and corrected: the two errors *oppose*, and the second provably interpolates
  between the first's output and the correct value. See #3.
- **"`compute_self_sufficiency` drops K-itemsets under mixed-tier input"** was
  predicted, measured at 0/454, reclassified as derived-but-unmeasured — and then
  **measured after all** at 248 frequent items, where it turned out to be the
  crash in #21 rather than the predicted silent drop. See #3.
- **"`_build_result_df` is the shared emission point every mining path routes
  through"** was asserted twice, carried into an earlier draft of this report as
  the recommended fix location for #1, and is false. See #20.
- **"The padding fix touches no test"** was asserted, then retracted as breaking
  `tests/test_sparse.py:183`, then the retraction itself was withdrawn. The
  original conclusion is correct — that fixture builds its own 100-column frame
  and never calls `build_boolean_matrix` — but the *reason* first given for it
  ("the fixtures are small") was wrong. Two reviewers gave **opposite wrong
  answers about the same pair of test files**, each from checking one and
  generalising to the other. It is the cleanest example in the review of the
  failure mode the review is documenting, and the reason the fix section now
  records the invariant ("no test couples to the name format") rather than the
  observation.

### Provenance

Recorded because it is the strongest evidence for how much of this report to
trust. Across the review the four reviewers withdrew or narrowed **more than a
dozen** claims — one reviewer accounts for seven of them alone, which is worth
stating rather than smoothing over: one correctness finding retracted outright after a 300-trial fuzz
contradicted it (the `groups.rs` prune), two retracted as not-defects
(`cooccurrence.rs`, `batch_size="auto"`), two narrowed to the half that measured
true (the MKL `LD_LIBRARY_PATH` and version-skew claims), one recommendation
withdrawn on policy grounds (the gate `sorted()`), one fix location retracted as
factually wrong (`_build_result_df`), one fix replaced by a simpler one (#1's
30-line refactor → one-line padding), one over-claim about that fix being
test-neutral, one sign error (#3 vs #12), one citation "correction" that
introduced an error of its own, and one severity reversed twice before settling
(#1: HIGH → MEDIUM → HIGH).

Two of those were reviewers correcting themselves *against their own interest*.

**The pattern in the errors is the same one the report documents in the code, and
it recurred five times across three reviewers.** Each failed by *generalising from
a match to a mechanism* without checking what produced the match: a grep hit on
`("i_5","i_10")` read as breakage without opening the fixture that defines it; a
docstring (`core/result.py:3-6`) read as a routing fact; a `0/454` null result
read as absence rather than as a fixture too small to construct the case; "no
test couples to the name format" inferred from matching call sites while one does
couple; and an enumeration asserted from a result set that had been silently
truncated.

The mechanism is more specific than carelessness: in **four of the five the
evidence was silently incomplete rather than wrong** — two head-limited greps
that omitted the decisive line, a null result bounded by fixture size, and an
inference from one file of two. Only the docstring case came from believing a
written claim outright. So: *a grep says a string is present, a docstring says
what someone believed, a zero says nothing was observed, and a truncated result
set says nothing about what it omitted — none of them says why.*

That is precisely the defect class this report files as #19 and #20: written
invariants nobody traced to a mechanism — each true of the cases its author
checked. **The review reproduced the codebase's own failure mode five times while
documenting it**, which is worth stating plainly rather than softening. It kept
striking whichever reviewer had most recently warned another against it, in one
case a single message later — because complete and truncated evidence are
indistinguishable at the point of use, so the warning does not transfer.

**What finally resolved it was not a more careful reading.** Three reviewers
produced three different static answers about which test breaks under the #1 fix.
The question was settled by *running it with a control* — and the control was the
part that mattered, because a bare green would have been indistinguishable from a
patch that never applied, and all three would have accepted it. Incomplete
evidence and unverified instrumentation fail the same way: both produce a result
indistinguishable from the one you wanted. That is the same gap that lets
`tests/test_tier_equivalence.py` pass 9/9 against everything here, and it is the
standard the fixes should be verified against.

It is worth being accurate about the lesson rather than the flattering version of
it. Execution did not simply beat static reading: the reviewer with no shell
produced both several withdrawn claims *and* four predictions that held (the
digit-class threshold, the ratio sign, the pip-only MKL branch, the padding fix),
all from static reading. **What separated the good ones was being committed to a
falsifiable number before anyone ran anything**, which is what made them cheap to
kill or confirm. That is the note to carry into how the fixes get verified.

## Handed to the streaming review (root cause outside section 1's scope)

- **`streaming/async_pipeline.py:407` and `:659` use `int(min_support * n_transactions)`
  — floor, not ceil.** `_min_count` (`core/result.py:15-17`) is the canonical
  helper and is correctly imported by `core/apriori.py:44`, `core/matrix.py:28`,
  `gpu/row_split.py:21`, `gpu/mining.py:23`, `streaming/son.py:42` and
  `streaming/multi_gpu.py:56`. The async pipeline instead re-implements it with
  the wrong rounding: at `min_support=0.0155, n=1000` that is 15 against the
  canonical 16, so the async streaming path keeps itemsets the documented rule
  excludes. Four copies of a threshold CLAUDE.md declares mandatory, one already
  drifted.

---

# Section 2: GPU path + CUDA kernels

Reviewed by the same four-lens panel, unanimous **BLOCK**. Two reviewers had two
RTX 3090s and ran the reproductions; the other two worked from the sources.

**Baseline.** `bench/selfcheck.py` compiles and smoke-launches all 19 registered
kernels on both devices — READY. Full suite including slow tests: **479 passed,
6 skipped**. `-m "not slow"`: 471 passed, 4 skipped. Every defect below is live
against that.

## The headline: Defect A does not reproduce at HEAD

`docs/specs/et_miner_fix_spec.md` Defect A — "the row-split path drops frequent,
closed, apriori-valid itemsets from K=5", severity high — **was refuted by two
reviewers independently.** It should be marked resolved in that spec.

`docs/recon/` does not exist at HEAD; the spec's own repro script
(`docs/recon/et_miner_repro_row_split_drop.py`) is gone, so both reviewers
rebuilt it from the written description.

**At the spec's exact conditions** — 120,000 rows / 32 items / `min_support` 0.01
/ `max_length` 6 / `ET_MINER_MAX_CHUNK_CANDS=5000`, `prune_equal_support=True`:

```
GPU row-split output == free-sets computed independently from the complete lattice
714,841 == 714,841     0 missing / 0 extra / 0 count mismatches

holds on: 1 GPU; 2 GPUs; under a full column-relabelling permutation;
          with 3 extra unrelated columns appended;
          ET_MINER_KERNEL_VARIANT=legacy; ET_MINER_FILTER_IMPL=cpu and cupy;
          ET_MINER_DISABLE_NCCL=1; ET_MINER_TILED_MIN_GROUP_PAIRS=0
chunk caps 500 / 5000 / 100000 all identical
          (3,037 / 2,141 / 2,141 chunk-filter events — chunking genuinely exercised)
```

**Independently, on different data, at a different scale, against a different
oracle** — 40,000 × 32 nested-vocabulary (child implies parent, so the free-set
prune actually engages), `min_support` 0.02, full lattice 158,607, free-sets
8,573: **21 gated/unpruned configurations and 5 layout-sensitivity
configurations, 0 missing / 0 extra / 0 count mismatches** — across
`ET_MINER_MAX_CHUNK_CANDS` of 5000, 500, 137 and 1, 1 and 2 GPUs, NCCL on and
off, both kernel variants, all three filter implementations, dense and sparse
CSR, and appending 3 or 7 unrelated columns.

> **Which chunk cap is the strongest evidence — and it is not the smallest one.**
> `plan_group_chunks` routes both mega groups (pairs > cap) *and* tiny groups
> (pairs < `tiled_min_group_pairs`, default 64) to the **legacy** kernel; only the
> middle class reaches shared/tiled. At `cap=1` every group falls into one of those
> two, so **the shared/tiled kernel is never launched at all**. Measured on a real
> K=5 level (18,439 itemsets, 2,452 prefix groups, 92,784 candidates):
>
> ```
>       cap   chunks   tiled   legacy   % candidates via the tiled kernel
>         1    92784       0    92784        0.0%   <- legacy-only degenerate corner
>       137      953     223      730       21.5%
>       500      352     216      136       67.2%   <- best mix: both kernels, many boundaries
>      5000      232     116      116       67.2%
>    100000      232     116      116       67.2%
> ```
>
> So **`cap=500` is the strongest single configuration** (best kernel mix) and
> `cap=137` gives the most chunk boundaries; `cap=1` belongs in the list only as a
> legacy-only corner case. This also explains why caps of 5000 and 100000 produced
> identical chunk-filter counts: above ~500 the plan is dominated by class-run
> boundaries, not by the cap.

**Instrumentation, stated honestly.** The first round of multi-GPU runs did **not**
assert the device count, and the multi-GPU wrappers silently degrade to one device
(see #32) — so a degraded run would have looked like a pass. The key configurations
were therefore **re-run** with `assert cp.cuda.runtime.getDeviceCount() == 2`, an
assertion that row-split really built two shards summing to 40,000, and a spy on
the realised chunk plan. These are the numbers the report stands on:

```
  n_gpus=2 cap=-     8,573/8,573 miss=0 extra=0 cntdiff=0 | shards [20000, 20000] | 15 tiled + 18 legacy
  n_gpus=2 cap=5000  8,573/8,573 miss=0 extra=0 cntdiff=0 | shards [20000, 20000] | 15 tiled + 19 legacy
  n_gpus=2 cap=500   8,573/8,573 miss=0 extra=0 cntdiff=0 | shards [20000, 20000] | 24 tiled + 31 legacy
  n_gpus=2 cap=137   8,573/8,573 miss=0 extra=0 cntdiff=0 | shards [20000, 20000] | 44 tiled + 120 legacy
  n_gpus=1 cap=500   8,573/8,573 miss=0 extra=0 cntdiff=0 | shards [40000]        | 24 tiled + 31 legacy
  every run: "visible devices = 2", shard assertion PASSED
```

The two harnesses differ in every respect that matters: data generation (nested
implication tree vs exact duplicate columns), scale, and oracle (CPU tier vs
single-GPU legacy bitvec with host-side free-set derivation). Both compared
**itemsets and absolute counts**.

Supporting evidence from the same reviewers: **900 route runs** (6 GPU routes × 3
chunk sizes × 2 kernel variants × 25 random datasets) all exactly equal to
`efficient-apriori` called with the mandated `(min_count − 0.5)/N` and an explicit
`max_length` — **total route disagreements: 0**. `compute-sanitizer`
memcheck / racecheck / synccheck / initcheck clean on every route.

Two other items from that spec are also fixed: **Defect B** (already recorded in
section 1), and the spec's third item — `level_callback` reporting
`n_candidates = 0` on the dense branch — is fixed at `gpu/row_split.py:487` and
`:560` (`_n_cands_cb = tc`).

### Why it was real at `25d0957` and is not at HEAD

The negative result is explained, not merely asserted. Commit `9ec6b8c` introduced
the two-population split — `prev_frequent_flat` (what the level emits and the next
level generates from) versus `prev_full_flat` (what every subset test resolves
against) — and its docstring at `gpu/row_split.py:96-101` names the old failure in
so many words:

> *"Emitting a level and then generating from a smaller one is what silently
> dropped frequent, apriori-valid itemsets from K=5 on: the output advertised
> itemsets the run would never extend."*

That sentence **is** Defect A. It was fixed two commits before HEAD and documented
in the code at the site of the fix.

### `docs/specs/et_miner_fix_spec.md` should be amended in place

It is checked in, it presents Defect A as open with a "Done when" checklist, and
it points at a reproduction under `docs/recon/` that no longer exists. Left as it
stands it will keep sending future sessions after a defect that was fixed two
commits ago. Its Defect B and its third item are also fixed.

### And the same defect class is still live — in the anchor path

**#22 below is the sound pattern's own counter-example, in the same function.**
`prune_non_free` gets the two-population treatment; the anchor filter, added
later, does not — it emits a level and generates from a smaller one, exactly what
the docstring above warns against. The difference in outcome is one property:
**freeness is anti-monotone, anchoredness is not.** That single line explains why
the same code shape is safe for one flag and catastrophic for the other, and it is
why a static reading that cleared the free-set path did not clear the anchor
path.

## Summary — Section 2

| # | Severity | Defect | Where |
|---|---|---|---|
| 22 | **HIGH** | `anchor_items` prunes the level the *next* level generates from — loses up to 99.3% of the itemsets it advertises | `gpu/row_split.py:529-531`, `:628-630`, `:471-482` |
| 23 | **HIGH** | Four multi-GPU kernels silently truncate survivors at *any* magnitude where their single-GPU siblings raise | `kernels/k3plus.py:193`,`:447`, `kernels/gpu_resident.py:338`,`:491` |
| 24 | MEDIUM | `_warn_result_truncation` itself tolerates ≤5% loss — drops itemsets non-deterministically with only a log line | `kernels/loader.py:32-40` |
| 25 | LOW | K≥3 kernels read an uninitialised `s_items[]` slot and write one past the array; two different caps, neither enforced host-side | `_src/k3plus_dense.cu:36-41` + 3 siblings |
| 26 | MEDIUM | The row-split miner returns three different `itemset` dtypes, and the common one is incompatible with every other tier | `gpu/row_split.py:733-756` |
| 27 | LOW | One `try` wraps two independent cleanups with a bare `pass`; a failure in the first silently skips the second | `gpu/row_split.py:704-708` |
| 28 | LOW | `_check_memory_guard` returns a truncated lattice as if complete — and the row-split miner has no such guard at all | `gpu/mining.py:503-520`, `:759-761`, `:775` |
| 29 | MEDIUM | The apriori prune builds a full Python set of the previous level unconditionally, before a Rust path that never reads it | `gpu/row_split.py:436`, `:546` |
| 30 | MEDIUM | `del bitvecs_gpu` at the density transition frees nothing, and logs that it did | `gpu/mining.py:622-624` |
| 31 | MEDIUM | K≥3 dense group arrays (~40 GB) have no `finally`, where the sparse twin has one | `gpu/row_split.py:585-616` |
| 32 | MEDIUM | `_apriori_from_bitvecs` declares `n_gpus` and never reads it — the route is chosen by ambient device visibility, silently, in both directions | `gpu/mining.py:411`, `gpu/dispatch.py:36`,`:57`,`:128` |

---

## 22. `anchor_items` prunes the level the next level generates from

**Severity: HIGH.** Found by all four reviewers; measured by two.

The anchor mask is applied to `current_flat` — `gpu/row_split.py:529-531` (K=2),
`:628-630` (K≥3 dense), `:471-482` (sparse CSR), via `_anchor_keep_mask` /
`_apply_anchor_filter` (`gpu/mining.py:218-259`). That same filtered array then
becomes **both** downstream populations:

```python
gpu/row_split.py:651   full_flat = current_flat            # -> prev_full_flat (:699), the subset oracle
gpu/row_split.py:697   prev_frequent_flat = current_flat   # -> the generation base
```

**The root cause, stated precisely.** The prefix-join does *not* need the filtering
predicate to be anti-monotone — it needs the surviving family to be **closed under
the two prefix-parents**: to generate `X = prefix + (s_i, s_j)` it needs
`prefix+s_i` and `prefix+s_j` retained, not every subset of `X`. In an *arbitrary*
column layout the anchored family has no such closure — an anchored candidate's two
prefix-parents may both be unanchored, get dropped a level early, and `X` can never
be generated. Separately, the apriori oracle *does* need the anchor-dropping
(k−1)-subsets, and those are unanchored by construction.

So the current code is unsound in two independent ways at once. This is the same
defect class `gpu/row_split.py:94-107` documents as having been fixed for
free-sets; the anchor filter simply bypasses that design.

*(The distinction matters for the fix: prefix-parent closure is the weaker property,
and it can be **induced** by a column remap. That is why a remap looked promising —
but the *oracle* needs a different set — the anchor-*dropping* subsets — and a
restricted generation base never produces those. Both halves of that reasoning are
right; the conjunction does not follow. See the remediation section for the
measurement.)*

### The full 2×2 — it loses 95–99% in every configuration users can reach

`apriori()` sets both gates from the single `prune_equal_support` flag
(`core/apriori.py:508-509`: `prune_non_free=prune_equal_support`,
`prune_apriori=prune_equal_support`), so **both gates on** is what the public API
produces. 40,000 rows / 32 items, `min_support` 0.05, `max_length` 5; unanchored
reference 106,029 frequent / 87,459 free-sets:

```
  LOW  [0,1,2,3]      gates=OFF   expected 59,191 anchored   got 59,219   MISSING      0 ( 0.0%)
  LOW  [0,1,2,3]      gates=ON    expected 44,537            got  2,267   MISSING 42,298 (95.0%)
  HIGH [28,29,30,31]  gates=OFF   expected 61,591            got    463   MISSING 61,156 (99.3%)
  HIGH [28,29,30,31]  gates=ON    expected 46,624            got    431   MISSING 46,221 (99.1%)
```

**Only the top-left cell is safe, and it is not a configuration the public API can
produce.** Sort position determines *which* channel dominates, never whether the
feature works: through `apriori()` the loss is 95–99% at any anchor position. A
reader given only the gates-off contrast would conclude "put the anchors at low
column indices and you are fine" — which is false.

### The mechanism, isolated by anchor position (gates OFF)

Same data (40,000 rows / 32 items, `min_support` 0.05, `max_length` 5), three
anchor sets, **`prune_non_free` and `prune_apriori` both False**:

```
anchors=[0,1,2,3]      expected 59,191 anchored frequent itemsets   got 59,219 rows   MISSING      0 (0.0%)
anchors=[28,29,30,31]  expected 61,591                              got    463 rows   MISSING 61,156 (99.3%)
anchors=[15,16]        expected 28,232                              got  1,186 rows   MISSING 27,076 (95.9%)
count mismatches: 0 in all three
```

With the gates off, anchors that sort **first** lose nothing — every prefix-join
parent retains them — while anchors that sort **last** are almost entirely lost.
That isolates channel 2 cleanly, and it is what proves the mechanism is positional
rather than a coincidence of the data.

**Reproduced through the public entry point**, on a dataset with three planted
anchored triples whose parent pairs are unanchored:

```
mine_two_phase, anchors from phase 1 = [10..19]
phase-2 free-sets containing an anchor: 273;  mine_two_phase returned 280
MISSING 3 per-K {3: 3}, e.g. (0,1,15), (2,3,16), (4,5,17)
```

Each missing triple is frequent, free, anchored **and** apriori-valid; its prefix
parent — `(0,1)`, `(2,3)`, `(4,5)` — contains no anchor and was dropped at K=2.

### Two channels, each independently sufficient

An earlier reading had channel 1 as a minor increment on top of channel 2. That was
an artefact of anchor choice, and the discriminating experiment settles it. Running
the gates *separately* through the internal entry point (the public flag ties them),
6,007 rows × 20 columns, nested vocabulary:

```
anchors = LOWEST four [0,1,2,3]     (neutralises channel 2 — the anchor is the
                                     minimum element, so it sits in the prefix and
                                     both prefix-join parents retain it)
  gates OFF     free=F apriori=F    mined 3,687 / 3,687 expected   MISSING     0
  apriori ONLY  free=F apriori=T    mined   930 / 3,687            MISSING 2,757  {3:73, 4:1015, 5:1669}

anchors = HIGHEST four [16..19]
  gates OFF                         mined   206 / 2,824            MISSING 2,618
  apriori ONLY                      mined   206 / 2,824            MISSING 2,618   <- identical
```

With low anchors, channel 2 is silent and **channel 1 alone loses 2,757**. With
high anchors the two rows are identical, because channel 2 removes the itemsets
before the oracle can reject them. **Both channels are independently sufficient;
they are separable only by anchor sort position** — which is why a reviewer
looking from either configuration could see only one of them.

The casualties were then attributed **mechanically rather than by argument**, on
the 42,298 lost in the LOW/gates-ON cell:

```
  with an UNANCHORED prefix-join PARENT (channel 2):        0 / 42,298   ( 0.0%)
  with ANY unanchored (k-1)-SUBSET     (channel 1):    33,006 / 42,298   (78.0%)
  with NEITHER -> cascade (second-order, same root cause):  9,292   (100.0% verified)
  unexplained:                                                  0   ( 0.0%)
  K=3 example (3,4,5): pair subsets [(3,4),(3,5),(4,5)] — unanchored: [(4,5)]
```

**The accounting closes at 100% with nothing residual.** The "cascade" label was
initially asserted by inference; it was then tested — does each such itemset have
a (k−1)-subset that is anchored but *absent from the level the run actually
mined*? — and came back **9,292 of 9,292**. So the two channels plus their
second-order cascade account for every lost itemset; there is no third mechanism.

`(4,5)` carries no anchor, the mask removes it from the K=2 level, and
`_prune_groups_apriori`'s K=3 test at `gpu/mining.py:352-357` then rejects
`(3,4,5)`. Channel 2 contributes exactly **zero** here, because with low-sorting
anchors the anchor is the minimum element, sits in the prefix, and both
prefix-join parents retain it.

*One figure not to overstate:* the survivors of the apriori-only configuration are
**predominantly**, not exactly, the ≥2-anchor itemsets — measured 570 of 930
(61.3%). `_prune_groups_apriori` marks validity per suffix **slot**, not per pair
(`gpu/mining.py:373-374`), so a suffix kept for one valid pair drags every other
pair in its group along and single-anchor itemsets ride in as a by-product.
**That is the same per-slot under-pruning #19 files against the docstring at
`gpu/mining.py:262-268`** — one code property seen from opposite sides: there it
means the prune keeps more than it claims, here it means the damage is near-total
rather than total.

**Channel 1 is self-amplifying.** Splitting the 2,757 by whether an anchored parent
was already missing at k−1:

```
DIRECT  (all anchored parents survived; killed by the oracle itself):   762   {3:73, 4:689}
CASCADE (>=1 anchored parent already lost at k-1)                   : 1,995   {4:326, 5:1669}
  direct  e.g. (0,4,5,6)   — frequent-but-unanchored (k-1)-subset [(4,5,6)]
  cascade e.g. (0,1,3,4,5) — anchored parents already missing at K=4
```

A candidate pruned at level k is never counted, so it is absent from the generation
base at k+1 regardless of anchoring. Direct kills stop at K=4 in this fixture;
all of K=5 is compounding.

**Why the suite does not catch it.** `tests/test_row_split_e2e.py:102-106` is the
only anchor test, and it deliberately sets `phase2_support == phase1_support` so
that "every frequent item is an anchor" — the mask is then all-True and the filter
is a no-op. `grep -rn anchor tests/` returns that one docstring line. Nothing
anywhere runs the two-phase path with distinct supports, which is the entire point
of two-phase mining.

### Fix — and a trap in the obvious one

**The correctness fix:** filter only what the level *emits*
(`gpu/row_split.py:670-672`), leaving both `current_flat` and `full_flat`
unfiltered — i.e. treat anchoring as an **output selector**, not a mining gate. It
cannot be a generation restriction either, because the property it restricts on is
not anti-monotone. Apply it at K=1 too (`:357-360`), which today emits every
frequent item because `_anchor_keep_mask` returns `None` for k<2
(`gpu/mining.py:223`); under this fix that stops being a separate inconsistency and
becomes one line of the same patch, and it is free — generation reads
`prev_frequent_flat`, not the emitted array.

> ⚠ **This fix will make Phase 2 much slower, and whoever applies it must be told
> why that is correct.** The anchor filter is documented as cutting candidate
> explosion "from billions to a tractable search space" (`gpu/mining.py:234-236`).
> Leaving the generating level unfiltered means Phase 2 mines the **full** lattice
> at its ultra-low support and merely post-filters — possibly intractable at
> `mine_two_phase`'s own default `phase2_support=0.00001` (`gpu/row_split.py:762`).
> A fixer who applies this patch, watches Phase 2 OOM and reverts it will have
> restored a 99.3% silent data loss to fix a performance problem.
>
> **The accounting, stated carefully.** The filter runs *after* counting at every
> level — `:513-521` then `:529-531` (K=2), `:601-609` then `:628-630` (K≥3 dense),
> `:456-464` then `:471-482` (sparse) — so it never removed a single candidate from
> the level it is applied at. Its entire performance contribution came from
> shrinking the *next* level's generation base, and **as currently arranged that is
> exactly the unsound part**: the speed was purchased by silently discarding up to
> 99.3% of the correct answer.
>
> **And it is not recoverable.** A pruning-preserving variant was proposed twice,
> and measured to fail — see below. The candidate-space reduction and the apriori
> prune are mutually exclusive, so the fixed feature delivers no speedup.

### The fix — emit-only, and it is the only correct one
**VALIDATED end-to-end.** Filter only at
`gpu/row_split.py:670-672`, leaving `current_flat` and `full_flat` unfiltered; add
the mask at the K=1 emit site (`:357-360`); drop the `k < 2` clause in
`_anchor_keep_mask` (`gpu/mining.py:223`); leave `n_freq` alone. Applied to a
**copy** of the tree and run under `PYTHONPATH` (`git status --porcelain src/`
clean — no repository file touched):

```
getDeviceCount()=2   et_miner loaded from .../scratchpad/patched/et_miner

 gates=OFF anchors=[28,29,30,31]: expected 61,591 got 61,591 | missing=0 extra=0 countmm=0 | maxK 5 vs 5 OK | K=1 non-anchor 0
 gates=OFF anchors=[0,1,2,3]    : expected 59,191 got 59,191 | missing=0 extra=0 countmm=0 | maxK 5 vs 5 OK | K=1 non-anchor 0
 gates=OFF anchors=[15,16]      : expected 28,232 got 28,232 | missing=0 extra=0 countmm=0 | maxK 5 vs 5 OK | K=1 non-anchor 0
 gates=ON  anchors=[28,29,30,31]: expected 46,624 got 46,624 | missing=0 extra=0 countmm=0 | maxK 5 vs 5 OK | K=1 non-anchor 0
 gates=ON  anchors=[0,1,2,3]    : expected 44,537 got 44,537 | missing=0 extra=0 countmm=0 | maxK 5 vs 5 OK | K=1 non-anchor 0
 gates=ON  anchors=[15,16]      : expected 24,590 got 24,590 | missing=0 extra=0 countmm=0 | maxK 5 vs 5 OK | K=1 non-anchor 0

mine_two_phase repro:
  phase-2 free-sets containing an anchor: 273; mine_two_phase returned 273
  MISSING 0 — the three planted triples are back;  non-anchor rows emitted: 0

full suite against the patched copy: 479 passed, 6 skipped  (identical to HEAD)
```

All six anchor × gate configurations exact, including the two that were 95.0% and
99.3% broken.

**Also verified on the sparse CSR branch**, which those six did not cover — and it
matters, because the block the patch deletes at `gpu/row_split.py:471-482` is the
sparse one, filtering `_surv` in lockstep with `current_flat`, and `_surv` feeds
`materialize_survivors`, whose per-shard length check at
`gpu/sparse_csr.py:310-318` **raises** rather than warns:

```
n_gpus=1 (shards [1,1]) and n_gpus=2 (shards [2,2]), sparse_from_k=3 forced,
{gates off, gates on} x anchors {[28,29,30,31], [0,1,2,3], [15,16]}
  all twelve -> missing=0 extra=0 countmm=0, maxK 5 vs 5 OK, K=1 non-anchor 0
  dense->sparse transitions=1 and materialize_survivors calls=2 per run
  the per-shard length assertion never fired
```

So the fix is verified on **every branch it touches**: dense K=2, dense K≥3,
sparse CSR at one and two shards, and the public `mine_two_phase` entry point. **This is the only proposed fix measured to close both channels** —
the gates-ON rows went from 42,298 and 46,221 missing to zero, which a
hoist-only variant would not do for the gates-OFF rows and an emit-mask-only
variant would not do for the gates-ON rows. The `maxK 5 vs 5` column is the
falsifier for the `n_freq` coupling: it held everywhere.

**Cost:** the generating level stays unfiltered, so Phase 2 mines the full lattice
at `phase2_support` and post-filters — likely intractable at `mine_two_phase`'s
default `0.00001`. It also costs no GPU work to remove, because the filter always
ran *after* `run_chunked_dense_level` and never saved a kernel launch.

### A pruning-preserving fix was sought, measured, and does not exist

Two variants were proposed to keep the candidate reduction. Both were implemented
or faithfully simulated, and both fail.

**hoist + remap** — hoist `full_flat = current_flat` above the anchor filter *and*
remap anchor columns to `[0, n_anchors)`. An early simulation showed it exact
(321/321), but that simulation fed the oracle the **true complete lattice level**.
A real hoist can only preserve what was *generated*, and generation runs off the
anchor-filtered level. **Implemented for real** in a scratch copy — `full_flat` /
`full_counts` captured before the anchor mask in all three branches, carrying the
lexsort with it, nothing else changed, `git status --porcelain src/` clean:

```
6007x20, min_support 0.05, max_length 5, gates ON, expected = anchored free-sets K>=2
  REMAP [0,1,2,3]  hoist=off       mined 126/321  MISSING 195  {3:47, 4:125, 5:23}
  REMAP [0,1,2,3]  hoist=FAITHFUL  mined 192/321  MISSING 129  {      4:106, 5:23}   <- the real fix
  REMAP [0,1,2,3]  hoist=ideal     mined 321/321  MISSING   0  {}                    <- the flawed simulation
```

The per-K column is the argument made visible: **the hoist repairs K=3 completely**
(47 → 0), because the K=3 oracle needs only the complete K=2 level and K=2's
pre-filter level *is* complete (K=1 is never anchor-filtered). **From K=4 it
fails**, because level k−1 was itself generated from an anchor-restricted base.
hoist+remap loses 129 of 321, and the residue grows with K. It is not a fix.

> **Why the simulation misled, and it is this report's recurring shape again.** The
> simulated oracle and a real one differ by exactly the itemsets that were *never
> generated* — a substitute that looks equivalent and is not. At 321 itemsets that
> difference is invisible; at 44,537 it is dominant. The fixture that motivated the
> claim could not have falsified it.

**Independently confirmed on a fixture ~130× larger**, where the two hypotheses are
separated by three orders of magnitude rather than by tens of itemsets. A
*hoist-only* patch (snapshot `current_flat` before each anchor filter, sorted to
match the level-end lexsort, used for `full_flat`; `current_flat` still masked):

```
anchors=[0,1,2,3] gates=ON    expected 44,537   got 3,980
  hoist-only missing per-K: {4: 7066, 5: 33519}          total 40,585
  unpatched baseline      : {3: 314, 4: 7825, 5: 34159}  total 42,298
      -> K=3: 314 -> 0    K=4: 7,825 -> 7,066    K=5: 34,159 -> 33,519
      -> recovers 1,713 of 42,298 = 4.1%

anchors=[28,29,30,31] gates=ON  hoist-only {3:1272, 4:8780, 5:36169} = 46,221
                                unpatched  {3:1272, 4:8780, 5:36169} = 46,221
      -> IDENTICAL: the hoist changes literally nothing, because it never
         touches the generation base

attribution of the 40,585 still missing after the hoist:
  with an unanchored prefix-join PARENT:       0    <- an anti-monotonicity
  with an unanchored (k-1)-SUBSET:        32,692       obstruction would show up here
  with NEITHER -> cascade (verified 100.0%):   7,893
  unexplained:                                     0
```

Cascade verified at **7,893 of 7,893** here too, and the bookkeeping closes
*across* the two runs as well as within each: hoisted, the cascade sits entirely
at K=5; unpatched it is K=4 (759) and K=5 (8,533). The hoist repairs K=3, which
removes the K=4 cascade layer — and those 759 are exactly the unpatched-minus-hoisted
K=4 difference (7,825 − 7,066). That independently confirms why K≥4 moves under
the hoist at all.

**That attribution falsifies the anti-monotonicity framing directly**, rather than
merely failing to support it. Zero parent-level failures means the prefix-join is
*not* the barrier — generation closure is intact and the remap does give it what
it needs. The loss is entirely in the oracle, which needs subsets a restricted
generation base never produces.

*The 4.1% is from the most favourable faithful implementation available:* the
snapshot was explicitly sorted before use, so it is not depressed by the
lexsort trap below. A hoist given its sorted oracle still recovers 4.1%.

**The hoist can only ever repair the level immediately below the first filtered
one.** Concretely: `(4,5,6)` carries no anchor, so its parents `(4,5)` and `(4,6)`
were removed from the K=2 level; it was never a K=3 candidate, so it is absent
from K=3's `full_flat` even after the hoist, and the K=4 oracle still cannot see
it.

A prediction was committed in advance — "K=3 → 0; K=4 and K=5 stay at roughly 125
and 23" — and K=3 and K=5 landed exactly. K=4 moved 125 → 106 for a second-order
reason worth recording: recovering the K=3 candidates enlarges the K=3 *emitted*
level, which then feeds K=4 generation. A knock-on effect, not a direct one. The
real implementation reproduced the faithful simulation to the itemset.

**The general account, which forecloses the design space rather than these two
proposals.** The apriori prune must test the (k−1)-subsets that *drop* the anchor
(`candidate[:d] + candidate[d+1:]` over the prefix, `gpu/mining.py:363-368`; at
`d=0` that removes the minimum element — under a remap, the anchor itself). Those
subsets are unanchored, and they exist in `prev_full_flat` only if they were
generated and counted, which happens only if the generation base was unfiltered.
**Restricting the generation base by a predicate that is not anti-monotone starves
the apriori oracle of exactly the subsets it must test, and no column ordering
repairs it — ordering changes which subsets go missing, never whether they do.**

A nuance worth keeping, because it explains why the remap looked promising: the
*generator* needs only closure under the two prefix-parents, which is strictly
weaker than anti-monotonicity, and a remap does induce it. The *oracle* needs more,
and nothing induces that. So the candidate-space reduction the feature exists for
and the apriori prune are **mutually exclusive**.

**Consequence, stated so it survives a reader trying to falsify it.** The anchor
filter can only be an **output selector** *in the configuration that matters*. A
generation restriction is separately achievable — the remap does make every
anchored itemset's minimum an anchor, so both prefix-join parents are retained and
generation stays complete; with **both gates off** that is sound, and the measured
gates-off row is 0 missing in every run. What it cannot do is coexist with the
subset prunes, because the apriori test needs the (k−1)-subsets that are *not*
anchored and a restricted generation base never produces them.

`mine_two_phase` forces `prune_equal_support=True` (`gpu/row_split.py:866-867`),
so **the configuration where the restriction would be sound is not the feature's
configuration**. Its documented purpose at `gpu/mining.py:236-238` ("reducing
candidate explosion") is therefore not delivered by the fixed version either, and
`mine_two_phase` needs redesigning or documenting as post-filter-only. The reader
should not conclude the fix traded correctness for lost performance: in the
configuration that matters there was no sound performance to lose.

*(With the free-set prune on but the apriori prune off, a remap-based restriction
under-prunes rather than losing itemsets — a wrong output of a milder class. That
combination is not reachable through `apriori()`, which ties both flags.)* If candidate reduction is genuinely needed, the
one sound shape left is *per-anchor conditional databases* — for each anchor `a`,
mine the projection onto transactions containing `a`, which is downward-closed
within itself, then union and dedupe. One run per anchor is the cost.

> **Implementation caveat for anyone attempting a hoist anyway.** The level-end
> lexsort is at `gpu/row_split.py:641`, *below* the anchor filter at `:628-630`,
> while `full_flat = current_flat` is at `:651`. Hoisting `full_flat` above the
> filter also lifts it above that sort, producing an unsorted `prev_full_flat` that
> `_prune_non_free_mask`'s Rust path rejects with
> `ValueError: prev_flat must be sorted lexicographically by row for the
> free-set-prune binary search`. A hoist must carry the sort with it. The guard
> raises loudly rather than binary-searching unsorted data silently — that one is
> working as intended.

---

## 23. Four multi-GPU kernels silently truncate survivors at any magnitude

**Severity: HIGH.** Found by three reviewers; measured on two visible devices.

```python
kernels/k3plus.py:447        n = min(n, gpu_max)   # LIVE  (dispatch_k3plus_fused:129 <- mining.py:729)
kernels/gpu_resident.py:338  n = min(n, gpu_max)   # LIVE  (:157 <- mining.py:916)
kernels/gpu_resident.py:491  n = min(n, gpu_max)   # LIVE  (:189 <- mining.py:942)
kernels/k3plus.py:193        n = min(n, gpu_max)   # exported but unreachable in-tree
                                                   #   (only via bare dispatch_k3plus, 0 callers)
```

**Three live sites, one exported-but-unreachable** — stated precisely so a
maintainer who checks the fourth, finds nothing calls it, and starts doubting the
other three does not have to. It also means one of the four fan-out functions and
its dispatcher are already dead weight, which lowers the bar for deleting them.

Their single-GPU siblings (`k3plus.py:84`, `:305`, `gpu_resident.py:135`, `:233`)
call `_warn_result_truncation`, which **raises** above 5% loss. `k2.py:191` calls
it on the multi-GPU path too — which is what makes these four drift rather than a
deliberate convention.

Measured, two visible devices, 4,096 rows × 40 columns, 9,880 K=3 survivors,
per-GPU slice 4,940:

```
per-GPU overflow    multi-GPU                       single-GPU, same data
      3.0%          298 lost, SILENT                297 lost, logger.warning
     10.0%          988 lost, SILENT                RuntimeError
     60.0%        5,928 lost, SILENT                —
```

**The 10% row is the point:** the same overflow the single-GPU path refuses to
proceed on, the multi-GPU path completes silently, 988 frequent itemsets short.
There is no window, no warning, and no magnitude at which these four become an
error. A silent 60% loss cannot be caught by a log grep. The truncated level then
feeds candidate generation, so the loss compounds at every deeper K.

**Production trigger:** >10M survivors in a single GPU's candidate slice at one
level, reached automatically via `dispatch_k3plus_fused` / `dispatch_k3plus_gpu_resident`
at ≥500,000 candidates with >1 GPU. That was not reachable on this hardware; both
reviewers exercised the identical code path with a reduced `max_results`.

> **Reproduction note — two traps that produce false negatives.**
> 1. `max_results` is a **per-GPU** cap (`gpu_max = min(n_gpu_cands, max_results)`)
>    and the per-GPU slice is `ceil(total / n_gpus)`, so a cap at or above the
>    slice cannot overflow however small it looks against the total.
> 2. Setting `CUDA_VISIBLE_DEVICES=0` makes `getDeviceCount()` return 1, and the
>    multi-GPU wrappers then silently degrade to the single-GPU path through
>    `n_gpus = min(n_gpus, available_gpus, ...)` at `kernels/k3plus.py:119-122`,
>    `:373-374` and `kernels/gpu_resident.py:416-417`. The test prints the
>    single-GPU `RuntimeError`s and looks like it has refuted its own hypothesis
>    while never having exercised the code under test. **Assert
>    `cupy.cuda.runtime.getDeviceCount() == 2` inside the script.**
>
> Both traps caught a reviewer during this review.

### Fix, ranked — and #23 alone is hollow

**Must ship, 4 lines:** replace each bare clamp with
`n = _warn_result_truncation(n, gpu_max, f"filtered_kernel (device {device_id})")`.

**Must ship *with* it, or the first fix is hollow: #24.** The helper itself
tolerates ≤5% silently, so calling it at four more sites without removing that
tolerance simply extends a 5% silent-loss window to all eight. These are not
independent fixes.

**And that closes the argument for deletion.** Once the tolerance is gone, the
fan-out becomes multi-GPU mining with a hard 10M-survivors-per-GPU-per-level
ceiling that raises above it. The row-split path has no such ceiling — allcounts
plus `compact_threshold_filter` sizes exactly to the survivor count
(`kernels/filter.py:139-177`) — and `core/apriori.py:495` already routes every
genuine `n_gpus > 1` request there. So after the minimal fix the fan-out is a
strictly capability-limited version of a path that already exists, and which also
round-trips the entire bitvec matrix through host RAM.

**What deletion would cost, stated honestly:** it removes multi-GPU from
`gpu_resident` mode entirely (there is no row-split gpu-resident path —
`core/apriori.py:402-413` and `:517-528` route `gpu_resident=True` straight to
`_apriori_from_bitvecs_gpu_resident`), and removes throughput-only multi-GPU from
the `bitvecs=` entry point. Neither is a **capacity** loss: the fan-out replicates
the full bitvec to every device, so it never let anyone mine anything that did not
already fit on one card. The four names are exported from `et_miner.gpu.kernels`,
so the safe sequence is **fix → log → deprecate → delete**, not delete now.

**The middle option, not recommended but available:** make the fan-out
overflow-safe rather than deleting it. The kernels already keep counting past
capacity (`_src/k3plus_fullyfused.cu:83-87` increments `n_results` unconditionally
and gates only the write), so the exact count is available, and
`kernels/shared_tiled.py:154-189` already implements the re-allocate-and-rerun
protocol — ~15 lines per site. It is not recommended because it would make four
more copies of a protocol that exists once correctly, which is the duplication
that produced this drift in the first place. If it is chosen anyway, note each
device must re-allocate to **its own** reported count, not a global one.

---

## 24. `_warn_result_truncation` itself tolerates ≤5% loss

**Severity: MEDIUM.** Measured.

```python
kernels/loader.py:32-40
    if n_actual > max_results:
        overflow_pct = (n_actual - max_results) / n_actual * 100
        if overflow_pct > 5:
            raise RuntimeError(...)
        logger.warning(msg)
    return min(n_actual, max_results)
```

So even the "correct" single-GPU call sites drop frequent itemsets whenever the
overflow is under 5% — the window `10,000,000 < n ≤ 10,526,315` at one level.

```
max_results=9,880 (no overflow):  returned 9,880 of 9,880 ->   0 lost
max_results=9,583 (3% overflow):  returned 9,583 of 9,880 -> 297 itemsets SILENTLY LOST
   WARNING | Result truncation: 9,880 found but buffer=9,583 (3.0% lost). filtered_kernel
max_results=8,892 (10% overflow): RuntimeError raised

three repeats at 3% overflow: sizes [9583, 9583, 9583]
                              identical kept-set? False   pairwise symmetric difference 30-32
```

**The dropped set is non-deterministic** — the kernels append via `atomicAdd` — so
these routes disagree with the row-split path, with the CPU tiers, and **with
themselves run twice**. The tier-equivalence chain CLAUDE.md mandates cannot hold
at that scale even against itself.

There is no percentage of silently-lost frequent itemsets that is acceptable for a
miner whose contract is exactness. Either make any overflow raise, or adopt the
two-pass protocol at `kernels/shared_tiled.py:154-189`.

---

## 25. K≥3 kernels read an uninitialised `s_items[]` slot and write one past the array

**Severity: MEDIUM.** Found by all four reviewers; measured by two.

Each kernel declares `__shared__ int s_items[64]` immediately followed by
`__shared__ unsigned long long warp_sums[8]`. **There are two different effective
caps, and neither is enforced on the host.**

**(a) The three group kernels** — `_src/k3plus_dense.cu:36-41`,
`_src/k3plus_gpu_resident.cu:40-45`, `_src/k3plus_fullyfused.cu:41-46`. The fill
loop is guarded `threadIdx.x < prefix_len && threadIdx.x < 62`, but **thread 0
separately writes `s_items[prefix_len]` and `s_items[prefix_len+1]`**, which at
`prefix_len == 62` land exactly on slots 62 and 63. So these are **correct at
K=64** and break at K=65, where slot 62 is never written (and read as a column
index) while `s_items[prefix_len+1]` writes index **64** — one past the array,
into `warp_sums`, corrupting the block reduction too.

Both directions were verified, which is what pins the boundary:

```
prefix-sensitive (640x72, column 62 all-zero; a missed PREFIX slot flips the answer):
  count_k3plus_dense  K=63 prefix_len=61  expected 640  got 640  OK
  count_k3plus_dense  K=64 prefix_len=62  expected 640  got 640  OK
  count_k3plus_dense  K=65 prefix_len=63  expected   0  got 640  *** WRONG ***
  gpu_resident        K=65                expected   0  got 640  *** WRONG ***

suffix-sensitive (640x70, suffixes = 62,63; a missed SUFFIX slot flips the answer):
  K=64 prefix_len=62  expected 0  got 0  OK   <- proves thread 0 DID write slots 62 and 63
```

**(b) `count_itemsets_fused_k3plus`** — `_src/k3plus_fused.cu:27-30` caches the
*whole* itemset with no separate suffix write, and its own comment says "supports
up to K=62". Its guard `threadIdx.x < k && threadIdx.x < 62` leaves slots 62-63
unwritten while the AND loop reads `s_items[0..k-1]`:

```
count_itemsets_fused_k3plus  K=62  expected 0  got   0  OK
count_itemsets_fused_k3plus  K=63  expected 0  got 640  *** WRONG ***
count_itemsets_fused_k3plus  K=64  expected 0  got 640  *** WRONG ***
```

> **Which K fails depends on where the discriminating item sits — and that was
> pinned down.** Re-probing with the all-zero column placed in a **prefix** slot
> versus a **suffix** slot, on two matrix widths (8 clean probes, no luck
> involved):
>
> ```
>  640x72 and 640x80, only column 62 all-zero
>     k  | zero col in a PREFIX slot          | zero col in a SUFFIX slot
>        | true  dense       gpu_resident     | true  dense    gpu_resident
>    62  |   0   0 OK        0 OK             |   0   0 OK     0 OK
>    63  |   0   0 OK        0 OK             |   0   0 OK     0 OK
>    64  |   0   0 OK        0 OK             |   0   0 OK     0 OK
>    65  |   0 640 BAD     640 BAD            |   0   0 OK     0 OK
>  identical on both widths
> ```
>
> At `prefix_len = 63` the unwritten slot `s_items[62]` is a **prefix** slot, so a
> prefix-borne item is silently skipped while a suffix-borne one still gets ANDed
> via thread 0's write to `s_items[63]`. **The k=65 failure is therefore
> deterministic for prefix-slot items**, not merely undefined. Earlier runs that
> saw different failing K sets (63/64/66 wrong with 65 returning 0) were reading
> whatever the leftover shared contents happened to be for a *suffix*-borne
> discriminator — so a fix must be verified with the discriminating column in a
> prefix slot, or it can appear to pass.

**Only `shared_tiled.py:42-47` (`_assert_k_cap`) enforces anything**, raising
`ValueError("shared kernel supports K <= 62 ...")` on the identical shape. The
legacy, fully-fused and gpu-resident wrappers enforce nothing —
`count_k3plus_allcounts` (`kernels/k3plus.py:630-710`),
`count_k3plus_fully_fused` (`:222`), `count_itemsets_fused_k3plus` (`:14`) and
`count_k3plus_gpu_resident` (`gpu_resident.py:67`) all pass the shape through.

### Fix — host-check only, zero `.cu` edits

`_assert_k_cap` (`kernels/shared_tiled.py:42-47`) already exists and already
raises. **Call it from the four unguarded wrappers** — `kernels/k3plus.py:677-710`
(legacy branch), `:246-298`, `:14-91`, and `gpu_resident.py:86-129` — imposing a
uniform `k <= 62`.

62 is already correct for `count_itemsets_fused_k3plus` as written, and sits two
levels below the group kernels' true k≤64 capacity. That conservatism costs
nothing precisely *because* those levels are unreachable — the same fact that makes
this LOW — and it introduces no new undefined behaviour. **One guard, four call
sites, no fifth copy of the rule, and no kernel source touched.** If a proposed
patch edits `.cu` bounds, block it.

> ⚠ **Do not widen the device guard.** An earlier recommendation here was to
> change `_src/k3plus_fused.cu:30` to `threadIdx.x < k && threadIdx.x < 64`. It was
> compiled and measured on a modified copy:
>
> ```
>  640x72, only column 62 all-zero, candidate = {k-1 all-ones cols} + col 62
>     k   true   HEAD(<62)     patched(<64)
>    62      0     0   OK         0   OK
>    63      0   640 WRONG        0   OK
>    64      0     0   OK         0   OK
>    65      0     0   ?        640   ?      <- both undefined; see below
>    66      0    73 ?           21   ?
> ```
>
> **Read that k=65 row carefully — it is not evidence of a regression.** Both
> outcomes are undefined behaviour: at HEAD threads write `s_items[0..61]` while
> the loop reads i=0..64, so slots 62-63 are uninitialised *and* index 64 is past
> the array; patched, threads write `s_items[0..63]` and the loop still reads index
> 64. The `0` at HEAD is the same luck noted above, not a correct answer.
> Comparing two undefined outcomes, one run each, establishes nothing — and
> "widening made it worse" would be the wrong lesson to bank.
>
> **The conclusion survives on better footing:** widening extends *correct*
> behaviour from k≤62 to k≤64 but leaves k≥65 silently corrupt, so it is a
> capacity increase dressed as a fix. Since those levels are unreachable anyway,
> the capacity is worth nothing and the risk is real. Likewise do not widen the
> three group kernels' fill guard — that does not address `prefix_len == 63`, where
> thread 0 still writes index 64, while making the code look repaired.

### Severity: LOW — and how it got there is worth recording

This rating moved in both directions before settling, and it settled on a checked
fact rather than a vote. The case for MEDIUM was that
`count_itemsets_fused_k3plus` takes a **caller-supplied** candidate list, with `k`
read from `offsets[cand+1] - offsets[cand]` (`_src/k3plus_fused.cu:22-24`) — so
its k≥63 input would require nothing of the lattice. **That leg was checked and
failed.** The live caller is `streaming/async_pipeline.py:327`, and three lines
above it `:313` reads `candidates = _generate_candidates(prev_frequent, k)` inside
an ordinary apriori level loop — lattice-derived, so the same bar applies. The
only other caller-supplied-list entry point, bare `dispatch_k3plus`
(`gpu/dispatch.py:76`), has **zero callers** in `src/`, `tests/` or `bench/`;
verified. It is exported at `kernels/__init__.py:58` and unreachable in-tree.

So there is no in-tree path where `k` is not lattice-bound, and reaching k=63 or
k=65 requires a frequent 62- or 64-itemset — all 2⁶² subsets frequent, which no run
completes. `ET_MINER_KERNEL_VARIANT=legacy` and the planner's `use_legacy` routing
change *which kernel runs*, not *which K is reached*, so they do not lower the bar;
the default variant is "shared" (`gpu/dispatch.py:23`), which raises cleanly.

**LOW** — and worth fixing anyway, because the fix is four host-side lines calling
a guard that already exists.

---

## 26. The row-split miner returns three different `itemset` dtypes

**Severity: MEDIUM** (values are unaffected — only the schema differs). Measured.

`gpu/row_split.py:733-744` builds the result through Arrow from `flat_values`,
which is **int32** because `col_to_item_arr` is `np.zeros(n_cols, dtype=np.int32)`
(`:166`). The fallback at `:747-756` builds from `arr[i].tolist()` — Python ints,
which Polars infers as **Int64**. `_empty_result()` (`:368`, `:721`) is also
Int64. And the switch between the first two is invisible, because `:745` is:

```python
except (ImportError, Exception):
```

`Exception` subsumes `ImportError`, so the tuple is redundant *and* every failure
inside the Arrow block is swallowed — a `MemoryError` from `np.concatenate` at
campaign scale, or an offset overflow in `pa.LargeListArray.from_arrays`, silently
changes the returned schema instead of surfacing.

```
row-split, in-memory return   : Schema([('itemset', List(Int32)), ('support', Float64)])  (696 rows)
row-split, output_dir return  : Schema([('itemset', List(Int64)), ('support', Float64)])
CPU tier (_build_result_df)   : Schema([('itemset', List(Int64)), ('support', Float64)])  (696 rows)

pl.concat(gpu, cpu) -> SchemaError: type Int64 is incompatible with expected type Int32

replaying both branches verbatim on the same arrays:
  fast == slow schema?               False
  fast == other tiers?               False
  fast == its own empty-result?      False
  values identical:                  True
```

So one function returns three different schemas depending on a path the caller
cannot observe, and the common one is incompatible with the rest of the library.
The tier-equivalence tests cannot see it because `_result_to_counted_set` calls
`.to_list()`, which erases the dtype.

This is #20's documentation defect with a behavioural consequence: because the two
real return paths bypass `_build_result_df`, they emit a schema `_build_result_df`
never would. Routing row-split through the shared builder — or casting
`flat_values` to int64 — closes both at once. Separately, narrow `:745` to
`except ImportError` so a genuine Arrow failure surfaces.

---

## 27. One `try` wraps two independent cleanups with a bare `pass`

**Severity: LOW.** Measured.

```python
gpu/row_split.py:704-708
        try:
            free_groups(_sparse_groups_gpu)
            sparse_state.release()
        except Exception:
            pass
```

`free_groups` does `with cp.cuda.Device(did): g.clear(); pool.free_all_blocks()`
(`gpu/sparse_csr.py:208-214`), which raises on a CUDA context or device error —
precisely the abort path this `finally` exists to handle. When it does,
`sparse_state.release()` never runs, so the resident CSR tidset shards (the largest
allocation in sparse mode) are never explicitly freed, and the bare `pass`
discards the reason.

```
injecting a raising free_groups into a real single-GPU sparse-CSR run (sparse_from_k=3):
  exception propagated:      injected: device free failed (call 1)
  call sequence:             ['replace', 'CsrShard.free']
  free_groups() attempts:    2
  sparse_state.release() ran? False
  -> shards LEFT to refcounting (release() skipped by the shared try)
```

Refcounting is a real backstop — the arrays die with the frame once the traceback
is dropped — which is why this is LOW rather than a hard leak. But it violates
"released correctly on every path including error paths", and nothing appears in
the logs. Give each cleanup its own `try` and log the exception rather than
`pass`.

---

## 28. `_check_memory_guard` returns a truncated lattice as if it were complete

**Severity: LOW** — it cannot fire at the defaults (`max_vram_gb=70` on a 24 GB
card, `max_ram_gb=800` on a 125 GB host), so it needs an explicit user setting.
The guard's whole purpose is to be set, and when it fires the contract break is
total.

`_check_memory_guard` (`gpu/mining.py:503-520`) returns True on pressure, the
caller `break`s out of the level loop at `:759-761` with only a `logger.warning`,
and control falls straight through to `_build_result_df(results)` at `:775`. The
caller receives a DataFrame that stops at some K, with nothing on it to say so.

Reproduced:

```
4001 rows x 22 cols, min_support 0.05, max_length 6
  default guards (max_ram_gb=800): 31,160 itemsets, K levels [1,2,3,4,5,6]
  max_ram_gb=0.5                 : 31,160 itemsets, K levels [1,2,3,4,5,6]   (guard not tripped)
  max_ram_gb=0.0                 :    249 itemsets, K levels [1,2] only
                                     -> 30,911 missing (99.2%), no exception, plain DataFrame
```

It needs the user to set a low guard, so it is not silent-by-default — but the
guard's whole purpose is to be set, and when it fires the contract break is total.

It is also a **route-equivalence break of the same family as #23 and #24**: the
row-split miner has no such guard at all, so `_apriori_from_bitvecs` can silently
stop at a K where `_apriori_row_split_multi_gpu` continues — same data, same
flags, different lattice.

**#23, #24 and #28 are three instances of one failure mode:** the miner returns a
silently incomplete lattice through a value-returning API, with only a log line.
One is a buffer clamp, one a tolerance, one a memory guard, and all three end with
a normal-looking DataFrame. The single recommendation covering all three: an
exactness-contract miner must raise, or return an explicit `partial` signal —
never a bare truncated result.

*Related, not filed:* `gpu/mining.py:490` wraps `pool.used_bytes()` in
`except Exception` → `vram_gb = 0.0`, so the guard can never fire on VRAM at all.
That can only make a run return *more* results, so it cannot lose itemsets.

---

## 29. The apriori prune builds a full Python set before a Rust path that never reads it

**Severity: MEDIUM** (downgraded from HIGH by its own author on re-derived
arithmetic, before any peer challenged it).

```python
gpu/row_split.py:436   prev_freq_set = set(map(tuple, prev_full_flat.tolist()))   # sparse branch
gpu/row_split.py:546   prev_freq_set = set(map(tuple, prev_full_flat.tolist()))   # K>=3 dense branch
```

Both are evaluated eagerly as arguments. `_prune_groups_apriori`
(`gpu/mining.py:289-323`) tries `et_miner_rust.prune_groups_apriori` first, using
only `prev_flat_np`, and returns at `:314` **without ever reading
`prev_frequent_set`** — which is consumed only in the Python fallback at
`:331-338`.

That was proved rather than read off: passing a tripwire object in place of the
set, raising on `__contains__` / `__iter__` / `__len__`, the Rust path **completed
normally**.

```
rust available: True
Rust path with a tripwire sentinel -> the set is NEVER read (126 -> 126 candidates)
same call with the real set        -> 126 -> 126 candidates, identical: True
Python fallback                    -> prev_frequent_set WAS read (required only here)

held after gc, k=6:
  n=  250,000  0.11 GB  built in 0.17s   -> 0.44 GB/M
  n=1,000,000  0.46 GB  built in 0.85s   -> 0.46 GB/M
  n=4,000,000  1.74 GB  built in 3.69s   -> 0.44 GB/M
what the Rust path actually consumes (np.ascontiguousarray, n=1M): 0.000 GB, 0.000s  [zero-copy]
```

That is an **upper bound** — RSS also counts the numpy source array. The figure to
quote is the allocation the code is itself responsible for, measured at k=5 with
the whole build timed:

```
          N     time   peak GB   B/row
  1,000,000     3.3s     0.36      357
  5,000,000    18.4s     1.75      350
 10,000,000    35.3s     3.50      350
```

**350 B/row, 3.50 GB at 10M itemsets, and 35.3 seconds per pruned K level** — on a
value the Rust branch returns without ever reading it. Extrapolating to the 430M
the code's own comment plans for: ~150 GB and ~25 minutes **per level**. On a
large-RAM host the wall-clock is the sharper edge, not the memory, and it is the
reason this does not drop below MEDIUM.

Cost, **measured by RSS** at k=6 rather than estimated:

```
   n_itemsets   tolist MB   set MB   total MB   B/itemset
      100,000        29.5     13.9       43.4         455
      400,000        82.8     59.0      141.8         372
    1,600,000       325.1    241.6      566.6         371
```

Two numbers matter and they differ. **Peak** — the `.tolist()` temporary and the
set are both live during construction, because `tolist()` materialises fully
before `map` runs — is 371 B/itemset, i.e. **~3.7 GB at 10M itemsets and ~160 GB
at the 430M the code's own comment plans for** (`gpu/mining.py:121-123`).
Steady-state (set only, once the list is collected) is 158 B/itemset: 1.6 GB and
68 GB. **Peak is the one that OOMs**, and it scales with k.

**MEDIUM as filed, HIGH in the K≥7 campaign regime** — stated as an escalation
condition rather than a hedge. At 100M itemsets this is ~37 GB and at 430M ~160 GB,
and `gpu/row_split.py:190-191` independently documents host RAM as the binding
constraint in exactly that regime ("Prevents 680 GB CPU RAM accumulation at K=7+
scale"). Note also that the Python fallback the set exists for is **dead code on
every box matching CLAUDE.md's documented setup** (`cd rust_ext && maturin develop
--release`) — i.e. every campaign box, not an exotic configuration.

Against a default `max_ram_gb=800` (`core/apriori.py:249`), 3.5 GB is not a HIGH
resource defect — which is why the rating moved. **The stronger ground is that it
is unconditional O(N) GIL-bound Python work** — N tuple allocations and N hashes
per level — thrown away untouched whenever the Rust extension is present, directly
defeating what the same function advertises ("Rust fast path: HashSet + Rayon
parallel, GIL-free", `gpu/mining.py:288`). It escalates to HIGH at ≥100M-itemset
levels.

The in-tree counter-example is one function away: `_prune_non_free_mask`
(`gpu/mining.py:178-215`) passes only numpy and builds its dict lazily inside
`_prune_non_free_mask_python` (`:145-154`). Fix: make the set lazy at the two call
sites, or build it inside the fallback.

---

## 30. `del bitvecs_gpu` at the density transition frees nothing, and logs that it did

**Severity: MEDIUM.**

```python
gpu/mining.py:622-624
    del bitvecs_gpu
    cp.get_default_memory_pool().free_all_blocks()
    logger.debug("    Freed bitvec VRAM, ... tid entries in the CSR shard")
```

`bitvecs_gpu` is **parameter 1** of `_apriori_from_bitvecs` (`gpu/mining.py:403`).
Deleting the local binding cannot release the array: the caller's frame holds a
live reference (`core/apriori.py:514`, `:531-532`), and on the `bitvecs=` route the
array is **contractually caller-owned** (`core/apriori.py:304-306`) and must not be
freed here at all. So `free_all_blocks()` reclaims nothing and the transition runs
at bitvec **+** CSR peak — precisely the peak it exists to avoid.

**The false log line is its own harm.** On a memory-pressure path it is the
operator's only evidence, and it asserts something that did not happen.

**One defect, three sites — settled by measurement.** The sibling site at
`gpu/row_split.py:418-422` was initially filed as a tidiness flag on the grounds
that `bitvecs_list.clear()` at `:422` does the work. That holds on **one** of its
two routes:

```
bitvec is 0.60 MB
route A (row_split builds bitvecs_list itself, :149)
    -> pool holds 0.00 MB after the run   -> released by bitvecs_list.clear()
route B (caller supplies bitvecs_list, core/apriori.py:387-393)
    -> pool holds 0.60 MB, caller's array still usable=True   -> STILL HELD

and for gpu/mining.py:622-624 directly:
    pool used after the dense->sparse transition == bitvec size
    caller's bitvecs_gpu still usable: True   -> bitvec VRAM STILL HELD
```

So on the caller-supplied route `row_split.py:418-422` behaves exactly as
`mining.py:622-624` does. Record it as **one bug with three sites** —
`gpu/mining.py:622-624`, `gpu/row_split.py:418-422`, and the caller-ownership
aggravation at `core/apriori.py:304-306` / `:387-393` — rather than as a bug plus
a flag. Severity is bounded by route A genuinely releasing, and by the caller
being able to free it on route B; **the log line claiming a release that happened
on neither is the part that misleads.**

---

## 31. K≥3 dense group arrays (~40 GB) have no `finally`, where the sparse twin has one

**Severity: MEDIUM.** Two independent sources.

`all_groups_gpu` — described at `kernels/k3plus.py:597` as ~40 GB at K=8, resident
on **every** GPU — is uploaded at `gpu/row_split.py:585-587` and released only by
straight-line `del` at `:612-616`. The function's own `finally` at `:702-708`
covers `_sparse_groups_gpu` and the CSR shards but **not** this. Any exception
from `run_chunked_dense_level` between the two (a kernel launch failure, a CuPy
OOM in `_filter_compact`, or the `compact_threshold pass disagreement`
`RuntimeError` at `kernels/filter.py:180`) skips the release.

Measured by injecting a failure immediately after the group upload:

```
exception propagated: injected: kernel launch failed at K=3, after the group upload
group dicts uploaded before the failure: 1
still holding device arrays after the unwind: 1/1
   -> NOT released by any cleanup (relies on refcounting once the traceback dies)
```

The claim is deliberately structural rather than about traceback lifetime — CPython
clears `except ... as e` bindings and usually drops the frame promptly, so an
earlier "the retry starts with tens of GB already gone" framing was withdrawn. What
stands is the asymmetry: **the sibling implementation of the identical concern
makes it deterministic** (`gpu/mining.py:637-665`, `groups_gpu = upload_groups_to_shards(...)`
then `try: ... finally: free_groups(groups_gpu)`), while this one leaves release on
the exception path to interpreter teardown timing — and `free_all_blocks()` is never
called, so the pool never returns the ~40 GB to the driver.

This and #27 are the same class — error-path cleanup that the sibling module gets
right — and are best fixed together.

---

## 32. `_apriori_from_bitvecs` declares `n_gpus` and never reads it

**Severity: MEDIUM.** Found by two reviewers.

`n_gpus` appears exactly once in `gpu/mining.py` — at `:411`, in the signature.
The body never mentions it: `:685` calls `dispatch_k2(...)` and `:729` calls
`dispatch_k3plus_fused(...)`, neither passing it. `gpu/dispatch.py` decides the
route from the **physical** device count instead — `should_use_multi_gpu` reads
`get_gpu_count()` at `:34-36`, and `:57` and `:128` then call the multi-GPU
kernels with `get_gpu_count()`. `core/apriori.py:425` and `:540` do pass
`n_gpus=n_gpus` in, so the parameter is dropped **at the route boundary**, not
merely vestigial. `_apriori_from_bitvecs_gpu_resident` (`gpu/mining.py:822`) has
the same behaviour without even declaring the parameter.

**The caller's GPU budget is ignored in both directions.** `n_gpus=1` on a 2-GPU
box does not keep the run on the truncation-free single-GPU kernels — it lands on
the fan-out path of #23, and additionally pays a full host round-trip of the
bitvec matrix (`kernels/k2.py:130` `bitvecs_np = bitvecs_gpu.get()`, then `:150`
`cp.array(bitvecs_np)` per device; same at `kernels/k3plus.py:133`, `:382`,
`gpu_resident.py:283`, `:420`), which can OOM the host on a matrix that fit
comfortably in one GPU. `n_gpus=8` on a 2-GPU box silently uses 2. So a parameter
that reads as a resource cap is in practice a correctness-relevant route selector
the caller cannot set.

> **This review produced a live demonstration of the hazard.** A reviewer set
> `CUDA_VISIBLE_DEVICES=0` to avoid contending for a card. `getDeviceCount()` then
> returned 1, the multi-GPU wrappers silently degraded to the single-GPU path
> through `n_gpus = min(n_gpus, available_gpus, ...)`
> (`kernels/k3plus.py:119-122`, `:373-374`, `gpu_resident.py:416-417`), and a test
> of #23 printed the single-GPU `RuntimeError`s and looked like it had refuted its
> own hypothesis — having never executed the code under test.
>
> That clamp is defensive and correct in itself and is **not** filed. The point is
> the property it exposes: **which route executes is a function of ambient device
> visibility, not of the caller's request, and the switch is silent in both
> directions.** The reviewers hit the downgrade direction and got a false negative;
> a user hits the upgrade direction and gets every card after asking for one, on
> the silently-truncating wrappers of #23.

Measured directly, with hooks on both kernel entry points — one prefix group of
1,099 suffixes giving 603,051 candidates, `ET_MINER_KERNEL_VARIANT=legacy`:

```
getDeviceCount() = 2
dispatch_k3plus_fused(...) took: ['MULTI-GPU(n_gpus=2)']    # the caller asked for 1 GPU
```

**This materially raises #23's reachability.** A user does not have to ask for
multi-GPU to land on the four bare-clamp sites — asking for *one* GPU on a
two-GPU box is enough. #23's "production trigger needs a multi-GPU run" is
therefore weaker than it looks, and on a shared box this is also the concurrency
hazard in its own right: a job told to use one card takes both.

**The loop closes, and nothing records which way it went.** `gpu/dispatch.py:36`
and `:73` decide to *enter* the fan-out from `get_gpu_count()`, ignoring the
caller's `n_gpus`. Five sites then decide to *leave* it, silently, reconciling the
requested configuration with the available one:

```
kernels/k3plus.py:118-119        available_gpus = getDeviceCount(); n_gpus = min(n_gpus, available_gpus, n_candidates)
kernels/k3plus.py:370-371        same
kernels/gpu_resident.py:275-276  same
kernels/gpu_resident.py:413-414  same
kernels/k2.py:116-117            same
```

`gpu/dispatch.py` contains **no logging at all** — verified, not inferred. So the
route is chosen by the box on the way in, unchosen by the box on the way out, and
no log line anywhere records which one ran. Contrast the row-split path, which
announces its configuration (`gpu/row_split.py:185-187`).

**That is what makes #23 undiagnosable in production, not merely in a harness.** A
campaign that lost itemsets to the silent clamp on a 2-GPU box produces a log
indistinguishable from the 1-GPU run that *would have raised*. Anyone reproducing
it on one card gets a clean run and concludes the data changed. One log line per
site recording the resolved GPU count is near-free, makes #23 diagnosable after
the fact, and is what would have caught this review's own harness error.

(The `min()` clamps are defensive and correct in themselves and are **not** filed.
The finding is the absent observability around them.)

Also worth noting: `tests/test_tier_equivalence.py:163` is named "single-GPU
legacy", but on a multi-GPU box `apriori(..., use_gpu=True)` with `n_gpus=1`
reaches the multi-GPU fused kernels through `get_gpu_count()`. **The test does not
measure what its name says.**

---

## Section 2 negative results — what was checked and found correct

- **Every `except` in `gpu/**` was audited** (20 of them) for whether it can change
  the answer rather than the speed. **Only `row_split.py:745` can** (#26). Verified
  empirically, not by argument, for the three that select a whole alternative
  implementation: `nccl.py:52` (staged D2D reduce instead of ncclReduce —
  `ET_MINER_DISABLE_NCCL=1` byte-identical on 714,841 itemsets); `filter.py:65`
  (conservative routing — `ET_MINER_FILTER_IMPL` compact/cupy/cpu all identical on
  the same 714,841); and the Rust→Python fallbacks at `kernels/k3plus.py:766`,
  `gpu/mining.py:137`, `:213`, `:324` — forcing `get_rust_ext` to return None for a
  whole run gave `prune_equal_support=False: 5,218 vs 5,218` and
  `prune_equal_support=True: 4,510 vs 4,510`, identical.
- **No GPU file re-implements the min-count threshold.** All three entry points
  import the shared helper (`gpu/row_split.py:21`+`:135`, `gpu/mining.py:23`+`:469`,
  `:860`). The GPU set inherits #11's off-by-one but adds no second drift of the
  kind found in `streaming/async_pipeline.py`.
- **The triangular-inverse candidate decode is exact.** `floor(0.5 + sqrt(0.25 + 2p))`
  shared by `_src/_decode_common.cu:28-30` and `kernels/decode.py:90-91` agrees
  with exact integer arithmetic at every row boundary; first float64 failure
  computed at j = 134,217,729 (pair index ≈ 2⁵³), unreachable.
- **The int32 dense-count guard CLAUDE.md asks for exists and is sufficient** —
  `gpu/row_split.py:129-133` and `gpu/mining.py:450-453`; counts are bounded by
  `n_transactions < 2³¹`, and the ≤8-GPU int32 NCCL/staged SUM is bounded by the
  same value.
- **The chunk planner is an exact non-overlapping cover** — 4,000-trial fuzz and
  6,000 fuzzed plans, independently, 0 failures; boundaries land on
  `cumulative_pairs` entries for non-legacy chunks, and `shared_tiled.py:50-59`
  raises rather than mis-serving a misaligned chunk.
- **Row-split partial counts sum to the true global count** at every awkward row
  count and both balance modes — re-run with the shard split asserted rather than
  assumed (`getDeviceCount() == 2`, `len(shards) == n_gpus`, shards on distinct
  devices):

  ```
  1001r/2gpu/rows [501,500]   1001r/2gpu/nnz [407,594]   129r/2gpu [65,64]
    64r/2gpu [32,32]            65r/2gpu [33,32]        4097r/2gpu/nnz [1641,2456]
   777r/1gpu [777]
  K=2 dense, K=3 dense and K=3 sparse-CSR partial sums exact in all 7 cases. FAILURES: none
  ```
- **CSR warp kernels** vs numpy set intersection across empty / 1 / 32 / 33 / 64 /
  65 / duplicate / identical rows: 0/400 failures.
- **Shared/tiled vs legacy vs numpy** across group sizes straddling `TILE_T=32`
  ({2,3,31,32,33,63,64,65} × 6 chunk granularities): 0/150 failures.
- **`_prune_groups_apriori` Rust == Python** on 300 fuzzed levels, neither dropping
  a valid candidate — independently reproducing section 1's result.
- **The two multi-GPU links of CLAUDE.md's tier chain ARE tested against the
  oracle at HEAD**, with the shard count recorded rather than assumed: 5 fresh
  seeds × multi-GPU legacy and multi-GPU sparse CSR, `bitvec shards=[2]`,
  `csr shards=[2]`, 0 disagreements against efficient-apriori. The Defect A 2-GPU
  arm likewise re-ran with `bitvec shards built: [2]` and 714,841 == 714,841.
- **CLAUDE.md CUDA constraints hold**: all 19 `extern "C" __global__` entry points
  in `_src/*.cu` are registered in `loader.py::_KERNEL_FILES` and vice versa; no
  `#include`, `cub::`, `template<`, cooperative groups, `memcpy_async` or
  `extern __shared__` anywhere in `_src/`; every counting kernel is launched with
  `blockDim.x == 256`; `shared_tiled`'s static shared memory is ~17.4 KB, well
  under 48 KB.
- **`__ballot_sync(__activemask(), ...)`** early-exit — the discouraged idiom, used
  in four kernels — was stress-tested under maximal intra-warp divergence
  (`n_u64s = 15,626` with a 10-of-256 tail, block-sparse structure): all five
  routes returned identical exact counts. Value-safe as written.
- **NCCL communicators — re-verified with instrumentation after the first run's
  validity was questioned.** `reduce_sum_to_gpu0` returns immediately at
  `gpu/nccl.py:185-186` when there is one shard, so a run on a single visible
  device would show no growth *while measuring nothing*. Re-run with the
  degradation made impossible to hide — asserting `getDeviceCount() == 2`,
  asserting row-split built 2 shards, and wrapping the collectives to count entries:

  ```
  CUDA_VISIBLE_DEVICES='<unset>'   getDeviceCount()=2
  5 runs: fds 75 -> 75      free VRAM dev0 24,891 -> 24,891 MiB   dev1 24,891 -> 24,891 MiB
  shard counts each reduce saw: [2]      early returns (<=1 shard): 0
  125 ncclReduce collectives executed;   10 communicators created and dropped
  ```

  Every reduce saw two shards and none hit the early return, so the collective
  genuinely ran. **No growth in fds or VRAM**, despite there being no explicit
  `destroy()` call anywhere in the tree. *Stated precisely:* host `maxrss` does
  creep (1,074 → 1,140 MB over five runs, decelerating) — that is Python/allocator
  growth, not fds and not VRAM, and it does not support an NCCL leak, but "no
  growth" without qualification would be wrong.
- **`_prune_groups_apriori` copying `groups=groups_info.groups` unchanged** into
  the pruned result (`gpu/mining.py:321`, `:397`) is unreachable today — both call
  sites use `build_k3plus_groups_from_flat`, which sets `groups=None`
  (`kernels/k3plus.py:763`, `:844`). Latent, LOW, **but sharper than "stale
  metadata"**: `build_k3plus_groups` does set it to the real list
  (`kernels/k3plus.py:590`), and `count_k3plus_fully_fused` reads
  `groups_info.groups` at `:253` and passes `len(groups)` as the kernel's
  `n_groups` at `:289`. Route a tuple-built `groups_info` through the prune and
  `n_groups` becomes the **pre**-prune count while `cumulative_pairs` is
  post-prune, so `_decode_common.cu:11-21`'s `_find_group` binary-searches
  `[0, n_groups-1]` over a shorter array — an out-of-bounds read and wrong group
  assignment, not a cosmetic inconsistency. One-word fix: `groups=None` at both
  sites. Drift note: `count_k3plus_allcounts` already derives the same quantity
  correctly from the arrays (`kernels/k3plus.py:702`,
  `len(groups_info.cumulative_pairs) - 1`) — one quantity, two sources, one
  stale-able. Not hypothetical as a *shape*: the tests already pass
  `build_k3plus_groups` output through the prune
  (`tests/test_group_src_rows.py:145`, `:151`), so the combination is one
  production call site away.

## Section 2 — a documentation defect in CLAUDE.md itself, not in the code

`_src/bitvec_extract_tids.cu:19` uses `__ffsll`, which is **not** on CLAUDE.md's
allowed-intrinsic list. Investigated and **the code is right, the list is stale**:
the loop walks an `unsigned long long word` (`:15-22`), so `__ffs` would truncate
to 32 bits and silently lose every tid in the upper half of each word. `__ffsll`
is sm_20+, compiles under NVRTC with no arch flags, and is already smoke-launched
by `bench/selfcheck.py`.

The list is incomplete in a second and much larger place: **`__popc`** is used in
**10 of the 16 `.cu` files** — including `compact_threshold.cu`, `csr_warp.cu`,
`shared_tiled.cu` and every counting kernel — and is likewise absent, while its
64-bit sibling `__popcll` is listed. A full audit of every `__`-intrinsic in
`_src/` against the list:

```
  __activemask      LISTED        __ffsll   NOT LISTED   bitvec_extract_tids.cu
  __ballot_sync     LISTED        __popc    NOT LISTED   10 of 16 .cu files
  __ffs             LISTED
  __popcll          LISTED
  __shfl_down_sync  LISTED
  __shfl_sync       LISTED
```

(`atomicOr` in `csr_to_bitvec.cu` is 64-bit here so it is covered by "64-bit
atomics", though the phrase does not name it.)

Recommended amendment to CLAUDE.md: `__popcll, __popc, __ffs, __ffsll,
__shfl_down_sync/__shfl_sync, __ballot_sync, __activemask, 64-bit atomics` — and
state the list's *intent* (sm_60+, NVRTC-compilable with no arch flags) so future
additions can be judged rather than pattern-matched against a fixed list.

## Section 2 — explicitly NOT verified

Recorded because a check that could not be run is not a pass.

- **sm_90 *runtime* behaviour.** Compilation for sm_90 is verified (below); nothing
  was ever *executed* on Hopper, because both available devices were RTX 3090
  (sm_86). Correctness and performance on sm_90 hardware remain untested.
- **The production trigger for #23** (>10M survivors in one GPU's slice at one
  level) was not reachable on this hardware. Both reviewers exercised the identical
  code path with a reduced `max_results` instead.

### sm_90 compilation — verified

CLAUDE.md's constraint is a **build** constraint ("sources must build on sm_86 as
well as sm_90"), and that is testable without Hopper hardware. All 15 kernel
sources and all 19 entry points compile for sm_90, confirmed two independent ways:

- `nvcc -arch=compute_90 -ptx`: 15/15 sources, every entry point present in the
  emitted PTX (`compute_86` likewise 15/15).
- `cupy.cuda.compiler.compile_using_nvrtc(arch='90')` — the same NVRTC path
  `cupy.RawKernel` uses at runtime: 15/15 sources, 19/19 entry-point symbols, with
  the architecture **read back out of the cubin ELF header** rather than assumed:
  `cubin ELF e_flags sm = [90]` (and `[86]` for `arch='86'`).

> **A near-miss worth recording, because it is this report's recurring theme
> again.** The first attempt passed `--gpu-architecture=compute_90` as a
> `RawModule` *option*. That silently does nothing: CuPy appends its own `-arch=86`
> afterwards and NVRTC honours the last one. It surfaced only because
> `compute_90a` happened to fail loudly — `nvrtc: warning:
> "--gpu-architecture(-arch)=90" followed by "--gpu-architecture(-arch)=86"`.
> **Anyone re-verifying this must read the architecture back out of the artifact,
> not trust the flag.** A silently-ignored option and a working one produce the
> same green.

---

# Section 3: streaming / SON, IO, config and CLI

Same four-lens panel, unanimous **BLOCK**. Entries below are those confirmed by
the architecture and alternatives reviewers; the numerics and correctness
reviewers were still running when this draft was written and their findings are
folded in above their own headings.

## Summary — Section 3

| # | Severity | Defect | Where |
|---|---|---|---|
| 33 | **HIGH** | **Multi-GPU streaming returns wrong supports on any 2+ GPU box, no fault injection** — every chunk on GPU≥1 dies and the failure is swallowed | `gpu/csr_bitvec.py:371`, `streaming/multi_gpu.py:370-371`,`:527-528` |
| 34 | **HIGH** | A failed chunk in SON pass 1 is skipped while pass 2 still counts it — silent false negatives | `streaming/son.py:257-259` |
| 35 | **HIGH** | SON dedups candidates by a tuple whose item order is *chunk-local*, so one itemset becomes several rows | `streaming/son.py:644`, `:297-298` |
| 36 | **HIGH** | `async_pipeline` compares per-wave popcounts against a whole-dataset threshold — reports **zero itemsets** at its own defaults | `streaming/async_pipeline.py:406-407` |
| 37 | **HIGH** | Local partitioned flush uploads every K level to the same GCS object names — only the last K survives | `io/flush.py:258-260` |
| 38 | **HIGH** | The low-disk fallback leaves an empty `frequent_k{k}/` that shadows the file it just wrote | `io/flush.py:158-159` |
| 39 | **HIGH** | Partitioned flush has no completion marker and never clears the directory — a partial or stale level reads as complete | `io/flush.py:139-248` |
| 40 | MEDIUM | The CLI removes every terminal log sink: `-v`, `-q` and all error messages produce no visible output | `cli.py:277-282` |
| 41 | MEDIUM | `--streaming` materialises the entire input into RAM before streaming it | `cli.py:155-156`, `:180-186` |
| 42 | MEDIUM | Both streaming entry points drop caller parameters in their single-chunk fast path | `streaming/son.py:183-194`, `multi_gpu.py:225-235` |
| 43 | MEDIUM | Every GCS upload failure is swallowed — no caller can learn a level never reached the bucket | `io/gcs.py:250`,`:253`,`:263-264`,`:309-313` |
| 44 | MEDIUM | NVMe staging copies are never deleted, and the disk pre-check sizes only the current level | `io/flush.py:153-159`, `:250-257` |
| 45 | MEDIUM | The low-disk fallback skips both the GCS upload and the local backup | `io/flush.py:194` |
| 46 | MEDIUM | An environment override equal to the default value is silently discarded | `config.py:376-388` |
| 47 | MEDIUM | `local_support_factor` is unvalidated — a value above 1.0 silently breaks SON's superset guarantee | `streaming/son.py:86`, `:198` |
| 48 | MEDIUM | `chunk_size` is unvalidated — a negative value returns an empty frame with no error | `streaming/son.py:197`, `:325-329` |
| 49 | MEDIUM | `memory_budget_gb` sizes chunks from a hardcoded vocabulary estimate, and one of its parameters is dead | `streaming/son.py:53-76`, `:163-169` |
| 50 | MEDIUM | Four config fields are parsed, validated and exposed — and never reach the miner | `config.py:57-89` vs `cli.py:170-193` |
| 51 | MEDIUM | `--min-lift` filters nothing at or below its own documented default | `cli.py:203-204` |
| 52 | MEDIUM | `ET_MINER_GCS_CREDENTIALS` documented as a service-account JSON but the uploader requires authorized_user — and the `KeyError` is swallowed into a **silent identity substitution** | `_env.py:22` vs `io/gcs.py:167-176`,`:249` |
| 53 | MEDIUM | The ramdisk space check ignores that tmpfs is RAM, and generation materialises the whole wave first | `streaming/ramdisk.py:336-373` |
| 54 | LOW | `--min-support 0`, `--max-length 0`, `--chunk-size 0` are silently replaced by config values | `cli.py:170-171`, `:178` |
| 55 | LOW | `setup_ramdisk` dereferences `get_ramdisk_info`'s documented `None` return | `streaming/ramdisk.py:104-110` |
| 56 | LOW | The ramdisk self-test reads a 17-byte header from a 21-byte format | `streaming/ramdisk.py:636` |
| 57 | MEDIUM | `pyarrow_gcs_filesystem` permanently mutates `GOOGLE_APPLICATION_CREDENTIALS` as a side effect of a getter | `io/gcs.py:154-157` |
| 58 | MEDIUM | `ET_FLUSH_THREADS=0` divides by zero; `-1` indexes an empty `bounds` — both after the level is mined | `io/flush.py:141-144`, `:196`, `:229` |
| 59 | LOW | `_env` integer getters reject the literals their own docstring prints (`1e8`, `5e7`) | `_env.py:53`, `:65` |
| 60 | LOW | `delete_wave` reports success while leaving an orphan `list_waves` cannot see | `streaming/ramdisk.py:561-577`, `:596-601` |
| 61 | LOW | `generate_wave_to_disk` emits invalid CSR — duplicate and unsorted indices; inflates every nnz-derived figure | `streaming/ramdisk.py:373` |
| 62 | LOW | `get_ramdisk_info` parses `df` without `-k`, so sizes are environment-dependent | `streaming/ramdisk.py:234-259` |

### The mandated correctness gate does not cover either streaming path

CLAUDE.md's tier-equivalence chain names Tier 1 Polars, Tier 2 Rust, single-GPU
legacy, multi-GPU legacy, shared multi-GPU, single-GPU sparse CSR, multi-GPU
sparse CSR and efficient-apriori. **SON streaming and multi-GPU streaming are not
in it.** That is the structural reason every defect in this section is live
against a green suite — and it is a stronger version of Section 1's finding that
the gate *hides* defect #1. Here the gate does not look at all.

It compounds for #35: `tests/test_tier_equivalence.py:41-44` normalises to a set
of `(tuple(sorted(...)), count)`, which would collapse a duplicate row even on the
paths it does cover.

### `streaming/async_pipeline.py` and `streaming/ramdisk.py` are unreachable code

**Verified with a plain `grep -r` over the whole repository including
`.gitignore`d trees** — the review's own first attempt used a ripgrep-backed tool
that honours `.gitignore` (which excludes `docs/`, `logs/`, `results/`,
`papers/`), and the claim was correctly flagged as unverified until re-run:

```
grep -rnE "^\s*(from|import).*(async_pipeline|ramdisk)" . --include=*.py   -> 0 matches
symbol mentions outside the two files: README.md:25 (prose)
                                       src/et_miner/__init__.py:6 (prose)
                                       tests/test_stream_sync.py:281 (a print string;
                                         the test defines its own suppress_null_sync at :303)
```

Between them these two modules carry **seven** of this section's confirmed
defects — #36, #53, #55, #56, #59, #60, #61, #62 — none of which can currently
harm anyone, because nothing imports them. **Deleting both (~1,500 lines) closes
seven confirmed defects at zero behavioural risk**, removes both drifted
min-count copies, and reduces "three streaming implementations that must agree"
to two. That is a better return than fixing any of them individually.

*It does not bury a GPU defect:* `count_itemsets_fused_k3plus` stays live and
covered independently — `gpu/dispatch.py:90-97` calls it and
`tests/test_fused_k3plus_kernel.py` exercises it directly at eleven sites. If
anything the dependency runs the other way and argues *for* deletion:
`async_pipeline.py:306-307` is the only caller outside `gpu/` that reaches a fused
kernel without going through `gpu/dispatch.py`, so removing it also removes a
bypass of the dispatch layer that Section 2 found to be unlogged (#32).

> **Frame it as an API removal, not a dead-code cleanup.** Both modules ship in
> the wheel, both declare `__all__` (`async_pipeline.py:52-60`,
> `ramdisk.py:42-50`), and both are advertised in prose — `README.md:25` and
> `src/et_miner/__init__.py:6` ("ramdisk helpers … live in their subpackages").
> `from et_miner.streaming.async_pipeline import run_streams_benchmark` works
> today for any installed user. Two conditions on the deletion: **update
> `README.md:25` and `src/et_miner/__init__.py:6`**, or the docs will describe
> modules that no longer exist — the same documentation-drift class this report
> files elsewhere; and **`tests/test_stream_sync.py`** survives untouched (it
> imports nothing from the package) but its docstring and print strings reference
> `async_pipeline.py` and would dangle.
>
> Both modules carry `if __name__ == '__main__'` blocks
> (`ramdisk.py:610`, `async_pipeline.py:862`), which is positive evidence they
> were built for *direct invocation* — so a static import scan is necessary but
> not sufficient. A separate search of `bench/` and `runs/` for invocation forms
> (`python -m …`, `streams_benchmark`, `wave_to_disk`, `/mnt/ramdisk`) returned no
> matches, and `runs/` holds the actual campaign artifacts and job logs. That is
> the stronger evidence.
>
> If the deletion is taken, the defects it removes should be recorded as **"removed
> with the module", not "fixed"**.
>
> **The limit of the evidence, stated rather than glossed:** both modules ship in
> the installed package, so nothing in this tree can establish whether an
> *external* consumer imports them. "No in-tree importer" is verified; "no
> importer" is an assumption. Within the tree the search was exhaustive — six
> files mention either module by any name or symbol (the two modules, this report,
> the two prose lines, and the test) — and `applications/`, 1,364 untracked files
> and the likeliest hiding place for a live consumer, is clean. The scan covered
> `.venv`, so no installed package shadows either name.
>
> **But the assumption is narrower than "no external consumer exists":** these
> modules are *reachable but never exported*. `streaming/__init__.py:6-9`'s
> `__all__` is exactly `["apriori_streaming", "apriori_streaming_multi_gpu"]`;
> the top-level `__all__` (`src/et_miner/__init__.py:81-119`) lists 28 names and
> none comes from either module; `pyproject.toml:7,28` is `0.1.0` / "Development
> Status :: 4 - Beta" with no deprecation policy, no `py.typed` and no API docs;
> and the single console entry point imports neither. So what a reader must accept
> is only: *no external code imports a module this package has never exported, at
> 0.1.0-beta.* If even that is uncomfortable, a four-line `raise ImportError` shim
> pointing at the git SHA, kept for one release, removes the objection without
> blocking the change.

**Deletion checklist — six edit sites, no test changes:**

1. Delete `streaming/async_pipeline.py` (886 lines) and `streaming/ramdisk.py`
   (657) — 1,543 lines.
2. Edit `README.md:25` (advertises "CUDA-streams pipeline, ramdisk").
3. Edit `src/et_miner/__init__.py:6` (advertises "ramdisk helpers, and benchmark
   harnesses").
4. **No `__init__.py` export changes** — `streaming/__init__.py:3-9` never
   referenced either module.
5. **No test changes.** `tests/test_stream_sync.py`'s only `et_miner` import is
   `from et_miner.gpu.csr_build import ...` at `:396`; its `suppress_null_sync` is
   a local `def` at `:303`. It collects 5 tests before and after.
6. Nothing in `bench/` — `bench/selfcheck.py:103` takes `get_popcount_kernel`
   from `et_miner.gpu.kernels`.

*One more instance of this report's recurring theme, found while verifying the
above:* a `grep -vE '^\./(...)'` exclusion written to filter the two modules out of
the results **matched nothing**, because that grep emits paths without a leading
`./`. Harmless — the conclusion was unaffected — but it is an instrument that
silently did nothing, and the output looked identical either way.


**#33, #34 and #35 are the SON algorithm's three correctness defects, and they
compound**: a chunk can be lost, its loss is unreported, and what survives can be
emitted twice.

---

## 33. Multi-GPU streaming returns wrong supports on any multi-GPU machine

**Severity: HIGH.** This is the worst defect in the report: it is on the **default
path**, needs **no fault injection**, and returns **wrong numbers**. Reproduced
independently by the review lead on the first attempt.

```
2x RTX 3090.  300 rows, every row [1,2], chunk_size=100 (3 chunks), min_support=0.3
truth        : {(1,): 1.0,      (2,): 1.0,      (1,2): 1.0}
n_gpus=1     : {(1,): 1.0,      (2,): 1.0,      (1,2): 1.0}      <- control, exact
n_gpus=2     : {(1,): 0.6667,   (2,): 0.6667,   (1,2): 0.6667}   <- 1/3 of the data lost
```

**The mechanism — and it is not what it first looks like.** The helper does not
*fail to inherit* the caller's device; it **explicitly overrides** it.
`gpu/bitvec.py:65` calls `build_bitvecs_gpu_from_scipy(csr)` with **no device
argument**, and `build_bitvecs_gpu` then opens its own
`with cp.cuda.Device(device_id)` (`gpu/csr_bitvec.py:243`) with
`device_id: int = 0`. So the bitvec is built on device 0 *successfully* and
returned to a caller running inside `with cp.cuda.Device(gpu_id)`
(`streaming/multi_gpu.py:276`, `:430`). The counting kernel then launches on
`gpu_id` against a device-0 array.

Two consequences a fixer must know:

- **The actionable line is `gpu/bitvec.py:65`, not the exception handlers.** The
  origin is upstream of both `as_completed` blocks.
- **The `except Exception` → Rust fallback at `gpu/bitvec.py:68-69` does not
  rescue this**, because on the no-raise path nothing is caught.

`n_gpus=1` measures exact precisely because the current device already *is* 0, so
the override is a no-op. **The correct in-tree pattern is one file away**:
`gpu/csr_bitvec.py:512` passes `device_id=gpu_id` explicitly, which is why the
row-split path is unaffected.

In practice it does raise, and every chunk scheduled on a non-zero GPU then dies:

```
ERROR | multi_gpu:371 | Chunk 1 on GPU 1 failed: The device where the array resides (0)
                        is different from the current device (1). Peer access is
                        unavailable between these devices.
ERROR | multi_gpu:528 | Chunk 1 on GPU 1 failed: (same)
INFO  | multi_gpu:541 | Pass 2 complete: 3/3 candidates are globally frequent
```

**And both loops swallow it** — `streaming/multi_gpu.py:370-371` (pass 1) and
`:527-528` (pass 2) catch `Exception`, log, and continue with no re-raise and no
failed-chunk tracking. `:533-548` then computes `min_count_threshold` against the
full `n_total` and returns a normal DataFrame.

**Two consequences, both severe:**

1. **The returned supports are wrong**, not merely incomplete — undercounted by
   the fraction of chunks that landed on GPU≥1. With 8 GPUs, 7/8 of the data is
   silently discarded.
2. **The documented multi-GPU speedup does not exist.** Only GPU 0 ever completes
   work; the other devices fail immediately and their chunks are dropped.

**The single-GPU sibling has no handler at all** (`streaming/son.py:381-452`), so
the same class of failure aborts one path and silently corrupts the other — the
two implementations of one algorithm disagree on their most important error
contract. The only trace is a `logger.error` which, per #40, never reaches the
terminal.

*Scope:* confirmed at `n_gpus=2`. The mechanism — a hardcoded `device_id=0` — 
predicts it for all N, but N≥3 was not tested.

**Fix — two independent changes, and they can land in either order.**

1. **`gpu/bitvec.py:65`**: thread the caller's device through to
   `build_bitvecs_gpu_from_scipy`, matching `gpu/csr_bitvec.py:512`.
2. **`streaming/multi_gpu.py:370-371` and `:527-528`**: count failed chunks and
   raise, because **a support computed over 2/3 of the data is not a support.**

*Recorded because it was investigated and disproved:* fixing the dropped `n_gpus`
at `multi_gpu.py:225-235` (#42) is **not** blocked on fixing the device bug first.
Forwarding `n_gpus` routes to `apriori(n_gpus>1, use_gpu=True)` →
`_apriori_row_split_multi_gpu`, which passes `device_id=gpu_id` explicitly. The
row-split path is correct, so the two fixes are independent.

---

## 34. A failed chunk in SON pass 1 is skipped while pass 2 still counts it

**Severity: HIGH.**

```python
streaming/son.py:251-259   except Exception as e:
                               logger.warning("Chunk {} failed: {}", ...)
                               continue        # past candidate_itemsets.add
```

The `continue` drops that chunk's contribution to `candidate_itemsets`, but the
pass-2 loop at `:381` iterates all `n_chunks` unconditionally and the final
threshold at `:460` uses the full `n_total`.

**SON's entire guarantee is "frequent globally ⇒ frequent in at least one
chunk".** If the chunk that witnessed an itemset is the one that failed, the
itemset never enters `candidate_itemsets` and can never be recovered in pass 2 —
a false negative that no later stage can detect. The realistic trigger is a
MemoryError on a wide 40M-row chunk, which is precisely the workload streaming
exists to serve.

`exceptions.ChunkProcessingError` (`exceptions.py:237-260`) is defined and
exported **for exactly this situation, and is raised nowhere in the tree.** The
same pattern appears at `streaming/multi_gpu.py:291-293` and `:370-371`.

---

## 35. SON dedups candidates by a chunk-local tuple, so one itemset becomes several rows

**Severity: HIGH.**

`_mine_chunk_frequent` returns itemsets as ordered tuples of item ids whose order
derives from **that chunk's own column names**, and `apriori_streaming` uses those
tuples as the cross-chunk dedup key (`streaming/son.py:644` → `:297-298`).

**There are *two* pass-1 producers, which matters for where the fix goes.**
`_mine_chunk_frequent` builds its tuples at `:644`, but `_mine_chunk_gpu_resident`
(`:504-536`, used whenever `gpu_resident=True`) returns tuples straight off the GPU
result frame at `:530` — `[tuple(x) for x in result_df["itemset"].to_list()]` — with
no ordering guarantee, and never touches `:644`. Both funnel into the same merge at
`:283-298`, **so the fix must go at the merge, not at `:644`.**

`build_boolean_matrix` assigns `i_{idx}` positionally over *each chunk's*
surviving frequent items (`core/matrix.py:135-137`), and `son.py:252-256` calls it
per chunk — so the mapping differs per chunk. Combined with #1's string ordering
(`"i_10" < "i_2"`), the same item set arriving from two chunks with different
local vocabularies becomes two distinct tuples.

**Example:** chunk 0's locally frequent items are ids `[1..12]`, so item 3 → `i_2`
and item 11 → `i_10`; chunk 1's are `[3, 11]`, so item 3 → `i_0` and item 11 →
`i_1`. `candidate_itemsets` then holds both `(11, 3)` and `(3, 11)`. Pass 2 builds
a distinct column tuple for each (`:364-366`), counts each separately, and emits
one result row each (`:462-466`) with identical support.

The user gets **duplicate itemsets** in the output frame, `_build_support_lookup`
in `core/rules.py` sees two keys for one set, and pass-2 counting work doubles for
every affected itemset. **The direct path cannot produce this** — it has one
column mapping. This is the cross-chunk incoherence that #1 creates *only here*,
not a restatement of it.

Measured three ways:

```
auditor-str  200 rows of range(11) + 200 rows of [2,10], chunk_size=200:
    SON rows: 67   distinct: 66   duplicated: {(2,10): [(2,10),(10,2)]}
    non-streaming apriori, same data (CONTROL): 66 rows, 0 duplicated

math-str  4000 rows, item 0 locally frequent in chunk 0 only (shifting every column index):
    direct apriori   rows= 91  distinct= 91  DUPLICATED= 0
    SON (2 chunks)   rows=103  distinct= 91  DUPLICATED=12
    k=2 rows: direct 78, SON 90; true frequent pairs C(13,2) = 78

math-str  realistic drifting Zipf, 24k rows, vocab 300, 10 chunks, min_support 0.005:
    19 of 19,019 pairs emitted under >1 ordering in pass 1
    3 of 3,094 distinct itemsets duplicated in the output (3,097 rows)
    CONTROL, identical per-chunk column map (all 120 items frequent everywhere): 0
```

**The control is what makes it a mechanism rather than an anomaly**: with an
identical per-chunk column map the duplication vanishes entirely, exactly as
predicted.

`tests/test_tier_equivalence.py:41-44` normalises to a set of
`(tuple(sorted(...)), count)`, so duplicate output rows are **structurally
invisible** to it — and `apriori_streaming` is not in the mandated chain in the
first place.

**Fix — at the merge point, where both producers funnel:**

```python
# son.py:298          was: candidate_itemsets.add(itemset)
candidate_itemsets.add(tuple(sorted(itemset)))

# multi_gpu.py:313    was: return set(local_frequent), local_items
return {tuple(sorted(t)) for t in local_frequent}, local_items
```

Sorted-in gives sorted-out: `global_counts` is keyed off `candidate_itemsets`
(`son.py:357`), `candidate_cols` / `col_to_itemset` derive from those keys
(`:364-369`), and the result frame is built from them (`:463-466`). Same chain at
`multi_gpu.py:403-421`, `:536-539`.

**The transferable lesson: canonicalise at the funnel, not inside one producer.**
The obvious site (`:644`) is inside `_mine_chunk_frequent` and would have silently
missed `_mine_chunk_gpu_resident` — and a third producer added later would be
missed the same way.

> ⚠ **FIX-ORDERING HAZARD — the duplicate is currently *masking* #2, and the naive
> fix makes rule output strictly worse.** The direction is the opposite of what one
> would guess. A pair emitted only as `(10, 2)` is never found by
> `core/rules.py:89`'s sorted query, so `rhs_support = 0.0` and `:90` sets
> `lift = 0.0`. The duplicate row adds the `(2, 10)` key and the lookup *succeeds*.
>
> Measured on the deterministic SON frame at `max_length=3` — 461 rows, 377
> distinct, 84 duplicate rows:
>
> ```
> as-emitted   ->  2146 rules,  1718 distinct keys
> dedup-first  ->  1674 rules   <- collapse to an arbitrary representative
> dedup-asc    ->  1718 rules
> dedup-desc   ->  1586 rules
> keep-asc vs keep-desc: only-in-asc = 132,  lift-differs = 132
>    e.g. ([0] -> [10,11]): lift = 1.0 with one representative, 0.0 with the other
> ```
>
> Two equally legitimate readings of "keep one arbitrarily" differ by **132 rules
> and 132 lift values**. Confidence never differs; only lift and rule presence.
>
> **So `tuple(sorted(...))` is not the tidy choice — it is the only safe one.**
> Collapsing to an arbitrary representative would silently collapse 132 of 1,718
> rules' lift from 1.0 to 0.0, making the SON fix *strictly worse than leaving the
> bug in place*.
>
> **The word "sorted" is load-bearing and must survive editing.** If this
> recommendation is ever compressed to "canonicalise" or "dedup", it becomes the
> fix that makes rule output worse.
>
> **And sorting dissolves the ordering question entirely** — no sequencing against
> #2 is needed. `core/rules.py:43` keys `_build_support_lookup` on
> `tuple(row["itemset"])` *as emitted*, while `:81` and `:89` query
> `tuple(sorted(...))`. If SON emits sorted tuples the keys are sorted, the queries
> hit, and **#2 is fixed for the SON path by the same one-line change.** There is
> no intermediate state where lift collapses. Nothing else consumes intra-tuple
> order: `Rule` already re-sorts at `rules.py:94`, `_build_result_df` is
> order-agnostic, and #3's positional join is about row alignment rather than tuple
> order.
>
> *Data-dependent, so the report does not over-claim:* on drifting Zipf data at
> `max_length=3` all four dedup variants were identical (4,589 rules, zero
> differences). The effect is real and unbounded in magnitude, not universal — the
> same profile as the duplication rate itself.
>
> *And a check that could not have failed:* the first attempt at this used a
> `max_length=2` frame and came back completely clean. That was **inconclusive, not
> a falsification** — at `max_length=2` every LHS and RHS is a singleton, so
> `rules.py:81`/`:89` only ever query 1-tuples, which are trivially sorted, and the
> defect cannot bite. Same shape as the tier-equivalence gate's `set()`
> normalisation and the oracle deriving its threshold from the expression under
> test.

---

## 36. `async_pipeline` compares per-wave popcounts against a whole-dataset threshold

**Severity: HIGH.** This is the handed-over floor-vs-ceil lead, and the threshold
scaling is the larger half of it.

```python
streaming/async_pipeline.py:406   rows_per_wave = (n_transactions + n_waves - 1) // n_waves
streaming/async_pipeline.py:407   min_count = int(min_support * n_transactions)
```

`min_count` is computed from the **total** row count, then passed unchanged into
`_compute_on_stream` (`:511-517`), where it is compared against `col_counts`
popcounted over a **single wave's** bitvecs (`:294-298`; `bitvecs` has
`ceil(wave.n_rows/64)` u64s). The threshold is therefore `n_waves` × too strict.

**At the module's own documented defaults** — `n_cols=1000`, `avg_items=10`, so
each item appears in ≈1% of rows, `min_support=0.01` — the per-wave count is
≈`0.01·rows_per_wave` while the threshold is `0.01·n_waves·rows_per_wave`.
`frequent_mask` is empty, `total_itemsets` is **0**, and the benchmark's reported
throughput measures data generation only.

Three errors compound at that one line:
1. **Scaling** — global threshold applied per wave (8× stricter at the defaults).
2. **Rounding** — `int()` floors where the canonical `_min_count`
   (`core/result.py:15-17`) ceils. At `min_support=0.0155, n=1000` that is 15
   against 16. This is the **fourth copy** of a threshold CLAUDE.md declares
   mandatory, and the only one that drifted; `son.py:42` and `multi_gpu.py:56`
   import the helper correctly, `async_pipeline.py` imports nothing from
   `core.result`.
3. **Aggregation** — `total_itemsets` sums per-wave counts (`:302`, `:324`,
   `:331`, `:524-527`), so an itemset frequent in all eight waves is counted eight
   times. The returned figure is not a count of distinct frequent itemsets under
   any threshold, and the dict reports `min_support` verbatim beside it.

Identical structure in the sequential baseline at `:658-659` → `:702-704`.

**Measured end-to-end**, 2×RTX 3090 with the device count asserted:

```
N=800,000  n_cols=200  avg_items=10  min_support=0.048  max_length=3
min_count as computed at :407 = 38,400

 n_waves | rows/wave | itemsets reported
       1 |   800,000 |    200
       2 |   400,000 |      0
       4 |   200,000 |      0
       8 |   100,000 |      0        <- the default
```

Same data, same threshold, only the pipelining knob changed. Recomputing the k=1
column counts directly from the same generator: at `n_waves=8` the correct
per-wave `min_count` is 5,000 and there are **7** frequent 1-itemsets; as coded
there are **0**.

The floor-vs-ceil half is separately measurable: **2,194 of 5,000 sampled (s, N)
pairs (43.9%)** disagree with the canonical `_min_count`. A worked case at
`min_support=0.048003, N=100,000`: `int()` gives 4,800 against the canonical
4,801, and the extra survivor is column 182 with count exactly 4,800 — support
0.048000, strictly below the requested 0.048003.

**Nothing in the repository calls these functions** — `streaming/__init__.py:1-9`
does not re-export them and the only invocation is the module's own `__main__` —
but they are in `__all__` at `:52-60`, so they are the module's public surface,
and **any published throughput or itemset figure from them is not comparable to
the miner's.**

---

## 37. Local partitioned flush uploads every K level to the same GCS object names

**Severity: HIGH.**

```python
io/flush.py:250-257   # is_remote branch — CORRECT
    join_gs_uri(output_dir, f"frequent_k{k_level}/{os.path.basename(p)}")
    uploader.upload_to_uri(p, gs_dest)

io/flush.py:258-260   # local branch — no K level anywhere
    uploader.upload(p)
```

`GCSUploader.upload` (`io/gcs.py:290-295`) forwards to `upload_file(str(path),
self.prefix)`, which builds `gs_uri = f"{bucket}/{prefix}/{local.name}"`
(`:237-239`) — **basename only**. The uploader is constructed once for the whole
K-loop with a single prefix (`gpu/row_split.py:206-210`), and every level's parts
are named `part_000.parquet … part_00N.parquet` (`io/flush.py:43`).

So K=2's parts overwrite K=1's, K=3's overwrite K=2's, and the bucket ends up
holding **one level's parts under names that claim to be the whole run.** Locally
the data is intact, so the loss is invisible until someone reads the bucket.

The single-table branch (`:128-129`) is correct, because its filename
`frequent_k{k}.parquet` already carries K. **One rule, two copies, one drifted.**
Fires only above the 100M-row parallel threshold — i.e. exactly the
large-campaign configuration the branch was written for.

---

## 38. The low-disk fallback leaves an empty directory that shadows its own output

**Severity: HIGH.**

`os.makedirs(local_part_dir, exist_ok=True)` at `io/flush.py:159` runs
**unconditionally, before** the disk check at `:161-166`. When the check fails,
the fallback writes the real data to `output_dir/frequent_k{k}.parquet` and
returns at `:194` — leaving `output_dir/frequent_k{k}/` in place and empty.

**Every downstream consumer dispatches on `isdir` and prefers the directory:**

```python
io/gcs.py:108-110    if os.path.isdir(part_dir): return part_dir      # resolve_k_parquet
core/rules.py:122-126  if Path(path).is_dir(): part_files = sorted(Path(path).glob("part_*.parquet"))
```

So rule generation for that K yields **zero rows, silently** (empty glob → empty
iterator → no error), and `resume_from_k` (`gpu/row_split.py:246`) reads an empty
dataset and mines K+1 from nothing. The correct data sits intact two paths away.

The trigger is the "disk filled up at K=7" case the branch exists to handle.

---

## 39. Partitioned flush has no completion marker and never clears the directory

**Severity: HIGH.**

Two independent paths to a corrupt level:

**(a) Interrupted write.** The cleanup at `io/flush.py:235-240` deletes partial
parts only on a *Python-level exception*, and only for `range(len(bounds))` of the
current run. A SIGKILL — OOM killer, vast.ai preemption — leaves whatever was
fsynced.

**(b) Changed part count.** `n_threads = min(_env.flush_threads(), max(1,
(os.cpu_count() or 4) // 4))` (`:141-144`), so the number of `part_00X.parquet`
files a level has depends on `ET_FLUSH_THREADS` **and the machine's core count**.
`os.makedirs(..., exist_ok=True)` at `:159` never clears, so a run producing 2
parts into a directory holding 4 leaves `part_002` and `part_003` from the
previous run in place.

Either way the directory is read whole — `pq.read_table(dir)` at
`gpu/row_split.py:255`, the `part_*.parquet` glob at `core/rules.py:124` — so the
level is silently **truncated** (a) or silently **mixed across two runs at two
different thresholds** (b).

**Nothing records `n`, a part count, or a success flag anywhere.**
`resolve_k_parquet` (`io/gcs.py:93-112`) accepts any directory that exists, and on
`gs://` accepts any prefix pyarrow synthesises into a `FileType.Directory` —
including one left by a single orphaned part. That is the direct answer to what a
resumed run assumes about what the interrupted run wrote: **it assumes
completeness, and has no way to check.**

**Fix:** write parts into `frequent_k{k}.tmp-{pid}/`, `os.replace` the directory
into place, and write a `_SUCCESS` file holding `n` and the part count; make
`resolve_k_parquet` require `_SUCCESS` before preferring a directory over a file.

---

# Section 4: found during PR 6 remediation (2026-09-10)

Numbered N-series because they were found while fixing the numbered defects
rather than in the original audit. N10 and N20 are fixed in PR 6; N21 and N22
are filed, not fixed.

## N20. The single-GPU K>=3 and K=2 bodies launched on the ambient device

`count_k3plus_gpu_resident` and `count_pairs_fused_k2_gpu_resident` allocated
their output buffers and launched on the **ambient** current device while
reading `bitvecs_gpu` from whatever device the caller had put it on.

For K>=3 this was masked by accident: `build_prefix_groups_gpu` ran first and
raised `ValueError: The device where the array resides (1) is different from
the current device (0)` before the body could launch. Making that helper
device-following — correct in itself — removed the raise without moving the
body, and the fault became `CUDA_ERROR_ILLEGAL_ADDRESS`, which poisons the CUDA
context process-wide. A lost campaign, not a lost level, and every gate stayed
green over it.

Both now take the wrapper + `_impl` split and pin to `bitvecs_gpu.device.id`.
Co-residency is asserted separately by `loader.py::_assert_home`, **above** the
routing — below `if n_gpus <= 1: return ...` it never ran on a one-GPU host or
a single-candidate level, which is every call `gpu/dispatch.py` makes on a
single-GPU box.

Co-residency and ambient-pinning are two different properties and neither
implies the other. Fixing only the first is how the K=2 twin shipped: it passed
the guard and still aborted.

**Fixed in PR 6** (`4344be8`, `6e32c5c`). Controls: dropping either pin gives
`cudaErrorIllegalAddress` and a dead context; dropping either guard lets the
mixed-device input reach the kernel.

## N21. Seven test files claiming `n_gpus=2` pass on one GPU

PR 5's `_resolve_gpus` caps the request at the number of devices actually
present, so with `CUDA_VISIBLE_DEVICES=0` all 121 tests across the seven files
that pass `n_gpus=2` still pass — having exercised the single-GPU path.
`test_multi_gpu_legacy_matches_oracle` and its two siblings are in that set.

Only `tests/test_gpu_device_affinity.py` carries the `multigpu` marker, applied
per test. The other seven degrade silently.

**Filed, not fixed.** The fix is either the marker on every test that means it,
or an assertion that the route taken was the route named.

## N22. `output_dir=""` writes outside the caller's control and returns empty

Reachable from the documented public API: `core/apriori.py`'s route check gates
on `output_dir is not None`, so `""` passes validation and is forwarded, and
nothing downstream checks for emptiness.

Every `output_dir` guard in `gpu/row_split.py` tests truthiness except the one
that decides flush-vs-defer, which tests `is None` — and that is the one
matching the API contract. `""` is the single value that splits them: the flush
branch is taken, then the deferred branch's early return fires, and the caller
gets an empty DataFrame.

The destination depends on level size, because the two flush branches build
paths with different functions:

| branch | builder | `output_dir=""` gives |
|---|---|---|
| single-table (below `ET_MINER_FLUSH_PARALLEL_THRESHOLD`) | `io/gcs.py::join_gs_uri` | `/frequent_kK.parquet` — **filesystem root** |
| parallel (at or above it) | `os.path.join` | `frequent_kK/` — process CWD |
| large level, low-disk fallback | `join_gs_uri` | root again |

`join_gs_uri` does `base.rstrip("/")` then `f"{base}/{suffix}"`, so an empty
base synthesises a leading slash. Measured: an ordinary call writes
`/frequent_k1.parquet` … `/frequent_k3.parquet` and returns height 0 with CWD
untouched. The threshold defaults to 100,000,000 rows, so the CWD branch needs
either the env knob or a single K level in the nine-figure regime — i.e.
unreachable where it would be caught early, reachable where it would hurt most.

**Filed, not fixed.** Fix upstream rather than at the six guards: reject a
falsy non-None `output_dir` at validation, and all of them agree without any
being edited.
