# Behaviour-change impact assessment (PR 0a)

**Purpose.** Six fixes in the remediation programme change mined output: **#11**
(min-count threshold), **#12** (self-sufficiency aggregate), **#13** (batch
length filter), **#22** (`anchor_items` semantics), **#24** (result truncation
now raises), **#51** (`--min-lift`). The repository has been public since August
and campaign artifacts exist on this box, so before any of those land we must
know which already-produced numbers move.

**Method.** Every statement below is measured on this box against the artifacts
actually present, not inferred. The reusable tool is
`bench/baseline/min_count_impact.py`.

---

## What is on this box

| Path | What it is |
|---|---|
| `runs/20260902T0000Z/phase2/extract_2026_01/transactions_214m_base.parquet` | base214m **input**, 90,506,840 rows, schema `protein_id: str, items: list[i64]` |
| `runs/20260902T0000Z/phase2/extract_2026_01/transactions_214m_base_multi.parquet` | the same input after `load_transactions(min_items=2)`, **76,890,945 rows** |
| `runs/20260902T0000Z/phase2/scripts/__pycache__/son_run.cpython-310.pyc` | the mining driver; **source deleted**, bytecode survives |
| `applications/` | build artifacts only (`af-extract/target/`, `__pycache__/`) — no source that calls the miner |

**No mined lattice is stored in this tree.** `runs/` holds inputs and extraction
intermediates. Nothing here needs re-mining; the exposure is to figures already
published elsewhere.

### What the driver did, recovered from bytecode

`son_run.py`'s docstring and constants give the run shape exactly:

- `et_miner.apriori_streaming` — **SON streaming, single GPU**
- `chunk_size = 40_000_000`, `local_support_factor = 0.9`, GPU on
- `min_support` was a **required CLI argument** (`--min-support`), so the value
  used is *not* recoverable from this tree. The docstring's own example is
  `0.001`.
- It reports: itemset count, K-distribution, max K, wall-clock, and optionally a
  rule count at a confidence threshold. It does **not** call
  `compute_self_sufficiency` and does **not** use `anchor_items` or
  `mine_two_phase`.

At 76,890,945 rows and `chunk_size = 40,000,000` the run is **exactly two
chunks** — which is the precondition for #35.

---

## Verdict per fix

| Fix | base214m | Basis |
|---|---|---|
| **#11** min-count | **unaffected** | 0 shifts / 189,999 thresholds at N = 76,890,945 |
| **#12** self-sufficiency | **not used** | driver never calls `compute_self_sufficiency` |
| **#13** length filter | **unaffected** | SON pass 2 already passes `enable_length_filter=False`; pass 1 mines uniform-length levels |
| **#22** anchors | **not used** | no `anchor_items`, no `mine_two_phase` |
| **#24** truncation | **unaffected** | SON pass-2 GPU counting goes through `count_support_gpu_bitvec`, which allocates no `max_results` buffer; `_warn_result_truncation` is confined to `kernels/{k2,k3plus,gpu_resident}.py` |
| **#51** `--min-lift` | **not used** | library call, not the CLI |
| **#35** SON duplicates | **unaffected — verified mechanically** | see below |

### #11 — exhaustive, not sampled

Every `min_support` with ≤5 significant digits in `[1e-6, 1)` was evaluated under
`math.ceil(s*N)` against `math.ceil(Fraction(str(s))*N)`:

```
$ uv run python bench/baseline/min_count_impact.py --exhaustive --n-rows <N>

base214m (post min_items=2)    N= 76,890,945  tested=189,999  SHIFTS=0
base214m raw                   N= 90,506,840  tested=189,999  SHIFTS=3      (0.275, 0.55, 0.675)
smoke preset                   N=     60,000  tested=189,999  SHIFTS=414    (0.00205, 0.00395, 0.0041)
deep_k / skewed_rows           N=  1,000,000  tested=189,999  SHIFTS=2,457  (0.000123, 0.000246, 0.000253)
stress_k2                      N=  2,000,000  tested=189,999  SHIFTS=2,457  (0.000123, 0.000246, 0.000253)
oom_regression                 N=    500,000  tested=189,999  SHIFTS=1,733  (0.000246, 0.000492, 0.000506)
```

**The zero at N = 76,890,945 is a property of that row count, not a general
safety** — the same sweep finds 2,457 shifting thresholds at N = 1,000,000.
Any *other* campaign must be re-checked with the tool rather than assumed safe
by analogy.

The three shifting values at the raw row count (0.275, 0.55, 0.675) are far above
any plausible mining threshold and the raw file is not what was mined.

### #35 — the two chunks diverge, and it still cannot fire

The run's two chunks build their own `i_N` column maps over their own locally
frequent items (`core/matrix.py:135-137`), so #35 fires when a shared item's
**column index** differs between chunks. Measured at three plausible thresholds
(local threshold = `min_support × 0.9`):

```
min_support  |V0|   |V1|   identical   only-in-c0   only-in-c1
0.0005       1002   1002   yes         -            -
0.001         751    751   no          [255]        [254]
0.002         317    319   no          []           [739, 740]
```

The vocabularies differ at 0.001 and 0.002 — but **no shared item's column index
moves**, and the reason is arithmetic rather than luck:

- at **0.001**, items 255 and 254 each have exactly **250 shared items below
  them**, so the two divergent items occupy the *same rank*; every shared item
  keeps its index in both chunks.
- at **0.002**, chunk 1's extras (739, 740) sort **above all 317** shared items,
  so they only append columns at the end.

Cross-checked a second way: of the 79,800 / 50,086 shared-item pairs tested, **0**
flip their unpadded-string order between the chunks. With no order flip, one
itemset cannot become two rows.

**Conclusion: the base214m itemset count is not inflated by #35** at any of the
three thresholds tested. It is *not* cleared for an untested threshold — rerun
the vocabulary comparison if the real value turns out to be something else.

---

## Disposition

- **No re-mining is required for anything stored in this tree.**
- **No figure derived from the base214m run moves** under any of the six
  output-changing fixes, on the evidence above.
- **Open, and the user must close it:** the exact `--min-support` used for the
  published base214m figures. Everything above tests 0.0005 / 0.001 / 0.002. If
  the real value is outside that set, re-run:
  `uv run python bench/baseline/min_count_impact.py --n-rows 76890945 --min-support <S>`
  and the vocabulary comparison in this document's method section.
- **Anything mined outside this tree** — the V2 preprint tables, pattern.poker
  campaigns — is not covered here and must be checked against the tool, because
  the #11 clearance is row-count-specific.
