# Session Handoff: ET-Miner, 2026-10-18 13:36

**Branch:** `perf/moonshot-polish` @ `6066744` (in sync with origin)
**Tree:** clean
**Focus:** GPU-route moonshots campaign finished and merged (PR #35, per the owner); next: work through the open optimization items
**Reason:** end of session / task switch

## Verify on Resume
Run `git status --short --branch` and `git rev-parse --short HEAD`. HEAD should be `6066744` on `perf/moonshot-polish`, tree clean. The owner says PR #35 is merged into `main`, so start from an up-to-date `main` (`git fetch && git checkout main && git pull`), not from this branch.

## Current State
- PR #35 (`perf/moonshot-polish` → `main`) merged by the owner. It holds the whole stack: baseline harness, M1, M2, X0 table, M7, M5b, polish, Phase 3, the adversarial-review fixes and the council fixes.
- The last work was the council's three open should-items (`6066744`); gates green, GitHub CI green.
- Nothing is in flight. No timed run is running.
- The open optimization items are listed in `bench/moonshots/REPORT.md`: § Findings 1–11 and § Council review items 7–8. The ones worth doing are listed under Next Steps below.

## Next Steps (open optimization items; pick with the owner, one branch per idea off `main`)
1. (should) **CuPy pool fragmentation across calls in one process** (REPORT Findings 3, predates the campaign): a second `apriori(stress_k2, max_length=2)` on one 12 GB GPU raises OutOfMemoryError. Fix: free the pool's cached blocks at run start. Cheap, no timed run needed beyond a check.
2. (should) **Name the 2-GPU teardown tail** (Findings 2): it is bimodal, ~0.07 or ~0.5 s. Add span phases for NCCL destroy, the uploader drain and the pools first. A 1-GPU dispatch for small work would change what `n_gpus` means, so that part is the owner's call.
3. (should) **M2 dispatch statistic** (Findings 4): replace r3 by (R2 + ρ²·T3) / (slots × words); ρ²T3 predicted sk2ml3's and oom2ml3's increments within 10 %. Needs its own sweep (`bench/moonshots/k3_crossover.py`) and a PROTOCOL first.
4. (should) **M7 per-level switch** (Findings 5): or002 pays +0.08 s on 1 GPU for bounds that settle nothing. The kernels would have to report how many candidates they settled.
5. (nice, council) **Deep params cost registers**: every K>=3 counting kernel now has 1408 B/thread of local memory, and the tiled kernels went from 46 to 66 registers even with the bounds off (sm_86 tiled occupancy 5 → 3 blocks/SM by arithmetic). The speed effect is unmeasured; measure it on a tiled-heavy regime.
6. (nice, council) **Lex row order is unconditional** (`core/apriori.py`, `row_order=_env.row_order() or "lex"`). SPEC § M5 (b) planned a dispatch on the zero-tile share, and the evidence comes from dsl only. Measure on real data (Online Retail, larger row counts) before keeping lex as the default.
7. (nice) **M5b permuted `ids_to_bitvec`** costs 0.15 s on dsl against 0.02 s unpermuted (atomics on scattered words) (Findings 7).
8. (nice, council) **Host fallback builder**: point the host CSR path (`core/apriori.py` host branch → `matrix._build_csr_from_transactions`, explode + join + COO, 20.7 s input on dsl) at `core/cpu_miner.build_transaction_csr` (bincount + LUT). That builder filters columns by row count, not occurrences, so `col_to_item` can differ; adjust the byte-identity tests in `tests/test_input_layer.py`.
9. (nice, council) **Unserved column read in memory**: a column with span > 2²² is collected and its chunks read before the span check, so the host build no longer streams it (~8 B/entry extra peak RAM). Check the span first, or fall back to the original LazyFrame.
10. (nice) `KEEP_ROWS_BYTES = 28` is about 2× conservative (measured 13.4 B/entry canonical), so rows could be kept more often. Lex holds `new_pos` (4 B/row) during the build (Findings 11).
11. Small code cleanups (council): `core/apriori.py` repeats ~18 kwargs across the device and host miner calls; `build_device_input(row_order="keep")` defaults differ from the route's "lex"; the occupancy peak is ~98 MiB per block, not the 64 MiB that `OCC_BLOCK_BYTES` documents.

## Session Instructions
- Campaign brief rules stay binding for any follow-up idea:
  - efficient-apriori is the oracle;
  - every new path gets pinned tier legs on 1 and 2 GPUs (forced chunks, count inference, free sets where served);
  - knobs are `ET_MINER_*` pins only, documented in `_env.py`; no new public params, no default change, no new runtime dependency without the owner's word;
  - `PROTOCOL-<id>.md` committed before an idea's first timed run;
  - ≥ 3 reps rep-major, fresh process per config, never compare across boxes, no tracked-file edits or commits during a timed run;
  - one branch per idea; commit and push after each verified step; no PR or merge without the owner's word.
- GPU budget ≤ 1 GPU-hour per idea (owner, 2026-10-17); stop and ask when exceeded.
- Chat in Dutch, concise, plain language. Files, code and commits in English.
- Reports are numbered lists tagged (must-fix)/(should)/(nice), sorted in that order.
- Owner chose: stacked branches, `deep_k20` in PRESETS, lattice dumps kept.

## Decisions & Rationale
- **X0 (`use_generator_pruning` default) not flipped**: owner's decision, still unanswered. On the final tree it takes dsl from 3.86 to 2.40 s, costs nothing measurable elsewhere, and leaves the output identical.
- **M6, M9, M3, M10 dropped**: their stakes after M2 are below 1 s (SPEC amendments). M4, M5a, M8, M11 and M12 were dropped in Phase 1.
- **Rows give way out of memory, not a second estimate** (council): `DeviceRows.pairs/upload` and `_HostRows.upload` return None on OutOfMemoryError and release the rows; K=2 and K=3 then count from the bitvectors.
- **`_subset_index` degrades per layer** (council must-fix): when the K−1 index fits but the older levels do not, only SUBSET_BOUNDS goes. The whole index goes only when the K−1 index alone does not fit, as on `main`.
- **No timed run for the review/council fixes**: a spy showed the regimes run the same work (no position reuse, dsl occupancy in one block, every level keeps its bounds on dsl-infer).

## Dead Ends
- Do not retry the M2 triangle-list layout with binary search: 1.1 s; the slot-by-pair layout with bitmap ranks replaced it (kernel 1.56 → 0.39 s) (from compaction summary).
- Do not retry scattering M5b's bitvectors by target position: it does not help (`REPORT-M5b.md`).
- Do not import cupy at module or install time in `bench/` or tests: CI is CPU-only (`.github/workflows`, `uv run pytest -q -m "not slow"`). `bench/level_split.py` broke CI this way (fixed in `fa3f021`).
- Do not `pkill` by a pattern that matches your own shell (from compaction summary).

## Key Findings
- CI has no CuPy. To reproduce locally, run pytest with `sys.modules["cupy"] = None` and `sys.modules["cupyx"] = None` set before `pytest.main` (`-m "not slow"`): 894 passed, 557 skipped at `6066744`.
- The `test_support_00*` tests (Online Retail) do not finish with the dataset present; every local gate uses `-k "not test_support_00"`.
- CuPy's pinned pool must be freed in tests (`_free_pools` in `tests/test_input_layer.py`), or it locks host RAM (from compaction summary).
- In `tile_occupancy` and `DeviceRows`, release memory only after the `except` block exits: the traceback keeps the failed allocation's arrays alive (`input_layer.py` `DeviceRows._out_of_memory`).
- `min_support=0.0`: the GPU route emits count-0 itemsets of real items (predates the campaign, Findings 10). The device layer no longer invents absent ids (`input_layer.py`, `freq = ... & (occurrences > 0)`).

## Blockers & Pending Decisions
- Owner: flip the `use_generator_pruning` default (X0)?
- Owner: a 1-GPU dispatch for small work on `n_gpus=2` (changes what `n_gpus` means)?
- Owner: which open items to take first (Next Steps).

## Test Status
At `6066744` (2026-10-18 ~13:30):
- `uv run pytest tests/test_tier_equivalence.py -q`: 91 passed.
- `uv run pytest -q -m "not slow" -k "not test_support_00"`: 1615 passed.
- No-CuPy run (as CI): 894 passed, 557 skipped.
- `uv run pytest -q -m "gpu and slow" -k "not test_support_00"`: 17 passed.
- `bench/selfcheck.py`: READY (16 kernels, 2 devices).
- `ruff check src tests bench`: clean. GitHub CI: pass.

## What Was Done (this session)
- Adversarial review: no wrong result; memory findings fixed (`88e2532`):
  - blocked `_canonicalize` (`CANON_BLOCK_ENTRIES`) and `_keep_bytes` in `gpu/input_layer.py`;
  - occupancy temporaries capped by `OCC_BLOCK_BYTES`, with an OOM fallback (`gpu/kernels/occupancy.py`, `row_split._occupancy`);
  - rows released before dense levels; memory guard drops the rows;
  - device layer keeps rows when row-wise K=3 may run under a dense K=2 pin.
- PR #35 opened. The council (4 reviewers, 3 rounds) found one must-fix, fixed in `c1f8b38`: `row_split._subset_index` degrades per layer, plus the `min_support=0` column rule and the `_env.py` K3 doc.
- CI fix: `bench/level_split.py` imports cupy lazily (`fa3f021`).
- Council should-items (`6066744`):
  - out-of-memory contract for both row sources;
  - `_drop_rows` as the single release point (`_k2_src` renamed `_rows_src`);
  - `_keep_levels` and the guard dropping `_complete_levels` (`gpu/row_split.py`).
- Reports: `bench/moonshots/REPORT.md` (§ Adversarial review, § Council review of PR #35, Gates), `SPEC.md` M1 amendment, `CHANGELOG.md`.

## Commits This Session
- `6066744` Council open items: rows give way out of memory; one owner; levels kept for the bounds only
- `7f18830` Moonshots report: the subset-index window as index_nbytes gives it
- `fa3f021` bench: level split imports cupy only when the GPU bitvec build runs
- `c1f8b38` Council review: keep the subset test when only the bounds do not fit
- `88e2532` Moonshots: fix the adversarial review's memory findings
- `f103fdc` Moonshots report: gate counts as run (from compaction summary)
- `390039d` Moonshots: final report; tile_occupancy runs the bitvec guard (from compaction summary)
- `ba158d9` Phase 3: reduced matrix, baseline src against the final tree (from compaction summary)

## Key Files
| File | Role |
|------|------|
| `src/et_miner/gpu/input_layer.py` | M1 device input layer: `build_device_input`, `DeviceRows`, `_canonicalize`, `_keep_bytes`, `_row_order` |
| `src/et_miner/gpu/row_split.py` | Row-split miner: `_rows_src`/`_drop_rows`, `_subset_index`, `_count_k2_rows`, `_count_k3_rows`, `_occupancy`, `_check_memory_guard` |
| `src/et_miner/gpu/row_split_chunks.py` | Dispatches `k2_counts_rows` / `k3_counts_rows` (`K2_ROWS_MAX_R`, `K3_ROWS_MAX_R`), chunk budget |
| `src/et_miner/gpu/kernels/{k3_rows,occupancy,subset_index}.py`, `_src/*.cu` | M2, M5b and M7 kernels and wrappers |
| `src/et_miner/core/apriori.py` | Device/host input-layer dispatch before the miner call |
| `src/et_miner/_env.py` | Every `ET_MINER_*` pin, documented |
| `bench/consolidation_matrix.py`, `bench/runner.py` | Moonshot matrices (`--mode moonshots-*`), `MOONSHOT_REGIMES`, `WORKLOADS` |
| `bench/level_split.py`, `bench/moonshots/{decide,split_table,k3_crossover}.py` | Level split, decision rule 2, split tables, M2 sweep |
| `bench/moonshots/REPORT.md`, `SPEC.md`, `PROTOCOL-*.md` | Final report with every open item, design and amendments, protocols |

## Extra Context
- Focus for the next session: **open optimization punten afwerken**, the items in Next Steps. Confirm the order with the owner first, then one branch per idea off `main`, with a PROTOCOL before any timed run.
- Box: 2× RTX 3060 12 GB, EPYC 7282; thread pools at 16 (`MOONSHOT_THREADS`); `NCCL_P2P_DISABLE=1`. Phase 0 ran on an A4000/Pentium box and is never compared.
- The Phase 3 numbers on this box (`main` src → final, 1 GPU): dsl 29.39 → 3.86 s, dsl-infer 25.07 → 2.40 s, sk2ml3 27.68 → 1.77 s, oom2ml3 2.70 → 0.38 s. These are the reference for any follow-up idea.
