# Moonshots for the GPU route: final report

The campaign built five ideas and one polish. Each was measured against
today's path and kept by its decision rule. Together they make the GPU
route 7–16× faster on its heavy regimes, with every result unchanged:
same itemsets, same counts, same signatures in every regime.

- **Box.** 2× RTX 3060 12 GB, EPYC 7282, thread pools at 16. Every number
  below is from this box (Phase 0's box, an A4000 with a Pentium, is not
  compared). Medians of 3.
- **Branches.** Each idea stacks on the previous one, on the owner's choice:
  `perf/moonshot-baseline` → `-m1` → `-m2` → `-x0` → `-m7` → `-m5b` →
  `-polish`. None is merged, and there is no PR.

## What the stack does, end to end (Phase 3, `../results/2026-10-18-moonshots-phase3/`)

`main`'s `src/` against the final tree, the same harness and pins, every knob
at its default:

| regime | GPUs | `main` [min, max] | final [min, max] | final / main |
|---|---|---|---|---|
| dsl | 1 | 29.39 [29.34, 30.21] | 3.86 [3.85, 3.90] | 0.131 |
| dsl, count inference | 1 | 25.07 [24.75, 25.75] | 2.40 [2.35, 2.50] | 0.096 |
| dsl, free sets | 1 | 24.31 [24.00, 25.28] | 2.76 [2.73, 2.77] | 0.114 |
| sk2ml3 | 1 | 27.68 [27.55, 27.76] | 1.77 [1.74, 1.78] | 0.064 |
| oom2ml3 | 1 | 2.70 [2.66, 2.70] | 0.38 [0.37, 0.39] | 0.139 |
| skew / deepk | 1 | 1.20 / 0.87 | 0.17 / 0.15 | 0.14 / 0.18 |
| or003 / or002 / smoke | 1 | 0.20 / 0.36 / 0.08 | 0.11 / 0.27 / 0.06 | 0.53–0.75 |
| dsl / dsl-infer / dsl-free | 2 | 26.88 / 24.12 / 23.61 | 3.23 / 2.72 / 2.75 | 0.11–0.12 |
| sk2ml3 / oom2ml3 | 2 | 20.36 / 3.13 | 2.33 / 1.12 | 0.11 / 0.36 |

No regime is slower; must-fix: none. The baseline run used 0.405 GPU-hours and
the final run 0.178.

## Per idea

| idea | decision | where it wins (rule 2) | report | GPU-hours |
|---|---|---|---|---|
| M1 device input layer | **kept**, the dispatch for integer ids spanning ≤ 2²² | 11 of 20 regimes: dsl ×6, sk2ml3 ×2, oom2ml3, skew | `REPORT-M1.md` | 0.725 |
| M2 row-wise K=3 | **kept** behind the r3 dispatch (`K3_ROWS_MAX_R = 2.5e-4`, swept) | sk2ml3 and oom2ml3 on 1 and 2 GPUs | `REPORT-M2.md` | 0.649 |
| M10 item order | **not built** | its stake (dense K=3 generation) went with M2 | `PROTOCOL-M2.md` | — |
| X0 count inference by default | **table only; the owner decides** | dsl on 1 and 2 GPUs; ties elsewhere | `REPORT-X0.md` | 0.271 |
| M7 deduction bounds | **kept**, under count inference | dsl-infer on 1 and 2 GPUs | `REPORT-M7.md` | 0.301 |
| M5b lex row order + tile skip | **kept** for the device input layer | dsl on 1 and 2 GPUs, dsl free sets on 1 | `REPORT-M5b.md` | 0.442 |
| polish (span check, argsort, lazy occupancy) | **kept** | dsl on 1 GPU; every regime lower or tied | `REPORT-POLISH.md` | 0.178 |
| M6, M9, M3 | **not built** | stakes after M2 ≤ 0.70, 0.37, 0.12 s | `SPEC.md` amendments | — |
| Phase 3 reduced matrix | — | — | `PROTOCOL-PHASE3.md` | 0.583 |

M4, M5a, M8, M11 and M12 were dropped in Phase 1 or are outside the approved
order. `deep_k20` joined `PRESETS` for a future M4. The ideas' own budget
(≤ 1 GPU-hour each) held for all of them.

Exactness carried every step:
- each new path has pinned tier legs on 1 and 2 GPUs; the gate grew from 66
  to 91 legs, all passing;
- each has property tests against brute force or efficient-apriori, on the
  brief's edge cases;
- the device input layer's bitvectors are byte-identical to the host path's
  on every preset.

## Findings

1. **(should)** *X0: the owner's decision.* `use_generator_pruning=True` takes
   dsl from 3.86 to 2.40 s on the final tree (one GPU) and costs nothing
   measurable elsewhere (`REPORT-X0.md`). The output is identical by
   construction. The default stays `False` until the owner decides.
2. **(should)** *Two GPUs rarely pay on this box after the stack.* dsl is 3.23 s
   on two GPUs against 3.86 on one, but sk2ml3 is 2.33 against 1.77, and every
   small regime is slower on two. Fixed costs dominate the short walls:
   - NCCL init;
   - the per-level reduce;
   - a bimodal teardown tail of about 0.07 or 0.5 s (X0 finding 2,
     unexplained).

   Naming the tail (span phases for the NCCL destroy, the uploader drain, the
   pools) comes first. A one-GPU dispatch for small work would change
   `n_gpus`' meaning, which is the owner's call.
3. **(should)** *CuPy's pool fragments across calls in one process*
   (`REPORT-M2.md` finding 3, predates the campaign). A second
   `apriori(stress_k2, max_length=2)` on one 12 GB GPU raises
   OutOfMemoryError on either input layer. Freeing the pool's cached blocks
   at the start of a run would fix it.
4. **(should)** *M2's dispatch leaves wins on the table.* r3 alone does not
   order the swept points. Adding the increments estimate gives
   (R2 + ρ² T3) / (slots × words); ρ² T3 predicts sk2ml3's and oom2ml3's
   increments to 10 %. It would admit mid-range wins (`REPORT-M2.md`
   finding 1). It needs its own sweep.
5. **(should)** *M7 could stop where it does not settle.* or002 pays 0.08 s on
   one GPU for bounds that settle nothing there (`REPORT-M7.md` finding 1).
   A per-level switch needs the kernels to report what they settled.
6. **(should)** *String item ids cannot pass the GPU route* (by code reading,
   unverified). `row_split.py` copies `col_to_item` into an int64 array.
   That is outside the brief's scope and is left as a follow-up.
7. **(nice)** M5b's permuted `ids_to_bitvec` costs 0.15 s on dsl against
   0.02 s unpermuted, from atomics on scattered words. Scattering by target
   position does not help (`REPORT-M5b.md`).
8. **(nice)** Process: M2's crossover sweep ran before `PROTOCOL-M2.md` existed.
   Its procedure was in `k3_crossover.py`'s docstring, committed before it ran
   (`REPORT-M2.md` finding 5).
9. **(nice)** `uv run ty check` is not available in this project's
   environment. Lint (`ruff`) and the tests are the static gates used.

## Gates on the final tree (`perf/moonshot-polish`)

| check | result |
|---|---|
| `uv run pytest tests/test_tier_equivalence.py -q` | 91 passed |
| `uv run pytest -q -m "not slow" -k "not test_support_00"` | 1586 passed, 1 failed: `test_every_bitvec_wrapper_runs_the_guard` (`tile_occupancy`). After the guard fix that test and `tests/test_row_order.py` pass (38 passed); the whole suite was not rerun. |
| `uv run pytest -q -m "gpu and slow" -k "not test_support_00"` | 17 passed |
| `uv run python bench/selfcheck.py` | READY: 16 kernels on 2 devices |
| `uv run ruff check src tests bench` | clean |

`-k "not test_support_00"` deselects the Online Retail support tests, as in
every earlier gate. They do not finish with Online Retail present.

The `tile_occupancy` guard: `tests/test_kernel_guard_claims.py` requires every
wrapper that takes `bitvecs_gpu` to run `_assert_bitvecs`. M5b's
`tile_occupancy` did not. The fix sits on `perf/moonshot-polish`, so
`perf/moonshot-m5b` alone still fails that test.
