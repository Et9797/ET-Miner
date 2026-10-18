# Phase 3 polish: report

**Decision: the polish stays.** It loses no regime and wins dsl on one GPU
by rule 2 (4.92 → 3.88 s). Every other regime ties, mostly lower. The
signatures match M5b's in every regime.

Protocol: `PROTOCOL-POLISH.md`.
- Data: `../results/2026-10-18-moonshots-polish/`:
  - `raw.jsonl`: the after-run, at `0efcaa9`;
  - `before.jsonl`: M5b's `lex` arm, at `797b54c`;
  - `compare.jsonl` and `decision.md`: the two together.

## Results (median of 3, seconds; same box, the two runs hours apart)

| regime | GPUs | before [min, max] | after [min, max] | input before → after | after/before |
|---|---|---|---|---|---|
| dsl | 1 | 4.92 [4.90, 4.93] | 3.88 [3.88, 3.90] | 2.33 → 1.38 | 0.789 (**wins**) |
| dsl, count inference | 1 | 3.46 [3.41, 3.47] | 2.48 [2.46, 2.50] | 2.34 → 1.37 | 0.716 |
| dsl, free sets | 1 | 3.76 [3.75, 3.79] | 2.79 [2.75, 2.83] | 2.34 → 1.38 | 0.741 |
| dsl | 2 | 4.11 [4.09, 4.11] | 3.26 [3.24, 3.27] | 2.07 → 1.22 | 0.795 |
| dsl, count inference / free sets | 2 | 3.59 / 3.60 | 2.73 / 2.75 | 2.07 → 1.21 | 0.76 |
| sk2ml3 | 1 / 2 | 2.00 / 2.41 | 1.78 / 1.82 | 0.37 → 0.28 | 0.89 / 0.75 |
| deepk, skew, oom2ml3 | 1 / 2 | | | −0.03 to −0.04 | 0.78–0.97 |
| smoke, or003, or002 | 1 / 2 | ≤ 0.59 | ≤ 0.58 | | 0.98–1.16 (≤ 0.01 s) |

- dsl's input falls by 0.95 s: the span check (about 0.75 s) and the order's
  sort (about 0.2 s).
- sk2ml3 sheds the occupancy it never read. Its one-GPU wall is back at M5b's
  `keep` arm (1.78 s).
- The dsl rows tie at 0.72–0.80× because they save under 1 s each. Only dsl
  on one GPU crosses rule 2's 1 s.

Budget used: 0.178 GPU-hours.
