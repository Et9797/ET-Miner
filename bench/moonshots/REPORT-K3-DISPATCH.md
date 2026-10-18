# K=3 dispatch statistic: report

**Decision: the row-wise K=3 dispatch reads s = (R2 + ρ² T3) / (slots × words)
below 1.0e-2.** It replaces r3 = R2 / (generated × words) below 2.5e-4.

On the sweep's 42 points, the K=3 time lost against the faster kernel falls
from 16.2 s to 5 ms. On the held-out points it is 0 s. No regime of the matrix
changes kernel, and every result is unchanged.

The gain lies outside the matrix: mid-range data, where the old rule sent
levels to the slower kernel.
- **Rows with random pair structure:** up to 6.4 s per level (8.6 → 2.2 s).
- **Short rows over 4M transactions:** 0.2–2.1 s.

Protocol: `PROTOCOL-K3-DISPATCH.md` with Amendment 1. Data:
`../results/2026-10-18-k3-dispatch/` (sweep, held-out points) and
`../results/2026-10-18-k3-dispatch-final/` (final check). Branch
`perf/m2-dispatch-stat` off `main` at `37819e2`.

## What changed

- `row_split_chunks.k3_rows_statistic` computes s.
  - R2 and T3 are Σ_rows C(len, 2) and C(len, 3) over the active columns.
  - ρ = P2 / R2, with P2 the K=2 level's summed counts.
  - ρ² T3 estimates the row-wise kernel's increments, which r3 left out
    (`REPORT-M2.md` finding 1).
- The row sources (`DeviceRows`, `_HostRows`) give T3 beside R2 in one pass
  (`pairs_triples`).
- `k3_counts_rows(s, slots, budget)` picks rows below `K3_ROWS_MAX_S = 1.0e-2`
  when the slots fit one chunk. `ET_MINER_K3_KERNEL=dense|rows` pins it, as
  before.
- The miner logs r3, s and ρ at K=3 whichever kernel runs, and the level
  split times `pairs_triples` as `k2_dispatch`.
- `k3_crossover.py` has two new grids (`--grid dispatch`, `--grid heldout`)
  and a regret analysis.
- Tests:
  - the statistic and the dispatch;
  - row triples on the host CSR, and on the device layer against it;
  - the tier chain's row-wise K=3 dispatch leg, now under s (smoke's s is
    2.4, so the leg raises the threshold to 10).

## Results (2× RTX 3060 / EPYC 7282, GPU 0, median of 3 per kernel, K=3 level)

**Regret** is the K=3 time a rule loses against the faster kernel, summed
over the points.

### The sweep (42 of 43 points, `k3_crossover.txt`)

| rule | threshold | regret |
|---|---|---|
| r3, before this change | 2.5e-4 | 16,241 ms |
| r3, best threshold | 2.0e-3 | 4,166 ms |
| s over the generated candidates (the protocol's P) | 6.8e-3 | 3,437 ms |
| **s over the slots** | **1.0e-2** | **5 ms** |
| s with the counted increments, over the generated candidates | 3.7e-3 | 2,253 ms |

- P passed rule 1.
- Nearly all of P's regret is one Zipf point, F1600-L48 (rows 9.3 s, dense
  5.9 s), which P sent to rows. On Zipf rows the generated candidates are
  4–5× the slots. P then read the dense level as larger than it is.
- Amendment 1 tested the slots' variant on points the sweep did not contain.

Where the old rule lost most, and the new one picks rows:

| point | r3 | s (slots) | dense | rows |
|---|---|---|---|---|
| uniform N1M F1600-L48, ρ 0.53 | 3.9e-4 | 2.1e-3 | 8.60 s | 2.25 s |
| uniform N1M F1600-L64, ρ 0.52 | 7.0e-4 | 4.7e-3 | 9.32 s | 5.20 s |
| uniform N4M F450-L12 | 2.8e-4 | 1.2e-3 | 2.77 s | 0.65 s |
| uniform N1M F1200-L48 | 2.5e-4 | 4.1e-3 | 14.21 s | 13.52 s |
| uniform N4M F300-L8 | 4.0e-4 | 1.2e-3 | 0.85 s | 0.14 s |

### Held-out points (Amendment 1, 4 of 8, `k3_heldout.txt`)

All four are Zipf points where s over the generated candidates picks rows and
s over the slots picks dense. Dense won all four:

| point | s (generated) | s (slots) | dense | rows |
|---|---|---|---|---|
| N1M F1000-L32 | 4.9e-3 | 1.9e-2 | 1.43 s | 1.92 s |
| N1M F1200-L40 | 5.4e-3 | 2.2e-2 | 2.67 s | 4.36 s |
| N1M F1600-L40 | 2.5e-3 | 1.1e-2 | 5.06 s | 5.21 s |
| N1M F1600-L44 | 3.2e-3 | 1.3e-2 | 5.37 s | 6.98 s |

- Regret at the frozen thresholds:
  - s over the slots: 0 ms;
  - s over the generated candidates: 3,953 ms;
  - r3 before this change: 151 ms.
- Rule 1 of the amendment holds: 0 ≤ 0.9 × 3,953, and the saving is above
  1,000 ms.
- The owner chose to adopt it on these four points.

### The matrix (final check)

One rep of the 20 moonshot configs, every knob unset:
- every signature equals Phase 3's, and no run logged a fallback;
- K=3 ran row-wise in sk2ml3 and oom2ml3 on one and two GPUs, dense in the
  other 16, as before.

Measured once to K=3 beforehand, the regimes sit far from the threshold:
- the dense ones at s ≥ 0.045 (or002 the closest);
- sk2ml3 at 2.1e-4 and oom2ml3 at 8.6e-4.

## Findings

1. **(should)** *Why the dense level follows the slots is unmeasured.* The
   dense kernels iterate the generated candidates, but their time tracks the
   slots far better.
   - A plausible cause is that the subset test makes a pruned candidate
     cheap.
   - The point meant to show it (`zipfrev-N1M-F1600-L32`: the item ids
     reversed, generated = slots) did not run.
2. **(should)** *The held-out decision rests on 4 of its 8 points.* The cap
   stopped the run (0.183 of 0.15 GPU-hours; the estimates were 20–30 %
   low).
   - Two disagreeing points did not run: N4M F600-L16 and N2M F1000-L32. To
     reverse the decision, rows would have to win them by more than 2.95 s
     together. Their dense levels are planned at about 1.2 and 2.9 s, and
     the same shape at N1M lost row-wise (1.35×). That is an argument, not a
     measurement.
   - The sweep's last point (`zipf-N1M-F2400-L32`, rows by every rule) did
     not run either.
3. **(nice)** *ρ² T3 estimates the increments well on Zipf rows* (5 % high).
   On uniform rows with ρ ≈ 1/2 it is 1.8× high, where ρ³ fits (independent
   pairs). The dispatch's threshold absorbs that on the swept points.
4. **(nice)** *The matrix gains nothing.* The change pays on mid-range K=3
   levels: tens of items per row over hundreds to thousands of frequent
   items. No regime of the matrix has such a level.
5. **(nice)** *The dense K=3 level is 20–25 % slower at F1600 than in M2's
   sweep* (F1600-L32: 26.8 s against 21.7 s at `313dfc9`, another tree). It
   is unexplained. The council's open item on the deep parameters' register
   cost (`REPORT.md`, PR #35 council, item 7) is a candidate.
6. **(nice)** *Process.* The first held-out start appended the environment
   capture to a tracked `env.txt`, which would have stamped every row dirty.
   It was stopped within its first point (about 20 s of GPU time, no row),
   the capture was committed, and the run restarted. The final check wrote to
   a directory of its own for that reason.

## Budget

0.950 GPU-hours of 1:
- sweep 0.724 (cap 0.7, checked between points);
- held-out points 0.183 (cap 0.15);
- the aborted start about 0.006;
- final check 0.037.

The untimed diagnostics (dry runs of the sweep's child, the matrix regimes'
statistics) took a few GPU-minutes more.

## Gates (`b520037`)

| check | result |
|---|---|
| `uv run pytest tests/test_tier_equivalence.py -q` | 91 passed |
| `uv run pytest -q -m "not slow" -k "not test_support_00"` | 1629 passed |
| `uv run pytest -q -m "gpu and slow" -k "not test_support_00"` | 17 passed |
| not-slow with CuPy hidden, as on the CPU-only CI runner | 898 passed, 563 skipped |
| `uv run ruff check src tests bench` | clean |
