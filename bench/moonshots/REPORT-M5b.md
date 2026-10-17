# M5b — Row reordering and the tile-occupancy skip: report

**Decision: the lex row order stays the default for the device input layer.**
It wins 3 of the 20 regimes by rule 2 (dsl on one and two GPUs, dsl with free
sets on one) and loses none. The final check (unset) runs lex everywhere and
matches the `lex` arm. No signature mismatch, no logged fallback.

Protocol: `PROTOCOL-M5b.md`. Data: `../results/2026-10-17-moonshots-m5b/`, at
`797b54c` on `perf/moonshot-m5b` (stacked on M7, X0, M2, M1).

## Results (2× RTX 3060 / EPYC 7282, median of 3, seconds)

| regime | GPUs | keep [min, max] | lex [min, max] | lex/keep | rule 2 |
|---|---|---|---|---|---|
| dsl | 1 | 10.87 [10.69, 10.91] | 4.92 [4.90, 4.93] | 0.453 | **lex wins** (−5.95 s) |
| dsl, free sets | 1 | 5.29 [5.28, 5.33] | 3.76 [3.75, 3.79] | 0.711 | **lex wins** (−1.53 s) |
| dsl | 2 | 6.88 [6.87, 7.39] | 4.11 [4.09, 4.11] | 0.596 | **lex wins** (−2.77 s) |
| dsl, free sets | 2 | 4.33 [4.30, 4.34] | 3.60 [3.60, 3.61] | 0.833 | tie (0.73 s saved) |
| dsl, count inference | 1 / 2 | 3.67 / 3.35 | 3.46 / 3.59 | 0.94 / 1.07 | tie |
| sk2ml3 | 1 / 2 | 1.78 / 1.88 | 2.00 / 2.41 | 1.12 / 1.28 | tie (+0.22 s on one GPU) |
| oom2ml3, deepk, skew, smoke, or003, or002 | 1 / 2 | | | 0.84–1.07 | tie |

On dsl (one GPU) the K≥2 levels fall from 8.9 s to 2.6 s: the per-candidate
and group kernels skip the tiles their items do not occupy. The order costs
0.41 s more input (1.92 → 2.33 s). With count inference and the bounds (M7)
dsl's counting was already small, so the order saves 0.75 s of levels and
pays its 0.42 s input.

Final check: 20 of 20 ok, signatures consistent. dsl 4.91 / 4.08 s, dsl free
sets 3.76 / 3.59, sk2ml3 1.99 / 1.93 (one / two GPUs).

Budget used: 0.442 GPU-hours.

## Findings

1. **(should)** The order's 0.41 s on dsl splits as:
   - `cp.lexsort` over the two keys, 0.25 s;
   - the permuted `ids_to_bitvec`, 0.15 s against 0.02 s: atomics on
     scattered words.

   Two stable radix argsorts give the same order in about 0.05 s: by the low
   key, then the high key gathered through it. Scattering by target position
   instead does not help (0.14 s; the rows are then read uncoalesced).
2. **(should)** sk2ml3 pays 0.22 s on one GPU:
   - 0.06 s of input for the order;
   - about 0.11 s of K=1, which waits on the asynchronous occupancy kernels
     (35,000 columns × 31,250 words).

   Its K=3 counts row-wise, so the occupancy is never read. Building it lazily,
   at the first dense K≥3 level that runs the per-candidate or group kernel,
   removes that cost.
3. **(nice)** On two GPUs sk2ml3's lex median (2.41) and dsl-infer's (3.59)
   carry the bimodal 2-GPU tail (X0 finding 2): their levels match the keep
   arm's.
4. **(nice)** The tiled kernel and K=2 do not read the occupancy; dsl spends
   0.12 s and 0.11 s there.
