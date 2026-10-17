# X0 — Count inference by default: report

**The table, for the owner's decision.** `use_generator_pruning=True` wins
dsl by rule 2, on one GPU and on two, and ties every other complete-lattice
regime. It loses none. Signatures are identical in every regime. The default
stays `False` until the owner decides.

Protocol: `PROTOCOL-X0.md`. Data: `../results/2026-10-17-moonshots-x0/`
(`decision.md`, `split.md`, `raw.jsonl`), at `68c7d27` on
`perf/moonshot-x0`. That tree is M1 + M2: the device input layer and
row-wise K=3 by dispatch.

## Results (2× RTX 3060 / EPYC 7282, median of 3, seconds)

| regime | GPUs | inference off [min, max] | on [min, max] | on/off | rule 2 |
|---|---|---|---|---|---|
| dsl | 1 | 10.65 [10.63, 10.68] | 5.83 [5.81, 5.87] | 0.547 | **on wins** (−4.82 s) |
| dsl | 2 | 6.83 [6.83, 6.84] | 4.83 [4.37, 4.84] | 0.707 | **on wins** (−2.00 s) |
| sk2ml3 | 1 / 2 | 1.79 / 1.88 | 1.87 / 2.36 | 1.05 / 1.26 | tie |
| oom2ml3 | 1 / 2 | 0.38 / 1.15 | 0.39 / 1.15 | 1.03 / 1.01 | tie |
| deepk | 1 / 2 | 0.17 / 0.67 | 0.17 / 0.67 | 1.03 / 1.01 | tie |
| skew | 1 / 2 | 0.20 / 0.68 | 0.19 / 0.68 | 0.97 / 1.00 | tie |
| or003 | 1 / 2 | 0.10 / 0.58 | 0.09 / 0.58 | 0.90 / 1.00 | tie |
| or002 | 1 / 2 | 0.29 / 0.59 | 0.30 / 1.08 | 1.02 / 1.82 | tie |
| smoke | 1 / 2 | 0.05 / 0.58 | 0.05 / 0.58 | 0.98 / 1.01 | tie |

The 2-GPU medians of sk2ml3 (+0.48 s) and or002 (+0.49 s) with inference
on come from the tail after the last level, not from mining:
- the input and the levels agree within 0.1 s per rep;
- the tail is either about 0.07 s or about 0.5 s, in both arms, rep by rep.

On one GPU, where the tail is 0.01 s, both regimes tie at 1.02–1.05×.

Budget used: 0.271 GPU-hours (the brief's estimate was 0.1 for one GPU; the
table also runs on two).

## Findings

1. **(should)** Flipping the default (`use_generator_pruning=True`) is the
   owner's decision. On this table it saves 4.8 s of dsl's 10.7 s on one GPU
   and costs nothing measurable elsewhere. The output is identical by
   construction (Pascal inference is exact) and by the signatures here.
2. **(should)** On two GPUs the tail is bimodal, in every campaign so far
   (M1 finding 4, M2 finding 4, here): about 0.07 s or about 0.5 s, with the
   same levels. It decides the 2-GPU medians of the small regimes.
   Teardown candidates: the NCCL communicators, the GCS uploader's drain, the
   pools. Timing them as span phases would name it.
