# GPU campaign findings — 2×RTX 3090, 2026-08-31

Branch under test: `claude/density-adaptive-gpu-transition-g0i3t5` (memory +
throughput optimization campaign). Box: 2× RTX 3090 24 GB (sm_86, driver
580.159.03, CuPy 14.1.1, no P2P — NCCL over SHM), vast.ai. Raw data:
`raw.jsonl`; per-config medians: `report.md`; environment: `env.txt`.

## Correctness (the part that gates everything else)

- **Tier-equivalence chain: 7/7.** Tier 1 Polars == Tier 2 Rust ==
  single-GPU == multi-GPU legacy == shared multi-GPU == efficient-apriori,
  exact itemsets AND absolute counts, on the smoke preset (CLAUDE.md policy).
- **Full GPU test suite: 114 passed / 0 failed** — first-ever on-device run;
  includes the OOM-regression under per-device CuPy pool limits, forced
  NCCL-fallback equivalence, forced multi-chunk equivalence, density-auto
  vs dense equality, and shared-vs-legacy kernel bit-equality.
- **Cross-config refutation:** all result signatures (sha256 over sorted
  itemset:count) identical across kernel variant × GPU count × filter impl ×
  NCCL mode × row balance × density mode — with ONE deliberate exception,
  which is finding #2 below.

## Finding 1 — shared/tiled kernel: 8.8× / 16.3× on the stress workload

`stress_k2`: 2M transactions × 35K items, min_count 30, mined to K=3 —
612M K=2 candidates plus **~76 billion** K=3 candidates, every one exactly
counted (no sampling):

| Pipeline | legacy | shared/tiled | speedup |
|---|---|---|---|
| 1× 3090 (fused) | 2238.8 s | 253.3 s (±1 s over 3 reps) | **8.8×** |
| 2× 3090 (dense row-split) | 1996.7 s | 122.3 s | **16.3×** |

≈300M exact support counts/s on one 3090, ≈630M/s on two. Shared also
scales ~2× across GPUs where legacy barely moved (counting was the
bottleneck; once it is cheap, row-split parallelism shows). On small/deep
data (`deep_k`, `smoke`) shared is neutral (0.97–1.07×): per-level work is
tiny and sub-64-pair groups route to the legacy kernel by design.
**Decision: `ET_MINER_KERNEL_VARIANT=auto` keeps resolving to `shared`.**

## Finding 2 — the legacy sampled prefilter provably drops itemsets

`stressk2-legacy-1g` (the only config where the sampled prefilter engages)
mined **9,285 fewer K=3 itemsets (−0.7% of that level)** than the seven
exact configs, which all agree bit-for-bit. Root cause is pre-existing: the
prefilter rejects candidates whose sampled estimate is < 0.7×min_count
*without exact recount* — a silent false-negative path that had never been
measured until this campaign's refutation harness compared signatures.
**Decision: the prefilter is now OPT-IN (`ET_MINER_ENABLE_PREFILTER=1`)
and off by default.** The raw.jsonl divergence rows are retained as the
evidence; the equivalence checker's failure on this campaign is this
finding, not a harness defect.

## Finding 3 — density-auto transition: correct, but the CSR path needs its
Python bottleneck fixed before it pays off

`deepk-density-auto` produced identical results to dense mining (hashes
equal) and the transition fired as designed — but took 10.4 s vs 1.3 s
dense: post-transition levels (K=5–9) spend seconds in the sparse path's
host-side pair-building loop (a known, explicitly out-of-scope bottleneck).
The n/32 crossover optimizes memory, not yet time, at this scale.
**Decision: defaults unchanged** (`apriori(sparse_from_k=None)`); revisit
after the CSR pair-building loop is vectorized.

> Follow-up (2026-09-01): the attribution above was wrong — the host
> tidset rebuild, not the pair loop, cost ~2.5 s per level. With the
> sparse levels made GPU-resident and row-split, `deepk-density-auto`
> runs in 1.34 s (== dense) with the identical signature. See
> `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md`.

## Smaller readings

- **Filter impls** (compact/cupy/cpu) indistinguishable at this scale
  (89.6/89.7/90.2 s): survivors are MBs and the cpu impl's full-array D2H
  is only 2.4 GB here. `compact` stays default — it is never worse and is
  structurally required at the 80 GB-D2H scale this box cannot reach.
- **NCCL fallback** (staged D2D): works, 0.9 s vs 1.3 s with NCCL on tiny
  data (init overhead dominates); exercised for equivalence, not speed.
- **Row balance**: `nnz` shows no win over `rows` on the clustered preset
  (1.5 vs 1.4 s) — `rows` stays default, as the anti-balancing analysis
  predicted (dense cost scales with rows, not nnz).
- **Two-phase mining**: first-ever verified run, equals plain row-split at
  equal supports.
- All heavy runs show clock throttling (⚠) — 3090 power/thermal limits;
  expected, and why medians over reps are reported.
