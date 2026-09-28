# Consolidation findings

One row per decision point: candidates, the winner of each regime under rule 2,
the decision, and the evidence (config ids; cells are median [min, max] seconds
from `report.md`). Protocol and decision rule: `bench/consolidation/PROTOCOL.md`,
fixed before the campaign ran and applied as written.

## Conditions

- **Box.** 2× RTX 3060 12 GB (sm_86, no P2P), Ryzen 5 5600X 6C/12T, 31 GB RAM
  (`env.txt`), not the 2× RTX 3090 24 GB the brief assumed; `deep_sparse_large`
  is sized for 12 GB. sm_90 is untested.
- **Revisions.** Campaign rows at `e4bb3ae`; supplementary rows at `933ac98`
  (harness-only changes since: the supplement mode, the report fix, the sweep
  script).
- **Device loss.** GPU 1 (`0000:2B:00.0`) fell off the bus at 20:38:57 UTC
  during `dsl-C2-shared#r0`; `nvidia-smi` reports "Unable to determine the
  device handle ... Unknown Error" and a reset is impossible inside the
  container. The 1-GPU configs resumed on device 0. Every 2-GPU config that had
  not run by then is missing: C2 has one rep on smoke/deepk/skew/oom2/sk2ml2
  and none on dsl, the pair/candidate-split routes (`Asplit2`, `Bsplit2`) have
  none, `E2` has one rep on deepk and a timeout on dsl, and the DP10 arms have
  one rep each. Those rows were re-measured on the second box (next bullet).
- **Second box.** The 2-GPU rows (DP6, DP10, `E2`) were measured on
  2026-09-28 on a second vast.ai container: 2× RTX A4000 16 GB (sm_86, P2P
  through the host bridge), the same Ryzen 5 5600X, 46 GB RAM, CuPy 14.1.1,
  NCCL 2.31 (`../2026-09-28-consolidation-2gpu/`, `env.txt` there). Only
  within-box comparisons are made: every DP6 and DP10 cell compares 2-GPU
  configs measured there, and no cell compares a 3060 number with an A4000
  number. On that box NCCL's P2P transport hung about one run in three at
  the first collective (both ranks enqueue, rank 1 completes, rank 0 never
  does; `../2026-09-28-consolidation-2gpu/nccl-hang/`), so every row there
  was measured with `NCCL_P2P_DISABLE=1` (0 hangs in 30 repro runs and 82
  config-reps). The code path is unchanged (NCCL ring reduce); only the
  transport between the two devices differs.
- **Correctness.** Every regime produced exactly one signature across all GPU,
  SON and CPU configs (15 regimes, 459 ok config-reps, `report.md` §Signatures);
  no ok row logged a fallback. The 29 non-ok config-reps are 18 out-of-memory
  errors (sparse layouts on dsl), 2 timeouts (dsl SON, 600 s cap by protocol)
  and 9 max_length=3 gate skips. The 2-GPU rows on the second box: one
  signature per regime in all 7 regimes, equal to the Phase 2 signature (72
  ok config-reps); the 10 non-ok rows are 9 gate skips and the dsl `E2`
  timeout (600 s cap).
- **Budget.** Campaign 4.68 GPU-hours (process time × devices, all 495 rows
  including the incident), calibration ≈ 0.72, microbench dumps ≈ 0.1,
  supplementary 1.04 (kernel sweep ≈ 0.40, oom2 to K=3 0.64). Total ≈ 6.5 of the 8 allowed.
  The 2-GPU rows on the second box added 3.44 GPU-hours (1.72 h box time,
  82 rows), so the campaign as a whole used ≈ 10 GPU-hours: the overrun is
  the re-measurement the device loss forced.

## Decisions

| DP | Candidates | Winner per regime (rule 2) | Decision | Evidence |
|---|---|---|---|---|
| DP1 in-core miner | A single-GPU bitvec, B GPU-resident, C row-split (each at its fastest variant) | C: oom2 (8.22 [8.21, 8.24] vs A 10.34 [10.27, 10.41]), dsl (24.19 [23.95, 24.27] vs B 26.98 [26.88, 27.00]), or002 (0.35 [0.35, 0.39] vs B 2.66 [2.63, 2.71]). A: sk2ml2 (56.21 [56.20, 56.37] vs C 453.57 [453.55, 453.61]) and sk2ml3 (734.66, the only config the max_length=3 gate admitted). B: none. | C is the one in-core miner. B goes (rule 3). A wins exactly where the pair counts do not fit one dense chunk on one device and C falls back to the per-candidate kernel (oom2's 450M pairs fit: C 7.64 vs A 7.99 s at K=2; sk2ml2's 612M next to 8.75 GB of bitvecs do not: A 47.21 vs C 450.99). Rule 4: that case is selected by candidate count and free VRAM before the level runs, so C counts it the way A does, with the fused tiled kernel; the rest of A goes. | `{oom2,sk2ml2,sk2ml3,dsl,or002}-{A1-shared,A1-legacy,B1,C1-shared,C1-legacy,C1-tiny0}` |
| DP2 K=2 counting | tiled (dense and fused) vs per-candidate (dense and fused) | Tiled: oom2 (A1 7.99 [7.98, 8.01] vs 83.06 [83.03, 83.06]; C1 7.64 [7.62, 7.64] vs 82.49 [82.49, 82.49]), sk2ml2 (A1 47.21 [47.11, 47.37] vs 453.16 [453.16, 453.18]). Per-candidate: none. C2 agrees on one rep (oom2 4.27 vs 42.11, sk2ml2 25.60 vs 328.71). | Tiled K=2. The per-candidate dense K=2 stays only for pair spaces chunked across several GPUs (DP4). The per-candidate fused K=2 (`pairs_k2.cu`) loses its callers with A and B. | K=2 level of `{oom2,sk2ml2}-{A1,C1,C2}-{shared,legacy}` |
| DP3 K≥3 counting | tiled, per-candidate (dense in C, fused in A), B's `count_k3plus_gpu_resident` | Within C1: per-candidate on dsl (14.63 [14.58, 14.74] vs tiled 76.39 [76.38, 76.43]); tiled none. Within A1: tiled on or002 (1.26 [1.24, 1.26] vs 2.61 [2.59, 2.61]) and sk2ml3 (675.65, unopposed); per-candidate on dsl (17.97 [17.93, 18.05] vs 108.04 [108.02, 108.10]). B's kernel: none. | Both kernels win somewhere, so rule 4: dispatch per prefix group on its pair count. Crossover (kernel sweep, supplementary): the tiled kernel's time over the per-candidate kernel's does not depend on the row count from 7,813 to 312,500 words (both kernels scale with the words) and falls with the suffixes per group and with K. Parity sits at 120 pairs per group at K=3, 91 at K=4, 66 at K=5, 45 at K=6 and ≈ 23 at K=8; at 2 suffixes the tiled kernel is 41× slower at K=3 and 15× at K=6. At 570 words the crossovers sit lower, with every launch under 10 ms. The dsl loss of tiled in C1 is not the threshold: at K=4–9 the fragmentation guard dropped the small-group routing and tiled every group (the run log of `dsl-C1-shared#r0`, not committed, shows `chunk 1/1` at each of those levels), and on 2–4-suffix groups the tiled kernel is 15–45× slower. The dispatch therefore partitions a level by kernel instead of interleaving chunks. B's kernel goes with B. | DP3 tables; `crossover.jsonl`; `supplement/` |
| DP4 what tiled can't serve | C1-shared (small groups, mega-groups and multi-chunk K=2 per-candidate), C1-tiny0 (small groups tiled too), C1-legacy (all per-candidate) | C1-shared: oom2 (8.22 [8.21, 8.24] vs 83.09). C1-legacy: dsl (24.19 [23.95, 24.27] vs 86.00 [85.95, 86.01]). C1-tiny0: none. | Keep the per-candidate dense kernel as the named fallback (small groups; mega-groups and multi-chunk K=2 on several GPUs) and keep small-group routing (tiny0 wins nothing). Port the one-device oversize case to the fused tiled kernel (DP1: A1-shared vs C1-shared on sk2ml2). | `{oom2,dsl}-C1-*`, `sk2ml2-{A1,C1}-shared` |
| DP5 layout | dense, `sparse_from_k="auto"`, `sparse_from_k=3` | Dense: dsl (every sparse layout runs out of memory on 12 GB: the CSR shards hold Σ support × 4 B per level). A1-legacy-auto: or002 (2.71 [2.69, 2.73] vs 4.57 [4.55, 4.62]), inside A only; C1 dense takes 0.82 there. Elsewhere none. | Dense only; the sparse CSR layout goes and `"auto"` does not become the default. It wins no regime in the surviving miner and has no capability dense lacks: its transition needs the dense bitvecs first. | DP5 tables |
| DP6 multi-GPU | C row-split + NCCL vs A/B pair/candidate split (2-GPU rows measured on the second box, `../2026-09-28-consolidation-2gpu/`) | C2-shared: oom2 (Σ K≥2 3.40 [3.38, 3.41] vs Asplit2 37.25 [37.20, 37.26], Bsplit2 36.39 [36.35, 36.40], C2-legacy 35.35 [35.31, 35.36]), sk2ml2 (17.22 [17.15, 17.23] vs 225.08 [223.73, 227.38], 224.67 [222.45, 227.63], 188.87 [188.86, 188.87]), sk2ml3 (781.87 [779.92, 785.05], unopposed: every other candidate failed the max_length=3 gate). C2-legacy: dsl (5.65 [5.63, 5.65] vs 27.69 [27.67, 27.94]; the DP3 kernel question, settled by the per-group dispatch). Splits: none. | C's row split with the NCCL reduce is the multi-GPU scheme. The pair/candidate splits win nothing and stay gone with A and B (DP1); `bitvecs=` on several GPUs uses C's row split. Wall on the A4000s: sk2ml2 C2-shared 20.00 [19.98, 20.05] against 232.85–237.90 for the splits. Observed, not a decision: sk2ml3 on two A4000s takes 784.87 s wall, where the fused one-GPU path took 711.57 s on one 3060 (Phase 4); across two GPUs the oversize K=3 groups run on the per-candidate kernel (DP4), not the fused tiled one — see `REPORT.md`, open risks. | `*-C2-*`, `*-Asplit2-shared`, `*-Bsplit2` |
| DP7 out-of-core | SON single-GPU pass 1 on B + pass 2 `count_itemsets_batch` (`D1-resident`), both passes on the per-level GPU counter (`D1-gpu`), CPU (`D1-cpu`); multi-GPU (`E2`) | D1-resident: deepk (3.69 [3.36, 3.76] vs D1-cpu 12.21 [11.98, 12.26]), dsl (194.75, one rep, vs a >600 s timeout for D1-gpu and E2) | SON stays (rule 6). Pass 1 moves to C per chunk; pass 2 keeps `count_itemsets_batch` (pass 2: deepk 1.70 s vs 6.84 s for the per-level counter; dsl 132.6 s vs a timeout); the per-level counter's Python loop goes; E gets the same pair on each of its devices. Overhead against the in-core winner: deepk 3.69 vs 0.43 s (8.6×), dsl 194.75 vs 24.19 s (8.1×). | `{deepk,dsl}-{D1-*,E2}` |
| DP8 CPU tier | Polars, sparse + Rust, sparse without Rust | Polars: deepk (2.64 [2.44, 2.67] vs 5.46 [5.15, 5.58]), skew (3.52 [3.31, 3.52] vs 6.71 [6.54, 7.00]). Sparse + Rust: or003 (39.81 [39.04, 39.97] vs 44.77 [44.53, 44.77]). Sparse without Rust: none. | Keep both. `sparse=None` is already a rule-4 dispatch that picks the winner in every regime that has one (sparse when C(n,2) > 100K, i.e. n ≥ 448 frequent items, or n·rows/8 > 1 GiB): F-auto deepk 2.61, skew 3.49, or003 39.43. | DP8 tables |
| DP9 Rust host roles | R2 group build, R3 Apriori group prune, R4 free-set prune: Rust vs fallback; `prune_apriori` on vs off | Per call the fallbacks are slower everywhere: R2 2.1–13×, R3 3.9–18.6× (0.9× on the 4-candidate K=16 call), R4 6.1–27×; R3 on stress_k2 K=3 takes 487 s in Rust (fallback > 300 s cap). Prune vs no prune: no regime either way (largest gap 0.31 s on dsl). Supplementary: on oom2 to K=3 the prune costs 63.6 s of a 93.1 s K=3 level (C1-shared 101.25 [100.33, 102.08] vs C1-shared-noprune 37.62 [37.48, 37.65]). | R2 and R4 stay. Neither the prune nor `prune_apriori=False` wins a regime, a tie everywhere, so rule 5 keeps the smaller code: the prune goes and R3 with it (rule 7: the role disappears). | DP9 tables, microbench |
| DP10 leftover A/B arms | filter `compact`/`cupy`/`cpu`; row balance `rows`/`nnz` (measured on the second box, `../2026-09-28-consolidation-2gpu/`) | None (rule 2). Filter on sk2ml2 (C2-shared): compact 20.00 [19.98, 20.05], cupy 19.05 [18.06, 19.05], cpu 19.99 [19.00, 20.07]; on dsl (C2-legacy): 17.91 [17.90, 18.41], 18.49 [17.50, 19.23], 18.83 [17.96, 19.05]. Row balance: skew rows 1.07 [1.06, 1.09] vs nnz 1.07 [1.07, 1.07]; dsl 17.91 [17.90, 18.41] vs 18.14 [18.02, 18.80]. | A tie everywhere, so rule 5 keeps the least code on each axis. Row balance: `nnz` goes (`ET_MINER_ROW_BALANCE=nnz` and `balance="nnz"` raise naming the rows split). Filter: the sliced CuPy filter is the one implementation (`threshold_filter`); the `compact_threshold` kernel (two launches, a host sort at 48 B/survivor, a host-RAM probe) and the whole-array CPU path go, and `ET_MINER_FILTER_IMPL` raises. Its capability is kept: measured per 64M-element slice on an A4000, 94 ms and 1.1 B/element of extra VRAM at a 1% pass rate, 13 B/element (794 MiB) with every element surviving, under the budget's 1 GiB margin, and a slice that does not fit is filtered on the host (tested). Not measured: the 10B-candidate levels of the AlphaFold regime, where the slices cost ≈ 15 s per level against two kernel passes. | `*-C2-*-filter-*`, `*-C2-legacy-balance-nnz`, `{sk2ml2,dsl,skew}-C2-{shared,legacy}` |

**Rule 5 on DP10.** Both axes tie in every regime, so rule 5 ("keep
whatever serves the most routes with the least code") decides, as it did on
DP9. Every filter implementation serves the same one caller (the dense
chunk filter on GPU 0) and the sliced CuPy filter is the smallest; the same
holds for the rows cut against the nnz cut. The peak-VRAM tiebreak is not
reached.

**Rust, per role.** R1 (CPU K>2 counting) stays: without it the sparse CPU
route's K≥3 levels are 4–36× slower (F-sparse vs F-sparse-norust) and DP8
keeps that route. R2 stays, R4 stays, R3 disappears with the prune. No
`refactor/drop-rust` branch.

## Supplementary measurements (not pre-registered)

The pre-registered matrix left no regime with wide K≥3 groups in which both K≥3
kernels ran on the row-split miner: the max_length=3 gate skipped `C1-shared` on
stress_k2 (its K=2 had fallen back to the per-candidate kernel), and the 2-GPU
configs were lost with the device. Two measurements were added after the
campaign, with the harness changes committed first (`d6ecea7`, `933ac98`); they inform the
rule-4 crossover and never remove anything on their own.

**Kernel sweep** (`bench/kernel_crossover.py`, `crossover.jsonl`, 131 points,
every point bit-identical between the two kernels). Synthetic prefix groups over
correlated random bitvecs, ≈ 200,000 candidates per point, K ∈ {3, 4, 5, 6, 8},
2–512 suffixes per group, 570 / 7,813 / 31,250 / 312,500 words (36K–20M rows).
Tiled time over per-candidate time at 31,250 words:

| suffixes (pairs) | 8 (28) | 10 (45) | 12 (66) | 14 (91) | 16 (120) | 20 (190) |
|---|---|---|---|---|---|---|
| K=3 | 4.20 | 2.57 | 1.84 | 1.33 | 1.03 | 0.66 |
| K=4 | 2.98 | 1.85 | 1.36 | 0.98 | 0.78 | 0.47 |
| K=5 | 2.13 | 1.33 | 0.98 | 0.72 | 0.57 | 0.36 |
| K=6 | 1.44 | 0.92 | 0.68 | 0.52 | 0.41 | 0.26 |
| K=8 | 0.83 | 0.50 | 0.39 | 0.30 | 0.23 | 0.15 |

The 7,813- and 312,500-word points of the full sweep agree with these within 12%.

**oom_regression to K=3** (`supplement/`, 3 reps each, rev `933ac98`, one
signature across all 12 config-reps). K=3 is 2.08B candidates in 432 prefix
groups (median 192 suffixes) over 7,813 words.

| config | wall s | K=2 s | K=3 s |
|---|---|---|---|
| A1-shared | 41.28 [41.11, 41.31] | 8.61 | 29.95 |
| C1-shared | 101.25 [100.33, 102.08] | 7.61 | 93.08 |
| C1-shared-noprune | 37.62 [37.48, 37.65] | 7.51 | 29.51 |
| C1-legacy | 575.27 [574.26, 575.80] | 82.49 | 492.21 |

Within C1 the tiled kernel wins this regime under rule 2 (K=3 93.08 vs 492.21);
without the host-side prune C1 matches A1, which has none.

## Open

- DP6 and DP10 need the second GPU. The missing configs are the 2-GPU rows of
  the campaign matrix at `e4bb3ae` (`--only C2`, `--only split2`, `--only E2`
  on `bench/runner.py --mode consolidation`, which resumes past every ok row);
  they have to run from a checkout of that revision, since Phase 3 removes the
  split routes.
- The multi-GPU tier-chain legs skip on one device; they must run on the
  restored box.
