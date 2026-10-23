# PLAN — CPU reproduction (`CPU_REPRO_GOAL.md`, Phases 2–3)

Status: **frozen 2026-10-20T16:00Z** (22.5 h after the goal was set). Re-estimates after each run are
appended under "Re-estimates"; the queue below changes only by those recorded decisions.

Data: `data/extract_2026_01/transactions_214m_base_multi.parquet` (76,890,945 rows; RESULTS CPU-017).
Threads: 6 (`scripts/run_py.sh`, `n_jobs=6`, every pool pinned, `CUDA_VISIBLE_DEVICES=`). Outputs:
`data/mined/<ID>/` (per-K `parquet/k*.parquet` with itemset, count, support; `meta.json`). Logs:
`logs/<ID>_<UTC>.log`. Markers: `state/<ID>.done`. Runner: `scripts/queue.sh <RUN>` (one run at a time,
so no two runs share the CPU).

## Thresholds

Asserted by the harness before every mine: `ceil(Fraction(str(s)) * N) == min_count`.

| run | min_support | min_count |
|---|---|---|
| G0 | 5e-05 on 1,000,000 rows | 50 |
| C1 / S1 / R1 Base | 1e-3 | 76,891 |
| C2 / S2 Super | 1e-4 | 7,690 |
| C3 / C3G / S3 / X1 Power, N5 / N100 | 1e-5 | 769 |
| C4 Blitz | 1e-6 | 77 |
| C5 Ultra | 2e-7 | 16 |
| C6 Opus | 1e-7 | 8 |
| M4 | (4 − 0.5) / N | 4 |
| P1T (SON pass 1 at Base on one 40M-row chunk) | 9e-4 on the first 40,000,000 rows | 36,000 |

## Budget

- `BUDGET_HOURS` 96 for the Phase 2–3 mining. Used by the gate and the sizing runs: 9.08 h
  (G0 0.08 h, C1 0.08 h, C2 1.77 h, C3 7.11 h, P1T 0.04 h; `data/mined/*/meta.json`).
  Remaining: **86.9 h**.
- The `/goal` guard (120 h since 2026-10-19T17:23:32Z) leaves 97.5 h at the freeze; minus 8 h for
  Phase 4 that is 89.5 h, so the mining budget is the smaller, 86.9 h.

## Sizing model (`scripts/sizing.py` → `data/sizing/sizing.json`)

The CPU route counts far fewer candidates than the GPU miner (C3, K=3: 224,019 vs 1,855,572), and its
time per candidate depends on the level K (through the row space it counts over) much more than on the
threshold: C2 and C3 agree per level within about 1.5× (K=5: 63 vs 43 ms; K=8: 180 vs 205 ms per
candidate), and both drop to about 0.2 ms per candidate from K=9 on.

- In-core run T: load + fixed overhead + Σ_K F_T(K) · r(K) · c(K). F_T(K) is the GPU log's frequent
  count at K for T's threshold (`data/sizing/gpu_levels.json` from `single_gpu/full_campaign_1gpu.log`
  and `minc4_mining.log`), r(K) C3's CPU candidates per frequent itemset, c(K) C3's seconds per
  candidate. Levels deeper than K=14 use r = 1 and c of C3's deepest level with ≥ 1,000 candidates (K=10).
- Check: C2 predicted from C3 = 5,865 s, measured 6,382 s (ratio 1.09); C3 predicted from C2 =
  28,420 s, measured 25,604 s (0.90). Safety factor on every unmeasured run: max(1.5, ratios) = **1.5**.
- SON S_T = 1.5 · (2 chunks · P1T · C_T / C1 + C_T): pass 1 from P1T (153.0 s for one 40M-row chunk at
  Base, local min_count 36,000), pass 2 taken as one in-core pass at T.
- X1 = S3's estimate (replaced by S3's measured wall once known).
- One null permutation = 1.5 · (in-core model on the GPU null's mean K distribution + 2 · C3 load) = 12.5 min.
- C3G (C3 with `use_generator_pruning=True`) ≤ C3's wall. R1, A1: 0.1 h each.
- Repetitions: every timed engine run under 15 minutes runs 3 times (`<ID>`, `<ID>_r2`, `<ID>_r3`;
  median, [min, max]); a run of 15 minutes or more runs once and is flagged. The oracle, compares,
  checks and null permutations run once. `scripts/if_short.sh` skips a repetition when the first run
  took ≥ 15 minutes.

| run | estimate (h, incl. repetitions) | basis |
|---|---|---|
| C1 / C2 / C3 | 0.23 / 1.77 / 7.11 | measured (C1 × 3) |
| S1 / S2 / S3 | 0.73 / 5.62 / 22.55 | model (S1 × 3) |
| R1 | 0.30 | 3 × 0.1 h |
| N5 / N100 | 1.04 / 19.74 | 5 and 95 permutations × 12.5 min |
| C3G | 7.11 | ≤ C3 |
| C4 Blitz | 80.1 | model |
| C5 Ultra | 421.0 | model |
| C6 Opus | 709.6 | model |
| M4 | 1,236.8 | model |
| X1 | 22.55 | = S3 |

## Selection

- **Minimum core** (G0 done; C1–C3 done): repetitions of G0 and C1, R1 × 3, S1 (× 3 if under 15 min),
  S2, S3, and the compares: 29.4 h ≤ 86.9 h. It fits, so no stop under Budget.
- **Additions, in the brief's order.** Reading of "add runs in this order while the summed estimates
  fit": each run in the order is added when its estimate fits what remains, and a run that does not fit
  is dropped (`not-run (budget)`) before the next one is considered; the order is never changed.
  1. N5 (1.04 h): fits → added. Remaining after core + N5: 56.4 h.
  2. **Count-inference fallback (decided now, before C4):** C6 (709.6 h) does not fit with defaults, so
     C3G runs (≤ 7.11 h; remaining after it: 49.3 h). If C3G's tables equal C3's (signature and full
     compare) and its wall is ≥ 10% lower, C4–C6 and M4 use `use_generator_pruning=True` and are
     re-estimated with C3G's per-level costs; otherwise they keep the defaults.
  3. After C3G, C4 → C6 with A1 → C5 → N100 → M4 → X1 are decided in that order against the remaining
     budget and recorded under "Re-estimates". With the default estimates: C4 (80.1 h) does not fit,
     C6 (709.6 h) and A1 do not fit, C5 (421.0 h) does not fit, N100 (19.74 h) fits (29.6 h left),
     M4 (1,236.8 h) does not fit, X1 (22.55 h) fits (7.0 h left).
  4. A1 belongs to C6 (Opus: the K=22 itemset and P-026) and C4 (Blitz: P-030, P-031). If C4 runs and
     C6 does not, the Blitz decode (`decode_intermediate_k.py`) runs and the K=22 part is `not-run`.
- **Re-estimate after every run** (brief Phase 2 step 4): a run past 2× its estimate continues while the
  new estimate fits what remains; otherwise it is killed, its per-level log kept, and it is marked
  `not-run (budget)`.

## Queue

Lines starting with `#` are not executed. C4, C5, C6, A1, N100, M4 and X1 are listed and enabled (or left
dropped) by the decision after C3G.

```queue
!G0_incore | ./run_py.sh cpu_mine.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/G0/incore --min-count 50 --min-support 5e-05 --head 1000000 --label G0_incore
!G0_son | ./run_py.sh cpu_son.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/G0/son --min-count 50 --min-support 5e-05 --chunk-size 250000 --chunks 4 --head 1000000 --label G0_son
!G0_ea | ./run_py.sh ea_oracle.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/G0/ea --min-count 50 --head 1000000 --label G0_ea
!G0_cmp_incore | ./run_py.sh compare.py ../data/mined/G0/incore/parquet ../data/mined/G0/ea/parquet --n 1000000 --threshold 50 --data ../data/extract_2026_01/transactions_214m_base_multi.parquet --head 1000000 --out ../data/mined/G0/cmp_incore_ea.json --label G0_incore_vs_ea
!G0_cmp_son | ./run_py.sh compare.py ../data/mined/G0/son/parquet ../data/mined/G0/ea/parquet --n 1000000 --threshold 50 --data ../data/extract_2026_01/transactions_214m_base_multi.parquet --head 1000000 --out ../data/mined/G0/cmp_son_ea.json --label G0_son_vs_ea
!G0_check | ./run_py.sh g0_check.py ../data/mined/G0 ../../20260902T0000Z/phase3/2026_01/validate_row_split.log ../data/phase1_verification.json --out ../data/mined/G0/G0.json
C1 | ./run_py.sh cpu_mine.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/C1 --min-count 76891 --min-support 1e-3 --label C1
C2 | ./run_py.sh cpu_mine.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/C2 --min-count 7690 --min-support 1e-4 --label C2
C3 | ./run_py.sh cpu_mine.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/C3 --min-count 769 --min-support 1e-5 --label C3
P1T | ./run_py.sh cpu_mine.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/P1T --min-count 36000 --min-support 9e-4 --head 40000000 --label P1T
C1_cmp | ./run_py.sh compare.py ../data/mined/C1/parquet ../../20260902T0000Z/phase3/2026_01/opus/parquet --n 76890945 --ref-min-count 76891 --threshold 76891 --data ../data/extract_2026_01/transactions_214m_base_multi.parquet --translate ../../20260902T0000Z/phase2/extract_2026_01/item_mapping_214m_base.parquet ../data/extract_2026_01/item_mapping_214m_base.parquet --out ../data/mined/C1/cmp_gpu_opus.json --label C1_vs_GPU_opus_filtered
C2_cmp | ./run_py.sh compare.py ../data/mined/C2/parquet ../../20260902T0000Z/phase3/2026_01/opus/parquet --n 76890945 --ref-min-count 7690 --threshold 7690 --data ../data/extract_2026_01/transactions_214m_base_multi.parquet --translate ../../20260902T0000Z/phase2/extract_2026_01/item_mapping_214m_base.parquet ../data/extract_2026_01/item_mapping_214m_base.parquet --out ../data/mined/C2/cmp_gpu_opus.json --label C2_vs_GPU_opus_filtered
C3_cmp | ./run_py.sh compare.py ../data/mined/C3/parquet ../../20260902T0000Z/phase3/2026_01/opus/parquet --n 76890945 --ref-min-count 769 --threshold 769 --data ../data/extract_2026_01/transactions_214m_base_multi.parquet --translate ../../20260902T0000Z/phase2/extract_2026_01/item_mapping_214m_base.parquet ../data/extract_2026_01/item_mapping_214m_base.parquet --out ../data/mined/C3/cmp_gpu_opus.json --label C3_vs_GPU_opus_filtered
G0_incore_r2 | ./run_py.sh cpu_mine.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/G0/incore_r2 --min-count 50 --min-support 5e-05 --head 1000000 --label G0_incore_r2
G0_incore_r3 | ./run_py.sh cpu_mine.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/G0/incore_r3 --min-count 50 --min-support 5e-05 --head 1000000 --label G0_incore_r3
G0_son_r2 | ./run_py.sh cpu_son.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/G0/son_r2 --min-count 50 --min-support 5e-05 --chunk-size 250000 --chunks 4 --head 1000000 --label G0_son_r2
G0_son_r3 | ./run_py.sh cpu_son.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/G0/son_r3 --min-count 50 --min-support 5e-05 --chunk-size 250000 --chunks 4 --head 1000000 --label G0_son_r3
C1_r2 | ./run_py.sh cpu_mine.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/C1_r2 --min-count 76891 --min-support 1e-3 --label C1_r2
C1_r3 | ./run_py.sh cpu_mine.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/C1_r3 --min-count 76891 --min-support 1e-3 --label C1_r3
R1 | ./run_py.sh rules.py ../data/mined/C1/parquet ../data/mined/R1 --n 76890945 --min-confidence 0.5 --label R1
R1_r2 | ./run_py.sh rules.py ../data/mined/C1/parquet ../data/mined/R1_r2 --n 76890945 --min-confidence 0.5 --label R1_r2
R1_r3 | ./run_py.sh rules.py ../data/mined/C1/parquet ../data/mined/R1_r3 --n 76890945 --min-confidence 0.5 --label R1_r3
S1 | ./run_py.sh cpu_son.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/S1 --min-count 76891 --min-support 1e-3 --chunk-size 40000000 --chunks 2 --label S1
S1_cmp_C1 | ./run_py.sh compare.py ../data/mined/S1/parquet ../data/mined/C1/parquet --n 76890945 --threshold 76891 --data ../data/extract_2026_01/transactions_214m_base_multi.parquet --out ../data/mined/S1/cmp_C1.json --label S1_vs_C1
S1_cmp_gpu | ./run_py.sh compare.py ../data/mined/S1/parquet ../../20260902T0000Z/phase3/2026_01/exp/son_base_itemsets.parquet --n 76890945 --threshold 76891 --data ../data/extract_2026_01/transactions_214m_base_multi.parquet --translate ../../20260902T0000Z/phase2/extract_2026_01/item_mapping_214m_base.parquet ../data/extract_2026_01/item_mapping_214m_base.parquet --out ../data/mined/S1/cmp_gpu_son.json --label S1_vs_GPU_son_base
S1_r2 | ./if_short.sh ../data/mined/S1/meta.json ./run_py.sh cpu_son.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/S1_r2 --min-count 76891 --min-support 1e-3 --chunk-size 40000000 --chunks 2 --label S1_r2
S1_r3 | ./if_short.sh ../data/mined/S1/meta.json ./run_py.sh cpu_son.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/S1_r3 --min-count 76891 --min-support 1e-3 --chunk-size 40000000 --chunks 2 --label S1_r3
S2 | ./run_py.sh cpu_son.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/S2 --min-count 7690 --min-support 1e-4 --chunk-size 40000000 --chunks 2 --label S2
S2_cmp_C2 | ./run_py.sh compare.py ../data/mined/S2/parquet ../data/mined/C2/parquet --n 76890945 --threshold 7690 --data ../data/extract_2026_01/transactions_214m_base_multi.parquet --out ../data/mined/S2/cmp_C2.json --label S2_vs_C2
S2_cmp_gpu | ./run_py.sh compare.py ../data/mined/S2/parquet ../../20260902T0000Z/phase3/2026_01/exp/son_super_itemsets.parquet --n 76890945 --threshold 7690 --data ../data/extract_2026_01/transactions_214m_base_multi.parquet --translate ../../20260902T0000Z/phase2/extract_2026_01/item_mapping_214m_base.parquet ../data/extract_2026_01/item_mapping_214m_base.parquet --out ../data/mined/S2/cmp_gpu_son.json --label S2_vs_GPU_son_super
S3 | ./run_py.sh cpu_son.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/S3 --min-count 769 --min-support 1e-5 --chunk-size 40000000 --chunks 2 --label S3
S3_cmp_C3 | ./run_py.sh compare.py ../data/mined/S3/parquet ../data/mined/C3/parquet --n 76890945 --threshold 769 --data ../data/extract_2026_01/transactions_214m_base_multi.parquet --out ../data/mined/S3/cmp_C3.json --label S3_vs_C3
S3_cmp_gpu | ./run_py.sh compare.py ../data/mined/S3/parquet ../../20260902T0000Z/phase3/2026_01/opus/parquet --n 76890945 --ref-min-count 769 --threshold 769 --data ../data/extract_2026_01/transactions_214m_base_multi.parquet --translate ../../20260902T0000Z/phase2/extract_2026_01/item_mapping_214m_base.parquet ../data/extract_2026_01/item_mapping_214m_base.parquet --out ../data/mined/S3/cmp_gpu_opus.json --label S3_vs_GPU_opus_filtered
N5 | ./run_py.sh null_model.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/N --runs 100 --perms 0-4 --min-count 769 --seed 42 --check-seeds ../../20260902T0000Z/phase3/2026_01/exp_null100/experiment_null_model_20260902_115339.json
N5_verdict | ./run_py.sh null_verdict.py ../data/mined/N ../data/mined/C3/meta.json ../../20260902T0000Z/phase3/2026_01/exp_null100/experiment_null_model_20260902_115339.json --mode n5 --out ../data/mined/N/verdict_n5.json
C3G | ./run_py.sh cpu_mine.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/C3G --min-count 769 --min-support 1e-5 --use-generator-pruning --label C3G
C3G_cmp_C3 | ./run_py.sh compare.py ../data/mined/C3G/parquet ../data/mined/C3/parquet --n 76890945 --threshold 769 --data ../data/extract_2026_01/transactions_214m_base_multi.parquet --out ../data/mined/C3G/cmp_C3.json --label C3G_vs_C3
C4 | ./run_py.sh cpu_mine.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/C4 --min-count 77 --min-support 1e-6 --use-generator-pruning --label C4
C4_cmp | ./run_py.sh compare.py ../data/mined/C4/parquet ../../20260902T0000Z/phase3/2026_01/opus/parquet --n 76890945 --ref-min-count 77 --threshold 77 --data ../data/extract_2026_01/transactions_214m_base_multi.parquet --translate ../../20260902T0000Z/phase2/extract_2026_01/item_mapping_214m_base.parquet ../data/extract_2026_01/item_mapping_214m_base.parquet --out ../data/mined/C4/cmp_gpu_opus.json --label C4_vs_GPU_opus_filtered
C4_cmp_blitz | ./run_py.sh compare.py ../data/mined/C4/parquet ../../20260902T0000Z/phase3/2026_01/blitz/parquet --n 76890945 --threshold 77 --data ../data/extract_2026_01/transactions_214m_base_multi.parquet --translate ../../20260902T0000Z/phase2/extract_2026_01/item_mapping_214m_base.parquet ../data/extract_2026_01/item_mapping_214m_base.parquet --out ../data/mined/C4/cmp_gpu_blitz.json --label C4_vs_GPU_blitz
C6 | ./run_py.sh cpu_mine.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/C6 --min-count 8 --min-support 1e-7 --use-generator-pruning --label C6
C6_cmp | ./run_py.sh compare.py ../data/mined/C6/parquet ../../20260902T0000Z/phase3/2026_01/opus/parquet --n 76890945 --ref-min-count 8 --threshold 8 --data ../data/extract_2026_01/transactions_214m_base_multi.parquet --translate ../../20260902T0000Z/phase2/extract_2026_01/item_mapping_214m_base.parquet ../data/extract_2026_01/item_mapping_214m_base.parquet --out ../data/mined/C6/cmp_gpu_opus.json --label C6_vs_GPU_opus
A1_k22 | ./a1_k22.sh ..
A1_k22_check | ./run_py.sh a1_check.py "$(ls ../data/mined/A1/k22/analysis_deepest_itemset_*.json | tail -1)" ../../20260902T0000Z/phase3/2026_01/exp/analysis_deepest_itemset_20260902_062222.json --out ../data/mined/A1/k22_check.json
A1_p026_c6 | ./run_py.sh triple_support.py ../data/mined/C6/parquet ../data/extract_2026_01/item_mapping_214m_base.parquet PF00270 GO:0003724 GO:0045087 --out ../data/mined/A1/p026_c6.json
A1_decode | mkdir -p ../data/mined/A1 && ./run_py.sh decode_intermediate_k.py --json ../data/mined/A1/intermediate_k_decode.json
A1_p026 | ./run_py.sh triple_support.py ../data/mined/C4/parquet ../data/extract_2026_01/item_mapping_214m_base.parquet PF00270 GO:0003724 GO:0045087 --out ../data/mined/A1/p026.json
C5 | ./run_py.sh cpu_mine.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/C5 --min-count 16 --min-support 2e-7 --use-generator-pruning --label C5
C5_cmp | ./run_py.sh compare.py ../data/mined/C5/parquet ../../20260902T0000Z/phase3/2026_01/opus/parquet --n 76890945 --ref-min-count 16 --threshold 16 --data ../data/extract_2026_01/transactions_214m_base_multi.parquet --translate ../../20260902T0000Z/phase2/extract_2026_01/item_mapping_214m_base.parquet ../data/extract_2026_01/item_mapping_214m_base.parquet --out ../data/mined/C5/cmp_gpu_opus.json --label C5_vs_GPU_opus_filtered
N100 | ./run_py.sh null_model.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/N --runs 100 --perms 5-99 --min-count 769 --seed 42 --check-seeds ../../20260902T0000Z/phase3/2026_01/exp_null100/experiment_null_model_20260902_115339.json
N100_verdict | ./run_py.sh null_verdict.py ../data/mined/N ../data/mined/C3/meta.json ../../20260902T0000Z/phase3/2026_01/exp_null100/experiment_null_model_20260902_115339.json --mode n100 --out ../data/mined/N/verdict_n100.json
M4 | ./run_py.sh cpu_mine.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/M4 --min-count 4 --use-generator-pruning --label M4
M4_cmp | ./run_py.sh compare.py ../data/mined/M4/parquet ../../20260902T0000Z/phase3/2026_01/minc4/parquet --n 76890945 --threshold 4 --data ../data/extract_2026_01/transactions_214m_base_multi.parquet --translate ../../20260902T0000Z/phase2/extract_2026_01/item_mapping_214m_base.parquet ../data/extract_2026_01/item_mapping_214m_base.parquet --out ../data/mined/M4/cmp_gpu_minc4.json --label M4_vs_GPU_minc4
X1 | ./run_py.sh cpu_son.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/X1 --min-count 769 --min-support 1e-5 --chunk-size 19222737 --chunks 4 --label X1
X1_cmp_C3 | ./run_py.sh compare.py ../data/mined/X1/parquet ../data/mined/C3/parquet --n 76890945 --threshold 769 --data ../data/extract_2026_01/transactions_214m_base_multi.parquet --out ../data/mined/X1/cmp_C3.json --label X1_vs_C3
P1T_r2 | ./run_py.sh cpu_mine.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/P1T_r2 --min-count 36000 --min-support 9e-4 --head 40000000 --label P1T_r2
P1T_r3 | ./run_py.sh cpu_mine.py ../data/extract_2026_01/transactions_214m_base_multi.parquet ../data/mined/P1T_r3 --min-count 36000 --min-support 9e-4 --head 40000000 --label P1T_r3
```

## Re-estimates

- 2026-10-20T18:21Z, after S2 (measured 7,104.9 s vs 20,229 s estimated): on the CPU route SON's pass 2 has
  nothing to count (every pass-1 candidate is exact within the partition bound), so S_T ≈ 1.11 · C_T
  (S1/C1 = 1.06, S2/C2 = 1.11). S3 re-estimated at 1.5 · 1.11 · 25,604 s = 11.9 h (was 22.55 h); X1 the same
  until S3 is measured. Mining used so far 11.46 h (sum of finished queue runs in `logs/queue_*.log`);
  remaining 84.5 h. No change to the selection.
- 2026-10-21T02:32Z, after S3 (measured 29,451.8 s vs 11.9 h re-estimated): minimum core complete. Mining used
  19.64 h; remaining 76.4 h. X1 (S3 with 4 chunks, lower local threshold) re-estimated at 1.5 · S3 = 12.3 h.
  No change to the selection.
- 2026-10-21T03:46Z, after N5 (measured 4,426 s vs 1.04 h estimated): 14.8 min per permutation. N100 (95 more)
  re-estimated at 95 × 14.8 min = 23.4 h (measured, no safety factor). Peak RSS 24.6 GB per permutation (of 30.65 GiB).
  Mining used 20.87 h; remaining 75.1 h. No change to the selection.
- 2026-10-21T05:20Z, decision after C3G (`data/mined/C3G/cmp_C3.json`: identical, signature 3e025003… = C3's;
  wall 5,476.1 s vs 25,602.7 s, 78.6 % lower ≥ 10 %): **C4–C6 and M4 use `use_generator_pruning=True`.**
  Re-estimated from C3G's per-level costs (`data/sizing/sizing_C3G.json`, safety 1.5): C4 12.81 h, C5 61.72 h,
  C6 137.85 h, M4 246.31 h. Mining used 22.40 h; remaining 73.6 h (the 120 h guard leaves 120 − 35.9 − 8 = 76.1 h).
  In the brief's order:
  1. C4 (12.81 h) fits → queued with its compares (vs the Opus tables filtered at 77 and vs `blitz/parquet`);
     remaining 60.8 h.
  2. C6 (137.85 h) does not fit → `not-run (budget)`. A1 with it: the K=22 analysis (`analyze_k22_proteins.py`,
     P-007..P-009) is not-run; the Blitz decode (P-030, P-031) runs on C4 (`scripts/decode_intermediate_k.py`, a
     copy with paths changed only: `scripts/decode_intermediate_k.py.diff`), and P-026 (count 1,234 ≥ 77) is read
     from C4's K=3 table (`scripts/triple_support.py`) instead of Opus's. 0.1 h; remaining 60.7 h.
  3. C5 (61.72 h) does not fit 60.7 h by this estimate. C4 is the measured run closest to C5's regime, so C5 is
     decided again from C4's per-level costs once C4 is measured; then N100, M4 and X1 in that order.
- 2026-10-21T08:20Z, after C4 (measured 10,443.8 s, 2.90 h; the C3G-based model had predicted 8.54 h, ratio 0.34;
  identical to the GPU Blitz tables): C4 is the measured run closest to the deeper thresholds, so C5, C6 and M4 are
  re-estimated from C4's per-level costs (`data/sizing/sizing_C4.json`, safety 1.5): C6 22.18 h, C5 15.35 h,
  M4 34.34 h. Mining used 25.30 h; remaining 70.7 h (the guard leaves 120 − 38.9 − 8 = 73.2 h). The 05:20Z drop of C6
  rested on the C3G-based estimate; with C4 measured, the order is applied again from C6:
  1. C6 (22.18 h) with A1 (K=22 analysis 0.1 h; P-026 also read from C6's K=3 table, as the brief says) fits →
     queued; remaining 48.4 h.
  2. C5 (15.35 h) fits → queued; remaining 33.1 h.
  3. N100 (95 × 14.8 min = 23.4 h, measured) fits → queued; remaining 9.7 h.
  4. M4 (34.34 h) does not fit → `not-run (budget)`.
  5. X1 (12.3 h) does not fit → `not-run (budget)` unless the runs above finish early enough (checked after each).
  The Blitz decode of A1 (P-030, P-031) is identical to `phase4/citation_audit/intermediate_k_decode.json`
  (levels, maxima, member sets, patterns).
- 2026-10-21T13:20Z, after C6 (measured 17,670.1 s, 4.91 h vs 22.18 h estimated; identical to the GPU Opus tables)
  and A1 (identical): the C4-based model overestimated C6 3×, and a C6-based model underestimates C4 1.8×:
  count inference saves more the lower the threshold. Each remaining in-core run is therefore estimated from the
  measured run nearest its threshold, C6 (min_count 8) for C5 (16) and M4 (4) (`data/sizing/sizing_C6.json`,
  safety 1.5): C5 5.20 h, M4 11.16 h. Mining used 30.25 h; remaining 65.75 h (guard: 120 − 43.9 − 8 = 68.1 h).
  C5 (running), N100 (23.4 h), M4 (11.16 h) and X1 (12.3 h) sum to 52.1 h and fit, in that order: M4 and X1 are
  re-admitted to the queue (they start when the running queue has finished N100; the 2× rule applies to each).
- 2026-10-21T17:28Z, after C5 (measured 15,199.8 s, 4.22 h vs 5.20 h estimated; identical): mining used 34.48 h;
  remaining 61.5 h. N100 (23.4 h), M4 (11.16 h) and X1 (12.3 h) still fit (46.9 h). No change.
- 2026-10-21T18:45Z, after the interim review: (1) the 13:20Z re-admission of M4 and X1 only edited PLAN.md, and the
  running queue had read its list at 08:17Z, so a restart is chained: `scripts/after_pid.sh 629549 scripts/queue.sh .`
  (PID 704457, log `logs/queue_20261021T1838Z.log`) starts the queue again when the current one exits after
  N100_verdict; it skips every done run and then runs M4, M4_cmp, X1, X1_cmp_C3. (2) The safety factor is re-derived
  for the C6-based model from the other count-inference runs predicted by it (`data/sizing/sizing_C6_v2.json`):
  measured / predicted = 1.22 (C5, min_count 16), 2.74 (C4, 77), 6.00 (C3G, 769). The miss grows with the distance
  from C6's threshold (count inference saves more at lower thresholds), and M4 (4) lies on the far side of C6 from all
  three. M4 is estimated with the factor 2.74 (the largest among the count-inference runs at or below Blitz, as the
  review proposed): 11.16 / 1.5 · 2.74 = 20.4 h, so its 2× limit is 40.8 h or the remaining budget, whichever comes
  first. N100 (23.4 h) + M4 (20.4 h) + X1 (12.3 h) = 56.1 h ≤ 61.5 h: the selection stands. With 6.00, M4 would be
  44.7 h and would not fit; this is recorded as an open item for Et with D-011.
- 2026-10-21T18:50Z: Et approved the D-011 reading, the re-admissions and the factor 2.74 for M4 (deviations D-014).
- 2026-10-22T16:57Z, after N100 (measured 23.45 h vs 23.4 h; consistent): mining used 57.94 h; remaining 38.06 h.
  M4 (20.4 h) + X1 (12.3 h) = 32.7 h fit; the guard leaves 120 − 71.6 − 8 = 40.4 h. M4 started 16:55Z (queue PID
  704457, log `logs/queue_20261021T1838Z.log`). M4's 2× limit: 40.8 h, or earlier if its new estimate no longer fits.
- 2026-10-22T22:47Z, after M4 (measured 20,653.4 s, 5.74 h vs 20.4 h; 48,007,493 itemsets, K 22): M4_cmp ran out of
  memory (D-015); `compare.py` now works level by level, and M4_cmp re-runs right after X1 (`scripts/after_pid.sh
  704457 scripts/queue.sh .`, PID 856813, log `logs/queue_20261022T2246Z.log`). X1 started 2026-10-22T22:43Z.
  Mining used 63.7 h (M4 5.74 h added); remaining 32.3 h; X1 (12.3 h) fits.
- 2026-10-23T08:50Z, end of Phase 3: X1 (measured 16,322.1 s, 4.53 h vs 12.3 h) and M4_cmp done; every run of the queue
  completed. Mining used 68.32 h of 96 h (sum of finished queue runs in `logs/queue_*.log`, the failed first M4 compare
  included). No run of the plan was dropped.
- 2026-10-23T09:01Z, final review: P1T (153 s) had run once although the repetition rule covers it; P1T_r2 and
  P1T_r3 are added at the end of the queue (about 5 minutes).

## Not run

None: every run of the plan completed (C6, M4 and X1 after re-admission, PLAN.md Re-estimates; Et's approval D-014).
