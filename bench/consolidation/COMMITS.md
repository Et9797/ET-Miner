# The commits after `93224ef` (made on 2026-09-28 on the owner's word)

One commit per decision point, in this order. Each block gives the files and
the message. Shared files (`_env.py`, `bench/runner.py`, `bench/README.md`,
`CHANGELOG.md`, `row_split.py`) were committed in intermediate versions so
that each commit carries only its own change; the lists below name where a
file's final text landed.

## 1. The 2-GPU measurements (DP6 data, the DP10 tables, the NCCL hang evidence)

```
git add bench/results/2026-09-28-consolidation-2gpu
git commit -F - <<'EOF'
consolidation: DP6 and DP10 measured on two GPUs

The 2-GPU rows of the Phase 2 matrix (C2, Asplit2/Bsplit2, E2 and the
DP10 arms) at e4bb3ae, on 2x RTX A4000 16 GB: 82 config-reps, 72 ok,
3.44 GPU-hours, one signature per regime, identical to Phase 2.

DP6: C2-shared wins oom2 (K>=2 3.40 s vs Asplit2 37.25 / Bsplit2 36.39),
sk2ml2 (17.22 vs 225.08 / 224.67) and sk2ml3 (unopposed); the splits win
nothing. DP10: every arm ties its baseline (rule 2 finds no winner).

nccl-hang/: on this box NCCL's P2P transport through the AM4 host bridge
hangs about one run in three at the first ncclReduce (rank 1 completes,
rank 0 never does, traced at e4bb3ae and HEAD); every row here was
measured with NCCL_P2P_DISABLE=1, which gave 0 hangs in 30 runs.
EOF
```

## 2. DP10: the nnz row balance

```
git add src/et_miner/_env.py src/et_miner/gpu/csr_bitvec.py tests/test_balance_split.py
git commit -F - <<'EOF'
consolidation: DP10 nnz row balance removed

ET_MINER_ROW_BALANCE=nnz and balance="nnz" raise a ValueError naming
the rows split; "rows" stays a no-op. On two GPUs the nnz cut tied the
rows split in both regimes built for it (skewed_rows 1.07 vs 1.07 s,
deep_sparse_large 18.14 vs 17.91 s, medians of 3), so rule 5 keeps the
smaller code. The searchsorted cut and its feasibility check are gone.
EOF
```

(`bench/runner.py`, `bench/README.md` and `CHANGELOG.md` carry both DP10
changes; they go with commit 3.)

## 3. DP10: one survivor filter

```
git add src/et_miner/gpu/kernels/filter.py src/et_miner/gpu/kernels/loader.py \
        src/et_miner/gpu/kernels/__init__.py src/et_miner/gpu/kernels/_src/compact_threshold.cu \
        src/et_miner/gpu/row_split_chunks.py src/et_miner/gpu/row_split.py \
        src/et_miner/gpu/memory_budget.py bench/selfcheck.py bench/runner.py bench/report.py \
        bench/README.md CHANGELOG.md tests/test_threshold_filter.py tests/test_compact_threshold.py \
        tests/test_chunked_dense.py tests/test_tier_equivalence.py tests/test_runtime_dependencies.py
git commit -F - <<'EOF'
consolidation: DP10 one survivor filter, the compact kernel and cpu path removed

The dense survivor filter is the sliced cp.nonzero path
(gpu/kernels/filter.py::threshold_filter, with a per-slice host fallback
when a slice does not fit the device); the compact_threshold kernel, the
whole-array CPU path, ET_MINER_FILTER_IMPL and compact_threshold_filter's
impl= are gone. On two GPUs the three implementations tied in both
regimes built for them (stress_k2 K<=2 20.00 / 19.05 / 19.99 s,
deep_sparse_large 17.91 / 18.49 / 18.83 s), so rule 5 keeps the least
code. Six registered kernels remain; the selfcheck launches each one.
Measured per 64M-element slice on an A4000: 94 ms, 1.1 B/element of
extra VRAM at a 1% pass rate, 13 B/element with every element surviving,
under the chunk budget's 1 GiB margin.
EOF
```

Note: `bench/README.md` and `CHANGELOG.md` also contain the peer-copy fix
text of commit 4; either accept that overlap or stage those two files with
`git add -p`.

## 4. The peer-copy probe (finding 21)

```
git add src/et_miner/gpu/nccl.py tests/test_peer_copy_probe.py
git commit -F - <<'EOF'
gpu: probe device-to-device copies; stage through the host when they do not land

On a box whose PCIe P2P drops writes (2x RTX A4000 behind a Ryzen AM4
host bridge) cudaMemcpy, cudaMemcpyPeer and CuPy assignment from GPU 1
to GPU 0 return success with the destination untouched: the staged D2D
reduce summed only GPU 0's shard, bitvecs= sharded across two GPUs mined
stale pool memory, and NCCL's P2P transport hung at the first collective.
gpu/nccl.py::peer_copy_works probes each device pair once (4 KiB
pattern); a failed probe routes the staged reduce and the prebuilt-bitvec
shards through host memory and starts NCCL with NCCL_P2P_DISABLE=1
unless the caller set it. test_nccl_fallback_forced and the device
affinity tests now pass on that box.
EOF
```

## 5. The verify matrix on the final tree, and the reports

```
git add bench/results/2026-09-28-consolidation-verify-2gpu bench/consolidation_matrix.py \
        bench/results/2026-09-27-consolidation/FINDINGS.md bench/consolidation/REPORT.md \
        bench/consolidation/HANDOFF.md bench/consolidation/COMMITS.md .claude/handoffs
git commit -F - <<'EOF'
consolidation: verify matrix on two A4000s; DP6/DP10 recorded; handoff

The verify matrix at the final tree on the second box (78 config-reps,
all ok, every regime's signature equal to Phase 2), with a two-GPU
stress_k2 K=3 row added to the matrix: 306 s on two A4000s against
593 s fused on one and 785 s at e4bb3ae. FINDINGS.md and REPORT.md
record DP6 and DP10, the second box, the NCCL hang and the P2P
corruption incident; HANDOFF.md lists the open work.
EOF
```

Every message ends with the trailers the session was given:

```
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015GktjkfBjZXPt91ZCz8K47
```
