# NCCL hang on the 2× RTX A4000 box (2026-09-28)

Every row of `bench/results/2026-09-28-consolidation-2gpu/` was measured with
`NCCL_P2P_DISABLE=1` in the runner's environment (inherited by every child;
`../NCCL_ENV.txt`). Why:

- On this box the two A4000s reach each other P2P through the CPU's host
  bridge (`nvidia-smi topo -m`: PHB; `topo.txt`) and NCCL 2.31 picks the
  `P2P/direct pointer` transport (`nccl-debug-hung-run.txt`).
- With that transport, about one two-GPU run in three hangs at the first
  collective, the K=2 `ncclReduce` to root 0. The 3060 box of the Phase 2
  campaign had no P2P at all (NCCL used SHM there) and never hung.
- Traced repro (`repro_nnz2.py`, the campaign child's warm-up: `smoke`
  head(20000), min_support 0.02, max_length 3, 2 GPUs): both ranks enqueue the
  reduce with the same count (2,211 int32), each on its own device and
  communicator; rank 1's kernel completes in 51 ms; rank 0's never completes
  (`trace-e4bb3ae-rows-1.txt`, `trace-e4bb3ae-nnz-2.txt`, `trace-HEAD-rows-1.txt`).
  Same shape at the campaign revision `e4bb3ae` and at HEAD, with
  `ET_MINER_ROW_BALANCE=rows` and `=nnz`. No NCCL warning is logged.
- Thread dump of an untraced hang (`repro-e4bb3ae-rows-1.txt`, faulthandler):
  main thread in `_nccl_reduce_sum_to_root` waiting on rank 0's future, rank 0
  in `stream.synchronize()`, every other worker idle.
- Rates (`repro-summary.txt`, `trace-summary.txt`): P2P on, 4 hangs in 14 runs
  at `e4bb3ae` (rows 2/8, nnz 2/6), 1 in 1 at HEAD. P2P off
  (`p2poff-summary.txt`): 0 hangs in 30 runs (`e4bb3ae` rows 10, nnz 10; HEAD
  rows 10).

`aborted-p2p-attempt/` holds the first campaign attempt, with P2P on: four ok
rows (the DP10 filter arms) and `skew-C2-legacy-balance-nnz#r0`, which hung in
the warm-up and timed out at 1,800 s. Those rows are not used: the whole
directory was re-measured under one transport.

The code path is the same either way (NCCL ring reduce); only NCCL's transport
between the two devices changes. `ET_MINER_DISABLE_NCCL=1` (the staged D2D
fallback) was not used, so the measured path is the branch's default.
