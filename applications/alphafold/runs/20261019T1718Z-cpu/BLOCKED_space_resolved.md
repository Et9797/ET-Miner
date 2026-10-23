# BLOCKED — stop condition: Space (RESOLVED)

Resolved 2026-10-19: Et: "Werk met het huidige geheugen" (work with the current memory).
The campaign continues on this box; see `deviations.md` D-001.

Campaign: `CPU_REPRO_GOAL.md`. Stopped at Phase 0, step 1 (box audit), 2026-10-19.

## Cause

Phase 0 step 1 requires >= 200 GB free disk and >= 64 GiB RAM and says to stop under
**Space** if they can't be met. Disk is met; RAM is not: this box has 30.65 GiB.

## Evidence

All values are in `env.json` (this directory).

| need | required | this box | met |
|---|---|---|---|
| RAM | >= 64 GiB | 30.65 GiB (`MemTotal: 32134476 kB`), 2 GiB swap | no |
| cgroup memory limit | — | `max` at every level of the user cgroup chain | — |
| free disk on `/` | >= 200 GB | 511.9 GB | yes |

For reference, the GPU run's box had ~69.6 GiB cgroup RAM (`RESULTS.md` line 9).
Box: AMD Ryzen 5 4600G (6 cores / 12 threads), Ubuntu 22.04.5, no NVIDIA device
(`/dev/nvidia*` absent, `nvidia-smi` not installed).

Nothing else ran: no engine worktree, no BigQuery read, no download, no mining.

## What Et must do

One of:
1. Run the campaign on a box with >= 64 GiB RAM (and >= 200 GB free disk).
2. Amend the RAM need in `CPU_REPRO_GOAL.md` Phase 0 step 1 if 30.65 GiB is acceptable
   for this box. Runs that do not fit would then fall under the per-run **Space** stop.

Also decide on `runs/20261012T1559Z-cpu/`: an untracked earlier attempt (scripts, logs,
`data/uniprot_trembl_2026_01.reduced.dat.gz`, `data/metadata/`). It was left untouched as
the brief requires; the brief as written rebuilds every input under the new `<RUN>/data/`.

## How to resume

After the box or the brief changes: start a fresh `/goal` on branch
`alphafold-cpu-reproduction`, begin at Phase 0 step 1 and re-run the audit into
`env.json` (this run directory may be reused or a new `runs/<UTC>-cpu/` created).
