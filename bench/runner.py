"""Campaign driver: runs the benchmark matrix, one subprocess per config.

Each config runs in a fresh process (fresh CUDA context — OOM and pool
fragmentation cannot leak between runs) inside its own process group, so a
timeout kill takes the nvidia-smi sampler down with it. Results append to
<out>/raw.jsonl; re-invoking skips configs already recorded (resume).

Cross-config refutation: every config in the same equivalence group
(preset, max_length, min_support, two_phase) must report identical result
signatures — kernel variant, filter impl, GPU count, NCCL mode, row
balance, and density mode are all result-preserving by contract. Any
divergence fails the campaign.

Usage: python bench/runner.py --mode smoke|full [--out DIR] [--max-hours H]
       [--only SUBSTR] [--skip SUBSTR]
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "bench" / "results" / "campaign"
CHILD = REPO / "bench" / "child_run.py"


def _gpu_count() -> int:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


def _cfg(id_, preset, *, variant="legacy", filter_impl=None, n_gpus=2, balance=None,
         disable_nccl=False, sparse_from_k=None, max_length=None, min_support=None,
         two_phase=False, rep=0, timeout_s=1800):
    env = {"ET_MINER_KERNEL_VARIANT": variant}
    if n_gpus == 1:
        # Make "1 GPU" mean it: dispatch auto-detects physical devices and
        # would otherwise route big levels to the pair-split multi-GPU path,
        # silently muddying the 1g-vs-2g benchmark axis (observed on-box).
        env["CUDA_VISIBLE_DEVICES"] = "0"
    if filter_impl:
        env["ET_MINER_FILTER_IMPL"] = filter_impl
    if balance:
        env["ET_MINER_ROW_BALANCE"] = balance
    if disable_nccl:
        env["ET_MINER_DISABLE_NCCL"] = "1"
    return {
        "id": f"{id_}#r{rep}",
        "preset": preset,
        "env": env,
        "n_gpus": n_gpus,
        "sparse_from_k": sparse_from_k,
        "max_length": max_length,
        "min_support": min_support,
        "two_phase": two_phase,
        "timeout_s": timeout_s,
    }


def build_matrix(mode: str, n_dev: int) -> list[dict]:
    gpus = [1, 2] if n_dev >= 2 else [1]
    cfgs: list[dict] = []
    if mode == "smoke":
        for v in ("legacy", "shared"):
            for g in gpus:
                cfgs.append(_cfg(f"smoke-{v}-{g}g", "smoke", variant=v, n_gpus=g, timeout_s=900))
        for v in ("legacy", "shared"):
            cfgs.append(
                _cfg(f"stressk2ml2-{v}-{max(gpus)}g", "stress_k2", variant=v, n_gpus=max(gpus),
                     max_length=2, timeout_s=1800)
            )
        return cfgs

    # full — on-box recalibration: stress_k2's K=3 is ~76B candidates, so a
    # single legacy stress run costs ~37 min. Legacy stress gets ONE rep
    # (the slow baseline needs no variance estimate at that cost); shared
    # and deep_k keep 3. One-off axes run FIRST so a --max-hours stop can
    # only ever shed redundant reps, never whole measurement axes.
    for impl in ("compact", "cupy", "cpu"):
        cfgs.append(_cfg(f"stressk2-filter-{impl}", "stress_k2", filter_impl=impl,
                         n_gpus=max(gpus), max_length=2, timeout_s=1800))
    cfgs.append(_cfg("deepk-nonccl", "deep_k", disable_nccl=True, n_gpus=max(gpus)))
    cfgs.append(_cfg("deepk-density-auto", "deep_k", sparse_from_k="auto", n_gpus=max(gpus)))
    cfgs.append(_cfg("deepk-density-auto-1g", "deep_k", sparse_from_k="auto", n_gpus=1))
    cfgs.append(_cfg("twophase-smoke", "smoke", two_phase=True, n_gpus=max(gpus)))
    for rep in range(2):
        for bal in ("rows", "nnz"):
            cfgs.append(_cfg(f"skew-{bal}", "skewed_rows", balance=bal, n_gpus=max(gpus), rep=rep))
    for g in gpus:
        cfgs.append(_cfg(f"stressk2-legacy-{g}g", "stress_k2", variant="legacy", n_gpus=g,
                         max_length=3, rep=0, timeout_s=3600))
    for rep in range(3):
        for g in gpus:
            cfgs.append(_cfg(f"stressk2-shared-{g}g", "stress_k2", variant="shared", n_gpus=g,
                             max_length=3, rep=rep, timeout_s=3600))
            for v in ("legacy", "shared"):
                if rep == 0 or v == "shared":
                    cfgs.append(_cfg(f"deepk-{v}-{g}g", "deep_k", variant=v, n_gpus=g, rep=rep))
    return cfgs


def _git_rev() -> str:
    """`<sha>` at HEAD, suffixed `-dirty` when the tree carries uncommitted edits.

    Stamped onto every row so a resumed campaign can say WHICH code produced
    each number. Without it a replayed run prints the same "consistent" line as
    a fresh one: the smoke gate did exactly that across seven commits, and the
    green line was recomputed from JSON predating all of them.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=30, cwd=REPO
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=30, cwd=REPO,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha or "unknown"
    except Exception:
        return "unknown"


def capture_environment(out_dir: Path) -> None:
    # Re-captured per invocation, not once per directory: a resumed campaign
    # runs on a different revision than the one that started it, and the old
    # behaviour (return early if the file exists) froze env.txt at the first
    # run's HEAD forever.
    env_file = out_dir / "env.txt"
    blocks = []
    for cmd in (
        ["git", "rev-parse", "HEAD"],
        ["nvidia-smi"],
        [sys.executable, "-m", "pip", "freeze"],
    ):
        try:
            blocks.append(f"$ {' '.join(cmd)}\n" + subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, cwd=REPO
            ).stdout)
        except Exception as e:
            blocks.append(f"$ {' '.join(cmd)} FAILED: {e}")
    with env_file.open("a") as f:
        f.write(f"\n\n===== captured {time.strftime('%Y-%m-%dT%H:%M:%S')} rev={_git_rev()} =====\n")
        f.write("\n\n".join(blocks))


def run_config(cfg: dict, out_dir: Path) -> dict:
    print(f"→ {cfg['id']} (timeout {cfg['timeout_s']}s) env={cfg['env']}", flush=True)
    safe_id = cfg["id"].replace("#", "_")
    log_path = out_dir / f"{safe_id}.log"
    result_path = out_dir / f"{safe_id}.result.json"
    cfg = {**cfg, "result_path": str(result_path)}
    with log_path.open("w") as log_f:
        proc = subprocess.Popen(
            [sys.executable, str(CHILD), json.dumps(cfg)],
            stdout=subprocess.PIPE,
            stderr=log_f,
            text=True,
            cwd=REPO,
            start_new_session=True,  # own process group: timeout kill reaps the sampler too
        )
        try:
            stdout, _ = proc.communicate(timeout=cfg["timeout_s"])
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return {"id": cfg["id"], "config": cfg, "status": "timeout", "rev": _git_rev()}
    # Result file first (immune to NCCL's raw fd-1 writes splicing the
    # child's stdout); stdout scan as debug fallback.
    rev = _git_rev()
    if result_path.exists():
        try:
            return {**json.loads(result_path.read_text()), "rev": rev}
        except json.JSONDecodeError:
            pass
    for line in reversed(stdout.strip().splitlines() or [""]):
        if line.startswith("{"):
            try:
                return {**json.loads(line), "rev": rev}
            except json.JSONDecodeError:
                break
    return {"id": cfg["id"], "config": cfg, "status": f"no-result (rc={proc.returncode})", "rev": rev}


def group_key(cfg: dict) -> tuple:
    return (cfg["preset"], cfg.get("max_length"), cfg.get("min_support"), bool(cfg.get("two_phase")))


def check_equivalence(rows: list[dict]) -> list[str]:
    problems = []
    groups: dict[tuple, dict] = {}
    for r in rows:
        if r.get("status") != "ok" or "itemset_hash" not in r:
            continue
        key = group_key(r["config"])
        sig = (r["n_itemsets"], r["sum_counts"], r["itemset_hash"])
        if key in groups and groups[key][0] != sig:
            problems.append(
                f"SIGNATURE DIVERGENCE in group {key}: {groups[key][1]} {groups[key][0]} vs {r['id']} {sig}"
            )
        groups.setdefault(key, (sig, r["id"]))
        if not r.get("motifs_ok", True):
            problems.append(f"MOTIF RECOVERY FAILED: {r['id']}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], required=True)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--only", default=None, help="run only configs whose id contains this")
    ap.add_argument("--skip", default=None, help="skip configs whose id contains this")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    capture_environment(out_dir)
    raw = out_dir / "raw.jsonl"

    done_ids = set()
    rows: list[dict] = []
    if raw.exists():
        for line in raw.read_text().splitlines():
            try:
                r = json.loads(line)
                rows.append(r)
                if r.get("status") == "ok":
                    done_ids.add(r["id"])
            except json.JSONDecodeError:
                pass

    n_dev = _gpu_count()
    if n_dev == 0:
        print("no CUDA devices — nothing to run")
        return 2
    matrix = build_matrix(args.mode, n_dev)
    matrix_ids = [
        c["id"]
        for c in matrix
        if not (args.only and args.only not in c["id"]) and not (args.skip and args.skip in c["id"])
    ]
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None

    here = _git_rev()
    replayed: list[str] = []
    failed: list[str] = []
    fresh = 0
    for cfg in matrix:
        # Filters BEFORE the done check: reversed, a `--only smoke-gpu1` run
        # counted every other already-done config into `replayed` and warned
        # about configs it was never asked to gate.
        if args.only and args.only not in cfg["id"]:
            continue
        if args.skip and args.skip in cfg["id"]:
            continue
        if cfg["id"] in done_ids:
            was = next((r.get("rev", "unrecorded") for r in rows if r.get("id") == cfg["id"]), "unrecorded")
            print(f"skip (done): {cfg['id']}  [replayed from rev {was}]")
            replayed.append(was)
            continue
        if deadline and time.time() > deadline:
            print("max-hours reached — stopping (resume with the same command)")
            break
        result = run_config(cfg, out_dir)
        if result.get("status") == "ok":
            fresh += 1
        else:
            failed.append(f"{cfg['id']} ({result.get('status')})")
        rows.append(result)
        with raw.open("a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"   {result.get('status')} wall={result.get('wall_s')}s peak={result.get('peak_vram_mb')}")

    problems = check_equivalence(rows)
    if problems:
        print("\nCAMPAIGN CORRECTNESS FAILURES:")
        for p in problems:
            print(f"  {p}")
        return 1

    # The tick is emitted ONLY when it is a statement about this revision.
    #
    # `check_equivalence` skips every row whose status is not "ok", so a matrix
    # in which every config crashed compares nothing and reports no problems.
    # Stamping such a run with the current revision made it read
    # character-for-character like an honest fresh pass -- a stronger false
    # claim than the stale-replay case this reporting was written to fix, and
    # `bench/run_smoke.sh` runs under `set -euo pipefail`, so exit 0 IS the
    # gate passing. Hence: the tick requires that every config in this
    # invocation ran here, at this revision, and produced an ok row.
    #
    # Exit code stays 0 for a replay -- resuming a multi-hour campaign is
    # legitimate and failing it would break the resume this file exists to
    # support. What changes is that the line no longer says "consistent" about
    # a run that established nothing.
    stale_rows = [r for r in replayed if r != here]
    # `fresh == len(matrix_ids)` alone is true when BOTH are zero, so a
    # `--only` that matches nothing printed the tick over an empty selection --
    # the same vacuity one level down from the one this predicate exists to
    # stop. Selecting nothing is not gating everything.
    if not matrix_ids:
        print("\nequivalence groups: NOT GATED — no config matched the filters (--only/--skip).")
        return 0
    covered = fresh == len(matrix_ids) and not replayed and not failed
    if covered:
        print(f"\nequivalence groups consistent ✓  ({fresh} run here at {here})")
        return 0

    print(f"\nequivalence groups: NOT GATED at {here} — no ✓ emitted.")
    print(f"  {fresh} of {len(matrix_ids)} configs ran here and returned ok.")
    if failed:
        print(f"  {len(failed)} did not return ok: {', '.join(failed)}")
    if stale_rows:
        print(
            f"  {len(stale_rows)} of {len(replayed)} replayed rows were produced at a "
            f"revision other than {here} ({', '.join(sorted(set(stale_rows)))}) and say "
            f"nothing about the current tree."
        )
    elif replayed:
        print(f"  {len(replayed)} rows replayed from this same revision.")
    print("  Re-run with a fresh --out to gate this revision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
