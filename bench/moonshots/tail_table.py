"""The 2-GPU tail of a run, split: the NCCL communicators' destroy against the rest.

Reads the runner's raw.jsonl (``--mode moonshots-teardown``) and prints, per
config, the median with [min, max] over its ok reps of the wall, the tail
(from the last level's callback to the return), its ``tail_nccl`` and
``tail_other`` (level_split.py), and the destroy's share of the wall. The last
line pools every row. A config whose reps disagree on the itemset hash is
flagged, and with ``--reference`` so is one whose hash differs from the
reference rows of its regime and GPU count (the id without its arm).

Usage: uv run python bench/moonshots/tail_table.py RAW_JSONL [--reference JSONL]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict

COLUMNS = ("wall_s", "tail", "tail_nccl", "tail_other")


def _cell(values: list[float]) -> str:
    return f"{statistics.median(values):.3f} [{min(values):.3f}, {max(values):.3f}]"


def _regime(config_id: str) -> str:
    """``dsl-C2-infer-teardown#r0`` -> ``dsl-C2-infer``: the config without its arm and rep."""
    return config_id.split("#")[0].rsplit("-", 1)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--reference", default=None)
    args = ap.parse_args()
    reference: dict[str, set[str]] = defaultdict(set)
    if args.reference:
        for line in open(args.reference):
            r = json.loads(line)
            if r.get("status") == "ok":
                reference[_regime(r["id"])].add(r["itemset_hash"])
    rows: dict[str, list[dict]] = defaultdict(list)
    for line in open(args.raw):
        r = json.loads(line)
        if r.get("status") == "ok":
            rows[r["id"].split("#")[0]].append(r)

    def values(r: dict) -> dict[str, float]:
        span = r["timings"]["span"]
        return {"wall_s": r["wall_s"], **{c: span[c] for c in COLUMNS[1:]}}

    print("| config | reps | wall s | tail s | tail_nccl s | tail_other s | nccl / wall |")
    print("|---|---|---|---|---|---|---|")
    pooled: dict[str, list[float]] = defaultdict(list)
    for base_id, reps in rows.items():
        v = [values(r) for r in reps]
        for x in v:
            for c in COLUMNS:
                pooled[c].append(x[c])
        share = statistics.median(x["tail_nccl"] / x["wall_s"] for x in v)
        hashes = {r["itemset_hash"] for r in reps}
        known = reference.get(_regime(base_id))
        flag = "" if len(hashes) == 1 and (not args.reference or hashes == known) else " SIGNATURE MISMATCH"
        cells = " | ".join(_cell([x[c] for x in v]) for c in COLUMNS)
        print(f"| {base_id}{flag} | {len(reps)} | {cells} | {share:.0%} |")
    if pooled:
        n = len(pooled["wall_s"])
        print(f"| all rows | {n} | {' | '.join(_cell(pooled[c]) for c in COLUMNS)} | |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
