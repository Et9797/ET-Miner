"""Wall split of a moonshots baseline run: where each regime's time goes.

Reads the runner's raw.jsonl (``--mode moonshots-baseline`` or an idea's mode) and prints, per
config, the median over its ok reps (with [min, max] for the wall) of: the
wall, the input phases before K=1, the levels (K=1 and the K>=2 phases summed
over levels), the result frame and the tail. Signatures must agree within a
config's reps and within each equivalence group.

Usage: uv run python bench/moonshots/split_table.py RAW_JSONL [--json]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict

INPUT = ("input_count", "input_k1", "input_explode", "input_csr", "input_widen", "input_h2d", "input_bitvec",
         "input_dev_chunks", "input_dev_copy", "input_dev_k1", "input_dev_bitvec", "input_other")
LEVEL = ("count_rows", "count_percand", "count_group", "count_tiled", "count_fused", "count_csr", "reduce",
         "compact", "filter", "group_build", "group_upload", "budget", "k2_dispatch", "k2_input", "decode", "sort",
         "free_prune", "transition", "materialize", "other")


def summarize(row: dict) -> dict:
    t = row.get("timings", {})
    span = t.get("span", {})
    split = t.get("level_split", {})
    out = {"wall": row["wall_s"], "read": t.get("read_s")}
    for p in INPUT:
        out[p] = span.get(p, 0.0)
    out["input"] = span.get("input", 0.0)
    out["k1"] = sum(lv["ms"] for lv in row["levels"] if lv["k"] == 1) / 1000
    for p in LEVEL:
        out[p] = sum(phases.get(p, 0.0) for k, phases in split.items() if int(k) >= 2)
    out["levels_k2plus"] = row["levels_ms_k2plus"] / 1000
    out["frame"] = span.get("frame", 0.0)
    out["tail"] = span.get("tail", 0.0)
    out["per_level"] = {int(k): phases["level"] for k, phases in split.items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    by_cfg: dict[str, list[dict]] = defaultdict(list)
    sigs: dict[str, set] = defaultdict(set)
    groups: dict[tuple, set] = defaultdict(set)
    for line in open(args.raw):
        row = json.loads(line)
        if row.get("status") != "ok":
            continue
        cfg = row["config"]
        by_cfg[cfg["base_id"]].append(row)
        sig = (row.get("n_itemsets"), row.get("sum_counts"), row.get("itemset_hash"))
        sigs[cfg["base_id"]].add(sig)
        groups[(cfg["preset"], cfg.get("max_length"), cfg["min_support"], bool(cfg.get("prune_equal_support")))].add(sig)
    bad = [k for k, v in sigs.items() if len(v) > 1] + [str(k) for k, v in groups.items() if len(v) > 1]
    table = {}
    for base, rows in by_cfg.items():
        parts = [summarize(r) for r in rows]
        med = {k: statistics.median(p[k] for p in parts) for k in parts[0] if k != "per_level" and parts[0][k] is not None}
        med["wall_min"] = min(p["wall"] for p in parts)
        med["wall_max"] = max(p["wall"] for p in parts)
        med["reps"] = len(parts)
        levels = defaultdict(list)
        for p in parts:
            for k, v in p["per_level"].items():
                levels[k].append(v)
        med["per_level"] = {k: statistics.median(v) for k, v in sorted(levels.items())}
        med["throttle"] = sorted({x for r in rows for x in r.get("throttle_reasons", [])})
        med["peak_vram_mb"] = max(max(r["peak_vram_mb"].values(), default=0) for r in rows)
        med["peak_rss_mb"] = max(r["peak_rss_mb"] for r in rows)
        table[base] = med
    if args.json:
        print(json.dumps({"signature_mismatch": bad, "configs": table}, indent=1))
        return 0
    print(f"signature mismatches: {bad or 'none'}\n")
    print("| config | reps | wall [min, max] | input | count/k1/explode/csr | widen/h2d/bitvec "
          "| device chunks/copy/k1/bitvec | other | K=1 | K>=2 | frame | tail |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for base, m in table.items():
        print(
            f"| {base} | {m['reps']} | {m['wall']:.2f} [{m['wall_min']:.2f}, {m['wall_max']:.2f}] | {m['input']:.2f} | "
            f"{m['input_count']:.2f}/{m['input_k1']:.2f}/{m['input_explode']:.2f}/{m['input_csr']:.2f} | "
            f"{m['input_widen']:.2f}/{m['input_h2d']:.2f}/{m['input_bitvec']:.2f} | "
            f"{m['input_dev_chunks']:.2f}/{m['input_dev_copy']:.2f}/{m['input_dev_k1']:.2f}/{m['input_dev_bitvec']:.2f} | "
            f"{m['input_other']:.2f} | {m['k1']:.2f} | {m['levels_k2plus']:.2f} | {m['frame']:.2f} | {m['tail']:.2f} |"
        )
    print()
    print("| config | count_rows | percand | group | tiled | fused | reduce | compact | filter | groups | other |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for base, m in table.items():
        print(
            f"| {base} | {m['count_rows']:.2f} | {m['count_percand']:.2f} | {m['count_group']:.2f} | "
            f"{m['count_tiled']:.2f} | {m['count_fused']:.2f} | {m['reduce']:.2f} | {m['compact']:.2f} | "
            f"{m['filter']:.2f} | {m['group_build'] + m['group_upload']:.2f} | {m['other']:.2f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
