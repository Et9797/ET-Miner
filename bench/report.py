"""Render bench/results/<dir>/raw.jsonl into a markdown report."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "bench" / "results" / "campaign"


def _fmt_s(x) -> str:
    return f"{x:.1f}" if isinstance(x, (int, float)) else "—"


def _base_id(run_id: str) -> str:
    return run_id.split("#", 1)[0]


def load(raw: Path) -> list[dict]:
    rows = []
    for line in raw.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out_dir = Path(args.out)
    raw = out_dir / "raw.jsonl"
    if not raw.exists():
        print(f"no results at {raw}")
        return 1
    rows = load(raw)

    ok = [r for r in rows if r.get("status") == "ok"]
    bad = [r for r in rows if r.get("status") != "ok"]

    by_config: dict[str, list[dict]] = defaultdict(list)
    for r in ok:
        by_config[_base_id(r["id"])].append(r)

    lines: list[str] = []
    lines.append("# GPU campaign report\n")
    lines.append(f"Runs: {len(ok)} ok, {len(bad)} failed/timeout. Raw data: `raw.jsonl`, env: `env.txt`.\n")

    lines.append("## Wall time by config (median over reps)\n")
    lines.append("| config | preset | variant | filter | gpus | reps | median s | min s | peak VRAM MB | throttled |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for base in sorted(by_config):
        rs = by_config[base]
        cfg = rs[0]["config"]
        walls = [r["wall_s"] for r in rs]
        peak = max((max(r.get("peak_vram_mb", {"0": 0}).values(), default=0) for r in rs), default=0)
        throttled = any(r.get("throttle_reasons") for r in rs)
        lines.append(
            f"| {base} | {cfg['preset']} | {cfg['env'].get('ET_MINER_KERNEL_VARIANT', '-')} "
            f"| {cfg['env'].get('ET_MINER_FILTER_IMPL', 'compact')} | {cfg.get('n_gpus')} "
            f"| {len(rs)} | {_fmt_s(statistics.median(walls))} | {_fmt_s(min(walls))} "
            f"| {peak} | {'⚠' if throttled else ''} |"
        )
    lines.append("")

    # Shared-vs-legacy A/B per (preset, gpus)
    lines.append("## Kernel variant A/B (legacy → shared, median wall)\n")
    lines.append("| preset | gpus | legacy s | shared s | speedup |")
    lines.append("|---|---|---|---|---|")
    ab: dict[tuple, dict[str, float]] = defaultdict(dict)
    for base, rs in by_config.items():
        cfg = rs[0]["config"]
        v = cfg["env"].get("ET_MINER_KERNEL_VARIANT")
        if v in ("legacy", "shared") and len(cfg["env"]) == 1 and not cfg.get("two_phase"):
            ab[(cfg["preset"], cfg.get("n_gpus"), cfg.get("max_length"))][v] = statistics.median(
                [r["wall_s"] for r in rs]
            )
    for (preset, gpus, _ml), d in sorted(ab.items()):
        if "legacy" in d and "shared" in d:
            speed = d["legacy"] / d["shared"] if d["shared"] else float("nan")
            lines.append(f"| {preset} | {gpus} | {_fmt_s(d['legacy'])} | {_fmt_s(d['shared'])} | {speed:.2f}× |")
    lines.append("")

    # Per-level breakdown for the slowest ok run of each preset
    lines.append("## Per-level timings (slowest run per preset)\n")
    slowest: dict[str, dict] = {}
    for r in ok:
        p = r["config"]["preset"]
        if p not in slowest or r["wall_s"] > slowest[p]["wall_s"]:
            slowest[p] = r
    for p, r in sorted(slowest.items()):
        lines.append(f"**{p}** ({r['id']}, {r['wall_s']}s):\n")
        lines.append("| K | candidates | frequent | ms |")
        lines.append("|---|---|---|---|")
        for lv in r.get("levels", []):
            lines.append(f"| {lv['k']} | {lv['n_candidates']:,} | {lv['n_frequent']:,} | {lv['ms']:.0f} |")
        lines.append("")

    if bad:
        lines.append("## Failures / timeouts\n")
        for r in bad:
            lines.append(f"- `{r['id']}`: {r.get('status')}")
        lines.append("")

    report = out_dir / "report.md"
    report.write_text("\n".join(lines))
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
