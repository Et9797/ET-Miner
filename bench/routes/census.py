"""Which counting kernels each regime ran, per level, from the level split of the runner's rows.

Usage: uv run python bench/routes/census.py RAW.jsonl
"""

from __future__ import annotations

import json
import sys

PHASES = ("count_rows", "count_percand", "count_group", "count_tiled", "count_fused", "count_csr", "occupancy")


def main() -> int:
    for line in open(sys.argv[1]):
        r = json.loads(line)
        ls = r.get("timings", {}).get("level_split") or {}
        levels = []
        for k, ph in sorted(ls.items(), key=lambda x: int(x[0])):
            used = [p.replace("count_", "") for p in PHASES if ph.get(p, 0) > 0.0005]
            levels.append(f"K{k}:{'+'.join(used) or '-'}")
        vram = max((r.get("peak_vram_mb") or {}).values(), default=0)
        print(f"{r['id']:<34} {r.get('status', '')[:30]:<10} wall={r.get('wall_s')} vram={vram} "
              f"rss={r.get('peak_rss_mb')} " + " ".join(levels))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
