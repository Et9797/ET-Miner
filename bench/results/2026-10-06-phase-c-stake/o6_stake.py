"""O6's stake per level: the work of the K≥3 kernels on a lattice dump, set against their measured times.

Reads a complete lattice dump (`<name>.lattice.parquet` with its `.json`
sidecar, written by `bench/runner.py --mode waste`) and a runner jsonl with
`timings.level_split`. Per level k ≥ 3 it rebuilds the prefix groups from level
k−1, applies the subset test (and with `--infer` count inference), and splits
the groups at `TILED_MIN_GROUP_PAIRS` as `row_split` does. It prints:

- per-candidate groups: generated and counted candidates, groups holding a
  counted candidate, row sweeps (k rows per counted candidate), and the row
  sweeps of a group-per-block kernel ((k−2) prefix rows plus the suffixes a
  counted pair uses, once per group);
- tiled groups: counted candidates and counted tile-pairs (≥ 1 counted pair)
  at tile sizes 32, 16 and 8, with the share of their slots that hold a
  counted pair;
- the per-candidate groups' counted tile-pairs at tile size 8;
- the measured `count_percand` and `count_tiled` (median over the config's
  reps), the per-candidate kernel's row-sweep rate in GB/s, and the tiled
  kernel's ns per counted 32×32 tile-pair per word.

Exact: the candidates, the subset test and the free flags come from the dump.

Usage: uv run python o6_stake.py DUMP_PREFIX RAW_JSONL CONFIG [--infer]
"""

import collections
import json
import math
import statistics
import sys

import polars as pl

TILED_MIN_GROUP_PAIRS = {2: 120, 3: 120, 4: 91, 5: 66, 6: 45, 7: 32, 8: 23}  # gpu/row_split_chunks.py
TILES = (32, 16, 8)


def min_pairs(k: int) -> int:
    return TILED_MIN_GROUP_PAIRS.get(k, TILED_MIN_GROUP_PAIRS[8])


def measured(raw: str, config: str) -> dict[int, dict[str, float]]:
    by_k: dict[int, dict[str, list[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(raw):
        r = json.loads(line)
        if r["config"]["base_id"] != config or r["status"] != "ok":
            continue
        for k, d in r["timings"]["level_split"].items():
            for p in ("count_percand", "count_tiled"):
                by_k[int(k)][p].append(d[p])
    return {k: {p: statistics.median(v) for p, v in d.items()} for k, d in by_k.items()}


def tile_pairs(pairs: list[tuple[int, int]], t: int) -> set[tuple[int, int]]:
    return {(i // t, j // t) for i, j in pairs}


def stake(prefix: str, raw: str, config: str, infer: bool) -> None:
    meta = json.loads(open(prefix + ".json").read())
    words = math.ceil(meta["n_rows"] / 64)
    df = pl.read_parquet(prefix + ".parquet")
    levels: dict[int, dict[tuple, int]] = collections.defaultdict(dict)
    for s, c in zip(df["itemset"].to_list(), df["count"].to_list()):
        levels[len(s)][tuple(sorted(s))] = c
    times = measured(raw, config)
    print(f"== {prefix.split('/')[-1]} {config}{' --infer' if infer else ''} N={meta['n_rows']:,} words={words:,}")
    cols = (
        "K", "gen", "pc_gen", "pc_cnt", "pc_grp", "pc_rows", "grp_rows", "pc_tp8", "tl_cnt",
        "tp32", "use32%", "tp16", "use16%", "tp8", "use8%", "percand_s", "tiled_s", "pc_GB/s", "ns/tp32w",
    )  # fmt: skip
    print(" ".join(f"{c:>10s}" for c in cols))
    total: collections.Counter = collections.Counter()
    for k in sorted(levels):
        big_k = k + 1
        if big_k < 3 or big_k not in times:
            continue
        prev = levels[k]
        free = (
            {y: all(levels[k - 1].get(y[:i] + y[i + 1 :], -1) != c for i in range(k)) for y, c in prev.items()}
            if infer
            else None
        )
        groups = collections.defaultdict(list)
        for y in sorted(prev):
            groups[y[:-1]].append(y[-1])
        acc: collections.Counter = collections.Counter()
        for pref, sufs in groups.items():
            s = len(sufs)
            if s < 2:
                continue
            counted = []
            for j in range(1, s):
                for i in range(j):
                    a, b = sufs[i], sufs[j]
                    subsets = [pref[:d] + pref[d + 1 :] + (a, b) for d in range(len(pref))]
                    if any(x not in prev for x in subsets):
                        continue  # skipped by the subset test
                    if infer and not all(free[x] for x in subsets + [pref + (a,), pref + (b,)]):
                        continue  # count inferred
                    counted.append((i, j))
            n_pairs = s * (s - 1) // 2
            if n_pairs < min_pairs(big_k):
                acc["pc_gen"] += n_pairs
                acc["pc_cnt"] += len(counted)
                acc["pc_rows"] += big_k * len(counted)
                if counted:
                    acc["pc_grp"] += 1
                    acc["grp_rows"] += (big_k - 2) + len({i for i, _ in counted} | {j for _, j in counted})
                acc["pc_tp8"] += len(tile_pairs(counted, 8))
            else:
                acc["tl_gen"] += n_pairs
                acc["tl_cnt"] += len(counted)
                for t in TILES:
                    acc[f"tp{t}"] += len(tile_pairs(counted, t))
        acc["gen"] = acc["pc_gen"] + acc["tl_gen"]
        m = times[big_k]
        pc_rate = acc["pc_rows"] * words * 8 / m["count_percand"] / 1e9 if m["count_percand"] > 1e-3 else math.nan
        tl_rate = m["count_tiled"] / (acc["tp32"] * words) * 1e9 if acc["tp32"] else math.nan
        use = {t: 100 * acc["tl_cnt"] / (acc[f"tp{t}"] * t * t) if acc[f"tp{t}"] else math.nan for t in TILES}
        row = (
            big_k, acc["gen"], acc["pc_gen"], acc["pc_cnt"], acc["pc_grp"], acc["pc_rows"], acc["grp_rows"],
            acc["pc_tp8"], acc["tl_cnt"], acc["tp32"], f"{use[32]:.1f}", acc["tp16"], f"{use[16]:.1f}", acc["tp8"],
            f"{use[8]:.1f}", f"{m['count_percand']:.3f}", f"{m['count_tiled']:.3f}", f"{pc_rate:.0f}", f"{tl_rate:.2f}",
        )  # fmt: skip
        print(" ".join(f"{v:>10}" for v in row))
        total.update(acc)
        total["percand_s"] += m["count_percand"]
        total["tiled_s"] += m["count_tiled"]
    slots = {t: total[f"tp{t}"] * t * t for t in TILES}
    print(
        f"   total: counted {total['pc_cnt']:,} per-candidate in {total['pc_grp']:,} groups, {total['tl_cnt']:,} tiled; "
        f"row sweeps per-candidate {total['pc_rows']:,}, group-per-block {total['grp_rows']:,} "
        f"({100 * total['grp_rows'] / max(1, total['pc_rows']):.0f} %); per-candidate groups at tile 8: "
        f"{total['pc_tp8'] * 64:,} slots"
    )
    print(
        "   tiled slots: "
        + ", ".join(f"T={t} {slots[t]:,} (use {100 * total['tl_cnt'] / max(1, slots[t]):.1f} %)" for t in TILES)
        + f"; measured percand {total['percand_s']:.2f} s, tiled {total['tiled_s']:.2f} s"
    )


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--infer"]
    stake(*args, infer="--infer" in sys.argv)
