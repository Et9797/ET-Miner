"""Upper-bound stakes of the moonshot ideas from the Phase 0 run and the Phase 1 stake outputs.

Reads the baseline's ``raw.jsonl`` and ``microbench.jsonl`` and the stake
scripts' outputs in one directory (``replay-*.jsonl``, ``census-*.jsonl``,
``k3rows-*.json``, ``components-*.jsonl``, ``tiles-*.jsonl``, ``dedup-*.jsonl``,
``input-floor-*.json``, ``replay-dsl-order-*.jsonl``) and prints, per idea and
regime, the seconds a perfect implementation could save at most (the formula
of each idea's SPEC entry) against the regime's median wall, as JSON lines.

Usage: uv run python bench/moonshots/stakes.py BASELINE_DIR STAKES_DIR
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REGIMES = {
    "smoke": "smoke-C1-base", "deepk": "deepk-C1-base", "skew": "skew-C1-base", "or003": "or003-C1-base",
    "or002": "or002-C1-base", "dsl": "dsl-C1-base", "dsl-infer": "dsl-C1-infer-base", "dsl-free": "dsl-C1-free-base",
    "oom2ml3": "oom2ml3-C1-base", "sk2ml3": "sk2ml3-C1-base",
}
COUNT = ("count_percand", "count_group", "count_tiled", "count_rows", "count_fused")
DSL_WORDS = 312_500


def medians(raw: Path) -> dict:
    """Per config: wall, span phases and per-level phase medians."""
    rows = defaultdict(list)
    for line in raw.open():
        r = json.loads(line)
        if r["status"] == "ok":
            rows[r["config"]["base_id"]].append(r)
    out = {}
    for base, rs in rows.items():
        lv = defaultdict(lambda: defaultdict(list))
        for r in rs:
            for k, ph in r["timings"]["level_split"].items():
                for p, v in ph.items():
                    lv[int(k)][p].append(v)
        span = defaultdict(list)
        for r in rs:
            for p, v in r["timings"]["span"].items():
                span[p].append(v)
        out[base] = {
            "wall": statistics.median(r["wall_s"] for r in rs),
            "span": {p: statistics.median(v) for p, v in span.items()},
            "levels": {k: {p: statistics.median(v) for p, v in ph.items()} for k, ph in sorted(lv.items())},
        }
    return out


def jl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.open()] if path.exists() else []


def counting(level: dict) -> float:
    return sum(level.get(p, 0.0) for p in COUNT)


def emit(idea: str, regime: str, saving: float, wall: float, **detail) -> None:
    saving, wall = float(saving), float(wall)
    print(json.dumps({"idea": idea, "regime": regime, "saving_s": round(saving, 3), "wall_s": round(wall, 3),
                      "share": round(saving / wall, 3) if wall else None,
                      "rule2_possible": saving >= 1.0 and saving >= 0.1 * wall, **detail}))


def main() -> int:
    base, st = Path(sys.argv[1]), Path(sys.argv[2])
    m = medians(base / "raw.jsonl")
    mb = {(r["probe"], r.get("fill")): r for r in jl(base / "microbench.jsonl")}
    h2d = mb[("h2d_pinned", None)]["rate"]
    ratio_bmma = mb[("bmma_m16n8k256", None)]["rate"] / mb[("popc64", None)]["rate"]

    # M1: input phases above the host floor (explode to int32 + pinned copies + the bitvector kernel).
    floors = {"dsl": jl(st / "input-floor-deep_sparse_large.json"), "sk2ml3": jl(st / "input-floor-stress_k2.json")}
    per_nnz = floors["dsl"][0]["explode_int32_chunked"] / floors["dsl"][0]["nnz"]
    for reg, cfg in REGIMES.items():
        c = m[cfg]
        span = c["span"]
        key = "dsl" if reg.startswith("dsl") else reg
        fl = floors.get(key)
        nnz = fl[0]["nnz"] if fl else None
        explode = fl[0]["explode_int32_chunked"] if fl else None
        n_rows = fl[0]["n_rows"] if fl else None
        if explode is None:
            continue
        floor = explode + (4 * nnz + 4 * n_rows) / h2d + span.get("input_bitvec", 0.0)
        emit("M1", reg, span["input"] - floor, c["wall"], input_s=round(span["input"], 3), floor_s=round(floor, 3))
    datasets = {"smoke": "smoke", "deepk": "deep_k", "skew": "skewed_rows", "oom2ml3": "oom_regression",
                "or003": "online_retail", "or002": "online_retail"}
    repo = Path(__file__).resolve().parents[2]
    for reg, ds in datasets.items():
        c = m[REGIMES[reg]]
        if ds == "online_retail":
            import polars as pl

            items = pl.read_parquet(repo / "datasets" / "online_retail_ii" / "transactions.parquet")["items"]
            nnz, n_rows = int(items.list.len().sum()), len(items)
        else:
            side = json.loads((repo / "datasets" / "synth" / f"{ds}.json").read_text())
            nnz, n_rows = side["nnz"], side["spec"]["n_rows"]
        floor = nnz * per_nnz + (4 * nnz + 4 * n_rows) / h2d + c["span"].get("input_bitvec", 0.0)
        emit("M1", reg, c["span"]["input"] - floor, c["wall"], input_s=round(c["span"]["input"], 3),
             floor_s=round(floor, 3), note="floor at dsl's explode rate per entry")

    # M2: K=3 from the rows on the explosions, at the regime's own row-wise K=2 rate per row pair.
    for reg in ("oom2ml3", "sk2ml3"):
        c = m[REGIMES[reg]]
        k3 = jl(st / f"k3rows-{reg}.json")
        if not k3:
            continue
        k3 = k3[0]
        rate = c["levels"][2]["count_rows"] / k3["row_pairs"]
        model = (k3["row_pairs"] + k3["increments"]) * rate
        t3 = counting(c["levels"][3])
        emit("M2", reg, t3 - model, c["wall"], k3_counting_s=round(t3, 3), rows_model_s=round(model, 3),
             increments=k3["increments"], row_pairs=k3["row_pairs"], ns_per_unit=round(rate * 1e9, 3))
    # M2 on the deep levels: subsets the rows hold against the counted work.
    for reg in ("dsl", "deepk", "skew", "or002"):
        rows = [r for r in jl(st / f"replay-{reg}.jsonl") if "full" in r]
        worst = min((r["rows"]["row_subsets"] / max(1, r["full"]["counted"] * r["words"]) for r in rows if r["full"]["counted"]),
                    default=None)
        emit("M2", reg, 0.0, m[REGIMES[reg]]["wall"], min_row_subsets_per_counted_word=worst,
             note="row-wise K>=3 enumerates C(active items, k) per row")

    # M3: the popcount part of the tiled and group kernels at the measured b1 rate.
    tiled_zero = mb[("kernel_tiled", "zero")]["ns_per_unit_word"]
    group_zero = mb[("kernel_group", "zero")]["ns_per_unit_word"]
    group_dense = mb[("kernel_group", "dense")]["ns_per_unit_word"]
    for reg in ("sk2ml3", "oom2ml3"):
        c = m[REGIMES[reg]]
        cen = {r["order"]: r for r in jl(st / f"census-{reg}.jsonl")}["orig"]
        t = c["levels"][3]["count_tiled"]
        per = t / (cen["tiled_counted_tilepairs"] * cen["words"]) * 1e9
        share = max(0.0, (per - tiled_zero) / per)
        emit("M3", reg, t * share * (1 - 1 / ratio_bmma), c["wall"], tiled_s=round(t, 3),
             ns_per_tilepair_word=round(per, 3), compute_share=round(share, 3), bmma_over_popc=round(ratio_bmma, 1))
    for reg in ("dsl", "dsl-infer", "dsl-free"):
        c = m[REGIMES[reg]]
        g = sum(lv.get("count_group", 0) for lv in c["levels"].values())
        t = sum(lv.get("count_tiled", 0) for lv in c["levels"].values())
        share = (group_dense - group_zero) / group_dense
        emit("M3", reg, (g * share + t) * (1 - 1 / ratio_bmma), c["wall"], group_s=round(g, 3), tiled_s=round(t, 3),
             group_compute_share_dense=round(share, 3))

    # M4: levels from K0 on, when every component fits a 2^30 table; the finisher's own cost is not subtracted.
    for reg in ("smoke", "deepk", "skew", "or003", "or002", "dsl", "dsl-free"):
        comps = jl(st / f"components-{reg}.jsonl")
        c = m[REGIMES[reg]]
        best = None
        for row in comps:
            if row["max_items"] <= 30 and row["components"]:
                after = sum(lv["level"] for k, lv in c["levels"].items() if k >= row["k0"])
                best = best or (row, after)
        if best:
            row, after = best
            emit("M4", reg, after, c["wall"], k0=row["k0"], max_items=row["max_items"], components=row["components"],
                 table_bytes=row["table_bytes"], itemsets_from_k0=row["itemsets_from_k0"])
        else:
            emit("M4", reg, 0.0, c["wall"], note="no K0 where every component has <= 30 items",
                 smallest_max_items=min((r["max_items"] for r in comps), default=None))

    # M5a: each level's counting at the share of rows it keeps (free rebuilds, any threshold).
    for reg in ("dsl", "dsl-infer", "dsl-free", "deepk", "skew", "or002"):
        src = "dsl" if reg.startswith("dsl") else reg
        rep = {r["k"]: r for r in jl(st / f"replay-{src}.jsonl") if "rows" in r}
        c = m[REGIMES[reg]]
        s = sum(counting(lv) * (1 - rep[k]["rows"]["kept_share"]) for k, lv in c["levels"].items() if k in rep)
        half = sum(counting(lv) * (1 - rep[k]["rows"]["kept_share"]) for k, lv in c["levels"].items()
                   if k in rep and rep[k]["rows"]["kept_share"] <= 0.5)
        emit("M5a", reg, s, c["wall"], at_half_rule_s=round(half, 3))

    # M5b: zero prefix word-tiles, per row order, times each level's counting.
    tiles = defaultdict(dict)
    for r in jl(st / "tiles-dsl.jsonl"):
        tiles[r["order"]][r["k"]] = r["zero_prefix_tile_share"]
    for order, shares in tiles.items():
        for reg in ("dsl", "dsl-infer", "dsl-free"):
            c = m[REGIMES[reg]]
            s = sum(counting(lv) * shares.get(k, 0.0) for k, lv in c["levels"].items())
            emit("M5b", reg, s, c["wall"], order=order)
    # M12 / M5b on the K=3 explosions: staged row tiles skipped, times the staging share of the tiled kernel.
    for reg in ("sk2ml3", "oom2ml3"):
        c = m[REGIMES[reg]]
        t = c["levels"][3]["count_tiled"]
        cen = {r["order"]: r for r in jl(st / f"census-{reg}.jsonl")}["orig"]
        per = t / (cen["tiled_counted_tilepairs"] * cen["words"]) * 1e9
        staging = min(1.0, tiled_zero / per)
        for r in jl(st / f"tiles-{reg}.jsonl"):
            cut = 1 - r["with_prefix_and_suffix_skip"] / r["staged_row_tiles_today"]
            cut_p = 1 - r["with_prefix_skip"] / r["staged_row_tiles_today"]
            emit("M12" if r["order"] == "orig" else "M5b", reg, t * staging * cut, c["wall"], order=r["order"],
                 staged_cut=round(cut, 3), prefix_only_cut=round(cut_p, 3), staging_share=round(staging, 3))

    # M7: counted candidates the deduction rules settle, times each level's counting under count inference.
    rep = {r["k"]: r for r in jl(st / "replay-dsl.jsonl") if "ndi" in r}
    for reg in ("dsl-infer",):
        c = m[REGIMES[reg]]
        s = 0.0
        per_level = {}
        for k, lv in c["levels"].items():
            r = rep.get(k)
            if r and r["ndi"]["considered"]:
                share = (r["ndi"]["skip_infrequent"] + r["ndi"]["exact"]) / r["ndi"]["considered"]
                s += counting(lv) * share
                per_level[k] = round(share, 3)
        emit("M7", reg, s, c["wall"], settled_share_by_k=per_level)
    for reg in ("or002", "or003", "deepk", "skew"):
        rep_r = {r["k"]: r for r in jl(st / f"replay-{reg}.jsonl") if "ndi" in r}
        c = m[REGIMES[reg]]
        s = sum(counting(lv) * (rep_r[k]["ndi"]["skip_infrequent"] + rep_r[k]["ndi"]["exact"]) / rep_r[k]["ndi"]["considered"]
                for k, lv in c["levels"].items() if k in rep_r and rep_r[k]["ndi"]["considered"])
        emit("M7", reg, s, c["wall"], note="without count inference the regime counts the inferable ones too")

    # M8: the next level's counting, minus the triples' popcounts at the group kernel's dense ALU rate.
    alu = (group_dense - group_zero) * 1e-9
    for reg, src in (("dsl", "dsl"), ("dsl-infer", "dsl")):
        rep = {r["k"]: r for r in jl(st / f"replay-{src}.jsonl") if "full" in r}
        c = m[REGIMES[reg]]
        s = 0.0
        for k, lv in c["levels"].items():
            if k + 1 in c["levels"] and k in rep and k >= 3:
                extra = rep[k]["triples_sum"] * rep[k]["words"] * alu
                s += max(0.0, counting(c["levels"][k + 1]) - extra)
        emit("M8", reg, s, c["wall"])

    # M9: share of written entries a pigeonhole reduce flags on 2 GPUs (the reduce itself needs a 2-GPU box).
    for reg in ("oom2ml3", "sk2ml3"):
        k3 = jl(st / f"k3rows-{reg}.json")
        cen = {r["order"]: r for r in jl(st / f"census-{reg}.jsonl")}.get("orig")
        if k3 and cen:
            emit("M9", reg, 0.0, m[REGIMES[reg]]["wall"], flagged=k3[0]["flagged"], written=cen["counted"],
                 flagged_share=round(k3[0]["flagged"] / cen["counted"], 5), note="saving = (reduce+compact) x (1 - share) on 2 GPUs")

    # M10: K=3 under ascending support, at today's per-unit rates of each kernel.
    pc_rate = None
    dsl = m["dsl-C1-base"]["levels"]
    rep_dsl = {r["k"]: r for r in jl(st / "replay-dsl.jsonl") if "full" in r}
    reads = sum(rep_dsl[k]["full"]["percand_row_reads"] for k in range(5, 12))
    pc_rate = sum(dsl[k]["count_percand"] for k in range(5, 12)) / (reads * DSL_WORDS)
    # The group kernel's time per level ~ staged rows x a + listed pairs x b (least squares over dsl's levels).
    import numpy as np

    ks = [k for k in range(3, 14) if k in rep_dsl and k in dsl]
    A = np.array([[rep_dsl[k]["full"]["group_staged_rows"], rep_dsl[k]["full"]["group_listed_pairs"]] for k in ks], float)
    y = np.array([dsl[k]["count_group"] for k in ks])
    (stage_rate, pair_rate), *_ = np.linalg.lstsq(A * DSL_WORDS, y, rcond=None)
    for reg in ("oom2ml3", "sk2ml3"):
        c = m[REGIMES[reg]]
        cen = {r["order"]: r for r in jl(st / f"census-{reg}.jsonl")}
        t_today = counting(c["levels"][3])
        per_tp = c["levels"][3]["count_tiled"] / (cen["orig"]["tiled_counted_tilepairs"] * cen["orig"]["words"])
        for order in ("asc", "deg"):
            o = cen[order]
            w = o["words"]
            model = (o["tiled_counted_tilepairs"] * w * per_tp + o["group_staged_rows"] * w * stage_rate
                     + o["group_listed_pairs"] * w * pair_rate + o["percand_counted"] * 3 * w * pc_rate)
            emit("M10", reg, t_today - model, c["wall"], order=order, k3_counting_s=round(t_today, 3),
                 model_s=round(model, 3), generated=o["generated"], tilepairs=o["tiled_counted_tilepairs"],
                 group_pairs=o["group_listed_pairs"])
    for order in ("asc", "desc"):
        rep_o = {r["k"]: r for r in jl(st / f"replay-dsl-order-{order}.jsonl") if "full" in r}
        for reg, regime in (("dsl", "full"), ("dsl-infer", "infer")):
            c = m[REGIMES[reg]]
            today = model_o = 0.0
            for k, r in rep_dsl.items():
                if k not in rep_o:
                    continue
                for rr, acc in ((r, "today"), (rep_o[k], "o")):
                    f = rr[regime]
                    t = (f["percand_row_reads"] * pc_rate + f["group_staged_rows"] * stage_rate
                         + f["group_listed_pairs"] * pair_rate) * rr["words"]
                    if acc == "today":
                        today += t
                    else:
                        model_o += t
            emit("M10", reg, today - model_o, c["wall"], order=order, model_today_s=round(today, 3),
                 model_order_s=round(model_o, 3))

    # M11: distinct projected rows.
    for reg in ("or003", "or002", "dsl", "deepk", "skew", "smoke"):
        for r in jl(st / f"dedup-{reg}.jsonl"):
            if r["level"] == "frequent":
                emit("M11", reg, 0.0, m[REGIMES[reg]]["wall"], dedup_ratio=round(r["ratio"], 3),
                     note="build only if ratio / weight planes >= 4 on a real workload")

    # X0: count inference against the default on dsl (other regimes: measure).
    emit("X0", "dsl", m["dsl-C1-base"]["wall"] - m["dsl-C1-infer-base"]["wall"], m["dsl-C1-base"]["wall"])
    print(json.dumps({"constants": {"bmma_over_popc64": ratio_bmma, "pc_s_per_row_word": pc_rate,
                                    "group_stage_s_per_row_word": float(stage_rate),
                                    "group_pair_s_per_word": float(pair_rate), "h2d_pinned": h2d,
                                    "explode_s_per_nnz": per_nnz}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
