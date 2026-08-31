#!/usr/bin/env python3
"""Compute closed + maximal frequent-itemset counts (Major 5).

Post-processing pass over the per-K ``frequent_k{K}.parquet`` files produced by
the mine. Reports, per K and in total:

    total    — all frequent itemsets (the number already in the paper)
    closed   — no frequent superset has *equal* support
    maximal  — no frequent superset exists at all

Both are derived in a single downward K <-> K+1 pass, exploiting anti-monotonicity:
a superset's support is always <= the subset's. So for a K-itemset with support s,
looking only at its (K+1)-supersets (their drop-1 subsets are exactly the K-subsets):

    non-maximal  <=>  at least one frequent (K+1)-superset exists
    non-closed   <=>  at least one (K+1)-superset has support == s
                      (equivalently: max superset support == s)

Checking only K+1 is complete: if a frequent superset of size > K+1 existed, then by
downward closure a frequent (K+1)-superset containing the itemset exists too.

This mirrors ``filter_self_sufficient.py`` (which joins K with K-1); here we join K
with K+1. The maximal count independently cross-checks the closed count from a
``prune_equal_support=True`` mining run.

CAVEAT — exhaustiveness: the deepest level Kmax has no K+1 parquet, so all its
itemsets are reported closed + maximal. That is exact only if the mine terminated
naturally (no frequent (Kmax+1)-itemset existed). If the run was length-capped
(``--max-length L``), the top level's closed/maximal are an UPPER BOUND. Pass
``--capped-top-level`` to flag this in the output.

Usage (on the box, after the campaign wrote per-K parquets):
    python3 compute_maximal.py \
        --results-dir /workspace/results_214m/parquet \
        --n-transactions 85000000 \
        --output-json /workspace/results_214m/closed_maximal.json \
        -v
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from et_miner.rules import (  # noqa: E402
    _explode_drop1,
    _iter_row_groups,
    _list_to_scalar_cols,
)

_LEVEL_RE = re.compile(r"frequent_k(\d+)(?:\.parquet)?$")


def discover_levels(results_dir: Path) -> dict[int, Path]:
    """Map K -> path of each ``frequent_k{K}`` parquet (file or partitioned dir)."""
    levels: dict[int, Path] = {}
    for entry in sorted(Path(results_dir).iterdir()):
        m = _LEVEL_RE.search(entry.name)
        if not m:
            continue
        k = int(m.group(1))
        # A single-file .parquet and a partitioned directory should not both exist;
        # if they do, prefer the directory (parallel-flush output is authoritative).
        if k not in levels or entry.is_dir():
            levels[k] = entry
    return levels


def _count_rows(parquet_path: Path) -> int:
    """Total row count across a parquet file or partitioned directory."""
    path = Path(parquet_path)
    files = sorted(path.glob("part_*.parquet")) if path.is_dir() else [path]
    return sum(pq.ParquetFile(str(f)).metadata.num_rows for f in files)


def _superset_support_lookup(kp1_parquet: Path, k: int, chunk_size: int) -> pl.DataFrame:
    """Build {K-subset -> max (K+1)-superset support} from the K+1 parquet.

    Streams the (K+1) parquet in row-group chunks, explodes each (K+1)-itemset into
    its K drop-1 subsets (each carrying the superset's support), and aggregates the
    max superset support per distinct K-subset. Returns a DataFrame keyed by scalar
    columns i0..i{k-1} with column ``max_superset_support``.
    """
    join_cols = [f"i{j}" for j in range(k)]
    partials: list[pl.DataFrame] = []

    for chunk in _iter_row_groups(kp1_parquet, chunk_size):
        # drop-1 of a (K+1)-itemset -> K-subset in `antecedent`, superset support in `support`
        exploded = _explode_drop1(chunk, k + 1)
        grouped = (
            exploded.with_columns(_list_to_scalar_cols("antecedent", k))
            .group_by(join_cols)
            .agg(pl.col("support").max().alias("max_superset_support"))
        )
        partials.append(grouped)
        del chunk, exploded

    if not partials:
        return pl.DataFrame(
            schema={**{c: pl.Int64 for c in join_cols}, "max_superset_support": pl.Float64}
        )

    # Merge per-chunk partial maxima into a global max per K-subset.
    return (
        pl.concat(partials, how="vertical")
        .group_by(join_cols)
        .agg(pl.col("max_superset_support").max())
    )


def compute_level(
    k: int,
    k_parquet: Path,
    kp1_parquet: Path | None,
    chunk_size: int,
    rel_tol: float,
) -> dict:
    """Return total/closed/maximal counts for one K level."""
    join_cols = [f"i{j}" for j in range(k)]

    if kp1_parquet is None:
        # Deepest level: no supersets in the mined set -> all closed + maximal.
        total = _count_rows(k_parquet)
        logger.info(f"K={k}: {total:,} itemsets, no K+1 level -> all closed + maximal")
        return {"k": k, "total": total, "closed": total, "maximal": total, "top_level": True}

    t0 = time.perf_counter()
    lookup = _superset_support_lookup(kp1_parquet, k, chunk_size)
    logger.debug(f"K={k}: superset lookup built ({len(lookup):,} K-subsets covered) "
                 f"in {time.perf_counter() - t0:.1f}s")

    total = 0
    non_maximal = 0
    non_closed = 0

    for chunk in _iter_row_groups(k_parquet, chunk_size):
        n = len(chunk)
        total += n
        keyed = chunk.with_columns(_list_to_scalar_cols("itemset", k))
        joined = keyed.join(lookup, on=join_cols, how="left")

        covered = joined.filter(pl.col("max_superset_support").is_not_null())
        non_maximal += covered.height

        # non-closed: a superset has equal support. support is anti-monotone so the
        # superset support can only equal (not exceed) the subset support; compare
        # with a relative tolerance to absorb float division noise.
        nc = covered.filter(
            (pl.col("support") - pl.col("max_superset_support")).abs()
            <= rel_tol * pl.col("support")
        ).height
        non_closed += nc
        del chunk, keyed, joined, covered

    result = {
        "k": k,
        "total": total,
        "closed": total - non_closed,
        "maximal": total - non_maximal,
        "top_level": False,
    }
    logger.info(
        f"K={k}: {total:,} total | {result['closed']:,} closed | {result['maximal']:,} maximal"
    )
    return result


def compute_closed_maximal(
    results_dir: Path,
    chunk_size: int = 1_000_000,
    rel_tol: float = 1e-9,
    capped_top_level: bool = False,
) -> dict:
    """Compute closed + maximal counts across all discovered K levels."""
    levels = discover_levels(results_dir)
    if not levels:
        raise FileNotFoundError(f"No frequent_k*.parquet found in {results_dir}")

    # An interior gap (e.g. K=1,2,3,5 with K=4 missing) makes every K below the gap
    # falsely look top-level (all closed+maximal), silently inflating counts — because
    # compute_level treats a missing K+1 as "no supersets". Only the true Kmax may be
    # top-level, so enforce contiguity over the discovered range.
    kmin, kmax = min(levels), max(levels)
    if sorted(levels) != list(range(kmin, kmax + 1)):
        missing = sorted(set(range(kmin, kmax + 1)) - set(levels))
        raise ValueError(
            f"Non-contiguous K levels {sorted(levels)}: missing {missing}. An interior "
            f"gap makes every K below it falsely appear top-level (all closed+maximal), "
            f"inflating counts. Re-mine or restore the missing per-K parquet(s) first."
        )
    logger.info(f"Discovered K levels: {sorted(levels)} (Kmax={kmax})")

    per_level = []
    for k in sorted(levels):
        kp1 = levels.get(k + 1)
        per_level.append(compute_level(k, levels[k], kp1, chunk_size, rel_tol))

    totals = {
        "total": sum(r["total"] for r in per_level),
        "closed": sum(r["closed"] for r in per_level),
        "maximal": sum(r["maximal"] for r in per_level),
    }

    summary = {
        "results_dir": str(results_dir),
        "k_levels": sorted(levels),
        "kmax": kmax,
        "per_level": per_level,
        "totals": totals,
        "capped_top_level": capped_top_level,
    }

    logger.info("=" * 60)
    logger.info("CLOSED + MAXIMAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total frequent itemsets: {totals['total']:,}")
    logger.info(f"  Closed:                  {totals['closed']:,}")
    logger.info(f"  Maximal:                 {totals['maximal']:,}")
    if capped_top_level:
        logger.warning(
            f"  Run was length-capped at K={kmax}: its {per_level[-1]['closed']:,} closed / "
            f"{per_level[-1]['maximal']:,} maximal are UPPER BOUNDS (frequent K={kmax+1} "
            f"supersets may exist but were not mined)."
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Closed + maximal itemset counts (Major 5)")
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="Directory holding frequent_k{K}.parquet files (or partitioned dirs)",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Write summary JSON here")
    parser.add_argument("--chunk-size", type=int, default=1_000_000, help="Rows per chunk")
    parser.add_argument(
        "--rel-tol", type=float, default=1e-9,
        help="Relative tolerance for closed support-equality (default: %(default)s)",
    )
    parser.add_argument(
        "--capped-top-level", action="store_true",
        help="Flag that the run was length-capped: top-level closed/maximal are upper bounds",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not args.results_dir.exists():
        logger.error(f"results-dir not found: {args.results_dir}")
        return 1

    summary = compute_closed_maximal(
        args.results_dir,
        chunk_size=args.chunk_size,
        rel_tol=args.rel_tol,
        capped_top_level=args.capped_top_level,
    )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2))
        logger.info(f"Summary written to {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
