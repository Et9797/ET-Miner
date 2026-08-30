"""Seeded synthetic transaction generator for tests and GPU benchmarks.

Produces AlphaFold-shaped transaction data (Zipf-skewed item popularity,
Poisson row lengths, optional contiguous nnz clustering, optional planted
motifs) directly as CSR arrays, fully vectorized. Rows are guaranteed to
hold **unique, sorted** items by construction: all (row, item) draws are
deduplicated through one global ``np.unique`` over ``row * vocab + item``
keys — a duplicate-emitting generator would silently diverge the mining
tiers from the efficient-apriori oracle, which collapses duplicates via
index sets.

Planted motifs give oracle-independent ground truth at depths the
pure-Python oracle cannot reach: each motif's items are force-added to a
known set of rows, so the motif and every subset of it must be mined with
support >= the planted count (background noise can only add occurrences —
assertions use >=, never ==).

CLI (box setup pre-generates benchmark datasets)::

    python -m et_miner.synthetic --preset all --out datasets/synth/
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

__all__ = [
    "SynthSpec",
    "GeneratedData",
    "PRESETS",
    "generate_csr",
    "generate_transactions",
    "estimate_level_sizes",
    "check_preset_purpose",
]


@dataclass(frozen=True)
class SynthSpec:
    """Parameters for one synthetic dataset (all randomness from ``seed``)."""

    name: str
    n_rows: int
    vocab_size: int
    #: Zipf exponent for item popularity (weights ∝ 1/rank^zipf_a).
    zipf_a: float
    #: Poisson mean for row lengths, clipped to [row_len_min, row_len_max].
    row_len_mean: float
    row_len_min: int = 1
    row_len_max: int = 64
    #: Contiguous cluster: the first ``skew_frac`` of rows get their length
    #: multiplied by ``skew_mult`` (drives the skewed-rows balance A/B).
    skew_frac: float = 0.0
    skew_mult: float = 1.0
    #: Planted motifs: ``motif_count`` disjoint itemsets of ``motif_size``
    #: items, each force-added to ``motif_penetration`` of all rows.
    motif_count: int = 0
    motif_size: int = 0
    motif_penetration: float = 0.0
    #: The support threshold benchmarks/tests mine this preset at.
    min_support: float = 0.01
    seed: int = 42

    @property
    def min_count(self) -> int:
        return math.ceil(self.min_support * self.n_rows)


class GeneratedData(NamedTuple):
    indptr: np.ndarray  # int64, n_rows + 1
    indices: np.ndarray  # int32 item ids, unique + sorted within each row
    n_rows: int
    n_cols: int
    #: [(motif_items tuple, planted_row_count int), ...] — realized mined
    #: support for each motif and every subset is >= planted_row_count.
    planted: list[tuple[tuple[int, ...], int]]


def _item_weights(spec: SynthSpec) -> np.ndarray:
    w = 1.0 / np.power(np.arange(1, spec.vocab_size + 1, dtype=np.float64), spec.zipf_a)
    return w / w.sum()


def _motif_items(spec: SynthSpec) -> list[np.ndarray]:
    """Disjoint motif itemsets from the mid-tail (not trivially frequent)."""
    if spec.motif_count <= 0 or spec.motif_size <= 0:
        return []
    lo = spec.vocab_size // 10
    hi = max(lo + spec.motif_count * spec.motif_size, spec.vocab_size // 2)
    if hi > spec.vocab_size:
        raise ValueError(f"{spec.name}: vocab too small for {spec.motif_count}×{spec.motif_size} motifs")
    pool = np.arange(lo, hi, dtype=np.int64)
    rng = np.random.default_rng(spec.seed + 1)
    chosen = rng.choice(pool, size=spec.motif_count * spec.motif_size, replace=False)
    return [np.sort(chosen[i * spec.motif_size : (i + 1) * spec.motif_size]) for i in range(spec.motif_count)]


def generate_csr(spec: SynthSpec) -> GeneratedData:
    """Generate the dataset as CSR arrays (vectorized; ~seconds at 2M rows)."""
    rng = np.random.default_rng(spec.seed)
    weights = _item_weights(spec)

    lens = rng.poisson(spec.row_len_mean, size=spec.n_rows)
    if spec.skew_frac > 0 and spec.skew_mult != 1.0:
        n_skew = int(spec.n_rows * spec.skew_frac)
        lens[:n_skew] = np.round(lens[:n_skew] * spec.skew_mult).astype(lens.dtype)
    lens = np.clip(lens, spec.row_len_min, spec.row_len_max)

    total_draws = int(lens.sum())
    row_ids = np.repeat(np.arange(spec.n_rows, dtype=np.int64), lens)
    # Weighted sampling with replacement via inverse-CDF (fast at 10^8 draws);
    # within-row duplicates are removed by the global unique below.
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    draws = np.searchsorted(cdf, rng.random(total_draws), side="right").astype(np.int64)

    # Planted motifs: force-add each motif's items to its chosen rows.
    planted: list[tuple[tuple[int, ...], int]] = []
    extra_rows: list[np.ndarray] = []
    extra_items: list[np.ndarray] = []
    for m_idx, motif in enumerate(_motif_items(spec)):
        n_planted = int(round(spec.motif_penetration * spec.n_rows))
        m_rng = np.random.default_rng(spec.seed + 100 + m_idx)
        rows = m_rng.choice(spec.n_rows, size=n_planted, replace=False).astype(np.int64)
        extra_rows.append(np.repeat(rows, len(motif)))
        extra_items.append(np.tile(motif, n_planted))
        planted.append((tuple(int(x) for x in motif), n_planted))
    if extra_rows:
        row_ids = np.concatenate([row_ids] + extra_rows)
        draws = np.concatenate([draws] + extra_items)

    # Global dedupe + sort: unique (row, item) keys → per-row unique sorted
    # items by construction.
    keys = row_ids * spec.vocab_size + draws
    keys = np.unique(keys)
    out_rows = keys // spec.vocab_size
    out_items = (keys % spec.vocab_size).astype(np.int32)

    counts = np.bincount(out_rows, minlength=spec.n_rows)
    indptr = np.zeros(spec.n_rows + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return GeneratedData(indptr, out_items, spec.n_rows, spec.vocab_size, planted)


def generate_transactions(spec: SynthSpec):
    """The dataset as a Polars DataFrame with an ``items`` list column."""
    import polars as pl
    import pyarrow as pa

    data = generate_csr(spec)
    arrow_list = pa.LargeListArray.from_arrays(data.indptr, data.indices.astype(np.int64))
    df = pl.DataFrame({"items": pl.Series("items", arrow_list)})
    return df, data


def estimate_level_sizes(spec: SynthSpec) -> dict:
    """Cheap analytic self-check: expected K=1 survivors and K=2 candidates.

    Uses the per-item inclusion probability p_i = 1 - (1 - w_i)^E[len]
    (independent-draw approximation — accurate to a few percent for the
    presets here). Motif items additionally appear in their planted rows.
    """
    weights = _item_weights(spec)
    mean_len = float(np.clip(spec.row_len_mean, spec.row_len_min, spec.row_len_max))
    p_item = 1.0 - np.power(1.0 - weights, mean_len)
    exp_counts = spec.n_rows * p_item
    for motif, n_planted in [(m, int(round(spec.motif_penetration * spec.n_rows))) for m in _motif_items(spec)]:
        exp_counts[motif] += n_planted
    n1 = int((exp_counts >= spec.min_count).sum())
    return {
        "expected_k1_survivors": n1,
        "expected_k2_candidates": n1 * (n1 - 1) // 2,
        "min_count": spec.min_count,
    }


def check_preset_purpose(spec: SynthSpec) -> None:
    """Assert a preset still exercises what it exists for. Raises on drift."""
    est = estimate_level_sizes(spec)
    if spec.name == "stress_k2":
        if est["expected_k2_candidates"] <= 100_000_000:
            raise AssertionError(
                f"stress_k2 expects >100M K=2 candidates to exercise the large-filter path, "
                f"got ~{est['expected_k2_candidates']:,}"
            )
    if spec.motif_count > 0:
        planted_count = int(round(spec.motif_penetration * spec.n_rows))
        if planted_count < spec.min_count:
            raise AssertionError(
                f"{spec.name}: planted support {planted_count} < min_count {spec.min_count} — "
                f"motif recovery would pass vacuously"
            )


# ── Presets ────────────────────────────────────────────────────────────────
# smoke is calibrated so the efficient-apriori oracle (pure Python, the
# mandated smoke-run correctness baseline) finishes in a couple of minutes
# while the mining still reaches K>=4. stress_k2 forces >100M K=2
# candidates within 2×24 GB. deep_k plants 10-item motifs (K>=6 mining +
# the measured-density transition mid-run). skewed_rows clusters nnz for
# the balance A/B. oom_regression is sized so its dense counts exceed a
# test-set memory-pool limit, proving budget-driven chunking.

PRESETS: dict[str, SynthSpec] = {
    "smoke": SynthSpec(
        name="smoke",
        n_rows=60_000,
        vocab_size=2_000,
        zipf_a=1.05,
        row_len_mean=9,
        row_len_max=40,
        motif_count=3,
        motif_size=5,
        motif_penetration=0.02,
        min_support=0.01,
        seed=42,
    ),
    "stress_k2": SynthSpec(
        name="stress_k2",
        n_rows=2_000_000,
        vocab_size=35_000,
        zipf_a=0.75,
        row_len_mean=24,
        row_len_max=64,
        min_support=0.000015,
        seed=1337,
    ),
    "deep_k": SynthSpec(
        name="deep_k",
        n_rows=1_000_000,
        vocab_size=4_000,
        zipf_a=1.1,
        row_len_mean=14,
        row_len_max=48,
        motif_count=4,
        motif_size=10,
        motif_penetration=0.03,
        min_support=0.02,
        seed=777,
    ),
    "skewed_rows": SynthSpec(
        name="skewed_rows",
        n_rows=1_000_000,
        vocab_size=4_000,
        zipf_a=1.1,
        row_len_mean=12,
        row_len_max=64,
        skew_frac=0.2,
        skew_mult=5.0,
        motif_count=2,
        motif_size=8,
        motif_penetration=0.03,
        min_support=0.02,
        seed=778,
    ),
    "oom_regression": SynthSpec(
        name="oom_regression",
        n_rows=500_000,
        vocab_size=30_000,
        zipf_a=0.7,
        row_len_mean=20,
        row_len_max=64,
        min_support=0.00003,
        seed=99,
    ),
}


def _main(argv=None) -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Generate synthetic benchmark datasets")
    parser.add_argument("--preset", required=True, choices=[*PRESETS, "all"])
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = list(PRESETS) if args.preset == "all" else [args.preset]
    for name in names:
        spec = PRESETS[name]
        check_preset_purpose(spec)
        df, data = generate_transactions(spec)
        path = out / f"{name}.parquet"
        df.write_parquet(path)
        sidecar = {
            "spec": dataclasses.asdict(spec),
            "planted": [{"items": list(m), "planted_count": c} for m, c in data.planted],
            "estimate": estimate_level_sizes(spec),
            "nnz": int(data.indptr[-1]),
        }
        (out / f"{name}.json").write_text(json.dumps(sidecar, indent=2))
        print(f"  {name}: {spec.n_rows:,} rows, nnz={int(data.indptr[-1]):,} → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
