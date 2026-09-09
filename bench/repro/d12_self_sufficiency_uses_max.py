"""#12 -- compute_self_sufficiency divides by max where its own semantics need min.

The docstring says a ratio close to 1.0 means "the K-th item adds almost no
information beyond what the (K-1)-subset already captures -- the itemset is
near-closed and can be filtered out", and that "ratios well below 1.0 indicate
genuine combinatorial signal".

Since support_K <= support(W) for every (K-1)-subset W, requiring
support_K == max(subset supports) forces ALL subsets to share a support -- a
degenerate corner, not the near-closed family described. The condition actually
described is support_K == min(subset supports).

So a maximally redundant itemset scores low and reads as "genuine signal":
anyone filtering on this ratio keeps the redundancy and discards the signal.

Recalibrating the cutoff is not available, because under max the ratio is not
monotone in the property described: two itemsets that are EQUALLY redundant by
the engine's own predicate score differently. It is the wrong kind of
aggregation, not a mis-scaled one.

The package already implements the correct predicate twice, both as the min
test (core/apriori.py's _prune_equal_support, and groups.rs).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def reproduce() -> tuple[bool, str]:
    sys.path.insert(0, str(REPO / "src"))
    import polars as pl

    from et_miner.core.rules import compute_self_sufficiency

    # {1,2,3} at 0.30 with subsets {1,2}=0.30, {1,3}=0.90, {2,3}=0.95.
    # Item 3 is fully implied by {1,2}: maximally redundant.
    with tempfile.TemporaryDirectory() as td:
        k3 = Path(td) / "k3.parquet"
        k2 = Path(td) / "k2.parquet"
        pl.DataFrame({"itemset": [[1, 2, 3]], "support": [0.30]}).write_parquet(k3)
        pl.DataFrame(
            {"itemset": [[1, 2], [1, 3], [2, 3]], "support": [0.30, 0.90, 0.95]}
        ).write_parquet(k2)
        out = compute_self_sufficiency(k3, k2)

    ratio = out["self_sufficiency_ratio"][0]
    agg = out["min_k_minus1_support"][0]

    # Under min the ratio is 1.0 -- "near-closed, filter it out". Under max it
    # is 0.30/0.95 = 0.3158 -- "genuine combinatorial signal". Exactly backwards.
    live = abs(ratio - 1.0) > 1e-9
    return live, (
        f"maximally redundant itemset: aggregate={agg:.4f}, ratio={ratio:.4f} "
        f"(min-based would be {0.30 / 0.30:.4f}); "
        f"{'reads as genuine signal -- backwards' if live else 'reads as near-closed'}"
    )


if __name__ == "__main__":
    live, ev = reproduce()
    print(("LIVE  " if live else "FIXED ") + ev)
    raise SystemExit(0 if live else 1)
