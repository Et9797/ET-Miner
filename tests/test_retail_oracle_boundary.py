"""The benchmark's oracle route keeps the integer support boundary at Online Retail's scale."""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

from consolidation_run import _mine
from child_run import result_signatures


def test_retail_oracle_route_keeps_the_integer_boundary():
    from et_miner import apriori

    # Same row count and threshold counts as Online Retail, with a small
    # vocabulary so both supports can be checked exactly in CPU CI.
    df = pl.DataFrame({"items": [[0, 1, 2]] * 4 + [[3, 4, 5]] * 2 + [[99]] * (36_422 - 6)})
    for s in (0.0001, 0.00005):
        oracle = _mine({"route": "EA"}, df, s, 6, None, None, {})
        cpu = apriori(df, min_support=s, max_length=6)
        assert result_signatures(oracle, df.height) == result_signatures(cpu, df.height)
