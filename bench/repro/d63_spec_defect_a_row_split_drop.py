"""Spec Defect A -- "the row-split path drops frequent, apriori-valid itemsets from K=5".

`docs/specs/et_miner_fix_spec.md` records this at severity **high** and presented
it as open, with a reproduction under `docs/recon/` that no longer exists. Two
reviewers refuted it independently at `22cb1dc`, but "two reviewers agreed" is
not a standing gate: it cannot be re-run, and it came from the same process that
produced several factual errors elsewhere in the same report.

This is that gate. `63` is a filename slot, not a BUGS_FOUND number -- Defect A
is a spec item and is not one of the 62.

WHAT "GATED" MEANS -- THE SPEC'S OWN METRIC WAS THE WRONG NOTION
----------------------------------------------------------------
`prune_equal_support=True` returns the frequent **free-sets** (generators): X is
free iff no proper subset has the SAME support (Bastide et al. 2000). It is
deliberately not closed-itemset mining, which asks about equal-support
SUPERSETS and which this engine never implemented.

The spec's "Done when" counted itemsets that were "frequent + closed +
apriori-valid", so it measured something the engine does not compute -- a
closed-based count is non-zero against a generator-based output even when the
output is exactly right. That is part of why the defect looked live.

Checking the immediate (k-1)-subsets is sufficient: support is anti-monotone, so
if any subset ties, some (k-1)-subset on the chain ties too.

THE ORACLE, AND WHY IT IS NOT efficient-apriori AT THIS SCALE
-------------------------------------------------------------
CLAUDE.md makes efficient-apriori the canonical oracle, and it stays the root of
trust here -- but it cannot run at the spec's conditions. Measured on this box,
20,000 rows, max_length 6, the same nested vocabulary:

    14 items  13.5 s      16 items  26.3 s      18 items  47.9 s

which is a doubling per two items. Extrapolated to the spec's 32 items it is
~1.7 h at 20,000 rows and ~10 h at 120,000. Not a gate anyone will run.

So the oracle is TWO TIERS:

  Tier A (the verdict) -- an exhaustive host-side count at the spec's exact
    conditions. Every subset up to max_length is enumerated by DFS with an
    incremental AND: 1,149,016 nodes, 7.6 s. There is NO candidate generation
    and no pruning of any kind, so an itemset missing from this lattice was
    counted and failed the threshold -- it can never have gone ungenerated.

    That property is the reason for exhaustive enumeration rather than a second
    apriori. Defect A is a CANDIDATE-GENERATION defect ("whole prefix groups
    vanish"). An oracle that also builds candidates by prefix-join could share
    the blind spot; one that enumerates every subset cannot.

  Tier B (trust in Tier A) -- efficient-apriori vs the exhaustive counter at
    16 items / 20,000 rows, where it does run. If the two disagree the script
    reports the ORACLE as broken and does not judge the engine at all.

Runtime: about 40 s total plus the mine. It does not need the `slow` marker.
"""

from __future__ import annotations

import itertools
import math
import os
import sys
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# The spec's exact conditions.
N_ROWS = 120_000
N_ITEMS = 32
MIN_SUPPORT = 0.01
MAX_LENGTH = 6
CHUNK_CANDS = "5000"
SEED = 17

# Tier B: the largest configuration in which efficient-apriori is affordable.
VALIDATE_ROWS = 20_000
VALIDATE_ITEMS = 16


def _transactions(n_rows: int, n_items: int, seed: int = SEED):
    """Nested vocabulary: a child item implies its parent.

    The spec calls for this because it is what makes the equal-support prune
    engage at all -- on a flat random vocabulary almost every frequent itemset
    is a generator and the gate becomes a no-op.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    parent = {c: c // 2 for c in range(1, n_items)}
    rows = []
    for _ in range(n_rows):
        r = set(rng.choice(n_items, size=int(rng.integers(4, 10)), replace=False).tolist())
        for _ in range(4):  # close under the implication a few levels up
            for it in list(r):
                p = parent.get(it)
                if p is not None and rng.random() < 0.8:
                    r.add(p)
        rows.append(tuple(sorted(r)))
    return rows


def _pack(rows, n_rows: int, n_items: int):
    import numpy as np

    n_u64 = (n_rows + 63) // 64
    dense = np.zeros((n_items, n_rows), dtype=np.uint8)
    for r, tx in enumerate(rows):
        for it in tx:
            dense[it, r] = 1
    out = np.zeros((n_items, n_u64), dtype=np.uint64)
    for c in range(n_items):
        bits = np.packbits(dense[c], bitorder="little")
        buf = np.zeros(n_u64 * 8, dtype=np.uint8)
        buf[: len(bits)] = bits
        out[c] = buf.view(np.uint64)
    return out


def _exhaustive_lattice(packed, n_items: int, min_count: int, max_len: int):
    """{itemset: count} for EVERY subset up to max_len that meets min_count.

    DFS with an incremental AND down the recursion, so each of the
    sum_k C(n_items, k) nodes costs one AND against its parent's accumulator.
    No generation, no pruning: absence here means "counted and failed".
    """
    import numpy as np

    n_u64 = packed.shape[1]
    scratch = [np.empty(n_u64, dtype=np.uint64) for _ in range(max_len + 2)]
    lattice: dict[tuple[int, ...], int] = {}
    cur: list[int] = []
    nodes = 0

    def rec(start: int, depth: int, acc):
        nonlocal nodes
        for c in range(start, n_items):
            nxt = scratch[depth]
            if acc is None:
                np.copyto(nxt, packed[c])
            else:
                np.bitwise_and(acc, packed[c], out=nxt)
            nodes += 1
            count = int(np.bitwise_count(nxt).sum())
            cur.append(c)
            if count >= min_count:
                lattice[tuple(cur)] = count
            if depth < max_len:
                rec(c + 1, depth + 1, nxt)
            cur.pop()

    rec(0, 1, None)
    return lattice, nodes


def _free_sets(lattice):
    """Generators: no (k-1)-subset of equal support."""
    free = {}
    for iset, count in lattice.items():
        if len(iset) == 1 or not any(
            lattice.get(iset[:i] + iset[i + 1 :]) == count for i in range(len(iset))
        ):
            free[iset] = count
    return free


def _free_sets_bruteforce(lattice):
    """Independent restatement of `_free_sets`, for Tier B to check it against.

    `_free_sets` builds each (k-1)-subset by index slicing, `iset[:i] +
    iset[i+1:]`, and that arithmetic is the whole of it. This gate's verdict is
    not "which itemsets are frequent" -- Tier B already covers that -- but
    "which are generators", and that verdict is produced entirely by the line
    above. An off-by-one there would move the verdict silently, so it is
    restated through `itertools.combinations`, which shares no index arithmetic
    with it.

    Returns `(free, n_absent)`. `n_absent` counts subsets missing from the
    lattice, and must be 0: the exhaustive lattice is downward closed over the
    frequent region, so every (k-1)-subset of a frequent itemset is itself
    frequent and present. `_free_sets` relies on that -- `.get()` returning
    None compares unequal to `count` and quietly votes "free" -- so Tier B
    asserts it rather than inheriting it.
    """
    free = {}
    n_absent = 0
    for iset, count in lattice.items():
        if len(iset) == 1:
            free[iset] = count
            continue
        equal_support_parent = False
        for sub in itertools.combinations(iset, len(iset) - 1):
            got = lattice.get(sub)
            if got is None:
                n_absent += 1
            elif got == count:
                equal_support_parent = True
        if not equal_support_parent:
            free[iset] = count
    return free, n_absent


def _validate_oracle():
    """Tier B -- efficient-apriori vs the exhaustive counter, where both run."""
    try:
        from efficient_apriori import apriori as ea_apriori
    except ImportError:
        return None, "efficient_apriori not installed"

    rows = _transactions(VALIDATE_ROWS, VALIDATE_ITEMS)
    # Deliberately NOT et_miner.core.result._min_count, and this is not an
    # un-done tidy-up: that docstring's "all three implementations must move
    # together" rule is about the three inside the miner. This is the oracle.
    # Importing the miner's helper would let one bug sit in both the code under
    # test and the thing testing it, which is the entire failure mode a
    # separate oracle exists to prevent.
    #
    # `Fraction(str(s))` is the same contract the miner states -- the shortest
    # decimal that round-trips to the float -- computed independently.
    # `math.ceil(MIN_SUPPORT * ROWS)` is wrong on the boundary: 0.07 * 10000 is
    # 700.0000000000001 in binary64 and ceils to 701 where the exact ceiling is
    # 700, dropping an itemset the mandated oracle keeps.
    #
    # LATENT at today's constants, not active: measured, 0.01 * 120_000 and
    # 0.01 * 20_000 are both exact in binary64, so naive and exact agree here
    # and this gate has never miscounted. It is fixed because the next person
    # to edit MIN_SUPPORT should not have to know that.
    min_count = math.ceil(Fraction(str(MIN_SUPPORT)) * VALIDATE_ROWS)
    # Both conventions are load-bearing: the -0.5 makes the float boundary exact
    # for integer counts, and the explicit max_length stops the silent default
    # of 8 from truncating the oracle.
    ea_sets, _ = ea_apriori(
        rows,
        min_support=(min_count - 0.5) / VALIDATE_ROWS,
        min_confidence=1.0,
        max_length=MAX_LENGTH,
    )
    want = {tuple(sorted(i)): int(c) for lvl in ea_sets.values() for i, c in lvl.items()}
    got, _ = _exhaustive_lattice(
        _pack(rows, VALIDATE_ROWS, VALIDATE_ITEMS), VALIDATE_ITEMS, min_count, MAX_LENGTH
    )
    if want != got:
        return False, (
            f"ORACLE DISAGREES with efficient-apriori: "
            f"{len(set(want) - set(got))} missing, {len(set(got) - set(want))} extra"
        )

    # Tier B used to stop here, validating the LATTICE and shipping the
    # generator layer on trust -- and the generator layer is what the verdict
    # below actually consumes. `_free_sets` sat outside the oracle's root of
    # trust while being the last thing to touch both sides of the comparison.
    free_sliced = _free_sets(got)
    free_brute, n_absent = _free_sets_bruteforce(got)
    if n_absent:
        return False, (
            f"ORACLE BROKEN: the exhaustive lattice is not downward closed -- "
            f"{n_absent:,} (k-1)-subsets of frequent itemsets are absent, so "
            "`_free_sets` would score them as generators by default"
        )
    if free_sliced != free_brute:
        only_sliced = sorted(set(free_sliced) - set(free_brute))
        only_brute = sorted(set(free_brute) - set(free_sliced))
        return False, (
            f"ORACLE DISAGREES with itself on generators: "
            f"{len(only_sliced)} only in _free_sets, {len(only_brute)} only in "
            f"the brute-force restatement (e.g. {(only_sliced or only_brute)[:2]})"
        )
    return True, (
        f"oracle validated against efficient-apriori ({len(want):,} itemsets); "
        f"generator layer cross-checked ({len(free_sliced):,} free sets)"
    )


def _mined_free_sets(rows, n_rows: int):
    import polars as pl

    from et_miner import apriori

    # Restored, not leaked: run_all.py runs every repro in ONE process, so a
    # chunk cap left in os.environ silently reshapes whichever gate runs next.
    _prev_cap = os.environ.get("ET_MINER_MAX_CHUNK_CANDS")
    os.environ["ET_MINER_MAX_CHUNK_CANDS"] = CHUNK_CANDS
    try:
        df = pl.DataFrame({"items": [list(r) for r in rows]})
        res = apriori(
            df,
            min_support=MIN_SUPPORT,
            item_col="items",
            use_gpu=True,
            max_length=MAX_LENGTH,
            prune_equal_support=True,
        )
    finally:
        if _prev_cap is None:
            os.environ.pop("ET_MINER_MAX_CHUNK_CANDS", None)
        else:
            os.environ["ET_MINER_MAX_CHUNK_CANDS"] = _prev_cap
    return {
        tuple(sorted(int(i) for i in iset)): round(sup * n_rows)
        for iset, sup in zip(res["itemset"].to_list(), res["support"].to_list())
    }


def reproduce():
    """Returns (is_live, evidence) -- the bench/repro contract."""
    try:
        import cupy as cp
    except ImportError:
        return False, "SKIPPED: no cupy"

    n_devices = cp.cuda.runtime.getDeviceCount()
    if n_devices < 1:
        return False, "SKIPPED: no CUDA device"

    ok, note = _validate_oracle()
    if ok is False:
        # Do not judge the engine with an oracle that failed its own check.
        return True, f"ORACLE BROKEN, engine not assessed -- {note}"

    rows = _transactions(N_ROWS, N_ITEMS)
    # Exact ceiling, computed independently of the miner -- see _validate_oracle.
    min_count = math.ceil(Fraction(str(MIN_SUPPORT)) * N_ROWS)
    lattice, nodes = _exhaustive_lattice(
        _pack(rows, N_ROWS, N_ITEMS), N_ITEMS, min_count, MAX_LENGTH
    )
    want = _free_sets(lattice)
    got = _mined_free_sets(rows, N_ROWS)

    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    cntdiff = sorted(k for k in set(want) & set(got) if want[k] != got[k])

    is_live = bool(missing or extra or cntdiff)
    evidence = (
        f"devices={n_devices} nodes={nodes:,} lattice={len(lattice):,} "
        f"free={len(want):,} mined={len(got):,} "
        f"missing={len(missing)} extra={len(extra)} cntdiff={len(cntdiff)}"
    )
    if missing:
        evidence += f" first_missing={missing[0]}"
    if ok is None:
        evidence += f" [{note}]"
    return is_live, evidence


if __name__ == "__main__":
    live, ev = reproduce()
    print(f"{'LIVE ' if live else 'FIXED'}  {ev}")
    sys.exit(1 if live else 0)
