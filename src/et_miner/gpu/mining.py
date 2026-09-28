"""Host-side level helpers for the row-split GPU miner.

The free-set (non-free) prune, the Apriori group prune, the anchor output
selector and the sortedness check, each with its Rust fast path where one
exists. Import-safe without CuPy or the Rust extension.
"""

from __future__ import annotations

from loguru import logger


def _rust_prune_fn(et_miner_rust, name: str):
    """Resolve a free-set prune entry point, tolerating a pre-0.3.0 wheel.

    0.3.0 renamed ``prune_closed_flat*`` to ``prune_non_free_flat*`` — the test
    was always the free-set one, only the word was wrong. The old name is tried
    second so an unrebuilt wheel keeps working (identical semantics).
    """
    return getattr(et_miner_rust, name, None) or getattr(et_miner_rust, name.replace("non_free", "closed"), None)


def _rows_sorted(flat) -> bool:
    """True iff the rows of an (n, k) integer array are in STRICT lexicographic
    ascending order — vectorized O(n·k), no sort.

    Semantics: for every adjacent pair the first differing column must be
    ascending. Duplicate adjacent rows make ``argmax`` over ``a != b`` return
    column 0, where ``a < b`` is false, so duplicates count as UNSORTED and
    merely trigger a (harmless) redundant sort. Rows are unique itemsets, so
    this is by design, not a defect. Used to skip the level-end lexsort that
    the free-set-prune binary search requires when the level is already sorted.
    """
    import numpy as np

    n = len(flat)
    if n < 2:
        return True
    a = np.asarray(flat[:-1])
    b = np.asarray(flat[1:])
    neq = a != b
    first = neq.argmax(axis=1)
    rows = np.arange(n - 1)
    return bool(np.all(a[rows, first] < b[rows, first]))


def _prune_non_free_mask_python(current_flat, current_counts, prev_flat, prev_counts):
    """Pure-Python free-set keep-mask (True = keep). Dict-based, so it
    does not need prev_flat sorted."""
    import numpy as np

    # Build lookup: bytes(subset) → count. Bytes keys are faster than tuple keys.
    prev_flat_c = np.ascontiguousarray(prev_flat)
    prev_lookup = {}
    for i in range(len(prev_flat_c)):
        prev_lookup[prev_flat_c[i].tobytes()] = int(prev_counts[i])

    k = current_flat.shape[1]
    mask = np.ones(len(current_flat), dtype=bool)

    # For each drop position d, generate all subsets by removing column d
    for d in range(k):
        subsets = np.ascontiguousarray(np.delete(current_flat, d, axis=1))
        for idx in range(len(subsets)):
            if not mask[idx]:
                continue
            prev_count = prev_lookup.get(subsets[idx].tobytes())
            if prev_count is not None and prev_count == int(current_counts[idx]):
                mask[idx] = False

    n_before = len(current_flat)
    n_after = int(mask.sum())
    if n_before > n_after:
        logger.debug(
            f"    Free-set pruning: {n_before:,} → {n_after:,} ({100 * (1 - n_after / n_before):.1f}% non-free removed)"
        )
    return mask


def _prune_non_free_mask(current_flat, current_counts, prev_flat, prev_counts):
    """Free-set keep-mask of a level (True = keep).

    An itemset is a free-set (generator) when no (k-1)-subset has the same
    count [Bastide et al. 2000]. ``prev_flat``/``prev_counts`` must be the
    COMPLETE previous level, or the test misses subsets that were themselves
    pruned and under-prunes. A mask rather than filtered arrays, because the
    sparse-CSR path also carries the survivor index array and filters it in
    lockstep. Rust ``prune_non_free_flat`` when available (``prev_flat`` must be
    row-sorted), else the dict-based Python fallback.
    """
    import numpy as np

    n = len(current_flat)
    if prev_flat is None or prev_counts is None or len(prev_flat) == 0 or n == 0:
        return np.ones(n, dtype=bool)
    try:
        from et_miner.backends import get_rust_ext

        et_miner_rust = get_rust_ext()
        if et_miner_rust is None:
            raise ImportError("et_miner_rust not built")
        _mask_fn = _rust_prune_fn(et_miner_rust, "prune_non_free_flat")
        if _mask_fn is None:
            raise AttributeError("et_miner_rust exposes no free-set prune")
        mask = np.asarray(
            _mask_fn(
                np.ascontiguousarray(current_flat, dtype=np.int32),
                np.ascontiguousarray(current_counts, dtype=np.int64),
                np.ascontiguousarray(prev_flat, dtype=np.int32),
                np.ascontiguousarray(prev_counts, dtype=np.int64),
            ),
            dtype=bool,
        )
        n_after = int(mask.sum())
        if n > n_after:
            logger.debug(
                f"    Free-set pruning: {n:,} → {n_after:,} ({100 * (1 - n_after / n):.1f}% non-free removed) [rust-mask]"
            )
        return mask
    except (ImportError, AttributeError):
        pass
    return _prune_non_free_mask_python(current_flat, current_counts, prev_flat, prev_counts)


def _anchor_keep_mask(current_flat, anchor_col_arr, k):
    """Boolean keep-mask for itemsets containing >= 1 anchor column, or None
    when no filtering applies (no anchors, or nothing to filter).

    K=1 is masked like any other level. The `k < 2` early return this used to
    carry made a two-phase run emit every frequent item at K=1 while filtering
    every deeper level -- an inconsistency that only made sense while anchoring
    was a mining gate. As an output selector it is uniform, and free: the
    generation base is a different array.
    """
    import numpy as np

    if anchor_col_arr is None or len(current_flat) == 0:
        return None
    anchor_mask = np.zeros(len(current_flat), dtype=bool)
    for col in range(k):
        anchor_mask |= np.isin(current_flat[:, col], anchor_col_arr)
    return anchor_mask


def _prune_groups_apriori(groups_info, prev_frequent_set, k, prev_flat_np=None):
    """Prune prefix groups by removing suffix pairs whose (k-1)-subsets are not all frequent.

    Validity is tested per PAIR and then recorded per SUFFIX SLOT: for a
    candidate prefix + [s_i, s_j] the k-2 prefix-drop subsets are checked (the
    two suffix-drop subsets are rows of the level the group was built from, so
    they are frequent by construction), and a valid pair marks BOTH of its
    suffixes as keepers. The group is then rebuilt from every surviving suffix
    with total_candidates recomputed over all C(m,2) pairs among them, which
    re-admits pairs just found invalid.

    So this **over-approximates**: a suffix kept for one valid pair drags every
    other pair in its group along. It is not the per-suffix all-subsets test an
    earlier version of this docstring described — that claim carried a
    verification badge and was wrong about both implementations (this one and
    rust_ext/src/core/groups.rs, which is identical in shape).

    The behaviour is sound, and the surplus is provably one-sided: by
    anti-monotonicity a re-admitted pair cannot reach min_count, so the cost is
    wasted counting and never a wrong output. Fuzzed over 300 random previous
    levels against a brute-force enumeration: 0/300 lost a valid candidate,
    83/300 carried extra ones.

    The GPU dense kernel counts ALL pairs within a group. By removing invalid
    suffixes, we reduce the group sizes and thus the candidate count.

    Rust fast path: HashSet + Rayon parallel, GIL-free. Falls back to Python if
    the Rust extension is not available.

    Which argument is authoritative, and why it matters
    ---------------------------------------------------
    `prev_flat_np` wins whenever it is not None; `prev_frequent_set` is then
    never read. That is not a new rule -- it is what the Rust fast path below
    has always done, since it takes the flat array and returns without touching
    the set. Writing it down turns a silent asymmetry into a contract.

    The consequence is that `prev_frequent_set` may be None. It used to be built
    eagerly by both callers in gpu/row_split.py, at `set(map(tuple,
    prev_full_flat.tolist()))`, and then handed to a Rust call that never looked
    at it. On any build with the extension present it was pure cost. Now the
    Python fallback derives it from `prev_flat_np` at the one place that reads
    it.

    MEASURED, 10M itemsets at k=5, VmHWM delta: **~9 s and ~370 B per itemset**,
    i.e. ~3.7 GB of host RAM for a set nothing reads (370 B x 1e7; decimal GB,
    matching the convention BUGS_FOUND.md uses for the same quantity). Time
    scales with k: ~7 s at k=3, ~12 s at k=7. An earlier revision of this
    docstring claimed ~35 s per level; that number was never measured and is
    ~4x high, and an earlier one said ~3.5 GB, stale from a 350 B/row regime.

    Two significant figures on the BYTES, because the figure is regime- rather
    than run-dependent: three consecutive runs agreed to 0.1 B, so the
    allocator moves it by nothing. An independent re-measurement read 388 B,
    and the 20 B/itemset gap is deterministic, not noise -- it is the 10M x 5
    int32 input (2.0e8 B / 1e7) sitting inside one measurement's baseline and
    outside the other's. What genuinely varies is the ID distribution, and it
    varies by 160 B rather than by a last digit: see the regime note below.

    One significant figure on the TIME, because that one is noisy -- 8.3 to
    10.8 s across runs on the same box.

    The byte figure is quoted with its regime because it is not a constant.
    Item IDs below 257 are CPython singletons, so the tuples share them and the
    same measurement gives ~210 B/itemset. ~370 B is the no-sharing case, which
    is the one this line exists for -- a vocabulary small enough to intern is
    also small enough that the set never gets big.

    No cross-check is performed when both are supplied and disagree: comparing
    them would cost exactly the set this change removes. `prev_flat_np` wins,
    and that is the documented behaviour rather than an accident.

    Args:
        groups_info: K3PlusGroups namedtuple.
        prev_frequent_set: set of tuples of frequent (k-1)-itemsets, or None to
            derive it from `prev_flat_np` if and when the Python fallback runs.
            Ignored entirely when `prev_flat_np` is not None.
        k: current itemset size.
        prev_flat_np: numpy int32 (n_prev, k-1) array. Authoritative when given;
            drives the Rust fast path and, failing that, the fallback's set.

    Raises:
        ValueError: if both `prev_frequent_set` and `prev_flat_np` are None --
            there is then no previous level to resolve subsets against, and
            pruning nothing silently would look like a level with no invalid
            candidates.

    Returns:
        Pruned K3PlusGroups or None if all candidates pruned.
    """
    import numpy as np
    from et_miner.gpu.kernels import K3PlusGroups

    if prev_frequent_set is None and prev_flat_np is None:
        raise ValueError(
            "_prune_groups_apriori needs a previous level: pass prev_flat_np "
            "(preferred) or prev_frequent_set. Both None would prune nothing "
            "and be indistinguishable from a level with no invalid candidates."
        )

    # --- Rust fast path: HashSet + Rayon parallel, GIL-free ---
    if prev_flat_np is not None:
        try:
            from et_miner.backends import get_rust_ext

            et_miner_rust = get_rust_ext()
            if et_miner_rust is None:
                raise ImportError("et_miner_rust not built")

            pf = np.ascontiguousarray(prev_flat_np, dtype=np.int32)
            src_rows = getattr(groups_info, "suffix_src_rows", None)
            result = et_miner_rust.prune_groups_apriori(
                np.ascontiguousarray(groups_info.prefix_items),
                np.ascontiguousarray(groups_info.prefix_offsets),
                np.ascontiguousarray(groups_info.suffixes),
                np.ascontiguousarray(groups_info.suffix_offsets),
                np.ascontiguousarray(groups_info.cumulative_pairs),
                int(groups_info.total_candidates),
                pf,
                None if src_rows is None else np.ascontiguousarray(src_rows, dtype=np.int64),
            )
            if result is None:
                return None
            if len(result) != 7:  # pre-0.2.0 wheel would also have rejected the 8th argument
                raise TypeError("prune_groups_apriori returned a 6-tuple")
            pi, po, sf, so, cp_arr, sr, tc = result
            return K3PlusGroups(
                prefix_items=np.asarray(pi),
                prefix_offsets=np.asarray(po),
                suffixes=np.asarray(sf),
                suffix_offsets=np.asarray(so),
                cumulative_pairs=np.asarray(cp_arr),
                total_candidates=int(tc),
                groups=groups_info.groups,
                suffix_src_rows=None if src_rows is None else np.asarray(sr, dtype=np.int64),
            )
        except (ImportError, AttributeError) as exc:
            # Raised above for a missing extension, and by getattr on a wheel
            # too old to carry the symbol. Both leave the answer intact and
            # both cost ~9x on this level, so neither may be silent.
            from et_miner.gpu.kernels.k3plus import _warn_missing_rust_once

            _warn_missing_rust_once(f"_prune_groups_apriori ({exc})")
        except TypeError:  # stale wheel: no suffix_src_rows parameter / 6-tuple result
            from et_miner.gpu.kernels.k3plus import _warn_stale_rust_once

            _warn_stale_rust_once("prune_groups_apriori has no suffix_src_rows")

    # --- Python fallback ---
    # The only reader of prev_frequent_set. Derive it here rather than at the
    # call sites, so the Rust path above never pays for a set it does not read.
    #
    # Derived whenever prev_flat_np is given, NOT only when the set is None:
    # the precedence documented above has to hold on a build without the Rust
    # extension too. Honouring a passed-in set here would make the answer
    # depend on whether the wheel happened to be present -- the exact
    # two-paths-to-one-answer shape this change is meant to remove. Every
    # in-tree caller that passes both derives the set from the same array, so
    # this is a no-op for them.
    if prev_flat_np is not None:
        prev_frequent_set = set(map(tuple, prev_flat_np.tolist()))

    prefix_items = groups_info.prefix_items
    prefix_offsets = groups_info.prefix_offsets
    suffixes = groups_info.suffixes
    suffix_offsets = groups_info.suffix_offsets
    src_rows = getattr(groups_info, "suffix_src_rows", None)
    n_groups = len(suffix_offsets) - 1

    new_pi, new_po, new_sf, new_so, new_sr = [], [0], [], [0], []
    new_total = 0
    new_cp = [0]  # MUST start with 0 — every consumer assumes cumulative_pairs[0] == 0

    for g in range(n_groups):
        pstart, pend = int(prefix_offsets[g]), int(prefix_offsets[g + 1])
        sstart, send = int(suffix_offsets[g]), int(suffix_offsets[g + 1])
        prefix = tuple(int(x) for x in prefix_items[pstart:pend])
        gsuf = [int(x) for x in suffixes[sstart:send]]
        # Per-SLOT validity (not a set of values) so suffix_src_rows can be
        # filtered slot-for-slot alongside the suffixes.
        valid = [False] * len(gsuf)

        if k == 3:
            # EXACT: for K=3, prefix has 1 element. Only check = (s_i, s_j) in prev_set.
            for i in range(len(gsuf)):
                for j in range(i + 1, len(gsuf)):
                    if (gsuf[i], gsuf[j]) in prev_frequent_set:
                        valid[i] = True
                        valid[j] = True
        else:
            # CONSERVATIVE: for K>=4, check k-2 subsets (drop each prefix element).
            # Keep suffixes that participate in at least one valid pair.
            for i in range(len(gsuf)):
                for j in range(i + 1, len(gsuf)):
                    candidate = prefix + (gsuf[i], gsuf[j])
                    all_freq = True
                    for d in range(len(prefix)):
                        subset = candidate[:d] + candidate[d + 1 :]
                        if subset not in prev_frequent_set:
                            all_freq = False
                            break
                    if all_freq:
                        valid[i] = True
                        valid[j] = True

        kept = sorted((idx for idx in range(len(gsuf)) if valid[idx]), key=lambda idx: gsuf[idx])
        if len(kept) >= 2:
            new_pi.extend(prefix)
            new_po.append(len(new_pi))
            new_sf.extend(gsuf[idx] for idx in kept)
            new_so.append(len(new_sf))
            if src_rows is not None:
                new_sr.extend(int(src_rows[sstart + idx]) for idx in kept)
            n_pairs = len(kept) * (len(kept) - 1) // 2
            new_total += n_pairs
            new_cp.append(new_total)

    if new_total == 0:
        return None

    return K3PlusGroups(
        prefix_items=np.array(new_pi, dtype=np.int32),
        prefix_offsets=np.array(new_po, dtype=np.int64),
        suffixes=np.array(new_sf, dtype=np.int32),
        suffix_offsets=np.array(new_so, dtype=np.int64),
        cumulative_pairs=np.array(new_cp, dtype=np.int64),
        total_candidates=new_total,
        groups=groups_info.groups,  # preserve original group metadata
        suffix_src_rows=None if src_rows is None else np.array(new_sr, dtype=np.int64),
    )
