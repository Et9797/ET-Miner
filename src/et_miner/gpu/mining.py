"""Single-GPU bitvec mining: the GPU-resident Apriori inner loops.

Holds the bitvec-based mining paths (_apriori_from_bitvecs and the
GPU-resident variant), tidset conversion, and the GPU-side pruning helpers.
Import-safe without CuPy — cupy and the CUDA kernels are imported inside
functions, so a CPU-only install can import this module and fails with a
clear error only when a GPU path is actually invoked.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

import polars as pl
from loguru import logger

from et_miner.core.profiling import ProfilingSession
from et_miner.core.result import (
    _build_result_df,
    _empty_result,
    _min_count,
)
from et_miner.gpu.density import DENSITY_CROSSOVER, SPARSE_AUTO, should_transition_to_sparse




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


def _prune_non_free_flat(current_flat, current_counts, prev_flat, prev_counts):
    """Keep only the free-sets (generators) of a level.

    An itemset is a free-set when no proper subset has the same support
    [Bastide et al. 2000, Pascal]. If dropping one item leaves the count
    unchanged, that item is implied by the rest and the itemset is not free.
    Free-sets are anti-monotone (every subset of a free-set is free), so the
    next level can be generated from the survivors alone without losing any
    free-set — which is why the caller may hand the survivors forward.

    NOT closed-itemset mining: closed asks about equal-support *supersets*,
    this asks about equal-support *subsets*. ``prev_flat``/``prev_counts`` must
    be the COMPLETE previous level, not a previously pruned one, or the test
    misses subsets that were themselves pruned and under-prunes.

    Operates on raw integer counts (not float support) for exact comparison.
    Uses bytes-key dict for O(1) lookup of (k-1)-subsets.

    Args:
        current_flat: numpy int32 (n, k) — current level frequent itemsets.
        current_counts: numpy int64 (n,) — raw support counts.
        prev_flat: numpy int32 (m, k-1) — previous level frequent itemsets.
        prev_counts: numpy int64 (m,) — raw support counts for previous level.

    Returns:
        Tuple of (pruned_flat, pruned_counts) with the non-free itemsets removed.
    """
    import numpy as np

    if prev_flat is None or prev_counts is None or len(prev_flat) == 0 or len(current_flat) == 0:
        return current_flat, current_counts

    # --- Rust fast path: HashMap + Rayon parallel, GIL-free ---
    try:
        from et_miner.backends import get_rust_ext

        et_miner_rust = get_rust_ext()
        if et_miner_rust is None:
            raise ImportError("et_miner_rust not built")

        cf = np.ascontiguousarray(current_flat, dtype=np.int32)
        cc = np.ascontiguousarray(current_counts, dtype=np.int64)
        pf = np.ascontiguousarray(prev_flat, dtype=np.int32)
        pc = np.ascontiguousarray(prev_counts, dtype=np.int64)
        n_before = len(current_flat)
        k = current_flat.shape[1]

        # Compact path returns pruned arrays directly, skipping the
        # single-threaded numpy fancy-index that bottlenecked at 30-60s on 430M
        # K=6 rows. Sequential extend_from_slice in Rust ~10-13× faster.
        _compact = _rust_prune_fn(et_miner_rust, "prune_non_free_flat_compact")
        if _compact is not None:
            flat_1d, pruned_counts, n_kept = _compact(cf, cc, pf, pc)
            if n_before > n_kept:
                logger.debug(
                    f"    Free-set pruning: {n_before:,} → {n_kept:,} ({100 * (1 - n_kept / n_before):.1f}% non-free removed) [rust-compact]"
                )
            # Reshape (n_kept * k,) → (n_kept, k) — zero-copy view on
            # C-contiguous source. Empty case yields (0, k) shape, not (0,),
            # so prev_frequent_flat.shape[1] stays k for the next K level.
            return flat_1d.reshape((n_kept, k)), pruned_counts

        # Legacy path: bool mask + Python fancy-index (slow at high K)
        _mask_fn = _rust_prune_fn(et_miner_rust, "prune_non_free_flat")
        if _mask_fn is None:
            raise AttributeError("et_miner_rust exposes no free-set prune")
        mask = _mask_fn(cf, cc, pf, pc)
        n_after = int(mask.sum())
        if n_before > n_after:
            logger.debug(
                f"    Free-set pruning: {n_before:,} → {n_after:,} ({100 * (1 - n_after / n_before):.1f}% non-free removed) [rust-mask]"
            )
        return current_flat[mask], current_counts[mask]
    except (ImportError, AttributeError):
        pass

    # --- Python fallback ---
    mask = _prune_non_free_mask_python(current_flat, current_counts, prev_flat, prev_counts)
    return current_flat[mask], current_counts[mask]


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
    """Keep-mask form of :func:`_prune_non_free_flat` (True = keep / open), for
    callers that must filter more than the two arrays in lockstep (the
    sparse-CSR path also carries the survivor index array). Rust
    ``prune_non_free_flat`` when available (prev_flat must be row-sorted),
    else the dict-based Python fallback."""
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

    MEASURED, 10M itemsets at k=5, VmHWM delta: **~8.9 s and ~367 B per
    itemset**, i.e. ~3.5 GB of host RAM for a set nothing reads. Time scales
    with k: 7.2 s at k=3, 12.0 s at k=7. An earlier revision of this docstring
    claimed ~35 s per level; that number was never measured and is ~4x high.

    The byte figure is quoted with its regime because it is not a constant.
    Item IDs below 257 are CPython singletons, so the tuples share them and the
    same measurement gives ~208 B/itemset. 367 B is the no-sharing case, which
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
        except (ImportError, AttributeError):
            pass
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


def _apriori_from_bitvecs(
    bitvecs_gpu,  # CuPy array shape (n_cols, ceil(n_rows/64)) dtype uint64
    col_to_item: dict[int, int],  # column index -> item ID
    n_transactions: int,
    min_support: float,
    max_length: int | None,
    batch_size: int | None,
    profile: bool,
    level_callback: Callable[[int, int, int, float], None] | None,
    n_gpus: int = 1,  # honoured: forwarded to dispatch as a cap, see gpu/dispatch._resolve_gpus
    max_ram_gb: float = 800.0,
    max_vram_gb: float = 70.0,
    sparse_from_k: int | str | None = None,
) -> pl.DataFrame | tuple[pl.DataFrame, ProfilingSession]:
    """Run Apriori directly from pre-built GPU bitvectors.

    This is the fast path for streaming GPU processing where bitvectors
    are already built in GPU memory. Avoids DataFrame conversion overhead.

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s) with packed bits.
        col_to_item: Mapping from column index to original item ID.
        n_transactions: Total number of transactions (for support calculation).
        min_support: Minimum support threshold (0.0-1.0).
        max_length: Maximum itemset length (None = unlimited).
        batch_size: Unused. The dense levels size their own chunks from the
            measured VRAM budget (gpu.row_split_chunks.compute_chunk_budget);
            the parameter is kept only so the positional call sites in
            core.apriori keep working.
        profile: If True, return profiling metrics alongside results.
        level_callback: Optional callback for per-level progress updates.
        sparse_from_k: Dense→sparse CSR transition. Int = fixed K-level
            (floored to 3), "auto" = transition when the previous level's
            measured mean support falls below the byte-cost crossover
            (n_transactions/32 — see et_miner.gpu.density), None = never.

    Returns:
        If profile=False: DataFrame with columns [itemset, support].
        If profile=True: Tuple of (DataFrame, ProfilingSession).
    """
    import numpy as np

    try:
        import cupy as cp
    except ImportError:
        raise ImportError("CuPy is required for bitvecs parameter. Install with: pip install cupy-cuda12x")

    # int32 tidset indices can't represent transaction IDs > 2^31
    if n_transactions > np.iinfo(np.int32).max:
        raise ValueError(
            f"n_transactions={n_transactions:,} exceeds int32 max. CSR tidset indices require int64 upgrade."
        )

    from et_miner.gpu.kernels import build_k3plus_groups_from_flat, decode_k3plus_flat, get_popcount_kernel
    from et_miner.gpu.sparse_csr import (
        SparseMiningState,
        convert_shards_to_csr,
        free_groups,
        log_new_shards,
        materialize_survivors,
        run_sparse_level,
        upload_groups_to_shards,
    )

    session = ProfilingSession() if profile else None

    n_cols, n_u64s = bitvecs_gpu.shape
    min_count_threshold = _min_count(min_support, n_transactions)

    # Effective max length: can't exceed number of columns
    effective_max_length = min(
        max_length if max_length else float("inf"),
        n_cols,
    )

    def _get_memory_gb():
        """Get (system_ram_gb, gpu_vram_gb) usage."""
        ram_gb = 0.0
        vram_gb = 0.0
        try:
            import psutil

            ram_gb = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        except ImportError:
            pass
        try:
            pool = cp.get_default_memory_pool()
            vram_gb = pool.used_bytes() / (1024**3)
        except Exception as e:  # noqa: BLE001 — reported, never swallowed
            # This used to be a bare `pass`, leaving vram_gb at 0.0, so the VRAM
            # half of the guard below could never fire at all. A guard that
            # cannot trip is worse than no guard: it reads as protection.
            logger.warning(f"    VRAM usage unavailable ({e}); the VRAM memory guard is inactive")
        return ram_gb, vram_gb

    def _log_level(k, n_cand, n_freq, elapsed_s, cumulative_itemsets):
        ram_gb, vram_gb = _get_memory_gb()
        msg = (
            f"  K={k}: candidates={n_cand:,} → frequent={n_freq:,} "
            f"({elapsed_s:.1f}s) | cumulative={cumulative_itemsets:,} | "
            f"RAM={ram_gb:.1f}GB VRAM={vram_gb:.1f}GB"
        )
        logger.info(msg)

    def _check_memory_guard(k, cumulative_itemsets):
        """Raise if a memory guard has tripped.

        This used to return True and let the caller `break` out of the level
        loop with only a logger.warning, falling straight through to
        _build_result_df -- so the caller received a DataFrame that stopped at
        some K, with nothing on it to say so. Measured: max_ram_gb=0.0 returned
        249 itemsets where the complete answer is 31,160, a 99.2% loss, with no
        exception and a perfectly ordinary-looking frame.

        It needs the user to set a low guard, so it is not silent by default --
        but the guard's whole purpose is to be set, and when it fires the
        contract break is total. A miner whose contract is exactness must raise
        or return an explicit partial signal, never a bare truncated result.
        This is the same failure mode as the result-buffer clamps: an
        incomplete lattice handed back through a value-returning API.
        """
        ram_gb, vram_gb = _get_memory_gb()
        for used, limit, what in ((ram_gb, max_ram_gb, "RAM"), (vram_gb, max_vram_gb, "VRAM")):
            if used > limit:
                # Only remedies reachable on THIS route. output_dir is refused
                # here by _validate_route_support -- the per-K flush belongs to
                # the row-split miner -- so advising it unqualified sent the
                # user to a ValueError after an hours-long run.
                raise MemoryError(
                    f"Memory guard tripped at K={k}: {what}={used:.1f}GB exceeds the "
                    f"max_{what.lower()}_gb={limit}GB limit after {cumulative_itemsets:,} "
                    f"itemsets. The lattice is INCOMPLETE at this point, so it is not "
                    f"returned. Raise max_{what.lower()}_gb, lower max_length, or "
                    f"re-run on the row-split miner (n_gpus>1 or "
                    f"prune_equal_support=True), which accepts output_dir and "
                    f"flushes each level as it completes."
                )

    _t_total_start = time.perf_counter()
    _cumulative_itemsets = 0
    logger.info(
        f"APRIORI BITVEC: {n_cols} cols, {n_transactions:,} txns, "
        f"min_support={min_support:.8f} (min_count={min_count_threshold}), "
        f"max_length={effective_max_length}"
    )

    # Phase 1: k=1 support counting via popcount on each column
    _k1_start = time.perf_counter()
    if session:
        session.start_phase("k1_support")

    # Popcount each column to get 1-itemset support
    # Each column is a bitvector of shape (n_u64s,)
    results: list[tuple[list[int], float]] = []
    prev_frequent: list[tuple[int, ...]] = []  # Use int indices, not str column names
    prev_counts: dict[tuple[int, ...], int] = {}

    # GPU popcount using __popcll hardware intrinsic (183x faster than Python)
    popcount_kernel = get_popcount_kernel()
    popcounts = popcount_kernel(bitvecs_gpu.view(cp.uint64))
    col_counts = cp.sum(popcounts.reshape(n_cols, -1), axis=1, dtype=cp.int64).get()

    # Filter frequent 1-itemsets
    for col_idx in range(n_cols):
        count = col_counts[col_idx]
        if count >= min_count_threshold:
            item_id = col_to_item[col_idx]
            support = count / n_transactions
            results.append(([item_id], support))
            prev_frequent.append((col_idx,))
            prev_counts[(col_idx,)] = count

    if session:
        session.end_phase(n_frequent=len(prev_frequent))

    # Call level callback for k=1
    _k1_elapsed = time.perf_counter() - _k1_start
    _cumulative_itemsets += len(prev_frequent)
    _log_level(1, n_cols, len(prev_frequent), _k1_elapsed, _cumulative_itemsets)
    if level_callback:
        level_callback(1, n_cols, len(prev_frequent), _k1_elapsed * 1000)

    # Pathological support warning
    freq_ratio = len(prev_frequent) / n_cols if n_cols > 0 else 0.0
    if freq_ratio > 0.90 or min_count_threshold <= 1:
        import warnings

        warnings.warn(
            f"{freq_ratio:.0%} of items are frequent (min_count={min_count_threshold}). "
            f"Mining may produce combinatorially explosive results. "
            f"Consider raising min_support.",
            stacklevel=3,
        )

    if not prev_frequent:
        if profile:
            return _empty_result(), session
        return _empty_result()

    # Phase 2: k >= 2 using CUDA kernels
    k = 2

    # Sparse CSR state (GPU-resident shard, see gpu.sparse_csr); sticky once
    # active. prev_frequent_flat mirrors prev_frequent in the shard's row order.
    sparse_state = SparseMiningState()
    prev_frequent_flat = None
    col_to_item_arr = np.zeros(n_cols, dtype=np.int64)
    for _c, _item in col_to_item.items():
        col_to_item_arr[_c] = _item

    while k <= effective_max_length and len(prev_frequent) >= k:
        _k_start = time.perf_counter()

        # Dense→sparse transition: fixed K-level or measured density ("auto").
        # One-way — the bitvecs are freed below, tidset_offsets keeps it sticky.
        _go_sparse = False
        if not sparse_state.active and sparse_from_k is not None:
            _mean_count = None
            if sparse_from_k == SPARSE_AUTO and prev_counts:
                _mean_count = sum(prev_counts.values()) / len(prev_counts)
            _go_sparse = should_transition_to_sparse(
                sparse_from_k, k, n_transactions=n_transactions, mean_count=_mean_count
            )

        if _go_sparse:
            _trigger = (
                f"measured mean support {_mean_count / n_transactions:.4%} < {DENSITY_CROSSOVER:.4%} crossover"
                if sparse_from_k == SPARSE_AUTO
                else f"fixed sparse_from_k={sparse_from_k}"
            )
            logger.info(f"  ═══ DENSITY TRANSITION at K={k} ({_trigger}): dense bitvec → sparse CSR ═══")
            prev_frequent_flat = np.array(prev_frequent, dtype=np.int32)
            prev_counts_flat = (
                np.array([prev_counts[t] for t in prev_frequent], dtype=np.int64) if prev_counts else None
            )
            sparse_state.shards = convert_shards_to_csr(
                [(bitvecs_gpu, int(bitvecs_gpu.device.id), n_transactions)], prev_frequent_flat, prev_counts_flat
            )
            # `bitvecs_gpu` is this function's PARAMETER -- the caller still
            # holds the array, so `del` drops one reference of two and frees
            # nothing. The old message claimed the VRAM was freed. #30.
            #
            # The pool call stays (it is not scoped to allocations made here,
            # and the CSR conversion above did allocate); only the claim goes.
            del bitvecs_gpu
            cp.get_default_memory_pool().free_all_blocks()
            logger.debug(
                f"    CSR shard built with {sparse_state.shards[0].nnz:,} tid entries; "
                "the bitvecs are the caller's and were not released"
            )

        if sparse_state.active:
            # ═══ SPARSE CSR LEVEL (GPU-resident shard, see gpu.sparse_csr) ═══
            groups_info = build_k3plus_groups_from_flat(prev_frequent_flat, with_src_rows=True)

            current_frequent: list[tuple[int, ...]] = []
            current_counts: dict[tuple[int, ...], int] = {}
            _csr_cands = 0

            if groups_info is not None and groups_info.total_candidates > 0:
                _csr_cands = groups_info.total_candidates
                logger.info(f"  K={k}: {_csr_cands:,} candidates (CSR sparse mode)")
                groups_gpu = upload_groups_to_shards(groups_info, sparse_state.shards)
                try:
                    surv, counts = run_sparse_level(
                        sparse_state.shards,
                        groups_info,
                        groups_gpu,
                        min_count_threshold,
                        nccl_comms=None,
                        use_nccl=False,
                        level_label=f"K={k}",
                    )
                    n_freq = len(surv)
                    if n_freq > 0:
                        current_flat = decode_k3plus_flat(surv, groups_info, k)
                        items = col_to_item_arr[current_flat]
                        results.extend(zip(items.tolist(), (counts / n_transactions).tolist()))
                        current_frequent = [tuple(r) for r in current_flat.tolist()]
                        current_counts = dict(zip(current_frequent, counts.tolist()))
                        # Rebuild the shard for K+1 in the same row order (skip at max_length)
                        if k < effective_max_length and n_freq >= k + 1:
                            sparse_state.replace(
                                materialize_survivors(
                                    sparse_state.shards, groups_gpu, surv, counts, level_label=f"K={k}"
                                )
                            )
                            log_new_shards(sparse_state.shards, n_freq)
                        prev_frequent_flat = current_flat
                finally:
                    free_groups(groups_gpu)

            _k_elapsed = time.perf_counter() - _k_start
            _cumulative_itemsets += len(current_frequent)
            _log_level(k, _csr_cands, len(current_frequent), _k_elapsed, _cumulative_itemsets)
            if level_callback:
                level_callback(k, _csr_cands, len(current_frequent), _k_elapsed * 1000)

        # === FUSED K=2 FAST PATH ===
        # Single kernel launch: pair gen + AND + popcount + filter on GPU.
        # Eliminates all Python overhead (28.8M tuples, numpy arrays, flatten loop).
        elif k == 2:
            freq_cols = sorted(p[0] for p in prev_frequent)
            n_pairs = len(freq_cols) * (len(freq_cols) - 1) // 2

            if session:
                session.start_phase("k2_fused_gpu")

            from et_miner.gpu.dispatch import dispatch_k2

            pairs, counts = dispatch_k2(
                bitvecs_gpu, freq_cols, n_u64s, min_count_threshold, n_gpus=n_gpus
            )

            if session:
                session.end_phase(n_candidates=n_pairs, n_frequent=len(pairs))

            current_frequent: list[tuple[int, ...]] = []
            current_counts: dict[tuple[int, ...], int] = {}

            for idx, (col_i, col_j) in enumerate(pairs):
                count = int(counts[idx])
                support = count / n_transactions
                item_list = [col_to_item[col_i], col_to_item[col_j]]
                results.append((item_list, support))
                current_frequent.append((col_i, col_j))
                current_counts[(col_i, col_j)] = count

            # Pair cache infrastructure ready but not yet wired to K>=3 kernels.
            # Disabled to avoid 13.6 GB VRAM waste. Re-enable when kernel integration is done.

            _k_elapsed = time.perf_counter() - _k_start
            _cumulative_itemsets += len(current_frequent)
            _log_level(k, n_pairs, len(current_frequent), _k_elapsed, _cumulative_itemsets)
            if level_callback:
                level_callback(k, n_pairs, len(current_frequent), _k_elapsed * 1000)

        else:
            # === FULLY-FUSED K>=3 PATH ===
            # Candidate generation + count + filter ALL on GPU in ONE kernel.
            # Prefix groups built on CPU (O(n_frequent)), transferred to GPU,
            # candidates generated on-the-fly via triangular number inverse.
            # No Apriori pruning needed: anti-monotone property guarantees
            # non-frequent candidates fail min_count check in-kernel.
            if session:
                session.start_phase(f"k{k}_fully_fused_gpu")

            from et_miner.gpu.dispatch import dispatch_k3plus_fused

            # Estimate candidate count from prefix groups (level logging only)
            _prefix_groups: dict[tuple, int] = {}
            for itemset in prev_frequent:
                _p = itemset[:-1]
                _prefix_groups[_p] = _prefix_groups.get(_p, 0) + 1
            _est_cands = sum(g * (g - 1) // 2 for g in _prefix_groups.values())

            frequent_candidates, counts = dispatch_k3plus_fused(
                bitvecs_gpu, prev_frequent, k, n_u64s, min_count_threshold, n_gpus=n_gpus
            )

            # Build results from frequent candidates (already filtered by kernel)
            current_frequent = []
            current_counts = {}

            for idx, candidate in enumerate(frequent_candidates):
                count = int(counts[idx])
                support = count / n_transactions
                item_list = [col_to_item[c] for c in candidate]
                results.append((item_list, support))
                current_frequent.append(candidate)
                current_counts[candidate] = count

            if session:
                session.end_phase(n_frequent=len(current_frequent))

            _k_elapsed = time.perf_counter() - _k_start
            _cumulative_itemsets += len(current_frequent)
            _log_level(k, _est_cands, len(current_frequent), _k_elapsed, _cumulative_itemsets)
            if level_callback:
                level_callback(k, len(frequent_candidates), len(current_frequent), _k_elapsed * 1000)

        if not current_frequent:
            logger.info(
                f"  K={k}: 0 frequent — EXHAUSTED. Total: {_cumulative_itemsets:,} itemsets in {time.perf_counter() - _t_total_start:.1f}s"
            )
            break

        # Memory guard: check BEFORE starting the next level. Raises rather
        # than breaking -- see _check_memory_guard.
        _check_memory_guard(k, _cumulative_itemsets)

        prev_frequent = current_frequent
        prev_counts = current_counts
        k += 1
    else:
        # While loop ended because k > effective_max_length or not enough frequent items
        logger.info(
            f"  Mining complete: K={k - 1} reached max_length={effective_max_length}. "
            f"Total: {_cumulative_itemsets:,} itemsets "
            f"in {time.perf_counter() - _t_total_start:.1f}s"
        )

    sparse_state.release()
    result_df = _build_result_df(results)
    if profile:
        return result_df, session
    return result_df


def _build_results_from_gpu(gpu_results, col_to_item, n_transactions):
    """Bulk transfer GPU results to CPU and build result list.

    Single PCIe transfer per K-level at the end, using vectorized numpy
    col_to_item mapping instead of per-element Python dict lookups.

    Args:
        gpu_results: List of (itemsets_gpu, counts_gpu) CuPy arrays per K-level.
        col_to_item: Dict mapping column index to item ID.
        n_transactions: Total number of transactions.

    Returns:
        List of (item_list, support) tuples.
    """
    import numpy as np

    # Build vectorized lookup array: col_idx -> item_id
    if col_to_item:
        max_col = max(col_to_item.keys())
        col_lookup = np.zeros(max_col + 1, dtype=np.int64)
        for col_idx, item_id in col_to_item.items():
            col_lookup[col_idx] = item_id
    else:
        col_lookup = np.array([], dtype=np.int64)

    results = []
    for itemsets_gpu, counts_gpu in gpu_results:
        # Bulk .get() — one PCIe transfer per level
        itemsets_np = itemsets_gpu.get()  # (n, k) int32
        counts_np = counts_gpu.get()  # (n,) int64

        # Vectorized col->item mapping
        item_ids = col_lookup[itemsets_np]  # (n, k) int64

        for i in range(len(counts_np)):
            support = counts_np[i] / n_transactions
            results.append((item_ids[i].tolist(), support))

    return results


def _apriori_from_bitvecs_gpu_resident(
    bitvecs_gpu,
    col_to_item: dict[int, int],
    n_transactions: int,
    min_support: float,
    max_length: int | None,
    profile: bool,
    level_callback: Callable[[int, int, int, float], None] | None,
    n_gpus: int = 1,
) -> "pl.DataFrame | tuple[pl.DataFrame, ProfilingSession]":
    """Fully GPU-resident Apriori: all frequent itemsets stay in VRAM.

    Zero PCIe round trips per K-level (only ~12 bytes for loop control).
    Single bulk transfer at the end to build the result DataFrame.

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s) with packed bits.
        col_to_item: Mapping from column index to original item ID.
        n_transactions: Total number of transactions.
        min_support: Minimum support threshold (0.0-1.0).
        max_length: Maximum itemset length (None = unlimited).
        profile: If True, return profiling metrics alongside results.
        level_callback: Optional callback for per-level progress updates.

    Returns:
        If profile=False: DataFrame with columns [itemset, support].
        If profile=True: Tuple of (DataFrame, ProfilingSession).
    """
    try:
        import cupy as cp
    except ImportError:
        raise ImportError("CuPy required for gpu_resident mode")

    from et_miner.gpu.kernels import get_popcount_kernel
    from et_miner.gpu.dispatch import dispatch_k2_gpu_resident, dispatch_k3plus_gpu_resident

    session = ProfilingSession() if profile else None

    n_cols, n_u64s = bitvecs_gpu.shape
    min_count_threshold = _min_count(min_support, n_transactions)

    effective_max_length = min(
        max_length if max_length else float("inf"),
        n_cols,
    )

    # Accumulate (CuPy itemsets, CuPy counts) per level — all in VRAM
    gpu_results = []

    # === K=1: popcount -> freq_cols_gpu (n,1) in VRAM ===
    _k1_start = time.perf_counter()
    if session:
        session.start_phase("k1_support_gpu_resident")

    popcount_kernel = get_popcount_kernel()
    popcounts = popcount_kernel(bitvecs_gpu.view(cp.uint64))
    col_counts_gpu = cp.sum(popcounts.reshape(n_cols, -1), axis=1, dtype=cp.int64)

    # Filter frequent on GPU
    freq_mask = col_counts_gpu >= min_count_threshold
    freq_col_indices = cp.where(freq_mask)[0].astype(cp.int32)  # VRAM
    freq_counts = col_counts_gpu[freq_mask]  # VRAM

    n_frequent_k1 = len(freq_col_indices)

    if n_frequent_k1 > 0:
        # Store K=1 results as (n, 1) array in VRAM
        k1_itemsets = freq_col_indices.reshape(-1, 1)  # (n, 1) VRAM
        gpu_results.append((k1_itemsets, freq_counts))

    if session:
        session.end_phase(n_frequent=n_frequent_k1)

    if level_callback:
        _k1_duration_ms = (time.perf_counter() - _k1_start) * 1000
        level_callback(1, n_cols, n_frequent_k1, _k1_duration_ms)

    if n_frequent_k1 == 0:
        if profile:
            return _empty_result(), session
        return _empty_result()

    # === K=2: fused kernel -> pair_items (n,2) in VRAM ===
    k = 2
    prev_freq_gpu = k1_itemsets  # (n, 1) — sorted freq column indices

    while k <= effective_max_length and len(prev_freq_gpu) >= k:
        _k_start = time.perf_counter()

        if k == 2:
            if session:
                session.start_phase("k2_fused_gpu_resident")

            n_pairs = n_frequent_k1 * (n_frequent_k1 - 1) // 2

            pair_itemsets, pair_counts = dispatch_k2_gpu_resident(
                bitvecs_gpu, freq_col_indices, n_u64s, min_count_threshold, n_gpus=n_gpus
            )

            if pair_itemsets is not None:
                gpu_results.append((pair_itemsets, pair_counts))
                n_frequent_k2 = len(pair_itemsets)
                prev_freq_gpu = pair_itemsets  # (n, 2) for next iteration
            else:
                n_frequent_k2 = 0

            if session:
                session.end_phase(n_candidates=n_pairs, n_frequent=n_frequent_k2)

            if level_callback:
                _k_duration_ms = (time.perf_counter() - _k_start) * 1000
                level_callback(k, n_pairs, n_frequent_k2, _k_duration_ms)

            if n_frequent_k2 == 0:
                break

        else:
            # === K>=3: build_groups_gpu -> count kernel -> decode kernel -> sort ===
            if session:
                session.start_phase(f"k{k}_gpu_resident")

            freq_itemsets, freq_counts = dispatch_k3plus_gpu_resident(
                bitvecs_gpu, prev_freq_gpu, n_u64s, min_count_threshold, n_gpus=n_gpus
            )

            if freq_itemsets is not None:
                gpu_results.append((freq_itemsets, freq_counts))
                n_frequent_k = len(freq_itemsets)
                prev_freq_gpu = freq_itemsets  # (n, k) for next iteration
            else:
                n_frequent_k = 0

            if session:
                session.end_phase(n_frequent=n_frequent_k)

            if level_callback:
                _k_duration_ms = (time.perf_counter() - _k_start) * 1000
                level_callback(k, 0, n_frequent_k, _k_duration_ms)

            if n_frequent_k == 0:
                break

        k += 1

    # === END: single bulk transfer, build DataFrame ===
    results = _build_results_from_gpu(gpu_results, col_to_item, n_transactions)
    result_df = _build_result_df(results)

    if profile:
        return result_df, session
    return result_df




