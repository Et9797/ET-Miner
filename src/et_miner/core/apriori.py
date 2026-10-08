"""apriori(): parameter validation and the routing to a mining route.

Routes: the CPU miner (et_miner.core.cpu_miner, the default), the row-split
GPU miner (et_miner.gpu.row_split; use_gpu=True or bitvecs=), and SON
streaming on one or more devices (et_miner.streaming; streaming=True). This
module validates the parameters, rejects combinations a route cannot honour,
and holds list-based reference implementations of the free-set test and
Pascal inference (core.cpu_miner applies both to arrays).

Usage:
    from et_miner import apriori

    itemsets = apriori(df, min_support=0.01)
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from math import comb
from typing import Any, Literal

import polars as pl
from loguru import logger

from et_miner import _env
from et_miner.gpu.density import validate_sparse_from_k

from .profiling import ProfilingSession
from .result import (
    _empty_result,
)

def _warn_complexity(
    n_frequent_items: int,
    min_support: float,
) -> None:
    """Warn user if computation might be expensive."""
    # Estimate k=2 candidates (n choose 2)
    n_pairs = comb(n_frequent_items, 2) if n_frequent_items > 1 else 0

    # Warning thresholds (tuned from benchmarks)
    if n_pairs > 10_000_000:
        warnings.warn(
            f"Very large computation ahead!\n"
            f"  Frequent items: {n_frequent_items:,}\n"
            f"  Candidate pairs: {n_pairs:,}\n"
            f"  Estimated time: >30 minutes\n"
            f"  Consider increasing min_support (currently {min_support})",
            UserWarning,
            stacklevel=3,
        )
    elif n_pairs > 1_000_000:
        warnings.warn(
            f"Large computation: {n_pairs:,} candidate pairs.\n  This may take several minutes.",
            UserWarning,
            stacklevel=3,
        )


def _validate_parameters(
    min_support: float,
    max_length: int | None,
    batch_size: int | None,
    sparse_from_k: int | str | None = None,
    prune_apriori: bool = True,
) -> None:
    """Validate apriori parameters with clear error messages."""
    # min_support: must be numeric in [0.0, 1.0]
    if not isinstance(min_support, (int, float)):
        raise TypeError(f"min_support must be numeric, got {type(min_support).__name__}")
    if not 0.0 <= min_support <= 1.0:
        raise ValueError(
            f"min_support must be in range [0.0, 1.0], got {min_support}\n"
            f"  Hint: Support is a proportion (e.g., 0.01 = 1%)"
        )

    # max_length: must be positive int or None
    if max_length is not None:
        if not isinstance(max_length, int):
            raise TypeError(f"max_length must be int or None, got {type(max_length).__name__}")
        if max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {max_length}")

    # batch_size: must be positive int or None
    if batch_size is not None:
        if not isinstance(batch_size, int):
            raise TypeError(f"batch_size must be int or None, got {type(batch_size).__name__}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    validate_sparse_from_k(sparse_from_k)
    if not isinstance(prune_apriori, bool):
        raise TypeError(f"prune_apriori must be bool, got {type(prune_apriori).__name__}")


def _prune_equal_support(
    current_frequent: list[tuple[str, ...]],
    current_counts: dict[tuple[str, ...], int],
    prev_counts: dict[tuple[str, ...], int],
) -> list[tuple[str, ...]]:
    """Keep only the free-sets (generators) of a level (reference implementation).

    The CPU miner applies the same test to arrays (``core.cpu_miner``), and
    ``tests/test_cpu_miner.py`` checks its free-sets against a brute force;
    nothing calls this function on a mining route.

    An itemset is a *free-set* when no proper subset has the same support
    [Bastide et al. 2000, Pascal]. If support({A,B,C}) == support({A,B}) then C
    is implied by {A,B} and {A,B,C} is not free. Free-sets are anti-monotone —
    every subset of a free-set is free — so the next level can be generated
    from the survivors alone without losing a single free-set.

    NOT closed-itemset mining: closed asks about equal-support *supersets*,
    this asks about equal-support *subsets*, and the two select different
    itemsets. ``prev_counts`` must hold every frequent candidate of the
    previous level, not just its free-sets (the caller never prunes it): an
    equal-count (k-1)-subset of a candidate generated from free parents is
    always among those candidates, while the free level alone can miss it and
    under-prune.

    Compares raw integer counts, exactly as the GPU path does — a float
    comparison with a tolerance made the two tiers disagree at the boundary.

    Args:
        current_frequent: Frequent k-itemsets from current iteration.
        current_counts: Integer counts for the current k-itemsets.
        prev_counts: Integer counts of every frequent candidate of the previous level.

    Returns:
        The free-sets, in input order.
    """
    if not prev_counts:
        return current_frequent

    free = []
    for itemset in current_frequent:
        is_free = True
        current_count = current_counts[itemset]

        # Not free if the count equals any (k-1)-subset's count
        for i in range(len(itemset)):
            subset = itemset[:i] + itemset[i + 1 :]
            if prev_counts.get(subset) == current_count:
                is_free = False
                break

        if is_free:
            free.append(itemset)

    return free




def _infer_count_from_subsets(
    candidate: tuple[str, ...],
    prev_counts: dict[tuple[str, ...], int],
    prev_free: set[tuple[str, ...]] | None,
) -> int | None:
    """Infer a candidate's exact count from its (k-1)-subsets, or None (reference implementation).

    The CPU miner applies the same rule to arrays (``core.cpu_miner``).

    Pascal [Bastide et al. 2000]: if a (k-1)-subset Y of X is *not* free — some
    Z ⊊ Y has rows(Z) = rows(Y) — then for X = Y ∪ {a} we have
    rows(X) = rows(Z ∪ {a}), and any (k-1)-subset W with Z ∪ {a} ⊆ W ⊂ X
    satisfies rows(W) = rows(X). So X is not free either and

        count(X) = min{count(W) : W ⊂ X, |W| = k-1}

    exactly — no counting pass needed. When every (k-1)-subset is free nothing
    can be inferred and the candidate must be counted.

    This replaces an earlier rule that inferred whenever ALL subsets happened to
    share a count, which is unsound: support({A,B}) = support({A,C}) =
    support({B,C}) = 30 with support({A,B,C}) = 15 was its own documented
    counterexample. Here all three pairs are free, so no inference is made.

    Args:
        candidate: Candidate k-itemset to check.
        prev_counts: Integer counts of every frequent candidate of the previous level.
        prev_free: The free (generator) (k-1)-itemsets. None disables inference.

    Returns:
        The exact count, or None when the candidate has to be counted.
    """
    k = len(candidate)
    if k < 2 or prev_free is None:
        return None

    subset_counts: list[int] = []
    licensed = False
    for i in range(k):
        subset = candidate[:i] + candidate[i + 1 :]
        count = prev_counts.get(subset)
        if count is None:
            # A subset is not frequent — the candidate cannot be either, and we
            # have no minimum to take. Let the caller count it.
            return None
        subset_counts.append(count)
        if subset not in prev_free:
            licensed = True

    if not licensed:
        return None

    inferred = min(subset_counts)
    logger.debug(f"INFERRED {candidate}: non-free subset licenses count={inferred}")
    return inferred




def _validate_route_support(
    *,
    streaming: bool,
    use_gpu: bool,
    has_bitvecs: bool,
    n_gpus: int,
    profile: bool,
    gpu_resident: bool,
    prune_equal_support: bool,
    use_generator_pruning: bool,
    anchor_items: set | None,
    output_dir: str | None,
    resume_from_k: int | None,
    memory_budget_gb: float | None,
    prune_apriori: bool = True,
) -> None:
    """Reject parameter/route combinations the chosen route cannot honour.

    apriori() takes a wide parameter set and dispatches to one of four routes
    (the row-split GPU miner, single- and multi-GPU SON streaming, the CPU
    miner), not all of which implement all of it. Every mismatch below used to be
    SILENT: the caller passed the parameter, the route dropped it, and nothing
    in the return value or the logs said so. Measured on a 400-row fixture:

      streaming=True + prune_equal_support  -> the complete 214-itemset lattice
                                               where the gated answer is 176
      anchor_items on the CPU route         -> 214 rows, identical to unanchored
      output_dir on the CPU route           -> files written: []
      profile=True + streaming + n_gpus>1   -> a bare DataFrame, which then
                                               UNPACKS into two Series, so
                                               `result, session = apriori(...)`
                                               succeeds and hands back a column
                                               of itemsets and a column of floats

    The file already raised for one such combination (profile + pruning on a GPU
    path); this generalises that to every one of them. Raising is the right
    answer rather than best-effort forwarding: a miner whose contract is
    exactness should not quietly answer a different question.

    Where a route CAN honour a parameter it is forwarded instead -- output_dir
    and resume_from_k on the bitvecs route, and memory_budget_gb on multi-GPU
    streaming -- so this only fires where the capability genuinely does not
    exist.
    """
    if gpu_resident:
        raise ValueError(
            "gpu_resident was removed: use_gpu=True (or bitvecs=) mines on the row-split miner, "
            "which keeps candidates and counts on the GPU. Drop the argument."
        )

    # prune_equal_support / use_generator_pruning: the row-split miner alone
    # implements the gates, and it is only reachable via use_gpu or bitvecs=.
    if streaming and (prune_equal_support or use_generator_pruning):
        which = " and ".join(
            n for n, v in (("prune_equal_support", prune_equal_support),
                           ("use_generator_pruning", use_generator_pruning)) if v
        )
        raise ValueError(
            f"streaming=True cannot be combined with {which}: the SON streaming "
            "paths do not implement the pruning gates, and would return the "
            "complete frequent lattice instead of the free-sets. Drop one of "
            "the two, or mine without streaming."
        )

    # anchor_items is implemented by the row-split miner for transactions
    # input only. The bitvecs route REFUSES it rather than forwarding: that
    # route forwards output_dir and resume_from_k, so anchoring there would put
    # anchoring on a route that also persists and resumes -- and an anchored
    # per-K parquet is not a resumable mining state (see the resume_from_k
    # check below). Refusing keeps everything that route flushes a complete
    # level for a structural reason, rather than only while a second guard
    # holds. Nothing in the tree pairs bitvecs= with anchor_items.
    if anchor_items is not None and not (use_gpu and not streaming and not has_bitvecs):
        raise ValueError(
            "anchor_items requires the row-split miner, reached with "
            "use_gpu=True, streaming=False and transactions (not bitvecs=): "
            "only that miner implements anchor filtering, and every other route "
            "would silently return the complete, differently-shaped lattice."
        )

    # profile builds a ProfilingSession, which every route but multi-GPU SON does.
    if profile and streaming and n_gpus > 1 and not has_bitvecs:
        raise ValueError(
            "profile=True cannot be combined with streaming=True and "
            "n_gpus>1: the multi-GPU streaming path builds no "
            "ProfilingSession and would return a bare DataFrame, which "
            "unpacks silently into two Series."
        )

    # The row-split miner serves bitvecs= (that branch runs first) and use_gpu
    # without streaming.
    reaches_row_split = has_bitvecs or (use_gpu and not streaming)

    # The CPU route generates candidates with the subset test built in, and SON
    # mines its chunks with the test on; only the row-split miner can skip it.
    if use_generator_pruning and not prune_apriori:
        raise ValueError(
            "use_generator_pruning needs prune_apriori=True: on the row-split miner the count "
            "inference runs inside the same subset test, and without it every candidate is counted."
        )
    if not prune_apriori and not reaches_row_split:
        raise ValueError(
            "prune_apriori=False requires the row-split miner, reached with bitvecs= or with "
            "use_gpu=True and streaming=False: the CPU route and SON always test every "
            "candidate's (k-1)-subsets, so the flag would be ignored there."
        )

    # output_dir / resume_from_k are the row-split miner's per-K parquet flush.
    if output_dir is not None or resume_from_k is not None:
        which = " and ".join(
            n for n, v in (("output_dir", output_dir), ("resume_from_k", resume_from_k))
            if v is not None
        )
        if not reaches_row_split:
            raise ValueError(
                f"{which} requires the row-split miner, reached with bitvecs= or "
                "with use_gpu=True and streaming=False. On this route the per-K "
                "parquet flush does not run: output_dir would stay empty and "
                "resume_from_k would silently re-mine from K=1."
            )


    # A persisted K-level is a valid resume artifact IFF it is the complete
    # frequent level at that K -- resume reloads it as BOTH the generation base
    # and the subset oracle for every level above.
    #
    # Anchoring emits a strict subset of the level, so an anchored parquet is a
    # REPORT, not a mining state. Resuming from one reconstructs exactly the
    # restricted generation base that made anchoring unsound in the first place:
    # measured 57 itemsets against 838 for the same call without resume, a 93%
    # silent loss, every level after the resume point wrong.
    #
    # The free-set prune violates the same rule in the SAFE direction (it emits
    # a subset that is closed under the operations the next level needs) and
    # says so where it resumes; anchoring violates it in the unsafe direction,
    # because anchoredness is not anti-monotone. Only the unsafe one is refused.
    #
    # Refusing the READ rather than the write is deliberate: mine_two_phase
    # already produces anchored artifacts on disk, so guarding the writer would
    # break that feature, while guarding the reader closes the hole whoever
    # wrote the file -- including a user resuming by hand from a directory
    # mine_two_phase left behind.
    if resume_from_k is not None and anchor_items is not None:
        raise ValueError(
            "resume_from_k cannot be combined with anchor_items: an anchored "
            "per-K parquet holds only the itemsets containing an anchor, so it "
            "is a report rather than a resumable mining state. Resuming from it "
            "would rebuild the next level from a restricted base and silently "
            "lose most of the lattice. Re-mine without resume_from_k, or resume "
            "from a run that was not anchored."
        )

    # memory_budget_gb sizes the SON chunk; nothing else reads it.
    if memory_budget_gb is not None and not streaming:
        raise ValueError(
            "memory_budget_gb requires streaming=True: it overrides chunk_size "
            "for the SON paths and is read nowhere else."
        )


def apriori(
    transactions: pl.LazyFrame | pl.DataFrame | None = None,
    min_support: float = 0.5,
    max_length: int | None = None,
    item_col: str = "items",
    use_gpu: bool = False,
    batch_size: int | None = 10_000,
    profile: bool = False,
    show_progress: bool = False,
    warn_complexity: bool = True,
    prune_equal_support: bool = False,
    use_generator_pruning: bool = False,
    prune_apriori: bool = True,
    sparse: bool | None = None,
    n_jobs: int = 1,
    enable_length_filter: bool = True,
    # Streaming parameters (SON algorithm)
    streaming: bool = False,
    chunk_size: int = 10_000_000,
    n_gpus: int = 1,
    memory_budget_gb: float | None = None,
    progress_callback: Callable[[str, int, int, dict[str, Any]], None] | None = None,
    level_callback: Callable[[int, int, int, float], None] | None = None,
    # Removed: True raises ValueError
    gpu_resident: bool = False,
    # Pre-built bitvector input (GPU fast path)
    bitvecs: tuple | None = None,
    # Per-K Parquet flush (prevents CPU RAM OOM at K=7+ scale)
    output_dir: str | None = None,
    # Resume from K=N+1 by loading K=N frequent itemsets from parquet
    resume_from_k: int | None = None,
    # Memory guard limits for exhaustive mining
    max_ram_gb: float = 800.0,
    max_vram_gb: float = 70.0,
    # ESCO: dense→sparse CSR transition, fixed K or measured density
    sparse_from_k: int | Literal["auto"] | None = None,
    # V3 B6: restrict candidates to anchor neighborhoods (two-phase mining)
    anchor_items: set | None = None,
) -> pl.DataFrame | tuple[pl.DataFrame, ProfilingSession]:
    """Find the frequent itemsets (or free-sets) of a transaction list column.

    The CPU route (the default) builds one CSR of the frequent items and mines
    every level from it; use_gpu, bitvecs= and streaming select the other routes.

    Args:
        transactions: Transaction data with item lists.
        min_support: Minimum support threshold (0.0-1.0).
        max_length: Maximum itemset length (None = unlimited).
        item_col: Column name with item lists.
        use_gpu: Mine on the GPU with the row-split miner (requires CuPy).
        batch_size: Candidates per batch for SON streaming's CPU counter. None = no
            batching. No effect on the CPU route.
        profile: If True, return profiling metrics alongside results.
        show_progress: If True, display progress bars where a route has them
            (requires tqdm).
        warn_complexity: If True, warn when candidate pairs > 1M.
        prune_equal_support: Mine frequent FREE-SETS (generators) instead of the
            complete lattice. An itemset is free when no proper subset has the
            same support [Bastide et al. 2000]; free-sets are anti-monotone, so
            each level is generated from the previous level's free-sets alone —
            3-50x fewer candidates on dense data. The returned lattice is
            exactly the free-sets: what is emitted is what the next level is
            generated from, so the support of every omitted frequent itemset
            equals that of one of its subsets. False (default) returns the
            complete frequent lattice. This is NOT closed-itemset mining, which
            asks about equal-support supersets.
        use_generator_pruning: Infer support from (k-1)-subsets instead of counting.
            Pascal [Bastide et al. 2000]: a candidate with a non-free (k-1)-subset
            is itself non-free and its support is exactly the minimum of its
            (k-1)-subset supports, so it never has to be counted. Exact, no
            impact on results. CPU route and the row-split GPU miner; on the
            GPU it needs ``prune_apriori`` and applies to complete-lattice
            runs (in a free-set run such candidates are not free and are
            skipped anyway).
        prune_apriori: Skip candidates with an infrequent (k-1)-subset before
            counting them (Apriori's subset test; exact, no impact on results).
            The row-split GPU miner tests every K>=3 candidate on the device
            against the previous level and leaves the ones it settles
            uncounted; in a free-set run the test runs against the free level,
            which also skips every candidate that cannot be free. False counts
            every candidate the prefix groups generate, and is honoured only by
            the row-split miner (the CPU route and SON always test).
        sparse: Deprecated on the CPU route, where it no longer selects a counting
            engine (a non-None value warns and is ignored). For streaming=True:
            True = scipy CSR, False = Polars, None = auto.
        n_jobs: Parallel workers for counting. 1 = sequential, -1 = all CPUs.
        enable_length_filter: Skip transactions shorter than k when counting
            k-itemsets. Set to False to disable (results are the same).
        streaming: Use SON algorithm for chunked processing. Memory becomes
            O(chunk_size × n_items) instead of O(total × n_items).
        chunk_size: Transactions per chunk when streaming=True. Default 10M.
        n_gpus: GPUs for the row-split miner and for streaming (default 1).
            Requires CuPy.
        memory_budget_gb: Auto-calculate chunk_size to fit this budget.
            Overrides chunk_size. Only used when streaming=True.
        progress_callback: Streaming progress callback
            (phase, chunk_idx, n_chunks, metrics).
        level_callback: Per-level callback
            (k, n_candidates, n_frequent, duration_ms).

            **n_candidates is route-dependent and the routes do not agree.**
            The CPU path applies the full per-candidate subset test before
            counting, so it reports candidates that survived it. The GPU path
            reports every candidate its prefix groups generate, including the
            ones its device-side subset test (``prune_apriori``) then leaves
            uncounted, so it reports more candidates than the CPU path for the
            same input at the same level. Both are honest counts of what that
            route enumerated; neither is "the" candidate count. A consumer
            doing per-level cost accounting should treat the figure as
            comparable within a route and not across routes.

            n_frequent and duration_ms mean the same thing everywhere.
        bitvecs: Pre-built GPU bitvectors tuple (bitvecs_gpu, col_to_item, n_transactions).
            Skips DataFrame conversion; transactions must be None when provided.
            The array is READ-ONLY to the engine: ET-Miner writes only to
            bitvectors it builds itself, never to one it is handed, so the
            caller may reuse it after the call without copying it first.
        sparse_from_k: In-core GPU paths only — an int switches to sparse CSR
            tidsets from that K (at least 3); "auto" switches when the previous
            level's mean count falls below n_transactions/32. None (default)
            keeps dense bitvectors. The transition is one-way. CPU mining
            ignores this GPU setting; SON streaming rejects it.
        max_ram_gb / max_vram_gb: Memory guards for the GPU routes
            (use_gpu=True without streaming, or bitvecs=): MemoryError between
            K-levels once host RSS or the largest device's CuPy pool exceeds
            them. Candidate chunks are sized from measured free VRAM either way.
        gpu_resident: Removed; True raises ValueError. The row-split miner
            keeps candidates and counts on the GPU.

    Returns:
        DataFrame with itemset (List[Int64]) and support (Float64) columns.

        Every itemset is emitted as an **ascending tuple of item ids**, on every
        route. That is a contract, not an accident: both parquet consumers in
        core.rules join K against K-1 positionally, and _build_support_lookup is
        keyed on the itemset, so a producer emitting [10, 2] where another emits
        [2, 10] silently loses rows and corrupts numbers across routes. The
        order is also stable across min_support values, so a stored artifact or
        a frozen digest keyed on the emitted list stays valid when the threshold
        moves.

        Row order within the frame is NOT part of the contract -- only the order
        of items within each itemset.

        If profile=True: tuple of (DataFrame, ProfilingSession).

    Example:
        >>> result = apriori(df, min_support=0.5)
        >>> result = apriori(df, min_support=0.001, use_gpu=True)
        >>> result, session = apriori(df, min_support=0.1, profile=True)
        >>> result = apriori(df, min_support=0.0001, n_jobs=-1)
        >>> result = apriori(huge_df, min_support=0.001, streaming=True, n_gpus=8)
    """
    _validate_parameters(min_support, max_length, batch_size, sparse_from_k, prune_apriori)
    _env.reject_removed_knobs()

    if streaming and bitvecs is None and sparse_from_k is not None:
        raise ValueError("sparse_from_k requires in-core GPU mining; SON streaming does not implement the transition")

    # Every parameter/route mismatch is rejected here, ABOVE the routing, so a
    # route cannot silently drop something the caller asked for. This has to run
    # before `if streaming:` — that branch returns long before the GPU routing
    # is reached, which is how streaming came to swallow prune_equal_support.
    _validate_route_support(
        streaming=streaming,
        use_gpu=use_gpu,
        has_bitvecs=bitvecs is not None,
        n_gpus=n_gpus,
        profile=profile,
        gpu_resident=gpu_resident,
        prune_equal_support=prune_equal_support,
        use_generator_pruning=use_generator_pruning,
        prune_apriori=prune_apriori,
        anchor_items=anchor_items,
        output_dir=output_dir,
        resume_from_k=resume_from_k,
        memory_budget_gb=memory_budget_gb,
    )

    # Route to bitvecs fast path if pre-built GPU bitvectors provided
    if bitvecs is not None:
        # Validate: cannot provide both bitvecs and transactions
        if transactions is not None:
            raise ValueError(
                "Cannot provide both 'transactions' and 'bitvecs'. "
                "Use 'bitvecs' for pre-built GPU bitvectors or 'transactions' for DataFrame input."
            )

        # Unpack and validate bitvecs tuple
        if not isinstance(bitvecs, tuple) or len(bitvecs) != 3:
            raise ValueError("bitvecs must be a tuple of (bitvecs_gpu, col_to_item, n_transactions)")

        bitvecs_gpu, col_to_item, n_trans = bitvecs

        # Validate bitvecs_gpu is a CuPy array
        if not hasattr(bitvecs_gpu, "__cuda_array_interface__"):
            raise TypeError(
                f"bitvecs_gpu must be a CuPy array (has __cuda_array_interface__), got {type(bitvecs_gpu).__name__}"
            )

        # Validate shape
        if bitvecs_gpu.ndim != 2:
            raise ValueError(f"bitvecs_gpu must be 2D (n_cols, n_u64s), got {bitvecs_gpu.ndim}D")

        # Validate n_u64s matches n_transactions
        expected_n_u64s = math.ceil(n_trans / 64)
        if bitvecs_gpu.shape[1] != expected_n_u64s:
            raise ValueError(
                f"bitvecs_gpu.shape[1] ({bitvecs_gpu.shape[1]}) must equal "
                f"ceil(n_transactions/64) = ceil({n_trans}/64) = {expected_n_u64s}"
            )

        # Validate col_to_item keys match n_cols
        if len(col_to_item) != bitvecs_gpu.shape[0]:
            raise ValueError(
                f"col_to_item has {len(col_to_item)} keys but bitvecs_gpu has {bitvecs_gpu.shape[0]} columns"
            )

        # The kernels index the array as col * n_u64s + word.
        if bitvecs_gpu.dtype != "uint64" or not bitvecs_gpu.flags.c_contiguous:
            raise ValueError(
                f"bitvecs_gpu must be a C-contiguous uint64 array (cupy.ascontiguousarray), got "
                f"{bitvecs_gpu.dtype}, c_contiguous={bitvecs_gpu.flags.c_contiguous}"
            )

        from et_miner.gpu.row_split import _apriori_row_split_multi_gpu, shard_prebuilt_bitvecs

        return _apriori_row_split_multi_gpu(
            None,
            col_to_item,
            n_trans,
            min_support,
            max_length,
            max(1, n_gpus),
            level_callback,
            bitvecs_list=shard_prebuilt_bitvecs(bitvecs_gpu, n_trans, max(1, n_gpus)),
            output_dir=output_dir,
            resume_from_k=resume_from_k,
            prune_non_free=prune_equal_support,
            prune_apriori=prune_apriori,
            infer_counts=use_generator_pruning,
            sparse_from_k=sparse_from_k,
            profile=profile,
            max_ram_gb=max_ram_gb,
            max_vram_gb=max_vram_gb,
        )

    # Validate transactions is provided when bitvecs is not
    if transactions is None:
        raise ValueError("Either 'transactions' or 'bitvecs' must be provided")

    # Route to streaming implementation if requested
    if streaming:
        # Multi-GPU streaming if n_gpus > 1
        if n_gpus > 1:
            from et_miner.streaming.multi_gpu import apriori_streaming_multi_gpu

            return apriori_streaming_multi_gpu(
                transactions,
                min_support=min_support,
                max_length=max_length,
                item_col=item_col,
                n_gpus=n_gpus,
                chunk_size=chunk_size,
                memory_budget_gb=memory_budget_gb,
                batch_size=batch_size,
                show_progress=show_progress,
                sparse=sparse,
                n_jobs=n_jobs,
                progress_callback=progress_callback,
            )

        # Single-GPU/CPU streaming
        from et_miner.streaming.son import apriori_streaming

        return apriori_streaming(
            transactions,
            min_support=min_support,
            max_length=max_length,
            item_col=item_col,
            chunk_size=chunk_size,
            memory_budget_gb=memory_budget_gb,
            use_gpu=use_gpu,
            gpu_resident=gpu_resident,
            batch_size=batch_size,
            profile=profile,
            show_progress=show_progress,
            sparse=sparse,
            n_jobs=n_jobs,
            progress_callback=progress_callback,
        )

    lf = transactions.lazy() if isinstance(transactions, pl.DataFrame) else transactions

    # ── GPU direct path: transactions → CSR → GPU bitvecs ──────────────
    # Bypasses the dense boolean matrix entirely.
    # For 205M × 1006: ~5 GB CPU + ~25 GB GPU  instead of  206 GB CPU.
    if use_gpu:
        from .matrix import _build_csr_from_transactions

        csr_result = _build_csr_from_transactions(lf, min_support, item_col)
        if csr_result is None:
            if profile:
                return _empty_result(), ProfilingSession()
            return _empty_result()

        csr, idx_to_item, n_trans = csr_result

        from et_miner.gpu.row_split import _apriori_row_split_multi_gpu

        return _apriori_row_split_multi_gpu(
            csr,
            idx_to_item,
            n_trans,
            min_support,
            max_length,
            max(1, n_gpus),
            level_callback,
            output_dir=output_dir,
            resume_from_k=resume_from_k,
            prune_non_free=prune_equal_support,  # free-sets: emit == generate
            prune_apriori=prune_apriori,
            infer_counts=use_generator_pruning,
            sparse_from_k=sparse_from_k,
            anchor_items=anchor_items,  # V3 B6: two-phase anchor filtering
            profile=profile,
            max_ram_gb=max_ram_gb,
            max_vram_gb=max_vram_gb,
        )

    # ── CPU path: one CSR of the frequent items, mined level by level ──
    if sparse is not None:
        warnings.warn(
            "sparse= no longer selects a counting engine on the CPU route: every level is counted "
            "from one CSR whatever its value, and the argument is ignored there. It still applies "
            "to streaming=True and will be removed from the CPU route in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
    from .cpu_miner import mine_cpu

    return mine_cpu(
        lf,
        min_support,
        max_length,
        item_col,
        prune_equal_support=prune_equal_support,
        use_generator_pruning=use_generator_pruning,
        enable_length_filter=enable_length_filter,
        n_jobs=n_jobs,
        level_callback=level_callback,
        profile=profile,
        warn_complexity=warn_complexity,
    )

