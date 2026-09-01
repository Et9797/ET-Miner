"""Measured-density decision logic for the dense→sparse GPU transition.

The GPU miners hold the transaction set of each frequent itemset in one of
two layouts:

- **dense bitvec**: ``ceil(n_transactions/64)`` uint64 words per itemset.
  Cost is fixed at ~``n_transactions/8`` bytes no matter how few
  transactions actually contain the itemset, and every counting kernel
  sweeps all words.
- **sparse CSR tidset**: one int32 per supporting transaction, so
  ``4 * count`` bytes per itemset. Intersection kernels touch only the
  stored tids.

Byte-for-byte the layouts cross over where ``4 * mean_count ==
n_transactions / 8``, i.e. at mean support ``n_transactions / 32``
(~3.1% density); kernel work scales the same way. ``sparse_from_k="auto"``
applies that crossover to the mean support count of the previous level —
which both mining loops already carry for free — instead of a fixed K.

The transition is one-way by construction: both callers free the bitvecs
when they convert, and keep sparse mode sticky afterwards regardless of
what later levels measure.

Everything in this module is pure CPU math and imports nothing from
et_miner, so it is unit-testable without a GPU and safe to import from any
tier (no import-cycle risk).
"""

from __future__ import annotations

from typing import Final

#: Sentinel value for ``sparse_from_k``: transition on measured density.
SPARSE_AUTO: Final = "auto"

#: Mean-support fraction below which CSR tidsets beat dense bitvecs
#: (4 bytes/tid < n_transactions/8 bytes/bitvec ⇔ mean_count < n/32).
DENSITY_CROSSOVER: Final[float] = 1.0 / 32.0

#: Sparse CSR counting builds K>=3 prefix groups, and the fused K=2 kernel
#: is always faster than pair intersection — never transition below K=3.
MIN_SPARSE_K: Final[int] = 3


def validate_sparse_from_k(value: object) -> int | str | None:
    """Validate a public ``sparse_from_k`` argument and return it unchanged.

    Accepts ``None`` (never transition), an int K-level (fixed transition,
    floored to :data:`MIN_SPARSE_K` at use), or :data:`SPARSE_AUTO`.

    Raises:
        ValueError: For any other string.
        TypeError: For any other type (bool included).
    """
    if value is None:
        return None
    if isinstance(value, str):
        if value == SPARSE_AUTO:
            return value
        raise ValueError(f"sparse_from_k accepts an int K-level, {SPARSE_AUTO!r}, or None — got {value!r}")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"sparse_from_k must be an int K-level, {SPARSE_AUTO!r}, or None — got {type(value).__name__}")
    return value


def should_transition_to_sparse(
    sparse_from_k: int | str | None,
    k: int,
    *,
    n_transactions: int,
    mean_count: float | None = None,
) -> bool:
    """Decide whether level ``k`` should switch from dense bitvecs to CSR.

    Args:
        sparse_from_k: ``None`` = never, int = fixed K-level (floored to
            :data:`MIN_SPARSE_K`), :data:`SPARSE_AUTO` = measured density.
        k: The level about to be mined.
        n_transactions: Total transaction count (bitvec width in rows).
        mean_count: Mean support count of the previous level's frequent
            itemsets. ``None`` = unknown (e.g. first level after a resume
            without counts) — in auto mode that defers the transition until
            counts are available.

    Returns:
        True when level ``k`` should run on sparse CSR tidsets. Callers
        must treat a True result as sticky — the conversion frees the
        dense bitvecs.
    """
    if sparse_from_k is None or k < MIN_SPARSE_K:
        return False
    if sparse_from_k == SPARSE_AUTO:
        if mean_count is None or n_transactions <= 0:
            return False
        return mean_count < n_transactions * DENSITY_CROSSOVER
    return k >= sparse_from_k
