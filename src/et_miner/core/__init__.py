"""Core mining algorithms: Apriori, candidate generation, counting, rules."""

from et_miner.core.apriori import apriori
from et_miner.core.matrix import (
    build_boolean_matrix,
    count_support_batched,
    count_support_vectorized,
)
from et_miner.core.sparse import count_support_sparse
from et_miner.core.rules import (
    Rule,
    compute_self_sufficiency,
    generate_rules,
    generate_rules_drop1,
    generate_rules_top_n,
)

__all__ = [
    "apriori",
    "build_boolean_matrix",
    "count_support_batched",
    "count_support_sparse",
    "count_support_vectorized",
    "Rule",
    "compute_self_sufficiency",
    "generate_rules",
    "generate_rules_drop1",
    "generate_rules_top_n",
]
