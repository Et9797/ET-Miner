"""Custom exceptions for et-miner.

This module provides a hierarchy of exceptions for better error handling
and debugging. All exceptions inherit from PolarsAprioriError for easy
catching of library-specific errors.

Example:
    >>> from et_miner.exceptions import NoRulesFoundError
    >>> try:
    ...     rules = mine_rules(data, min_support=0.99)
    ... except NoRulesFoundError as e:
    ...     print(f"Try lowering min_support: {e}")
"""


class PolarsAprioriError(Exception):
    """Base exception for all et-miner errors.

    All library-specific exceptions inherit from this class, making it easy
    to catch any et-miner error:

        try:
            result = some_operation()
        except PolarsAprioriError as e:
            logger.error(f"et-miner error: {e}")
    """

    pass


# =============================================================================
# Mining Errors
# =============================================================================


class MiningError(PolarsAprioriError):
    """Base class for errors during the mining process."""

    pass


class NoFrequentItemsetsError(MiningError):
    """Raised when no frequent itemsets are found.

    This typically happens when min_support is too high for the dataset.

    Attributes:
        min_support: The minimum support threshold that was used
        total_transactions: Number of transactions in the dataset
    """

    def __init__(
        self,
        message: str = "No frequent itemsets found",
        min_support: float | None = None,
        total_transactions: int | None = None,
    ):
        self.min_support = min_support
        self.total_transactions = total_transactions

        if min_support is not None:
            message = f"{message} (min_support={min_support})"
        if total_transactions is not None:
            message = f"{message} with {total_transactions} transactions"
        message += ". Try lowering min_support."

        super().__init__(message)


class NoRulesFoundError(MiningError):
    """Raised when no association rules meet the thresholds.

    This can happen when:
    - min_confidence is too high
    - min_lift is too high
    - The itemsets don't have strong associations

    Attributes:
        min_confidence: The confidence threshold used
        min_lift: The lift threshold used (if any)
        num_itemsets: Number of frequent itemsets available
    """

    def __init__(
        self,
        message: str = "No association rules found",
        min_confidence: float | None = None,
        min_lift: float | None = None,
        num_itemsets: int | None = None,
    ):
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.num_itemsets = num_itemsets

        details = []
        if min_confidence is not None:
            details.append(f"min_confidence={min_confidence}")
        if min_lift is not None:
            details.append(f"min_lift={min_lift}")
        if num_itemsets is not None:
            details.append(f"from {num_itemsets} itemsets")

        if details:
            message = f"{message} ({', '.join(details)})"
        message += ". Try lowering thresholds."

        super().__init__(message)


class InsufficientDataError(MiningError):
    """Raised when there's not enough data for meaningful mining.

    Attributes:
        required: Minimum required transactions/items
        actual: Actual count in the dataset
        data_type: What's insufficient ('transactions', 'items', etc.)
    """

    def __init__(
        self,
        message: str = "Insufficient data for mining",
        required: int | None = None,
        actual: int | None = None,
        data_type: str = "transactions",
    ):
        self.required = required
        self.actual = actual
        self.data_type = data_type

        if required is not None and actual is not None:
            message = f"{message}: need at least {required} {data_type}, got {actual}"
        elif actual is not None:
            message = f"{message}: only {actual} {data_type}"

        super().__init__(message)


# =============================================================================
# Predicate Errors
# =============================================================================


class PredicateError(PolarsAprioriError):
    """Base class for predicate-related errors."""

    pass


class PredicateMismatchError(PredicateError):
    """Raised when predicates in rules don't match expected predicates.

    This is a critical error that indicates mining and querying are using
    different predicate definitions, which causes silent query expansion failures.

    Attributes:
        unknown_predicates: List of predicate names that weren't recognized
        available_predicates: List of valid predicate names
    """

    def __init__(
        self,
        message: str = "Predicate mismatch detected",
        unknown_predicates: list[str] | None = None,
        available_predicates: list[str] | None = None,
    ):
        self.unknown_predicates = unknown_predicates or []
        self.available_predicates = available_predicates

        if unknown_predicates:
            message = f"{message}: unknown predicates {unknown_predicates}"
        message += ". Ensure rules were mined with the same PredicateRegistry."

        super().__init__(message)


class InvalidPredicatePatternError(PredicateError):
    """Raised when a predicate pattern is invalid.

    Attributes:
        predicate_name: Name of the predicate with the invalid pattern
        pattern: The invalid pattern string
        reason: Why the pattern is invalid
    """

    def __init__(
        self,
        predicate_name: str,
        pattern: str,
        reason: str = "invalid regex",
    ):
        self.predicate_name = predicate_name
        self.pattern = pattern
        self.reason = reason

        message = f"Invalid pattern for '{predicate_name}': {reason}"
        super().__init__(message)


# =============================================================================
# Embedding Errors
# =============================================================================


class EmbeddingError(PolarsAprioriError):
    """Base class for embedding-related errors."""

    pass


class EmbeddingModelNotAvailableError(EmbeddingError):
    """Raised when sentence-transformers is not installed.

    Provides installation instructions.
    """

    def __init__(self, feature: str = "embeddings"):
        message = (
            f"sentence-transformers is required for {feature}. "
            "Install with: pip install 'et-miner[semantic]' "
            "or: pip install sentence-transformers"
        )
        super().__init__(message)


class EmbeddingDimensionMismatchError(EmbeddingError):
    """Raised when embedding dimensions don't match.

    Attributes:
        expected_dim: Expected embedding dimension
        actual_dim: Actual embedding dimension
    """

    def __init__(self, expected_dim: int, actual_dim: int):
        self.expected_dim = expected_dim
        self.actual_dim = actual_dim

        message = f"Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}"
        super().__init__(message)


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(PolarsAprioriError):
    """Base class for configuration-related errors."""

    pass


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration values are invalid.

    Attributes:
        key: The configuration key with invalid value
        value: The invalid value
        reason: Why it's invalid
    """

    def __init__(self, key: str, value: object, reason: str):
        self.key = key
        self.value = value
        self.reason = reason

        message = f"Invalid configuration for '{key}': {reason} (got {value!r})"
        super().__init__(message)


class ConfigFileNotFoundError(ConfigurationError):
    """Raised when a configuration file is not found.

    Attributes:
        path: Path to the missing file
    """

    def __init__(self, path: str):
        self.path = path
        message = f"Configuration file not found: {path}"
        super().__init__(message)


# =============================================================================
# Cache Errors
# =============================================================================


class CacheError(PolarsAprioriError):
    """Base class for caching-related errors."""

    pass


class CacheCorruptedError(CacheError):
    """Raised when the cache is corrupted or invalid.

    Attributes:
        cache_path: Path to the corrupted cache
        reason: Description of the corruption
    """

    def __init__(self, cache_path: str, reason: str = "unknown"):
        self.cache_path = cache_path
        self.reason = reason

        message = f"Cache corrupted at {cache_path}: {reason}"
        super().__init__(message)


class CacheVersionMismatchError(CacheError):
    """Raised when cache version doesn't match current version.

    Attributes:
        cache_version: Version found in cache
        current_version: Current expected version
    """

    def __init__(self, cache_version: str, current_version: str):
        self.cache_version = cache_version
        self.current_version = current_version

        message = (
            f"Cache version mismatch: cache is v{cache_version}, "
            f"current version is v{current_version}. Consider clearing the cache."
        )
        super().__init__(message)


# =============================================================================
# Streaming Errors
# =============================================================================


class StreamingError(PolarsAprioriError):
    """Base class for streaming-related errors."""

    pass


class ChunkProcessingError(StreamingError):
    """Raised when processing a chunk fails during streaming.

    Attributes:
        chunk_index: Index of the failed chunk
        total_chunks: Total number of chunks (if known)
    """

    def __init__(
        self,
        message: str = "Chunk processing failed",
        chunk_index: int | None = None,
        total_chunks: int | None = None,
    ):
        self.chunk_index = chunk_index
        self.total_chunks = total_chunks

        if chunk_index is not None:
            if total_chunks is not None:
                message = f"{message} (chunk {chunk_index + 1}/{total_chunks})"
            else:
                message = f"{message} (chunk {chunk_index})"

        super().__init__(message)


# =============================================================================
# Convenience re-exports
# =============================================================================

__all__ = [
    # Base
    "PolarsAprioriError",
    # Mining
    "MiningError",
    "NoFrequentItemsetsError",
    "NoRulesFoundError",
    "InsufficientDataError",
    # Predicates
    "PredicateError",
    "PredicateMismatchError",
    "InvalidPredicatePatternError",
    # Embeddings
    "EmbeddingError",
    "EmbeddingModelNotAvailableError",
    "EmbeddingDimensionMismatchError",
    # Configuration
    "ConfigurationError",
    "InvalidConfigurationError",
    "ConfigFileNotFoundError",
    # Cache
    "CacheError",
    "CacheCorruptedError",
    "CacheVersionMismatchError",
    # Streaming
    "StreamingError",
    "ChunkProcessingError",
]
