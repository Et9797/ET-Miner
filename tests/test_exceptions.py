"""Tests for the exception hierarchy."""

import pytest

from et_miner.exceptions import (
    PolarsAprioriError,
    MiningError,
    NoFrequentItemsetsError,
    NoRulesFoundError,
    InsufficientDataError,
    PredicateError,
    PredicateMismatchError,
    InvalidPredicatePatternError,
    EmbeddingError,
    EmbeddingModelNotAvailableError,
    EmbeddingDimensionMismatchError,
    ConfigurationError,
    InvalidConfigurationError,
    ConfigFileNotFoundError,
    CacheError,
    CacheCorruptedError,
    CacheVersionMismatchError,
    StreamingError,
    ChunkProcessingError,
)


class TestExceptionHierarchy:
    """Test that exception hierarchy is correct."""

    def test_all_inherit_from_base(self):
        """All exceptions should inherit from PolarsAprioriError."""
        exceptions = [
            MiningError,
            NoFrequentItemsetsError,
            NoRulesFoundError,
            InsufficientDataError,
            PredicateError,
            PredicateMismatchError,
            InvalidPredicatePatternError,
            EmbeddingError,
            EmbeddingModelNotAvailableError,
            EmbeddingDimensionMismatchError,
            ConfigurationError,
            InvalidConfigurationError,
            ConfigFileNotFoundError,
            CacheError,
            CacheCorruptedError,
            CacheVersionMismatchError,
            StreamingError,
            ChunkProcessingError,
        ]

        for exc_class in exceptions:
            assert issubclass(exc_class, PolarsAprioriError), \
                f"{exc_class.__name__} should inherit from PolarsAprioriError"

    def test_mining_errors_inherit_from_mining_error(self):
        """Mining-related exceptions should inherit from MiningError."""
        assert issubclass(NoFrequentItemsetsError, MiningError)
        assert issubclass(NoRulesFoundError, MiningError)
        assert issubclass(InsufficientDataError, MiningError)

    def test_predicate_errors_inherit_from_predicate_error(self):
        """Predicate-related exceptions should inherit from PredicateError."""
        assert issubclass(PredicateMismatchError, PredicateError)
        assert issubclass(InvalidPredicatePatternError, PredicateError)

    def test_embedding_errors_inherit_from_embedding_error(self):
        """Embedding-related exceptions should inherit from EmbeddingError."""
        assert issubclass(EmbeddingModelNotAvailableError, EmbeddingError)
        assert issubclass(EmbeddingDimensionMismatchError, EmbeddingError)


class TestMiningExceptions:
    """Test mining-related exceptions."""

    def test_no_frequent_itemsets_basic(self):
        """Test NoFrequentItemsetsError with basic message."""
        exc = NoFrequentItemsetsError()
        assert "No frequent itemsets" in str(exc)
        assert "Try lowering min_support" in str(exc)

    def test_no_frequent_itemsets_with_details(self):
        """Test NoFrequentItemsetsError with details."""
        exc = NoFrequentItemsetsError(
            min_support=0.5,
            total_transactions=100,
        )
        assert "min_support=0.5" in str(exc)
        assert "100 transactions" in str(exc)
        assert exc.min_support == 0.5
        assert exc.total_transactions == 100

    def test_no_rules_found_basic(self):
        """Test NoRulesFoundError with basic message."""
        exc = NoRulesFoundError()
        assert "No association rules" in str(exc)
        assert "Try lowering thresholds" in str(exc)

    def test_no_rules_found_with_details(self):
        """Test NoRulesFoundError with details."""
        exc = NoRulesFoundError(
            min_confidence=0.9,
            min_lift=3.0,
            num_itemsets=50,
        )
        assert "min_confidence=0.9" in str(exc)
        assert "min_lift=3.0" in str(exc)
        assert "50 itemsets" in str(exc)

    def test_insufficient_data(self):
        """Test InsufficientDataError."""
        exc = InsufficientDataError(
            required=100,
            actual=10,
            data_type="transactions",
        )
        assert "100" in str(exc)
        assert "10" in str(exc)
        assert "transactions" in str(exc)


class TestPredicateExceptions:
    """Test predicate-related exceptions."""

    def test_predicate_mismatch_basic(self):
        """Test PredicateMismatchError with basic message."""
        exc = PredicateMismatchError()
        assert "Predicate mismatch" in str(exc)
        assert "PredicateRegistry" in str(exc)

    def test_predicate_mismatch_with_unknown(self):
        """Test PredicateMismatchError with unknown predicates."""
        exc = PredicateMismatchError(
            unknown_predicates=["mentions_typo", "mentions_old"],
        )
        assert "mentions_typo" in str(exc)
        assert "mentions_old" in str(exc)
        assert exc.unknown_predicates == ["mentions_typo", "mentions_old"]

    def test_invalid_predicate_pattern(self):
        """Test InvalidPredicatePatternError."""
        exc = InvalidPredicatePatternError(
            predicate_name="test_pred",
            pattern="[invalid(",
            reason="unmatched bracket",
        )
        assert "test_pred" in str(exc)
        assert "unmatched bracket" in str(exc)


class TestEmbeddingExceptions:
    """Test embedding-related exceptions."""

    def test_model_not_available(self):
        """Test EmbeddingModelNotAvailableError provides install instructions."""
        exc = EmbeddingModelNotAvailableError("semantic predicates")
        assert "sentence-transformers" in str(exc)
        assert "pip install" in str(exc)
        assert "semantic predicates" in str(exc)

    def test_dimension_mismatch(self):
        """Test EmbeddingDimensionMismatchError."""
        exc = EmbeddingDimensionMismatchError(expected_dim=384, actual_dim=768)
        assert "384" in str(exc)
        assert "768" in str(exc)
        assert exc.expected_dim == 384
        assert exc.actual_dim == 768


class TestConfigurationExceptions:
    """Test configuration-related exceptions."""

    def test_invalid_configuration(self):
        """Test InvalidConfigurationError."""
        exc = InvalidConfigurationError(
            key="min_support",
            value=-0.5,
            reason="must be between 0 and 1",
        )
        assert "min_support" in str(exc)
        assert "-0.5" in str(exc)
        assert "between 0 and 1" in str(exc)

    def test_config_file_not_found(self):
        """Test ConfigFileNotFoundError."""
        exc = ConfigFileNotFoundError("/path/to/config.toml")
        assert "/path/to/config.toml" in str(exc)
        assert exc.path == "/path/to/config.toml"


class TestCacheExceptions:
    """Test cache-related exceptions."""

    def test_cache_corrupted(self):
        """Test CacheCorruptedError."""
        exc = CacheCorruptedError(
            cache_path="/tmp/cache.db",
            reason="invalid schema",
        )
        assert "/tmp/cache.db" in str(exc)
        assert "invalid schema" in str(exc)

    def test_cache_version_mismatch(self):
        """Test CacheVersionMismatchError."""
        exc = CacheVersionMismatchError(
            cache_version="1.0",
            current_version="2.0",
        )
        assert "1.0" in str(exc)
        assert "2.0" in str(exc)
        assert "clearing the cache" in str(exc)


class TestStreamingExceptions:
    """Test streaming-related exceptions."""

    def test_chunk_processing_error_basic(self):
        """Test ChunkProcessingError with basic message."""
        exc = ChunkProcessingError()
        assert "Chunk processing failed" in str(exc)

    def test_chunk_processing_error_with_index(self):
        """Test ChunkProcessingError with chunk index."""
        exc = ChunkProcessingError(
            chunk_index=5,
            total_chunks=10,
        )
        assert "chunk 6/10" in str(exc)  # 1-indexed for display


class TestExceptionCatching:
    """Test that exceptions can be caught at various hierarchy levels."""

    def test_catch_by_base_class(self):
        """Test catching by PolarsAprioriError catches all."""
        with pytest.raises(PolarsAprioriError):
            raise NoFrequentItemsetsError()

        with pytest.raises(PolarsAprioriError):
            raise PredicateMismatchError()

        with pytest.raises(PolarsAprioriError):
            raise EmbeddingModelNotAvailableError()

    def test_catch_by_category(self):
        """Test catching by category catches related errors."""
        with pytest.raises(MiningError):
            raise NoFrequentItemsetsError()

        with pytest.raises(MiningError):
            raise NoRulesFoundError()

        with pytest.raises(EmbeddingError):
            raise EmbeddingModelNotAvailableError()

    def test_specific_catch_doesnt_catch_unrelated(self):
        """Test that specific catches don't catch unrelated errors."""
        with pytest.raises(MiningError):
            # This should NOT be caught by EmbeddingError
            try:
                raise NoRulesFoundError()
            except EmbeddingError:
                pytest.fail("EmbeddingError shouldn't catch MiningError")
