"""Configuration management for et-miner.

Provides TOML-based configuration with environment variable overrides
and sensible defaults.

Example:
    >>> from et_miner.config import Config, load_config
    >>>
    >>> # Load from file
    >>> config = load_config("et-miner.toml")
    >>>
    >>> # Or use defaults with env overrides
    >>> config = Config.from_env()
    >>>
    >>> # Access settings
    >>> config.apriori.min_support
    0.01
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from et_miner.exceptions import (
    ConfigFileNotFoundError,
    InvalidConfigurationError,
)

# Try to import tomllib (Python 3.11+) or tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[import-not-found]
    except ImportError:
        tomllib = None  # type: ignore[assignment]


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class AprioriConfig:
    """Configuration for Apriori algorithm parameters."""

    min_support: float = 0.01
    """Minimum support threshold (0.0 to 1.0)."""

    max_length: int | None = None
    """Maximum itemset length. None means no limit."""

    sparse: bool = True
    """Use sparse matrix operations when beneficial."""

    n_jobs: int = -1
    """Number of parallel workers. -1 means use all CPUs."""

    def validate(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.min_support <= 1.0:
            raise InvalidConfigurationError("min_support", self.min_support, "must be between 0.0 and 1.0")
        if self.max_length is not None and self.max_length < 1:
            raise InvalidConfigurationError("max_length", self.max_length, "must be at least 1")


@dataclass
class RAGConfig:
    """Configuration for RAG integration features."""

    lift_threshold: float = 1.5
    """Minimum lift for rule consideration in query expansion."""

    confidence_threshold: float = 0.6
    """Minimum confidence for rule consideration."""

    semantic_threshold: float = 0.5
    """Similarity threshold for semantic predicate activation."""

    alpha: float = 0.7
    """Weight for vector similarity in hybrid scoring (0.0 to 1.0)."""

    max_expansions: int = 5
    """Maximum predicates to add during expansion."""

    def validate(self) -> None:
        """Validate configuration values."""
        if self.lift_threshold < 1.0:
            raise InvalidConfigurationError("lift_threshold", self.lift_threshold, "must be at least 1.0")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise InvalidConfigurationError(
                "confidence_threshold", self.confidence_threshold, "must be between 0.0 and 1.0"
            )
        if not 0.0 <= self.semantic_threshold <= 1.0:
            raise InvalidConfigurationError(
                "semantic_threshold", self.semantic_threshold, "must be between 0.0 and 1.0"
            )
        if not 0.0 <= self.alpha <= 1.0:
            raise InvalidConfigurationError("alpha", self.alpha, "must be between 0.0 and 1.0")


@dataclass
class StreamingConfig:
    """Configuration for streaming Apriori."""

    chunk_size: int = 100_000
    """Number of transactions per chunk."""

    memory_budget_mb: int = 1024
    """Target memory budget in MB."""

    prefetch_chunks: int = 2
    """Number of chunks to prefetch."""

    def validate(self) -> None:
        """Validate configuration values."""
        if self.chunk_size < 1000:
            raise InvalidConfigurationError("chunk_size", self.chunk_size, "must be at least 1000")
        if self.memory_budget_mb < 64:
            raise InvalidConfigurationError("memory_budget_mb", self.memory_budget_mb, "must be at least 64")


@dataclass
class CacheConfig:
    """Configuration for caching."""

    enabled: bool = True
    """Whether caching is enabled."""

    directory: str = ".et_miner_cache"
    """Directory for cache files."""

    max_size_mb: int = 512
    """Maximum cache size in MB."""

    ttl_hours: int = 24
    """Time-to-live for cache entries in hours."""

    def validate(self) -> None:
        """Validate configuration values."""
        if self.max_size_mb < 1:
            raise InvalidConfigurationError("max_size_mb", self.max_size_mb, "must be positive")
        if self.ttl_hours < 1:
            raise InvalidConfigurationError("ttl_hours", self.ttl_hours, "must be positive")


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = "INFO"
    """Log level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL)."""

    format: str = "default"
    """Log format preset (default, compact, json)."""

    file: str | None = None
    """Path to log file. None means stderr only."""

    def validate(self) -> None:
        """Validate configuration values."""
        valid_levels = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
        if self.level.upper() not in valid_levels:
            raise InvalidConfigurationError("level", self.level, f"must be one of {valid_levels}")
        valid_formats = {"default", "compact", "json"}
        if self.format not in valid_formats:
            raise InvalidConfigurationError("format", self.format, f"must be one of {valid_formats}")


@dataclass
class Config:
    """Main configuration container for et-miner.

    Example:
        >>> config = Config()
        >>> config.apriori.min_support = 0.05
        >>> config.validate()
    """

    apriori: AprioriConfig = field(default_factory=AprioriConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> None:
        """Validate all configuration sections."""
        self.apriori.validate()
        self.rag.validate()
        self.streaming.validate()
        self.cache.validate()
        self.logging.validate()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create Config from a dictionary (e.g., parsed TOML).

        Args:
            data: Dictionary with configuration sections

        Returns:
            Config instance

        Example:
            >>> data = {"apriori": {"min_support": 0.05}}
            >>> config = Config.from_dict(data)
        """
        config = cls()

        if "apriori" in data:
            for key, value in data["apriori"].items():
                if hasattr(config.apriori, key):
                    setattr(config.apriori, key, value)

        if "rag" in data:
            for key, value in data["rag"].items():
                if hasattr(config.rag, key):
                    setattr(config.rag, key, value)

        if "streaming" in data:
            for key, value in data["streaming"].items():
                if hasattr(config.streaming, key):
                    setattr(config.streaming, key, value)

        if "cache" in data:
            for key, value in data["cache"].items():
                if hasattr(config.cache, key):
                    setattr(config.cache, key, value)

        if "logging" in data:
            for key, value in data["logging"].items():
                if hasattr(config.logging, key):
                    setattr(config.logging, key, value)

        return config

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        """Load configuration from a TOML file.

        Args:
            path: Path to TOML configuration file

        Returns:
            Config instance

        Raises:
            ConfigFileNotFoundError: If file doesn't exist
            InvalidConfigurationError: If TOML is invalid

        Example:
            >>> config = Config.from_file("et-miner.toml")
        """
        if tomllib is None:
            raise ImportError(
                "tomllib or tomli is required for TOML parsing. Install with: pip install tomli (Python < 3.11)"
            )

        path = Path(path)
        if not path.exists():
            raise ConfigFileNotFoundError(str(path))

        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except Exception as e:
            raise InvalidConfigurationError("file", str(path), f"failed to parse TOML: {e}")

        config = cls.from_dict(data)
        config.validate()
        return config

    @classmethod
    def from_env(cls, prefix: str = "ET_MINER") -> "Config":
        """Create Config with environment variable overrides.

        Environment variables are named: {PREFIX}_{SECTION}_{KEY}
        For example: ET_MINER_APRIORI_MIN_SUPPORT=0.05

        Args:
            prefix: Environment variable prefix

        Returns:
            Config instance with env overrides applied

        Example:
            >>> # With ET_MINER_RAG_LIFT_THRESHOLD=2.0 set
            >>> config = Config.from_env()
            >>> config.rag.lift_threshold
            2.0
        """
        config = cls()

        env_mappings = {
            # Apriori
            f"{prefix}_APRIORI_MIN_SUPPORT": ("apriori", "min_support", float),
            f"{prefix}_APRIORI_MAX_LENGTH": ("apriori", "max_length", lambda x: None if x == "" else int(x)),
            f"{prefix}_APRIORI_SPARSE": ("apriori", "sparse", lambda x: x.lower() in ("true", "1", "yes")),
            f"{prefix}_APRIORI_N_JOBS": ("apriori", "n_jobs", int),
            # RAG
            f"{prefix}_RAG_LIFT_THRESHOLD": ("rag", "lift_threshold", float),
            f"{prefix}_RAG_CONFIDENCE_THRESHOLD": ("rag", "confidence_threshold", float),
            f"{prefix}_RAG_SEMANTIC_THRESHOLD": ("rag", "semantic_threshold", float),
            f"{prefix}_RAG_ALPHA": ("rag", "alpha", float),
            f"{prefix}_RAG_MAX_EXPANSIONS": ("rag", "max_expansions", int),
            # Streaming
            f"{prefix}_STREAMING_CHUNK_SIZE": ("streaming", "chunk_size", int),
            f"{prefix}_STREAMING_MEMORY_BUDGET_MB": ("streaming", "memory_budget_mb", int),
            # Cache
            f"{prefix}_CACHE_ENABLED": ("cache", "enabled", lambda x: x.lower() in ("true", "1", "yes")),
            f"{prefix}_CACHE_DIRECTORY": ("cache", "directory", str),
            f"{prefix}_CACHE_MAX_SIZE_MB": ("cache", "max_size_mb", int),
            # Logging
            f"{prefix}_LOGGING_LEVEL": ("logging", "level", str),
            f"{prefix}_LOGGING_FORMAT": ("logging", "format", str),
            f"{prefix}_LOGGING_FILE": ("logging", "file", lambda x: None if x == "" else x),
        }

        for env_var, (section, key, converter) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    converted = converter(value)
                    section_obj = getattr(config, section)
                    setattr(section_obj, key, converted)
                except (ValueError, TypeError) as e:
                    raise InvalidConfigurationError(env_var, value, f"failed to parse: {e}")

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of config
        """
        return {
            "apriori": {
                "min_support": self.apriori.min_support,
                "max_length": self.apriori.max_length,
                "sparse": self.apriori.sparse,
                "n_jobs": self.apriori.n_jobs,
            },
            "rag": {
                "lift_threshold": self.rag.lift_threshold,
                "confidence_threshold": self.rag.confidence_threshold,
                "semantic_threshold": self.rag.semantic_threshold,
                "alpha": self.rag.alpha,
                "max_expansions": self.rag.max_expansions,
            },
            "streaming": {
                "chunk_size": self.streaming.chunk_size,
                "memory_budget_mb": self.streaming.memory_budget_mb,
                "prefetch_chunks": self.streaming.prefetch_chunks,
            },
            "cache": {
                "enabled": self.cache.enabled,
                "directory": self.cache.directory,
                "max_size_mb": self.cache.max_size_mb,
                "ttl_hours": self.cache.ttl_hours,
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "file": self.logging.file,
            },
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def load_config(
    path: str | Path | None = None,
    use_env: bool = True,
) -> Config:
    """Load configuration from file and/or environment.

    Searches for config file in this order:
    1. Explicit path if provided
    2. ./et-miner.toml
    3. ~/.config/et-miner/config.toml

    Environment variables override file settings.

    Args:
        path: Explicit path to config file (optional)
        use_env: Whether to apply environment variable overrides

    Returns:
        Config instance
    """
    config = None

    # Try explicit path first
    if path is not None:
        config = Config.from_file(path)

    # Try default locations
    if config is None:
        default_paths = [
            Path("et-miner.toml"),
            Path.home() / ".config" / "et-miner" / "config.toml",
        ]
        for default_path in default_paths:
            if default_path.exists():
                try:
                    config = Config.from_file(default_path)
                    break
                except Exception:
                    continue

    # Fall back to defaults
    if config is None:
        config = Config()

    # Apply environment overrides
    if use_env:
        env_config = Config.from_env()
        # Merge env config on top of file config
        # (This is a simple merge - env values override file values)
        config_dict = config.to_dict()
        env_dict = env_config.to_dict()

        for section in config_dict:
            for key in config_dict[section]:
                # Check if env was explicitly set (non-default value)
                default_config = Config()
                default_section = getattr(default_config, section)
                default_value = getattr(default_section, key)
                env_section = getattr(env_config, section)
                env_value = getattr(env_section, key)

                # If env value differs from default, it was explicitly set
                if env_value != default_value:
                    section_obj = getattr(config, section)
                    setattr(section_obj, key, env_value)

    config.validate()
    return config


def get_default_config() -> Config:
    return Config()


# =============================================================================
# Module-level config instance
# =============================================================================

_global_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance.

    Loads config on first access using load_config().

    Returns:
        Global Config instance
    """
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def set_config(config: Config) -> None:
    """Set the global configuration instance.

    Args:
        config: Config to use globally
    """
    global _global_config
    config.validate()
    _global_config = config


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "Config",
    "AprioriConfig",
    "RAGConfig",
    "StreamingConfig",
    "CacheConfig",
    "LoggingConfig",
    "load_config",
    "get_config",
    "set_config",
    "get_default_config",
]
