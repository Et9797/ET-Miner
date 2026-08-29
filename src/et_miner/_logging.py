"""Structured logging for ET-Miner.

Provides a configured loguru logger with sensible defaults and easy
configuration for different use cases. loguru is a hard runtime dependency
(declared in pyproject), so no stdlib fallback is needed.

Example:
    >>> from et_miner._logging import logger
    >>> logger.info("Mining started", min_support=0.01)
    >>> logger.debug("Found itemsets", count=150)

Configuration:
    >>> from et_miner._logging import configure_logging
    >>> configure_logging(level="DEBUG", format="json")
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger as _loguru_logger

from et_miner import _env

if TYPE_CHECKING:
    from loguru import Logger


# =============================================================================
# Logger Configuration
# =============================================================================

# Default format for console output
DEFAULT_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# Compact format for production
COMPACT_FORMAT = "{time:HH:mm:ss} | {level: <8} | {message}"

# JSON format for structured logging
JSON_FORMAT = (
    '{{"time":"{time:YYYY-MM-DD HH:mm:ss.SSS}", '
    '"level":"{level}", '
    '"module":"{name}", '
    '"function":"{function}", '
    '"line":{line}, '
    '"message":"{message}"}}'
)


# The main logger instance
logger: "Logger" = _loguru_logger.bind(library="et-miner")


def configure_logging(
    topic: str = "et-miner",
    level: str = "INFO",
    format: str = "compact",
    stderr: bool = False,
    stdout: bool = False,
    serialize: bool = False,
    rotation: str = "10 MB",
    retention: int = 3,
    log_dir: str | Path | None = None,
) -> None:
    """Configure topic-based file logging.

    Logs are written to {log_dir}/{topic}/{topic}.log with automatic
    rotation. The directory defaults to $ET_MINER_LOG_DIR, falling back to
    ~/.cache/et-miner/logs — never inside the installed package. Terminal
    output (stderr/stdout) is OFF by default — tail the log files.

    Args:
        topic: Project topic (e.g. 'et-miner')
        level: Log level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Format preset ('default', 'compact', 'json') or custom format string
        stderr: Also log to stderr (default: False)
        stdout: Also log to stdout (default: False)
        serialize: If True, log as JSON
        rotation: Log file rotation size (default: '10 MB')
        retention: Number of rotated files to keep (default: 3)
        log_dir: Base directory for log files (default: $ET_MINER_LOG_DIR
            or ~/.cache/et-miner/logs)

    Example:
        >>> configure_logging(topic="et-miner", level="DEBUG")
        >>> configure_logging(topic="et-miner", stdout=True)
    """
    # Determine format string
    if format == "default":
        fmt = DEFAULT_FORMAT
    elif format == "compact":
        fmt = COMPACT_FORMAT
    elif format == "json":
        fmt = JSON_FORMAT
        serialize = True
    else:
        fmt = format

    # Remove existing handlers
    _loguru_logger.remove()

    # File sink — topic-based directory
    base_dir = Path(log_dir) if log_dir is not None else _env.log_dir()
    topic_dir = base_dir / topic
    topic_dir.mkdir(parents=True, exist_ok=True)
    log_file = topic_dir / f"{topic}.log"

    _loguru_logger.add(
        str(log_file),
        format=fmt,
        level=level.upper(),
        rotation=rotation,
        retention=retention,
        serialize=serialize,
    )

    # Optional terminal sinks
    if stderr:
        _loguru_logger.add(sys.stderr, format=fmt, level=level.upper(), colorize=True)
    if stdout:
        _loguru_logger.add(sys.stdout, format=fmt, level=level.upper(), colorize=True)


def disable_logging() -> None:
    """Disable all et-miner logging.

    Useful for tests or when embedding in other applications.
    """
    _loguru_logger.disable("et_miner")


def enable_logging() -> None:
    """Re-enable et-miner logging after disable_logging()."""
    _loguru_logger.enable("et_miner")


# =============================================================================
# Context Managers
# =============================================================================


class LogContext:
    """Context manager for adding context to log messages.

    Example:
        >>> with LogContext(operation="mining", min_support=0.01):
        ...     logger.info("Started")  # Includes operation and min_support
        ...     logger.info("Completed")
    """

    def __init__(self, **context: Any):
        self.context = context
        self._token = None

    def __enter__(self) -> "LogContext":
        global logger
        logger = _loguru_logger.bind(**self.context)
        return self

    def __exit__(self, *args: Any) -> None:
        global logger
        logger = _loguru_logger.bind(library="et-miner")


# =============================================================================
# Convenience Functions
# =============================================================================


def log_mining_start(
    min_support: float,
    total_transactions: int,
    max_length: int | None = None,
) -> None:
    """Log the start of a mining operation."""
    logger.info(
        "Mining started",
        min_support=min_support,
        transactions=total_transactions,
        max_length=max_length,
    )


def log_mining_complete(
    itemsets: int,
    rules: int,
    elapsed_seconds: float,
) -> None:
    """Log completion of a mining operation."""
    logger.success(
        "Mining completed",
        itemsets=itemsets,
        rules=rules,
        elapsed=f"{elapsed_seconds:.2f}s",
    )


def log_phase_timing(phase: str, elapsed_seconds: float, **extra: Any) -> None:
    """Log timing for a specific phase."""
    logger.debug(
        f"Phase completed: {phase}",
        elapsed=f"{elapsed_seconds:.3f}s",
        **extra,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "logger",
    "configure_logging",
    "disable_logging",
    "enable_logging",
    "LogContext",
    "log_mining_start",
    "log_mining_complete",
    "log_phase_timing",
    "DEFAULT_FORMAT",
    "COMPACT_FORMAT",
    "JSON_FORMAT",
]
