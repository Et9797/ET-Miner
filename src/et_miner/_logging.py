"""Structured logging for ET-Miner.

Provides a configured loguru logger with sensible defaults and easy
configuration for different use cases.

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

# Try to import loguru, fall back to stdlib logging
try:
    from loguru import logger as _loguru_logger

    HAS_LOGURU = True
except ImportError:
    HAS_LOGURU = False
    _loguru_logger = None  # type: ignore[assignment]

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


class LoguruFallback:
    """Minimal stdlib logging fallback when loguru is not installed.

    Provides the same interface as loguru but uses stdlib logging.
    """

    def __init__(self):
        import logging

        self._logger = logging.getLogger("et_miner")
        self._configured = False

    def _ensure_configured(self):
        if not self._configured:
            import logging

            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
            )
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)
            self._configured = True

    def _format_message(self, message: str, **kwargs: Any) -> str:
        if kwargs:
            extra = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            return f"{message} | {extra}"
        return message

    def trace(self, message: str, **kwargs: Any) -> None:
        self._ensure_configured()
        self._logger.debug(self._format_message(f"[TRACE] {message}", **kwargs))

    def debug(self, message: str, **kwargs: Any) -> None:
        self._ensure_configured()
        self._logger.debug(self._format_message(message, **kwargs))

    def info(self, message: str, **kwargs: Any) -> None:
        self._ensure_configured()
        self._logger.info(self._format_message(message, **kwargs))

    def success(self, message: str, **kwargs: Any) -> None:
        self._ensure_configured()
        self._logger.info(self._format_message(f"[SUCCESS] {message}", **kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        self._ensure_configured()
        self._logger.warning(self._format_message(message, **kwargs))

    def error(self, message: str, **kwargs: Any) -> None:
        self._ensure_configured()
        self._logger.error(self._format_message(message, **kwargs))

    def critical(self, message: str, **kwargs: Any) -> None:
        self._ensure_configured()
        self._logger.critical(self._format_message(message, **kwargs))

    def exception(self, message: str, **kwargs: Any) -> None:
        self._ensure_configured()
        self._logger.exception(self._format_message(message, **kwargs))

    def bind(self, **kwargs: Any) -> "LoguruFallback":
        # Fallback doesn't support bind, return self
        return self

    def opt(self, **kwargs: Any) -> "LoguruFallback":
        # Fallback doesn't support opt, return self
        return self


def _create_logger() -> "Logger | LoguruFallback":
    """Create the appropriate logger based on available packages."""
    if HAS_LOGURU:
        # Create a child logger for et-miner
        return _loguru_logger.bind(library="et-miner")
    return LoguruFallback()


# The main logger instance
logger: "Logger | LoguruFallback" = _create_logger()


def _get_log_dir() -> Path:
    """Resolve the project-level logs/ directory."""
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir


def configure_logging(
    topic: str = "et-miner",
    level: str = "INFO",
    format: str = "compact",
    stderr: bool = False,
    stdout: bool = False,
    serialize: bool = False,
    rotation: str = "10 MB",
    retention: int = 3,
) -> None:
    """Configure topic-based file logging.

    Logs are written to logs/{topic}/{topic}.log with automatic rotation.
    Terminal output (stderr/stdout) is OFF by default — tail the log files.

    Args:
        topic: Project topic (e.g. 'et-miner')
        level: Log level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Format preset ('default', 'compact', 'json') or custom format string
        stderr: Also log to stderr (default: False)
        stdout: Also log to stdout (default: False)
        serialize: If True, log as JSON
        rotation: Log file rotation size (default: '10 MB')
        retention: Number of rotated files to keep (default: 3)

    Example:
        >>> configure_logging(topic="et-miner", level="DEBUG")
        >>> configure_logging(topic="et-miner", stdout=True)
    """
    if not HAS_LOGURU:
        import logging

        level_map = {
            "TRACE": logging.DEBUG, "DEBUG": logging.DEBUG,
            "INFO": logging.INFO, "SUCCESS": logging.INFO,
            "WARNING": logging.WARNING, "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        if isinstance(logger, LoguruFallback):
            logger._logger.setLevel(level_map.get(level.upper(), logging.INFO))
        return

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
    log_dir = _get_log_dir() / topic
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{topic}.log"

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
    if HAS_LOGURU:
        _loguru_logger.disable("et_miner")
    elif isinstance(logger, LoguruFallback):
        import logging

        logger._logger.setLevel(logging.CRITICAL + 1)


def enable_logging() -> None:
    """Re-enable et-miner logging after disable_logging()."""
    if HAS_LOGURU:
        _loguru_logger.enable("et_miner")
    elif isinstance(logger, LoguruFallback):
        import logging

        logger._logger.setLevel(logging.INFO)


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
        if HAS_LOGURU:
            global logger
            logger = _loguru_logger.bind(**self.context)
        return self

    def __exit__(self, *args: Any) -> None:
        if HAS_LOGURU:
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
    "HAS_LOGURU",
    "DEFAULT_FORMAT",
    "COMPACT_FORMAT",
    "JSON_FORMAT",
]
