"""Progress utilities for live terminal output during Apriori execution.

Provides callback factories for real-time k-level progress tracking.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class LevelStats:
    """Statistics for a single k-level."""

    k: int
    n_candidates: int
    n_frequent: int
    duration_ms: float


@dataclass
class ProgressTracker:
    """Tracks progress across all k-levels.

    Accumulates statistics and provides summary methods.
    """

    start_time: float = field(default_factory=time.perf_counter)
    levels: list[LevelStats] = field(default_factory=list)

    def record_level(
        self, k: int, n_candidates: int, n_frequent: int, duration_ms: float
    ) -> None:
        """Record statistics for a completed k-level."""
        self.levels.append(
            LevelStats(
                k=k,
                n_candidates=n_candidates,
                n_frequent=n_frequent,
                duration_ms=duration_ms,
            )
        )

    @property
    def elapsed_seconds(self) -> float:
        """Total elapsed time since start."""
        return time.perf_counter() - self.start_time

    @property
    def total_candidates(self) -> int:
        """Total candidates generated across all levels."""
        return sum(level.n_candidates for level in self.levels)

    @property
    def total_frequent(self) -> int:
        """Total frequent itemsets found across all levels."""
        return sum(level.n_frequent for level in self.levels)

    def summary(self) -> str:
        """Generate a summary of all k-levels."""
        lines = []
        for level in self.levels:
            lines.append(
                f"  k={level.k}: {level.n_candidates:,} candidates "
                f"→ {level.n_frequent:,} frequent ({level.duration_ms:.1f}ms)"
            )
        lines.append(f"\n  Total: {self.total_frequent:,} itemsets in {self.elapsed_seconds:.2f}s")
        return "\n".join(lines)


def create_live_level_printer(
    prefix: str = "  ",
    show_elapsed: bool = True,
) -> tuple[Callable[[int, int, int, float], None], ProgressTracker]:
    """Create a callback that prints live k-level statistics.

    Args:
        prefix: String prefix for each line (default: 2 spaces for indentation).
        show_elapsed: Whether to show cumulative elapsed time.

    Returns:
        Tuple of (callback function, ProgressTracker instance).
        The callback can be passed directly to apriori(level_callback=...).
        The tracker can be used to access accumulated statistics after completion.

    Example:
        >>> callback, tracker = create_live_level_printer()
        >>> result = apriori(df, min_support=0.01, level_callback=callback)
        >>> print(f"Total itemsets: {tracker.total_frequent}")
    """
    tracker = ProgressTracker()

    def printer(k: int, n_candidates: int, n_frequent: int, duration_ms: float) -> None:
        """Print live progress for a k-level."""
        tracker.record_level(k, n_candidates, n_frequent, duration_ms)

        if show_elapsed:
            elapsed = tracker.elapsed_seconds
            logger.info(
                f"{prefix}k={k}: {n_candidates:,} candidates → "
                f"{n_frequent:,} frequent ({duration_ms:.1f}ms) "
                f"[total: {elapsed:.1f}s]"
            )
        else:
            logger.info(
                f"{prefix}k={k}: {n_candidates:,} candidates → "
                f"{n_frequent:,} frequent ({duration_ms:.1f}ms)"
            )

    return printer, tracker


def create_silent_tracker() -> tuple[Callable[[int, int, int, float], None], ProgressTracker]:
    """Create a callback that tracks statistics without printing.

    Useful when you want to collect statistics but control output yourself.

    Returns:
        Tuple of (callback function, ProgressTracker instance).
    """
    tracker = ProgressTracker()

    def recorder(k: int, n_candidates: int, n_frequent: int, duration_ms: float) -> None:
        """Record level statistics without output."""
        tracker.record_level(k, n_candidates, n_frequent, duration_ms)

    return recorder, tracker
