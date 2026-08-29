"""Memory profiling utilities for et-miner.

This module provides tools for tracking memory usage during Apriori execution,
useful for benchmarking, debugging, and understanding memory bottlenecks.
"""

from __future__ import annotations

import resource
import time
from dataclasses import dataclass, field


@dataclass
class PhaseMetrics:
    """Metrics for a single execution phase."""

    name: str
    duration_ms: float
    memory_delta_mb: float
    peak_memory_mb: float
    extra: dict[str, int | float | str] = field(default_factory=dict)


@dataclass
class ProfilingSession:
    """Session for collecting memory and timing metrics.

    Tracks RSS (Resident Set Size) memory usage across different phases
    of Apriori execution. Uses resource.getrusage() for cross-platform
    memory measurement.

    Example:
        >>> session = ProfilingSession()
        >>> session.start_phase("matrix_build")
        >>> # ... do matrix building ...
        >>> session.end_phase(n_items=1000, n_transactions=100000)
        >>> print(session.summary())
    """

    phases: list[PhaseMetrics] = field(default_factory=list)
    _current_phase: str | None = field(default=None, repr=False)
    _phase_start_time: float | None = field(default=None, repr=False)
    _phase_start_memory: float | None = field(default=None, repr=False)

    @staticmethod
    def _get_rss_mb() -> float:
        """Get current RSS memory in MB."""
        # maxrss is in KB on Linux, bytes on macOS
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux returns KB, convert to MB
        return rss / 1024

    def start_phase(self, name: str) -> None:
        """Start timing a new phase.

        Args:
            name: Identifier for this phase (e.g., "matrix_build", "k2_support").
        """
        if self._current_phase is not None:
            raise RuntimeError(
                f"Phase '{self._current_phase}' already in progress. "
                "Call end_phase() first."
            )
        self._current_phase = name
        self._phase_start_time = time.perf_counter()
        self._phase_start_memory = self._get_rss_mb()

    def end_phase(self, **extra: int | float | str) -> PhaseMetrics:
        """End the current phase and record metrics.

        Args:
            **extra: Additional metrics to record (e.g., n_candidates=1000).

        Returns:
            PhaseMetrics for the completed phase.
        """
        if self._current_phase is None:
            raise RuntimeError("No phase in progress. Call start_phase() first.")

        end_time = time.perf_counter()
        end_memory = self._get_rss_mb()

        metrics = PhaseMetrics(
            name=self._current_phase,
            duration_ms=(end_time - self._phase_start_time) * 1000,
            memory_delta_mb=end_memory - self._phase_start_memory,
            peak_memory_mb=end_memory,
            extra=dict(extra),
        )

        self.phases.append(metrics)
        self._current_phase = None
        self._phase_start_time = None
        self._phase_start_memory = None

        return metrics

    def summary(self) -> str:
        """Generate a human-readable summary of all phases.

        Returns:
            Formatted string with phase timings and memory usage.
        """
        if not self.phases:
            return "No phases recorded."

        lines = ["Phase Profiling Summary", "=" * 60]

        total_time_s = sum(p.duration_ms for p in self.phases) / 1000
        max_memory = max(p.peak_memory_mb for p in self.phases)

        for phase in self.phases:
            extra_str = ", ".join(f"{k}={v}" for k, v in phase.extra.items())
            duration_s = phase.duration_ms / 1000
            lines.append(
                f"  {phase.name:20s} | "
                f"{duration_s:8.3f}s | "
                f"Δ{phase.memory_delta_mb:+8.1f}MB | "
                f"peak {phase.peak_memory_mb:8.1f}MB"
            )
            if extra_str:
                lines.append(f"    {extra_str}")

        lines.append("-" * 60)
        lines.append(f"  {'TOTAL':20s} | {total_time_s:8.3f}s | peak {max_memory:8.1f}MB")

        return "\n".join(lines)

    def __str__(self) -> str:
        """Return formatted summary when printed."""
        return self.summary()

    def __repr__(self) -> str:
        """Return formatted summary for repr() calls (e.g., in tuples)."""
        return self.__str__()

    def to_dict(self) -> dict:
        """Convert session to dictionary for serialization.

        Returns:
            Dictionary with all phase metrics.
        """
        return {
            "phases": [
                {
                    "name": p.name,
                    "duration_ms": p.duration_ms,
                    "memory_delta_mb": p.memory_delta_mb,
                    "peak_memory_mb": p.peak_memory_mb,
                    **p.extra,
                }
                for p in self.phases
            ],
            "total_duration_ms": sum(p.duration_ms for p in self.phases),
            "peak_memory_mb": max((p.peak_memory_mb for p in self.phases), default=0),
        }
