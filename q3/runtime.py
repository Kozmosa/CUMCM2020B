"""Wall-clock, memory, and progress budgets for long Q3 searches."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Callable


class BudgetExceeded(RuntimeError):
    pass


def current_rss_bytes() -> int:
    """Return current resident memory on Linux, falling back to max RSS."""
    statm = Path("/proc/self/statm")
    try:
        fields = statm.read_text(encoding="ascii").split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value * (1024 if os.name != "darwin" else 1)


@dataclass(frozen=True)
class BudgetSnapshot:
    elapsed_seconds: float
    rss_bytes: int
    states: int
    profiles: int
    closing: bool


class BudgetManager:
    """Cooperative resource limiter with a deterministic closeout window."""

    def __init__(
        self,
        *,
        wall_seconds: float | None = None,
        memory_bytes: int | None = None,
        max_states: int | None = None,
        max_profiles: int | None = None,
        closeout_seconds: float = 3600.0,
        checkpoint_callback: Callable[[], object] | None = None,
    ) -> None:
        for value, label in (
            (wall_seconds, "wall_seconds"),
            (memory_bytes, "memory_bytes"),
            (max_states, "max_states"),
            (max_profiles, "max_profiles"),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{label} must be positive")
        if closeout_seconds < 0:
            raise ValueError("closeout_seconds cannot be negative")
        self.wall_seconds = wall_seconds
        self.memory_bytes = memory_bytes
        self.max_states = max_states
        self.max_profiles = max_profiles
        self.closeout_seconds = closeout_seconds
        self.checkpoint_callback = checkpoint_callback
        self.started = monotonic()
        self._cancelled = threading.Event()
        self._closeout_started = False

    def cancel(self) -> None:
        self._cancelled.set()

    def set_checkpoint_callback(self, callback: Callable[[], object] | None) -> None:
        self.checkpoint_callback = callback

    def snapshot(self, *, states: int = 0, profiles: int = 0) -> BudgetSnapshot:
        elapsed = monotonic() - self.started
        closing = (
            self.wall_seconds is not None
            and elapsed >= max(0.0, self.wall_seconds - self.closeout_seconds)
        )
        return BudgetSnapshot(elapsed, current_rss_bytes(), states, profiles, closing)

    def check(self, *, states: int = 0, profiles: int = 0) -> BudgetSnapshot:
        snap = self.snapshot(states=states, profiles=profiles)
        if self._cancelled.is_set():
            raise BudgetExceeded("search cancellation requested")
        if self.max_states is not None and states >= self.max_states:
            raise BudgetExceeded(f"state budget reached {states:,}")
        if self.max_profiles is not None and profiles >= self.max_profiles:
            raise BudgetExceeded(f"profile budget reached {profiles:,}")
        if self.memory_bytes is not None and snap.rss_bytes >= self.memory_bytes:
            raise BudgetExceeded(
                f"RSS {snap.rss_bytes / 2**30:.2f} GiB reached the memory budget"
            )
        if snap.closing and not self._closeout_started:
            self._closeout_started = True
            if self.checkpoint_callback is not None:
                self.checkpoint_callback()
        if self.wall_seconds is not None and snap.elapsed_seconds >= self.wall_seconds:
            raise BudgetExceeded(
                f"wall-clock budget reached {snap.elapsed_seconds:.1f} seconds"
            )
        return snap
