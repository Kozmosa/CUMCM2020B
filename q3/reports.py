"""Public result and policy types shared by the Q3.1 and Q3.2 solvers."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from .model import Action


@dataclass(frozen=True, slots=True)
class PolicyEntry:
    """A pure action or a finite probability distribution over actions."""

    action: Action | None = None
    distribution: tuple[tuple[Action, float], ...] = ()

    def __post_init__(self) -> None:
        if (self.action is None) == (len(self.distribution) == 0):
            raise ValueError("PolicyEntry requires exactly one of action/distribution")
        if self.action is not None:
            return
        total = 0.0
        seen: set[Action] = set()
        for action, probability in self.distribution:
            if action in seen:
                raise ValueError("mixed policy contains a duplicate action")
            if probability < 0.0:
                raise ValueError("mixed policy probabilities cannot be negative")
            seen.add(action)
            total += probability
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"mixed policy probabilities sum to {total}, not 1")

    @classmethod
    def pure(cls, action: Action) -> "PolicyEntry":
        return cls(action=action)

    @classmethod
    def mixed(
        cls, distribution: tuple[tuple[Action, float], ...]
    ) -> "PolicyEntry":
        cleaned = tuple((action, float(p)) for action, p in distribution if p > 0.0)
        total = sum(p for _, p in cleaned)
        if total <= 0.0:
            raise ValueError("mixed policy must contain positive probability mass")
        normalized = tuple((action, p / total) for action, p in cleaned)
        return cls(distribution=normalized)

    @property
    def is_pure(self) -> bool:
        return self.action is not None


@dataclass(frozen=True)
class SolveReport:
    status: str
    value_lower: tuple[float, ...]
    value_upper: tuple[float, ...]
    success: tuple[float, ...]
    max_regret_lower: float
    max_regret_upper: float
    selection_complete: bool
    policy: Mapping[str, Any] = field(default_factory=dict)
    stats: Mapping[str, Any] = field(default_factory=dict)
    checkpoint: str | None = None


@dataclass(frozen=True)
class Q32SolveResult(SolveReport):
    backend: str = "adaptive"
    player_regret_lower: tuple[float, ...] = ()
    player_regret_upper: tuple[float, ...] = ()


def json_safe(value: Any) -> Any:
    """Convert reports to strict-JSON data without losing infinite gaps."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [json_safe(item) for item in value]
    return value
