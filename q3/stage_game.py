"""Exact pure-strategy Nash detection and deterministic equilibrium selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .model import Action


class NoPureEquilibrium(RuntimeError):
    pass


@dataclass(frozen=True)
class PureEquilibrium:
    index: tuple[int, ...]
    actions: tuple[Action, ...]
    value: tuple[float, ...]
    success: tuple[float, ...]


def pure_nash_indices(
    payoff: np.ndarray, valid: np.ndarray, *, atol: float = 1e-10
) -> np.ndarray:
    """Return all pure Nash indices under a coupled joint-feasibility mask."""
    if payoff.ndim < 2:
        raise ValueError("payoff requires action axes and one player axis")
    action_shape = payoff.shape[:-1]
    n_players = payoff.shape[-1]
    if len(action_shape) != n_players:
        raise ValueError("one action axis is required per player")
    if valid.shape != action_shape:
        raise ValueError("valid mask shape does not match payoff action axes")

    nash = valid.copy()
    for player in range(n_players):
        player_payoff = np.where(valid, payoff[..., player], -np.inf)
        best = np.max(player_payoff, axis=player, keepdims=True)
        best_response = valid & (player_payoff >= best - atol)
        nash &= best_response
    return np.argwhere(nash)


def verify_pure_equilibrium(
    payoff: np.ndarray,
    valid: np.ndarray,
    index: tuple[int, ...],
    *,
    atol: float = 1e-10,
) -> tuple[bool, str]:
    if not valid[index]:
        return False, "candidate joint action is infeasible"
    n_players = payoff.shape[-1]
    for player in range(n_players):
        slicer: list[int | slice] = list(index)
        slicer[player] = slice(None)
        feasible = valid[tuple(slicer)]
        alternatives = payoff[tuple(slicer) + (player,)]
        best = np.max(np.where(feasible, alternatives, -np.inf))
        current = payoff[index + (player,)]
        if current < best - atol:
            return False, f"player {player} can improve {current} -> {best}"
    return True, "OK"


def _joint_action_code(actions: Sequence[Action]) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(action.code for action in actions)


def select_pure_equilibrium(
    indices: np.ndarray,
    payoff: np.ndarray,
    success: np.ndarray,
    action_sets: Sequence[Sequence[Action]],
) -> PureEquilibrium:
    if indices.size == 0:
        raise NoPureEquilibrium("complete pure-strategy scan found no Nash equilibrium")

    candidates: list[PureEquilibrium] = []
    for row in indices:
        index = tuple(int(x) for x in row)
        actions = tuple(action_sets[i][index[i]] for i in range(len(index)))
        candidates.append(
            PureEquilibrium(
                index=index,
                actions=actions,
                value=tuple(float(x) for x in payoff[index]),
                success=tuple(float(x) for x in success[index]),
            )
        )

    # Maximise success count, minimum payoff, total payoff; then minimise code.
    candidates.sort(key=lambda equilibrium: _joint_action_code(equilibrium.actions))
    return max(
        candidates,
        key=lambda equilibrium: (
            sum(equilibrium.success),
            min(equilibrium.value),
            sum(equilibrium.value),
        ),
    )
