"""Lossless player-permutation canonicalisation for symmetric Q3 states."""

from __future__ import annotations

from .model import Action, JointState, StateValue


def _player_key(player: object) -> tuple[int, int, int, int, int, int]:
    return (
        int(player.status),
        player.position,
        player.water,
        player.food,
        player.cash_scaled,
        player.fixed_payoff_scaled,
    )


def canonicalize_state(state: JointState) -> tuple[JointState, tuple[int, ...]]:
    """Return canonical state and mapping canonical index -> original index."""
    ordered = sorted(enumerate(state), key=lambda item: (_player_key(item[1]), item[0]))
    canonical_to_original = tuple(original for original, _ in ordered)
    canonical = tuple(player for _, player in ordered)
    return canonical, canonical_to_original


def value_to_original(value: StateValue, canonical_to_original: tuple[int, ...]) -> StateValue:
    n = len(canonical_to_original)
    values = [0.0] * n
    success = [0.0] * n
    for canonical_i, original_i in enumerate(canonical_to_original):
        values[original_i] = value.value[canonical_i]
        success[original_i] = value.success[canonical_i]
    return StateValue(tuple(values), tuple(success))


def actions_to_original(
    actions: tuple[Action, ...], canonical_to_original: tuple[int, ...]
) -> tuple[Action, ...]:
    original: list[Action | None] = [None] * len(actions)
    for canonical_i, original_i in enumerate(canonical_to_original):
        original[original_i] = actions[canonical_i]
    if any(action is None for action in original):
        raise AssertionError("invalid player permutation")
    return tuple(action for action in original if action is not None)
