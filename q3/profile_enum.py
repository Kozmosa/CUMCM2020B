"""Structured, lossless joint-action block enumeration for Q3."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import islice, product
from math import prod

import numpy as np

from .data import Q3Config, Weather
from .model import Action, JointState
from .transition import (
    apply_player_action_given_counts,
    count_interactions_scalar,
)


@dataclass(frozen=True, order=True, slots=True)
class ActionSkeleton:
    """Action fields that determine multiplayer interaction counts."""

    kind: int
    destination: int
    is_buyer: bool


@dataclass(frozen=True, slots=True)
class IndexedAction:
    index: int
    action: Action


@dataclass(frozen=True)
class StructuredProfileBlock:
    indices: np.ndarray
    profiles: tuple[tuple[Action, ...], ...]


@dataclass
class StructuredEnumerationStats:
    raw_profiles: int = 0
    feasible_profiles: int = 0
    skeleton_combinations: int = 0
    pruned_skeleton_combinations: int = 0

    @property
    def profiles_pruned(self) -> int:
        return self.raw_profiles - self.feasible_profiles


def action_skeleton(action: Action) -> ActionSkeleton:
    return ActionSkeleton(int(action.kind), action.destination, action.is_buyer)


def group_actions_by_skeleton(
    actions: Sequence[Action],
) -> tuple[tuple[ActionSkeleton, tuple[IndexedAction, ...]], ...]:
    grouped: dict[ActionSkeleton, list[IndexedAction]] = {}
    for index, action in enumerate(actions):
        grouped.setdefault(action_skeleton(action), []).append(
            IndexedAction(index, action)
        )
    return tuple((skeleton, tuple(grouped[skeleton])) for skeleton in sorted(grouped))


def _filtered_group_actions(
    cfg: Q3Config,
    state: JointState,
    weather: Weather,
    skeleton_groups: Sequence[tuple[ActionSkeleton, Sequence[IndexedAction]]],
) -> tuple[tuple[IndexedAction, ...], ...]:
    representatives = tuple(group[1][0].action for group in skeleton_groups)
    counts = count_interactions_scalar(state, representatives)
    filtered: list[tuple[IndexedAction, ...]] = []
    for player_index, (_, entries) in enumerate(skeleton_groups):
        feasible = tuple(
            entry
            for entry in entries
            if apply_player_action_given_counts(
                cfg,
                state[player_index],
                entry.action,
                weather,
                edge_count=counts.edge[player_index],
                mine_count=counts.mine[player_index],
                village_buyer_count=counts.village_buyers[player_index],
            )
            is not None
        )
        filtered.append(feasible)
    return tuple(filtered)


def iter_structured_profile_blocks(
    cfg: Q3Config,
    state: JointState,
    action_sets: Sequence[Sequence[Action]],
    weather: Weather,
    *,
    block_size: int,
    stats: StructuredEnumerationStats | None = None,
) -> Iterator[StructuredProfileBlock]:
    """Yield every jointly feasible profile exactly once in bounded blocks.

    Actions are first partitioned by fields that determine congestion and
    village price multipliers.  Inside a joint skeleton those multipliers are
    fixed, so feasibility separates by player and can be checked before the
    purchase-quantity Cartesian product is formed.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if len(action_sets) != cfg.n_players or len(state) != cfg.n_players:
        raise ValueError("state/action player count does not match configuration")

    enumeration_stats = stats if stats is not None else StructuredEnumerationStats()
    enumeration_stats.raw_profiles += prod(len(actions) for actions in action_sets)
    grouped_sets = tuple(group_actions_by_skeleton(actions) for actions in action_sets)
    index_buffer: list[list[int]] = []
    profile_buffer: list[tuple[Action, ...]] = []

    def flush() -> StructuredProfileBlock:
        indices = np.asarray(index_buffer, dtype=np.int64)
        profiles = tuple(profile_buffer)
        index_buffer.clear()
        profile_buffer.clear()
        return StructuredProfileBlock(indices=indices, profiles=profiles)

    for skeleton_groups in product(*grouped_sets):
        enumeration_stats.skeleton_combinations += 1
        filtered = _filtered_group_actions(cfg, state, weather, skeleton_groups)
        if any(not entries for entries in filtered):
            enumeration_stats.pruned_skeleton_combinations += 1
            continue
        feasible_count = prod(len(entries) for entries in filtered)
        enumeration_stats.feasible_profiles += feasible_count
        for row in product(*filtered):
            index_buffer.append([entry.index for entry in row])
            profile_buffer.append(tuple(entry.action for entry in row))
            if len(profile_buffer) == block_size:
                yield flush()
    if profile_buffer:
        yield flush()


def iter_index_blocks(
    action_counts: Sequence[int], *, block_size: int
) -> Iterator[np.ndarray]:
    """Yield a plain full Cartesian product of action indices in bounded blocks."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    iterator = product(*(range(count) for count in action_counts))
    while True:
        rows = tuple(islice(iterator, block_size))
        if not rows:
            break
        yield np.asarray(rows, dtype=np.int64)


def profiles_from_indices(
    action_sets: Sequence[Sequence[Action]], indices: np.ndarray
) -> tuple[tuple[Action, ...], ...]:
    if indices.ndim != 2 or indices.shape[1] != len(action_sets):
        raise ValueError("profile index matrix has the wrong shape")
    return tuple(
        tuple(
            action_sets[player][int(row[player])] for player in range(len(action_sets))
        )
        for row in indices
    )
