"""Sparse exact short-horizon Q3.2 solver for pure-strategy MPE.

This backend intentionally materialises small payoff tensors.  Explicit limits
prevent accidental full level-6 enumeration on a workstation.  Crossing a
limit raises an exception; no action, state, or weather branch is truncated.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Sequence

import numpy as np

from .action_enum import (
    ActionEnumerationLimitExceeded,
    enumerate_individual_actions,
    enumerate_initial_purchases_bounded,
)
from .canonical import actions_to_original, canonicalize_state, value_to_original
from .data import Q3Config, Weather
from .model import Action, JointState, StateValue, all_absorbed, initial_joint_state, terminal_state_value
from .stage_game import NoPureEquilibrium, pure_nash_indices, select_pure_equilibrium
from .transition import apply_initial_purchases, apply_joint_transition_batch


class ResourceLimitExceeded(RuntimeError):
    """The exact search exceeded an explicit safety budget without truncation."""


@dataclass(frozen=True)
class SolverLimits:
    max_actions_per_player: int = 4_096
    max_joint_profiles: int = 250_000
    max_cached_states: int = 100_000


@dataclass
class SolverStats:
    states_solved: int = 0
    cache_hits: int = 0
    stage_games: int = 0
    joint_profiles: int = 0
    invalid_profiles: int = 0
    pure_equilibria: int = 0
    max_actions_seen: int = 0
    max_joint_profiles_seen: int = 0


@dataclass(frozen=True)
class InitialSolveResult:
    value: StateValue
    purchases: tuple[Action, ...]
    post_purchase_state: JointState


class ExactQ3Solver:
    def __init__(
        self,
        cfg: Q3Config,
        *,
        limits: SolverLimits | None = None,
        equilibrium_atol: float = 1e-10,
    ) -> None:
        self.cfg = cfg
        self.limits = limits or SolverLimits()
        self.equilibrium_atol = equilibrium_atol
        self.stats = SolverStats()
        self._value_cache: dict[tuple[int, JointState], StateValue] = {}
        self._policy_cache: dict[tuple[int, JointState, Weather], tuple[Action, ...]] = {}

    def clear(self) -> None:
        self.stats = SolverStats()
        self._value_cache.clear()
        self._policy_cache.clear()

    def _check_action_budget(self, action_sets: Sequence[Sequence[Action]]) -> tuple[int, ...]:
        counts = tuple(len(actions) for actions in action_sets)
        self.stats.max_actions_seen = max(self.stats.max_actions_seen, max(counts))
        if any(count > self.limits.max_actions_per_player for count in counts):
            raise ResourceLimitExceeded(
                f"exact action count {counts} exceeds per-player limit "
                f"{self.limits.max_actions_per_player}"
            )
        joint_count = prod(counts)
        self.stats.max_joint_profiles_seen = max(
            self.stats.max_joint_profiles_seen, joint_count
        )
        if joint_count > self.limits.max_joint_profiles:
            raise ResourceLimitExceeded(
                f"exact joint profile count {joint_count:,} for actions {counts} exceeds "
                f"limit {self.limits.max_joint_profiles:,}; chunked exact backend required"
            )
        return counts

    def solve_state(self, day: int, state: JointState) -> StateValue:
        """Solve from an end-of-day ``day`` state in the caller's player order."""
        if not 0 <= day <= self.cfg.deadline:
            raise ValueError("day is outside the configured horizon")
        canonical, canonical_to_original = canonicalize_state(state)
        canonical_value = self._solve_canonical(day, canonical)
        return value_to_original(canonical_value, canonical_to_original)

    def _solve_canonical(self, day: int, state: JointState) -> StateValue:
        key = (day, state)
        cached = self._value_cache.get(key)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached
        if len(self._value_cache) >= self.limits.max_cached_states:
            raise ResourceLimitExceeded(
                f"exact state cache reached {self.limits.max_cached_states:,} entries"
            )

        if day == self.cfg.deadline or all_absorbed(state):
            result = terminal_state_value(self.cfg, state)
            self._value_cache[key] = result
            self.stats.states_solved += 1
            return result

        expected_value = np.zeros(self.cfg.n_players, dtype=np.float64)
        expected_success = np.zeros(self.cfg.n_players, dtype=np.float64)
        for weather in self.cfg.weather_order:
            probability = float(self.cfg.p_weather[weather])
            equilibrium = self._solve_stage(day, state, weather)
            expected_value += probability * np.asarray(equilibrium.value)
            expected_success += probability * np.asarray(equilibrium.success)
            self._policy_cache[(day, state, weather)] = equilibrium.actions

        result = StateValue(tuple(expected_value), tuple(expected_success))
        self._value_cache[key] = result
        self.stats.states_solved += 1
        return result

    def _solve_stage(self, day: int, state: JointState, weather: Weather):
        try:
            action_sets = tuple(
                enumerate_individual_actions(
                    self.cfg,
                    player,
                    weather,
                    max_actions=self.limits.max_actions_per_player,
                )
                for player in state
            )
        except ActionEnumerationLimitExceeded as exc:
            raise ResourceLimitExceeded(str(exc)) from exc
        action_shape = self._check_action_budget(action_sets)
        profiles = tuple(product(*action_sets))
        batch = apply_joint_transition_batch(self.cfg, state, profiles, weather)

        payoff = np.full(action_shape + (self.cfg.n_players,), -np.inf, dtype=np.float64)
        success = np.zeros(action_shape + (self.cfg.n_players,), dtype=np.float64)
        valid = batch.valid.reshape(action_shape)
        self.stats.stage_games += 1
        self.stats.joint_profiles += len(profiles)
        self.stats.invalid_profiles += int((~batch.valid).sum())

        for flat_index in np.flatnonzero(batch.valid):
            successor = batch.successors[int(flat_index)]
            if successor is None:
                raise AssertionError("valid vectorized transition has no successor")
            continuation = self.solve_state(day + 1, successor)
            index = np.unravel_index(int(flat_index), action_shape)
            payoff[index] = continuation.value
            success[index] = continuation.success

        indices = pure_nash_indices(
            payoff, valid, atol=self.equilibrium_atol
        )
        self.stats.pure_equilibria += len(indices)
        if len(indices) == 0:
            raise NoPureEquilibrium(
                f"no pure equilibrium at day={day}, weather={weather}, state={state}"
            )
        return select_pure_equilibrium(indices, payoff, success, action_sets)

    def policy_for(
        self, day: int, state: JointState, weather: Weather
    ) -> tuple[Action, ...]:
        canonical, canonical_to_original = canonicalize_state(state)
        if (day, canonical) not in self._value_cache:
            self.solve_state(day, state)
        actions = self._policy_cache[(day, canonical, weather)]
        return actions_to_original(actions, canonical_to_original)

    def solve_initial_purchases(self) -> InitialSolveResult:
        """Solve the exact day-0 purchase stage when it fits configured limits."""
        state = initial_joint_state(self.cfg)
        try:
            action_sets = tuple(
                enumerate_initial_purchases_bounded(
                    self.cfg,
                    player,
                    max_actions=self.limits.max_actions_per_player,
                )
                for player in state
            )
        except ActionEnumerationLimitExceeded as exc:
            raise ResourceLimitExceeded(str(exc)) from exc
        action_shape = self._check_action_budget(action_sets)
        payoff = np.full(action_shape + (self.cfg.n_players,), -np.inf, dtype=np.float64)
        success = np.zeros(action_shape + (self.cfg.n_players,), dtype=np.float64)
        valid = np.zeros(action_shape, dtype=bool)

        for index in np.ndindex(action_shape):
            actions = tuple(action_sets[i][index[i]] for i in range(self.cfg.n_players))
            post = apply_initial_purchases(self.cfg, state, actions)
            if post is None:
                continue
            continuation = self.solve_state(0, post)
            valid[index] = True
            payoff[index] = continuation.value
            success[index] = continuation.success

        indices = pure_nash_indices(payoff, valid, atol=self.equilibrium_atol)
        equilibrium = select_pure_equilibrium(indices, payoff, success, action_sets)
        post = apply_initial_purchases(self.cfg, state, equilibrium.actions)
        if post is None:
            raise AssertionError("selected initial equilibrium is infeasible")
        return InitialSolveResult(
            StateValue(equilibrium.value, equilibrium.success), equilibrium.actions, post
        )
