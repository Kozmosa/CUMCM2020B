"""Sparse exact Q3.2 solver with dense and checkpointable chunked backends."""

from __future__ import annotations

import os
import pickle
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from math import prod
from pathlib import Path
from typing import Self

import numpy as np

from .action_enum import (
    ActionEnumerationLimitExceeded,
    enumerate_individual_actions,
    enumerate_initial_purchases_bounded,
)
from .canonical import actions_to_original, canonicalize_state, value_to_original
from .data import Q3Config, Weather
from .model import (
    Action,
    JointState,
    PlayerState,
    StateValue,
    all_absorbed,
    initial_joint_state,
    terminal_state_value,
)
from .profile_enum import (
    StructuredEnumerationStats,
    iter_index_blocks,
    iter_structured_profile_blocks,
    profiles_from_indices,
)
from .pruning import (
    BestResponsePruningCertificate,
    RelaxedSinglePlayerUpperBound,
)
from .stage_game import (
    ChunkedSearchProgress,
    NoPureEquilibrium,
    ProfileBatchEvaluation,
    PureEquilibrium,
    chunked_pure_nash_search,
    pure_nash_indices,
    select_pure_equilibrium,
    select_pure_equilibrium_candidates,
)
from .transition import apply_initial_purchases, apply_joint_transition_batch

CHECKPOINT_VERSION = 1


class ResourceLimitExceeded(RuntimeError):
    """The exact search exceeded an explicit safety budget without truncation."""


class SearchCancelled(RuntimeError):
    """Cooperative cancellation requested by the CLI or embedding process."""


@dataclass(frozen=True)
class SolverLimits:
    max_actions_per_player: int = 1_000_000
    max_joint_profiles: int = 250_000
    max_cached_states: int = 100_000
    profile_chunk_size: int = 4_096
    max_stage_profiles_evaluated: int = 5_000_000

    def __post_init__(self) -> None:
        if (
            min(
                self.max_actions_per_player,
                self.max_joint_profiles,
                self.max_cached_states,
                self.profile_chunk_size,
                self.max_stage_profiles_evaluated,
            )
            <= 0
        ):
            raise ValueError("all solver limits must be positive")


@dataclass
class SolverStats:
    states_solved: int = 0
    cache_hits: int = 0
    cache_waits: int = 0
    stage_games: int = 0
    action_enumerations: int = 0
    action_set_reuses: int = 0
    dense_stage_games: int = 0
    chunked_stage_games: int = 0
    raw_joint_profiles: int = 0
    joint_profiles: int = 0
    structured_profiles_pruned: int = 0
    upper_bound_profiles: int = 0
    best_response_profiles_pruned: int = 0
    pruning_certificates_recorded: int = 0
    invalid_profiles: int = 0
    transition_batches: int = 0
    unique_successors: int = 0
    duplicate_successors: int = 0
    pure_equilibria: int = 0
    opponent_pairs_completed: int = 0
    max_actions_seen: int = 0
    max_joint_profiles_seen: int = 0
    checkpoint_writes: int = 0
    checkpoint_loads: int = 0


@dataclass(frozen=True)
class InitialSolveResult:
    value: StateValue
    purchases: tuple[Action, ...]
    post_purchase_state: JointState


@dataclass
class _InFlight:
    owner_ident: int
    event: threading.Event
    error: BaseException | None = None


class ExactQ3Solver:
    def __init__(
        self,
        cfg: Q3Config,
        *,
        limits: SolverLimits | None = None,
        equilibrium_atol: float = 1e-10,
        workers: int = 1,
        checkpoint_path: str | os.PathLike[str] | None = None,
        checkpoint_every_states: int = 0,
        checkpoint_every_pairs: int = 0,
        enable_bound_pruning: bool = True,
        bound_pruning_slack: float = 1e-6,
        record_pruning_certificates: bool = False,
        max_pruning_certificates: int = 10_000,
    ) -> None:
        if workers <= 0:
            raise ValueError("workers must be positive")
        if checkpoint_every_states < 0 or checkpoint_every_pairs < 0:
            raise ValueError("checkpoint intervals cannot be negative")
        if max_pruning_certificates < 0 or bound_pruning_slack < 0:
            raise ValueError(
                "max_pruning_certificates and bound_pruning_slack cannot be negative"
            )
        self.cfg = cfg
        self.limits = limits or SolverLimits()
        self.equilibrium_atol = equilibrium_atol
        self.workers = workers
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.checkpoint_every_states = checkpoint_every_states
        self.checkpoint_every_pairs = checkpoint_every_pairs
        self.enable_bound_pruning = enable_bound_pruning
        self.bound_pruning_slack = bound_pruning_slack
        self.record_pruning_certificates = record_pruning_certificates
        self.max_pruning_certificates = max_pruning_certificates
        self.pruning_certificates: list[BestResponsePruningCertificate] = []
        self._upper_bound = RelaxedSinglePlayerUpperBound.build(cfg)
        self.stats = SolverStats()
        self._value_cache: dict[tuple[int, JointState], StateValue] = {}
        self._policy_cache: dict[
            tuple[int, JointState, Weather], tuple[Action, ...]
        ] = {}
        self._stage_progress: dict[object, ChunkedSearchProgress] = {}
        self._inflight: dict[tuple[int, JointState], _InFlight] = {}
        self._cache_lock = threading.RLock()
        self._checkpoint_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._owner_ident = threading.get_ident()
        self._last_checkpoint_states = 0
        self._last_checkpoint_pairs: dict[object, int] = {}
        self._successor_executor = (
            ThreadPoolExecutor(max_workers=workers, thread_name_prefix="q3-state")
            if workers > 1
            else None
        )

    def close(self) -> None:
        if self._successor_executor is not None:
            self._successor_executor.shutdown(wait=True, cancel_futures=True)
            self._successor_executor = None

    def request_stop(self) -> None:
        self._cancel_event.set()

    def _check_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise SearchCancelled("Q3 search cancellation requested")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def clear(self) -> None:
        with self._cache_lock:
            if self._inflight:
                raise RuntimeError(
                    "cannot clear the solver while states are being evaluated"
                )
            self.stats = SolverStats()
            self._value_cache.clear()
            self._policy_cache.clear()
            self._stage_progress.clear()
            self.pruning_certificates.clear()
            self._cancel_event.clear()
            self._last_checkpoint_states = 0
            self._last_checkpoint_pairs.clear()

    def _update_stats(self, **increments: int) -> None:
        with self._cache_lock:
            for field, amount in increments.items():
                setattr(self.stats, field, getattr(self.stats, field) + amount)

    def _update_max_stats(self, *, actions: int = 0, profiles: int = 0) -> None:
        with self._cache_lock:
            self.stats.max_actions_seen = max(self.stats.max_actions_seen, actions)
            self.stats.max_joint_profiles_seen = max(
                self.stats.max_joint_profiles_seen, profiles
            )

    def _check_action_budget(
        self, action_sets: Sequence[Sequence[Action]]
    ) -> tuple[int, ...]:
        counts = tuple(len(actions) for actions in action_sets)
        self._update_max_stats(actions=max(counts, default=0), profiles=prod(counts))
        if any(count > self.limits.max_actions_per_player for count in counts):
            raise ResourceLimitExceeded(
                f"exact action count {counts} exceeds per-player limit "
                f"{self.limits.max_actions_per_player}"
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
        self._check_cancelled()
        key = (day, state)
        ident = threading.get_ident()
        while True:
            with self._cache_lock:
                cached = self._value_cache.get(key)
                if cached is not None:
                    self.stats.cache_hits += 1
                    return cached
                marker = self._inflight.get(key)
                if marker is None:
                    if (
                        len(self._value_cache) + len(self._inflight)
                        >= self.limits.max_cached_states
                    ):
                        raise ResourceLimitExceeded(
                            f"exact state cache reached {self.limits.max_cached_states:,} entries"
                        )
                    marker = _InFlight(ident, threading.Event())
                    self._inflight[key] = marker
                    break
                if marker.owner_ident == ident:
                    raise RuntimeError(
                        "recursive state dependency unexpectedly formed a cycle"
                    )
                self.stats.cache_waits += 1
            marker.event.wait()
            if marker.error is not None:
                raise marker.error

        try:
            result = self._compute_canonical(day, state)
        except BaseException as exc:
            with self._cache_lock:
                marker.error = exc
                self._inflight.pop(key, None)
                marker.event.set()
            raise

        with self._cache_lock:
            self._value_cache[key] = result
            self.stats.states_solved += 1
            self._inflight.pop(key, None)
            marker.event.set()
            states_solved = self.stats.states_solved
        self._maybe_checkpoint_states(states_solved)
        return result

    def _compute_canonical(self, day: int, state: JointState) -> StateValue:
        if day == self.cfg.deadline or all_absorbed(state):
            return terminal_state_value(self.cfg, state)

        expected_value = np.zeros(self.cfg.n_players, dtype=np.float64)
        expected_success = np.zeros(self.cfg.n_players, dtype=np.float64)
        for weather in self.cfg.weather_order:
            probability = float(self.cfg.p_weather[weather])
            equilibrium = self._solve_stage(day, state, weather)
            expected_value += probability * np.asarray(equilibrium.value)
            expected_success += probability * np.asarray(equilibrium.success)
            with self._cache_lock:
                self._policy_cache[(day, state, weather)] = equilibrium.actions
        return StateValue(tuple(expected_value), tuple(expected_success))

    def _solve_unique_successors(
        self, day: int, states: Sequence[JointState]
    ) -> tuple[StateValue, ...]:
        if (
            self._successor_executor is not None
            and threading.get_ident() == self._owner_ident
            and len(states) > 1
        ):
            values = self._successor_executor.map(
                lambda successor: self._solve_canonical(day, successor), states
            )
            return tuple(values)
        return tuple(self._solve_canonical(day, successor) for successor in states)

    def _evaluate_successors(
        self,
        continuation_day: int,
        valid: np.ndarray,
        successors: Sequence[JointState | None],
    ) -> ProfileBatchEvaluation:
        batch_size = len(successors)
        payoff = np.full((batch_size, self.cfg.n_players), -np.inf, dtype=np.float64)
        success = np.zeros((batch_size, self.cfg.n_players), dtype=np.float64)
        canonical_rows: list[tuple[int, JointState, tuple[int, ...]]] = []
        unique: dict[JointState, None] = {}
        for row in np.flatnonzero(valid):
            successor = successors[int(row)]
            if successor is None:
                raise AssertionError("valid transition has no successor")
            canonical, mapping = canonicalize_state(successor)
            canonical_rows.append((int(row), canonical, mapping))
            unique.setdefault(canonical, None)

        unique_states = tuple(unique)
        unique_values = self._solve_unique_successors(continuation_day, unique_states)
        value_by_state = dict(zip(unique_states, unique_values, strict=True))
        for row, canonical, mapping in canonical_rows:
            value = value_to_original(value_by_state[canonical], mapping)
            payoff[row] = value.value
            success[row] = value.success

        unique_count = len(unique_states)
        valid_count = len(canonical_rows)
        self._update_stats(
            unique_successors=unique_count,
            duplicate_successors=valid_count - unique_count,
        )
        return ProfileBatchEvaluation(valid.copy(), payoff, success)

    def _evaluate_day_profiles(
        self,
        day: int,
        state: JointState,
        weather: Weather,
        profiles: Sequence[Sequence[Action]],
    ) -> ProfileBatchEvaluation:
        batch = apply_joint_transition_batch(self.cfg, state, profiles, weather)
        self._update_stats(
            joint_profiles=len(profiles),
            invalid_profiles=int((~batch.valid).sum()),
            transition_batches=1,
        )
        return self._evaluate_successors(day + 1, batch.valid, batch.successors)

    def _upper_bounds_from_successors(
        self,
        continuation_day: int,
        valid: np.ndarray,
        successors: Sequence[JointState | None],
        player: int,
    ) -> np.ndarray:
        bounds = np.full(len(successors), -np.inf, dtype=np.float64)
        for row in np.flatnonzero(valid):
            successor = successors[int(row)]
            if successor is None:
                raise AssertionError("valid upper-bound transition has no successor")
            bounds[int(row)] = self._upper_bound.value(
                continuation_day, successor[player]
            )
        self._update_stats(upper_bound_profiles=len(successors))
        return bounds

    def _day_profile_upper_bounds(
        self,
        day: int,
        state: JointState,
        weather: Weather,
        action_sets: Sequence[Sequence[Action]],
        indices: np.ndarray,
        player: int,
    ) -> np.ndarray:
        self._check_cancelled()
        profiles = profiles_from_indices(action_sets, indices)
        batch = apply_joint_transition_batch(self.cfg, state, profiles, weather)
        return self._upper_bounds_from_successors(
            day + 1, batch.valid, batch.successors, player
        )

    def _pruning_callback(self, stage: str, day: int, weather: str):
        def record(
            player: int,
            indices: np.ndarray,
            upper_bounds: np.ndarray,
            exact_lower_bound: float,
        ) -> None:
            count = len(indices)
            self._update_stats(best_response_profiles_pruned=count)
            if not self.record_pruning_certificates or count == 0:
                return
            with self._cache_lock:
                remaining = self.max_pruning_certificates - len(
                    self.pruning_certificates
                )
                if remaining <= 0:
                    return
                for row, upper_bound in zip(
                    indices[:remaining], upper_bounds[:remaining], strict=True
                ):
                    certificate = BestResponsePruningCertificate(
                        stage=stage,
                        day=day,
                        weather=weather,
                        player=player,
                        profile_index=tuple(int(x) for x in row),
                        upper_bound=float(upper_bound),
                        exact_lower_bound=float(exact_lower_bound),
                        safety_margin=float(
                            exact_lower_bound
                            - self.equilibrium_atol
                            - self.bound_pruning_slack
                            - float(upper_bound)
                        ),
                    )
                    self.pruning_certificates.append(certificate)
                self.stats.pruning_certificates_recorded = len(
                    self.pruning_certificates
                )

        return record

    def _stage_evaluator(
        self,
        day: int,
        state: JointState,
        weather: Weather,
        action_sets: Sequence[Sequence[Action]],
    ):
        evaluated = 0
        budget_lock = threading.Lock()

        def evaluate(indices: np.ndarray) -> ProfileBatchEvaluation:
            nonlocal evaluated
            self._check_cancelled()
            with budget_lock:
                requested = len(indices)
                if evaluated + requested > self.limits.max_stage_profiles_evaluated:
                    raise ResourceLimitExceeded(
                        f"stage evaluation reached {evaluated:,} profiles and the next "
                        f"block would exceed limit "
                        f"{self.limits.max_stage_profiles_evaluated:,}"
                    )
                evaluated += requested
            profiles = profiles_from_indices(action_sets, indices)
            return self._evaluate_day_profiles(day, state, weather, profiles)

        return evaluate

    def _solve_stage_dense(
        self,
        day: int,
        state: JointState,
        weather: Weather,
        action_sets: Sequence[Sequence[Action]],
        action_shape: tuple[int, ...],
    ) -> PureEquilibrium:
        payoff = np.full(
            action_shape + (self.cfg.n_players,), -np.inf, dtype=np.float64
        )
        success = np.zeros(action_shape + (self.cfg.n_players,), dtype=np.float64)
        valid = np.zeros(action_shape, dtype=bool)
        enumeration_stats = StructuredEnumerationStats()
        evaluated = 0
        for block in iter_structured_profile_blocks(
            self.cfg,
            state,
            action_sets,
            weather,
            block_size=self.limits.profile_chunk_size,
            stats=enumeration_stats,
        ):
            if (
                evaluated + len(block.profiles)
                > self.limits.max_stage_profiles_evaluated
            ):
                raise ResourceLimitExceeded(
                    f"dense stage would exceed exact evaluation limit "
                    f"{self.limits.max_stage_profiles_evaluated:,}"
                )
            evaluated += len(block.profiles)
            evaluation = self._evaluate_day_profiles(
                day, state, weather, block.profiles
            )
            coordinates = tuple(
                block.indices[:, player] for player in range(self.cfg.n_players)
            )
            valid[coordinates] = evaluation.valid
            payoff[coordinates] = evaluation.payoff
            success[coordinates] = evaluation.success

        self._update_stats(structured_profiles_pruned=enumeration_stats.profiles_pruned)
        indices = pure_nash_indices(payoff, valid, atol=self.equilibrium_atol)
        if len(indices) == 0:
            raise NoPureEquilibrium(
                f"no pure equilibrium at day={day}, weather={weather}, state={state}"
            )
        return select_pure_equilibrium(indices, payoff, success, action_sets)

    def _stage_progress_callback(self, key: object):
        def update(progress: ChunkedSearchProgress) -> None:
            with self._cache_lock:
                previous = self._stage_progress.get(key)
                previous_pair = (
                    previous.next_opponent_pair if previous is not None else 0
                )
                self._stage_progress[key] = progress
                self.stats.opponent_pairs_completed += (
                    progress.next_opponent_pair - previous_pair
                )
            self._maybe_checkpoint_pairs(key, progress.next_opponent_pair)

        return update

    def _solve_stage_chunked(
        self,
        day: int,
        state: JointState,
        weather: Weather,
        action_sets: Sequence[Sequence[Action]],
    ) -> PureEquilibrium:
        key = ("day", day, state, weather)
        with self._cache_lock:
            progress = self._stage_progress.get(key)
        workers = self.workers if threading.get_ident() == self._owner_ident else 1
        equilibria = chunked_pure_nash_search(
            action_sets,
            self._stage_evaluator(day, state, weather, action_sets),
            chunk_size=self.limits.profile_chunk_size,
            atol=self.equilibrium_atol,
            bound_slack=self.bound_pruning_slack,
            workers=workers,
            upper_bounder=(
                (
                    lambda indices, player: self._day_profile_upper_bounds(
                        day,
                        state,
                        weather,
                        action_sets,
                        indices,
                        player,
                    )
                )
                if self.enable_bound_pruning
                else None
            ),
            pruning_callback=(
                self._pruning_callback("day", day, weather)
                if self.enable_bound_pruning
                else None
            ),
            progress=progress,
            progress_callback=self._stage_progress_callback(key),
        )
        with self._cache_lock:
            self._stage_progress.pop(key, None)
            self._last_checkpoint_pairs.pop(key, None)
        if not equilibria:
            raise NoPureEquilibrium(
                f"no pure equilibrium at day={day}, weather={weather}, state={state}"
            )
        return select_pure_equilibrium_candidates(equilibria)

    def _solve_stage(
        self, day: int, state: JointState, weather: Weather
    ) -> PureEquilibrium:
        try:
            actions_by_state: dict[PlayerState, tuple[Action, ...]] = {}
            action_sets_list: list[tuple[Action, ...]] = []
            enumerations = 0
            reuses = 0
            for player in state:
                actions = actions_by_state.get(player)
                if actions is None:
                    actions = enumerate_individual_actions(
                        self.cfg,
                        player,
                        weather,
                        max_actions=self.limits.max_actions_per_player,
                    )
                    actions_by_state[player] = actions
                    enumerations += 1
                else:
                    reuses += 1
                action_sets_list.append(actions)
            action_sets = tuple(action_sets_list)
            self._update_stats(
                action_enumerations=enumerations,
                action_set_reuses=reuses,
            )
        except ActionEnumerationLimitExceeded as exc:
            raise ResourceLimitExceeded(str(exc)) from exc
        action_shape = self._check_action_budget(action_sets)
        raw_profiles = prod(action_shape)
        self._update_stats(stage_games=1, raw_joint_profiles=raw_profiles)
        if raw_profiles <= self.limits.max_joint_profiles:
            self._update_stats(dense_stage_games=1)
            equilibrium = self._solve_stage_dense(
                day, state, weather, action_sets, action_shape
            )
        else:
            self._update_stats(chunked_stage_games=1)
            equilibrium = self._solve_stage_chunked(day, state, weather, action_sets)
        self._update_stats(pure_equilibria=1)
        return equilibrium

    def policy_for(
        self, day: int, state: JointState, weather: Weather
    ) -> tuple[Action, ...]:
        canonical, canonical_to_original = canonicalize_state(state)
        if (day, canonical) not in self._value_cache:
            self.solve_state(day, state)
        with self._cache_lock:
            actions = self._policy_cache[(day, canonical, weather)]
        return actions_to_original(actions, canonical_to_original)

    def _evaluate_initial_indices(
        self,
        action_sets: Sequence[Sequence[Action]],
        indices: np.ndarray,
    ) -> ProfileBatchEvaluation:
        state = initial_joint_state(self.cfg)
        profiles = profiles_from_indices(action_sets, indices)
        successors: list[JointState | None] = []
        valid = np.zeros(len(profiles), dtype=bool)
        for row, actions in enumerate(profiles):
            post = apply_initial_purchases(self.cfg, state, actions)
            successors.append(post)
            valid[row] = post is not None
        self._update_stats(
            joint_profiles=len(profiles),
            invalid_profiles=int((~valid).sum()),
            transition_batches=1,
        )
        return self._evaluate_successors(0, valid, successors)

    def _initial_profile_upper_bounds(
        self,
        action_sets: Sequence[Sequence[Action]],
        indices: np.ndarray,
        player: int,
    ) -> np.ndarray:
        self._check_cancelled()
        state = initial_joint_state(self.cfg)
        profiles = profiles_from_indices(action_sets, indices)
        successors: list[JointState | None] = []
        valid = np.zeros(len(profiles), dtype=bool)
        for row, actions in enumerate(profiles):
            post = apply_initial_purchases(self.cfg, state, actions)
            successors.append(post)
            valid[row] = post is not None
        return self._upper_bounds_from_successors(0, valid, successors, player)

    def _initial_evaluator(self, action_sets: Sequence[Sequence[Action]]):
        evaluated = 0
        budget_lock = threading.Lock()

        def evaluate(indices: np.ndarray) -> ProfileBatchEvaluation:
            nonlocal evaluated
            self._check_cancelled()
            with budget_lock:
                requested = len(indices)
                if evaluated + requested > self.limits.max_stage_profiles_evaluated:
                    raise ResourceLimitExceeded(
                        f"initial stage reached {evaluated:,} evaluated profiles and the "
                        f"next block would exceed limit "
                        f"{self.limits.max_stage_profiles_evaluated:,}"
                    )
                evaluated += requested
            return self._evaluate_initial_indices(action_sets, indices)

        return evaluate

    def solve_initial_purchases(self) -> InitialSolveResult:
        """Solve the exact day-0 purchase game with dense or chunked search."""
        state = initial_joint_state(self.cfg)
        try:
            shared_actions = enumerate_initial_purchases_bounded(
                self.cfg,
                state[0],
                max_actions=self.limits.max_actions_per_player,
            )
            action_sets = tuple(shared_actions for _ in state)
            self._update_stats(
                action_enumerations=1,
                action_set_reuses=max(0, len(state) - 1),
            )
        except ActionEnumerationLimitExceeded as exc:
            raise ResourceLimitExceeded(str(exc)) from exc
        action_shape = self._check_action_budget(action_sets)
        raw_profiles = prod(action_shape)
        self._update_stats(stage_games=1, raw_joint_profiles=raw_profiles)
        evaluator = self._initial_evaluator(action_sets)

        if raw_profiles <= self.limits.max_joint_profiles:
            self._update_stats(dense_stage_games=1)
            payoff = np.full(
                action_shape + (self.cfg.n_players,), -np.inf, dtype=np.float64
            )
            success = np.zeros(action_shape + (self.cfg.n_players,), dtype=np.float64)
            valid = np.zeros(action_shape, dtype=bool)
            for indices in iter_index_blocks(
                action_shape, block_size=self.limits.profile_chunk_size
            ):
                evaluation = evaluator(indices)
                coordinates = tuple(
                    indices[:, player] for player in range(self.cfg.n_players)
                )
                valid[coordinates] = evaluation.valid
                payoff[coordinates] = evaluation.payoff
                success[coordinates] = evaluation.success
            indices = pure_nash_indices(payoff, valid, atol=self.equilibrium_atol)
            equilibrium = select_pure_equilibrium(indices, payoff, success, action_sets)
        else:
            self._update_stats(chunked_stage_games=1)
            key = ("initial",)
            with self._cache_lock:
                progress = self._stage_progress.get(key)
            workers = self.workers if threading.get_ident() == self._owner_ident else 1
            equilibria = chunked_pure_nash_search(
                action_sets,
                evaluator,
                chunk_size=self.limits.profile_chunk_size,
                atol=self.equilibrium_atol,
                bound_slack=self.bound_pruning_slack,
                workers=workers,
                upper_bounder=(
                    (
                        lambda indices, player: self._initial_profile_upper_bounds(
                            action_sets, indices, player
                        )
                    )
                    if self.enable_bound_pruning
                    else None
                ),
                pruning_callback=(
                    self._pruning_callback("initial", 0, "")
                    if self.enable_bound_pruning
                    else None
                ),
                progress=progress,
                progress_callback=self._stage_progress_callback(key),
            )
            with self._cache_lock:
                self._stage_progress.pop(key, None)
                self._last_checkpoint_pairs.pop(key, None)
            equilibrium = select_pure_equilibrium_candidates(equilibria)

        self._update_stats(pure_equilibria=1)
        post = apply_initial_purchases(self.cfg, state, equilibrium.actions)
        if post is None:
            raise AssertionError("selected initial equilibrium is infeasible")
        return InitialSolveResult(
            StateValue(equilibrium.value, equilibrium.success),
            equilibrium.actions,
            post,
        )

    def _checkpoint_payload(self) -> dict[str, object]:
        with self._cache_lock:
            return {
                "version": CHECKPOINT_VERSION,
                "config": self.cfg,
                "value_cache": dict(self._value_cache),
                "policy_cache": dict(self._policy_cache),
                "stage_progress": dict(self._stage_progress),
                "pruning_certificates": tuple(self.pruning_certificates),
                "stats": self.stats,
            }

    def save_checkpoint(self, path: str | os.PathLike[str] | None = None) -> Path:
        target = Path(path) if path is not None else self.checkpoint_path
        if target is None:
            raise ValueError("checkpoint path is not configured")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(
            f".{target.name}.tmp-{os.getpid()}-{threading.get_ident()}"
        )
        with self._checkpoint_lock:
            self._update_stats(checkpoint_writes=1)
            payload = self._checkpoint_payload()
            try:
                with temporary.open("wb") as handle:
                    pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink()
        return target

    def load_checkpoint(self, path: str | os.PathLike[str] | None = None) -> None:
        source = Path(path) if path is not None else self.checkpoint_path
        if source is None:
            raise ValueError("checkpoint path is not configured")
        with source.open("rb") as handle:
            payload = pickle.load(handle)
        if (
            not isinstance(payload, dict)
            or payload.get("version") != CHECKPOINT_VERSION
        ):
            raise ValueError("unsupported Q3 checkpoint format")
        if payload.get("config") != self.cfg:
            raise ValueError("checkpoint configuration does not match this solver")
        with self._cache_lock:
            if self._inflight:
                raise RuntimeError(
                    "cannot load a checkpoint while states are being evaluated"
                )
            self._value_cache = dict(payload["value_cache"])
            self._policy_cache = dict(payload["policy_cache"])
            self._stage_progress = dict(payload["stage_progress"])
            self.pruning_certificates = list(payload.get("pruning_certificates", ()))
            loaded_stats = payload["stats"]
            normalized_stats = SolverStats()
            for field in fields(SolverStats):
                if hasattr(loaded_stats, field.name):
                    setattr(
                        normalized_stats,
                        field.name,
                        getattr(loaded_stats, field.name),
                    )
            self.stats = normalized_stats
            self.stats.pruning_certificates_recorded = len(self.pruning_certificates)
            self.stats.checkpoint_loads += 1
            self._last_checkpoint_states = self.stats.states_solved
            self._last_checkpoint_pairs = {
                key: progress.next_opponent_pair
                for key, progress in self._stage_progress.items()
            }

    def _maybe_checkpoint_states(self, states_solved: int) -> None:
        if (
            self.checkpoint_path is None
            or self.checkpoint_every_states <= 0
            or states_solved - self._last_checkpoint_states
            < self.checkpoint_every_states
        ):
            return
        with self._cache_lock:
            if (
                states_solved - self._last_checkpoint_states
                < self.checkpoint_every_states
            ):
                return
            self._last_checkpoint_states = states_solved
        self.save_checkpoint()

    def _maybe_checkpoint_pairs(self, key: object, completed_pairs: int) -> None:
        if self.checkpoint_path is None or self.checkpoint_every_pairs <= 0:
            return
        with self._cache_lock:
            previous = self._last_checkpoint_pairs.get(key, 0)
            if completed_pairs - previous < self.checkpoint_every_pairs:
                return
            self._last_checkpoint_pairs[key] = completed_pairs
        self.save_checkpoint()
