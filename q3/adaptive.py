"""Adaptive restricted-game backend and public Q3.2 solve interface."""

from __future__ import annotations

import os
import pickle
from dataclasses import asdict, dataclass
from itertools import permutations, product
from math import prod
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .action_enum import (
    ActionEnumerationLimitExceeded,
    IndividualActionArrays,
    enumerate_individual_actions,
    enumerate_individual_action_arrays,
    enumerate_initial_purchases_bounded,
)
from .canonical import actions_to_original, canonicalize_state
from .data import Q3Config, Weather
from .model import (
    Action,
    ActionKind,
    JointState,
    PlayerState,
    StateValue,
    Status,
    all_absorbed,
    initial_joint_state,
    terminal_state_value,
)
from .profile_enum import action_skeleton, iter_index_blocks, profiles_from_indices
from .purchase_oracle import (
    PurchaseLatticeOracle,
    PurchaseOracleOptions,
)
from .reports import PolicyEntry, Q32SolveResult
from .runtime import BudgetManager
from .stage_game import (
    MixedEquilibrium,
    NoPureEquilibrium,
    ProfileBatchEvaluation,
    PureEquilibrium,
    minimize_nashconv,
    pure_nash_indices,
    select_pure_equilibrium,
    select_pure_equilibrium_candidates,
)
from .stochastic_dp import (
    ExactQ3Solver,
    InitialSolveResult,
    ResourceLimitExceeded,
    SearchCancelled,
    SolverLimits,
)
from .transition import (
    BatchTransitionArrays,
    apply_initial_purchases,
    apply_unilateral_transition_arrays,
    apply_unilateral_transition_encoded_arrays,
)

if TYPE_CHECKING:
    from .heuristic import HeuristicOptions


@dataclass(frozen=True)
class AdaptiveOptions:
    initial_candidates: int = 24
    village_candidates_per_skeleton: int = 12
    max_initial_candidates: int = 128
    max_village_candidates_per_skeleton: int = 64
    additions_per_player: int = 8
    max_restricted_rounds: int = 32
    equilibrium: str = "pure-mixed"
    seed: int = 20260728
    warm_initial_actions: tuple[Action, ...] = ()
    purchase_oracle_backend: str = "auto"
    purchase_oracle_threads: int | None = None
    purchase_cuda_device: int = 0
    purchase_cuda_min_actions: int = 131_072
    purchase_parallel_min_actions: int = 32_768

    def __post_init__(self) -> None:
        if min(
            self.initial_candidates,
            self.village_candidates_per_skeleton,
            self.max_initial_candidates,
            self.max_village_candidates_per_skeleton,
            self.additions_per_player,
            self.max_restricted_rounds,
        ) <= 0:
            raise ValueError("adaptive candidate and round limits must be positive")
        if self.equilibrium not in {"pure", "pure-mixed"}:
            raise ValueError("equilibrium must be 'pure' or 'pure-mixed'")
        PurchaseOracleOptions(
            backend=self.purchase_oracle_backend,
            threads=self.purchase_oracle_threads,
            cuda_device=self.purchase_cuda_device,
            cuda_min_actions=self.purchase_cuda_min_actions,
            parallel_min_actions=self.purchase_parallel_min_actions,
        )


@dataclass
class AdaptiveStats:
    restricted_rounds: int = 0
    restricted_profiles: int = 0
    full_deviation_actions: int = 0
    deviation_actions_pruned: int = 0
    deviation_actions_evaluated: int = 0
    candidates_added: int = 0
    certified_stages: int = 0
    approximate_mixed_stages: int = 0
    max_stage_regret_upper: float = 0.0
    purchase_oracle_calls: int = 0
    purchase_oracle_actions: int = 0
    purchase_oracle_cpu_calls: int = 0
    purchase_oracle_cuda_calls: int = 0
    purchase_oracle_seconds: float = 0.0
    purchase_regions_pruned: int = 0
    purchase_oracle_cache_entries: int = 0
    purchase_oracle_cache_bytes: int = 0


@dataclass(frozen=True)
class AdaptiveStageOutcome:
    value: tuple[float, ...]
    success: tuple[float, ...]
    policy: tuple[PolicyEntry, ...]
    actions: tuple[Action, ...] | None
    status: str
    regret_lower: tuple[float, ...]
    regret_upper: tuple[float, ...]


@dataclass(frozen=True)
class AdaptiveInitialResult:
    value: StateValue
    policy: tuple[PolicyEntry, ...]
    purchases: tuple[Action, ...] | None
    post_purchase_state: JointState | None
    status: str
    regret_lower: tuple[float, ...]
    regret_upper: tuple[float, ...]


def _action_targets(limit: int) -> tuple[tuple[float, float], ...]:
    if limit <= 0:
        return ()
    load_levels = (0.0, 0.25, 0.5, 0.75, 1.0)
    ratios = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)
    targets = tuple(product(load_levels, ratios))
    if limit >= len(targets):
        return targets
    positions = np.linspace(0, len(targets) - 1, limit, dtype=int)
    return tuple(targets[int(index)] for index in positions)


def deterministic_action_candidates(
    cfg: Q3Config,
    actions: Sequence[Action],
    *,
    per_skeleton: int,
    warm_actions: Sequence[Action] = (),
) -> tuple[Action, ...]:
    """Select deterministic water/food strata while retaining every skeleton."""
    if len(actions) <= per_skeleton:
        return tuple(actions)

    def skeleton_key(action: Action) -> tuple[int, int, bool]:
        return (int(action.kind), action.destination, action.is_buyer)

    grouped: dict[tuple[int, int, bool], list[Action]] = {}
    for action in actions:
        grouped.setdefault(skeleton_key(action), []).append(action)
    action_set = set(actions)
    warm_by_skeleton: dict[tuple[int, int, bool], set[Action]] = {}
    for action in warm_actions:
        if action in action_set:
            warm_by_skeleton.setdefault(skeleton_key(action), set()).add(action)
    selected: set[Action] = set()
    for skeleton, group in grouped.items():
        group.sort(key=lambda action: action.code)
        group_selected = set(warm_by_skeleton.get(skeleton, ()))
        if len(group) <= per_skeleton:
            group_selected.update(group)
            selected.update(group_selected)
            continue
        group_selected.add(group[0])
        group_selected.add(group[-1])
        water_load = np.fromiter(
            (cfg.water_weight * action.buy_water for action in group),
            dtype=np.float64,
            count=len(group),
        )
        food_load = np.fromiter(
            (cfg.food_weight * action.buy_food for action in group),
            dtype=np.float64,
            count=len(group),
        )
        load = water_load + food_load
        max_load = max(1.0, float(np.max(load)))
        ratio = np.full(len(group), 0.5, dtype=np.float64)
        np.divide(water_load, load, out=ratio, where=load != 0.0)
        normalized_load = load / max_load
        targets = _action_targets(max(0, per_skeleton - 2))
        for load_target, ratio_target in targets:
            metric = np.abs(normalized_load - load_target) + np.abs(
                ratio - ratio_target
            )
            group_selected.add(group[int(np.argmin(metric))])
        if len(group_selected) < per_skeleton:
            for index in np.linspace(0, len(group) - 1, per_skeleton, dtype=int):
                group_selected.add(group[int(index)])
                if len(group_selected) >= per_skeleton:
                    break
        selected.update(group_selected)
    return tuple(sorted(selected, key=lambda action: action.code))


def deterministic_array_action_candidates(
    cfg: Q3Config,
    actions: IndividualActionArrays,
    *,
    per_skeleton: int,
) -> tuple[Action, ...]:
    """Array-backed equivalent of deterministic_action_candidates."""
    if per_skeleton <= 0:
        raise ValueError("per_skeleton must be positive")
    if len(actions) <= per_skeleton:
        return actions.action_tuple()
    selected: set[int] = set()
    targets = _action_targets(max(0, per_skeleton - 2))
    for start, stop in actions.skeleton_ranges:
        count = stop - start
        if count <= per_skeleton:
            selected.update(range(start, stop))
            continue
        selected.add(start)
        selected.add(stop - 1)
        water_load = (
            cfg.water_weight * actions.buy_water[start:stop].astype(np.float64)
        )
        food_load = (
            cfg.food_weight * actions.buy_food[start:stop].astype(np.float64)
        )
        load = water_load + food_load
        max_load = max(1.0, float(np.max(load)))
        ratio = np.full(count, 0.5, dtype=np.float64)
        np.divide(water_load, load, out=ratio, where=load != 0.0)
        normalized_load = load / max_load
        for load_target, ratio_target in targets:
            metric = np.abs(normalized_load - load_target) + np.abs(
                ratio - ratio_target
            )
            selected.add(start + int(np.argmin(metric)))
        if sum(start <= index < stop for index in selected) < per_skeleton:
            for index in np.linspace(start, stop - 1, per_skeleton, dtype=int):
                selected.add(int(index))
                if sum(start <= item < stop for item in selected) >= per_skeleton:
                    break
    return tuple(actions.action_at(index) for index in sorted(selected))


class AdaptiveQ3Solver(ExactQ3Solver):
    def __init__(
        self,
        cfg: Q3Config,
        *,
        limits: SolverLimits | None = None,
        options: AdaptiveOptions | None = None,
        quality_target: float = 10.0,
        budget_manager: BudgetManager | None = None,
        **kwargs,
    ) -> None:
        if quality_target < 0:
            raise ValueError("quality_target cannot be negative")
        super().__init__(
            cfg,
            limits=limits,
            budget_manager=budget_manager,
            **kwargs,
        )
        self.options = options or AdaptiveOptions()
        self.quality_target = quality_target
        self.adaptive_stats = AdaptiveStats()
        self._purchase_oracle = PurchaseLatticeOracle(
            cfg,
            self._upper_bound,
            PurchaseOracleOptions(
                backend=self.options.purchase_oracle_backend,
                threads=self.options.purchase_oracle_threads,
                cuda_device=self.options.purchase_cuda_device,
                cuda_min_actions=self.options.purchase_cuda_min_actions,
                parallel_min_actions=self.options.purchase_parallel_min_actions,
            ),
        )
        self._policy_entry_cache: dict[
            tuple[bytes, Weather], tuple[PolicyEntry, ...]
        ] = {}

    def clear(self) -> None:
        super().clear()
        self.adaptive_stats = AdaptiveStats()
        self._policy_entry_cache.clear()

    def close(self) -> None:
        self._purchase_oracle.close()
        super().close()

    def _write_v2_checkpoint(self, directory: Path) -> None:
        super()._write_v2_checkpoint(directory)
        with self._cache_lock:
            payload = {
                "options": self.options,
                "quality_target": self.quality_target,
                "stats": AdaptiveStats(**asdict(self.adaptive_stats)),
                "policy_entry_cache": dict(self._policy_entry_cache),
            }
        with (directory / "adaptive.pkl").open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())

    def load_checkpoint(self, path=None) -> None:
        super().load_checkpoint(path)
        source = Path(path) if path is not None else self.checkpoint_path
        if source is not None and source.is_dir() and (source / "adaptive.pkl").exists():
            with (source / "adaptive.pkl").open("rb") as handle:
                payload = pickle.load(handle)
            raw_options = payload.get("options")
            loaded_options = (
                AdaptiveOptions(**vars(raw_options))
                if isinstance(raw_options, AdaptiveOptions)
                else raw_options
            )
            if loaded_options != self.options:
                raise ValueError("adaptive checkpoint options do not match")
            raw_stats = payload["stats"]
            self.adaptive_stats = (
                AdaptiveStats(**vars(raw_stats))
                if isinstance(raw_stats, AdaptiveStats)
                else AdaptiveStats(**raw_stats)
            )
            self._policy_entry_cache = {
                key: entries
                for key, entries in self._normalize_policy_mapping(
                    dict(payload.get("policy_entry_cache", {}))
                ).items()
                if any(entry.action is None for entry in entries)
            }

    def _compute_canonical(self, day: int, state: JointState) -> StateValue:
        if day == self.cfg.deadline or all_absorbed(state):
            return terminal_state_value(self.cfg, state)
        expected_value = np.zeros(self.cfg.n_players, dtype=np.float64)
        expected_success = np.zeros(self.cfg.n_players, dtype=np.float64)
        state_key = self._state_key(day, state)
        for weather in self.cfg.weather_order:
            probability = float(self.cfg.p_weather[weather])
            outcome = self._solve_adaptive_stage(day, state, weather)
            expected_value += probability * np.asarray(outcome.value)
            expected_success += probability * np.asarray(outcome.success)
            with self._cache_lock:
                key = (state_key, weather)
                if outcome.actions is not None:
                    self._policy_cache[key] = outcome.actions
                else:
                    self._policy_entry_cache[key] = outcome.policy
        return StateValue(tuple(expected_value), tuple(expected_success))

    def policy_entries_for(
        self, day: int, state: JointState, weather: Weather
    ) -> tuple[PolicyEntry, ...]:
        canonical, canonical_to_original = canonicalize_state(state)
        state_key = self._state_key(day, canonical)
        if state_key not in self._value_cache:
            self.solve_state(day, state)
        key = (state_key, weather)
        with self._cache_lock:
            entries = self._policy_entry_cache.get(key)
            if entries is None:
                entries = tuple(
                    PolicyEntry.pure(action) for action in self._policy_cache[key]
                )
        output: list[PolicyEntry | None] = [None] * len(entries)
        for canonical_index, original_index in enumerate(canonical_to_original):
            output[original_index] = entries[canonical_index]
        return tuple(entry for entry in output if entry is not None)

    def policy_for(
        self, day: int, state: JointState, weather: Weather
    ) -> tuple[Action, ...]:
        entries = self.policy_entries_for(day, state, weather)
        actions = []
        for entry in entries:
            if entry.action is not None:
                actions.append(entry.action)
            else:
                actions.append(max(entry.distribution, key=lambda item: item[1])[0])
        return tuple(actions)

    def _full_action_sets(
        self, state: JointState, weather: Weather
    ) -> tuple[tuple[Action, ...], ...]:
        by_state: dict[PlayerState, tuple[Action, ...]] = {}
        output = []
        for player in state:
            actions = by_state.get(player)
            if actions is None:
                try:
                    actions = enumerate_individual_actions(
                        self.cfg,
                        player,
                        weather,
                        max_actions=self.limits.max_actions_per_player,
                    )
                except ActionEnumerationLimitExceeded as exc:
                    raise ResourceLimitExceeded(str(exc)) from exc
                by_state[player] = actions
            output.append(actions)
        return tuple(output)

    def _full_action_array_spaces(
        self, state: JointState, weather: Weather
    ) -> tuple[IndividualActionArrays, ...]:
        by_state: dict[PlayerState, IndividualActionArrays] = {}
        output = []
        for player in state:
            actions = by_state.get(player)
            if actions is None:
                try:
                    actions = enumerate_individual_action_arrays(
                        self.cfg,
                        player,
                        weather,
                        max_actions=self.limits.max_actions_per_player,
                    )
                except ActionEnumerationLimitExceeded as exc:
                    raise ResourceLimitExceeded(str(exc)) from exc
                by_state[player] = actions
            output.append(actions)
        return tuple(output)

    def _initial_candidates(
        self, full_actions: Sequence[Sequence[Action]]
    ) -> tuple[tuple[Action, ...], ...]:
        return self._candidate_sets(
            full_actions,
            per_skeleton=min(
                self.options.initial_candidates,
                self.options.max_initial_candidates,
            ),
            warm_actions=self.options.warm_initial_actions,
        )

    def _stage_candidates(
        self, full_actions: Sequence[Sequence[Action]]
    ) -> tuple[tuple[Action, ...], ...]:
        return self._candidate_sets(
            full_actions,
            per_skeleton=min(
                self.options.village_candidates_per_skeleton,
                self.options.max_village_candidates_per_skeleton,
            ),
        )

    def _candidate_sets(
        self,
        full_actions: Sequence[Sequence[Action]],
        *,
        per_skeleton: int,
        warm_actions: Sequence[Action] = (),
    ) -> tuple[tuple[Action, ...], ...]:
        by_identity: dict[int, tuple[Action, ...]] = {}
        output: list[tuple[Action, ...]] = []
        for actions in full_actions:
            identity = id(actions)
            candidates = by_identity.get(identity)
            if candidates is None:
                candidates = deterministic_action_candidates(
                    self.cfg,
                    actions,
                    per_skeleton=per_skeleton,
                    warm_actions=warm_actions,
                )
                by_identity[identity] = candidates
            output.append(candidates)
        return tuple(output)

    def _restricted_tensor(
        self,
        action_sets: Sequence[Sequence[Action]],
        evaluator,
        *,
        symmetric_state: JointState | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shape = tuple(len(actions) for actions in action_sets)
        profiles = prod(shape)
        if profiles > self.limits.max_stage_profiles_evaluated:
            raise ResourceLimitExceeded(
                f"restricted game has {profiles:,} profiles, above the stage budget"
            )
        payoff = np.full(shape + (self.cfg.n_players,), -np.inf, dtype=np.float64)
        success = np.zeros(shape + (self.cfg.n_players,), dtype=np.float64)
        valid = np.zeros(shape, dtype=bool)
        player_permutations = self._restricted_player_permutations(
            action_sets, symmetric_state
        )
        evaluated_profiles = 0
        if len(player_permutations) == 1:
            for indices in iter_index_blocks(
                shape, block_size=self.limits.profile_chunk_size
            ):
                self._check_cancelled()
                evaluation = evaluator(indices)
                coordinates = tuple(
                    indices[:, player] for player in range(self.cfg.n_players)
                )
                valid[coordinates] = evaluation.valid
                payoff[coordinates] = evaluation.payoff
                success[coordinates] = evaluation.success
                evaluated_profiles += len(indices)
        else:
            representatives = []
            for index in product(*(range(size) for size in shape)):
                canonical = min(
                    tuple(index[permutation[player]] for player in range(self.cfg.n_players))
                    for permutation in player_permutations
                )
                if index == canonical:
                    representatives.append(index)
            for start in range(
                0, len(representatives), self.limits.profile_chunk_size
            ):
                self._check_cancelled()
                indices = np.asarray(
                    representatives[start : start + self.limits.profile_chunk_size],
                    dtype=np.int64,
                )
                evaluation = evaluator(indices)
                evaluated_profiles += len(indices)
                for row, representative in enumerate(indices):
                    index = tuple(int(value) for value in representative)
                    for permutation in player_permutations:
                        transformed = tuple(
                            index[permutation[player]]
                            for player in range(self.cfg.n_players)
                        )
                        valid[transformed] = evaluation.valid[row]
                        payoff[transformed] = evaluation.payoff[row, permutation]
                        success[transformed] = evaluation.success[row, permutation]
        self.adaptive_stats.restricted_profiles += evaluated_profiles
        return payoff, success, valid

    def _restricted_player_permutations(
        self,
        action_sets: Sequence[Sequence[Action]],
        state: JointState | None,
    ) -> tuple[tuple[int, ...], ...]:
        identity = tuple(range(self.cfg.n_players))
        if state is None:
            return (identity,)
        output = []
        for permutation in permutations(range(self.cfg.n_players)):
            if all(
                state[player] == state[permutation[player]]
                and action_sets[player] == action_sets[permutation[player]]
                for player in range(self.cfg.n_players)
            ):
                output.append(tuple(permutation))
        return tuple(output) if output else (identity,)

    def _restricted_stage_equilibrium(
        self,
        day: int,
        state: JointState,
        weather: Weather,
        action_sets: Sequence[Sequence[Action]],
    ) -> PureEquilibrium | MixedEquilibrium:
        evaluator = self._stage_evaluator(day, state, weather, action_sets)
        symmetric = self._symmetric_restricted_pure(
            state, action_sets, evaluator
        )
        if symmetric is not None:
            return symmetric
        if prod(len(actions) for actions in action_sets) > 50_000:
            iterative = self._iterative_restricted_pure(action_sets, evaluator)
            if iterative is not None:
                return iterative
        payoff, success, valid = self._restricted_tensor(
            action_sets, evaluator, symmetric_state=state
        )
        indices = pure_nash_indices(payoff, valid, atol=self.equilibrium_atol)
        if len(indices):
            return select_pure_equilibrium(
                indices, payoff, success, action_sets
            )
        if self.options.equilibrium == "pure":
            raise NoPureEquilibrium(
                f"restricted adaptive game has no pure equilibrium at day={day}"
            )
        return minimize_nashconv(
            payoff,
            valid=valid,
            success=success,
            seed=self.options.seed + day,
        )

    def _iterative_restricted_pure(
        self,
        action_sets: Sequence[Sequence[Action]],
        evaluator,
    ) -> PureEquilibrium | None:
        """Find and fully verify a candidate pure NE before tensor fallback."""
        starts = (
            tuple(0 for _ in action_sets),
            tuple((len(actions) - 1) // 2 for actions in action_sets),
            tuple(len(actions) - 1 for actions in action_sets),
        )
        equilibria: dict[tuple[int, ...], PureEquilibrium] = {}
        for start in starts:
            for order in permutations(range(self.cfg.n_players)):
                current = list(start)
                seen = {tuple(current)}
                for _ in range(32):
                    changed = False
                    failed = False
                    last_evaluation = None
                    last_player = -1
                    for player in order:
                        indices = np.repeat(
                            np.asarray(current, dtype=np.int64)[None, :],
                            len(action_sets[player]),
                            axis=0,
                        )
                        indices[:, player] = np.arange(
                            len(action_sets[player]), dtype=np.int64
                        )
                        evaluation = evaluator(indices)
                        self.adaptive_stats.restricted_profiles += len(indices)
                        feasible = np.flatnonzero(evaluation.valid)
                        if len(feasible) == 0:
                            failed = True
                            break
                        payoffs = evaluation.payoff[feasible, player]
                        best_value = float(np.max(payoffs))
                        best_rows = feasible[
                            payoffs >= best_value - self.equilibrium_atol
                        ]
                        best_action = int(best_rows[0])
                        current_action = current[player]
                        current_valid = bool(evaluation.valid[current_action])
                        current_value = (
                            float(evaluation.payoff[current_action, player])
                            if current_valid
                            else -np.inf
                        )
                        if (
                            not current_valid
                            or best_value > current_value + self.equilibrium_atol
                        ):
                            current[player] = best_action
                            changed = True
                        last_evaluation = evaluation
                        last_player = player
                    if failed:
                        break
                    index = tuple(current)
                    if not changed:
                        if last_evaluation is None or last_player < 0:
                            break
                        row = current[last_player]
                        if not last_evaluation.valid[row]:
                            break
                        actions = tuple(
                            action_sets[player][index[player]]
                            for player in range(self.cfg.n_players)
                        )
                        equilibria[index] = PureEquilibrium(
                            index,
                            actions,
                            tuple(
                                float(value)
                                for value in last_evaluation.payoff[row]
                            ),
                            tuple(
                                float(value)
                                for value in last_evaluation.success[row]
                            ),
                        )
                        break
                    if index in seen:
                        break
                    seen.add(index)
            if equilibria:
                break
        if not equilibria:
            return None
        return select_pure_equilibrium_candidates(tuple(equilibria.values()))

    def _symmetric_restricted_pure(
        self,
        state: JointState,
        action_sets: Sequence[Sequence[Action]],
        evaluator,
    ) -> PureEquilibrium | None:
        if any(player != state[0] for player in state[1:]) or any(
            actions != action_sets[0] for actions in action_sets[1:]
        ):
            return None
        action_count = len(action_sets[0])
        diagonal = np.repeat(
            np.arange(action_count, dtype=np.int64)[:, None],
            self.cfg.n_players,
            axis=1,
        )
        diagonal_evaluation = evaluator(diagonal)
        self.adaptive_stats.restricted_profiles += len(diagonal)
        candidate_indices = np.flatnonzero(diagonal_evaluation.valid)
        if len(candidate_indices) == 0:
            return None

        deviations_per_candidate = max(0, action_count - 1)
        if deviations_per_candidate:
            deviation_indices = np.empty(
                (
                    len(candidate_indices) * deviations_per_candidate,
                    self.cfg.n_players,
                ),
                dtype=np.int64,
            )
            all_actions = np.arange(action_count, dtype=np.int64)
            for block_index, candidate_index in enumerate(candidate_indices):
                start = block_index * deviations_per_candidate
                stop = start + deviations_per_candidate
                deviation_indices[start:stop] = int(candidate_index)
                deviation_indices[start:stop, 0] = all_actions[
                    all_actions != int(candidate_index)
                ]
            deviation_evaluation = evaluator(deviation_indices)
            self.adaptive_stats.restricted_profiles += len(deviation_indices)
        else:
            deviation_evaluation = None

        equilibria = []
        for block_index, candidate_index in enumerate(candidate_indices):
            value = diagonal_evaluation.payoff[int(candidate_index)]
            best_response = float(value[0])
            if deviation_evaluation is not None:
                start = block_index * deviations_per_candidate
                stop = start + deviations_per_candidate
                feasible = deviation_evaluation.valid[start:stop]
                best_response = max(
                    best_response,
                    float(
                        np.max(
                            np.where(
                                feasible,
                                deviation_evaluation.payoff[start:stop, 0],
                                -np.inf,
                            )
                        )
                    )
                )
            if value[0] < best_response - self.equilibrium_atol:
                continue
            index = tuple(int(candidate_index) for _ in range(self.cfg.n_players))
            action = action_sets[0][int(candidate_index)]
            equilibria.append(
                PureEquilibrium(
                    index=index,
                    actions=tuple(action for _ in range(self.cfg.n_players)),
                    value=tuple(float(item) for item in value),
                    success=tuple(
                        float(item)
                        for item in diagonal_evaluation.success[int(candidate_index)]
                    ),
                )
            )
        if not equilibria:
            return None
        return select_pure_equilibrium_candidates(equilibria)

    def _scan_pure_stage_deviations(
        self,
        day: int,
        state: JointState,
        weather: Weather,
        equilibrium: PureEquilibrium,
        full_actions: Sequence[Sequence[Action] | IndividualActionArrays],
        candidate_sets: Sequence[set[Action]],
    ) -> tuple[tuple[float, ...], tuple[float, ...], list[list[Action]]]:
        lower: list[float] = []
        upper: list[float] = []
        additions: list[list[Action]] = []
        symmetric_results: dict[
            tuple[PlayerState, Action, frozenset[Action]],
            tuple[float, float, tuple[Action, ...]],
        ] = {}
        for player in range(self.cfg.n_players):
            symmetry_key = (
                state[player],
                equilibrium.actions[player],
                frozenset(candidate_sets[player]),
            )
            symmetric = symmetric_results.get(symmetry_key)
            if symmetric is not None:
                player_lower, player_upper, player_additions = symmetric
                lower.append(player_lower)
                upper.append(player_upper)
                additions.append(list(player_additions))
                continue
            current = equilibrium.value[player]
            best_exact = current
            best_upper = current
            profitable: list[tuple[float, Action]] = []
            actions = full_actions[player]
            self.adaptive_stats.full_deviation_actions += len(actions)
            if (
                isinstance(actions, IndividualActionArrays)
                and self.enable_bound_pruning
                and self.options.purchase_oracle_backend != "off"
            ):
                threshold = (
                    current
                    - self.equilibrium_atol
                    - self.bound_pruning_slack
                )
                screened = self._purchase_oracle.screen(
                    day + 1,
                    state,
                    equilibrium.actions,
                    player,
                    actions,
                    weather,
                    threshold,
                )
                self.adaptive_stats.purchase_oracle_calls += 1
                self.adaptive_stats.purchase_oracle_actions += len(actions)
                self.adaptive_stats.purchase_oracle_seconds += (
                    screened.elapsed_seconds
                )
                self.adaptive_stats.purchase_regions_pruned += (
                    screened.regions_pruned
                )
                self.adaptive_stats.purchase_oracle_cache_entries = (
                    self._purchase_oracle.cache_entries
                )
                self.adaptive_stats.purchase_oracle_cache_bytes = (
                    self._purchase_oracle.cache_bytes
                )
                if screened.backend == "cuda":
                    self.adaptive_stats.purchase_oracle_cuda_calls += 1
                else:
                    self.adaptive_stats.purchase_oracle_cpu_calls += 1
                self.adaptive_stats.deviation_actions_pruned += (
                    screened.pruned_count
                )
                self._update_stats(upper_bound_profiles=len(actions))
                self._sync_upper_bound_stats()
                if np.isfinite(screened.max_pruned_bound):
                    best_upper = max(best_upper, screened.max_pruned_bound)

                survivor_indices = screened.survivor_indices
                for survivor_start in range(
                    0, len(survivor_indices), self.limits.profile_chunk_size
                ):
                    self._check_cancelled()
                    selected = survivor_indices[
                        survivor_start : survivor_start
                        + self.limits.profile_chunk_size
                    ]
                    batch = apply_unilateral_transition_encoded_arrays(
                        self.cfg,
                        state,
                        equilibrium.actions,
                        player,
                        actions.kind[selected],
                        actions.destination[selected],
                        actions.buy_water[selected],
                        actions.buy_food[selected],
                        weather,
                    )
                    if not np.all(batch.valid):
                        raise AssertionError(
                            "purchase oracle retained an invalid deviation"
                        )
                    keep = batch.valid.copy()
                    evaluation = self._evaluate_transition_arrays(
                        day + 1, state, batch, keep
                    )
                    self.adaptive_stats.deviation_actions_evaluated += len(
                        selected
                    )
                    for row, action_index in enumerate(selected):
                        value = float(evaluation.payoff[row, player])
                        best_exact = max(best_exact, value)
                        action = actions.action_at(int(action_index))
                        if (
                            value > current + self.equilibrium_atol
                            and action not in candidate_sets[player]
                        ):
                            profitable.append((value, action))
                profitable.sort(key=lambda item: (-item[0], item[1].code))
                player_additions = tuple(
                    action
                    for _, action in profitable[
                        : self.options.additions_per_player
                    ]
                )
                player_lower = max(0.0, best_exact - current)
                player_upper = max(
                    0.0, max(best_exact, best_upper) - current
                )
                additions.append(list(player_additions))
                lower.append(player_lower)
                upper.append(player_upper)
                symmetric_results[symmetry_key] = (
                    player_lower,
                    player_upper,
                    player_additions,
                )
                continue
            scan_chunk_size = (
                max(self.limits.profile_chunk_size, 65_536)
                if isinstance(actions, IndividualActionArrays)
                else self.limits.profile_chunk_size
            )
            for start in range(0, len(actions), scan_chunk_size):
                self._check_cancelled()
                stop = min(start + scan_chunk_size, len(actions))
                if isinstance(actions, IndividualActionArrays):
                    batch = apply_unilateral_transition_encoded_arrays(
                        self.cfg,
                        state,
                        equilibrium.actions,
                        player,
                        actions.kind[start:stop],
                        actions.destination[start:stop],
                        actions.buy_water[start:stop],
                        actions.buy_food[start:stop],
                        weather,
                    )
                else:
                    block = actions[start:stop]
                    batch = apply_unilateral_transition_arrays(
                        self.cfg,
                        state,
                        equilibrium.actions,
                        player,
                        block,
                        weather,
                    )
                bounds = self._unilateral_upper_bounds(day + 1, state, batch, player)
                keep = batch.valid & (
                    bounds >= current - self.equilibrium_atol - self.bound_pruning_slack
                )
                pruned = batch.valid & ~keep
                if np.any(pruned):
                    best_upper = max(best_upper, float(np.max(bounds[pruned])))
                self.adaptive_stats.deviation_actions_pruned += int(
                    np.sum(pruned)
                )
                evaluation = self._evaluate_transition_arrays(
                    day + 1, state, batch, keep
                )
                self.adaptive_stats.deviation_actions_evaluated += int(np.sum(keep))
                for row in np.flatnonzero(keep):
                    value = float(evaluation.payoff[int(row), player])
                    best_exact = max(best_exact, value)
                    action = (
                        actions.action_at(start + int(row))
                        if isinstance(actions, IndividualActionArrays)
                        else block[int(row)]
                    )
                    if (
                        value > current + self.equilibrium_atol
                        and action not in candidate_sets[player]
                    ):
                        profitable.append((value, action))
            profitable.sort(key=lambda item: (-item[0], item[1].code))
            player_additions = tuple(
                action
                for _, action in profitable[: self.options.additions_per_player]
            )
            player_lower = max(0.0, best_exact - current)
            player_upper = max(0.0, max(best_exact, best_upper) - current)
            additions.append(list(player_additions))
            lower.append(player_lower)
            upper.append(player_upper)
            symmetric_results[symmetry_key] = (
                player_lower,
                player_upper,
                player_additions,
            )
        return tuple(lower), tuple(upper), additions

    def _unilateral_upper_bounds(
        self,
        continuation_day: int,
        state: JointState,
        batch: BatchTransitionArrays,
        player: int,
    ) -> np.ndarray:
        bounds = np.full(len(batch.valid), -np.inf, dtype=np.float64)
        rows = np.flatnonzero(batch.valid)
        if len(rows) == 0:
            self._update_stats(upper_bound_profiles=len(batch.valid))
            return bounds
        original = state[player]
        if original.status is not Status.ACTIVE:
            bounds[rows] = original.fixed_payoff_scaled / self.cfg.money_scale
            self._update_stats(upper_bound_profiles=len(batch.valid))
            return bounds

        kinds = batch.kind[rows, player]
        failed = kinds == int(ActionKind.FAIL)
        if np.any(failed):
            bounds[rows[failed]] = (
                original.cash_scaled - self.cfg.failure_penalty_scaled
            ) / self.cfg.money_scale

        active_rows = rows[~failed]
        if len(active_rows):
            position = batch.position[active_rows, player]
            water = batch.water[active_rows, player]
            food = batch.food[active_rows, player]
            cash = batch.cash_scaled[active_rows, player]
            finished = position == self.cfg.end
            if np.any(finished):
                terminal_rows = active_rows[finished]
                refund_scaled = (
                    self.cfg.money_scale
                    * (
                        self.cfg.water_price * water[finished].astype(np.int64)
                        + self.cfg.food_price * food[finished].astype(np.int64)
                    )
                ) // 2
                bounds[terminal_rows] = (
                    cash[finished] + refund_scaled
                ) / self.cfg.money_scale

            continuing_rows = active_rows[~finished]
            if len(continuing_rows):
                continuing_position = position[~finished]
                continuing_cash = cash[~finished]
                if continuation_day >= self.cfg.deadline:
                    residual = np.full(
                        len(continuing_rows),
                        -float(self.cfg.failure_penalty),
                        dtype=np.float64,
                    )
                else:
                    continuing_water = water[~finished]
                    continuing_food = food[~finished]
                    resource_ids = self._upper_bound.resources.id_grid[
                        continuing_water, continuing_food
                    ]
                    residual = self._upper_bound.residuals_by_id(
                        continuation_day,
                        continuing_position,
                        resource_ids,
                    )
                cash_value = continuing_cash / self.cfg.money_scale
                bounds[continuing_rows] = np.nextafter(
                    cash_value + residual, np.inf
                )
        self._update_stats(upper_bound_profiles=len(batch.valid))
        self._sync_upper_bound_stats()
        return bounds

    def _mixed_stage_outcome(
        self,
        equilibrium: MixedEquilibrium,
        action_sets: Sequence[Sequence[Action]],
    ) -> AdaptiveStageOutcome:
        policy = tuple(
            PolicyEntry.mixed(
                tuple(
                    (action, probability)
                    for action, probability in zip(
                        action_sets[player],
                        equilibrium.probabilities[player],
                        strict=True,
                    )
                    if probability > 1e-12
                )
            )
            for player in range(self.cfg.n_players)
        )
        self.adaptive_stats.approximate_mixed_stages += 1
        self.adaptive_stats.max_stage_regret_upper = max(
            self.adaptive_stats.max_stage_regret_upper,
            max(equilibrium.regrets, default=0.0),
        )
        return AdaptiveStageOutcome(
            equilibrium.value,
            equilibrium.success,
            policy,
            None,
            "APPROX_MIXED",
            equilibrium.regrets,
            tuple(float("inf") for _ in equilibrium.regrets),
        )

    def _solve_adaptive_stage(
        self, day: int, state: JointState, weather: Weather
    ) -> AdaptiveStageOutcome:
        at_village = any(
            player.position in self.cfg.villages
            for player in state
            if player.status.value == 0
        )
        if not at_village:
            full_actions: tuple[
                Sequence[Action] | IndividualActionArrays, ...
            ] = self._full_action_sets(state, weather)
            candidates = tuple(tuple(actions) for actions in full_actions)
        else:
            full_actions = self._full_action_array_spaces(state, weather)
            per_skeleton = min(
                self.options.village_candidates_per_skeleton,
                self.options.max_village_candidates_per_skeleton,
            )
            candidates = tuple(
                deterministic_array_action_candidates(
                    self.cfg,
                    actions,
                    per_skeleton=per_skeleton,
                )
                for actions in full_actions
            )
        full_coverage = all(
            len(candidate_actions) == len(actions)
            and all(
                candidate_actions[index]
                == (
                    actions.action_at(index)
                    if isinstance(actions, IndividualActionArrays)
                    else actions[index]
                )
                for index in range(len(actions))
            )
            for candidate_actions, actions in zip(
                candidates, full_actions, strict=True
            )
        )
        if not full_coverage:
            full_actions = None
        candidate_sets = [set(actions) for actions in candidates]
        last_pure: PureEquilibrium | None = None
        last_lower = tuple(float("inf") for _ in range(self.cfg.n_players))
        last_upper = last_lower
        for _ in range(self.options.max_restricted_rounds):
            self.adaptive_stats.restricted_rounds += 1
            equilibrium = self._restricted_stage_equilibrium(
                day, state, weather, candidates
            )
            if isinstance(equilibrium, MixedEquilibrium):
                return self._mixed_stage_outcome(equilibrium, candidates)
            last_pure = equilibrium
            if full_coverage:
                zero_regret = tuple(0.0 for _ in range(self.cfg.n_players))
                self.adaptive_stats.certified_stages += 1
                return AdaptiveStageOutcome(
                    equilibrium.value,
                    equilibrium.success,
                    tuple(
                        PolicyEntry.pure(action) for action in equilibrium.actions
                    ),
                    equilibrium.actions,
                    "CERTIFIED_PURE",
                    zero_regret,
                    zero_regret,
                )
            round_full_actions = (
                self._full_action_array_spaces(state, weather)
                if at_village
                else self._full_action_sets(state, weather)
            )
            lower, upper, additions = self._scan_pure_stage_deviations(
                day,
                state,
                weather,
                equilibrium,
                round_full_actions,
                candidate_sets,
            )
            del round_full_actions
            last_lower, last_upper = lower, upper
            added = 0
            candidate_lists = [list(actions) for actions in candidates]
            for player, new_actions in enumerate(additions):
                for action in new_actions:
                    if action in candidate_sets[player]:
                        continue
                    skeleton = action_skeleton(action)
                    skeleton_count = sum(
                        action_skeleton(existing) == skeleton
                        for existing in candidate_lists[player]
                    )
                    if skeleton_count >= self.options.max_village_candidates_per_skeleton:
                        continue
                    candidate_sets[player].add(action)
                    candidate_lists[player].append(action)
                    added += 1
                candidate_lists[player].sort(key=lambda action: action.code)
            candidates = tuple(tuple(actions) for actions in candidate_lists)
            self.adaptive_stats.candidates_added += added
            if added == 0 and max(lower, default=0.0) <= self.equilibrium_atol:
                self.adaptive_stats.certified_stages += 1
                self.adaptive_stats.max_stage_regret_upper = max(
                    self.adaptive_stats.max_stage_regret_upper,
                    max(upper, default=0.0),
                )
                return AdaptiveStageOutcome(
                    equilibrium.value,
                    equilibrium.success,
                    tuple(PolicyEntry.pure(action) for action in equilibrium.actions),
                    equilibrium.actions,
                    "CERTIFIED_PURE",
                    lower,
                    upper,
                )
            if added == 0:
                return AdaptiveStageOutcome(
                    equilibrium.value,
                    equilibrium.success,
                    tuple(PolicyEntry.pure(action) for action in equilibrium.actions),
                    equilibrium.actions,
                    "APPROX_PURE",
                    lower,
                    upper,
                )
        if last_pure is None:
            raise NoPureEquilibrium("adaptive stage produced no reportable equilibrium")
        return AdaptiveStageOutcome(
            last_pure.value,
            last_pure.success,
            tuple(PolicyEntry.pure(action) for action in last_pure.actions),
            last_pure.actions,
            "APPROX_PURE",
            last_lower,
            last_upper,
        )

    def _restricted_initial_equilibrium(
        self, action_sets: Sequence[Sequence[Action]]
    ) -> PureEquilibrium | MixedEquilibrium:
        state = initial_joint_state(self.cfg)
        evaluator = self._initial_evaluator(action_sets)
        symmetric = self._symmetric_restricted_pure(
            state, action_sets, evaluator
        )
        if symmetric is not None:
            return symmetric
        payoff, success, valid = self._restricted_tensor(
            action_sets,
            evaluator,
            symmetric_state=state,
        )
        indices = pure_nash_indices(payoff, valid, atol=self.equilibrium_atol)
        if len(indices):
            return select_pure_equilibrium(indices, payoff, success, action_sets)
        if self.options.equilibrium == "pure":
            raise NoPureEquilibrium("restricted initial game has no pure equilibrium")
        return minimize_nashconv(
            payoff, valid=valid, success=success, seed=self.options.seed
        )

    def _scan_pure_initial_deviations(
        self,
        equilibrium: PureEquilibrium,
        full_actions: Sequence[Sequence[Action]],
        candidate_sets: Sequence[set[Action]],
    ) -> tuple[tuple[float, ...], tuple[float, ...], list[list[Action]]]:
        state = initial_joint_state(self.cfg)
        lower: list[float] = []
        upper: list[float] = []
        additions: list[list[Action]] = []
        symmetric_results: dict[
            tuple[PlayerState, Action, int, frozenset[Action]],
            tuple[float, float, tuple[Action, ...]],
        ] = {}
        for player in range(self.cfg.n_players):
            symmetry_key = (
                state[player],
                equilibrium.actions[player],
                id(full_actions[player]),
                frozenset(candidate_sets[player]),
            )
            symmetric = symmetric_results.get(symmetry_key)
            if symmetric is not None:
                player_lower, player_upper, player_additions = symmetric
                lower.append(player_lower)
                upper.append(player_upper)
                additions.append(list(player_additions))
                continue
            current = equilibrium.value[player]
            best_exact = current
            best_upper = current
            profitable: list[tuple[float, Action]] = []
            actions = full_actions[player]
            self.adaptive_stats.full_deviation_actions += len(actions)
            if (
                self.enable_bound_pruning
                and self.options.purchase_oracle_backend != "off"
            ):
                started = perf_counter()
                buy_water = np.fromiter(
                    (action.buy_water for action in actions),
                    dtype=np.int32,
                    count=len(actions),
                )
                buy_food = np.fromiter(
                    (action.buy_food for action in actions),
                    dtype=np.int32,
                    count=len(actions),
                )
                original = state[player]
                next_water = original.water + buy_water
                next_food = original.food + buy_food
                cost_scaled = self.cfg.money_scale * (
                    self.cfg.water_price * buy_water.astype(np.int64)
                    + self.cfg.food_price * buy_food.astype(np.int64)
                )
                next_cash = original.cash_scaled - cost_scaled
                resource_ids = self._upper_bound.resources.id_grid[
                    next_water, next_food
                ]
                positions = np.full(
                    len(actions), original.position, dtype=np.int16
                )
                residual = self._upper_bound.residuals_by_id(
                    0, positions, resource_ids
                )
                bounds = np.nextafter(
                    next_cash.astype(np.float64) / self.cfg.money_scale
                    + residual,
                    np.inf,
                )
                threshold = (
                    current
                    - self.equilibrium_atol
                    - self.bound_pruning_slack
                )
                keep = bounds >= threshold
                pruned = ~keep
                if np.any(pruned):
                    best_upper = max(
                        best_upper, float(np.max(bounds[pruned]))
                    )
                survivor_indices = np.flatnonzero(keep)
                self.adaptive_stats.deviation_actions_pruned += int(
                    np.sum(pruned)
                )
                self.adaptive_stats.purchase_oracle_calls += 1
                self.adaptive_stats.purchase_oracle_actions += len(actions)
                self.adaptive_stats.purchase_oracle_cpu_calls += 1
                self.adaptive_stats.purchase_oracle_seconds += (
                    perf_counter() - started
                )
                self._update_stats(upper_bound_profiles=len(actions))
                self._sync_upper_bound_stats()
                for survivor_start in range(
                    0, len(survivor_indices), self.limits.profile_chunk_size
                ):
                    self._check_cancelled()
                    selected = survivor_indices[
                        survivor_start : survivor_start
                        + self.limits.profile_chunk_size
                    ]
                    successors: list[JointState | None] = []
                    for action_index in selected:
                        profile = list(equilibrium.actions)
                        profile[player] = actions[int(action_index)]
                        post = apply_initial_purchases(self.cfg, state, profile)
                        if post is None:
                            raise AssertionError(
                                "initial purchase oracle retained an invalid action"
                            )
                        successors.append(post)
                    selected_valid = np.ones(len(selected), dtype=bool)
                    evaluation = self._evaluate_successors(
                        0, selected_valid, successors
                    )
                    self.adaptive_stats.deviation_actions_evaluated += len(
                        selected
                    )
                    for row, action_index in enumerate(selected):
                        value = float(evaluation.payoff[row, player])
                        best_exact = max(best_exact, value)
                        action = actions[int(action_index)]
                        if (
                            value > current + self.equilibrium_atol
                            and action not in candidate_sets[player]
                        ):
                            profitable.append((value, action))
                profitable.sort(key=lambda item: (-item[0], item[1].code))
                player_additions = tuple(
                    action
                    for _, action in profitable[
                        : self.options.additions_per_player
                    ]
                )
                player_lower = max(0.0, best_exact - current)
                player_upper = max(
                    0.0, max(best_exact, best_upper) - current
                )
                additions.append(list(player_additions))
                lower.append(player_lower)
                upper.append(player_upper)
                symmetric_results[symmetry_key] = (
                    player_lower,
                    player_upper,
                    player_additions,
                )
                continue
            for start in range(0, len(actions), self.limits.profile_chunk_size):
                self._check_cancelled()
                block = actions[start : start + self.limits.profile_chunk_size]
                successors: list[JointState | None] = []
                valid = np.zeros(len(block), dtype=bool)
                for row, action in enumerate(block):
                    profile = list(equilibrium.actions)
                    profile[player] = action
                    post = apply_initial_purchases(self.cfg, state, profile)
                    successors.append(post)
                    valid[row] = post is not None
                bounds = self._upper_bounds_from_successors(
                    0, valid, successors, player
                )
                keep = valid & (
                    bounds >= current - self.equilibrium_atol - self.bound_pruning_slack
                )
                pruned = valid & ~keep
                if np.any(pruned):
                    best_upper = max(best_upper, float(np.max(bounds[pruned])))
                self.adaptive_stats.deviation_actions_pruned += int(
                    np.sum(pruned)
                )
                evaluation = self._evaluate_successors(0, keep, successors)
                self.adaptive_stats.deviation_actions_evaluated += int(np.sum(keep))
                for row in np.flatnonzero(keep):
                    value = float(evaluation.payoff[int(row), player])
                    best_exact = max(best_exact, value)
                    action = block[int(row)]
                    if (
                        value > current + self.equilibrium_atol
                        and action not in candidate_sets[player]
                    ):
                        profitable.append((value, action))
            profitable.sort(key=lambda item: (-item[0], item[1].code))
            player_additions = tuple(
                action
                for _, action in profitable[: self.options.additions_per_player]
            )
            player_lower = max(0.0, best_exact - current)
            player_upper = max(0.0, max(best_exact, best_upper) - current)
            additions.append(list(player_additions))
            lower.append(player_lower)
            upper.append(player_upper)
            symmetric_results[symmetry_key] = (
                player_lower,
                player_upper,
                player_additions,
            )
        return tuple(lower), tuple(upper), additions

    def solve_initial_adaptive(self) -> AdaptiveInitialResult:
        state = initial_joint_state(self.cfg)
        try:
            shared = enumerate_initial_purchases_bounded(
                self.cfg,
                state[0],
                max_actions=self.limits.max_actions_per_player,
            )
        except ActionEnumerationLimitExceeded as exc:
            raise ResourceLimitExceeded(str(exc)) from exc
        full_actions = tuple(shared for _ in state)
        candidates = self._initial_candidates(full_actions)
        candidate_sets = [set(actions) for actions in candidates]
        last_pure: PureEquilibrium | None = None
        last_lower = tuple(float("inf") for _ in state)
        last_upper = last_lower
        for _ in range(self.options.max_restricted_rounds):
            self.adaptive_stats.restricted_rounds += 1
            equilibrium = self._restricted_initial_equilibrium(candidates)
            if isinstance(equilibrium, MixedEquilibrium):
                policy = tuple(
                    PolicyEntry.mixed(
                        tuple(
                            (action, probability)
                            for action, probability in zip(
                                candidates[player],
                                equilibrium.probabilities[player],
                                strict=True,
                            )
                            if probability > 1e-12
                        )
                    )
                    for player in range(self.cfg.n_players)
                )
                return AdaptiveInitialResult(
                    StateValue(equilibrium.value, equilibrium.success),
                    policy,
                    None,
                    None,
                    "APPROX_MIXED",
                    equilibrium.regrets,
                    tuple(float("inf") for _ in equilibrium.regrets),
                )
            last_pure = equilibrium
            lower, upper, additions = self._scan_pure_initial_deviations(
                equilibrium, full_actions, candidate_sets
            )
            last_lower, last_upper = lower, upper
            candidate_lists = [list(actions) for actions in candidates]
            added = 0
            for player, new_actions in enumerate(additions):
                for action in new_actions:
                    if action not in candidate_sets[player]:
                        if (
                            len(candidate_lists[player])
                            >= self.options.max_initial_candidates
                        ):
                            continue
                        candidate_sets[player].add(action)
                        candidate_lists[player].append(action)
                        added += 1
                candidate_lists[player].sort(key=lambda action: action.code)
            candidates = tuple(tuple(actions) for actions in candidate_lists)
            self.adaptive_stats.candidates_added += added
            if added == 0:
                post = apply_initial_purchases(self.cfg, state, equilibrium.actions)
                if post is None:
                    raise AssertionError("certified initial profile is infeasible")
                status = (
                    "CERTIFIED_PURE"
                    if max(upper, default=0.0)
                    <= self.equilibrium_atol + self.bound_pruning_slack
                    else "APPROX_PURE"
                )
                return AdaptiveInitialResult(
                    StateValue(equilibrium.value, equilibrium.success),
                    tuple(PolicyEntry.pure(action) for action in equilibrium.actions),
                    equilibrium.actions,
                    post,
                    status,
                    lower,
                    upper,
                )
        if last_pure is None:
            raise NoPureEquilibrium("adaptive initial game produced no equilibrium")
        post = apply_initial_purchases(self.cfg, state, last_pure.actions)
        return AdaptiveInitialResult(
            StateValue(last_pure.value, last_pure.success),
            tuple(PolicyEntry.pure(action) for action in last_pure.actions),
            last_pure.actions,
            post,
            "APPROX_PURE",
            last_lower,
            last_upper,
        )


def _policy_payload(entries: Sequence[PolicyEntry]) -> tuple[dict[str, object], ...]:
    output = []
    for entry in entries:
        if entry.action is not None:
            output.append({"action": entry.action.label(), "probability": 1.0})
        else:
            output.append(
                {
                    "distribution": tuple(
                        {"action": action.label(), "probability": probability}
                        for action, probability in entry.distribution
                    )
                }
            )
    return tuple(output)


def solve_q3_2(
    config: Q3Config,
    backend: str = "adaptive",
    limits: SolverLimits | None = None,
    quality_target: float = 10.0,
    *,
    adaptive_options: AdaptiveOptions | None = None,
    heuristic_options: HeuristicOptions | None = None,
    budget_manager: BudgetManager | None = None,
    checkpoint: str | None = None,
    resume: bool = False,
    workers: int = 1,
    bound_threads: int | None = None,
    checkpoint_every_states: int = 0,
    checkpoint_every_pairs: int = 0,
    enable_bound_pruning: bool = True,
    bound_pruning_slack: float = 1e-6,
    record_pruning_certificates: bool = False,
    max_pruning_certificates: int = 10_000,
) -> Q32SolveResult:
    if backend == "heuristic":
        if checkpoint is not None or resume:
            raise ValueError("heuristic backend does not use exact-state checkpoints")
        from .heuristic import solve_q3_2_heuristic

        return solve_q3_2_heuristic(
            config,
            options=heuristic_options,
            quality_target=quality_target,
            budget_manager=budget_manager,
        )
    if backend not in {"exact", "adaptive"}:
        raise ValueError("backend must be 'exact', 'adaptive', or 'heuristic'")
    solver_class = ExactQ3Solver if backend == "exact" else AdaptiveQ3Solver
    kwargs = dict(
        limits=limits,
        workers=workers,
        bound_threads=bound_threads,
        checkpoint_path=checkpoint,
        checkpoint_every_states=checkpoint_every_states,
        checkpoint_every_pairs=checkpoint_every_pairs,
        enable_bound_pruning=enable_bound_pruning,
        bound_pruning_slack=bound_pruning_slack,
        record_pruning_certificates=record_pruning_certificates,
        max_pruning_certificates=max_pruning_certificates,
        budget_manager=budget_manager,
    )
    if backend == "adaptive":
        kwargs.update(options=adaptive_options, quality_target=quality_target)
    solver = solver_class(config, **kwargs)
    try:
        if resume:
            solver.load_checkpoint()
        if backend == "exact":
            exact: InitialSolveResult = solver.solve_initial_purchases()
            policy = {
                "initial": _policy_payload(
                    tuple(PolicyEntry.pure(action) for action in exact.purchases)
                ),
                "feedback_states": len(solver._policy_cache),
            }
            result = Q32SolveResult(
                status="EXACT_SELECTED",
                value_lower=exact.value.value,
                value_upper=exact.value.value,
                success=exact.value.success,
                max_regret_lower=0.0,
                max_regret_upper=0.0,
                selection_complete=True,
                policy=policy,
                stats=asdict(solver.stats),
                checkpoint=checkpoint,
                backend=backend,
                player_regret_lower=tuple(0.0 for _ in exact.value.value),
                player_regret_upper=tuple(0.0 for _ in exact.value.value),
            )
        else:
            adaptive: AdaptiveInitialResult = solver.solve_initial_adaptive()
            upper_value = tuple(
                value + regret
                for value, regret in zip(
                    adaptive.value.value, adaptive.regret_upper, strict=True
                )
            )
            status = adaptive.status
            if status == "CERTIFIED_PURE" and max(
                adaptive.regret_upper, default=0.0
            ) > quality_target:
                status = "APPROX_PURE"
            stats = asdict(solver.stats)
            stats["adaptive"] = asdict(solver.adaptive_stats)
            result = Q32SolveResult(
                status=status,
                value_lower=adaptive.value.value,
                value_upper=upper_value,
                success=adaptive.value.success,
                max_regret_lower=max(adaptive.regret_lower, default=0.0),
                max_regret_upper=max(adaptive.regret_upper, default=0.0),
                selection_complete=False,
                policy={
                    "initial": _policy_payload(adaptive.policy),
                    "feedback_states": len(
                        set(solver._policy_cache) | set(solver._policy_entry_cache)
                    ),
                },
                stats=stats,
                checkpoint=checkpoint,
                backend=backend,
                player_regret_lower=adaptive.regret_lower,
                player_regret_upper=adaptive.regret_upper,
            )
        if checkpoint:
            solver.save_checkpoint()
        return result
    except KeyboardInterrupt:
        solver.request_stop()
        if checkpoint:
            solver.save_checkpoint()
        stats = asdict(solver.stats)
        if isinstance(solver, AdaptiveQ3Solver):
            stats["adaptive"] = asdict(solver.adaptive_stats)
        return Q32SolveResult(
            status="SEARCH_STOPPED",
            value_lower=tuple(float("-inf") for _ in range(config.n_players)),
            value_upper=tuple(float("inf") for _ in range(config.n_players)),
            success=tuple(0.0 for _ in range(config.n_players)),
            max_regret_lower=0.0,
            max_regret_upper=float("inf"),
            selection_complete=False,
            policy={"error": "search interrupted"},
            stats=stats,
            checkpoint=checkpoint,
            backend=backend,
            player_regret_lower=tuple(0.0 for _ in range(config.n_players)),
            player_regret_upper=tuple(float("inf") for _ in range(config.n_players)),
        )
    except (ResourceLimitExceeded, NoPureEquilibrium, SearchCancelled) as exc:
        if checkpoint:
            solver.save_checkpoint()
        stats = asdict(solver.stats)
        if isinstance(solver, AdaptiveQ3Solver):
            stats["adaptive"] = asdict(solver.adaptive_stats)
        return Q32SolveResult(
            status="SEARCH_STOPPED",
            value_lower=tuple(float("-inf") for _ in range(config.n_players)),
            value_upper=tuple(float("inf") for _ in range(config.n_players)),
            success=tuple(0.0 for _ in range(config.n_players)),
            max_regret_lower=0.0,
            max_regret_upper=float("inf"),
            selection_complete=False,
            policy={"error": str(exc)},
            stats=stats,
            checkpoint=checkpoint,
            backend=backend,
            player_regret_lower=tuple(0.0 for _ in range(config.n_players)),
            player_regret_upper=tuple(float("inf") for _ in range(config.n_players)),
        )
    finally:
        solver.close()
