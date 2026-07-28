"""Adaptive restricted-game backend and public Q3.2 solve interface."""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from itertools import product
from math import prod
from pathlib import Path
from typing import Sequence

import numpy as np

from .action_enum import (
    ActionEnumerationLimitExceeded,
    enumerate_individual_actions,
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
    all_absorbed,
    initial_joint_state,
    terminal_state_value,
)
from .profile_enum import action_skeleton, iter_index_blocks, profiles_from_indices
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
)
from .stochastic_dp import (
    ExactQ3Solver,
    InitialSolveResult,
    ResourceLimitExceeded,
    SearchCancelled,
    SolverLimits,
)
from .transition import apply_initial_purchases, apply_joint_transition_batch


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
    grouped: dict[object, list[Action]] = {}
    for action in actions:
        grouped.setdefault(action_skeleton(action), []).append(action)
    selected: set[Action] = set(action for action in warm_actions if action in actions)
    for group in grouped.values():
        group.sort(key=lambda action: action.code)
        if len(group) <= per_skeleton:
            selected.update(group)
            continue
        selected.add(group[0])
        selected.add(group[-1])
        max_load = max(
            1,
            max(
                cfg.water_weight * action.buy_water
                + cfg.food_weight * action.buy_food
                for action in group
            ),
        )
        targets = _action_targets(max(0, per_skeleton - 2))
        for load_target, ratio_target in targets:
            def distance(action: Action) -> tuple[float, tuple[int, int, int, int]]:
                water_load = cfg.water_weight * action.buy_water
                food_load = cfg.food_weight * action.buy_food
                load = water_load + food_load
                ratio = water_load / load if load else 0.5
                metric = abs(load / max_load - load_target) + abs(
                    ratio - ratio_target
                )
                return metric, action.code

            selected.add(min(group, key=distance))
        if len([action for action in selected if action in group]) < per_skeleton:
            for index in np.linspace(0, len(group) - 1, per_skeleton, dtype=int):
                selected.add(group[int(index)])
                if len([action for action in selected if action in group]) >= per_skeleton:
                    break
    return tuple(sorted(selected, key=lambda action: action.code))


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
        self._policy_entry_cache: dict[
            tuple[int, JointState, Weather], tuple[PolicyEntry, ...]
        ] = {}
        self._stage_outcome_cache: dict[
            tuple[int, JointState, Weather], AdaptiveStageOutcome
        ] = {}

    def clear(self) -> None:
        super().clear()
        self.adaptive_stats = AdaptiveStats()
        self._policy_entry_cache.clear()
        self._stage_outcome_cache.clear()

    def _write_v2_checkpoint(self, directory: Path) -> None:
        super()._write_v2_checkpoint(directory)
        with (directory / "adaptive.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "options": self.options,
                    "quality_target": self.quality_target,
                    "stats": self.adaptive_stats,
                    "policy_entry_cache": self._policy_entry_cache,
                    "stage_outcome_cache": self._stage_outcome_cache,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load_checkpoint(self, path=None) -> None:
        super().load_checkpoint(path)
        source = Path(path) if path is not None else self.checkpoint_path
        if source is not None and source.is_dir() and (source / "adaptive.pkl").exists():
            with (source / "adaptive.pkl").open("rb") as handle:
                payload = pickle.load(handle)
            if payload.get("options") != self.options:
                raise ValueError("adaptive checkpoint options do not match")
            self.adaptive_stats = payload["stats"]
            self._policy_entry_cache = dict(payload["policy_entry_cache"])
            self._stage_outcome_cache = dict(payload["stage_outcome_cache"])

    def _compute_canonical(self, day: int, state: JointState) -> StateValue:
        if day == self.cfg.deadline or all_absorbed(state):
            return terminal_state_value(self.cfg, state)
        expected_value = np.zeros(self.cfg.n_players, dtype=np.float64)
        expected_success = np.zeros(self.cfg.n_players, dtype=np.float64)
        for weather in self.cfg.weather_order:
            probability = float(self.cfg.p_weather[weather])
            outcome = self._solve_adaptive_stage(day, state, weather)
            expected_value += probability * np.asarray(outcome.value)
            expected_success += probability * np.asarray(outcome.success)
            self._policy_entry_cache[(day, state, weather)] = outcome.policy
            self._stage_outcome_cache[(day, state, weather)] = outcome
            if outcome.actions is not None:
                self._policy_cache[(day, state, weather)] = outcome.actions
        return StateValue(tuple(expected_value), tuple(expected_success))

    def policy_entries_for(
        self, day: int, state: JointState, weather: Weather
    ) -> tuple[PolicyEntry, ...]:
        canonical, canonical_to_original = canonicalize_state(state)
        if (day, canonical) not in self._value_cache:
            self.solve_state(day, state)
        entries = self._policy_entry_cache[(day, canonical, weather)]
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

    def _initial_candidates(
        self, full_actions: Sequence[Sequence[Action]]
    ) -> tuple[tuple[Action, ...], ...]:
        return tuple(
            deterministic_action_candidates(
                self.cfg,
                actions,
                per_skeleton=min(
                    self.options.initial_candidates,
                    self.options.max_initial_candidates,
                ),
                warm_actions=self.options.warm_initial_actions,
            )
            for actions in full_actions
        )

    def _stage_candidates(
        self, full_actions: Sequence[Sequence[Action]]
    ) -> tuple[tuple[Action, ...], ...]:
        return tuple(
            deterministic_action_candidates(
                self.cfg,
                actions,
                per_skeleton=min(
                    self.options.village_candidates_per_skeleton,
                    self.options.max_village_candidates_per_skeleton,
                ),
            )
            for actions in full_actions
        )

    def _restricted_tensor(
        self,
        action_sets: Sequence[Sequence[Action]],
        evaluator,
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
        for indices in iter_index_blocks(
            shape, block_size=self.limits.profile_chunk_size
        ):
            self._check_cancelled()
            evaluation = evaluator(indices)
            coordinates = tuple(indices[:, player] for player in range(self.cfg.n_players))
            valid[coordinates] = evaluation.valid
            payoff[coordinates] = evaluation.payoff
            success[coordinates] = evaluation.success
        self.adaptive_stats.restricted_profiles += profiles
        return payoff, success, valid

    def _restricted_stage_equilibrium(
        self,
        day: int,
        state: JointState,
        weather: Weather,
        action_sets: Sequence[Sequence[Action]],
    ) -> PureEquilibrium | MixedEquilibrium:
        evaluator = self._stage_evaluator(day, state, weather, action_sets)
        payoff, success, valid = self._restricted_tensor(action_sets, evaluator)
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

    def _scan_pure_stage_deviations(
        self,
        day: int,
        state: JointState,
        weather: Weather,
        equilibrium: PureEquilibrium,
        full_actions: Sequence[Sequence[Action]],
        candidate_sets: Sequence[set[Action]],
    ) -> tuple[tuple[float, ...], tuple[float, ...], list[list[Action]]]:
        lower: list[float] = []
        upper: list[float] = []
        additions: list[list[Action]] = []
        for player in range(self.cfg.n_players):
            current = equilibrium.value[player]
            best_exact = current
            best_upper = current
            profitable: list[tuple[float, Action]] = []
            actions = full_actions[player]
            self.adaptive_stats.full_deviation_actions += len(actions)
            for start in range(0, len(actions), self.limits.profile_chunk_size):
                block = actions[start : start + self.limits.profile_chunk_size]
                profiles = []
                for action in block:
                    profile = list(equilibrium.actions)
                    profile[player] = action
                    profiles.append(tuple(profile))
                batch = apply_joint_transition_batch(self.cfg, state, profiles, weather)
                bounds = self._upper_bounds_from_successors(
                    day + 1, batch.valid, batch.successors, player
                )
                keep = batch.valid & (
                    bounds >= current - self.equilibrium_atol - self.bound_pruning_slack
                )
                pruned = batch.valid & ~keep
                if np.any(pruned):
                    best_upper = max(best_upper, float(np.max(bounds[pruned])))
                self.adaptive_stats.deviation_actions_pruned += int(
                    np.sum(pruned)
                )
                evaluation = self._evaluate_successors(
                    day + 1, keep, batch.successors
                )
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
            additions.append(
                [
                    action
                    for _, action in profitable[: self.options.additions_per_player]
                ]
            )
            lower.append(max(0.0, best_exact - current))
            upper.append(max(0.0, max(best_exact, best_upper) - current))
        return tuple(lower), tuple(upper), additions

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
        full_actions = self._full_action_sets(state, weather)
        if not any(
            player.position in self.cfg.villages
            for player in state
            if player.status.value == 0
        ):
            candidates = tuple(tuple(actions) for actions in full_actions)
        else:
            candidates = self._stage_candidates(full_actions)
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
            lower, upper, additions = self._scan_pure_stage_deviations(
                day,
                state,
                weather,
                equilibrium,
                full_actions,
                candidate_sets,
            )
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
        payoff, success, valid = self._restricted_tensor(
            action_sets, self._initial_evaluator(action_sets)
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
        for player in range(self.cfg.n_players):
            current = equilibrium.value[player]
            best_exact = current
            best_upper = current
            profitable: list[tuple[float, Action]] = []
            actions = full_actions[player]
            self.adaptive_stats.full_deviation_actions += len(actions)
            for start in range(0, len(actions), self.limits.profile_chunk_size):
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
            additions.append(
                [action for _, action in profitable[: self.options.additions_per_player]]
            )
            lower.append(max(0.0, best_exact - current))
            upper.append(max(0.0, max(best_exact, best_upper) - current))
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
    budget_manager: BudgetManager | None = None,
    checkpoint: str | None = None,
    resume: bool = False,
    workers: int = 1,
) -> Q32SolveResult:
    if backend not in {"exact", "adaptive"}:
        raise ValueError("backend must be 'exact' or 'adaptive'")
    solver_class = ExactQ3Solver if backend == "exact" else AdaptiveQ3Solver
    kwargs = dict(
        limits=limits,
        workers=workers,
        checkpoint_path=checkpoint,
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
                    "feedback_states": len(solver._policy_entry_cache),
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
