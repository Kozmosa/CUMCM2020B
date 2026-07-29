"""Submission-gated policy-response solver for the Q3.2 heuristic backend.

The training game grows from a small interpretable seed library.  At every
round it solves the accumulated empirical normal-form game, searches all short
simple routes for profitable policy responses, and appends the strongest
responses.  A disjoint weather holdout then audits the selected profile.

This remains an empirical result over a broad parameterized policy class.  It
does not claim a full Markov-perfect-equilibrium certificate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations_with_replacement, permutations
from math import sqrt
from statistics import NormalDist
from time import perf_counter
from typing import Sequence

import numpy as np

from .data import Q3Config, Weather
from .heuristic import (
    HeuristicOptions,
    HeuristicPolicy,
    _fit_inventory,
    _minimum_regret_profile,
    _player_mapping,
    _policy_for_route,
    _policy_payload,
    _replay_payload,
    _select_pure_profile,
    _simulate_profile,
    _validate_policy,
    _weather_scenarios,
    generate_heuristic_policies,
)
from .reports import Q32SolveResult
from .runtime import BudgetExceeded, BudgetManager, peak_rss_bytes
from .stage_game import independent_mixture_values, minimize_nashconv, pure_nash_indices


@dataclass(frozen=True, slots=True)
class _TrainingSelection:
    profile: tuple[int, ...]
    probabilities: tuple[np.ndarray, ...]
    kind: str
    pure_count: int
    value: tuple[float, ...]
    success: tuple[float, ...]
    regret: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _ResponseEvaluation:
    policy: HeuristicPolicy
    mean_gain: float
    mean_value: float
    success: float


@dataclass(frozen=True, slots=True)
class _AuditResult:
    value_lower: tuple[float, ...]
    value_mean: tuple[float, ...]
    value_upper: tuple[float, ...]
    success: tuple[float, ...]
    success_lower: tuple[float, ...]
    regret_lower: tuple[float, ...]
    regret_mean: tuple[float, ...]
    regret_upper: tuple[float, ...]
    best_deviations: tuple[dict[str, object], ...]
    profiles: int
    day_steps: int


class _TrainingGame:
    """Canonical-profile sample cache for an append-only symmetric library."""

    def __init__(
        self,
        cfg: Q3Config,
        scenarios: Sequence[Sequence[Weather]],
        budget_manager: BudgetManager | None,
    ) -> None:
        self.cfg = cfg
        self.scenarios = scenarios
        self.budget_manager = budget_manager
        self.samples: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}
        self.day_steps = 0
        self.profile_evaluations = 0

    def ensure(self, policies: Sequence[HeuristicPolicy]) -> None:
        count = len(policies)
        episodes = len(self.scenarios)
        for index in combinations_with_replacement(range(count), self.cfg.n_players):
            if index in self.samples:
                continue
            if self.budget_manager is not None:
                self.budget_manager.check(profiles=self.profile_evaluations)
            profile = tuple(policies[action] for action in index)
            payoff = np.empty((episodes, self.cfg.n_players), dtype=np.float64)
            success = np.empty((episodes, self.cfg.n_players), dtype=np.uint8)
            for episode, weather in enumerate(self.scenarios):
                result = _simulate_profile(self.cfg, profile, weather)
                payoff[episode] = result.payoff
                success[episode] = result.success
                self.day_steps += result.day_steps
            self.samples[index] = payoff, success
            self.profile_evaluations += 1

    def profile_samples(
        self, profile: tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        canonical = tuple(sorted(profile))
        payoff, success = self.samples[canonical]
        mapping = _player_mapping(canonical, profile)
        return payoff[:, mapping], success[:, mapping]

    def mean_tensors(
        self, policies: Sequence[HeuristicPolicy]
    ) -> tuple[np.ndarray, np.ndarray]:
        count = len(policies)
        shape = (count,) * self.cfg.n_players
        payoff = np.empty(shape + (self.cfg.n_players,), dtype=np.float64)
        success = np.empty_like(payoff)
        for canonical, (profile_payoff, profile_success) in self.samples.items():
            mean_payoff = np.mean(profile_payoff, axis=0)
            mean_success = np.mean(profile_success, axis=0)
            for target in set(permutations(canonical)):
                mapping = _player_mapping(canonical, target)
                payoff[target] = mean_payoff[list(mapping)]
                success[target] = mean_success[list(mapping)]
        return payoff, success


def enumerate_submission_routes(
    cfg: Q3Config, max_moves: int
) -> tuple[tuple[int, ...], ...]:
    """Enumerate all loop-free start/end routes up to ``max_moves``."""
    distances = _distances_from(cfg, cfg.end)
    routes: list[tuple[int, ...]] = []

    def visit(node: int, path: tuple[int, ...]) -> None:
        moves = len(path) - 1
        if moves + distances.get(node, max_moves + 1) > max_moves:
            return
        if node == cfg.end:
            routes.append(path)
            return
        for neighbor in sorted(cfg.adj[node]):
            if neighbor not in path:
                visit(neighbor, path + (neighbor,))

    visit(cfg.start, (cfg.start,))
    return tuple(sorted(routes, key=lambda route: (len(route), route)))


def _distances_from(cfg: Q3Config, target: int) -> dict[int, int]:
    distances = {target: 0}
    frontier = [target]
    while frontier:
        node = frontier.pop(0)
        for neighbor in cfg.adj[node]:
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                frontier.append(neighbor)
    return distances


def _policy_key(policy: HeuristicPolicy) -> tuple[object, ...]:
    return (
        policy.route,
        policy.initial_water,
        policy.initial_food,
        policy.mine_days,
        policy.village_water_target,
        policy.village_food_target,
        policy.yield_when_crowded,
        policy.mine_only_alone,
    )


def _route_policy(
    cfg: Q3Config,
    route: tuple[int, ...],
    route_index: int,
    mine_days: int,
    safety_factor: float,
    *,
    name_suffix: str = "",
    yield_when_crowded: bool = False,
    mine_only_alone: bool = False,
) -> HeuristicPolicy:
    policy = _policy_for_route(
        cfg,
        family=f"route{route_index:04d}",
        route=route,
        route_index=0,
        mine_days=mine_days,
        safety_factor=safety_factor,
    )
    return replace(
        policy,
        name=(
            f"response-r{route_index:04d}-m{mine_days}-s{safety_factor:g}"
            f"{name_suffix}"
            f"-y{int(yield_when_crowded)}-a{int(mine_only_alone)}"
        ),
        yield_when_crowded=yield_when_crowded,
        mine_only_alone=mine_only_alone,
    )


def _anchored_policy(
    cfg: Q3Config,
    route: tuple[int, ...],
    route_index: int,
    mine_days: int,
    anchor: tuple[int, int],
    safety_factor: float,
    *,
    yield_when_crowded: bool = False,
    mine_only_alone: bool = False,
) -> HeuristicPolicy:
    policy = _route_policy(
        cfg,
        route,
        route_index,
        mine_days,
        safety_factor,
        name_suffix=f"-w{anchor[0]}-f{anchor[1]}",
        yield_when_crowded=yield_when_crowded,
        mine_only_alone=mine_only_alone,
    )
    water, food = _fit_inventory(cfg, *anchor, price_factor=1)
    return replace(policy, initial_water=water, initial_food=food)


def _deduplicate_policies(
    policies: Sequence[HeuristicPolicy],
) -> tuple[HeuristicPolicy, ...]:
    output: list[HeuristicPolicy] = []
    seen: set[tuple[object, ...]] = set()
    for policy in policies:
        key = _policy_key(policy)
        if key not in seen:
            seen.add(key)
            output.append(policy)
    return tuple(output)


def _select_training_equilibrium(
    payoff: np.ndarray,
    success: np.ndarray,
    options: HeuristicOptions,
) -> _TrainingSelection:
    valid = np.ones(payoff.shape[:-1], dtype=bool)
    pure_rows = pure_nash_indices(payoff, valid, atol=options.pure_tolerance)
    if len(pure_rows):
        profile = _select_pure_profile(
            [tuple(int(value) for value in row) for row in pure_rows],
            payoff,
            success,
        )
        probabilities = tuple(
            np.eye(payoff.shape[player], dtype=np.float64)[profile[player]]
            for player in range(payoff.shape[-1])
        )
        kind = "pure"
    else:
        profile = _minimum_regret_profile(payoff, success)
        probabilities = tuple(
            np.eye(payoff.shape[player], dtype=np.float64)[profile[player]]
            for player in range(payoff.shape[-1])
        )
        kind = "minimum-regret-pure"
    values, successes, regrets = independent_mixture_values(
        payoff, probabilities, valid=valid, success=success
    )
    return _TrainingSelection(
        profile=profile,
        probabilities=probabilities,
        kind=kind,
        pure_count=len(pure_rows),
        value=values,
        success=successes,
        regret=regrets,
    )


def _scenario_window(
    scenarios: Sequence[Sequence[Weather]], count: int, offset: int
) -> tuple[Sequence[Weather], ...]:
    count = min(count, len(scenarios))
    start = (offset * count) % len(scenarios)
    return tuple(scenarios[(start + index) % len(scenarios)] for index in range(count))


def _screen_policies(
    cfg: Q3Config,
    current_profile: Sequence[HeuristicPolicy],
    player: int,
    candidates: Sequence[HeuristicPolicy],
    scenarios: Sequence[Sequence[Weather]],
    baseline: np.ndarray,
    budget_manager: BudgetManager | None,
    profile_counter: list[int],
    day_counter: list[int],
) -> tuple[_ResponseEvaluation, ...]:
    evaluations: list[_ResponseEvaluation] = []
    for candidate in candidates:
        if budget_manager is not None:
            budget_manager.check(profiles=profile_counter[0])
        profile = list(current_profile)
        profile[player] = candidate
        payoff = np.empty(len(scenarios), dtype=np.float64)
        success = np.empty(len(scenarios), dtype=np.float64)
        for episode, weather in enumerate(scenarios):
            result = _simulate_profile(cfg, profile, weather)
            payoff[episode] = result.payoff[player]
            success[episode] = result.success[player]
            day_counter[0] += result.day_steps
        profile_counter[0] += 1
        evaluations.append(
            _ResponseEvaluation(
                candidate,
                float(np.mean(payoff - baseline)),
                float(np.mean(payoff)),
                float(np.mean(success)),
            )
        )
    return tuple(
        sorted(
            evaluations,
            key=lambda item: (
                item.mean_gain,
                item.success,
                item.mean_value,
                item.policy.name,
            ),
            reverse=True,
        )
    )


def _response_candidates_for_player(
    cfg: Q3Config,
    routes: Sequence[tuple[int, ...]],
    policies: Sequence[HeuristicPolicy],
    selection: _TrainingSelection,
    player: int,
    training: _TrainingGame,
    options: HeuristicOptions,
    round_index: int,
    profile_counter: list[int],
    day_counter: list[int],
) -> tuple[_ResponseEvaluation, ...]:
    current_profile = tuple(policies[index] for index in selection.profile)
    scenarios = _scenario_window(
        training.scenarios, options.response_screening_episodes, round_index
    )
    training_payoff, _ = training.profile_samples(selection.profile)
    window_size = len(scenarios)
    start = (round_index * window_size) % len(training.scenarios)
    indices = [
        (start + index) % len(training.scenarios) for index in range(window_size)
    ]
    baseline = training_payoff[indices, player]

    robust_safety = max(options.response_safety_factors)
    route_bases: list[HeuristicPolicy] = []
    for route_index, route in enumerate(routes):
        route_bases.append(
            _route_policy(cfg, route, route_index, 0, robust_safety)
        )
        route_bases.append(
            _route_policy(
                cfg,
                route,
                route_index,
                0,
                robust_safety,
                yield_when_crowded=True,
            )
        )
        for anchor in options.inventory_anchors:
            route_bases.append(
                _anchored_policy(
                    cfg, route, route_index, 0, anchor, robust_safety
                )
            )
        if any(node in cfg.mines for node in route):
            route_bases.append(
                _route_policy(cfg, route, route_index, 1, robust_safety)
            )
            for anchor in options.inventory_anchors:
                route_bases.append(
                    _anchored_policy(
                        cfg, route, route_index, 1, anchor, robust_safety
                    )
                )
    base_evaluations = _screen_policies(
        cfg,
        current_profile,
        player,
        _deduplicate_policies(route_bases),
        scenarios,
        baseline,
        training.budget_manager,
        profile_counter,
        day_counter,
    )
    route_lookup = {route: index for index, route in enumerate(routes)}
    selected_routes: list[int] = []
    for evaluation in base_evaluations:
        route_index = route_lookup[evaluation.policy.route]
        if route_index not in selected_routes:
            selected_routes.append(route_index)
        if len(selected_routes) >= options.response_route_candidates:
            break

    refined: list[HeuristicPolicy] = []
    for route_index in selected_routes:
        route = routes[route_index]
        mine_days_values = (
            options.response_mine_days
            if any(node in cfg.mines for node in route)
            else (0,)
        )
        for mine_days in mine_days_values:
            interaction_modes = [(False, False), (True, False)]
            if mine_days > 0:
                interaction_modes.extend([(False, True), (True, True)])
            for yield_when_crowded, mine_only_alone in interaction_modes:
                for safety_factor in options.response_safety_factors:
                    refined.append(
                        _route_policy(
                            cfg,
                            route,
                            route_index,
                            mine_days,
                            safety_factor,
                            yield_when_crowded=yield_when_crowded,
                            mine_only_alone=mine_only_alone,
                        )
                    )
                for anchor in options.inventory_anchors:
                    refined.append(
                        _anchored_policy(
                            cfg,
                            route,
                            route_index,
                            mine_days,
                            anchor,
                            robust_safety,
                            yield_when_crowded=yield_when_crowded,
                            mine_only_alone=mine_only_alone,
                        )
                    )
    refined_evaluations = _screen_policies(
        cfg,
        current_profile,
        player,
        _deduplicate_policies(refined),
        scenarios,
        baseline,
        training.budget_manager,
        profile_counter,
        day_counter,
    )
    return refined_evaluations[: options.response_audit_candidates]


def _mixed_training_selection(
    payoff: np.ndarray,
    success: np.ndarray,
    options: HeuristicOptions,
) -> _TrainingSelection:
    valid = np.ones(payoff.shape[:-1], dtype=bool)
    finite = payoff[np.isfinite(payoff)]
    scale = max(1.0, float(np.max(np.abs(finite))) if finite.size else 1.0)
    mixed = minimize_nashconv(
        payoff / scale,
        valid=valid,
        success=success,
        seed=options.seed,
        starts=options.mixed_starts,
        max_iterations=options.mixed_max_iterations,
    )
    probabilities = tuple(np.asarray(block) for block in mixed.probabilities)
    values, successes, regrets = independent_mixture_values(
        payoff, probabilities, valid=valid, success=success
    )
    profile = tuple(int(np.argmax(block)) for block in probabilities)
    return _TrainingSelection(
        profile=profile,
        probabilities=probabilities,
        kind="mixed",
        pure_count=0,
        value=values,
        success=successes,
        regret=regrets,
    )


def _normal_interval(
    samples: np.ndarray, confidence: float
) -> tuple[float, float, float]:
    mean = float(np.mean(samples))
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    halfwidth = z * float(np.std(samples, ddof=1)) / sqrt(len(samples))
    return mean - halfwidth, mean, mean + halfwidth


def _wilson_lower(successes: int, total: int, confidence: float) -> float:
    z = NormalDist().inv_cdf(confidence)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    radius = z * sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return max(0.0, (center - radius) / denominator)


def _audit_profile(
    cfg: Q3Config,
    policies: Sequence[HeuristicPolicy],
    selection: _TrainingSelection,
    response_candidates: Sequence[Sequence[HeuristicPolicy]],
    scenarios: Sequence[Sequence[Weather]],
    confidence: float,
    budget_manager: BudgetManager | None,
) -> _AuditResult:
    profile = tuple(policies[index] for index in selection.profile)
    episodes = len(scenarios)
    baseline_payoff = np.empty((episodes, cfg.n_players), dtype=np.float64)
    baseline_success = np.empty((episodes, cfg.n_players), dtype=np.uint8)
    day_steps = 0
    for episode, weather in enumerate(scenarios):
        result = _simulate_profile(cfg, profile, weather)
        baseline_payoff[episode] = result.payoff
        baseline_success[episode] = result.success
        day_steps += result.day_steps

    audit_sets: list[tuple[HeuristicPolicy, ...]] = []
    for player in range(cfg.n_players):
        candidates = list(policies)
        candidates.extend(response_candidates[player])
        audit_sets.append(_deduplicate_policies(candidates))
    tests = max(1, sum(len(candidates) for candidates in audit_sets))
    alpha = 1.0 - confidence
    z = NormalDist().inv_cdf(1.0 - alpha / tests)
    regret_lower: list[float] = []
    regret_mean: list[float] = []
    regret_upper: list[float] = []
    best_deviations: list[dict[str, object]] = []
    profiles = 0
    for player, candidates in enumerate(audit_sets):
        best_lower = 0.0
        best_mean = 0.0
        best_upper = 0.0
        best_payload: dict[str, object] = {
            "player": player + 1,
            "policy": profile[player].name,
            "mean_gain": 0.0,
            "lower": 0.0,
            "upper": 0.0,
        }
        for candidate in candidates:
            if _policy_key(candidate) == _policy_key(profile[player]):
                continue
            if budget_manager is not None:
                budget_manager.check(profiles=profiles)
            deviating_profile = list(profile)
            deviating_profile[player] = candidate
            difference = np.empty(episodes, dtype=np.float64)
            candidate_success = 0
            for episode, weather in enumerate(scenarios):
                result = _simulate_profile(cfg, deviating_profile, weather)
                difference[episode] = (
                    result.payoff[player] - baseline_payoff[episode, player]
                )
                candidate_success += int(result.success[player] > 0.5)
                day_steps += result.day_steps
            profiles += 1
            mean = float(np.mean(difference))
            halfwidth = z * float(np.std(difference, ddof=1)) / sqrt(episodes)
            lower = mean - halfwidth
            upper = mean + halfwidth
            best_lower = max(best_lower, lower)
            best_mean = max(best_mean, mean)
            if upper > best_upper:
                best_upper = upper
                best_payload = {
                    "player": player + 1,
                    "policy": candidate.name,
                    "mean_gain": mean,
                    "lower": lower,
                    "upper": upper,
                    "success": candidate_success / episodes,
                }
        regret_lower.append(max(0.0, best_lower))
        regret_mean.append(max(0.0, best_mean))
        regret_upper.append(max(0.0, best_upper))
        best_deviations.append(best_payload)

    value_intervals = tuple(
        _normal_interval(baseline_payoff[:, player], confidence)
        for player in range(cfg.n_players)
    )
    successes = tuple(
        float(np.mean(baseline_success[:, player]))
        for player in range(cfg.n_players)
    )
    success_lower = tuple(
        _wilson_lower(
            int(np.sum(baseline_success[:, player])), episodes, confidence
        )
        for player in range(cfg.n_players)
    )
    return _AuditResult(
        value_lower=tuple(interval[0] for interval in value_intervals),
        value_mean=tuple(interval[1] for interval in value_intervals),
        value_upper=tuple(interval[2] for interval in value_intervals),
        success=successes,
        success_lower=success_lower,
        regret_lower=tuple(regret_lower),
        regret_mean=tuple(regret_mean),
        regret_upper=tuple(regret_upper),
        best_deviations=tuple(best_deviations),
        profiles=profiles + 1,
        day_steps=day_steps,
    )


def _player_policy_payload(
    policies: Sequence[HeuristicPolicy], selection: _TrainingSelection
) -> tuple[dict[str, object], ...]:
    output = []
    for player, probabilities in enumerate(selection.probabilities):
        support = tuple(
            {
                "policy": policies[index].name,
                "probability": float(probability),
            }
            for index, probability in enumerate(probabilities)
            if probability > 1e-9
        )
        output.append({"player": player + 1, "support": support})
    return tuple(output)


def _aggregate_audits(audits: Sequence[_AuditResult]) -> _AuditResult:
    players = len(audits[0].success)
    best_deviations = []
    for player in range(players):
        best_deviations.append(
            max(
                (audit.best_deviations[player] for audit in audits),
                key=lambda payload: float(payload.get("upper", 0.0)),
            )
        )
    return _AuditResult(
        value_lower=tuple(
            min(audit.value_lower[player] for audit in audits)
            for player in range(players)
        ),
        value_mean=tuple(
            float(np.mean([audit.value_mean[player] for audit in audits]))
            for player in range(players)
        ),
        value_upper=tuple(
            max(audit.value_upper[player] for audit in audits)
            for player in range(players)
        ),
        success=tuple(
            float(np.mean([audit.success[player] for audit in audits]))
            for player in range(players)
        ),
        success_lower=tuple(
            min(audit.success_lower[player] for audit in audits)
            for player in range(players)
        ),
        regret_lower=tuple(
            max(audit.regret_lower[player] for audit in audits)
            for player in range(players)
        ),
        regret_mean=tuple(
            max(audit.regret_mean[player] for audit in audits)
            for player in range(players)
        ),
        regret_upper=tuple(
            max(audit.regret_upper[player] for audit in audits)
            for player in range(players)
        ),
        best_deviations=tuple(best_deviations),
        profiles=sum(audit.profiles for audit in audits),
        day_steps=sum(audit.day_steps for audit in audits),
    )


def _stopped_result(
    config: Q3Config,
    options: HeuristicOptions,
    started: float,
    error: Exception,
    policies: Sequence[HeuristicPolicy],
) -> Q32SolveResult:
    return Q32SolveResult(
        status="SEARCH_STOPPED",
        value_lower=tuple(float("-inf") for _ in range(config.n_players)),
        value_upper=tuple(float("inf") for _ in range(config.n_players)),
        success=tuple(0.0 for _ in range(config.n_players)),
        max_regret_lower=0.0,
        max_regret_upper=float("inf"),
        selection_complete=False,
        policy={
            "error": str(error),
            "scope": "broad_parameterized_policy_response_search",
        },
        stats={
            "policy_count": len(policies),
            "training_episodes": options.episodes,
            "audit_episodes": options.audit_episodes,
            "audit_replicates": options.audit_replicates,
            "stability_episodes": options.stability_episodes,
            "stability_replicates": options.stability_replicates,
            "elapsed_seconds": perf_counter() - started,
            "peak_rss_bytes": peak_rss_bytes(),
        },
        checkpoint=None,
        backend="heuristic",
        player_regret_lower=tuple(0.0 for _ in range(config.n_players)),
        player_regret_upper=tuple(float("inf") for _ in range(config.n_players)),
    )


def solve_submission_heuristic(
    config: Q3Config,
    *,
    options: HeuristicOptions,
    budget_manager: BudgetManager | None = None,
) -> Q32SolveResult:
    """Train and independently audit a submission-scale empirical equilibrium."""
    started = perf_counter()
    policies = list(generate_heuristic_policies(config, options))
    try:
        for policy in policies:
            _validate_policy(config, policy)
        if budget_manager is not None:
            budget_manager.check()
        training_scenarios = _weather_scenarios(config, options)
        training = _TrainingGame(config, training_scenarios, budget_manager)
        routes = enumerate_submission_routes(config, options.route_max_moves)
        if not routes:
            raise ValueError("submission route universe is empty")
        response_profile_counter = [0]
        response_day_counter = [0]
        response_history: list[dict[str, object]] = []
        audit_candidates: list[list[_ResponseEvaluation]] = [
            [] for _ in range(config.n_players)
        ]
        stable_rounds = 0
        response_complete = False
        policy_cap_reached = False
        selection: _TrainingSelection | None = None
        mean_payoff: np.ndarray | None = None
        mean_success: np.ndarray | None = None

        for round_index in range(options.response_rounds):
            training.ensure(policies)
            mean_payoff, mean_success = training.mean_tensors(policies)
            selection = _select_training_equilibrium(
                mean_payoff, mean_success, options
            )
            round_responses: list[tuple[_ResponseEvaluation, ...]] = []
            for player in range(config.n_players):
                responses = _response_candidates_for_player(
                    config,
                    routes,
                    policies,
                    selection,
                    player,
                    training,
                    options,
                    round_index,
                    response_profile_counter,
                    response_day_counter,
                )
                round_responses.append(responses)
                audit_candidates[player].extend(responses)

            existing = {_policy_key(policy) for policy in policies}
            additions: list[_ResponseEvaluation] = []
            for responses in round_responses:
                for response in responses:
                    key = _policy_key(response.policy)
                    if (
                        response.mean_gain > options.response_training_regret
                        and key not in existing
                    ):
                        additions.append(response)
                        existing.add(key)
                        break
            additions.sort(
                key=lambda response: (
                    response.mean_gain,
                    response.success,
                    response.policy.name,
                ),
                reverse=True,
            )
            room = options.max_policies - len(policies)
            selected_additions = additions[
                : min(room, options.response_additions_per_round)
            ]
            response_history.append(
                {
                    "round": round_index + 1,
                    "policy_count": len(policies),
                    "equilibrium": selection.kind,
                    "pure_equilibria": selection.pure_count,
                    "training_regret": selection.regret,
                    "best_response_gain": tuple(
                        responses[0].mean_gain if responses else 0.0
                        for responses in round_responses
                    ),
                    "added": tuple(
                        response.policy.name for response in selected_additions
                    ),
                }
            )
            if selected_additions:
                policies.extend(response.policy for response in selected_additions)
                stable_rounds = 0
                continue
            if additions and room <= 0:
                policy_cap_reached = True
                break
            stable_rounds += 1
            if stable_rounds >= options.response_stable_rounds:
                response_complete = True
                break

        training.ensure(policies)
        mean_payoff, mean_success = training.mean_tensors(policies)
        selection = _select_training_equilibrium(mean_payoff, mean_success, options)
        if selection.pure_count == 0 and options.equilibrium == "pure-mixed":
            selection = _mixed_training_selection(mean_payoff, mean_success, options)

        if selection.kind == "mixed":
            stats = {
                "route_universe": len(routes),
                "policy_count": len(policies),
                "training_episodes": options.episodes,
                "audit_episodes": options.audit_episodes,
                "stability_episodes": options.stability_episodes,
                "stability_replicates": options.stability_replicates,
                "response_complete": False,
                "audit_skipped": "submission gate currently requires a pure profile",
                "response_history": tuple(response_history),
                "elapsed_seconds": perf_counter() - started,
                "peak_rss_bytes": peak_rss_bytes(),
                "regret_scope": "broad_parameterized_policy_response_search",
                "full_action_regret_certified": False,
            }
            return Q32SolveResult(
                status="EMPIRICAL_EQ_NOT_READY",
                value_lower=selection.value,
                value_upper=selection.value,
                success=selection.success,
                max_regret_lower=max(selection.regret, default=0.0),
                max_regret_upper=float("inf"),
                selection_complete=False,
                policy={
                    "scope": "broad_parameterized_policy_response_search",
                    "equilibrium": "mixed",
                    "players": _player_policy_payload(policies, selection),
                    "library": tuple(_policy_payload(policy) for policy in policies),
                },
                stats=stats,
                checkpoint=None,
                backend="heuristic",
                player_regret_lower=selection.regret,
                player_regret_upper=tuple(
                    float("inf") for _ in range(config.n_players)
                ),
            )

        # Search once more against the final selected profile.  Responses found
        # against intermediate profiles remain useful audit candidates, but
        # they cannot substitute for a targeted search after the last library
        # expansion and equilibrium re-selection.
        for player in range(config.n_players):
            audit_candidates[player].extend(
                _response_candidates_for_player(
                    config,
                    routes,
                    policies,
                    selection,
                    player,
                    training,
                    options,
                    options.response_rounds,
                    response_profile_counter,
                    response_day_counter,
                )
            )

        selected_role_keys = sorted(
            repr(_policy_key(policies[index])) for index in selection.profile
        )
        stability_results: list[dict[str, object]] = []
        stability_complete = True
        stability_day_steps = 0
        stability_profiles = 0
        for replicate in range(options.stability_replicates):
            stability_options = replace(
                options,
                episodes=options.stability_episodes,
                seed=options.seed + 10_000 + replicate,
            )
            stability_scenarios = _weather_scenarios(config, stability_options)
            stability_game = _TrainingGame(
                config, stability_scenarios, budget_manager
            )
            stability_game.ensure(policies)
            stability_payoff, stability_success = stability_game.mean_tensors(
                policies
            )
            stability_selection = _select_training_equilibrium(
                stability_payoff, stability_success, options
            )
            role_keys = sorted(
                repr(_policy_key(policies[index]))
                for index in stability_selection.profile
            )
            matches = (
                stability_selection.kind == "pure"
                and role_keys == selected_role_keys
            )
            stability_complete &= matches
            stability_day_steps += stability_game.day_steps
            stability_profiles += stability_game.profile_evaluations
            stability_results.append(
                {
                    "seed": stability_options.seed,
                    "equilibrium": stability_selection.kind,
                    "pure_equilibria": stability_selection.pure_count,
                    "representative_profile": tuple(
                        policies[index].name
                        for index in stability_selection.profile
                    ),
                    "role_structure_matches": matches,
                    "value": stability_selection.value,
                    "success": stability_selection.success,
                    "regret": stability_selection.regret,
                }
            )

        final_audit_candidates_list: list[tuple[HeuristicPolicy, ...]] = []
        for evaluations in audit_candidates:
            ordered = sorted(
                evaluations,
                key=lambda evaluation: (
                    evaluation.mean_gain,
                    evaluation.success,
                    evaluation.policy.name,
                ),
                reverse=True,
            )
            final_audit_candidates_list.append(
                _deduplicate_policies(
                    tuple(evaluation.policy for evaluation in ordered)
                )[: options.response_audit_candidates]
            )
        final_audit_candidates = tuple(final_audit_candidates_list)
        audit_runs: list[_AuditResult] = []
        first_audit_scenarios: tuple[tuple[Weather, ...], ...] | None = None
        for replicate in range(options.audit_replicates):
            audit_options = replace(
                options,
                episodes=options.audit_episodes,
                seed=options.seed + replicate + 1,
            )
            audit_scenarios = _weather_scenarios(config, audit_options)
            if first_audit_scenarios is None:
                first_audit_scenarios = audit_scenarios
            audit_runs.append(
                _audit_profile(
                    config,
                    policies,
                    selection,
                    final_audit_candidates,
                    audit_scenarios,
                    options.confidence,
                    budget_manager,
                )
            )
        audit = _aggregate_audits(audit_runs)
        ready = (
            selection.kind == "pure"
            and response_complete
            and not policy_cap_reached
            and stability_complete
            and max(audit.regret_mean, default=0.0)
            <= options.submission_mean_regret
            and max(audit.regret_upper, default=0.0)
            <= options.submission_upper_regret
            and min(audit.success_lower, default=0.0)
            >= options.submission_success_lower
        )
        status = (
            "SUBMISSION_READY_EMPIRICAL_EQ"
            if ready
            else "EMPIRICAL_EQ_NOT_READY"
        )
        representative_profile = tuple(
            policies[index] for index in selection.profile
        )
        assert first_audit_scenarios is not None
        representative = _simulate_profile(
            config,
            representative_profile,
            first_audit_scenarios[0],
            capture=True,
        )
        stats: dict[str, object] = {
            "route_universe": len(routes),
            "route_max_moves": options.route_max_moves,
            "policy_count": len(policies),
            "initial_policy_count": min(
                options.initial_policies, options.max_policies
            ),
            "training_episodes": options.episodes,
            "audit_episodes": options.audit_episodes,
            "audit_replicates": options.audit_replicates,
            "stability_episodes": options.stability_episodes,
            "stability_replicates": options.stability_replicates,
            "confidence": options.confidence,
            "training_canonical_profiles": training.profile_evaluations,
            "training_joint_profiles": int(len(policies) ** config.n_players),
            "training_day_steps": training.day_steps,
            "response_profiles_screened": response_profile_counter[0],
            "response_day_steps": response_day_counter[0],
            "response_complete": response_complete,
            "response_stable_rounds": stable_rounds,
            "policy_cap_reached": policy_cap_reached,
            "response_history": tuple(response_history),
            "training_equilibrium": selection.kind,
            "training_pure_equilibria": selection.pure_count,
            "training_value_mean": selection.value,
            "training_success_mean": selection.success,
            "training_policy_regret_mean": selection.regret,
            "stability_complete": stability_complete,
            "stability_profiles": stability_profiles,
            "stability_day_steps": stability_day_steps,
            "stability_results": tuple(stability_results),
            "audit_profiles": audit.profiles,
            "audit_day_steps": audit.day_steps,
            "audit_value_mean": audit.value_mean,
            "audit_success_lower": audit.success_lower,
            "audit_policy_regret_mean": audit.regret_mean,
            "audit_best_deviations": audit.best_deviations,
            "audit_replicate_results": tuple(
                {
                    "value_mean": result.value_mean,
                    "success": result.success,
                    "success_lower": result.success_lower,
                    "regret_mean": result.regret_mean,
                    "regret_upper": result.regret_upper,
                }
                for result in audit_runs
            ),
            "submission_mean_regret_target": options.submission_mean_regret,
            "submission_upper_regret_target": options.submission_upper_regret,
            "submission_success_lower_target": options.submission_success_lower,
            "submission_quality_met": ready,
            "regret_scope": "broad_parameterized_policy_response_search",
            "value_bounds": "independent_holdout_normal_confidence_interval",
            "success_bounds": "independent_holdout_one_sided_wilson_lower",
            "exact_joint_transition_rules": True,
            "full_action_regret_certified": False,
            "elapsed_seconds": perf_counter() - started,
            "peak_rss_bytes": peak_rss_bytes(),
        }
        return Q32SolveResult(
            status=status,
            value_lower=audit.value_lower,
            value_upper=audit.value_upper,
            success=audit.success,
            max_regret_lower=max(audit.regret_lower, default=0.0),
            max_regret_upper=max(audit.regret_upper, default=0.0),
            selection_complete=False,
            policy={
                "scope": "broad_parameterized_policy_response_search",
                "equilibrium": selection.kind,
                "selection_complete_within_policy_class": response_complete,
                "players": _player_policy_payload(policies, selection),
                "library": tuple(_policy_payload(policy) for policy in policies),
                "audit_response_candidates": tuple(
                    tuple(policy.name for policy in candidates)
                    for candidates in final_audit_candidates
                ),
                "representative_profile": tuple(
                    policy.name for policy in representative_profile
                ),
                "representative_weather": tuple(first_audit_scenarios[0]),
                "representative_value": representative.payoff,
                "representative_success": representative.success,
                "representative_replay": _replay_payload(
                    config, representative.replay
                ),
            },
            stats=stats,
            checkpoint=None,
            backend="heuristic",
            player_regret_lower=audit.regret_lower,
            player_regret_upper=audit.regret_upper,
        )
    except BudgetExceeded as exc:
        return _stopped_result(config, options, started, exc, policies)
