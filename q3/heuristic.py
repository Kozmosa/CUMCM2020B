"""Finite-policy Monte Carlo backend for a contest-scale Q3.2 solution.

The exact and adaptive backends solve a much stronger state-by-state dynamic
game.  This module keeps the lossless Q3 transition rules, but replaces the
full feedback-strategy space with a small deterministic library of route,
purchase, and mining policies.  A common set of weather scenarios estimates
the resulting finite normal-form game.

The reported regret is therefore only against deviations inside the generated
policy library.  It is a statistical estimate, not a full-action certificate.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations_with_replacement, permutations, product
from math import ceil, floor, prod, sqrt
from statistics import NormalDist
from time import perf_counter
from typing import Sequence, cast

import numpy as np

from .data import Q3Config, Weather
from .model import (
    Action,
    ActionKind,
    FAIL_ACTION,
    INACTIVE_ACTION,
    JointState,
    PlayerState,
    Status,
    all_absorbed,
    initial_joint_state,
    terminal_state_value,
    weight_ok,
)
from .reports import Q32SolveResult
from .runtime import BudgetExceeded, BudgetManager, peak_rss_bytes
from .stage_game import (
    independent_mixture_values,
    minimize_nashconv,
    pure_nash_indices,
)
from .transition import (
    apply_initial_purchases,
    apply_player_action_given_counts,
    count_interactions_scalar,
)


@dataclass(frozen=True, slots=True)
class HeuristicOptions:
    """Controls the finite policy library and Monte Carlo empirical game."""

    episodes: int = 500
    confidence: float = 0.95
    max_policies: int = 16
    route_variants_per_family: int = 2
    safety_factors: tuple[float, ...] = (1.0, 1.75, 2.5)
    mine_day_choices: tuple[int, ...] = (2, 4)
    equilibrium: str = "pure-mixed"
    pure_tolerance: float = 5.0
    seed: int = 20260728
    mixed_starts: int = 4
    mixed_max_iterations: int = 500

    def __post_init__(self) -> None:
        if self.episodes < 2:
            raise ValueError("heuristic episodes must be at least two")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("heuristic confidence must be in (0, 1)")
        if self.max_policies <= 0 or self.max_policies > 32:
            raise ValueError("heuristic max_policies must be in 1..32")
        if self.route_variants_per_family <= 0:
            raise ValueError("route_variants_per_family must be positive")
        if not self.safety_factors or any(value <= 0.0 for value in self.safety_factors):
            raise ValueError("safety_factors must be positive")
        if any(value < 0 for value in self.mine_day_choices):
            raise ValueError("mine_day_choices cannot be negative")
        if self.equilibrium not in {"pure", "pure-mixed"}:
            raise ValueError("heuristic equilibrium must be pure or pure-mixed")
        if self.pure_tolerance < 0.0:
            raise ValueError("pure_tolerance cannot be negative")
        if self.mixed_starts <= 0 or self.mixed_max_iterations <= 0:
            raise ValueError("mixed optimizer limits must be positive")


@dataclass(frozen=True, slots=True)
class HeuristicPolicy:
    """A compact weather-contingent policy with a committed route template."""

    name: str
    family: str
    route: tuple[int, ...]
    initial_water: int
    initial_food: int
    mine_days: int = 0
    village_water_target: int = 0
    village_food_target: int = 0
    safety_factor: float = 1.0

    def initial_action(self) -> Action:
        return Action(
            ActionKind.INITIAL_BUY,
            buy_water=self.initial_water,
            buy_food=self.initial_food,
        )


@dataclass(frozen=True, slots=True)
class HeuristicReplayDay:
    day: int
    weather: Weather
    actions: tuple[Action, ...]
    state: JointState


@dataclass(frozen=True, slots=True)
class _SimulationResult:
    payoff: tuple[float, ...]
    success: tuple[float, ...]
    day_steps: int
    replay: tuple[HeuristicReplayDay, ...] = ()


def _distances_to(cfg: Q3Config, target: int) -> dict[int, int]:
    distances = {target: 0}
    queue = deque([target])
    while queue:
        node = queue.popleft()
        for neighbor in cfg.adj[node]:
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def _shortest_paths(
    cfg: Q3Config, source: int, target: int, *, limit: int
) -> tuple[tuple[int, ...], ...]:
    if source == target:
        return ((source,),)
    distances = _distances_to(cfg, target)
    if source not in distances:
        return ()

    def branch_paths(first: int) -> list[tuple[int, ...]]:
        branch: list[tuple[int, ...]] = []

        def visit(node: int, path: tuple[int, ...]) -> None:
            if len(branch) >= limit:
                return
            if node == target:
                branch.append(path)
                return
            next_distance = distances[node] - 1
            for neighbor in sorted(cfg.adj[node]):
                if distances.get(neighbor) == next_distance:
                    visit(neighbor, path + (neighbor,))

        visit(first, (source, first))
        return branch

    first_hops = [
        neighbor
        for neighbor in sorted(cfg.adj[source])
        if distances.get(neighbor) == distances[source] - 1
    ]
    branches = [branch_paths(neighbor) for neighbor in first_hops]
    output: list[tuple[int, ...]] = []
    offset = 0
    while len(output) < limit:
        added = False
        for branch in branches:
            if offset < len(branch):
                output.append(branch[offset])
                added = True
                if len(output) >= limit:
                    break
        if not added:
            break
        offset += 1
    return tuple(output)


def _routes_via(
    cfg: Q3Config, waypoints: Sequence[int], *, limit: int
) -> tuple[tuple[int, ...], ...]:
    anchors = (cfg.start, *waypoints, cfg.end)
    segments = []
    for source, target in zip(anchors, anchors[1:]):
        paths = _shortest_paths(cfg, source, target, limit=limit)
        if not paths:
            return ()
        segments.append(paths)
    output: list[tuple[int, ...]] = []
    index_combinations = sorted(
        product(*(range(len(segment)) for segment in segments)),
        key=lambda index: (
            max(index, default=0),
            sum(index),
            tuple(-value for value in index),
        ),
    )
    for index in index_combinations:
        selected = tuple(
            segments[segment][choice]
            for segment, choice in enumerate(index)
        )
        route = selected[0]
        for segment in selected[1:]:
            route += segment[1:]
        if len(set(route)) != len(route):
            continue
        if route not in output:
            output.append(route)
        if len(output) >= limit:
            break
    return tuple(output)


def _mean_consumption(cfg: Q3Config, *, action: str, resource: str) -> float:
    consume = cfg.water_consume if resource == "water" else cfg.food_consume
    if action == "mine":
        return sum(float(cfg.p_weather[w]) * 3.0 * consume[w] for w in cfg.p_weather)
    nonstorm = 1.0 - float(cfg.p_weather.get("sandstorm", 0.0))
    if nonstorm <= 0.0:
        return float("inf")
    moving = sum(
        float(probability) * 2.0 * consume[weather]
        for weather, probability in cfg.p_weather.items()
        if weather != "sandstorm"
    ) / nonstorm
    storm_wait = (
        float(cfg.p_weather.get("sandstorm", 0.0))
        * consume.get("sandstorm", 0)
        / nonstorm
    )
    return moving + storm_wait


def _fit_inventory(
    cfg: Q3Config,
    water: int,
    food: int,
    *,
    price_factor: int,
) -> tuple[int, int]:
    water = max(0, int(water))
    food = max(0, int(food))
    load = cfg.water_weight * water + cfg.food_weight * food
    cost = price_factor * (cfg.water_price * water + cfg.food_price * food)
    scale = 1.0
    if load > cfg.weight_limit:
        scale = min(scale, cfg.weight_limit / load)
    if cost > cfg.init_cash:
        scale = min(scale, cfg.init_cash / cost)
    if scale < 1.0:
        water = floor(water * scale)
        food = floor(food * scale)
    while (
        cfg.water_weight * water + cfg.food_weight * food > cfg.weight_limit
        or price_factor * (cfg.water_price * water + cfg.food_price * food)
        > cfg.init_cash
    ):
        if water * cfg.water_weight >= food * cfg.food_weight and water:
            water -= 1
        elif food:
            food -= 1
        else:
            break
    return water, food


def _resource_requirement(
    cfg: Q3Config,
    *,
    moves: int,
    mine_days: int,
    safety_factor: float,
) -> tuple[int, int]:
    water = safety_factor * (
        moves * _mean_consumption(cfg, action="move", resource="water")
        + mine_days * _mean_consumption(cfg, action="mine", resource="water")
    )
    food = safety_factor * (
        moves * _mean_consumption(cfg, action="move", resource="food")
        + mine_days * _mean_consumption(cfg, action="mine", resource="food")
    )
    if not np.isfinite(water) or not np.isfinite(food):
        return _fit_inventory(cfg, cfg.weight_limit, cfg.weight_limit, price_factor=1)
    return ceil(water), ceil(food)


def _policy_for_route(
    cfg: Q3Config,
    *,
    family: str,
    route: tuple[int, ...],
    route_index: int,
    mine_days: int,
    safety_factor: float,
) -> HeuristicPolicy:
    village_indices = [
        index for index, node in enumerate(route) if node in cfg.villages
    ]
    mine_indices = [index for index, node in enumerate(route) if node in cfg.mines]
    first_village = min(village_indices) if village_indices else None
    if first_village is None:
        initial_moves = len(route) - 1
        initial_mine_days = mine_days
        village_target = (0, 0)
    else:
        initial_moves = first_village
        mine_before_village = bool(
            mine_indices and min(mine_indices) < first_village
        )
        initial_mine_days = mine_days if mine_before_village else 0
        remaining_mine_days = mine_days - initial_mine_days
        village_target = _resource_requirement(
            cfg,
            moves=len(route) - 1 - first_village,
            mine_days=remaining_mine_days,
            safety_factor=safety_factor,
        )
        village_target = _fit_inventory(
            cfg, *village_target, price_factor=4
        )
    initial = _resource_requirement(
        cfg,
        moves=initial_moves,
        mine_days=initial_mine_days,
        safety_factor=safety_factor,
    )
    initial = _fit_inventory(cfg, *initial, price_factor=1)
    suffix = f"r{route_index + 1}-m{mine_days}-s{safety_factor:g}"
    return HeuristicPolicy(
        name=f"{family}-{suffix}",
        family=family,
        route=route,
        initial_water=initial[0],
        initial_food=initial[1],
        mine_days=mine_days,
        village_water_target=village_target[0],
        village_food_target=village_target[1],
        safety_factor=safety_factor,
    )


def generate_heuristic_policies(
    cfg: Q3Config, options: HeuristicOptions | None = None
) -> tuple[HeuristicPolicy, ...]:
    """Generate a deterministic, balanced finite library of route policies."""
    options = options or HeuristicOptions()
    families: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = [
        ("direct", (), (0,)),
    ]
    for village in sorted(cfg.villages):
        families.append((f"village{village}", (village,), (0,)))
    for mine in sorted(cfg.mines):
        mine_days = tuple(sorted(set(options.mine_day_choices))) or (0,)
        families.append((f"mine{mine}", (mine,), mine_days))
    for village in sorted(cfg.villages):
        for mine in sorted(cfg.mines):
            families.append(
                (
                    f"village{village}-mine{mine}",
                    (village, mine),
                    tuple(sorted(set(options.mine_day_choices))) or (0,),
                )
            )

    by_family: list[list[HeuristicPolicy]] = []
    safety_values = tuple(
        dict.fromkeys(
            (
                min(options.safety_factors),
                max(options.safety_factors),
                *options.safety_factors,
            )
        )
    )
    for family, waypoints, mine_days_values in families:
        routes = _routes_via(
            cfg, waypoints, limit=options.route_variants_per_family
        )
        candidates: list[HeuristicPolicy] = []
        combinations = sorted(
            product(
                range(len(mine_days_values)),
                range(len(safety_values)),
                range(len(routes)),
            ),
            key=lambda index: (max(index), sum(index), index),
        )
        for mine_index, safety_index, route_index in combinations:
            candidates.append(
                _policy_for_route(
                    cfg,
                    family=family,
                    route=routes[route_index],
                    route_index=route_index,
                    mine_days=mine_days_values[mine_index],
                    safety_factor=safety_values[safety_index],
                )
            )
        if candidates:
            by_family.append(candidates)

    selected: list[HeuristicPolicy] = []
    seen: set[tuple[object, ...]] = set()
    positions = [0] * len(by_family)
    while len(selected) < options.max_policies:
        added = False
        for family_index, candidates in enumerate(by_family):
            while positions[family_index] < len(candidates):
                policy = candidates[positions[family_index]]
                positions[family_index] += 1
                key = (
                    policy.route,
                    policy.initial_water,
                    policy.initial_food,
                    policy.mine_days,
                    policy.village_water_target,
                    policy.village_food_target,
                )
                if key in seen:
                    continue
                seen.add(key)
                selected.append(policy)
                added = True
                break
            if len(selected) >= options.max_policies:
                break
        if not added:
            break
    if not selected:
        raise ValueError("the map produced no start-to-end heuristic route")
    for policy in selected:
        _validate_policy(cfg, policy)
    return tuple(selected)


def _validate_policy(cfg: Q3Config, policy: HeuristicPolicy) -> None:
    if not policy.name:
        raise ValueError("heuristic policy name cannot be empty")
    if not policy.route or policy.route[0] != cfg.start or policy.route[-1] != cfg.end:
        raise ValueError(f"policy {policy.name} route must connect start to end")
    if any(
        target not in cfg.adj[source]
        for source, target in zip(policy.route, policy.route[1:])
    ):
        raise ValueError(f"policy {policy.name} route contains a non-edge")
    if len(set(policy.route)) != len(policy.route):
        raise ValueError(f"policy {policy.name} route must be simple")
    if min(
        policy.initial_water,
        policy.initial_food,
        policy.mine_days,
        policy.village_water_target,
        policy.village_food_target,
    ) < 0:
        raise ValueError(f"policy {policy.name} contains a negative quantity")
    if not weight_ok(cfg, policy.initial_water, policy.initial_food):
        raise ValueError(f"policy {policy.name} initial load exceeds capacity")
    initial_cost = (
        cfg.water_price * policy.initial_water
        + cfg.food_price * policy.initial_food
    )
    if initial_cost > cfg.init_cash:
        raise ValueError(f"policy {policy.name} cannot afford its initial purchase")


def _purchase_toward_target(
    cfg: Q3Config, player: PlayerState, policy: HeuristicPolicy
) -> tuple[int, int]:
    if player.position not in cfg.villages:
        return 0, 0
    water = max(0, policy.village_water_target - player.water)
    food = max(0, policy.village_food_target - player.food)
    if water == 0 and food == 0:
        return 0, 0
    remaining_weight = (
        cfg.weight_limit
        - cfg.water_weight * player.water
        - cfg.food_weight * player.food
    )
    remaining_cash = player.cash_scaled
    requested_weight = cfg.water_weight * water + cfg.food_weight * food
    requested_cost = cfg.money_scale * 4 * (
        cfg.water_price * water + cfg.food_price * food
    )
    scale = 1.0
    if requested_weight > remaining_weight:
        scale = min(scale, remaining_weight / requested_weight)
    if requested_cost > remaining_cash:
        scale = min(scale, remaining_cash / requested_cost)
    if scale < 1.0:
        water = floor(water * scale)
        food = floor(food * scale)
    while (
        cfg.water_weight * water + cfg.food_weight * food > remaining_weight
        or cfg.money_scale
        * 4
        * (cfg.water_price * water + cfg.food_price * food)
        > remaining_cash
    ):
        if water * cfg.water_weight >= food * cfg.food_weight and water:
            water -= 1
        elif food:
            food -= 1
        else:
            break
    return water, food


def _route_next(policy: HeuristicPolicy, position: int) -> int | None:
    try:
        index = policy.route.index(position)
    except ValueError:
        return None
    if index + 1 >= len(policy.route):
        return None
    return policy.route[index + 1]


def _heuristic_action(
    cfg: Q3Config,
    state: JointState,
    player_index: int,
    weather: Weather,
    policy: HeuristicPolicy,
    mined_days: int,
) -> Action:
    player = state[player_index]
    if player.status is not Status.ACTIVE:
        return INACTIVE_ACTION
    buy_water, buy_food = _purchase_toward_target(cfg, player, policy)
    if weather == "sandstorm":
        kind = (
            ActionKind.MINE
            if player.position in cfg.mines and mined_days < policy.mine_days
            else ActionKind.STAY
        )
        destination = 0
    elif player.position in cfg.mines and mined_days < policy.mine_days:
        kind = ActionKind.MINE
        destination = 0
    else:
        destination = _route_next(policy, player.position) or 0
        kind = ActionKind.MOVE if destination else ActionKind.STAY

    return Action(
        kind,
        destination=destination,
        buy_water=buy_water,
        buy_food=buy_food,
    )


def _resolve_actions(
    cfg: Q3Config,
    state: JointState,
    actions: Sequence[Action],
    weather: Weather,
) -> tuple[tuple[Action, ...], JointState]:
    """Apply deterministic stay/fail fallbacks until the profile is feasible."""
    resolved = list(actions)
    for _ in range(cfg.n_players + 2):
        counts = count_interactions_scalar(state, resolved)
        next_players: list[PlayerState | None] = []
        invalid: list[int] = []
        for player, (player_state, action) in enumerate(
            zip(state, resolved, strict=True)
        ):
            next_player = apply_player_action_given_counts(
                cfg,
                player_state,
                action,
                weather,
                edge_count=counts.edge[player],
                mine_count=counts.mine[player],
                village_buyer_count=counts.village_buyers[player],
            )
            next_players.append(next_player)
            if next_player is None:
                invalid.append(player)
        if not invalid:
            return tuple(resolved), cast(JointState, tuple(next_players))
        for player in invalid:
            action = resolved[player]
            if action.kind in {ActionKind.MOVE, ActionKind.MINE}:
                resolved[player] = Action(
                    ActionKind.STAY,
                    buy_water=action.buy_water,
                    buy_food=action.buy_food,
                )
            else:
                resolved[player] = FAIL_ACTION
    raise AssertionError("heuristic fallback resolution did not converge")


def _simulate_profile(
    cfg: Q3Config,
    profile: Sequence[HeuristicPolicy],
    weather: Sequence[Weather],
    *,
    capture: bool = False,
) -> _SimulationResult:
    if len(profile) != cfg.n_players:
        raise ValueError("heuristic profile player count does not match config")
    for policy in profile:
        _validate_policy(cfg, policy)
    initial = initial_joint_state(cfg)
    state = apply_initial_purchases(
        cfg, initial, tuple(policy.initial_action() for policy in profile)
    )
    if state is None:
        raise AssertionError("validated heuristic initial purchase became infeasible")
    mined_days = [0] * cfg.n_players
    records: list[HeuristicReplayDay] = []
    day_steps = 0
    for day, current_weather in enumerate(weather[: cfg.deadline], start=1):
        if all_absorbed(state):
            break
        proposed = tuple(
            _heuristic_action(
                cfg,
                state,
                player,
                current_weather,
                profile[player],
                mined_days[player],
            )
            for player in range(cfg.n_players)
        )
        actions, next_state = _resolve_actions(
            cfg, state, proposed, current_weather
        )
        for player, action in enumerate(actions):
            if action.kind is ActionKind.MINE:
                mined_days[player] += 1
        state = next_state
        day_steps += 1
        if capture:
            records.append(HeuristicReplayDay(day, current_weather, actions, state))
    value = terminal_state_value(cfg, state)
    return _SimulationResult(value.value, value.success, day_steps, tuple(records))


def simulate_heuristic_profile(
    cfg: Q3Config,
    profile: Sequence[HeuristicPolicy],
    weather: Sequence[Weather],
) -> _SimulationResult:
    """Public deterministic replay helper for tests and report generation."""
    return _simulate_profile(cfg, profile, weather, capture=True)


def _weather_scenarios(
    cfg: Q3Config, options: HeuristicOptions
) -> tuple[tuple[Weather, ...], ...]:
    if cfg.weather_sequence is not None:
        return tuple(tuple(cfg.weather_sequence) for _ in range(options.episodes))
    order = cfg.weather_order
    probabilities = np.asarray([cfg.p_weather[w] for w in order], dtype=np.float64)
    rng = np.random.default_rng(options.seed)
    sampled = rng.choice(
        np.asarray(order, dtype=object),
        size=(options.episodes, cfg.deadline),
        p=probabilities,
    )
    return tuple(tuple(str(item) for item in row) for row in sampled)


def _player_mapping(
    source_profile: tuple[int, ...], target_profile: tuple[int, ...]
) -> tuple[int, ...]:
    pools: dict[int, deque[int]] = {}
    for player, policy in enumerate(source_profile):
        pools.setdefault(policy, deque()).append(player)
    return tuple(pools[policy].popleft() for policy in target_profile)


def _empirical_game(
    cfg: Q3Config,
    policies: Sequence[HeuristicPolicy],
    scenarios: Sequence[Sequence[Weather]],
    budget_manager: BudgetManager | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    count = len(policies)
    episodes = len(scenarios)
    shape = (count,) * cfg.n_players
    payoff = np.empty(shape + (episodes, cfg.n_players), dtype=np.float64)
    success = np.empty(shape + (episodes, cfg.n_players), dtype=np.float64)
    canonical_profiles = 0
    day_steps = 0
    for profile_index in combinations_with_replacement(
        range(count), cfg.n_players
    ):
        if budget_manager is not None:
            budget_manager.check(profiles=canonical_profiles)
        profile = tuple(policies[index] for index in profile_index)
        profile_payoff = np.empty((episodes, cfg.n_players), dtype=np.float64)
        profile_success = np.empty((episodes, cfg.n_players), dtype=np.float64)
        for episode, weather in enumerate(scenarios):
            result = _simulate_profile(cfg, profile, weather)
            profile_payoff[episode] = result.payoff
            profile_success[episode] = result.success
            day_steps += result.day_steps
        for target in sorted(set(permutations(profile_index))):
            mapping = _player_mapping(profile_index, target)
            payoff[target] = profile_payoff[:, mapping]
            success[target] = profile_success[:, mapping]
        canonical_profiles += 1
    return payoff, success, {
        "canonical_profiles_simulated": canonical_profiles,
        "joint_profiles_filled": int(prod(shape)),
        "simulated_day_steps": day_steps,
    }


def _profile_regrets(payoff: np.ndarray, profile: tuple[int, ...]) -> tuple[float, ...]:
    n_players = payoff.shape[-1]
    regrets = []
    for player in range(n_players):
        slicer: list[int | slice] = list(profile)
        slicer[player] = slice(None)
        best = float(np.max(payoff[tuple(slicer) + (player,)]))
        current = float(payoff[profile + (player,)])
        regrets.append(max(0.0, best - current))
    return tuple(regrets)


def _select_pure_profile(
    indices: Sequence[tuple[int, ...]],
    payoff: np.ndarray,
    success: np.ndarray,
) -> tuple[int, ...]:
    ordered = sorted(indices)
    return max(
        ordered,
        key=lambda index: (
            float(np.sum(success[index])),
            float(np.min(payoff[index])),
            float(np.sum(payoff[index])),
        ),
    )


def _minimum_regret_profile(
    payoff: np.ndarray, success: np.ndarray
) -> tuple[int, ...]:
    shape = payoff.shape[:-1]
    profiles = list(np.ndindex(shape))
    minimum = min(max(_profile_regrets(payoff, profile)) for profile in profiles)
    candidates = [
        profile
        for profile in profiles
        if max(_profile_regrets(payoff, profile)) <= minimum + 1e-12
    ]
    return _select_pure_profile(candidates, payoff, success)


def _mixture_episode_samples(
    samples: np.ndarray,
    probabilities: Sequence[np.ndarray],
    player: int,
) -> np.ndarray:
    n_players = len(probabilities)
    episodes = samples.shape[-2]
    output = np.zeros(episodes, dtype=np.float64)
    for index in np.ndindex(*(len(p) for p in probabilities)):
        weight = 1.0
        for current_player in range(n_players):
            weight *= probabilities[current_player][index[current_player]]
        if weight:
            output += weight * samples[index + (slice(None), player)]
    return output


def _deviation_episode_samples(
    samples: np.ndarray,
    probabilities: Sequence[np.ndarray],
    player: int,
    deviation: int,
) -> np.ndarray:
    episodes = samples.shape[-2]
    output = np.zeros(episodes, dtype=np.float64)
    opponents = tuple(index for index in range(len(probabilities)) if index != player)
    for opponent_actions in np.ndindex(
        *(len(probabilities[opponent]) for opponent in opponents)
    ):
        profile = [0] * len(probabilities)
        profile[player] = deviation
        weight = 1.0
        for opponent, action in zip(opponents, opponent_actions, strict=True):
            profile[opponent] = action
            weight *= probabilities[opponent][action]
        if weight:
            output += weight * samples[tuple(profile) + (slice(None), player)]
    return output


def _confidence_interval(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    mean = float(np.mean(samples))
    if len(samples) < 2:
        return float("-inf"), float("inf")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    halfwidth = z * float(np.std(samples, ddof=1)) / sqrt(len(samples))
    return mean - halfwidth, mean + halfwidth


def _regret_intervals(
    payoff_samples: np.ndarray,
    probabilities: Sequence[np.ndarray],
    confidence: float,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    n_players = len(probabilities)
    action_count = payoff_samples.shape[0]
    tests = max(1, n_players * action_count)
    alpha = 1.0 - confidence
    z = NormalDist().inv_cdf(1.0 - alpha / (2.0 * tests))
    lower: list[float] = []
    mean_regret: list[float] = []
    upper: list[float] = []
    for player in range(n_players):
        baseline = _mixture_episode_samples(payoff_samples, probabilities, player)
        player_lower = 0.0
        player_mean = 0.0
        player_upper = 0.0
        for deviation in range(action_count):
            difference = (
                _deviation_episode_samples(
                    payoff_samples, probabilities, player, deviation
                )
                - baseline
            )
            mean = float(np.mean(difference))
            halfwidth = z * float(np.std(difference, ddof=1)) / sqrt(len(difference))
            player_lower = max(player_lower, mean - halfwidth)
            player_mean = max(player_mean, mean)
            player_upper = max(player_upper, mean + halfwidth)
        lower.append(max(0.0, player_lower))
        mean_regret.append(max(0.0, player_mean))
        upper.append(max(0.0, player_upper))
    return tuple(lower), tuple(mean_regret), tuple(upper)


def _policy_payload(policy: HeuristicPolicy) -> dict[str, object]:
    return {
        "name": policy.name,
        "family": policy.family,
        "route": policy.route,
        "initial_purchase": {
            "water": policy.initial_water,
            "food": policy.initial_food,
        },
        "mine_days": policy.mine_days,
        "village_target": {
            "water": policy.village_water_target,
            "food": policy.village_food_target,
        },
        "safety_factor": policy.safety_factor,
    }


def _state_payload(
    cfg: Q3Config, state: JointState
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "status": player.status.name,
            "position": player.position,
            "water": player.water,
            "food": player.food,
            "cash": player.cash_scaled / cfg.money_scale,
            "fixed_payoff": player.fixed_payoff_scaled / cfg.money_scale,
        }
        for player in state
    )


def _replay_payload(
    cfg: Q3Config, replay: Sequence[HeuristicReplayDay]
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "day": record.day,
            "weather": record.weather,
            "actions": tuple(action.label() for action in record.actions),
            "state": _state_payload(cfg, record.state),
        }
        for record in replay
    )


def solve_q3_2_heuristic(
    config: Q3Config,
    *,
    options: HeuristicOptions | None = None,
    policies: Sequence[HeuristicPolicy] | None = None,
    quality_target: float = 10.0,
    budget_manager: BudgetManager | None = None,
) -> Q32SolveResult:
    """Solve the finite empirical policy game without claiming full regret."""
    options = options or HeuristicOptions()
    started = perf_counter()
    policy_library = (
        tuple(policies)
        if policies is not None
        else generate_heuristic_policies(config, options)
    )
    for policy in policy_library:
        _validate_policy(config, policy)
    if not policy_library:
        raise ValueError("heuristic policy library cannot be empty")
    try:
        scenarios = _weather_scenarios(config, options)
        payoff_samples, success_samples, simulation_stats = _empirical_game(
            config, policy_library, scenarios, budget_manager
        )
        mean_payoff = np.mean(payoff_samples, axis=-2)
        mean_success = np.mean(success_samples, axis=-2)
        valid = np.ones(mean_payoff.shape[:-1], dtype=bool)
        pure_rows = pure_nash_indices(
            mean_payoff, valid, atol=options.pure_tolerance
        )
        equilibrium_kind: str
        if len(pure_rows):
            profile = _select_pure_profile(
                [tuple(int(value) for value in row) for row in pure_rows],
                mean_payoff,
                mean_success,
            )
            probabilities = tuple(
                np.eye(len(policy_library), dtype=np.float64)[profile[player]]
                for player in range(config.n_players)
            )
            equilibrium_kind = "pure"
            status = "HEURISTIC_PURE"
        elif options.equilibrium == "pure":
            profile = _minimum_regret_profile(mean_payoff, mean_success)
            probabilities = tuple(
                np.eye(len(policy_library), dtype=np.float64)[profile[player]]
                for player in range(config.n_players)
            )
            equilibrium_kind = "minimum-regret-pure"
            status = "HEURISTIC_PURE"
        else:
            finite = mean_payoff[np.isfinite(mean_payoff)]
            scale = max(1.0, float(np.max(np.abs(finite))) if finite.size else 1.0)
            mixed = minimize_nashconv(
                mean_payoff / scale,
                valid=valid,
                success=mean_success,
                seed=options.seed,
                starts=options.mixed_starts,
                max_iterations=options.mixed_max_iterations,
            )
            probabilities = tuple(
                np.asarray(block, dtype=np.float64) for block in mixed.probabilities
            )
            equilibrium_kind = "mixed"
            status = "HEURISTIC_MIXED"

        values, successes, empirical_regrets = independent_mixture_values(
            mean_payoff,
            probabilities,
            valid=valid,
            success=mean_success,
        )
        value_lower: list[float] = []
        value_upper: list[float] = []
        for player in range(config.n_players):
            samples = _mixture_episode_samples(
                payoff_samples, probabilities, player
            )
            lower, upper = _confidence_interval(samples, options.confidence)
            value_lower.append(lower)
            value_upper.append(upper)
        regret_lower, regret_mean, regret_upper = _regret_intervals(
            payoff_samples, probabilities, options.confidence
        )

        representative_indices = tuple(
            int(np.argmax(probability)) for probability in probabilities
        )
        representative_profile = tuple(
            policy_library[index] for index in representative_indices
        )
        representative = _simulate_profile(
            config, representative_profile, scenarios[0], capture=True
        )
        player_policy = []
        for player, probability in enumerate(probabilities):
            support = tuple(
                {
                    "policy": policy_library[index].name,
                    "probability": float(value),
                }
                for index, value in enumerate(probability)
                if value > 1e-9
            )
            player_policy.append(
                {
                    "player": player + 1,
                    "support": support,
                }
            )
        stats: dict[str, object] = {
            **simulation_stats,
            "policy_count": len(policy_library),
            "episodes_per_profile": options.episodes,
            "weather_scenarios": len(scenarios),
            "confidence": options.confidence,
            "pure_tolerance": options.pure_tolerance,
            "empirical_pure_equilibria": len(pure_rows),
            "finite_library_pure_equilibrium_found": bool(len(pure_rows)),
            "empirical_value_mean": values,
            "empirical_success_mean": successes,
            "empirical_policy_regret_mean": empirical_regrets,
            "empirical_policy_regret_interval_mean": regret_mean,
            "regret_scope": "generated_finite_policy_library",
            "value_bounds": "normal_approximation_confidence_interval",
            "exact_joint_transition_rules": True,
            "full_action_regret_certified": False,
            "empirical_quality_target_met": max(regret_upper, default=0.0)
            <= quality_target,
            "peak_rss_bytes": peak_rss_bytes(),
            "elapsed_seconds": perf_counter() - started,
        }
        return Q32SolveResult(
            status=status,
            value_lower=tuple(value_lower),
            value_upper=tuple(value_upper),
            success=tuple(float(value) for value in successes),
            max_regret_lower=max(regret_lower, default=0.0),
            max_regret_upper=max(regret_upper, default=0.0),
            selection_complete=False,
            policy={
                "scope": "generated_finite_policy_library",
                "equilibrium": equilibrium_kind,
                "selection_complete_within_library": equilibrium_kind != "mixed",
                "players": tuple(player_policy),
                "library": tuple(_policy_payload(policy) for policy in policy_library),
                "representative_profile": tuple(
                    policy.name for policy in representative_profile
                ),
                "representative_weather": tuple(scenarios[0]),
                "representative_value": representative.payoff,
                "representative_success": representative.success,
                "representative_replay": _replay_payload(
                    config, representative.replay
                ),
            },
            stats=stats,
            checkpoint=None,
            backend="heuristic",
            player_regret_lower=regret_lower,
            player_regret_upper=regret_upper,
        )
    except BudgetExceeded as exc:
        return Q32SolveResult(
            status="SEARCH_STOPPED",
            value_lower=tuple(float("-inf") for _ in range(config.n_players)),
            value_upper=tuple(float("inf") for _ in range(config.n_players)),
            success=tuple(0.0 for _ in range(config.n_players)),
            max_regret_lower=0.0,
            max_regret_upper=float("inf"),
            selection_complete=False,
            policy={"error": str(exc), "scope": "finite_policy_library"},
            stats={
                "policy_count": len(policy_library),
                "episodes_per_profile": options.episodes,
                "regret_scope": "generated_finite_policy_library",
                "elapsed_seconds": perf_counter() - started,
                "peak_rss_bytes": peak_rss_bytes(),
            },
            checkpoint=None,
            backend="heuristic",
            player_regret_lower=tuple(0.0 for _ in range(config.n_players)),
            player_regret_upper=tuple(float("inf") for _ in range(config.n_players)),
        )
