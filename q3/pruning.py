"""Only conservative, deterministic Q3 pruning helpers."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .data import Q3Config
from .model import PlayerState, Status


def shortest_distance_to_end(cfg: Q3Config) -> dict[int, int]:
    """Optimistic BFS distance, ignoring weather and all player interaction."""
    distance = {node: 10**9 for node in cfg.nodes}
    distance[cfg.end] = 0
    queue: deque[int] = deque([cfg.end])
    while queue:
        node = queue.popleft()
        for neighbor in cfg.adj[node]:
            if distance[neighbor] > distance[node] + 1:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    return distance


def success_is_deadline_impossible(
    position: int, *, day: int, cfg: Q3Config, distance: dict[int, int]
) -> bool:
    """Prove only that success is impossible; do not discard cash-seeking actions."""
    return distance[position] > cfg.deadline - day


def _distance_from(cfg: Q3Config, source: int) -> dict[int, int]:
    distance = {node: 10**9 for node in cfg.nodes}
    distance[source] = 0
    queue: deque[int] = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in cfg.adj[node]:
            if distance[neighbor] > distance[node] + 1:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    return distance


def optimistic_initial_resource_requirements(
    cfg: Q3Config,
) -> tuple[tuple[int, int], ...]:
    """Return relaxed resource pairs that can reach an economic opportunity.

    A positive day-0 purchase that cannot reach the end or a village, and
    cannot both reach and work at a mine, inevitably fails without changing
    cash.  The zero purchase then gives strictly higher failure payoff.  The
    requirements below are deliberately optimistic: solo travel, independently
    minimum positive-weather consumption, and no congestion.
    """
    movable_weather = tuple(
        weather
        for weather in cfg.weather_order
        if weather != "sandstorm" and cfg.p_weather[weather] > 0
    )
    positive_weather = tuple(
        weather for weather in cfg.weather_order if cfg.p_weather[weather] > 0
    )
    if not movable_weather or not positive_weather:
        return ()
    move_water = 2 * min(cfg.water_consume[weather] for weather in movable_weather)
    move_food = 2 * min(cfg.food_consume[weather] for weather in movable_weather)
    mine_water = 3 * min(cfg.water_consume[weather] for weather in positive_weather)
    mine_food = 3 * min(cfg.food_consume[weather] for weather in positive_weather)
    distance = _distance_from(cfg, cfg.start)

    requirements: set[tuple[int, int]] = set()
    for target in cfg.villages | {cfg.end}:
        moves = distance[target]
        if moves <= cfg.deadline:
            requirements.add((moves * move_water, moves * move_food))
    for target in cfg.mines:
        moves = distance[target]
        if moves + 1 <= cfg.deadline:
            requirements.add(
                (
                    moves * move_water + mine_water,
                    moves * move_food + mine_food,
                )
            )

    # Remove componentwise weaker requirements; satisfying a stronger pair
    # never adds a new feasible purchase.
    minimal = []
    for candidate in sorted(requirements):
        if any(
            other[0] <= candidate[0] and other[1] <= candidate[1] and other != candidate
            for other in requirements
        ):
            continue
        minimal.append(candidate)
    return tuple(minimal)


def initial_purchase_is_potentially_useful(
    water: int,
    food: int,
    requirements: tuple[tuple[int, int], ...],
) -> bool:
    if water == 0 and food == 0:
        return True
    return any(
        water >= req_water and food >= req_food for req_water, req_food in requirements
    )


def _max_refund_scaled(cfg: Q3Config) -> int:
    best = 0
    max_water = cfg.weight_limit // cfg.water_weight
    for water in range(max_water + 1):
        remaining = cfg.weight_limit - cfg.water_weight * water
        food = remaining // cfg.food_weight
        refund = (
            cfg.money_scale * cfg.water_price * water
            + cfg.money_scale * cfg.food_price * food
        ) // 2
        best = max(best, refund)
    return best


def _add_up(left: float, right: float) -> float:
    return math.nextafter(left + right, math.inf)


def _mul_fraction_up(value: float, probability: Fraction) -> float:
    numerator = math.nextafter(value * probability.numerator, math.inf)
    return math.nextafter(numerator / probability.denominator, math.inf)


@dataclass(frozen=True)
class RelaxedSinglePlayerUpperBound:
    """Resource-free single-player relaxation used only as a proof bound.

    The relaxation keeps the map, horizon, weather observation order,
    sandstorm movement rule, terminal penalty, and mine locations.  It removes
    all resource/cash purchase constraints, road congestion, mine sharing, and
    terminal-inventory restrictions.  Consequently every real Q3 continuation
    maps to an action sequence in this relaxation with no smaller payoff.
    """

    cfg: Q3Config
    residual: np.ndarray
    max_refund: float

    @classmethod
    def build(cls, cfg: Q3Config) -> RelaxedSinglePlayerUpperBound:
        node_count = max(cfg.nodes)
        residual = np.full(
            (cfg.deadline + 1, node_count + 1),
            -float(cfg.failure_penalty),
            dtype=np.float64,
        )
        max_refund = math.nextafter(_max_refund_scaled(cfg) / cfg.money_scale, math.inf)
        residual[:, cfg.end] = max_refund
        probabilities = tuple(
            (weather, Fraction(str(cfg.p_weather[weather])))
            for weather in cfg.weather_order
        )
        probability_sum = sum(
            (probability for _, probability in probabilities), Fraction(0, 1)
        )

        for day in range(cfg.deadline - 1, -1, -1):
            for position in cfg.nodes:
                if position == cfg.end:
                    continue
                weather_bounds: list[tuple[float, Fraction]] = []
                for weather, probability in probabilities:
                    best = float(residual[day + 1, position])
                    if position in cfg.mines:
                        best = max(
                            best,
                            _add_up(best, float(cfg.mine_income)),
                        )
                    if weather != "sandstorm":
                        for destination in cfg.adj[position]:
                            best = max(best, float(residual[day + 1, destination]))
                    weather_bounds.append((best, probability))
                if probability_sum == 1:
                    expected = 0.0
                    for best, probability in weather_bounds:
                        expected = _add_up(
                            expected, _mul_fraction_up(best, probability)
                        )
                else:
                    # If custom float probabilities do not recover an exact
                    # decimal partition of one, the maximum weather branch is
                    # still an upper bound for every convex combination.
                    expected = max(best for best, _ in weather_bounds)
                residual[day, position] = expected
        return cls(cfg=cfg, residual=residual, max_refund=max_refund)

    def value(self, day: int, player: PlayerState) -> float:
        if not 0 <= day <= self.cfg.deadline:
            raise ValueError("upper-bound day is outside the horizon")
        if player.status is Status.FINISHED or player.status is Status.FAILED:
            return player.fixed_payoff_scaled / self.cfg.money_scale
        cash = player.cash_scaled / self.cfg.money_scale
        return _add_up(cash, float(self.residual[day, player.position]))


@dataclass(frozen=True)
class BestResponsePruningCertificate:
    stage: str
    day: int
    weather: str
    player: int
    profile_index: tuple[int, ...]
    upper_bound: float
    exact_lower_bound: float
    safety_margin: float
