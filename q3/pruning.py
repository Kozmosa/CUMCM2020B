"""Only conservative, deterministic Q3 pruning helpers."""

from __future__ import annotations

from collections import deque

from .data import Q3Config


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
