"""Only conservative, deterministic Q3 pruning helpers."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from .data import Q3Config
from .model import PlayerState, Status
from .resource_index import ResourceIndex

try:  # The exact scalar recurrence remains available without Numba.
    from numba import get_num_threads, njit, prange, set_num_threads

    NUMBA_BOUND_AVAILABLE = True
except ImportError:  # pragma: no cover - Numba is part of the formal environment.
    get_num_threads = None
    njit = None
    prange = range
    set_num_threads = None
    NUMBA_BOUND_AVAILABLE = False


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


if NUMBA_BOUND_AVAILABLE:

    @njit(cache=True, inline="always")
    def _add_up_numba(left: float, right: float) -> float:
        return math.nextafter(left + right, math.inf)


    @njit(cache=True, inline="always")
    def _mul_fraction_up_numba(
        value: float, numerator: int, denominator: int
    ) -> float:
        scaled = math.nextafter(value * numerator, math.inf)
        return math.nextafter(scaled / denominator, math.inf)


    @njit(cache=True, parallel=True)
    def _fill_resource_bound_layer_numba(
        output,
        next_layer,
        positions,
        position_to_row,
        water_by_id,
        food_by_id,
        id_grid,
        adjacency,
        degree,
        is_mine,
        end,
        terminal_refund,
        fallback_next,
        next_is_deadline,
        weather_count,
        base_water,
        base_food,
        weather_is_storm,
        probability_numerator,
        probability_denominator,
        use_expectation,
        mine_income,
        failure_penalty,
    ):
        resource_count = len(water_by_id)
        for flat_index in prange(len(positions) * resource_count):
            position_row = flat_index // resource_count
            resource_id = flat_index - position_row * resource_count
            position = int(positions[position_row])
            water = int(water_by_id[resource_id])
            food = int(food_by_id[resource_id])
            result = 0.0 if use_expectation else -math.inf

            for weather_index in range(weather_count):
                consume_water = int(base_water[weather_index])
                consume_food = int(base_food[weather_index])
                best = -failure_penalty

                next_water = water - consume_water
                next_food = food - consume_food
                if next_water >= 0 and next_food >= 0:
                    next_resource = int(id_grid[next_water, next_food])
                    future = (
                        -failure_penalty
                        if next_is_deadline
                        else float(next_layer[position_row, next_resource])
                    )
                    best = max(best, _add_up_numba(0.0, future))

                if is_mine[position]:
                    next_water = water - 3 * consume_water
                    next_food = food - 3 * consume_food
                    if next_water >= 0 and next_food >= 0:
                        next_resource = int(id_grid[next_water, next_food])
                        future = (
                            -failure_penalty
                            if next_is_deadline
                            else float(next_layer[position_row, next_resource])
                        )
                        best = max(best, _add_up_numba(mine_income, future))

                if not weather_is_storm[weather_index]:
                    next_water = water - 2 * consume_water
                    next_food = food - 2 * consume_food
                    if next_water >= 0 and next_food >= 0:
                        next_resource = int(id_grid[next_water, next_food])
                        for neighbor_index in range(int(degree[position])):
                            destination = int(adjacency[position, neighbor_index])
                            if destination == end:
                                future = float(terminal_refund[next_resource])
                            elif next_is_deadline:
                                future = -failure_penalty
                            else:
                                destination_row = int(position_to_row[destination])
                                if destination_row >= 0:
                                    future = float(
                                        next_layer[destination_row, next_resource]
                                    )
                                else:
                                    future = float(fallback_next[destination])
                            best = max(best, _add_up_numba(0.0, future))

                if use_expectation:
                    branch = _mul_fraction_up_numba(
                        best,
                        int(probability_numerator[weather_index]),
                        int(probability_denominator[weather_index]),
                    )
                    result = _add_up_numba(result, branch)
                else:
                    result = max(result, best)

            output[position_row, resource_id] = result

else:
    _fill_resource_bound_layer_numba = None


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


@dataclass
class ResourceAwareSinglePlayerUpperBound:
    """Upward-rounded relaxation indexed by ``(day, position, resource_id)``.

    Away from villages it keeps the exact inventory, load, map, weather,
    consumption, deadline, mining income, and terminal refund.  Congestion and
    mine sharing are removed.  At villages future purchases and their cash
    cost are relaxed completely; this is deliberately optimistic and avoids a
    quadratic inventory closure while remaining a valid proof bound.

    With Numba, requested day suffixes are built bottom-up into compact
    ``float64`` layers.  Every cell in a layer is independent once the next
    day is available, so the expensive recurrence runs through ``prange`` and
    never enters a Python dictionary.  The scalar packed-key memo is retained
    as a lossless fallback for installations without Numba.
    """

    cfg: Q3Config
    resources: ResourceIndex
    fallback: RelaxedSinglePlayerUpperBound
    requested_threads: int | None
    _positions: np.ndarray
    _position_to_row: np.ndarray
    _adjacency: np.ndarray
    _degree: np.ndarray
    _is_mine: np.ndarray
    _weather_count: np.ndarray
    _base_water: np.ndarray
    _base_food: np.ndarray
    _weather_is_storm: np.ndarray
    _probability_numerator: np.ndarray
    _probability_denominator: np.ndarray
    _use_expectation: np.ndarray
    _terminal_refund_by_id: np.ndarray
    _layers: list[np.ndarray | None]
    _lowest_built_day: int
    _memo: dict[int, float]
    _build_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    build_seconds: float = 0.0
    build_threads: int = 0

    @classmethod
    def build(
        cls, cfg: Q3Config, *, threads: int | None = None
    ) -> "ResourceAwareSinglePlayerUpperBound":
        if threads is not None and threads <= 0:
            raise ValueError("upper-bound threads must be positive")
        resources = ResourceIndex.build(cfg)
        max_position = max(cfg.nodes)
        positions = np.asarray(
            sorted(set(cfg.nodes) - set(cfg.villages) - {cfg.end}),
            dtype=np.int16,
        )
        position_to_row = np.full(max_position + 1, -1, dtype=np.int16)
        for row, position in enumerate(positions):
            position_to_row[int(position)] = row

        max_degree = max(len(cfg.adj[position]) for position in cfg.nodes)
        adjacency = np.zeros((max_position + 1, max_degree), dtype=np.int16)
        degree = np.zeros(max_position + 1, dtype=np.int8)
        for position in cfg.nodes:
            neighbors = cfg.adj[position]
            degree[position] = len(neighbors)
            adjacency[position, : len(neighbors)] = neighbors
        is_mine = np.zeros(max_position + 1, dtype=np.bool_)
        for position in cfg.mines:
            is_mine[position] = True

        max_weather_count = max(1, len(cfg.weather_order))
        shape = (cfg.deadline, max_weather_count)
        weather_count = np.zeros(cfg.deadline, dtype=np.int8)
        base_water = np.zeros(shape, dtype=np.int16)
        base_food = np.zeros(shape, dtype=np.int16)
        weather_is_storm = np.zeros(shape, dtype=np.bool_)
        probability_numerator = np.ones(shape, dtype=np.int64)
        probability_denominator = np.ones(shape, dtype=np.int64)
        use_expectation = np.ones(cfg.deadline, dtype=np.bool_)
        for day in range(cfg.deadline):
            if cfg.weather_sequence is not None:
                weather_values = ((cfg.weather_sequence[day], Fraction(1, 1)),)
            else:
                weather_values = tuple(
                    (
                        weather,
                        Fraction(str(cfg.p_weather[weather])),
                    )
                    for weather in cfg.weather_order
                )
            weather_count[day] = len(weather_values)
            use_expectation[day] = (
                sum(
                    (probability for _, probability in weather_values),
                    Fraction(0, 1),
                )
                == 1
            )
            for weather_index, (weather, probability) in enumerate(weather_values):
                base_water[day, weather_index] = cfg.water_consume[weather]
                base_food[day, weather_index] = cfg.food_consume[weather]
                weather_is_storm[day, weather_index] = weather == "sandstorm"
                probability_numerator[day, weather_index] = probability.numerator
                probability_denominator[day, weather_index] = probability.denominator

        refund_scaled = (
            cfg.money_scale
            * (
                cfg.water_price * resources.water.astype(np.int64)
                + cfg.food_price * resources.food.astype(np.int64)
            )
        ) // 2
        terminal_refund_by_id = np.nextafter(
            refund_scaled.astype(np.float64) / cfg.money_scale,
            np.inf,
        )
        return cls(
            cfg=cfg,
            resources=resources,
            fallback=RelaxedSinglePlayerUpperBound.build(cfg),
            requested_threads=threads,
            _positions=positions,
            _position_to_row=position_to_row,
            _adjacency=adjacency,
            _degree=degree,
            _is_mine=is_mine,
            _weather_count=weather_count,
            _base_water=base_water,
            _base_food=base_food,
            _weather_is_storm=weather_is_storm,
            _probability_numerator=probability_numerator,
            _probability_denominator=probability_denominator,
            _use_expectation=use_expectation,
            _terminal_refund_by_id=terminal_refund_by_id,
            _layers=[None] * cfg.deadline,
            _lowest_built_day=cfg.deadline,
            _memo={},
        )

    def _terminal_refund(self, water: int, food: int) -> float:
        resource_id = self.resources.encode(water, food)
        return float(self._terminal_refund_by_id[resource_id])

    def _weather_values(self, day: int) -> tuple[tuple[str, Fraction], ...]:
        if self.cfg.weather_sequence is not None:
            return ((self.cfg.weather_sequence[day], Fraction(1, 1)),)
        return tuple(
            (weather, Fraction(str(self.cfg.p_weather[weather])))
            for weather in self.cfg.weather_order
        )

    def _packed_key(self, day: int, position: int, resource_id: int) -> int:
        return (
            (day * len(self._position_to_row) + position)
            * len(self.resources.water)
            + resource_id
        )

    def _sparse_residual(self, day: int, position: int, resource_id: int) -> float:
        key = self._packed_key(day, position, resource_id)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        water, food = self.resources.decode(resource_id)
        if position == self.cfg.end:
            result = self._terminal_refund(water, food)
        elif day >= self.cfg.deadline:
            result = -float(self.cfg.failure_penalty)
        elif position in self.cfg.villages:
            # Free replenishment and free future village purchases strictly
            # enlarge the feasible set.  The resource-free residual already
            # includes the largest possible refund and solo mine income.
            result = float(self.fallback.residual[day, position])
        else:
            branches: list[tuple[float, Fraction]] = []
            for weather, probability in self._weather_values(day):
                base_w = self.cfg.water_consume[weather]
                base_f = self.cfg.food_consume[weather]
                best = -float(self.cfg.failure_penalty)

                def continuation(
                    multiplier: int, destination: int, income: float = 0.0
                ) -> float | None:
                    next_w = water - multiplier * base_w
                    next_f = food - multiplier * base_f
                    if next_w < 0 or next_f < 0:
                        return None
                    if destination == self.cfg.end:
                        future = self._terminal_refund(next_w, next_f)
                    else:
                        next_id = self.resources.encode(next_w, next_f)
                        future = self._sparse_residual(day + 1, destination, next_id)
                    return _add_up(income, future)

                stayed = continuation(1, position)
                if stayed is not None:
                    best = max(best, stayed)
                if position in self.cfg.mines:
                    mined = continuation(3, position, float(self.cfg.mine_income))
                    if mined is not None:
                        best = max(best, mined)
                if weather != "sandstorm":
                    for destination in self.cfg.adj[position]:
                        moved = continuation(2, destination)
                        if moved is not None:
                            best = max(best, moved)
                branches.append((best, probability))

            probability_sum = sum(
                (probability for _, probability in branches), Fraction(0, 1)
            )
            if probability_sum == 1:
                result = 0.0
                for branch, probability in branches:
                    result = _add_up(result, _mul_fraction_up(branch, probability))
            else:
                result = max(branch for branch, _ in branches)
        self._memo[key] = result
        return result

    def _selected_thread_count(self) -> int:
        if not NUMBA_BOUND_AVAILABLE:
            return 0
        available = int(get_num_threads())
        return min(available, self.requested_threads or 64)

    def _ensure_day(self, day: int) -> None:
        if not NUMBA_BOUND_AVAILABLE or day >= self.cfg.deadline:
            return
        if self._layers[day] is not None:
            return
        with self._build_lock:
            if self._layers[day] is not None:
                return
            selected_threads = self._selected_thread_count()
            previous_threads = int(get_num_threads())
            started = time.perf_counter()
            try:
                if selected_threads != previous_threads:
                    set_num_threads(selected_threads)
                while self._lowest_built_day > day:
                    build_day = self._lowest_built_day - 1
                    next_is_deadline = build_day + 1 >= self.cfg.deadline
                    next_layer = (
                        np.empty((0, 0), dtype=np.float64)
                        if next_is_deadline
                        else self._layers[build_day + 1]
                    )
                    if next_layer is None:
                        raise AssertionError("resource-bound day dependency is missing")
                    output = np.empty(
                        (len(self._positions), len(self.resources.water)),
                        dtype=np.float64,
                    )
                    _fill_resource_bound_layer_numba(
                        output,
                        next_layer,
                        self._positions,
                        self._position_to_row,
                        self.resources.water,
                        self.resources.food,
                        self.resources.id_grid,
                        self._adjacency,
                        self._degree,
                        self._is_mine,
                        self.cfg.end,
                        self._terminal_refund_by_id,
                        self.fallback.residual[build_day + 1],
                        next_is_deadline,
                        int(self._weather_count[build_day]),
                        self._base_water[build_day],
                        self._base_food[build_day],
                        self._weather_is_storm[build_day],
                        self._probability_numerator[build_day],
                        self._probability_denominator[build_day],
                        bool(self._use_expectation[build_day]),
                        float(self.cfg.mine_income),
                        float(self.cfg.failure_penalty),
                    )
                    self._layers[build_day] = output
                    self._lowest_built_day = build_day
            finally:
                if selected_threads != previous_threads:
                    set_num_threads(previous_threads)
                self.build_seconds += time.perf_counter() - started
                self.build_threads = selected_threads

    @property
    def cache_entries(self) -> int:
        if NUMBA_BOUND_AVAILABLE:
            return sum(layer.size for layer in self._layers if layer is not None)
        return len(self._memo)

    @property
    def cache_bytes(self) -> int:
        if NUMBA_BOUND_AVAILABLE:
            return self._terminal_refund_by_id.nbytes + sum(
                layer.nbytes for layer in self._layers if layer is not None
            )
        return 0

    def residuals_by_id(
        self,
        day: int,
        positions: np.ndarray,
        resource_ids: np.ndarray,
    ) -> np.ndarray:
        """Return a batch of residual bounds without Python per-state lookup."""
        if not 0 <= day <= self.cfg.deadline:
            raise ValueError("upper-bound day is outside the horizon")
        positions = np.asarray(positions, dtype=np.int64)
        resource_ids = np.asarray(resource_ids, dtype=np.int64)
        if positions.shape != resource_ids.shape:
            raise ValueError("position and resource-id arrays have different shapes")
        if np.any(resource_ids < 0) or np.any(
            resource_ids >= len(self.resources.water)
        ):
            raise ValueError("resource id is outside the feasible inventory index")
        output = np.full(positions.shape, -float(self.cfg.failure_penalty))
        at_end = positions == self.cfg.end
        output[at_end] = self._terminal_refund_by_id[resource_ids[at_end]]
        if day >= self.cfg.deadline:
            return output

        village = np.isin(positions, tuple(self.cfg.villages))
        for position in self.cfg.villages:
            output[positions == position] = self.fallback.residual[day, position]
        active = ~(at_end | village)
        if not np.any(active):
            return output
        if NUMBA_BOUND_AVAILABLE:
            self._ensure_day(day)
            layer = self._layers[day]
            if layer is None:
                raise AssertionError("resource-bound day layer was not built")
            rows = self._position_to_row[positions[active]]
            if np.any(rows < 0):
                raise ValueError("position is outside the configured map")
            output[active] = layer[rows, resource_ids[active]]
        else:
            output[active] = np.fromiter(
                (
                    self._sparse_residual(day, int(position), int(resource_id))
                    for position, resource_id in zip(
                        positions[active], resource_ids[active], strict=True
                    )
                ),
                dtype=np.float64,
                count=int(np.sum(active)),
            )
        return output

    def _residual(self, day: int, position: int, resource_id: int) -> float:
        return float(
            self.residuals_by_id(
                day,
                np.asarray([position], dtype=np.int64),
                np.asarray([resource_id], dtype=np.int64),
            )[0]
        )

    def value_by_id(
        self, day: int, position: int, resource_id: int, cash_scaled: int
    ) -> float:
        if not 0 <= day <= self.cfg.deadline:
            raise ValueError("upper-bound day is outside the horizon")
        cash = cash_scaled / self.cfg.money_scale
        return _add_up(cash, self._residual(day, position, resource_id))

    def value(self, day: int, player: PlayerState) -> float:
        if player.status is Status.FINISHED or player.status is Status.FAILED:
            return player.fixed_payoff_scaled / self.cfg.money_scale
        resource_id = self.resources.encode(player.water, player.food)
        return self.value_by_id(day, player.position, resource_id, player.cash_scaled)


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
