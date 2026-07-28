"""Exact unilateral purchase screening on CPU/Numba and NVIDIA CUDA.

The adaptive solver repeatedly fixes the opponents' actions and scans one
player's complete integer purchase lattice.  Materialising a full joint
transition batch for every lattice point is unnecessary: interaction counts
depend only on the action skeleton and on whether the purchase is zero or
positive.  These kernels nevertheless validate the complete joint profile
for every point, then evaluate the same resource-aware continuation upper
bound used by the scalar solver.  Consequently a rejected point is certified
not to be a profitable deviation; no resources, cash, or actions are rounded.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .action_enum import IndividualActionArrays
from .data import Q3Config, Weather
from .model import Action, ActionKind, JointState, Status
from .pruning import ResourceAwareSinglePlayerUpperBound
from .transition import (
    apply_player_action_given_counts,
    count_interactions_scalar,
)

try:
    from numba import njit

    NUMBA_PURCHASE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional acceleration dependency
    NUMBA_PURCHASE_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorate(function):
            return function

        return decorate


try:
    from numba import cuda
    from numba.cuda import libdevice

    CUDA_PURCHASE_AVAILABLE = bool(cuda.is_available())
except (ImportError, RuntimeError):  # pragma: no cover - optional GPU dependency
    cuda = None
    libdevice = None
    CUDA_PURCHASE_AVAILABLE = False


_STATUS_ACTIVE = int(Status.ACTIVE)
_ACTION_INACTIVE = int(ActionKind.INACTIVE)
_ACTION_FAIL = int(ActionKind.FAIL)
_ACTION_STAY = int(ActionKind.STAY)
_ACTION_MINE = int(ActionKind.MINE)
_ACTION_MOVE = int(ActionKind.MOVE)
_ACTION_INITIAL_BUY = int(ActionKind.INITIAL_BUY)


@dataclass(frozen=True, slots=True)
class PurchaseOracleOptions:
    backend: str = "auto"
    threads: int | None = None
    cuda_device: int = 0
    cuda_min_actions: int = 131_072
    parallel_min_actions: int = 32_768

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "cpu", "cuda", "off"}:
            raise ValueError("purchase oracle backend must be auto/cpu/cuda/off")
        if self.threads is not None and self.threads <= 0:
            raise ValueError("purchase oracle threads must be positive")
        if self.cuda_device < 0:
            raise ValueError("CUDA device must be non-negative")
        if self.cuda_min_actions <= 0 or self.parallel_min_actions <= 0:
            raise ValueError("purchase oracle thresholds must be positive")


@dataclass(frozen=True, slots=True)
class PurchaseScreenResult:
    survivor_indices: np.ndarray
    survivor_bounds: np.ndarray
    valid_count: int
    pruned_count: int
    max_pruned_bound: float
    regions_pruned: int
    backend: str
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class _BoundMaxPyramid:
    levels: tuple[np.ndarray, ...]


def _coarsen_maximum(values: np.ndarray) -> np.ndarray:
    rows, columns = values.shape
    output = np.full(
        ((rows + 1) // 2, (columns + 1) // 2),
        -np.inf,
        dtype=np.float64,
    )
    for row_offset in (0, 1):
        for column_offset in (0, 1):
            source = values[row_offset::2, column_offset::2]
            target = output[: source.shape[0], : source.shape[1]]
            np.maximum(target, source, out=target)
    return output


def _topology_arrays(cfg: Q3Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = max(cfg.nodes) + 1
    adjacency = np.zeros((size, size), dtype=np.bool_)
    is_mine = np.zeros(size, dtype=np.bool_)
    is_village = np.zeros(size, dtype=np.bool_)
    for source, destinations in cfg.adj.items():
        adjacency[source, destinations] = True
    for position in cfg.mines:
        is_mine[position] = True
    for position in cfg.villages:
        is_village[position] = True
    return adjacency, is_mine, is_village


@njit(cache=True)
def _profile_bound_one(
    row,
    deviating_player,
    continuation_day,
    deadline,
    end,
    failure_penalty,
    status,
    source,
    water0,
    food0,
    cash0,
    fixed_payoff,
    base_kind,
    base_destination,
    base_buy_water,
    base_buy_food,
    deviation_kind,
    deviation_destination,
    deviation_buy_water,
    deviation_buy_food,
    adjacency,
    is_mine_node,
    is_village_node,
    weather_is_storm,
    base_water,
    base_food,
    money_scale,
    water_price,
    food_price,
    water_weight,
    food_weight,
    weight_limit,
    mine_income,
    id_grid,
    position_to_row,
    bound_layer,
    village_residual,
    terminal_refund,
):
    n = len(status)
    deviator_next_position = int(source[deviating_player])
    deviator_next_water = int(water0[deviating_player])
    deviator_next_food = int(food0[deviating_player])
    deviator_next_cash = int(cash0[deviating_player])
    deviator_kind = int(deviation_kind[row])

    for i in range(n):
        kind_i = (
            int(deviation_kind[row])
            if i == deviating_player
            else int(base_kind[i])
        )
        destination_i = (
            int(deviation_destination[row])
            if i == deviating_player
            else int(base_destination[i])
        )
        buy_water_i = (
            int(deviation_buy_water[row])
            if i == deviating_player
            else int(base_buy_water[i])
        )
        buy_food_i = (
            int(deviation_buy_food[row])
            if i == deviating_player
            else int(base_buy_food[i])
        )
        buyer_i = buy_water_i + buy_food_i > 0

        if status[i] != _STATUS_ACTIVE:
            if kind_i != _ACTION_INACTIVE or buyer_i:
                return -math.inf
            continue
        if kind_i == _ACTION_FAIL:
            if buyer_i:
                return -math.inf
            continue
        if kind_i == _ACTION_INACTIVE or kind_i == _ACTION_INITIAL_BUY:
            return -math.inf
        if buyer_i and not is_village_node[source[i]]:
            return -math.inf
        if kind_i == _ACTION_MINE and not is_mine_node[source[i]]:
            return -math.inf
        if kind_i == _ACTION_MOVE:
            if weather_is_storm or not adjacency[source[i], destination_i]:
                return -math.inf
        elif kind_i != _ACTION_STAY and kind_i != _ACTION_MINE:
            return -math.inf

        edge_count = 0
        mine_count = 0
        buyer_count = 0
        for j in range(n):
            if status[j] != _STATUS_ACTIVE:
                continue
            kind_j = (
                int(deviation_kind[row])
                if j == deviating_player
                else int(base_kind[j])
            )
            destination_j = (
                int(deviation_destination[row])
                if j == deviating_player
                else int(base_destination[j])
            )
            buy_water_j = (
                int(deviation_buy_water[row])
                if j == deviating_player
                else int(base_buy_water[j])
            )
            buy_food_j = (
                int(deviation_buy_food[row])
                if j == deviating_player
                else int(base_buy_food[j])
            )
            if (
                kind_i == _ACTION_MOVE
                and kind_j == _ACTION_MOVE
                and source[i] == source[j]
                and destination_i == destination_j
            ):
                edge_count += 1
            if (
                kind_i == _ACTION_MINE
                and kind_j == _ACTION_MINE
                and source[i] == source[j]
            ):
                mine_count += 1
            if (
                buyer_i
                and buy_water_j + buy_food_j > 0
                and source[i] == source[j]
            ):
                buyer_count += 1

        price_factor = 0
        if buyer_i:
            if buyer_count <= 0:
                return -math.inf
            price_factor = 2 if buyer_count == 1 else 4
        purchase_cost = (
            money_scale
            * price_factor
            * (water_price * buy_water_i + food_price * buy_food_i)
        )
        water_pre = int(water0[i]) + buy_water_i
        food_pre = int(food0[i]) + buy_food_i
        cash_after_buy = int(cash0[i]) - purchase_cost
        if (
            cash_after_buy < 0
            or water_weight * water_pre + food_weight * food_pre > weight_limit
        ):
            return -math.inf

        multiplier = 0
        income = 0
        next_position = int(source[i])
        if kind_i == _ACTION_STAY:
            multiplier = 1
        elif kind_i == _ACTION_MINE:
            if mine_count <= 0:
                return -math.inf
            multiplier = 3
            income = money_scale * mine_income // mine_count
        else:
            if edge_count <= 0:
                return -math.inf
            multiplier = 2 * edge_count
            next_position = destination_i
        next_water = water_pre - multiplier * base_water
        next_food = food_pre - multiplier * base_food
        if next_water < 0 or next_food < 0:
            return -math.inf
        if i == deviating_player:
            deviator_next_position = next_position
            deviator_next_water = next_water
            deviator_next_food = next_food
            deviator_next_cash = cash_after_buy + income

    if status[deviating_player] != _STATUS_ACTIVE:
        return fixed_payoff[deviating_player] / money_scale
    if deviator_kind == _ACTION_FAIL:
        return (cash0[deviating_player] - money_scale * failure_penalty) / money_scale

    resource_id = int(id_grid[deviator_next_water, deviator_next_food])
    if resource_id < 0:
        return -math.inf
    if deviator_next_position == end:
        return (deviator_next_cash / money_scale) + terminal_refund[resource_id]
    if continuation_day >= deadline:
        residual = -failure_penalty
    elif is_village_node[deviator_next_position]:
        residual = village_residual[deviator_next_position]
    else:
        position_row = int(position_to_row[deviator_next_position])
        if position_row < 0:
            return -math.inf
        residual = bound_layer[position_row, resource_id]
    return math.nextafter(deviator_next_cash / money_scale + residual, math.inf)


@njit(cache=True, nogil=True)
def _screen_profile_bounds_serial(
    deviating_player,
    continuation_day,
    deadline,
    end,
    failure_penalty,
    status,
    source,
    water0,
    food0,
    cash0,
    fixed_payoff,
    base_kind,
    base_destination,
    base_buy_water,
    base_buy_food,
    deviation_kind,
    deviation_destination,
    deviation_buy_water,
    deviation_buy_food,
    adjacency,
    is_mine_node,
    is_village_node,
    weather_is_storm,
    base_water,
    base_food,
    money_scale,
    water_price,
    food_price,
    water_weight,
    food_weight,
    weight_limit,
    mine_income,
    id_grid,
    position_to_row,
    bound_layer,
    village_residual,
    terminal_refund,
):
    action_count = len(deviation_kind)
    output = np.empty(action_count, dtype=np.float64)
    for row in range(action_count):
        output[row] = _profile_bound_one(
            row,
            deviating_player,
            continuation_day,
            deadline,
            end,
            failure_penalty,
            status,
            source,
            water0,
            food0,
            cash0,
            fixed_payoff,
            base_kind,
            base_destination,
            base_buy_water,
            base_buy_food,
            deviation_kind,
            deviation_destination,
            deviation_buy_water,
            deviation_buy_food,
            adjacency,
            is_mine_node,
            is_village_node,
            weather_is_storm,
            base_water,
            base_food,
            money_scale,
            water_price,
            food_price,
            water_weight,
            food_weight,
            weight_limit,
            mine_income,
            id_grid,
            position_to_row,
            bound_layer,
            village_residual,
            terminal_refund,
        )
    return output


if cuda is not None:

    @cuda.jit(device=True, inline=True)
    def _cuda_action_column(
        player, deviating_player, row, base_column, deviation_column
    ):
        if player == deviating_player:
            return deviation_column[row]
        return base_column[player]


    @cuda.jit
    def _screen_profile_bounds_cuda(
        output,
        deviating_player,
        continuation_day,
        deadline,
        end,
        failure_penalty,
        status,
        source,
        water0,
        food0,
        cash0,
        fixed_payoff,
        base_kind,
        base_destination,
        base_buy_water,
        base_buy_food,
        deviation_kind,
        deviation_destination,
        deviation_buy_water,
        deviation_buy_food,
        adjacency,
        is_mine_node,
        is_village_node,
        weather_is_storm,
        base_water,
        base_food,
        money_scale,
        water_price,
        food_price,
        water_weight,
        food_weight,
        weight_limit,
        mine_income,
        id_grid,
        position_to_row,
        bound_layer,
        village_residual,
        terminal_refund,
    ):
        row = cuda.grid(1)
        if row >= len(output):
            return
        n = len(status)
        dev_position = int(source[deviating_player])
        dev_water = int(water0[deviating_player])
        dev_food = int(food0[deviating_player])
        dev_cash = int(cash0[deviating_player])
        dev_kind = int(deviation_kind[row])
        valid = True

        for i in range(n):
            kind_i = int(_cuda_action_column(i, deviating_player, row, base_kind, deviation_kind))
            destination_i = int(_cuda_action_column(i, deviating_player, row, base_destination, deviation_destination))
            buy_w_i = int(_cuda_action_column(i, deviating_player, row, base_buy_water, deviation_buy_water))
            buy_f_i = int(_cuda_action_column(i, deviating_player, row, base_buy_food, deviation_buy_food))
            buyer_i = buy_w_i + buy_f_i > 0
            if status[i] != _STATUS_ACTIVE:
                if kind_i != _ACTION_INACTIVE or buyer_i:
                    valid = False
                continue
            if kind_i == _ACTION_FAIL:
                if buyer_i:
                    valid = False
                continue
            if kind_i == _ACTION_INACTIVE or kind_i == _ACTION_INITIAL_BUY:
                valid = False
                continue
            if buyer_i and not is_village_node[source[i]]:
                valid = False
                continue
            if kind_i == _ACTION_MINE and not is_mine_node[source[i]]:
                valid = False
                continue
            if kind_i == _ACTION_MOVE:
                if weather_is_storm or not adjacency[source[i], destination_i]:
                    valid = False
                    continue
            elif kind_i != _ACTION_STAY and kind_i != _ACTION_MINE:
                valid = False
                continue

            edge_count = 0
            mine_count = 0
            buyer_count = 0
            for j in range(n):
                if status[j] != _STATUS_ACTIVE:
                    continue
                kind_j = int(_cuda_action_column(j, deviating_player, row, base_kind, deviation_kind))
                destination_j = int(_cuda_action_column(j, deviating_player, row, base_destination, deviation_destination))
                buy_w_j = int(_cuda_action_column(j, deviating_player, row, base_buy_water, deviation_buy_water))
                buy_f_j = int(_cuda_action_column(j, deviating_player, row, base_buy_food, deviation_buy_food))
                if kind_i == _ACTION_MOVE and kind_j == _ACTION_MOVE and source[i] == source[j] and destination_i == destination_j:
                    edge_count += 1
                if kind_i == _ACTION_MINE and kind_j == _ACTION_MINE and source[i] == source[j]:
                    mine_count += 1
                if buyer_i and buy_w_j + buy_f_j > 0 and source[i] == source[j]:
                    buyer_count += 1

            price_factor = 0
            if buyer_i:
                if buyer_count <= 0:
                    valid = False
                    continue
                price_factor = 2 if buyer_count == 1 else 4
            purchase_cost = money_scale * price_factor * (water_price * buy_w_i + food_price * buy_f_i)
            water_pre = int(water0[i]) + buy_w_i
            food_pre = int(food0[i]) + buy_f_i
            cash_after_buy = int(cash0[i]) - purchase_cost
            if cash_after_buy < 0 or water_weight * water_pre + food_weight * food_pre > weight_limit:
                valid = False
                continue
            multiplier = 0
            income = 0
            next_position = int(source[i])
            if kind_i == _ACTION_STAY:
                multiplier = 1
            elif kind_i == _ACTION_MINE:
                if mine_count <= 0:
                    valid = False
                    continue
                multiplier = 3
                income = money_scale * mine_income // mine_count
            else:
                if edge_count <= 0:
                    valid = False
                    continue
                multiplier = 2 * edge_count
                next_position = destination_i
            next_water = water_pre - multiplier * base_water
            next_food = food_pre - multiplier * base_food
            if next_water < 0 or next_food < 0:
                valid = False
                continue
            if i == deviating_player:
                dev_position = next_position
                dev_water = next_water
                dev_food = next_food
                dev_cash = cash_after_buy + income

        if not valid:
            output[row] = -math.inf
        elif status[deviating_player] != _STATUS_ACTIVE:
            output[row] = fixed_payoff[deviating_player] / money_scale
        elif dev_kind == _ACTION_FAIL:
            output[row] = (cash0[deviating_player] - money_scale * failure_penalty) / money_scale
        else:
            resource_id = int(id_grid[dev_water, dev_food])
            if resource_id < 0:
                output[row] = -math.inf
            elif dev_position == end:
                output[row] = dev_cash / money_scale + terminal_refund[resource_id]
            else:
                if continuation_day >= deadline:
                    residual = -failure_penalty
                elif is_village_node[dev_position]:
                    residual = village_residual[dev_position]
                else:
                    position_row = int(position_to_row[dev_position])
                    if position_row < 0:
                        output[row] = -math.inf
                        return
                    residual = bound_layer[position_row, resource_id]
                output[row] = libdevice.nextafter(
                    dev_cash / money_scale + residual, math.inf
                )


class PurchaseLatticeOracle:
    """Screen complete unilateral action lattices without Python transitions."""

    def __init__(
        self,
        cfg: Q3Config,
        upper_bound: ResourceAwareSinglePlayerUpperBound,
        options: PurchaseOracleOptions | None = None,
    ) -> None:
        self.cfg = cfg
        self.upper_bound = upper_bound
        self.options = options or PurchaseOracleOptions()
        self.adjacency, self.is_mine, self.is_village = _topology_arrays(cfg)
        self.cuda_available = CUDA_PURCHASE_AVAILABLE
        self._cuda_static_cache: dict[int, tuple[np.ndarray, object]] = {}
        self._cpu_executor: ThreadPoolExecutor | None = None
        self._pyramid_cache: OrderedDict[
            tuple[int, int, int], _BoundMaxPyramid
        ] = OrderedDict()
        self._pyramid_cache_limit = 128
        if self.options.backend == "cuda" and not self.cuda_available:
            raise RuntimeError("CUDA purchase backend requested but unavailable")
        if self.cuda_available:
            if self.options.cuda_device >= len(cuda.gpus):
                raise ValueError("CUDA purchase device index is out of range")

    def _bound_arrays(self, continuation_day: int):
        if continuation_day < self.cfg.deadline:
            self.upper_bound._ensure_day(continuation_day)
            layer = self.upper_bound._layers[continuation_day]
            if layer is None:
                layer = np.empty((0, 0), dtype=np.float64)
        else:
            layer = np.empty((0, 0), dtype=np.float64)
        return (
            layer,
            np.asarray(
                self.upper_bound.fallback.residual[
                    min(continuation_day, self.cfg.deadline)
                ],
                dtype=np.float64,
            ),
        )

    def _arguments(
        self,
        continuation_day: int,
        state: JointState,
        base_actions: tuple[Action, ...],
        player: int,
        actions: IndividualActionArrays,
        weather: Weather,
    ) -> tuple:
        layer, village_residual = self._bound_arrays(continuation_day)
        return (
            player,
            continuation_day,
            self.cfg.deadline,
            self.cfg.end,
            self.cfg.failure_penalty,
            np.asarray([int(item.status) for item in state], dtype=np.int8),
            np.asarray([item.position for item in state], dtype=np.int16),
            np.asarray([item.water for item in state], dtype=np.int32),
            np.asarray([item.food for item in state], dtype=np.int32),
            np.asarray([item.cash_scaled for item in state], dtype=np.int64),
            np.asarray([item.fixed_payoff_scaled for item in state], dtype=np.int64),
            np.asarray([int(action.kind) for action in base_actions], dtype=np.int8),
            np.asarray([action.destination for action in base_actions], dtype=np.int16),
            np.asarray([action.buy_water for action in base_actions], dtype=np.int32),
            np.asarray([action.buy_food for action in base_actions], dtype=np.int32),
            actions.kind,
            actions.destination,
            actions.buy_water,
            actions.buy_food,
            self.adjacency,
            self.is_mine,
            self.is_village,
            weather == "sandstorm",
            self.cfg.water_consume[weather],
            self.cfg.food_consume[weather],
            self.cfg.money_scale,
            self.cfg.water_price,
            self.cfg.food_price,
            self.cfg.water_weight,
            self.cfg.food_weight,
            self.cfg.weight_limit,
            self.cfg.mine_income,
            self.upper_bound.resources.id_grid,
            self.upper_bound._position_to_row,
            layer,
            village_residual,
            self.upper_bound._terminal_refund_by_id,
        )

    def _residual_by_resource(
        self, continuation_day: int, position: int
    ) -> np.ndarray:
        if position == self.cfg.end:
            return self.upper_bound._terminal_refund_by_id
        if continuation_day >= self.cfg.deadline:
            return np.full(
                len(self.upper_bound.resources.water),
                -float(self.cfg.failure_penalty),
                dtype=np.float64,
            )
        if position in self.cfg.villages:
            return np.full(
                len(self.upper_bound.resources.water),
                float(
                    self.upper_bound.fallback.residual[
                        continuation_day, position
                    ]
                ),
                dtype=np.float64,
            )
        self.upper_bound._ensure_day(continuation_day)
        layer = self.upper_bound._layers[continuation_day]
        if layer is None:
            raise AssertionError("resource upper-bound layer was not built")
        position_row = int(self.upper_bound._position_to_row[position])
        if position_row < 0:
            raise ValueError("position is outside the resource-bound map")
        return layer[position_row]

    def _pyramid(
        self, continuation_day: int, position: int, price_factor: int
    ) -> _BoundMaxPyramid:
        key = (continuation_day, position, price_factor)
        cached = self._pyramid_cache.get(key)
        if cached is not None:
            self._pyramid_cache.move_to_end(key)
            return cached
        residual = self._residual_by_resource(continuation_day, position)
        resources = self.upper_bound.resources
        purchase_value = price_factor * (
            self.cfg.water_price * resources.water.astype(np.float64)
            + self.cfg.food_price * resources.food.astype(np.float64)
        )
        transformed = np.nextafter(residual - purchase_value, np.inf)
        base = np.full(resources.id_grid.shape, -np.inf, dtype=np.float64)
        base[resources.water, resources.food] = transformed
        levels = [base]
        while levels[-1].shape != (1, 1):
            levels.append(_coarsen_maximum(levels[-1]))
        result = _BoundMaxPyramid(tuple(levels))
        self._pyramid_cache[key] = result
        if len(self._pyramid_cache) > self._pyramid_cache_limit:
            self._pyramid_cache.popitem(last=False)
        return result

    @staticmethod
    def _find_action_index(
        actions: IndividualActionArrays,
        start: int,
        stop: int,
        buy_water: int,
        buy_food: int,
    ) -> int:
        low = start
        high = stop
        target = (buy_water, buy_food)
        while low < high:
            middle = (low + high) // 2
            current = (
                int(actions.buy_water[middle]),
                int(actions.buy_food[middle]),
            )
            if current < target:
                low = middle + 1
            else:
                high = middle
        if (
            low < stop
            and int(actions.buy_water[low]) == buy_water
            and int(actions.buy_food[low]) == buy_food
        ):
            return low
        return -1

    def _screen_positive_skeleton(
        self,
        continuation_day: int,
        state: JointState,
        base_actions: tuple[Action, ...],
        player: int,
        actions: IndividualActionArrays,
        start: int,
        stop: int,
        weather: Weather,
        threshold: float,
    ) -> tuple[list[int], float, int]:
        representative = actions.action_at(start)
        if not representative.is_buyer:
            raise AssertionError("positive skeleton starts with a zero purchase")
        profile = list(base_actions)
        profile[player] = representative
        counts = count_interactions_scalar(state, profile)
        for opponent in range(self.cfg.n_players):
            if opponent == player:
                continue
            if apply_player_action_given_counts(
                self.cfg,
                state[opponent],
                profile[opponent],
                weather,
                edge_count=counts.edge[opponent],
                mine_count=counts.mine[opponent],
                village_buyer_count=counts.village_buyers[opponent],
            ) is None:
                return [], -math.inf, 0

        original = state[player]
        kind = representative.kind
        if kind is ActionKind.STAY:
            multiplier = 1
            next_position = original.position
            income_scaled = 0
        elif kind is ActionKind.MINE:
            mine_count = counts.mine[player]
            if mine_count <= 0:
                return [], -math.inf, 0
            multiplier = 3
            next_position = original.position
            income_scaled = self.cfg.money_scale * self.cfg.mine_income // mine_count
        elif kind is ActionKind.MOVE:
            edge_count = counts.edge[player]
            if edge_count <= 0:
                return [], -math.inf, 0
            multiplier = 2 * edge_count
            next_position = representative.destination
            income_scaled = 0
        else:
            return [], -math.inf, 0

        buyer_count = counts.village_buyers[player]
        if buyer_count <= 0:
            return [], -math.inf, 0
        price_factor = 2 if buyer_count == 1 else 4
        consume_water = multiplier * self.cfg.water_consume[weather]
        consume_food = multiplier * self.cfg.food_consume[weather]
        minimum_water = max(0, original.water - consume_water)
        minimum_food = max(0, original.food - consume_food)
        maximum_pre_weight = self.cfg.weight_limit
        pyramid = self._pyramid(
            continuation_day, next_position, price_factor
        )
        constant = (
            (original.cash_scaled + income_scaled) / self.cfg.money_scale
            + price_factor
            * (
                self.cfg.water_price * (original.water - consume_water)
                + self.cfg.food_price * (original.food - consume_food)
            )
        )
        survivors: list[int] = []
        max_pruned = -math.inf
        pruned_nodes = 0
        top_level = len(pyramid.levels) - 1
        stack = [
            (top_level, row, column)
            for row in range(pyramid.levels[top_level].shape[0])
            for column in range(pyramid.levels[top_level].shape[1])
        ]
        grid_rows, grid_columns = pyramid.levels[0].shape
        while stack:
            level, row, column = stack.pop()
            scale = 1 << level
            low_water = row * scale
            low_food = column * scale
            high_water = min(grid_rows - 1, low_water + scale - 1)
            high_food = min(grid_columns - 1, low_food + scale - 1)
            if high_water < minimum_water or high_food < minimum_food:
                continue
            feasible_low_water = max(low_water, minimum_water)
            feasible_low_food = max(low_food, minimum_food)
            buy_water_min = feasible_low_water + consume_water - original.water
            buy_food_min = feasible_low_food + consume_food - original.food
            if buy_water_min < 0 or buy_food_min < 0:
                continue
            if (
                self.cfg.water_weight * (feasible_low_water + consume_water)
                + self.cfg.food_weight * (feasible_low_food + consume_food)
                > maximum_pre_weight
            ):
                continue
            minimum_cost_scaled = (
                self.cfg.money_scale
                * price_factor
                * (
                    self.cfg.water_price * buy_water_min
                    + self.cfg.food_price * buy_food_min
                )
            )
            if minimum_cost_scaled > original.cash_scaled:
                continue
            node_score = float(pyramid.levels[level][row, column])
            if not np.isfinite(node_score):
                continue
            node_bound = math.nextafter(constant + node_score, math.inf)
            if node_bound < threshold:
                max_pruned = max(max_pruned, node_bound)
                pruned_nodes += 1
                continue
            if level:
                child_level = level - 1
                child_shape = pyramid.levels[child_level].shape
                for child_row in (2 * row, 2 * row + 1):
                    for child_column in (2 * column, 2 * column + 1):
                        if (
                            child_row < child_shape[0]
                            and child_column < child_shape[1]
                        ):
                            stack.append(
                                (child_level, child_row, child_column)
                            )
                continue

            buy_water = row + consume_water - original.water
            buy_food = column + consume_food - original.food
            if buy_water < 0 or buy_food < 0 or buy_water + buy_food == 0:
                continue
            if (
                self.cfg.water_weight * (row + consume_water)
                + self.cfg.food_weight * (column + consume_food)
                > self.cfg.weight_limit
            ):
                continue
            purchase_cost_scaled = (
                self.cfg.money_scale
                * price_factor
                * (
                    self.cfg.water_price * buy_water
                    + self.cfg.food_price * buy_food
                )
            )
            if purchase_cost_scaled > original.cash_scaled:
                continue
            action_index = self._find_action_index(
                actions, start, stop, buy_water, buy_food
            )
            if action_index >= 0:
                survivors.append(action_index)
        return survivors, max_pruned, pruned_nodes

    def _cpu_bounds(self, arguments: tuple, action_count: int) -> tuple[np.ndarray, str]:
        requested = self.options.threads or min(64, os.cpu_count() or 1)
        if (
            not NUMBA_PURCHASE_AVAILABLE
            or action_count < self.options.parallel_min_actions
            or requested == 1
        ):
            return _screen_profile_bounds_serial(*arguments), "numba-cpu"
        # Compile once before dispatch.  The nopython kernel releases the GIL,
        # so these are native scans; recursive successor solving remains
        # single-threaded and deterministic.
        warm = list(arguments)
        for index in (15, 16, 17, 18):
            warm[index] = warm[index][:1]
        _screen_profile_bounds_serial(*warm)
        if self._cpu_executor is None:
            self._cpu_executor = ThreadPoolExecutor(
                max_workers=requested, thread_name_prefix="q3-purchase"
            )
        worker_count = min(requested, action_count)
        edges = np.linspace(0, action_count, worker_count + 1, dtype=np.int64)

        def submit(start: int, stop: int):
            chunk = list(arguments)
            for index in (15, 16, 17, 18):
                chunk[index] = chunk[index][start:stop]
            return self._cpu_executor.submit(
                _screen_profile_bounds_serial, *chunk
            )

        futures = [
            submit(int(edges[index]), int(edges[index + 1]))
            for index in range(worker_count)
            if edges[index] < edges[index + 1]
        ]
        return np.concatenate([future.result() for future in futures]), "numba-threaded"

    def close(self) -> None:
        if self._cpu_executor is not None:
            self._cpu_executor.shutdown(wait=True, cancel_futures=True)
            self._cpu_executor = None
        self._cuda_static_cache.clear()
        self._pyramid_cache.clear()

    @property
    def cache_entries(self) -> int:
        return len(self._pyramid_cache)

    @property
    def cache_bytes(self) -> int:
        return sum(
            level.nbytes
            for pyramid in self._pyramid_cache.values()
            for level in pyramid.levels
        )

    def _cuda_bounds(self, arguments: tuple, action_count: int) -> np.ndarray:
        cuda.select_device(self.options.cuda_device)
        device_arguments = []
        static_array_indices = {19, 20, 21, 32, 33, 34, 36}
        for index, argument in enumerate(arguments):
            if isinstance(argument, np.ndarray):
                if index in static_array_indices:
                    cache_key = id(argument)
                    cached = self._cuda_static_cache.get(cache_key)
                    if cached is None:
                        device = cuda.to_device(np.ascontiguousarray(argument))
                        self._cuda_static_cache[cache_key] = (argument, device)
                    else:
                        _, device = cached
                    device_arguments.append(device)
                else:
                    device_arguments.append(
                        cuda.to_device(np.ascontiguousarray(argument))
                    )
            else:
                device_arguments.append(argument)
        output = cuda.device_array(action_count, dtype=np.float64)
        threads = 256
        blocks = (action_count + threads - 1) // threads
        _screen_profile_bounds_cuda[blocks, threads](output, *device_arguments)
        return output.copy_to_host()

    def screen(
        self,
        continuation_day: int,
        state: JointState,
        base_actions: tuple[Action, ...],
        player: int,
        actions: IndividualActionArrays,
        weather: Weather,
        threshold: float,
    ) -> PurchaseScreenResult:
        started = perf_counter()
        arguments = self._arguments(
            continuation_day, state, base_actions, player, actions, weather
        )
        has_positive_lattice = any(
            int(actions.buy_water[start]) + int(actions.buy_food[start]) > 0
            for start, _ in actions.skeleton_ranges
        )
        if self.options.backend in {"auto", "cpu"} and has_positive_lattice:
            survivor_indices: list[int] = []
            generic_indices: list[int] = []
            max_pruned = -math.inf
            regions_pruned = 0
            for start, stop in actions.skeleton_ranges:
                if int(actions.buy_water[start]) + int(actions.buy_food[start]) > 0:
                    selected, skeleton_pruned, skeleton_regions = (
                        self._screen_positive_skeleton(
                            continuation_day,
                            state,
                            base_actions,
                            player,
                            actions,
                            start,
                            stop,
                            weather,
                            threshold,
                        )
                    )
                    survivor_indices.extend(selected)
                    max_pruned = max(max_pruned, skeleton_pruned)
                    regions_pruned += skeleton_regions
                else:
                    generic_indices.extend(range(start, stop))

            exact_indices = np.asarray(
                sorted((*generic_indices, *survivor_indices)), dtype=np.int64
            )
            exact_bounds = np.empty(0, dtype=np.float64)
            if len(exact_indices):
                exact_arguments = list(arguments)
                for column, argument_index in enumerate((15, 16, 17, 18)):
                    exact_arguments[argument_index] = (
                        actions.kind,
                        actions.destination,
                        actions.buy_water,
                        actions.buy_food,
                    )[column][exact_indices]
                exact_bounds = _screen_profile_bounds_serial(*exact_arguments)
            exact_valid = np.isfinite(exact_bounds)
            exact_keep = exact_valid & (exact_bounds >= threshold)
            exact_pruned = exact_valid & ~exact_keep
            if np.any(exact_pruned):
                max_pruned = max(
                    max_pruned, float(np.max(exact_bounds[exact_pruned]))
                )
            retained_indices = exact_indices[exact_keep]
            retained_bounds = exact_bounds[exact_keep]
            # Positive regions rejected by the pyramid are validly bounded,
            # while points failing joint cash/resource checks are harmless.
            pruned_count = max(0, len(actions) - len(retained_indices))
            return PurchaseScreenResult(
                survivor_indices=retained_indices,
                survivor_bounds=retained_bounds.copy(),
                valid_count=len(actions),
                pruned_count=pruned_count,
                max_pruned_bound=max_pruned,
                regions_pruned=regions_pruned,
                backend="numba-pyramid",
                elapsed_seconds=perf_counter() - started,
            )
        use_cuda = (
            self.options.backend == "cuda"
            and self.cuda_available
            and len(actions) >= self.options.cuda_min_actions
        )
        if use_cuda:
            bounds = self._cuda_bounds(arguments, len(actions))
            backend = "cuda"
        else:
            bounds, backend = self._cpu_bounds(arguments, len(actions))
        valid = np.isfinite(bounds)
        keep = valid & (bounds >= threshold)
        pruned = valid & ~keep
        survivor_indices = np.flatnonzero(keep).astype(np.int64, copy=False)
        max_pruned = (
            float(np.max(bounds[pruned])) if np.any(pruned) else -math.inf
        )
        return PurchaseScreenResult(
            survivor_indices=survivor_indices,
            survivor_bounds=bounds[keep].copy(),
            valid_count=int(np.sum(valid)),
            pruned_count=int(np.sum(pruned)),
            max_pruned_bound=max_pruned,
            regions_pruned=0,
            backend=backend,
            elapsed_seconds=perf_counter() - started,
        )
