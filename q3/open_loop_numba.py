"""Compact Numba frontier kernel for the three-player, no-village Q3.1 game."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Q3Config
from .model import ActionKind, PlayerState, Status
from .resource_index import ResourceIndex

try:
    from numba import njit

    NUMBA_OPEN_LOOP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the optional wheel
    NUMBA_OPEN_LOOP_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorate(function):
            return function

        return decorate


_DEV_BITS = 21
_OPP_BITS = 23
_OPP_LOW_BITS = 20
_DEV_MASK = (1 << _DEV_BITS) - 1
_OPP_MASK = (1 << _OPP_BITS) - 1
_OPP_LOW_MASK = (1 << _OPP_LOW_BITS) - 1
_NEG_PAYOFF = -(1 << 62)
_STATUS_ACTIVE = 0
_STATUS_FINISHED = 1
_STATUS_FAILED = 2
_ACTION_FAIL = 1
_ACTION_STAY = 2
_ACTION_MINE = 3
_ACTION_MOVE = 4


@dataclass(frozen=True)
class CompactKernelConfig:
    resources: ResourceIndex
    neighbors: np.ndarray
    degree: np.ndarray
    is_mine: np.ndarray

    @classmethod
    def build(cls, cfg: Q3Config) -> "CompactKernelConfig":
        if cfg.n_players != 3:
            raise ValueError("compact Q3.1 kernel requires exactly three players")
        if cfg.villages:
            raise ValueError("compact Q3.1 kernel currently requires no villages")
        if max(cfg.nodes) >= 16:
            raise ValueError("compact Q3.1 node ids must fit four bits")
        resources = ResourceIndex.build(cfg)
        if len(resources.water) >= 1 << 17:
            raise ValueError("compact Q3.1 resource ids must fit seventeen bits")
        max_degree = max(len(cfg.adj[node]) for node in cfg.nodes)
        neighbors = np.full((max(cfg.nodes) + 1, max_degree), -1, dtype=np.int8)
        degree = np.zeros(max(cfg.nodes) + 1, dtype=np.int8)
        is_mine = np.zeros(max(cfg.nodes) + 1, dtype=np.bool_)
        for node in cfg.nodes:
            adjacent = cfg.adj[node]
            degree[node] = len(adjacent)
            neighbors[node, : len(adjacent)] = adjacent
            is_mine[node] = node in cfg.mines
        return cls(resources, neighbors, degree, is_mine)


def pack_frontier_state(
    resources: ResourceIndex,
    deviator: PlayerState,
    opponent0: PlayerState,
    opponent1: PlayerState,
) -> tuple[np.uint64, np.uint8]:
    if deviator.status is not Status.ACTIVE:
        raise ValueError("frontier contains only active deviator states")

    def resource_id(player: PlayerState) -> int:
        return resources.encode(player.water, player.food) if player.status is Status.ACTIVE else 0

    dev_code = deviator.position | (resource_id(deviator) << 4)
    opp0_code = (
        int(opponent0.status)
        | (opponent0.position << 2)
        | (resource_id(opponent0) << 6)
    )
    opp1_code = (
        int(opponent1.status)
        | (opponent1.position << 2)
        | (resource_id(opponent1) << 6)
    )
    low = (
        dev_code
        | (opp0_code << _DEV_BITS)
        | ((opp1_code & _OPP_LOW_MASK) << (_DEV_BITS + _OPP_BITS))
    )
    high = opp1_code >> _OPP_LOW_BITS
    return np.uint64(low), np.uint8(high)


@njit(cache=True, inline="always")
def _decode_key(low, high):
    dev_code = low & np.uint64(_DEV_MASK)
    opp0_code = (low >> np.uint64(_DEV_BITS)) & np.uint64(_OPP_MASK)
    opp1_code = (
        (low >> np.uint64(_DEV_BITS + _OPP_BITS)) & np.uint64(_OPP_LOW_MASK)
    ) | (np.uint64(high) << np.uint64(_OPP_LOW_BITS))
    dpos = int(dev_code & np.uint64(15))
    drid = int(dev_code >> np.uint64(4))
    o0status = int(opp0_code & np.uint64(3))
    o0pos = int((opp0_code >> np.uint64(2)) & np.uint64(15))
    o0rid = int(opp0_code >> np.uint64(6))
    o1status = int(opp1_code & np.uint64(3))
    o1pos = int((opp1_code >> np.uint64(2)) & np.uint64(15))
    o1rid = int(opp1_code >> np.uint64(6))
    return dpos, drid, o0status, o0pos, o0rid, o1status, o1pos, o1rid


@njit(cache=True, inline="always")
def _pack_key(dpos, drid, o0status, o0pos, o0rid, o1status, o1pos, o1rid):
    dev_code = np.uint64(dpos) | (np.uint64(drid) << np.uint64(4))
    opp0_code = (
        np.uint64(o0status)
        | (np.uint64(o0pos) << np.uint64(2))
        | (np.uint64(o0rid) << np.uint64(6))
    )
    opp1_code = (
        np.uint64(o1status)
        | (np.uint64(o1pos) << np.uint64(2))
        | (np.uint64(o1rid) << np.uint64(6))
    )
    low = (
        dev_code
        | (opp0_code << np.uint64(_DEV_BITS))
        | (
            (opp1_code & np.uint64(_OPP_LOW_MASK))
            << np.uint64(_DEV_BITS + _OPP_BITS)
        )
    )
    high = np.uint8(opp1_code >> np.uint64(_OPP_LOW_BITS))
    return low, high


@njit(cache=True, inline="always")
def _is_adjacent(source, destination, neighbors, degree):
    for index in range(int(degree[source])):
        if int(neighbors[source, index]) == destination:
            return True
    return False


@njit(cache=True, inline="always")
def _advance_opponent(
    status,
    position,
    resource_id,
    action_kind,
    destination,
    edge_count,
    weather_is_storm,
    base_water,
    base_food,
    end,
    water_by_id,
    food_by_id,
    id_grid,
    neighbors,
    degree,
    is_mine,
):
    if status != _STATUS_ACTIVE:
        return status, 0, 0
    if action_kind == _ACTION_FAIL:
        return _STATUS_FAILED, 0, 0
    if action_kind == _ACTION_STAY:
        multiplier = 1
        next_position = position
    elif action_kind == _ACTION_MINE:
        if not is_mine[position]:
            return _STATUS_FAILED, 0, 0
        multiplier = 3
        next_position = position
    elif action_kind == _ACTION_MOVE:
        if weather_is_storm or not _is_adjacent(
            position, destination, neighbors, degree
        ):
            return _STATUS_FAILED, 0, 0
        multiplier = 2 * edge_count
        next_position = destination
    else:
        return _STATUS_FAILED, 0, 0
    water = int(water_by_id[resource_id]) - multiplier * base_water
    food = int(food_by_id[resource_id]) - multiplier * base_food
    if water < 0 or food < 0:
        return _STATUS_FAILED, 0, 0
    if next_position == end:
        return _STATUS_FINISHED, 0, 0
    return _STATUS_ACTIVE, next_position, int(id_grid[water, food])


@njit(cache=True, inline="always")
def _evaluate_action(
    dpos,
    drid,
    cash,
    o0status,
    o0pos,
    o0rid,
    o1status,
    o1pos,
    o1rid,
    dev_kind,
    dev_destination,
    opp_kind,
    opp_destination,
    weather_is_storm,
    base_water,
    base_food,
    day_is_deadline,
    end,
    mine_income_scaled,
    failure_penalty_scaled,
    money_scale,
    water_price,
    food_price,
    water_by_id,
    food_by_id,
    id_grid,
    neighbors,
    degree,
    is_mine,
):
    o0move = o0status == _STATUS_ACTIVE and opp_kind[0] == _ACTION_MOVE
    o1move = o1status == _STATUS_ACTIVE and opp_kind[1] == _ACTION_MOVE
    devmove = dev_kind == _ACTION_MOVE

    dev_edge_count = 0
    if devmove:
        dev_edge_count = 1
        if o0move and o0pos == dpos and int(opp_destination[0]) == dev_destination:
            dev_edge_count += 1
        if o1move and o1pos == dpos and int(opp_destination[1]) == dev_destination:
            dev_edge_count += 1
    o0_edge_count = 0
    if o0move:
        o0_edge_count = 1
        if devmove and dpos == o0pos and dev_destination == int(opp_destination[0]):
            o0_edge_count += 1
        if (
            o1move
            and o1pos == o0pos
            and int(opp_destination[1]) == int(opp_destination[0])
        ):
            o0_edge_count += 1
    o1_edge_count = 0
    if o1move:
        o1_edge_count = 1
        if devmove and dpos == o1pos and dev_destination == int(opp_destination[1]):
            o1_edge_count += 1
        if (
            o0move
            and o0pos == o1pos
            and int(opp_destination[0]) == int(opp_destination[1])
        ):
            o1_edge_count += 1

    dev_mine_count = 0
    if dev_kind == _ACTION_MINE:
        dev_mine_count = 1
        if (
            o0status == _STATUS_ACTIVE
            and opp_kind[0] == _ACTION_MINE
            and o0pos == dpos
        ):
            dev_mine_count += 1
        if (
            o1status == _STATUS_ACTIVE
            and opp_kind[1] == _ACTION_MINE
            and o1pos == dpos
        ):
            dev_mine_count += 1

    dev_water = int(water_by_id[drid])
    dev_food = int(food_by_id[drid])
    if dev_kind == _ACTION_STAY:
        multiplier = 1
        next_position = dpos
        income = 0
    elif dev_kind == _ACTION_MINE:
        multiplier = 3
        next_position = dpos
        income = mine_income_scaled // dev_mine_count
    else:
        multiplier = 2 * dev_edge_count
        next_position = dev_destination
        income = 0
    next_water = dev_water - multiplier * base_water
    next_food = dev_food - multiplier * base_food
    if next_water < 0 or next_food < 0:
        return 1, cash - failure_penalty_scaled, 0, np.uint64(0), np.uint8(0), cash
    next_cash = cash + income
    if next_position == end:
        refund = (
            money_scale * water_price * next_water
            + money_scale * food_price * next_food
        ) // 2
        return 1, next_cash + refund, 1, np.uint64(0), np.uint8(0), next_cash
    if day_is_deadline:
        return 1, next_cash - failure_penalty_scaled, 0, np.uint64(0), np.uint8(0), next_cash

    next_drid = int(id_grid[next_water, next_food])
    n0status, n0pos, n0rid = _advance_opponent(
        o0status,
        o0pos,
        o0rid,
        int(opp_kind[0]),
        int(opp_destination[0]),
        o0_edge_count,
        weather_is_storm,
        base_water,
        base_food,
        end,
        water_by_id,
        food_by_id,
        id_grid,
        neighbors,
        degree,
        is_mine,
    )
    n1status, n1pos, n1rid = _advance_opponent(
        o1status,
        o1pos,
        o1rid,
        int(opp_kind[1]),
        int(opp_destination[1]),
        o1_edge_count,
        weather_is_storm,
        base_water,
        base_food,
        end,
        water_by_id,
        food_by_id,
        id_grid,
        neighbors,
        degree,
        is_mine,
    )
    next_low, next_high = _pack_key(
        next_position,
        next_drid,
        n0status,
        n0pos,
        n0rid,
        n1status,
        n1pos,
        n1rid,
    )
    return 0, 0, 0, next_low, next_high, next_cash


@njit(cache=True, inline="always")
def _hash_slot(low, high, mask):
    value = low ^ (np.uint64(high) * np.uint64(0x9E3779B97F4A7C15))
    value ^= value >> np.uint64(30)
    value *= np.uint64(0xBF58476D1CE4E5B9)
    value ^= value >> np.uint64(27)
    value *= np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    return np.int64(value & np.uint64(mask))


@njit(cache=True, inline="always")
def _insert_state(
    low,
    high,
    cash,
    parent,
    action_code,
    table_low,
    table_high,
    table_cash,
    table_parent,
    table_action,
    occupied,
    mask,
    count,
    max_states,
):
    slot = _hash_slot(low, high, mask)
    while occupied[slot]:
        if table_low[slot] == low and table_high[slot] == high:
            if cash > table_cash[slot] or (
                cash == table_cash[slot]
                and (
                    action_code < table_action[slot]
                    or (
                        action_code == table_action[slot]
                        and parent < table_parent[slot]
                    )
                )
            ):
                table_cash[slot] = cash
                table_parent[slot] = parent
                table_action[slot] = action_code
            return count, 0
        slot = np.int64((slot + 1) & mask)
    if count >= max_states:
        return count, 1
    occupied[slot] = 1
    table_low[slot] = low
    table_high[slot] = high
    table_cash[slot] = cash
    table_parent[slot] = parent
    table_action[slot] = action_code
    return count + 1, 0


@njit(cache=True)
def expand_frontier_numba(
    current_low,
    current_high,
    current_cash,
    opp_kind,
    opp_destination,
    weather_is_storm,
    base_water,
    base_food,
    day_is_deadline,
    end,
    mine_income_scaled,
    failure_penalty_scaled,
    money_scale,
    water_price,
    food_price,
    water_by_id,
    food_by_id,
    id_grid,
    neighbors,
    degree,
    is_mine,
    table_low,
    table_high,
    table_cash,
    table_parent,
    table_action,
    occupied,
    max_states,
):
    count = 0
    overflow = 0
    best_payoff = np.int64(_NEG_PAYOFF)
    best_success = np.int8(0)
    best_parent = np.int32(-1)
    best_action = np.uint8(255)
    mask = len(occupied) - 1

    for row in range(len(current_low)):
        (
            dpos,
            drid,
            o0status,
            o0pos,
            o0rid,
            o1status,
            o1pos,
            o1rid,
        ) = _decode_key(current_low[row], current_high[row])
        water = int(water_by_id[drid])
        food = int(food_by_id[drid])
        has_action = False

        # At most one stay, one mine, and degree[dpos] move actions.
        for action_slot in range(2 + int(degree[dpos])):
            if action_slot == 0:
                dev_kind = _ACTION_STAY
                destination = 0
                individually_feasible = water >= base_water and food >= base_food
            elif action_slot == 1:
                dev_kind = _ACTION_MINE
                destination = 0
                individually_feasible = (
                    is_mine[dpos]
                    and water >= 3 * base_water
                    and food >= 3 * base_food
                )
            else:
                dev_kind = _ACTION_MOVE
                destination = int(neighbors[dpos, action_slot - 2])
                individually_feasible = (
                    not weather_is_storm
                    and water >= 2 * base_water
                    and food >= 2 * base_food
                )
            if not individually_feasible:
                continue
            has_action = True
            terminal, payoff, success, next_low, next_high, next_cash = _evaluate_action(
                dpos,
                drid,
                current_cash[row],
                o0status,
                o0pos,
                o0rid,
                o1status,
                o1pos,
                o1rid,
                dev_kind,
                destination,
                opp_kind,
                opp_destination,
                weather_is_storm,
                base_water,
                base_food,
                day_is_deadline,
                end,
                mine_income_scaled,
                failure_penalty_scaled,
                money_scale,
                water_price,
                food_price,
                water_by_id,
                food_by_id,
                id_grid,
                neighbors,
                degree,
                is_mine,
            )
            action_code = np.uint8((dev_kind << 4) | destination)
            if terminal:
                if payoff > best_payoff or (
                    payoff == best_payoff
                    and (
                        success > best_success
                        or (
                            success == best_success
                            and (
                                action_code < best_action
                                or (
                                    action_code == best_action
                                    and row < best_parent
                                )
                            )
                        )
                    )
                ):
                    best_payoff = payoff
                    best_success = success
                    best_parent = row
                    best_action = action_code
            else:
                count, failed = _insert_state(
                    next_low,
                    next_high,
                    next_cash,
                    np.int32(row),
                    action_code,
                    table_low,
                    table_high,
                    table_cash,
                    table_parent,
                    table_action,
                    occupied,
                    mask,
                    count,
                    max_states,
                )
                if failed:
                    overflow = 1
                    return (
                        count,
                        overflow,
                        best_payoff,
                        best_success,
                        best_parent,
                        best_action,
                    )

        if not has_action:
            payoff = current_cash[row] - failure_penalty_scaled
            action_code = np.uint8(_ACTION_FAIL << 4)
            if payoff > best_payoff or (
                payoff == best_payoff
                and (
                    action_code < best_action
                    or (action_code == best_action and row < best_parent)
                )
            ):
                best_payoff = payoff
                best_success = 0
                best_parent = row
                best_action = action_code

    return count, overflow, best_payoff, best_success, best_parent, best_action


def hash_capacity(max_states: int) -> int:
    capacity = 1
    target = max(2, max_states * 2)
    while capacity < target:
        capacity <<= 1
    return capacity


def allocate_hash_table(max_states: int):
    capacity = hash_capacity(max_states)
    return (
        np.zeros(capacity, dtype=np.uint64),
        np.zeros(capacity, dtype=np.uint8),
        np.zeros(capacity, dtype=np.int64),
        np.zeros(capacity, dtype=np.int32),
        np.zeros(capacity, dtype=np.uint8),
        np.zeros(capacity, dtype=np.uint8),
    )


def compact_hash_table(table, count: int):
    table_low, table_high, table_cash, table_parent, table_action, occupied = table
    slots = np.flatnonzero(occupied)
    if len(slots) != count:
        raise AssertionError("compact frontier hash count mismatch")
    return (
        table_low[slots].copy(),
        table_high[slots].copy(),
        table_cash[slots].copy(),
        table_parent[slots].copy(),
        table_action[slots].copy(),
    )


def clear_hash_table(table) -> None:
    table[-1].fill(0)


def decode_action_code(code: int):
    kind = ActionKind(code >> 4)
    destination = code & 15
    return kind, destination
