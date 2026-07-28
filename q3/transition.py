"""Independent scalar and lossless NumPy-vectorized Q3 joint transitions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .data import Q3Config, Weather
from .interaction import count_interactions_batch
from .model import (
    Action,
    ActionKind,
    JointState,
    PlayerState,
    Status,
    fail_player,
    finish_player,
    weight_ok,
)


@dataclass(frozen=True, slots=True)
class InteractionCounts:
    edge: tuple[int, ...]
    mine: tuple[int, ...]
    village_buyers: tuple[int, ...]


@dataclass(frozen=True)
class BatchTransitionResult:
    valid: np.ndarray
    successors: tuple[JointState | None, ...]
    edge_count: np.ndarray
    mine_count: np.ndarray
    village_buyer_count: np.ndarray


def _validate_profile_size(
    cfg: Q3Config, state: JointState, actions: Sequence[Action]
) -> None:
    if len(state) != cfg.n_players or len(actions) != cfg.n_players:
        raise ValueError("state/action player count does not match configuration")


def count_interactions_scalar(
    state: JointState, actions: Sequence[Action]
) -> InteractionCounts:
    n = len(state)
    edge = [0] * n
    mine = [0] * n
    buyers = [0] * n
    for i, (player_i, action_i) in enumerate(zip(state, actions, strict=True)):
        if player_i.status is not Status.ACTIVE:
            continue
        if action_i.kind is ActionKind.MOVE:
            edge[i] = sum(
                player_j.status is Status.ACTIVE
                and action_j.kind is ActionKind.MOVE
                and player_j.position == player_i.position
                and action_j.destination == action_i.destination
                for player_j, action_j in zip(state, actions, strict=True)
            )
        if action_i.kind is ActionKind.MINE:
            mine[i] = sum(
                player_j.status is Status.ACTIVE
                and action_j.kind is ActionKind.MINE
                and player_j.position == player_i.position
                for player_j, action_j in zip(state, actions, strict=True)
            )
        if action_i.is_buyer:
            buyers[i] = sum(
                player_j.status is Status.ACTIVE
                and action_j.is_buyer
                and player_j.position == player_i.position
                for player_j, action_j in zip(state, actions, strict=True)
            )
    return InteractionCounts(tuple(edge), tuple(mine), tuple(buyers))


def apply_initial_purchases(
    cfg: Q3Config, state: JointState, actions: Sequence[Action]
) -> JointState | None:
    _validate_profile_size(cfg, state, actions)
    out: list[PlayerState] = []
    for player, action in zip(state, actions, strict=True):
        if (
            player.status is not Status.ACTIVE
            or action.kind is not ActionKind.INITIAL_BUY
        ):
            return None
        water = player.water + action.buy_water
        food = player.food + action.buy_food
        cost = cfg.money_scale * (
            cfg.water_price * action.buy_water + cfg.food_price * action.buy_food
        )
        if cost > player.cash_scaled or not weight_ok(cfg, water, food):
            return None
        out.append(
            PlayerState(
                Status.ACTIVE,
                position=player.position,
                water=water,
                food=food,
                cash_scaled=player.cash_scaled - cost,
            )
        )
    return tuple(out)


def apply_player_action_given_counts(
    cfg: Q3Config,
    player: PlayerState,
    action: Action,
    weather: Weather,
    *,
    edge_count: int,
    mine_count: int,
    village_buyer_count: int,
) -> PlayerState | None:
    """Apply one player's action after the joint interaction counts are fixed.

    The interaction counts depend only on the action skeleton (main action,
    destination, and whether a purchase is positive), not on purchase amounts.
    Keeping this scalar rule in one place lets the structured profile iterator
    remove infeasible Cartesian-product entries before continuation values are
    requested while remaining identical to the reference transition.
    """
    if player.status is not Status.ACTIVE:
        if action.kind is ActionKind.INACTIVE and not action.is_buyer:
            return player
        return None

    if action.kind is ActionKind.FAIL:
        if action.is_buyer:
            return None
        return fail_player(cfg, player)
    if action.kind in (ActionKind.INACTIVE, ActionKind.INITIAL_BUY):
        return None
    if action.is_buyer and player.position not in cfg.villages:
        return None
    if action.kind is ActionKind.MINE and player.position not in cfg.mines:
        return None
    if action.kind is ActionKind.MOVE:
        if weather == "sandstorm" or action.destination not in cfg.adj[player.position]:
            return None
    elif action.kind not in (ActionKind.STAY, ActionKind.MINE):
        return None

    if action.is_buyer:
        if village_buyer_count <= 0:
            return None
        price_factor = 2 if village_buyer_count == 1 else 4
    else:
        price_factor = 0
    purchase_cost = (
        cfg.money_scale
        * price_factor
        * (cfg.water_price * action.buy_water + cfg.food_price * action.buy_food)
    )
    water = player.water + action.buy_water
    food = player.food + action.buy_food
    cash = player.cash_scaled - purchase_cost
    if cash < 0 or not weight_ok(cfg, water, food):
        return None

    if action.kind is ActionKind.STAY:
        multiplier = 1
        destination = player.position
        income = 0
    elif action.kind is ActionKind.MINE:
        if mine_count <= 0:
            return None
        multiplier = 3
        destination = player.position
        income = cfg.money_scale * cfg.mine_income // mine_count
    else:
        if edge_count <= 0:
            return None
        multiplier = 2 * edge_count
        destination = action.destination
        income = 0

    water -= multiplier * cfg.water_consume[weather]
    food -= multiplier * cfg.food_consume[weather]
    if water < 0 or food < 0:
        return None
    next_player = PlayerState(
        Status.ACTIVE,
        position=destination,
        water=water,
        food=food,
        cash_scaled=cash + income,
    )
    if destination == cfg.end:
        next_player = finish_player(cfg, next_player)
    return next_player


def apply_joint_transition_scalar(
    cfg: Q3Config,
    state: JointState,
    actions: Sequence[Action],
    weather: Weather,
) -> JointState | None:
    """Apply one simultaneous day, returning None for an illegal profile."""
    _validate_profile_size(cfg, state, actions)
    counts = count_interactions_scalar(state, actions)
    out: list[PlayerState] = []

    for i, (player, action) in enumerate(zip(state, actions, strict=True)):
        next_player = apply_player_action_given_counts(
            cfg,
            player,
            action,
            weather,
            edge_count=counts.edge[i],
            mine_count=counts.mine[i],
            village_buyer_count=counts.village_buyers[i],
        )
        if next_player is None:
            return None
        out.append(next_player)
    return tuple(out)


def apply_joint_transition_batch(
    cfg: Q3Config,
    state: JointState,
    profiles: Sequence[Sequence[Action]],
    weather: Weather,
) -> BatchTransitionResult:
    """Vectorize interaction counts, feasibility checks, and numeric updates."""
    batch = len(profiles)
    n = cfg.n_players
    if len(state) != n:
        raise ValueError("state player count does not match configuration")
    if batch == 0:
        empty = np.empty((0, n), dtype=np.int16)
        return BatchTransitionResult(
            np.empty(0, dtype=bool), (), empty, empty.copy(), empty.copy()
        )
    if any(len(profile) != n for profile in profiles):
        raise ValueError("profile player count does not match configuration")

    kind = np.asarray(
        [[int(action.kind) for action in profile] for profile in profiles],
        dtype=np.int8,
    )
    destination = np.asarray(
        [[action.destination for action in profile] for profile in profiles],
        dtype=np.int16,
    )
    buy_w = np.asarray(
        [[action.buy_water for action in profile] for profile in profiles],
        dtype=np.int32,
    )
    buy_f = np.asarray(
        [[action.buy_food for action in profile] for profile in profiles],
        dtype=np.int32,
    )
    status = np.asarray([int(player.status) for player in state], dtype=np.int8)
    source = np.asarray([player.position for player in state], dtype=np.int16)
    water0 = np.asarray([player.water for player in state], dtype=np.int32)
    food0 = np.asarray([player.food for player in state], dtype=np.int32)
    cash0 = np.asarray([player.cash_scaled for player in state], dtype=np.int64)

    active = status == int(Status.ACTIVE)
    active_b = np.broadcast_to(active, (batch, n))
    is_move = kind == int(ActionKind.MOVE)
    is_mine = kind == int(ActionKind.MINE)
    is_stay = kind == int(ActionKind.STAY)
    is_fail = kind == int(ActionKind.FAIL)
    is_inactive = kind == int(ActionKind.INACTIVE)
    is_buyer = (buy_w + buy_f) > 0

    edge_count, mine_count, buyer_count = count_interactions_batch(
        kind, source, destination, is_buyer, active
    )

    player_valid = np.ones((batch, n), dtype=bool)
    player_valid[:, ~active] &= is_inactive[:, ~active] & ~is_buyer[:, ~active]
    player_valid[:, active] &= ~is_inactive[:, active]
    player_valid &= kind != int(ActionKind.INITIAL_BUY)
    player_valid &= ~(is_fail & is_buyer)

    for i, player in enumerate(state):
        if player.status is not Status.ACTIVE:
            continue
        allowed_move = np.isin(destination[:, i], np.asarray(cfg.adj[player.position]))
        player_valid[:, i] &= ~is_move[:, i] | allowed_move
        if weather == "sandstorm":
            player_valid[:, i] &= ~is_move[:, i]
        if player.position not in cfg.mines:
            player_valid[:, i] &= ~is_mine[:, i]
        if player.position not in cfg.villages:
            player_valid[:, i] &= ~is_buyer[:, i]
        player_valid[:, i] &= (
            is_stay[:, i] | is_mine[:, i] | is_move[:, i] | is_fail[:, i]
        )

    price_factor = np.where(is_buyer, np.where(buyer_count == 1, 2, 4), 0)
    purchase_cost = (
        cfg.money_scale
        * price_factor.astype(np.int64)
        * (
            cfg.water_price * buy_w.astype(np.int64)
            + cfg.food_price * buy_f.astype(np.int64)
        )
    )
    water_pre = water0[None, :] + buy_w
    food_pre = food0[None, :] + buy_f
    cash_after_buy = cash0[None, :] - purchase_cost
    player_valid &= cash_after_buy >= 0
    player_valid &= (
        cfg.water_weight * water_pre + cfg.food_weight * food_pre <= cfg.weight_limit
    )

    multiplier = np.zeros((batch, n), dtype=np.int16)
    multiplier[is_stay] = 1
    multiplier[is_mine] = 3
    multiplier[is_move] = 2 * edge_count[is_move]
    base_w = cfg.water_consume[weather]
    base_f = cfg.food_consume[weather]
    water_next = water_pre - multiplier * base_w
    food_next = food_pre - multiplier * base_f
    player_valid &= is_fail | ~active_b | ((water_next >= 0) & (food_next >= 0))

    income = np.zeros((batch, n), dtype=np.int64)
    for i in range(n):
        rows = is_mine[:, i]
        income[rows, i] = cfg.money_scale * cfg.mine_income // mine_count[rows, i]
    cash_next = cash_after_buy + income
    position_next = np.broadcast_to(source, (batch, n)).copy()
    position_next[is_move] = destination[is_move]

    valid = player_valid.all(axis=1)
    successors: list[JointState | None] = [None] * batch
    for row in np.flatnonzero(valid):
        next_players: list[PlayerState] = []
        for i, player in enumerate(state):
            action_kind = ActionKind(int(kind[row, i]))
            if player.status is not Status.ACTIVE:
                next_players.append(player)
            elif action_kind is ActionKind.FAIL:
                next_players.append(fail_player(cfg, player))
            else:
                next_player = PlayerState(
                    Status.ACTIVE,
                    position=int(position_next[row, i]),
                    water=int(water_next[row, i]),
                    food=int(food_next[row, i]),
                    cash_scaled=int(cash_next[row, i]),
                )
                if next_player.position == cfg.end:
                    next_player = finish_player(cfg, next_player)
                next_players.append(next_player)
        successors[row] = tuple(next_players)
    return BatchTransitionResult(
        valid=valid,
        successors=tuple(successors),
        edge_count=edge_count,
        mine_count=mine_count,
        village_buyer_count=buyer_count,
    )
