"""Exact individual action and purchase enumeration for Q3."""

from __future__ import annotations

from itertools import chain
from typing import Iterator

from .data import Q3Config, Weather
from .model import (
    FAIL_ACTION,
    INACTIVE_ACTION,
    Action,
    ActionKind,
    PlayerState,
    Status,
    weight_ok,
)


class ActionEnumerationLimitExceeded(RuntimeError):
    """Exact enumeration crossed a caller-provided safety limit."""


def _check_limit(size: int, max_actions: int | None) -> None:
    if max_actions is not None and size > max_actions:
        raise ActionEnumerationLimitExceeded(
            f"exact action enumeration exceeded safety limit {max_actions}"
        )


def _purchase_options(
    cfg: Q3Config,
    state: PlayerState,
    *,
    price_factor: int,
    positive_only: bool = False,
) -> Iterator[tuple[int, int]]:
    """Enumerate all exact affordable post-inventory increments."""
    max_w = (cfg.weight_limit - cfg.water_weight * state.water) // cfg.water_weight
    max_f_by_weight = (
        cfg.weight_limit - cfg.food_weight * state.food
    ) // cfg.food_weight
    price_w = price_factor * cfg.water_price * cfg.money_scale
    price_f = price_factor * cfg.food_price * cfg.money_scale
    for buy_w in range(max_w + 1):
        remaining_weight = (
            cfg.weight_limit
            - cfg.water_weight * (state.water + buy_w)
            - cfg.food_weight * state.food
        )
        if remaining_weight < 0:
            break
        max_f_weight = min(max_f_by_weight, remaining_weight // cfg.food_weight)
        remaining_cash = state.cash_scaled - price_w * buy_w
        if remaining_cash < 0:
            break
        max_f_cash = remaining_cash // price_f if price_f else max_f_weight
        for buy_f in range(min(max_f_weight, max_f_cash) + 1):
            if positive_only and buy_w + buy_f == 0:
                continue
            yield buy_w, buy_f


def enumerate_initial_purchases(cfg: Q3Config, state: PlayerState) -> tuple[Action, ...]:
    if state.status is not Status.ACTIVE or state.position != cfg.start:
        raise ValueError("initial purchase requires an active player at the start")
    actions: list[Action] = []
    for buy_w, buy_f in _purchase_options(cfg, state, price_factor=1):
        actions.append(Action(ActionKind.INITIAL_BUY, buy_water=buy_w, buy_food=buy_f))
    return tuple(actions)


def _main_actions(cfg: Q3Config, state: PlayerState, weather: Weather) -> list[Action]:
    actions = [Action(ActionKind.STAY)]
    if state.position in cfg.mines:
        actions.append(Action(ActionKind.MINE))
    if weather != "sandstorm":
        actions.extend(
            Action(ActionKind.MOVE, destination=destination)
            for destination in cfg.adj[state.position]
        )
    return actions


def _minimum_multiplier(action: Action) -> int:
    if action.kind is ActionKind.STAY:
        return 1
    if action.kind is ActionKind.MINE:
        return 3
    if action.kind is ActionKind.MOVE:
        return 2
    raise ValueError(f"no consumption multiplier for {action.kind}")


def enumerate_individual_actions(
    cfg: Q3Config,
    state: PlayerState,
    weather: Weather,
    *,
    max_actions: int | None = None,
) -> tuple[Action, ...]:
    """Enumerate a lossless superset of jointly feasible actions.

    Positive village purchases are enumerated using the cheapest possible
    village price (2x).  When multiple players buy, the joint transition
    applies 4x prices and rejects unaffordable profiles.  Likewise moves are
    filtered here only at the single-traveller multiplier; congestion is
    checked after the complete joint action is known.
    """
    if state.status is not Status.ACTIVE:
        return (INACTIVE_ACTION,)

    purchases: Iterator[tuple[int, int]] = iter(((0, 0),))
    if state.position in cfg.villages:
        purchases = chain(
            purchases,
            _purchase_options(cfg, state, price_factor=2, positive_only=True),
        )

    base_w = cfg.water_consume[weather]
    base_f = cfg.food_consume[weather]
    actions: list[Action] = []
    for buy_w, buy_f in purchases:
        post_w = state.water + buy_w
        post_f = state.food + buy_f
        if not weight_ok(cfg, post_w, post_f):
            continue
        for main in _main_actions(cfg, state, weather):
            multiplier = _minimum_multiplier(main)
            if post_w < multiplier * base_w or post_f < multiplier * base_f:
                continue
            actions.append(
                Action(
                    main.kind,
                    destination=main.destination,
                    buy_water=buy_w,
                    buy_food=buy_f,
                )
            )
            _check_limit(len(actions), max_actions)

    if not actions:
        return (FAIL_ACTION,)
    actions.sort(key=lambda action: action.code)
    return tuple(actions)


def enumerate_initial_purchases_bounded(
    cfg: Q3Config, state: PlayerState, *, max_actions: int | None = None
) -> tuple[Action, ...]:
    if state.status is not Status.ACTIVE or state.position != cfg.start:
        raise ValueError("initial purchase requires an active player at the start")
    actions: list[Action] = []
    for buy_w, buy_f in _purchase_options(cfg, state, price_factor=1):
        actions.append(Action(ActionKind.INITIAL_BUY, buy_water=buy_w, buy_food=buy_f))
        _check_limit(len(actions), max_actions)
    return tuple(actions)
