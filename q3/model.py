"""Q3 state, action, fixed-point money, and terminal-payoff definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

from .data import Q3Config


class Status(IntEnum):
    ACTIVE = 0
    FINISHED = 1
    FAILED = 2


class ActionKind(IntEnum):
    INACTIVE = 0
    FAIL = 1
    STAY = 2
    MINE = 3
    MOVE = 4
    INITIAL_BUY = 5


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    destination: int = 0
    buy_water: int = 0
    buy_food: int = 0

    def __post_init__(self) -> None:
        if self.buy_water < 0 or self.buy_food < 0:
            raise ValueError("purchase quantities must be non-negative")
        if self.kind is ActionKind.MOVE and self.destination <= 0:
            raise ValueError("move action requires a positive destination")
        if self.kind is not ActionKind.MOVE and self.destination != 0:
            raise ValueError("only move actions may carry a destination")

    @property
    def is_buyer(self) -> bool:
        return self.buy_water + self.buy_food > 0

    @property
    def code(self) -> tuple[int, int, int, int]:
        return (int(self.kind), self.destination, self.buy_water, self.buy_food)

    def label(self) -> str:
        if self.kind is ActionKind.MOVE:
            main = f"move:{self.destination}"
        else:
            main = self.kind.name.lower()
        if self.is_buyer:
            return f"{main}+W{self.buy_water}/F{self.buy_food}"
        return main


INACTIVE_ACTION = Action(ActionKind.INACTIVE)
FAIL_ACTION = Action(ActionKind.FAIL)


@dataclass(frozen=True, slots=True)
class PlayerState:
    status: Status
    position: int = 0
    water: int = 0
    food: int = 0
    cash_scaled: int = 0
    fixed_payoff_scaled: int = 0

    def __post_init__(self) -> None:
        if self.water < 0 or self.food < 0:
            raise ValueError("resource inventory cannot be negative")
        if self.status is Status.ACTIVE and self.position <= 0:
            raise ValueError("active player requires a positive position")

    @property
    def absorbed(self) -> bool:
        return self.status is not Status.ACTIVE


JointState = tuple[PlayerState, ...]


@dataclass(frozen=True, slots=True)
class StateValue:
    value: tuple[float, ...]
    success: tuple[float, ...]


def weight(cfg: Q3Config, water: int, food: int) -> int:
    return cfg.water_weight * water + cfg.food_weight * food


def weight_ok(cfg: Q3Config, water: int, food: int) -> bool:
    return water >= 0 and food >= 0 and weight(cfg, water, food) <= cfg.weight_limit


def terminal_payoff_scaled(
    cfg: Q3Config, cash_scaled: int, water: int, food: int
) -> int:
    refund = (
        cfg.money_scale * cfg.water_price * water
        + cfg.money_scale * cfg.food_price * food
    ) // 2
    return cash_scaled + refund


def failure_payoff_scaled(cfg: Q3Config, cash_scaled: int) -> int:
    return cash_scaled - cfg.failure_penalty_scaled


def finish_player(cfg: Q3Config, state: PlayerState) -> PlayerState:
    payoff = terminal_payoff_scaled(cfg, state.cash_scaled, state.water, state.food)
    return PlayerState(Status.FINISHED, fixed_payoff_scaled=payoff)


def fail_player(cfg: Q3Config, state: PlayerState) -> PlayerState:
    payoff = failure_payoff_scaled(cfg, state.cash_scaled)
    return PlayerState(Status.FAILED, fixed_payoff_scaled=payoff)


def initial_joint_state(cfg: Q3Config) -> JointState:
    state = PlayerState(
        Status.ACTIVE,
        position=cfg.start,
        cash_scaled=cfg.init_cash_scaled,
    )
    return tuple(state for _ in range(cfg.n_players))


def terminal_state_value(cfg: Q3Config, state: JointState) -> StateValue:
    values: list[float] = []
    success: list[float] = []
    for player in state:
        if player.status is Status.FINISHED:
            payoff = player.fixed_payoff_scaled
            succeeded = 1.0
        elif player.status is Status.FAILED:
            payoff = player.fixed_payoff_scaled
            succeeded = 0.0
        else:
            payoff = failure_payoff_scaled(cfg, player.cash_scaled)
            succeeded = 0.0
        values.append(payoff / cfg.money_scale)
        success.append(succeeded)
    return StateValue(tuple(values), tuple(success))


def all_absorbed(state: Iterable[PlayerState]) -> bool:
    return all(player.absorbed for player in state)
