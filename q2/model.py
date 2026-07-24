"""Q2 game-rule helpers: consumption multipliers, prices, penalties."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from .data import LevelConfig, Weather


class ActionType(Enum):
    STAY = 0
    MINE = 1
    MOVE = 2


class VillagePurchaseMode(str, Enum):
    """When the player may buy water/food at a village (same semantics as Q1).

    start_of_day  : buy only if already at the village at the beginning of the day;
                    arriving today unlocks purchase tomorrow. Matches Q1.md.
    after_arrival : buy after the day's action, if ending the day at a village.
    """

    START_OF_DAY = "start_of_day"
    AFTER_ARRIVAL = "after_arrival"


def action_multiplier(action: ActionType) -> int:
    return {ActionType.STAY: 1, ActionType.MOVE: 2, ActionType.MINE: 3}[action]


def base_consume(cfg: LevelConfig, weather: Weather) -> Tuple[int, int]:
    return cfg.water_consume[weather], cfg.food_consume[weather]


def consume(cfg: LevelConfig, weather: Weather, action: ActionType) -> Tuple[int, int]:
    w0, f0 = base_consume(cfg, weather)
    m = action_multiplier(action)
    return m * w0, m * f0


def village_prices(cfg: LevelConfig) -> Tuple[int, int]:
    """Village unit prices = 2 × base prices."""
    return 2 * cfg.water_price, 2 * cfg.food_price


def terminal_value(cfg: LevelConfig, cash: int, water: int, food: int) -> float:
    """Final objective: cash + half-price refund of leftover resources."""
    return cash + 0.5 * cfg.water_price * water + 0.5 * cfg.food_price * food


def weight_ok(cfg: LevelConfig, water: int, food: int) -> bool:
    return (
        water >= 0
        and food >= 0
        and cfg.water_weight * water + cfg.food_weight * food <= cfg.weight_limit
    )


@dataclass(frozen=True)
class DayRecord:
    """One day of a sampled trajectory (Result.xlsx-style and debugging)."""

    day: int
    region: int
    cash: int
    water: int
    food: int
    action: str = ""
    bought_water: int = 0
    bought_food: int = 0
    weather: str = ""