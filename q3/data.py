"""Q3 level configurations built from the verified Q2 maps and parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from q2.data import LevelConfig as Q2LevelConfig
from q2.data import level3 as q2_level3
from q2.data import level4 as q2_level4

Weather = str


@dataclass(frozen=True)
class Q3Config:
    name: str
    n_players: int
    deadline: int
    mine_income: int
    p_weather: Mapping[Weather, float]
    weight_limit: int
    init_cash: int
    water_weight: int
    food_weight: int
    water_price: int
    food_price: int
    water_consume: Mapping[Weather, int]
    food_consume: Mapping[Weather, int]
    start: int
    end: int
    mines: frozenset[int]
    villages: frozenset[int]
    adj: Mapping[int, tuple[int, ...]]
    nodes: tuple[int, ...]
    money_scale: int = 6
    failure_penalty: int = 1_000_000
    weather_sequence: tuple[Weather, ...] | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.n_players <= 3:
            raise ValueError("the exact fixed-point implementation supports 1..3 players")
        if self.money_scale % 6:
            raise ValueError("money_scale must be divisible by 6")
        if self.start not in self.nodes or self.end not in self.nodes:
            raise ValueError("start/end must be map nodes")
        if self.deadline <= 0 or self.weight_limit <= 0:
            raise ValueError("deadline and weight_limit must be positive")
        positive_mass = sum(float(p) for p in self.p_weather.values())
        if abs(positive_mass - 1.0) > 1e-12:
            raise ValueError(f"weather probabilities sum to {positive_mass}, not 1")
        for weather, probability in self.p_weather.items():
            if probability < 0:
                raise ValueError(f"negative weather probability for {weather}")
            if weather not in self.water_consume or weather not in self.food_consume:
                raise ValueError(f"missing consumption for weather {weather}")
        if self.weather_sequence is not None:
            if len(self.weather_sequence) != self.deadline:
                raise ValueError(
                    "known weather sequence length must equal the deadline"
                )
            for weather in self.weather_sequence:
                if weather not in self.water_consume or weather not in self.food_consume:
                    raise ValueError(f"unknown weather in sequence: {weather}")

    @property
    def init_cash_scaled(self) -> int:
        return self.money_scale * self.init_cash

    @property
    def failure_penalty_scaled(self) -> int:
        return self.money_scale * self.failure_penalty

    @property
    def weather_order(self) -> tuple[Weather, ...]:
        canonical = ("sunny", "hot", "sandstorm")
        ordered = [w for w in canonical if self.p_weather.get(w, 0.0) > 0]
        ordered.extend(
            sorted(w for w, p in self.p_weather.items() if p > 0 and w not in canonical)
        )
        return tuple(ordered)


def _from_q2(base: Q2LevelConfig, *, name: str) -> Q3Config:
    return Q3Config(
        name=name,
        n_players=3,
        deadline=base.deadline,
        mine_income=base.mine_income,
        p_weather=dict(base.p_weather),
        weight_limit=base.weight_limit,
        init_cash=base.init_cash,
        water_weight=base.water_weight,
        food_weight=base.food_weight,
        water_price=base.water_price,
        food_price=base.food_price,
        water_consume=dict(base.water_consume),
        food_consume=dict(base.food_consume),
        start=base.start,
        end=base.end,
        mines=base.mines,
        villages=base.villages,
        adj=dict(base.adj),
        nodes=base.nodes,
        weather_sequence=(
            tuple(base.weather[1:]) if name == "level5" else None
        ),
    )


def level5() -> Q3Config:
    """Known-weather fifth level; shared rules are useful to Q3.1."""
    return _from_q2(q2_level3(), name="level5")


def level6() -> Q3Config:
    """Stochastic sixth level used by Q3.2."""
    return _from_q2(q2_level4(), name="level6")


def tiny_level6(*, n_players: int = 3, deadline: int = 2) -> Q3Config:
    """A tiny exact instance for smoke tests; it does not alter level-6 rules.

    The map is start 1 -- mine/village 2 -- end 3.  Resource units are lighter
    only to keep exhaustive purchase tests small.
    """
    return Q3Config(
        name="tiny-level6",
        n_players=n_players,
        deadline=deadline,
        mine_income=12,
        p_weather={"sunny": 1.0, "hot": 0.0, "sandstorm": 0.0},
        weight_limit=12,
        init_cash=40,
        water_weight=1,
        food_weight=1,
        water_price=1,
        food_price=1,
        water_consume={"sunny": 1, "hot": 2, "sandstorm": 2},
        food_consume={"sunny": 1, "hot": 2, "sandstorm": 2},
        start=1,
        end=3,
        mines=frozenset({2}),
        villages=frozenset({2}),
        adj={1: (2,), 2: (1, 3), 3: (2,)},
        nodes=(1, 2, 3),
        failure_penalty=1_000,
    )
