"""Sample a trajectory under the optimal Q2 policy for a given weather sequence.

Day 0 applies the optimal purchase ``res.best_q0``; each following day observes
``weather[t]`` and acts via ``q2.policy.best_action``.  Returns the list of
DayRecord plus the realised terminal value (None if the player fails).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .data import LevelConfig, Weather
from .dp import SolveResult
from .model import DayRecord, terminal_value, village_prices
from .policy import best_action


def simulate(
    cfg: LevelConfig,
    res: SolveResult,
    weather: Tuple[Weather, ...] | None = None,
) -> Tuple[List[DayRecord], Optional[float]]:
    """Drive one trajectory; ``weather[t]`` is day t's weather (1-indexed)."""
    if weather is None:
        weather = cfg.weather
    p_w_v, p_f_v = village_prices(cfg)

    q0_w, q0_f = res.best_q0
    cash = cfg.init_cash - cfg.water_price * q0_w - cfg.food_price * q0_f
    water, food, pos = q0_w, q0_f, cfg.start
    recs = [DayRecord(0, pos, cash, water, food, "buy0", q0_w, q0_f, "")]

    value: Optional[float] = None
    for t in range(1, cfg.deadline + 1):
        theta = weather[t]
        ch = best_action(cfg, res, t, pos, water, food, theta)
        if ch.action == "done":
            break
        if ch.action == "fail":
            recs.append(DayRecord(t, pos, cash, water, food, "fail", 0, 0, theta))
            break
        cash += ch.income - p_w_v * ch.buy_w - p_f_v * ch.buy_f
        water, food, pos = ch.water, ch.food, ch.dest
        recs.append(
            DayRecord(t, pos, cash, water, food, ch.action, ch.buy_w, ch.buy_f, theta)
        )
        if pos == cfg.end:
            value = terminal_value(cfg, cash, water, food)
            break
    return recs, value
