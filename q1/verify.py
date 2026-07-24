"""Replay a DP trajectory under the game rules and check consistency."""

from __future__ import annotations

from typing import List, Tuple

from .data import LevelConfig
from .model import DayRecord, VillagePurchaseMode, terminal_value, village_prices


def verify_trajectory(
    cfg: LevelConfig,
    traj: List[DayRecord],
    purchase_mode: VillagePurchaseMode,
    expected_value: float | None = None,
    tol: float = 1e-6,
) -> Tuple[bool, str]:
    """Return (ok, message). On success message contains the terminal value."""
    if not traj or traj[0].day != 0:
        return False, "trajectory must start at day 0"

    d0 = traj[0]
    if d0.region != cfg.start:
        return False, f"day0 region {d0.region} != start {cfg.start}"

    cost0 = cfg.water_price * d0.water + cfg.food_price * d0.food
    if d0.cash != cfg.init_cash - cost0:
        return False, f"day0 cash mismatch: {d0.cash} vs {cfg.init_cash - cost0}"
    if cfg.water_weight * d0.water + cfg.food_weight * d0.food > cfg.weight_limit:
        return False, "day0 overweight"

    cash, water, food, pos = d0.cash, d0.water, d0.food, d0.region
    p_w_v, p_f_v = village_prices(cfg)

    for rec in traj[1:]:
        t = rec.day
        weather = cfg.weather[t]
        is_storm = weather == "sandstorm"
        bw, bf = cfg.water_consume[weather], cfg.food_consume[weather]

        # --- start-of-day village purchase ---
        if purchase_mode is VillagePurchaseMode.START_OF_DAY and pos in cfg.villages:
            buy_w, buy_f = rec.bought_water, rec.bought_food
            cost = p_w_v * buy_w + p_f_v * buy_f
            if cost > cash:
                return False, f"day {t}: cannot afford start buy"
            cash -= cost
            water += buy_w
            food += buy_f
            if cfg.water_weight * water + cfg.food_weight * food > cfg.weight_limit:
                return False, f"day {t}: overweight after start buy"

        # --- main action ---
        action = rec.action.split("+")[0]
        if action == "stay":
            mult, new_pos, income = 1, pos, 0
        elif action == "mine":
            if pos not in cfg.mines:
                return False, f"day {t}: mine not at mine node {pos}"
            mult, new_pos, income = 3, pos, cfg.mine_income
        elif action.startswith("move:"):
            if is_storm:
                return False, f"day {t}: move during sandstorm"
            dest = int(action.split(":")[1])
            if dest not in cfg.adj[pos]:
                return False, f"day {t}: {pos} not adjacent to {dest}"
            mult, new_pos, income = 2, dest, 0
        elif action == "buy_only":
            mult, new_pos, income = 0, pos, 0
        else:
            return False, f"day {t}: unknown action {rec.action}"

        need_w, need_f = mult * bw, mult * bf
        if water < need_w or food < need_f:
            return False, f"day {t}: insufficient resources for {action}"
        water -= need_w
        food -= need_f
        cash += income
        pos = new_pos

        # --- after-arrival village purchase ---
        if purchase_mode is VillagePurchaseMode.AFTER_ARRIVAL and pos in cfg.villages:
            if "+buy" in rec.action or rec.bought_water or rec.bought_food:
                buy_w, buy_f = rec.bought_water, rec.bought_food
                cost = p_w_v * buy_w + p_f_v * buy_f
                if cost > cash:
                    return False, f"day {t}: cannot afford after buy"
                cash -= cost
                water += buy_w
                food += buy_f
                if cfg.water_weight * water + cfg.food_weight * food > cfg.weight_limit:
                    return False, f"day {t}: overweight after arrival buy"

        if (pos, cash, water, food) != (rec.region, rec.cash, rec.water, rec.food):
            return (
                False,
                f"day {t}: state mismatch sim=({pos},{cash},{water},{food}) "
                f"rec=({rec.region},{rec.cash},{rec.water},{rec.food})",
            )

        if pos == cfg.end:
            val = terminal_value(cfg, cash, water, food)
            if expected_value is not None and abs(val - expected_value) > tol:
                return False, f"terminal value {val} != expected {expected_value}"
            if rec is not traj[-1]:
                return False, "trajectory continues after reaching end"
            return True, f"OK value={val:.1f} arrival_day={t}"

    return False, "never reached end"
