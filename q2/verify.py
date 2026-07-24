"""Replay a simulated Q2 trajectory under the game rules and check consistency.

Two levels of checking:
  * rule replay (like Q1): consumption, adjacency, storm ban, mine/village
    placement, weight limit, village prices, and cash never going negative
    (this validates the cash-never-binds assumption behind V = C + U);
  * policy consistency: every recorded action must equal the argmax returned
    by ``q2.policy.best_action`` at the realised state (requires the solve
    result; pass ``res=None`` to skip).

Purchase timing follows the mode: start_of_day buys before the action at the
morning village; after_arrival buys after the action at the destination
village (consumption is therefore checked against pre-buy inventory).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .data import LevelConfig
from .dp import SolveResult
from .model import DayRecord, VillagePurchaseMode, terminal_value, village_prices
from .policy import best_action


def verify_trajectory(
    cfg: LevelConfig,
    traj: List[DayRecord],
    purchase_mode: VillagePurchaseMode,
    res: Optional[SolveResult] = None,
    weather: Optional[Tuple[str, ...]] = None,
) -> Tuple[bool, str]:
    """Return (ok, message). On success message contains the terminal value."""
    if weather is None:
        weather = cfg.weather
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

    def apply_buy(t: int, buy_w: int, buy_f: int, where: int) -> Optional[str]:
        nonlocal cash, water, food
        if where not in cfg.villages:
            return f"day {t}: purchase not allowed at node {where}"
        cost = p_w_v * buy_w + p_f_v * buy_f
        if cost > cash:
            return f"day {t}: cannot afford purchase (cash {cash})"
        cash -= cost
        water += buy_w
        food += buy_f
        if cfg.water_weight * water + cfg.food_weight * food > cfg.weight_limit:
            return f"day {t}: overweight after purchase"
        return None

    for rec in traj[1:]:
        t = rec.day
        theta = weather[t]
        is_storm = theta == "sandstorm"
        bw, bf = cfg.water_consume[theta], cfg.food_consume[theta]
        buy_w, buy_f = rec.bought_water, rec.bought_food

        # --- policy consistency (action must be the DP argmax here) ---
        if res is not None:
            ch = best_action(cfg, res, t, pos, water, food, theta)
            if (ch.action, ch.buy_w, ch.buy_f) != (
                rec.action,
                rec.bought_water,
                rec.bought_food,
            ):
                return (
                    False,
                    f"day {t}: action {rec.action}+W{rec.bought_water}/F{rec.bought_food}"
                    f" != policy argmax {ch.action}+W{ch.buy_w}/F{ch.buy_f}",
                )

        # --- start_of_day purchase at the morning village ---
        if purchase_mode is VillagePurchaseMode.START_OF_DAY and (buy_w or buy_f):
            err = apply_buy(t, buy_w, buy_f, pos)
            if err:
                return False, err

        # --- main action ---
        action = rec.action
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
        elif action == "fail":
            return True, "OK (failed: no feasible action)"
        else:
            return False, f"day {t}: unknown action {action}"

        need_w, need_f = mult * bw, mult * bf
        if water < need_w or food < need_f:
            return False, f"day {t}: insufficient resources for {action}"
        water -= need_w
        food -= need_f
        cash += income
        pos = new_pos
        if cash < 0:
            return False, f"day {t}: cash went negative ({cash})"

        # --- after_arrival purchase at the destination village ---
        if purchase_mode is VillagePurchaseMode.AFTER_ARRIVAL and (buy_w or buy_f):
            err = apply_buy(t, buy_w, buy_f, pos)
            if err:
                return False, err

        if (pos, cash, water, food) != (rec.region, rec.cash, rec.water, rec.food):
            return (
                False,
                f"day {t}: state mismatch sim=({pos},{cash},{water},{food}) "
                f"rec=({rec.region},{rec.cash},{rec.water},{rec.food})",
            )

        if pos == cfg.end:
            val = terminal_value(cfg, cash, water, food)
            if rec is not traj[-1]:
                return False, "trajectory continues after reaching end"
            return True, f"OK value={val:.1f} arrival_day={t}"

    return False, "never reached end"
