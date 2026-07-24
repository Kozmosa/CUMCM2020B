"""Optimal-policy query for Q2.

Given the value tables U[t][v, w, f] computed by ``q2.dp.solve``,
``best_action`` re-evaluates every legal action at a *single* realised state
(day, node, inventory, observed weather) with exactly the same semantics as
the DP (including village purchase folding), and returns the argmax together
with the chosen buy amounts.  ``simulate`` drives a full trajectory with it.

Index convention: ``day`` is the current day (1..T).  The state (v, W, F) is
the end-of-day-(day-1) state; the action taken on ``day`` lands at
end-of-day-``day`` whose value is read from ``U[day]`` (the DP fills U[t]
from U[t+1] via the day-(t+1) action).
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Tuple

import numpy as np

from .data import LevelConfig, Weather
from .model import VillagePurchaseMode
from .dp import SolveResult, _NINF, _weight_mask


class Choice(NamedTuple):
    action: str  # "stay" | "mine" | "move:<u>" | "done" | "fail"
    dest: int  # node at end of day
    water: int  # inventory at end of day (after any arrival buy)
    food: int
    income: int  # mining income earned today
    buy_w: int  # water bought today (morning at v, or on arrival at dest)
    buy_f: int
    value: float  # U value of the chosen continuation (excl. cash)


def best_action(
    cfg: LevelConfig,
    res: SolveResult,
    day: int,
    v: int,
    W: int,
    F: int,
    theta: Weather,
) -> Choice:
    """Optimal action for (day, v, W, F) under observed weather ``theta``.

    Mirrors ``q2.dp.solve`` pointwise:
      * plain destinations read ``U[day][u, W - cw, F - cf]`` (+ mine income);
      * after_arrival at a village dest: lift the dest layer at the arrival
        inventory (buy on arrival);
      * start_of_day at a village source: morning buy q (after observing
        theta), then act — scan the post-buy inventory (W + q) >= (W, F).
    """
    if v == cfg.end:
        return Choice("done", v, W, F, 0, 0, 0, 0.0)

    w_max = cfg.weight_limit // cfg.water_weight
    f_max = cfg.weight_limit // cfg.food_weight
    mask = _weight_mask(cfg, w_max, f_max)
    p_w_v, p_f_v = 2 * cfg.water_price, 2 * cfg.food_price
    Ud = res.U[day]  # end-of-day value layer

    bw, bf = cfg.water_consume[theta], cfg.food_consume[theta]
    is_storm = theta == "sandstorm"
    at_village_src = (
        res.purchase_mode is VillagePurchaseMode.START_OF_DAY and v in cfg.villages
    )

    candidates: List[Choice] = []

    def plain(u: int, cw: int, cf: int, income: int, tag: str) -> None:
        """No purchase involved on either side."""
        if W < cw or F < cf:
            return
        val = float(Ud[u, W - cw, F - cf])
        if val <= _NINF * 0.5:
            return
        candidates.append(
            Choice(tag, u, W - cw, F - cf, income, 0, 0, val + income)
        )

    def arrival_buy(u: int, cw: int, cf: int, income: int, tag: str) -> None:
        """after_arrival: land at village u with (W-cw, F-cf), then buy q."""
        if W < cw or F < cf:
            return
        wa, fa = W - cw, F - cf
        plane = Ud[u]
        sub = plane[wa:, fa:]
        qw = np.arange(sub.shape[0], dtype=np.float32)
        qf = np.arange(sub.shape[1], dtype=np.float32)
        val = sub - p_w_v * qw[:, None] - p_f_v * qf[None, :]
        valid = mask[wa:, fa:] & (sub > _NINF * 0.5)
        val = np.where(valid, val, np.float32(-np.inf))
        i = int(np.argmax(val))
        if not np.isfinite(val.flat[i]):
            return
        bw_i, bf_i = np.unravel_index(i, val.shape)
        candidates.append(
            Choice(
                tag, u, wa + int(bw_i), fa + int(bf_i), income,
                int(bw_i), int(bf_i), float(val.flat[i]) + income,
            )
        )

    def morning_buy(u: int, cw: int, cf: int, income: int, tag: str) -> None:
        """start_of_day: buy q at village v, then act with consumption (cw, cf).

        Scan the post-buy inventory (w', f') >= (W, F); the action requires
        w' >= cw, f' >= cf and lands at (w' - cw, f' - cf).
        """
        qw = np.arange(w_max + 1 - W, dtype=np.int32)
        qf = np.arange(f_max + 1 - F, dtype=np.int32)
        wp = W + qw  # post-buy inventory candidates
        fp = F + qf
        ok = (wp[:, None] >= cw) & (fp[None, :] >= cf) & mask[W:, F:]
        iw = np.clip(wp - cw, 0, None)
        jf = np.clip(fp - cf, 0, None)
        sub = Ud[u][iw[:, None], jf[None, :]]
        val = (
            sub
            - p_w_v * qw[:, None].astype(np.float32)
            - p_f_v * qf[None, :].astype(np.float32)
        )
        valid = ok & (sub > _NINF * 0.5)
        val = np.where(valid, val, np.float32(-np.inf))
        i = int(np.argmax(val))
        if not np.isfinite(val.flat[i]):
            return
        bw_i, bf_i = np.unravel_index(i, val.shape)
        candidates.append(
            Choice(
                tag, u, int(wp[bw_i]) - cw, int(fp[bf_i]) - cf, income,
                int(bw_i), int(bf_i), float(val.flat[i]) + income,
            )
        )

    def consider(u: int, cw: int, cf: int, income: int, tag: str) -> None:
        if at_village_src:
            morning_buy(u, cw, cf, income, tag)
        elif res.purchase_mode is VillagePurchaseMode.AFTER_ARRIVAL and u in cfg.villages:
            arrival_buy(u, cw, cf, income, tag)
        else:
            plain(u, cw, cf, income, tag)

    consider(v, bw, bf, 0, "stay")
    if v in cfg.mines:
        consider(v, 3 * bw, 3 * bf, cfg.mine_income, "mine")
    if not is_storm:
        for u in cfg.adj[v]:
            consider(u, 2 * bw, 2 * bf, 0, f"move:{u}")

    if not candidates:
        return Choice("fail", v, W, F, 0, 0, 0, -res.M)
    return max(candidates, key=lambda c: c.value)
