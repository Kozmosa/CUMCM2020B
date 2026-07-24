"""Forward DP for Q1: known weather, single player, maximise terminal funds.

State at end of day t
---------------------
    (v, W, F)  ->  max cash C

Cash is monotone for feasibility and for the objective, so keeping only the
maximum C at each (t, v, W, F) is exact (no multi-objective Pareto needed).

Timeline
--------
    day 0 : buy at start only (no move / no consume).
    day t : [optional village buy] -> one action -> consume / mine income
            -> [optional village buy] -> if v == end, score and stop.

Village buy timing is controlled by ``VillagePurchaseMode`` (see model.py).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data import LevelConfig
from .model import DayRecord, VillagePurchaseMode, terminal_value

# Unreachable sentinel in cash tables (must stay well below any real cash).
NEG = np.int32(-(10**9))

# Parent tuple layout (shared by day-0 and day-t dicts):
#   (prev_v, prev_w, prev_f, action, buy_w, buy_f, cash_after)
Parent = Tuple[int, int, int, str, int, int, int]


@dataclass
class SolveResult:
    best_value: float
    best_cash_at_end: int
    best_water: int
    best_food: int
    arrival_day: int
    trajectory: List[DayRecord]
    purchase_mode: VillagePurchaseMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shortest_to_end(cfg: LevelConfig) -> np.ndarray:
    """BFS hop-count to the end node (optimistic: ignores sandstorms).

    Used only for pruning: if dist[v] > remaining days, v cannot finish in time.
    """
    n = max(cfg.nodes)
    dist = np.full(n + 1, 10**9, dtype=np.int32)
    dist[cfg.end] = 0
    q = deque([cfg.end])
    while q:
        u = q.popleft()
        for v in cfg.adj[u]:
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _weight_mask(cfg: LevelConfig, w_max: int, f_max: int) -> np.ndarray:
    """True where water_weight*W + food_weight*F <= weight_limit."""
    w = np.arange(w_max + 1, dtype=np.int32)[:, None]
    f = np.arange(f_max + 1, dtype=np.int32)[None, :]
    return cfg.water_weight * w + cfg.food_weight * f <= cfg.weight_limit


def buy_knapsack(
    cash: np.ndarray, mask: np.ndarray, p_w: int, p_f: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand a (W+1, F+1) cash layer by buying any amount of water then food.

    For every cell (w, f) computes
        max_{w0<=w, f0<=f}  cash[w0,f0] - p_w*(w-w0) - p_f*(f-f0)
    subject to the weight mask.  Two 1-D sweeps are exact because unit prices
    are constant.

    Returns
    -------
    new_cash, origin_w, origin_f
        origin_* maps each cell back to the pre-buy inventory that realises it
        (needed to recover buy amounts for parent pointers / verification).
    """
    out = cash.copy()
    w_max, f_max = out.shape[0] - 1, out.shape[1] - 1
    ow = np.broadcast_to(np.arange(w_max + 1, dtype=np.int16)[:, None], out.shape).copy()
    of = np.broadcast_to(np.arange(f_max + 1, dtype=np.int16)[None, :], out.shape).copy()

    # Buy water: increasing w
    for w in range(w_max):
        src = out[w]
        cand = src - np.int32(p_w)
        better = (src > NEG) & mask[w + 1] & (cand > out[w + 1])
        out[w + 1, better] = cand[better]
        ow[w + 1, better] = ow[w, better]
        of[w + 1, better] = of[w, better]

    # Buy food: increasing f
    for f in range(f_max):
        src = out[:, f]
        cand = src - np.int32(p_f)
        better = (src > NEG) & mask[:, f + 1] & (cand > out[:, f + 1])
        out[better, f + 1] = cand[better]
        ow[better, f + 1] = ow[better, f]
        of[better, f + 1] = of[better, f]

    return out, ow, of


# ---------------------------------------------------------------------------
# Main DP
# ---------------------------------------------------------------------------


def solve(
    cfg: LevelConfig,
    purchase_mode: VillagePurchaseMode = VillagePurchaseMode.START_OF_DAY,
    verbose: bool = True,
) -> SolveResult:
    """Forward DP; return optimal terminal value and day-by-day trajectory."""
    T = cfg.deadline
    w_max = cfg.weight_limit // cfg.water_weight  # 400
    f_max = cfg.weight_limit // cfg.food_weight  # 600
    n = max(cfg.nodes)

    # cash[v, w, f] = max cash at end of current day (node id indexes axis 0).
    shape = (n + 1, w_max + 1, f_max + 1)
    mask = _weight_mask(cfg, w_max, f_max)
    dist_end = _shortest_to_end(cfg)
    p_w_v, p_f_v = 2 * cfg.water_price, 2 * cfg.food_price  # village prices

    # parents[t][(v,w,f)] -> Parent tuple for backtracking the best path.
    parents: List[Dict[Tuple[int, int, int], Parent]] = [dict() for _ in range(T + 1)]

    cash = np.full(shape, NEG, dtype=np.int32)

    # ---- day 0: purchase at start, no action ----
    seed = np.full((w_max + 1, f_max + 1), NEG, dtype=np.int32)
    seed[0, 0] = np.int32(cfg.init_cash)
    day0, _, _ = buy_knapsack(seed, mask, cfg.water_price, cfg.food_price)
    cash[cfg.start] = day0
    ww, ff = np.where(day0 > NEG)
    for w, f in zip(ww.tolist(), ff.tolist()):
        parents[0][(cfg.start, w, f)] = (
            cfg.start, 0, 0, "buy0", w, f, int(day0[w, f])
        )

    if verbose:
        print(f"[{cfg.name}/{purchase_mode.value}] day0 states={ww.size}")

    best_value = float("-inf")
    best_cash = -1
    best_key: Optional[Tuple[int, int, int, int]] = None  # (t, v, w, f)

    def consider_finish(t: int, layer: np.ndarray) -> None:
        """Score every state that sits on the end node after day t."""
        nonlocal best_value, best_cash, best_key
        for w, f in zip(*np.where(layer[cfg.end] > NEG)):
            c = int(layer[cfg.end, w, f])
            val = terminal_value(cfg, c, int(w), int(f))
            if val > best_value + 1e-12 or (
                abs(val - best_value) <= 1e-12 and c > best_cash
            ):
                best_value, best_cash = val, c
                best_key = (t, cfg.end, int(w), int(f))

    # ---- days 1 .. T ----
    for t in range(1, T + 1):
        weather = cfg.weather[t]
        is_storm = weather == "sandstorm"
        bw = cfg.water_consume[weather]
        bf = cfg.food_consume[weather]
        # base, 2x (move), 3x (mine)
        cons = {"stay": (bw, bf), "move": (2 * bw, 2 * bf), "mine": (3 * bw, 3 * bf)}

        # work = cash available at the *start* of day t (after optional buy).
        # origin_w/f recover pre-buy inventory for parent pointers.
        work = cash.copy()
        origin_w = np.broadcast_to(
            np.arange(w_max + 1, dtype=np.int16)[None, :, None], shape
        ).copy()
        origin_f = np.broadcast_to(
            np.arange(f_max + 1, dtype=np.int16)[None, None, :], shape
        ).copy()

        if purchase_mode is VillagePurchaseMode.START_OF_DAY:
            for v in cfg.villages:
                after, ow, of = buy_knapsack(cash[v], mask, p_w_v, p_f_v)
                work[v] = after
                origin_w[v] = ow
                origin_f[v] = of

        next_cash = np.full(shape, NEG, dtype=np.int32)
        next_parents: Dict[Tuple[int, int, int], Parent] = {}

        def apply_transition(
            v_from: int,
            v_to: int,
            src: np.ndarray,
            cw: int,
            cf: int,
            income: int,
            action: str,
        ) -> None:
            """Vectorised transition over the whole (W, F) plane.

            Destination inventory is (w_from - cw, f_from - cf); cash gains
            ``income`` (mine only).  Only cells that strictly improve next_cash
            get a parent pointer.
            """
            if cw >= src.shape[0] or cf >= src.shape[1]:
                return
            block = src[cw:, cf:]
            if income:
                valid = block > NEG
                block = block.astype(np.int32, copy=True)
                block[valid] += np.int32(income)
                block[~valid] = NEG

            dest = next_cash[v_to, : w_max + 1 - cw, : f_max + 1 - cf]
            better = block > dest
            if not better.any():
                return
            dest[better] = block[better]

            # Sparse parent write for improved cells only.
            ww, ff = np.where(better)
            w_from, f_from = ww + cw, ff + cf
            c_to = block[better]
            o_w = origin_w[v_from, w_from, f_from]
            o_f = origin_f[v_from, w_from, f_from]
            for i in range(ww.size):
                next_parents[(v_to, int(ww[i]), int(ff[i]))] = (
                    v_from,
                    int(o_w[i]),
                    int(o_f[i]),
                    action,
                    int(w_from[i] - o_w[i]),  # buy_w
                    int(f_from[i] - o_f[i]),  # buy_f
                    int(c_to[i]),
                )

        for v in cfg.nodes:
            if v == cfg.end:
                continue  # finished earlier; do not act from the end
            src = work[v]
            if not (src > NEG).any():
                continue

            cw, cf = cons["stay"]
            apply_transition(v, v, src, cw, cf, 0, "stay")

            # Mine only if the day *starts* at a mine (arrival day cannot mine).
            if v in cfg.mines:
                cw, cf = cons["mine"]
                apply_transition(v, v, src, cw, cf, cfg.mine_income, "mine")

            if not is_storm:
                cw, cf = cons["move"]
                for u in cfg.adj[v]:
                    # Optimistic prune: need at least dist_end[u] days after today.
                    if dist_end[u] > T - t:
                        continue
                    apply_transition(v, u, src, cw, cf, 0, f"move:{u}")

        # after_arrival: buy at villages after the day's action.
        if purchase_mode is VillagePurchaseMode.AFTER_ARRIVAL:
            for v in cfg.villages:
                before = next_cash[v]
                after, ow, of = buy_knapsack(before, mask, p_w_v, p_f_v)
                improved = after > before
                next_cash[v] = after
                if not improved.any():
                    continue
                for w, f in zip(*np.where(improved)):
                    w, f = int(w), int(f)
                    w0, f0 = int(ow[w, f]), int(of[w, f])
                    buy_w, buy_f = w - w0, f - f0
                    c_after = int(after[w, f])
                    if (v, w0, f0) in next_parents:
                        pv = next_parents[(v, w0, f0)]
                        next_parents[(v, w, f)] = (
                            pv[0], pv[1], pv[2], pv[3] + "+buy",
                            buy_w, buy_f, c_after,
                        )
                    else:
                        next_parents[(v, w, f)] = (
                            v, w0, f0, "buy_only", buy_w, buy_f, c_after,
                        )

        consider_finish(t, next_cash)
        cash = next_cash
        parents[t] = next_parents

        if verbose:
            print(
                f"[{cfg.name}/{purchase_mode.value}] day {t:2d} {weather:9s} "
                f"reach={int(np.sum(cash > NEG)):7d} best={best_value:.1f}"
            )

    if best_key is None:
        raise RuntimeError(
            f"No feasible path for {cfg.name} under mode={purchase_mode.value}"
        )

    return SolveResult(
        best_value=best_value,
        best_cash_at_end=best_cash,
        best_water=best_key[2],
        best_food=best_key[3],
        arrival_day=best_key[0],
        trajectory=_backtrack(cfg, parents, best_key),
        purchase_mode=purchase_mode,
    )


def _backtrack(
    cfg: LevelConfig,
    parents: List[Dict[Tuple[int, int, int], Parent]],
    best_key: Tuple[int, int, int, int],
) -> List[DayRecord]:
    """Walk parent pointers from the best end-state back to day 0."""
    t, v, w, f = best_key
    recs: List[DayRecord] = []
    while True:
        meta = parents[t].get((v, w, f))
        if meta is None:
            raise RuntimeError(f"missing parent day={t} state=({v},{w},{f})")
        prev_v, prev_w, prev_f, action, buy_w, buy_f, c_after = meta
        recs.append(
            DayRecord(
                day=t,
                region=v,
                cash=c_after,
                water=w,
                food=f,
                action=action,
                bought_water=buy_w,
                bought_food=buy_f,
                weather=cfg.weather[t] if t > 0 else "",
            )
        )
        if t == 0:
            break
        v, w, f = prev_v, prev_w, prev_f
        t -= 1
    recs.reverse()
    return recs
