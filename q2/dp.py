"""Backward induction DP for Q2 (unknown future weather, single player).

Model (Solution/Q2.md)
----------------------
Player observes theta_t *before* acting on day t; future days are i.i.d.
Bellman recursion is

    V_t(s) = sum_theta p(theta) * max_a  V_{t+1}(g(s, a, theta))   (v != end, t<T)
    V_T(s) = J(s)        if v == end           (cash + 1/2 p_W W + 1/2 p_F F)
            -M           otherwise             (failure penalty, replaces -inf)

State decomposition
-------------------
J = cash + 1/2 (p_W W + p_F F) is linear in cash, and cash is only altered by
mining income (+R) and village purchases (-p q). In these levels the cash
constraint never binds (the bottleneck is the 1200 kg weight limit; verified in
the replay), so we factor

    V_t(v, W, F, C) = C + U_t(v, W, F)

where U_t stores *future* net cash contributions:
    - mining income      +R
    - village purchase   -p_W q_W - p_F q_F
    - terminal refund    +(1/2)(p_W W + p_F F)   if the end is reached
    - failure penalty    -M                      otherwise

Index convention (same as Q1)
-----------------------------
s_t is the state at the *end* of day t.  Day t+1's action consumes (cw, cf)
and moves s_t -> s_{t+1}.  U[t][v, w, f] is indexed by the end-of-day-t
inventory (w, f); an action with consumption (cw, cf) therefore reads
U[t+1][dest, w - cw, f - cf].  Days run 1..T, s_0 is the state right after
the day-0 purchase, s_T is terminal.

Village purchase (reverse knapsack / "lift")
--------------------------------------------
Buying q at a village maps the current inventory (w, f) to a *higher*
inventory (w + q_W, f + q_F) at linear cost.  The lifted plane

    lifted[w, f] = max_{q >= 0}  U[w + q_W, f + q_F] - p_W q_W - p_F q_F

is computed by two 1-D sweeps that propagate values from high inventory to
low inventory (exact because the cost is additive and axis-separable).  The
sweeps track the argmax origin cell so the success-probability plane P can be
gathered at the same origin (P stays exactly consistent with the U-optimal
buy).

Mode semantics (same flag as Q1)
--------------------------------
start_of_day  : buy at the village where the player *starts* the day, after
                observing today's weather, before acting.  Implemented as:
                first take the per-weather max over actions (plain shifted
                reads of U[t+1]), then lift the resulting plane once.
                max and lift commute, so this equals per-action lifting, and
                the weight mask then applies to the post-buy / pre-action
                inventory, as the rules require.
after_arrival : buy at the village where the player *ends* the day.  The
                destination layer U[t+1][u] is lifted (once per day, theta-
                independent) whenever u is a village; staying at a village
                therefore also allows buying, matching Q1.

The day-0 purchase is a direct argmax of  cash_after(q) + U[0][start, q]
over the weight mask (buying from (0, 0) at base prices needs no lift).

Outputs: optimal expected terminal funds V0, the success probability under
the optimal policy, and the optimal day-0 purchase.  The full U table is kept
for policy probing / simulation (P is rolled two-layer and not stored).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .data import LevelConfig, Weather
from .model import VillagePurchaseMode

# Failure penalty (replaces -inf). Large enough to dominate reachable values.
DEFAULT_M = float(1.0e6)

# Unreachable sentinel for the U table; the action layer maps cells that can
# still *act* (but are doomed) to -M, while truly unreachable cells stay at
# _NINF and never propagate.
_NINF = float(-1.0e30)


@dataclass
class SolveResult:
    # Optimal expected terminal funds at day 0; equals cash_after_q0 + U[0].
    V0: float
    # Success probability under the optimal policy with the chosen day-0 buy.
    prob_succ: float
    # Day-0 purchase achieving V0 (water, food).
    best_q0: Tuple[int, int]
    M: float
    purchase_mode: VillagePurchaseMode
    level: str
    # Per-day value tables U[t][v, w, f] (kept for policy probing / simulation).
    U: np.ndarray


# ---------------------------------------------------------------------------
# Geometry / masks
# ---------------------------------------------------------------------------


def _shortest_to_end(cfg: LevelConfig) -> np.ndarray:
    """BFS hop-count to end (optimistic: ignores storms). For pruning only."""
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


# ---------------------------------------------------------------------------
# Lift (village purchase): propagate values from high to low inventory
# ---------------------------------------------------------------------------


def lift_up(
    U: np.ndarray, mask: np.ndarray, p_w: float, p_f: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """lifted[w, f] = max_{q >= 0} U[w + q_W, f + q_F] - p_W q_W - p_F q_F.

    Two 1-D sweeps from high to low inventory are exact because the cost is
    additive and axis-separable.  The weight mask is enforced on the *target*
    (post-buy) inventory: the chain may only pass through in-mask cells.

    Returns (lifted, origin_w, origin_f); origin_* maps each cell to the
    post-buy inventory realising the max, so a parallel plane (e.g. success
    probability) can be gathered at exactly the U-optimal origin.
    """
    out = U.astype(np.float32, copy=True)
    w_max, f_max = out.shape[0] - 1, out.shape[1] - 1
    ow = np.broadcast_to(np.arange(w_max + 1, dtype=np.int16)[:, None], out.shape).copy()
    of = np.broadcast_to(np.arange(f_max + 1, dtype=np.int16)[None, :], out.shape).copy()

    # Water: out[w] = max(out[w], out[w + 1] - p_w), target w + 1 must be in-mask.
    for w in range(w_max - 1, -1, -1):
        cand = out[w + 1] - np.float32(p_w)
        better = mask[w + 1] & (cand > out[w])
        out[w, better] = cand[better]
        ow[w, better] = ow[w + 1, better]
        of[w, better] = of[w + 1, better]

    # Food: out[:, f] = max(out[:, f], out[:, f + 1] - p_f).
    for f in range(f_max - 1, -1, -1):
        cand = out[:, f + 1] - np.float32(p_f)
        better = mask[:, f + 1] & (cand > out[:, f])
        out[better, f] = cand[better]
        ow[better, f] = ow[better, f + 1]
        of[better, f] = of[better, f + 1]

    return out, ow, of


# ---------------------------------------------------------------------------
# Action application (per (theta, v), take the max over actions cellwise)
# ---------------------------------------------------------------------------


def _apply_action(
    out_u: np.ndarray,
    out_p: np.ndarray,
    src_u: np.ndarray,
    src_p: np.ndarray,
    cw: int,
    cf: int,
    income: int,
) -> None:
    """Improve (out_u, out_p) in place by acting ``a`` from node ``v_from``.

    out_u[w, f] is the value at the source node holding (w, f) at the end of
    day t.  Consuming (cw, cf) and earning ``income`` lands the player at the
    destination with inventory (w - cw, f - cf), whose next-day value is
    ``src_u[w - cw, f - cf]`` (the destination layer of U[t+1], already
    lifted if the destination is a village in after_arrival mode).  We read
    the source plane from (0, 0) up to (w_max + 1 - cw, f_max + 1 - cf) and
    slot it into out_u at offset (cw, cf).
    """
    if cw >= src_u.shape[0] or cf >= src_u.shape[1]:
        return
    # Source region: end-of-day-(t+1) inventory indices (w_after, f_after).
    src_block_u = src_u[: src_u.shape[0] - cw, : src_u.shape[1] - cf]
    src_block_p = src_p[: src_p.shape[0] - cw, : src_p.shape[1] - cf]
    valid = src_block_u > _NINF * 0.5
    cand_u = np.where(valid, src_block_u + np.float32(income), np.float32(0.0))
    cand_p = np.where(valid, src_block_p, np.float32(0.0))

    dh, dw = src_block_u.shape
    dest_u = out_u[cw : cw + dh, cf : cf + dw]
    dest_p = out_p[cw : cw + dh, cf : cf + dw]

    better = valid & (cand_u > dest_u)
    dest_u[better] = cand_u[better]
    dest_p[better] = cand_p[better]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def solve(
    cfg: LevelConfig,
    purchase_mode: VillagePurchaseMode = VillagePurchaseMode.START_OF_DAY,
    M: float = DEFAULT_M,
    verbose: bool = True,
) -> SolveResult:
    """Backward-induct to optimal expected terminal funds.

    The day-0 purchase is optimised over (q_W, q_F). See module docstring.
    """
    T = cfg.deadline
    w_max = cfg.weight_limit // cfg.water_weight
    f_max = cfg.weight_limit // cfg.food_weight
    n = max(cfg.nodes)
    # cash table layout: U[t][v, w, f] with v indexed by node id 1..n
    cash_shape = (n + 1, w_max + 1, f_max + 1)
    dist_end = _shortest_to_end(cfg)
    mask = _weight_mask(cfg, w_max, f_max)
    p_w_v, p_f_v = 2 * cfg.water_price, 2 * cfg.food_price
    theta_list: Tuple[Weather, ...] = tuple(k for k, p in cfg.p_weather.items() if p > 0)
    p_list: Tuple[float, ...] = tuple(cfg.p_weather[k] for k in theta_list)

    U = np.full((T + 1,) + cash_shape, _NINF, dtype=np.float32)
    # Success probability is rolled two-layer (only P[t + 1] is needed).
    P_next = np.zeros(cash_shape, dtype=np.float32)

    # Terminal payoff residual (cash contribution handled by the +C offset):
    # U[T][end, w, f] = 0.5 (p_W w + p_F f). Failure (not at end) leaves the
    # cell at _NINF; the action layer maps reachable-but-doomed into -M.
    end_J = 0.5 * (
        cfg.water_price * np.arange(w_max + 1, dtype=np.float32)[:, None]
        + cfg.food_price * np.arange(f_max + 1, dtype=np.float32)[None, :]
    )
    U[T, cfg.end] = end_J
    P_next[cfg.end] = 1.0

    Mf = np.float32(M)

    # Backward recurrence.
    for t in range(T - 1, -1, -1):
        remaining = T - t  # days left including day t + 1
        # Once at the end the player has finished; keep terminal J at all t.
        U[t, cfg.end] = end_J

        # after_arrival: lift each village destination layer once per day
        # (theta-independent, since the buy happens before tomorrow's weather).
        lifted: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        if purchase_mode is VillagePurchaseMode.AFTER_ARRIVAL:
            for u in cfg.villages:
                lu, ow, of = lift_up(U[t + 1][u], mask, p_w_v, p_f_v)
                lp = P_next[u][ow, of]
                np.minimum(lp, 1.0, out=lp)
                lifted[u] = (lu, lp)

        P_cur = np.zeros(cash_shape, dtype=np.float32)
        P_cur[cfg.end] = 1.0

        for v in cfg.nodes:
            if v == cfg.end:
                continue
            # V_t(v, .) = sum_theta p(theta) * max_a V_{t+1}(g(., a, theta))
            Uacc_v = np.zeros((w_max + 1, f_max + 1), dtype=np.float32)
            Pacc_v = np.zeros((w_max + 1, f_max + 1), dtype=np.float32)

            for theta, p_theta in zip(theta_list, p_list):
                is_storm = theta == "sandstorm"
                bw, bf = cfg.water_consume[theta], cfg.food_consume[theta]

                # Best achievable U under this theta, for every (W, F).
                # Start at -M (no action selected yet); improvements overwrite.
                U_theta = np.full((w_max + 1, f_max + 1), -Mf, dtype=np.float32)
                P_theta = np.zeros((w_max + 1, f_max + 1), dtype=np.float32)

                def dest_layer(
                    u: int, _U=U, _P=P_next, _lifted=lifted, _t=t
                ) -> Tuple[np.ndarray, np.ndarray]:
                    if u in _lifted:
                        return _lifted[u]
                    return _U[_t + 1][u], _P[u]

                # stay (and mine) keep the player at v.
                uu, pp = dest_layer(v)
                _apply_action(U_theta, P_theta, uu, pp, bw, bf, 0)
                if v in cfg.mines:
                    _apply_action(U_theta, P_theta, uu, pp, 3 * bw, 3 * bf, cfg.mine_income)

                if not is_storm:
                    for u in cfg.adj[v]:
                        # Optimistic prune: after moving, T - (t + 1) days remain.
                        if dist_end[u] > remaining - 1:
                            continue
                        uu, pp = dest_layer(u)
                        _apply_action(U_theta, P_theta, uu, pp, 2 * bw, 2 * bf, 0)

                # start_of_day: morning buy at v (after observing theta), i.e.
                # lift the per-action max plane at village prices.
                if (
                    purchase_mode is VillagePurchaseMode.START_OF_DAY
                    and v in cfg.villages
                ):
                    U_theta, ow, of = lift_up(U_theta, mask, p_w_v, p_f_v)
                    P_theta = P_theta[ow, of]

                Uacc_v += np.float32(p_theta) * U_theta
                Pacc_v += np.float32(p_theta) * P_theta

            U[t][v] = Uacc_v
            P_cur[v] = Pacc_v

        P_next = P_cur

        if verbose:
            n_reach = int(np.sum(U[t] > -M * 0.9))
            print(
                f"[{cfg.name}/{purchase_mode.value}/M={M:.0f}] t={t:2d} reach={n_reach:7d}"
            )

    # ---- Day 0 purchase optimisation ----
    # Buying from (0, 0) at base prices: obj(q) = cash_after(q) + U[0][start, q].
    base = U[0][cfg.start]
    cash_grid = (
        cfg.init_cash
        - cfg.water_price * np.arange(w_max + 1, dtype=np.float32)[:, None]
        - cfg.food_price * np.arange(f_max + 1, dtype=np.float32)[None, :]
    )
    obj = cash_grid + base
    feasible = mask & (cash_grid >= 0) & (base > -M * 0.9)
    if not feasible.any():
        raise RuntimeError("no feasible day-0 purchase")
    obj_feas = np.where(feasible, obj, np.float32(-np.inf))
    q_w, q_f = map(int, np.unravel_index(np.argmax(obj_feas), obj_feas.shape))
    V0 = float(obj[q_w, q_f])
    prob_succ = float(min(P_next[cfg.start, q_w, q_f], 1.0))

    return SolveResult(
        V0=V0,
        prob_succ=prob_succ,
        best_q0=(q_w, q_f),
        M=M,
        purchase_mode=purchase_mode,
        level=cfg.name,
        U=U,
    )
