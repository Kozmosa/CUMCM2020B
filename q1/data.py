"""Level parameters, weather sequences, and map graphs for Q1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, Tuple


Weather = str  # "sunny" | "hot" | "sandstorm"


@dataclass(frozen=True)
class LevelConfig:
    """All fixed parameters of one game level."""

    name: str
    weight_limit: int
    init_cash: int
    deadline: int  # T: must arrive on day tau <= T
    mine_income: int
    water_weight: int
    food_weight: int
    water_price: int  # base price (day-0 start); village = 2x
    food_price: int
    water_consume: Dict[Weather, int]  # base boxes / day by weather
    food_consume: Dict[Weather, int]
    weather: Tuple[Weather, ...]  # weather[d] for day d = 1..T; index 0 unused
    start: int
    end: int
    mines: frozenset[int]
    villages: frozenset[int]
    adj: Dict[int, Tuple[int, ...]]  # node -> neighbours (no self-loops)
    nodes: Tuple[int, ...]


# Levels 1 and 2 share the same 30-day weather forecast.
_WEATHER_L1L2: Tuple[Weather, ...] = (
    "",  # day 0 placeholder
    "hot", "hot", "sunny", "sandstorm", "sunny",
    "hot", "sandstorm", "sunny", "hot", "hot",
    "sandstorm", "hot", "sunny", "hot", "hot",
    "hot", "sandstorm", "sandstorm", "hot", "hot",
    "sunny", "sunny", "hot", "sunny", "sandstorm",
    "hot", "sunny", "sunny", "hot", "hot",
)

# Level-1 undirected edges (27 nodes). Cross-checked against the official map
# and public solutions (e.g. skr2005/mcm_2020b_p1, youngzhou1999/CUMCM2020B).
_EDGES_L1: Tuple[Tuple[int, int], ...] = (
    (1, 2), (1, 25),
    (2, 3),
    (3, 4), (3, 25),
    (4, 5), (4, 24), (4, 25),
    (5, 6), (5, 24),
    (6, 7), (6, 23), (6, 24),
    (7, 8), (7, 22),
    (8, 9), (8, 22),
    (9, 10), (9, 15), (9, 16), (9, 17), (9, 21), (9, 22),
    (10, 11), (10, 13), (10, 15),
    (11, 12), (11, 13),
    (12, 13), (12, 14),
    (13, 14), (13, 15),
    (14, 15), (14, 16),
    (15, 16),
    (16, 17), (16, 18),
    (17, 18), (17, 21),
    (18, 19), (18, 20),
    (19, 20),
    (20, 21),
    (21, 22), (21, 23), (21, 27),
    (22, 23),
    (23, 24), (23, 26),
    (24, 25), (24, 26),
    (25, 26),
    (26, 27),
)


def _adj_from_edges(
    edges: Tuple[Tuple[int, int], ...], n_nodes: int
) -> Dict[int, Tuple[int, ...]]:
    g: Dict[int, Set[int]] = {i: set() for i in range(1, n_nodes + 1)}
    for a, b in edges:
        g[a].add(b)
        g[b].add(a)
    return {i: tuple(sorted(ns)) for i, ns in g.items()}


def _hex_grid_adj(rows: int = 8, cols: int = 8) -> Dict[int, Tuple[int, ...]]:
    """Level-2 map: 8x8 staggered (hex-offset) grid, id = y*cols + x + 1.

    Even rows connect diagonally to (y±1, x-1); odd rows to (y±1, x+1).
    Orthogonal neighbours always connect when in bounds.
    """

    def yx_to_id(y: int, x: int) -> int | None:
        if 0 <= y < rows and 0 <= x < cols:
            return y * cols + x + 1
        return None

    adj: Dict[int, Set[int]] = {}
    for y in range(rows):
        for x in range(cols):
            i = yx_to_id(y, x)
            assert i is not None
            neigh: Set[int] = set()
            for ny, nx in ((y, x - 1), (y, x + 1), (y - 1, x), (y + 1, x)):
                j = yx_to_id(ny, nx)
                if j is not None:
                    neigh.add(j)
            extras = (
                ((y - 1, x + 1), (y + 1, x + 1))
                if y % 2 == 1
                else ((y - 1, x - 1), (y + 1, x - 1))
            )
            for ny, nx in extras:
                j = yx_to_id(ny, nx)
                if j is not None:
                    neigh.add(j)
            adj[i] = neigh
    return {i: tuple(sorted(ns)) for i, ns in adj.items()}


_COMMON = dict(
    weight_limit=1200,
    init_cash=10000,
    deadline=30,
    mine_income=1000,
    water_weight=3,
    food_weight=2,
    water_price=5,
    food_price=10,
    water_consume={"sunny": 5, "hot": 8, "sandstorm": 10},
    food_consume={"sunny": 7, "hot": 6, "sandstorm": 10},
    weather=_WEATHER_L1L2,
)


def level1() -> LevelConfig:
    n = 27
    return LevelConfig(
        name="level1",
        start=1,
        end=27,
        mines=frozenset({12}),
        villages=frozenset({15}),
        adj=_adj_from_edges(_EDGES_L1, n),
        nodes=tuple(range(1, n + 1)),
        **_COMMON,
    )


def level2() -> LevelConfig:
    n = 64
    adj = _hex_grid_adj(8, 8)
    # Sanity checks against known public-solution assertions.
    assert set(adj[1]) == {2, 9}
    assert set(adj[30]) == {22, 23, 29, 31, 38, 39}
    assert set(adj[64]) == {56, 63}
    return LevelConfig(
        name="level2",
        start=1,
        end=64,
        mines=frozenset({30, 55}),
        villages=frozenset({39, 62}),
        adj=adj,
        nodes=tuple(range(1, n + 1)),
        **_COMMON,
    )


def load_level(which: str) -> LevelConfig:
    if which in ("1", "level1", "l1", "first"):
        return level1()
    if which in ("2", "level2", "l2", "second"):
        return level2()
    raise ValueError(f"unknown level: {which}")
