"""Level parameters, maps, and default weather distributions for Q2.

Q2 differs from Q1 only in that the player observes the day's weather before
acting but the future weather is i.i.d.  See ``Solution/Q2.md``.

Map data
--------
The official Word appendix draws levels 3 and 4 only schematically (no VML
connector lines), so adjacency cannot be extracted from it.  We use the
constraint-correct maps from public reference solutions
(``Jackory/CUMCM2020B/code/Map3.txt``, ``Map4.txt``), which are one-line-per-
node tables whose rows are numbered so that line ``i`` describes node ``i``.
This matches the problem prose ``1 → 2,3,...,12 → mine 9 → end 13``.

L3 has a single non-symmetric entry (node 5 does not list node 7 as a neighbour
though node 7 lists node 5).  We keep both directions here (i.e. add 5↔7);
experiments show removing it does not change the optimum strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

Weather = str  # "sunny" | "hot" | "sandstorm"


# One row per node id (1..N). Row i: ``tag nb1 nb2 ...`` where
#   tag: s=start  p=plain  c=village  k=mine  z=end
# adjacency is undirected (we symmetrise below).
_L3_TABLE: Tuple[str, ...] = (
    "s 2 4 5",       # 1
    "p 1 3 4",       # 2
    "p 2 4 8 9",     # 3
    "p 1 2 3 5 6 7", # 4
    "p 1 4 6 7",     # 5
    "p 4 5 7 12 13", # 6
    "p 4 6 11 12",   # 7
    "p 3 9",         # 8
    "k 3 8 10 11",   # 9
    "p 9 11 13",     # 10
    "p 7 9 10 12 13",# 11
    "p 6 7 11 13",   # 12
    "z 6 10 11 12",  # 13
)

_L4_TABLE: Tuple[str, ...] = (
    "s 2 6",          # 1
    "p 1 3 7",        # 2
    "p 2 4 8",        # 3
    "p 3 5 9",        # 4
    "p 4 10",         # 5
    "p 1 7 11",       # 6
    "p 2 6 8 12",     # 7
    "p 3 7 9 13",     # 8
    "p 4 8 10 14",    # 9
    "p 5 9 15",       # 10
    "p 6 12 16",      # 11
    "p 7 11 13 17",   # 12
    "p 8 12 14 18",   # 13
    "c 9 13 15 19",   # 14
    "p 10 14 20",     # 15
    "p 11 17 21",     # 16
    "p 12 16 18 22",  # 17
    "k 13 17 19 23",  # 18
    "p 14 18 20 24",  # 19
    "p 15 19 25",     # 20
    "p 16 22",        # 21
    "p 17 21 23",     # 22
    "p 18 22 24",     # 23
    "p 19 23 25",     # 24
    "z 20 24",        # 25
)


def _parse_table(table: Tuple[str, ...]) -> Tuple[
    int, int, frozenset[int], frozenset[int], Dict[int, Tuple[int, ...]]
]:
    """Return (start, end, mines, villages, adj) with symmetrised adjacency."""
    adj: Dict[int, Set[int]] = {i: set() for i in range(1, len(table) + 1)}
    start = end = -1
    mines: Set[int] = set()
    villages: Set[int] = set()
    for i, row in enumerate(table, start=1):
        parts = row.split()
        tag = parts[0]
        nbs = [int(x) for x in parts[1:]]
        if tag == "s":
            start = i
        elif tag == "z":
            end = i
        elif tag == "k":
            mines.add(i)
        elif tag == "c":
            villages.add(i)
        for u in nbs:
            adj[i].add(u)
            adj[u].add(i)
    if start < 0 or end < 0:
        raise ValueError("table missing start/end markers")
    return (
        start,
        end,
        frozenset(mines),
        frozenset(villages),
        {i: tuple(sorted(adj[i])) for i in adj},
    )


# Common resource / consumption rules for all Q2 levels.
_Q2_RES = dict(
    weight_limit=1200,
    init_cash=10000,
    water_weight=3,
    food_weight=2,
    water_price=5,
    food_price=10,
    water_consume={"sunny": 3, "hot": 9, "sandstorm": 10},
    food_consume={"sunny": 4, "hot": 9, "sandstorm": 10},
)


@dataclass(frozen=True)
class LevelConfig:
    """All fixed parameters of one Q2 game level."""

    name: str
    deadline: int  # T (last allowed arrival day)
    mine_income: int
    weather: Tuple[Weather, ...]  # observational/demonstration sequence; weather[d] for d=1..T
    p_weather: Dict[Weather, float]  # iid distribution used by DP
    weight_limit: int
    init_cash: int
    water_weight: int
    food_weight: int
    water_price: int
    food_price: int
    water_consume: Dict[Weather, int]
    food_consume: Dict[Weather, int]
    start: int
    end: int
    mines: frozenset[int]
    villages: frozenset[int]
    adj: Dict[int, Tuple[int, ...]]
    nodes: Tuple[int, ...]


# Probability distribution defaults (chosen in our Q2 discussion):
#   L3 has no sandstorm; use the observed empirical frequency 6 sunny / 4 hot.
#   L4 uses p = (0.5, 0.4, 0.1); sensitivity sweep varies p_storm.
_P_L3: Dict[Weather, float] = {"sunny": 0.6, "hot": 0.4, "sandstorm": 0.0}
_P_L4: Dict[Weather, float] = {"sunny": 0.5, "hot": 0.4, "sandstorm": 0.1}

# Demonstration weather sequence (used only for sample trajectories).
# L3 actual (10 days, no storms):
_WEATHER_L3: Tuple[Weather, ...] = (
    "",
    "sunny", "hot", "sunny", "sunny", "sunny",
    "sunny", "hot", "hot", "hot", "hot",
)
# L4: problem text gives no concrete sequence; pick one plausible 30-day realisation
# drawn from the same iid distribution as used by the DP (for demonstration only).
_WEATHER_L4: Tuple[Weather, ...] = (
    "",
    "sunny", "hot", "sunny", "sunny", "hot",
    "sunny", "sunny", "sandstorm", "hot", "sunny",
    "hot", "sunny", "hot", "sunny", "hot",
    "sunny", "sunny", "hot", "sunny", "hot",
    "sunny", "sunny", "hot", "hot", "sunny",
    "hot", "sunny", "sunny", "hot", "sunny",
)


def level3(p_weather: Dict[Weather, float] | None = None) -> LevelConfig:
    start, end, mines, villages, adj = _parse_table(_L3_TABLE)
    n = len(_L3_TABLE)
    return LevelConfig(
        name="level3",
        deadline=10,
        mine_income=200,
        weather=_WEATHER_L3,
        p_weather=p_weather if p_weather is not None else dict(_P_L3),
        **_Q2_RES,
        start=start,
        end=end,
        mines=mines,
        villages=villages,
        adj=adj,
        nodes=tuple(range(1, n + 1)),
    )


def level4(p_weather: Dict[Weather, float] | None = None) -> LevelConfig:
    start, end, mines, villages, adj = _parse_table(_L4_TABLE)
    n = len(_L4_TABLE)
    return LevelConfig(
        name="level4",
        deadline=30,
        mine_income=1000,
        weather=_WEATHER_L4,
        p_weather=p_weather if p_weather is not None else dict(_P_L4),
        **_Q2_RES,
        start=start,
        end=end,
        mines=mines,
        villages=villages,
        adj=adj,
        nodes=tuple(range(1, n + 1)),
    )


def load_level(
    which: str, p_weather: Dict[Weather, float] | None = None
) -> LevelConfig:
    if which in ("3", "level3", "l3", "third"):
        return level3(p_weather)
    if which in ("4", "level4", "l4", "fourth"):
        return level4(p_weather)
    raise ValueError(f"unknown level: {which}")