# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CUMCM 2020B — "Crossing the Desert" (穿越沙漠)**

A mathematical modeling competition problem about optimal strategy in a desert-crossing game. **Q1** (single player, known weather) and **Q2** (single player, only current-day weather known) are implemented.

## Commands

```bash
# Run solver (both levels, start_of_day mode)
uv run python -m q1.solve

# Level-specific
uv run python -m q1.solve --level 1
uv run python -m q1.solve --level 2

# Village purchase timing variants
uv run python -m q1.solve --purchase-mode start_of_day
uv run python -m q1.solve --purchase-mode after_arrival

# Quiet mode (suppress per-day DP progress)
uv run python -m q1.solve --quiet

# Use a custom output path
uv run python -m q1.solve --out path/to/out.xlsx

# Programmatic usage
uv run python -c "from q1 import solve, load_level; print(solve(load_level('1')).best_value)"

# Q2 stochastic solver (levels 3 and 4)
uv run python -m q2.solve --quiet
uv run python -m q2.solve --quiet --sensitivity
```

Dependencies are managed via `uv` (Python ≥3.13). Install with `uv sync`.

## Code Architecture

### Module layout (`q1/`)

| Module | Responsibility |
|--------|---------------|
| `data.py` | `LevelConfig` dataclass with all game parameters; level-specific map graphs (level 1 = explicit edge list for 27 nodes; level 2 = 8×8 staggered hex grid for 64 nodes) |
| `model.py` | Game rule layer — consumption multipliers, village prices, terminal value function, weight limit check; enums (`ActionType`, `VillagePurchaseMode`); `DayRecord` dataclass |
| `dp.py` | Core forward-DP solver — `buy_knapsack()` for exact village purchase, per-day vectorized transitions via numpy slices, parent-pointer backtracking |
| `verify.py` | Independent trajectory replay checker (reimplements game rules to verify DP output end-to-end) |
| `solve.py` | CLI entry point (`argparse`); prints trajectory tables; writes `Result.xlsx` in the official column layout (A–E for L1, G–K for L2) |
| `__init__.py` | Public exports: `load_level`, `solve`, `level1`, `level2`, `VillagePurchaseMode` |

### Algorithm (Q1 forward DP)

The state space is `cash[v, W, F]` — the maximum cash attainable at a given (node, water, food). Cash is monotone w.r.t. feasibility and the objective, so keeping only the max is exact.

- **Day 0**: Initial knapsack purchase at base prices (no action, no consumption).
- **Day t (1..T)**: Optional village buy → one action (stay/move/mine) → consume → mine income → optional village buy (`after_arrival` mode) → if at end node, score.
- **Village purchase**: solved via two 1D sweeps (`buy_knapsack` — water then food), exact because unit prices are constant.
- **Pruning**: BFS shortest-path-to-end bound; end-node absorption; NEG sentinel for unreachable states.
- **Backtracking**: parent pointers record `(prev_v, prev_W, prev_F, action, buy_W, buy_F, cash)` per improved state.

### Village purchase modes

- `START_OF_DAY` (default) — buy only if the player **starts** the day at a village (was already there at end of previous day). Matches the mathematical model in `docs/Solution/Q1.md`.
- `AFTER_ARRIVAL` — act first, then buy if the **ending** position is a village. Looser reading of "经过或在村庄停留时可随时购买".

Both modes yield the same terminal value for level 1 (10470), differing only in which day purchases occur.

### Results

| Level | Optimal value | Arrival day | End state |
|-------|--------------|-------------|-----------|
| 1     | 10470        | 24          | (10470, 0, 0) |
| 2     | 12730        | 30          | (12730, 0, 0) |

### Documentation

- `docs/Q1Solve.md` — Detailed implementation notes (Chinese), including algorithm explanation, code structure, and key design decisions.
- `docs/Solution/Q1.md` — Full mathematical model (Chinese, LaTeX), Bellman equation derivation.
- `docs/2020B/problem.md` — Original problem statement.
- `docs/2020B/appendix.md` — All 6 levels' parameters, maps, weather tables.

### Key design decisions

- **Forward DP not backward Bellman**: cash range is large (10000+ mining income), so a backward table over (v, W, F, C) would be too large. Forward DP with (v, W, F) → max C is equivalent and tractable.
- **numpy vectorization**: per-node transitions are done via array slicing (`src[cw:, cf:]` → `dest[:w_max+1-cw, :f_max+1-cf]`), avoiding Python loops over (W, F) cells.
- **int32 cash**: `NEG = -10**9` sentinel for unreachable states, well below any real cash value.
- **No (W, F) Pareto pruning**: holding more resources consumes weight capacity that could be used for cheaper village purchases, so "more is better" does not hold across the resource plane.

### Future extensions needed

- **Q2**: Weather-unknown (online) optimal strategy — needs a different algorithmic approach (stochastic DP or robust optimization).
- **Q3**: Multi-player game with congestion effects — requires game-theoretic modeling.
