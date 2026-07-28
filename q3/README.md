# Q3 exact solver — first implementation stage

This directory contains the exact, lossless core for question 3:

- fixed-point (`scale = 6`) money and terminal payoffs;
- active/finished/failed player states;
- exact individual action and purchase enumeration;
- simultaneous road, mine, and village interactions;
- independent scalar and NumPy-vectorized transitions;
- pure-strategy Nash detection on coupled feasible action sets;
- deterministic multiple-equilibrium selection;
- player-permutation canonicalization and sparse stochastic recursion;
- explicit CPU/memory safety limits that stop instead of approximating.

The current workstation-safe backend materializes the complete payoff tensor
for a small stage game.  It is intended for correctness tests and short-horizon
states.  The large purchase games at day 0 and villages still require the
documented chunked exact backend before a full 30-day level-6 run is practical.

## Commands

Run all tests:

```bash
.venv/bin/python -m unittest discover -s q3/tests -v
```

Run the safe three-player smoke solve:

```bash
.venv/bin/python -m q3.solve_q3_2 --mode smoke
```

Solve a small late level-6 state under explicit limits:

```bash
.venv/bin/python -m q3.solve_q3_2 \
  --mode level6-state --day 29 --position 24 --water 60 --food 60
```

Attempting the full initial-purchase game is allowed only through explicit
limits.  Exceeding them returns `RESOURCE_LIMIT` without selecting Top-K
actions, quantizing resources, or enabling any other approximation.

Numba is not installed in the current Python 3.13 environment, so the batch
kernels automatically use the NumPy fallback here.  `interaction.py` already
contains an optional cached `@njit` implementation with identical inputs and
outputs; installing a compatible Numba build on the 128-core machine activates
it without changing solver semantics.
