# Q3 game solvers

The correctness argument for state compression, lossless pruning, parallel
execution, and checkpoints is collected in [[Q3-Exactness]].

This directory now contains the shared exact rule core plus Q3.1 and Q3.2 solvers:

- fixed-point (`scale = 6`) money and terminal payoffs;
- active/finished/failed player states;
- exact individual action and purchase enumeration;
- simultaneous road, mine, and village interactions;
- independent scalar and NumPy-vectorized transitions;
- pure-strategy Nash detection on coupled feasible action sets;
- deterministic multiple-equilibrium selection;
- player-permutation canonicalization and sparse stochastic recursion;
- lossless action-skeleton filtering before purchase Cartesian products;
- resource-aware, outward-rounded single-player upper bounds for certified best-response pruning;
- blockwise exact best-response and pure-Nash search for large stage games;
- block-local canonical successor deduplication and configurable workers;
- v2 directory checkpoints with per-day NumPy layers and v1 pickle migration;
- explicit CPU/memory safety limits that stop instead of approximating.
- known-weather open-loop replay and best responses that retain every opponent's full state;
- adaptive restricted games with full unilateral-deviation scans and explicit regret bounds;
- finite-support mixed fallback through deterministic NashConv minimization.

Small stage games use a dense payoff tensor.  Larger games automatically switch
to a bounded-memory exact best-response scan.  A proven optimistic-reachability
test removes 31,676 strictly dominated day-0 purchases, leaving 88,925 actions
per player and about 7.032e14 ordered profiles.  Identical player states share
one immutable action tuple.  A complete 30-day level-6 solve therefore remains
a long-running research computation even though the previous tensor-memory
blocker has been removed.

## Commands

Run all tests:

```bash
.venv/bin/python -m unittest discover -s q3/tests -v
```

Run the safe three-player smoke solve:

```bash
uv run python -m q3.solve_q3_2 --backend adaptive --mode smoke
```

Run the Q3.1 smoke solve or the official fifth level:

```bash
uv run python -m q3.solve_q3_1 --tiny --output q3/output/q3_1_smoke
uv run python -m q3.solve_q3_1 --output q3/output/q3_1_level5
```

Run the baseline plus the four non-duplicate one-factor sensitivity points:

```bash
uv run python -m q3.sensitivity --output q3/output/sensitivity
```

Solve a small late level-6 state under explicit limits:

```bash
uv run python -m q3.solve_q3_2 --backend exact \
  --mode level6-state --day 29 --position 24 --water 60 --food 60
```

Run with parallel successor evaluation and an atomic checkpoint:

```bash
uv run python -m q3.solve_q3_2 --backend adaptive \
  --mode level6-state --day 27 --position 22 --water 120 --food 120 \
  --workers 16 --max-states 200000 \
  --checkpoint /tmp/q3-day27.chk --checkpoint-every-states 20000
```

Resume the same calculation:

```bash
uv run python -m q3.solve_q3_2 --backend adaptive \
  --mode level6-state --day 27 --position 22 --water 120 --food 120 \
  --workers 16 --max-states 200000 \
  --checkpoint /tmp/q3-day27.chk --resume
```

Compare the certified bound pruning against the unpruned exact scan:

```bash
.venv/bin/python -m q3.solve_q3_2 --mode smoke \
  --max-profiles 1 --chunk-size 1 --disable-bound-pruning
.venv/bin/python -m q3.solve_q3_2 --mode smoke \
  --max-profiles 1 --chunk-size 1 \
  --record-pruning-certificates --max-pruning-certificates 1000
```

On CPython 3.13t the thread backend can execute Python recursion without the
GIL.  Numba/llvmlite currently has no compatible wheel in this environment, so
the free-threaded environment uses the exact NumPy interaction fallback:

```bash
uv python install 3.13t
uv venv .venv-ft --python 3.13t
uv pip install --python .venv-ft/bin/python 'numpy>=2.4.6,<2.5'
.venv-ft/bin/python -m q3.solve_q3_2 --mode smoke --workers 16
```

Attempting the full initial-purchase game is allowed only through explicit
limits.  The formal adaptive command is:

```bash
uv run python -m q3.solve_q3_2 \
  --backend adaptive --mode level6-initial \
  --quality-regret 10 --wall-hours 24 --memory-gib 256 \
  --equilibrium pure-mixed \
  --checkpoint q3/output/level6.chk \
  --output q3/output/level6
```

Exceeding a budget returns `SEARCH_STOPPED` with an explicit unresolved gap;
it never labels a finite candidate set as an exact solution.

Numba is included in the UV environment and activates the cached `@njit`
interaction-counting kernel automatically.  `interaction.py` retains an
identical NumPy fallback for environments where Numba is unavailable.  This
accelerates a tested hot kernel without changing solver semantics, but the
full solve still needs stronger proven value bounds to reduce the enormous
day-0 best-response search.
