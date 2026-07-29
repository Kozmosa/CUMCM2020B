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
- explicit CPU/memory safety limits that stop instead of approximating;
- known-weather open-loop replay and best responses that retain every opponent's full state;
- a compact 67-bit/Numba Q3.1 frontier that removes opponent cash only when the map has no villages;
- adaptive restricted games with full unilateral-deviation scans and explicit regret bounds;
- compact structure-of-arrays village action spaces and fused Numba unilateral scans;
- an exact purchase-lattice max-pyramid oracle with CPU/Numba and optional CUDA screening;
- symmetric-stage reduction and fully verified best-response fixed-point shortcuts;
- finite-support mixed fallback through deterministic NashConv minimization.
- a submission-gated policy-response Monte Carlo backend with exact joint rule
  replay, complete short-route screening, independent holdout audits, and
  player-symmetry reduction.

Small stage games use a dense payoff tensor.  Larger games automatically switch
to a bounded-memory exact best-response scan.  A proven optimistic-reachability
test removes 31,676 strictly dominated day-0 purchases, leaving 88,925 actions
per player and about 7.032e14 ordered profiles.  The adaptive backend does not
enumerate those ordered profiles: it solves small restricted games and certifies
the selected pure profile against the complete unilateral action space.  A full
30-day level-6 solve is still a budgeted computation, but the previous tensor,
Python-action-object, and duplicate-checkpoint blockers have been removed.

## Commands

Run all tests:

```bash
.venv/bin/python -m unittest discover -s q3/tests -v
```

Run the safe three-player smoke solve:

```bash
uv run python -m q3.solve_q3_2 --backend adaptive --mode smoke
```

Run the submission-gated policy-response Q3.2 solver:

```bash
.venv/bin/python -m q3.solve_q3_2 \
  --backend heuristic --mode level6-initial \
  --heuristic-episodes 5000 \
  --heuristic-audit-episodes 100000 \
  --heuristic-audit-replicates 3 \
  --heuristic-max-policies 32 \
  --workers 16 \
  --wall-hours 8 --memory-gib 16 \
  --equilibrium pure-mixed --output q3/output/q3_2_submission
```

This backend keeps the exact multiplayer transition and payoff rules.  It
screens all 804 simple level-6 routes of at most 12 moves, expands a cached
empirical game with profitable policy responses, and audits the final profile
on three disjoint weather holdouts.  It exits successfully only when success
lower bounds and the 100/200-yuan regret gates pass.  It remains a broad
policy-class result rather than a full-action Nash certificate.  See
[`docs/Q3-2-Heuristic.md`](../docs/Q3-2-Heuristic.md) for the complete list of
simplifications.

On ordinary CPython 3.13 the heuristic simulation hot path uses multiple
worker processes (up to 16 by default) rather than Python threads, so it is not
serialized by the GIL. Progress is emitted as JSONL throughout training,
response search, stability checks, and holdout audits; redirect stdout/stderr
to `run.log` and use `tail -f` for live monitoring.

Run the Q3.1 smoke solve or the official fifth level:

```bash
uv run python -m q3.solve_q3_1 --tiny --output q3/output/q3_1_smoke
uv run python -m q3.solve_q3_1 --output q3/output/q3_1_level5
```

The official fifth level now returns a certified pure profile with zero full
unilateral regret.  Use `--benchmark-best-response` to profile one exact
oracle call, or `--disable-compact-numba` to retain the scalar reference.

Run the baseline plus the four non-duplicate one-factor sensitivity points:

```bash
uv run python -m q3.sensitivity --output q3/output/sensitivity
```

Solve a small late level-6 state under explicit limits:

```bash
uv run python -m q3.solve_q3_2 --backend exact \
  --mode level6-state --day 29 --position 24 --water 60 --food 60
```

Run a late state with an atomic checkpoint and explicit state limit:

```bash
uv run python -m q3.solve_q3_2 --backend adaptive \
  --mode level6-state --day 27 --position 22 --water 120 --food 120 \
  --workers 1 --max-states 200000 \
  --checkpoint /tmp/q3-day27.chk --checkpoint-every-states 20000
```

Resume the same calculation:

```bash
uv run python -m q3.solve_q3_2 --backend adaptive \
  --mode level6-state --day 27 --position 22 --water 120 --food 120 \
  --workers 1 --max-states 200000 \
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

The supported formal environment is ordinary CPython 3.13 with the GIL and
Numba enabled.  Recreate it with:

```bash
uv sync --python 3.13
.venv/bin/python -c 'import sys; print(sys.version, sys._is_gil_enabled())'
```

The CLI automatically uses one Python worker when the GIL is enabled.  Setting
more workers on ordinary CPython usually slows recursive state evaluation;
Numba and NumPy still accelerate the large numeric batches internally.

Attempting the full initial-purchase game is allowed only through explicit
limits.  The formal adaptive command remains available for research-grade
full-action certification:

```bash
uv run python -m q3.solve_q3_2 \
  --backend adaptive --mode level6-initial \
  --quality-regret 10 --wall-hours 24 --memory-gib 256 \
  --equilibrium pure-mixed --workers 1 --bound-threads 64 \
  --purchase-oracle auto --purchase-threads 64 \
  --max-states 30000000 --max-stage-evaluations 50000000 \
  --checkpoint q3/output/level6.chk \
  --checkpoint-every-states 1000000 \
  --output q3/output/level6
```

Exceeding a budget returns `SEARCH_STOPPED` with an explicit unresolved gap;
it never labels a finite candidate set as an exact solution.

The first formal adaptive experiment was safely stopped after 38,095 seconds
(10.58 hours).  It retained 5,131,301 non-terminal states, settled 138,876,780
terminal leaves, scanned about 1.804e11 unilateral-deviation upper bounds,
peaked near 7.07 GiB RSS, and wrote an approximately 1.8 GiB resumable v2
checkpoint under `q3/output/level6-formal/`.  It had not yet produced a root
policy or finite regret gap, so the result remains `SEARCH_STOPPED`.

Numba is included in the UV environment and activates the fused transition
kernel and the day-layer resource-bound DP automatically.  Python successor
workers remain at one under the GIL; `--bound-threads` controls independent
native `prange` work and defaults to at most 64 threads.  `transition.py`
retains an identical NumPy fallback.
The compact village action representation stores integer columns rather than
hundreds of thousands of Python objects and materializes only selected or
profitable actions.

`--purchase-oracle auto` uses the exact max-pyramid region oracle.  It stores
an outward-rounded maximum for each rectangular water/food region and rejects
the whole region when that maximum cannot improve the current payoff.  The
remaining leaves are checked by the same fused transition kernel before
recursion.  `--purchase-oracle cuda` enables the complete CUDA point scanner
for explicit GPU profiling; on the reference A800, branch divergence and many
short launches make it slower than the region oracle, so CUDA is not selected
by `auto`.  `--purchase-oracle off` retains the old complete point scan as a
reference implementation.

## Current CPython 3.13 profile

On the reference 128-core/A800 host, using one Python worker:

- day-29 village state: 5.19 s before compact action arrays, 0.84 s after that
  change, and 0.71 s after terminal array settlement;
- the resource-aware upper bound is stored in lazy `float64` day layers.  A
  complete level-6 suffix is at most about 636 MiB; the five-minute root probe
  needed only day 28--29, 45.3 MiB, built in 0.051 s with 64 native threads;
- before the region oracle, a 60-second root probe scanned 213.6 million
  unilateral deviations and evaluated 60,290 states;
- with the exact max-pyramid oracle, the same 60-second probe certifies 770.2
  million deviations and evaluates 254,841 states: 3.61x deviation throughput
  and 4.23x state throughput.  The oracle itself takes 2.15 seconds, prunes
  99.998% of purchase actions, and uses 331 MiB peak RSS;
- the new 60-second probe reaches almost the same 255k state evaluations as
  the previous five-minute point-scan probe, an end-to-end improvement of
  about 4.9x at that frontier;
- terminal leaves are settled directly and are not retained.  The probe made
  258,237 state evaluations, including 243,495 terminal settlements, while
  retaining only 14,742 non-terminal value states;
- migrating the previous 101,534-row checkpoint drops its 87,566 terminal
  rows and rewrites packed policy keys, reducing it from 21 MiB to 4.9 MiB.
  Long-run v2 storage should be budgeted at roughly 0.2--0.3 KiB per retained
  non-terminal state plus policy metadata.

The short probe did not predict the later state distribution accurately: the
10.58-hour formal run was still expanding the recursive game and had not
reached the root certificate.  Completion time for the full adaptive backend
is therefore unresolved and remains gated by the reported root regret upper
bound, not by elapsed time alone.  For a contest-scale answer, use the
submission-gated heuristic backend and accept only
`SUBMISSION_READY_EMPIRICAL_EQ`; intermediate empirical equilibria are not a
paper-ready conclusion.
