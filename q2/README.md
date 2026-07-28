# Q2 solver

Unknown-future-weather single-player optimal strategy (backward induction DP;
only the current day's weather is observed before acting).

The implemented objective is a **single penalty-adjusted expected payoff**:
successful paths receive cash plus the half-price resource refund, while a
failed path receives its remaining cash minus `M`. `P_succ` is the success
probability associated with the payoff-optimal policy; it is not itself the
primary optimisation objective.

```bash
uv run python -m q2.solve                          # both levels, start_of_day
uv run python -m q2.solve --level 4
uv run python -m q2.solve --purchase-mode after_arrival
uv run python -m q2.solve --M 1e9                  # failure-penalty choice
uv run python -m q2.solve --sensitivity            # M / p_storm sweeps
uv run python -m q2.solve --quiet
```

See [`docs/Q2Solve.md`](../docs/Q2Solve.md) for the full implementation note.
