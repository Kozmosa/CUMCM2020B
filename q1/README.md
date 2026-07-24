# Q1 solver

Known-weather single-player optimal strategy (forward DP).

```bash
uv run python -m q1.solve                          # both levels, start_of_day
uv run python -m q1.solve --level 1
uv run python -m q1.solve --purchase-mode after_arrival
uv run python -m q1.solve --quiet
```

See [`docs/Q1Solve.md`](../docs/Q1Solve.md) for the full implementation note.
