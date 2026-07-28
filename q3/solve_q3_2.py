"""Safe command-line entry point for the Q3.2 exact solver."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from time import perf_counter

from .data import level6, tiny_level6
from .interaction import NUMBA_AVAILABLE
from .model import PlayerState, Status
from .stochastic_dp import ExactQ3Solver, ResourceLimitExceeded, SolverLimits


def _smoke_state(cfg) -> tuple[PlayerState, ...]:
    # End of day 1 at the mine/village.  With one day remaining, all three
    # players can exactly afford the congested move to the end.
    player = PlayerState(
        Status.ACTIVE,
        position=2,
        water=6,
        food=6,
        cash_scaled=cfg.init_cash_scaled,
    )
    return tuple(player for _ in range(cfg.n_players))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact pure-strategy Q3.2 solver")
    parser.add_argument(
        "--mode",
        choices=("smoke", "level6-state", "level6-initial"),
        default="smoke",
        help=(
            "smoke: tiny one-step validation; level6-state: solve one specified "
            "symmetric state; level6-initial: attempt exact day-0 game under limits"
        ),
    )
    parser.add_argument("--day", type=int, default=29)
    parser.add_argument("--position", type=int, default=24)
    parser.add_argument("--water", type=int, default=60)
    parser.add_argument("--food", type=int, default=60)
    parser.add_argument("--cash", type=int, default=10_000)
    parser.add_argument("--max-actions", type=int, default=4_096)
    parser.add_argument("--max-profiles", type=int, default=250_000)
    parser.add_argument("--max-states", type=int, default=100_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    limits = SolverLimits(args.max_actions, args.max_profiles, args.max_states)
    if args.mode == "smoke":
        cfg = tiny_level6()
        day = 1
        state = _smoke_state(cfg)
    else:
        cfg = level6()
        day = args.day
        player = PlayerState(
            Status.ACTIVE,
            position=args.position,
            water=args.water,
            food=args.food,
            cash_scaled=args.cash * cfg.money_scale,
        )
        state = tuple(player for _ in range(cfg.n_players))

    solver = ExactQ3Solver(cfg, limits=limits)
    started = perf_counter()
    try:
        if args.mode == "level6-initial":
            result = solver.solve_initial_purchases()
            payload = {
                "value": result.value.value,
                "success": result.value.success,
                "purchases": [action.label() for action in result.purchases],
            }
        else:
            value = solver.solve_state(day, state)
            payload = {
                "value": value.value,
                "success": value.success,
                "policy": {
                    weather: [
                        action.label() for action in solver.policy_for(day, state, weather)
                    ]
                    for weather in cfg.weather_order
                },
            }
    except ResourceLimitExceeded as exc:
        print(f"RESOURCE_LIMIT: {exc}")
        print("The exact search was stopped without truncation or approximation.")
        return 2

    payload["elapsed_seconds"] = perf_counter() - started
    payload["numba_available"] = NUMBA_AVAILABLE
    payload["stats"] = asdict(solver.stats)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
