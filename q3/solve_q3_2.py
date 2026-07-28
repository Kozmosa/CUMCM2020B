"""Safe command-line entry point for the Q3.2 exact solver."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from dataclasses import asdict
from time import perf_counter

from .data import level6, tiny_level6
from .interaction import NUMBA_AVAILABLE
from .model import PlayerState, Status
from .stage_game import NoPureEquilibrium
from .stochastic_dp import (
    ExactQ3Solver,
    ResourceLimitExceeded,
    SearchCancelled,
    SolverLimits,
)


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
    parser.add_argument("--max-actions", type=int, default=1_000_000)
    parser.add_argument("--max-profiles", type=int, default=250_000)
    parser.add_argument("--max-states", type=int, default=100_000)
    parser.add_argument("--chunk-size", type=int, default=4_096)
    parser.add_argument("--max-stage-evaluations", type=int, default=5_000_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every-states", type=int, default=0)
    parser.add_argument("--checkpoint-every-pairs", type=int, default=0)
    parser.add_argument("--disable-bound-pruning", action="store_true")
    parser.add_argument("--bound-pruning-slack", type=float, default=1e-6)
    parser.add_argument("--record-pruning-certificates", action="store_true")
    parser.add_argument("--max-pruning-certificates", type=int, default=1_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    limits = SolverLimits(
        max_actions_per_player=args.max_actions,
        max_joint_profiles=args.max_profiles,
        max_cached_states=args.max_states,
        profile_chunk_size=args.chunk_size,
        max_stage_profiles_evaluated=args.max_stage_evaluations,
    )
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

    solver = ExactQ3Solver(
        cfg,
        limits=limits,
        workers=args.workers,
        checkpoint_path=args.checkpoint,
        checkpoint_every_states=args.checkpoint_every_states,
        checkpoint_every_pairs=args.checkpoint_every_pairs,
        enable_bound_pruning=not args.disable_bound_pruning,
        bound_pruning_slack=args.bound_pruning_slack,
        record_pruning_certificates=args.record_pruning_certificates,
        max_pruning_certificates=args.max_pruning_certificates,
    )

    def handle_termination(signum, frame) -> None:
        solver.request_stop()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_termination)
    if args.resume:
        if not args.checkpoint:
            parser.error("--resume requires --checkpoint")
        solver.load_checkpoint()
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
                        action.label()
                        for action in solver.policy_for(day, state, weather)
                    ]
                    for weather in cfg.weather_order
                },
            }
    except (ResourceLimitExceeded, NoPureEquilibrium) as exc:
        solver.request_stop()
        if args.checkpoint:
            solver.save_checkpoint()
        stopped = {
            "status": "SEARCH_STOPPED",
            "error": str(exc),
            "elapsed_seconds": perf_counter() - started,
            "numba_available": NUMBA_AVAILABLE,
            "gil_enabled": (
                sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
            ),
            "stats": asdict(solver.stats),
        }
        if args.checkpoint:
            stopped["checkpoint"] = args.checkpoint
        if args.record_pruning_certificates:
            stopped["pruning_certificates"] = [
                asdict(certificate) for certificate in solver.pruning_certificates
            ]
        solver.close()
        print(json.dumps(stopped, ensure_ascii=False, indent=2))
        return 2
    except (KeyboardInterrupt, SearchCancelled):
        solver.request_stop()
        if args.checkpoint:
            solver.save_checkpoint()
        interrupted = {
            "status": "INTERRUPTED",
            "elapsed_seconds": perf_counter() - started,
            "stats": asdict(solver.stats),
        }
        if args.checkpoint:
            interrupted["checkpoint"] = args.checkpoint
        if args.record_pruning_certificates:
            interrupted["pruning_certificates"] = [
                asdict(certificate) for certificate in solver.pruning_certificates
            ]
        solver.close()
        print(json.dumps(interrupted, ensure_ascii=False, indent=2))
        return 130

    payload["elapsed_seconds"] = perf_counter() - started
    payload["numba_available"] = NUMBA_AVAILABLE
    payload["gil_enabled"] = (
        sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
    )
    payload["stats"] = asdict(solver.stats)
    if args.checkpoint:
        solver.save_checkpoint()
        payload["checkpoint"] = args.checkpoint
    if args.record_pruning_certificates:
        payload["pruning_certificates"] = [
            asdict(certificate) for certificate in solver.pruning_certificates
        ]
    solver.close()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
