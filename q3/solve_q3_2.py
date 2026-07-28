"""Command-line entry point for the exact and adaptive Q3.2 backends."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

from .adaptive import AdaptiveOptions, AdaptiveQ3Solver, solve_q3_2
from .data import level6, tiny_level6
from .interaction import NUMBA_AVAILABLE
from .model import PlayerState, Status
from .runtime import BudgetManager
from .reports import json_safe
from .stage_game import NoPureEquilibrium
from .stochastic_dp import (
    ExactQ3Solver,
    ResourceLimitExceeded,
    SearchCancelled,
    SolverLimits,
)


def _smoke_state(cfg) -> tuple[PlayerState, ...]:
    player = PlayerState(
        Status.ACTIVE,
        position=2,
        water=6,
        food=6,
        cash_scaled=cfg.init_cash_scaled,
    )
    return tuple(player for _ in range(cfg.n_players))


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CUMCM 2020B Q3.2 solver")
    parser.add_argument("--backend", choices=("exact", "adaptive"), default="adaptive")
    parser.add_argument(
        "--mode",
        choices=("smoke", "level6-state", "level6-initial"),
        default="smoke",
    )
    parser.add_argument("--day", type=int, default=29)
    parser.add_argument("--position", type=int, default=24)
    parser.add_argument("--water", type=int, default=60)
    parser.add_argument("--food", type=int, default=60)
    parser.add_argument("--cash", type=int, default=10_000)
    parser.add_argument("--penalty", type=int, default=1_000_000)
    parser.add_argument("--p-storm", type=float, default=0.1)
    parser.add_argument("--quality-regret", type=float, default=10.0)
    parser.add_argument("--wall-hours", type=float, default=24.0)
    parser.add_argument("--memory-gib", type=float, default=256.0)
    parser.add_argument("--equilibrium", choices=("pure", "pure-mixed"), default="pure-mixed")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--initial-candidates", type=int, default=24)
    parser.add_argument("--village-candidates", type=int, default=12)
    parser.add_argument("--max-initial-candidates", type=int, default=128)
    parser.add_argument("--max-village-candidates", type=int, default=64)
    parser.add_argument("--additions-per-player", type=int, default=8)
    parser.add_argument("--max-restricted-rounds", type=int, default=32)
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
    parser.add_argument("--output", type=Path)
    return parser


def _write_output(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    target = path
    if path.suffix.lower() != ".json":
        path.mkdir(parents=True, exist_ok=True)
        target = path / "result.json"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0.0 <= args.p_storm < 1.0:
        parser.error("--p-storm must be in [0, 1)")
    remaining = 1.0 - args.p_storm
    cfg = tiny_level6() if args.mode == "smoke" else level6()
    if args.mode != "smoke":
        cfg = replace(
            cfg,
            failure_penalty=args.penalty,
            p_weather={
                "sunny": remaining * 5.0 / 9.0,
                "hot": remaining * 4.0 / 9.0,
                "sandstorm": args.p_storm,
            },
        )
    limits = SolverLimits(
        max_actions_per_player=args.max_actions,
        max_joint_profiles=args.max_profiles,
        max_cached_states=args.max_states,
        profile_chunk_size=args.chunk_size,
        max_stage_profiles_evaluated=args.max_stage_evaluations,
    )
    options = AdaptiveOptions(
        initial_candidates=args.initial_candidates,
        village_candidates_per_skeleton=args.village_candidates,
        max_initial_candidates=args.max_initial_candidates,
        max_village_candidates_per_skeleton=args.max_village_candidates,
        additions_per_player=args.additions_per_player,
        max_restricted_rounds=args.max_restricted_rounds,
        equilibrium=args.equilibrium,
        seed=args.seed,
    )
    memory_soft_gib = min(args.memory_gib, 240.0)
    budget = BudgetManager(
        wall_seconds=args.wall_hours * 3600.0,
        memory_bytes=int(memory_soft_gib * 2**30),
        max_states=args.max_states,
        closeout_seconds=min(3600.0, args.wall_hours * 1800.0),
    )
    started = perf_counter()

    if args.mode == "level6-initial":
        report = solve_q3_2(
            cfg,
            args.backend,
            limits,
            args.quality_regret,
            adaptive_options=options,
            budget_manager=budget,
            checkpoint=args.checkpoint,
            resume=args.resume,
            workers=args.workers,
        )
        payload = asdict(report)
        payload.update(
            {
                "elapsed_seconds": perf_counter() - started,
                "config": {
                    "name": cfg.name,
                    "failure_penalty": cfg.failure_penalty,
                    "p_weather": dict(cfg.p_weather),
                },
                "git_commit": _git_commit(),
                "seed": args.seed,
                "resume_command": (
                    f"python -m q3.solve_q3_2 --backend {args.backend} "
                    f"--mode level6-initial --checkpoint {args.checkpoint} --resume"
                    if args.checkpoint
                    else None
                ),
            }
        )
        _write_output(args.output, payload)
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if report.status in {"EXACT_SELECTED", "CERTIFIED_PURE"} else 2

    if args.mode == "smoke":
        day = 1
        state = _smoke_state(cfg)
    else:
        day = args.day
        player = PlayerState(
            Status.ACTIVE,
            position=args.position,
            water=args.water,
            food=args.food,
            cash_scaled=args.cash * cfg.money_scale,
        )
        state = tuple(player for _ in range(cfg.n_players))

    solver_class = ExactQ3Solver if args.backend == "exact" else AdaptiveQ3Solver
    kwargs = dict(
        limits=limits,
        workers=args.workers,
        checkpoint_path=args.checkpoint,
        checkpoint_every_states=args.checkpoint_every_states,
        checkpoint_every_pairs=args.checkpoint_every_pairs,
        enable_bound_pruning=not args.disable_bound_pruning,
        bound_pruning_slack=args.bound_pruning_slack,
        record_pruning_certificates=args.record_pruning_certificates,
        max_pruning_certificates=args.max_pruning_certificates,
        budget_manager=budget,
    )
    if args.backend == "adaptive":
        kwargs.update(options=options, quality_target=args.quality_regret)
    solver = solver_class(cfg, **kwargs)

    def handle_termination(signum, frame) -> None:
        solver.request_stop()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_termination)
    if args.resume:
        if not args.checkpoint:
            parser.error("--resume requires --checkpoint")
        solver.load_checkpoint()
    try:
        value = solver.solve_state(day, state)
        if isinstance(solver, AdaptiveQ3Solver):
            policy = {
                weather: [
                    (
                        {"action": entry.action.label(), "probability": 1.0}
                        if entry.action is not None
                        else {
                            "distribution": [
                                {"action": action.label(), "probability": probability}
                                for action, probability in entry.distribution
                            ]
                        }
                    )
                    for entry in solver.policy_entries_for(day, state, weather)
                ]
                for weather in cfg.weather_order
            }
        else:
            policy = {
                weather: [
                    action.label() for action in solver.policy_for(day, state, weather)
                ]
                for weather in cfg.weather_order
            }
        payload = {
            "status": "CERTIFIED_PURE" if args.backend == "adaptive" else "EXACT_SELECTED",
            "backend": args.backend,
            "value": value.value,
            "success": value.success,
            "policy": policy,
        }
    except (ResourceLimitExceeded, NoPureEquilibrium) as exc:
        payload = {"status": "SEARCH_STOPPED", "error": str(exc)}
    except (KeyboardInterrupt, SearchCancelled):
        payload = {"status": "INTERRUPTED"}
    payload.update(
        {
            "elapsed_seconds": perf_counter() - started,
            "numba_available": NUMBA_AVAILABLE,
            "gil_enabled": (
                sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
            ),
            "stats": asdict(solver.stats),
            "git_commit": _git_commit(),
            "seed": args.seed,
        }
    )
    if isinstance(solver, AdaptiveQ3Solver):
        payload["adaptive_stats"] = asdict(solver.adaptive_stats)
    if args.checkpoint:
        solver.save_checkpoint()
        payload["checkpoint"] = args.checkpoint
    if args.record_pruning_certificates:
        payload["pruning_certificates"] = [
            asdict(certificate) for certificate in solver.pruning_certificates
        ]
    solver.close()
    _write_output(args.output, payload)
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if payload["status"] in {"EXACT_SELECTED", "CERTIFIED_PURE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
