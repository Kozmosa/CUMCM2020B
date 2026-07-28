"""Command-line entry point for the known-weather Q3.1 game."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from .data import level5, tiny_level6
from .open_loop import EquilibriumOptions, Q31Limits, solve_q3_1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve CUMCM 2020B Q3.1")
    parser.add_argument("--tiny", action="store_true", help="run a tiny known-weather smoke game")
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--max-frontier-states", type=int, default=2_000_000)
    parser.add_argument("--output", type=Path, default=Path("q3/output/q3_1"))
    return parser


def _payload(result) -> dict[str, object]:
    return {
        "status": result.status,
        "regret": result.regret,
        "selection_complete": result.selection_complete,
        "iterations": dict(result.iterations),
        "payoff": result.replay.payoff,
        "success": result.replay.success,
        "strategies": [
            {
                "initial_purchase": strategy.initial_purchase.label(),
                "actions": [action.label() for action in strategy.actions],
            }
            for strategy in result.strategies
        ],
        "mixed_probabilities": result.mixed_probabilities,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = tiny_level6() if args.tiny else level5()
    if cfg.weather_sequence is None:
        cfg = replace(
            cfg, weather_sequence=tuple("sunny" for _ in range(cfg.deadline))
        )
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        result = solve_q3_1(
            cfg,
            Q31Limits(max_frontier_states=args.max_frontier_states),
            EquilibriumOptions(
                max_gauss_seidel_rounds=args.max_rounds,
                max_double_oracle_rounds=args.max_rounds,
            ),
        )
    except (RuntimeError, MemoryError) as exc:
        stopped = {
            "status": "SEARCH_STOPPED",
            "error": str(exc),
            "max_frontier_states": args.max_frontier_states,
            "resume_command": (
                "python -m q3.solve_q3_1 "
                f"--max-rounds {args.max_rounds} "
                f"--max-frontier-states {args.max_frontier_states * 2} "
                f"--output {args.output}"
            ),
        }
        (args.output / "result.json").write_text(
            json.dumps(stopped, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(stopped, ensure_ascii=False, indent=2))
        return 2
    payload = _payload(result)
    (args.output / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output / "daily.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["day", "weather", "player", "action", "status", "position", "water", "food", "cash"])
        for record in result.replay.days:
            for player, (action, state) in enumerate(
                zip(record.actions, record.state_after, strict=True), start=1
            ):
                writer.writerow(
                    [record.day, record.weather, player, action.label(), state.status.name, state.position, state.water, state.food, state.cash_scaled / cfg.money_scale]
                )
    lines = [
        "# Q3.1 求解结果",
        "",
        f"- 状态：`{result.status}`",
        f"- 支付：`{result.replay.payoff}`",
        f"- 最大 regret：`{max(result.regret):.6g}` 元",
        f"- 完整均衡选择：`{str(result.selection_complete).lower()}`",
    ]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.status == "CERTIFIED_PURE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
