"""Reproducible level-6 baseline and one-factor sensitivity sweep."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, replace
from pathlib import Path

from .adaptive import AdaptiveOptions, solve_q3_2
from .data import level6
from .model import Action, ActionKind
from .runtime import BudgetManager
from .reports import json_safe
from .stochastic_dp import SolverLimits


def _initial_action(label: str) -> Action:
    match = re.fullmatch(r"initial_buy(?:\+W(\d+)/F(\d+))?", label)
    if match is None:
        raise ValueError(f"cannot parse initial action label {label!r}")
    return Action(
        ActionKind.INITIAL_BUY,
        buy_water=int(match.group(1) or 0),
        buy_food=int(match.group(2) or 0),
    )


def _warm_actions(report) -> tuple[Action, ...]:
    entries = report.policy.get("initial", ())
    actions = []
    for entry in entries:
        label = entry.get("action") if isinstance(entry, dict) else None
        if label is not None:
            actions.append(_initial_action(str(label)))
    return tuple(actions)


def run_sensitivity(
    output: Path,
    *,
    limits: SolverLimits,
    baseline_hours: float = 24.0,
    sensitivity_hours: float = 8.0,
    memory_gib: float = 256.0,
    quality_target: float = 10.0,
    seed: int = 20260728,
) -> list[dict[str, object]]:
    points = [
        ("baseline", 1_000_000, 0.10),
        ("M-1e4", 10_000, 0.10),
        ("M-1e9", 1_000_000_000, 0.10),
        ("storm-0.05", 1_000_000, 0.05),
        ("storm-0.15", 1_000_000, 0.15),
    ]
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    warm: tuple[Action, ...] = ()
    for index, (name, penalty, p_storm) in enumerate(points):
        remaining = 1.0 - p_storm
        cfg = replace(
            level6(),
            failure_penalty=penalty,
            p_weather={
                "sunny": remaining * 5.0 / 9.0,
                "hot": remaining * 4.0 / 9.0,
                "sandstorm": p_storm,
            },
        )
        hours = baseline_hours if index == 0 else sensitivity_hours
        checkpoint = output / f"{name}.chk"
        report = solve_q3_2(
            cfg,
            "adaptive",
            limits,
            quality_target,
            adaptive_options=AdaptiveOptions(
                seed=seed,
                warm_initial_actions=warm,
            ),
            budget_manager=BudgetManager(
                wall_seconds=hours * 3600.0,
                memory_bytes=int(min(memory_gib, 240.0) * 2**30),
                max_states=limits.max_cached_states,
                closeout_seconds=min(3600.0, hours * 1800.0),
            ),
            checkpoint=str(checkpoint),
        )
        if index == 0:
            warm = _warm_actions(report)
        payload = asdict(report)
        payload.update(
            {"name": name, "failure_penalty": penalty, "p_storm": p_storm}
        )
        (output / f"{name}.json").write_text(
            json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        rows.append(payload)

    markdown = [
        "# Q3.2 敏感性结果",
        "",
        "| 点 | M | 沙暴概率 | 状态 | 支付下界 | 成功率 | regret 上界 |",
        "|---|---:|---:|---|---|---|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['name']} | {row['failure_penalty']} | {row['p_storm']:.2f} "
            f"| {row['status']} | {row['value_lower']} | {row['success']} "
            f"| {row['max_regret_upper']} |"
        )
    (output / "sensitivity.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Q3.2 baseline/sensitivity sweep")
    parser.add_argument("--output", type=Path, default=Path("q3/output/sensitivity"))
    parser.add_argument("--baseline-hours", type=float, default=24.0)
    parser.add_argument("--sensitivity-hours", type=float, default=8.0)
    parser.add_argument("--memory-gib", type=float, default=256.0)
    parser.add_argument("--quality-regret", type=float, default=10.0)
    parser.add_argument("--max-actions", type=int, default=1_000_000)
    parser.add_argument("--max-states", type=int, default=30_000_000)
    parser.add_argument("--max-stage-evaluations", type=int, default=50_000_000)
    args = parser.parse_args(argv)
    run_sensitivity(
        args.output,
        limits=SolverLimits(
            max_actions_per_player=args.max_actions,
            max_cached_states=args.max_states,
            max_stage_profiles_evaluated=args.max_stage_evaluations,
        ),
        baseline_hours=args.baseline_hours,
        sensitivity_hours=args.sensitivity_hours,
        memory_gib=args.memory_gib,
        quality_target=args.quality_regret,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
