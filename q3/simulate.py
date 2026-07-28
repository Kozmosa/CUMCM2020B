"""Forward execution of a solved Q3.2 feedback policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .data import Weather
from .model import Action, JointState, all_absorbed
from .stochastic_dp import ExactQ3Solver
from .transition import apply_joint_transition_scalar


@dataclass(frozen=True)
class JointDayRecord:
    day: int
    weather: Weather
    state: JointState
    actions: tuple[Action, ...]


def simulate_policy(
    solver: ExactQ3Solver,
    initial_state: JointState,
    weather: Sequence[Weather],
    *,
    start_day: int = 0,
) -> tuple[JointDayRecord, ...]:
    state = initial_state
    records: list[JointDayRecord] = []
    for day, current_weather in enumerate(weather, start=start_day + 1):
        if day > solver.cfg.deadline or all_absorbed(state):
            break
        actions = solver.policy_for(day - 1, state, current_weather)
        next_state = apply_joint_transition_scalar(
            solver.cfg, state, actions, current_weather
        )
        if next_state is None:
            raise AssertionError("stored equilibrium policy produced an illegal transition")
        records.append(JointDayRecord(day, current_weather, next_state, actions))
        state = next_state
    return tuple(records)
