"""Independent correctness checks for Q3 transitions and equilibria."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .data import Q3Config, Weather
from .model import Action, JointState
from .stage_game import verify_pure_equilibrium
from .transition import apply_joint_transition_batch, apply_joint_transition_scalar


def compare_scalar_and_vectorized(
    cfg: Q3Config,
    state: JointState,
    profiles: Sequence[Sequence[Action]],
    weather: Weather,
) -> tuple[bool, str]:
    batch = apply_joint_transition_batch(cfg, state, profiles, weather)
    for index, profile in enumerate(profiles):
        scalar = apply_joint_transition_scalar(cfg, state, profile, weather)
        vectorized = batch.successors[index]
        if (scalar is not None) != bool(batch.valid[index]):
            return False, f"profile {index}: validity mismatch"
        if scalar != vectorized:
            return False, f"profile {index}: successor mismatch {scalar} != {vectorized}"
    return True, "OK"


def verify_selected_stage_equilibrium(
    payoff: np.ndarray,
    valid: np.ndarray,
    index: tuple[int, ...],
    *,
    atol: float = 1e-10,
) -> tuple[bool, str]:
    return verify_pure_equilibrium(payoff, valid, index, atol=atol)
