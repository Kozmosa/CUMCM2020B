"""Exact multi-player kernels for CUMCM 2020B question 3.

The package provides the lossless rule engine, dense and chunked pure-strategy
stage-game backends, sparse stochastic recursion, parallel successor solving,
and resumable checkpoints.  Full level-6 computation remains protected by
explicit resource limits because its exact day-0 profile count is enormous.
"""

from .data import Q3Config, level5, level6, tiny_level6
from .model import Action, ActionKind, JointState, PlayerState, Status
from .stage_game import NoPureEquilibrium
from .stochastic_dp import (
    ExactQ3Solver,
    ResourceLimitExceeded,
    SearchCancelled,
    SolverLimits,
)

__all__ = [
    "Action",
    "ActionKind",
    "ExactQ3Solver",
    "JointState",
    "NoPureEquilibrium",
    "PlayerState",
    "Q3Config",
    "ResourceLimitExceeded",
    "SearchCancelled",
    "SolverLimits",
    "Status",
    "level5",
    "level6",
    "tiny_level6",
]
