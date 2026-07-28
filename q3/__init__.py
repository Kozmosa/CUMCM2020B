"""Exact multi-player kernels for CUMCM 2020B question 3.

The package currently provides the lossless rule engine, pure-strategy stage
game solver, and a sparse short-horizon stochastic solver.  Full level-6
initial-purchase enumeration is deliberately protected by explicit resource
limits until the chunked 128-core backend is completed.
"""

from .data import Q3Config, level5, level6, tiny_level6
from .model import Action, ActionKind, JointState, PlayerState, Status
from .stage_game import NoPureEquilibrium
from .stochastic_dp import ExactQ3Solver, ResourceLimitExceeded, SolverLimits

__all__ = [
    "Action",
    "ActionKind",
    "ExactQ3Solver",
    "JointState",
    "NoPureEquilibrium",
    "PlayerState",
    "Q3Config",
    "ResourceLimitExceeded",
    "SolverLimits",
    "Status",
    "level5",
    "level6",
    "tiny_level6",
]
