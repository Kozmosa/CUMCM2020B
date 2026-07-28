"""Exact multi-player kernels for CUMCM 2020B question 3.

The package provides the lossless rule engine, dense and chunked pure-strategy
stage-game backends, sparse stochastic recursion, parallel successor solving,
and resumable checkpoints.  Full level-6 computation remains protected by
explicit resource limits because its exact day-0 profile count is enormous.
"""

from .data import Q3Config, level5, level6, tiny_level6
from .model import Action, ActionKind, JointState, PlayerState, Status
from .open_loop import (
    BestResponseResult,
    EquilibriumOptions,
    JointReplayResult,
    OpenLoopStrategy,
    Q31Limits,
    Q31SolveResult,
    solve_q3_1,
)
from .reports import PolicyEntry, Q32SolveResult, SolveReport
from .stage_game import NoPureEquilibrium
from .adaptive import AdaptiveOptions, AdaptiveQ3Solver, solve_q3_2
from .stochastic_dp import (
    ExactQ3Solver,
    ResourceLimitExceeded,
    SearchCancelled,
    SolverLimits,
)

__all__ = [
    "Action",
    "ActionKind",
    "AdaptiveOptions",
    "AdaptiveQ3Solver",
    "BestResponseResult",
    "EquilibriumOptions",
    "ExactQ3Solver",
    "JointReplayResult",
    "JointState",
    "NoPureEquilibrium",
    "OpenLoopStrategy",
    "PolicyEntry",
    "PlayerState",
    "Q31Limits",
    "Q31SolveResult",
    "Q32SolveResult",
    "Q3Config",
    "ResourceLimitExceeded",
    "SearchCancelled",
    "SolveReport",
    "SolverLimits",
    "Status",
    "level5",
    "level6",
    "tiny_level6",
    "solve_q3_1",
    "solve_q3_2",
]
