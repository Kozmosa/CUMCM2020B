"""Lossless batch interaction counts with an optional Numba JIT backend."""

from __future__ import annotations

import numpy as np

from .model import ActionKind

MOVE_CODE = int(ActionKind.MOVE)
MINE_CODE = int(ActionKind.MINE)

try:  # Numba is optional because the current Python 3.13 environment lacks it.
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on the current workstation
    njit = None
    NUMBA_AVAILABLE = False


def _count_interactions_loop(
    kind: np.ndarray,
    source: np.ndarray,
    destination: np.ndarray,
    is_buyer: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Small fixed-player loop suitable for Numba compilation."""
    batch, n_players = kind.shape
    edge = np.zeros((batch, n_players), dtype=np.int16)
    mine = np.zeros((batch, n_players), dtype=np.int16)
    village = np.zeros((batch, n_players), dtype=np.int16)
    for row in range(batch):
        for i in range(n_players):
            if not active[i]:
                continue
            for j in range(n_players):
                if not active[j]:
                    continue
                if (
                    kind[row, i] == MOVE_CODE
                    and kind[row, j] == MOVE_CODE
                    and source[i] == source[j]
                    and destination[row, i] == destination[row, j]
                ):
                    edge[row, i] += 1
                if (
                    kind[row, i] == MINE_CODE
                    and kind[row, j] == MINE_CODE
                    and source[i] == source[j]
                ):
                    mine[row, i] += 1
                if is_buyer[row, i] and is_buyer[row, j] and source[i] == source[j]:
                    village[row, i] += 1
    return edge, mine, village


if NUMBA_AVAILABLE:
    _count_interactions_numba = njit(cache=True)(_count_interactions_loop)
else:
    _count_interactions_numba = None


def count_interactions_batch(
    kind: np.ndarray,
    source: np.ndarray,
    destination: np.ndarray,
    is_buyer: np.ndarray,
    active: np.ndarray,
    *,
    prefer_numba: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count same directed edge, mine, and village buyers exactly."""
    if prefer_numba and _count_interactions_numba is not None:
        return _count_interactions_numba(kind, source, destination, is_buyer, active)

    is_move = kind == MOVE_CODE
    is_mine = kind == MINE_CODE
    active_pair = active[None, :, None] & active[None, None, :]
    same_source = source[None, :, None] == source[None, None, :]
    same_destination = destination[:, :, None] == destination[:, None, :]
    edge = (
        is_move[:, :, None]
        & is_move[:, None, :]
        & active_pair
        & same_source
        & same_destination
    ).sum(axis=2, dtype=np.int16)
    mine = (
        is_mine[:, :, None]
        & is_mine[:, None, :]
        & active_pair
        & same_source
    ).sum(axis=2, dtype=np.int16)
    village = (
        is_buyer[:, :, None]
        & is_buyer[:, None, :]
        & active_pair
        & same_source
    ).sum(axis=2, dtype=np.int16)
    return edge, mine, village
