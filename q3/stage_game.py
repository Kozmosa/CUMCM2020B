"""Exact pure-strategy Nash detection and deterministic equilibrium selection."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .model import Action


class NoPureEquilibrium(RuntimeError):
    pass


@dataclass(frozen=True)
class PureEquilibrium:
    index: tuple[int, ...]
    actions: tuple[Action, ...]
    value: tuple[float, ...]
    success: tuple[float, ...]


@dataclass(frozen=True)
class ProfileBatchEvaluation:
    valid: np.ndarray
    payoff: np.ndarray
    success: np.ndarray


@dataclass(frozen=True)
class ChunkedSearchProgress:
    pivot_player: int
    opponent_players: tuple[int, int]
    opponent_counts: tuple[int, int]
    next_opponent_pair: int
    equilibria: tuple[PureEquilibrium, ...] = ()


def pure_nash_indices(
    payoff: np.ndarray, valid: np.ndarray, *, atol: float = 1e-10
) -> np.ndarray:
    """Return all pure Nash indices under a coupled joint-feasibility mask."""
    if payoff.ndim < 2:
        raise ValueError("payoff requires action axes and one player axis")
    action_shape = payoff.shape[:-1]
    n_players = payoff.shape[-1]
    if len(action_shape) != n_players:
        raise ValueError("one action axis is required per player")
    if valid.shape != action_shape:
        raise ValueError("valid mask shape does not match payoff action axes")

    nash = valid.copy()
    for player in range(n_players):
        player_payoff = np.where(valid, payoff[..., player], -np.inf)
        best = np.max(player_payoff, axis=player, keepdims=True)
        best_response = valid & (player_payoff >= best - atol)
        nash &= best_response
    return np.argwhere(nash)


def verify_pure_equilibrium(
    payoff: np.ndarray,
    valid: np.ndarray,
    index: tuple[int, ...],
    *,
    atol: float = 1e-10,
) -> tuple[bool, str]:
    if not valid[index]:
        return False, "candidate joint action is infeasible"
    n_players = payoff.shape[-1]
    for player in range(n_players):
        slicer: list[int | slice] = list(index)
        slicer[player] = slice(None)
        feasible = valid[tuple(slicer)]
        alternatives = payoff[tuple(slicer) + (player,)]
        best = np.max(np.where(feasible, alternatives, -np.inf))
        current = payoff[index + (player,)]
        if current < best - atol:
            return False, f"player {player} can improve {current} -> {best}"
    return True, "OK"


def _joint_action_code(
    actions: Sequence[Action],
) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(action.code for action in actions)


def select_pure_equilibrium_candidates(
    candidates: Sequence[PureEquilibrium],
) -> PureEquilibrium:
    if not candidates:
        raise NoPureEquilibrium("complete pure-strategy scan found no Nash equilibrium")
    ordered = sorted(
        candidates, key=lambda equilibrium: _joint_action_code(equilibrium.actions)
    )
    return max(
        ordered,
        key=lambda equilibrium: (
            sum(equilibrium.success),
            min(equilibrium.value),
            sum(equilibrium.value),
        ),
    )


def select_pure_equilibrium(
    indices: np.ndarray,
    payoff: np.ndarray,
    success: np.ndarray,
    action_sets: Sequence[Sequence[Action]],
) -> PureEquilibrium:
    if indices.size == 0:
        raise NoPureEquilibrium("complete pure-strategy scan found no Nash equilibrium")

    candidates: list[PureEquilibrium] = []
    for row in indices:
        index = tuple(int(x) for x in row)
        actions = tuple(action_sets[i][index[i]] for i in range(len(index)))
        candidates.append(
            PureEquilibrium(
                index=index,
                actions=actions,
                value=tuple(float(x) for x in payoff[index]),
                success=tuple(float(x) for x in success[index]),
            )
        )

    # Maximise success count, minimum payoff, total payoff; then minimise code.
    return select_pure_equilibrium_candidates(candidates)


def _decode_pair(flat_index: int, counts: tuple[int, int]) -> tuple[int, int]:
    return divmod(flat_index, counts[1])


def _validate_evaluation(
    evaluation: ProfileBatchEvaluation, batch_size: int, n_players: int
) -> None:
    if evaluation.valid.shape != (batch_size,):
        raise ValueError("profile evaluator returned an invalid feasibility shape")
    expected = (batch_size, n_players)
    if evaluation.payoff.shape != expected or evaluation.success.shape != expected:
        raise ValueError("profile evaluator returned an invalid value shape")


def chunked_pure_nash_search(
    action_sets: Sequence[Sequence[Action]],
    evaluator: Callable[[np.ndarray], ProfileBatchEvaluation],
    *,
    chunk_size: int,
    atol: float = 1e-10,
    workers: int = 1,
    progress: ChunkedSearchProgress | None = None,
    progress_callback: Callable[[ChunkedSearchProgress], None] | None = None,
) -> tuple[PureEquilibrium, ...]:
    """Find every pure Nash equilibrium with bounded profile memory.

    One player's complete best-response set is scanned for each fixed opponent
    pair.  Only those candidates are subjected to the other two unilateral
    deviation scans.  Every feasible deviation is still evaluated exactly.
    """
    if len(action_sets) != 3:
        raise ValueError("the chunked backend currently requires exactly three players")
    if chunk_size <= 0 or workers <= 0:
        raise ValueError("chunk_size and workers must be positive")
    counts = tuple(len(actions) for actions in action_sets)
    if any(count == 0 for count in counts):
        return ()

    pivot = max(range(3), key=lambda player: (counts[player], -player))
    opponents = tuple(player for player in range(3) if player != pivot)
    opponent_counts = (counts[opponents[0]], counts[opponents[1]])
    if progress is None:
        next_pair = 0
        equilibria: list[PureEquilibrium] = []
    else:
        expected = (pivot, opponents, opponent_counts)
        actual = (
            progress.pivot_player,
            progress.opponent_players,
            progress.opponent_counts,
        )
        if actual != expected:
            raise ValueError(
                "checkpointed stage-search shape does not match action sets"
            )
        next_pair = progress.next_opponent_pair
        equilibria = list(progress.equilibria)

    total_pairs = opponent_counts[0] * opponent_counts[1]
    if not 0 <= next_pair <= total_pairs:
        raise ValueError("checkpointed opponent-pair offset is outside the search")

    def evaluate_scan(
        base_index: tuple[int, int, int], player: int
    ) -> tuple[float, list[tuple[int, tuple[float, ...], tuple[float, ...]]]]:
        best_value = -np.inf
        best_rows: list[tuple[int, tuple[float, ...], tuple[float, ...]]] = []
        for start in range(0, counts[player], chunk_size):
            stop = min(start + chunk_size, counts[player])
            indices = np.tile(np.asarray(base_index, dtype=np.int64), (stop - start, 1))
            indices[:, player] = np.arange(start, stop, dtype=np.int64)
            evaluation = evaluator(indices)
            _validate_evaluation(evaluation, len(indices), 3)
            feasible_rows = np.flatnonzero(evaluation.valid)
            for row in feasible_rows:
                value = float(evaluation.payoff[int(row), player])
                action_index = start + int(row)
                candidate = (
                    action_index,
                    tuple(float(x) for x in evaluation.payoff[int(row)]),
                    tuple(float(x) for x in evaluation.success[int(row)]),
                )
                if value > best_value:
                    best_value = value
                    best_rows.append(candidate)
                elif value >= best_value - atol:
                    best_rows.append(candidate)
            if best_rows:
                best_rows = [
                    row for row in best_rows if row[1][player] >= best_value - atol
                ]
        return best_value, best_rows

    def search_pair(flat_pair: int) -> tuple[PureEquilibrium, ...]:
        pair = _decode_pair(flat_pair, opponent_counts)
        base = [0, 0, 0]
        base[opponents[0]] = pair[0]
        base[opponents[1]] = pair[1]
        _, pivot_candidates = evaluate_scan(tuple(base), pivot)
        found: list[PureEquilibrium] = []
        for pivot_index, payoff, success in pivot_candidates:
            candidate_index = list(base)
            candidate_index[pivot] = pivot_index
            is_equilibrium = True
            for player in opponents:
                best_value, _ = evaluate_scan(tuple(candidate_index), player)
                if payoff[player] < best_value - atol:
                    is_equilibrium = False
                    break
            if not is_equilibrium:
                continue
            index = tuple(candidate_index)
            actions = tuple(action_sets[player][index[player]] for player in range(3))
            found.append(PureEquilibrium(index, actions, payoff, success))
        return tuple(found)

    pair_batch_size = max(1, workers * 4)
    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        while next_pair < total_pairs:
            stop = min(next_pair + pair_batch_size, total_pairs)
            pair_indices = range(next_pair, stop)
            if executor is None:
                batch_results = map(search_pair, pair_indices)
            else:
                batch_results = executor.map(search_pair, pair_indices)
            for result in batch_results:
                equilibria.extend(result)
            next_pair = stop
            if progress_callback is not None:
                progress_callback(
                    ChunkedSearchProgress(
                        pivot,
                        opponents,
                        opponent_counts,
                        next_pair,
                        tuple(equilibria),
                    )
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return tuple(equilibria)
