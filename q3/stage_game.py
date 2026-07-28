"""Exact pure-strategy Nash detection and deterministic equilibrium selection."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import minimize

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
class MixedEquilibrium:
    probabilities: tuple[tuple[float, ...], ...]
    value: tuple[float, ...]
    regrets: tuple[float, ...]
    nash_conv: float
    success: tuple[float, ...] = ()


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


def _profile_probability(probabilities: Sequence[np.ndarray]) -> np.ndarray:
    weight = np.asarray(1.0)
    for player, probability in enumerate(probabilities):
        shape = [1] * len(probabilities)
        shape[player] = len(probability)
        weight = weight * probability.reshape(shape)
    return np.asarray(weight, dtype=np.float64)


def independent_mixture_values(
    payoff: np.ndarray,
    probabilities: Sequence[Sequence[float]],
    *,
    valid: np.ndarray | None = None,
    success: np.ndarray | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Return expected payoff, success, and unilateral regrets.

    Coupled-infeasible profiles receive a deterministic very-low utility.  In
    the intended adaptive use, support expansion keeps their probability at
    zero; the convention mainly makes the optimizer robust on arbitrary test
    games.
    """
    n_players = payoff.shape[-1]
    action_shape = payoff.shape[:-1]
    if len(action_shape) != n_players:
        raise ValueError("one action axis is required per player")
    arrays = tuple(np.asarray(p, dtype=np.float64) for p in probabilities)
    if tuple(len(p) for p in arrays) != action_shape:
        raise ValueError("mixed strategy shape does not match payoff tensor")
    if valid is None:
        valid = np.ones(action_shape, dtype=bool)
    if valid.shape != action_shape:
        raise ValueError("valid mask shape does not match payoff tensor")

    finite = payoff[np.isfinite(payoff)]
    floor = (float(np.min(finite)) - 1e6) if finite.size else -1e12
    effective = np.where(valid[..., None], payoff, floor)
    profile_weight = _profile_probability(arrays)
    values = tuple(
        float(np.sum(profile_weight * effective[..., player]))
        for player in range(n_players)
    )
    success_values: tuple[float, ...] = ()
    if success is not None:
        success_values = tuple(
            float(np.sum(profile_weight * success[..., player]))
            for player in range(n_players)
        )

    regrets: list[float] = []
    for player in range(n_players):
        action_values = np.empty(action_shape[player], dtype=np.float64)
        opponents = tuple(i for i in range(n_players) if i != player)
        opponent_shape = tuple(action_shape[i] for i in opponents)
        for action in range(action_shape[player]):
            total = 0.0
            for opponent_index in np.ndindex(opponent_shape):
                index = [0] * n_players
                index[player] = action
                probability = 1.0
                for opponent, opponent_action in zip(
                    opponents, opponent_index, strict=True
                ):
                    index[opponent] = opponent_action
                    probability *= arrays[opponent][opponent_action]
                total += probability * effective[tuple(index) + (player,)]
            action_values[action] = total
        regrets.append(max(0.0, float(np.max(action_values)) - values[player]))
    return values, success_values, tuple(regrets)


def minimize_nashconv(
    payoff: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    success: np.ndarray | None = None,
    seed: int = 20260728,
    starts: int = 8,
    max_iterations: int = 2_000,
) -> MixedEquilibrium:
    """Deterministically minimize NashConv over independent finite mixtures."""
    n_players = payoff.shape[-1]
    action_shape = payoff.shape[:-1]
    if len(action_shape) != n_players or any(count <= 0 for count in action_shape):
        raise ValueError("invalid normal-form payoff shape")
    offsets = np.cumsum((0,) + action_shape)

    def unpack(vector: np.ndarray) -> tuple[np.ndarray, ...]:
        return tuple(
            vector[offsets[player] : offsets[player + 1]]
            for player in range(n_players)
        )

    def objective(vector: np.ndarray) -> float:
        probabilities = unpack(vector)
        _, _, regrets = independent_mixture_values(
            payoff, probabilities, valid=valid, success=success
        )
        return float(sum(regrets))

    constraints = tuple(
        {
            "type": "eq",
            "fun": lambda vector, player=player: float(
                np.sum(vector[offsets[player] : offsets[player + 1]]) - 1.0
            ),
        }
        for player in range(n_players)
    )
    bounds = tuple((0.0, 1.0) for _ in range(int(offsets[-1])))
    rng = np.random.default_rng(seed)
    initial: list[np.ndarray] = [
        np.concatenate(
            [np.full(count, 1.0 / count, dtype=np.float64) for count in action_shape]
        )
    ]
    # Pure starts expose easy equilibria; seeded Dirichlet starts cover games
    # such as matching pennies where the uniform point is already optimal.
    pure_profiles = list(product(*(range(count) for count in action_shape)))
    for profile in pure_profiles[: max(0, starts // 2)]:
        vector = np.zeros(int(offsets[-1]), dtype=np.float64)
        for player, action in enumerate(profile):
            vector[offsets[player] + action] = 1.0
        initial.append(vector)
    while len(initial) < starts:
        initial.append(
            np.concatenate([rng.dirichlet(np.ones(count)) for count in action_shape])
        )

    best_vector = initial[0]
    best_objective = objective(best_vector)
    for vector in initial[:starts]:
        result = minimize(
            objective,
            vector,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": max_iterations, "ftol": 1e-12, "disp": False},
        )
        candidate = np.asarray(result.x if result.success else vector, dtype=np.float64)
        # Remove numerical simplex drift before reporting or comparing.
        for player in range(n_players):
            block = candidate[offsets[player] : offsets[player + 1]]
            np.maximum(block, 0.0, out=block)
            total = float(np.sum(block))
            if total <= 0.0:
                block[:] = 1.0 / len(block)
            else:
                block /= total
        score = objective(candidate)
        if score < best_objective - 1e-12:
            best_objective = score
            best_vector = candidate.copy()

    probabilities = unpack(best_vector)
    values, successes, regrets = independent_mixture_values(
        payoff, probabilities, valid=valid, success=success
    )
    return MixedEquilibrium(
        probabilities=tuple(tuple(float(x) for x in p) for p in probabilities),
        value=values,
        success=successes,
        regrets=regrets,
        nash_conv=float(sum(regrets)),
    )


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
    bound_slack: float = 0.0,
    workers: int = 1,
    upper_bounder: Callable[[np.ndarray, int], np.ndarray] | None = None,
    pruning_callback: Callable[[int, np.ndarray, np.ndarray, float], None]
    | None = None,
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
    if chunk_size <= 0 or workers <= 0 or bound_slack < 0:
        raise ValueError(
            "chunk_size/workers must be positive and bound_slack non-negative"
        )
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
            if upper_bounder is not None and np.isfinite(best_value):
                upper_bounds = np.asarray(
                    upper_bounder(indices, player), dtype=np.float64
                )
                if upper_bounds.shape != (len(indices),):
                    raise ValueError("upper bounder returned an invalid shape")
                keep = upper_bounds >= best_value - atol - bound_slack
                if pruning_callback is not None and not keep.all():
                    pruning_callback(
                        player,
                        indices[~keep].copy(),
                        upper_bounds[~keep].copy(),
                        best_value,
                    )
                indices = indices[keep]
                if len(indices) == 0:
                    continue
            evaluation = evaluator(indices)
            _validate_evaluation(evaluation, len(indices), 3)
            feasible_rows = np.flatnonzero(evaluation.valid)
            for row in feasible_rows:
                value = float(evaluation.payoff[int(row), player])
                action_index = int(indices[int(row), player])
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
