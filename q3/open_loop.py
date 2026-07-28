"""Known-weather open-loop game and exact best responses for Q3.1."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import permutations, product
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from .action_enum import (
    enumerate_individual_actions,
    enumerate_initial_purchases_bounded,
)
from .data import Q3Config
from .model import (
    INACTIVE_ACTION,
    Action,
    ActionKind,
    JointState,
    PlayerState,
    Status,
    fail_player,
    failure_payoff_scaled,
    initial_joint_state,
    weight_ok,
)
from .stage_game import minimize_nashconv, pure_nash_indices
from .transition import (
    InteractionCounts,
    apply_player_action_given_counts,
    count_interactions_scalar,
)


@dataclass(frozen=True, slots=True)
class OpenLoopStrategy:
    initial_purchase: Action
    actions: tuple[Action, ...]

    def __post_init__(self) -> None:
        if self.initial_purchase.kind is not ActionKind.INITIAL_BUY:
            raise ValueError("open-loop initial action must be INITIAL_BUY")
        if any(action.kind is ActionKind.INITIAL_BUY for action in self.actions):
            raise ValueError("day actions cannot contain INITIAL_BUY")

    @property
    def code(self) -> tuple[tuple[int, int, int, int], ...]:
        return (self.initial_purchase.code,) + tuple(
            action.code for action in self.actions
        )


@dataclass(frozen=True, slots=True)
class JointDayReplay:
    day: int
    weather: str
    state_before: JointState
    actions: tuple[Action, ...]
    interactions: InteractionCounts
    forced_failures: tuple[bool, ...]
    state_after: JointState


@dataclass(frozen=True)
class JointReplayResult:
    states: tuple[JointState, ...]
    days: tuple[JointDayReplay, ...]
    payoff: tuple[float, ...]
    payoff_scaled: tuple[int, ...]
    success: tuple[bool, ...]
    legal: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class BestResponseResult:
    player: int
    strategy: OpenLoopStrategy
    payoff: float
    payoff_scaled: int
    current_payoff: float
    full_deviation_gain: float
    parent_pointers: tuple[Mapping[tuple[int, ...], tuple[tuple[int, ...] | None, Action]], ...]
    states_examined: int
    frontier_sizes: tuple[int, ...] = ()
    day_seconds: tuple[float, ...] = ()
    backend: str = "scalar"


@dataclass(frozen=True)
class Q31Limits:
    max_frontier_states: int = 2_000_000
    max_initial_actions: int = 1_000_000
    max_restricted_profiles: int = 2_000_000
    use_compact_numba: bool = True


@dataclass(frozen=True)
class EquilibriumOptions:
    max_gauss_seidel_rounds: int = 20
    max_double_oracle_rounds: int = 20
    equilibrium_atol: float = 1e-10
    mixed_fallback: bool = True
    seed: int = 20260728


@dataclass(frozen=True)
class Q31SolveResult:
    strategies: tuple[OpenLoopStrategy, ...]
    equilibrium_type: str
    regret: tuple[float, ...]
    iterations: Mapping[str, int]
    replay: JointReplayResult
    strategy_sets: tuple[tuple[OpenLoopStrategy, ...], ...]
    mixed_probabilities: tuple[tuple[float, ...], ...] = ()
    selection_complete: bool = False

    @property
    def status(self) -> str:
        return self.equilibrium_type


def _known_weather(cfg: Q3Config) -> tuple[str, ...]:
    if cfg.weather_sequence is None:
        raise ValueError("Q3.1 requires config.weather_sequence")
    return cfg.weather_sequence


def _initial_players(
    cfg: Q3Config, strategies: Sequence[OpenLoopStrategy]
) -> tuple[JointState | None, tuple[str, ...]]:
    base = initial_joint_state(cfg)
    players: list[PlayerState] = []
    errors: list[str] = []
    for player, (state, strategy) in enumerate(zip(base, strategies, strict=True)):
        action = strategy.initial_purchase
        water = state.water + action.buy_water
        food = state.food + action.buy_food
        cost = cfg.money_scale * (
            cfg.water_price * action.buy_water + cfg.food_price * action.buy_food
        )
        if cost > state.cash_scaled or not weight_ok(cfg, water, food):
            errors.append(f"player {player} has an infeasible initial purchase")
            continue
        players.append(
            PlayerState(
                Status.ACTIVE,
                position=cfg.start,
                water=water,
                food=food,
                cash_scaled=state.cash_scaled - cost,
            )
        )
    if errors:
        return None, tuple(errors)
    return tuple(players), ()


def apply_open_loop_day(
    cfg: Q3Config,
    state: JointState,
    scheduled_actions: Sequence[Action],
    weather: str,
) -> tuple[JointState, InteractionCounts, tuple[Action, ...], tuple[bool, ...]]:
    """Synchronously execute a precommitted day, forcing failed plans to die.

    Interaction counts are formed from all active players' scheduled actions
    before resource feasibility is checked.  Thus a deviator can increase an
    opponent's road consumption and make that opponent fail on the same day.
    """
    effective = tuple(
        action if player.status is Status.ACTIVE else INACTIVE_ACTION
        for player, action in zip(state, scheduled_actions, strict=True)
    )
    counts = count_interactions_scalar(state, effective)
    output: list[PlayerState] = []
    forced: list[bool] = []
    for index, (player, action) in enumerate(zip(state, effective, strict=True)):
        successor = apply_player_action_given_counts(
            cfg,
            player,
            action,
            weather,
            edge_count=counts.edge[index],
            mine_count=counts.mine[index],
            village_buyer_count=counts.village_buyers[index],
        )
        if successor is None and player.status is Status.ACTIVE:
            successor = fail_player(cfg, player)
            forced.append(True)
        else:
            if successor is None:
                raise AssertionError("absorbed player rejected the inactive action")
            forced.append(False)
        output.append(successor)
    return tuple(output), counts, effective, tuple(forced)


def replay_joint_strategy(
    cfg: Q3Config, strategies: Sequence[OpenLoopStrategy]
) -> JointReplayResult:
    weather = _known_weather(cfg)
    if len(strategies) != cfg.n_players:
        raise ValueError("strategy profile has the wrong player count")
    if any(len(strategy.actions) != cfg.deadline for strategy in strategies):
        raise ValueError("every open-loop strategy must cover the full deadline")
    state, errors = _initial_players(cfg, strategies)
    if state is None:
        failure = tuple(-float(cfg.failure_penalty) for _ in strategies)
        return JointReplayResult((), (), failure, tuple(
            -cfg.failure_penalty_scaled for _ in strategies
        ), tuple(False for _ in strategies), False, errors)

    states = [state]
    records: list[JointDayReplay] = []
    for day, day_weather in enumerate(weather, start=1):
        scheduled = tuple(strategy.actions[day - 1] for strategy in strategies)
        next_state, counts, effective, forced = apply_open_loop_day(
            cfg, state, scheduled, day_weather
        )
        records.append(
            JointDayReplay(
                day,
                day_weather,
                state,
                effective,
                counts,
                forced,
                next_state,
            )
        )
        state = next_state
        states.append(state)

    payoff_scaled: list[int] = []
    success: list[bool] = []
    for player in state:
        if player.status is Status.FINISHED:
            payoff_scaled.append(player.fixed_payoff_scaled)
            success.append(True)
        elif player.status is Status.FAILED:
            payoff_scaled.append(player.fixed_payoff_scaled)
            success.append(False)
        else:
            payoff_scaled.append(failure_payoff_scaled(cfg, player.cash_scaled))
            success.append(False)
    return JointReplayResult(
        tuple(states),
        tuple(records),
        tuple(value / cfg.money_scale for value in payoff_scaled),
        tuple(payoff_scaled),
        tuple(success),
        True,
    )


def _best_response_key(state: JointState, player: int) -> tuple[int, ...]:
    key: list[int] = []
    for index, item in enumerate(state):
        if index == player:
            key.extend(
                (
                    int(item.status),
                    item.position,
                    item.water,
                    item.food,
                    item.fixed_payoff_scaled,
                )
            )
        else:
            key.extend(
                (
                    int(item.status),
                    item.position,
                    item.water,
                    item.food,
                    item.cash_scaled,
                    item.fixed_payoff_scaled,
                )
            )
    return tuple(key)


def _terminal_player_payoff(cfg: Q3Config, player: PlayerState) -> int:
    if player.status is Status.ACTIVE:
        return failure_payoff_scaled(cfg, player.cash_scaled)
    return player.fixed_payoff_scaled


def _best_response_open_loop_scalar(
    cfg: Q3Config,
    player: int,
    profile: Sequence[OpenLoopStrategy],
    *,
    limits: Q31Limits | None = None,
) -> BestResponseResult:
    """Exact sparse DP over deviator state plus every opponent's full state."""
    limits = limits or Q31Limits()
    if not 0 <= player < cfg.n_players:
        raise ValueError("player index is outside the strategy profile")
    current = replay_joint_strategy(cfg, profile)
    weather = _known_weather(cfg)
    base = initial_joint_state(cfg)
    parents: list[
        dict[tuple[int, ...], tuple[tuple[int, ...] | None, Action]]
    ] = [dict() for _ in range(cfg.deadline + 1)]
    frontier: dict[tuple[int, ...], JointState] = {}
    initial_actions = enumerate_initial_purchases_bounded(
        cfg,
        base[player],
        max_actions=limits.max_initial_actions,
        prune_strictly_dominated=True,
    )
    for initial in initial_actions:
        trial = list(profile)
        trial[player] = OpenLoopStrategy(
            initial, tuple(INACTIVE_ACTION for _ in range(cfg.deadline))
        )
        state, _ = _initial_players(cfg, trial)
        if state is None:
            continue
        key = _best_response_key(state, player)
        incumbent = frontier.get(key)
        if incumbent is None or state[player].cash_scaled > incumbent[player].cash_scaled:
            frontier[key] = state
            parents[0][key] = (None, initial)
    if len(frontier) > limits.max_frontier_states:
        raise RuntimeError("Q3.1 initial best-response frontier exceeded its state limit")

    states_examined = len(frontier)
    best_terminal: tuple[int, int, tuple[int, ...]] | None = None
    terminal_day = cfg.deadline

    def terminal_is_better(payoff: int, day: int) -> bool:
        return best_terminal is None or (payoff, -day) > (
            best_terminal[0],
            -best_terminal[1],
        )

    for day, day_weather in enumerate(weather, start=1):
        next_frontier: dict[tuple[int, ...], JointState] = {}
        for previous_key, state in frontier.items():
            own_state = state[player]
            if own_state.status is not Status.ACTIVE:
                payoff = own_state.fixed_payoff_scaled
                if terminal_is_better(payoff, day - 1):
                    best_terminal = (payoff, day - 1, previous_key)
                    terminal_day = day - 1
                continue
            own_actions = enumerate_individual_actions(cfg, own_state, day_weather)
            for own_action in own_actions:
                scheduled = [
                    strategy.actions[day - 1] for strategy in profile
                ]
                scheduled[player] = own_action
                successor, _, _, _ = apply_open_loop_day(
                    cfg, state, scheduled, day_weather
                )
                own_successor = successor[player]
                if own_successor.status is not Status.ACTIVE:
                    payoff = own_successor.fixed_payoff_scaled
                    if terminal_is_better(payoff, day):
                        key = _best_response_key(successor, player)
                        parents[day][key] = (previous_key, own_action)
                        best_terminal = (payoff, day, key)
                        terminal_day = day
                    continue
                key = _best_response_key(successor, player)
                incumbent = next_frontier.get(key)
                replace_state = (
                    incumbent is None
                    or own_successor.cash_scaled > incumbent[player].cash_scaled
                )
                if replace_state:
                    next_frontier[key] = successor
                    parents[day][key] = (previous_key, own_action)
        frontier = next_frontier
        states_examined += len(frontier)
        if len(frontier) > limits.max_frontier_states:
            raise RuntimeError(
                f"Q3.1 day {day} frontier exceeded {limits.max_frontier_states:,} states"
            )

    for key, state in frontier.items():
        payoff = _terminal_player_payoff(cfg, state[player])
        if terminal_is_better(payoff, cfg.deadline):
            best_terminal = (payoff, cfg.deadline, key)
            terminal_day = cfg.deadline
    if best_terminal is None:
        raise RuntimeError("best-response search produced no terminal state")

    payoff_scaled, _, key = best_terminal
    recovered: list[Action] = []
    for day in range(terminal_day, 0, -1):
        previous_key, action = parents[day][key]
        recovered.append(action)
        if previous_key is None:
            raise AssertionError("day action unexpectedly has no parent")
        key = previous_key
    initial_parent, initial_action = parents[0][key]
    if initial_parent is not None:
        raise AssertionError("initial best-response parent is malformed")
    recovered.reverse()
    recovered.extend(INACTIVE_ACTION for _ in range(cfg.deadline - len(recovered)))
    strategy = OpenLoopStrategy(initial_action, tuple(recovered))
    payoff = payoff_scaled / cfg.money_scale
    current_payoff = current.payoff[player]
    public_parents = tuple(
        {key_: (value[0], value[1]) for key_, value in layer.items()}
        for layer in parents
    )
    return BestResponseResult(
        player,
        strategy,
        payoff,
        payoff_scaled,
        current_payoff,
        payoff - current_payoff,
        public_parents,
        states_examined,
    )


def _compact_kernel_eligible(
    cfg: Q3Config, profile: Sequence[OpenLoopStrategy], limits: Q31Limits
) -> bool:
    if not limits.use_compact_numba or cfg.n_players != 3 or cfg.villages:
        return False
    if max(cfg.nodes) >= 16:
        return False
    return not any(
        action.is_buyer
        for strategy in profile
        for action in strategy.actions
    )


def _best_response_open_loop_compact(
    cfg: Q3Config,
    player: int,
    profile: Sequence[OpenLoopStrategy],
    *,
    limits: Q31Limits,
) -> BestResponseResult:
    from .open_loop_numba import (
        CompactKernelConfig,
        NUMBA_OPEN_LOOP_AVAILABLE,
        allocate_hash_table,
        compact_hash_table,
        decode_action_code,
        expand_frontier_numba,
        pack_frontier_state,
    )

    if not NUMBA_OPEN_LOOP_AVAILABLE:
        return _best_response_open_loop_scalar(cfg, player, profile, limits=limits)
    current_replay = replay_joint_strategy(cfg, profile)
    weather = _known_weather(cfg)
    kernel = CompactKernelConfig.build(cfg)
    opponents = tuple(index for index in range(cfg.n_players) if index != player)
    initial_probe = list(profile)
    initial_probe[player] = OpenLoopStrategy(
        Action(ActionKind.INITIAL_BUY),
        tuple(INACTIVE_ACTION for _ in range(cfg.deadline)),
    )
    post_profile, errors = _initial_players(cfg, initial_probe)
    if post_profile is None:
        raise ValueError("opponent profile has an infeasible initial purchase: " + "; ".join(errors))
    opponent0 = post_profile[opponents[0]]
    opponent1 = post_profile[opponents[1]]
    base_player = initial_joint_state(cfg)[player]
    initial_actions = enumerate_initial_purchases_bounded(
        cfg,
        base_player,
        max_actions=limits.max_initial_actions,
        prune_strictly_dominated=True,
    )
    if len(initial_actions) > limits.max_frontier_states:
        raise RuntimeError(
            "Q3.1 initial compact frontier exceeded its state limit"
        )

    current_low = np.empty(len(initial_actions), dtype=np.uint64)
    current_high = np.empty(len(initial_actions), dtype=np.uint8)
    current_cash = np.empty(len(initial_actions), dtype=np.int64)
    initial_water = np.empty(len(initial_actions), dtype=np.int16)
    initial_food = np.empty(len(initial_actions), dtype=np.int16)
    for row, action in enumerate(initial_actions):
        cash = base_player.cash_scaled - cfg.money_scale * (
            cfg.water_price * action.buy_water + cfg.food_price * action.buy_food
        )
        deviator = PlayerState(
            Status.ACTIVE,
            position=cfg.start,
            water=action.buy_water,
            food=action.buy_food,
            cash_scaled=cash,
        )
        low, high = pack_frontier_state(
            kernel.resources, deviator, opponent0, opponent1
        )
        current_low[row] = low
        current_high[row] = high
        current_cash[row] = cash
        initial_water[row] = action.buy_water
        initial_food[row] = action.buy_food

    parent_layers: list[np.ndarray | None] = [None]
    action_layers: list[np.ndarray | None] = [None]
    states_examined = len(current_low)
    frontier_sizes = [len(current_low)]
    day_seconds: list[float] = []
    best_terminal: tuple[int, int, int, int, int] | None = None
    # payoff, success, day, source row, compact action code
    max_degree = max(len(cfg.adj[node]) for node in cfg.nodes)

    for day, day_weather in enumerate(weather, start=1):
        day_started = perf_counter()
        opponent_kind = np.asarray(
            [int(profile[index].actions[day - 1].kind) for index in opponents],
            dtype=np.int8,
        )
        opponent_destination = np.asarray(
            [profile[index].actions[day - 1].destination for index in opponents],
            dtype=np.int8,
        )
        raw_upper = max(1, len(current_low) * (2 + max_degree))
        day_limit = min(limits.max_frontier_states, raw_upper)
        table = allocate_hash_table(day_limit)
        (
            count,
            overflow,
            payoff,
            success,
            source,
            action_code,
        ) = expand_frontier_numba(
            current_low,
            current_high,
            current_cash,
            opponent_kind,
            opponent_destination,
            day_weather == "sandstorm",
            cfg.water_consume[day_weather],
            cfg.food_consume[day_weather],
            day == cfg.deadline,
            cfg.end,
            cfg.money_scale * cfg.mine_income,
            cfg.failure_penalty_scaled,
            cfg.money_scale,
            cfg.water_price,
            cfg.food_price,
            kernel.resources.water,
            kernel.resources.food,
            kernel.resources.id_grid,
            kernel.neighbors,
            kernel.degree,
            kernel.is_mine,
            *table,
            day_limit,
        )
        if int(payoff) > -(1 << 61):
            candidate = (
                int(payoff),
                int(success),
                day,
                int(source),
                int(action_code),
            )
            if best_terminal is None or candidate[:2] > best_terminal[:2]:
                best_terminal = candidate
        if overflow:
            raise RuntimeError(
                f"Q3.1 day {day} compact frontier exceeded "
                f"{limits.max_frontier_states:,} states"
            )
        day_seconds.append(perf_counter() - day_started)
        if day == cfg.deadline or count == 0:
            break
        (
            next_low,
            next_high,
            next_cash,
            next_parent,
            next_action,
        ) = compact_hash_table(table, int(count))
        parent_layers.append(next_parent)
        action_layers.append(next_action)
        states_examined += len(next_low)
        frontier_sizes.append(len(next_low))
        current_low, current_high, current_cash = next_low, next_high, next_cash

    if best_terminal is None:
        raise RuntimeError("compact best-response search produced no terminal state")
    payoff_scaled, _, terminal_day, current_index, terminal_code = best_terminal
    reverse_actions: list[Action] = []

    def decode(code: int) -> Action:
        kind, destination = decode_action_code(code)
        return Action(
            kind,
            destination=destination if kind is ActionKind.MOVE else 0,
        )

    reverse_actions.append(decode(terminal_code))
    for day in range(terminal_day - 1, 0, -1):
        parents = parent_layers[day]
        actions = action_layers[day]
        if parents is None or actions is None:
            raise AssertionError("compact best-response parent layer is missing")
        reverse_actions.append(decode(int(actions[current_index])))
        current_index = int(parents[current_index])
    initial_action = Action(
        ActionKind.INITIAL_BUY,
        buy_water=int(initial_water[current_index]),
        buy_food=int(initial_food[current_index]),
    )
    reverse_actions.reverse()
    reverse_actions.extend(
        INACTIVE_ACTION for _ in range(cfg.deadline - len(reverse_actions))
    )
    strategy = OpenLoopStrategy(initial_action, tuple(reverse_actions))
    verified = list(profile)
    verified[player] = strategy
    replay = replay_joint_strategy(cfg, verified)
    if replay.payoff_scaled[player] != payoff_scaled:
        raise AssertionError(
            "compact best-response payoff disagrees with independent replay: "
            f"{payoff_scaled} != {replay.payoff_scaled[player]}"
        )
    payoff_value = payoff_scaled / cfg.money_scale
    current_payoff = current_replay.payoff[player]
    return BestResponseResult(
        player=player,
        strategy=strategy,
        payoff=payoff_value,
        payoff_scaled=payoff_scaled,
        current_payoff=current_payoff,
        full_deviation_gain=payoff_value - current_payoff,
        parent_pointers=tuple({} for _ in range(cfg.deadline + 1)),
        states_examined=states_examined,
        frontier_sizes=tuple(frontier_sizes),
        day_seconds=tuple(day_seconds),
        backend="compact-numba",
    )


def best_response_open_loop(
    cfg: Q3Config,
    player: int,
    profile: Sequence[OpenLoopStrategy],
    *,
    limits: Q31Limits | None = None,
) -> BestResponseResult:
    limits = limits or Q31Limits()
    if _compact_kernel_eligible(cfg, profile, limits):
        return _best_response_open_loop_compact(
            cfg, player, profile, limits=limits
        )
    return _best_response_open_loop_scalar(cfg, player, profile, limits=limits)


def _single_player_seed(cfg: Q3Config, limits: Q31Limits) -> OpenLoopStrategy:
    if cfg.n_players == 3 and not cfg.villages:
        dummy = OpenLoopStrategy(
            Action(ActionKind.INITIAL_BUY),
            (Action(ActionKind.FAIL),)
            + tuple(INACTIVE_ACTION for _ in range(cfg.deadline - 1)),
        )
        # Failed dummy opponents never interact.  The same exact compact BR
        # oracle therefore produces a single-player seed without materializing
        # Q1's large Python parent dictionaries.
        return best_response_open_loop(
            cfg,
            0,
            tuple(dummy for _ in range(cfg.n_players)),
            limits=limits,
        ).strategy

    solo = replace(cfg, n_players=1)
    dummy = OpenLoopStrategy(
        Action(ActionKind.INITIAL_BUY),
        tuple(Action(ActionKind.FAIL) for _ in range(cfg.deadline)),
    )
    return best_response_open_loop(solo, 0, (dummy,), limits=limits).strategy


def _profile_regret(
    cfg: Q3Config,
    profile: tuple[OpenLoopStrategy, ...],
    limits: Q31Limits,
) -> tuple[tuple[float, ...], tuple[BestResponseResult, ...]]:
    replay = replay_joint_strategy(cfg, profile)
    responses = tuple(
        best_response_open_loop(cfg, player, profile, limits=limits)
        for player in range(cfg.n_players)
    )
    regret = tuple(
        max(0.0, response.payoff - replay.payoff[player])
        for player, response in enumerate(responses)
    )
    return regret, responses


def _restricted_game(
    cfg: Q3Config,
    strategy_sets: Sequence[Sequence[OpenLoopStrategy]],
    max_profiles: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[OpenLoopStrategy, ...]]]:
    shape = tuple(len(strategies) for strategies in strategy_sets)
    count = int(np.prod(shape, dtype=np.int64))
    if count > max_profiles:
        raise RuntimeError(
            f"restricted Q3.1 game has {count:,} profiles, above {max_profiles:,}"
        )
    payoff = np.empty(shape + (cfg.n_players,), dtype=np.float64)
    success = np.empty(shape + (cfg.n_players,), dtype=np.float64)
    profiles: list[tuple[OpenLoopStrategy, ...]] = []
    for index in product(*(range(size) for size in shape)):
        profile = tuple(strategy_sets[p][index[p]] for p in range(cfg.n_players))
        replay = replay_joint_strategy(cfg, profile)
        payoff[index] = replay.payoff
        success[index] = replay.success
        profiles.append(profile)
    return payoff, success, profiles


def _select_restricted_pure(
    indices: np.ndarray,
    payoff: np.ndarray,
    success: np.ndarray,
    strategy_sets: Sequence[Sequence[OpenLoopStrategy]],
) -> tuple[OpenLoopStrategy, ...]:
    candidates = []
    for row in indices:
        index = tuple(int(x) for x in row)
        profile = tuple(strategy_sets[p][index[p]] for p in range(len(index)))
        candidates.append((index, profile))
    candidates.sort(key=lambda item: tuple(strategy.code for strategy in item[1]))
    _, selected = max(
        candidates,
        key=lambda item: (
            float(np.sum(success[item[0]])),
            float(np.min(payoff[item[0]])),
            float(np.sum(payoff[item[0]])),
        ),
    )
    return selected


def solve_q3_1(
    config: Q3Config,
    limits: Q31Limits | None = None,
    equilibrium_options: EquilibriumOptions | None = None,
) -> Q31SolveResult:
    limits = limits or Q31Limits()
    options = equilibrium_options or EquilibriumOptions()
    _known_weather(config)
    seed = _single_player_seed(config, limits)
    strategy_sets: list[list[OpenLoopStrategy]] = [[seed] for _ in range(config.n_players)]
    seen_sets = [set(strategies) for strategies in strategy_sets]
    gauss_rounds = 0
    oracle_rounds = 0
    best_profile = tuple(seed for _ in range(config.n_players))
    best_regret = tuple(float("inf") for _ in range(config.n_players))

    for order in permutations(range(config.n_players)):
        profile = [seed for _ in range(config.n_players)]
        seen_profiles: set[tuple[OpenLoopStrategy, ...]] = set()
        for _ in range(options.max_gauss_seidel_rounds):
            gauss_rounds += 1
            frozen = tuple(profile)
            if frozen in seen_profiles:
                break
            seen_profiles.add(frozen)
            improved = False
            for player in order:
                current_replay = replay_joint_strategy(config, profile)
                response = best_response_open_loop(
                    config, player, profile, limits=limits
                )
                if response.payoff > current_replay.payoff[player] + options.equilibrium_atol:
                    profile[player] = response.strategy
                    improved = True
                if response.strategy not in seen_sets[player]:
                    seen_sets[player].add(response.strategy)
                    strategy_sets[player].append(response.strategy)
            candidate = tuple(profile)
            regret, _ = _profile_regret(config, candidate, limits)
            if max(regret) < max(best_regret):
                best_profile, best_regret = candidate, regret
            if max(regret) <= options.equilibrium_atol:
                return Q31SolveResult(
                    candidate,
                    "CERTIFIED_PURE",
                    regret,
                    {"gauss_seidel_rounds": gauss_rounds, "double_oracle_rounds": 0},
                    replay_joint_strategy(config, candidate),
                    tuple(tuple(items) for items in strategy_sets),
                    selection_complete=False,
                )
            if not improved:
                break

    for _ in range(options.max_double_oracle_rounds):
        oracle_rounds += 1
        payoff, success, _ = _restricted_game(
            config, strategy_sets, limits.max_restricted_profiles
        )
        valid = np.ones(payoff.shape[:-1], dtype=bool)
        indices = pure_nash_indices(payoff, valid, atol=options.equilibrium_atol)
        if len(indices):
            candidate = _select_restricted_pure(
                indices, payoff, success, strategy_sets
            )
            regret, responses = _profile_regret(config, candidate, limits)
            if max(regret) < max(best_regret):
                best_profile, best_regret = candidate, regret
            if max(regret) <= options.equilibrium_atol:
                return Q31SolveResult(
                    candidate,
                    "CERTIFIED_PURE",
                    regret,
                    {
                        "gauss_seidel_rounds": gauss_rounds,
                        "double_oracle_rounds": oracle_rounds,
                    },
                    replay_joint_strategy(config, candidate),
                    tuple(tuple(items) for items in strategy_sets),
                    selection_complete=False,
                )
            added = False
            for player, response in enumerate(responses):
                if regret[player] > options.equilibrium_atol and response.strategy not in seen_sets[player]:
                    seen_sets[player].add(response.strategy)
                    strategy_sets[player].append(response.strategy)
                    added = True
            if added:
                continue
        break

    mixed_probabilities: tuple[tuple[float, ...], ...] = ()
    equilibrium_type = "APPROX_PURE"
    if options.mixed_fallback:
        payoff, success, _ = _restricted_game(
            config, strategy_sets, limits.max_restricted_profiles
        )
        mixed = minimize_nashconv(
            payoff,
            valid=np.ones(payoff.shape[:-1], dtype=bool),
            success=success,
            seed=options.seed,
        )
        mixed_probabilities = mixed.probabilities
        if mixed.nash_conv < sum(best_regret):
            equilibrium_type = "APPROX_MIXED"
            best_regret = mixed.regrets
            modal = tuple(
                strategy_sets[player][int(np.argmax(probability))]
                for player, probability in enumerate(mixed.probabilities)
            )
            best_profile = modal
    return Q31SolveResult(
        best_profile,
        equilibrium_type,
        best_regret,
        {
            "gauss_seidel_rounds": gauss_rounds,
            "double_oracle_rounds": oracle_rounds,
        },
        replay_joint_strategy(config, best_profile),
        tuple(tuple(items) for items in strategy_sets),
        mixed_probabilities=mixed_probabilities,
        selection_complete=False,
    )
