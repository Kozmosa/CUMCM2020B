from __future__ import annotations

import json
import pickle
import unittest
from dataclasses import replace
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

import numpy as np

from q3.action_enum import (
    enumerate_individual_action_arrays,
    enumerate_individual_actions,
    enumerate_initial_purchases,
)
from q3.adaptive import (
    AdaptiveOptions,
    AdaptiveQ3Solver,
    deterministic_array_action_candidates,
    deterministic_action_candidates,
    solve_q3_2,
)
from q3.data import Q3Config, level5, level6, tiny_level6
from q3.heuristic import (
    HeuristicOptions,
    HeuristicPolicy,
    generate_heuristic_policies,
    simulate_heuristic_profile,
    solve_q3_2_heuristic,
)
from q3.submission_heuristic import _load_warm_start, enumerate_submission_routes
from q3.model import Action, ActionKind, PlayerState, Status, terminal_state_value
from q3.open_loop import (
    OpenLoopStrategy,
    Q31Limits,
    best_response_open_loop,
    replay_joint_strategy,
)
from q3.pruning import ResourceAwareSinglePlayerUpperBound
from q3.purchase_oracle import (
    CUDA_PURCHASE_AVAILABLE,
    PurchaseLatticeOracle,
    PurchaseOracleOptions,
)
from q3.reports import PolicyEntry
from q3.runtime import BudgetExceeded, BudgetManager
from q3.resource_index import ResourceIndex
from q3.stage_game import (
    ProfileBatchEvaluation,
    minimize_nashconv,
    pure_nash_indices,
    select_pure_equilibrium,
)
from q3.stochastic_dp import ExactQ3Solver, SolverLimits
from q3.storage import CompactStateCodec, LayerCache, PackedStateCodec
from q3.transition import (
    apply_joint_profile_arrays,
    apply_joint_transition_batch,
    apply_unilateral_transition_arrays,
    apply_unilateral_transition_encoded_arrays,
    materialize_transition_successors,
)


def one_step_config(*, n_players: int = 1) -> Q3Config:
    return Q3Config(
        name="one-step-known",
        n_players=n_players,
        deadline=1,
        mine_income=0,
        p_weather={"sunny": 1.0},
        weight_limit=6,
        init_cash=20,
        water_weight=1,
        food_weight=1,
        water_price=1,
        food_price=1,
        water_consume={"sunny": 1},
        food_consume={"sunny": 1},
        start=1,
        end=2,
        mines=frozenset(),
        villages=frozenset(),
        adj={1: (2,), 2: (1,)},
        nodes=(1, 2),
        failure_penalty=100,
        weather_sequence=("sunny",),
    )


class PublicTypeTests(unittest.TestCase):
    def test_level5_carries_official_known_weather(self) -> None:
        self.assertEqual(
            level5().weather_sequence,
            (
                "sunny",
                "hot",
                "sunny",
                "sunny",
                "sunny",
                "sunny",
                "hot",
                "hot",
                "hot",
                "hot",
            ),
        )

    def test_policy_entry_validates_pure_and_mixed(self) -> None:
        stay = Action(ActionKind.STAY)
        move = Action(ActionKind.MOVE, destination=2)
        self.assertTrue(PolicyEntry.pure(stay).is_pure)
        mixed = PolicyEntry.mixed(((stay, 2.0), (move, 1.0)))
        self.assertAlmostEqual(sum(p for _, p in mixed.distribution), 1.0)
        with self.assertRaises(ValueError):
            PolicyEntry(action=stay, distribution=((move, 1.0),))

    def test_compact_codec_round_trip_absorbed_and_negative_payoff(self) -> None:
        cfg = replace(tiny_level6(), failure_penalty=1_000_000_000)
        codec = CompactStateCodec(ResourceIndex.build(cfg), cfg.n_players)
        state = (
            PlayerState(Status.ACTIVE, position=2, water=3, food=4, cash_scaled=7),
            PlayerState(Status.FINISHED, fixed_payoff_scaled=123),
            PlayerState(Status.FAILED, fixed_payoff_scaled=-6_000_000_007),
        )
        self.assertEqual(codec.decode(codec.encode(state)), state)
        packed = PackedStateCodec(codec.resources, cfg.n_players)
        self.assertEqual(packed.decode(packed.encode(30, state)), (30, state))
        self.assertEqual(len(packed.encode(30, state)), 68)
        cache = LayerCache(codec)
        from q3.model import StateValue

        value = StateValue((1.0, 2.0, -1e9), (0.0, 1.0, 0.0))
        cache.put(state, value)
        self.assertEqual(cache.get(state), value)
        self.assertEqual(cache.arrays()[0].shape, (1, 15))

    def test_budget_manager_closeout_and_stop(self) -> None:
        calls: list[bool] = []
        manager = BudgetManager(
            wall_seconds=2.0,
            closeout_seconds=1.0,
            checkpoint_callback=lambda: calls.append(True),
        )
        manager.started -= 1.5
        manager.check()
        self.assertEqual(calls, [True])
        manager.started -= 1.0
        with self.assertRaises(BudgetExceeded):
            manager.check()


class HeuristicBackendTests(unittest.TestCase):
    def test_submission_route_universe_covers_all_short_level6_paths(self) -> None:
        routes = enumerate_submission_routes(level6(), 12)
        self.assertEqual(len(routes), 804)
        self.assertIn((1, 6, 11, 16, 17, 18, 23, 24, 25), routes)

    def test_generated_level6_library_is_bounded_valid_and_route_diverse(self) -> None:
        cfg = level6()
        options = HeuristicOptions(
            episodes=2,
            max_policies=16,
            route_variants_per_family=2,
        )
        policies = generate_heuristic_policies(cfg, options)
        self.assertLessEqual(len(policies), options.max_policies)
        self.assertTrue(any(policy.route[1] == 2 for policy in policies))
        self.assertTrue(any(policy.route[1] == 6 for policy in policies))
        self.assertTrue(any(policy.mine_days > 0 for policy in policies))
        self.assertEqual(
            {policy.mine_days for policy in policies if policy.mine_days > 0},
            {2, 4},
        )
        self.assertTrue(
            any(policy.village_water_target > 0 for policy in policies)
        )
        for policy in policies:
            self.assertEqual(policy.route[0], cfg.start)
            self.assertEqual(policy.route[-1], cfg.end)
            self.assertLessEqual(
                cfg.water_weight * policy.initial_water
                + cfg.food_weight * policy.initial_food,
                cfg.weight_limit,
            )

    def test_submission_warm_start_restores_library_and_audit_response(self) -> None:
        cfg = level6()
        routes = enumerate_submission_routes(cfg, 12)
        seed_policy = generate_heuristic_policies(
            cfg, HeuristicOptions(episodes=2, max_policies=1)
        )[0]
        with TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"
            result.write_text(
                json.dumps(
                    {
                        "policy": {
                            "library": [
                                {
                                    "name": seed_policy.name,
                                    "family": seed_policy.family,
                                    "route": seed_policy.route,
                                    "initial_purchase": {
                                        "water": seed_policy.initial_water,
                                        "food": seed_policy.initial_food,
                                    },
                                    "mine_days": seed_policy.mine_days,
                                    "village_target": {
                                        "water": seed_policy.village_water_target,
                                        "food": seed_policy.village_food_target,
                                    },
                                    "safety_factor": seed_policy.safety_factor,
                                    "yield_when_crowded": False,
                                    "mine_only_alone": False,
                                }
                            ]
                        },
                        "stats": {
                            "audit_best_deviations": [
                                {
                                    "policy": "response-r0056-m0-s2.5-y1-a0",
                                    "mean_gain": 100.0,
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            policies, stats = _load_warm_start(cfg, routes, str(result))
        self.assertEqual(len(policies), 2)
        self.assertEqual(stats["library"], 1)
        self.assertEqual(stats["audit_deviations_added"], 1)
        restored = policies[1]
        self.assertEqual(restored.name, "response-r0056-m0-s2.5-y1-a0")
        self.assertEqual(restored.route, routes[56])
        self.assertTrue(restored.yield_when_crowded)

    def test_heuristic_policy_cap_allows_final_run_size(self) -> None:
        self.assertEqual(HeuristicOptions(episodes=2, max_policies=48).max_policies, 48)
        with self.assertRaises(ValueError):
            HeuristicOptions(episodes=2, max_policies=65)

    def test_heuristic_replay_uses_exact_transition_and_terminal_refund(self) -> None:
        cfg = one_step_config(n_players=1)
        policy = HeuristicPolicy(
            name="direct",
            family="test",
            route=(1, 2),
            initial_water=3,
            initial_food=3,
        )
        result = simulate_heuristic_profile(cfg, (policy,), ("sunny",))
        self.assertEqual(result.success, (1.0,))
        self.assertEqual(result.payoff, (15.0,))
        self.assertEqual(result.replay[0].actions[0].kind, ActionKind.MOVE)
        self.assertEqual(result.replay[0].state[0].status, Status.FINISHED)

    def test_response_policy_can_yield_to_avoid_shared_start(self) -> None:
        cfg = replace(
            one_step_config(n_players=1),
            n_players=2,
            weight_limit=24,
            init_cash=100,
        )
        yielding = HeuristicPolicy(
            name="yielding",
            family="test",
            route=(1, 2),
            initial_water=8,
            initial_food=8,
            yield_when_crowded=True,
        )
        moving = replace(yielding, name="moving", yield_when_crowded=False)
        result = simulate_heuristic_profile(
            cfg, (yielding, moving), ("sunny",)
        )
        self.assertEqual(result.replay[0].actions[0].kind, ActionKind.STAY)
        self.assertEqual(result.replay[0].actions[1].kind, ActionKind.MOVE)
        self.assertEqual(result.success, (0.0, 1.0))

    def test_heuristic_solver_selects_best_policy_deterministically(self) -> None:
        cfg = one_step_config(n_players=1)
        success = HeuristicPolicy(
            name="success",
            family="test",
            route=(1, 2),
            initial_water=2,
            initial_food=2,
        )
        failure = HeuristicPolicy(
            name="failure",
            family="test",
            route=(1, 2),
            initial_water=1,
            initial_food=1,
        )
        options = HeuristicOptions(
            episodes=8,
            audit_episodes=16,
            audit_replicates=1,
            stability_episodes=4,
            stability_replicates=1,
            max_policies=2,
            initial_policies=2,
            route_max_moves=1,
            response_screening_episodes=4,
            response_route_candidates=1,
            response_audit_candidates=2,
            response_additions_per_round=1,
            response_rounds=2,
            response_stable_rounds=2,
            response_training_regret=0.0,
            submission_mean_regret=0.0,
            submission_upper_regret=0.0,
            submission_success_lower=0.8,
            equilibrium="pure",
            seed=7,
        )
        first = solve_q3_2_heuristic(
            cfg, options=options, policies=(success, failure)
        )
        second = solve_q3_2(
            cfg,
            "heuristic",
            quality_target=1e-9,
            heuristic_options=options,
            workers=2,
        )
        repeated = solve_q3_2_heuristic(
            cfg, options=options, policies=(success, failure)
        )
        self.assertEqual(first.status, "HEURISTIC_PURE")
        self.assertEqual(first.success, (1.0,))
        self.assertEqual(first.value_lower, (16.0,))
        self.assertEqual(first.value_upper, (16.0,))
        self.assertEqual(first.max_regret_upper, 0.0)
        self.assertEqual(
            first.policy["representative_profile"], ("success",)
        )
        self.assertEqual(first.value_lower, repeated.value_lower)
        self.assertEqual(first.policy, repeated.policy)
        self.assertEqual(second.backend, "heuristic")
        self.assertEqual(second.status, "SUBMISSION_READY_EMPIRICAL_EQ")
        self.assertEqual(
            second.stats["regret_scope"],
            "broad_parameterized_policy_response_search",
        )
        self.assertTrue(second.stats["submission_quality_met"])

    def test_heuristic_budget_stop_has_no_checkpoint_claim(self) -> None:
        budget = BudgetManager(wall_seconds=60.0)
        budget.cancel()
        report = solve_q3_2_heuristic(
            one_step_config(n_players=1),
            options=HeuristicOptions(
                episodes=4,
                audit_episodes=4,
                audit_replicates=1,
                stability_episodes=2,
                stability_replicates=1,
                max_policies=2,
                initial_policies=2,
                route_max_moves=1,
            ),
            budget_manager=budget,
        )
        self.assertEqual(report.status, "SEARCH_STOPPED")
        self.assertIsNone(report.checkpoint)

    def test_submission_gate_rejects_profitable_response_at_policy_cap(self) -> None:
        cfg = replace(
            one_step_config(n_players=1),
            p_weather={"sunny": 0.9, "hot": 0.1},
            water_consume={"sunny": 1, "hot": 3},
            food_consume={"sunny": 1, "hot": 3},
            weight_limit=12,
            weather_sequence=None,
        )
        report = solve_q3_2_heuristic(
            cfg,
            options=HeuristicOptions(
                episodes=100,
                audit_episodes=20,
                audit_replicates=1,
                stability_episodes=20,
                stability_replicates=1,
                max_policies=1,
                initial_policies=1,
                route_max_moves=1,
                response_screening_episodes=100,
                response_route_candidates=1,
                response_audit_candidates=2,
                response_additions_per_round=1,
                response_rounds=1,
                response_stable_rounds=1,
                response_training_regret=1.0,
                submission_success_lower=0.5,
                seed=11,
            ),
        )
        self.assertEqual(report.status, "EMPIRICAL_EQ_NOT_READY")
        self.assertTrue(report.stats["policy_cap_reached"])
        self.assertFalse(report.stats["response_complete"])
        self.assertFalse(report.stats["submission_quality_met"])


class OpenLoopTests(unittest.TestCase):
    def test_deviation_can_force_opponent_failure(self) -> None:
        cfg = replace(
            tiny_level6(n_players=2, deadline=2),
            weather_sequence=("sunny", "sunny"),
        )
        inactive = Action(ActionKind.INACTIVE)
        player0_stays = OpenLoopStrategy(
            Action(ActionKind.INITIAL_BUY, buy_water=4, buy_food=4),
            (Action(ActionKind.STAY), inactive),
        )
        player0_moves = OpenLoopStrategy(
            Action(ActionKind.INITIAL_BUY, buy_water=4, buy_food=4),
            (Action(ActionKind.MOVE, destination=2), inactive),
        )
        player1 = OpenLoopStrategy(
            Action(ActionKind.INITIAL_BUY, buy_water=2, buy_food=2),
            (Action(ActionKind.MOVE, destination=2), inactive),
        )
        solo_edge = replay_joint_strategy(cfg, (player0_stays, player1))
        congested = replay_joint_strategy(cfg, (player0_moves, player1))
        self.assertIs(solo_edge.states[1][1].status, Status.ACTIVE)
        self.assertIs(congested.states[1][1].status, Status.FAILED)
        self.assertTrue(congested.days[0].forced_failures[1])

    def test_best_response_matches_exhaustive_one_step_game(self) -> None:
        cfg = one_step_config()
        dummy = OpenLoopStrategy(
            Action(ActionKind.INITIAL_BUY), (Action(ActionKind.FAIL),)
        )
        response = best_response_open_loop(cfg, 0, (dummy,))
        best = -float("inf")
        for initial in enumerate_initial_purchases(cfg, PlayerState(
            Status.ACTIVE, position=cfg.start, cash_scaled=cfg.init_cash_scaled
        )):
            post_water = initial.buy_water
            post_food = initial.buy_food
            state = PlayerState(
                Status.ACTIVE,
                position=cfg.start,
                water=post_water,
                food=post_food,
                cash_scaled=cfg.init_cash_scaled
                - cfg.money_scale * (post_water + post_food),
            )
            for action in enumerate_individual_actions(cfg, state, "sunny"):
                strategy = OpenLoopStrategy(initial, (action,))
                best = max(best, replay_joint_strategy(cfg, (strategy,)).payoff[0])
        self.assertEqual(response.payoff, best)
        self.assertEqual(
            replay_joint_strategy(cfg, (response.strategy,)).payoff[0], best
        )

    def test_compact_numba_best_response_matches_scalar(self) -> None:
        cfg = replace(
            tiny_level6(n_players=3, deadline=2),
            villages=frozenset(),
            weather_sequence=("sunny", "sunny"),
        )
        profiles = (
            tuple(
                OpenLoopStrategy(
                    Action(ActionKind.INITIAL_BUY, buy_water=6, buy_food=6),
                    (
                        Action(ActionKind.MOVE, destination=2),
                        Action(ActionKind.MOVE, destination=3),
                    ),
                )
                for _ in range(3)
            ),
            (
                OpenLoopStrategy(
                    Action(ActionKind.INITIAL_BUY, buy_water=4, buy_food=4),
                    (Action(ActionKind.STAY), Action(ActionKind.MOVE, destination=2)),
                ),
                OpenLoopStrategy(
                    Action(ActionKind.INITIAL_BUY, buy_water=6, buy_food=6),
                    (Action(ActionKind.MOVE, destination=2), Action(ActionKind.MINE)),
                ),
                OpenLoopStrategy(
                    Action(ActionKind.INITIAL_BUY, buy_water=2, buy_food=2),
                    (
                        Action(ActionKind.MOVE, destination=2),
                        Action(ActionKind.MOVE, destination=3),
                    ),
                ),
            ),
        )
        for profile in profiles:
            for player in range(3):
                scalar = best_response_open_loop(
                    cfg,
                    player,
                    profile,
                    limits=Q31Limits(
                        max_frontier_states=100_000,
                        use_compact_numba=False,
                    ),
                )
                compact = best_response_open_loop(
                    cfg,
                    player,
                    profile,
                    limits=Q31Limits(
                        max_frontier_states=100_000,
                        use_compact_numba=True,
                    ),
                )
                self.assertEqual(compact.payoff_scaled, scalar.payoff_scaled)
                trial = list(profile)
                trial[player] = compact.strategy
                self.assertEqual(
                    replay_joint_strategy(cfg, trial).payoff_scaled[player],
                    scalar.payoff_scaled,
                )

    def test_compact_numba_matches_scalar_on_random_profiles(self) -> None:
        cfg = replace(
            tiny_level6(n_players=3, deadline=2),
            villages=frozenset(),
            weather_sequence=("sunny", "sunny"),
        )
        random = Random(20260728)
        day_actions = (
            Action(ActionKind.FAIL),
            Action(ActionKind.STAY),
            Action(ActionKind.MINE),
            Action(ActionKind.MOVE, destination=1),
            Action(ActionKind.MOVE, destination=2),
            Action(ActionKind.MOVE, destination=3),
        )
        purchases = tuple(
            Action(ActionKind.INITIAL_BUY, buy_water=water, buy_food=food)
            for water, food in ((0, 0), (2, 2), (4, 4), (6, 6), (8, 4))
        )
        for _ in range(12):
            profile = tuple(
                OpenLoopStrategy(
                    random.choice(purchases),
                    tuple(random.choice(day_actions) for _ in range(cfg.deadline)),
                )
                for _ in range(cfg.n_players)
            )
            player = random.randrange(cfg.n_players)
            scalar = best_response_open_loop(
                cfg,
                player,
                profile,
                limits=Q31Limits(
                    max_frontier_states=100_000,
                    use_compact_numba=False,
                ),
            )
            compact = best_response_open_loop(
                cfg,
                player,
                profile,
                limits=Q31Limits(
                    max_frontier_states=100_000,
                    use_compact_numba=True,
                ),
            )
            self.assertEqual(compact.payoff_scaled, scalar.payoff_scaled)


class BoundsAdaptiveAndCheckpointTests(unittest.TestCase):
    def test_unilateral_transition_arrays_match_general_batch(self) -> None:
        cfg = tiny_level6(n_players=3, deadline=2)
        state = tuple(
            PlayerState(
                Status.ACTIVE,
                position=2,
                water=8,
                food=8,
                cash_scaled=cfg.init_cash_scaled,
            )
            for _ in range(cfg.n_players)
        )
        actions = enumerate_individual_actions(cfg, state[0], "sunny")
        deviations = actions[: min(32, len(actions))]
        base = tuple(actions[0] for _ in range(cfg.n_players))
        profiles = []
        for deviation in deviations:
            profile = list(base)
            profile[1] = deviation
            profiles.append(tuple(profile))

        compact = apply_unilateral_transition_arrays(
            cfg, state, base, 1, deviations, "sunny"
        )
        general = apply_joint_transition_batch(cfg, state, profiles, "sunny")
        general_arrays = apply_joint_profile_arrays(cfg, state, profiles, "sunny")
        np.testing.assert_array_equal(compact.valid, general.valid)
        np.testing.assert_array_equal(general_arrays.valid, general.valid)
        np.testing.assert_array_equal(general_arrays.position, compact.position)
        np.testing.assert_array_equal(general_arrays.water, compact.water)
        np.testing.assert_array_equal(general_arrays.food, compact.food)
        np.testing.assert_array_equal(
            general_arrays.cash_scaled, compact.cash_scaled
        )
        np.testing.assert_array_equal(compact.edge_count, general.edge_count)
        np.testing.assert_array_equal(compact.mine_count, general.mine_count)
        np.testing.assert_array_equal(
            compact.village_buyer_count, general.village_buyer_count
        )
        self.assertEqual(
            materialize_transition_successors(cfg, state, compact),
            general.successors,
        )
        solver = ExactQ3Solver(cfg)
        try:
            terminal = solver._evaluate_transition_arrays(
                cfg.deadline, state, general_arrays
            )
            for row in np.flatnonzero(general.valid):
                expected = terminal_state_value(cfg, general.successors[int(row)])
                self.assertEqual(tuple(terminal.payoff[int(row)]), expected.value)
                self.assertEqual(tuple(terminal.success[int(row)]), expected.success)
        finally:
            solver.close()

    def test_vectorized_action_candidates_match_scalar_reference(self) -> None:
        cfg = tiny_level6(n_players=3, deadline=2)
        state = PlayerState(
            Status.ACTIVE,
            position=2,
            water=0,
            food=0,
            cash_scaled=cfg.init_cash_scaled,
        )
        actions = enumerate_individual_actions(cfg, state, "sunny")
        warm = actions[3:6]

        def reference(per_skeleton: int) -> tuple[Action, ...]:
            from q3.adaptive import _action_targets
            from q3.profile_enum import action_skeleton

            grouped: dict[object, list[Action]] = {}
            for action in actions:
                grouped.setdefault(action_skeleton(action), []).append(action)
            selected = set(action for action in warm if action in actions)
            for group in grouped.values():
                group.sort(key=lambda action: action.code)
                if len(group) <= per_skeleton:
                    selected.update(group)
                    continue
                selected.add(group[0])
                selected.add(group[-1])
                max_load = max(
                    1,
                    max(
                        cfg.water_weight * action.buy_water
                        + cfg.food_weight * action.buy_food
                        for action in group
                    ),
                )
                for load_target, ratio_target in _action_targets(
                    max(0, per_skeleton - 2)
                ):
                    def distance(
                        action: Action,
                    ) -> tuple[float, tuple[int, int, int, int]]:
                        water_load = cfg.water_weight * action.buy_water
                        food_load = cfg.food_weight * action.buy_food
                        load = water_load + food_load
                        ratio = water_load / load if load else 0.5
                        return (
                            abs(load / max_load - load_target)
                            + abs(ratio - ratio_target),
                            action.code,
                        )

                    selected.add(min(group, key=distance))
                if len([action for action in selected if action in group]) < per_skeleton:
                    for index in np.linspace(
                        0, len(group) - 1, per_skeleton, dtype=int
                    ):
                        selected.add(group[int(index)])
                        if (
                            len([action for action in selected if action in group])
                            >= per_skeleton
                        ):
                            break
            return tuple(sorted(selected, key=lambda action: action.code))

        for per_skeleton in (3, 5, 12):
            expected = reference(per_skeleton)
            self.assertEqual(
                deterministic_action_candidates(
                    cfg,
                    actions,
                    per_skeleton=per_skeleton,
                    warm_actions=warm,
                ),
                expected,
            )

    def test_compact_action_arrays_match_object_enumeration_and_candidates(self) -> None:
        cfg = tiny_level6(n_players=3, deadline=2)
        state = PlayerState(
            Status.ACTIVE,
            position=2,
            water=0,
            food=0,
            cash_scaled=cfg.init_cash_scaled,
        )
        for weather in cfg.weather_order:
            expected = enumerate_individual_actions(cfg, state, weather)
            compact = enumerate_individual_action_arrays(cfg, state, weather)
            self.assertEqual(compact.action_tuple(), expected)
            for per_skeleton in (3, 5, 12):
                self.assertEqual(
                    deterministic_array_action_candidates(
                        cfg, compact, per_skeleton=per_skeleton
                    ),
                    deterministic_action_candidates(
                        cfg, expected, per_skeleton=per_skeleton
                    ),
                )

    def test_resource_aware_upper_bound_dominates_exact(self) -> None:
        cfg = tiny_level6(n_players=1)
        bound = ResourceAwareSinglePlayerUpperBound.build(cfg)
        solver = ExactQ3Solver(cfg)
        try:
            for day in (0, 1):
                for position in (1, 2):
                    for water, food in ((0, 0), (3, 3), (6, 6)):
                        player = PlayerState(
                            Status.ACTIVE,
                            position=position,
                            water=water,
                            food=food,
                            cash_scaled=cfg.init_cash_scaled,
                        )
                        exact = solver.solve_state(day, (player,)).value[0]
                        self.assertLessEqual(exact, bound.value(day, player))
        finally:
            solver.close()

    def test_resource_bound_batch_and_threads_are_deterministic(self) -> None:
        cfg = tiny_level6(n_players=1)
        serial = ResourceAwareSinglePlayerUpperBound.build(cfg, threads=1)
        parallel = ResourceAwareSinglePlayerUpperBound.build(cfg, threads=4)
        resource_ids = np.asarray(
            [
                serial.resources.encode(0, 0),
                serial.resources.encode(3, 3),
                serial.resources.encode(6, 6),
                serial.resources.encode(9, 3),
            ],
            dtype=np.int64,
        )
        positions = np.asarray([1, 2, 1, 2], dtype=np.int64)
        expected = np.asarray(
            [
                serial._residual(0, int(position), int(resource_id))
                for position, resource_id in zip(
                    positions, resource_ids, strict=True
                )
            ]
        )
        np.testing.assert_array_equal(
            serial.residuals_by_id(0, positions, resource_ids), expected
        )
        np.testing.assert_array_equal(
            parallel.residuals_by_id(0, positions, resource_ids), expected
        )
        self.assertGreater(serial.cache_entries, 0)
        self.assertGreater(serial.cache_bytes, 0)

    def test_purchase_oracle_matches_complete_transition_bounds(self) -> None:
        cfg = tiny_level6(n_players=3, deadline=3)
        player = PlayerState(
            Status.ACTIVE,
            position=2,
            water=3,
            food=4,
            cash_scaled=cfg.init_cash_scaled,
        )
        state = tuple(player for _ in range(cfg.n_players))
        action_spaces = tuple(
            enumerate_individual_actions(cfg, item, "sunny") for item in state
        )
        compact = enumerate_individual_action_arrays(cfg, player, "sunny")
        solver = AdaptiveQ3Solver(cfg)
        oracle = PurchaseLatticeOracle(
            cfg,
            solver._upper_bound,
            PurchaseOracleOptions(backend="cpu", parallel_min_actions=1),
        )
        try:
            rng = Random(20260728)
            profiles = [
                tuple(rng.choice(actions) for actions in action_spaces)
                for _ in range(12)
            ]
            profiles.append(
                (
                    Action(ActionKind.STAY),
                    Action(ActionKind.MINE),
                    Action(ActionKind.MOVE, destination=3),
                )
            )
            for base in profiles:
                batch = apply_unilateral_transition_encoded_arrays(
                    cfg,
                    state,
                    base,
                    0,
                    compact.kind,
                    compact.destination,
                    compact.buy_water,
                    compact.buy_food,
                    "sunny",
                )
                expected = solver._unilateral_upper_bounds(1, state, batch, 0)
                finite = expected[np.isfinite(expected)]
                for threshold in (-float("inf"), float(np.median(finite))):
                    screened = oracle.screen(
                        1, state, base, 0, compact, "sunny", threshold
                    )
                    expected_indices = np.flatnonzero(
                        np.isfinite(expected) & (expected >= threshold)
                    )
                    np.testing.assert_array_equal(
                        screened.survivor_indices, expected_indices
                    )
                    np.testing.assert_array_equal(
                        screened.survivor_bounds, expected[expected_indices]
                    )
                    omitted = np.isfinite(expected) & (expected < threshold)
                    if np.any(omitted):
                        self.assertGreaterEqual(
                            screened.max_pruned_bound,
                            float(np.max(expected[omitted])),
                        )
        finally:
            solver.close()

    @unittest.skipUnless(CUDA_PURCHASE_AVAILABLE, "CUDA is unavailable")
    def test_cuda_purchase_oracle_matches_cpu_bit_for_bit(self) -> None:
        cfg = tiny_level6(n_players=3, deadline=3)
        player = PlayerState(
            Status.ACTIVE,
            position=2,
            water=3,
            food=4,
            cash_scaled=cfg.init_cash_scaled,
        )
        state = tuple(player for _ in range(cfg.n_players))
        base = (
            Action(ActionKind.MOVE, destination=3),
            Action(ActionKind.STAY, buy_water=1),
            Action(ActionKind.MINE, buy_food=1),
        )
        actions = enumerate_individual_action_arrays(cfg, player, "sunny")
        bound = ResourceAwareSinglePlayerUpperBound.build(cfg)
        cpu = PurchaseLatticeOracle(
            cfg, bound, PurchaseOracleOptions(backend="cpu")
        ).screen(1, state, base, 0, actions, "sunny", -float("inf"))
        gpu = PurchaseLatticeOracle(
            cfg,
            bound,
            PurchaseOracleOptions(backend="cuda", cuda_min_actions=1),
        ).screen(1, state, base, 0, actions, "sunny", -float("inf"))
        np.testing.assert_array_equal(gpu.survivor_indices, cpu.survivor_indices)
        np.testing.assert_array_equal(gpu.survivor_bounds, cpu.survivor_bounds)

    def test_terminal_leaves_are_evaluated_without_cache_entries(self) -> None:
        cfg = tiny_level6()
        active_state = tuple(
            PlayerState(
                Status.ACTIVE,
                position=2,
                water=6,
                food=6,
                cash_scaled=cfg.init_cash_scaled,
            )
            for _ in range(cfg.n_players)
        )
        failed_state = tuple(
            PlayerState(Status.FAILED, fixed_payoff_scaled=-cfg.money_scale)
            for _ in range(cfg.n_players)
        )
        solver = ExactQ3Solver(cfg)
        try:
            deadline_value = solver.solve_state(cfg.deadline, active_state)
            self.assertEqual(
                solver.solve_state(cfg.deadline, active_state), deadline_value
            )
            solver.solve_state(1, failed_state)
            self.assertEqual(len(solver._value_cache), 0)
            self.assertEqual(solver.stats.cache_hits, 0)
            self.assertEqual(solver.stats.states_solved, 3)
            self.assertEqual(solver.stats.terminal_evaluations, 3)
        finally:
            solver.close()

    def test_adaptive_matches_exact_when_candidates_cover_all_actions(self) -> None:
        cfg = one_step_config()
        limits = SolverLimits(1_000, 1_000_000, 100_000, 128, 1_000_000)
        options = AdaptiveOptions(
            initial_candidates=128,
            village_candidates_per_skeleton=64,
            max_initial_candidates=128,
            max_village_candidates_per_skeleton=64,
        )
        exact = solve_q3_2(cfg, "exact", limits, 1e-9)
        adaptive = solve_q3_2(
            cfg, "adaptive", limits, 1e-9, adaptive_options=options
        )
        self.assertEqual(exact.status, "EXACT_SELECTED")
        self.assertEqual(adaptive.status, "CERTIFIED_PURE")
        self.assertEqual(adaptive.value_lower, exact.value_lower)
        self.assertEqual(adaptive.success, exact.success)
        self.assertEqual(adaptive.max_regret_upper, 0.0)

    def test_level6_late_state_adaptive_matches_exact(self) -> None:
        cfg = level6()
        player = PlayerState(
            Status.ACTIVE,
            position=24,
            water=60,
            food=60,
            cash_scaled=cfg.init_cash_scaled,
        )
        state = tuple(player for _ in range(cfg.n_players))
        limits = SolverLimits(1_000, 250_000, 20_000, 256, 100_000)
        exact = ExactQ3Solver(cfg, limits=limits)
        adaptive = AdaptiveQ3Solver(cfg, limits=limits)
        try:
            exact_value = exact.solve_state(29, state)
            adaptive_value = adaptive.solve_state(29, state)
            np.testing.assert_allclose(
                adaptive_value.value, exact_value.value, atol=1e-10, rtol=0.0
            )
            np.testing.assert_allclose(
                adaptive_value.success, exact_value.success, atol=1e-10, rtol=0.0
            )
            for weather in cfg.weather_order:
                self.assertEqual(
                    adaptive.policy_for(29, state, weather),
                    exact.policy_for(29, state, weather),
                )
        finally:
            exact.close()
            adaptive.close()

    def test_symmetric_shortcut_matches_full_restricted_tensor(self) -> None:
        cfg = tiny_level6(n_players=3, deadline=2)
        player = PlayerState(
            Status.ACTIVE,
            position=2,
            water=6,
            food=6,
            cash_scaled=cfg.init_cash_scaled,
        )
        state = tuple(player for _ in range(cfg.n_players))
        solver = AdaptiveQ3Solver(cfg)
        try:
            action_sets = solver._full_action_sets(state, "sunny")
            evaluator = solver._stage_evaluator(1, state, "sunny", action_sets)
            symmetric = solver._symmetric_restricted_pure(
                state, action_sets, evaluator
            )
            self.assertIsNotNone(symmetric)
            payoff, success, valid = solver._restricted_tensor(
                action_sets, evaluator, symmetric_state=None
            )
            dense = select_pure_equilibrium(
                pure_nash_indices(payoff, valid),
                payoff,
                success,
                action_sets,
            )
            self.assertEqual(symmetric, dense)
        finally:
            solver.close()

    def test_iterative_pure_shortcut_returns_fully_verified_equilibrium(self) -> None:
        cfg = one_step_config(n_players=3)
        stay = Action(ActionKind.STAY)
        move = Action(ActionKind.MOVE, destination=2)
        action_sets = tuple((stay, move) for _ in range(cfg.n_players))

        def evaluator(indices: np.ndarray) -> ProfileBatchEvaluation:
            payoff = np.zeros((len(indices), cfg.n_players), dtype=np.float64)
            success = np.zeros_like(payoff)
            unanimous = np.all(indices == indices[:, :1], axis=1)
            payoff[unanimous] = 1.0
            success[unanimous] = 1.0
            return ProfileBatchEvaluation(
                np.ones(len(indices), dtype=bool), payoff, success
            )

        solver = AdaptiveQ3Solver(cfg)
        try:
            result = solver._iterative_restricted_pure(action_sets, evaluator)
            self.assertIsNotNone(result)
            self.assertIn(result.index, ((0, 0, 0), (1, 1, 1)))
            for player in range(cfg.n_players):
                indices = np.repeat(
                    np.asarray(result.index, dtype=np.int64)[None, :],
                    len(action_sets[player]),
                    axis=0,
                )
                indices[:, player] = np.arange(len(action_sets[player]))
                evaluation = evaluator(indices)
                self.assertGreaterEqual(
                    result.value[player],
                    float(np.max(evaluation.payoff[:, player])),
                )
        finally:
            solver.close()

    def test_adaptive_workers_are_deterministic(self) -> None:
        cfg = tiny_level6(n_players=3, deadline=2)
        player = PlayerState(
            Status.ACTIVE,
            position=2,
            water=6,
            food=6,
            cash_scaled=cfg.init_cash_scaled,
        )
        state = tuple(player for _ in range(cfg.n_players))
        serial = AdaptiveQ3Solver(cfg, workers=1)
        parallel = AdaptiveQ3Solver(cfg, workers=4)
        try:
            self.assertEqual(serial.solve_state(1, state), parallel.solve_state(1, state))
            for weather in cfg.weather_order:
                self.assertEqual(
                    serial.policy_entries_for(1, state, weather),
                    parallel.policy_entries_for(1, state, weather),
                )
        finally:
            serial.close()
            parallel.close()

    def test_matching_pennies_mixed_fallback(self) -> None:
        payoff = np.zeros((2, 2, 2), dtype=np.float64)
        payoff[:, :, 0] = ((1.0, -1.0), (-1.0, 1.0))
        payoff[:, :, 1] = -payoff[:, :, 0]
        result = minimize_nashconv(payoff, starts=4)
        self.assertAlmostEqual(result.probabilities[0][0], 0.5, places=7)
        self.assertAlmostEqual(result.probabilities[1][0], 0.5, places=7)
        self.assertLess(result.nash_conv, 1e-9)

    def test_v2_directory_checkpoint_and_v1_migration(self) -> None:
        cfg = tiny_level6()
        state = tuple(
            PlayerState(
                Status.ACTIVE,
                position=2,
                water=6,
                food=6,
                cash_scaled=cfg.init_cash_scaled,
            )
            for _ in range(cfg.n_players)
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            v2 = root / "checkpoint"
            solver = ExactQ3Solver(cfg, checkpoint_path=v2)
            expected = solver.solve_state(1, state)
            expected_policy = solver.policy_for(1, state, "sunny")
            solver.save_checkpoint()
            self.assertTrue(v2.is_dir())
            self.assertEqual(
                json.loads((v2 / "manifest.json").read_text())["version"], 2
            )
            manifest = json.loads((v2 / "manifest.json").read_text())
            self.assertNotIn(
                cfg.deadline,
                {int(layer["day"]) for layer in manifest["layers"]},
            )

            legacy = root / "legacy.pkl"
            payload = solver._checkpoint_payload()
            payload["version"] = 1
            payload["value_cache"][(cfg.deadline, state)] = solver.solve_state(
                cfg.deadline, state
            )
            with legacy.open("wb") as handle:
                pickle.dump(payload, handle)
            solver.close()

            restored = ExactQ3Solver(cfg, checkpoint_path=v2)
            restored.load_checkpoint()
            self.assertEqual(restored.solve_state(1, state), expected)
            self.assertEqual(restored.policy_for(1, state, "sunny"), expected_policy)
            restored.close()

            migrated = ExactQ3Solver(cfg, checkpoint_path=legacy)
            migrated.load_checkpoint()
            self.assertTrue(
                all(
                    migrated._state_codec.decode(key)[0] != cfg.deadline
                    for key in migrated._value_cache
                )
            )
            self.assertEqual(migrated.solve_state(1, state), expected)
            migrated.close()

    def test_initial_solver_periodic_checkpoint_and_cancel(self) -> None:
        cfg = one_step_config()
        limits = SolverLimits(1_000, 1_000_000, 100_000, 128, 1_000_000)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            periodic = root / "periodic"
            report = solve_q3_2(
                cfg,
                "adaptive",
                limits,
                1e-9,
                checkpoint=str(periodic),
                checkpoint_every_states=1,
            )
            self.assertTrue(periodic.is_dir())
            self.assertGreaterEqual(report.stats["checkpoint_writes"], 2)

            cancelled = root / "cancelled"
            budget = BudgetManager(wall_seconds=60.0)
            budget.cancel()
            stopped = solve_q3_2(
                cfg,
                "adaptive",
                limits,
                1e-9,
                budget_manager=budget,
                checkpoint=str(cancelled),
            )
            self.assertEqual(stopped.status, "SEARCH_STOPPED")
            self.assertTrue(cancelled.is_dir())


if __name__ == "__main__":
    unittest.main()
