from __future__ import annotations

import json
import pickle
import unittest
from dataclasses import replace
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

import numpy as np

from q3.action_enum import enumerate_individual_actions, enumerate_initial_purchases
from q3.adaptive import AdaptiveOptions, AdaptiveQ3Solver, solve_q3_2
from q3.data import Q3Config, level5, level6, tiny_level6
from q3.model import Action, ActionKind, PlayerState, Status
from q3.open_loop import (
    OpenLoopStrategy,
    Q31Limits,
    best_response_open_loop,
    replay_joint_strategy,
)
from q3.pruning import ResourceAwareSinglePlayerUpperBound
from q3.reports import PolicyEntry
from q3.runtime import BudgetExceeded, BudgetManager
from q3.resource_index import ResourceIndex
from q3.stage_game import minimize_nashconv
from q3.stochastic_dp import ExactQ3Solver, SolverLimits
from q3.storage import CompactStateCodec, LayerCache


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

            legacy = root / "legacy.pkl"
            payload = solver._checkpoint_payload()
            payload["version"] = 1
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
            self.assertEqual(migrated.solve_state(1, state), expected)
            migrated.close()


if __name__ == "__main__":
    unittest.main()
