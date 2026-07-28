from __future__ import annotations

import unittest
from random import Random

import numpy as np

from q3.action_enum import enumerate_individual_actions
from q3.canonical import canonicalize_state, value_to_original
from q3.data import Q3Config, tiny_level6
from q3.model import Action, ActionKind, PlayerState, StateValue, Status
from q3.resource_index import ResourceIndex
from q3.stage_game import pure_nash_indices, select_pure_equilibrium, verify_pure_equilibrium
from q3.stochastic_dp import ExactQ3Solver, SolverLimits
from q3.transition import (
    apply_joint_transition_batch,
    apply_joint_transition_scalar,
    count_interactions_scalar,
)
from q3.verify import compare_scalar_and_vectorized


def active(cfg: Q3Config, position: int, water: int = 5, food: int = 5, cash: int = 40):
    return PlayerState(
        Status.ACTIVE,
        position=position,
        water=water,
        food=food,
        cash_scaled=cash * cfg.money_scale,
    )


class TransitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = tiny_level6()

    def test_same_directed_edge_congestion(self) -> None:
        state = tuple(active(self.cfg, 1, water=6, food=6) for _ in range(3))
        actions = tuple(Action(ActionKind.MOVE, destination=2) for _ in range(3))
        counts = count_interactions_scalar(state, actions)
        self.assertEqual(counts.edge, (3, 3, 3))
        successor = apply_joint_transition_scalar(self.cfg, state, actions, "sunny")
        self.assertIsNotNone(successor)
        assert successor is not None
        self.assertEqual(tuple(player.water for player in successor), (0, 0, 0))
        self.assertEqual(tuple(player.food for player in successor), (0, 0, 0))

    def test_opposite_edges_do_not_interact(self) -> None:
        state = (active(self.cfg, 1), active(self.cfg, 2), active(self.cfg, 2))
        actions = (
            Action(ActionKind.MOVE, destination=2),
            Action(ActionKind.MOVE, destination=1),
            Action(ActionKind.STAY),
        )
        counts = count_interactions_scalar(state, actions)
        self.assertEqual(counts.edge, (1, 1, 0))

    def test_mine_income_is_shared_exactly(self) -> None:
        state = tuple(active(self.cfg, 2) for _ in range(3))
        actions = tuple(Action(ActionKind.MINE) for _ in range(3))
        successor = apply_joint_transition_scalar(self.cfg, state, actions, "sunny")
        self.assertIsNotNone(successor)
        assert successor is not None
        expected_income_scaled = self.cfg.money_scale * self.cfg.mine_income // 3
        self.assertEqual(
            tuple(player.cash_scaled for player in successor),
            tuple(self.cfg.init_cash_scaled + expected_income_scaled for _ in range(3)),
        )

    def test_two_village_buyers_pay_four_times_base_price(self) -> None:
        state = tuple(active(self.cfg, 2) for _ in range(3))
        actions = (
            Action(ActionKind.STAY, buy_water=1),
            Action(ActionKind.STAY, buy_water=1),
            Action(ActionKind.STAY),
        )
        successor = apply_joint_transition_scalar(self.cfg, state, actions, "sunny")
        self.assertIsNotNone(successor)
        assert successor is not None
        cost = 4 * self.cfg.water_price * self.cfg.money_scale
        self.assertEqual(successor[0].cash_scaled, self.cfg.init_cash_scaled - cost)
        self.assertEqual(successor[1].cash_scaled, self.cfg.init_cash_scaled - cost)
        self.assertEqual(successor[2].cash_scaled, self.cfg.init_cash_scaled)

    def test_storm_move_is_illegal(self) -> None:
        cfg = tiny_level6()
        state = tuple(active(cfg, 1) for _ in range(3))
        actions = tuple(Action(ActionKind.MOVE, destination=2) for _ in range(3))
        self.assertIsNone(apply_joint_transition_scalar(cfg, state, actions, "sandstorm"))

    def test_scalar_vectorized_equivalence(self) -> None:
        state = tuple(active(self.cfg, 2) for _ in range(3))
        action_sets = [
            (
                Action(ActionKind.STAY),
                Action(ActionKind.MINE),
                Action(ActionKind.MOVE, destination=3),
            )
            for _ in range(3)
        ]
        profiles = [
            (a0, a1, a2)
            for a0 in action_sets[0]
            for a1 in action_sets[1]
            for a2 in action_sets[2]
        ]
        ok, message = compare_scalar_and_vectorized(
            self.cfg, state, profiles, "sunny"
        )
        self.assertTrue(ok, message)

    def test_random_village_profiles_match_scalar_reference(self) -> None:
        state = tuple(active(self.cfg, 2, water=3, food=3) for _ in range(3))
        action_sets = [
            enumerate_individual_actions(
                self.cfg, player, "sunny", max_actions=512
            )
            for player in state
        ]
        random = Random(20260728)
        profiles = [
            tuple(random.choice(actions) for actions in action_sets)
            for _ in range(250)
        ]
        ok, message = compare_scalar_and_vectorized(
            self.cfg, state, profiles, "sunny"
        )
        self.assertTrue(ok, message)

    def test_absorbed_players_are_ignored_by_interactions(self) -> None:
        finished = PlayerState(Status.FINISHED, fixed_payoff_scaled=123)
        failed = PlayerState(Status.FAILED, fixed_payoff_scaled=-456)
        state = (finished, failed, active(self.cfg, 2, water=3, food=3))
        profile = (
            Action(ActionKind.INACTIVE),
            Action(ActionKind.INACTIVE),
            Action(ActionKind.MINE),
        )
        scalar = apply_joint_transition_scalar(self.cfg, state, profile, "sunny")
        batch = apply_joint_transition_batch(self.cfg, state, (profile,), "sunny")
        self.assertTrue(batch.valid[0])
        self.assertEqual(scalar, batch.successors[0])
        self.assertEqual(batch.mine_count[0].tolist(), [0, 0, 1])


class EnumerationAndCompressionTests(unittest.TestCase):
    def test_forced_failure_only_when_no_individual_action_exists(self) -> None:
        cfg = tiny_level6()
        state = active(cfg, 1, water=0, food=0)
        actions = enumerate_individual_actions(cfg, state, "sunny")
        self.assertEqual(len(actions), 1)
        self.assertIs(actions[0].kind, ActionKind.FAIL)

    def test_resource_index_is_dense_over_feasible_pairs(self) -> None:
        cfg = tiny_level6()
        index = ResourceIndex.build(cfg)
        pairs = {(int(w), int(f)) for w, f in zip(index.water, index.food, strict=True)}
        expected = {
            (w, f)
            for w in range(cfg.weight_limit + 1)
            for f in range(cfg.weight_limit + 1)
            if w + f <= cfg.weight_limit
        }
        self.assertEqual(pairs, expected)
        for resource_id, pair in enumerate(zip(index.water, index.food, strict=True)):
            self.assertEqual(index.encode(int(pair[0]), int(pair[1])), resource_id)

    def test_canonical_value_round_trip(self) -> None:
        cfg = tiny_level6()
        state = (
            active(cfg, 2, cash=30),
            active(cfg, 1, cash=40),
            active(cfg, 2, cash=20),
        )
        _, mapping = canonicalize_state(state)
        canonical_value = StateValue((10.0, 20.0, 30.0), (0.0, 0.5, 1.0))
        original = value_to_original(canonical_value, mapping)
        for canonical_i, original_i in enumerate(mapping):
            self.assertEqual(original.value[original_i], canonical_value.value[canonical_i])


class StageGameTests(unittest.TestCase):
    def test_pure_nash_with_coupled_feasibility(self) -> None:
        payoff = np.asarray(
            [
                [[3.0, 3.0], [0.0, 4.0]],
                [[4.0, 0.0], [2.0, 2.0]],
            ]
        )
        valid = np.asarray([[True, False], [False, True]])
        indices = pure_nash_indices(payoff, valid)
        self.assertEqual({tuple(row) for row in indices}, {(0, 0), (1, 1)})
        for row in indices:
            ok, message = verify_pure_equilibrium(payoff, valid, tuple(row))
            self.assertTrue(ok, message)

    def test_deterministic_selection_prefers_success(self) -> None:
        payoff = np.asarray(
            [
                [[3.0, 3.0], [0.0, 0.0]],
                [[0.0, 0.0], [2.0, 2.0]],
            ]
        )
        success = np.zeros_like(payoff)
        success[1, 1] = (1.0, 1.0)
        valid = np.ones((2, 2), dtype=bool)
        indices = pure_nash_indices(payoff, valid)
        action_sets = (
            (Action(ActionKind.STAY), Action(ActionKind.MOVE, destination=2)),
            (Action(ActionKind.STAY), Action(ActionKind.MOVE, destination=2)),
        )
        selected = select_pure_equilibrium(indices, payoff, success, action_sets)
        self.assertEqual(selected.index, (1, 1))


class SparseSolverTests(unittest.TestCase):
    def test_three_player_one_step_smoke(self) -> None:
        cfg = tiny_level6()
        state = tuple(active(cfg, 2, water=6, food=6) for _ in range(3))
        solver = ExactQ3Solver(
            cfg,
            limits=SolverLimits(
                max_actions_per_player=128,
                max_joint_profiles=100_000,
                max_cached_states=20_000,
            ),
        )
        result = solver.solve_state(1, state)
        self.assertEqual(result.success, (1.0, 1.0, 1.0))
        actions = solver.policy_for(1, state, "sunny")
        self.assertEqual(
            tuple((action.kind, action.destination) for action in actions),
            tuple((ActionKind.MOVE, 3) for _ in range(3)),
        )

    def test_one_player_initial_purchase_stage(self) -> None:
        cfg = Q3Config(
            name="one-step",
            n_players=1,
            deadline=1,
            mine_income=0,
            p_weather={"sunny": 1.0},
            weight_limit=8,
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
        )
        solver = ExactQ3Solver(
            cfg,
            limits=SolverLimits(128, 10_000, 10_000),
        )
        result = solver.solve_initial_purchases()
        self.assertEqual(
            (result.purchases[0].buy_water, result.purchases[0].buy_food),
            (2, 2),
        )
        self.assertEqual(result.value.success, (1.0,))


if __name__ == "__main__":
    unittest.main()
