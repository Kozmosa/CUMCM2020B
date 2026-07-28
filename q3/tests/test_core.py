from __future__ import annotations

import unittest
from itertools import product
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

import numpy as np

from q3.action_enum import (
    enumerate_individual_actions,
    enumerate_initial_purchases_bounded,
)
from q3.canonical import canonicalize_state, value_to_original
from q3.data import Q3Config, level6, tiny_level6
from q3.model import Action, ActionKind, PlayerState, StateValue, Status
from q3.profile_enum import StructuredEnumerationStats, iter_structured_profile_blocks
from q3.pruning import (
    RelaxedSinglePlayerUpperBound,
    optimistic_initial_resource_requirements,
)
from q3.resource_index import ResourceIndex
from q3.stage_game import (
    ProfileBatchEvaluation,
    chunked_pure_nash_search,
    pure_nash_indices,
    select_pure_equilibrium,
    verify_pure_equilibrium,
)
from q3.stochastic_dp import ExactQ3Solver, ResourceLimitExceeded, SolverLimits
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
        self.assertIsNone(
            apply_joint_transition_scalar(cfg, state, actions, "sandstorm")
        )

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
        ok, message = compare_scalar_and_vectorized(self.cfg, state, profiles, "sunny")
        self.assertTrue(ok, message)

    def test_random_village_profiles_match_scalar_reference(self) -> None:
        state = tuple(active(self.cfg, 2, water=3, food=3) for _ in range(3))
        action_sets = [
            enumerate_individual_actions(self.cfg, player, "sunny", max_actions=512)
            for player in state
        ]
        random = Random(20260728)
        profiles = [
            tuple(random.choice(actions) for actions in action_sets) for _ in range(250)
        ]
        ok, message = compare_scalar_and_vectorized(self.cfg, state, profiles, "sunny")
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
            self.assertEqual(
                original.value[original_i], canonical_value.value[canonical_i]
            )

    def test_structured_blocks_equal_full_joint_feasibility(self) -> None:
        cfg = tiny_level6()
        state = tuple(active(cfg, 2, water=5, food=5) for _ in range(3))
        action_sets = tuple(
            enumerate_individual_actions(cfg, player, "sunny", max_actions=128)
            for player in state
        )
        indexed_profiles = tuple(
            product(*(range(len(actions)) for actions in action_sets))
        )
        profiles = tuple(
            tuple(action_sets[player][index[player]] for player in range(3))
            for index in indexed_profiles
        )
        expected_batch = apply_joint_transition_batch(cfg, state, profiles, "sunny")
        expected = {
            indexed_profiles[row] for row in np.flatnonzero(expected_batch.valid)
        }

        stats = StructuredEnumerationStats()
        actual: set[tuple[int, ...]] = set()
        for block in iter_structured_profile_blocks(
            cfg,
            state,
            action_sets,
            "sunny",
            block_size=7,
            stats=stats,
        ):
            actual.update(tuple(int(x) for x in row) for row in block.indices)
            block_batch = apply_joint_transition_batch(
                cfg, state, block.profiles, "sunny"
            )
            self.assertTrue(block_batch.valid.all())
        self.assertEqual(actual, expected)
        self.assertEqual(stats.raw_profiles, len(indexed_profiles))
        self.assertEqual(stats.feasible_profiles, len(expected))
        self.assertGreater(stats.profiles_pruned, 0)

    def test_initial_purchase_dominance_pruning_is_optimistic_and_lossless(
        self,
    ) -> None:
        cfg = level6()
        state = PlayerState(
            Status.ACTIVE,
            position=cfg.start,
            cash_scaled=cfg.init_cash_scaled,
        )
        requirements = optimistic_initial_resource_requirements(cfg)
        self.assertEqual(requirements, ((30, 40),))
        full = enumerate_initial_purchases_bounded(
            cfg,
            state,
            max_actions=200_000,
            prune_strictly_dominated=False,
        )
        reduced = enumerate_initial_purchases_bounded(
            cfg,
            state,
            max_actions=200_000,
            prune_strictly_dominated=True,
        )
        self.assertIn(Action(ActionKind.INITIAL_BUY), reduced)
        self.assertTrue(
            all(
                (not action.is_buyer)
                or (action.buy_water >= 30 and action.buy_food >= 40)
                for action in reduced
            )
        )
        self.assertEqual(len(full), 120_601)
        self.assertEqual(len(reduced), 88_925)


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

    def test_chunked_search_matches_dense_random_games(self) -> None:
        random = np.random.default_rng(20260728)
        action_sets = tuple(
            tuple(
                Action(ActionKind.MOVE, destination=index + 1) for index in range(count)
            )
            for count in (2, 3, 2)
        )
        for _ in range(20):
            payoff = random.normal(size=(2, 3, 2, 3))
            success = random.random(size=(2, 3, 2, 3))
            valid = random.random(size=(2, 3, 2)) > 0.25

            def evaluator(
                indices: np.ndarray,
                valid: np.ndarray = valid,
                payoff: np.ndarray = payoff,
                success: np.ndarray = success,
            ) -> ProfileBatchEvaluation:
                coordinates = tuple(indices[:, player] for player in range(3))
                return ProfileBatchEvaluation(
                    valid[coordinates], payoff[coordinates], success[coordinates]
                )

            dense_indices = {
                tuple(int(x) for x in row) for row in pure_nash_indices(payoff, valid)
            }
            chunked_indices = {
                equilibrium.index
                for equilibrium in chunked_pure_nash_search(
                    action_sets,
                    evaluator,
                    chunk_size=2,
                    workers=2,
                )
            }
            self.assertEqual(chunked_indices, dense_indices)


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

    def test_relaxed_upper_bound_dominates_exact_single_player_values(self) -> None:
        cfg = tiny_level6(n_players=1)
        upper_bound = RelaxedSinglePlayerUpperBound.build(cfg)
        solver = ExactQ3Solver(cfg)
        try:
            for day in (0, 1):
                for position in (1, 2):
                    for water, food in ((0, 0), (3, 3), (6, 6)):
                        player = active(
                            cfg,
                            position,
                            water=water,
                            food=food,
                            cash=40,
                        )
                        exact = solver.solve_state(day, (player,)).value[0]
                        relaxed = upper_bound.value(day, player)
                        self.assertLessEqual(exact, relaxed)
        finally:
            solver.close()

    def test_bound_pruning_matches_unpruned_chunked_search(self) -> None:
        cfg = tiny_level6()
        state = tuple(active(cfg, 2, water=6, food=6) for _ in range(3))
        limits = SolverLimits(128, 1, 20_000, 1, 100_000)
        pruned = ExactQ3Solver(
            cfg,
            limits=limits,
            workers=2,
            enable_bound_pruning=True,
            record_pruning_certificates=True,
        )
        unpruned = ExactQ3Solver(
            cfg,
            limits=limits,
            workers=2,
            enable_bound_pruning=False,
        )
        try:
            self.assertEqual(
                pruned.solve_state(1, state), unpruned.solve_state(1, state)
            )
            self.assertEqual(
                pruned.policy_for(1, state, "sunny"),
                unpruned.policy_for(1, state, "sunny"),
            )
            self.assertGreater(pruned.stats.best_response_profiles_pruned, 0)
            self.assertLess(pruned.stats.joint_profiles, unpruned.stats.joint_profiles)
            self.assertTrue(pruned.pruning_certificates)
            self.assertTrue(
                all(
                    certificate.safety_margin > 0
                    for certificate in pruned.pruning_certificates
                )
            )
        finally:
            pruned.close()
            unpruned.close()

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

    def test_three_player_chunked_initial_stage_matches_dense(self) -> None:
        cfg = Q3Config(
            name="three-player-initial",
            n_players=3,
            deadline=1,
            mine_income=0,
            p_weather={"sunny": 1.0},
            weight_limit=4,
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
        dense = ExactQ3Solver(cfg, limits=SolverLimits(128, 100, 10_000, 8, 10_000))
        chunked = ExactQ3Solver(
            cfg,
            limits=SolverLimits(128, 1, 10_000, 2, 10_000),
            workers=2,
        )
        try:
            dense_result = dense.solve_initial_purchases()
            chunked_result = chunked.solve_initial_purchases()
            self.assertEqual(chunked_result, dense_result)
            self.assertGreaterEqual(chunked.stats.action_enumerations, 1)
            self.assertGreaterEqual(chunked.stats.action_set_reuses, 2)
            self.assertGreaterEqual(chunked.stats.chunked_stage_games, 1)
        finally:
            dense.close()
            chunked.close()

    def test_chunked_parallel_backend_matches_dense_backend(self) -> None:
        cfg = tiny_level6()
        state = tuple(active(cfg, 2, water=6, food=6) for _ in range(3))
        dense = ExactQ3Solver(
            cfg,
            limits=SolverLimits(128, 100_000, 20_000, 64, 100_000),
        )
        chunked = ExactQ3Solver(
            cfg,
            limits=SolverLimits(128, 1, 20_000, 2, 100_000),
            workers=2,
        )
        try:
            dense_value = dense.solve_state(1, state)
            chunked_value = chunked.solve_state(1, state)
            self.assertEqual(chunked_value, dense_value)
            self.assertEqual(
                chunked.policy_for(1, state, "sunny"),
                dense.policy_for(1, state, "sunny"),
            )
            self.assertEqual(chunked.stats.chunked_stage_games, 1)
            self.assertGreater(chunked.stats.opponent_pairs_completed, 0)
            self.assertGreater(dense.stats.duplicate_successors, 0)
        finally:
            dense.close()
            chunked.close()

    def test_checkpoint_round_trip_preserves_value_and_policy(self) -> None:
        cfg = tiny_level6()
        state = tuple(active(cfg, 2, water=6, food=6) for _ in range(3))
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "q3.chk"
            solver = ExactQ3Solver(cfg, checkpoint_path=checkpoint)
            try:
                expected = solver.solve_state(1, state)
                expected_policy = solver.policy_for(1, state, "sunny")
                solver.save_checkpoint()
            finally:
                solver.close()

            restored = ExactQ3Solver(cfg, checkpoint_path=checkpoint)
            try:
                restored.load_checkpoint()
                self.assertEqual(restored.solve_state(1, state), expected)
                self.assertEqual(
                    restored.policy_for(1, state, "sunny"), expected_policy
                )
                self.assertEqual(restored.stats.checkpoint_loads, 1)
            finally:
                restored.close()

    def test_pruning_certificates_survive_checkpoint(self) -> None:
        cfg = tiny_level6()
        state = tuple(active(cfg, 2, water=6, food=6) for _ in range(3))
        limits = SolverLimits(128, 1, 20_000, 1, 100_000)
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "q3-pruning.chk"
            solver = ExactQ3Solver(
                cfg,
                limits=limits,
                checkpoint_path=checkpoint,
                record_pruning_certificates=True,
            )
            try:
                solver.solve_state(1, state)
                solver.save_checkpoint()
                expected = tuple(solver.pruning_certificates)
                self.assertTrue(expected)
            finally:
                solver.close()

            restored = ExactQ3Solver(
                cfg,
                limits=limits,
                checkpoint_path=checkpoint,
                record_pruning_certificates=True,
            )
            try:
                restored.load_checkpoint()
                self.assertEqual(tuple(restored.pruning_certificates), expected)
                self.assertEqual(
                    restored.stats.pruning_certificates_recorded, len(expected)
                )
            finally:
                restored.close()

    def test_chunked_stage_progress_resumes_after_resource_stop(self) -> None:
        cfg = tiny_level6()
        state = tuple(active(cfg, 2, water=6, food=6) for _ in range(3))
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "q3-progress.chk"
            limited = ExactQ3Solver(
                cfg,
                limits=SolverLimits(128, 1, 20_000, 2, 80),
                checkpoint_path=checkpoint,
                checkpoint_every_pairs=1,
            )
            try:
                with self.assertRaises(ResourceLimitExceeded):
                    limited.solve_state(1, state)
            finally:
                limited.close()
            self.assertTrue(checkpoint.exists())

            resumed = ExactQ3Solver(
                cfg,
                limits=SolverLimits(128, 1, 20_000, 2, 100_000),
                checkpoint_path=checkpoint,
            )
            dense = ExactQ3Solver(cfg)
            try:
                resumed.load_checkpoint()
                self.assertEqual(
                    resumed.solve_state(1, state), dense.solve_state(1, state)
                )
                self.assertGreater(resumed.stats.checkpoint_loads, 0)
            finally:
                resumed.close()
                dense.close()


if __name__ == "__main__":
    unittest.main()
