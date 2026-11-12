"""Unit tests for the components of the weekly trading meta-decision model based
on a vectorised real-coded genetic algorithm (GA).
"""

import unittest
import numpy as np
import pandas as pd

from src.agents.orchestrator_agent import (
    WeeklySignalAggregator,
    PortfolioFitnessEvaluator,
    GeneticMetaOptimizer,
    WalkForwardEngine,
    OrchestratorAgent,
    generate_synthetic_market_data,
)


class TestWeeklySignalAggregator(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.T = 52
        self.N = 4
        self.signals = np.random.choice([-1.0, 0.0, 1.0], size=(self.T, self.N))
        self.confidences = np.random.uniform(0.1, 1.0, size=(self.T, self.N))
        self.weights = np.array([0.4, 0.3, 0.2, 0.1])
        self.alphas = np.array([1.0, 1.5, 0.8, 2.0])
        self.theta_long = 0.25
        self.theta_short = 0.30
        self.aggregator = WeeklySignalAggregator(
            weights=self.weights,
            alphas=self.alphas,
            theta_long=self.theta_long,
            theta_short=self.theta_short,
        )

    def test_weights_scale_invariance(self):
        """Check that unnormalised weights are always normalised to sum to 1."""
        raw_w = np.array([10.0, 20.0, 30.0, 40.0])
        agg = WeeklySignalAggregator(weights=raw_w, alphas=self.alphas)
        self.assertAlmostEqual(np.sum(agg.weights), 1.0, places=6)
        np.testing.assert_allclose(agg.weights, [0.1, 0.2, 0.3, 0.4])

    def test_2d_aggregation_shapes_and_bounds(self):
        """Check the shapes and value ranges of Z, S_out, C_out and P."""
        Z, S_out, C_out, P = self.aggregator.aggregate_2d(self.signals, self.confidences)

        self.assertEqual(len(Z), self.T)
        self.assertEqual(len(S_out), self.T)
        self.assertEqual(len(C_out), self.T)
        self.assertEqual(len(P), self.T)

        self.assertTrue(np.all(Z >= -1.0) and np.all(Z <= 1.0))
        self.assertTrue(set(np.unique(S_out)).issubset({-1, 0, 1}))
        self.assertTrue(np.all(C_out >= 0.0) and np.all(C_out <= 1.0))
        self.assertTrue(np.all(P >= -1.0) and np.all(P <= 1.0))

        # Position sizing: P = S_out * C_out (when S_out == 0, P must be 0).
        np.testing.assert_allclose(P, S_out * C_out)
        self.assertTrue(np.all(P[S_out == 0] == 0.0))

    def test_flat_agent_braking_effect(self):
        """Check the damping effect of flat agents (s_i = 0).

        If an agent has a flat signal (0) but high confidence, the denominator
        increases and |Z_t| decreases, acting as an active risk filter.
        """
        # Case A: agent 1 long, agent 2 flat with confidence 0.0.
        # Case B: agent 1 long, agent 2 flat with confidence 1.0.
        w = np.array([0.5, 0.5])
        a = np.array([1.0, 1.0])
        agg = WeeklySignalAggregator(weights=w, alphas=a, theta_long=0.20, theta_short=0.20)

        sig = np.array([[1.0, 0.0]])
        conf_no_brake = np.array([[0.8, 0.0]])
        conf_with_brake = np.array([[0.8, 1.0]])

        Z_no_brake, _, _, _ = agg.aggregate_2d(sig, conf_no_brake)
        Z_with_brake, _, _, _ = agg.aggregate_2d(sig, conf_with_brake)

        self.assertGreater(Z_no_brake[0], Z_with_brake[0])
        self.assertAlmostEqual(Z_no_brake[0], 1.0, places=4)
        # With damping, the denominator includes the weight of the flat agent.
        self.assertAlmostEqual(Z_with_brake[0], (0.5 * 0.8) / (0.5 * 0.8 + 0.5 * 1.0), places=4)

    def test_3d_population_broadcasting_equivalence(self):
        """Check that the 3D population computation matches aggregate_2d."""
        chrom = np.concatenate([self.weights, self.alphas, [self.theta_long, self.theta_short]])
        pop = np.vstack([chrom, chrom])  # M=2 identical individuals.

        P_pop, S_pop, C_pop = WeeklySignalAggregator.aggregate_3d_population(
            signals=self.signals,
            confidences=self.confidences,
            chromosomes=pop,
        )

        _, S_2d, C_2d, P_2d = self.aggregator.aggregate_2d(self.signals, self.confidences)

        self.assertEqual(P_pop.shape, (2, self.T))
        np.testing.assert_allclose(P_pop[0], P_2d)
        np.testing.assert_array_equal(S_pop[0], S_2d)
        np.testing.assert_allclose(C_pop[0], C_2d)


class TestPortfolioFitnessEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = PortfolioFitnessEvaluator(
            transaction_fee=0.0010,  # 10 bps
            annualization_factor=52,
            min_active_ratio=0.15,
            min_trades=5,
            lambda_concentration=0.05,
        )

    def test_turnover_and_net_return_calculation(self):
        """Check turnover and net returns under causal execution."""
        P = np.array([0.5, 0.5, -0.5, 0.0])
        S = np.array([1, 1, -1, 0])
        r = np.array([0.02, -0.01, 0.03, 0.01])
        w = np.array([0.5, 0.5])

        # Position P_t becomes effective from the following week.
        # Positions held:   [0.0, 0.5, 0.5, -0.5]
        # Turnover:           [0.0, 0.5, 0.0, 1.0]
        # Net returns:      [0.0, -0.0055, 0.015, -0.006]

        metrics = self.evaluator.evaluate_single(P, S, r, w)

        expected_turnover = (0.0 + 0.5 + 0.0 + 1.0) / 4.0
        expected_net_returns = np.array([
            0.0,
            0.5 * (-0.01) - 0.001 * 0.5,
            0.5 * 0.03,
            -0.5 * 0.01 - 0.001 * 1.0,
        ])

        expected_cumulative_return = np.prod(
            1.0 + expected_net_returns
        ) - 1.0

        self.assertAlmostEqual(
            metrics["avg_turnover"],
            expected_turnover,
            places=4,
        )

        self.assertAlmostEqual(
            metrics["cumulative_return"],
            expected_cumulative_return,
            delta=0.00005,
        )

    def test_activity_constraint_penalty(self):
        """Check that activity below the minimum (15%) or too few trades receives a fitness of -10.0.
        """
        T = 100
        # Flat position for almost the whole series (only 5 active weeks = 5%).
        P = np.zeros(T)
        P[:5] = 0.5
        S = np.zeros(T, dtype=int)
        S[:5] = 1
        r = np.random.normal(0.01, 0.02, size=T)
        w = np.array([0.5, 0.5])

        metrics = self.evaluator.evaluate_single(P, S, r, w)
        self.assertEqual(metrics["fitness"], -10.0)


class TestGeneticMetaOptimizer(unittest.TestCase):
    def test_genetic_optimization_convergence(self):
        """Check that the GA runs and improves fitness relative to the starting point.
        """
        T, N = 100, 3
        np.random.seed(42)
        S = np.random.choice([-1.0, 0.0, 1.0], size=(T, N))
        C = np.random.uniform(0.3, 0.9, size=(T, N))
        # Positive returns correlated with column 0.
        r = S[:, 0] * 0.02 + np.random.normal(0.0, 0.01, size=T)

        opt = GeneticMetaOptimizer(
            n_agents=N,
            population_size=40,
            n_generations=15,
            random_seed=42,
            verbose=False,
        )
        best_chrom, metrics, history = opt.optimize(S, C, r)

        self.assertEqual(len(best_chrom), 2 * N + 2)
        self.assertGreater(metrics["fitness"], -5.0)
        self.assertGreaterEqual(history[-1]["best_fitness"], history[0]["best_fitness"])

        decoded = opt.decode_chromosome(best_chrom)
        self.assertAlmostEqual(np.sum(decoded["weights"]), 1.0, places=4)
        self.assertTrue(0.05 <= decoded["theta_long"] <= 0.85)
        self.assertTrue(0.05 <= decoded["theta_short"] <= 0.85)


class TestWalkForwardEngine(unittest.TestCase):
    def test_walk_forward_pipeline(self):
        """Check that the walk-forward runs correctly without temporal gaps."""
        df_market, signals_pool = generate_synthetic_market_data(T=260, random_seed=42)
        engine = WalkForwardEngine(
            n_agents=len(signals_pool),
            train_window_weeks=104,
            test_window_weeks=26,
            step_weeks=26,
            population_size=30,
            n_generations=10,
            random_seed=42,
            verbose=False,
        )
        res = engine.run(signals_pool, df_market["return_1w"])

        self.assertIn("folds_df", res)
        self.assertIn("oos_series", res)
        self.assertIn("summary_metrics", res)

        df_folds = res["folds_df"]
        self.assertGreater(len(df_folds), 1)

        oos_series = res["oos_series"]
        self.assertEqual(len(oos_series), len(df_folds) * 26)
        self.assertIn("net_return", oos_series.columns)
        self.assertIn("equity_curve", oos_series.columns)


if __name__ == "__main__":
    unittest.main()
