"""Regression tests for genetic optimizer fitness and weekly causality (II.6).

Place this file in the repository's tests/ directory, then run:
    python3 -m unittest discover -s tests -p 'test_ga_fitness_II6.py' -v

These tests cover timing, costs, drawdown, vectorized/individual consistency,
walk-forward boundaries, and objective propagation.
"""

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.agents.base_agent import compute_objective_score
from src.agents.orchestrator_agent import (
    GeneticMetaOptimizer,
    OrchestratorOptimizer,
    PortfolioFitnessEvaluator,
    WalkForwardEngine,
)


class TestCausalFitness(unittest.TestCase):
    def test_no_reward_from_same_week_return(self):
        evaluator = PortfolioFitnessEvaluator(
            transaction_fee=0.0, min_active_ratio=0.0, min_trades=0,
        )
        positions = np.ones(6)
        signals = np.ones(6, dtype=int)
        returns = np.array([0.25, 0.0, 0.0, 0.0, 0.0, 0.0])

        result = evaluator.evaluate_single(positions, signals, returns, np.array([1.0]))
        self.assertAlmostEqual(result["cumulative_return"], 0.0, places=8)
        self.assertAlmostEqual(result["sharpe_ratio"], 0.0, places=8)

    def test_turnover_costs_and_activity_use_held_position(self):
        evaluator = PortfolioFitnessEvaluator(
            transaction_fee=0.01, min_active_ratio=0.0, min_trades=0,
        )
        positions = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
        signals = positions.astype(int)
        result = evaluator.evaluate_single(
            positions, signals, np.zeros(len(positions)), np.array([1.0])
        )

        # Executed positions: [0, 1, 1, -1, -1]. Turnover: [0, 1, 0, 2, 0].
        self.assertAlmostEqual(result["avg_turnover"], 3.0 / 5.0, places=4)
        self.assertAlmostEqual(result["cumulative_return"], (1 - 0.01) * (1 - 0.02) - 1, places=4)
        self.assertEqual(result["trades"], 2)
        self.assertAlmostEqual(result["active_ratio"], 0.8, places=4)

    def test_drawdown_includes_initial_capital(self):
        evaluator = PortfolioFitnessEvaluator(
            transaction_fee=0.0, min_active_ratio=0.0, min_trades=0,
        )
        result = evaluator.evaluate_single(
            positions=np.ones(4),
            signals=np.ones(4, dtype=int),
            returns=np.array([-0.20, 0.0, 0.0, 0.0]),
            weights=np.array([1.0]),
            initial_position=1.0,
            previous_position=1.0,
            initial_signal=1,
        )
        self.assertAlmostEqual(result["max_drawdown"], 0.20, places=4)
        self.assertAlmostEqual(result["cumulative_return"], -0.20, places=4)

    def test_sharpe_mdd_objective_is_actually_used(self):
        weights = np.array([0.7, 0.3])
        evaluator = PortfolioFitnessEvaluator(
            transaction_fee=0.0,
            lambda_concentration=0.1,
            min_active_ratio=0.0,
            min_trades=0,
            objective_metric="sharpe_mdd",
        )
        result = evaluator.evaluate_single(
            positions=np.ones(6),
            signals=np.ones(6, dtype=int),
            returns=np.array([0.0, 0.10, -0.04, 0.05, -0.01, 0.02]),
            weights=weights,
        )
        objective = compute_objective_score(result, metric_type="sharpe_mdd")
        expected = objective - 0.1 * np.sum(weights ** 2)
        self.assertAlmostEqual(result["fitness"], expected, delta=0.001)
        self.assertGreater(result["max_drawdown"], 0.0)

    def test_population_matches_single_for_each_chromosome(self):
        positions = np.array([
            [0.7, -0.5] * 10,
            [0.6] * 20,
            [0.8] * 5 + [-0.8] * 5 + [0.8] * 5 + [-0.8] * 5,
        ], dtype=float)
        signals = np.sign(positions).astype(int)
        returns = np.array([0.02, -0.03, 0.01, 0.04, -0.02] * 4)
        weights = np.array([[0.7, 0.3], [0.2, 0.8], [0.5, 0.5]])
        chromosomes = np.column_stack([
            weights, np.ones((3, 2)), np.full((3, 2), 0.25),
        ])
        evaluator = PortfolioFitnessEvaluator(
            transaction_fee=0.002,
            min_active_ratio=0.3,
            min_trades=2,
            lambda_concentration=0.05,
            objective_metric="sharpe_mdd",
        )

        population_scores = evaluator.evaluate_population(
            positions, signals, returns, chromosomes, n_agents=2,
        )
        for i in range(len(positions)):
            individual_score = evaluator.evaluate_single(
                positions[i], signals[i], returns, weights[i],
            )["fitness"]
            self.assertAlmostEqual(population_scores[i], individual_score, delta=0.00011)
        self.assertEqual(population_scores[1], evaluator.penalty_fitness)

    def test_walkforward_first_validation_return_uses_last_train_signal(self):
        dates = pd.date_range("2010-01-01", periods=5, freq="W-FRI")
        signal_values = [1, -1, 1, 1, -1]
        source = pd.DataFrame({
            "signal": signal_values,
            "confidence": np.ones(5),
        }, index=dates)
        returns = pd.Series([0.0, 0.0, 0.10, 0.02, 0.03], index=dates)
        evaluator = PortfolioFitnessEvaluator(
            transaction_fee=0.001,
            min_active_ratio=0.0,
            min_trades=0,
        )
        engine = WalkForwardEngine(
            n_agents=1,
            train_window_weeks=2,
            test_window_weeks=3,
            step_weeks=3,
            population_size=20,
            n_generations=5,
            evaluator=evaluator,
            verbose=False,
        )
        chosen = np.array([1.0, 1.0, 0.25, 0.25])
        train_metrics = {
            "sharpe_ratio": 0.0, "max_drawdown": 0.0, "active_ratio": 0.5,
        }
        with patch.object(
            GeneticMetaOptimizer, "optimize",
            return_value=(chosen, train_metrics, []),
        ):
            result = engine.run({"SyntheticAgent": source}, returns)

        oos = result["oos_series"]
        self.assertEqual(list(oos.index), list(dates[2:]))
        # The final training signal is -1, so the first test week's held position is -1.
        self.assertAlmostEqual(oos["position"].iloc[0], -1.0, places=8)
        self.assertAlmostEqual(oos["target_position"].iloc[0], 1.0, places=8)
        self.assertAlmostEqual(oos["net_return"].iloc[0], -0.101, places=8)
        # Following week holds previous test signal (+1), and pays turnover of 2.
        self.assertAlmostEqual(oos["position"].iloc[1], 1.0, places=8)
        self.assertAlmostEqual(oos["net_return"].iloc[1], 0.02 - 0.002, places=8)

    def test_orchestrator_forwards_requested_objective(self):
        dates = pd.date_range("2014-01-03", periods=12, freq="W-FRI")
        signals = {
            "A": pd.DataFrame({
                "signal": [1, -1] * 6, "confidence": [0.8] * 12,
            }, index=dates),
            "B": pd.DataFrame({
                "signal": [-1, 1] * 6, "confidence": [0.7] * 12,
            }, index=dates),
        }
        returns = pd.Series([0.01, -0.01] * 6, index=dates)
        fake_metrics = {
            "fitness": 1.0, "sharpe_ratio": 0.9, "max_drawdown": 0.1,
            "cumulative_return": 0.1, "annualized_return": 0.2,
            "profit_factor": 1.2, "trades": 8,
        }
        decoded = {
            "weights": np.array([0.5, 0.5]),
            "alphas": np.array([1.0, 1.0]),
            "theta_long": 0.25,
            "theta_short": 0.25,
        }

        with (
            patch(
                "src.agents.orchestrator_agent.PortfolioFitnessEvaluator",
                wraps=PortfolioFitnessEvaluator,
            ) as evaluator_cls,
            patch("src.agents.orchestrator_agent.GeneticMetaOptimizer") as ga_cls,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            ga = ga_cls.return_value
            ga.optimize.return_value = (
                np.array([0.5, 0.5, 1.0, 1.0, 0.25, 0.25]),
                fake_metrics,
                [{"generation": 1, "best_fitness": 1.0}],
            )
            ga.decode_chromosome.return_value = decoded
            optimizer = OrchestratorOptimizer(
                max_train_year=2016, objective_metric="calmar",
            )
            optimizer.optimize(signals, returns, asset_name="TEST", method="ga")
            self.assertEqual(evaluator_cls.call_args.kwargs["objective_metric"], "calmar")


if __name__ == "__main__":
    unittest.main()
