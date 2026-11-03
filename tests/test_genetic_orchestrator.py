"""
test_genetic_orchestrator.py

Test unitari per i componenti del Meta-Decisore di Trading su Timeframe Settimanale
basato su Algoritmo Genetico (GA) a codifica reale vettorizzato.
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
        """Verifica che i pesi non normalizzati vengano sempre normalizzati a somma 1."""
        raw_w = np.array([10.0, 20.0, 30.0, 40.0])
        agg = WeeklySignalAggregator(weights=raw_w, alphas=self.alphas)
        self.assertAlmostEqual(np.sum(agg.weights), 1.0, places=6)
        np.testing.assert_allclose(agg.weights, [0.1, 0.2, 0.3, 0.4])

    def test_2d_aggregation_shapes_and_bounds(self):
        """Verifica le dimensioni e i vincoli di range di Z, S_out, C_out e P."""
        Z, S_out, C_out, P = self.aggregator.aggregate_2d(self.signals, self.confidences)

        self.assertEqual(len(Z), self.T)
        self.assertEqual(len(S_out), self.T)
        self.assertEqual(len(C_out), self.T)
        self.assertEqual(len(P), self.T)

        self.assertTrue(np.all(Z >= -1.0) and np.all(Z <= 1.0))
        self.assertTrue(set(np.unique(S_out)).issubset({-1, 0, 1}))
        self.assertTrue(np.all(C_out >= 0.0) and np.all(C_out <= 1.0))
        self.assertTrue(np.all(P >= -1.0) and np.all(P <= 1.0))

        # Position Sizing: P = S_out * C_out (quando S_out == 0, P deve essere 0)
        np.testing.assert_allclose(P, S_out * C_out)
        self.assertTrue(np.all(P[S_out == 0] == 0.0))

    def test_flat_agent_braking_effect(self):
        """
        Verifica l'effetto freno degli agenti Flat (s_i = 0):
        Se un agente ha segnale flat (0) ma confidenza elevata,
        il denominatore aumenta e |Z_t| si riduce, agendo da filtro attivo del rischio.
        """
        # Caso A: Agente 1 Long, Agente 2 Flat con confidenza 0.0
        # Caso B: Agente 1 Long, Agente 2 Flat con confidenza 1.0
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
        # Con il freno, il denominatore include il peso dell'agente flat
        self.assertAlmostEqual(Z_with_brake[0], (0.5 * 0.8) / (0.5 * 0.8 + 0.5 * 1.0), places=4)

    def test_3d_population_broadcasting_equivalence(self):
        """Verifica che il calcolo vettorizzato 3D per una popolazione coincida con aggregate_2d."""
        chrom = np.concatenate([self.weights, self.alphas, [self.theta_long, self.theta_short]])
        pop = np.vstack([chrom, chrom])  # M=2 individui identici

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
        """Verifica il calcolo esatto del turnover e la detrazione della fee."""
        P = np.array([0.5, 0.5, -0.5, 0.0])
        S = np.array([1, 1, -1, 0])
        r = np.array([0.02, -0.01, 0.03, 0.01])
        w = np.array([0.5, 0.5])

        # Delta P:
        # t=0: |0.5 - 0| = 0.5 -> fee = 0.0010 * 0.5 = 0.0005 -> R_net[0] = 0.5*0.02 - 0.0005 = 0.0095
        # t=1: |0.5 - 0.5| = 0.0 -> fee = 0.0 -> R_net[1] = 0.5*-0.01 = -0.0050
        # t=2: |-0.5 - 0.5| = 1.0 -> fee = 0.0010 -> R_net[2] = -0.5*0.03 - 0.0010 = -0.0160
        # t=3: |0 - (-0.5)| = 0.5 -> fee = 0.0005 -> R_net[3] = 0.0 - 0.0005 = -0.0005
        metrics = self.evaluator.evaluate_single(P, S, r, w)
        self.assertAlmostEqual(metrics["avg_turnover"], (0.5 + 0.0 + 1.0 + 0.5) / 4.0, places=4)

    def test_activity_constraint_penalty(self):
        """Verifica che un'attività inferiore al minimo (15%) o pochi trade riceva -10.0 di fitness."""
        T = 100
        # Posizione flat per quasi tutta la serie (solo 5 settimane attive = 5%)
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
        """Verifica che il GA esegua l'ottimizzazione e migliori la fitness rispetto alla partenza."""
        T, N = 100, 3
        np.random.seed(42)
        S = np.random.choice([-1.0, 0.0, 1.0], size=(T, N))
        C = np.random.uniform(0.3, 0.9, size=(T, N))
        # Rendimenti positivi correlati con la colonna 0
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
        """Verifica l'esecuzione corretta e senza salti temporali della Walk-Forward."""
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
