"""
test_svm_agent.py

Test unitari per SVMAgent, SVMOptimizer e SVMWalkForwardTrainer.
Verifica:
1. Inizializzazione e conformità interfaccia BaseAgent.
2. Fit su dati di training (In-Sample <= max_year).
3. Predict con output discreto in {-1, 0, 1} e assenza di NaN.
4. Predict with confidence in [0.50, 1.00] per segnali attivi e 0.0 per Flat.
5. Salvataggio e caricamento corretto dei parametri JSON e dei modelli serializzati (.joblib).
6. Ottimizzazione su griglia tramite SVMOptimizer.
7. Workflow Walk-Forward con SVMWalkForwardTrainer.
"""

import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.agents.SVM_agent import SVMAgent, SVMOptimizer, SVMWalkForwardTrainer


class TestSVMAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        dates = pd.date_range("2010-01-01", "2022-12-31", freq="W-FRI")
        n = len(dates)

        # Generazione serie sintetica con trend e feature macro realistiche
        trend = np.linspace(100, 250, n)
        noise = np.random.normal(0, 3, n)
        close = trend + np.cumsum(noise)

        cls.df = pd.DataFrame(
            {
                "SPY_close": close,
                "SPY_return_1w": pd.Series(close).pct_change().fillna(0.0).values,
                "curve_10y_2y": np.random.normal(1.2, 0.5, n),
                "curve_10y_2y_change_1w": np.random.normal(0.0, 0.05, n),
                "curve_10y_3m": np.random.normal(1.5, 0.4, n),
                "curve_10y_3m_change_1w": np.random.normal(0.0, 0.05, n),
                "treasury_10y_change_1w": np.random.normal(0.0, 0.08, n),
                "treasury_2y_change_1w": np.random.normal(0.0, 0.08, n),
                "rates__policy_rate": np.random.uniform(0.5, 3.0, n),
                "cpi_yoy": np.random.normal(2.1, 0.3, n),
                "vix_level": np.random.uniform(12.0, 28.0, n),
                "breakeven_5y_change_1w": np.random.normal(0.0, 0.04, n),
                "rate_volatility_13w": np.random.uniform(0.1, 0.8, n),
            },
            index=dates,
        )

    def test_initialization(self):
        agent = SVMAgent(target_asset="SPY", C=2.0, gamma="scale", kernel="rbf", prob_threshold=0.53)
        self.assertEqual(agent.target_asset, "SPY")
        self.assertEqual(agent.C, 2.0)
        self.assertEqual(agent.gamma, "scale")
        self.assertEqual(agent.kernel, "rbf")
        self.assertEqual(agent.prob_threshold, 0.53)
        self.assertIsNone(agent.model)

    def test_fit_and_predict(self):
        agent = SVMAgent(target_asset="SPY", C=1.0, gamma="scale", prob_threshold=0.52)
        agent.fit(self.df, max_year=2018)

        self.assertIsNotNone(agent.model)
        self.assertIsNotNone(agent.scaler)
        self.assertGreater(agent.support_vector_stats["n_support_vectors"], 0)

        # Previsione discreta
        signals = agent.predict(self.df)
        self.assertEqual(len(signals), len(self.df))
        self.assertFalse(signals.isna().any())
        self.assertTrue(set(signals.unique()).issubset({-1, 0, 1}))

        # Previsione continua con confidenza
        sigs, conf = agent.predict_with_confidence(self.df)
        self.assertEqual(len(sigs), len(conf))
        self.assertTrue((conf >= 0.0).all())
        self.assertTrue((conf <= 1.0).all())

        # Segnali Flat (0) devono avere confidenza 0.0
        flat_mask = sigs == 0
        if flat_mask.any():
            self.assertTrue((conf[flat_mask] == 0.0).all())

        # Segnali attivi (!= 0) devono avere confidenza >= 0.50
        active_mask = sigs != 0
        if active_mask.any():
            self.assertTrue((conf[active_mask] >= 0.50).all())

    def test_save_and_load(self):
        agent = SVMAgent(target_asset="SPY", C=1.5, gamma=0.05, prob_threshold=0.54)
        agent.fit(self.df, max_year=2018)
        pred_before = agent.predict(self.df)

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "svm_test_params.json"
            agent.save_params(json_path)

            self.assertTrue(json_path.exists())
            joblib_path = json_path.with_suffix(".joblib")
            self.assertTrue(joblib_path.exists())

            # Nuovo agente che carica i parametri
            loaded_agent = SVMAgent(target_asset="SPY")
            loaded_agent.load_params(json_path)

            self.assertEqual(loaded_agent.C, 1.5)
            self.assertEqual(loaded_agent.gamma, 0.05)
            self.assertEqual(loaded_agent.prob_threshold, 0.54)

            pred_after = loaded_agent.predict(self.df)
            pd.testing.assert_series_equal(pred_before, pred_after)

    def test_optimizer(self):
        train_df = self.df[self.df.index.year <= 2017]
        val_df = self.df[self.df.index.year == 2018]

        opt = SVMOptimizer(
            target_asset="SPY",
            C_candidates=[0.5, 1.0],
            gamma_candidates=["scale"],
            threshold_candidates=[0.51, 0.52],
            objective_metric="sharpe_mdd",
        )
        res = opt.fit(train_df, val_df)
        self.assertIn("C", res)
        self.assertIn("gamma", res)
        self.assertIn("prob_threshold", res)
        self.assertIn("train_score", res)
        self.assertIsNotNone(opt.best_agent)

    def test_walk_forward_trainer(self):
        opt = SVMOptimizer(
            target_asset="SPY",
            C_candidates=[0.5, 1.0],
            gamma_candidates=["scale"],
            threshold_candidates=[0.51],
        )
        trainer = SVMWalkForwardTrainer(
            optimizer=opt,
            train_window_years=3,
            target_asset="SPY",
        )
        res = trainer.run(self.df, max_year=2016)
        self.assertIn("C", res)
        self.assertIn("gamma", res)
        self.assertIn("prob_threshold", res)
        self.assertIsNotNone(trainer.final_agent)


if __name__ == "__main__":
    unittest.main()
