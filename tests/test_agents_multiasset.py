"""
test_agents_multiasset.py

Suite di unit test per la verifica del caricamento dati degli agenti (src/data_loader.py)
e del corretto funzionamento dei singoli agenti sul paniere a 4 asset (SPY, IEF, GLD, SHY).

Esecuzione:
    python -m unittest tests/test_agents_multiasset.py
"""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.agents.base_agent import BaseAgent
from src.data_loader import (
    load_agent_dataset,
    load_agent_macro_data,
    load_agent_market_data,
    prepare_asset_data,
)
from src.utils import ASSETS
from src.agents.EMA_EMW_agent import SMAEMAAgent
from src.agents.ZScore_agent import ZScoreAgent
from src.agents.macro_yield_curve_agent import MacroYieldCurveAgent
from src.agents.random_forest_agent import RandomForestAgent
from src.agents.orchestrator_agent import OrchestratorAgent, OrchestratorOptimizer


class TestAgentDataLoading(unittest.TestCase):
    """Test per le funzionalità di caricamento e normalizzazione dati in BaseAgent."""

    def test_load_market_data_default(self):
        df = load_agent_market_data(frequency="weekly")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertTrue(df.index.is_monotonic_increasing)

        # Verifica presenza di tutti i 4 asset
        for asset in ASSETS:
            self.assertIn(f"{asset}_close", df.columns)
            self.assertIn(f"{asset}_open", df.columns)

    def test_load_market_data_filtered_assets_and_dates(self):
        df = load_agent_market_data(
            assets=["SPY", "GLD"],
            start_date="2006-01-01",
            end_date="2010-12-31",
        )
        self.assertGreaterEqual(df.index.min(), pd.Timestamp("2006-01-01"))
        self.assertLessEqual(df.index.max(), pd.Timestamp("2010-12-31"))

        # Solo SPY e GLD
        for c in df.columns:
            self.assertTrue(c.startswith("SPY_") or c.startswith("GLD_"))
        self.assertIn("SPY_close", df.columns)
        self.assertIn("GLD_close", df.columns)
        self.assertNotIn("IEF_close", df.columns)

    def test_load_macro_data(self):
        macro_df = load_agent_macro_data()
        self.assertIsInstance(macro_df, pd.DataFrame)
        expected_macro = ["curve_10y_2y", "treasury_10y_change_1w", "vix_level"]
        for col in expected_macro:
            self.assertIn(col, macro_df.columns)

    def test_load_complete_dataset(self):
        complete = load_agent_dataset(include_macro=True)
        # Contiene sia prezzi che macro
        for asset in ASSETS:
            self.assertIn(f"{asset}_close", complete.columns)
        self.assertIn("curve_10y_2y", complete.columns)

    def test_prepare_asset_data(self):
        complete = load_agent_dataset()
        spy_view = prepare_asset_data(complete, asset="SPY")
        self.assertIn("close", spy_view.columns)
        self.assertIn("open", spy_view.columns)
        np.testing.assert_allclose(spy_view["close"].values, complete["SPY_close"].values)

    def test_resolve_price_col(self):
        agent = SMAEMAAgent(target_asset="SPY")
        df_multi = pd.DataFrame({"SPY_close": [10.0, 11.0], "IEF_close": [20.0, 21.0]})
        col = agent.resolve_price_col(df_multi)
        self.assertEqual(col, "SPY_close")

        df_generic = pd.DataFrame({"close": [10.0, 11.0]})
        col_gen = agent.resolve_price_col(df_generic)
        self.assertEqual(col_gen, "close")

        with self.assertRaises(KeyError):
            agent.resolve_price_col(pd.DataFrame({"volume": [1, 2]}))


class TestIndividualAgentsMultiAsset(unittest.TestCase):
    """Test di inferenza e compatibilità multi-asset per SMA, ZScore e Macro."""

    @classmethod
    def setUpClass(cls):
        cls.df = load_agent_dataset()

    def test_sma_ema_agent_on_all_four_assets(self):
        for asset in ASSETS:
            agent = SMAEMAAgent(sma_window=20, ema_window=10, target_asset=asset)
            signals, conf = agent.predict_with_confidence(self.df)

            self.assertEqual(len(signals), len(self.df))
            self.assertEqual(len(conf), len(self.df))
            self.assertTrue(set(signals.unique()).issubset({-1, 0, 1}))
            self.assertTrue((conf >= 0.0).all() and (conf <= 1.0).all())

            # Test caricamento dati diretto da istanza
            inst_df = agent.load_data()
            inst_signals = agent.predict(inst_df)
            pd.testing.assert_series_equal(signals, inst_signals)

    def test_zscore_agent_on_all_four_assets(self):
        for asset in ASSETS:
            agent = ZScoreAgent(window=20, threshold=1.5, target_asset=asset)
            signals, conf = agent.predict_with_confidence(self.df)

            self.assertEqual(len(signals), len(self.df))
            self.assertEqual(len(conf), len(self.df))
            self.assertTrue(set(signals.unique()).issubset({-1, 0, 1}))
            self.assertTrue((conf >= 0.0).all() and (conf <= 1.0).all())

    def test_macro_yield_curve_agent_multi_asset(self):
        agent = MacroYieldCurveAgent()
        multi_sig = agent.predict_multi_asset(self.df)

        expected_cols = [f"{asset}_signal" for asset in ASSETS]
        for col in expected_cols:
            self.assertIn(col, multi_sig.columns)
            self.assertTrue(set(multi_sig[col].unique()).issubset({-1, 0, 1}))

    def test_random_forest_agent_multi_asset(self):
        for asset in ["SPY", "IEF"]:
            rf = RandomForestAgent(target_asset=asset, n_estimators=30, max_depth=2, min_samples_leaf=20)
            rf.fit(self.df, max_year=2021)
            sig, conf = rf.predict_with_confidence(self.df)

            self.assertEqual(len(sig), len(self.df))
            self.assertEqual(len(conf), len(self.df))
            self.assertTrue(set(sig.unique()).issubset({-1, 0, 1}))
            self.assertTrue((conf >= 0.0).all() and (conf <= 1.0).all())


    def test_orchestrator_signals_pool_includes_rf(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_dir = root / "src" / "agents" / "config"
            cfg_dir.mkdir(parents=True)

            cfg = cfg_dir / "random_forest_params_spy.json"
            cfg.write_text("{}", encoding="utf-8")

            dummy_signals = pd.Series(0, index=self.df.index, dtype=int)
            dummy_conf = pd.Series(0.0, index=self.df.index)

            with (
                patch(
                    "src.agents.orchestrator_agent.get_project_root",
                    return_value=root,
                ),
                patch(
                    "src.agents.orchestrator_agent._validate_training_provenance"
                ) as validator,
                patch(
                    "src.agents.orchestrator_agent.RandomForestAgent.load_params"
                ),
                patch(
                    "src.agents.orchestrator_agent.RandomForestAgent.predict_with_confidence",
                    return_value=(dummy_signals, dummy_conf),
                ),
            ):
                orch = OrchestratorAgent(
                    target_asset="SPY",
                    include_rf=True,
                    include_svm=False,
                )
                pool = orch._build_signals_pool_from_df(self.df)

                validator.assert_called_once_with(cfg, root)

            self.assertIn("RandomForest_Agent", pool)

            for sig_df in pool.values():
                self.assertIn("signal", sig_df.columns)
                self.assertIn("confidence", sig_df.columns)
                self.assertEqual(len(sig_df), len(self.df))


    def test_orchestrator_agent_on_all_four_assets(self):
        for asset in ASSETS:
            orch = OrchestratorAgent(target_asset=asset, include_rf=False, include_svm=False)
            sig, conf = orch.predict_with_confidence(self.df)

            self.assertEqual(len(sig), len(self.df))
            self.assertEqual(len(conf), len(self.df))
            self.assertTrue(set(sig.unique()).issubset({-1, 0, 1}))
            self.assertTrue((conf >= 0.0).all() and (conf <= 1.0).all())

    def test_orchestrator_predict_multi_asset(self):
        orch = OrchestratorAgent(include_rf=False, include_svm=False)
        multi = orch.predict_multi_asset(self.df)

        for asset in ASSETS:
            self.assertIn(f"{asset}_signal", multi.columns)
            self.assertIn(f"{asset}_confidence", multi.columns)
            self.assertTrue(set(multi[f"{asset}_signal"].unique()).issubset({-1, 0, 1}))
            self.assertTrue((multi[f"{asset}_confidence"] >= 0.0).all() and (multi[f"{asset}_confidence"] <= 1.0).all())

    def test_orchestrator_combine_and_optimize_arbitrary_k_signals(self):
        """Verifica che l'Orchestratore supporti un numero qualsiasi K di segnali/agenti in input."""
        from src.agents.orchestrator_agent import OrchestratorOptimizer

        dates = self.df.index[:100]
        # Genera segnali per 5 agenti arbitrari
        custom_signals = {}
        for i in range(1, 6):
            s = pd.Series(np.random.choice([-1, 0, 1], size=len(dates)), index=dates)
            c = pd.Series(np.random.uniform(0.5, 1.0, size=len(dates)), index=dates)
            custom_signals[f"Agent_{i}"] = pd.DataFrame({"signal": s, "confidence": c}, index=dates)

        # 1. Test combine con K=5
        orch = OrchestratorAgent()
        res = orch.combine(custom_signals)
        self.assertEqual(len(res), len(dates))
        self.assertIn("signal", res.columns)
        self.assertIn("confidence", res.columns)
        self.assertTrue(set(res["signal"].unique()).issubset({-1, 0, 1}))

        # 2. Test ottimizzazione con K=5
        ret = pd.Series(np.random.normal(0.001, 0.02, size=len(dates)), index=dates)
        opt = OrchestratorOptimizer(max_train_year=2021)
        best_orch, df_summary = opt.optimize(custom_signals, ret, asset_name="TEST_K5")
        self.assertEqual(len(best_orch.weights), 5)
        self.assertAlmostEqual(sum(best_orch.weights.values()), 1.0, places=3)
        self.assertGreater(len(df_summary), 100)

    def test_orchestrator_register_agent(self):
        """Verifica la registrazione dinamica di nuovi agenti personalizzati."""
        orch = OrchestratorAgent(include_rf=False, include_svm=False)

        class DummyAgent(BaseAgent):
            def predict(self, df):
                return pd.Series(1, index=df.index)

        orch.register_agent("CustomSentimentAgent", DummyAgent(name="Dummy"))
        pool = orch._build_signals_pool_from_df(self.df)

        self.assertIn("CustomSentimentAgent", pool)
        self.assertEqual(len(pool["CustomSentimentAgent"]), len(self.df))
        self.assertEqual(int(pool["CustomSentimentAgent"]["signal"].iloc[0]), 1)


if __name__ == "__main__":
    unittest.main()
