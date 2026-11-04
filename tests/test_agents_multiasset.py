"""
test_agents_multiasset.py

Suite di unit test per la verifica del caricamento dati su BaseAgent
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

from src.agents.base_agent import (
    BaseAgent,
    DEFAULT_ASSETS,
    DEFAULT_TRAIN_START,
    calculate_performance_metrics,
    compute_objective_score,
)
from src.agents.EMA_EMW_agent import SMAEMAAgent
from src.agents.ZScore_agent import ZScoreAgent
from src.agents.macro_yield_curve_agent import MacroYieldCurveAgent
from src.agents.random_forest_agent import RandomForestAgent
from src.agents.orchestrator_agent import OrchestratorAgent, OrchestratorOptimizer


class _CasoDatiMultiAssetSintetici(unittest.TestCase):
    """Isola i test da CSV generati e dalle serie di mercato reali."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cartella = TemporaryDirectory()
        cls.addClassCleanup(cartella.cleanup)
        root = Path(cartella.name)

        processed = root / "data" / "processed"
        macro_dir = root / "data" / "raw" / "macro"
        processed.mkdir(parents=True)
        macro_dir.mkdir(parents=True)
        (root / "src").mkdir()
        (root / "config.yaml").write_text("{}\n", encoding="utf-8")

        # Serie interamente artificiali e deterministiche, solo 2006-2016.
        date = pd.date_range("2006-01-06", "2016-12-30", freq="W-FRI")
        t = np.arange(len(date), dtype=float)
        prezzi = pd.DataFrame(index=date)
        prezzi.index.name = "Date"
        for asset, base, fase in (
            ("SPY", 100.0, 0.0),
            ("IEF", 82.0, 0.7),
            ("GLD", 92.0, 1.3),
            ("SHY", 71.0, 2.1),
        ):
            close = base + 0.08 * t + 2.7 * np.sin(t / 3.1 + fase) + 1.1 * np.sin(t / 11.0 + fase)
            open_ = close * (1.0 + 0.001 * np.cos(t / 4.0 + fase))
            prezzi[f"{asset}_open"] = open_
            prezzi[f"{asset}_high"] = np.maximum(open_, close) * 1.003
            prezzi[f"{asset}_low"] = np.minimum(open_, close) * 0.997
            prezzi[f"{asset}_close"] = close
            prezzi[f"{asset}_volume"] = 1000000.0 + 2500.0 * t
            prezzi[f"{asset}_return_1w"] = pd.Series(close, index=date).pct_change().fillna(0.0).to_numpy()
        prezzi.to_csv(processed / "master_weekly.csv")

        date_macro = pd.date_range("2005-12-01", "2016-12-30", freq="B")
        m = np.arange(len(date_macro), dtype=float)
        serie_macro = {
            "DGS10": 2.3 + 0.35 * np.sin(m / 37.0),
            "DGS2": 1.5 + 0.22 * np.cos(m / 51.0),
            "DGS3MO": 0.85 + 0.12 * np.cos(m / 65.0),
            "DGS30": 3.2 + 0.29 * np.sin(m / 72.0),
            "VIXCLS": 19.0 + 5.0 * np.sin(m / 43.0),
            "T10YIE": 2.0 + 0.2 * np.sin(m / 63.0),
            "T5YIE": 1.8 + 0.15 * np.cos(m / 49.0),
        }
        for nome, valori in serie_macro.items():
            s = pd.DataFrame({nome: valori}, index=date_macro)
            # L'indice non nominato mantiene il contratto merge_asof di BaseAgent.
            s.index.name = None
            s.to_csv(macro_dir / f"{nome}.csv")

        # Restituisce il root temporaneo anche ai metodi di istanza degli agenti.
        root_base = patch("src.agents.base_agent.get_project_root", return_value=root)
        root_base.start()
        cls.addClassCleanup(root_base.stop)
        root_orchestrator = patch("src.agents.orchestrator_agent.get_project_root", return_value=root)
        root_orchestrator.start()
        cls.addClassCleanup(root_orchestrator.stop)


class TestBaseAgentDataLoading(_CasoDatiMultiAssetSintetici):
    """Test per le funzionalità di caricamento e normalizzazione dati in BaseAgent."""

    def test_load_market_data_default(self):
        df = BaseAgent.load_market_data(frequency="weekly")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertTrue(df.index.is_monotonic_increasing)

        # Verifica presenza di tutti i 4 asset
        for asset in DEFAULT_ASSETS:
            self.assertIn(f"{asset}_close", df.columns)
            self.assertIn(f"{asset}_open", df.columns)

    def test_load_market_data_filtered_assets_and_dates(self):
        df = BaseAgent.load_market_data(
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
        macro_df = BaseAgent.load_macro_data()
        self.assertIsInstance(macro_df, pd.DataFrame)
        expected_macro = ["curve_10y_2y", "treasury_10y_change_1w", "vix_level"]
        for col in expected_macro:
            self.assertIn(col, macro_df.columns)

    def test_load_complete_dataset(self):
        complete = BaseAgent.load_complete_dataset(include_macro=True)
        # Contiene sia prezzi che macro
        for asset in DEFAULT_ASSETS:
            self.assertIn(f"{asset}_close", complete.columns)
        self.assertIn("curve_10y_2y", complete.columns)

    def test_prepare_asset_data(self):
        complete = BaseAgent.load_complete_dataset()
        spy_view = BaseAgent.prepare_asset_data(complete, asset="SPY")
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


class TestIndividualAgentsMultiAsset(_CasoDatiMultiAssetSintetici):
    """Test di inferenza e compatibilità multi-asset per SMA, ZScore e Macro."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.df = BaseAgent.load_complete_dataset()

    def test_sma_ema_agent_on_all_four_assets(self):
        for asset in DEFAULT_ASSETS:
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
        for asset in DEFAULT_ASSETS:
            agent = ZScoreAgent(window=20, threshold=1.5, target_asset=asset)
            signals, conf = agent.predict_with_confidence(self.df)

            self.assertEqual(len(signals), len(self.df))
            self.assertEqual(len(conf), len(self.df))
            self.assertTrue(set(signals.unique()).issubset({-1, 0, 1}))
            self.assertTrue((conf >= 0.0).all() and (conf <= 1.0).all())

    def test_macro_yield_curve_agent_multi_asset(self):
        agent = MacroYieldCurveAgent()
        multi_sig = agent.predict_multi_asset(self.df)

        expected_cols = [f"{asset}_signal" for asset in DEFAULT_ASSETS]
        for col in expected_cols:
            self.assertIn(col, multi_sig.columns)
            self.assertTrue(set(multi_sig[col].unique()).issubset({-1, 0, 1}))

    def test_random_forest_agent_multi_asset(self):
        for asset in ["SPY", "IEF"]:
            rf = RandomForestAgent(target_asset=asset, n_estimators=30, max_depth=2, min_samples_leaf=20)
            rf.fit(self.df, max_year=2016)
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
        for asset in DEFAULT_ASSETS:
            orch = OrchestratorAgent(target_asset=asset, include_rf=False, include_svm=False)
            sig, conf = orch.predict_with_confidence(self.df)

            self.assertEqual(len(sig), len(self.df))
            self.assertEqual(len(conf), len(self.df))
            self.assertTrue(set(sig.unique()).issubset({-1, 0, 1}))
            self.assertTrue((conf >= 0.0).all() and (conf <= 1.0).all())

    def test_orchestrator_predict_multi_asset(self):
        orch = OrchestratorAgent(include_rf=False, include_svm=False)
        multi = orch.predict_multi_asset(self.df)

        for asset in DEFAULT_ASSETS:
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
