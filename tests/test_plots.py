"""
test_plots.py

Test unitari per il modulo di visualizzazione src/plots.py.
Verifica:
1. Caricamento dei segnali per un asset tramite load_all_signals_for_asset.
2. Generazione del grafico di distribuzione segnali (plot_agent_signal_distributions).
3. Salvataggio su file e controllo dimensioni/integrita del plot generato.
4. Funzionamento sia con input dizionario esplicito che con caricamento automatico da cartella signals.
"""

import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.plots import load_all_signals_for_asset, plot_agent_signal_distributions, plot_all_assets_signal_distributions


class TestPlotsModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", "2021-12-31", freq="W-FRI")
        n = len(dates)

        # Genera segnali sintetici per 4 agenti fittizi
        cls.dummy_signals = {
            "Agent_Trend": pd.DataFrame({
                "signal": np.random.choice([-1, 0, 1], size=n, p=[0.2, 0.3, 0.5]),
                "confidence": np.random.uniform(0.5, 1.0, size=n),
            }, index=dates),
            "Agent_MeanRev": pd.DataFrame({
                "signal": np.random.choice([-1, 0, 1], size=n, p=[0.4, 0.2, 0.4]),
                "confidence": np.random.uniform(0.5, 1.0, size=n),
            }, index=dates),
            "Agent_SVM": pd.DataFrame({
                "signal": np.random.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3]),
                "confidence": np.random.uniform(0.5, 1.0, size=n),
            }, index=dates),
            "Orchestrator_Agent": pd.DataFrame({
                "signal": np.random.choice([-1, 0, 1], size=n, p=[0.25, 0.25, 0.50]),
                "confidence": np.random.uniform(0.6, 1.0, size=n),
            }, index=dates),
        }

    @classmethod
    def _scrivi_segnali_sintetici(cls, root: Path):
        """Scrive i CSV di test nel root temporaneo, senza usare data/signals locale."""
        signals_dir = root / "data" / "signals"
        signals_dir.mkdir(parents=True, exist_ok=True)
        files = {
            "signals_sma_ema_spy.csv": "Agent_Trend",
            "signals_zscore_spy.csv": "Agent_MeanRev",
            "signals_svm_spy.csv": "Agent_SVM",
            "signals_macro_spy.csv": "Agent_Trend",
            "signals_orchestrator_spy.csv": "Orchestrator_Agent",
        }
        for filename, key in files.items():
            cls.dummy_signals[key].to_csv(signals_dir / filename)

    def tearDown(self):
        plt.close("all")

    def test_plot_with_explicit_signals_dict(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_png = Path(tmp_dir) / "test_distribution.png"

            fig, ax = plot_agent_signal_distributions(
                asset="TEST_ASSET",
                signals_dict=self.dummy_signals,
                save_path=out_png,
                show_plot=False,
            )

            self.assertIsNotNone(fig)
            self.assertIsNotNone(ax)
            self.assertTrue(out_png.exists())
            self.assertGreater(out_png.stat().st_size, 5000)

    def test_plot_loads_from_signals_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._scrivi_segnali_sintetici(root)
            out_png = root / "test_spy_distribution.png"

            # Esercita il caricamento automatico usando solo CSV temporanei.
            with patch("src.plots.PROJECT_ROOT", root):
                fig, ax = plot_agent_signal_distributions(
                    asset="SPY", save_path=out_png, show_plot=False,
                )

            self.assertIsNotNone(fig)
            self.assertIsNotNone(ax)
            self.assertTrue(out_png.exists())
            self.assertGreater(out_png.stat().st_size, 10000)

    def test_load_all_signals_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._scrivi_segnali_sintetici(root)

            with patch("src.plots.PROJECT_ROOT", root):
                signals = load_all_signals_for_asset("SPY")

            self.assertIsInstance(signals, dict)
            self.assertGreaterEqual(len(signals), 4)
            for df in signals.values():
                self.assertIn("signal", df.columns)
                self.assertIn("confidence", df.columns)


if __name__ == "__main__":
    unittest.main()
