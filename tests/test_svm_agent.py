"""Test dell'agente SVM (src/agents/SVM_agent.py).

I test usano soltanto dati sintetici. Non leggono dati di mercato reali e
non consultano validation o OOS.

Invarianti controllati:
- le variabili usano solo informazioni passate;
- l'etichetta usa il futuro ma è assente dove il futuro non è disponibile;
- walk-forward: modificare i dati successivi a un anno non cambia le
  previsioni fino a quell'anno;
- purge: le etichette che "toccano" l'anno da prevedere non entrano
  nell'allenamento di quell'anno;
- segnali in {-1, 0, +1}, confidenza in [0, 1], opzione long-only;
- risultati riproducibili;
- trainer: segnali out-of-fold solo dal primo anno di test, cutoff applicato
  prima di costruire le etichette, modello finale salvabile e ricaricabile;
- nessun riempimento all'indietro: righe incomplete -> segnale neutro.

Esecuzione dalla radice del repository:
    python3 -m unittest discover -s tests -p 'test_*.py' -v
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import build_asset_features
from src.agents.SVM_agent import (
    SVMAgent,
    SVMWalkForwardTrainer,
    build_direction_target,
    get_svm_settings,
    scores_to_signals,
    walk_forward_svm,
)

TICKERS = ["SPY", "IEF", "GLD", "SHY"]


def make_settings(**overrides) -> dict:
    settings = {
        "horizon_weeks": 4,
        "first_test_year": 2012,
        "min_train_weeks": 156,
        "cv_splits": 3,
        "C_values": [0.1, 1.0],
        "gamma_values": ["scale"],
        "class_weight": "none",
        "neutral_band": 0.25,
        "long_only": False,
        "use_macro_features": False,
    }
    settings.update(overrides)
    return settings


def make_weekly_close(seed: int = 0, end: str = "2016-12-30") -> pd.DataFrame:
    """Prezzi sintetici settimanali con un leggero drift positivo."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-07", end, freq="W-FRI")
    drifts = {"SPY": 0.0015, "IEF": 0.0008, "GLD": 0.0005, "SHY": 0.0002}
    vols = {"SPY": 0.022, "IEF": 0.010, "GLD": 0.020, "SHY": 0.002}
    data = {
        t: 100 * np.exp(np.cumsum(rng.normal(drifts[t], vols[t], len(dates))))
        for t in TICKERS
    }
    return pd.DataFrame(data, index=dates)


class TestFeaturesAndTarget(unittest.TestCase):
    def test_features_use_only_past_information(self):
        close = make_weekly_close()
        cutoff = pd.Timestamp("2010-06-25")
        changed = close.copy()
        changed.loc[changed.index > cutoff] *= 1.5

        original = build_asset_features(close, "SPY")
        modified = build_asset_features(changed, "SPY")
        pd.testing.assert_frame_equal(original.loc[:cutoff], modified.loc[:cutoff])

    def test_target_values_and_missing_prices(self):
        dates = pd.date_range("2016-11-04", periods=6, freq="W-FRI")
        close = pd.Series([100.0, 110.0, np.nan, 90.0, 120.0, 130.0], index=dates)
        label = build_direction_target(close, horizon=1)
        self.assertEqual(label.iloc[0], 1.0)            # 110 > 100
        self.assertTrue(pd.isna(label.iloc[1]))         # prezzo futuro mancante
        self.assertTrue(pd.isna(label.iloc[2]))         # prezzo attuale mancante
        self.assertEqual(label.iloc[3], 1.0)            # 120 > 90
        self.assertTrue(pd.isna(label.iloc[-1]))        # futuro non disponibile

    def test_target_is_missing_where_future_is_unknown(self):
        label = build_direction_target(make_weekly_close()["SPY"], horizon=4)
        self.assertTrue(label.iloc[-4:].isna().all())
        self.assertTrue(label.iloc[:-4].notna().all())
        self.assertTrue(set(label.dropna().unique()).issubset({0.0, 1.0}))


class TestWalkForward(unittest.TestCase):
    def setUp(self):
        self.settings = make_settings()
        self.close = make_weekly_close()
        self.features = build_asset_features(self.close, "SPY")
        self.label = build_direction_target(self.close["SPY"], 4)

    def run_wf(self, features=None, label=None):
        return walk_forward_svm(
            self.features if features is None else features,
            self.label if label is None else label,
            self.settings, first_year=2012, last_year=2016,
        )

    def test_predictions_start_at_first_test_year(self):
        predictions, history = self.run_wf()
        self.assertTrue(predictions.loc[:"2011", "score"].isna().all())
        self.assertTrue(predictions.loc["2012":, "score"].notna().any())
        self.assertEqual(list(history["year"]), [2012, 2013, 2014, 2015, 2016])

    def test_later_data_does_not_change_earlier_predictions(self):
        cutoff = pd.Timestamp("2014-12-31")
        changed_close = self.close.copy()
        changed_close.loc[changed_close.index > cutoff] *= 0.5
        features = build_asset_features(changed_close, "SPY")
        label = build_direction_target(changed_close["SPY"], 4)

        original, _ = self.run_wf()
        modified, _ = self.run_wf(features, label)
        pd.testing.assert_frame_equal(original.loc[:cutoff], modified.loc[:cutoff])

    def test_labels_overlapping_the_predicted_year_are_purged(self):
        first_2014 = self.label.loc["2014"].index[0]
        position = self.label.index.get_loc(first_2014)
        overlapping = self.label.index[position - 3: position]  # orizzonte 4: etichette che finiscono nel 2014

        flipped = self.label.copy()
        flipped.loc[overlapping] = 1.0 - flipped.loc[overlapping]

        original, _ = self.run_wf()
        modified, _ = self.run_wf(label=flipped)
        pd.testing.assert_frame_equal(original.loc["2014"], modified.loc["2014"])

    def test_reproducible(self):
        first, _ = self.run_wf()
        second, _ = self.run_wf()
        pd.testing.assert_frame_equal(first, second)


class TestSignals(unittest.TestCase):
    def test_signals_and_confidence_ranges(self):
        z = pd.Series([-1.0, -0.1, 0.0, 0.2, 0.3, 2.0, np.nan])
        signals, confidence = scores_to_signals(z, neutral_band=0.25, long_only=False)
        self.assertEqual(list(signals), [-1, 0, 0, 0, 1, 1, 0])
        self.assertTrue(((confidence >= 0) & (confidence <= 1)).all())
        self.assertTrue((confidence[signals == 0] == 0).all())

    def test_long_only_removes_shorts(self):
        z = pd.Series([-2.0, -0.5, 0.5])
        signals, _ = scores_to_signals(z, neutral_band=0.25, long_only=True)
        self.assertEqual(list(signals), [0, 0, 1])

    def test_agent_interface(self):
        close = make_weekly_close()
        agent = SVMAgent("SPY", make_settings()).fit_walk_forward(close, None, 2012, 2016)
        signals, confidence = agent.predict_with_confidence(close)
        self.assertTrue(set(signals.unique()).issubset({-1, 0, 1}))
        self.assertTrue((signals.loc[:"2011"] == 0).all())
        self.assertEqual(len(signals), len(close))
        self.assertTrue(((confidence >= 0) & (confidence <= 1)).all())


def make_panel(close: pd.DataFrame) -> pd.DataFrame:
    """Pannello come quello di train.py: colonne <ETF>_close sulle date di decisione."""
    return close.rename(columns=lambda t: f"{t}_close")


class TestTrainerAndFinalModel(unittest.TestCase):
    def setUp(self):
        self.settings = make_settings()
        self.panel = make_panel(make_weekly_close())

    def run_trainer(self, panel, max_year=2016):
        trainer = SVMWalkForwardTrainer("SPY", settings=self.settings)
        trainer.run(panel, max_year=max_year)
        return trainer

    def test_oof_signals_cover_only_test_years(self):
        trainer = self.run_trainer(self.panel)
        oof = trainer.oof_signals
        self.assertEqual(list(oof.columns), ["signal", "confidence"])
        self.assertEqual(oof.index.min().year, 2012)
        self.assertEqual(oof.index.max(), self.panel.index.max())
        self.assertFalse(oof.isna().any().any())
        self.assertTrue(oof["signal"].isin([-1, 0, 1]).all())
        self.assertTrue(oof["confidence"].between(0, 1).all())
        self.assertEqual([h["year"] for h in trainer.history], [2012, 2013, 2014, 2015, 2016])
        self.assertIsNotNone(trainer.final_agent)

    def test_cutoff_precedes_target_generation(self):
        full = make_panel(make_weekly_close(end="2017-12-29"))
        full.loc[full.index.year >= 2017, "SPY_close"] += 500.0  # salto estremo dopo il cutoff
        from_full = self.run_trainer(full, max_year=2016)
        from_cut = self.run_trainer(full.loc[full.index.year <= 2016], max_year=2016)
        pd.testing.assert_frame_equal(from_full.oof_signals, from_cut.oof_signals)
        self.assertEqual(from_full.final_agent.params["n_samples_train"],
                         from_cut.final_agent.params["n_samples_train"])

    def test_final_model_never_uses_unknown_labels(self):
        trainer = self.run_trainer(self.panel)
        last_known = self.panel.index[-1 - self.settings["horizon_weeks"]]
        self.assertEqual(trainer.final_agent.params["train_end"], last_known.date().isoformat())

    def test_rejects_unsorted_or_duplicate_dates(self):
        for broken in (self.panel.iloc[::-1], pd.concat([self.panel, self.panel.iloc[[-1]]])):
            with self.assertRaisesRegex(ValueError, "Indice temporale"):
                self.run_trainer(broken)

    def test_single_class_is_rejected(self):
        rising = self.panel.copy()
        rising["SPY_close"] = 100.0 + np.arange(len(rising))
        agent = SVMAgent("SPY", self.settings)
        closes = rising.rename(columns=lambda c: c.replace("_close", ""))
        with self.assertRaisesRegex(ValueError, "classi del target"):
            agent.fit_final(closes, None)

    def test_missing_features_give_neutral_signal_not_backfill(self):
        trainer = self.run_trainer(self.panel)
        query = self.panel.loc["2015"].copy()
        signals, confidence = trainer.final_agent.predict_with_confidence(query)
        # Con meno di 52 settimane di storia le variabili mancano: segnale neutro.
        self.assertTrue((signals == 0).all())
        self.assertTrue((confidence == 0).all())
        full_signals, _ = trainer.final_agent.predict_with_confidence(self.panel)
        self.assertTrue((full_signals.loc[:"2005"] == 0).all())

    def test_save_and_load_reproduce_predictions(self):
        trainer = self.run_trainer(self.panel)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "svm_params_spy.json"
            trainer.save_final_agent_config(path)
            self.assertTrue(path.with_suffix(".joblib").exists())
            loaded = SVMAgent("SPY", self.settings).load_params(path)
            original = trainer.final_agent.predict_with_confidence(self.panel)
            reloaded = loaded.predict_with_confidence(self.panel)
            pd.testing.assert_series_equal(original[0], reloaded[0])
            pd.testing.assert_series_equal(original[1], reloaded[1])


class TestSettings(unittest.TestCase):
    def test_missing_settings_are_rejected(self):
        config = {"strategies": {"C": {"svm_agent": make_settings()}}}
        get_svm_settings(config)
        del config["strategies"]["C"]["svm_agent"]["horizon_weeks"]
        with self.assertRaises(KeyError):
            get_svm_settings(config)


if __name__ == "__main__":
    unittest.main()
