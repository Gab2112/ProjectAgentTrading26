"""Contract tests for actual-session, train-only decision features.

These tests deliberately do not modify historical Level B parameters, signals,
portfolio weights, or frozen backtesting artifacts.
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data_transformation import (
    _macro_series_mapping,
    build_decision_feature_panel,
    build_training_decision_features,
    build_weekly_decision_dates,
    run_data_transformation,
)
from src.macro_availability import align_daily_macro_panel


class TestDecisionFeatures(unittest.TestCase):
    def setUp(self):
        self.sessions = pd.bdate_range("2020-01-02", periods=200, name="Date")
        n = np.arange(len(self.sessions), dtype=float)
        data = {}
        for i, asset in enumerate(("SPY", "IEF", "GLD", "SHY")):
            data[f"{asset}_open"] = 100 + 10 * i + .1 * n
            data[f"{asset}_close"] = 100 + 10 * i + .2 * n
        self.market = pd.DataFrame(data, index=self.sessions)
        self.decisions = build_weekly_decision_dates(self.market).index

    def test_dates_are_actual_sessions_not_calendar_fridays(self):
        missing_friday = pd.Timestamp("2020-01-10")
        truncated = self.market.drop(missing_friday)
        decisions = build_weekly_decision_dates(truncated).index
        self.assertIn(pd.Timestamp("2020-01-09"), decisions)
        panel = build_decision_feature_panel(truncated, decisions)
        self.assertTrue(panel.index.equals(decisions))
        self.assertNotIn(missing_friday, panel.index)

    def test_close_and_return_are_known_at_decision_close(self):
        panel = build_decision_feature_panel(self.market, self.decisions)
        self.assertEqual(panel.index.name, "Date")
        self.assertEqual(panel.loc[self.decisions[2], "SPY_close"],
                         self.market.loc[self.decisions[2], "SPY_close"])
        expected_return = (self.market.loc[self.decisions[2], "SPY_close"] /
                           self.market.loc[self.decisions[1], "SPY_close"] - 1)
        self.assertAlmostEqual(panel.loc[self.decisions[2], "SPY_return_1w"],
                               expected_return)
        self.assertTrue(np.isnan(panel.loc[self.decisions[0], "SPY_return_1w"]))

    def test_momentum_21_and_63_use_only_past_sessions(self):
        panel = build_decision_feature_panel(self.market, self.decisions)
        date = self.decisions[20]
        pos = self.market.index.get_loc(date)
        for window in (21, 63):
            expected = (self.market.iloc[pos]["SPY_close"] /
                        self.market.iloc[pos-window]["SPY_close"] - 1)
            self.assertAlmostEqual(panel.loc[date, f"SPY_momentum_{window}d"], expected)

    def test_market_future_mutation_cannot_change_past_prefix(self):
        early = self.decisions[:25]
        original = build_decision_feature_panel(self.market, early)
        altered = self.market.copy()
        altered.loc[altered.index > early[-1], "SPY_close"] *= 20
        recomputed = build_decision_feature_panel(altered, early)
        pd.testing.assert_frame_equal(original, recomputed)

    def test_rejects_nontrading_decision(self):
        invalid = pd.DatetimeIndex([pd.Timestamp("2020-01-04")], name="Date")
        with self.assertRaises(ValueError):
            build_decision_feature_panel(self.market, invalid)

    def test_rejects_unsorted_duplicate_and_timestamped_dates(self):
        for decisions in (
            self.decisions[:3][::-1],
            self.decisions[:3].append(self.decisions[-1:]).append(self.decisions[-1:]),
            self.decisions[:3] + pd.Timedelta(hours=1),
        ):
            with self.subTest(decisions=str(decisions)):
                with self.assertRaises(ValueError):
                    build_decision_feature_panel(self.market, decisions)

    def test_rejects_missing_week_in_one_week_return(self):
        skipped = self.decisions.delete(6)
        with self.assertRaises(ValueError):
            build_decision_feature_panel(self.market, skipped)

    def test_rejects_midweek_date_when_friday_traded(self):
        invalid = self.decisions[:10].delete(4).append(
            pd.DatetimeIndex([self.decisions[4] - pd.Timedelta(days=1)])
        ).sort_values()
        with self.assertRaises(ValueError):
            build_decision_feature_panel(self.market, invalid)

    def test_rejects_nan_market_close(self):
        altered = self.market.copy()
        altered.iloc[10, altered.columns.get_loc("GLD_close")] = np.nan
        with self.assertRaises(ValueError):
            build_decision_feature_panel(altered, self.decisions)

    def test_daily_macro_proxy_excludes_same_date_observation(self):
        decision = pd.DatetimeIndex([pd.Timestamp("2020-05-15")], name="Date")
        obs = pd.Series([11.0, 88.0],
                        index=pd.DatetimeIndex(["2020-05-14", "2020-05-15"]))
        panel = align_daily_macro_panel(decision, {"VIXCLS": obs})
        self.assertEqual(panel.loc[decision[0], "VIXCLS"], 11.0)
        self.assertEqual(panel.loc[decision[0], "VIXCLS_observation_date"],
                         pd.Timestamp("2020-05-14"))

    def test_macro_panel_reports_observation_provenance(self):
        decisions = self.decisions[5:10]
        macro = pd.Series(np.arange(len(self.sessions), dtype=float), index=self.sessions)
        panel = build_decision_feature_panel(
            self.market, decisions, {"stress": macro, "inflation": macro + 1}
        )
        self.assertIn("stress_observation_date", panel.columns)
        self.assertIn("inflation_observation_date", panel.columns)
        self.assertTrue((panel["stress_observation_date"].to_numpy()
                         < panel.index.to_numpy()).all())

    def test_macro_future_mutation_cannot_change_past_prefix(self):
        early = self.decisions[:25]
        original_macro = pd.Series(np.arange(len(self.sessions), dtype=float),
                                   index=self.sessions)
        original = build_decision_feature_panel(self.market, early,
                                                {"vix": original_macro})
        altered = original_macro.copy()
        altered.loc[altered.index >= early[-1]] *= 100
        fresh = build_decision_feature_panel(self.market, early, {"vix": altered})
        pd.testing.assert_frame_equal(original, fresh)

    def test_macro_proxy_rejects_stale_history(self):
        decision = pd.DatetimeIndex([pd.Timestamp("2020-05-15")], name="Date")
        old = pd.Series([10.], index=pd.DatetimeIndex(["2020-04-01"]))
        with self.assertRaises(ValueError):
            align_daily_macro_panel(decision, {"vix": old})

    def test_training_panel_excludes_validation_and_future(self):
        config = {
            "evaluation": {"evaluate_oos": False},
            "validation": {"development_only": True},
        }
        train_start = pd.Timestamp("2020-04-01")
        train_end = pd.Timestamp("2020-07-31")
        with patch("src.data_transformation.get_time_windows", return_value={
            "train_start": train_start, "train_end": train_end,
        }):
            panel = build_training_decision_features(
                config, self.market, self.decisions
            )
        self.assertGreaterEqual(panel.index.min(), train_start)
        self.assertLessEqual(panel.index.max(), train_end)
        self.assertTrue(panel.index.equals(self.decisions[
            (self.decisions >= train_start) & (self.decisions <= train_end)
        ]))

    def test_training_unauthorized_oos_is_rejected(self):
        config = {
            "evaluation": {"evaluate_oos": True},
            "validation": {"development_only": False},
        }
        with self.assertRaises(ValueError):
            build_training_decision_features(config, self.market, self.decisions)

    def test_export_writes_only_training_decision_features(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "data" / "raw" / "market"
            raw.mkdir(parents=True)
            for ticker in ("SPY", "IEF", "GLD", "SHY"):
                pd.DataFrame({
                    "Open": self.market[f"{ticker}_open"],
                    "Close": self.market[f"{ticker}_close"],
                }).to_csv(raw / f"{ticker}.csv", index_label="Date")

            config = {
                "universe": {"traded_assets": {
                    t: {"ticker": t} for t in ("SPY", "IEF", "GLD", "SHY")
                }},
                "paths": {"raw_data": "data/raw", "processed_data": "data/processed"},
                "pipeline": {"active_level": "A1"},
                "evaluation": {"evaluate_oos": False},
                "validation": {
                    "development_only": True,
                    "train_start": "2020-04-01", "train_end": "2020-07-31",
                    "validation_start": "2020-08-01", "validation_end": "2020-09-30",
                    "oos_start": "2020-10-01",
                },
                "data": {"start_date": "2020-01-02"},
            }
            with patch("src.data_transformation.get_time_windows", return_value={
                "train_start": pd.Timestamp("2020-04-01"),
                "train_end": pd.Timestamp("2020-07-31"),
            }):
                run_data_transformation(config, root)
            output_path = root / "data/processed/decision_features_training.csv"
            self.assertTrue(output_path.is_file())
            result = pd.read_csv(output_path, index_col="Date", parse_dates=["Date"])
            self.assertGreaterEqual(result.index.min(), pd.Timestamp("2020-04-01"))
            self.assertLessEqual(result.index.max(), pd.Timestamp("2020-07-31"))
            self.assertIn("SPY_momentum_21d", result.columns)
            self.assertIn("SPY_return_1w", result.columns)

    def test_stress_is_in_macro_mapping_for_opt_in_usage(self):
        config = {"data": {
            "rates": {"treasury_10y": "DGS10"},
            "inflation": {"breakeven_10y": "T10YIE"},
            "stress": {"equity_implied_volatility": "VIXCLS"},
        }}
        self.assertEqual(_macro_series_mapping(config), {
            "rates__treasury_10y": "DGS10",
            "inflation__breakeven_10y": "T10YIE",
            "stress__equity_implied_volatility": "VIXCLS",
        })


if __name__ == "__main__":
    unittest.main()
