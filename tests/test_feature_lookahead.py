"""Regression checks for historical feature availability (audit II.3)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.agents.base_agent import BaseAgent
from src.agents.SVM_agent import SVMAgent


class TestFeatureLookahead(unittest.TestCase):
    def test_svm_missing_early_feature_is_not_backfilled(self):
        dates = pd.date_range("2015-01-02", periods=8, freq="W-FRI")
        df = pd.DataFrame({
            "SPY_close": [100, 101, 99, 102, 100, 104, 101, 105],
            "factor": [np.nan, np.nan, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0],
        }, index=dates)
        agent = SVMAgent(target_asset="SPY", feature_cols=["factor"])
        X, _, _ = agent._extract_xy(df.copy())
        self.assertTrue(X.loc[dates[:2], "factor"].isna().all())
        self.assertEqual(float(X.loc[dates[3], "factor"]), 2.0)

    def test_svm_predict_missing_early_feature_is_neutral(self):
        dates = pd.date_range("2014-01-03", periods=120, freq="W-FRI")
        t = np.arange(len(dates), dtype=float)
        df = pd.DataFrame({
            "SPY_close": 100 + 5 * np.sin(t / 3) + t * 0.1,
            "factor": np.sin(t / 7),
        }, index=dates)
        agent = SVMAgent(target_asset="SPY", feature_cols=["factor"])
        agent.fit(df, max_year=None)
        query = df.iloc[:5].copy()
        query.iloc[0, query.columns.get_loc("factor")] = np.nan
        proba = agent.predict_proba(query)
        self.assertEqual(float(proba.iloc[0]), 0.50)
        self.assertTrue(proba.iloc[1:].between(0.0, 1.0).all())

    def _load_macro_fixture(self, change_future_cpi=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "src").mkdir()
            macro = root / "data" / "raw" / "macro"
            macro.mkdir(parents=True)
            (root / "config.yaml").write_text(
                yaml.safe_dump({"data": {"release_lags_days": {
                    "FEDFUNDS": 35, "CPIAUCSL": 45,
                }}}), encoding="utf-8")
            daily = pd.date_range("2004-01-01", "2006-03-31", freq="B")
            pd.DataFrame({"Date": daily, "DGS10": np.full(len(daily), 4.0)}).to_csv(
                macro / "DGS10.csv", index=False)
            months = pd.date_range("2004-01-01", periods=26, freq="MS")
            cpi = np.arange(100.0, 126.0)
            if change_future_cpi:
                cpi[14:] += 1000.0
            pd.DataFrame({"Date": months, "CPIAUCSL": cpi}).to_csv(
                macro / "CPIAUCSL.csv", index=False)
            pd.DataFrame({"Date": ["2005-01-01"], "FEDFUNDS": [3.0]}).to_csv(
                macro / "FEDFUNDS.csv", index=False)
            decisions = pd.DatetimeIndex([
                "2005-02-04", "2005-02-07", "2005-02-14", "2005-02-15",
                "2005-02-18", "2005-04-01",
            ])
            return BaseAgent.load_macro_data(root_dir=root, decision_dates=decisions)

    def test_fedfunds_released_only_after_configured_lag(self):
        panel = self._load_macro_fixture()
        self.assertTrue(pd.isna(panel.loc["2005-02-04", "rates__policy_rate"]))
        self.assertEqual(float(panel.loc["2005-02-07", "rates__policy_rate"]), 3.0)

    def test_cpi_yoy_uses_months_and_release_lag(self):
        panel = self._load_macro_fixture()
        self.assertTrue(pd.isna(panel.loc["2005-02-14", "cpi_yoy"]))
        self.assertAlmostEqual(float(panel.loc["2005-02-15", "cpi_yoy"]), 0.12)

    def test_future_cpi_changes_do_not_change_past_feature(self):
        original = self._load_macro_fixture()
        changed = self._load_macro_fixture(change_future_cpi=True)
        pd.testing.assert_series_equal(
            original.loc[:"2005-02-18", "cpi_yoy"],
            changed.loc[:"2005-02-18", "cpi_yoy"],
        )


if __name__ == "__main__":
    unittest.main()
