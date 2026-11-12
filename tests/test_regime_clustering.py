"""Tests of the regime clustering (Level B preparation).

The tests use synthetic data only. They do not read real market data and do
not consult validation or OOS results.

Invariants checked:
- the variables use past information only;
- the model is fitted on the training period only: later data does not
  change the clusters assigned to training weeks;
- probabilities are valid and labels are ordered in a stable way;
- the share of "unrecognised" weeks in the training period respects the quantile;
- the risk diagnostics use only windows within the training period;
- decision dates (including Thursdays) are mapped to their own week.

Run from the repository root:
    python3 -m unittest discover -s tests -p 'test_*.py' -v
"""

import unittest
from copy import deepcopy

import numpy as np
import pandas as pd

from src.features import build_cluster_features
from src.regime_clustering import (
    align_to_decision_dates,
    build_risk_summary_train,
    get_clustering_settings,
    run_cluster_model,
)

TRAIN_END = pd.Timestamp("2016-12-31")


def make_settings() -> dict:
    return {
        "n_clusters": 3,
        "covariance_type": "diag",
        "unknown_quantile": 0.02,
        "order_by": "treasury_2y_change_13w",
        "change_weeks": 13,
        "volatility_weeks": 13,
        "zscore_weeks": 156,
        "zscore_min_weeks": 104,
        "features": [
            "treasury_2y_change_13w",
            "curve_10y_2y_change_13w",
            "rate_volatility_13w",
            "treasury_2y_zscore_3y",
            "curve_10y_2y_zscore_3y",
        ],
    }


def make_weekly_yields(seed: int = 0) -> pd.DataFrame:
    """Synthetic weekly (W-FRI) yields from 2004 to 2021."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2004-01-02", "2021-12-31", freq="W-FRI")
    y2 = 2.0 + np.cumsum(rng.normal(0, 0.08, len(dates)))
    y10 = y2 + 1.5 + np.cumsum(rng.normal(0, 0.04, len(dates)))
    return pd.DataFrame({"treasury_2y": y2, "treasury_10y": y10}, index=dates)


def make_features(seed: int = 0) -> pd.DataFrame:
    return build_cluster_features(make_weekly_yields(seed), make_settings())


class TestClusterFeatures(unittest.TestCase):
    def test_features_use_only_past_information(self):
        yields = make_weekly_yields()
        cutoff = pd.Timestamp("2012-06-29")
        changed = yields.copy()
        changed.loc[changed.index > cutoff] += 5.0  # modified future

        original = build_cluster_features(yields, make_settings())
        modified = build_cluster_features(changed, make_settings())
        pd.testing.assert_frame_equal(original.loc[:cutoff], modified.loc[:cutoff])

    def test_zscore_needs_minimum_history(self):
        features = make_features()
        first_valid = features["treasury_2y_zscore_3y"].first_valid_index()
        self.assertEqual(features.index.get_loc(first_valid), 104 - 1)

    def test_thursday_decision_maps_to_its_week(self):
        features = make_features()
        friday = pd.Timestamp("2015-04-03")       # Good Friday: market closed
        thursday = pd.Timestamp("2015-04-02")     # actual decision
        aligned = align_to_decision_dates(features, pd.DatetimeIndex([thursday]))
        pd.testing.assert_series_equal(
            aligned.iloc[0], features.loc[friday], check_names=False
        )
        self.assertEqual(aligned.index[0], thursday)


class TestClusterModel(unittest.TestCase):
    def setUp(self):
        self.settings = make_settings()
        self.features = make_features()
        self.regimes = run_cluster_model(self.features, TRAIN_END, self.settings, seed=42)

    def test_probabilities_are_valid(self):
        probs = self.regimes.filter(like="p_cluster_")
        self.assertEqual(probs.shape[1], 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)
        self.assertTrue(self.regimes["cluster"].isin([0, 1, 2]).all())
        self.assertTrue(self.regimes["known"].dtype == bool)

    def test_training_clusters_do_not_depend_on_later_data(self):
        changed = self.features.copy()
        changed.loc[changed.index > TRAIN_END] *= 10.0
        regimes_changed = run_cluster_model(changed, TRAIN_END, self.settings, seed=42)

        train = self.regimes.index <= TRAIN_END
        pd.testing.assert_frame_equal(
            self.regimes.loc[train], regimes_changed.loc[regimes_changed.index <= TRAIN_END]
        )

    def test_labels_ordered_by_rate_change(self):
        train = self.regimes.loc[self.regimes.index <= TRAIN_END, "cluster"]
        means = self.features.loc[train.index, "treasury_2y_change_13w"].groupby(train).mean()
        self.assertTrue(means.is_monotonic_increasing)

    def test_unknown_share_in_training_matches_quantile(self):
        known_train = self.regimes.loc[self.regimes.index <= TRAIN_END, "known"]
        self.assertAlmostEqual((~known_train).mean(), 0.02, delta=0.01)

    def test_reproducible_with_same_seed(self):
        again = run_cluster_model(self.features, TRAIN_END, self.settings, seed=42)
        pd.testing.assert_frame_equal(self.regimes, again)


class TestRiskSummary(unittest.TestCase):
    def test_uses_only_windows_inside_training(self):
        dates = pd.bdate_range("2015-01-01", "2017-06-30")
        rng = np.random.default_rng(1)
        prices = pd.DataFrame(
            {"SPY_close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))},
            index=dates,
        )
        decisions = pd.DatetimeIndex(pd.Series(dates, index=dates).resample("W-FRI").last().values)
        regimes = pd.DataFrame({"cluster": 0, "known": True}, index=decisions)

        base = build_risk_summary_train(regimes, prices, ["SPY"], TRAIN_END)

        shocked = prices.copy()
        after = shocked.index > TRAIN_END
        shocked.loc[after, "SPY_close"] *= np.exp(np.cumsum(rng.normal(0, 0.2, after.sum())))
        modified = build_risk_summary_train(regimes, shocked, ["SPY"], TRAIN_END)

        pd.testing.assert_frame_equal(base, modified)


class TestSettings(unittest.TestCase):
    def test_missing_settings_are_rejected(self):
        config = {"strategies": {"B": {"clustering": make_settings()}}}
        broken = deepcopy(config)
        del broken["strategies"]["B"]["clustering"]["n_clusters"]
        get_clustering_settings(config)
        with self.assertRaises(KeyError):
            get_clustering_settings(broken)

    def test_order_by_must_be_a_feature(self):
        config = {"strategies": {"B": {"clustering": make_settings()}}}
        config["strategies"]["B"]["clustering"]["order_by"] = "non_esiste"
        with self.assertRaises(ValueError):
            get_clustering_settings(config)


if __name__ == "__main__":
    unittest.main()
