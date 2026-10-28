"""Audit causale sintetico delle feature macro e dell'interfaccia regimi.

Da collocare in tests/test_macro_causality.py. Non legge dati reali,
non usa validation/OOS, non genera o modifica alcun output del progetto.

NOTA: i test verificano causalita' computazionale e interfacce.
NON certificano la disponibilita' storica point-in-time dei dati FRED.
"""

import unittest

import numpy as np
import pandas as pd

from src.macro_availability import (
    align_daily_observation_proxy,
    align_verified_daily_series,
)
from src.macro_availability import build_macro_feature_levels, classify_macro_regimes
from src.regime_allocator import build_regime_budget_panel, validate_regime_series


class TestMacroObservationAlignment(unittest.TestCase):
    def test_previous_observation_used_not_same_day(self):
        decisions = pd.DatetimeIndex(["2016-01-08", "2016-01-15"], name="Date")
        observed = pd.Series(
            [15., 999., 18.],
            index=pd.to_datetime(["2016-01-07", "2016-01-08", "2016-01-14"]),
        )
        aligned = align_daily_observation_proxy(decisions, observed)
        self.assertEqual(aligned.loc[decisions[0], "observation_date"], pd.Timestamp("2016-01-07"))
        self.assertEqual(aligned.loc[decisions[1], "observation_date"], pd.Timestamp("2016-01-14"))
        self.assertEqual(aligned.loc[decisions[0], "value"], 15.)
        self.assertEqual(aligned.loc[decisions[1], "value"], 18.)
        self.assertTrue((aligned["observation_date"].to_numpy() < decisions.to_numpy()).all())

    def test_only_same_day_value_is_unavailable(self):
        decisions = pd.DatetimeIndex(["2016-01-08"])
        observed = pd.Series([77.], index=pd.to_datetime(["2016-01-08"]))
        with self.assertRaisesRegex(ValueError, "Missing or stale"):
            align_daily_observation_proxy(decisions, observed)

    def test_stale_macro_value_rejected(self):
        decisions = pd.DatetimeIndex(["2016-01-15"])
        observed = pd.Series([77.], index=pd.to_datetime(["2016-01-04"]))
        with self.assertRaisesRegex(ValueError, "Missing or stale"):
            align_daily_observation_proxy(decisions, observed)

    def test_future_updates_cannot_change_past_alignment(self):
        decisions = pd.DatetimeIndex(["2016-01-08", "2016-01-15"])
        initial = pd.Series([12., 13.], index=pd.to_datetime(["2016-01-07", "2016-01-14"]))
        later = pd.concat([
            initial,
            pd.Series([9999.], index=pd.to_datetime(["2016-01-20"])),
        ])
        a = align_daily_observation_proxy(decisions, initial)
        b = align_daily_observation_proxy(decisions, later)
        pd.testing.assert_frame_equal(a, b)

    def test_duplicate_or_unsorted_macro_observations_rejected(self):
        decisions = pd.DatetimeIndex(["2016-01-15"])
        examples = [
            pd.Series([12., 13.], index=pd.to_datetime(["2016-01-14", "2016-01-14"])),
            pd.Series([12., 13.], index=pd.to_datetime(["2016-01-14", "2016-01-07"])),
        ]
        for observed in examples:
            with self.subTest(index=observed.index.tolist()):
                with self.assertRaises(ValueError):
                    align_daily_observation_proxy(decisions, observed)

    def test_release_time_alignment_uses_publication_instant(self):
        decisions = pd.DatetimeIndex([
            "2020-01-02 15:00:00+00:00",
            "2020-01-02 17:00:00+00:00",
            "2020-01-03 18:00:00+00:00",
        ])
        events = pd.DataFrame({
            "observation_date": ["2020-01-01", "2020-01-02"],
            "available_at": ["2020-01-02T16:00:00Z", "2020-01-03T12:00:00Z"],
            "value": [10., 20.],
        })
        result = align_verified_daily_series(decisions, events)
        self.assertTrue(np.isnan(result.iloc[0]["value"]))
        self.assertEqual(float(result.iloc[1]["value"]), 10.)
        self.assertEqual(float(result.iloc[2]["value"]), 20.)
        self.assertTrue(
            (result.dropna(subset=["value"])["available_at"] <=
             result.dropna(subset=["value"]).index).all()
        )

    def test_release_alignment_rejects_missing_timezones(self):
        decisions = pd.DatetimeIndex(["2020-01-03 18:00:00"])
        events = pd.DataFrame({
            "observation_date": ["2020-01-01"],
            "available_at": ["2020-01-02T16:00:00Z"],
            "value": [10.],
        })
        with self.assertRaises(ValueError):
            align_verified_daily_series(decisions, events)

    def test_revisions_are_not_silently_treated_as_one_vintage(self):
        decisions = pd.DatetimeIndex(["2020-01-05 12:00:00+00:00"])
        events = pd.DataFrame({
            "observation_date": ["2020-01-01", "2020-01-01"],
            "available_at": ["2020-01-02T16:00:00Z", "2020-01-04T16:00:00Z"],
            "value": [10., 12.],
        })
        with self.assertRaisesRegex(ValueError, "Revisions require"):
            align_verified_daily_series(decisions, events)


class TestMacroRegimeThresholds(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2014-01-03", periods=83, freq="W-FRI", name="Date")
        n = len(self.dates)
        self.vix = pd.Series(14. + np.linspace(0, 6, n), index=self.dates - pd.Timedelta(days=1))
        self.be = pd.Series(2. + 0.1 * np.sin(np.arange(n) / 3.0), index=self.dates - pd.Timedelta(days=1))

    def test_macro_levels_never_use_same_date_or_future_observation(self):
        features = build_macro_feature_levels(self.dates, self.vix, self.be)
        for field in ("vix_observation_date", "breakeven_observation_date"):
            self.assertTrue((features[field].to_numpy() < features.index.to_numpy()).all())

    def test_first_regime_requiring_52_past_weeks(self):
        features = build_macro_feature_levels(self.dates, self.vix, self.be)
        classified = classify_macro_regimes(features, window_weeks=52)
        self.assertTrue(classified["regime"].iloc[:52].isna().all())
        self.assertTrue(classified["regime"].iloc[52:].notna().all())

    def test_threshold_excludes_current_observation(self):
        features = build_macro_feature_levels(self.dates, self.vix, self.be)
        original = classify_macro_regimes(features, window_weeks=52)
        row = 65
        expected = float(features["vix_level"].iloc[row - 52:row].median())
        self.assertAlmostEqual(original["vix_threshold"].iloc[row], expected)
        altered = features.copy(deep=True)
        altered.iloc[row, altered.columns.get_loc("vix_level")] = 99999.
        rerun = classify_macro_regimes(altered, window_weeks=52)
        self.assertEqual(original["vix_threshold"].iloc[row], rerun["vix_threshold"].iloc[row])

    def test_future_macro_changes_leave_historical_regimes_unchanged(self):
        original_features = build_macro_feature_levels(self.dates, self.vix, self.be)
        baseline = classify_macro_regimes(original_features, window_weeks=52)
        cutoff = 64
        future_vix = self.vix.copy()
        future_be = self.be.copy()
        future_vix.iloc[cutoff + 1:] *= 30.
        future_be.iloc[cutoff + 1:] += 100.
        altered_features = build_macro_feature_levels(self.dates, future_vix, future_be)
        altered = classify_macro_regimes(altered_features, window_weeks=52)
        pd.testing.assert_frame_equal(baseline.iloc[:cutoff + 1], altered.iloc[:cutoff + 1])


class TestClusteringRegimeContract(unittest.TestCase):
    def setUp(self):
        self.dates = pd.DatetimeIndex(["2015-01-09", "2015-01-16", "2015-01-23"], name="Date")
        self.labels = pd.Series(["cluster_0", "cluster_1", "cluster_0"], index=self.dates)
        self.policy = {
            "cluster_0": {"SPY": .25, "IEF": .30, "GLD": .20, "SHY": .25},
            "cluster_1": {"SPY": .15, "IEF": .20, "GLD": .30, "SHY": .35},
        }

    def run_contract(self, labels=None, policy=None):
        return build_regime_budget_panel(
            regimes=self.labels if labels is None else labels,
            decision_dates=self.dates,
            train_start=pd.Timestamp("2015-01-01"),
            train_end=pd.Timestamp("2015-12-31"),
            budget_policy=self.policy if policy is None else policy,
        )

    def test_cluster_labels_map_to_explicit_asset_budgets(self):
        budgets = self.run_contract()
        self.assertTrue(budgets.index.equals(self.dates))
        self.assertEqual(list(budgets.columns), ["SPY", "IEF", "GLD", "SHY"])
        np.testing.assert_allclose(budgets.sum(axis=1), 1.)
        self.assertAlmostEqual(budgets.loc[self.dates[1], "SHY"], .35)
        pd.testing.assert_series_equal(budgets.iloc[0], budgets.iloc[2], check_names=False)

    def test_missing_mapping_rejected(self):
        with self.assertRaisesRegex(ValueError, "No budget policy"):
            self.run_contract(policy={"cluster_0": self.policy["cluster_0"]})

    def test_unallocated_negative_and_invalid_budgets_rejected(self):
        for bad in [
            {"SPY": .1, "IEF": .3, "GLD": .2, "SHY": .2},
            {"SPY": -.1, "IEF": .3, "GLD": .3, "SHY": .5},
            {"SPY": .1, "IEF": .3, "GLD": .2},
        ]:
            with self.subTest(bad=bad):
                p = {**self.policy, "cluster_1": bad}
                with self.assertRaises(ValueError):
                    self.run_contract(policy=p)

    def test_dates_outside_training_or_misaligned_rejected(self):
        bad_labels = self.labels.copy(deep=True)
        bad_labels.index = self.dates + pd.Timedelta(days=1)
        with self.assertRaises(ValueError):
            self.run_contract(labels=bad_labels)
        with self.assertRaises(ValueError):
            validate_regime_series(self.labels, self.dates, "2016-01-01", "2016-12-31")

    def test_empty_or_missing_labels_rejected(self):
        bad_labels = self.labels.copy(deep=True)
        bad_labels.iloc[1] = ""
        with self.assertRaises(ValueError):
            self.run_contract(labels=bad_labels)
        bad_labels.iloc[1] = np.nan
        with self.assertRaises(ValueError):
            self.run_contract(labels=bad_labels)


if __name__ == "__main__":
    unittest.main()
