"""Regression tests for the Level B refactor (synthetic TRAINING ONLY).

These tests use temporary files and never read or overwrite frozen project
results. The original B_v1/B_v2 CSVs are checked by the existing main.py runs.
"""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.regime_allocator import (
    ASSETS,
    apply_signed_volatility_cap,
    apply_volatility_cap_panel,
    remaining_allocation_weight,
)
from src.regime_strategy import build_level_b_training, build_level_b_v2_training


class TestPostCapAccounting(TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2015-01-05", periods=145, name="Date")
        t = np.arange(len(self.dates))
        self.returns = pd.DataFrame(
            {
                "SPY": .025 * np.where(t % 2 == 0, 1.0, -1.0),
                "IEF": .003 * np.sin(t / 5),
                "GLD": .01 * np.sin(t / 7),
                "SHY": .0001 * np.sin(t / 11),
            },
            index=self.dates,
        )

    def test_panel_cap_matches_original_single_date_routine(self):
        weights = pd.DataFrame(
            [[1., 0., 0., 0.], [0., .5, .5, 0.]],
            index=self.dates[[-8, -1]], columns=ASSETS,
        )
        panel, risk = apply_volatility_cap_panel(weights, self.returns)
        self.assertEqual(list(risk.columns), ["vol_before", "vol_after", "scaling_factor"])
        for date in weights.index:
            expected, expected_risk = apply_signed_volatility_cap(
                weights.loc[date], self.returns, date,
            )
            pd.testing.assert_series_equal(panel.loc[date], expected, check_names=False)
            self.assertEqual(risk.loc[date].to_dict(), expected_risk)

    def test_final_cash_is_computed_after_cap(self):
        date = self.dates[-1]
        raw = pd.DataFrame([[1., 0., 0., 0.]], index=[date], columns=ASSETS)
        final, risk = apply_volatility_cap_panel(raw, self.returns)
        cash = remaining_allocation_weight(final, long_only=True)
        gross_unused = remaining_allocation_weight(final, long_only=False)
        self.assertLess(float(risk.loc[date, "scaling_factor"]), 1.0)
        self.assertGreater(float(cash.iloc[0]), 0.0)
        self.assertAlmostEqual(float(cash.iloc[0]), 1.0 - float(final.loc[date].sum()))
        self.assertAlmostEqual(float(cash.iloc[0]), float(gross_unused.iloc[0]))
        self.assertEqual(1.0 - float(raw.loc[date].sum()), 0.0)

    def test_short_positions_do_not_have_naive_cash_weight(self):
        signed = pd.DataFrame([[-.2, .2, 0., 0.]], index=self.dates[-1:], columns=ASSETS)
        idle = remaining_allocation_weight(signed, long_only=False)
        self.assertAlmostEqual(float(idle.iloc[0]), .6)
        with self.assertRaisesRegex(ValueError, "Long-only"):
            remaining_allocation_weight(signed, long_only=True)

    def test_rejects_oversized_or_invalid_target_panels(self):
        bad = pd.DataFrame([[1.1, 0., 0., 0.]], index=self.dates[-1:], columns=ASSETS)
        with self.assertRaises(ValueError):
            remaining_allocation_weight(bad, long_only=True)
        with self.assertRaises(ValueError):
            apply_volatility_cap_panel(bad.rename(columns={"GLD": "XXX"}), self.returns)


class TestIndependentRegimeStrategies(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        (cls.root / "data/processed").mkdir(parents=True)
        (cls.root / "data/raw/macro").mkdir(parents=True)
        dates = pd.bdate_range("2004-01-02", "2006-04-28", name="Date")
        t = np.arange(len(dates), dtype=float)
        prices = {
            "SPY": 100 * np.exp(.0007*t + .12*np.sin(t/19) + .06*np.sin(t/7.1)),
            "IEF": 96 * np.exp(.0002*t + .035*np.sin(t/28) + .018*np.cos(t/11)),
            "GLD": 102 * np.exp(.0003*t + .09*np.cos(t/19) + .045*np.sin(t/10)),
            "SHY": 99 * np.exp(.00007*t + .004*np.sin(t/34)),
        }
        daily = pd.DataFrame({f"{a}_close": v for a, v in prices.items()}, index=dates)
        daily.to_csv(cls.root / "data/processed/market_daily.csv", index_label="Date")
        decision_dates = pd.Series(dates, index=dates).resample("W-FRI").last().dropna()
        pd.DataFrame({"decision_date": decision_dates.values},
                     index=pd.DatetimeIndex(decision_dates.values, name="Date")).to_csv(
                         cls.root / "data/processed/weekly_decision_dates.csv", index_label="Date")
        for name, vals in (
            ("VIXCLS", 17 + 5*np.sin(t/23) + 2*np.cos(t/8)),
            ("T10YIE", 2.1 + .35*np.cos(t/29) + .2*np.sin(t/10)),
        ):
            pd.DataFrame({name: vals}, index=dates).to_csv(
                cls.root / f"data/raw/macro/{name}.csv", index_label="Date")
        cls.config = {
            "evaluation": {"evaluate_oos": False},
            "validation": {
                "development_only": True, "train_start": "2005-05-20",
                "train_end": "2006-03-31", "validation_start": "2006-04-01",
                "validation_end": "2006-09-30", "oos_start": "2007-01-01",
            },
            "data": {"start_date": "2004-01-01"},
            "paths": {"processed_data": "data/processed", "raw_data": "data/raw"},
        }
        cls.reference = build_level_b_v2_training(cls.config, cls.root)
        cls.decisions = cls.reference["regimes"].index
        cls.labels = pd.Series(
            np.where(np.arange(len(cls.decisions)) % 2, "cluster_1", "cluster_0"),
            index=cls.decisions, name="regime",
        )
        cls.policy = {
            "cluster_0": {"SPY": .4, "IEF": .2, "GLD": .2, "SHY": .2},
            "cluster_1": {"SPY": .15, "IEF": .35, "GLD": .30, "SHY": .20},
        }

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_bv1_remains_signed_and_risk_diagnostics_unchanged(self):
        result = build_level_b_training(self.config, self.root)
        self.assertEqual(list(result["risk"].columns),
                         ["vol_before", "vol_after", "scaling_factor"])
        self.assertTrue((result["risk"]["vol_after"] <= .08 + 1e-8).all())
        pd.testing.assert_series_equal(
            result["unused_gross_budget_final"],
            remaining_allocation_weight(result["target_weights"], long_only=False),
        )

    def test_bv2_is_independent_of_bv1_and_its_macro_filter(self):
        with patch("src.regime_strategy.build_level_b_training", side_effect=RuntimeError("Bv1 called")), \
             patch("src.regime_strategy.apply_macro_regime_filter", side_effect=RuntimeError("macro filter called")):
            result = build_level_b_v2_training(self.config, self.root)
        pd.testing.assert_frame_equal(result["target_weights"], self.reference["target_weights"])
        self.assertTrue((result["target_weights"] >= -1e-10).all().all())
        self.assertEqual(list(result["risk"].columns),
                         ["vol_before", "vol_after", "scaling_factor"])
        np.testing.assert_allclose(result["cash_weight_final"].to_numpy(),
                                   1 - result["target_weights"].sum(axis=1).to_numpy())

    def test_external_cluster_requires_explicit_valid_policy(self):
        with self.assertRaisesRegex(ValueError, "budget_policy"):
            build_level_b_v2_training(self.config, self.root, regimes_override=self.labels)
        with self.assertRaisesRegex(ValueError, "budget_policy"):
            build_level_b_v2_training(self.config, self.root, budget_policy=self.policy)

    def test_external_cluster_mapping_and_long_flat_risk(self):
        result = build_level_b_v2_training(
            self.config, self.root,
            regimes_override=self.labels, budget_policy=self.policy,
        )
        pd.testing.assert_series_equal(result["regimes"], self.labels.astype(object))
        self.assertAlmostEqual(float(result["budgets"].loc[self.decisions[0], "SPY"]), .4)
        self.assertAlmostEqual(float(result["budgets"].loc[self.decisions[1], "IEF"]), .35)
        self.assertTrue(np.allclose(result["budgets"].sum(axis=1), 1.0))
        self.assertTrue((result["target_weights"] >= -1e-10).all().all())
        self.assertTrue((result["risk"]["vol_after"] <= .08 + 1e-8).all())

    def test_reject_unmapped_cluster_and_future_dates(self):
        invalid = self.labels.copy()
        invalid.iloc[0] = "cluster_99"
        with self.assertRaisesRegex(ValueError, "No budget policy"):
            build_level_b_v2_training(self.config, self.root,
                                      regimes_override=invalid, budget_policy=self.policy)
        shifted = self.labels.copy()
        shifted.index = shifted.index + pd.Timedelta(days=7)
        with self.assertRaises(ValueError):
            build_level_b_v2_training(self.config, self.root,
                                      regimes_override=shifted, budget_policy=self.policy)

    def test_training_lock_remains_enforced(self):
        config = deepcopy(self.config)
        config["evaluation"]["evaluate_oos"] = True
        with self.assertRaisesRegex(ValueError, "training-only"):
            build_level_b_v2_training(config, self.root)

    def test_external_clusters_do_not_require_deterministic_macro_files(self):
        macro_path = self.root / "data/raw/macro/VIXCLS.csv"
        hidden = macro_path.with_suffix(".hidden")
        macro_path.rename(hidden)
        try:
            result = build_level_b_v2_training(
                self.config, self.root,
                regimes_override=self.labels, budget_policy=self.policy,
            )
            self.assertEqual(len(result["regimes"]), len(self.labels))
        finally:
            hidden.rename(macro_path)
