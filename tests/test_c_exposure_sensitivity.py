"""Unit tests: pre-declared Level C exposure what-if (synthetic training data)."""
from __future__ import annotations

from unittest import TestCase

import numpy as np
import pandas as pd

from src.regime_allocator import ASSETS
from src.run_c_exposure_sensitivity import make_scenario_weights


class TestExposureSensitivity(TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2015-01-01", periods=145, name="Date")
        self.decisions = self.dates[130:145:7]
        x = np.random.default_rng(17).normal(0, 0.004, (len(self.dates), 4))
        self.returns = pd.DataFrame(x, index=self.dates, columns=ASSETS)
        self.end = self.dates[-1]
        self.raw = pd.DataFrame([
            [0.25, 0.20, 0.0, 0.25],
            [0.0, 0.0, 0.0, 0.0],
            [0.25, 0.0, -0.30, 0.35],
        ], index=self.decisions, columns=ASSETS)

    def scenario(self, factor, cap, raw=None, returns=None):
        return make_scenario_weights(
            self.raw if raw is None else raw,
            self.returns if returns is None else returns,
            exposure_factor=factor, volatility_cap=cap, train_end=self.end,
        )

    def test_baseline_uses_existing_signed_cap_without_amplification(self):
        from src.regime_allocator import apply_volatility_cap_panel
        target, _, limited = self.scenario(1.0, 0.08)
        expected, _ = apply_volatility_cap_panel(self.raw, self.returns, window=126, target_volatility=0.08)
        np.testing.assert_allclose(target.to_numpy(), expected.to_numpy())
        self.assertFalse(limited.any())

    def test_2x_never_leverages_and_keeps_short_direction(self):
        target, risk, limited = self.scenario(2.0, 0.12)
        self.assertTrue((target.abs().sum(axis=1) <= 1.0 + 1e-10).all())
        self.assertLess(target.iloc[2]["GLD"], 0)
        self.assertTrue((risk["vol_after"] <= 0.12 + 1e-8).all())
        self.assertTrue(limited.iloc[0])
        self.assertTrue(limited.iloc[2])

    def test_no_new_positions_where_signals_are_zero(self):
        target, _, _ = self.scenario(2.0, 0.08)
        self.assertTrue((target.iloc[1] == 0).all())

    def test_lifting_vol_cap_can_restore_risk_only_from_raw_weights(self):
        # Strong returns make 8% cap bind, 12% lets original raw weight grow.
        rng = np.random.default_rng(7)
        high = self.returns * 6
        low_target, _, _ = self.scenario(1.0, 0.08, returns=high)
        high_target, _, _ = self.scenario(1.0, 0.12, returns=high)
        self.assertGreater(high_target.iloc[0].abs().sum(), low_target.iloc[0].abs().sum())

    def test_future_dates_rejected_instead_of_used(self):
        future = pd.DataFrame(.03, index=[self.end + pd.offsets.BDay(1)], columns=ASSETS)
        extended = pd.concat([self.returns, future])
        with self.assertRaisesRegex(ValueError, "TRAINING"):
            self.scenario(1.0, 0.08, returns=extended)

    def test_invalid_gross_exposure_or_multiplier_rejected(self):
        bad = self.raw.copy()
        bad.iloc[0] = [0.9, 0.8, 0.0, 0.25]
        with self.assertRaisesRegex(ValueError, "gross"):
            self.scenario(1.0, 0.08, raw=bad)
        with self.assertRaisesRegex(ValueError, "factor"):
            self.scenario(0.0, 0.08)

    def test_bad_calendar_or_missing_decision_return_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing historical"):
            self.scenario(1.0, 0.08, returns=self.returns.drop(index=self.decisions[1]))
        with self.assertRaisesRegex(ValueError, "unique"):
            self.scenario(1.0, 0.08, returns=self.returns.iloc[::-1])


if __name__ == "__main__":
    import unittest
    unittest.main()

class TestExposureRunnerWiring(TestCase):
    def setUp(self):
        self.sessions = pd.bdate_range("2015-01-01", periods=146, name="Date")
        self.end = self.sessions[-2]
        self.decisions = self.sessions[[130, 137, 144]]
        rng = np.random.default_rng(109)
        self.market = pd.DataFrame(index=self.sessions)
        for a in ASSETS:
            closes = 100.0 * np.exp(np.cumsum(rng.normal(0, .002, len(self.sessions))))
            self.market[f"{a}_open"] = closes
            self.market[f"{a}_close"] = closes
        aligned = self.market.loc[:self.end, [f"{a}_close" for a in ASSETS]].copy()
        aligned.columns = list(ASSETS)
        self.returns = aligned.pct_change(fill_method=None).iloc[1:]
        self.raw1 = pd.DataFrame([
            [0.25, 0.20, 0.0, 0.25],
            [0.0, 0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0, 0.35],
        ], index=self.decisions, columns=ASSETS)
        self.raw2 = self.raw1.copy()
        self.raw2.iloc[-1, 2] = -0.30
        from src.regime_allocator import apply_volatility_cap_panel
        self.target1, _ = apply_volatility_cap_panel(self.raw1, self.returns)
        self.target2, _ = apply_volatility_cap_panel(self.raw2, self.returns)
        self.cfg = {
            "paths": {"processed_data": "data/processed", "results": "data/results"},
            "evaluation": {"annualization_factor": 252, "risk_free_rate": 0.0},
            "execution": {"initial_capital": 100000.0, "commission_bps": 2,
                          "slippage_bps": 1},
            "strategies": {"B": {"annual_borrow_rate": 0.01,
                                  "additional_margin_rate": 0.50}},
        }
        self.calls = []

    def fake_simulate(self, **kwargs):
        self.calls.append(kwargs)
        dates = kwargs["open_prices"].index
        initial = kwargs["initial_capital"]
        return pd.DataFrame({
            "nav": np.full(len(dates), initial),
            "daily_return": np.zeros(len(dates)),
            "cash": np.full(len(dates), initial),
            "margin_shortfall": np.zeros(len(dates)),
            "gross_exposure": np.zeros(len(dates)),
            "transaction_cost": np.zeros(len(dates)),
            "borrow_fee": np.zeros(len(dates)),
        }, index=dates)

    def test_runner_16_scenarios_and_cutoff(self):
        from unittest.mock import patch
        import src.run_c_exposure_sensitivity as module
        with patch.object(module, "_guard_training", return_value=(self.sessions[0], self.end)), \
             patch.object(module, "load_dataset", return_value=self.market), \
             patch.object(module, "build_level_c_training",
                          return_value={"raw_weights": self.raw1, "target_weights": self.target1}), \
             patch.object(module, "build_level_c_v2_training",
                          return_value={"raw_weights": self.raw2, "target_weights": self.target2}), \
             patch.object(module, "simulate_longshort_daily", side_effect=self.fake_simulate):
            result = module.run_c_exposure_sensitivity(self.cfg, ".")
        self.assertEqual(len(result["summary"]), 16)
        self.assertEqual(len(self.calls), 16)
        self.assertTrue(all(x["open_prices"].index.max() <= self.end for x in self.calls))
        self.assertTrue(all(x["annual_borrow_rate"] == 0.0 for x in self.calls[:8]))
        self.assertTrue(all(x["annual_borrow_rate"] == 0.01 for x in self.calls[8:]))
        self.assertEqual(result["summary"].loc["C_v1_x1_vol8", "zero_exposure_weeks"], 1)

    def test_baseline_mismatch_rejected(self):
        from unittest.mock import patch
        import src.run_c_exposure_sensitivity as module
        bad = self.target1.copy()
        bad.iloc[0, 0] += 0.01
        with patch.object(module, "_guard_training", return_value=(self.sessions[0], self.end)), \
             patch.object(module, "load_dataset", return_value=self.market), \
             patch.object(module, "build_level_c_training",
                          return_value={"raw_weights": self.raw1, "target_weights": bad}), \
             patch.object(module, "build_level_c_v2_training",
                          return_value={"raw_weights": self.raw2, "target_weights": self.target2}), \
             patch.object(module, "simulate_longshort_daily", side_effect=self.fake_simulate):
            with self.assertRaisesRegex(AssertionError, "baseline mismatch"):
                module.run_c_exposure_sensitivity(self.cfg, ".")
