"""Training-only matched B/C comparison tests with synthetic data.

The actual A/B simulator's execution accounting already has separate tests;
these tests exercise the NEW wrapper's alignment, safeguards and arguments.
"""
from __future__ import annotations

from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.run_training_comparison as module
from src.utils import ASSETS


class TestMatchedTrainingComparison(TestCase):
    def setUp(self):
        self.sessions = pd.bdate_range("2014-01-06", periods=11, name="Date")
        self.decisions = self.sessions[[0, 5, 10]]
        self.end = self.sessions[-1]
        self.config = {
            "validation": {"development_only": True},
            "evaluation": {"evaluate_oos": False, "annualization_factor": 252,
                           "risk_free_rate": 0.0},
            "execution": {"execution_rule": "next_available_trading_open",
                          "signal_frequency": "weekly", "cost_basis": "traded_notional",
                          "initial_capital": 100000.0, "commission_bps": 2,
                          "slippage_bps": 1},
            "strategies": {"B": {"cash_interest_rate": 0.0,
                                 "annual_borrow_rate": 0.01,
                                 "additional_margin_rate": 0.50}},
            "paths": {"processed_data": "data/processed", "results": "data/results"},
        }
        self.b1 = pd.DataFrame(
            [[-0.25, 0.25, 0.0, 0.0], [-0.25, 0.25, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            index=self.decisions, columns=ASSETS,
        )
        self.b2 = pd.DataFrame(
            [[0.25, 0.0, 0.0, 0.0], [0.25, 0.0, 0.0, 0.0], [0.25, 0.0, 0.0, 0.0]],
            index=self.decisions, columns=ASSETS,
        )
        self.c1 = self.b2.copy()
        self.c1.loc[:, "SPY"] *= 0.5
        self.c2 = self.c1.copy()
        self.c2.loc[self.decisions[1], "IEF"] = -0.30
        self.market = pd.DataFrame(index=self.sessions)
        for a in ASSETS:
            self.market[f"{a}_open"] = 100.0
            self.market[f"{a}_close"] = 100.0
        self.call_parameters = []

    def fake_simulate(self, **kwargs):
        self.call_parameters.append(kwargs)
        dates = kwargs["open_prices"].index
        total = len(dates)
        r = np.full(total, 0.001, dtype=float)
        nav = kwargs["initial_capital"] * np.cumprod(1.0 + r)
        rate = kwargs["annual_borrow_rate"]
        short_frac = 0.1 if (kwargs["target_weights"] < 0).any().any() else 0.0
        return pd.DataFrame({
            "nav": nav, "daily_return": r, "cash": nav * (1-short_frac),
            "borrow_fee": np.full(total, rate * 0.1),
            "transaction_cost": np.full(total, 1.0),
            "turnover": np.full(total, 0.01),
            "executed": [False] + [True]*(total-1),
            "gross_exposure": np.full(total, 0.5),
            "net_exposure": np.full(total, 0.3),
            "short_exposure": np.full(total, short_frac),
            "margin_shortfall": np.zeros(total),
        }, index=dates)

    def run_with_mocks(self, *, c=None, b1=None, market=None, config=None):
        c = c if c is not None else {
            "target_weights": self.c2,
            "c_v1_target_weights_same_dates": self.c1,
            "b_v2_target_weights_same_dates": self.b2,
        }
        b1 = b1 if b1 is not None else self.b1
        market = market if market is not None else self.market
        config = config if config is not None else self.config
        with patch.object(module, "get_time_windows", return_value={
            "train_start": self.sessions[0], "train_end": self.end,
        }), patch.object(module, "is_oos_enabled", return_value=False), \
             patch.object(module, "build_level_c_v2_training", return_value=c), \
             patch.object(module, "build_level_b_training", return_value={"target_weights": b1}), \
             patch.object(module, "load_dataset", return_value=market), \
             patch.object(module, "simulate_longshort_daily", side_effect=self.fake_simulate):
            return module.run_training_comparison(config, ".")

    def test_matched_four_portfolios_and_next_open(self):
        result = self.run_with_mocks()
        self.assertEqual(list(result["summary"].index), list(module.PORTFOLIOS))
        self.assertEqual(len(self.call_parameters), 4)
        self.assertEqual(result["summary"].loc["C_v2", "decisions"], 3)
        self.assertEqual(result["summary"].loc["C_v2", "scheduled_rebalances"], 2)
        self.assertEqual(result["execution_schedules"]["C_v2"].index.min(), self.sessions[1])
        self.assertEqual(result["last_accounted_session"], self.end)
        self.assertEqual([x["annual_borrow_rate"] for x in self.call_parameters],
                         [.01, 0.0, 0.0, .01])
        self.assertTrue(all(x["commission_bps"] == 2 and x["slippage_bps"] == 1
                            for x in self.call_parameters))
        self.assertTrue(all(x["additional_margin_rate"] == .50
                            for x in self.call_parameters))
        self.assertTrue(result["summary"].loc[["B_v1", "C_v2"],
                     "mean_cash_fraction_longflat"].isna().all())
        self.assertTrue(result["summary"].loc[["B_v2", "C_v1"],
                     "mean_cash_fraction_longflat"].notna().all())

    def test_reject_missing_b_v1_decision(self):
        with self.assertRaisesRegex(ValueError, "missing.*decision"):
            self.run_with_mocks(b1=self.b1.iloc[:-1])

    def test_reject_short_position_in_c_v1(self):
        bad = self.c1.copy()
        bad.loc[self.decisions[0], "GLD"] = -0.20
        with self.assertRaisesRegex(ValueError, "Long/Flat"):
            self.run_with_mocks(c={
                "target_weights": self.c2,
                "c_v1_target_weights_same_dates": bad,
                "b_v2_target_weights_same_dates": self.b2,
            })

    def test_reject_inconsistent_c_decision_dates(self):
        bad = self.c1.iloc[1:]
        with self.assertRaisesRegex(ValueError, "mismatched decision"):
            self.run_with_mocks(c={
                "target_weights": self.c2,
                "c_v1_target_weights_same_dates": bad,
                "b_v2_target_weights_same_dates": self.b2,
            })

    def test_future_market_rows_are_excluded(self):
        extra_date = self.end + pd.offsets.BDay(1)
        future = pd.DataFrame(99999.0, index=pd.DatetimeIndex([extra_date], name="Date"),
                              columns=self.market.columns)
        extended = pd.concat([self.market, future])
        self.run_with_mocks(market=extended)
        self.assertTrue(all(args["open_prices"].index.max() == self.end
                            for args in self.call_parameters))
        self.assertTrue(all(args["close_prices"].index.max() == self.end
                            for args in self.call_parameters))

    def test_reject_leverage_in_b_v1(self):
        bad = self.b1.copy()
        bad.loc[self.decisions[0], "SPY"] = -0.90
        with self.assertRaisesRegex(ValueError, "leveraged"):
            self.run_with_mocks(b1=bad)

    def test_training_lock_and_wrong_execution(self):
        bad = dict(self.config)
        bad["evaluation"] = {**self.config["evaluation"], "evaluate_oos": True}
        with self.assertRaisesRegex(ValueError, "Training-only lock"):
            module._guard_training(bad)
        bad = dict(self.config)
        bad["execution"] = {**self.config["execution"], "execution_rule": "same_close"}
        with patch.object(module, "is_oos_enabled", return_value=False), \
             patch.object(module, "get_time_windows", return_value={
                 "train_start": self.sessions[0], "train_end": self.end,
             }):
            with self.assertRaisesRegex(ValueError, "Execution policy"):
                module._guard_training(bad)

    def test_output_saves_to_new_diagnostic_folder_only(self):
        result = self.run_with_mocks()
        with TemporaryDirectory() as temp:
            path = module.save_comparison(result, self.config, temp)
            self.assertEqual(path.name, "bc_training_matched")
            self.assertTrue((path / "comparison_metrics.csv").is_file())
            self.assertTrue((path / "provenance.json").is_file())
            for name in module.PORTFOLIOS:
                self.assertTrue((path / name / "backtest_daily.csv").is_file())
                self.assertTrue((path / name / "target_weights.csv").is_file())
                self.assertTrue((path / name / "execution_schedule.csv").is_file())


if __name__ == "__main__":
    import unittest
    unittest.main()
