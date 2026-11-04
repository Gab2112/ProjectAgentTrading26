"""Targeted A/B/C comparison checks; synthetic data, no historical holdout access."""
from __future__ import annotations

from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.run_abc_training_comparison as module
from src.utils import ASSETS


class TestABCMatchedTrainingComparison(TestCase):
    def setUp(self):
        self.days = pd.bdate_range("2014-01-06", periods=11, name="Date")
        self.decisions = self.days[[0, 5, 10]]
        self.warmup = pd.bdate_range(end=self.days[0] - pd.offsets.BDay(1),
                                     periods=160, name="Date")
        dates = self.warmup.append(self.days)
        self.market = pd.DataFrame(index=dates)
        for a in ASSETS:
            prices = np.full(len(dates), 100.0)
            if a == "SPY":
                prices[-len(self.days):] = 100.0 + np.arange(len(self.days))
            self.market[f"{a}_open"] = prices
            self.market[f"{a}_close"] = prices
        self.config = {
            "validation": {"development_only": True},
            "evaluation": {"evaluate_oos": False, "risk_free_rate": 0.0,
                           "annualization_factor": 252},
            "execution": {"execution_rule": "next_available_trading_open",
                          "signal_frequency": "weekly", "cost_basis": "traded_notional",
                          "initial_capital": 100000.0, "commission_bps": 2.0,
                          "slippage_bps": 1.0},
            "strategies": {"B": {"cash_interest_rate": 0.0,
                                 "annual_borrow_rate": 0.01,
                                 "additional_margin_rate": 0.50},
                            "A1": {"enabled": True}, "A2": {"enabled": True}},
            "benchmarks": {
                "spy_buy_and_hold": {"weights": {"SPY": 1.0}, "rebalance": "none"},
                "sixty_forty": {"weights": {"SPY": .6, "IEF": .4},
                                "rebalance": "weekly"},
                "equal_weight": {"weights": dict.fromkeys(ASSETS, .25),
                                 "rebalance": "weekly"},
                "equal_weight_buy_and_hold": {"weights": dict.fromkeys(ASSETS, .25),
                                              "rebalance": "none"},
            },
            "paths": {"processed_data": "data/processed", "results": "data/results"},
        }
        self.b_names = ("B_v1", "B_v2", "C_v1", "C_v2")
        base_series = pd.Series(0.0, index=self.days)
        from src.evaluator import calculate_metrics
        metric = calculate_metrics(base_series, 0.0, 252)
        self.bc_result = {
            "summary": pd.DataFrame.from_records([
                {"strategy": name, **metric, "final_nav": 100000.0,
                 "initial_capital": 100000.0, "decisions": 3,
                 "scheduled_rebalances": 2, "nonzero_trades": 0,
                 "total_turnover": 0.0, "transaction_costs": 0.0,
                 "borrow_fees": 0.0, "mean_gross_exposure": 0.0,
                 "mean_net_exposure": 0.0, "mean_short_exposure": 0.0,
                 "mean_cash_fraction_longflat": np.nan}
                for name in self.b_names]).set_index("strategy"),
            "target_weights": {
                name: pd.DataFrame(0.0, index=self.decisions,
                                   columns=ASSETS) for name in self.b_names},
            "backtests": {name: pd.DataFrame(index=self.days)
                          for name in self.b_names},
            "execution_schedules": {name: pd.DataFrame(index=self.days[1:3])
                                    for name in self.b_names},
        }
        self.a2_calls = []

    def fake_a1(self, config, decisions, tickers):
        result = {}
        for name, spec in config["benchmarks"].items():
            dates = decisions if spec["rebalance"] == "weekly" else decisions[:1]
            allocation = [float(spec["weights"].get(a, 0.0)) for a in tickers]
            result[name] = pd.DataFrame([allocation] * len(dates),
                                        index=dates, columns=tickers)
        return result

    def fake_a2(self, daily, decisions, config, tickers):
        self.a2_calls.append((daily.copy(), decisions.copy()))
        frame = pd.DataFrame(.25, index=decisions, columns=tickers)
        report = pd.DataFrame({"stato": ["equal_weight"] * len(decisions)},
                              index=decisions)
        return frame, report

    @staticmethod
    def synthetic_simulate(*, open_prices, close_prices, target_weights,
                           initial_capital, commission_bps, slippage_bps,
                           annual_borrow_rate, additional_margin_rate):
        """Local wrapper test double. Production uses the tested shared engine."""
        from src.execution_engine import schedule_next_open_executions
        scheduled = schedule_next_open_executions(target_weights, open_prices.index)
        shares = np.zeros(len(ASSETS), dtype=float)
        cash = float(initial_capital)
        previous_nav = cash
        rows = []
        for date in open_prices.index:
            p_open = open_prices.loc[date].to_numpy(dtype=float)
            p_close = close_prices.loc[date].to_numpy(dtype=float)
            cost, turnover, executed = 0.0, 0.0, False
            if date in scheduled.index:
                pre_nav = cash + float(shares @ p_open)
                weights = scheduled.loc[date].to_numpy(dtype=float)
                traded = float(np.abs(pre_nav * weights - shares * p_open).sum())
                cost = traded * (commission_bps + slippage_bps) / 10000.0
                new_nav = pre_nav - cost
                shares = new_nav * weights / p_open
                cash = new_nav - float(shares @ p_open)
                turnover = traded / pre_nav
                executed = traded > 1e-10
            nav = cash + float(shares @ p_close)
            gross = float(np.abs(shares * p_close).sum() / nav)
            rows.append({
                "Date": date, "nav": nav, "daily_return": nav / previous_nav - 1.0,
                "cash": cash, "transaction_cost": cost, "borrow_fee": 0.0,
                "turnover": turnover, "executed": executed,
                "gross_exposure": gross, "net_exposure": gross,
                "short_exposure": 0.0, "margin_shortfall": 0.0,
            })
            previous_nav = nav
        return pd.DataFrame(rows).set_index("Date")

    def run_case(self, *, market=None, a1=None, a2=None, bc=None, config=None):
        with patch.object(module, "_guard_training", return_value=(self.days[0], self.days[-1])), \
             patch.object(module, "_validate_universe", return_value=list(ASSETS)), \
             patch.object(module, "run_training_comparison", return_value=(bc or self.bc_result)), \
             patch.object(module, "load_dataset", return_value=(self.market if market is None else market)), \
             patch.object(module, "build_a1_weights", side_effect=(a1 or self.fake_a1)), \
             patch.object(module, "build_a2_weights", side_effect=(a2 or self.fake_a2)), \
             patch.object(module, "simulate_longshort_daily", side_effect=self.synthetic_simulate):
            return module.run_abc_training_comparison(config or self.config, ".")

    def test_nine_strategies_and_same_daily_calendar(self):
        result = self.run_case()
        self.assertEqual(len(result["summary"]), 9)
        self.assertEqual(len(self.a2_calls), 1)
        for backtest in result["backtests"].values():
            self.assertTrue(backtest.index.equals(self.days))
        self.assertEqual(len(result["target_weights"]["A1_spy_buy_and_hold"]), 1)
        self.assertEqual(len(result["target_weights"]["A1_equal_weight_buy_and_hold"]), 1)
        self.assertEqual(len(result["target_weights"]["A2_volatility_control"]), 3)
        self.assertEqual(len(result["execution_schedules"]["A1_spy_buy_and_hold"]), 1)
        self.assertEqual(len(result["execution_schedules"]["A1_equal_weight"]), 2)

    def test_buy_and_hold_not_implicitly_rebalanced(self):
        result = self.run_case()
        name = "A1_spy_buy_and_hold"
        self.assertEqual(result["summary"].loc[name, "scheduled_rebalances"], 1)
        self.assertEqual(result["execution_schedules"][name].index[0], self.days[1])
        self.assertAlmostEqual(result["backtests"][name]["nav"].iloc[0], 100000.0)
        self.assertGreater(result["backtests"][name]["nav"].iloc[-1], 100000.0)

    def test_a2_uses_2009_warmup_but_never_future(self):
        next_date = self.days[-1] + pd.offsets.BDay(1)
        future = pd.DataFrame(999.0, index=pd.DatetimeIndex([next_date], name="Date"),
                              columns=self.market.columns)
        result = self.run_case(market=pd.concat([self.market, future]))
        self.assertEqual(self.a2_calls[0][0].index.min(), self.warmup[0])
        self.assertEqual(self.a2_calls[0][0].index.max(), self.days[-1])
        self.assertEqual(result["last_accounted_session"], self.days[-1])

    def test_reject_a2_wrong_decision_dates(self):
        def bad_a2(daily, decisions, config, tickers):
            frame = pd.DataFrame(.25, index=decisions[:-1], columns=tickers)
            return frame, pd.DataFrame(index=decisions[:-1])
        with self.assertRaisesRegex(ValueError, "target dates"):
            self.run_case(a2=bad_a2)

    def test_reject_short_or_incomplete_a1(self):
        def bad_a1(config, decisions, tickers):
            frames = self.fake_a1(config, decisions, tickers)
            frames["sixty_forty"].iloc[0, :] = [-.1, .4, .3, .4]
            return frames
        with self.assertRaisesRegex(ValueError, "long-only"):
            self.run_case(a1=bad_a1)

    def test_refuse_disabled_a2(self):
        config = {**self.config, "strategies": {**self.config["strategies"],
                  "A2": {"enabled": False}}}
        with self.assertRaisesRegex(ValueError, "A2.*not enabled"):
            self.run_case(config=config)

    def test_refuse_market_calendar_mismatch(self):
        bad = self.market.drop(index=self.days[4])
        with self.assertRaisesRegex(ValueError, "Daily A/B/C calendars"):
            self.run_case(market=bad)

    def test_save_new_outputs_and_preserve_bc_metrics(self):
        result = self.run_case()
        self.assertAlmostEqual(result["summary"].loc["C_v2", "final_nav"], 100000.0)
        with TemporaryDirectory() as tmp:
            folder = module.save_abc_comparison(result, self.config, tmp)
            self.assertEqual(folder.name, "abc_training_matched")
            self.assertTrue((folder / "comparison_metrics.csv").is_file())
            self.assertTrue((folder / "a2_risk_diagnostics.csv").is_file())
            self.assertTrue((folder / "provenance.json").is_file())
            self.assertTrue((folder / "A1_spy_buy_and_hold" / "backtest_daily.csv").is_file())
            self.assertTrue((folder / "C_v2" / "target_weights.csv").is_file())


if __name__ == "__main__":
    import unittest
    unittest.main()
