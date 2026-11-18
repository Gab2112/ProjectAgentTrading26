"""Tests of the execution engine, the backtest interface and the metrics.

Covered properties:
- a decision taken at the close of day t is executed at the open of day t+1;
- the per-asset signal convention: +1 earns the asset return, -1 earns minus
  the asset return, 0 earns nothing;
- transaction costs, short borrowing fees and collateral are accounted for;
- prices after a date cannot change the NAV up to that date (no look-ahead);
- leverage above 100% gross exposure is rejected;
- the reporting metrics match independent formulas.
"""

import unittest

import numpy as np
import pandas as pd

from src.backtest import run_signal_backtest, run_weight_backtest
from src.evaluator import calculate_metrics
from src.execution_engine import simulate_longshort_daily
from src.utils import ASSETS

CONFIG = {
    "execution": {
        "initial_capital": 100_000.0,
        "commission_bps": 0.0,
        "slippage_bps": 0.0,
        "annual_borrow_rate": 0.0,
        "additional_margin_rate": 0.5,
    }
}


def market(prices_spy_open, prices_spy_close):
    """Daily panel in the market_daily.csv format; other ETFs are constant."""
    dates = pd.bdate_range("2020-01-06", periods=len(prices_spy_open), name="Date")
    data = {}
    for asset in ASSETS:
        if asset == "SPY":
            data["SPY_open"], data["SPY_close"] = prices_spy_open, prices_spy_close
        else:
            data[f"{asset}_open"] = [50.0] * len(dates)
            data[f"{asset}_close"] = [50.0] * len(dates)
    return pd.DataFrame(data, index=dates)


class TestExecutionTiming(unittest.TestCase):
    def setUp(self):
        # The close of day 0 jumps by 10%: a strategy that could trade at the
        # same close would capture it; next-open execution must not.
        self.panel = market([100, 110, 110, 121], [110, 110, 121, 121])

    def test_decision_is_executed_at_next_open(self):
        signals = pd.Series([1], index=self.panel.index[:1])
        nav = run_signal_backtest(signals, "SPY", self.panel, CONFIG, start=self.panel.index[0])
        self.assertEqual(nav["executed"].tolist(), [False, True, False, False])
        # Bought at the open of day 1 (110); day-1 close is 110, so no gain yet.
        self.assertAlmostEqual(nav["nav"].iloc[1], 100_000.0)
        self.assertAlmostEqual(nav["nav"].iloc[-1], 110_000.0)

    def test_long_and_short_returns_follow_the_signal_convention(self):
        long_nav = run_signal_backtest(pd.Series([1], index=self.panel.index[:1]), "SPY", self.panel, CONFIG,
                                       start=self.panel.index[0])
        short_nav = run_signal_backtest(pd.Series([-1], index=self.panel.index[:1]), "SPY", self.panel, CONFIG,
                                        start=self.panel.index[0])
        flat_nav = run_signal_backtest(pd.Series([0], index=self.panel.index[:1]), "SPY", self.panel, CONFIG,
                                       start=self.panel.index[0])
        asset_return = 121 / 110 - 1
        self.assertAlmostEqual(long_nav["nav"].iloc[-1] / 100_000 - 1, asset_return)
        self.assertAlmostEqual(short_nav["nav"].iloc[-1] / 100_000 - 1, -asset_return)
        self.assertAlmostEqual(flat_nav["nav"].iloc[-1], 100_000.0)

    def test_future_prices_do_not_change_past_nav(self):
        signals = pd.Series([1, 0], index=self.panel.index[[0, 2]])
        base = run_signal_backtest(signals, "SPY", self.panel, CONFIG, start=self.panel.index[0])
        altered = self.panel.copy()
        altered.loc[altered.index[3], ["SPY_open", "SPY_close"]] = [500.0, 500.0]
        changed = run_signal_backtest(signals, "SPY", altered, CONFIG, start=self.panel.index[0])
        pd.testing.assert_series_equal(base["nav"].iloc[:3], changed["nav"].iloc[:3])

    def test_transaction_costs_are_charged_on_traded_notional(self):
        config = {"execution": {**CONFIG["execution"], "commission_bps": 2.0, "slippage_bps": 1.0}}
        nav = run_signal_backtest(pd.Series([1], index=self.panel.index[:1]), "SPY", self.panel, config,
                                  start=self.panel.index[0])
        cost = nav["transaction_cost"].iloc[1]
        self.assertAlmostEqual(cost / (nav["nav"].iloc[1] + cost), 3e-4, places=6)

    def test_leverage_is_rejected(self):
        weights = pd.DataFrame([[0.8, 0.4, 0.0, 0.0]], index=self.panel.index[:1], columns=list(ASSETS))
        with self.assertRaises(ValueError):
            run_weight_backtest(weights, self.panel, CONFIG, start=self.panel.index[0])


class TestShortAccounting(unittest.TestCase):
    def setUp(self):
        self.dates = pd.to_datetime(["2016-01-07", "2016-01-08", "2016-01-11", "2016-01-12"])
        self.opens = pd.DataFrame(100.0, index=self.dates, columns=list(ASSETS))
        self.closes = self.opens.copy()

    def simulate(self, weights, borrow_rate=0.0):
        target = pd.DataFrame([weights], index=self.dates[:1], columns=list(ASSETS))
        return simulate_longshort_daily(self.opens, self.closes, target, commission_bps=0.0,
                                        slippage_bps=0.0, annual_borrow_rate=borrow_rate)

    def test_short_proceeds_enter_the_cash_ledger(self):
        result = self.simulate([-0.4, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(result.iloc[1]["cash"], 140_000.0)
        self.assertAlmostEqual(result.iloc[1]["SPY_weight"], -0.4)
        self.assertAlmostEqual(result.iloc[1]["nav"], 100_000.0)

    def test_borrow_fee_uses_act_365_over_the_weekend(self):
        result = self.simulate([-0.4, 0.0, 0.0, 0.0], borrow_rate=0.01)
        self.assertAlmostEqual(result.loc[self.dates[2], "borrow_fee"], 40_000 * 0.01 * 3 / 365, places=8)

    def test_margin_shortfall_is_rejected(self):
        # A 50% rise leaves 200,000 of cash against 225,000 of required collateral.
        self.opens.loc[self.dates[2]:, "SPY"] = 150.0
        self.closes.loc[self.dates[2]:, "SPY"] = 150.0
        with self.assertRaisesRegex(RuntimeError, "Margin shortfall"):
            self.simulate([-1.0, 0.0, 0.0, 0.0])


class TestMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        dates = pd.bdate_range("2018-01-01", periods=600)
        self.returns = pd.Series(rng.normal(4e-4, 0.01, len(dates)), index=dates)
        self.market = pd.Series(rng.normal(3e-4, 0.012, len(dates)), index=dates)

    def test_metrics_match_independent_formulas(self):
        m = calculate_metrics(self.returns, 0.0, 252)
        r = self.returns.to_numpy()
        wealth = np.r_[1.0, np.cumprod(1 + r)]
        self.assertAlmostEqual(m["cagr"], wealth[-1] ** (252 / len(r)) - 1, places=12)
        self.assertAlmostEqual(m["sharpe_ratio"], r.mean() / r.std(ddof=1) * np.sqrt(252), places=12)
        self.assertAlmostEqual(m["max_drawdown"], (wealth / np.maximum.accumulate(wealth) - 1).min(), places=12)
        self.assertAlmostEqual(m["profit_factor"], r[r > 0].sum() / -r[r < 0].sum(), places=12)

    def test_risk_free_rate_lowers_the_sharpe_ratio(self):
        self.assertLess(calculate_metrics(self.returns, 0.03, 252)["sharpe_ratio"],
                        calculate_metrics(self.returns, 0.0, 252)["sharpe_ratio"])

    def test_beta_of_the_market_to_itself_is_one(self):
        m = calculate_metrics(self.market, 0.0, 252, market_returns=self.market)
        self.assertAlmostEqual(m["beta_to_market"], 1.0, places=12)
        self.assertAlmostEqual(m["correlation_to_market"], 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
