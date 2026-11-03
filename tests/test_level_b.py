"""Audit di equivalenza A/B e invarianti del Level B (solo dati sintetici).

Inserire questo file in tests/test_level_b.py nel repository ProjectAgentTrading26.
Non legge CSV raw, non scrive output e non accede a validation/OOS.

La comparazione A/B e' valida SOLO per pesi long-only che sommano esattamente
al 100%, stessi prezzi, stessi costi e nessun costo di prestito titoli.
Il motore A non supporta cash intenzionale ne' posizioni short.
"""

import unittest

import numpy as np
import pandas as pd

from src.evaluator import calculate_metrics
from src.longshort_backtester import (
    schedule_next_open_executions,
    simulate_longshort_daily,
)
from src.longshort_backtester import simula_portafoglio
from src.regime_allocator import (
    ASSETS,
    REGIME_WEIGHTS,
    build_regime_weight_panel,
    apply_signed_volatility_cap,
)


class SyntheticMarket:
    """Prezzi Open/Close deterministici, senza accesso a dati reali."""

    @staticmethod
    def make(n=48):
        dates = pd.bdate_range("2015-01-05", periods=n, name="Date")
        t = np.arange(n, dtype=float)
        opens = pd.DataFrame(
            {
                "SPY": 100 * np.exp(0.0013 * t + 0.014 * np.sin(t / 3.5)),
                "IEF": 95 * np.exp(-0.0003 * t + 0.006 * np.cos(t / 4.2)),
                "GLD": 122 * np.exp(0.0005 * t + 0.011 * np.sin(t / 5.6 + 1)),
                "SHY": 82 * np.exp(0.0001 * t + 0.002 * np.cos(t / 7.0)),
            },
            index=dates,
        )
        intraday = pd.DataFrame(
            {
                a: 1 + 0.0017 * np.sin(t / (3 + j) + j)
                for j, a in enumerate(ASSETS)
            },
            index=dates,
        )
        closes = opens * intraday
        return opens, closes

    @staticmethod
    def target_panel(dates):
        # Un target iniziale, successivi ribilanciamenti e uno NON eseguibile.
        at = [0, 5, 14, 27, 40, len(dates) - 1]
        weights = [
            [0.25, 0.25, 0.25, 0.25],
            [0.55, 0.15, 0.10, 0.20],
            [0.10, 0.50, 0.15, 0.25],
            [0.20, 0.10, 0.40, 0.30],
            [0.40, 0.20, 0.30, 0.10],
            [0.00, 0.00, 0.00, 1.00],
        ]
        return pd.DataFrame(weights, index=dates[at], columns=ASSETS)


class TestEquivalenceAB(unittest.TestCase):
    """Confronto numerico dei due motori nella loro zona comune."""

    def setUp(self):
        self.opens, self.closes = SyntheticMarket.make()
        self.targets = SyntheticMarket.target_panel(self.opens.index)

    def check_equivalent(self, commission_bps, slippage_bps):
        a = simula_portafoglio(
            aperture=self.opens,
            chiusure=self.closes,
            pesi_target=self.targets,
            capitale_iniziale=100_000.0,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
        )
        b = simulate_longshort_daily(
            open_prices=self.opens,
            close_prices=self.closes,
            target_weights=self.targets,
            initial_capital=100_000.0,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            annual_borrow_rate=0.0,
        )
        self.assertTrue(a.index.equals(b.index))
        for ac, bc in (
            ("nav", "nav"),
            ("rendimento_netto", "daily_return"),
            ("turnover", "turnover"),
            ("costo_transazione", "transaction_cost"),
        ):
            with self.subTest(metric=ac, fees=(commission_bps, slippage_bps)):
                np.testing.assert_allclose(
                    a[ac].to_numpy(), b[bc].to_numpy(),
                    rtol=1e-9, atol=1e-7 if ac == "nav" else 1e-10,
                )
        for asset in ASSETS:
            np.testing.assert_allclose(
                a[f"{asset}_peso_effettivo"].to_numpy(),
                b[f"{asset}_weight"].to_numpy(),
                rtol=1e-9, atol=1e-9,
                err_msg=f"Pesi effettivi differenti per {asset}",
            )
        # Anche le metriche devono coincidere quando il motore e' equivalente.
        for metric in (
            "cumulative_return", "cagr", "annualized_volatility",
            "sharpe_ratio", "sortino_ratio", "max_drawdown",
            "profit_factor_daily_returns",
        ):
            ma = calculate_metrics(a["rendimento_netto"], 0.0, 252)
            mb = calculate_metrics(b["daily_return"], 0.0, 252)
            self.assertAlmostEqual(ma[metric], mb[metric], places=9, msg=metric)
        # La serie NAV deve concordare anche con l'identita' dei rendimenti.
        np.testing.assert_allclose(
            np.prod(1 + b["daily_return"].to_numpy()),
            b["nav"].iloc[-1] / 100_000.0,
            rtol=1e-12, atol=1e-12,
        )

    def test_equivalence_without_costs(self):
        self.check_equivalent(0.0, 0.0)

    def test_equivalence_with_project_costs(self):
        self.check_equivalent(2.0, 1.0)

    def test_equivalence_with_higher_costs(self):
        self.check_equivalent(8.0, 3.0)

    def test_zero_notional_execution_flag_differs_but_not_nav(self):
        # Diagnostica di audit: A conta una decisione programmata come operazione;
        # B registra executed solo se il nozionale negoziato e' > 0.
        # Questa differenza NON modifica NAV, costi o rendimenti.
        dates = pd.bdate_range("2015-01-05", periods=6, name="Date")
        p = pd.DataFrame(100.0, index=dates, columns=ASSETS)
        targets = pd.DataFrame([[.25] * 4, [.25] * 4],
                               index=dates[[0, 2]], columns=ASSETS)
        a = simula_portafoglio(p, p, targets)
        b = simulate_longshort_daily(p, p, targets, annual_borrow_rate=0.0)
        np.testing.assert_allclose(a["nav"], b["nav"], atol=1e-9)
        self.assertAlmostEqual(a.loc[dates[3], "costo_transazione"], 0.0, places=9)
        self.assertAlmostEqual(b.loc[dates[3], "transaction_cost"], 0.0, places=9)
        self.assertTrue(bool(a.loc[dates[3], "operazione_eseguita"]))
        self.assertFalse(bool(b.loc[dates[3], "executed"]))

    def test_next_open_not_same_close(self):
        dates = pd.bdate_range("2016-01-04", periods=5, name="Date")
        o = pd.DataFrame(100.0, index=dates, columns=ASSETS)
        c = o.copy()
        o.loc[dates[1], "SPY"] = 110.0
        c.loc[dates[1], "SPY"] = 120.0
        o.loc[dates[2]:, "SPY"] = 120.0
        c.loc[dates[2]:, "SPY"] = 120.0
        w = pd.DataFrame([[1.0, 0.0, 0.0, 0.0]], index=dates[:1], columns=ASSETS)
        result = simulate_longshort_daily(o, c, w, commission_bps=0, slippage_bps=0)
        self.assertAlmostEqual(result.loc[dates[0], "nav"], 100_000.0)
        self.assertFalse(result.loc[dates[0], "executed"])
        self.assertTrue(result.loc[dates[1], "executed"])
        self.assertAlmostEqual(result.loc[dates[1], "nav"], 100_000 * 120 / 110)

    def test_end_of_training_decision_does_not_execute(self):
        d = self.opens.index
        early = pd.DataFrame([[.25, .25, .25, .25]], index=d[:1], columns=ASSETS)
        with_final = pd.concat([
            early,
            pd.DataFrame([[1.0, 0.0, 0.0, 0.0]], index=d[-1:], columns=ASSETS),
        ])
        a = simulate_longshort_daily(self.opens, self.closes, early)
        b = simulate_longshort_daily(self.opens, self.closes, with_final)
        np.testing.assert_allclose(a["nav"], b["nav"], rtol=0, atol=1e-9)
        self.assertEqual(int(a["executed"].sum()), int(b["executed"].sum()))

    def test_future_prices_cannot_change_past_nav(self):
        old = simulate_longshort_daily(self.opens, self.closes, self.targets)
        later_open = self.opens.copy()
        later_close = self.closes.copy()
        later_open.iloc[30:] *= 1.25
        later_close.iloc[30:] *= 0.82
        changed = simulate_longshort_daily(later_open, later_close, self.targets)
        np.testing.assert_allclose(old["nav"].iloc[:30], changed["nav"].iloc[:30], rtol=0, atol=1e-8)

    def test_future_targets_cannot_change_past_nav(self):
        early = self.targets.iloc[:2]
        late = self.targets.iloc[:5]
        a = simulate_longshort_daily(self.opens, self.closes, early)
        b = simulate_longshort_daily(self.opens, self.closes, late)
        cutoff = self.opens.index.get_loc(late.index[2]) + 1
        np.testing.assert_allclose(a["nav"].iloc[:cutoff], b["nav"].iloc[:cutoff], rtol=0, atol=1e-8)


class TestLevelBAccounting(unittest.TestCase):
    def setUp(self):
        self.dates = pd.to_datetime(["2016-01-07", "2016-01-08", "2016-01-11", "2016-01-12"])
        self.opens = pd.DataFrame(100.0, index=self.dates, columns=ASSETS)
        self.closes = self.opens.copy()

    def run_b(self, weights, borrow_rate=0.0):
        w = pd.DataFrame([weights], index=self.dates[:1], columns=ASSETS)
        return simulate_longshort_daily(
            self.opens, self.closes, w,
            commission_bps=0.0, slippage_bps=0.0,
            annual_borrow_rate=borrow_rate,
        )

    def test_flat_cash_not_implicitly_invested(self):
        b = self.run_b([0.3, 0.0, 0.2, 0.0])
        self.assertAlmostEqual(b.iloc[1]["cash"], 50_000.0)
        self.assertAlmostEqual(b.iloc[1]["net_exposure"], 0.5)
        self.assertAlmostEqual(b.iloc[1]["nav"], 100_000.0)
        self.assertAlmostEqual(b["nav"].iloc[-1], 100_000.0)

    def test_short_positions_and_cash_ledger(self):
        b = self.run_b([-0.4, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(b.iloc[1]["cash"], 140_000.0)
        self.assertAlmostEqual(b.iloc[1]["SPY_weight"], -0.4)
        self.assertAlmostEqual(b.iloc[1]["gross_exposure"], 0.4)
        self.assertAlmostEqual(b.iloc[1]["nav"], 100_000.0)

    def test_short_borrow_fee_act_365_weekend(self):
        b = self.run_b([-0.4, 0.0, 0.0, 0.0], borrow_rate=0.01)
        expected_weekend = 40_000 * 0.01 * 3 / 365
        self.assertAlmostEqual(b.loc[self.dates[2], "borrow_fee"], expected_weekend, places=8)
        self.assertAlmostEqual(b.loc[self.dates[2], "nav"], 100_000 - expected_weekend, places=8)
        self.assertAlmostEqual(b.loc[self.dates[1], "borrow_fee"], 0.0)

    def test_margin_shortfall_is_rejected(self):
        self.opens.loc[self.dates[2]:, "SPY"] = 150.0
        self.closes.loc[self.dates[2]:, "SPY"] = 150.0
        with self.assertRaisesRegex(RuntimeError, "Margin shortfall"):
            self.run_b([-1.0, 0.0, 0.0, 0.0])

    def test_no_leverage_or_invalid_targets(self):
        for bad in ([1.1, 0, 0, 0], [-0.8, -0.5, 0, 0], [np.nan, 0, 0, 0]):
            with self.subTest(weights=bad):
                with self.assertRaises(ValueError):
                    self.run_b(bad)

    def test_nav_daily_returns_and_weight_sum_consistent(self):
        b = self.run_b([.2, -.3, .1, .0])
        lagged = b["nav"].shift(1)
        lagged.iloc[0] = 100_000
        np.testing.assert_allclose(b["daily_return"], b["nav"] / lagged - 1, atol=1e-12)
        np.testing.assert_allclose(
            b["net_exposure"], b[[f"{a}_weight" for a in ASSETS]].sum(axis=1), atol=1e-12,
        )


class TestRegimeAllocationAndMetrics(unittest.TestCase):
    def test_regime_budgets_are_explicit_and_complete(self):
        dates = pd.bdate_range("2015-02-02", periods=4)
        regimes = pd.Series(list(REGIME_WEIGHTS), index=dates)
        result = build_regime_weight_panel(regimes)
        self.assertTrue(result.index.equals(dates))
        for regime, date in zip(regimes, dates):
            np.testing.assert_allclose(result.loc[date], REGIME_WEIGHTS[regime], atol=1e-14)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-14)

    def test_volatility_cap_never_increases_exposure_and_uses_past(self):
        days = pd.bdate_range("2015-01-05", periods=175)
        t = np.arange(len(days))
        returns = pd.DataFrame(
            {"SPY": 0.05 * np.where(t % 2 == 0, 1, -1),
             "IEF": 0.008 * np.sin(t),
             "GLD": 0.02 * np.cos(t),
             "SHY": 0.0001 * np.sin(t)},
            index=days,
        )
        w = pd.Series([.7, .1, .1, .1], index=ASSETS)
        date = days[140]
        scaled, risk = apply_signed_volatility_cap(w, returns, date, window=126, target_volatility=.08)
        self.assertLess(risk["scaling_factor"], 1)
        self.assertLessEqual(risk["vol_after"], .08 + 1e-12)
        self.assertLessEqual(scaled.abs().sum(), w.abs().sum() + 1e-12)
        future_changed = returns.copy()
        future_changed.loc[future_changed.index > date] *= 100
        stable, risk_stable = apply_signed_volatility_cap(w, future_changed, date, window=126, target_volatility=.08)
        np.testing.assert_allclose(scaled, stable, atol=1e-14)
        self.assertAlmostEqual(risk["vol_after"], risk_stable["vol_after"])

    def test_missing_126_session_warmup_rejected(self):
        dates = pd.bdate_range("2015-01-05", periods=40)
        returns = pd.DataFrame(0.001, index=dates, columns=ASSETS)
        w = pd.Series([.25] * 4, index=ASSETS)
        with self.assertRaisesRegex(ValueError, "Insufficient return history"):
            apply_signed_volatility_cap(w, returns, dates[-1], window=126)

    def test_common_metric_formulas_agree_with_nav(self):
        market_o, market_c = SyntheticMarket.make()
        target = SyntheticMarket.target_panel(market_o.index)
        result = simulate_longshort_daily(market_o, market_c, target)
        rf = pd.Series(0.0, index=result.index)
        m = calculate_metrics(result["daily_return"], rf, 252)
        expected_cagr = (result["nav"].iloc[-1] / 100_000) ** (252 / len(result)) - 1
        self.assertAlmostEqual(m["cagr"], expected_cagr, places=12)
        self.assertAlmostEqual(m["cumulative_return"], result["nav"].iloc[-1] / 100_000 - 1, places=12)
        nav = np.r_[100_000, result["nav"].to_numpy()]
        expected_mdd = (nav / np.maximum.accumulate(nav) - 1).min()
        self.assertAlmostEqual(m["max_drawdown"], expected_mdd, places=12)


if __name__ == "__main__":
    unittest.main()
