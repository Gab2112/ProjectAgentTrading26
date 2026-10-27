"""Test economici e operativi della pipeline Multi-Asset, Fase A (A1/A2).

La suite non legge prezzi storici reali, validation reale o OOS reale:
usa solamente configurazioni e dati di mercato sintetici. Il solo
``config.yaml`` del progetto puo' essere letto per verificarne i vincoli.

Sono separati:
- invarianti dei benchmark A1 e dell'allocazione A2;
- esecuzione al prossimo Open, quantita', NAV, turnover e costi;
- correttezza delle metriche giornaliere;
- integrazione tra portfolio, backtester ed evaluator;
- isolamento del training rispetto a validation e OOS.

I test numerici A2 richiedono CVXPY/CLARABEL. Se non installati, solo quei
casi sono saltati; i test della pipeline e del backtester restano eseguibili.

Esecuzione dalla radice del repository:
    python3 -m unittest discover -s tests -p 'test_*.py' -v
"""

from __future__ import annotations

from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from src.backtester import run_backtest
from src.data_loader import load_dataset
from src.evaluator import calculate_metrics, run_evaluation
from src.multiasset_backtester import simula_portafoglio
from src.portfolio import (
    _get_a2_parameters,
    build_a1_weights,
    build_a2_weights,
    run_portfolio,
)
from src.time_split import get_time_windows, is_oos_enabled
from src.utils import get_tickers


ASSETS = ("SPY", "IEF", "GLD", "SHY")
HAS_CVXPY = importlib.util.find_spec("cvxpy") is not None
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_config() -> dict:
    """Restituisce una configurazione sintetica coerente con la Fase A."""
    equal = {ticker: 0.25 for ticker in ASSETS}
    return {
        "universe": {
            "traded_assets": {
                "us_equities": {"ticker": "SPY"},
                "intermediate_treasuries": {"ticker": "IEF"},
                "gold": {"ticker": "GLD"},
                "short_treasuries": {"ticker": "SHY"},
            },
        },
        "portfolio": {
            "long_only": True,
            "fully_invested": True,
            "leverage": 1.0,
            "allow_short": False,
        },
        "benchmarks": {
            "spy_buy_and_hold": {
                "target_weights": {"SPY": 1.0},
                "rebalance": "none",
            },
            "sixty_forty": {
                "target_weights": {"SPY": 0.6, "IEF": 0.4},
                "rebalance": "weekly",
            },
            "equal_weight": {
                "target_weights": equal,
                "rebalance": "weekly",
            },
            "equal_weight_buy_and_hold": {
                "target_weights": equal,
                "rebalance": "none",
            },
        },
        "strategies": {
            "A1": {
                "enabled": True,
                "benchmark_names": [
                    "spy_buy_and_hold", "sixty_forty", "equal_weight",
                    "equal_weight_buy_and_hold",
                ],
            },
            "A2": {
                "enabled": True,
                "reference_benchmark": "equal_weight",
                "covariance_lookback_trading_days": 126,
                "covariance_annualization_factor": 252,
                "maximum_estimated_annual_volatility": 0.08,
                "objective": "minimum_squared_distance_from_reference",
                "optimization_library": "cvxpy",
                "solver": "CLARABEL",
                "infeasible_fallback": "minimum_variance",
            },
            "B": {"enabled": False},
            "C": {"enabled": False},
        },
        "validation": {
            "train_start": "2020-06-01",
            "train_end": "2020-06-30",
            "validation_start": "2020-07-01",
            "validation_end": "2020-07-31",
            "oos_start": "2020-08-01",
            "development_only": True,
        },
        "evaluation": {
            "returns_frequency": "daily",
            "annualization_factor": 252,
            "risk_free_rate_annual": 0.0,
            "evaluate_oos": False,
        },
        "execution": {
            "signal_frequency": "weekly",
            "execution_rule": "next_available_trading_open",
            "initial_capital": 100000.0,
            "commission_bps": 2.0,
            "slippage_bps": 1.0,
        },
        "paths": {
            "processed_data": "data/processed",
            "signals": "data/signals",
            "results": "data/results",
        },
        "pipeline": {"active_level": "A1", "legacy_treasury_enabled": False},
    }


def make_synthetic_market() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genera Open/Close coerenti e decisioni settimanali, senza Internet."""
    dates = pd.bdate_range("2019-10-01", "2020-08-31", name="Date")
    rng = np.random.default_rng(47)
    innovations = rng.normal(0, 1, size=(len(dates), 4))
    daily_returns = 0.0002 + innovations * np.array([0.006, 0.003, 0.005, 0.0005])

    panel = pd.DataFrame(index=dates)
    for j, ticker in enumerate(ASSETS):
        closes = 100 * np.cumprod(1 + daily_returns[:, j])
        opens = closes * (1 + 0.0002 * np.sin(np.arange(len(dates)) + j))
        panel[f"{ticker}_open"] = opens
        panel[f"{ticker}_close"] = closes

    weekly = dates.to_series().resample("W-FRI").last().dropna()
    decision_dates = pd.DatetimeIndex(weekly.to_numpy(), name="Date")
    decisions = pd.DataFrame(
        {"decision_date": decision_dates}, index=decision_dates
    )
    return panel, decisions


def write_processed_data(root: Path, panel: pd.DataFrame, decisions: pd.DataFrame) -> None:
    """Scrive soltanto i due CSV sintetici richiesti da portfolio.py."""
    destination = root / "data" / "processed"
    destination.mkdir(parents=True, exist_ok=True)
    panel.to_csv(destination / "market_daily.csv", index_label="Date")
    decisions.to_csv(destination / "weekly_decision_dates.csv", index_label="Date")


def fake_a2_allocator(captured: list[pd.DataFrame] | None = None):
    """Sostituisce esclusivamente il risolutore nei test di integrazione."""
    module = types.ModuleType("src.multiasset_allocator")

    def calculate(returns, finestra=126, volatilita_massima=0.08):
        if captured is not None:
            captured.append(returns.copy(deep=True))
        if len(returns) != finestra:
            raise AssertionError("L'ottimizzatore non ha ricevuto la finestra richiesta.")
        weights = pd.Series({ticker: 0.25 for ticker in ASSETS})
        info = {
            "stato": "equal_weight",
            "volatilita_ew": 0.03,
            "volatilita_target": 0.03,
            "limite_rispettato": True,
        }
        return weights, info

    module.calcola_pesi_a2 = calculate
    return module


class TestProjectConfiguration(unittest.TestCase):
    """Impedisce che il file YAML locale torni ai parametri Treasury."""

    def test_local_yaml_describes_the_agreed_phase_a(self):
        path = PROJECT_ROOT / "config.yaml"
        if not path.exists():
            self.skipTest("config.yaml non disponibile nel progetto di test.")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(get_tickers(config), list(ASSETS))
        self.assertEqual(
            {name for name, params in config["strategies"].items() if params.get("enabled")},
            {"A1", "A2", "B"},
        )
        self.assertFalse(is_oos_enabled(config))
        self.assertTrue(config["validation"].get("development_only"))
        self.assertEqual(get_time_windows(config)["train_start"], pd.Timestamp("2005-05-20"))
        self.assertEqual(get_time_windows(config)["train_end"], pd.Timestamp("2016-12-31"))
        self.assertEqual(get_time_windows(config)["validation_start"], pd.Timestamp("2017-01-01"))
        self.assertEqual(get_time_windows(config)["oos_start"], pd.Timestamp("2022-01-01"))
        self.assertEqual(_get_a2_parameters(config), (126, 0.08))
        self.assertEqual(config["execution"]["execution_rule"], "next_available_trading_open")
        self.assertEqual(config["evaluation"]["annualization_factor"], 252)

        decisions = pd.DatetimeIndex(pd.to_datetime(["2010-01-08", "2010-01-15"]))
        weights = build_a1_weights(config, decisions, list(ASSETS))
        self.assertEqual(len(weights), 4)
        implemented = Counter()
        for name, frame in weights.items():
            mode = config["benchmarks"][name]["rebalance"]
            vector = tuple(np.round(frame.iloc[0].to_numpy(dtype=float), 8))
            implemented[(mode, vector)] += 1
        self.assertEqual(
            implemented,
            Counter({
                ("none", (1.0, 0.0, 0.0, 0.0)): 1,
                ("weekly", (0.6, 0.4, 0.0, 0.0)): 1,
                ("weekly", (0.25, 0.25, 0.25, 0.25)): 1,
                ("none", (0.25, 0.25, 0.25, 0.25)): 1,
            }),
        )


class TestA1TargetWeights(unittest.TestCase):
    """Pesi long-only, buy-and-hold reale e ribilanciamento settimanale."""

    def setUp(self):
        self.config = make_config()
        self.decisions = pd.bdate_range("2020-06-05", periods=4, freq="W-FRI")
        self.weights = build_a1_weights(self.config, self.decisions, list(ASSETS))

    def test_all_four_benchmarks_have_expected_weights(self):
        self.assertEqual(set(self.weights), set(self.config["benchmarks"]))
        expected = {
            "spy_buy_and_hold": [1.0, 0.0, 0.0, 0.0],
            "sixty_forty": [0.6, 0.4, 0.0, 0.0],
            "equal_weight": [0.25] * 4,
            "equal_weight_buy_and_hold": [0.25] * 4,
        }
        for name, frame in self.weights.items():
            with self.subTest(portfolio=name):
                np.testing.assert_allclose(frame.iloc[0].to_numpy(dtype=float), expected[name])
                self.assertEqual(list(frame.columns), list(ASSETS))
                np.testing.assert_allclose(frame.sum(axis=1).to_numpy(), 1.0)
                self.assertTrue((frame >= 0).all().all())
                self.assertTrue((frame <= 1).all().all())

    def test_buy_and_hold_has_one_decision_weekly_has_four(self):
        for name, frame in self.weights.items():
            expected = 1 if self.config["benchmarks"][name]["rebalance"] == "none" else 4
            with self.subTest(portfolio=name):
                self.assertEqual(len(frame), expected)
                self.assertEqual(frame.index[0], self.decisions[0])

    def test_invalid_portfolio_weights_are_rejected(self):
        changes = (
            ("spy_buy_and_hold", {"SPY": 0.9}),
            ("sixty_forty", {"SPY": 0.6, "IEF": -0.1, "SHY": 0.5}),
            ("equal_weight", {"SPY": 0.25, "IEF": 0.25, "GLD": 0.25, "TLT": 0.25}),
        )
        for name, weights in changes:
            with self.subTest(portfolio=name):
                config = deepcopy(self.config)
                config["benchmarks"][name]["target_weights"] = weights
                with self.assertRaises(ValueError):
                    build_a1_weights(config, self.decisions, list(ASSETS))

    def test_invalid_rebalance_or_duplicate_selection_is_rejected(self):
        config = deepcopy(self.config)
        config["benchmarks"]["sixty_forty"]["rebalance"] = "daily"
        with self.assertRaises(ValueError):
            build_a1_weights(config, self.decisions, list(ASSETS))

        config = deepcopy(self.config)
        config["strategies"]["A1"]["benchmark_names"].append("sixty_forty")
        with self.assertRaises(ValueError):
            build_a1_weights(config, self.decisions, list(ASSETS))

    def test_invalid_a2_yaml_parameters_are_rejected(self):
        for path, value in (
            ("solver", "ECOS"),
            ("covariance_annualization_factor", 52),
            ("infeasible_fallback", "equal_weight"),
            ("maximum_estimated_annual_volatility", -1),
        ):
            with self.subTest(parameter=path):
                config = deepcopy(self.config)
                config["strategies"]["A2"][path] = value
                with self.assertRaises(ValueError):
                    _get_a2_parameters(config)


class TestA2Causality(unittest.TestCase):
    """La finestra dell'ottimizzatore termina alla chiusura della decisione."""

    def setUp(self):
        self.config = make_config()
        self.panel, self.decisions_frame = make_synthetic_market()
        windows = get_time_windows(self.config)
        self.decisions = self.decisions_frame.index[
            (self.decisions_frame.index >= windows["train_start"])
            & (self.decisions_frame.index <= windows["train_end"])
        ]

    def test_only_last_126_past_returns_reach_allocator(self):
        captured = []
        with patch.dict(sys.modules, {"src.multiasset_allocator": fake_a2_allocator(captured)}):
            weights, diagnostics = build_a2_weights(
                self.panel, self.decisions, self.config, list(ASSETS)
            )
        self.assertEqual(len(captured), len(self.decisions))
        self.assertEqual(len(weights), len(self.decisions))
        self.assertTrue((diagnostics["osservazioni_covarianza"] == 126).all())
        for date, history in zip(self.decisions, captured):
            with self.subTest(decision=date):
                self.assertEqual(len(history), 126)
                self.assertEqual(history.index[-1], date)
                self.assertTrue((history.index <= date).all())
                self.assertEqual(list(history.columns), list(ASSETS))
        np.testing.assert_allclose(weights.to_numpy(), 0.25)

    def test_changing_future_prices_does_not_change_past_inputs(self):
        first_run = []
        second_run = []
        altered = self.panel.copy(deep=True)
        # La variazione inizia dopo la prima chiusura decisionale.
        altered.loc[altered.index > self.decisions[0], "SPY_close"] *= 1.05
        with patch.dict(sys.modules, {"src.multiasset_allocator": fake_a2_allocator(first_run)}):
            build_a2_weights(self.panel, self.decisions, self.config, list(ASSETS))
        with patch.dict(sys.modules, {"src.multiasset_allocator": fake_a2_allocator(second_run)}):
            build_a2_weights(altered, self.decisions, self.config, list(ASSETS))

        pd.testing.assert_frame_equal(first_run[0], second_run[0])
        self.assertTrue(any(not a.equals(b) for a, b in zip(first_run[1:], second_run[1:])))

    def test_insufficient_warmup_is_an_error(self):
        shortened = self.panel.loc["2020-05-25":].copy()
        with patch.dict(sys.modules, {"src.multiasset_allocator": fake_a2_allocator()}):
            with self.assertRaisesRegex(ValueError, "Storico insufficiente"):
                build_a2_weights(shortened, self.decisions, self.config, list(ASSETS))


@unittest.skipUnless(HAS_CVXPY, "CVXPY non installato: test numerici A2 da eseguire sul Mac.")
class TestA2NumericalAllocator(unittest.TestCase):
    """Verifica il vero risolutore CVXPY/CLARABEL su rendimenti sintetici."""

    @staticmethod
    def returns(volatilities: list[float], *, seed: int = 8) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        data = rng.normal(size=(156, 4)) * np.asarray(volatilities, dtype=float)
        index = pd.bdate_range("2019-01-01", periods=len(data), name="Date")
        return pd.DataFrame(data, index=index, columns=ASSETS)

    def test_equal_weight_if_estimated_risk_below_limit(self):
        from src.multiasset_allocator import calcola_pesi_a2
        daily = self.returns([0.001, 0.0008, 0.0012, 0.0003])
        weights, info = calcola_pesi_a2(daily, finestra=126, volatilita_massima=0.08)
        np.testing.assert_allclose(weights.to_numpy(), [0.25] * 4, atol=1e-12)
        self.assertEqual(info["stato"], "equal_weight")
        self.assertTrue(info["limite_rispettato"])
        self.assertLess(info["volatilita_ew"], 0.08)

    def test_feasible_constraint_reduces_risk_with_actual_solver(self):
        from src.multiasset_allocator import calcola_pesi_a2
        daily = self.returns([0.025, 0.003, 0.015, 0.0006])
        weights, info = calcola_pesi_a2(daily, finestra=126, volatilita_massima=0.08)
        self.assertEqual(info["stato"], "ottimizzato")
        self.assertGreater(info["volatilita_ew"], 0.08)
        self.assertTrue(info["limite_rispettato"])
        self.assertLessEqual(info["volatilita_target"], 0.0800001)
        self.assertAlmostEqual(weights.sum(), 1.0, places=7)
        self.assertGreaterEqual(weights.min(), -1e-7)
        self.assertLess(weights["SPY"], 0.25)

    def test_infeasible_constraint_uses_minimum_variance_fallback(self):
        from src.multiasset_allocator import calcola_pesi_a2
        rng = np.random.default_rng(212)
        common = rng.normal(0, 0.025, size=156)
        noise = rng.normal(0, 0.0004, size=(156, 4))
        daily = pd.DataFrame(
            common[:, None] + noise,
            index=pd.bdate_range("2019-01-01", periods=156),
            columns=ASSETS,
        )
        weights, info = calcola_pesi_a2(daily, finestra=126, volatilita_massima=0.08)
        self.assertEqual(info["stato"], "fallback_minimum_variance")
        self.assertFalse(info["limite_rispettato"])
        self.assertGreater(info["volatilita_target"], 0.08)
        self.assertAlmostEqual(weights.sum(), 1.0, places=7)
        self.assertGreaterEqual(weights.min(), -1e-7)

    def test_covariance_uses_last_126_observations_and_252_days(self):
        from src.multiasset_allocator import calcola_pesi_a2
        daily = self.returns([0.004, 0.003, 0.006, 0.001])
        weights, info = calcola_pesi_a2(daily, finestra=126, volatilita_massima=1.0)
        cov_annual = daily.tail(126).cov().to_numpy(dtype=float) * 252
        expected_vol = float(np.sqrt(np.full(4, 0.25) @ cov_annual @ np.full(4, 0.25)))
        self.assertAlmostEqual(info["volatilita_ew"], expected_vol, places=12)
        np.testing.assert_allclose(weights.to_numpy(), 0.25)

        older_dates = pd.bdate_range(end=daily.index[0] - pd.Timedelta(days=1), periods=8)
        older = pd.DataFrame(0.10, index=older_dates, columns=ASSETS)
        _, with_older_history = calcola_pesi_a2(
            pd.concat([older, daily]), finestra=126, volatilita_massima=1.0
        )
        self.assertAlmostEqual(info["volatilita_ew"], with_older_history["volatilita_ew"])

    def test_invalid_prices_or_short_history_are_rejected_by_allocator(self):
        from src.multiasset_allocator import calcola_pesi_a2
        valid = self.returns([0.003] * 4)
        bad = valid.copy()
        bad.iloc[4, 0] = np.nan
        with self.assertRaises(ValueError):
            calcola_pesi_a2(bad, finestra=156)
        with self.assertRaises(ValueError):
            calcola_pesi_a2(valid.iloc[:100], finestra=126)
        with self.assertRaises(ValueError):
            calcola_pesi_a2(valid.drop(columns=["GLD"]), finestra=126)
        with self.assertRaises(ValueError):
            calcola_pesi_a2(valid, finestra=126, volatilita_massima=-0.1)



def make_toy_prices() -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Crea cinque sedute con un solo movimento noto di SPY."""
    dates = pd.bdate_range("2020-06-01", periods=5, name="Date")
    opens = pd.DataFrame(100.0, index=dates, columns=ASSETS)
    closes = opens.copy()
    closes.loc[dates[2], "SPY"] = 120.0
    opens.loc[dates[3]:, "SPY"] = 120.0
    closes.loc[dates[3]:, "SPY"] = 120.0
    return opens, closes, dates


class TestDailyBacktester(unittest.TestCase):
    """Contabilita' puntuale senza ribilanciamento giornaliero implicito."""

    def setUp(self):
        self.opens, self.closes, self.dates = make_toy_prices()
        self.initial = 1000.0
        self.first_target = pd.DataFrame(
            [[0.5, 0.5, 0.0, 0.0]],
            index=pd.DatetimeIndex([self.dates[0]], name="Date"),
            columns=ASSETS,
        )

    def simulate(self, target=None, commission=2.0, slippage=1.0):
        return simula_portafoglio(
            self.opens, self.closes,
            self.first_target if target is None else target,
            capitale_iniziale=self.initial,
            commission_bps=commission,
            slippage_bps=slippage,
        )

    def test_first_signal_trades_only_on_next_open(self):
        output = self.simulate()
        self.assertFalse(bool(output.loc[self.dates[0], "operazione_eseguita"]))
        self.assertTrue(bool(output.loc[self.dates[1], "operazione_eseguita"]))
        self.assertFalse(bool(output.loc[self.dates[2], "operazione_eseguita"]))
        self.assertAlmostEqual(output.loc[self.dates[0], "nav"], self.initial)
        self.assertAlmostEqual(output.loc[self.dates[0], "rendimento_netto"], 0)
        self.assertEqual(int(output["operazione_eseguita"].sum()), 1)

    def test_first_trade_cost_is_charged_on_notional(self):
        output = self.simulate()
        post = self.initial / (1 + 3.0 / 10000.0)
        fee = post * 3.0 / 10000.0
        first = output.loc[self.dates[1]]
        self.assertAlmostEqual(first["nav"], post, places=8)
        self.assertAlmostEqual(first["costo_transazione"], fee, places=8)
        self.assertAlmostEqual(first["turnover"], post / self.initial, places=8)
        self.assertAlmostEqual(first["rendimento_netto"], -fee / self.initial, places=9)
        self.assertAlmostEqual(first["rendimento_lordo"], 0.0, places=9)

    def test_share_quantity_drift_without_implicit_rebalancing(self):
        output = self.simulate(commission=0.0, slippage=0.0)
        day1, day2 = self.dates[1], self.dates[2]
        self.assertAlmostEqual(output.loc[day1, "nav"], 1000.0)
        self.assertAlmostEqual(output.loc[day2, "nav"], 1100.0)
        self.assertAlmostEqual(output.loc[day2, "rendimento_netto"], 0.10)
        self.assertAlmostEqual(output.loc[day2, "SPY_peso_effettivo"], 600.0 / 1100.0)
        self.assertAlmostEqual(output.loc[day2, "IEF_peso_effettivo"], 500.0 / 1100.0)
        self.assertFalse(bool(output.loc[day2, "operazione_eseguita"]))
        self.assertEqual(output.loc[day2, "costo_transazione"], 0)
        self.assertEqual(output.loc[day2, "turnover"], 0)
        # Le quantita' SPY restano uguali, anche se il suo peso cambia.
        qty1 = output.loc[day1, "SPY_peso_effettivo"] * output.loc[day1, "nav"] / 100
        qty2 = output.loc[day2, "SPY_peso_effettivo"] * output.loc[day2, "nav"] / 120
        self.assertAlmostEqual(qty1, qty2, places=10)

    def test_future_target_cannot_change_any_earlier_nav(self):
        target_a = self.first_target.copy()
        target_a.loc[self.dates[2]] = [1.0, 0.0, 0.0, 0.0]
        target_b = self.first_target.copy()
        target_b.loc[self.dates[2]] = [0.0, 1.0, 0.0, 0.0]
        first = self.simulate(target=target_a)
        second = self.simulate(target=target_b)
        pd.testing.assert_frame_equal(first.loc[:self.dates[2]], second.loc[:self.dates[2]])
        self.assertTrue(bool(first.loc[self.dates[3], "operazione_eseguita"]))
        self.assertNotAlmostEqual(
            first.loc[self.dates[3], "SPY_peso_effettivo"],
            second.loc[self.dates[3], "SPY_peso_effettivo"],
        )

    def test_signal_on_last_day_is_discarded(self):
        with_last = self.first_target.copy()
        with_last.loc[self.dates[-1]] = [1.0, 0.0, 0.0, 0.0]
        original = self.simulate()
        extended = self.simulate(target=with_last)
        pd.testing.assert_frame_equal(original, extended)
        only_last = with_last.loc[[self.dates[-1]]]
        with self.assertRaises(ValueError):
            self.simulate(target=only_last)

    def test_weights_are_long_only_and_fully_invested_after_trading(self):
        result = self.simulate()
        weight_cols = [f"{ticker}_peso_effettivo" for ticker in ASSETS]
        active = result.loc[self.dates[1]:, weight_cols]
        np.testing.assert_allclose(active.sum(axis=1), 1.0, atol=1e-9)
        self.assertTrue((active >= -1e-9).all().all())
        self.assertTrue((active <= 1 + 1e-9).all().all())
        self.assertTrue(np.isfinite(result["nav"]).all())
        self.assertTrue((result["nav"] > 0).all())

    def test_bad_signal_weights_are_rejected(self):
        for weights in (
            [0.5, 0.3, 0.0, 0.0],
            [1.1, -0.1, 0.0, 0.0],
            [np.nan, 0.0, 0.0, 1.0],
        ):
            with self.subTest(weights=weights):
                bad = self.first_target.copy()
                bad.iloc[0] = weights
                with self.assertRaises(ValueError):
                    self.simulate(target=bad)

    def test_bad_prices_and_execution_costs_are_rejected(self):
        original = self.opens.copy()
        try:
            self.opens.iloc[2, 0] = 0.0
            with self.assertRaises(ValueError):
                self.simulate()
            self.opens.iloc[2, 0] = np.nan
            with self.assertRaises(ValueError):
                self.simulate()
        finally:
            self.opens = original
        with self.assertRaises(ValueError):
            self.simulate(commission=-1.0)
        with self.assertRaises(ValueError):
            self.simulate(slippage=-1.0)

    def test_misaligned_calendars_or_signal_dates_are_rejected(self):
        bad_close = self.closes.iloc[1:].copy()
        with self.assertRaises(ValueError):
            simula_portafoglio(self.opens, bad_close, self.first_target)
        future = self.first_target.copy()
        future.index = pd.DatetimeIndex([pd.Timestamp("2020-06-08")])
        with self.assertRaises(ValueError):
            self.simulate(target=future)


class TestEvaluationMetrics(unittest.TestCase):
    """Rendimenti composti, rischio, Sharpe e Sortino di un campione noto."""

    def test_metrics_match_independent_formulas(self):
        values = np.array([0.10, -0.05, 0.02, -0.01, 0.03, -0.02])
        returns = pd.Series(values, index=pd.bdate_range("2020-06-01", periods=6))
        result = calculate_metrics(returns, risk_free=0.0, annualization_factor=252)
        wealth = np.prod(1 + values)
        std = values.std(ddof=1)
        downside = np.sqrt(np.mean(np.minimum(values, 0.0) ** 2))
        expected = {
            "cumulative_return": wealth - 1,
            "cagr": wealth ** (252 / len(values)) - 1,
            "annualized_volatility": std * np.sqrt(252),
            "sharpe_ratio": values.mean() / std * np.sqrt(252),
            "sortino_ratio": values.mean() / downside * np.sqrt(252),
            "max_drawdown": -0.05,
            "profit_factor_daily_returns": 0.15 / 0.08,
        }
        self.assertEqual(result["observations"], len(values))
        for key, target in expected.items():
            with self.subTest(metric=key):
                self.assertAlmostEqual(result[key], target, places=10)

    def test_risk_free_affects_risk_adjusted_metrics(self):
        returns = pd.Series(
            [0.01, -0.005, 0.006, 0.001],
            index=pd.bdate_range("2020-01-01", periods=4),
        )
        no_rf = calculate_metrics(returns, 0.0, 252)
        with_rf = calculate_metrics(returns, 0.0002, 252)
        self.assertLess(with_rf["sharpe_ratio"], no_rf["sharpe_ratio"])
        self.assertAlmostEqual(with_rf["cagr"], no_rf["cagr"])

    def test_invalid_returns_are_not_silently_imputed(self):
        index = pd.bdate_range("2020-01-01", periods=3)
        for values in ([0.01, np.nan, 0.02], [0.01, -1.0, 0.02]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    calculate_metrics(pd.Series(values, index=index), 0.0, 252)
        with self.assertRaises(ValueError):
            calculate_metrics(pd.Series([0.01], index=index[:1]), 0.0, 252)


class TestCSVDataLoader(unittest.TestCase):
    """Conserva i test utili del vecchio progetto sui CSV di input."""

    @staticmethod
    def read(csv_text: str, *, strict: bool = True) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic.csv"
            path.write_text(csv_text, encoding="utf-8")
            return load_dataset(path, strict=strict)

    def test_unsorted_dates_are_sorted_without_dropping_rows(self):
        result = self.read("Date,value\n2020-01-10,2\n2020-01-03,1\n")
        self.assertEqual(result["value"].tolist(), [1, 2])
        self.assertTrue(result.index.is_monotonic_increasing)

    def test_invalid_dates_are_rejected_in_strict_mode(self):
        with self.assertRaises(ValueError):
            self.read("Date,value\n2020-01-10,2\nnot-a-date,1\n")

    def test_duplicate_dates_are_rejected_in_strict_mode(self):
        with self.assertRaises(ValueError):
            self.read("Date,value\n2020-01-10,1\n2020-01-10,2\n")

    def test_permissive_mode_must_be_requested_explicitly(self):
        output = self.read(
            "Date,value\n2020-01-10,1\nnot-a-date,999\n"
            "2020-01-03,2\n2020-01-10,3\n",
            strict=False,
        )
        self.assertEqual(output["value"].tolist(), [2, 3])

    def test_missing_date_column_is_a_failure(self):
        with self.assertRaises(KeyError):
            self.read("timestamp,value\n2020-01-10,2\n")


class TestPipelineIntegration(unittest.TestCase):
    """A1 e A2 usano gli stessi prezzi sintetici e scrivono output distinti."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = make_config()
        self.panel, self.decisions = make_synthetic_market()
        write_processed_data(self.root, self.panel, self.decisions)

    def run_a1(self):
        with redirect_stdout(StringIO()):
            weights = run_portfolio(self.config, self.root, level="A1")
            backtests, reports = run_backtest(
                self.config, self.root, level="A1", portfolios=weights
            )
            metrics, curves = run_evaluation(
                self.config, self.root, level="A1", backtests=backtests
            )
        return weights, backtests, reports, metrics, curves

    def test_a1_end_to_end_train_only_and_distinct_outputs(self):
        weights, backtests, reports, metrics, curves = self.run_a1()
        self.assertEqual(set(backtests), set(self.config["benchmarks"]))
        self.assertEqual(set(metrics["portfolio"]), set(self.config["benchmarks"]))
        self.assertEqual(set(metrics["period"]), {"train"})
        self.assertEqual(set(metrics["level"]), {"A1"})
        self.assertEqual(set(curves["period"]), {"train"})
        self.assertEqual(set(curves["level"]), {"A1"})
        self.assertTrue((pd.to_datetime(curves["Date"]) <= pd.Timestamp("2020-06-30")).all())

        for name, frame in backtests.items():
            with self.subTest(strategy=name):
                self.assertGreaterEqual(frame.index.min(), pd.Timestamp("2020-06-01"))
                self.assertLessEqual(frame.index.max(), pd.Timestamp("2020-06-30"))
                self.assertEqual(reports[name]["periodo"], "training")
                self.assertEqual(reports[name]["operazioni_eseguite"], int(frame["operazione_eseguita"].sum()))
                if self.config["benchmarks"][name]["rebalance"] == "none":
                    self.assertEqual(reports[name]["operazioni_eseguite"], 1)
                self.assertTrue((self.root / "data" / "signals" / "A1" / name / "target_weights.csv").is_file())
                self.assertTrue((self.root / "data" / "results" / "A1" / name / "backtest_daily.csv").is_file())
                self.assertTrue((self.root / "data" / "results" / "A1" / name / "equity_curve.csv").is_file())
                raw_report = json.loads(
                    (self.root / "data" / "results" / "A1" / name / "execution_summary.json")
                    .read_text(encoding="utf-8")
                )
                self.assertEqual(raw_report["periodo"], "training")
        self.assertTrue((self.root / "data" / "results" / "A1" / "evaluation_metrics.csv").is_file())

    def test_reloading_saved_targets_reproduces_identical_backtest(self):
        _, first, _, _, _ = self.run_a1()
        with redirect_stdout(StringIO()):
            second, _ = run_backtest(self.config, self.root, level="A1")
        for name in first:
            with self.subTest(strategy=name):
                pd.testing.assert_frame_equal(first[name], second[name])

    def test_mutating_validation_and_oos_prices_changes_no_training_result(self):
        _, original, _, _, _ = self.run_a1()
        altered = self.panel.copy(deep=True)
        altered.loc[altered.index > pd.Timestamp("2020-06-30"), "SPY_close"] *= 25.0
        write_processed_data(self.root, altered, self.decisions)
        with redirect_stdout(StringIO()):
            weights = run_portfolio(self.config, self.root, level="A1")
            later, _ = run_backtest(
                self.config, self.root, level="A1", portfolios=weights
            )
        for name in original:
            with self.subTest(strategy=name):
                pd.testing.assert_frame_equal(original[name], later[name])

    def test_a2_integration_uses_the_same_backtest_engine_as_equal_weight(self):
        _, a1, _, _, _ = self.run_a1()
        with patch.dict(sys.modules, {"src.multiasset_allocator": fake_a2_allocator()}):
            with redirect_stdout(StringIO()):
                a2_weights = run_portfolio(self.config, self.root, level="A2")
        with redirect_stdout(StringIO()):
            a2, reports = run_backtest(
                self.config, self.root, level="A2", portfolios=a2_weights
            )
            metrics, curves = run_evaluation(
                self.config, self.root, level="A2", backtests=a2
            )
        self.assertEqual(list(a2), ["volatility_control"])
        pd.testing.assert_frame_equal(
            a1["equal_weight"], a2["volatility_control"]
        )
        self.assertEqual(reports["volatility_control"]["periodo"], "training")
        self.assertEqual(metrics.loc[0, "level"], "A2")
        self.assertEqual(set(curves["portfolio"]), {"volatility_control"})
        self.assertTrue((self.root / "data" / "signals" / "A2" / "volatility_control" / "diagnostics.csv").is_file())
        self.assertTrue((self.root / "data" / "results" / "A2" / "evaluation_metrics.csv").is_file())
        self.assertTrue((self.root / "data" / "results" / "A1" / "evaluation_metrics.csv").is_file())

    def test_last_training_day_signal_does_not_trade_in_validation(self):
        weights, original, _, _, _ = self.run_a1()
        extended = {name: frame.copy(deep=True) for name, frame in weights.items()}
        last_training_day = pd.Timestamp("2020-06-30")
        extended["equal_weight"].loc[last_training_day] = [0.25] * 4
        with redirect_stdout(StringIO()):
            updated, reports = run_backtest(
                self.config, self.root, level="A1", portfolios=extended
            )
        pd.testing.assert_frame_equal(original["equal_weight"], updated["equal_weight"])
        self.assertEqual(reports["equal_weight"]["decisioni_non_eseguibili"], 1)
        self.assertEqual(
            reports["equal_weight"]["operazioni_eseguite"],
            int(original["equal_weight"]["operazione_eseguita"].sum()),
        )

    def test_backtester_rejects_future_targets(self):
        weights, _, _, _, _ = self.run_a1()
        altered = {name: data.copy(deep=True) for name, data in weights.items()}
        altered["spy_buy_and_hold"].loc[pd.Timestamp("2020-07-01")] = [1, 0, 0, 0]
        with self.assertRaises(ValueError):
            run_backtest(self.config, self.root, level="A1", portfolios=altered)

    def test_evaluator_rejects_future_dated_results(self):
        _, original, _, _, _ = self.run_a1()
        bad = {name: data.copy(deep=True) for name, data in original.items()}
        future_date = pd.Timestamp("2020-07-01")
        bad["spy_buy_and_hold"].loc[future_date] = bad["spy_buy_and_hold"].iloc[-1]
        with self.assertRaises(ValueError):
            run_evaluation(self.config, self.root, level="A1", backtests=bad)

    def test_b_and_c_cannot_be_run_as_if_implemented(self):
        for level in ("B", "C"):
            with self.subTest(level=level):
                with self.assertRaises(NotImplementedError):
                    run_portfolio(self.config, self.root, level=level)
                with self.assertRaises(NotImplementedError):
                    run_backtest(self.config, self.root, level=level)
                with self.assertRaises(NotImplementedError):
                    run_evaluation(self.config, self.root, level=level)

    def test_explicit_oos_unlock_is_not_supported_in_level_a(self):
        config = deepcopy(self.config)
        config["validation"]["development_only"] = False
        config["evaluation"]["evaluate_oos"] = True
        with self.assertRaises(ValueError):
            run_backtest(config, self.root, level="A1")
        with self.assertRaises(ValueError):
            run_evaluation(config, self.root, level="A1")


if __name__ == "__main__":
    unittest.main()
