
"""
Test degli invarianti economici e temporali della pipeline.

Utilizza dati sintetici e CSV temporanei.
Non accede ai dataset reali né al Test Set OOS.
"""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from src.signal_generation import (
    build_duration_score,
    generate_asset_signals,
)
from src.portfolio import build_portfolio_weights
from src.backtester import (
    apply_execution_lag,
    calculate_strategy_returns,
)
from src.data_loader import load_dataset


def make_config():
    return {
        "universe": {
            "traded_assets": {
                "short_duration": {"ticker": "SHY"},
                "intermediate_duration": {"ticker": "IEF"},
                "long_duration": {"ticker": "TLT"},
            }
        },
        "signals": {
            "long_duration_threshold": 2,
            "short_duration_threshold": -2,
        },
        "portfolio": {
            "allocation_method": "equal_active_signal",
            "leverage": 1.0,
        },
        "execution": {
            "initial_capital": 100000.0,
            "commission_bps": 2,
            "slippage_bps": 1,
            "execution_rule": "next_available_period",
        },
    }


def make_index(n=4):
    return pd.date_range("2020-01-03", periods=n, freq="W-FRI")


def make_signals():
    return pd.DataFrame(
        {
            "SHY_signal": [0, 1, 0, 0],
            "IEF_signal": [0, 0, 1, 0],
            "TLT_signal": [1, -1, 0, 0],
        },
        index=make_index(),
    )


class TestSignalInvariants(unittest.TestCase):

    def setUp(self):
        self.config = make_config()

        self.states = pd.DataFrame(
            {
                "inflation_state": [
                    "falling", "rising",
                    "rising", "falling",
                ],
                "rate_state": [
                    "falling", "rising",
                    "falling", "rising",
                ],
                "curve_state": [
                    "normal", "normal",
                    "inverted", "inverted",
                ],
                "volatility_state": [
                    "low", "low", "high", "high",
                ],
            },
            index=make_index(),
        )

    def test_duration_score_economic_rules(self):
        result = build_duration_score(self.states)

        # Score grezzo: +3, -3, +2, 0.
        np.testing.assert_allclose(
            result["duration_score_raw"].to_numpy(
                dtype=float
            ),
            [3, -3, 2, 0],
        )

        # Alta volatilita: lo score +2 diventa +1.
        np.testing.assert_allclose(
            result["duration_score"].to_numpy(
                dtype=float
            ),
            [3, -3, 1, 0],
        )

    def test_signal_mapping(self):
        df = pd.DataFrame(
            {
                "duration_score": [
                    3, -3, 1, 0, np.nan,
                ]
            },
            index=make_index(5),
        )

        result = generate_asset_signals(
            df, self.config
        )

        self.assertEqual(
            result["SHY_signal"].tolist(),
            [0, 1, 0, 0, 0],
        )
        self.assertEqual(
            result["IEF_signal"].tolist(),
            [0, 0, 1, 1, 0],
        )
        self.assertEqual(
            result["TLT_signal"].tolist(),
            [1, -1, 0, 0, 0],
        )

        for asset in ("SHY", "IEF", "TLT"):
            values = set(result[f"{asset}_signal"])
            self.assertTrue(
                values.issubset({-1, 0, 1})
            )

    def test_future_states_do_not_change_past(self):
        short = build_duration_score(
            self.states.iloc[:3]
        )
        full = build_duration_score(self.states)

        pd.testing.assert_series_equal(
            short["duration_score"],
            full["duration_score"].iloc[:3],
        )

        changed = self.states.copy()
        changed.loc[
            changed.index[-1], "rate_state"
        ] = "falling"

        altered = build_duration_score(changed)

        pd.testing.assert_series_equal(
            full["duration_score"].iloc[:3],
            altered["duration_score"].iloc[:3],
        )

    def test_incomplete_regime_has_no_signal(self):
        states = self.states.copy()
        states.loc[
            states.index[0], "inflation_state"
        ] = None

        scored = build_duration_score(states)
        result = generate_asset_signals(
            scored, self.config
        )

        self.assertTrue(
            pd.isna(result.iloc[0]["duration_score"])
        )

        for asset in ("SHY", "IEF", "TLT"):
            self.assertEqual(
                result.iloc[0][f"{asset}_signal"],
                0,
            )


class TestPortfolioInvariants(unittest.TestCase):

    def setUp(self):
        self.config = make_config()
        self.signals = make_signals()

    def test_equal_active_signal_weights(self):
        result = build_portfolio_weights(
            self.signals, self.config
        )

        actual = result[
            ["SHY_weight", "IEF_weight", "TLT_weight"]
        ].to_numpy(dtype=float)

        expected = np.array([
            [0.0, 0.0, 1.0],
            [0.5, 0.0, -0.5],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ])

        np.testing.assert_allclose(
            actual, expected
        )

        np.testing.assert_allclose(
            result["gross_exposure"],
            [1.0, 1.0, 1.0, 0.0],
        )

        np.testing.assert_allclose(
            result["net_exposure"],
            [1.0, 0.0, 1.0, 0.0],
        )

        self.assertTrue(np.isfinite(actual).all())

    def test_configured_leverage_limit(self):
        config = deepcopy(self.config)
        config["portfolio"]["leverage"] = 0.5

        result = build_portfolio_weights(
            self.signals, config
        )

        self.assertTrue(
            (result["gross_exposure"] <= 0.5 + 1e-12).all()
        )

        self.assertAlmostEqual(
            result.iloc[0]["TLT_weight"],
            0.5,
        )

        self.assertEqual(
            result.iloc[-1]["gross_exposure"],
            0.0,
        )

    def test_invalid_leverage_rejected(self):
        config = deepcopy(self.config)
        config["portfolio"]["leverage"] = -1.0

        with self.assertRaises(ValueError):
            build_portfolio_weights(
                self.signals, config
            )


class TestBacktesterInvariants(unittest.TestCase):

    def setUp(self):
        self.config = make_config()

        self.portfolio = build_portfolio_weights(
            make_signals(), self.config
        )

        self.portfolio["SHY_return_1w"] = [
            0.10, 0.01, 0.02, 0.00,
        ]
        self.portfolio["IEF_return_1w"] = [
            0.20, 0.03, -0.01, 0.01,
        ]
        self.portfolio["TLT_return_1w"] = [
            0.30, 0.05, -0.04, 0.02,
        ]

    def test_execution_lag_one_week(self):
        result = apply_execution_lag(
            self.portfolio, self.config
        )

        for asset in ("SHY", "IEF", "TLT"):
            expected = (
                self.portfolio[f"{asset}_weight"]
                .shift(1)
                .fillna(0.0)
            )

            pd.testing.assert_series_equal(
                result[f"{asset}_applied_weight"],
                expected,
                check_names=False,
            )

        # Nessuna posizione nella prima settimana.
        for asset in ("SHY", "IEF", "TLT"):
            self.assertEqual(
                result.iloc[0][f"{asset}_applied_weight"],
                0.0,
            )

    def test_future_target_cannot_affect_past(self):
        original = apply_execution_lag(
            self.portfolio, self.config
        )

        changed = self.portfolio.copy()
        changed.loc[
            changed.index[2], "TLT_weight"
        ] = 0.75

        altered = apply_execution_lag(
            changed, self.config
        )

        columns = [
            f"{asset}_applied_weight"
            for asset in ("SHY", "IEF", "TLT")
        ]

        pd.testing.assert_frame_equal(
            original[columns].iloc[:3],
            altered[columns].iloc[:3],
        )

        self.assertNotEqual(
            original.iloc[3]["TLT_applied_weight"],
            altered.iloc[3]["TLT_applied_weight"],
        )

    def test_gross_returns_turnover_and_costs(self):
        lagged = apply_execution_lag(
            self.portfolio, self.config
        )

        result = calculate_strategy_returns(
            lagged, self.config
        )

        np.testing.assert_allclose(
            result["strategy_gross_return"],
            [0.0, 0.05, 0.03, 0.01],
        )

        np.testing.assert_allclose(
            result["turnover"],
            [0.0, 1.0, 2.0, 2.0],
        )

        np.testing.assert_allclose(
            result["transaction_cost"],
            [0.0, 0.0003, 0.0006, 0.0006],
        )

        np.testing.assert_allclose(
            result["strategy_net_return"],
            [0.0, 0.0497, 0.0294, 0.0094],
        )

        np.testing.assert_allclose(
            result["strategy_net_return"],
            result["strategy_gross_return"]
            - result["transaction_cost"],
        )

    def test_negative_transaction_cost_rejected(self):
        config = deepcopy(self.config)
        config["execution"]["commission_bps"] = -1

        lagged = apply_execution_lag(
            self.portfolio, config
        )

        with self.assertRaises(ValueError):
            calculate_strategy_returns(
                lagged, config
            )


class TestDataLoaderInvariants(unittest.TestCase):

    def load_synthetic_csv(
        self, csv_content, strict=True
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "synthetic.csv"
            path.write_text(
                csv_content, encoding="utf-8"
            )
            return load_dataset(
                path, strict=strict
            )

    def test_valid_dates_are_sorted(self):
        result = self.load_synthetic_csv(
            "Date,value\n"
            "2020-01-10,2\n"
            "2020-01-03,1\n"
        )

        self.assertTrue(
            result.index.is_monotonic_increasing
        )
        self.assertEqual(
            result["value"].tolist(),
            [1, 2],
        )

    def test_strict_rejects_invalid_dates(self):
        with self.assertRaises(ValueError):
            self.load_synthetic_csv(
                "Date,value\n"
                "2020-01-03,1\n"
                "invalid-date,2\n"
            )

    def test_strict_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            self.load_synthetic_csv(
                "Date,value\n"
                "2020-01-03,1\n"
                "2020-01-03,2\n"
            )

    def test_permissive_mode_is_explicit(self):
        result = self.load_synthetic_csv(
            "Date,value\n"
            "2020-01-10,1\n"
            "invalid-date,999\n"
            "2020-01-03,2\n"
            "2020-01-10,3\n",
            strict=False,
        )

        self.assertEqual(len(result), 2)

        self.assertEqual(
            result.loc[
                pd.Timestamp("2020-01-03"), "value"
            ],
            2,
        )

        # Con strict=False prevale l'ultimo duplicato.
        self.assertEqual(
            result.loc[
                pd.Timestamp("2020-01-10"), "value"
            ],
            3,
        )

    def test_missing_date_column_rejected(self):
        with self.assertRaises(KeyError):
            self.load_synthetic_csv(
                "timestamp,value\n"
                "2020-01-03,1\n"
            )


if __name__ == "__main__":
    unittest.main()
