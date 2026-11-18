"""Tests of the allocation rule, the validation tuning and the time split."""

import unittest
from copy import deepcopy

import numpy as np
import pandas as pd

from src.strategy import MultiAssetSignalAggregator, tune_allocation
from src.time_split import build_weekly_decision_dates, get_time_windows, slice_period
from src.utils import ASSETS

DATES = pd.DatetimeIndex(["2020-01-10", "2020-01-17", "2020-01-24"], name="Date")


def frame(rows):
    return pd.DataFrame(rows, index=DATES[: len(rows)], columns=list(ASSETS))


def aggregator(**overrides):
    params = dict(min_confidence=0.3, max_weight_per_asset=0.5, allow_short=False,
                  spy_default_long=True, rebalance_threshold=0.05)
    params.update(overrides)
    return MultiAssetSignalAggregator(**params)


class TestAllocationRule(unittest.TestCase):
    def test_spy_is_long_by_default_and_flat_on_confirmed_short(self):
        signals = frame([[0, 0, 0, 0], [-1, 0, 0, 0]])
        confidences = frame([[0.0, 0.0, 0.0, 0.0], [0.9, 0.0, 0.0, 0.0]])
        weights = aggregator().target_weights(signals, confidences)
        self.assertEqual(weights["SPY"].tolist(), [0.5, 0.0])

    def test_shorts_require_permission_and_confidence(self):
        signals = frame([[1, -1, -1, 0]])
        confidences = frame([[0.9, 0.9, 0.1, 0.0]])
        no_short = aggregator().target_weights(signals, confidences)
        with_short = aggregator(allow_short=True).target_weights(signals, confidences)
        self.assertEqual(no_short.iloc[0].tolist(), [0.5, 0.0, 0.0, 0.0])
        self.assertEqual(with_short.iloc[0].tolist(), [0.5, -0.5, 0.0, 0.0])

    def test_weights_are_capped_and_never_levered(self):
        rng = np.random.default_rng(1)
        dates = pd.date_range("2020-01-03", periods=200, freq="W-FRI")
        signals = pd.DataFrame(rng.integers(-1, 2, (200, 4)), index=dates, columns=list(ASSETS))
        confidences = pd.DataFrame(rng.random((200, 4)), index=dates, columns=list(ASSETS))
        for cap in (0.35, 0.5, 1.0):
            weights = aggregator(max_weight_per_asset=cap, allow_short=True).target_weights(signals, confidences)
            self.assertLessEqual(weights.abs().max().max(), cap + 1e-12)
            self.assertLessEqual(weights.abs().sum(axis=1).max(), 1.0 + 1e-12)

    def test_small_changes_do_not_trigger_rebalancing(self):
        # Moving from four to three active assets changes no weight by more than 0.25.
        agg = aggregator(rebalance_threshold=0.3, max_weight_per_asset=1.0)
        signals = frame([[1, 1, 1, 1], [1, 1, 1, 0]])
        confidences = frame([[1, 1, 1, 1], [1, 1, 1, 1]])
        weights = agg.target_weights(signals, confidences)
        self.assertEqual(weights.iloc[1].tolist(), weights.iloc[0].tolist())


class TestTimeSplit(unittest.TestCase):
    CONFIG = {"validation": {
        "train_start": "2005-05-20", "train_end": "2020-12-31",
        "validation_start": "2016-01-01", "validation_end": "2020-12-31",
        "oos_start": "2021-01-01", "oos_end": None,
    }}

    def test_validation_must_lie_inside_the_in_sample_window(self):
        config = deepcopy(self.CONFIG)
        config["validation"]["validation_end"] = "2021-06-30"
        with self.assertRaises(ValueError):
            get_time_windows(config)

    def test_oos_must_follow_the_in_sample_window(self):
        config = deepcopy(self.CONFIG)
        config["validation"]["oos_start"] = "2020-06-01"
        with self.assertRaises(ValueError):
            get_time_windows(config)

    def test_periods_do_not_overlap(self):
        dates = pd.date_range("2004-01-02", "2026-01-02", freq="W-FRI")
        data = pd.DataFrame({"x": range(len(dates))}, index=dates)
        in_sample, oos = slice_period(data, self.CONFIG, "in_sample"), slice_period(data, self.CONFIG, "oos")
        self.assertTrue(in_sample.index.max() < oos.index.min())

    def test_decision_dates_are_last_trading_session_of_each_week(self):
        sessions = pd.DatetimeIndex(["2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-13"])
        decisions = build_weekly_decision_dates(pd.DataFrame(index=sessions)).index
        # Good Friday 2020-04-10 was a holiday: the week ends on Thursday.
        self.assertEqual(list(decisions), [pd.Timestamp("2020-04-09"), pd.Timestamp("2020-04-13")])


class TestValidationTuning(unittest.TestCase):
    def test_tuning_ignores_data_after_the_validation_window(self):
        dates = pd.bdate_range("2019-01-01", "2021-06-30", name="Date")
        rng = np.random.default_rng(3)
        panel = {}
        for asset in ASSETS:
            close = 100 * np.cumprod(1 + rng.normal(2e-4, 0.01, len(dates)))
            panel[f"{asset}_open"], panel[f"{asset}_close"] = close, close
        market = pd.DataFrame(panel, index=dates)
        decisions = build_weekly_decision_dates(market).index
        signals = pd.DataFrame(rng.integers(-1, 2, (len(decisions), 4)), index=decisions, columns=list(ASSETS))
        confidences = pd.DataFrame(rng.random((len(decisions), 4)), index=decisions, columns=list(ASSETS))
        config = {
            "validation": {"train_start": "2019-01-01", "train_end": "2020-12-31",
                           "validation_start": "2020-01-01", "validation_end": "2020-12-31",
                           "oos_start": "2021-01-01", "oos_end": None},
            "allocation": {"spy_default_long": True, "rebalance_threshold": 0.05, "selection_metric": "sharpe_ratio",
                           "grid": {"min_confidence": [0.2, 0.5], "max_weight_per_asset": [0.5],
                                    "allow_short": [False, True]}},
            "execution": {"initial_capital": 100_000.0, "commission_bps": 2, "slippage_bps": 1,
                          "annual_borrow_rate": 0.01, "additional_margin_rate": 0.5},
            "evaluation": {"risk_free_rate": 0.0, "annualization_factor": 252},
        }
        selected, _ = tune_allocation(config, signals, confidences, market)

        altered_market = market.copy()
        altered_market.loc[altered_market.index >= "2021-01-01"] *= 3.0
        altered_signals = signals.copy()
        altered_signals.loc[altered_signals.index >= "2021-01-01"] *= -1
        altered, _ = tune_allocation(config, altered_signals, confidences, altered_market)
        self.assertEqual(selected, altered)


if __name__ == "__main__":
    unittest.main()
