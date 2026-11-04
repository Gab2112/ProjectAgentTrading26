"""Regression tests for SMA/EMA indicator warm-up and temporal isolation (II.4).

Run from repository root:
    python3 -m unittest discover -s tests -p 'test_sma_ema_warmup.py' -v
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.agents.EMA_EMW_agent import (
    SMAEMAAgent,
    SMAEMAOptimizer,
    calculate_performance_metrics,
)


class TestSMAEMAWarmup(unittest.TestCase):
    @staticmethod
    def panel(n=80):
        dates = pd.date_range('2014-01-03', periods=n, freq='W-FRI')
        t = np.arange(n, dtype=float)
        price = 100.0 + 0.55 * t + 0.30 * np.sin(t / 3.0)
        return pd.DataFrame({'SPY_close': price}, index=dates)

    @staticmethod
    def optimizer(dead_band=0.0):
        return SMAEMAOptimizer(
            price_col='SPY_close',
            sma_range=[8],
            ema_range=[3],
            dead_band=dead_band,
        )

    def test_agent_warmup_is_hold_with_zero_confidence(self):
        df = self.panel()
        for band in (0.0, 0.02):
            with self.subTest(dead_band=band):
                agent = SMAEMAAgent(
                    target_asset='SPY', sma_window=8,
                    ema_window=3, dead_band=band,
                )
                signal, confidence = agent.predict_with_confidence(df)
                self.assertTrue((signal.iloc[:7] == 0).all())
                self.assertTrue((confidence.iloc[:7] == 0.0).all())
                self.assertTrue(signal.iloc[7:].isin([-1, 0, 1]).all())

    def _assert_optimizer_neutral_before_first_valid_signal(self, band):
        df = self.panel()
        opt = self.optimizer(band)
        with patch(
            'src.agents.EMA_EMW_agent.calculate_performance_metrics',
            wraps=calculate_performance_metrics,
        ) as metrics:
            opt.fit(df)
        self.assertGreaterEqual(metrics.call_count, 1)
        train_returns = metrics.call_args_list[0].args[0]
        pd.testing.assert_index_equal(train_returns.index, df.index)
        np.testing.assert_allclose(train_returns.iloc[:8].to_numpy(), 0.0)
        self.assertGreater(train_returns.iloc[8:].abs().sum(), 0.0)

    def test_optimizer_neutral_warmup_without_dead_band(self):
        self._assert_optimizer_neutral_before_first_valid_signal(0.0)

    def test_optimizer_neutral_warmup_with_dead_band(self):
        self._assert_optimizer_neutral_before_first_valid_signal(0.001)

    def test_historical_warmup_enables_first_training_signal(self):
        df = self.panel()
        warmup, train = df.iloc[:20], df.iloc[20:]
        with patch(
            'src.agents.EMA_EMW_agent.calculate_performance_metrics',
            wraps=calculate_performance_metrics,
        ) as metrics:
            self.optimizer().fit(train, warmup_df=warmup)
        returns_with_history = metrics.call_args_list[0].args[0]
        self.assertEqual(len(returns_with_history), len(train))
        pd.testing.assert_index_equal(returns_with_history.index, train.index)
        self.assertGreater(returns_with_history.iloc[0], 0.0)

        with patch(
            'src.agents.EMA_EMW_agent.calculate_performance_metrics',
            wraps=calculate_performance_metrics,
        ) as metrics:
            self.optimizer().fit(train)
        returns_without_history = metrics.call_args_list[0].args[0]
        self.assertEqual(returns_without_history.iloc[0], 0.0)

    def test_optimizer_rejects_unordered_or_duplicate_dates(self):
        df = self.panel()
        for bad_df in (df.iloc[::-1], pd.concat([df, df.iloc[-1:]])):
            with self.subTest(index_type='duplicate' if bad_df.index.has_duplicates else 'unordered'):
                with self.assertRaises(ValueError):
                    self.optimizer().fit(bad_df)

    def test_future_validation_does_not_change_training_selection(self):
        df = self.panel(95)
        train, validation = df.iloc[:65], df.iloc[65:]
        opt_train = self.optimizer()
        opt_val = self.optimizer()
        result_train = opt_train.fit(train)
        result_val = opt_val.fit(train, val_df=validation)
        self.assertEqual(result_train['sma_window'], result_val['sma_window'])
        self.assertEqual(result_train['ema_window'], result_val['ema_window'])
        self.assertEqual(result_train['train_score'], result_val['train_score'])


if __name__ == '__main__':
    unittest.main()
