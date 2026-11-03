"""Regression tests: agent signals -> daily next-open execution, training only."""

import unittest

import numpy as np
import pandas as pd

from src.agents.agent_execution_bridge import (
    signals_to_target_weights,
    run_agent_training_execution_diagnostic,
)
from src.regime_allocator import ASSETS


class TestAgentExecutionBridgeII7(unittest.TestCase):
    def setUp(self):
        self.sessions = pd.bdate_range('2015-01-05', periods=21).difference(
            pd.DatetimeIndex(['2015-01-19'])  # MLK Day: market closed.
        )
        self.d1 = pd.Timestamp('2015-01-09')  # Friday after close
        self.d2 = pd.Timestamp('2015-01-16')
        self.signal_dates = pd.DatetimeIndex([self.d1, self.d2])
        self.spysig = pd.DataFrame(
            {'signal': [1, -1], 'confidence': [1.0, 1.0]},
            index=self.signal_dates,
        )
        self.config = {
            'validation': {
                'development_only': True,
                'train_start': '2015-01-05',
                'train_end': '2015-01-23',
            },
            'evaluation': {'evaluate_oos': False},
            'execution': {
                'execution_rule': 'next_available_trading_open',
                'signal_frequency': 'weekly',
                'cost_basis': 'traded_notional',
                'initial_capital': 100_000.0,
                'commission_bps': 2.0,
                'slippage_bps': 1.0,
            },
        }
        self.market_daily = pd.DataFrame(index=self.sessions)
        for asset in ASSETS:
            self.market_daily[f'{asset}_open'] = 100.0
            self.market_daily[f'{asset}_close'] = 100.0
        # A 10% gap occurs at the next Open; the strategy cannot earn it.
        self.market_daily.loc[self.sessions[5]:, 'SPY_open'] = 110.0
        self.market_daily.loc[self.sessions[5]:, 'SPY_close'] = 110.0
        self.market_daily.loc[self.sessions[6], 'SPY_close'] = 121.0

    def test_sell_is_flat_and_cash_is_allowed(self):
        weights = signals_to_target_weights({'SPY': self.spysig})
        self.assertEqual(list(weights.columns), list(ASSETS))
        self.assertAlmostEqual(weights.loc[self.d1, 'SPY'], 0.25)
        self.assertAlmostEqual(weights.loc[self.d2, 'SPY'], 0.0)
        self.assertTrue((weights.loc[:, ['IEF', 'GLD', 'SHY']] == 0.0).all().all())
        self.assertTrue((weights.sum(axis=1) <= 1.0).all())

    def test_next_open_execution_and_no_gap_capture(self):
        result = run_agent_training_execution_diagnostic(
            self.config, self.market_daily, {'SPY': self.spysig}
        )
        scheduled = result['execution_schedule']
        daily = result['daily_backtest']
        self.assertEqual(list(scheduled.index), [pd.Timestamp('2015-01-12'), pd.Timestamp('2015-01-20')])
        self.assertEqual(result['sample'], 'training_diagnostic_only_not_independent_oos')
        self.assertFalse(bool(daily.loc[self.d1, 'executed']))
        self.assertTrue(bool(daily.loc[pd.Timestamp('2015-01-12'), 'executed']))
        self.assertAlmostEqual(daily.loc[pd.Timestamp('2015-01-12'), 'nav'], 100000.0, delta=10.0)
        self.assertGreater(daily.loc[pd.Timestamp('2015-01-12'), 'transaction_cost'], 0.0)
        self.assertAlmostEqual(daily.loc[pd.Timestamp('2015-01-12'), 'transaction_cost'], 7.50, delta=0.10)
        self.assertGreater(daily.loc[pd.Timestamp('2015-01-13'), 'nav'], daily.loc[pd.Timestamp('2015-01-12'), 'nav'])

    def test_train_cutoff_excludes_future_market_prices(self):
        result = run_agent_training_execution_diagnostic(
            self.config, self.market_daily, {'SPY': self.spysig}
        )
        self.assertLessEqual(result['daily_backtest'].index.max(), pd.Timestamp('2015-01-23'))
        self.assertLessEqual(result['execution_schedule'].index.max(), pd.Timestamp('2015-01-23'))

    def test_reject_signal_outside_training(self):
        frame = self.spysig.copy()
        frame.index = pd.DatetimeIndex([self.d1, pd.Timestamp('2015-01-30')])
        with self.assertRaisesRegex(ValueError, 'fuori dal training'):
            run_agent_training_execution_diagnostic(
                self.config, self.market_daily, {'SPY': frame}
            )

    def test_reject_mismatched_asset_decision_dates(self):
        ief = self.spysig.iloc[:1].copy()
        with self.assertRaisesRegex(ValueError, 'stesse date'):
            signals_to_target_weights({'SPY': self.spysig, 'IEF': ief})

    def test_reject_nan_and_out_of_range_confidence(self):
        for confidence in (np.nan, -0.1, 1.5):
            bad = self.spysig.copy()
            bad.loc[self.d1, 'confidence'] = confidence
            with self.subTest(confidence=confidence):
                with self.assertRaises(ValueError):
                    signals_to_target_weights({'SPY': bad})

    def test_reject_oos_enabled_and_excess_budget(self):
        bad_config = {**self.config, 'evaluation': {'evaluate_oos': True}}
        with self.assertRaisesRegex(ValueError, 'OOS'):
            run_agent_training_execution_diagnostic(
                bad_config, self.market_daily, {'SPY': self.spysig}
            )
        with self.assertRaisesRegex(ValueError, 'budget'):
            signals_to_target_weights(
                {'SPY': self.spysig}, asset_budgets={'SPY': 1.1}
            )


if __name__ == '__main__':
    unittest.main()
