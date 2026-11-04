"""Unit/regression tests for Level C v1 using synthetic TRAINING-ONLY data.

These tests must run from the repository root using its real regime allocator:
    python3 -m unittest tests.test_level_c_allocator -v
or:
    python3 -m unittest discover -s tests -p 'test_level_c_allocator.py' -v
They do NOT read financial CSVs or access 2017+/2022+ observations.
"""

from __future__ import annotations

from unittest import TestCase

import numpy as np
import pandas as pd

from src.level_c_allocator import build_level_c_target_weights
from src.portfolio import apply_volatility_cap_panel
from src.utils import ASSETS


class TestLevelCAllocator(TestCase):
    def setUp(self):
        self.sessions = pd.bdate_range('2014-01-02', periods=330, name='Date')
        self.decisions = pd.DatetimeIndex(
            [self.sessions[250], self.sessions[255], self.sessions[260]], name='Date'
        )
        self.train_start = pd.Timestamp('2014-01-02')
        self.train_end = self.sessions[-1]
        rng = np.random.default_rng(2026)
        self.returns = pd.DataFrame(
            rng.standard_normal((330, 4)) * [0.016, 0.005, 0.012, 0.001],
            index=self.sessions,
            columns=ASSETS,
        )
        self.budgets = pd.DataFrame(
            np.tile([.25, .30, .20, .25], (3, 1)),
            index=self.decisions, columns=ASSETS
        )
        self.trend = pd.DataFrame(
            [[1, 1, -1, 1], [1, 0, 1, 1], [-1, 1, 1, 1]],
            index=self.decisions, columns=ASSETS,
        )
        self.agents = {
            asset: pd.DataFrame({'signal': [1, 1, 1], 'confidence': [.7, .5, .9]},
                                index=self.decisions)
            for asset in ASSETS
        }

    def run_c(self, **overrides):
        params = dict(
            signals_by_asset=self.agents,
            b_budgets=self.budgets,
            b_trend_signals=self.trend,
            daily_returns=self.returns,
            train_start=self.train_start,
            train_end=self.train_end,
        )
        params.update(overrides)
        return build_level_c_target_weights(**params)

    def test_budget_momentum_agent_gate(self):
        agents = {key: value.copy() for key, value in self.agents.items()}
        agents['IEF'].loc[self.decisions[0], 'signal'] = -1
        agents['SHY'].loc[self.decisions[1], 'signal'] = 0
        agents['GLD'].loc[self.decisions[2], 'confidence'] = 0
        res = self.run_c(signals_by_asset=agents)
        self.assertAlmostEqual(res['raw_weights'].loc[self.decisions[0], 'SPY'], .25)
        self.assertAlmostEqual(res['raw_weights'].loc[self.decisions[0], 'IEF'], 0)
        self.assertAlmostEqual(res['raw_weights'].loc[self.decisions[0], 'GLD'], 0)
        self.assertAlmostEqual(res['raw_weights'].loc[self.decisions[1], 'SHY'], 0)
        self.assertAlmostEqual(res['raw_weights'].loc[self.decisions[2], 'GLD'], 0)
        self.assertEqual(res['sample'], 'training_diagnostic_only_not_independent_oos')

    def test_all_positive_agents_recover_bv2_weights(self):
        expected_raw = self.budgets * self.trend.eq(1)
        expected, expected_risk = apply_volatility_cap_panel(expected_raw, self.returns)
        actual = self.run_c()
        pd.testing.assert_frame_equal(actual['target_weights'], expected)
        pd.testing.assert_frame_equal(actual['risk'], expected_risk)

    def test_agent_confidence_is_gate_not_fractional_sizing(self):
        agents = {k: p.copy() for k, p in self.agents.items()}
        agents['SPY'].loc[:, 'confidence'] = .001
        res = self.run_c(signals_by_asset=agents)
        self.assertAlmostEqual(res['raw_weights'].loc[self.decisions[0], 'SPY'], .25)

    def test_all_flat_means_cash_not_short(self):
        agents = {k: p.copy() for k, p in self.agents.items()}
        for df in agents.values():
            df.loc[:, 'signal'] = -1
        res = self.run_c(signals_by_asset=agents)
        self.assertTrue((res['target_weights'] == 0).all().all())
        self.assertTrue(np.allclose(res['cash_weight_final'].to_numpy(), 1.))
        self.assertTrue((res['risk']['vol_after'] == 0).all())

    def test_budget_volatility_and_cash_constraints(self):
        res = self.run_c()
        target = res['target_weights']
        self.assertTrue((target >= 0).all().all())
        self.assertTrue((target <= res['budgets'] + 1e-10).all().all())
        self.assertTrue((target.sum(axis=1) <= 1.0).all())
        self.assertTrue((res['risk']['vol_after'] <= .08 + 1e-8).all())
        np.testing.assert_allclose(res['cash_weight_final'], 1.0 - target.sum(axis=1))
        self.assertEqual(list(res['risk'].columns), ['vol_before', 'vol_after', 'scaling_factor'])
        self.assertTrue((res['risk']['scaling_factor'] <= 1).all())

    def test_strong_risk_triggers_single_cap(self):
        high = self.returns * 10
        res = self.run_c(daily_returns=high)
        self.assertTrue((res['risk']['scaling_factor'] < 1).any())
        for d in self.decisions:
            x = res['raw_weights'].loc[d].to_numpy(dtype=float)
            y = res['target_weights'].loc[d].to_numpy(dtype=float)
            np.testing.assert_allclose(y, x * res['risk'].loc[d, 'scaling_factor'])

    def test_future_returns_cannot_change_earlier_decisions(self):
        a = self.run_c()
        changed = self.returns.copy()
        changed.loc[self.sessions[self.sessions > self.decisions[0]], :] *= 50
        b = self.run_c(daily_returns=changed)
        pd.testing.assert_series_equal(
            a['target_weights'].iloc[0], b['target_weights'].iloc[0]
        )
        pd.testing.assert_series_equal(a['risk'].iloc[0], b['risk'].iloc[0])

    def test_changing_later_agent_signal_does_not_change_past(self):
        agents = {key: value.copy() for key, value in self.agents.items()}
        agents['SPY'].iloc[-1, agents['SPY'].columns.get_loc('signal')] = -1
        a, b = self.run_c(), self.run_c(signals_by_asset=agents)
        pd.testing.assert_series_equal(a['target_weights'].iloc[0], b['target_weights'].iloc[0])

    def test_missing_agent_or_date_is_error(self):
        missing_asset = {k: v for k, v in self.agents.items() if k != 'SHY'}
        with self.assertRaisesRegex(ValueError, 'requires all'):
            self.run_c(signals_by_asset=missing_asset)
        shifted = {k: v.copy() for k, v in self.agents.items()}
        shifted['IEF'] = shifted['IEF'].iloc[1:]
        with self.assertRaisesRegex(ValueError, 'match exactly'):
            self.run_c(signals_by_asset=shifted)

    def test_invalid_confidence_and_signals_are_error(self):
        for conf in (np.nan, -0.1, 1.2):
            agents = {k: v.copy() for k, v in self.agents.items()}
            agents['SPY'].loc[self.decisions[0], 'confidence'] = conf
            with self.subTest(conf=conf), self.assertRaises(ValueError):
                self.run_c(signals_by_asset=agents)
        agents = {k: v.copy() for k, v in self.agents.items()}
        agents['GLD'].loc[self.decisions[0], 'signal'] = 2
        with self.assertRaisesRegex(ValueError, 'signals must'):
            self.run_c(signals_by_asset=agents)

    def test_no_training_boundary_bypass(self):
        with self.assertRaisesRegex(ValueError, 'training boundaries'):
            self.run_c(train_end=pd.Timestamp('2014-08-01'))
        future = self.returns.copy()
        future.loc[self.train_end + pd.offsets.BDay(1)] = [.1, .1, .1, .1]
        with self.assertRaisesRegex(ValueError, 'future observations'):
            self.run_c(daily_returns=future)
        with self.assertRaisesRegex(ValueError, 'training boundaries'):
            self.run_c(b_budgets=self.budgets.rename(index={self.decisions[-1]: self.train_end + pd.offsets.BDay(1)}))

    def test_no_silent_missing_macro_regimes_or_invalid_budgets(self):
        bad = self.budgets.copy()
        bad.loc[self.decisions[0], 'SPY'] = np.nan
        with self.assertRaises(ValueError):
            self.run_c(b_budgets=bad)
        bad = self.budgets.copy()
        bad.loc[self.decisions[0], 'SPY'] = .5
        with self.assertRaisesRegex(ValueError, 'sum to 100%'):
            self.run_c(b_budgets=bad)
        bad = self.budgets.iloc[1:]
        with self.assertRaisesRegex(ValueError, 'B_v2 budgets/trend calendars differ'):
            self.run_c(b_budgets=bad)

    def test_insufficient_history_and_invalid_vol_params(self):
        with self.assertRaisesRegex(ValueError, 'Insufficient covariance'):
            self.run_c(daily_returns=self.returns.loc[:self.decisions[-1]].tail(100))
        with self.assertRaisesRegex(ValueError, 'covariance_window'):
            self.run_c(covariance_window=1)
        with self.assertRaisesRegex(ValueError, 'target_volatility'):
            self.run_c(target_volatility=-.08)

    def test_duplicate_and_unordered_dates_rejected(self):
        agents = {key: val.copy() for key, val in self.agents.items()}
        agents['SPY'].index = pd.DatetimeIndex([self.decisions[0], self.decisions[0], self.decisions[-1]])
        with self.assertRaises(ValueError):
            self.run_c(signals_by_asset=agents)
        bad = self.returns.iloc[::-1]
        with self.assertRaises(ValueError):
            self.run_c(daily_returns=bad)

    def test_replay_deterministic_outputs(self):
        a = self.run_c()
        b = self.run_c()
        pd.testing.assert_frame_equal(a['target_weights'], b['target_weights'])
        pd.testing.assert_frame_equal(a['risk'], b['risk'])


if __name__ == '__main__':
    import unittest
    unittest.main()
