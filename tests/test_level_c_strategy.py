"""Synthetic, non-historical data-adapter smoke tests (no validation/OOS)."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.level_c_strategy import build_level_c_training
from src.regime_allocator import ASSETS


class TestLevelCStrategyAdapter(TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.paths = {'signals': 'data/signals', 'processed_data': 'data/processed'}
        signal_dir = self.root / 'data/signals'
        signal_dir.mkdir(parents=True)
        self.sessions = pd.bdate_range('2014-01-02', periods=200, name='Date')
        self.decisions = self.sessions[[150, 155, 160]]
        self.start = pd.Timestamp('2014-01-02')
        self.end = self.sessions[-1]
        self.market = pd.DataFrame(index=self.sessions)
        for asset in ASSETS:
            self.market[f'{asset}_close'] = 100 * np.exp(np.arange(len(self.sessions)) * 0.0001)
            pd.DataFrame({'signal': [1, 0, 1], 'confidence': [.6, .8, .6]},
                         index=self.decisions).to_csv(
                signal_dir / f'signals_orchestrator_{asset.lower()}.csv'
            )
        self.budgets = pd.DataFrame(np.tile([.25, .30, .20, .25], (3, 1)),
                                    index=self.decisions, columns=ASSETS)
        self.trend = pd.DataFrame(1, index=self.decisions, columns=ASSETS)
        self.config = {
            'validation': {'development_only': True},
            'evaluation': {'evaluate_oos': False},
            'paths': self.paths,
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_reads_all_agents_and_builds_training_targets(self):
        with patch('src.level_c_strategy.get_time_windows', return_value={'train_start': self.start, 'train_end': self.end}), \
             patch('src.level_c_strategy.build_level_b_v2_training', return_value={
                 'budgets': self.budgets, 'trend_signals': self.trend,
                 'target_weights': self.budgets, 'regimes': pd.Series('low_low', index=self.decisions),
             }), \
             patch('src.level_c_strategy.load_dataset', return_value=self.market):
            result = build_level_c_training(self.config, self.root)
        self.assertEqual(list(result['target_weights'].columns), list(ASSETS))
        self.assertTrue((result['raw_weights'].iloc[1] == 0).all())
        self.assertEqual(len(result['target_weights']), 3)
        self.assertTrue(result['target_weights'].ge(0).all().all())
        self.assertEqual(result['sample'], 'training_diagnostic_only_not_independent_oos')
        self.assertEqual(result['regimes'].iloc[0], 'low_low')

    def test_oos_unlock_and_non_development_rejected(self):
        cfg = {**self.config, 'evaluation': {'evaluate_oos': True}}
        with self.assertRaises(ValueError):
            build_level_c_training(cfg, self.root)
        cfg = {**self.config, 'validation': {'development_only': False}}
        with self.assertRaises(ValueError):
            build_level_c_training(cfg, self.root)

    def test_missing_agent_export_rejected(self):
        (self.root / 'data/signals/signals_orchestrator_spy.csv').unlink()
        with patch('src.level_c_strategy.get_time_windows', return_value={'train_start': self.start, 'train_end': self.end}), \
             self.assertRaises(FileNotFoundError):
            build_level_c_training(self.config, self.root)
