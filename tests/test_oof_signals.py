"""Regression tests for RF/SVM out-of-fold signal production and meta-training (II.5).

Run from the repository root:
    python3 -m unittest discover -s tests -p 'test_oof_signals.py' -v

The tests check temporal separation, OOF-only inputs to the orchestrator,
and that the saved full-fit models are not used to score their own training rows.
"""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.agents import train as train_module
from src.agents.random_forest_agent import (
    RandomForestOptimizer,
    RandomForestWalkForwardTrainer,
)
from src.agents.SVM_agent import SVMWalkForwardTrainer


class TestOOFSignals(unittest.TestCase):
    @staticmethod
    def panel():
        idx = pd.date_range('2009-01-02', '2015-12-31', freq='W-FRI')
        t = np.arange(len(idx), dtype=float)
        return pd.DataFrame({
            'SPY_close': 100.0 + 0.11 * t + 5.0 * np.sin(t / 2.8),
            'known_factor': np.cos(t / 2.8),
        }, index=idx)

    @staticmethod
    def rf_trainer():
        opt = RandomForestOptimizer(
            target_asset='SPY',
            price_col='SPY_close',
            feature_cols=['known_factor'],
            depth_candidates=[2],
            min_leaf_candidates=[4],
            threshold_candidates=[0.52],
            n_estimators_candidates=[10],
        )
        return RandomForestWalkForwardTrainer(
            optimizer=opt, train_window_years=5, target_asset='SPY'
        )

    @staticmethod
    def svm_trainer():
        settings = {
            'horizon_weeks': 1, 'first_test_year': 2014, 'min_train_weeks': 52,
            'cv_splits': 2, 'C_values': [1.0], 'gamma_values': ['scale'],
            'class_weight': 'none', 'neutral_band': 0.25, 'long_only': False,
            'use_macro_features': False,
        }
        return SVMWalkForwardTrainer(target_asset='SPY', settings=settings)

    def _check_real_folds(self, trainer):
        df = self.panel()
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.run(df, max_year=2015)

        expected = df.index[df.index.year >= 2014]
        pd.testing.assert_index_equal(trainer.oof_signals.index, expected)
        self.assertEqual(list(trainer.oof_signals.columns), ['signal', 'confidence'])
        self.assertFalse(trainer.oof_signals.isna().any().any())
        self.assertTrue(trainer.oof_signals['signal'].isin([-1, 0, 1]).all())
        self.assertTrue(trainer.oof_signals['confidence'].between(0, 1).all())
        self.assertEqual([h['val_year'] for h in trainer.history], [2014, 2015])
        self.assertIsNotNone(trainer.final_agent)
        self.assertIsNot(trainer.final_agent, trainer.optimizer.best_agent)

    def test_random_forest_only_predicts_outside_each_fold_training(self):
        self._check_real_folds(self.rf_trainer())

    def test_svm_only_predicts_outside_each_fold_training(self):
        df = self.panel()
        for other in ('IEF', 'GLD', 'SHY'):
            df[f'{other}_close'] = df['SPY_close'] * 0.5 + 10.0
        trainer = self.svm_trainer()
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.run(df, max_year=2015)
        expected = df.index[df.index.year >= 2014]
        pd.testing.assert_index_equal(trainer.oof_signals.index, expected)
        self.assertEqual(list(trainer.oof_signals.columns), ['signal', 'confidence'])
        self.assertFalse(trainer.oof_signals.isna().any().any())
        self.assertTrue(trainer.oof_signals['signal'].isin([-1, 0, 1]).all())
        self.assertTrue(trainer.oof_signals['confidence'].between(0, 1).all())
        self.assertEqual([h['year'] for h in trainer.history], [2014, 2015])
        self.assertIsNotNone(trainer.final_agent)

    def _run_pipeline_with_mocked_other_agents(self, rf_index, svm_index):
        """Run the real train_single_asset wiring without expensive outer searches.

        The mock final models deliberately have no predict method. If train.py
        mistakenly scores using the full-fit models, this test will fail.
        """
        df = self.panel()
        all_dates = df.index
        captured = {}

        def fixed_signal(frame):
            return (
                pd.Series(1, index=frame.index, dtype=int),
                pd.Series(0.75, index=frame.index, dtype=float),
            )

        def fake_rf_run(trainer, data, max_year):
            trainer.oof_signals = pd.DataFrame({
                'signal': 1, 'confidence': 0.8,
            }, index=rf_index)
            trainer.final_agent = object()
            return {'mock': 'rf'}

        def fake_svm_run(trainer, data, max_year):
            trainer.oof_signals = pd.DataFrame({
                'signal': -1, 'confidence': 0.7,
            }, index=svm_index)
            trainer.final_agent = object()
            return {'mock': 'svm'}

        def fake_optimize(optimizer, signals_dict, returns_or_prices, asset_name='ASSET', method='ga'):
            captured['pool'] = signals_dict
            captured['returns'] = returns_or_prices
            optimizer.best_params = {'mock': True}
            fake_agent = MagicMock()
            fake_agent.combine.side_effect = lambda pool: pd.DataFrame({
                'signal': 0, 'confidence': 0.0,
            }, index=next(iter(pool.values())).index)
            return fake_agent, pd.DataFrame({'mock': [1]})

        sma_summary = pd.DataFrame([{
            'sma_window': 20, 'ema_window': 10, 'times_best': 1,
            'selection_pct': 100.0, 'mean_oos_sharpe': 0.0,
            'mean_oos_return_pct': 0.0, 'oos_win_rate_pct': 0.0,
        }])
        z_summary = pd.DataFrame([{
            'window': 20, 'threshold': 1.5, 'mode': 'momentum',
            'times_best': 1, 'selection_pct': 100.0, 'mean_oos_sharpe': 0.0,
            'mean_oos_return_pct': 0.0, 'oos_win_rate_pct': 0.0,
        }])

        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            root = Path(tmp)
            for folder in ('config', 'signals', 'results'):
                (root / folder).mkdir()

            for trainer_cls, summary in (
                (train_module.SMAEMATrainer, sma_summary),
                (train_module.ZScoreWalkForwardTrainer, z_summary),
            ):
                stack.enter_context(patch.object(trainer_cls, 'run', return_value={}))
                stack.enter_context(patch.object(trainer_cls, 'save_final_agent_config'))
                stack.enter_context(patch.object(trainer_cls, 'save_combinations_log'))
                stack.enter_context(patch.object(trainer_cls, 'save_combinations_summary'))
                stack.enter_context(patch.object(
                    trainer_cls, 'get_combinations_summary', return_value=summary
                ))

            for agent_cls in (train_module.SMAEMAAgent, train_module.ZScoreAgent):
                stack.enter_context(patch.object(agent_cls, 'load_params'))
                stack.enter_context(patch.object(
                    agent_cls, 'predict_with_confidence', side_effect=fixed_signal
                ))

            stack.enter_context(patch.object(
                train_module.MacroYieldCurveAgent, 'predict_multi_asset',
                side_effect=lambda frame, assets=None: pd.DataFrame({
                    'SPY_signal': np.ones(len(frame), dtype=int)
                }, index=frame.index),
            ))
            stack.enter_context(patch.object(
                train_module.MacroYieldCurveAgent, 'calculate_macro_score',
                side_effect=lambda frame: pd.Series(1.0, index=frame.index),
            ))

            stack.enter_context(patch.object(
                train_module.RandomForestWalkForwardTrainer, 'run', new=fake_rf_run
            ))
            stack.enter_context(patch.object(
                train_module.SVMWalkForwardTrainer, 'run', new=fake_svm_run
            ))
            stack.enter_context(patch.object(
                train_module.RandomForestWalkForwardTrainer, 'save_final_agent_config'
            ))
            stack.enter_context(patch.object(
                train_module.SVMWalkForwardTrainer, 'save_final_agent_config'
            ))
            stack.enter_context(patch.object(train_module, 'save_training_provenance'))
            stack.enter_context(patch.object(
                train_module.OrchestratorOptimizer, 'optimize', new=fake_optimize
            ))
            stack.enter_context(patch.object(
                train_module.OrchestratorOptimizer, 'save_final_config'
            ))

            with contextlib.redirect_stdout(io.StringIO()):
                output = train_module.train_single_asset(
                    asset='SPY', df=df,
                    config_dir=root / 'config',
                    signals_dir=root / 'signals',
                    results_dir=root / 'results',
                    max_train_year=2015,
                    train_rf=rf_index is not None,
                    train_svm=svm_index is not None,
                    train_orchestrator=True,
                )
            captured['output'] = output
            captured['dates'] = all_dates
            return captured

    def test_orchestrator_uses_oof_intersection_not_full_fit_models(self):
        dates = self.panel().index
        rf_idx = dates[dates.year >= 2014]
        svm_idx = dates[dates.year == 2015]
        result = self._run_pipeline_with_mocked_other_agents(rf_idx, svm_idx)
        pool = result['pool']
        expected = svm_idx
        self.assertEqual(set(pool), {
            'SMA_EMA_Agent', 'ZScore_Agent', 'Macro_YieldCurve_Agent',
            'RandomForest_Agent', 'SVM_Agent',
        })
        for signals in pool.values():
            pd.testing.assert_index_equal(signals.index, expected)
        pd.testing.assert_index_equal(result['returns'].index, expected)
        pd.testing.assert_index_equal(result['output']['signals_orchestrator'].index, expected)
        self.assertTrue((pool['RandomForest_Agent']['signal'] == 1).all())
        self.assertTrue((pool['SVM_Agent']['signal'] == -1).all())

    def test_pipeline_without_ml_keeps_all_training_dates(self):
        result = self._run_pipeline_with_mocked_other_agents(None, None)
        pool = result['pool']
        self.assertEqual(set(pool), {
            'SMA_EMA_Agent', 'ZScore_Agent', 'Macro_YieldCurve_Agent'
        })
        for signals in pool.values():
            pd.testing.assert_index_equal(signals.index, result['dates'])
        pd.testing.assert_index_equal(result['returns'].index, result['dates'])

    def test_pipeline_rejects_disjoint_oof_folds(self):
        dates = self.panel().index
        rf_idx = dates[dates.year == 2014]
        svm_idx = dates[dates.year == 2015]
        with self.assertRaisesRegex(ValueError, 'OOF'):
            self._run_pipeline_with_mocked_other_agents(rf_idx, svm_idx)


if __name__ == '__main__':
    unittest.main()
