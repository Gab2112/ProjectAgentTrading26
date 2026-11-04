"""Regression tests for future weekly return labels and cutoff isolation (II.2).

Random Forest only: the SVM agent has its own equivalent checks in
tests/test_svm_agent.py (target, cutoff, unknown labels, invalid dates).

Run from repository root:
    python3 -m unittest discover -s tests -p "test_ml_targets.py" -v
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.agents.random_forest_agent import RandomForestAgent


class TestMLFutureReturnTargets(unittest.TestCase):
    @staticmethod
    def _agent_classes():
        return (RandomForestAgent,)

    @staticmethod
    def _panel(n=160):
        dates = pd.date_range("2015-01-02", periods=n, freq="W-FRI")
        t = np.arange(n, dtype=float)
        return pd.DataFrame(
            {
                "SPY_close": 100.0 + 6.0 * np.sin(t / 2.7) + 0.05 * t,
                "known_factor": np.cos(t / 4.1),
            },
            index=dates,
        )

    @staticmethod
    def _new_agent(agent_type):
        kwargs = {"target_asset": "SPY", "feature_cols": ["known_factor"]}
        if agent_type is RandomForestAgent:
            kwargs.update({"n_estimators": 12, "max_depth": 2, "min_samples_leaf": 4})
        return agent_type(**kwargs)

    def test_target_refers_to_next_week_and_preserves_missing_label(self):
        dates = pd.date_range("2016-11-04", periods=5, freq="W-FRI")
        df = pd.DataFrame(
            {
                "SPY_close": [100.0, 110.0, 110.0, 90.0, 120.0],
                "known_factor": [0.0, 1.0, 2.0, 3.0, 4.0],
            },
            index=dates,
        )
        for agent_type in self._agent_classes():
            with self.subTest(agent=agent_type.__name__):
                _, y, future_returns = self._new_agent(agent_type)._extract_xy(df.copy())
                np.testing.assert_allclose(y.iloc[:4].to_numpy(), [1.0, 0.0, 0.0, 1.0])
                self.assertTrue(pd.isna(y.iloc[-1]))
                self.assertTrue(pd.isna(future_returns.iloc[-1]))
                self.assertAlmostEqual(future_returns.iloc[0], 0.10)
                self.assertAlmostEqual(future_returns.iloc[2], 90.0 / 110.0 - 1.0)

    def test_nan_price_does_not_turn_into_false_down_label(self):
        dates = pd.date_range("2016-11-04", periods=4, freq="W-FRI")
        df = pd.DataFrame(
            {
                "SPY_close": [100.0, np.nan, 115.0, 125.0],
                "known_factor": [1.0, 1.0, 1.0, 1.0],
            },
            index=dates,
        )
        for agent_type in self._agent_classes():
            with self.subTest(agent=agent_type.__name__):
                _, y, _ = self._new_agent(agent_type)._extract_xy(df.copy())
                self.assertTrue(pd.isna(y.iloc[0]))
                self.assertTrue(pd.isna(y.iloc[1]))
                self.assertEqual(y.iloc[2], 1.0)
                self.assertTrue(pd.isna(y.iloc[3]))

    def test_cutoff_precedes_target_generation(self):
        full = self._panel()
        cutoff = full.loc[full.index.year <= 2016].copy()
        # The first 2017 close is deliberately extreme: its direction must
        # never become the label of the final 2016 training observation.
        full.loc[full.index.year >= 2017, "SPY_close"] += 500.0

        for agent_type in self._agent_classes():
            with self.subTest(agent=agent_type.__name__):
                from_full = self._new_agent(agent_type).fit(full, max_year=2016)
                from_cutoff = self._new_agent(agent_type).fit(cutoff, max_year=None)
                features = from_cutoff.trained_features
                probe = cutoff[features]
                p_a = from_full.predict_proba(probe)
                p_b = from_cutoff.predict_proba(probe)
                np.testing.assert_allclose(p_a.values, p_b.values, atol=1e-12)

    def test_last_row_is_not_used_for_fit(self):
        df = self._panel()
        cutoff = df.loc[df.index.year <= 2016]
        expected_count = len(cutoff) - 1

        original_rf_fit = RandomForestClassifier.fit
        seen_rf = []

        def capture_rf(model, X, y, *args, **kwargs):
            seen_rf.append((X.index.copy(), len(y)))
            return original_rf_fit(model, X, y, *args, **kwargs)

        with patch.object(RandomForestClassifier, "fit", capture_rf):
            self._new_agent(RandomForestAgent).fit(df, max_year=2016)

        self.assertEqual(len(seen_rf), 1)
        self.assertEqual(seen_rf[0][1], expected_count)
        self.assertNotIn(cutoff.index[-1], seen_rf[0][0])

    def test_single_class_is_rejected_before_model_training(self):
        dates = pd.date_range("2015-01-02", periods=85, freq="W-FRI")
        rising = pd.DataFrame(
            {"SPY_close": 100.0 + np.arange(85), "known_factor": np.arange(85)},
            index=dates,
        )
        for agent_type in self._agent_classes():
            with self.subTest(agent=agent_type.__name__):
                with self.assertRaisesRegex(ValueError, "classi del target"):
                    self._new_agent(agent_type).fit(rising, max_year=None)

    def test_training_rejects_unsorted_or_duplicate_dates(self):
        df = self._panel(85)
        broken = df.iloc[::-1]
        duplicate = pd.concat([df, df.iloc[[-1]]])
        for agent_type in self._agent_classes():
            for invalid in (broken, duplicate):
                with self.subTest(agent=agent_type.__name__, rows=len(invalid)):
                    with self.assertRaisesRegex(ValueError, "Indice temporale"):
                        self._new_agent(agent_type).fit(invalid, max_year=None)


if __name__ == "__main__":
    unittest.main()
