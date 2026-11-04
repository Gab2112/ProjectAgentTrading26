"""Synthetic, training-only tests for experimental C_v2 Long/Short/Flat."""

from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.level_c_allocator import build_level_c_target_weights
from src.level_c_v2_strategy import (
    build_level_c_v2_target_weights,
    build_level_c_v2_training,
)
from src.regime_allocator import ASSETS


class TestLevelCV2(TestCase):
    def setUp(self):
        self.sessions = pd.bdate_range("2014-01-02", periods=310, name="Date")
        self.decisions = pd.DatetimeIndex(
            [self.sessions[220], self.sessions[225], self.sessions[230]], name="Date"
        )
        self.start = pd.Timestamp("2014-01-02")
        self.end = self.sessions[-1]
        rng = np.random.default_rng(177)
        self.returns = pd.DataFrame(
            rng.normal(size=(310, 4)) * [0.012, 0.005, 0.01, 0.001],
            index=self.sessions, columns=ASSETS,
        )
        self.budgets = pd.DataFrame(
            np.tile([0.25, 0.30, 0.20, 0.25], (3, 1)),
            index=self.decisions, columns=ASSETS,
        )
        self.trend = pd.DataFrame(
            [[1, -1, -1, 0], [-1, 1, 1, -1], [0, -1, 1, 1]],
            index=self.decisions, columns=ASSETS,
        )
        self.agents = {
            asset: pd.DataFrame({"signal": [1, 1, 1], "confidence": [.7, .7, .7]},
                                index=self.decisions)
            for asset in ASSETS
        }
        self.agents["IEF"].loc[:, "signal"] = [-1, 1, -1]
        self.agents["GLD"].loc[:, "signal"] = [1, -1, 1]
        self.agents["SHY"].loc[:, "signal"] = [-1, -1, 0]

    def build(self, *, agents=None, trend=None, returns=None, volatility=0.08):
        agents = self.agents if agents is None else agents
        trend = self.trend if trend is None else trend
        returns = self.returns if returns is None else returns
        v1 = build_level_c_target_weights(
            signals_by_asset=agents,
            b_budgets=self.budgets,
            b_trend_signals=trend,
            daily_returns=returns,
            train_start=self.start,
            train_end=self.end,
        )
        return build_level_c_v2_target_weights(
            v1, trend, returns,
            train_start=self.start,
            train_end=self.end,
            target_volatility=volatility,
        )

    def test_signed_consensus_and_disagreement(self):
        result = self.build()
        expected = pd.DataFrame(
            [[.25, -.30, 0, 0], [0, .30, 0, -.25], [0, -.30, .20, 0]],
            index=self.decisions, columns=ASSETS,
        )
        pd.testing.assert_frame_equal(result["raw_weights"], expected)
        self.assertTrue((result["target_weights"].loc[:, "IEF"] < 0).any())
        self.assertTrue(result["risk"]["vol_after"].le(.08 + 1e-8).all())

    def test_no_short_without_negative_momentum_and_agent(self):
        result = self.build()
        for date in self.decisions:
            for asset in ASSETS:
                if result["target_weights"].loc[date, asset] < 0:
                    self.assertEqual(self.trend.loc[date, asset], -1)
                    self.assertEqual(self.agents[asset].loc[date, "signal"], -1)

    def test_zero_confidence_prevents_any_exposure(self):
        agents = {asset: df.copy() for asset, df in self.agents.items()}
        for panel in agents.values():
            panel["confidence"] = 0.0
        result = self.build(agents=agents)
        self.assertTrue(result["target_weights"].eq(0).all().all())
        self.assertTrue(result["unused_gross_budget_final"].eq(1).all())

    def test_confidence_is_a_gate_not_a_fraction(self):
        agents = {asset: df.copy() for asset, df in self.agents.items()}
        agents["IEF"]["confidence"] = 0.001
        result = self.build(agents=agents)
        self.assertEqual(result["raw_weights"].loc[self.decisions[0], "IEF"], -.30)

    def test_signed_risk_overlay_scales_only_once(self):
        result = self.build(returns=self.returns * 20)
        self.assertTrue(result["risk"]["scaling_factor"].lt(1).any())
        for date in self.decisions:
            factor = result["risk"].loc[date, "scaling_factor"]
            np.testing.assert_allclose(
                result["target_weights"].loc[date].to_numpy(),
                result["raw_weights"].loc[date].to_numpy() * factor,
            )
        self.assertTrue(result["gross_target_exposure"].le(1 + 1e-10).all())
        self.assertTrue(result["unused_gross_budget_final"].ge(-1e-10).all())
        self.assertNotIn("cash_weight_final", result)  # Short proceeds != cash allocation.

    def test_future_returns_cannot_change_first_decision(self):
        original = self.build()
        changed = self.returns.copy()
        changed.loc[changed.index > self.decisions[0], :] *= 10
        alternate = self.build(returns=changed)
        pd.testing.assert_series_equal(
            original["target_weights"].iloc[0], alternate["target_weights"].iloc[0]
        )

    def test_future_agent_signal_cannot_change_first_decision(self):
        original = self.build()
        agents = {a: f.copy() for a, f in self.agents.items()}
        agents["IEF"].loc[self.decisions[-1], "signal"] = 1
        alternate = self.build(agents=agents)
        pd.testing.assert_series_equal(
            original["target_weights"].iloc[0], alternate["target_weights"].iloc[0]
        )

    def test_reject_trend_disagreement_with_c1(self):
        v1 = build_level_c_target_weights(
            self.agents, self.budgets, self.trend, self.returns,
            train_start=self.start, train_end=self.end,
        )
        corrupted = self.trend.copy()
        corrupted.loc[self.decisions[0], "SPY"] = -1
        with self.assertRaisesRegex(ValueError, "trend differs"):
            build_level_c_v2_target_weights(
                v1, corrupted, self.returns,
                train_start=self.start, train_end=self.end,
            )

    def test_reject_date_mismatch_and_future_input(self):
        v1 = build_level_c_target_weights(
            self.agents, self.budgets, self.trend, self.returns,
            train_start=self.start, train_end=self.end,
        )
        with self.assertRaisesRegex(ValueError, "dates must match"):
            build_level_c_v2_target_weights(
                v1, self.trend.iloc[1:], self.returns,
                train_start=self.start, train_end=self.end,
            )
        future = self.returns.copy()
        future.loc[self.end + pd.offsets.BDay(1)] = [0, 0, 0, 0]
        with self.assertRaisesRegex(ValueError, "training boundaries"):
            build_level_c_v2_target_weights(
                v1, self.trend, future,
                train_start=self.start, train_end=self.end,
            )

    def test_adapter_reuses_c1_dates_and_regime_budgets(self):
        """Smoke test of C_v2 assembly without reading external CSVs."""
        closes = pd.DataFrame(
            {
                f"{asset}_close": 100.0 * np.exp(
                    np.arange(len(self.sessions)) * [0.0004, -0.0002, 0.0003, -0.0001][i]
                )
                for i, asset in enumerate(ASSETS)
            },
            index=self.sessions,
        )
        synthetic_returns = closes.pct_change(fill_method=None).dropna()
        synthetic_returns.columns = list(ASSETS)
        v1 = build_level_c_target_weights(
            self.agents, self.budgets, self.trend, synthetic_returns,
            train_start=self.start, train_end=self.end,
        )
        v1["b_v2_target_weights_same_dates"] = self.budgets.copy()
        v1["regimes"] = pd.Series("low_low", index=self.decisions)
        cfg = {
            "validation": {"development_only": True},
            "evaluation": {"evaluate_oos": False},
            "paths": {"processed_data": "data/processed"},
        }
        with patch("src.level_c_v2_strategy.build_level_c_training", return_value=v1), \
             patch("src.level_c_v2_strategy.get_time_windows", return_value={
                 "train_start": self.start, "train_end": self.end,
             }), \
             patch("src.level_c_v2_strategy.load_dataset", return_value=closes), \
             patch("src.level_c_v2_strategy.build_trend_signal_panel", return_value=self.trend):
            out = build_level_c_v2_training(cfg, ".")
        self.assertTrue(out["target_weights"].index.equals(self.decisions))
        pd.testing.assert_frame_equal(out["b_v2_target_weights_same_dates"], self.budgets)
        self.assertTrue(out["target_weights"].lt(0).any().any())

    def test_adapter_rejects_oos_and_non_development(self):
        cfg = {"validation": {"development_only": True}, "evaluation": {"evaluate_oos": True}}
        with self.assertRaisesRegex(ValueError, "training-only"):
            build_level_c_v2_training(cfg)
        cfg = {"validation": {"development_only": False}, "evaluation": {"evaluate_oos": False}}
        with self.assertRaisesRegex(ValueError, "training-only"):
            build_level_c_v2_training(cfg)


if __name__ == "__main__":
    import unittest
    unittest.main()
