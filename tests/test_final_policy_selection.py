"""Guardrails for 2021-only policy ranking and archived 2022-2026 comparison."""
import unittest
from pathlib import Path
import numpy as np
from src.development_policy_selection import (
    DEVELOPMENT_END, POLICY_NAMES, evaluate_development,
    evaluate_final_policy_comparison,
)
from src.utils import ROOT_DIR

class FinalPolicySelectionTests(unittest.TestCase):
    def test_development_only_inputs_and_metrics(self):
        result = evaluate_development(ROOT_DIR)
        self.assertEqual(set(result.index), set(POLICY_NAMES))
        self.assertTrue((result['last_decision'] <= DEVELOPMENT_END).all())
        self.assertTrue((result['decisions'] == 626).all())
        self.assertAlmostEqual(float(result.loc['Equal_Active_4ETF','sharpe_ratio']), 1.352847, delta=.0001)
        self.assertAlmostEqual(float(result.loc['Equal_Active_4ETF','max_drawdown']), -.092512, delta=.0001)
        self.assertEqual(result['sharpe_ratio'].idxmax(), 'Equal_Active_4ETF')

    def test_archived_core_still_matches(self):
        development, later = evaluate_final_policy_comparison(ROOT_DIR)
        self.assertEqual(set(later.index),set(POLICY_NAMES))
        self.assertAlmostEqual(float(later.at['Core35_Satellite65','cagr']), .0587194608265, places=10)
        self.assertAlmostEqual(float(later.at['Regime_Adaptive_SMA200','sharpe_ratio']), .91199542712, places=9)

if __name__=='__main__':
    unittest.main()
