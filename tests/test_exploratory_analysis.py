"""Non-destructive regression tests for post-freeze research modes."""
import tempfile
import unittest
from pathlib import Path

from src.submission_runner import (
    _research_output_directory,
    calculate_archived_robustness,
    calculate_regime_adaptive_research,
)
from src.utils import ROOT_DIR


class ExploratoryResearchTests(unittest.TestCase):
    def test_temporal_subperiod_is_historical_subset(self):
        frame = calculate_archived_robustness(ROOT_DIR)
        self.assertEqual(len(frame), 9)
        sample = frame.loc[(frame.period == 'retrospective_2023_2026')
                           & (frame.strategy == 'Core35_Satellite65')].iloc[0]
        self.assertGreater(sample.cagr_pct, 14.0)
        self.assertLess(sample.cagr_pct, 15.0)
        self.assertEqual(sample.first_date, '2023-01-03')

    def test_output_directory_cannot_overwrite_release(self):
        for name in ('reports/research', 'data/processed/research', 'src/research'):
            with self.assertRaises(ValueError):
                _research_output_directory(ROOT_DIR/name, ROOT_DIR)

    def test_regime_replay_preserves_official_core(self):
        frame = calculate_regime_adaptive_research(ROOT_DIR)
        selected = frame[(frame['sample'] == 'retrospective_2022_2026')
                         & (frame.strategy == 'original')].iloc[0]
        self.assertAlmostEqual(selected.cagr_pct, 5.872, places=2)
        self.assertAlmostEqual(selected.sharpe, .551, places=2)
        dynamic = frame[(frame['sample'] == 'retrospective_2022_2026')
                        & (frame.strategy == 'regime_adaptive')].iloc[0]
        self.assertGreater(dynamic.stress_decisions, 0)
        self.assertEqual(dynamic.decisions, 249)


if __name__ == '__main__':
    unittest.main()
