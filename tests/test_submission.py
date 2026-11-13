"""Focused interface, integrity and frozen-parameter regression checks.

The full 275-test development checkpoint (five skips) belongs to the
collaboration Git history; this lean archive validates its own submission contract.
"""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import shutil
import tempfile
import unittest

from src.orchestrator_loader import load_frozen_orchestrator, AGENTS
from src.release_pipeline import audit_release, verify_package_manifest
from src.submission_validation import verify_release_contract, verify_signal_calendar
from src.utils import ASSETS, ROOT_DIR


class SubmissionTests(unittest.TestCase):
    def test_contract(self):
        spec = verify_release_contract(ROOT_DIR)
        self.assertEqual(spec['final_portfolio']['satellite_weight'], 0.65)

    def test_archive_integrity(self):
        with redirect_stdout(io.StringIO()):
            audit_release(ROOT_DIR)

    def test_signal_decision_calendar(self):
        self.assertEqual(verify_signal_calendar(ROOT_DIR, oos=True), 249)
        self.assertEqual(verify_signal_calendar(ROOT_DIR, oos=False), 626)

    def test_orchestrator_loader_keeps_two_thresholds(self):
        for asset in ASSETS:
            model = load_frozen_orchestrator(asset, ROOT_DIR)
            payload = json.loads((ROOT_DIR / 'src/agents/config' /
                                  f'orchestrator_params_{asset.lower()}.json').read_text())
            p = payload['best_params']
            self.assertEqual(set(model.weights), set(AGENTS))
            self.assertEqual(model.weights, p['weights'])
            self.assertEqual(model.alphas, p['alphas'])
            self.assertEqual(model.theta_long, p['threshold_long'])
            self.assertEqual(model.theta_short, p['threshold_short'])

    def test_missing_weekly_signal_is_detected(self):
        # Isolated, minimal dataset copy to exercise the boundary check.
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT_DIR/'data/processed', root/'data/processed')
            shutil.copytree(ROOT_DIR/'data/signals', root/'data/signals')
            shutil.copy2(ROOT_DIR/'config.yaml', root/'config.yaml')
            path = root/'data/signals/signals_orchestrator_spy.csv'
            frame = pd.read_csv(path)
            frame = frame.drop(index=30)
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'missing 1 weekly decisions'):
                verify_signal_calendar(root, oos=False)

    def test_processed_data_mutation_breaks_manifest(self):
        if not (ROOT_DIR/'PACKAGE_MANIFEST.json').exists():
            self.skipTest('Manifest generated during ZIP assembly')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'project'
            shutil.copytree(ROOT_DIR, root,
                            ignore=shutil.ignore_patterns('__pycache__', 'outputs', '.venv'))
            copy = root/'data/processed/market_daily.csv'
            copy.write_bytes(copy.read_bytes()+b'\n')
            with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
                verify_package_manifest(root)



if __name__ == '__main__':
    unittest.main()
