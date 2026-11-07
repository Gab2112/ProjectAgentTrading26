"""Regression tests for explicit release and development dispatch.

These tests use mocks, allowing GitHub CI without the separately distributed
frozen model binaries and OOS report files.
"""
import unittest
from io import StringIO
from contextlib import redirect_stderr
from unittest.mock import patch

import main


class TestReleaseCLI(unittest.TestCase):
    @patch('src.submission_validation.verify_release_contract')
    @patch('src.release_pipeline.audit_release')
    @patch.object(main, 'run_pipeline')
    def test_audit_does_not_run_legacy_pipeline(self, legacy, audit, contract):
        main.main(['--release-action', 'audit'])
        audit.assert_called_once_with(main.ROOT_DIR)
        legacy.assert_not_called()

    @patch('src.submission_validation.verify_release_contract')
    @patch('src.release_pipeline.show_frozen_results')
    @patch.object(main, 'run_pipeline')
    def test_default_is_final_frozen_results(self, legacy, show, contract):
        main.main([])
        show.assert_called_once_with(main.ROOT_DIR)
        legacy.assert_not_called()

    def test_reject_mixed_release_execution_modes(self):
        for mixed in (['--level', 'C'], ['--level', 'A1'],
                      ['--b-version', 'v1'], ['--config', 'config.yaml']):
            with self.subTest(mixed=mixed):
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as err:
                    main.main(['--release-action', 'audit']+mixed)
                self.assertEqual(err.exception.code, 2)

    @patch('src.submission_validation.verify_release_contract')
    @patch('src.release_pipeline.audit_release')
    @patch('src.submission_runner.run_development_in_copy')
    @patch.object(main, '_run_core_development')
    def test_all_runs_isolated_development_then_core(self, core, runner, audit, contract):
        main.main(['--level', 'ALL'])
        runner.assert_called_once_with('ALL')
        core.assert_called_once_with()
        audit.assert_called_once_with(main.ROOT_DIR)


if __name__ == '__main__':
    unittest.main()
