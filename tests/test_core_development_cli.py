"""Smoke tests for the new explicit Core-Satellite development entry points."""
import unittest
from io import StringIO
from contextlib import redirect_stderr
from unittest.mock import patch

import main


class TestCoreDevelopmentCLI(unittest.TestCase):
    @patch('src.submission_validation.verify_release_contract')
    @patch.object(main, '_run_core_development')
    def test_core_alias_dispatch(self, core, contract):
        main.main(['--core-development'])
        core.assert_called_once_with()

    @patch('src.submission_validation.verify_release_contract')
    @patch.object(main, '_run_core_development')
    def test_explicit_core_level_dispatch(self, core, contract):
        main.main(['--level', 'CORE'])
        core.assert_called_once_with()

    def test_incompatible_flags_rejected(self):
        for arguments in (['--level', 'A1'], ['--release-action', 'audit'],
                          ['--config', 'config.yaml'], ['--b-version', 'v2']):
            with self.subTest(arguments=arguments):
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as err:
                    main.main(['--core-development']+arguments)
                self.assertEqual(err.exception.code, 2)

    @patch('src.submission_validation.verify_release_contract')
    @patch('src.release_pipeline.audit_release')
    @patch('src.submission_runner.run_development_in_copy')
    @patch.object(main, 'run_pipeline')
    def test_c_development_is_not_core_or_legacy_dispatch(self, legacy, runner,
                                                           audit, contract):
        main.main(['--level', 'C'])
        runner.assert_called_once_with('C', b_version='v2', config_path=None)
        audit.assert_called_once_with(main.ROOT_DIR)
        legacy.assert_not_called()


if __name__ == '__main__':
    unittest.main()
