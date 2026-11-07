"""Regression tests for read-only release CLI dispatch."""

import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import patch

import main


class TestReleaseCLI(unittest.TestCase):

    @patch("src.release_pipeline.audit_release")
    @patch.object(main, "run_pipeline")
    def test_audit_does_not_run_pipeline(self, pipeline, audit):
        main.main(["--release-action", "audit"])

        audit.assert_called_once_with()
        pipeline.assert_not_called()

    @patch("src.release_pipeline.show_frozen_results")
    @patch.object(main, "run_pipeline")
    def test_results_do_not_run_pipeline(self, pipeline, results):
        main.main(["--release-action", "frozen-results"])

        results.assert_called_once_with()
        pipeline.assert_not_called()

    def test_reject_mixed_execution_modes(self):
        incompatible = [
            ["--level", "A1"],
            ["--level", "C"],
            ["--b-version", "v1"],
            ["--config", "config.yaml"],
        ]

        for options in incompatible:
            with self.subTest(options=options):
                with redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        main.main(
                            ["--release-action", "audit"] + options
                        )

                self.assertEqual(caught.exception.code, 2)

    @patch.object(main, "run_pipeline")
    def test_existing_all_dispatch_is_unchanged(self, pipeline):
        main.main(["--level", "ALL"])

        pipeline.assert_called_once_with(
            level="ALL",
            config_path=None,
            b_version=None,
        )


if __name__ == "__main__":
    unittest.main()
