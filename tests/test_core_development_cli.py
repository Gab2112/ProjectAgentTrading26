"""Tests for the frozen development CLI; no real backtests."""

import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import patch

import pandas as pd

import main


class TestCoreDevelopmentCLI(unittest.TestCase):

    @staticmethod
    def freeze():
        return {
            "status": "FROZEN_BEFORE_FINAL_OOS",
            "selected_strategy": "Core35_Satellite65",
            "training_end": "2021-12-31",
            "code_and_model_sha256": {"config.yaml": "synthetic-hash"},
        }

    @staticmethod
    def fake_result():
        names = [
            "Core35_Satellite65",
            "SPY_buy_hold",
            "Sixty_Forty",
        ]

        summary = pd.DataFrame({
            "cagr": [0.12, 0.14, 0.10],
            "annualized_volatility": [0.13, 0.17, 0.09],
            "sharpe_ratio": [0.9, 0.8, 1.1],
            "max_drawdown": [-0.18, -0.33, -0.19],
            "trades": [205, 1, 625],
        }, index=names)

        return {"summary": summary}

    def test_valid_development_dispatch(self):
        config = {"synthetic": True}

        with (
            patch(
                "src.release_pipeline.read_json",
                return_value=self.freeze(),
            ) as reader,
            patch(
                "src.release_pipeline.verify_hashes",
            ) as verify,
            patch(
                "src.core_satellite.run_core_satellite_training",
                return_value=self.fake_result(),
            ) as runner,
            patch.object(main, "load_config", return_value=config),
            patch.object(main, "validate_config") as validate,
            patch.object(main, "run_pipeline") as old_pipeline,
            redirect_stdout(StringIO()) as output,
        ):
            main.main(["--core-development"])

        reader.assert_called_once_with(
            main.ROOT_DIR / "reports/development_freeze_2021.json"
        )
        verify.assert_called_once_with(
            {"config.yaml": "synthetic-hash"},
            main.ROOT_DIR,
            "Frozen development",
        )
        validate.assert_called_once_with(config)
        runner.assert_called_once_with(config, main.ROOT_DIR)
        old_pipeline.assert_not_called()

        self.assertIn("Core35_Satellite65", output.getvalue())
        self.assertIn("DEVELOPMENT ONLY", output.getvalue())

    def test_modified_freeze_is_rejected(self):
        invalid = self.freeze()
        invalid["training_end"] = "2016-12-31"

        with (
            patch(
                "src.release_pipeline.read_json",
                return_value=invalid,
            ),
            patch(
                "src.release_pipeline.verify_hashes",
            ) as verify,
            patch(
                "src.core_satellite.run_core_satellite_training",
            ) as runner,
        ):
            with self.assertRaises(RuntimeError):
                main.main(["--core-development"])

        verify.assert_not_called()
        runner.assert_not_called()

    def test_hash_failure_blocks_execution(self):
        with (
            patch(
                "src.release_pipeline.read_json",
                return_value=self.freeze(),
            ),
            patch(
                "src.release_pipeline.verify_hashes",
                side_effect=RuntimeError("Integrity failure"),
            ),
            patch(
                "src.core_satellite.run_core_satellite_training",
            ) as runner,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "Integrity failure"
            ):
                main.main(["--core-development"])

        runner.assert_not_called()

    def test_incompatible_modes_are_rejected(self):
        combinations = [
            ["--level", "A1"],
            ["--level", "C"],
            ["--b-version", "v1"],
            ["--config", "config.yaml"],
            ["--release-action", "audit"],
        ]

        for arguments in combinations:
            with self.subTest(arguments=arguments):
                with redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        main.main(
                            ["--core-development"] + arguments
                        )

                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
