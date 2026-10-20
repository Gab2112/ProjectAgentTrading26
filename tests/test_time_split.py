
"""
Test degli invarianti temporali della pipeline.

Utilizza esclusivamente dati sintetici.
Non accede ai dataset reali né ai rendimenti OOS.
"""

import unittest
from copy import deepcopy

import pandas as pd

from src.time_split import (
    get_time_boundaries,
    is_oos_enabled,
    restrict_to_allowed_period,
    split_time_periods,
)


class TestTimeSplit(unittest.TestCase):

    def setUp(self):
        self.config = {
            "validation": {
                "train_end": "2016-12-31",
                "validation_end": "2021-12-31",
                "oos_start": "2022-01-01",
            },
            "evaluation": {
                "evaluate_oos": False,
            },
        }

        # Date sintetiche che attraversano i tre periodi.
        dates = pd.to_datetime([
            "2016-12-30",
            "2017-01-06",
            "2021-12-31",
            "2022-01-07",
        ])

        self.df = pd.DataFrame(
            {"value": [10, 20, 30, 40]},
            index=dates,
        )

    def test_valid_boundaries(self):
        boundaries = get_time_boundaries(self.config)

        self.assertEqual(
            boundaries,
            (
                pd.Timestamp("2016-12-31"),
                pd.Timestamp("2021-12-31"),
                pd.Timestamp("2022-01-01"),
            ),
        )

    def test_missing_boundary(self):
        config = deepcopy(self.config)
        del config["validation"]["oos_start"]

        with self.assertRaises(ValueError):
            get_time_boundaries(config)

    def test_invalid_boundary_order(self):
        config = deepcopy(self.config)
        config["validation"]["train_end"] = "2022-01-01"

        with self.assertRaises(ValueError):
            get_time_boundaries(config)

    def test_invalid_boundary_date(self):
        config = deepcopy(self.config)
        config["validation"]["train_end"] = "NaT"

        with self.assertRaises(ValueError):
            get_time_boundaries(config)

    def test_oos_flag_must_be_boolean(self):
        for invalid_value in ["false", 0, 1, None]:
            with self.subTest(value=invalid_value):
                config = deepcopy(self.config)
                config["evaluation"]["evaluate_oos"] = (
                    invalid_value
                )

                with self.assertRaises(ValueError):
                    is_oos_enabled(config)

    def test_oos_disabled_by_default(self):
        self.assertFalse(is_oos_enabled(self.config))

        config = deepcopy(self.config)
        config["evaluation"]["evaluate_oos"] = True

        self.assertTrue(is_oos_enabled(config))

    def test_restrict_excludes_oos(self):
        result = restrict_to_allowed_period(
            self.df,
            self.config,
        )

        self.assertEqual(len(result), 3)
        self.assertEqual(
            result.index.max(),
            pd.Timestamp("2021-12-31"),
        )

    def test_split_train_validation(self):
        periods = split_time_periods(
            self.df,
            self.config,
        )

        self.assertEqual(
            set(periods.keys()),
            {"train", "validation"},
        )

        self.assertEqual(len(periods["train"]), 1)
        self.assertEqual(len(periods["validation"]), 2)

        self.assertTrue(
            (periods["train"].index
             <= pd.Timestamp("2016-12-31")).all()
        )

        self.assertTrue(
            (periods["validation"].index
             <= pd.Timestamp("2021-12-31")).all()
        )

    def test_oos_requires_explicit_authorization(self):
        config = deepcopy(self.config)
        config["evaluation"]["evaluate_oos"] = True

        periods = split_time_periods(
            self.df,
            config,
        )

        self.assertIn("oos", periods)
        self.assertEqual(len(periods["oos"]), 1)

        self.assertEqual(
            periods["oos"].index.min(),
            pd.Timestamp("2022-01-07"),
        )

    def test_non_datetime_index_rejected(self):
        invalid_df = self.df.reset_index(drop=True)

        with self.assertRaises(TypeError):
            restrict_to_allowed_period(
                invalid_df,
                self.config,
            )

    def test_unsorted_index_rejected(self):
        invalid_df = self.df.iloc[[1, 0, 2, 3]]

        with self.assertRaises(ValueError):
            restrict_to_allowed_period(
                invalid_df,
                self.config,
            )

    def test_empty_allowed_period_rejected(self):
        oos_only = self.df.iloc[[3]]

        with self.assertRaises(ValueError):
            restrict_to_allowed_period(
                oos_only,
                self.config,
            )

    def test_empty_validation_rejected(self):
        no_validation = self.df.iloc[[0, 3]]

        with self.assertRaises(ValueError):
            split_time_periods(
                no_validation,
                self.config,
            )

    def test_empty_oos_rejected_when_enabled(self):
        config = deepcopy(self.config)
        config["evaluation"]["evaluate_oos"] = True

        with self.assertRaises(ValueError):
            split_time_periods(
                self.df.iloc[:3],
                config,
            )

    def test_original_dataframe_unchanged(self):
        original = self.df.copy(deep=True)

        restricted = restrict_to_allowed_period(
            self.df,
            self.config,
        )

        periods = split_time_periods(
            self.df,
            self.config,
        )

        pd.testing.assert_frame_equal(
            self.df,
            original,
        )

        restricted.iloc[0, 0] = -999
        periods["train"].iloc[0, 0] = -999

        pd.testing.assert_frame_equal(
            self.df,
            original,
        )


if __name__ == "__main__":
    unittest.main()
