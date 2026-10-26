"""Test temporali della ricerca Multi-Asset, Fase A.

I test usano soltanto date e valori sintetici. Non leggono dati di
mercato reali e non consultano i risultati della validation o dell'OOS.

Invarianti controllati:
- training e validation hanno confini espliciti e non sovrapposti;
- le osservazioni di warm-up possono precedere il training;
- con development_only=true l'OOS e' bloccato anche se evaluate_oos=true;
- gli split non modificano il dataset di partenza;
- date ambigue, duplicate o non ordinate vengono rifiutate.

Esecuzione dalla radice del repository:
    python3 -m unittest discover -s tests -p 'test_*.py' -v
"""

import unittest
from copy import deepcopy

import pandas as pd

from src.time_split import (
    get_time_boundaries,
    get_time_windows,
    is_oos_enabled,
    restrict_to_allowed_period,
    split_time_periods,
)


def make_config() -> dict:
    """Definisce il calendario sintetico con OOS bloccato."""
    return {
        "validation": {
            "train_start": "2016-12-28",
            "train_end": "2016-12-31",
            "validation_start": "2017-01-01",
            "validation_end": "2021-12-31",
            "oos_start": "2022-01-01",
            "development_only": True,
        },
        "evaluation": {"evaluate_oos": False},
    }


def make_dataset() -> pd.DataFrame:
    """Costruisce periodi sintetici, incluso il riscaldamento pre-training."""
    dates = pd.to_datetime(
        [
            "2016-12-27",  # Warm-up: non e' una valutazione.
            "2016-12-28",  # Primo giorno di training.
            "2016-12-30",  # Training, prima del confine inclusivo.
            "2017-01-02",  # Validation.
            "2021-12-31",  # Ultimo giorno di validation.
            "2022-01-03",  # OOS sintetico: normalmente nascosto.
            "2022-01-04",  # OOS sintetico: normalmente nascosto.
        ]
    )
    return pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7]}, index=dates)


class TestTimeBoundaries(unittest.TestCase):
    """Controlli sui parametri cronologici e sulle date limite."""

    def setUp(self) -> None:
        self.config = make_config()

    def test_all_boundaries_are_parsed(self):
        windows = get_time_windows(self.config)
        expected = {
            "train_start": pd.Timestamp("2016-12-28"),
            "train_end": pd.Timestamp("2016-12-31"),
            "validation_start": pd.Timestamp("2017-01-01"),
            "validation_end": pd.Timestamp("2021-12-31"),
            "oos_start": pd.Timestamp("2022-01-01"),
        }
        self.assertEqual(windows, expected)
        self.assertEqual(
            get_time_boundaries(self.config),
            (expected["train_end"], expected["validation_end"], expected["oos_start"]),
        )

    def test_optional_old_config_boundaries(self):
        config = deepcopy(self.config)
        del config["validation"]["train_start"]
        del config["validation"]["validation_start"]
        windows = get_time_windows(config)
        self.assertIsNone(windows["train_start"])
        self.assertEqual(windows["validation_start"], pd.Timestamp("2017-01-01"))

    def test_required_dates_cannot_be_missing(self):
        for field in ("train_end", "validation_end", "oos_start"):
            with self.subTest(field=field):
                config = deepcopy(self.config)
                del config["validation"][field]
                with self.assertRaises(ValueError):
                    get_time_windows(config)

    def test_invalid_boundary_formats_are_rejected(self):
        invalid = ("invalid-date", "NaT", "2016-12-31T12:30", "2016-12-31T00:00:00Z")
        for value in invalid:
            with self.subTest(value=value):
                config = deepcopy(self.config)
                config["validation"]["train_end"] = value
                with self.assertRaises(ValueError):
                    get_time_windows(config)

    def test_overlapping_or_reversed_periods_are_rejected(self):
        cases = (
            ("train_start", "2017-01-01"),
            ("train_end", "2017-01-01"),
            ("validation_start", "2016-12-31"),
            ("validation_start", "2022-01-01"),
            ("validation_end", "2022-01-01"),
            ("oos_start", "2021-12-31"),
        )
        for key, value in cases:
            with self.subTest(key=key, value=value):
                config = deepcopy(self.config)
                config["validation"][key] = value
                with self.assertRaises(ValueError):
                    get_time_windows(config)

    def test_validation_section_must_be_a_mapping(self):
        config = deepcopy(self.config)
        config["validation"] = None
        with self.assertRaises(TypeError):
            get_time_windows(config)


class TestOOSAuthorization(unittest.TestCase):
    """L'OOS resta bloccato finche' la ricerca e' in sviluppo."""

    def setUp(self) -> None:
        self.config = make_config()

    def test_oos_off_by_default(self):
        self.assertFalse(is_oos_enabled(self.config))
        config = deepcopy(self.config)
        del config["evaluation"]["evaluate_oos"]
        self.assertFalse(is_oos_enabled(config))

    def test_a_single_flag_cannot_unlock_development_oos(self):
        config = deepcopy(self.config)
        config["evaluation"]["evaluate_oos"] = True
        with self.assertRaises(ValueError):
            is_oos_enabled(config)
        with self.assertRaises(ValueError):
            restrict_to_allowed_period(make_dataset(), config)

    def test_double_explicit_authorization_on_synthetic_data(self):
        config = deepcopy(self.config)
        config["evaluation"]["evaluate_oos"] = True
        config["validation"]["development_only"] = False
        self.assertTrue(is_oos_enabled(config))
        periods = split_time_periods(make_dataset(), config)
        self.assertEqual(set(periods), {"train", "validation", "oos"})
        self.assertEqual(periods["oos"]["value"].tolist(), [6, 7])

    def test_boolean_flags_must_be_real_booleans(self):
        for field, section in (
            ("evaluate_oos", "evaluation"),
            ("development_only", "validation"),
        ):
            for value in ("false", "true", 0, 1, None):
                with self.subTest(field=field, value=value):
                    config = deepcopy(self.config)
                    config[section][field] = value
                    with self.assertRaises(ValueError):
                        is_oos_enabled(config)

    def test_authentication_sections_must_be_mappings(self):
        for section in ("validation", "evaluation"):
            with self.subTest(section=section):
                config = deepcopy(self.config)
                config[section] = []
                with self.assertRaises(TypeError):
                    is_oos_enabled(config)


class TestTimePartitions(unittest.TestCase):
    """Isolamento delle osservazioni in ciascun intervallo cronologico."""

    def setUp(self) -> None:
        self.config = make_config()
        self.dataset = make_dataset()

    def test_restriction_keeps_warmup_and_validation_but_hides_oos(self):
        allowed = restrict_to_allowed_period(self.dataset, self.config)
        self.assertEqual(allowed["value"].tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(allowed.index.max(), pd.Timestamp("2021-12-31"))

    def test_split_excludes_warmup_and_oos(self):
        periods = split_time_periods(self.dataset, self.config)
        self.assertEqual(set(periods), {"train", "validation"})
        self.assertEqual(periods["train"]["value"].tolist(), [2, 3])
        self.assertEqual(periods["validation"]["value"].tolist(), [4, 5])
        self.assertTrue(periods["train"].index.max() < periods["validation"].index.min())

    def test_temporal_gap_is_not_assigned_to_another_period(self):
        config = deepcopy(self.config)
        config["validation"]["validation_start"] = "2017-02-01"
        periods = split_time_periods(self.dataset, config)
        self.assertEqual(periods["train"]["value"].tolist(), [2, 3])
        self.assertEqual(periods["validation"]["value"].tolist(), [5])

    def test_future_observations_cannot_change_training_slice(self):
        original = split_time_periods(self.dataset, self.config)["train"]
        changed = self.dataset.copy()
        changed.loc[changed.index >= pd.Timestamp("2017-01-01"), "value"] = 99999
        later = split_time_periods(changed, self.config)["train"]
        pd.testing.assert_frame_equal(original, later)

    def test_inputs_and_outputs_are_independent_copies(self):
        original = self.dataset.copy(deep=True)
        allowed = restrict_to_allowed_period(self.dataset, self.config)
        periods = split_time_periods(self.dataset, self.config)
        allowed.iloc[0, 0] = -50
        periods["train"].iloc[0, 0] = -100
        pd.testing.assert_frame_equal(self.dataset, original)

    def test_bad_indexes_are_rejected(self):
        cases = {
            "non_temporal": self.dataset.reset_index(drop=True),
            "reversed": self.dataset.iloc[::-1],
            "duplicated": pd.concat([self.dataset, self.dataset.iloc[[-1]]]).sort_index(),
            "date_missing": self.dataset.set_axis(
                self.dataset.index.where(self.dataset.index != self.dataset.index[0], pd.NaT)
            ),
            "timezone": self.dataset.tz_localize("UTC"),
        }
        for name, data in cases.items():
            with self.subTest(case=name):
                error = TypeError if name == "non_temporal" else ValueError
                with self.assertRaises(error):
                    restrict_to_allowed_period(data, self.config)

    def test_empty_allowed_period_fails(self):
        with self.assertRaises(ValueError):
            restrict_to_allowed_period(self.dataset.iloc[-1:], self.config)

    def test_empty_training_or_validation_fails(self):
        for subset in (self.dataset.loc["2017-01-01":"2021-12-31"],
                       self.dataset.loc["2016-12-28":"2016-12-31"]):
            with self.subTest(start=subset.index[0]):
                with self.assertRaises(ValueError):
                    split_time_periods(subset, self.config)

    def test_empty_oos_after_explicit_unlock_fails(self):
        config = deepcopy(self.config)
        config["validation"]["development_only"] = False
        config["evaluation"]["evaluate_oos"] = True
        with self.assertRaises(ValueError):
            split_time_periods(self.dataset.loc[:"2021-12-31"], config)


if __name__ == "__main__":
    unittest.main()
