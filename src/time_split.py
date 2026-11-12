"""Centralised management of the project time periods.

Dates and permissions are read exclusively from ``config.yaml``. Existing
functions keep their interfaces so that other modules can migrate gradually.

The module also provides the decision calendar
(``build_weekly_decision_dates``): the last actual trading session of each week.

Core rules:
- training starts at ``train_start`` when that date is defined;
- validation starts at ``validation_start`` when that date is defined;
- data preceding the training window may remain available to warm up
  indicators (for example 126 sessions of covariance estimation);
- the OOS period is hidden during development and requires two explicit
  authorisations before it becomes accessible.

This module separates the periods, but it does NOT by itself prevent an
improper use of the validation set during parameter selection.
"""

import pandas as pd


def _parse_boundary(
    validation: dict,
    name: str,
    *,
    required: bool,
) -> pd.Timestamp | None:
    """Parse a date from the YAML file and validate its format."""
    value = validation.get(name)

    if value is None:
        if required:
            raise ValueError(f"Confine temporale obbligatorio mancante: {name}")
        return None

    try:
        boundary = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Data non valida per {name}: {value!r}") from exc

    if pd.isna(boundary):
        raise ValueError(f"La data {name} non puo' essere NaT.")
    if boundary.tzinfo is not None:
        raise ValueError(f"La data {name} deve essere priva di fuso orario.")
    if boundary != boundary.normalize():
        raise ValueError(f"La data {name} deve indicare un giorno, senza orario.")

    return boundary


def get_time_windows(config: dict) -> dict[str, pd.Timestamp | None]:
    """Read and validate all time boundaries, including the start dates.

    For compatibility with the previous configuration:
    - ``train_start`` is optional: if absent, training covers every
      observation available up to ``train_end``;
    - ``validation_start`` is optional: if absent, it is the first
      calendar day after ``train_end``.

    Boundaries are inclusive. Days are not assumed to be trading sessions:
    filters are applied to the actual dates present in the dataset.
    """
    validation = config.get("validation", {})
    if not isinstance(validation, dict):
        raise TypeError("La sezione 'validation' deve essere un dizionario.")

    train_start = _parse_boundary(validation, "train_start", required=False)
    train_end = _parse_boundary(validation, "train_end", required=True)
    validation_start = _parse_boundary(
        validation, "validation_start", required=False
    )
    validation_end = _parse_boundary(validation, "validation_end", required=False)
    oos_start = _parse_boundary(validation, "oos_start", required=True)

    if validation_start is not None and validation_end is None:
        raise ValueError("validation_start requires validation_end")
    if validation_end is not None and validation_start is None:
        validation_start = train_end + pd.Timedelta(days=1)

    if train_start is not None and train_start > train_end:
        raise ValueError("'train_start' deve precedere o coincidere con 'train_end'.")
    if not train_end < oos_start:
        raise ValueError("train_end must precede oos_start")
    if validation_end is not None and not (train_end < validation_start <= validation_end < oos_start):
        raise ValueError(
            "Ordine delle date non valido: e' necessario "
            "train_end < validation_start <= validation_end < oos_start."
        )

    return {
        "train_start": train_start,
        "train_end": train_end,
        "validation_start": validation_start,
        "validation_end": validation_end,
        "oos_start": oos_start,
    }


def get_time_boundaries(
    config: dict,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return the three historical boundaries, preserving the existing interface."""
    windows = get_time_windows(config)
    return (
        windows["train_end"],
        windows["validation_end"] if windows["validation_end"] is not None else windows["train_end"],
        windows["oos_start"],
    )


def is_oos_enabled(config: dict) -> bool:
    """Authorise the OOS period only once development is complete.

    Access requires ``evaluation.evaluate_oos: true`` and, when present,
    ``validation.development_only: false``. A contradictory configuration
    raises an error instead of unlocking the final test.
    """
    evaluation = config.get("evaluation", {})
    validation = config.get("validation", {})

    if not isinstance(evaluation, dict) or not isinstance(validation, dict):
        raise TypeError("Le sezioni 'evaluation' e 'validation' devono essere dizionari.")

    enabled = evaluation.get("evaluate_oos", False)
    development_only = validation.get("development_only", False)

    if not isinstance(enabled, bool):
        raise ValueError("'evaluate_oos' deve essere un valore booleano.")
    if not isinstance(development_only, bool):
        raise ValueError("'development_only' deve essere un valore booleano.")
    if enabled and development_only:
        raise ValueError(
            "Accesso OOS negato: 'development_only' e' ancora true. "
            "Il test finale va sbloccato solo dopo aver fissato il modello."
        )

    return enabled


def _validate_time_index(df: pd.DataFrame) -> None:
    """Check that the dates can be used without ambiguity."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Il dataset deve utilizzare un DatetimeIndex.")
    if df.index.hasnans:
        raise ValueError("L'indice temporale contiene date mancanti (NaT).")
    if df.index.tz is not None:
        raise ValueError("L'indice temporale deve essere privo di fuso orario.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Le date del dataset devono essere in ordine crescente.")
    if df.index.has_duplicates:
        raise ValueError("Il dataset contiene date duplicate.")


def restrict_to_allowed_period(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Exclude OOS data until it is explicitly authorised.

    During development, observations up to ``validation_end`` are kept.
    Dates before ``train_start`` remain available to compute indicators
    with a look-back window, without entering the training metrics
    computed by ``split_time_periods``.

    This function does not perform the train/validation split.
    """
    windows = get_time_windows(config)
    _validate_time_index(df)

    if is_oos_enabled(config):
        allowed = df.copy()
    else:
        cutoff = windows["validation_end"] if windows["validation_end"] is not None else windows["train_end"]
        allowed = df.loc[df.index <= cutoff].copy()

    if allowed.empty:
        raise ValueError("Nessuna osservazione nel periodo autorizzato.")

    return allowed


def split_time_periods(
    df: pd.DataFrame,
    config: dict,
) -> dict[str, pd.DataFrame]:
    """Split the dataset into training, validation and, optionally, OOS.

    The periods follow the start and end dates defined in the YAML file.
    No observation can belong to two periods. Dates falling in an unassigned
    interval are left out of every split. The OOS set is returned only when
    explicitly authorised.
    """
    windows = get_time_windows(config)
    data = restrict_to_allowed_period(df, config)

    train_mask = data.index <= windows["train_end"]
    if windows["train_start"] is not None:
        train_mask &= data.index >= windows["train_start"]

    periods = {"train": data.loc[train_mask].copy()}
    if windows["validation_end"] is not None:
        validation_mask = (
            (data.index >= windows["validation_start"])
            & (data.index <= windows["validation_end"])
        )
        periods["validation"] = data.loc[validation_mask].copy()

    if is_oos_enabled(config):
        periods["oos"] = data.loc[
            data.index >= windows["oos_start"]
        ].copy()

    for name, period in periods.items():
        if period.empty:
            raise ValueError(f"Periodo temporale senza osservazioni: {name}")

    return periods


def build_weekly_decision_dates(market_daily: pd.DataFrame) -> pd.DataFrame:
    """Identify the LAST actual trading session of each week.

    A week with a market holiday may end on Thursday. Signals are computed
    after that session closes and can be executed at the open of the next
    available session.
    """
    session_dates = pd.Series(market_daily.index, index=market_daily.index)
    dates = session_dates.resample("W-FRI").last().dropna()
    decisions = pd.DataFrame({"decision_date": pd.to_datetime(dates.values)})
    decisions.index = pd.DatetimeIndex(decisions["decision_date"], name="Date")
    # Unlike W-FRI labels, the index always refers to an actual trading session.
    return decisions
