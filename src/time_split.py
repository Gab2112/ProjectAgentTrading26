"""Temporal boundaries and decision calendar.

All dates are read from the ``validation`` section of ``config.yaml``:

- in-sample (IS): ``train_start`` .. ``train_end``. Every model is fitted and
  every hyperparameter is selected using observations in this window only;
- validation: ``validation_start`` .. ``validation_end``, the final
  sub-window of the IS period. It is used exclusively to select the
  portfolio-allocation parameters;
- out-of-sample (OOS): ``oos_start`` .. ``oos_end``. It is evaluated once,
  after every parameter has been fixed.

Observations preceding ``train_start`` may be used only as warm-up history
for rolling indicators; they never enter a performance statistic.

The module also defines the weekly decision calendar: the last actual trading
session of each calendar week.
"""

from __future__ import annotations

import pandas as pd


def _parse_boundary(section: dict, name: str, *, required: bool) -> pd.Timestamp | None:
    """Parse one date from the configuration and check its format."""
    value = section.get(name)
    if value is None:
        if required:
            raise ValueError(f"Missing mandatory time boundary: {name}")
        return None

    try:
        boundary = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid date for {name}: {value!r}") from exc

    if pd.isna(boundary):
        raise ValueError(f"The date {name} cannot be NaT.")
    if boundary.tzinfo is not None:
        raise ValueError(f"The date {name} must be timezone-naive.")
    if boundary != boundary.normalize():
        raise ValueError(f"The date {name} must be a calendar day without a time component.")
    return boundary


def get_time_windows(config: dict) -> dict[str, pd.Timestamp | None]:
    """Read and validate every time boundary (all boundaries are inclusive).

    Required ordering:
        train_start <= validation_start <= validation_end <= train_end < oos_start <= oos_end
    """
    section = config.get("validation", {})
    if not isinstance(section, dict):
        raise TypeError("The 'validation' section must be a dictionary.")

    windows = {
        "train_start": _parse_boundary(section, "train_start", required=True),
        "train_end": _parse_boundary(section, "train_end", required=True),
        "validation_start": _parse_boundary(section, "validation_start", required=True),
        "validation_end": _parse_boundary(section, "validation_end", required=True),
        "oos_start": _parse_boundary(section, "oos_start", required=True),
        "oos_end": _parse_boundary(section, "oos_end", required=False),
    }

    if not windows["train_start"] < windows["train_end"]:
        raise ValueError("train_start must precede train_end.")
    if not (
        windows["train_start"]
        <= windows["validation_start"]
        <= windows["validation_end"]
        <= windows["train_end"]
    ):
        raise ValueError("The validation window must lie inside the in-sample window.")
    if not windows["train_end"] < windows["oos_start"]:
        raise ValueError("train_end must precede oos_start.")
    if windows["oos_end"] is not None and not windows["oos_start"] <= windows["oos_end"]:
        raise ValueError("oos_start must precede oos_end.")

    return windows


def period_bounds(config: dict, period: str) -> tuple[pd.Timestamp, pd.Timestamp | None]:
    """Return the inclusive (start, end) bounds of ``period``.

    ``period`` is one of ``"in_sample"``, ``"validation"`` or ``"oos"``;
    an end of ``None`` means "up to the last available observation".
    """
    windows = get_time_windows(config)
    if period == "in_sample":
        return windows["train_start"], windows["train_end"]
    if period == "validation":
        return windows["validation_start"], windows["validation_end"]
    if period == "oos":
        return windows["oos_start"], windows["oos_end"]
    raise ValueError(f"Unknown period: {period!r}")


def slice_period(data: pd.DataFrame | pd.Series, config: dict, period: str):
    """Return the rows of ``data`` whose date belongs to ``period``."""
    start, end = period_bounds(config, period)
    mask = data.index >= start
    if end is not None:
        mask &= data.index <= end
    selected = data.loc[mask]
    if selected.empty:
        raise ValueError(f"No observations in the {period} period.")
    return selected


def build_weekly_decision_dates(market_daily: pd.DataFrame) -> pd.DataFrame:
    """Identify the LAST actual trading session of each calendar week.

    A week that contains a holiday may end on Thursday. Signals are computed
    after the close of that session and executed at the open of the next
    available trading session.
    """
    session_dates = pd.Series(market_daily.index, index=market_daily.index)
    dates = session_dates.resample("W-FRI").last().dropna()
    decisions = pd.DataFrame({"decision_date": pd.to_datetime(dates.values)})
    # Unlike the W-FRI labels, the index always refers to an actual session.
    decisions.index = pd.DatetimeIndex(decisions["decision_date"], name="Date")
    return decisions
