"""Temporal alignment of verified releases and conservative date-based proxies."""

from collections.abc import Mapping

import pandas as pd


def align_verified_daily_series(
    decisions: pd.DatetimeIndex,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Match each decision with the latest previously available value."""

    if not isinstance(decisions, pd.DatetimeIndex) or decisions.tz is None:
        raise ValueError("Decisions must have timezone-aware timestamps")

    if decisions.hasnans or decisions.has_duplicates or not decisions.is_monotonic_increasing:
        raise ValueError("Invalid decision timestamps")

    required = {"observation_date", "available_at", "value"}
    if not required.issubset(events.columns):
        raise ValueError("Missing publication metadata")

    if events.empty or events[list(required)].isna().any().any():
        raise ValueError("Missing observations or metadata")

    if any(pd.Timestamp(t).tzinfo is None for t in events["available_at"]):
        raise ValueError("Publication timestamps must include timezone")

    releases = events[
        ["observation_date", "available_at", "value"]
    ].copy()

    releases["observation_date"] = pd.to_datetime(
        releases["observation_date"], errors="raise"
    )
    releases["available_at"] = pd.to_datetime(
        releases["available_at"], utc=True, errors="raise"
    )
    releases["value"] = pd.to_numeric(
        releases["value"], errors="raise"
    )

    if releases["observation_date"].duplicated().any():
        raise ValueError("Revisions require vintage-aware processing")

    if releases["available_at"].duplicated().any():
        raise ValueError("Duplicate publication timestamps")

    left = pd.DataFrame({
        "decision_at": decisions.tz_convert("UTC")
    }).sort_values("decision_at")

    right = releases.sort_values("available_at")

    result = pd.merge_asof(
        left,
        right,
        left_on="decision_at",
        right_on="available_at",
        direction="backward",
    )

    return result.set_index("decision_at")

def align_daily_observation_proxy(
    decision_dates: pd.DatetimeIndex,
    series: pd.Series,
) -> pd.DataFrame:
    """Conservative date-based alignment. NOT point-in-time verified."""

    if not isinstance(decision_dates, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Series must have a DatetimeIndex")

    for index in (decision_dates, series.index):
        if (
            index.tz is not None
            or index.hasnans
            or index.has_duplicates
            or not index.is_monotonic_increasing
        ):
            raise ValueError("Invalid date index")

    observations = (
        series.dropna()
        .rename("value")
        .rename_axis("observation_date")
        .reset_index()
    )

    decisions = pd.DataFrame({"decision_date": decision_dates})

    aligned = pd.merge_asof(
        decisions,
        observations,
        left_on="decision_date",
        right_on="observation_date",
        direction="backward",
        allow_exact_matches=False,
        tolerance=pd.Timedelta(days=7),
    )

    if aligned["value"].isna().any():
        raise ValueError("Missing or stale macro observations")

    return aligned.set_index("decision_date")

def align_daily_macro_panel(
    decision_dates: pd.DatetimeIndex,
    daily_observations: Mapping[str, pd.Series],
) -> pd.DataFrame:
    """Align several *daily* FRED proxies on actual session decision dates.

    Every observation is strictly earlier than its decision date and at most
    seven calendar days old. Each value retains its observation date for audit.
    This does NOT establish publication timestamps or historical vintages.
    Monthly/low-frequency releases should use verified event availability,
    not this seven-day observation-date proxy.
    """
    if not isinstance(daily_observations, Mapping) or not daily_observations:
        raise ValueError("Expected a nonempty mapping of daily macro series")

    panel = pd.DataFrame(index=decision_dates)
    panel.index.name = "Date"
    for name, series in daily_observations.items():
        if (not isinstance(name, str) or not name.isidentifier()
                or name.endswith("_observation_date") or name in panel.columns):
            raise ValueError(f"Invalid or duplicate macro name: {name!r}")
        if not isinstance(series, pd.Series):
            raise TypeError(f"Daily macro input {name!r} must be a Series")
        aligned = align_daily_observation_proxy(decision_dates, series)
        panel[name] = pd.to_numeric(aligned["value"], errors="raise").to_numpy()
        panel[f"{name}_observation_date"] = aligned["observation_date"].to_numpy()
    return panel


# Macro feature construction on the available historical observation proxies.


