"""Discrete trading signals for Level B."""

import pandas as pd
import numpy as np

from src.regime_allocator import ASSETS


def validate_signal_panel(
    signals: pd.DataFrame,
) -> pd.DataFrame:
    """Validate discrete -1/0/+1 weekly trading signals."""

    if not isinstance(signals, pd.DataFrame):
        raise TypeError("Signals must be a DataFrame")

    if list(signals.columns) != list(ASSETS):
        raise ValueError("Invalid asset universe or column order")

    index = signals.index

    if (
        not isinstance(index, pd.DatetimeIndex)
        or signals.empty
        or index.hasnans
        or index.has_duplicates
        or index.tz is not None
        or not index.is_monotonic_increasing
    ):
        raise ValueError("Invalid decision dates")

    if signals.isna().any().any():
        raise ValueError("Missing trading signals")

    if not signals.isin([-1, 0, 1]).all().all():
        raise ValueError("Signals must be -1, 0, or +1")

    return signals.astype("int8")

def build_trend_signal_panel(
    daily_close: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Generate weekly trend signals from past adjusted closing prices."""

    if list(daily_close.columns) != list(ASSETS):
        raise ValueError("Invalid ETF columns")

    index = daily_close.index

    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.hasnans
        or index.has_duplicates
        or index.tz is not None
        or not index.is_monotonic_increasing
    ):
        raise ValueError("Invalid daily price dates")

    if not isinstance(decision_dates, pd.DatetimeIndex):
        raise TypeError("Decision dates must be a DatetimeIndex")

    prices = daily_close.astype(float)

    if (
        not np.isfinite(prices.to_numpy()).all()
        or (prices <= 0).any().any()
    ):
        raise ValueError("Invalid closing prices")

    positions = index.get_indexer(decision_dates)

    if (positions < 63).any():
        raise ValueError("Missing decisions or insufficient price history")

    momentum_21 = prices.div(prices.shift(21)).sub(1)
    momentum_63 = prices.div(prices.shift(63)).sub(1)

    short = momentum_21.loc[decision_dates]
    long = momentum_63.loc[decision_dates]

    positive = (short > 0) & (long > 0)
    negative = (short < 0) & (long < 0)

    signals = (
        positive.astype("int8")
        - negative.astype("int8")
    )

    return validate_signal_panel(signals)


def apply_macro_regime_filter(
    signals: pd.DataFrame,
    regimes: pd.Series,
) -> pd.DataFrame:
    """Apply experimental macro filters to trading signals."""

    signals = validate_signal_panel(signals)

    if not isinstance(regimes, pd.Series):
        raise TypeError("Regimes must be a Series")

    if not regimes.index.equals(signals.index):
        raise ValueError("Signal and regime dates must match")

    allowed = {
        "low_low", "low_high",
        "high_low", "high_high",
    }

    if regimes.isna().any() or not regimes.isin(allowed).all():
        raise ValueError("Missing or invalid macro regimes")

    filtered = signals.copy()

    # High relative VIX: suppress equity long positions.
    high_stress = regimes.isin(["high_low", "high_high"])

    filtered.loc[
        high_stress & filtered["SPY"].eq(1),
        "SPY",
    ] = 0

    # High relative inflation compensation:
    # suppress intermediate-duration Treasury longs.
    high_breakeven = regimes.isin(["low_high", "high_high"])

    filtered.loc[
        high_breakeven & filtered["IEF"].eq(1),
        "IEF",
    ] = 0

    return validate_signal_panel(filtered)