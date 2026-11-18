"""Rule-based macro-regime baseline strategies.

Contents:
- regime classification from the VIX and the 10-year breakeven inflation rate,
  each compared with its trailing median (``classify_macro_regimes``);
- fixed risk budgets per regime (``REGIME_WEIGHTS``, ``build_regime_weight_panel``);
- 21/63-session momentum signals and a macro filter on those signals
  (``build_trend_signal_panel``, ``apply_macro_regime_filter``);
- conversion of signals and budgets into signed weights (``build_signed_weight_panel``).

The budgets in ``REGIME_WEIGHTS`` are fixed by economic reasoning and were not
estimated from returns. The volatility cap applied afterwards is implemented
in ``src/portfolio.py``; the macro features in ``src/features.py``.
"""

import numpy as np
import pandas as pd

from src.utils import ASSETS


REGIME_WEIGHTS = {
    "low_low":  (0.25, 0.30, 0.20, 0.25),
    "low_high": (0.25, 0.20, 0.30, 0.25),
    "high_low": (0.15, 0.30, 0.20, 0.35),
    "high_high": (0.15, 0.20, 0.30, 0.35),
}


def regime_to_weights(regime: str) -> pd.Series:
    """Return long-only target weights for a valid macro regime."""

    if not isinstance(regime, str) or regime not in REGIME_WEIGHTS:
        raise ValueError(f"Invalid or unavailable regime: {regime!r}")

    weights = pd.Series(
        REGIME_WEIGHTS[regime],
        index=ASSETS,
        dtype="float64",
    )

    if (weights < 0).any() or abs(weights.sum() - 1.0) > 1e-10:
        raise ValueError("Invalid portfolio weights")

    return weights

def build_regime_weight_panel(
    regimes: pd.Series,
) -> pd.DataFrame:
    """Convert weekly regimes into portfolio target weights."""

    if not isinstance(regimes.index, pd.DatetimeIndex):
        raise TypeError("Regimes require a DatetimeIndex")

    if (
        regimes.empty
        or regimes.index.has_duplicates
        or not regimes.index.is_monotonic_increasing
        or regimes.isna().any()
    ):
        raise ValueError("Invalid regime series")

    weights = pd.DataFrame(
        [
            regime_to_weights(str(regime)).to_numpy()
            for regime in regimes
        ],
        index=regimes.index,
        columns=ASSETS,
        dtype=float,
    )

    if not (weights.sum(axis=1) - 1.0).abs().lt(1e-10).all():
        raise ValueError("Portfolio weights do not sum to one")

    return weights


def build_signed_weight_panel(
    signals: pd.DataFrame,
    budgets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Convert discrete signals and risk budgets into signed weights."""


    signals = validate_signal_panel(signals)

    if not isinstance(budgets, pd.DataFrame):
        raise TypeError("Budgets must be a DataFrame")

    if (
        not budgets.index.equals(signals.index)
        or list(budgets.columns) != list(ASSETS)
    ):
        raise ValueError("Budget and signal panels must align")

    budgets = budgets.astype(float)

    if (
        budgets.isna().any().any()
        or not budgets.map(
            lambda x: 0 <= x <= 1
        ).all().all()
        or (budgets.sum(axis=1) - 1).abs().max() > 1e-10
    ):
        raise ValueError("Invalid risk budgets")

    # -1 = short, 0 = neutral, +1 = long.
    signed_weights = budgets.mul(signals)

    # Gross exposure never exceeds 100% of equity.
    gross_exposure = signed_weights.abs().sum(axis=1)

    if (gross_exposure > 1 + 1e-10).any():
        raise ValueError("Gross exposure exceeds 100%")

    unused_budget = 1.0 - gross_exposure

    return signed_weights, unused_budget


# Weekly signal generation and validation.

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
    """Suppress SPY longs under high stress and IEF longs under high breakeven inflation."""

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


def classify_macro_regimes(
    features: pd.DataFrame,
    min_history: int = 52,
    window_weeks: int | None = None,
) -> pd.DataFrame:
    """Classify each week by VIX and breakeven inflation relative to their trailing medians.

    The thresholds use only past observations (shift by one week). FRED
    series are not point-in-time certified.
    """

    required = {"vix_level", "breakeven_10y"}
    if not required.issubset(features.columns):
        raise ValueError("Missing macro features")

    if min_history < 1:
        raise ValueError("Invalid minimum history")

    if window_weeks is not None and window_weeks < min_history:
        raise ValueError("Rolling window shorter than minimum history")

    result = pd.DataFrame(index=features.index)

    def past_median(column):
        past = features[column].shift(1)

        if window_weeks is None:
            return past.expanding(
                min_periods=min_history
            ).median()

        return past.rolling(
            window=window_weeks,
            min_periods=min_history
        ).median()

    result["vix_threshold"] = past_median("vix_level")

    result["breakeven_threshold"] = past_median(
        "breakeven_10y"
    )
    valid = result.notna().all(axis=1)

    result["stress_state"] = pd.Series(
        pd.NA, index=features.index, dtype="string"
    )
    result["inflation_comp_state"] = pd.Series(
        pd.NA, index=features.index, dtype="string"
    )

    result.loc[valid, "stress_state"] = (
        features.loc[valid, "vix_level"]
        .gt(result.loc[valid, "vix_threshold"])
        .map({True: "high", False: "low"})
    )

    result.loc[valid, "inflation_comp_state"] = (
        features.loc[valid, "breakeven_10y"]
        .gt(result.loc[valid, "breakeven_threshold"])
        .map({True: "high", False: "low"})
    )

    result["regime"] = result["stress_state"].str.cat(
        result["inflation_comp_state"], sep="_"
    )

    return result
