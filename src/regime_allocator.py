"""Experimental regime-based allocation for Level B."""

from collections.abc import Mapping

import pandas as pd
import numpy as np


ASSETS = ("SPY", "IEF", "GLD", "SHY")

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

def apply_signed_volatility_cap(
    weights: pd.Series,
    daily_returns: pd.DataFrame,
    decision_date: pd.Timestamp,
    window: int = 126,
    target_volatility: float = 0.08,
) -> tuple[pd.Series, dict]:
    """Cap estimated annualized volatility without changing signal directions."""

    if list(weights.index) != list(ASSETS):
        raise ValueError("Invalid weight universe")

    if list(daily_returns.columns) != list(ASSETS):
        raise ValueError("Invalid return universe")

    if (
        not isinstance(daily_returns.index, pd.DatetimeIndex)
        or daily_returns.index.has_duplicates
        or not daily_returns.index.is_monotonic_increasing
        or daily_returns.index.tz is not None
    ):
        raise ValueError("Invalid return dates")

    decision_date = pd.Timestamp(decision_date)

    if (
        decision_date.tz is not None
        or decision_date not in daily_returns.index
    ):
        raise ValueError("Invalid decision date")

    if window < 2 or not np.isfinite(target_volatility) or target_volatility <= 0:
        raise ValueError("Invalid risk parameters")

    w = weights.to_numpy(dtype=float)

    if (
        not np.isfinite(w).all()
        or np.abs(w).sum() > 1.0 + 1e-10
    ):
        raise ValueError("Invalid signed weights")

    # Only historical observations available at the decision close.
    history = daily_returns.loc[:decision_date].tail(window)

    if len(history) != window:
        raise ValueError("Insufficient return history")

    history = history.astype(float)

    if not np.isfinite(history.to_numpy()).all():
        raise ValueError("Invalid historical returns")

    covariance = history.cov().to_numpy(dtype=float) * 252

    variance = float(w @ covariance @ w)
    volatility_before = float(np.sqrt(max(variance, 0.0)))

    factor = (
        min(1.0, target_volatility / volatility_before)
        if volatility_before > 0
        else 1.0
    )

    adjusted = pd.Series(
        w * factor,
        index=ASSETS,
        dtype=float,
    )

    diagnostics = {
        "vol_before": volatility_before,
        "vol_after": volatility_before * factor,
        "scaling_factor": factor,
    }

    return adjusted, diagnostics

def apply_volatility_cap_panel(
    raw_weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
    *,
    window: int = 126,
    target_volatility: float = 0.08,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the existing point-in-time risk cap to every decision.

    This is an orchestration helper: the numerical algorithm and the three
    legacy risk-diagnostic columns stay unchanged for frozen B_v1/B_v2 CSVs.
    """
    if not isinstance(raw_weights, pd.DataFrame) or raw_weights.empty:
        raise ValueError("Expected nonempty target-weight panel")
    if list(raw_weights.columns) != list(ASSETS):
        raise ValueError("Invalid target-weight asset order")
    if not isinstance(raw_weights.index, pd.DatetimeIndex):
        raise ValueError("Target weights require dated decisions")
    if (
        raw_weights.index.hasnans
        or raw_weights.index.has_duplicates
        or not raw_weights.index.is_monotonic_increasing
        or raw_weights.index.tz is not None
    ):
        raise ValueError("Invalid target-weight decision dates")

    adjusted_rows: list[np.ndarray] = []
    risk_rows: list[dict] = []
    for date in raw_weights.index:
        adjusted, diagnostics = apply_signed_volatility_cap(
            weights=raw_weights.loc[date],
            daily_returns=daily_returns,
            decision_date=date,
            window=window,
            target_volatility=target_volatility,
        )
        adjusted_rows.append(adjusted.to_numpy(dtype=float))
        risk_rows.append(diagnostics)

    final_weights = pd.DataFrame(
        adjusted_rows, index=raw_weights.index, columns=ASSETS,
    )
    # Keep the frozen CSV column order: vol_before, vol_after, scaling_factor.
    risk = pd.DataFrame(risk_rows, index=raw_weights.index)
    return final_weights, risk


def remaining_allocation_weight(
    target_weights: pd.DataFrame,
    *,
    long_only: bool,
) -> pd.Series:
    """Return post-cap idle allocation; for Long-only, this is cash weight.

    With short positions, the remainder is *unused gross risk budget*, not
    accounting cash (short-sale proceeds and collateral are separate).
    """
    if not isinstance(target_weights, pd.DataFrame) or target_weights.empty:
        raise ValueError("Expected nonempty target weights")
    if list(target_weights.columns) != list(ASSETS):
        raise ValueError("Invalid target-weight asset order")
    values = target_weights.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite target weights")
    if long_only and (values < -1e-10).any():
        raise ValueError("Cash weight is defined only for Long-only weights")
    gross = np.abs(values).sum(axis=1)
    if (gross > 1.0 + 1e-10).any():
        raise ValueError("Gross target exposure exceeds 100%")

    unused = (1.0 - gross).clip(min=0.0)
    name = "cash_weight_final" if long_only else "unused_gross_budget_final"
    return pd.Series(unused, index=target_weights.index, name=name, dtype=float)


# Regime label validation and explicit cluster-budget contract.

def validate_regime_series(
    regimes: pd.Series,
    decision_dates: pd.DatetimeIndex,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> pd.Series:
    """Validate one known-at-decision regime label for every training decision.

    Labels must be nonempty strings; e.g. 'low_low' or 'cluster_0'.
    Matching dates enforce alignment, not point-in-time data provenance.
    """
    if not isinstance(regimes, pd.Series):
        raise TypeError("regimes must be a pandas Series")
    if not isinstance(decision_dates, pd.DatetimeIndex):
        raise TypeError("decision_dates must be a DatetimeIndex")

    for name, index in (
        ("regimes.index", regimes.index),
        ("decision_dates", decision_dates),
    ):
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError(f"{name} must be a DatetimeIndex")
        if (
            len(index) == 0
            or index.hasnans
            or index.has_duplicates
            or not index.is_monotonic_increasing
            or index.tz is not None
            or not index.equals(index.normalize())
        ):
            raise ValueError(f"{name} contains invalid, unordered or duplicated dates")

    if not regimes.index.equals(decision_dates):
        raise ValueError("regimes and decision_dates must match exactly")

    start = pd.Timestamp(train_start)
    end = pd.Timestamp(train_end)
    if (
        pd.isna(start)
        or pd.isna(end)
        or start.tzinfo is not None
        or end.tzinfo is not None
        or start != start.normalize()
        or end != end.normalize()
        or start > end
    ):
        raise ValueError("Invalid training boundaries")
    if decision_dates.min() < start or decision_dates.max() > end:
        raise ValueError("Regime decisions must stay inside training")

    values = regimes.tolist()
    if any(
        not isinstance(value, str)
        or not value
        or value != value.strip()
        for value in values
    ):
        raise ValueError("Regime labels must be nonempty, trimmed strings")

    return pd.Series(values, index=decision_dates, name="regime", dtype="object")


def build_regime_budget_panel(
    regimes: pd.Series,
    decision_dates: pd.DatetimeIndex,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    budget_policy: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Map dated regime labels to nonnegative, fully allocated *risk budgets*.

    'budget_policy' is explicit and fixed before any held-out evaluation.
    It is NOT inferred from labels or future portfolio returns.
    The downstream trend signals can subsequently leave some budget in cash.
    """
    validated = validate_regime_series(
        regimes, decision_dates, train_start, train_end
    )
    if not isinstance(budget_policy, Mapping) or not budget_policy:
        raise ValueError("budget_policy must be a nonempty mapping")

    missing_labels = set(validated.unique()) - set(budget_policy)
    if missing_labels:
        raise ValueError(f"No budget policy for labels: {sorted(missing_labels)}")

    normalized: dict[str, np.ndarray] = {}
    for label, by_asset in budget_policy.items():
        if not isinstance(label, str) or not label:
            raise ValueError("Budget policy labels must be nonempty strings")
        if not isinstance(by_asset, Mapping) or set(by_asset) != set(ASSETS):
            raise ValueError(f"Invalid asset budget for regime {label!r}")
        try:
            weights = np.array([by_asset[asset] for asset in ASSETS], dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Non-numeric budgets for regime {label!r}") from exc
        if (
            not np.isfinite(weights).all()
            or (weights < 0).any()
            or not np.isclose(weights.sum(), 1.0, atol=1e-10, rtol=0)
        ):
            raise ValueError(f"Invalid or unallocated budgets for regime {label!r}")
        normalized[label] = weights

    matrix = np.vstack([normalized[label] for label in validated])
    return pd.DataFrame(matrix, index=decision_dates, columns=list(ASSETS))


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
