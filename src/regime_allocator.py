"""Experimental regime-based allocation for Level B."""

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

    # Local import avoids circular dependency.
    from src.regime_signals import validate_signal_panel

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