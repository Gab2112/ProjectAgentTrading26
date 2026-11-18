"""Risk-based portfolio construction used by the baseline strategies.

- ``compute_volatility_control_weights``: long-only, fully invested weights
  closest to equal weight subject to an ex-ante annualised volatility limit
  (convex problem solved with CVXPY/CLARABEL; minimum variance is used when
  the limit is infeasible).
- ``build_volatility_control_weights``: applies the optimisation at every
  decision date using only the daily returns observed up to that close.
- ``apply_signed_volatility_cap`` / ``apply_volatility_cap_panel``: scale a
  signed weight vector proportionally so that its ex-ante volatility does not
  exceed a target, without changing the direction of any position.

Covariances are sample covariances of the last ``window`` daily returns,
annualised with 252 sessions. No function uses information after the
decision date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import ASSETS

SESSIONS_PER_YEAR = 252
WEIGHT_TOLERANCE = 1e-7
VOLATILITY_TOLERANCE = 1e-7


def _validated_returns(returns: pd.DataFrame, window: int, max_volatility: float) -> pd.DataFrame:
    """Validate the optimiser inputs and return the last ``window`` rows."""
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("Returns must be a pandas DataFrame.")
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)) or window < 2:
        raise ValueError("The window must be an integer of at least two sessions.")
    limit = float(max_volatility)
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("The volatility limit must be finite and positive.")
    if len(returns) < window:
        raise ValueError("Insufficient observations for the requested window.")
    index = returns.index
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.hasnans
        or index.tz is not None
        or not index.is_monotonic_increasing
        or index.has_duplicates
    ):
        raise ValueError("Returns require sorted, unique, timezone-naive dates.")
    if not set(ASSETS).issubset(returns.columns):
        raise ValueError(f"Returns must contain every ETF: {', '.join(ASSETS)}.")

    data = returns.loc[:, list(ASSETS)].tail(window).astype(float)
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Missing or invalid returns.")
    return data


def _checked_solution(weights: np.ndarray) -> np.ndarray:
    """Check that solver weights are long-only and fully invested (no silent normalisation)."""
    if weights is None:
        raise RuntimeError("The solver returned no solution.")
    values = np.asarray(weights, dtype=float).reshape(-1)
    if (
        len(values) != len(ASSETS)
        or not np.isfinite(values).all()
        or abs(values.sum() - 1.0) > WEIGHT_TOLERANCE
        or np.any(values < -WEIGHT_TOLERANCE)
        or np.any(values > 1.0 + WEIGHT_TOLERANCE)
    ):
        raise RuntimeError("The solution violates the weight constraints.")
    return values


def compute_volatility_control_weights(
    returns: pd.DataFrame,
    window: int,
    max_volatility: float,
) -> tuple[pd.Series, dict]:
    """Weights closest to equal weight whose ex-ante volatility is at most ``max_volatility``.

    If the equal-weight portfolio already satisfies the limit it is kept;
    otherwise ``||w - w_EW||^2`` is minimised subject to sum(w) = 1, w >= 0
    and sqrt(w' Sigma w) <= limit. If no such portfolio exists, the
    minimum-variance portfolio is returned and flagged.
    """
    import cvxpy as cp

    data = _validated_returns(returns, window, max_volatility)
    limit = float(max_volatility)
    cov = data.cov().to_numpy(dtype=float) * SESSIONS_PER_YEAR
    cov = (cov + cov.T) / 2.0
    reference = np.full(len(ASSETS), 1.0 / len(ASSETS))

    def volatility(weights: np.ndarray) -> float:
        return float(np.sqrt(max(float(weights @ cov @ weights), 0.0)))

    vol_ew = volatility(reference)
    if vol_ew <= limit:
        return pd.Series(reference, index=ASSETS), {
            "status": "equal_weight", "volatility_ew": vol_ew, "volatility_target": vol_ew, "limit_met": True,
        }

    # A PSD factorisation turns the variance constraint into a second-order cone.
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if eigenvalues.min() < -1e-10:
        raise ValueError("The covariance matrix is not positive semidefinite.")
    factor = np.diag(np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T

    w = cp.Variable(len(ASSETS))
    constraints = [cp.sum(w) == 1, w >= 0]
    # Expressing risk in percentage points improves numerical conditioning.
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(w - reference)),
        constraints + [cp.norm(100.0 * (factor @ w), 2) <= 100.0 * limit],
    )
    problem.solve(solver=cp.CLARABEL)

    if problem.status in ("optimal", "optimal_inaccurate"):
        weights, status = _checked_solution(w.value), "optimised"
    elif problem.status in ("infeasible", "infeasible_inaccurate"):
        fallback = cp.Problem(cp.Minimize(cp.sum_squares(100.0 * (factor @ w))), constraints)
        fallback.solve(solver=cp.CLARABEL)
        if fallback.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Minimum-variance fallback not solved: {fallback.status}")
        weights, status = _checked_solution(w.value), "fallback_minimum_variance"
    else:
        raise RuntimeError(f"Volatility-control optimisation not solved: {problem.status}")

    final_vol = volatility(weights)
    limit_met = final_vol <= limit + VOLATILITY_TOLERANCE
    if status == "optimised" and not limit_met:
        raise RuntimeError("The solution violates the volatility limit.")
    return pd.Series(weights, index=ASSETS), {
        "status": status, "volatility_ew": vol_ew, "volatility_target": final_vol, "limit_met": limit_met,
    }


def build_volatility_control_weights(
    daily_returns: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
    window: int,
    max_volatility: float,
) -> pd.DataFrame:
    """Apply ``compute_volatility_control_weights`` at every decision date.

    The return of the decision session is included because the decision is
    taken after that close; later returns are never used.
    """
    rows = []
    for decision in decision_dates:
        past = daily_returns.loc[:decision].tail(window)
        if len(past) < window:
            raise ValueError(f"Insufficient history at {decision.date()}: {window} daily returns are required.")
        weights, _ = compute_volatility_control_weights(past, window=window, max_volatility=max_volatility)
        rows.append(weights.rename(decision))
    frame = pd.DataFrame(rows)
    frame.index.name = "Date"
    return frame


def apply_signed_volatility_cap(
    weights: pd.Series,
    daily_returns: pd.DataFrame,
    decision_date: pd.Timestamp,
    window: int = 126,
    target_volatility: float = 0.08,
) -> tuple[pd.Series, dict]:
    """Scale signed weights so that ex-ante volatility does not exceed the target."""
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
    if decision_date.tz is not None or decision_date not in daily_returns.index:
        raise ValueError("Invalid decision date")
    if window < 2 or not np.isfinite(target_volatility) or target_volatility <= 0:
        raise ValueError("Invalid risk parameters")

    w = weights.to_numpy(dtype=float)
    if not np.isfinite(w).all() or np.abs(w).sum() > 1.0 + 1e-10:
        raise ValueError("Invalid signed weights")

    # Only observations available at the decision close.
    history = daily_returns.loc[:decision_date].tail(window).astype(float)
    if len(history) != window:
        raise ValueError("Insufficient return history")
    if not np.isfinite(history.to_numpy()).all():
        raise ValueError("Invalid historical returns")

    covariance = history.cov().to_numpy(dtype=float) * SESSIONS_PER_YEAR
    volatility_before = float(np.sqrt(max(float(w @ covariance @ w), 0.0)))
    factor = min(1.0, target_volatility / volatility_before) if volatility_before > 0 else 1.0

    adjusted = pd.Series(w * factor, index=ASSETS, dtype=float)
    return adjusted, {
        "vol_before": volatility_before,
        "vol_after": volatility_before * factor,
        "scaling_factor": factor,
    }


def apply_volatility_cap_panel(
    raw_weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
    *,
    window: int = 126,
    target_volatility: float = 0.08,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply ``apply_signed_volatility_cap`` at every decision date."""
    if not isinstance(raw_weights, pd.DataFrame) or raw_weights.empty:
        raise ValueError("Expected a non-empty target-weight panel")
    if list(raw_weights.columns) != list(ASSETS):
        raise ValueError("Invalid target-weight asset order")
    index = raw_weights.index
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.hasnans
        or index.has_duplicates
        or not index.is_monotonic_increasing
        or index.tz is not None
    ):
        raise ValueError("Invalid target-weight decision dates")

    adjusted_rows, risk_rows = [], []
    for date in index:
        adjusted, diagnostics = apply_signed_volatility_cap(
            weights=raw_weights.loc[date],
            daily_returns=daily_returns,
            decision_date=date,
            window=window,
            target_volatility=target_volatility,
        )
        adjusted_rows.append(adjusted.to_numpy(dtype=float))
        risk_rows.append(diagnostics)

    final_weights = pd.DataFrame(adjusted_rows, index=index, columns=ASSETS)
    return final_weights, pd.DataFrame(risk_rows, index=index)
