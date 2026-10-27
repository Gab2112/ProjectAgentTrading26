"""Shared regime-provider interface for Level B and future clustering.

The module validates a dated regime-label series and converts labels to
portfolio risk budgets using an explicitly supplied allocation policy.
It does not fit a clustering model, infer cluster economics, or certify
historical point-in-time data availability.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.regime_allocator import ASSETS


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
