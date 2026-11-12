"""Level C allocation rules (v1 Long/Flat and v2 Long/Short/Flat).

``build_level_c_target_weights`` implements C_v1 and
``build_level_c_v2_target_weights`` implements C_v2 (consensus between
momentum and the orchestrator). Data loading and strategy composition are
handled in ``src/level_c_strategy.py``.

Level C v1: training-only, long/flat B_v2 allocation gated by the AI agents.

Policy (decisions after the market close):
    raw_weight[i,t] = b_v2_budget[i,t] * (b_v2_trend[i,t] == +1)
                       * (agent_signal[i,t] == +1 and confidence[i,t] > 0)

The existing 126-session historical covariance overlay is applied EXACTLY ONCE.
Unused exposure is held in cash (0% yield under the project modelling
assumptions). This module does not train models, read files, select
parameters or run backtests.

Important: even when the underlying RF/SVM predictions are out-of-fold, the
output of the fitted orchestrator is not an independent OOS validation of
the ensemble.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.portfolio import apply_volatility_cap_panel
from src.regime_allocator import remaining_allocation_weight
from src.utils import ASSETS


_TOL = 1e-10


def _validated_dates(index: pd.Index, name: str) -> pd.DatetimeIndex:
    """Require strictly ordered, unique, timezone-naive, midnight decision/session dates.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
        raise ValueError(f"{name}: expected nonempty DatetimeIndex")
    if (
        index.hasnans
        or index.has_duplicates
        or not index.is_monotonic_increasing
        or index.tz is not None
        or not index.equals(index.normalize())
    ):
        raise ValueError(f"{name}: invalid, duplicated, or non-normalized dates")
    return index


def _validated_window(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    for name, value in (("train_start", start), ("train_end", end)):
        if not isinstance(value, pd.Timestamp):
            raise TypeError(f"{name} must be a pandas Timestamp")
        if pd.isna(value) or value.tzinfo is not None or value != value.normalize():
            raise ValueError(f"{name} must be a valid timezone-naive midnight")
    if start > end:
        raise ValueError("train_start must not exceed train_end")
    return start, end


def _asset_panel(panel: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(panel, pd.DataFrame) or list(panel.columns) != list(ASSETS):
        raise ValueError(f"{name}: columns must be exactly {list(ASSETS)} in this order")
    _validated_dates(panel.index, name)
    try:
        converted = panel.astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}: non-numeric values") from exc
    if not np.isfinite(converted.to_numpy(dtype=float)).all():
        raise ValueError(f"{name}: missing or non-finite values")
    return converted


def build_level_c_target_weights(
    signals_by_asset: Mapping[str, pd.DataFrame],
    b_budgets: pd.DataFrame,
    b_trend_signals: pd.DataFrame,
    daily_returns: pd.DataFrame,
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    covariance_window: int = 126,
    target_volatility: float = 0.08,
) -> dict:
    """Build Level C weekly target weights without accessing validation/OOS.

    Each signals_by_asset[asset] must have exactly the same dated decisions
    and at least the columns 'signal' (-1/0/+1) and 'confidence' ([0,1]).
    Missing observations are errors, never silently treated as FLAT.

    B_v2 budgets/trend may contain earlier training decisions (before ML OOF
    begins); their dates must include every agent decision. The daily returns
    include historical warm-up and the decision closes, not future observations.

    Returns weight panels, cash series, and the unchanged B risk diagnostics.
    """
    start, end = _validated_window(train_start, train_end)
    if isinstance(covariance_window, bool) or not isinstance(covariance_window, int) or covariance_window < 2:
        raise ValueError("covariance_window must be an integer >= 2")
    if (
        not isinstance(target_volatility, (float, int))
        or isinstance(target_volatility, bool)
        or not np.isfinite(target_volatility)
        or target_volatility <= 0
    ):
        raise ValueError("target_volatility must be positive and finite")

    if not isinstance(signals_by_asset, Mapping) or set(signals_by_asset) != set(ASSETS):
        raise ValueError(f"signals_by_asset requires all and only {list(ASSETS)}")

    agent_signals: dict[str, pd.Series] = {}
    confidences: dict[str, pd.Series] = {}
    decisions: pd.DatetimeIndex | None = None
    for asset in ASSETS:
        panel = signals_by_asset[asset]
        if not isinstance(panel, pd.DataFrame) or not {"signal", "confidence"}.issubset(panel.columns):
            raise ValueError(f"{asset}: expected signal/confidence DataFrame")
        index = _validated_dates(panel.index, f"{asset} agent decisions")
        if decisions is None:
            decisions = index
            if index.min() < start or index.max() > end:
                raise ValueError("Agent decisions exceed training boundaries")
        elif not index.equals(decisions):
            raise ValueError("Agent decision dates must match exactly across assets")
        try:
            signal = pd.to_numeric(panel["signal"], errors="raise").astype(float)
            confidence = pd.to_numeric(panel["confidence"], errors="raise").astype(float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{asset}: non-numeric signal/confidence") from exc
        if not np.isfinite(signal.to_numpy()).all() or not signal.isin((-1.0, 0.0, 1.0)).all():
            raise ValueError(f"{asset}: signals must be -1, 0, or +1")
        if (
            not np.isfinite(confidence.to_numpy()).all()
            or confidence.lt(0).any()
            or confidence.gt(1).any()
        ):
            raise ValueError(f"{asset}: confidence must lie in [0,1]")
        agent_signals[asset] = signal.astype("int8")
        confidences[asset] = confidence

    assert decisions is not None
    budgets = _asset_panel(b_budgets, "B_v2 budgets")
    trend = _asset_panel(b_trend_signals, "B_v2 trend signals")
    returns = _asset_panel(daily_returns, "daily returns")
    if budgets.index.min() < start or budgets.index.max() > end:
        raise ValueError("B_v2 budgets exceed training boundaries")
    if trend.index.min() < start or trend.index.max() > end:
        raise ValueError("B_v2 trend signals exceed training boundaries")
    if returns.index.max() > end:
        raise ValueError("Daily returns include future observations past train_end")

    if not budgets.index.equals(trend.index):
        raise ValueError("B_v2 budgets/trend calendars differ")
    if not decisions.isin(budgets.index).all():
        raise ValueError("B_v2 panels do not cover all agent decision dates")
    if not decisions.isin(returns.index).all():
        raise ValueError("Daily returns missing decision sessions")

    if budgets.lt(0).any().any() or budgets.gt(1 + _TOL).any().any():
        raise ValueError("B_v2 budgets must be in [0,1]")
    if not np.allclose(budgets.sum(axis=1).to_numpy(), 1.0, atol=_TOL, rtol=0):
        raise ValueError("B_v2 regime budgets must sum to 100%")
    if not trend.isin((-1.0, 0.0, 1.0)).all().all():
        raise ValueError("B_v2 trend signals must be -1, 0, or +1")

    for date in decisions:
        if len(returns.loc[:date].tail(covariance_window)) < covariance_window:
            raise ValueError(f"Insufficient covariance history at {date:%Y-%m-%d}")

    signals_frame = pd.DataFrame(agent_signals, index=decisions)[list(ASSETS)]
    confidence_frame = pd.DataFrame(confidences, index=decisions)[list(ASSETS)]
    trend_gate = trend.loc[decisions].eq(1).astype("int8")
    agent_gate = (signals_frame.eq(1) & confidence_frame.gt(0)).astype("int8")
    selected_budgets = budgets.loc[decisions].copy()

    raw_weights = selected_budgets.mul(trend_gate).mul(agent_gate).astype(float)
    raw_weights.index.name = "Date"
    target_weights, risk = apply_volatility_cap_panel(
        raw_weights=raw_weights,
        daily_returns=returns,
        window=covariance_window,
        target_volatility=float(target_volatility),
    )
    cash_weight_final = remaining_allocation_weight(target_weights, long_only=True)

    if (
        not np.isfinite(target_weights.to_numpy(dtype=float)).all()
        or target_weights.lt(-_TOL).any().any()
        or target_weights.gt(selected_budgets + _TOL).any().any()
        or target_weights.sum(axis=1).gt(1 + _TOL).any()
        or (risk["vol_after"] > float(target_volatility) + 1e-8).any()
        or not np.isfinite(cash_weight_final.to_numpy(dtype=float)).all()
        or cash_weight_final.lt(-_TOL).any()
    ):
        raise RuntimeError("Level C portfolio constraints violated")

    for panel in (signals_frame, confidence_frame, trend_gate, agent_gate,
                  selected_budgets, target_weights, risk):
        panel.index.name = "Date"
    cash_weight_final.index.name = "Date"

    return {
        "agent_signals": signals_frame,
        "agent_confidences": confidence_frame,
        "agent_gate": agent_gate,
        "trend_gate": trend_gate,
        "budgets": selected_budgets,
        "raw_weights": raw_weights,
        "target_weights": target_weights,
        "risk": risk,
        "cash_weight_final": cash_weight_final,
        "sample": "training_diagnostic_only_not_independent_oos",
        "policy": "B_v2_momentum_and_binary_agent_gate_v1",
    }


def build_level_c_v2_target_weights(
    c_v1: Mapping,
    signed_trend: pd.DataFrame,
    daily_returns: pd.DataFrame,
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    covariance_window: int = 126,
    target_volatility: float = 0.08,
) -> dict:
    """Create signed C_v2 targets from the same validated C_v1 inputs.

    IMPORTANT: pass the original C_v1 regime budgets (NOT the B_v2 target weights).
    ``signed_trend`` must contain the actual {-1, 0, +1} B_v2 momentum decisions.
    Historical returns may include pre-training warm-up but no post-training
    observations. No implicit reindexing, filling or sign imputation occurs.
    """
    start, end = _validated_window(train_start, train_end)
    if (
        isinstance(covariance_window, bool)
        or not isinstance(covariance_window, int)
        or covariance_window < 2
    ):
        raise ValueError("covariance_window must be an integer >= 2")
    if (
        isinstance(target_volatility, bool)
        or not isinstance(target_volatility, (float, int))
        or not np.isfinite(target_volatility)
        or target_volatility <= 0
    ):
        raise ValueError("target_volatility must be positive and finite")
    if not isinstance(c_v1, Mapping) or c_v1.get("policy") != "B_v2_momentum_and_binary_agent_gate_v1":
        raise ValueError("Expected an authenticated Level C v1 result")

    required = (
        "agent_signals", "agent_confidences", "trend_gate",
        "budgets", "target_weights",
    )
    if any(name not in c_v1 for name in required):
        raise ValueError("Incomplete Level C v1 result")

    agents = _asset_panel(c_v1["agent_signals"], "C_v1 agent signals")
    confidence = _asset_panel(c_v1["agent_confidences"], "C_v1 confidence")
    budgets = _asset_panel(c_v1["budgets"], "C_v1 regime budgets")
    old_targets = _asset_panel(c_v1["target_weights"], "C_v1 targets")
    old_trend_gate = _asset_panel(c_v1["trend_gate"], "C_v1 trend gate")
    trend = _asset_panel(signed_trend, "Signed 21/63-day trend")
    returns = _asset_panel(daily_returns, "Daily historical returns")

    decisions = old_targets.index
    for name, panel in (
        ("agent signals", agents), ("agent confidence", confidence),
        ("budgets", budgets), ("old trend gate", old_trend_gate),
        ("signed trend", trend),
    ):
        if not decisions.equals(panel.index):
            raise ValueError(f"C_v2 {name} dates must match C_v1 decisions")
    if decisions.min() < start or decisions.max() > end or returns.index.max() > end:
        raise ValueError("C_v2 would exceed training boundaries")
    if not decisions.isin(returns.index).all():
        raise ValueError("Missing historical returns at decision sessions")
    if not agents.isin((-1, 0, 1)).all().all() or not trend.isin((-1, 0, 1)).all().all():
        raise ValueError("Only discrete {-1, 0, +1} signals are supported")
    if confidence.lt(0).any().any() or confidence.gt(1).any().any():
        raise ValueError("Confidence must be in [0,1]")
    if budgets.lt(0).any().any() or budgets.gt(1 + _TOL).any().any():
        raise ValueError("Regime budgets out of range")
    if not np.allclose(budgets.sum(axis=1), 1.0, atol=_TOL, rtol=0):
        raise ValueError("Regime budgets must sum to 100%")
    if not np.array_equal(trend.eq(1).to_numpy(dtype=int), old_trend_gate.to_numpy(dtype=int)):
        raise ValueError("C_v2 trend differs from B_v2 trend used in C_v1")

    long_gate = trend.eq(1) & agents.eq(1) & confidence.gt(0)
    short_gate = trend.eq(-1) & agents.eq(-1) & confidence.gt(0)
    signed_gate = long_gate.astype("int8") - short_gate.astype("int8")
    raw_weights = budgets.mul(signed_gate).astype(float)
    raw_weights.index.name = "Date"

    targets, risk = apply_volatility_cap_panel(
        raw_weights=raw_weights,
        daily_returns=returns,
        window=covariance_window,
        target_volatility=float(target_volatility),
    )
    gross = targets.abs().sum(axis=1).rename("gross_target_exposure")
    net = targets.sum(axis=1).rename("net_target_exposure")
    unused_gross = (1.0 - gross).rename("unused_gross_budget_final")

    if (
        not np.isfinite(targets.to_numpy(dtype=float)).all()
        or targets.abs().gt(budgets + _TOL).any().any()
        or gross.gt(1 + _TOL).any()
        or unused_gross.lt(-_TOL).any()
        or risk["vol_after"].gt(float(target_volatility) + 1e-8).any()
        or risk["scaling_factor"].gt(1 + _TOL).any()
    ):
        raise RuntimeError("C_v2 volatility or gross-exposure constraints violated")

    return {
        "agent_signals": agents,
        "agent_confidences": confidence,
        "trend_signals": trend,
        "signed_gate": signed_gate,
        "budgets": budgets,
        "raw_weights": raw_weights,
        "target_weights": targets,
        "risk": risk,
        "gross_target_exposure": gross,
        "net_target_exposure": net,
        "unused_gross_budget_final": unused_gross,
        "sample": "training_diagnostic_only_not_independent_oos",
        "policy": "B_v2_budgets_plus_signed_momentum_agent_consensus_v2",
    }
