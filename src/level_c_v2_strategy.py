"""Level C v2 experimental Long/Short/Flat consensus strategy (training only).

Reuse the already-validated C_v1 inputs and B_v2 macro budgets, but require
AGREEMENT between 21/63-session momentum and Orchestrator direction:
  (+1, +1, positive confidence) -> LONG
  (-1, -1, positive confidence) -> SHORT
  all other combinations         -> FLAT

One shared 126-session covariance volatility cap (8%) is applied to the
SIGNED raw weights. Gross exposure stays <= 100%; remaining gross capacity is
NOT the cash ledger for a short portfolio. The historical backtest belongs to
Phase 4 and must use borrow costs and margin rules of the shared simulator.

This is an IN-SAMPLE research experiment: Orchestrator was optimized on the
same period even though upstream RF/SVM forecasts were out-of-fold.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_transformation import load_dataset
from src.level_c_allocator import _asset_panel, _validated_window
from src.level_c_strategy import build_level_c_training
from src.regime_allocator import (
    ASSETS,
    apply_volatility_cap_panel,
    build_trend_signal_panel,
)
from src.time_split import get_time_windows


_TOL = 1e-10


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
    """Create signed C_v2 targets from the identical, validated C_v1 inputs.

    IMPORTANT: pass C_v1's original regime budgets (NOT B_v2 target weights).
    signed_trend must contain actual {-1,0,+1} B_v2 momentum decisions.
    Historical returns may include pre-training warm-up but no post-training
    observations. No implicit reindex, filling, or sign imputation occurs.
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


def build_level_c_v2_training(config: dict, root_dir: str | Path = ".") -> dict:
    """Connect C_v2 to exactly the C_v1 Orchestrator dates, never 2017+."""
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")
    if (
        config.get("validation", {}).get("development_only") is not True
        or config.get("evaluation", {}).get("evaluate_oos") is not False
    ):
        raise ValueError("C_v2 is strictly training-only")

    # C_v1 performs strict four-asset signal validation and builds B_v2 once.
    c_v1 = build_level_c_training(config, root_dir)
    decisions = c_v1["target_weights"].index
    windows = get_time_windows(config)
    start, end = windows["train_start"], windows["train_end"]
    if start is None or end is None:
        raise ValueError("Training boundaries are required")

    root = Path(root_dir)
    market = load_dataset(
        root / config["paths"]["processed_data"] / "market_daily.csv", strict=True,
    ).loc[:end]
    closes = market[[f"{asset}_close" for asset in ASSETS]].astype(float).copy()
    closes.columns = list(ASSETS)
    signed_trend = build_trend_signal_panel(closes, decisions)
    returns = closes.pct_change(fill_method=None).dropna()

    result = build_level_c_v2_target_weights(
        c_v1, signed_trend, returns,
        train_start=start,
        train_end=end,
    )
    result["c_v1_target_weights_same_dates"] = c_v1["target_weights"].copy()
    result["b_v2_target_weights_same_dates"] = c_v1["b_v2_target_weights_same_dates"].copy()
    result["regimes"] = c_v1["regimes"].copy()
    return result
