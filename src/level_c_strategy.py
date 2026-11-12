"""Level C: costruzione delle strategie C_v1 e C_v2 (solo training).

``build_level_c_training`` (C_v1) e ``build_level_c_v2_training`` (C_v2).
Le regole dei pesi stanno in ``src/level_c_allocator.py``.

C_v1 - minimal training-only integration adapter for the Level C v1 allocator.

Reuses B_v2 regimes/budgets/momentum and existing on-disk Orchestrator
outputs; does NOT train models, execute trades, change frozen B outputs,
or authorize validation/OOS. The shared daily simulator is Phase 4.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader import load_dataset
from src.level_c_allocator import (
    build_level_c_target_weights,
    build_level_c_v2_target_weights,
)
from src.regime_allocator import build_trend_signal_panel
from src.regime_strategy import build_level_b_v2_training
from src.time_split import get_time_windows
from src.utils import ASSETS


def build_level_c_training(config: dict, root_dir: str | Path = ".") -> dict:
    """Load frozen training inputs and calculate Level C long/flat targets.

    Requires the four regenerated signals_orchestrator_<asset>.csv files.
    These are in-sample, ensemble-fitted outputs, NOT independent OOS signals.
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")
    if (
        config.get("validation", {}).get("development_only") is not True
        or config.get("evaluation", {}).get("evaluate_oos") is not False
    ):
        raise ValueError("Level C runs only with development_only=true and evaluate_oos=false")

    windows = get_time_windows(config)
    start, end = windows["train_start"], windows["train_end"]
    if start is None or end is None:
        raise ValueError("Explicit training boundaries are required")

    root = Path(root_dir)
    signal_dir = root / config["paths"]["signals"]
    signals_by_asset = {}
    for asset in ASSETS:
        path = signal_dir / f"signals_orchestrator_{asset.lower()}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing orchestrator signals: {path}")
        frame = pd.read_csv(path, index_col=0)
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise"), name="Date")
        signals_by_asset[asset] = frame

    # Generate the established baseline independently; no B_v1 path is called.
    b = build_level_b_v2_training(config, root)

    market = load_dataset(
        root / config["paths"]["processed_data"] / "market_daily.csv",
        strict=True,
    ).loc[:end]
    closes = market[[f"{asset}_close" for asset in ASSETS]].astype(float).copy()
    closes.columns = list(ASSETS)
    daily_returns = closes.pct_change(fill_method=None).dropna()

    result = build_level_c_target_weights(
        signals_by_asset=signals_by_asset,
        b_budgets=b["budgets"],
        b_trend_signals=b["trend_signals"],
        daily_returns=daily_returns,
        train_start=start,
        train_end=end,
    )
    # Diagnostics for comparisons on the EXACT same portfolio decision dates.
    decisions = result["target_weights"].index
    result["b_v2_target_weights_same_dates"] = b["target_weights"].loc[decisions].copy()
    result["regimes"] = b["regimes"].loc[decisions].copy()
    return result


# -----------------------------------------------------------------------------
# C_v2 (ex level_c_v2_strategy.py)
#
# Level C v2 experimental Long/Short/Flat consensus strategy (training only).
#
# Reuse the already-validated C_v1 inputs and B_v2 macro budgets, but require
# AGREEMENT between 21/63-session momentum and Orchestrator direction:
#   (+1, +1, positive confidence) -> LONG
#   (-1, -1, positive confidence) -> SHORT
#   all other combinations         -> FLAT
#
# One shared 126-session covariance volatility cap (8%) is applied to the
# SIGNED raw weights. Gross exposure stays <= 100%; remaining gross capacity is
# NOT the cash ledger for a short portfolio. The historical backtest belongs to
# Phase 4 and must use borrow costs and margin rules of the shared simulator.
#
# This is an IN-SAMPLE research experiment: Orchestrator was optimized on the
# same period even though upstream RF/SVM forecasts were out-of-fold.
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Retrospective research extension (not the archived official OOS candidate).
# -----------------------------------------------------------------------------

def build_regime_adaptive_core_targets(
    market_daily: pd.DataFrame,
    signals_by_asset: dict[str, pd.DataFrame],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    core_weight: float = 0.35,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Compare frozen Core targets with a B/C-gated conditional allocation.

    Normal regime (SPY close >= its past-200-session SMA): original Core.
    Defensive regime: only positive 21- AND 63-session price momentum and
    a +1 agent direction qualify for the satellite; the same original
    ranking/hysteresis rule then selects among qualifying ETFs.

    Both candidate selectors carry their own hysteresis state across dates;
    the regime switch chooses from those two independently constructed
    target series. This is a *retrospective* research convention, not the
    original frozen strategy or a newly independent OOS evaluation.
    """
    import numpy as np

    from src.frozen_satellite_selector import generate_dynamic_growth_matrix
    from src.core_satellite import (
        SATELLITE_ORDER,
        build_core_satellite_targets,
    )
    from src.time_split import build_weekly_decision_dates

    boundary = pd.Timestamp(end)
    if not 0.0 <= core_weight <= 1.0:
        raise ValueError('core_weight must be in [0, 1]')
    if set(signals_by_asset) != set(ASSETS):
        raise ValueError('Four ETF agent signal inputs are required')

    # Explicitly exclude observations after the requested endpoint.
    market = market_daily.loc[:boundary].copy()
    original = build_core_satellite_targets(
        market, signals_by_asset, start, end, core_weight=core_weight
    )
    decisions = original.index
    weekly = build_weekly_decision_dates(market).index
    closing_prices = market[[f'{asset}_close' for asset in ASSETS]].astype(float)
    closing_prices.columns = list(ASSETS)
    closing_prices = closing_prices.loc[:decisions[-1]]

    sma200 = closing_prices['SPY'].rolling(200, min_periods=200).mean()
    stress = closing_prices['SPY'].loc[decisions].lt(sma200.loc[decisions])
    if sma200.loc[decisions].isna().any() or stress.isna().any():
        raise ValueError('At least 200 historical trading closes required')
    stress.name = 'SPY_below_SMA200'

    trend = build_trend_signal_panel(closing_prices, decisions)
    filtered_signals: dict[str, pd.DataFrame] = {}
    for asset in SATELLITE_ORDER:
        frame = signals_by_asset[asset].loc[decisions].copy()
        eligible = trend[asset].eq(1) & frame['signal'].eq(1)
        frame.loc[~eligible, 'signal'] = 0
        filtered_signals[asset] = frame

    weekly_close = market.loc[weekly, [f'{asset}_close' for asset in ASSETS]]
    selected = generate_dynamic_growth_matrix(
        filtered_signals,
        df_features=weekly_close,
        default_vol=0.10,
        min_confidence=0.35,
        hysteresis_buffer=0.015,
    )
    defensive = selected.reindex(index=decisions, columns=list(ASSETS)).astype(float)
    defensive *= 1.0 - core_weight
    defensive['SPY'] += core_weight

    adaptive = original.copy()
    adaptive.loc[stress] = defensive.loc[stress]
    for name, weights in {'original': original, 'defensive': defensive,
                          'adaptive': adaptive}.items():
        values = weights.to_numpy(dtype=float)
        if (not np.isfinite(values).all()
                or (np.abs(values).sum(axis=1) > 1.0 + 1e-10).any()):
            raise ValueError(f'{name}: invalid target gross exposure')
    return {
        'original': original,
        'defensive_long_flat': defensive,
        'regime_adaptive': adaptive,
        'stress': stress,
    }
