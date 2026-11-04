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
