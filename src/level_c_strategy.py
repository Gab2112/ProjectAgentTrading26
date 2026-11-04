"""Minimal training-only integration adapter for the Level C v1 allocator.

Reuses B_v2 regimes/budgets/momentum and existing on-disk Orchestrator
outputs; does NOT train models, execute trades, change frozen B outputs,
or authorize validation/OOS. The shared daily simulator is Phase 4.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_transformation import load_dataset
from src.level_c_allocator import build_level_c_target_weights
from src.regime_allocator import ASSETS
from src.regime_strategy import build_level_b_v2_training
from src.time_split import get_time_windows


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
