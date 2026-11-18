"""Benchmarks and baseline strategies.

Benchmarks (passive):
- SPY Buy & Hold: the broad US equity market, the reference for the market
  exposure of the strategy (SPY is also one of the traded assets);
- Equal Weight Buy & Hold: one equal purchase of the four traded ETFs, never
  rebalanced; it represents a passive holder of the same universe.

Baselines (simple active rules, used to judge the added value of the agents):
- 60/40 SPY/IEF and Equal Weight, both rebalanced weekly;
- Volatility Control: weights closest to equal weight with ex-ante volatility
  at most ``baselines.volatility_control.target_volatility``;
- Macro Regime Long/Short and Long/Flat: 21/63-session momentum signals,
  regime risk budgets (VIX and breakeven inflation) and a volatility cap.

Every weight is computed with information available at the decision close.
Buy-and-hold portfolios contain a single decision at the start of the
evaluation period (see ``weights_for_period``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader import load_dataset
from src.features import build_macro_feature_levels
from src.portfolio import apply_volatility_cap_panel, build_volatility_control_weights
from src.regime_allocator import (
    apply_macro_regime_filter,
    build_regime_weight_panel,
    build_signed_weight_panel,
    build_trend_signal_panel,
    classify_macro_regimes,
)
from src.utils import ASSETS

BUY_AND_HOLD = ("SPY Buy & Hold", "Equal Weight Buy & Hold")


def _static(decisions: pd.DatetimeIndex, weights: dict) -> pd.DataFrame:
    frame = pd.DataFrame(0.0, index=decisions, columns=list(ASSETS))
    for asset, weight in weights.items():
        frame[asset] = float(weight)
    return frame


def build_baseline_weights(
    config: dict,
    root_dir: Path,
    market_daily: pd.DataFrame,
    decisions: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    """Target weights of every benchmark and baseline on ``decisions``."""
    settings = config["baselines"]
    closes = market_daily[[f"{a}_close" for a in ASSETS]].astype(float)
    closes.columns = list(ASSETS)
    daily_returns = closes.pct_change(fill_method=None).iloc[1:]

    weights = {
        "SPY Buy & Hold": _static(decisions, {"SPY": 1.0}),
        "Equal Weight Buy & Hold": _static(decisions, {a: 0.25 for a in ASSETS}),
        "60/40 SPY-IEF (weekly)": _static(decisions, settings["sixty_forty"]),
        "Equal Weight (weekly)": _static(decisions, settings["equal_weight"]),
    }

    vol = settings["volatility_control"]
    weights["Volatility Control"] = build_volatility_control_weights(
        daily_returns, decisions, int(vol["covariance_window"]), float(vol["target_volatility"])
    )

    # Macro-regime baselines: VIX and 10Y breakeven against trailing medians.
    regime_cfg = settings["macro_regime"]
    macro_dir = root_dir / config["paths"]["raw_data"] / "macro"

    def macro(series_id: str) -> pd.Series:
        frame = load_dataset(macro_dir / f"{series_id}.csv", strict=True)
        return pd.to_numeric(frame[series_id], errors="coerce")

    warmup_start = pd.Timestamp(config["data"]["start_date"]) + pd.Timedelta(days=7)
    warmup = pd.date_range(warmup_start, decisions.min() - pd.Timedelta(days=1), freq="W-FRI")
    levels = build_macro_feature_levels(warmup.union(decisions), macro("VIXCLS"), macro("T10YIE"))
    regimes = classify_macro_regimes(levels, window_weeks=int(regime_cfg["regime_window_weeks"])).loc[decisions, "regime"]
    budgets = build_regime_weight_panel(regimes)
    trend = build_trend_signal_panel(closes, decisions)

    for name, signals in (
        ("Macro Regime Long/Short", apply_macro_regime_filter(trend, regimes)),
        ("Macro Regime Long/Flat", trend.clip(lower=0)),
    ):
        raw, _ = build_signed_weight_panel(signals, budgets)
        capped, _ = apply_volatility_cap_panel(
            raw,
            daily_returns,
            window=int(regime_cfg["covariance_window"]),
            target_volatility=float(regime_cfg["target_volatility"]),
        )
        weights[name] = capped

    return weights


def weights_for_period(name: str, weights: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """Restrict weights to the period; buy-and-hold keeps only the first decision."""
    period = weights.loc[weights.index >= start]
    return period.iloc[:1] if name in BUY_AND_HOLD else period
