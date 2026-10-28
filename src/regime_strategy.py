"""Level B: independently built B_v1/B_v2 strategies with shared causal inputs.

The historical rule set and risk-cap mathematics are unchanged. Default
execution is training-only, with macro observations aligned by date but NOT
certified point-in-time. External clustering labels are optional ONLY for
B_v2, with an explicit budget policy supplied by the caller.
"""

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from src.data_transformation import load_dataset
from src.time_split import get_time_windows
from src.macro_availability import build_macro_feature_levels, classify_macro_regimes
from src.regime_allocator import (
    build_trend_signal_panel,
    apply_macro_regime_filter,
    build_regime_budget_panel,
    validate_regime_series,
    ASSETS,
    build_regime_weight_panel,
    build_signed_weight_panel,
    apply_volatility_cap_panel,
    remaining_allocation_weight,
)


def _load_common_training_inputs(
    config: dict,
    root_dir: str | Path,
    *,
    regimes_override: pd.Series | None = None,
    budget_policy: Mapping[str, Mapping[str, float]] | None = None,
) -> dict:
    """Load price history, 21/63-day momentum, and weekly regime budgets once."""
    if (
        config["evaluation"]["evaluate_oos"] is not False
        or config["validation"]["development_only"] is not True
    ):
        raise ValueError("Level B development must remain training-only")

    if (regimes_override is None) != (budget_policy is None):
        raise ValueError("External regime labels require an explicit budget_policy")

    root = Path(root_dir)
    windows = get_time_windows(config)
    start, end = windows["train_start"], windows["train_end"]
    if start is None:
        raise ValueError("Missing training start date")

    processed = root / config["paths"]["processed_data"]
    calendar = load_dataset(
        processed / "weekly_decision_dates.csv", strict=True,
    ).index
    decisions = pd.DatetimeIndex(
        calendar[(calendar >= start) & (calendar <= end)], name="Date",
    )
    if decisions.empty:
        raise ValueError("No training decisions")

    market = load_dataset(
        processed / "market_daily.csv", strict=True,
    ).loc[:end]
    closes = market[[f"{asset}_close" for asset in ASSETS]].copy()
    closes.columns = list(ASSETS)
    closes = closes.astype(float)
    if not decisions.isin(closes.index).all():
        raise ValueError("Missing market prices at decision dates")

    trend = build_trend_signal_panel(closes, decisions)

    if regimes_override is None:
        # Frozen deterministic macro path. Preserve the exact historical
        # warm-up convention and observation-date alignment.
        macro_dir = root / config["paths"]["raw_data"] / "macro"

        def macro(name: str) -> pd.Series:
            data = load_dataset(
                macro_dir / f"{name}.csv", strict=True,
            ).loc[:end]
            return pd.to_numeric(data[name], errors="coerce")

        warmup_start = (
            pd.Timestamp(config["data"]["start_date"])
            + pd.Timedelta(days=7)
        )
        warmup = pd.date_range(
            warmup_start,
            decisions.min() - pd.Timedelta(days=1),
            freq="W-FRI",
        )
        features = build_macro_feature_levels(
            warmup.union(decisions), macro("VIXCLS"), macro("T10YIE"),
        )
        regimes = classify_macro_regimes(
            features, window_weeks=52,
        ).loc[decisions, "regime"]
        if regimes.isna().any():
            raise ValueError("Unclassified training decisions")
        budgets = build_regime_weight_panel(regimes)
    else:
        # Regime labels from clustering are NOT assumed to be equivalent to
        # deterministic low_low/high_high states. They need explicit budgets.
        regimes = validate_regime_series(
            regimes_override, decisions, start, end,
        )
        budgets = build_regime_budget_panel(
            regimes, decisions, start, end, budget_policy,
        )

    daily_returns = closes.pct_change(fill_method=None).dropna()
    return {
        "regimes": regimes,
        "trend_signals": trend,
        "budgets": budgets,
        "daily_returns": daily_returns,
    }


def _build_portfolio(common: dict, *, long_flat: bool) -> dict:
    """Build one independent strategy without invoking its other variant."""
    trend, regimes, budgets = (
        common["trend_signals"], common["regimes"], common["budgets"]
    )
    signals = (
        trend.clip(lower=0)
        if long_flat
        else apply_macro_regime_filter(trend, regimes)
    )
    raw_weights, unused_budget = build_signed_weight_panel(signals, budgets)
    weights, risk = apply_volatility_cap_panel(
        raw_weights, common["daily_returns"],
        window=126, target_volatility=0.08,
    )

    if (
        (weights.abs().sum(axis=1) > 1 + 1e-10).any()
        or (risk["vol_after"] > 0.08 + 1e-8).any()
    ):
        raise RuntimeError("Level B risk constraints violated")
    if long_flat and ((weights < -1e-10).any().any() or
                      (weights.sum(axis=1) > 1 + 1e-10).any()):
        raise RuntimeError("B_v2 Long/Flat constraints violated")

    result = {
        "regimes": regimes,
        "trend_signals": trend,
        "signals": signals,
        "budgets": budgets,
        "signed_weights": raw_weights,
        "target_weights": weights,
        "risk": risk,  # Legacy three columns remain byte-compatible.
        # Legacy key retains its meaning: budget unused BEFORE risk scaling.
        "unused_budget": unused_budget,
        "unused_gross_budget_final": remaining_allocation_weight(
            weights, long_only=False,
        ),
    }
    if long_flat:
        result["cash_weight_final"] = remaining_allocation_weight(
            weights, long_only=True,
        )
    return result


def build_level_b_training(
    config: dict,
    root_dir: str | Path = ".",
) -> dict:
    """Reproduce frozen B_v1: momentum Long/Short and macro directional filter."""
    common = _load_common_training_inputs(config, root_dir)
    return _build_portfolio(common, long_flat=False)


def build_level_b_v2_training(
    config: dict,
    root_dir: str | Path = ".",
    *,
    regimes_override: pd.Series | None = None,
    budget_policy: Mapping[str, Mapping[str, float]] | None = None,
) -> dict:
    """Build B_v2 Long/Flat, optionally with explicitly mapped cluster labels.

    The default deterministic path reproduces frozen B_v2. Custom labels
    are validated for decision-date alignment, NOT for data publication
    vintages or look-ahead within the clustering model. Never evaluate
    custom regimes on 2022+ until their research protocol is frozen.
    """
    common = _load_common_training_inputs(
        config, root_dir,
        regimes_override=regimes_override,
        budget_policy=budget_policy,
    )
    return _build_portfolio(common, long_flat=True)
