"""Level B: exploratory training-only portfolio construction."""

from pathlib import Path

import pandas as pd

from src.data_loader import load_dataset
from src.time_split import get_time_windows
from src.macro_features import (
    build_macro_feature_levels,
    classify_macro_regimes,
)
from src.regime_signals import (
    build_trend_signal_panel,
    apply_macro_regime_filter,
)
from src.regime_allocator import (
    ASSETS,
    build_regime_weight_panel,
    build_signed_weight_panel,
    apply_signed_volatility_cap,
)


def build_level_b_training(
    config: dict,
    root_dir: str | Path = ".",
) -> dict:
    """Build Level B weights using training data only.

    Macro data are date-aligned proxies, not PIT-certified.
    No backtest is executed.
    """

    if (
        config["evaluation"]["evaluate_oos"] is not False
        or config["validation"]["development_only"] is not True
    ):
        raise ValueError("Level B development must remain training-only")

    root = Path(root_dir)
    windows = get_time_windows(config)
    start, end = windows["train_start"], windows["train_end"]

    if start is None:
        raise ValueError("Missing training start date")

    processed = root / config["paths"]["processed_data"]
    macro_dir = root / config["paths"]["raw_data"] / "macro"

    # 1. Actual weekly trading decisions, training only.
    calendar = load_dataset(
        processed / "weekly_decision_dates.csv",
        strict=True,
    ).index

    decisions = pd.DatetimeIndex(
        calendar[(calendar >= start) & (calendar <= end)],
        name="Date",
    )

    if decisions.empty:
        raise ValueError("No training decisions")

    # 2. Adjusted market prices, including historical warm-up.
    market = load_dataset(
        processed / "market_daily.csv",
        strict=True,
    ).loc[:end]

    closes = market[
        [f"{asset}_close" for asset in ASSETS]
    ].copy()

    closes.columns = list(ASSETS)
    closes = closes.astype(float)

    if not decisions.isin(closes.index).all():
        raise ValueError("Missing market prices at decision dates")

    # 3. Market momentum.
    trend = build_trend_signal_panel(closes, decisions)

    # 4. Historical macro observations.
    def macro(name: str) -> pd.Series:
        data = load_dataset(
            macro_dir / f"{name}.csv",
            strict=True,
        ).loc[:end]

        return pd.to_numeric(data[name], errors="coerce")

    # Preserve the original 2004 warm-up convention.
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
        warmup.union(decisions),
        macro("VIXCLS"),
        macro("T10YIE"),
    )

    regimes = classify_macro_regimes(
        features,
        window_weeks=52,
    ).loc[decisions, "regime"]

    if regimes.isna().any():
        raise ValueError("Unclassified training decisions")

    # 5. Macro-filtered directional signals.
    signals = apply_macro_regime_filter(trend, regimes)

    # 6. Regime budgets and signed weights.
    budgets = build_regime_weight_panel(regimes)

    signed, unused = build_signed_weight_panel(
        signals, budgets
    )

    # 7. Volatility cap using the previous 126 daily returns.
    returns = closes.pct_change(
        fill_method=None
    ).dropna()

    weight_rows = []
    risk_rows = []

    for date in decisions:
        weights, diagnostics = apply_signed_volatility_cap(
            weights=signed.loc[date],
            daily_returns=returns,
            decision_date=date,
            window=126,
            target_volatility=0.08,
        )

        weight_rows.append(weights.to_numpy())
        risk_rows.append(diagnostics)

    final_weights = pd.DataFrame(
        weight_rows,
        index=decisions,
        columns=ASSETS,
    )

    risk = pd.DataFrame(risk_rows, index=decisions)

    if (
        (final_weights.abs().sum(axis=1) > 1 + 1e-10).any()
        or (risk["vol_after"] > 0.08 + 1e-8).any()
    ):
        raise RuntimeError("Level B risk constraints violated")

    return {
        "regimes": regimes,
        "trend_signals": trend,
        "signals": signals,
        "budgets": budgets,
        "signed_weights": signed,
        "target_weights": final_weights,
        "risk": risk,
        "unused_budget": unused,
    }



def build_level_b_v2_training(
    config: dict,
    root_dir: str | Path = ".",
) -> dict:
    """Build the exploratory B_v2 Long/Flat candidate.

    - Momentum signals: 21/63-day trend.
    - Negative signals become Flat.
    - No macro directional filter.
    - Regime-dependent allocation budgets.
    - 8% volatility cap applied to actual Long/Flat weights.

    Training only. Macro PIT availability not certified.
    """

    baseline = build_level_b_training(config, root_dir)

    # Directional decisions from pure momentum.
    signals = baseline["trend_signals"].clip(lower=0)

    # Retain the existing regime-dependent budgets.
    budgets = baseline["budgets"]

    raw_weights, unused_budget = build_signed_weight_panel(
        signals,
        budgets,
    )

    # Historical market data, including indicator warm-up.
    end = get_time_windows(config)["train_end"]

    market = load_dataset(
        Path(root_dir)
        / config["paths"]["processed_data"]
        / "market_daily.csv",
        strict=True,
    ).loc[:end]

    closes = market[
        [f"{a}_close" for a in ASSETS]
    ].copy()

    closes.columns = list(ASSETS)

    daily_returns = closes.pct_change(
        fill_method=None
    ).dropna()

    weight_rows = []
    risk_rows = []

    for date in signals.index:
        weights, diagnostics = apply_signed_volatility_cap(
            weights=raw_weights.loc[date],
            daily_returns=daily_returns,
            decision_date=date,
            window=126,
            target_volatility=0.08,
        )

        weight_rows.append(weights.to_numpy(dtype=float))
        risk_rows.append(diagnostics)

    final_weights = pd.DataFrame(
        weight_rows,
        index=signals.index,
        columns=ASSETS,
    )

    risk = pd.DataFrame(
        risk_rows,
        index=signals.index,
    )

    # Long/Flat: no negative positions, no leverage.
    if (
        (final_weights < -1e-10).any().any()
        or (final_weights.sum(axis=1) > 1 + 1e-10).any()
        or (risk["vol_after"] > 0.08 + 1e-8).any()
    ):
        raise RuntimeError("B_v2 portfolio constraints violated")

    return {
        **baseline,
        "signals": signals,
        "signed_weights": raw_weights,
        "target_weights": final_weights,
        "risk": risk,
        "unused_budget": unused_budget,
    }