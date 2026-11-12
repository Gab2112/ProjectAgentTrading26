"""Core-Satellite training-only, using the shared daily execution engine."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.frozen_satellite_selector import generate_dynamic_growth_matrix
from src.data_loader import load_dataset
from src.evaluator import calculate_metrics
from src.execution_engine import simulate_longshort_daily
from src.run_training_comparison import _guard_training
from src.time_split import build_weekly_decision_dates
from src.utils import ASSETS, ROOT_DIR, load_config


# Tie-breaking order of the original satellite selector.
SATELLITE_ORDER = ("SHY", "IEF", "GLD", "SPY")


def build_core_satellite_targets(
    market_daily,
    signals_by_asset,
    train_start,
    train_end,
    core_weight=0.35,
):
    """Construct causal weekly targets without accessing future market rows."""

    start = pd.Timestamp(train_start)
    end = pd.Timestamp(train_end)

    if not 0.0 <= core_weight <= 1.0:
        raise ValueError("Invalid core weight")

    if set(signals_by_asset) != set(ASSETS):
        raise ValueError("Four ETF signals are required")

    market = market_daily.loc[:end].copy()

    if market.empty:
        raise ValueError("Empty training market data")

    weekly_dates = build_weekly_decision_dates(market).index
    decision_dates = weekly_dates[weekly_dates >= start]

    validated = {}

    for asset in SATELLITE_ORDER:
        frame = signals_by_asset[asset].copy()

        if (
            not isinstance(frame.index, pd.DatetimeIndex)
            or frame.index.hasnans
            or frame.index.has_duplicates
            or not frame.index.is_monotonic_increasing
            or frame.index.tz is not None
        ):
            raise ValueError(f"{asset}: invalid signal dates")

        frame = frame.loc[:end]

        if not {"signal", "confidence"}.issubset(frame.columns):
            raise ValueError(f"{asset}: missing signal columns")

        if (
            not frame["signal"].isin([-1, 0, 1]).all()
            or not frame["confidence"].between(0, 1).all()
        ):
            raise ValueError(f"{asset}: invalid signal or confidence")

        validated[asset] = frame
        decision_dates = decision_dates.intersection(frame.index)

    if len(decision_dates) < 2:
        raise ValueError("Insufficient common training decisions")

    if not decision_dates.isin(market.index).all():
        raise ValueError("Decisions must coincide with trading sessions")

    # Weekly closes use real last trading sessions, including Thursdays
    # preceding a Friday market holiday. All data are <= train_end.
    weekly_closes = market.loc[
        weekly_dates,
        [f"{a}_close" for a in ASSETS],
    ].astype(float)

    pool = {}

    for asset in SATELLITE_ORDER:
        cols = ["signal", "confidence"]
        if "volatility" in validated[asset]:
            cols.append("volatility")

        pool[asset] = validated[asset].loc[
            decision_dates, cols
        ].copy()

    # Original Satellite rule, including confidence, hysteresis,
    # SPY trend filter and optional volatility input.
    satellite = generate_dynamic_growth_matrix(
        asset_dfs=pool,
        df_features=weekly_closes,
        default_vol=0.10,
        min_confidence=0.35,
        hysteresis_buffer=0.015,
    )

    weights = satellite.reindex(
        columns=list(ASSETS), fill_value=0.0
    ) * (1.0 - core_weight)

    weights["SPY"] += core_weight

    exposure = weights.abs().sum(axis=1)

    if (
        not np.isfinite(weights.to_numpy()).all()
        or (exposure > 1.0 + 1e-10).any()
        or not weights.index.equals(decision_dates)
    ):
        raise ValueError("Invalid Core-Satellite allocations")

    return weights


def run_core_satellite_training(config, root_dir=ROOT_DIR):
    """Development-only comparison of Core-Satellite with SPY and a 60/40 portfolio on matched dates.
    """

    start, end = _guard_training(config)

    if config["strategies"]["C"].get("sentiment_enabled", False):
        raise ValueError("FinBERT historical inputs are not yet certified")

    root = Path(root_dir)

    market = load_dataset(
        root / config["paths"]["processed_data"] / "market_daily.csv",
        strict=True,
    ).loc[:end]

    signals = {}

    for asset in ASSETS:
        path = (
            root / config["paths"]["signals"]
            / f"signals_orchestrator_{asset.lower()}.csv"
        )

        signals[asset] = load_dataset(path, strict=True).loc[:end]

    core = build_core_satellite_targets(
        market, signals, start, end
    )

    dates = core.index

    spy = pd.DataFrame(
        0.0, index=dates[:1], columns=list(ASSETS)
    )
    spy["SPY"] = 1.0

    sixty_forty = pd.DataFrame(
        0.0, index=dates, columns=list(ASSETS)
    )
    sixty_forty["SPY"] = 0.60
    sixty_forty["IEF"] = 0.40

    targets = {
        "Core35_Satellite65": core,
        "SPY_buy_hold": spy,
        "Sixty_Forty": sixty_forty,
    }

    market = market.loc[dates.min():end]

    opens = market[[f"{a}_open" for a in ASSETS]].copy()
    closes = market[[f"{a}_close" for a in ASSETS]].copy()

    opens.columns = list(ASSETS)
    closes.columns = list(ASSETS)

    execution = config["execution"]
    b_settings = config["strategies"]["B"]

    annualization = int(config["evaluation"]["annualization_factor"])
    annual_rf = float(config["evaluation"]["risk_free_rate"])
    daily_rf = (1.0 + annual_rf) ** (1.0 / annualization) - 1.0

    backtests = {}
    records = []

    for name, weights in targets.items():
        bt = simulate_longshort_daily(
            open_prices=opens,
            close_prices=closes,
            target_weights=weights,
            initial_capital=float(execution["initial_capital"]),
            commission_bps=float(execution["commission_bps"]),
            slippage_bps=float(execution["slippage_bps"]),
            annual_borrow_rate=float(b_settings["annual_borrow_rate"]),
            additional_margin_rate=float(
                b_settings["additional_margin_rate"]
            ),
        )

        metrics = calculate_metrics(
            bt["daily_return"], daily_rf, annualization
        )

        records.append({
            "strategy": name,
            **metrics,
            "final_nav": float(bt["nav"].iloc[-1]),
            "trades": int(bt["executed"].sum()),
            "transaction_costs": float(bt["transaction_cost"].sum()),
            "borrow_fees": float(bt["borrow_fee"].sum()),
            "mean_gross_exposure": float(bt["gross_exposure"].mean()),
        })

        backtests[name] = bt

    summary = pd.DataFrame(records).set_index("strategy")

    return {
        "summary": summary,
        "backtests": backtests,
        "target_weights": targets,
    }


if __name__ == "__main__":
    config = load_config(ROOT_DIR / "config.yaml")
    result = run_core_satellite_training(config, ROOT_DIR)

    columns = [
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "trades",
    ]

    print("\nDEVELOPMENT ONLY: CORE-SATELLITE COMPARISON")
    print(result["summary"][columns].round(4).to_string())
