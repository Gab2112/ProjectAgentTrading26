"""Training-only what-if sensitivity: increase exposure in frozen Level C signals.

This experiment does NOT retrain models or select an improved strategy. Both
C_v1 and C_v2 use their PRE-volatility-cap raw weights. Each scenario scales
only pre-existing positions, imposes 100% gross exposure, then applies the
existing 126-session covariance-based volatility cap before next-open trading.

Run from repository root:
    python3 -m src.run_c_exposure_sensitivity
    python3 -m src.run_c_exposure_sensitivity --save

The 2017-2021 validation and 2022+ holdout are NOT evaluated here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data_loader import load_dataset
from src.evaluator import calculate_metrics
from src.execution_engine import simulate_longshort_daily
from src.level_c_strategy import build_level_c_training, build_level_c_v2_training
from src.portfolio import apply_volatility_cap_panel
from src.run_training_comparison import _guard_training
from src.utils import ASSETS

# Pre-declared diagnostic grid, not a train-optimized parameter search.
EXPOSURE_FACTORS = (1.0, 1.25, 1.5, 2.0)
VOLATILITY_CAPS = (0.08, 0.12)
SAMPLE = "2010-2016 TRAINING WHAT-IF; IN-SAMPLE ORCHESTRATOR; NOT INDEPENDENT VALIDATION"


def make_scenario_weights(
    raw_weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
    *,
    exposure_factor: float,
    volatility_cap: float,
    train_end: pd.Timestamp,
    window: int = 126,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Scale current directions, no new signals, no leverage, then cap risk.

    This is an exposure sensitivity, NOT a new method of selecting trades.
    If all four raw signals are zero, a scenario necessarily stays 100% cash.
    """
    if not isinstance(raw_weights, pd.DataFrame) or raw_weights.empty:
        raise ValueError("raw_weights must be a nonempty DataFrame")
    if not isinstance(daily_returns, pd.DataFrame) or daily_returns.empty:
        raise ValueError("daily_returns must be a nonempty DataFrame")
    if list(raw_weights.columns) != list(ASSETS) or list(daily_returns.columns) != list(ASSETS):
        raise ValueError("Expected asset order SPY, IEF, GLD, SHY")
    if any((not isinstance(x.index, pd.DatetimeIndex) or x.index.hasnans
            or x.index.has_duplicates or x.index.tz is not None
            or not x.index.is_monotonic_increasing or not x.index.equals(x.index.normalize()))
           for x in (raw_weights, daily_returns)):
        raise ValueError("Expected unique, ordered actual trading-session dates")
    cutoff = pd.Timestamp(train_end)
    if (pd.isna(cutoff) or cutoff.tzinfo is not None
            or raw_weights.index.max() > cutoff or daily_returns.index.max() > cutoff):
        raise ValueError("Only past and current TRAINING data are allowed")
    if not raw_weights.index.isin(daily_returns.index).all():
        raise ValueError("Missing historical return on a decision date")
    if (not np.isfinite(raw_weights.to_numpy(dtype=float)).all()
            or not np.isfinite(daily_returns.to_numpy(dtype=float)).all()):
        raise ValueError("Non-finite weights/returns")
    if isinstance(exposure_factor, bool) or not np.isfinite(exposure_factor) or exposure_factor < 1.0:
        raise ValueError("Exposure factor must be >= 1")
    if isinstance(volatility_cap, bool) or not np.isfinite(volatility_cap) or volatility_cap <= 0:
        raise ValueError("Invalid volatility cap")
    if (raw_weights.abs().sum(axis=1) > 1.0 + 1e-10).any():
        raise ValueError("Original raw gross exposures must be <= 1")

    amplified = raw_weights.astype(float) * float(exposure_factor)
    gross = amplified.abs().sum(axis=1)
    # Fixed no-leverage rule: shrink rows above 100% gross, preserve signs.
    gross_factor = (1.0 / gross.clip(lower=1.0)).astype(float)
    bounded = amplified.mul(gross_factor, axis=0)
    targets, risk = apply_volatility_cap_panel(
        bounded, daily_returns, window=window,
        target_volatility=float(volatility_cap),
    )
    if (not targets.index.equals(raw_weights.index)
            or not np.isfinite(targets.to_numpy(dtype=float)).all()
            or (targets.abs().sum(axis=1) > 1.0 + 1e-10).any()
            or (risk["vol_after"] > volatility_cap + 1e-8).any()
            or ((targets * raw_weights) < -1e-10).any().any()):
        raise RuntimeError("Risk or direction invariant failed")
    return targets, risk, (gross > 1.0 + 1e-10).rename("gross_limit_active")


def run_c_exposure_sensitivity(config: dict, root_dir: str | Path = ".") -> dict:
    """Run pre-declared exposure sensitivity on the matched C training sample."""
    _, end = _guard_training(config)
    root = Path(root_dir)
    c1 = build_level_c_training(config, root)
    c2 = build_level_c_v2_training(config, root)
    strategies = {"C_v1": c1, "C_v2": c2}
    decisions = c1["target_weights"].index
    if not decisions.equals(c2["target_weights"].index) or decisions.max() > end:
        raise ValueError("C variants must share training decision dates")

    market = load_dataset(
        root / config["paths"]["processed_data"] / "market_daily.csv", strict=True,
    ).loc[:end]
    daily_closes = market[[f"{a}_close" for a in ASSETS]].astype(float).copy()
    daily_closes.columns = list(ASSETS)
    returns = daily_closes.pct_change(fill_method=None).iloc[1:]
    live_market = market.loc[decisions.min():end]
    opens = live_market[[f"{a}_open" for a in ASSETS]].astype(float).copy()
    closes = live_market[[f"{a}_close" for a in ASSETS]].astype(float).copy()
    opens.columns = list(ASSETS)
    closes.columns = list(ASSETS)
    if (live_market.empty or live_market.index.min() != decisions.min()
            or not decisions.isin(live_market.index).all()
            or returns.index.max() > end):
        raise ValueError("Missing market coverage or training-period violation")

    exe = config["execution"]
    bconf = config["strategies"]["B"]
    capital = float(exe["initial_capital"])
    commission = float(exe["commission_bps"])
    slippage = float(exe["slippage_bps"])
    borrow = float(bconf["annual_borrow_rate"])
    margin = float(bconf["additional_margin_rate"])
    annual = int(config["evaluation"]["annualization_factor"])
    rf_ann = float(config["evaluation"]["risk_free_rate"])
    rf_daily = (1.0 + rf_ann) ** (1.0 / annual) - 1.0

    records, runs = [], {}
    for name, built in strategies.items():
        raw = built["raw_weights"]
        if not raw.index.equals(decisions):
            raise ValueError("The raw and final decision indexes must match")
        for cap in VOLATILITY_CAPS:
            for factor in EXPOSURE_FACTORS:
                weights, risk, gross_limited = make_scenario_weights(
                    raw, returns, exposure_factor=factor,
                    volatility_cap=cap, train_end=end,
                )
                # Replay of k=1, cap=8 must reproduce the current frozen strategy.
                if factor == 1.0 and cap == 0.08:
                    np.testing.assert_allclose(
                        weights.to_numpy(dtype=float),
                        built["target_weights"].to_numpy(dtype=float),
                        rtol=1e-11, atol=1e-12,
                        err_msg=f"{name}: baseline mismatch; do not trust scenarios",
                    )
                bt = simulate_longshort_daily(
                    open_prices=opens, close_prices=closes,
                    target_weights=weights, initial_capital=capital,
                    commission_bps=commission, slippage_bps=slippage,
                    annual_borrow_rate=borrow if name == "C_v2" else 0.0,
                    additional_margin_rate=margin,
                )
                if (not bt.index.equals(live_market.index)
                        or (bt["margin_shortfall"] > 1e-8).any()
                        or not np.isfinite(bt["nav"]).all()
                        or (bt["nav"] <= 0).any()):
                    raise RuntimeError(f"{name} {factor} {cap}: invalid backtest")
                np.testing.assert_allclose(
                    (1.0 + bt["daily_return"]).prod(),
                    bt["nav"].iloc[-1] / capital, rtol=1e-8,
                )
                m = calculate_metrics(bt["daily_return"], rf_daily, annual)
                scenario = f"{name}_x{factor:g}_vol{int(cap * 100)}"
                records.append({
                    "scenario": scenario,
                    "strategy": name,
                    "exposure_factor": factor,
                    "volatility_cap": cap,
                    "final_nav": float(bt["nav"].iloc[-1]),
                    "cagr": m["cagr"],
                    "annualized_volatility": m["annualized_volatility"],
                    "sharpe_ratio": m["sharpe_ratio"],
                    "max_drawdown": m["max_drawdown"],
                    "mean_target_gross": float(weights.abs().sum(axis=1).mean()),
                    "mean_realized_gross": float(bt["gross_exposure"].mean()),
                    "idle_cash_longflat": (float((bt["cash"] / bt["nav"]).mean())
                                           if name == "C_v1" else float("nan")),
                    "total_trading_costs": float(bt["transaction_cost"].sum()),
                    "total_borrow_fees": float(bt["borrow_fee"].sum()),
                    "zero_exposure_weeks": int(weights.abs().sum(axis=1).eq(0).sum()),
                    "gross_clipped_weeks": int(gross_limited.sum()),
                    "vol_capped_weeks": int((risk["scaling_factor"] < 1 - 1e-10).sum()),
                })
                runs[scenario] = {"target_weights": weights, "risk": risk, "backtest": bt}

    summary = pd.DataFrame.from_records(records).set_index("scenario")
    return {"summary": summary, "runs": runs,
            "first_decision": decisions.min(), "last_decision": decisions.max(),
            "last_accounted_session": live_market.index.max(), "sample": SAMPLE}


def save_sensitivity(result: dict, config: dict, root_dir: str | Path = ".") -> Path:
    """Write only ignored new diagnostics, never frozen A/B/C artifacts."""
    dest = Path(root_dir) / config["paths"]["results"] / "c_exposure_sensitivity_training"
    dest.mkdir(parents=True, exist_ok=True)
    result["summary"].to_csv(dest / "scenario_comparison.csv")
    meta = {"sample": result["sample"],
            "start": result["first_decision"].strftime("%Y-%m-%d"),
            "end": result["last_accounted_session"].strftime("%Y-%m-%d"),
            "exposure_factors": list(EXPOSURE_FACTORS),
            "volatility_caps": list(VOLATILITY_CAPS),
            "max_gross_exposure": 1.0,
            "scenario_type": "IN-SAMPLE exploratory sensitivity; NOT a selection rule",
            "validation_used": False, "oos_used": False}
    (dest / "provenance.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return dest


def main() -> None:
    """Command-line entry point: print the sensitivity table and optionally save it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="Save diagnostic metrics (ignored folder)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with (root / "config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    result = run_c_exposure_sensitivity(config, root)
    shown = result["summary"][[
        "strategy", "exposure_factor", "volatility_cap", "final_nav",
        "cagr", "annualized_volatility", "sharpe_ratio", "max_drawdown",
        "mean_realized_gross", "idle_cash_longflat",
        "total_trading_costs", "total_borrow_fees", "gross_clipped_weeks", "vol_capped_weeks",
    ]].copy()
    for col in ("cagr", "annualized_volatility", "max_drawdown",
                "mean_realized_gross", "idle_cash_longflat", "volatility_cap"):
        shown[col] *= 100.0
    shown.rename(columns={
        "cagr": "CAGR_%", "annualized_volatility": "RealizedVol_%",
        "max_drawdown": "MaxDD_%", "mean_realized_gross": "Gross_%",
        "idle_cash_longflat": "IdleCash_%_C1", "volatility_cap": "EstVolCap_%",
    }, inplace=True)
    print("\n=== C EXPOSURE SENSITIVITY (TRAINING / IN-SAMPLE) ===")
    print(f"Decisions {result['first_decision'].date()} - {result['last_decision'].date()}")
    print(shown.round(3).to_string())
    print("\nNote: C_v2 cash ledger includes short-sale proceeds; idle cash omitted.")
    print("IN-SAMPLE exploration only. No refitting, no 2017+ validation, no 2022+ OOS.")
    if args.save:
        print("Saved diagnostic summary:", save_sensitivity(result, config, root))


if __name__ == "__main__":
    main()
