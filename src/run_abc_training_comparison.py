"""Training-only comparison of A1, A2, B_v1, B_v2, C_v1 and C_v2.

Reuses the verified B/C comparison, existing A1/A2 target builders and the
same daily next-open simulator. Nothing is retrained or tuned; no 2017+ dates
are used for decisions, risk windows or backtesting.

    python3 -m src.run_abc_training_comparison
    python3 -m src.run_abc_training_comparison --save
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
from src.execution_engine import (
    schedule_next_open_executions, simulate_longshort_daily,
)
from src.portfolio import build_a1_weights, build_a2_weights, _validate_universe
from src.utils import ASSETS
from src.run_training_comparison import _guard_training, run_training_comparison

A2_NAME = "A2_volatility_control"
SAMPLE = "TRAINING / IN-SAMPLE (2010-2016); NOT independent validation"


def _validated_a_targets(name: str, frame: pd.DataFrame,
                         expected_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """A1/A2 remain long-only, fully invested; buy-and-hold has one decision."""
    if (not isinstance(frame, pd.DataFrame)
            or list(frame.columns) != list(ASSETS)
            or not frame.index.equals(expected_dates)):
        raise ValueError(f"{name}: target dates or ordered ETF columns mismatch")
    values = frame.to_numpy(dtype=float)
    if (not np.isfinite(values).all()
            or (values < -1e-7).any()
            or (values > 1 + 1e-7).any()
            or not np.allclose(values.sum(axis=1), 1.0, atol=1e-7, rtol=0)):
        raise ValueError(f"{name}: A1/A2 must be long-only and fully invested")
    # A2 can contain insignificant negative solver noise (<= 1e-7).
    if (values < 0).any():
        values = np.maximum(values, 0.0)
        values = values / values.sum(axis=1, keepdims=True)
        return pd.DataFrame(values, index=frame.index, columns=frame.columns)
    return frame.copy()


def run_abc_training_comparison(config: dict, root_dir: str | Path = ".") -> dict:
    """Evaluate 4 A1 benchmarks + A2 + 4 B/C variants with matched restart.

    All nine portfolios start from cash at the first C decision close, execute
    no earlier than next trading open and share precisely the same daily NAV
    calendar. The 2005-2009 observations are A2 indicator warm-up, not trades.
    """
    _, train_end = _guard_training(config)
    for level in ("A1", "A2"):
        if config.get("strategies", {}).get(level, {}).get("enabled") is not True:
            raise ValueError(f"{level}: not enabled in config")
    if list(_validate_universe(config)) != list(ASSETS):
        # Current YAML already orders SPY/IEF/GLD/SHY; never silently reorder.
        raise ValueError("Universe must be SPY, IEF, GLD, SHY in the expected order")
    root = Path(root_dir)

    bc = run_training_comparison(config, root)
    decisions = bc["target_weights"]["C_v2"].index
    if decisions.empty or decisions.max() > train_end:
        raise ValueError("A/B/C common decisions are outside training")
    first = decisions.min()

    # Build A2 indicators with historical warm-up, never using validation/OOS.
    market_history = load_dataset(
        root / config["paths"]["processed_data"] / "market_daily.csv",
        strict=True,
    ).loc[:train_end]
    if (market_history.empty or not decisions.isin(market_history.index).all()
            or market_history.index.has_duplicates
            or not market_history.index.is_monotonic_increasing):
        raise ValueError("A1/A2 market calendar does not match B/C decisions")

    tickers = list(ASSETS)
    a1 = build_a1_weights(config, decisions, tickers)
    a2, a2_diagnostics = build_a2_weights(market_history, decisions, config, tickers)
    targets: dict[str, pd.DataFrame] = {}
    for name, frame in a1.items():
        if name not in config.get("benchmarks", {}):
            raise ValueError(f"Unexpected benchmark {name}")
        mode = config["benchmarks"][name]["rebalance"]
        expected = decisions if mode == "weekly" else decisions[:1]
        targets[f"A1_{name}"] = _validated_a_targets(
            f"A1_{name}", frame, expected,
        )
    targets[A2_NAME] = _validated_a_targets(A2_NAME, a2, decisions)

    market = market_history.loc[first:train_end]
    if (not market.index.equals(bc["backtests"]["C_v2"].index)
            or market.index.min() != first):
        raise ValueError("Daily A/B/C calendars do not match")
    opens = market[[f"{a}_open" for a in ASSETS]].copy()
    closes = market[[f"{a}_close" for a in ASSETS]].copy()
    opens.columns = tickers
    closes.columns = tickers

    exe = config["execution"]
    initial = float(exe["initial_capital"])
    commission = float(exe["commission_bps"])
    slippage = float(exe["slippage_bps"])
    margin = float(config["strategies"]["B"]["additional_margin_rate"])
    factor = int(config["evaluation"]["annualization_factor"])
    rf_year = float(config["evaluation"]["risk_free_rate"])
    rf_daily = (1.0 + rf_year) ** (1.0 / factor) - 1.0

    additional_backtests = {}
    additional_schedules = {}
    records = []
    for name, frame in targets.items():
        schedule = schedule_next_open_executions(frame, opens.index)
        bt = simulate_longshort_daily(
            open_prices=opens, close_prices=closes, target_weights=frame,
            initial_capital=initial, commission_bps=commission,
            slippage_bps=slippage, annual_borrow_rate=0.0,
            additional_margin_rate=margin,
        )
        if (not bt.index.equals(market.index)
                or not np.isfinite(bt["nav"]).all()
                or (bt["nav"] <= 0).any()
                or (bt["margin_shortfall"] > 1e-8).any()):
            raise RuntimeError(f"{name}: invalid daily accounting")
        np.testing.assert_allclose(
            float((1.0 + bt["daily_return"]).prod()),
            float(bt["nav"].iloc[-1] / initial), rtol=1e-8, atol=1e-9,
        )
        metrics = calculate_metrics(bt["daily_return"], rf_daily, factor)
        records.append({
            "strategy": name, **metrics,
            "final_nav": float(bt["nav"].iloc[-1]),
            "initial_capital": initial,
            "decisions": len(frame),
            "scheduled_rebalances": len(schedule),
            "nonzero_trades": int(bt["executed"].sum()),
            "total_turnover": float(bt["turnover"].sum()),
            "transaction_costs": float(bt["transaction_cost"].sum()),
            "borrow_fees": float(bt["borrow_fee"].sum()),
            "mean_gross_exposure": float(bt["gross_exposure"].mean()),
            "mean_net_exposure": float(bt["net_exposure"].mean()),
            "mean_short_exposure": float(bt["short_exposure"].mean()),
            "mean_cash_fraction_longflat": float((bt["cash"] / bt["nav"]).mean()),
        })
        additional_backtests[name] = bt
        additional_schedules[name] = schedule

    # Keep previously tested B/C results unchanged, including borrow accounting.
    bc_names = list(bc["summary"].index)
    names = list(targets) + bc_names
    summary = pd.concat([
        pd.DataFrame.from_records(records).set_index("strategy"),
        bc["summary"],
    ], axis=0).loc[names]
    backtests = {**additional_backtests, **bc["backtests"]}
    schedules = {**additional_schedules, **bc["execution_schedules"]}
    all_targets = {**targets, **bc["target_weights"]}
    if (len(names) != len(set(names))
            or any(not backtests[k].index.equals(market.index) for k in names)):
        raise RuntimeError("Duplicate strategy or unmatched daily calendars")

    return {
        "summary": summary,
        "backtests": backtests,
        "target_weights": all_targets,
        "execution_schedules": schedules,
        "a2_diagnostics": a2_diagnostics,
        "sample": SAMPLE,
        "first_decision": first,
        "last_decision": decisions.max(),
        "last_accounted_session": market.index.max(),
    }


def save_abc_comparison(result: dict, config: dict,
                        root_dir: str | Path = ".") -> Path:
    """Write NEW ignored A/B/C diagnostics without touching frozen outputs."""
    path = Path(root_dir) / config["paths"]["results"] / "abc_training_matched"
    path.mkdir(parents=True, exist_ok=True)
    result["summary"].to_csv(path / "comparison_metrics.csv")
    result["a2_diagnostics"].to_csv(path / "a2_risk_diagnostics.csv", index_label="Date")
    for name in result["summary"].index:
        folder = path / name
        folder.mkdir(exist_ok=True)
        result["target_weights"][name].to_csv(folder / "target_weights.csv", index_label="Date")
        result["execution_schedules"][name].to_csv(folder / "execution_schedule.csv", index_label="Date")
        result["backtests"][name].to_csv(folder / "backtest_daily.csv", index_label="Date")
    metadata = {
        "sample": result["sample"],
        "first_decision": result["first_decision"].date().isoformat(),
        "last_decision": result["last_decision"].date().isoformat(),
        "last_accounted_session": result["last_accounted_session"].date().isoformat(),
        "reference_calendar": "C_v2 weekly decisions and common daily trading dates",
        "buy_and_hold": "One initial order; no recurring rebalance",
        "accounting": "Same next-open daily simulator, costs and restart for all portfolios",
        "holdout_used": False,
        "macro_point_in_time_verified": False,
        "independent_oos": False,
        "interpretation": "Orchestrator tuned on 2010-2016; apparent C performance is in-sample",
    }
    (path / "provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return path


def main() -> None:
    """Command-line entry point: print the comparison and optionally save it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="Write A/B/C training-only outputs")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with (root / "config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    result = run_abc_training_comparison(config, root)
    view = result["summary"][[
        "final_nav", "cagr", "annualized_volatility", "sharpe_ratio",
        "max_drawdown", "total_turnover", "transaction_costs", "borrow_fees",
        "mean_gross_exposure", "mean_cash_fraction_longflat",
    ]].copy()
    for key in ("cagr", "annualized_volatility", "max_drawdown",
                "mean_gross_exposure", "mean_cash_fraction_longflat"):
        view[key] *= 100.0
    view = view.rename(columns={
        "cagr": "CAGR_%", "annualized_volatility": "Vol_%",
        "max_drawdown": "Max_DD_%_signed", "mean_gross_exposure": "Gross_%",
        "mean_cash_fraction_longflat": "Idle_cash_%_longflat",
    })
    print("\n=== A/B/C MATCHED TRAINING COMPARISON (IN-SAMPLE) ===")
    print(f"Decisions: {result['first_decision'].date()} - "
          f"{result['last_decision'].date()} | last accounted: "
          f"{result['last_accounted_session'].date()}")
    print(view.round(3).to_string())
    print("\nShort portfolios: accounting cash is NOT idle cash (shown as NaN).")
    print("WARNING: C scores are IN-SAMPLE. 2017-2021 not independently evaluated; 2022+ LOCKED.")
    if args.save:
        path = save_abc_comparison(result, config, root)
        print("Diagnostics saved to:", path)


if __name__ == "__main__":
    main()
