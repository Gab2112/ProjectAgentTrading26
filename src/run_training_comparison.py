"""Matched 2010-2016 training-only backtest for B_v1, B_v2, C_v1, C_v2.

Use existing strategy constructors and the SAME next-open daily engine for all
four portfolios. No retraining, parameter optimization, or validation/OOS data.

Run from project root:
    python3 -m src.run_training_comparison
    python3 -m src.run_training_comparison --save

Or import ``run_training_comparison(config, root_dir)`` for tests.
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
from src.level_c_strategy import build_level_c_v2_training
from src.execution_engine import (
    schedule_next_open_executions,
    simulate_longshort_daily,
)
from src.regime_strategy import build_level_b_training
from src.time_split import get_time_windows, is_oos_enabled
from src.utils import ASSETS

PORTFOLIOS = ("B_v1", "B_v2", "C_v1", "C_v2")
SAMPLE_LABEL = "TRAINING / IN-SAMPLE diagnostic, NOT independent OOS"


def _guard_training(config: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Refuse validation/final test and accidental changes to execution rules."""
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")
    if (config.get("validation", {}).get("development_only") is not True
            or config.get("evaluation", {}).get("evaluate_oos") is not False
            or is_oos_enabled(config)):
        raise ValueError("Training-only lock: validation/OOS not authorized")
    w = get_time_windows(config)
    start, end = w["train_start"], w["train_end"]
    if start is None or end is None or start > end:
        raise ValueError("Missing or invalid training boundaries")

    execution = config.get("execution", {})
    if (execution.get("execution_rule") != "next_available_trading_open"
            or execution.get("signal_frequency") != "weekly"
            or execution.get("cost_basis") != "traded_notional"):
        raise ValueError("Execution policy does not match the shared engine")
    if float(config["strategies"]["B"]["cash_interest_rate"]) != 0.0:
        raise ValueError("Nonzero cash yield is not supported")
    return pd.Timestamp(start), pd.Timestamp(end)


def _validate_weights(
    label: str, weights: pd.DataFrame, decisions: pd.DatetimeIndex,
) -> pd.DataFrame:
    if not isinstance(weights, pd.DataFrame) or list(weights.columns) != list(ASSETS):
        raise ValueError(f"{label}: expected ordered SPY, IEF, GLD, SHY weights")
    if (not isinstance(weights.index, pd.DatetimeIndex)
            or not weights.index.equals(decisions)):
        raise ValueError(f"{label}: mismatched decision calendar")
    values = weights.to_numpy(dtype=float)
    if (not np.isfinite(values).all()
            or (np.abs(values).sum(axis=1) > 1.0 + 1e-10).any()):
        raise ValueError(f"{label}: invalid or leveraged allocations")
    if label in ("B_v2", "C_v1") and (values < -1e-10).any():
        raise ValueError(f"{label}: Long/Flat allocation contains a short")
    return weights.copy()


def run_training_comparison(config: dict, root_dir: str | Path = ".") -> dict:
    """Return matched, net-of-cost B/C backtests; no files are written.

    Sample starts at the first C_v1/C_v2 decision, usually 2010-01-08.
    No trade occurs on that Close: next actual trading Open is used.
    All four NAV series start from the same capital and cash state.
    """
    start, end = _guard_training(config)
    root = Path(root_dir)

    # C_v2 already builds C_v1 and B_v2 on the exact same agent-signal dates.
    c = build_level_c_v2_training(config, root)
    decisions = c["target_weights"].index
    if (not isinstance(decisions, pd.DatetimeIndex) or decisions.empty
            or decisions.hasnans or decisions.has_duplicates
            or not decisions.is_monotonic_increasing
            or decisions.tz is not None
            or not decisions.equals(decisions.normalize())
            or decisions.min() < start or decisions.max() > end):
        raise ValueError("Invalid C decision dates or training boundary")

    b_v1 = build_level_b_training(config, root)["target_weights"]
    if not decisions.isin(b_v1.index).all():
        raise ValueError("B_v1 is missing one or more common decision dates")
    targets = {
        "B_v1": b_v1.loc[decisions],
        "B_v2": c["b_v2_target_weights_same_dates"],
        "C_v1": c["c_v1_target_weights_same_dates"],
        "C_v2": c["target_weights"],
    }
    targets = {name: _validate_weights(name, frame, decisions)
               for name, frame in targets.items()}

    # Only the pre-defined common training range enters the backtester.
    market = load_dataset(
        root / config["paths"]["processed_data"] / "market_daily.csv",
        strict=True,
    ).loc[decisions.min():end]
    if (market.empty or market.index.min() != decisions.min()
            or market.index.max() > end
            or not decisions.isin(market.index).all()):
        raise ValueError("Market data do not match training decisions")
    opens = market[[f"{asset}_open" for asset in ASSETS]].copy()
    closes = market[[f"{asset}_close" for asset in ASSETS]].copy()
    opens.columns = list(ASSETS)
    closes.columns = list(ASSETS)

    exe = config["execution"]
    bconfig = config["strategies"]["B"]
    initial_capital = float(exe["initial_capital"])
    commission = float(exe["commission_bps"])
    slippage = float(exe["slippage_bps"])
    short_rate = float(bconfig["annual_borrow_rate"])
    margin_rate = float(bconfig["additional_margin_rate"])
    rf_annual = float(config["evaluation"]["risk_free_rate"])
    annualization = int(config["evaluation"]["annualization_factor"])
    rf_daily = (1.0 + rf_annual) ** (1.0 / annualization) - 1.0
    if (not np.isfinite([initial_capital, commission, slippage, short_rate,
                         margin_rate, rf_annual]).all()
            or initial_capital <= 0 or commission < 0 or slippage < 0
            or short_rate < 0 or not 0 <= margin_rate <= 1
            or rf_annual <= -1 or annualization <= 1):
        raise ValueError("Invalid simulation or metric configuration")

    backtests, schedules, records = {}, {}, []
    for name in PORTFOLIOS:
        borrow = short_rate if name in ("B_v1", "C_v2") else 0.0
        schedules[name] = schedule_next_open_executions(targets[name], opens.index)
        bt = simulate_longshort_daily(
            open_prices=opens,
            close_prices=closes,
            target_weights=targets[name],
            initial_capital=initial_capital,
            commission_bps=commission,
            slippage_bps=slippage,
            annual_borrow_rate=borrow,
            additional_margin_rate=margin_rate,
        )
        if (bt.empty or bt.index.max() > end
                or not np.isfinite(bt["nav"]).all()
                or (bt["nav"] <= 0).any()
                or (bt["margin_shortfall"] > 1e-8).any()):
            raise RuntimeError(f"{name}: invalid training backtest")
        np.testing.assert_allclose(
            float(np.prod(1.0 + bt["daily_return"])),
            float(bt["nav"].iloc[-1] / initial_capital),
            rtol=1e-8, atol=1e-9,
        )
        performance = calculate_metrics(bt["daily_return"], rf_daily, annualization)
        records.append({
            "strategy": name,
            **performance,
            "final_nav": float(bt["nav"].iloc[-1]),
            "initial_capital": initial_capital,
            "decisions": len(decisions),
            "scheduled_rebalances": len(schedules[name]),
            "nonzero_trades": int(bt["executed"].sum()),
            "total_turnover": float(bt["turnover"].sum()),
            "transaction_costs": float(bt["transaction_cost"].sum()),
            "borrow_fees": float(bt["borrow_fee"].sum()),
            "mean_gross_exposure": float(bt["gross_exposure"].mean()),
            "mean_net_exposure": float(bt["net_exposure"].mean()),
            "mean_short_exposure": float(bt["short_exposure"].mean()),
            # With shorts, cash ledger includes sale proceeds. Not idle cash.
            "mean_cash_fraction_longflat": (
                float((bt["cash"] / bt["nav"]).mean())
                if name in ("B_v2", "C_v1") else float("nan")
            ),
        })
        backtests[name] = bt

    reference = backtests[PORTFOLIOS[0]].index
    if any(not backtests[name].index.equals(reference) for name in PORTFOLIOS[1:]):
        raise RuntimeError("Backtests have different daily calendars")

    summary = pd.DataFrame.from_records(records).set_index("strategy").loc[list(PORTFOLIOS)]
    return {
        "summary": summary,
        "backtests": backtests,
        "target_weights": targets,
        "execution_schedules": schedules,
        "sample": SAMPLE_LABEL,
        "first_decision": decisions.min(),
        "last_decision": decisions.max(),
        "last_accounted_session": reference.max(),
    }


def save_comparison(result: dict, config: dict, root_dir: str | Path = ".") -> Path:
    """Save only NEW ignored diagnostic files, never frozen B artifacts."""
    directory = Path(root_dir) / config["paths"]["results"] / "bc_training_matched"
    directory.mkdir(parents=True, exist_ok=True)
    result["summary"].to_csv(directory / "comparison_metrics.csv")
    for name in PORTFOLIOS:
        sub = directory / name
        sub.mkdir(parents=True, exist_ok=True)
        result["target_weights"][name].to_csv(sub / "target_weights.csv", index_label="Date")
        result["execution_schedules"][name].to_csv(sub / "execution_schedule.csv", index_label="Date")
        result["backtests"][name].to_csv(sub / "backtest_daily.csv", index_label="Date")
    meta = {
        "sample": result["sample"],
        "first_decision": result["first_decision"].date().isoformat(),
        "last_decision": result["last_decision"].date().isoformat(),
        "last_accounted_session": result["last_accounted_session"].date().isoformat(),
        "execution": "next_available_trading_open",
        "no_2017_plus_used": True,
        "caveat": (
            "Orchestrator tuned on the same 2010-2016 data; this is not "
            "an independent evaluation. Macro data vintages not PIT-verified."
        ),
    }
    (directory / "provenance.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return directory


def main() -> None:
    """Command-line entry point: print the comparison and optionally save it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="Write training-only diagnostics")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with (root / "config.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    result = run_training_comparison(config, root)
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
        "max_drawdown": "Max_DD_%_signed",
        "mean_gross_exposure": "Gross_%",
        "mean_cash_fraction_longflat": "Idle_cash_%_longflat",
    })
    print("\n=== B/C MATCHED TRAINING COMPARISON (IN-SAMPLE) ===")
    print(f"Decisions: {result['first_decision'].date()} - "
          f"{result['last_decision'].date()} | last accounted: "
          f"{result['last_accounted_session'].date()}")
    print(view.round(3).to_string())
    print("\nShort portfolio cash is an accounting ledger, not idle cash.")
    print("WARNING: IN-SAMPLE: Orchestrator was tuned on these dates. 2022+ LOCKED.")
    if args.save:
        path = save_comparison(result, config, root)
        print(f"Diagnostics saved to {path}")


if __name__ == "__main__":
    main()
