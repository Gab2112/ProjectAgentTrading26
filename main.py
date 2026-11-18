"""ProjectAgentTrading26 - end-to-end pipeline.

Run from the repository root:

    python main.py

Every setting is read from config.yaml. The steps are executed in order:

1. Data acquisition (optional, ``pipeline.download_data``): download raw data
   from Yahoo Finance and FRED. The repository already contains the raw data,
   so this step is skipped by default and no download is required.
2. Data validation and preparation: adjusted-price checks, aligned daily
   panel, weekly decision calendar.
3. Agent training (optional, ``pipeline.train_agents``): walk-forward
   training of every agent and genetic-algorithm fit of the Orchestrator,
   on the in-sample period only. Saved models are reused otherwise.
4. Validation: selection of the allocation parameters on the validation
   window (the last years of the in-sample period).
5. Backtests and evaluation: agent strategy, benchmarks and baselines,
   per-asset signal tests, performance metrics and equity/drawdown charts,
   on three periods:
   a. in-sample 2005-2020 with the final models, i.e. evaluated on their own
      training data (an upper bound, reported only to measure overfitting);
   b. in-sample 2013-2020 with the walk-forward Orchestrator signals;
   c. out-of-sample, evaluated once.

All outputs are written to data/processed, data/signals and data/results.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import run_signal_backtest, run_weight_backtest
from src.baselines import build_baseline_weights, weights_for_period
from src.data_acquisition import run_data_acquisition
from src.data_transformation import run_data_transformation
from src.evaluator import calculate_metrics
from src.plots import (
    STRATEGY_NAME,
    equity_curve,
    plot_equity_and_drawdown,
    plot_signal_backtests,
    plot_signal_distribution,
    plot_validation_grid,
)
from src.strategy import (
    build_feature_panel,
    generate_signals_with_trained_models,
    load_in_sample_signals,
    make_aggregator,
    tune_allocation,
)
from src.time_split import get_time_windows
from src.utils import ASSETS, load_config

ROOT = Path(__file__).resolve().parent
PERIOD_LABELS = {
    "in_sample_full": "In-sample, final fit",
    "in_sample": "In-sample, walk-forward",
    "oos": "Out-of-sample",
}
PERCENT_COLUMNS = (
    "cumulative_return", "cagr", "annualized_volatility", "max_drawdown",
    "hit_rate", "mean_return_market_down_days",
)


def banner(text: str) -> None:
    print("\n" + "=" * 100 + f"\n  {text}\n" + "=" * 100)


def format_metrics(table: pd.DataFrame) -> pd.DataFrame:
    """Readable version of a metrics table (percentages for return-type columns)."""
    shown = table.copy()
    for column in PERCENT_COLUMNS:
        if column in shown:
            factor = 10_000.0 if column == "mean_return_market_down_days" else 100.0
            shown[column] = shown[column] * factor
    shown = shown.rename(columns={
        "cumulative_return": "Cum. ret. %", "cagr": "CAGR %", "annualized_volatility": "Vol. %",
        "sharpe_ratio": "Sharpe", "sortino_ratio": "Sortino", "max_drawdown": "Max DD %",
        "calmar_ratio": "Calmar", "profit_factor": "Profit f.", "hit_rate": "Hit rate %",
        "beta_to_market": "Beta SPY", "correlation_to_market": "Corr. SPY",
        "mean_return_market_down_days": "Ret. SPY-down (bp)", "turnover": "Turnover",
        "transaction_costs": "Costs $", "observations": "Days",
    })
    return shown.round(2)


def models_available(config: dict) -> bool:
    models = ROOT / config["paths"]["models"]
    signals = ROOT / config["paths"]["signals"]
    return all(
        (models / f"orchestrator_params_{a.lower()}.json").is_file()
        and (signals / f"signals_orchestrator_{a.lower()}.csv").is_file()
        for a in ASSETS
    )


def evaluate_period(
    period: str,
    config: dict,
    market_daily: pd.DataFrame,
    signals: pd.DataFrame,
    confidences: pd.DataFrame,
    allocation: dict,
    baseline_weights: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp | None,
    results_dir: Path,
) -> None:
    """Backtest the strategy, the benchmarks and the per-asset signals on one period."""
    evaluation = config["evaluation"]
    capital = float(config["execution"]["initial_capital"])
    label = PERIOD_LABELS[period]
    sig = signals.loc[(signals.index >= start) & ((signals.index <= end) if end is not None else True)]
    conf = confidences.loc[sig.index]

    strategy_weights = make_aggregator(config, allocation).target_weights(sig, conf)
    strategy_weights.to_csv(results_dir / f"target_weights_{period}.csv")

    navs = {STRATEGY_NAME: run_weight_backtest(strategy_weights, market_daily, config, start, end)}
    for name, weights in baseline_weights.items():
        navs[name] = run_weight_backtest(weights_for_period(name, weights, start), market_daily, config, start, end)

    market = navs["SPY Buy & Hold"]["daily_return"]
    rows = {}
    for name, nav in navs.items():
        metrics = calculate_metrics(
            nav["daily_return"], evaluation["risk_free_rate"], evaluation["annualization_factor"], market
        )
        metrics["turnover"] = float(nav["turnover"].sum())
        metrics["transaction_costs"] = float(nav["transaction_cost"].sum())
        rows[name] = metrics
    table = pd.DataFrame(rows).T
    table.index.name = "strategy"
    table.to_csv(results_dir / f"metrics_{period}.csv")
    pd.DataFrame({name: nav["nav"] for name, nav in navs.items()}).to_csv(results_dir / f"nav_{period}.csv")

    first_day = navs[STRATEGY_NAME].index[0]
    print(f"\n{label}: {first_day.date()} -> {navs[STRATEGY_NAME].index[-1].date()} "
          f"({len(sig)} weekly decisions)")
    print(format_metrics(table).to_string())

    curves = {name: equity_curve(nav, capital, first_day) for name, nav in navs.items()}
    plot_equity_and_drawdown(
        {k: v for k, v in curves.items() if k in (STRATEGY_NAME, "SPY Buy & Hold", "Equal Weight Buy & Hold", "60/40 SPY-IEF (weekly)")},
        f"{label} (vs benchmarks)",
        results_dir / "plots" / f"equity_{period}_benchmarks.png",
    )
    plot_equity_and_drawdown(curves, f"{label} (all strategies)", results_dir / "plots" / f"equity_{period}_all.png")
    plot_signal_distribution(sig, label, results_dir / "plots" / f"signal_distribution_{period}.png")

    # Per-asset signal backtests: +1 long, 0 flat, -1 short, full NAV on the asset.
    signal_rows, signal_curves = {}, {}
    for asset in ASSETS:
        signal_nav = run_signal_backtest(sig[asset], asset, market_daily, config, start, end)
        hold = sig[asset].iloc[:1] * 0 + 1
        hold_nav = run_signal_backtest(hold, asset, market_daily, config, start, end)
        for name, nav in ((f"{asset} signal", signal_nav), (f"{asset} Buy & Hold", hold_nav)):
            signal_rows[name] = calculate_metrics(
                nav["daily_return"], evaluation["risk_free_rate"], evaluation["annualization_factor"]
            )
        signal_curves[asset] = {
            f"{asset} signal": equity_curve(signal_nav, capital, first_day),
            f"{asset} Buy & Hold": equity_curve(hold_nav, capital, first_day),
        }
    signal_table = pd.DataFrame(signal_rows).T
    signal_table.index.name = "strategy"
    signal_table.to_csv(results_dir / f"signal_backtests_{period}.csv")
    plot_signal_backtests(signal_curves, label, results_dir / "plots" / f"signal_backtests_{period}.png")
    print(f"\nPer-asset signal backtests ({label}):")
    print(format_metrics(signal_table).to_string())


def main() -> None:
    config = load_config(ROOT / "config.yaml")
    seed = int(config["project"]["random_seed"])
    random.seed(seed)
    np.random.seed(seed)
    windows = get_time_windows(config)
    results_dir = ROOT / config["paths"]["results"]
    (results_dir / "plots").mkdir(parents=True, exist_ok=True)

    # 1. Data acquisition (optional).
    banner("STEP 1 - DATA ACQUISITION")
    if config["pipeline"]["download_data"]:
        run_data_acquisition(config, ROOT)
    else:
        print("Skipped: the raw data stored in data/raw are used (pipeline.download_data: false).")

    # 2. Data validation and preparation.
    banner("STEP 2 - DATA VALIDATION AND PREPARATION")
    run_data_transformation(config, ROOT)
    processed = ROOT / config["paths"]["processed_data"]
    market_daily = pd.read_csv(processed / "market_daily.csv", index_col="Date", parse_dates=["Date"])
    decisions = pd.read_csv(processed / "weekly_decision_dates.csv", index_col="Date", parse_dates=["Date"]).index

    # 3. Agent training on the in-sample period (optional).
    banner("STEP 3 - AGENT TRAINING (IN-SAMPLE ONLY)")
    if config["pipeline"]["train_agents"] or not models_available(config):
        from src.agents.train import run_all_assets
        run_all_assets(config, ROOT)
    else:
        print("Skipped: the trained models in src/agents/config and the in-sample signals in data/signals are used.")

    # 4. Validation of the allocation parameters.
    banner("STEP 4 - ALLOCATION PARAMETERS SELECTED ON THE VALIDATION WINDOW")
    is_signals, is_conf = load_in_sample_signals(config, ROOT)
    allocation, grid = tune_allocation(config, is_signals, is_conf, market_daily)
    validation_dir = results_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(validation_dir / "allocation_grid.csv", index=False)
    with (validation_dir / "selected_allocation.json").open("w", encoding="utf-8") as fh:
        json.dump(allocation, fh, indent=4)
    plot_validation_grid(grid, config["allocation"]["selection_metric"], results_dir / "plots" / "validation_grid.png")
    print(f"Validation window: {allocation['validation_start']} -> {allocation['validation_end']}")
    print(f"Selected: min_confidence={allocation['min_confidence']:.2f}, "
          f"max_weight_per_asset={allocation['max_weight_per_asset']:.2f}, allow_short={allocation['allow_short']} "
          f"(validation {allocation['selection_metric']} = {allocation['validation_value']:.2f})")

    # 5. Backtests and evaluation.
    banner("STEP 5 - BACKTESTS AND EVALUATION")
    baseline_weights = build_baseline_weights(
        config, ROOT, market_daily, decisions[decisions >= windows["train_start"]]
    )
    feature_panel = build_feature_panel(market_daily, decisions, ROOT)
    all_signals, all_conf = generate_signals_with_trained_models(feature_panel, config, ROOT)

    # 5a. Whole in-sample period with the final models. Every agent and the
    # Orchestrator are evaluated on the data they were fitted on, so these
    # figures are an optimistic upper bound and are not evidence of skill;
    # they are reported to measure the gap with the walk-forward results.
    print("\nNote: the 'final fit' figures apply the final models to their own training period "
          "(in-sample fit) and are therefore optimistic.")
    evaluate_period("in_sample_full", config, market_daily, all_signals, all_conf, allocation,
                    baseline_weights, windows["train_start"], windows["train_end"], results_dir)

    # 5b. In-sample period covered by the Orchestrator walk-forward.
    is_start = is_signals.index[0]
    print(f"\nNote: the walk-forward in-sample comparison starts on {is_start.date()}, the first week of the "
          f"Orchestrator walk-forward; earlier IS data are used to fit the first walk-forward models.")
    evaluate_period("in_sample", config, market_daily, is_signals, is_conf, allocation,
                    baseline_weights, is_start, windows["train_end"], results_dir)

    # 5c. Single out-of-sample evaluation.
    oos_mask = all_signals.index >= windows["oos_start"]
    if windows["oos_end"] is not None:
        oos_mask &= all_signals.index <= windows["oos_end"]
    oos_signals, oos_conf = all_signals.loc[oos_mask], all_conf.loc[oos_mask]
    pd.concat({"signal": oos_signals, "confidence": oos_conf}, axis=1).to_csv(
        ROOT / config["paths"]["signals"] / "oos_signals_orchestrator.csv"
    )
    oos_start = oos_signals.index[0]
    evaluate_period("oos", config, market_daily, oos_signals, oos_conf, allocation,
                    baseline_weights, oos_start, windows["oos_end"], results_dir)
    print(f"\nAll results were written to {results_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
