"""Reproducible training-only execution diagnostic for the four agent outputs.

Run from the repository root:
    python3 -m src.agents.run_agent_execution_diagnostic

Reads pre-existing signal CSVs. It NEVER trains models, changes strategy
parameters, or accesses validation/OOS market observations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.agents.agent_execution_bridge import run_agent_training_execution_diagnostic
from src.utils import ASSETS


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Run the execution diagnostic on the stored orchestrator signals and print the results.
    """
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    signal_directory = ROOT / config["paths"]["signals"]
    market_file = ROOT / config["paths"]["processed_data"] / "market_daily.csv"
    training_start = pd.Timestamp(config["validation"]["train_start"])
    oof_lower_bound = pd.Timestamp(year=training_start.year + 5, month=1, day=1)

    signals = {}
    for asset in ASSETS:
        path = signal_directory / f"signals_orchestrator_{asset.lower()}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Segnali orchestratore mancanti: {path}. "
                "Servono CSV rigenerati dalla pipeline con predizioni RF/SVM OOF."
            )
        frame = pd.read_csv(path, index_col=0)
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise"))
        if frame.empty or frame.index.min() < oof_lower_bound:
            raise ValueError(
                f"{path.name}: storico precedente alla prima finestra OOF "
                f"({oof_lower_bound:%Y-%m-%d}) o file vuoto. "
                "Verificare che i CSV siano stati rigenerati dopo la II.5."
            )
        signals[asset] = frame

    if not market_file.is_file():
        raise FileNotFoundError(f"Prezzi giornalieri assenti: {market_file}")
    daily_prices = pd.read_csv(market_file, index_col="Date", parse_dates=["Date"])

    result = run_agent_training_execution_diagnostic(
        config=config,
        market_daily=daily_prices,
        signals_by_asset=signals,
    )
    backtest = result["daily_backtest"]
    daily_returns = backtest["daily_return"].astype(float)
    initial_capital = float(config["execution"]["initial_capital"])
    final_nav = float(backtest["nav"].iloc[-1])
    n_sessions = len(backtest)
    vol_daily = float(daily_returns.std(ddof=1)) if n_sessions > 1 else 0.0

    wealth = np.concatenate(([initial_capital], backtest["nav"].to_numpy(dtype=float)))
    peaks = np.maximum.accumulate(wealth)
    maximum_drawdown = float((1.0 - wealth / peaks).max())

    summary = {
        "sample": result["sample"],
        "first_decision": str(result["target_weights"].index.min().date()),
        "last_decision": str(result["target_weights"].index.max().date()),
        "first_execution": str(result["execution_schedule"].index.min().date()),
        "last_execution": str(result["execution_schedule"].index.max().date()),
        "last_accounted_session": str(backtest.index.max().date()),
        "n_daily_sessions": n_sessions,
        "n_decisions": len(result["target_weights"]),
        "n_scheduled_rebalances": len(result["execution_schedule"]),
        "n_nonzero_trades": int(backtest["executed"].sum()),
        "initial_capital": initial_capital,
        "final_nav": final_nav,
        "cumulative_net_return": float(final_nav / initial_capital - 1.0),
        "annualized_net_return_252": float((final_nav / initial_capital) ** (252 / n_sessions) - 1.0),
        "annualized_volatility_252": float(vol_daily * np.sqrt(252)),
        "sharpe_ratio_252_rf0": (
            float(np.sqrt(252) * daily_returns.mean() / vol_daily)
            if vol_daily > 1e-12 else 0.0
        ),
        "maximum_drawdown": maximum_drawdown,
        "total_turnover": float(backtest["turnover"].sum()),
        "total_transaction_cost": float(backtest["transaction_cost"].sum()),
        "commission_bps": float(config["execution"]["commission_bps"]),
        "slippage_bps": float(config["execution"]["slippage_bps"]),
        "important_limit": (
            "In-sample diagnostic: orchestrator weights were optimized on this period. "
            "This is NOT independently validated performance."
        ),
    }

    output_directory = ROOT / config["paths"]["results"] / "agents_execution_ii7"
    output_directory.mkdir(parents=True, exist_ok=True)
    backtest.to_csv(output_directory / "daily_backtest.csv")
    result["target_weights"].to_csv(output_directory / "target_weights.csv")
    result["execution_schedule"].to_csv(output_directory / "execution_schedule.csv")
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nOutput diagnostico: {output_directory}")


if __name__ == "__main__":
    main()
