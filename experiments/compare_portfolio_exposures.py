from copy import deepcopy

import pandas as pd
import yaml

from src.regime_model import build_regime_dataset
from src.signal_generation import build_duration_score, generate_asset_signals
from src.portfolio import build_portfolio_weights
from src.backtester import apply_execution_lag, calculate_strategy_returns
from src.evaluator import (
    get_first_active_week,
    build_weekly_risk_free,
    calculate_metrics,
)
from src.utils import get_tickers

with open("config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

cfg = deepcopy(config)
cfg["regime"]["rate_state_source"] = "rates__treasury_30y"

features = pd.read_csv(
    "data/processed/features_weekly.csv",
    parse_dates=["Date"],
).set_index("Date").sort_index()

# Utilizziamo esclusivamente il train.
features = features.loc[
    features.index <= cfg["validation"]["train_end"]
].copy()

regimes = build_regime_dataset(features, cfg)
signals = generate_asset_signals(
    build_duration_score(regimes),
    cfg,
)

# Modifichiamo esclusivamente la posizione assunta
# quando il segnale originale e' ribassista sulla duration.
scenarios = {
    "A_50SHY_short50TLT": (1, -1),
    "B_100SHY": (1, 0),
    "C_short100TLT": (0, -1),
}

results = []
annualization = cfg["evaluation"]["annualization_factor"]

for name, (shy_signal, tlt_signal) in scenarios.items():
    candidate = signals.copy()

    bearish = (
        (candidate["SHY_signal"] == 1)
        & (candidate["TLT_signal"] == -1)
    )

    candidate.loc[bearish, "SHY_signal"] = shy_signal
    candidate.loc[bearish, "TLT_signal"] = tlt_signal

    portfolio = build_portfolio_weights(candidate, cfg)

    bt = calculate_strategy_returns(
        apply_execution_lag(portfolio, cfg),
        cfg,
    )

    first_active = get_first_active_week(
        bt, get_tickers(cfg)
    )

    period = bt.loc[bt.index >= first_active].copy()

    rf = build_weekly_risk_free(period, annualization)

    metrics = calculate_metrics(
        period["strategy_net_return"],
        rf,
        annualization,
    )

    results.append({
        "scenario": name,
        "weeks": len(period),
        "cumulative_pct": 100 * metrics["cumulative_return"],
        "cagr_pct": 100 * metrics["annualized_return"],
        "sharpe": metrics["sharpe_ratio"],
        "max_dd_pct": 100 * metrics["max_drawdown"],
        "turnover": period["turnover"].sum(),
    })

print("\nPORTFOLIO EXPOSURE COMPARISON - TRAIN ONLY")
print(
    pd.DataFrame(results)
    .set_index("scenario")
    .round(3)
    .to_string()
)

print("\nNo validation/OOS observations used.")
print("No files modified.")