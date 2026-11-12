"""Retrospective comparison of the Equal Active allocation from 2021-01-01 to October 2026.

Note: 2021 belongs to the development period; the official out-of-sample
evaluation of the frozen strategy starts on 2022-01-07.

Objective:
- Compare the Equal Active strategy on four assets (SPY, IEF, GLD and SHY,
  the latter as a cash proxy / short-term Treasury) with the buy-and-hold
  benchmarks (100% SPY and 100% SHY).
- Apply a fixed maximum weight (cap) of 50% per asset.
- Set transaction costs to zero (tc = 0.0).
- Compute risk/return metrics, the number of position changes and turnover.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Tuple, List

# Temporary directory for matplotlib.
os.environ["MPLCONFIGDIR"] = str(Path(os.environ.get("TEMP", "/tmp")) / "matplotlib_pat")

# Project path resolution.
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR if (CURRENT_DIR / "config.yaml").exists() else CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml

from src.agent_backtest import MultiAssetSignalAggregator
from src.agents.orchestrator_agent import OrchestratorAgent
from src.data_loader import load_agent_macro_data, load_daily_market_data
from src.features import build_decision_feature_panel
from src.time_split import build_weekly_decision_dates

warnings.filterwarnings("ignore")


# =====================================================================
# BACKTEST ENGINE WITHOUT COMMISSIONS, WITH POSITION-CHANGE COUNT
# =====================================================================

def run_backtest_no_costs(
        positions_matrix: pd.DataFrame,
        returns_matrix: pd.DataFrame,
        rebalance_threshold: float = 0.05,
        periods_per_year: int = 52,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Run the portfolio backtest without transaction costs (tc = 0.0).

    The number of effective position changes and the cumulative turnover are
    tracked explicitly.
    """
    common_idx = positions_matrix.index.intersection(returns_matrix.index)
    pos_target = positions_matrix.loc[common_idx]
    rets = returns_matrix.loc[common_idx]

    executed_positions = pd.DataFrame(0.0, index=common_idx, columns=pos_target.columns)
    current_pos = pd.Series(0.0, index=pos_target.columns)

    for t in common_idx:
        target = pos_target.loc[t]
        weight_changes = (target - current_pos).abs()

        if (weight_changes >= rebalance_threshold).any():
            current_pos = target.copy()

        executed_positions.loc[t] = current_pos

    # Gross returns (equal to net returns since tc = 0.0).
    shifted_pos = executed_positions.shift(1).fillna(0.0)
    pos_diff = executed_positions.diff().abs().fillna(0.0)
    turnover = pos_diff.sum(axis=1) / 2.0

    net_returns = (shifted_pos * rets).sum(axis=1)
    equity_curve = (1.0 + net_returns).cumprod()

    # Compute the metrics.
    total_return = equity_curve.iloc[-1] - 1.0
    ann_return = (equity_curve.iloc[-1]) ** (periods_per_year / len(equity_curve)) - 1.0
    ann_volatility = net_returns.std() * np.sqrt(periods_per_year)
    sharpe_ratio = ann_return / ann_volatility if ann_volatility != 0 else 0.0

    drawdown = (equity_curve - equity_curve.cummax()) / equity_curve.cummax()
    max_drawdown = drawdown.min()
    calmar_ratio = abs(ann_return / max_drawdown) if max_drawdown != 0 else 0.0

    # Count every week in which the portfolio composition actually changed.
    rebalance_weeks_count = int((pos_diff.sum(axis=1) > 1e-5).sum())

    metrics = pd.Series({
        "Rendimento Totale (%)": total_return * 100,
        "Rendimento Ann. (%)": ann_return * 100,
        "Volatilità Ann. (%)": ann_volatility * 100,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown (%)": max_drawdown * 100,
        "Calmar Ratio": calmar_ratio,
        "N° Variazioni di Posizione": float(rebalance_weeks_count),
        "Turnover Cumulato": turnover.sum(),
    })

    bt_dataframe = pd.DataFrame({
        "Return": net_returns,
        "Equity": equity_curve,
        "Turnover": turnover,
    })

    return bt_dataframe, metrics


# =====================================================================
# DATA LOADING AND SIGNAL GENERATION WITH THE TRAINED PARAMETERS
# =====================================================================

def build_full_dataset(project_root: Path, assets: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the daily market data and build the weekly panel for the requested assets.
    """
    with open(project_root / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    daily_market, _ = load_daily_market_data(config, project_root)
    decision_dates = build_weekly_decision_dates(daily_market).index

    df_panel = build_decision_feature_panel(daily_market, decision_dates)
    macro_df = load_agent_macro_data(decision_dates=df_panel.index, root_dir=project_root)
    df_full = df_panel.join(macro_df, how="left")

    returns_dict = {}
    for asset in assets:
        returns_dict[asset] = df_full[f"{asset}_close"].pct_change()
    returns_df = pd.DataFrame(returns_dict).dropna()

    return df_full, returns_df


def generate_orchestrator_signals(
        assets: List[str],
        df_features: pd.DataFrame,
        project_root: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the signals from 2021 onwards using the models and weights frozen after training.
    """
    signals_dict, confs_dict = {}, {}
    cfg_dir = project_root / "src/agents/config"

    for asset in assets:
        cfg_file = cfg_dir / f"orchestrator_params_{asset.lower()}.json"
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)

        orch = OrchestratorAgent(target_asset=asset)
        orch._apply_params(cfg_data.get("best_params", {}))

        pool = orch._build_signals_pool_from_df(df_features)
        combined = orch.combine(pool)

        signals_dict[asset] = combined["signal"]
        confs_dict[asset] = combined["confidence"]

    return pd.DataFrame(signals_dict), pd.DataFrame(confs_dict)


def run_comparison_3vs4_assets():
    """Run the comparison from 2021 onwards and print the metrics of each strategy."""
    all_assets = ["SPY", "IEF", "GLD", "SHY"]
    assets_3 = ["SPY", "IEF", "GLD"]
    assets_4 = ["SPY", "IEF", "GLD", "SHY"]

    oos_start_date = "2021-01-01"
    cap = 0.50  # Fixed 50% cap per asset

    print("=" * 115)
    print("      COMPARAZIONE DIRECTA: EQUAL ACTIVE 3 ASSET VS EQUAL ACTIVE 4 ASSET VS BUY & HOLD SPY")
    print("                             VALIDAZIONE OOS (2021 - 2026)")
    print("=" * 115)

    # 1. Load the full dataset.
    print("\n[1/3] Caricamento dati e feature settimanali estese fino a Ottobre 2026...")
    df_features, returns_full = build_full_dataset(PROJECT_ROOT, all_assets)

    # 2. Generate the signals for the four assets.
    print(f"[2/3] Generazione segnali OOS per gli asset: {all_assets}...")
    signals_full, confidences_full = generate_orchestrator_signals(all_assets, df_features, PROJECT_ROOT)

    common_idx = signals_full.index.intersection(returns_full.index)
    oos_idx = common_idx[common_idx >= oos_start_date]

    signals_oos = signals_full.loc[oos_idx]
    confs_oos = confidences_full.loc[oos_idx]
    returns_oos = returns_full.loc[oos_idx]

    print(f"  * Periodo OOS: {oos_idx.min():%Y-%m-%d} -> {oos_idx.max():%Y-%m-%d} ({len(oos_idx)} settimane)")

    results = {}


    # ------------------------------------------------------------------
    # 2. Equal Active strategy - 4 assets (SPY, IEF, GLD, SHY)
    # ------------------------------------------------------------------
    agg_4 = MultiAssetSignalAggregator(
        assets=assets_4,
        min_conf=0.20,
        rebalance_threshold=0.05,
        transaction_cost=0.0,
    )
    weights_4 = agg_4.build_target_weights(
        signals_df=signals_oos[assets_4],
        confidences_df=confs_oos[assets_4],
        returns_df=returns_oos[assets_4],
        method="equal_active",
        max_weight_per_asset=cap,
    )
    _, m_quant_4 = run_backtest_no_costs(weights_4, returns_oos[assets_4], rebalance_threshold=0.05)
    results["Equal Active (4 Asset: incl. SHY)"] = m_quant_4

    # ------------------------------------------------------------------
    # 3. Buy-and-hold benchmark (100% SPY)
    # ------------------------------------------------------------------
    pos_spy = pd.DataFrame(0.0, index=oos_idx, columns=assets_3)
    pos_spy["SPY"] = 1.0
    _, m_spy = run_backtest_no_costs(pos_spy, returns_oos[assets_3], rebalance_threshold=0.0)
    results["Buy & Hold (100% SPY)"] = m_spy

    # ------------------------------------------------------------------
    # 4. Buy-and-hold benchmark (100% SHY)
    # ------------------------------------------------------------------
    pos_shy = pd.DataFrame(0.0, index=oos_idx, columns=all_assets)
    pos_shy["SHY"] = 1.0
    _, m_shy = run_backtest_no_costs(pos_shy, returns_oos[all_assets], rebalance_threshold=0.0)
    results["Buy & Hold (100% SHY)"] = m_shy

    df_results = pd.DataFrame(results)

    # 3. Final output.
    print("\n[3/3] Risultati della comparazione OOS:")
    print("\n" + "=" * 115)
    print("                                TABELLA COMPARATIVA METRICHE OOS")
    print("=" * 115)
    print(df_results.round(2).to_string())
    print("-" * 115)


if __name__ == "__main__":
    run_comparison_3vs4_assets()