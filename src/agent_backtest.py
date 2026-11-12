"""In-sample (training) comparison of the Equal Active allocation (2005-05-20 to 2020-12-30).

Objective:
- Compare the Equal Active strategy on four assets (SPY, IEF, GLD and SHY,
  the latter as a cash proxy / short-term Treasury) with the buy-and-hold
  benchmark (100% SPY) and the defensive buy-and-hold benchmark (100% SHY).
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
from typing import Tuple, List, Dict, Optional

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

from src.agents.orchestrator_agent import OrchestratorAgent
from src.data_loader import load_agent_macro_data, load_daily_market_data
from src.features import build_decision_feature_panel
from src.time_split import build_weekly_decision_dates

warnings.filterwarnings("ignore")


# =====================================================================
# INTEGRATED MULTI-ASSET AGGREGATOR
# =====================================================================

class MultiAssetSignalAggregator:
    """Quantitative engine for signal aggregation and dynamic multi-asset allocation."""

    def __init__(
        self,
        assets: Optional[List[str]] = None,
        min_conf: float = 0.20,
        rebalance_threshold: float = 0.05,
        transaction_cost: float = 0.0,
        vol_window: int = 26,
    ):
        self.assets = assets or ["SPY", "IEF", "GLD", "SHY"]
        self.min_conf = float(min_conf)
        self.rebalance_threshold = float(rebalance_threshold)
        self.transaction_cost = float(transaction_cost)
        self.vol_window = int(vol_window)

    def determine_active_mask(
        self,
        signals_df: pd.DataFrame,
        confidences_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Determine the active state (True/False) of each asset.

        SPY follows an asymmetric rule (long unless a confirmed short signal);
        the other assets require a long signal with sufficient confidence.
        """
        active_mask = pd.DataFrame(False, index=signals_df.index, columns=self.assets)

        for asset in self.assets:
            if asset not in signals_df.columns:
                continue
            s = signals_df[asset]
            c = confidences_df[asset] if asset in confidences_df.columns else pd.Series(1.0, index=s.index)

            if asset == "SPY":
                # SPY: long by default; deactivated only on a confirmed short signal.
                short_signal = (s < 0) & (c >= self.min_conf)
                active_mask[asset] = ~short_signal
            else:
                # IEF, GLD, SHY: active only with a long signal (+1) and sufficient confidence.
                active_mask[asset] = (s == 1) & (c >= self.min_conf)

        return active_mask

    def build_target_weights(
        self,
        signals_df: pd.DataFrame,
        confidences_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        method: str = "equal_active",
        base_budgets: Optional[Dict[str, float]] = None,
        max_weight_per_asset: float = 0.50,
    ) -> pd.DataFrame:
        """Generate target weights by allocating capital equally across the active assets, subject to the per-asset cap.
        """
        common_idx = signals_df.index.intersection(returns_df.index)
        sig = signals_df.loc[common_idx]
        conf = confidences_df.loc[common_idx]

        active_mask = self.determine_active_mask(sig, conf)
        target_weights = pd.DataFrame(0.0, index=common_idx, columns=self.assets)

        for t in common_idx:
            active_assets_t = [a for a in self.assets if active_mask.loc[t, a]]

            if not active_assets_t:
                continue

            if method == "equal_active":
                w_each = 1.0 / len(active_assets_t)
                for a in active_assets_t:
                    target_weights.loc[t, a] = min(w_each, max_weight_per_asset)

                tot = target_weights.loc[t].sum()
                if 0 < tot < 1.0 and len(active_assets_t) > 1:
                    target_weights.loc[t] = target_weights.loc[t] / tot
            else:
                raise ValueError(f"Metodo {method} non supportato in questa configurazione.")

        return target_weights


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
    """Generate the in-sample signals using the models and weights defined during training.
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


def run_comparison_3vs4_assets_is():
    """Run the in-sample comparison of the Equal Active strategy with the buy-and-hold benchmarks and print the metrics.
    """
    all_assets = ["SPY", "IEF", "GLD", "SHY"]
    assets_4 = ["SPY", "IEF", "GLD", "SHY"]

    # Explicit in-sample (training) date range.
    is_start_date = "2005-05-20"
    is_end_date = "2020-12-30"
    cap = 0.50  # Fixed 50% cap per asset

    print("=" * 115)
    print("      COMPARAZIONE DIRETTA: EQUAL ACTIVE (3 VS 4 ASSET) VS BENCHMARK (SPY E SHY)")
    print(f"                             VALIDAZIONE IN-SAMPLE / TRAIN ({is_start_date} -> {is_end_date})")
    print("=" * 115)

    # 1. Load the full dataset.
    print("\n[1/3] Caricamento dati e feature settimanali per la finestra In-Sample...")
    df_features, returns_full = build_full_dataset(PROJECT_ROOT, all_assets)

    # 2. Generate the signals for the four assets.
    print(f"[2/3] Generazione segnali IS per gli asset: {all_assets}...")
    signals_full, confidences_full = generate_orchestrator_signals(all_assets, df_features, PROJECT_ROOT)

    common_idx = signals_full.index.intersection(returns_full.index)

    # Exact in-sample filter between the specified start and end dates.
    is_idx = common_idx[(common_idx >= is_start_date) & (common_idx <= is_end_date)]

    signals_is = signals_full.loc[is_idx]
    confs_is = confidences_full.loc[is_idx]
    returns_is = returns_full.loc[is_idx]

    print(f"  * Periodo IS Effettivo: {is_idx.min():%Y-%m-%d} -> {is_idx.max():%Y-%m-%d} ({len(is_idx)} settimane)")

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
        signals_df=signals_is[assets_4],
        confidences_df=confs_is[assets_4],
        returns_df=returns_is[assets_4],
        method="equal_active",
        max_weight_per_asset=cap,
    )
    _, m_quant_4 = run_backtest_no_costs(weights_4, returns_is[assets_4], rebalance_threshold=0.05)
    results["Equal Active (4 Asset: incl. SHY)"] = m_quant_4

    # ------------------------------------------------------------------
    # 3. Buy-and-hold benchmark (100% SPY)
    # ------------------------------------------------------------------
    pos_spy = pd.DataFrame(0.0, index=is_idx, columns=all_assets)
    pos_spy["SPY"] = 1.0
    _, m_spy = run_backtest_no_costs(pos_spy, returns_is[all_assets], rebalance_threshold=0.0)
    results["Buy & Hold (100% SPY)"] = m_spy

    # ------------------------------------------------------------------
    # 4. Buy-and-hold benchmark (100% SHY)
    # ------------------------------------------------------------------
    pos_shy = pd.DataFrame(0.0, index=is_idx, columns=all_assets)
    pos_shy["SHY"] = 1.0
    _, m_shy = run_backtest_no_costs(pos_shy, returns_is[all_assets], rebalance_threshold=0.0)
    results["Buy & Hold (100% SHY)"] = m_shy

    df_results = pd.DataFrame(results)

    # 3. Final output.
    print("\n[3/3] Risultati della comparazione IS (Train):")
    print("\n" + "=" * 115)
    print("                                TABELLA COMPARATIVA METRICHE IS")
    print("=" * 115)
    print(df_results.round(2).to_string())
    print("-" * 115)


if __name__ == "__main__":
    run_comparison_3vs4_assets_is()