"""
test_is_agent.py

Validazione In-Sample / Training (IS: 2005-05-20 -> 2020-12-30).
Obiettivo:
- Confronto diretto tra la strategia Equal Active a 3 Asset (SPY, IEF, GLD),
  la strategia Equal Active a 4 Asset (con SHY come Cash Proxy/Short-Term Treasury),
  il Benchmark Buy & Hold (100% SPY) e il Benchmark Difensivo Buy & Hold (100% SHY).
- Peso massimo (cap) fisso al 50% su singolo asset per entrambe le varianti quant.
- Costi di transazione azzerati (tc = 0.0).
- Calcolo delle metriche di rischio/rendimento, N° Variazioni di Posizione e Turnover.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Optional

# Imposta directory temporanea per matplotlib
os.environ["MPLCONFIGDIR"] = str(Path(os.environ.get("TEMP", "/tmp")) / "matplotlib_pat")

# Risoluzione percorsi di progetto
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
# AGGREGATORE MULTI-ASSET INTEGRATO (SOLUZIONE DEFINITIVA IMPORT ERROR)
# =====================================================================

class MultiAssetSignalAggregator:
    """
    Motore quantitativo di aggregazione segnali e asset allocation dinamica multi-asset.
    """

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
        """
        Determina lo stato attivo (True/False) per ciascun asset applicando
        la regola asimmetrica per SPY e la soglia di confidenza per gli altri.
        """
        active_mask = pd.DataFrame(False, index=signals_df.index, columns=self.assets)

        for asset in self.assets:
            if asset not in signals_df.columns:
                continue
            s = signals_df[asset]
            c = confidences_df[asset] if asset in confidences_df.columns else pd.Series(1.0, index=s.index)

            if asset == "SPY":
                # SPY: Long di base. Disattivo solo su segnale Short confermato
                short_signal = (s < 0) & (c >= self.min_conf)
                active_mask[asset] = ~short_signal
            else:
                # IEF, GLD, SHY: Attivo solo con segnale Long (+1) e confidenza sufficiente
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
        """
        Genera i pesi target del portafoglio allocando il capitale equamente
        sugli asset attivi rispettando il cap massimo per singolo asset.
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
# MOTORE DI BACKTEST SENZA COMMISSIONI MA CON CONTEGGIO VARIAZIONI
# =====================================================================

def run_backtest_no_costs(
    positions_matrix: pd.DataFrame,
    returns_matrix: pd.DataFrame,
    rebalance_threshold: float = 0.05,
    periods_per_year: int = 52,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Esegue il backtest del portafoglio senza costi di transazione (tc = 0.0),
    tracciando in modo esplicito il numero di variazioni di posizione effettive
    e il turnover cumulato.
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

    # Rendimenti lordi (uguali ai netti poiché tc = 0.0)
    shifted_pos = executed_positions.shift(1).fillna(0.0)
    pos_diff = executed_positions.diff().abs().fillna(0.0)
    turnover = pos_diff.sum(axis=1) / 2.0

    net_returns = (shifted_pos * rets).sum(axis=1)
    equity_curve = (1.0 + net_returns).cumprod()

    # Calcolo metriche
    total_return = equity_curve.iloc[-1] - 1.0
    ann_return = (equity_curve.iloc[-1]) ** (periods_per_year / len(equity_curve)) - 1.0
    ann_volatility = net_returns.std() * np.sqrt(periods_per_year)
    sharpe_ratio = ann_return / ann_volatility if ann_volatility != 0 else 0.0

    drawdown = (equity_curve - equity_curve.cummax()) / equity_curve.cummax()
    max_drawdown = drawdown.min()
    calmar_ratio = abs(ann_return / max_drawdown) if max_drawdown != 0 else 0.0

    # Conta tutte le settimane in cui la composizione di portafoglio è effettivamente cambiata
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
# CARICAMENTO DATI E GENERAZIONE SEGNALI CON PARAMETRI DA TRAIN
# =====================================================================

def build_full_dataset(project_root: Path, assets: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carica market data giornaliero e costruisce il panel settimanale per gli asset specificati."""
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
    """Genera i segnali per il periodo in-sample usando i modelli e i pesi definiti nel train."""
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
    all_assets = ["SPY", "IEF", "GLD", "SHY"]
    assets_4 = ["SPY", "IEF", "GLD", "SHY"]

    # Range temporale esplicito per l'In-Sample (Train)
    is_start_date = "2005-05-20"
    is_end_date = "2020-12-30"
    cap = 0.50  # Cap fisso al 50% per asset

    print("=" * 115)
    print("      COMPARAZIONE DIRETTA: EQUAL ACTIVE (3 VS 4 ASSET) VS BENCHMARK (SPY E SHY)")
    print(f"                             VALIDAZIONE IN-SAMPLE / TRAIN ({is_start_date} -> {is_end_date})")
    print("=" * 115)

    # 1. Caricamento Dati Completi
    print("\n[1/3] Caricamento dati e feature settimanali per la finestra In-Sample...")
    df_features, returns_full = build_full_dataset(PROJECT_ROOT, all_assets)

    # 2. Generazione Segnali per i 4 asset
    print(f"[2/3] Generazione segnali IS per gli asset: {all_assets}...")
    signals_full, confidences_full = generate_orchestrator_signals(all_assets, df_features, PROJECT_ROOT)

    common_idx = signals_full.index.intersection(returns_full.index)

    # Filtro In-Sample esatto tra start_date e end_date specificate
    is_idx = common_idx[(common_idx >= is_start_date) & (common_idx <= is_end_date)]

    signals_is = signals_full.loc[is_idx]
    confs_is = confidences_full.loc[is_idx]
    returns_is = returns_full.loc[is_idx]

    print(f"  * Periodo IS Effettivo: {is_idx.min():%Y-%m-%d} -> {is_idx.max():%Y-%m-%d} ({len(is_idx)} settimane)")

    results = {}

    # ------------------------------------------------------------------
    # 2. Strategy Equal Active - 4 Asset (SPY, IEF, GLD, SHY)
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
    # 3. Benchmark Buy & Hold (100% SPY)
    # ------------------------------------------------------------------
    pos_spy = pd.DataFrame(0.0, index=is_idx, columns=all_assets)
    pos_spy["SPY"] = 1.0
    _, m_spy = run_backtest_no_costs(pos_spy, returns_is[all_assets], rebalance_threshold=0.0)
    results["Buy & Hold (100% SPY)"] = m_spy

    # ------------------------------------------------------------------
    # 4. Benchmark Buy & Hold (100% SHY)
    # ------------------------------------------------------------------
    pos_shy = pd.DataFrame(0.0, index=is_idx, columns=all_assets)
    pos_shy["SHY"] = 1.0
    _, m_shy = run_backtest_no_costs(pos_shy, returns_is[all_assets], rebalance_threshold=0.0)
    results["Buy & Hold (100% SHY)"] = m_shy

    df_results = pd.DataFrame(results)

    # 3. Output finale
    print("\n[3/3] Risultati della comparazione IS (Train):")
    print("\n" + "=" * 115)
    print("                                TABELLA COMPARATIVA METRICHE IS")
    print("=" * 115)
    print(df_results.round(2).to_string())
    print("-" * 115)


if __name__ == "__main__":
    run_comparison_3vs4_assets_is()