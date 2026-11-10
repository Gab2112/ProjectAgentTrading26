"""
plots.py

Modulo di visualizzazione grafica e diagnostica per ProjectAgentTrading26.
Fornisce funzioni per analizzare la distribuzione dei segnali, la correlazione tra strategie,
la distribuzione dei pesi dell'Orchestratore e il confronto delle 4 curve di equity
(Equal Active 3 Asset, Equal Active 4 Asset con SHY, Buy & Hold SPY, Buy & Hold SHY).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Configurazione directory di cache per Matplotlib per prevenire warning su ambienti con permessi limitati
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib

# Configura backend headless se non è presente un display grafico interattivo
if not os.environ.get("DISPLAY") and sys.platform != "win32":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

# Risoluzione percorsi di progetto
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1] if CURRENT_DIR.name == "agents" else CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.orchestrator_agent import OrchestratorAgent
from src.data_loader import load_agent_macro_data, load_daily_market_data
from src.features import build_decision_feature_panel
from src.time_split import build_weekly_decision_dates

# Mappatura standard dei nomi degli agenti per leggibilità nei grafici
AGENT_DISPLAY_NAMES = {
    "signals_sma_ema": "SMA/EMA (Trend)",
    "signals_zscore": "Z-Score (Mean-Rev)",
    "signals_macro": "Macro Yield Curve",
    "signals_random_forest": "Random Forest",
    "signals_svm": "SVM Classifier",
    "signals_sentiment": "FinBERT Sentiment",
    "signals_orchestrator": "Orchestrator (GA)",
    "SMA_EMA_Agent": "SMA/EMA (Trend)",
    "ZScore_Agent": "Z-Score (Mean-Rev)",
    "Macro_YieldCurve_Agent": "Macro Yield Curve",
    "RandomForest_Agent": "Random Forest",
    "SVM_Agent": "SVM Classifier",
    "FinBERT_Agent": "FinBERT Sentiment",
    "OrchestratorAgent": "Orchestrator (GA)",
    "Orchestrator_Agent": "Orchestrator (GA)",
}

# Palette cromatica finanziaria standard
SIGNAL_COLORS = {
    -1: "#E74C3C",  # Short: Rosso Corallo / Crimson
    0: "#95A5A6",   # Flat: Grigio Ardesia / Slate
    1: "#27AE60",   # Long: Verde Smeraldo / Emerald
}

SIGNAL_LABELS = {
    -1: "Short (-1)",
    0: "Flat (0)",
    1: "Long (+1)",
}


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
        active_mask = pd.DataFrame(False, index=signals_df.index, columns=self.assets)

        for asset in self.assets:
            if asset not in signals_df.columns:
                continue
            s = signals_df[asset]
            c = confidences_df[asset] if asset in confidences_df.columns else pd.Series(1.0, index=s.index)

            if asset == "SPY":
                short_signal = (s < 0) & (c >= self.min_conf)
                active_mask[asset] = ~short_signal
            else:
                active_mask[asset] = (s == 1) & (c >= self.min_conf)

        return active_mask

    def build_target_weights(
        self,
        signals_df: pd.DataFrame,
        confidences_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        method: str = "equal_active",
        max_weight_per_asset: float = 0.50,
    ) -> pd.DataFrame:
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

        return target_weights


def _run_backtest_no_costs(
    positions_matrix: pd.DataFrame,
    returns_matrix: pd.DataFrame,
    rebalance_threshold: float = 0.05,
) -> pd.DataFrame:
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

    shifted_pos = executed_positions.shift(1).fillna(0.0)
    net_returns = (shifted_pos * rets).sum(axis=1)
    equity_curve = (1.0 + net_returns).cumprod()

    return pd.DataFrame({
        "Return": net_returns,
        "Equity": equity_curve,
    })


# =====================================================================
# 1. FUNZIONI DI CARICAMENTO SEGNALI
# =====================================================================
def load_all_signals_for_asset(
    asset: str = "SPY",
    signals_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, pd.DataFrame]:
    """Carica tutti i segnali disponibili per un determinato asset dalla cartella data/signals/."""
    asset_upper = asset.upper()
    asset_lower = asset.lower()
    sig_path = Path(signals_dir) if signals_dir else (PROJECT_ROOT / "data" / "signals")

    if not sig_path.exists():
        sig_path.mkdir(parents=True, exist_ok=True)

    signals_dict: Dict[str, pd.DataFrame] = {}

    expected_files = [
        ("signals_sma_ema", f"signals_sma_ema_{asset_lower}.csv"),
        ("signals_zscore", f"signals_zscore_{asset_lower}.csv"),
        ("signals_macro", f"signals_macro_{asset_lower}.csv"),
        ("signals_random_forest", f"signals_random_forest_{asset_lower}.csv"),
        ("signals_svm", f"signals_svm_{asset_lower}.csv"),
        ("signals_sentiment", f"signals_sentiment_{asset_lower}.csv"),
        ("signals_orchestrator", f"signals_orchestrator_{asset_lower}.csv"),
    ]

    for key, filename in expected_files:
        filepath = sig_path / filename
        display_name = AGENT_DISPLAY_NAMES.get(key, key)

        if filepath.exists():
            try:
                df_sig = pd.read_csv(filepath, index_col=0, parse_dates=True)
                if "signal" in df_sig.columns:
                    signals_dict[display_name] = df_sig
            except Exception as e:
                print(f"[AVVISO] Errore nel caricamento di {filepath}: {e}")

    return signals_dict


# =====================================================================
# 2. FUNZIONE PRINCIPALE: PLOT DISTRIBUZIONE DEI SEGNALI PER ASSET
# =====================================================================
def plot_agent_signal_distributions(
    asset: str = "SPY",
    signals_dict: Optional[Dict[str, Union[pd.DataFrame, pd.Series]]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = False,
    figsize: Tuple[int, int] = (16, 8),
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Crea un plot che visualizza e confronta la distribuzione dei segnali per tutti gli agenti."""
    asset_upper = asset.upper()

    if signals_dict is None or len(signals_dict) == 0:
        signals_dict = load_all_signals_for_asset(asset=asset_upper)

    if not signals_dict:
        raise ValueError(f"Nessun segnale trovato per l'asset '{asset_upper}' in data/signals/.")

    agent_names: List[str] = []
    counts_data: Dict[int, List[int]] = {-1: [], 0: [], 1: []}
    pcts_data: Dict[int, List[float]] = {-1: [], 0: [], 1: []}
    total_samples: List[int] = []

    date_start, date_end = None, None

    for raw_name, item in signals_dict.items():
        if isinstance(item, pd.DataFrame):
            s_series = item["signal"].astype(int) if "signal" in item.columns else item.iloc[:, 0].astype(int)
        elif isinstance(item, pd.Series):
            s_series = item.astype(int)
        else:
            s_series = pd.Series(item, dtype=int)

        if isinstance(s_series.index, pd.DatetimeIndex) and len(s_series.index) > 0:
            if date_start is None or s_series.index[0] < date_start:
                date_start = s_series.index[0]
            if date_end is None or s_series.index[-1] > date_end:
                date_end = s_series.index[-1]

        clean_name = AGENT_DISPLAY_NAMES.get(raw_name, raw_name)
        agent_names.append(clean_name)

        n_tot = len(s_series)
        total_samples.append(n_tot)

        for sig_val in [-1, 0, 1]:
            cnt = int((s_series == sig_val).sum())
            pct = (cnt / n_tot * 100.0) if n_tot > 0 else 0.0
            counts_data[sig_val].append(cnt)
            pcts_data[sig_val].append(pct)

    n_agents = len(agent_names)
    indices = np.arange(n_agents)
    bar_width = 0.25

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=figsize, dpi=120)

    rects_short = ax.bar(indices - bar_width, pcts_data[-1], width=bar_width, color=SIGNAL_COLORS[-1],
                         edgecolor="#922B21", linewidth=1.2, label=SIGNAL_LABELS[-1], alpha=0.92, zorder=3)
    rects_flat = ax.bar(indices, pcts_data[0], width=bar_width, color=SIGNAL_COLORS[0], edgecolor="#5D6D7E",
                        linewidth=1.2, label=SIGNAL_LABELS[0], alpha=0.88, zorder=3)
    rects_long = ax.bar(indices + bar_width, pcts_data[1], width=bar_width, color=SIGNAL_COLORS[1], edgecolor="#1E8449",
                        linewidth=1.2, label=SIGNAL_LABELS[1], alpha=0.92, zorder=3)

    def _add_bar_labels(rects, signal_key: int):
        for idx, rect in enumerate(rects):
            h = rect.get_height()
            cnt = counts_data[signal_key][idx]
            if h >= 2.0:
                ax.annotate(f"{h:.1f}%\n(n={cnt})", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4),
                            textcoords="offset points", ha="center", va="bottom", fontsize=8.0, fontweight="bold",
                            color="#2C3E50")
            elif h > 0.0:
                ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                            textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color="#555555")

    _add_bar_labels(rects_short, -1)
    _add_bar_labels(rects_flat, 0)
    _add_bar_labels(rects_long, 1)

    for idx, name in enumerate(agent_names):
        if "Orchestrator" in name:
            ax.axvspan(idx - 1.5 * bar_width, idx + 1.5 * bar_width, facecolor="#F4D03F", alpha=0.15, zorder=1,
                       linestyle="--", linewidth=1.5, edgecolor="#D4AC0D")
            ax.text(idx, -8.5, "[META-DECISORE]", ha="center", va="top", fontsize=8.5, fontweight="bold",
                    color="#B7950B")

    ax.set_xticks(indices)
    ax.set_xticklabels(agent_names, fontsize=10, fontweight="bold", rotation=15)
    ax.set_ylabel("Frequenza Relativa (%)", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))

    max_pct = max(max(pcts_data[-1]), max(pcts_data[0]), max(pcts_data[1]))
    ax.set_ylim(0, min(100.0, max(max_pct * 1.18, 50.0)))
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    t_samples = total_samples[0] if total_samples else 0
    t_range_str = f" | Periodo: {date_start.strftime('%Y-%m-%d')} -> {date_end.strftime('%Y-%m-%d')}" if (
                date_start and date_end) else ""

    main_title = title or f"Distribuzione dei Segnali per Agente - Asset: {asset_upper}"
    subtitle = f"Totale Settimane: T = {t_samples}{t_range_str}"

    fig.suptitle(main_title, fontsize=15, fontweight="bold", y=0.98, color="#1A252F")
    ax.set_title(subtitle, fontsize=10.5, color="#566573", pad=12)

    legend = ax.legend(loc="upper right", frameon=True, framealpha=0.95, facecolor="#FFFFFF", edgecolor="#BDC3C7",
                       fontsize=10.5, title="Segnale Operativo", title_fontsize=10.5)
    legend.get_title().set_fontweight("bold")

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    if save_path is None:
        plots_dir = PROJECT_ROOT / "data" / "results" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_path = plots_dir / f"signal_distribution_{asset.lower()}.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[{asset_upper}] Grafico distribuzione segnali salvato in: {save_path.resolve()}")
    plt.close(fig)

    return fig, ax


# =====================================================================
# 3. FUNZIONE RIPRISTINATA PER I TEST (plot_all_assets_signal_distributions)
# =====================================================================
def plot_all_assets_signal_distributions(
    assets: Optional[List[str]] = None,
    signals_dir: Optional[Union[str, Path]] = None,
    save_dir: Optional[Union[str, Path]] = None,
    show_plots: bool = False,
) -> Dict[str, Tuple[plt.Figure, plt.Axes]]:
    """Genera e salva automaticamente il plot della distribuzione dei segnali per tutti gli asset."""
    if assets is None:
        assets = ["SPY", "IEF", "GLD", "SHY"]

    plots_dir = Path(save_dir) if save_dir else (PROJECT_ROOT / "data" / "results" / "plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Tuple[plt.Figure, plt.Axes]] = {}

    for asset in assets:
        target_file = plots_dir / f"signal_distribution_{asset.lower()}.png"
        try:
            loaded_signals = load_all_signals_for_asset(asset=asset, signals_dir=signals_dir)
            fig, ax = plot_agent_signal_distributions(
                asset=asset,
                signals_dict=loaded_signals,
                save_path=target_file,
                show_plot=show_plots,
            )
            results[asset.upper()] = (fig, ax)
        except Exception as e:
            print(f"[{asset.upper()}] Errore durante la generazione del plot: {e}")

    return results


# =====================================================================
# 4. GRAFICO A TORTA DEI PESI DELL'ORCHESTRATORE PER ASSET
# =====================================================================
def plot_orchestrator_weights_pie(
    assets: Optional[List[str]] = None,
    config_dir: Optional[Union[str, Path]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = False,
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """Genera un pannello con grafici a torta dei pesi assegnati dall'Orchestratore (GA) agli agenti."""
    if assets is None:
        assets = ["SPY", "IEF", "GLD", "SHY"]

    cfg_path = Path(config_dir) if config_dir else (PROJECT_ROOT / "src" / "agents" / "config")

    n_assets = len(assets)
    ncols = 2 if n_assets > 1 else 1
    nrows = (n_assets + ncols - 1) // ncols

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=120)

    if n_assets == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    agent_color_map = {
        "SMA/EMA (Trend)": "#3498DB",
        "Z-Score (Mean-Rev)": "#9B59B6",
        "Macro Yield Curve": "#E67E22",
        "Random Forest": "#1ABC9C",
        "SVM Classifier": "#E74C3C",
        "FinBERT Sentiment": "#F1C40F",
    }

    for idx, asset in enumerate(assets):
        ax = axes_flat[idx]
        asset_upper = asset.upper()
        config_file = cfg_path / f"orchestrator_params_{asset.lower()}.json"

        weights_dict = {}
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    if "best_params" in content and "weights" in content["best_params"]:
                        weights_dict = content["best_params"]["weights"]
                    elif "weights" in content:
                        weights_dict = content["weights"]
            except Exception as e:
                print(f"[{asset_upper}] Errore lettura config Orchestratore: {e}")

        if not weights_dict:
            ax.text(0.5, 0.5, f"Configurazione assente\nper {asset_upper}", ha="center", va="center", fontsize=11,
                    fontweight="bold", color="#7F8C8D")
            ax.set_title(f"Asset: {asset_upper}", fontsize=13, fontweight="bold", pad=10)
            ax.axis("off")
            continue

        cleaned_weights = {AGENT_DISPLAY_NAMES.get(k, k): float(v) for k, v in weights_dict.items()}
        filtered_weights = {k: v for k, v in cleaned_weights.items() if v > 0.001}

        if not filtered_weights:
            ax.text(0.5, 0.5, f"Pesi azzerati per {asset_upper}", ha="center", va="center", fontsize=11,
                    color="#7F8C8D")
            ax.set_title(f"Asset: {asset_upper}", fontsize=13, fontweight="bold", pad=10)
            ax.axis("off")
            continue

        labels = list(filtered_weights.keys())
        sizes = list(filtered_weights.values())
        colors = [agent_color_map.get(l, "#BDC3C7") for l in labels]

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct=lambda p: f"{p:.1f}%" if p >= 2.0 else "",
            startangle=140,
            colors=colors,
            wedgeprops={"edgecolor": "#FFFFFF", "linewidth": 1.5, "antialiased": True},
            pctdistance=0.75,
        )

        for autotext in autotexts:
            autotext.set_color("#2C3E50")
            autotext.set_fontsize(9.5)
            autotext.set_fontweight="bold"

        for text in texts:
            text.set_fontsize(10)
            text.set_fontweight("bold")
            text.set_color("#34495E")

        ax.set_title(f"Orchestrator Weights — Asset: {asset_upper}", fontsize=13, fontweight="bold", pad=15,
                     color="#1A252F")

    for j in range(n_assets, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Allocazione dei Pesi dell'Orchestratore (GA) per Asset", fontsize=16, fontweight="bold", y=0.98,
                 color="#1A252F")
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    if save_path is None:
        plots_dir = PROJECT_ROOT / "data" / "results" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_path = plots_dir / "orchestrator_weights_pie_all_assets.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[Orchestrator Pie Chart] Grafico a torta pesi salvato in: {save_path.resolve()}")
    plt.close(fig)

    return fig


# =====================================================================
# 5. CONFRONTO STRATEGIA 4 ASSET (EQUAL ACTIVE) VS BENCHMARK (SPY & SHY)
# =====================================================================
def plot_equity_curves_comparison(
        start_date: str = "2021-01-01",
        end_date: Optional[str] = None,
        title_suffix: str = "Out-Of-Sample (2021 - 2026)",
        save_filename: str = "equity_curves_comparison.png",
) -> plt.Figure:
    """
    Genera e salva un grafico a doppio pannello ben evidenziato:
    - Subplot 1: Rendimento Percentuale Cumulato (%).
    - Subplot 2: Andamento temporale del Drawdown percentuale (MDD).

    Confronto mirato tra:
    1. Strategia Quant (4 Asset: SPY, IEF, GLD, SHY) [Cap 50%]
    2. Benchmark Buy & Hold (100% SPY)
    3. Benchmark Buy & Hold (100% SHY)
    """
    all_assets = ["SPY", "IEF", "GLD", "SHY"]
    assets_4 = ["SPY", "IEF", "GLD", "SHY"]
    cap = 0.50

    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    daily_market, _ = load_daily_market_data(config, PROJECT_ROOT)
    decision_dates = build_weekly_decision_dates(daily_market).index

    df_panel = build_decision_feature_panel(daily_market, decision_dates)
    macro_df = load_agent_macro_data(decision_dates=df_panel.index, root_dir=PROJECT_ROOT)
    df_full = df_panel.join(macro_df, how="left")

    returns_dict = {a: df_full[f"{a}_close"].pct_change() for a in all_assets}
    returns_full = pd.DataFrame(returns_dict).dropna()

    signals_dict, confs_dict = {}, {}
    cfg_dir = PROJECT_ROOT / "src" / "agents" / "config"

    for asset in all_assets:
        cfg_file = cfg_dir / f"orchestrator_params_{asset.lower()}.json"
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)

        orch = OrchestratorAgent(target_asset=asset)
        orch._apply_params(cfg_data.get("best_params", {}))

        pool = orch._build_signals_pool_from_df(df_full)
        combined = orch.combine(pool)

        signals_dict[asset] = combined["signal"]
        confs_dict[asset] = combined["confidence"]

    signals_full = pd.DataFrame(signals_dict)
    confidences_full = pd.DataFrame(confs_dict)

    common_idx = signals_full.index.intersection(returns_full.index)
    mask = common_idx >= start_date
    if end_date:
        mask = mask & (common_idx <= end_date)

    target_idx = common_idx[mask]

    sig_eval = signals_full.loc[target_idx]
    conf_eval = confidences_full.loc[target_idx]
    rets_eval = returns_full.loc[target_idx]

    # 1. Equal Active 4 Asset (SPY, IEF, GLD, SHY)
    agg_4 = MultiAssetSignalAggregator(assets=assets_4, min_conf=0.20, rebalance_threshold=0.05)
    w_4 = agg_4.build_target_weights(sig_eval[assets_4], conf_eval[assets_4], rets_eval[assets_4],
                                     max_weight_per_asset=cap)
    bt_4 = _run_backtest_no_costs(w_4, rets_eval[assets_4])

    # 2. Buy & Hold SPY
    pos_spy = pd.DataFrame(0.0, index=target_idx, columns=all_assets)
    pos_spy["SPY"] = 1.0
    bt_spy = _run_backtest_no_costs(pos_spy, rets_eval[all_assets])

    # 3. Buy & Hold SHY
    pos_shy = pd.DataFrame(0.0, index=target_idx, columns=all_assets)
    pos_shy["SHY"] = 1.0
    bt_shy = _run_backtest_no_costs(pos_shy, rets_eval[all_assets])

    # Dataframe Curve di Equity
    equity_df = pd.DataFrame({
        "Equal Active (4 Asset: SPY/IEF/GLD/SHY)": bt_4["Equity"],
        "Buy & Hold (100% SPY)": bt_spy["Equity"],
        "Buy & Hold (100% SHY)": bt_shy["Equity"],
    })

    # Dataframe Rendimento Percentuale Cumulato (%)
    returns_cum_df = (equity_df - 1.0) * 100.0

    # Dataframe Drawdown (%) per ciascuna strategia
    drawdown_df = (equity_df - equity_df.cummax()) / equity_df.cummax() * 100.0

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), dpi=150, sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})

    colors = {
        "Equal Active (4 Asset: SPY/IEF/GLD/SHY)": "#16A085",  # Verde Smeraldo Scuro / Teal
        "Buy & Hold (100% SPY)": "#E74C3C",  # Rosso Corallo
        "Buy & Hold (100% SHY)": "#7F8C8D",  # Grigio Ardesia
    }

    linestyles = {
        "Equal Active (4 Asset: SPY/IEF/GLD/SHY)": "-",
        "Buy & Hold (100% SPY)": "--",
        "Buy & Hold (100% SHY)": ":",
    }

    # ------------------------------------------------------------------
    # SUBPLOT 1: RENDIMENTO PERCENTUALE CUMULATO (%)
    # ------------------------------------------------------------------
    # Disegna prima i benchmark
    for col in ["Buy & Hold (100% SHY)", "Buy & Hold (100% SPY)"]:
        ax1.plot(
            returns_cum_df.index,
            returns_cum_df[col],
            label=col,
            color=colors[col],
            linestyle=linestyles[col],
            linewidth=1.8,
            alpha=0.85,
            zorder=2,
        )

    # Disegna la strategia a 4 asset in evidenza
    main_col = "Equal Active (4 Asset: SPY/IEF/GLD/SHY)"
    ax1.plot(
        returns_cum_df.index,
        returns_cum_df[main_col],
        label=main_col,
        color=colors[main_col],
        linestyle=linestyles[main_col],
        linewidth=3.0,
        zorder=4,
    )
    ax1.fill_between(returns_cum_df.index, returns_cum_df[main_col], 0, color=colors[main_col], alpha=0.08, zorder=1)

    ax1.axhline(0.0, color="#333333", linestyle="-", linewidth=0.8, alpha=0.7)
    ax1.set_title(f"Rendimento Cumulato e Rischio — Strategia 4 Asset vs Benchmark ({title_suffix})", fontsize=14,
                  fontweight="bold", pad=12)
    ax1.set_ylabel("Rendimento Cumulato (%)", fontsize=11, fontweight="bold")
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0, decimals=0))
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=10.5)

    # ------------------------------------------------------------------
    # SUBPLOT 2: ANDAMENTO DRAWDOWN (%)
    # ------------------------------------------------------------------
    for col in ["Buy & Hold (100% SHY)", "Buy & Hold (100% SPY)"]:
        ax2.plot(
            drawdown_df.index,
            drawdown_df[col],
            label=col,
            color=colors[col],
            linestyle=linestyles[col],
            linewidth=1.5,
            alpha=0.8,
            zorder=2,
        )

    ax2.plot(
        drawdown_df.index,
        drawdown_df[main_col],
        label=main_col,
        color=colors[main_col],
        linestyle=linestyles[main_col],
        linewidth=2.5,
        zorder=4,
    )
    ax2.fill_between(drawdown_df.index, drawdown_df[main_col], 0, color=colors[main_col], alpha=0.2, zorder=3)

    ax2.set_ylabel("Drawdown (%)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Data", fontsize=11, fontweight="bold")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0, decimals=0))
    ax2.grid(True, linestyle="--", alpha=0.5)

    plots_dir = PROJECT_ROOT / "data" / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_path = plots_dir / save_filename

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[Return & Drawdown Plot] Grafico salvato in: {save_path.resolve()}")
    plt.close(fig)

    return fig

# =====================================================================
# 6. FUNZIONE AUTOMATICA COMPLETA (FA TUTTO)
# =====================================================================
def run_all_plots(assets: Optional[List[str]] = None, show_plots: bool = False) -> None:
    """Esegue automaticamente tutti i plot diagnostici, di allocazione e le curve di equity per tutti gli asset."""
    if assets is None:
        assets = ["SPY", "IEF", "GLD", "SHY"]

    print("\n" + "=" * 90)
    print("      AVVIO GENERAZIONE AUTOMATICA DI TUTTI I GRAFICI E DIAGNOSTICA")
    print("=" * 90)

    # 1. Distribuzione segnali per ogni asset
    print("\n--- [1/3] Generazione Grafici Distribuzione Segnali ---")
    plot_all_assets_signal_distributions(assets=assets, show_plots=show_plots)

    # 2. Pie chart dei pesi dell'Orchestratore multi-asset
    print("\n--- [2/3] Generazione Grafico a Torta Pesi Orchestratore ---")
    try:
        plot_orchestrator_weights_pie(assets=assets, show_plot=show_plots)
    except Exception as e:
        print(f"[Orchestrator Pie] Errore nella generazione del grafico a torta: {e}")

    # 3. Curve di Equity (OOS e IS)
    print("\n--- [3/3] Generazione Grafici Curve di Equity Comparative ---")
    try:
        plot_equity_curves_comparison(
            start_date="2021-01-01",
            title_suffix="Out-Of-Sample (2021 - 2026)",
            save_filename="equity_curves_oos.png",
        )
        plot_equity_curves_comparison(
            start_date="2005-05-20",
            end_date="2020-12-30",
            title_suffix="In-Sample / Train (2005 - 2020)",
            save_filename="equity_curves_is.png",
        )
    except Exception as e:
        print(f"[Equity Curves] Errore durante il calcolo delle curve di equity: {e}")

    print("\n" + "=" * 90)
    print("  [OK] PROCESSO DI VISUALIZZAZIONE E PLOTTING COMPLETATO CON SUCCESSO.")
    print("  I grafici sono stati salvati nella cartella: data/results/plots/")
    print("=" * 90 + "\n")


# =====================================================================
# 7. ENTRY POINT CLI
# =====================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Script unificato di visualizzazione e diagnostica ProjectAgentTrading26.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Mostra i grafici a schermo tramite finestra interattiva",
    )
    parser.add_argument(
        "--asset",
        type=str,
        default="ALL",
        help="Asset specifico ('SPY', 'IEF', 'GLD', 'SHY') oppure 'ALL'. Default: 'ALL'",
    )

    args = parser.parse_args()

    if args.asset.upper() == "ALL":
        run_all_plots(show_plots=args.show)
    else:
        asset_target = args.asset.upper()
        print(f"\nEsecuzione plot mirato per l'asset: {asset_target}")
        loaded = load_all_signals_for_asset(asset=asset_target)
        if loaded:
            plot_agent_signal_distributions(asset=asset_target, signals_dict=loaded, show_plot=args.show)
        plot_orchestrator_weights_pie(assets=[asset_target], show_plot=args.show)