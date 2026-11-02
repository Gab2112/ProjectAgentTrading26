"""
plots.py

Modulo di visualizzazione grafica e diagnostica per ProjectAgentTrading26.
Fornisce funzioni per analizzare la distribuzione dei segnali, la correlazione tra strategie
e il comportamento comparativo di tutti gli agenti quantitativi:
- SMAEMAAgent (Trend-Following)
- ZScoreAgent (Mean-Reversion)
- MacroYieldCurveAgent (Macroeconomico)
- RandomForestAgent (Machine Learning non-lineare)
- SVMAgent (Machine Learning a margine massimo)
- OrchestratorAgent (Meta-Decisore di Sintesi)
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Configurazione directory di cache per Matplotlib per prevenire warning su ambienti con permessi limitati
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib
# Configura backend headless se non e presente un display grafico interattivo
if not os.environ.get("DISPLAY") and sys.platform != "win32":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

# Risoluzione percorsi di progetto
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Mappatura standard dei nomi degli agenti per leggibilita nei grafici
AGENT_DISPLAY_NAMES = {
    "signals_sma_ema": "SMA/EMA (Trend)",
    "signals_zscore": "Z-Score (Mean-Rev)",
    "signals_macro": "Macro Yield Curve",
    "signals_random_forest": "Random Forest",
    "signals_svm": "SVM Classifier",
    "signals_statistical": "Ensemble Statistico",
    "signals_orchestrator": "Orchestrator (GA)",
    "SMA_EMA_Agent": "SMA/EMA (Trend)",
    "ZScore_Agent": "Z-Score (Mean-Rev)",
    "Macro_YieldCurve_Agent": "Macro Yield Curve",
    "RandomForest_Agent": "Random Forest",
    "SVM_Agent": "SVM Classifier",
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
# 1. FUNZIONI DI CARICAMENTO SEGNALI
# =====================================================================
def load_all_signals_for_asset(
    asset: str = "SPY",
    signals_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Carica tutti i segnali disponibili per un determinato asset dalla cartella data/signals/.
    Se alcuni segnali (es. macro) non sono ancora stati esportati su CSV, tenta di generarli
    al volo per fornire un quadro completo.
    
    Parametri:
    -----------
    asset : str
        Ticker dell'asset (es. 'SPY', 'IEF', 'GLD', 'SHY').
    signals_dir : Path o str, opzionale
        Cartella contenente i file CSV dei segnali. Default: <PROJECT_ROOT>/data/signals.
        
    Ritorna:
    --------
    Dict[str, pd.DataFrame]
        Dizionario {NomeAgente: DataFrame con colonne ['signal', 'confidence']}.
    """
    asset_upper = asset.upper()
    asset_lower = asset.lower()
    sig_path = Path(signals_dir) if signals_dir else (PROJECT_ROOT / "data" / "signals")

    if not sig_path.exists():
        sig_path.mkdir(parents=True, exist_ok=True)

    signals_dict: Dict[str, pd.DataFrame] = {}

    # File attesi per ciascun agente
    expected_files = [
        ("signals_sma_ema", f"signals_sma_ema_{asset_lower}.csv"),
        ("signals_zscore", f"signals_zscore_{asset_lower}.csv"),
        ("signals_macro", f"signals_macro_{asset_lower}.csv"),
        ("signals_random_forest", f"signals_random_forest_{asset_lower}.csv"),
        ("signals_svm", f"signals_svm_{asset_lower}.csv"),
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

    # Fallback per Macro Yield Curve se il file non e presente
    macro_name = AGENT_DISPLAY_NAMES["signals_macro"]
    if macro_name not in signals_dict:
        try:
            from src.agents.base_agent import BaseAgent
            from src.agents.macro_yield_curve_agent import MacroYieldCurveAgent

            df_market = BaseAgent.load_complete_dataset(root_dir=PROJECT_ROOT)
            macro_agent = MacroYieldCurveAgent(target_asset=asset_upper)
            macro_pred = macro_agent.predict_multi_asset(df_market, assets=[asset_upper])
            
            sig_col = f"{asset_upper}_signal"
            if sig_col in macro_pred.columns:
                m_sig = macro_pred[sig_col]
                m_score = macro_agent.calculate_macro_score(df_market)
                m_conf = (m_score.abs() / 2.0).clip(0.0, 1.0).round(4)
                m_conf = pd.Series(np.where(m_sig != 0, np.maximum(0.50, m_conf), 0.0), index=df_market.index)
                
                df_macro = pd.DataFrame({"signal": m_sig, "confidence": m_conf}, index=df_market.index)
                signals_dict[macro_name] = df_macro
                # Salva per future consultazioni
                df_macro.to_csv(sig_path / f"signals_macro_{asset_lower}.csv")
        except Exception:
            pass

    return signals_dict


# =====================================================================
# 2. FUNZIONE PRINCIPALE: PLOT DISTRIBUZIONE DEI SEGNALI PER ASSET
# =====================================================================
def plot_agent_signal_distributions(
    asset: str = "SPY",
    signals_dict: Optional[Dict[str, Union[pd.DataFrame, pd.Series]]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = False,
    figsize: Tuple[int, int] = (15, 8),
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Crea un singolo plot per un asset che visualizza e confronta la distribuzione
    dei segnali discreti (-1 Short, 0 Flat, 1 Long) per tutti gli agenti di trading.
    
    Parametri:
    -----------
    asset : str
        Ticker dell'asset di riferimento (es. 'SPY').
    signals_dict : Dict[str, Union[pd.DataFrame, pd.Series]], opzionale
        Dizionario di segnali fornito dall'utente. Se None, carica automaticamente
        tutti i file salvati in data/signals/ per l'asset specificato.
    save_path : Path o str, opzionale
        Percorso in cui salvare l'immagine (PNG, SVG o PDF). Se None, salva di default
        in data/results/plots/signal_distribution_{asset}.png.
    show_plot : bool, default False
        Se True, chiama plt.show() per visualizzazione a schermo.
    figsize : Tuple[int, int], default (15, 8)
        Dimensioni della figura in pollici (larghezza, altezza).
    title : str, opzionale
        Titolo personalizzato per il grafico.
        
    Ritorna:
    --------
    Tuple[plt.Figure, plt.Axes]
        Riferimento alla figura e all'asse Matplotlib per ulteriori personalizzazioni.
    """
    asset_upper = asset.upper()

    # Carica i segnali se non forniti esplicitamente
    if signals_dict is None or len(signals_dict) == 0:
        signals_dict = load_all_signals_for_asset(asset=asset_upper)

    if not signals_dict:
        raise ValueError(
            f"Nessun segnale trovato per l'asset '{asset_upper}'. "
            f"Assicurati che i file CSV siano presenti in data/signals/ oppure fornisci un dizionario valido."
        )

    # Estrazione e calcolo delle frequenze per ciascun agente
    agent_names: List[str] = []
    counts_data: Dict[int, List[int]] = {-1: [], 0: [], 1: []}
    pcts_data: Dict[int, List[float]] = {-1: [], 0: [], 1: []}
    total_samples: List[int] = []

    # Determina l'intervallo temporale comune
    date_start = None
    date_end = None

    for raw_name, item in signals_dict.items():
        # Risoluzione della serie di segnali
        if isinstance(item, pd.DataFrame):
            if "signal" in item.columns:
                s_series = item["signal"].astype(int)
            else:
                s_series = item.iloc[:, 0].astype(int)
        elif isinstance(item, pd.Series):
            s_series = item.astype(int)
        else:
            s_series = pd.Series(item, dtype=int)

        # Traccia estremi temporali
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

    # Creazione della figura con stile elegante e professionale
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=figsize, dpi=120)

    # Disegno delle barre raggruppate: Short (-1), Flat (0), Long (+1)
    rects_short = ax.bar(
        indices - bar_width,
        pcts_data[-1],
        width=bar_width,
        color=SIGNAL_COLORS[-1],
        edgecolor="#922B21",
        linewidth=1.2,
        label=SIGNAL_LABELS[-1],
        alpha=0.92,
        zorder=3,
    )

    rects_flat = ax.bar(
        indices,
        pcts_data[0],
        width=bar_width,
        color=SIGNAL_COLORS[0],
        edgecolor="#5D6D7E",
        linewidth=1.2,
        label=SIGNAL_LABELS[0],
        alpha=0.88,
        zorder=3,
    )

    rects_long = ax.bar(
        indices + bar_width,
        pcts_data[1],
        width=bar_width,
        color=SIGNAL_COLORS[1],
        edgecolor="#1E8449",
        linewidth=1.2,
        label=SIGNAL_LABELS[1],
        alpha=0.92,
        zorder=3,
    )

    # Aggiunta etichette sopra ogni barra (Percentuale + Conteggio)
    def _add_bar_labels(rects, signal_key: int):
        for idx, rect in enumerate(rects):
            h = rect.get_height()
            cnt = counts_data[signal_key][idx]
            if h >= 2.0:
                ax.annotate(
                    f"{h:.1f}%\n(n={cnt})",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#2C3E50",
                )
            elif h > 0.0:
                ax.annotate(
                    f"{h:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.0,
                    color="#555555",
                )

    _add_bar_labels(rects_short, -1)
    _add_bar_labels(rects_flat, 0)
    _add_bar_labels(rects_long, 1)

    # Evidenziazione visiva dell'Orchestratore (se presente)
    for idx, name in enumerate(agent_names):
        if "Orchestrator" in name:
            ax.axvspan(
                idx - 1.5 * bar_width,
                idx + 1.5 * bar_width,
                facecolor="#F4D03F",
                alpha=0.15,
                zorder=1,
                linestyle="--",
                linewidth=1.5,
                edgecolor="#D4AC0D",
            )
            ax.text(
                idx,
                -8.5,
                "[META-DECISORE]",
                ha="center",
                va="top",
                fontsize=8.5,
                fontweight="bold",
                color="#B7950B",
            )

    # Personalizzazione assi e griglia
    ax.set_xticks(indices)
    ax.set_xticklabels(agent_names, fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequenza Relativa (%)", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))

    # Limite Y con margine per non tagliare le etichette
    max_pct = max(max(pcts_data[-1]), max(pcts_data[0]), max(pcts_data[1]))
    ax.set_ylim(0, min(100.0, max(max_pct * 1.18, 50.0)))
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    # Titolo e Sottotitolo
    t_samples = total_samples[0] if total_samples else 0
    t_range_str = ""
    if date_start and date_end:
        t_range_str = f" | Periodo: {date_start.strftime('%Y-%m-%d')} -> {date_end.strftime('%Y-%m-%d')}"

    main_title = title or f"Distribuzione dei Segnali per Agente - Asset: {asset_upper}"
    subtitle = f"Totale Settimane: T = {t_samples}{t_range_str}"
    
    fig.suptitle(main_title, fontsize=15, fontweight="bold", y=0.98, color="#1A252F")
    ax.set_title(subtitle, fontsize=10.5, color="#566573", pad=12)

    # Legenda elegante in alto a destra
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        facecolor="#FFFFFF",
        edgecolor="#BDC3C7",
        fontsize=10.5,
        title="Segnale Operativo",
        title_fontsize=10.5,
    )
    legend.get_title().set_fontweight("bold")

    # Tabella riassuntiva inserita in calce al grafico (Net Exposure & Activity Rate)
    summary_rows = []
    for idx, name in enumerate(agent_names):
        short_pct = pcts_data[-1][idx]
        flat_pct = pcts_data[0][idx]
        long_pct = pcts_data[1][idx]
        net_bias = long_pct - short_pct
        active_rate = long_pct + short_pct
        summary_rows.append(f"{name}: Bias Netto={net_bias:+5.1f}% | A Mercato={active_rate:4.1f}% | Flat={flat_pct:4.1f}%")

    # Ottimizzazione layout
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    # Salvataggio dell'immagine su disco
    if save_path is None:
        plots_dir = PROJECT_ROOT / "data" / "results" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_path = plots_dir / f"signal_distribution_{asset.lower()}.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[{asset_upper}] Grafico della distribuzione salvato con successo in: {save_path.resolve()}")

    if show_plot:
        plt.show()

    return fig, ax


# =====================================================================
# 3. FUNZIONE BATCH PER TUTTI GLI ASSET
# =====================================================================
def plot_all_assets_signal_distributions(
    assets: Optional[List[str]] = None,
    signals_dir: Optional[Union[str, Path]] = None,
    save_dir: Optional[Union[str, Path]] = None,
    show_plots: bool = False,
) -> Dict[str, Tuple[plt.Figure, plt.Axes]]:
    """
    Genera e salva automaticamente il plot della distribuzione dei segnali
    per tutti gli asset specificati (default: SPY, IEF, GLD, SHY).
    """
    if assets is None:
        assets = ["SPY", "IEF", "GLD", "SHY"]

    plots_dir = Path(save_dir) if save_dir else (PROJECT_ROOT / "data" / "results" / "plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Tuple[plt.Figure, plt.Axes]] = {}

    print("\n" + "=" * 80)
    print("      GENERAZIONE GRAFICI DISTRIBUZIONE SEGNALI PER AGENTE (MULTI-ASSET)")
    print("=" * 80)

    for asset in assets:
        target_file = plots_dir / f"signal_distribution_{asset.lower()}.png"
        try:
            fig, ax = plot_agent_signal_distributions(
                asset=asset,
                signals_dir=signals_dir,
                save_path=target_file,
                show_plot=show_plots,
            )
            results[asset.upper()] = (fig, ax)
        except Exception as e:
            print(f"[{asset.upper()}] Errore durante la generazione del plot: {e}")

    print("=" * 80 + "\n")
    return results


# =====================================================================
# 4. ENTRY POINT CLI
# =====================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Script di generazione grafici e diagnostica segnali.")
    parser.add_argument(
        "target",
        nargs="?",
        default="SPY",
        help="Asset da plottare ('SPY', 'IEF', 'GLD', 'SHY' o 'ALL'). Default: 'SPY'",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Mostra i grafici a schermo tramite finestra interattiva",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory di destinazione per salvare i grafici",
    )

    args = parser.parse_args()

    if args.target.upper() == "ALL":
        plot_all_assets_signal_distributions(
            assets=["SPY", "IEF", "GLD", "SHY"],
            save_dir=args.output_dir,
            show_plots=args.show,
        )
    else:
        plot_agent_signal_distributions(
            asset=args.target.upper(),
            save_path=(Path(args.output_dir) / f"signal_distribution_{args.target.lower()}.png") if args.output_dir else None,
            show_plot=args.show,
        )
