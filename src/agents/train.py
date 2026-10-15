"""
train.py

Script di training e tuning dei parametri per gli agenti quantitativi:
- SMAEMAAgent (Trend-Following)
- ZScoreAgent (Mean-Reversion con isteresi)

Caratteristiche:
- Risolve i percorsi in modo robusto rispetto alla radice del repository.
- Esegue l'ottimizzazione Walk-Forward su ciascun asset dell'universo (TLT, IEF, SHY).
- Rispetta il vincolo di ricerca: l'OOS (dal 2022 in poi) rimane rigorosamente bloccato (max_year=2021).
- Salva i file JSON di configurazione in src/agents/config/.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configurazione percorsi
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.agents.EMA_EMW_agent import SMAEMAOptimizer, WalkForwardTrainer as SMAEMATrainer
from src.agents.ZScore_agent import ZScoreOptimizer, ZScoreWalkForwardTrainer


def main(target_asset: str = "TLT", show_plots: bool = False, max_train_year: int = 2021):
    """
    Esegue il training e la ricerca dei parametri ottimali.

    Parametri:
        target_asset: Asset di riferimento (es. 'TLT', 'IEF', 'SHY').
        show_plots: Se True, visualizza la matrice di correlazione a schermo.
        max_train_year: Anno massimo per il walk-forward (default 2021 per rispettare l'OOS lock).
    """
    config_dir = CURRENT_DIR / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    data_path = PROJECT_ROOT / "data" / "processed" / "features_weekly.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset delle feature non trovato in: {data_path}. "
            f"Assicurati di aver eseguito le fasi precedenti della pipeline."
        )

    # 1. Carica il dataset settimanale
    df = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")
    print(f"[Train] Dataset caricato: {len(df)} righe dal {df.index[0].date()} al {df.index[-1].date()}")

    # 2. Analisi di correlazione
    features_of_interest = [
        "SHY_return_1w", "IEF_return_1w", "TLT_return_1w",
        "rates__treasury_2y", "rates__treasury_10y",
        "treasury_2y_change_1w", "treasury_10y_change_1w",
        "curve_10y_2y", "curve_10y_3m", "curve_30y_10y",
        "cpi_yoy", "core_cpi_yoy", "inflation__breakeven_10y",
        "rate_volatility_13w"
    ]
    available_features = [f for f in features_of_interest if f in df.columns]
    corr_df = df[available_features]
    corr_matrix = corr_df.corr(method="pearson")

    print(f"\n=== Matrice di Correlazione ({len(available_features)} feature) ===")
    print(corr_matrix.round(2))

    if show_plots:
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            cbar_kws={"label": "Correlation Coefficient"},
        )
        plt.title("Correlation Matrix - Treasury Yields, Spreads & Macro Drivers", fontsize=14, pad=15)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    # 3. Selezione colonna prezzo
    price_col = f"{target_asset}_close"
    if price_col not in df.columns:
        raise KeyError(
            f"Colonna '{price_col}' non trovata nel dataset. Colonne disponibili: {list(df.columns)}"
        )

    print(f"\n=== Avvio Training per Asset: {target_asset} (Colonna: {price_col}) ===")
    print(f"Vincolo OOS: Valutazione limitata fino all'anno {max_train_year} (OOS 2022+ bloccato)")

    # -----------------------------------------------------------------
    # A. TRAIN AGENTE SMA/EMA (Trend-Following)
    # -----------------------------------------------------------------
    print("\n>>> Avvio Walk-Forward: SMA/EMA Agent...")
    sma_opt = SMAEMAOptimizer(price_col=price_col)
    sma_trainer = SMAEMATrainer(optimizer=sma_opt, train_window_years=5, price_col=price_col)
    sma_trainer.run(df, max_year=max_train_year)
    sma_output_file = config_dir / f"sma_ema_params_{target_asset.lower()}.json"
    sma_trainer.save_final_agent_config(sma_output_file)

    # -----------------------------------------------------------------
    # B. TRAIN AGENTE Z-SCORE (Mean-Reversion con Isteresi)
    # -----------------------------------------------------------------
    print("\n>>> Avvio Walk-Forward: Z-Score Agent (con uscita a rientro alla media)...")
    zscore_opt = ZScoreOptimizer(price_col=price_col, exit_threshold=0.0)
    zscore_trainer = ZScoreWalkForwardTrainer(optimizer=zscore_opt, train_window_years=5, price_col=price_col)
    zscore_trainer.run(df, max_year=max_train_year)
    zscore_output_file = config_dir / f"zscore_params_{target_asset.lower()}.json"
    zscore_trainer.save_final_agent_config(zscore_output_file)

    print(f"\n[OK] Training completato con successo. Configurazioni salvate in: '{config_dir.resolve()}'.")


if __name__ == "__main__":
    # Possibilità di passare l'asset da riga di comando o default TLT
    asset = sys.argv[1] if len(sys.argv) > 1 else "TLT"
    main(target_asset=asset, show_plots=False, max_train_year=2021)