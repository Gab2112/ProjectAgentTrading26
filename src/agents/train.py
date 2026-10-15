from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Importa direttamente gli orchestratori/classi dai tuoi script esistenti
# (adatta i percorsi del modulo se i tuoi file si chiamano diversamente)
from EMA_EMW_agent import SMAEMAOptimizer, WalkForwardTrainer as SMAEMATrainer
from ZScore_agent import ZScoreOptimizer, ZScoreWalkForwardTrainer


def main():
    # Cartella di output per i file JSON di configurazione

    config_dir = Path("config")
    config_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carica il dataset
    df = pd.read_csv("../../data/processed/features_weekly.csv", parse_dates=['Date'])

    # 2. Seleziona un sottoinsieme logico di feature significative
    # (evita di includere tutti gli open/high/low/close che saturano la matrice)
    features_of_interest = [
        # Rendimenti ETF
        'SHY_return_1w', 'IEF_return_1w', 'TLT_return_1w',
        # Rendimenti tassi e variazioni
        'rates__treasury_2y', 'rates__treasury_10y',
        'treasury_2y_change_1w', 'treasury_10y_change_1w',
        # Spread di curva
        'curve_10y_2y', 'curve_10y_3m', 'curve_30y_10y',
        # Inflazione e aspettative
        'cpi_yoy', 'core_cpi_yoy', 'inflation__breakeven_10y',
        # Volatilità
        'rate_volatility_13w'
    ]

    # Se preferisci includere TUTTE le colonne numeriche indistintamente, usa invece:
    # corr_df = df.select_dtypes(include='number')
    corr_df = df[features_of_interest]

    # 3. Calcolo matrice di correlazione (Pearson di default)
    corr_matrix = corr_df.corr(method='pearson')

    # Stampa i valori più correlati in console (opzionale)
    print("=== Matrice di Correlazione ===")
    print(corr_matrix.round(2))

    # 4. Visualizzazione Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation Coefficient'}
    )

    plt.title('Correlation Matrix - Treasury Yields, Spreads & Macro Drivers', fontsize=14, pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    corr_df = df.select_dtypes(include='number')

    corr_matrix = corr_df.corr(method='pearson')

    # Stampa i valori più correlati in console (opzionale)
    print("=== Matrice di Correlazione ===")
    print(corr_matrix.round(2))

    # 4. Visualizzazione Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation Coefficient'}
    )

    plt.title('Correlation Matrix - Treasury Yields, Spreads & Macro Drivers', fontsize=14, pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # 2. Caricamento del dataset
    # Assicurati che l'indice sia un pd.DatetimeIndex e che esista la colonna 'close'
    df = pd.read_csv("../../data/processed/features_weekly.csv", parse_dates=["Date"], index_col="Date")
    # -----------------------------------------------------------------
    # A. TRAIN AGENTE SMA/EMA (5 anni train -> 1 anno val -> Moda)
    # -----------------------------------------------------------------
    print(">>> Avvio Train: SMA/EMA Agent...")
    sma_opt = SMAEMAOptimizer(price_col="close")
    sma_trainer = SMAEMATrainer(optimizer=sma_opt, train_window_years=5)
    sma_trainer.run(df)
    sma_trainer.save_final_agent_config(config_dir / "sma_ema_params.json")

    # -----------------------------------------------------------------
    # B. TRAIN AGENTE Z-SCORE (5 anni train -> 1 anno val -> Moda)
    # -----------------------------------------------------------------
    print("\n>>> Avvio Train: Z-Score Agent...")
    zscore_opt = ZScoreOptimizer(price_col="close")
    zscore_trainer = ZScoreWalkForwardTrainer(optimizer=zscore_opt, train_window_years=5)
    zscore_trainer.run(df)
    zscore_trainer.save_final_agent_config(config_dir / "zscore_params.json")

    print(f"\n[OK] Training completato con successo. Configurazioni salvate in '{config_dir.resolve()}'.")


if __name__ == "__main__":
    main()