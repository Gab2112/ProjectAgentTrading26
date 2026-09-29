"""
signal_generation.py - Modulo per la generazione dei segnali operativi di trading.
"""

import pandas as pd
import numpy as np


class SignalGenerator:
    """Genera segnali decisionali basati sul confronto tra prezzi e indicatori."""

    def __init__(self, sma_col_name: str = "SMA_20"):
        self.sma_col_name = sma_col_name

    def generate_signals(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Genera il segnale di trading:
          +1 se Close(t-1) > SMA(t)
          -1 se Close(t-1) < SMA(t)
           0 se Close(t-1) == SMA(t)

        Parameters:
            df_features (pd.DataFrame): DataFrame contenente 'Date', 'Close' e la colonna SMA.

        Returns:
            pd.DataFrame: DataFrame con le colonne originali più 'Close_Prev' e 'Signal'.
        """
        print(f"[3/5] Generazione segnali con shift temporale (Close ieri vs {self.sma_col_name} oggi)...")

        if df_features is None or df_features.empty:
            raise ValueError("Il DataFrame in ingresso è vuoto o None.")

        if self.sma_col_name not in df_features.columns:
            raise KeyError(f"Colonna '{self.sma_col_name}' non trovata nel DataFrame.")

        if "Close" not in df_features.columns:
            raise KeyError("Colonna 'Close' non presente nel DataFrame.")

        df_signals = df_features.copy()

        # 1. Chiusura del giorno precedente (t-1)
        df_signals["Close_Prev"] = df_signals["Close"].shift(1)

        # 2. Condizioni logiche
        condition_long = df_signals["Close_Prev"] > df_signals[self.sma_col_name]
        condition_short = df_signals["Close_Prev"] < df_signals[self.sma_col_name]

        # 3. Assegnazione segnali (1, -1, 0)
        df_signals["Signal"] = np.select(
            [condition_long, condition_short],
            [1, -1],
            default=0
        )

        # Il primo record non ha il giorno precedente, viene impostato neutrale (0)
        df_signals.loc[df_signals["Close_Prev"].isna(), "Signal"] = 0

        # Cast a intero
        df_signals["Signal"] = df_signals["Signal"].astype(int)

        print(f"Segnali calcolati. Distribuzione:\n{df_signals['Signal'].value_counts().to_dict()}")

        return df_signals