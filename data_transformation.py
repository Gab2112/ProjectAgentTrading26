"""
data_transformation.py - Modulo per la pulizia dei dati e feature engineering.
"""
from typing import Tuple
import pandas as pd
import numpy as np


class DataTransformer:
    """Pulisce, isola i prezzi di chiusura e calcola le feature di mercato."""

    def __init__(self):
        pass

    def clean_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Isola solo la colonna del prezzo di chiusura ('Close') ed elimina 
        i record con valori nulli (NaN).

        Supporta sia DataFrame a singolo ticker sia MultiIndex di yfinance.
        """
        print("[2/5] Pulizia dati: estrazione Close price e rimozione NaN...")

        if raw_data is None or raw_data.empty:
            raise ValueError("Il DataFrame in ingresso è vuoto o None.")

        # Gestione del caso MultiIndex di yfinance (quando si scaricano più ticker)
        if isinstance(raw_data.columns, pd.MultiIndex):
            # Cerca il livello contenente 'Close'
            if "Close" in raw_data.columns.get_level_values(0):
                df_close = raw_data["Close"].copy()
            elif "Close" in raw_data.columns.get_level_values(1):
                df_close = raw_data.xs("Close", axis=1, level=1).copy()
            else:
                raise KeyError("Colonna 'Close' non trovata nei livelli MultiIndex del DataFrame.")
        else:
            # Caso singolo ticker o tabella standard
            if "Close" in raw_data.columns:
                df_close = raw_data[["Close"]].copy()
            else:
                raise KeyError("Colonna 'Close' non presente nel DataFrame.")

        # Rimuove le righe che contengono anche solo un valore nullo
        cleaned_data = df_close.dropna()

        # Verifica che i dati siano ordinati per data crescente
        cleaned_data = cleaned_data.sort_index()

        print(f"Righe rimosse: {len(df_close) - len(cleaned_data)} | Record validi: {len(cleaned_data)}")
        return cleaned_data

    import pandas as pd

    def engineer_features(
            self, cleaned_data: pd.DataFrame, window: int = 20
    ) -> pd.DataFrame:
        """
        Calcola la media mobile sui prezzi di chiusura e struttura l'output
        con le sole colonne: 'Date', 'Close' e 'SMA_<window>'.

        Parameters:
            cleaned_data (pd.DataFrame): DataFrame con prezzi di chiusura indicizzati per data.
            window (int): Ampiezza della finestra temporale per la media mobile (default: 20).

        Returns:
            pd.DataFrame: DataFrame con colonne esatte ['Date', 'Close', f'SMA_{window}'].
        """
        print(f"[2/5] Feature engineering: calcolo SMA con finestra {window}...")

        if cleaned_data is None or cleaned_data.empty:
            raise ValueError("Il DataFrame cleaned_data fornito è vuoto o nullo.")

        if window <= 0:
            raise ValueError(f"Il parametro window deve essere > 0, ricevuto: {window}")

        df_features = cleaned_data.copy()

        # Se c'è una sola colonna ma non si chiama 'Close', rinominala per consistenza
        if len(df_features.columns) == 1 and df_features.columns[0] != "Close":
            df_features.columns = ["Close"]
        elif "Close" not in df_features.columns:
            raise KeyError("Colonna 'Close' non trovata in cleaned_data.")

        # 1. Calcolo della media mobile semplice (SMA)
        sma_col_name = f"SMA_{window}"
        df_features[sma_col_name] = df_features["Close"].rolling(window=window).mean()

        # 2. Rimozione dei primi record (NaN dovuti alla finestra di calcolo)
        df_features = df_features.dropna()

        # 3. Trasformazione dell'indice temporale in una colonna esplicita 'Date'
        df_features = df_features.reset_index()
        # Rinomina la colonna dell'indice se ha un nome diverso (es. 'Datetime', 'index', ecc.)
        date_col = df_features.columns[0]
        if date_col != "Date":
            df_features = df_features.rename(columns={date_col: "Date"})

        # 4. Selezione rigorosa delle tre colonne richieste
        expected_cols = ["Date", "Close", sma_col_name]
        df_final = df_features[expected_cols].copy()

        # =========================================================================
        # Check di validazione finale
        # =========================================================================
        assert list(df_final.columns) == expected_cols, (
            f"Errore struttura: colonne attese {expected_cols}, ottenute {list(df_final.columns)}"
        )
        assert not df_final.empty, "Errore: il DataFrame finale è vuoto dopo il dropna."
        assert df_final[expected_cols].isna().sum().sum() == 0, (
            "Errore: presenti valori nulli residui nel DataFrame finale."
        )

        print(f"Check superato con successo. Dimensioni finali: {df_final.shape}")
        return df_final


    def split_train_test(
            self,
            data: pd.DataFrame,
            train_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Esegue uno split cronologico (time-series split) senza shuffle.

        Parameters:
            data (pd.DataFrame): DataFrame indicizzato temporalmente e ordinato.
            train_ratio (float): Quota percentuale riservata al train set (default 0.8 -> 80%).

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
        """
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio deve essere un valore compreso tra 0 e 1.")

        if data.empty:
            raise ValueError("Il DataFrame passato allo split è vuoto.")

        # Calcolo dell'indice di taglio
        split_idx = int(len(data) * train_ratio)

        # Taglio rigorosamente cronologico
        train_df = data.iloc[:split_idx].copy()
        test_df = data.iloc[split_idx:].copy()

        # Log di verifica delle finestre temporali
        print(
            f"[Split Cronologico] Totale: {len(data)} righe | "
            f"Train: {len(train_df)} ({train_df.index.min().strftime('%Y-%m-%d')} -> {train_df.index.max().strftime('%Y-%m-%d')}) | "
            f"Test: {len(test_df)} ({test_df.index.min().strftime('%Y-%m-%d')} -> {test_df.index.max().strftime('%Y-%m-%d')})"
        )

        return train_df, test_df