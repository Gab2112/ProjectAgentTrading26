"""
EMA_EMW_agent.py

Agente statistico Trend-Following basato su Crossover tra Media Mobile Semplice (SMA)
e Media Mobile Esponenziale (EMA).
Eredita da BaseAgent per garantire un'interfaccia standardizzata.
"""

import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from .base_agent import BaseAgent


# =====================================================================
# 1. AGENTE INFERENZIALE SMA/EMA
# =====================================================================
class SMAEMAAgent(BaseAgent):
    """
    Agente inferenziale Trend-Following basato su Crossover SMA / EMA:
    - EMA (veloce) > SMA (lenta) -> BUY (+1)
    - EMA (veloce) <= SMA (lenta) -> SELL (-1)
    - Durante il periodo iniziale di warm-up -> HOLD (0)
    """

    def __init__(
        self,
        sma_window: Optional[int] = None,
        ema_window: Optional[int] = None,
        price_col: str = "close",
        output_format: str = "numeric",  # 'numeric' (-1, 0, 1) o 'string' ('SELL', 'HOLD', 'BUY')
    ):
        super().__init__(name="SMAEMAAgent", output_format=output_format)
        self.sma_window = sma_window
        self.ema_window = ema_window
        self.price_col = price_col

        if sma_window is not None and ema_window is not None:
            self.params = {
                "sma_window": int(sma_window),
                "ema_window": int(ema_window),
                "price_col": price_col,
            }

    def _apply_params(self, params: Dict[str, Any]) -> None:
        """Aggiorna i parametri interni dell'agente dopo il caricamento da JSON."""
        self.sma_window = int(params["sma_window"])
        self.ema_window = int(params["ema_window"])
        if "price_col" in params:
            self.price_col = str(params["price_col"])

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Calcola SMA ed EMA sui prezzi e restituisce i segnali operativi uniformati."""
        if self.sma_window is None or self.ema_window is None:
            raise ValueError(
                "Parametri non impostati. Inizializzali nel costruttore o usa .load_params()."
            )

        if self.price_col not in df.columns:
            raise KeyError(
                f"Colonna '{self.price_col}' non trovata nel DataFrame. "
                f"Colonne disponibili: {list(df.columns)}"
            )

        close = df[self.price_col].astype(float)
        sma = close.rolling(window=self.sma_window).mean()
        ema = close.ewm(span=self.ema_window, adjust=False).mean()

        # Condizione di Crossover: EMA > SMA -> BUY (+1), altrimenti SELL (-1)
        raw_signal = np.where(ema > sma, 1, -1)
        signals = pd.Series(raw_signal, index=df.index, dtype=int)

        # Durante il warm-up iniziale la SMA è NaN: impostiamo segnale a 0 (HOLD)
        signals[sma.isna()] = 0

        return self._format_signals(signals)


# Alias per retrocompatibilità
EMA_EMW_Agent = SMAEMAAgent


# =====================================================================
# 2. OTTIMIZZATORE SU SINGOLA FINESTRA (Train + Val)
# =====================================================================
class SMAEMAOptimizer:
    """
    Ottimizzatore su singolo blocco temporale:
    - Concatena train_df e val_df per garantire la continuità del warm-up.
    - Esegue una grid search sulle coppie (sma, ema) con vincolo ema < sma.
    - Applica shift(1) per evitare il look-ahead bias (si entra in posizione a t+1).
    - Valuta il rendimento cumulato della strategia unicamente sul periodo di validation.
    """

    def __init__(
        self,
        price_col: str = "close",
        sma_range: range = range(20, 101, 5),
        ema_range: range = range(5, 51, 5),
    ):
        self.price_col = price_col
        self.sma_range = sma_range
        self.ema_range = ema_range
        self.best_params: Optional[Dict[str, Any]] = None

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Any]:
        val_indices = val_df.index
        full_df = pd.concat([train_df, val_df])

        if self.price_col not in full_df.columns:
            raise KeyError(
                f"Colonna '{self.price_col}' non presente nei dati di train/validation."
            )

        close = full_df[self.price_col].astype(float)
        val_returns = val_df[self.price_col].pct_change().fillna(0.0)

        best_score = -np.inf
        best_params = None

        # Vincolo tecnico: EMA (veloce) deve avere finestra minore di SMA (lenta)
        param_grid = [
            (s, e) for s, e in itertools.product(self.sma_range, self.ema_range) if e < s
        ]

        for sma_w, ema_w in param_grid:
            sma = close.rolling(window=sma_w).mean()
            ema = close.ewm(span=ema_w, adjust=False).mean()

            # Posizione teorica: +1 se rialzista, -1 se ribassista
            pos = np.where(ema > sma, 1.0, -1.0)

            # SHIFT(1): il segnale calcolato alla chiusura di oggi opera sul rendimento di domani
            pos_series = (
                pd.Series(pos, index=full_df.index)
                .shift(1)
                .reindex(val_indices)
                .fillna(0.0)
            )

            strat_ret = pos_series * val_returns
            cum_score = float((1.0 + strat_ret).prod() - 1.0)

            if not np.isnan(cum_score) and cum_score > best_score:
                best_score = cum_score
                best_params = {
                    "sma_window": int(sma_w),
                    "ema_window": int(ema_w),
                    "price_col": self.price_col,
                    "val_return_pct": round(cum_score * 100, 2),
                }

        self.best_params = best_params
        return best_params

    def save_params(self, filepath: Union[str, Path]) -> None:
        """Salva i parametri ottimali su file JSON."""
        if self.best_params is None:
            raise ValueError("Nessun parametro trovato. Esegui prima .fit().")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.best_params, f, indent=4)


# =====================================================================
# 3. ORCHESTRATORE WALK-FORWARD (5 Anni Train -> 1 Anno Val/Test)
# =====================================================================
class WalkForwardTrainer:
    """
    Gestisce il ciclo di Walk-Forward:
    - Suddivide il DataFrame anno per anno.
    - Per ogni step estrae una finestra di 5 anni di training e l'anno successivo di validation.
    - Raccoglie la coppia (sma, ema) ottima per ciascun ciclo.
    - Trova la combinazione più frequente (moda statistica).
    """

    def __init__(
        self,
        optimizer: SMAEMAOptimizer,
        train_window_years: int = 5,
        price_col: str = "close",
    ):
        self.optimizer = optimizer
        self.train_window_years = train_window_years
        self.price_col = price_col
        self.history: List[Dict[str, Any]] = []
        self.selected_params: Optional[Dict[str, Any]] = None

    def run(self, df: pd.DataFrame, max_year: Optional[int] = None) -> Dict[str, Any]:
        """
        Esegue il loop rolling sull'intero dataset storico.

        Parametri:
            df: DataFrame storico con DatetimeIndex.
            max_year: Anno massimo consentito (es. 2021 per rispettare l'OOS lock).
        """
        data = df.copy()
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("L'indice del DataFrame deve essere un pd.DatetimeIndex.")

        data["year"] = data.index.year
        years = sorted(data["year"].unique())

        if max_year is not None:
            years = [y for y in years if y <= max_year]

        if len(years) <= self.train_window_years:
            raise ValueError(
                f"Servono almeno {self.train_window_years + 1} anni di dati. Presenti: {len(years)}."
            )

        self.history = []
        param_tuples = []

        print(
            f"Inizio Walk-Forward SMA/EMA: {len(years)} anni ({years[0]} - {years[-1]})\n"
        )

        for i in range(len(years) - self.train_window_years):
            train_span = years[i : i + self.train_window_years]
            val_year = years[i + self.train_window_years]

            train_df = data[data["year"].isin(train_span)]
            val_df = data[data["year"] == val_year]

            best = self.optimizer.fit(train_df, val_df)

            res = {
                "train_years": f"{train_span[0]}-{train_span[-1]}",
                "val_year": int(val_year),
                "sma_window": best["sma_window"],
                "ema_window": best["ema_window"],
                "val_return_pct": best["val_return_pct"],
            }
            self.history.append(res)
            param_tuples.append((best["sma_window"], best["ema_window"]))

            print(
                f"Train: [{res['train_years']}] -> Val {res['val_year']} | "
                f"Miglior coppia: SMA={res['sma_window']}, EMA={res['ema_window']} "
                f"(Rendimento Val: {res['val_return_pct']}%)"
            )

        # Selezione moda
        counts = Counter(param_tuples)
        most_common_pair, freq = counts.most_common(1)[0]

        self.selected_params = {
            "sma_window": most_common_pair[0],
            "ema_window": most_common_pair[1],
            "price_col": self.price_col,
            "frequency": f"{freq}/{len(param_tuples)} finestre",
            "all_combinations_tested": len(param_tuples),
        }

        print("\n" + "=" * 50)
        print(
            f"PARAMETRI SELEZIONATI (MODA): SMA={self.selected_params['sma_window']}, EMA={self.selected_params['ema_window']}"
        )
        print(f"Frequenza di apparizione: {self.selected_params['frequency']}")
        print("=" * 50)

        return self.selected_params

    def save_final_agent_config(self, filepath: Union[str, Path]) -> None:
        """Salva i parametri finali della moda in un JSON leggibile da SMAEMAAgent."""
        if self.selected_params is None:
            raise ValueError("Nessun parametro selezionato. Esegui prima .run().")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sma_window": self.selected_params["sma_window"],
                    "ema_window": self.selected_params["ema_window"],
                    "price_col": self.selected_params["price_col"],
                },
                f,
                indent=4,
            )
        print(f"Configurazione salvata con successo in: {path}")