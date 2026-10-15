"""
ZScore_agent.py

Agente statistico di Mean-Reversion basato su Z-Score rolling.
Implementa una logica a isteresi (state-machine) per correggere la chiusura prematura:
- Entry Long: Z-Score < -entry_threshold -> BUY (+1)
- Exit Long: Z-Score >= -exit_threshold (rientro alla media, default 0.0) -> FLAT (0)
- Entry Short: Z-Score > +entry_threshold -> SELL (-1)
- Exit Short: Z-Score <= +exit_threshold (rientro alla media, default 0.0) -> FLAT (0)
- All'interno della banda di transizione, la posizione preesistente viene mantenuta.
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
# LOGICA DI MEAN REVERSION A ISTERESI (STATO CONTINUO)
# =====================================================================
def compute_zscore_positions(
    z_scores: np.ndarray,
    entry_threshold: float,
    exit_threshold: float = 0.0,
) -> np.ndarray:
    """
    Calcola la serie delle posizioni operative applicando una corretta logica
    di Mean-Reversion a isteresi:
    - Entra in posizione quando Z supera la soglia estrema di deviazione.
    - Mantiene la posizione aperta finché Z non rientra verso la media (exit_threshold).
    - Evita l'uscita istantanea appena Z rientra di poco sotto entry_threshold.

    Parametri:
        z_scores: Array numpy con i valori dello Z-score.
        entry_threshold: Soglia di deviazione per l'ingresso (es. 1.5, 2.0).
        exit_threshold: Soglia di rientro verso la media per l'uscita (default 0.0 = media).

    Ritorna:
        np.ndarray di interi con valori in {-1, 0, 1}.
    """
    n = len(z_scores)
    positions = np.zeros(n, dtype=int)
    current_pos = 0

    for i in range(n):
        z = z_scores[i]

        if np.isnan(z):
            current_pos = 0
        elif current_pos == 0:
            if z < -entry_threshold:
                current_pos = 1  # BUY (prezzo anormalmente basso)
            elif z > entry_threshold:
                current_pos = -1  # SELL (prezzo anormalmente alto)
        elif current_pos == 1:
            # Siamo LONG: usciamo quando il prezzo risale alla media (o sopra -exit_threshold)
            if z >= -exit_threshold:
                current_pos = 0
                if z > entry_threshold:
                    current_pos = -1  # Cambio repentino di regime
        elif current_pos == -1:
            # Siamo SHORT: usciamo quando il prezzo riscende alla media (o sotto +exit_threshold)
            if z <= exit_threshold:
                current_pos = 0
                if z < -entry_threshold:
                    current_pos = 1  # Cambio repentino di regime

        positions[i] = current_pos

    return positions


# =====================================================================
# 1. AGENTE INFERENZIALE Z-SCORE
# =====================================================================
class ZScoreAgent(BaseAgent):
    """
    Agente inferenziale Mean-Reversion basato su Z-Score con isteresi di uscita:
    - Z < -threshold: prezzo anormalmente basso rispetto alla media mobile -> BUY (+1)
    - Z > +threshold: prezzo anormalmente alto rispetto alla media mobile -> SELL (-1)
    - Mantiene la posizione fino al ritorno alla media (|Z| <= exit_threshold).
    """

    def __init__(
        self,
        window: Optional[int] = None,
        threshold: Optional[float] = None,
        exit_threshold: float = 0.0,
        price_col: str = "close",
        output_format: str = "numeric",  # 'numeric' (-1, 0, 1) o 'string' ('SELL', 'HOLD', 'BUY')
    ):
        super().__init__(name="ZScoreAgent", output_format=output_format)
        self.window = window
        self.threshold = threshold
        self.exit_threshold = exit_threshold
        self.price_col = price_col

        if window is not None and threshold is not None:
            self.params = {
                "window": int(window),
                "threshold": float(threshold),
                "exit_threshold": float(exit_threshold),
                "price_col": price_col,
            }

    def _apply_params(self, params: Dict[str, Any]) -> None:
        """Aggiorna i parametri dell'agente dopo il caricamento da JSON."""
        self.window = int(params["window"])
        self.threshold = float(params["threshold"])
        self.exit_threshold = float(params.get("exit_threshold", 0.0))
        if "price_col" in params:
            self.price_col = str(params["price_col"])

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Calcola lo Z-Score e genera i segnali di mean reversion persistenti."""
        if self.window is None or self.threshold is None:
            raise ValueError(
                "Parametri non impostati. Inizializzali nel costruttore o usa .load_params()."
            )

        if self.price_col not in df.columns:
            raise KeyError(
                f"Colonna '{self.price_col}' non trovata nel DataFrame. "
                f"Colonne disponibili: {list(df.columns)}"
            )

        close = df[self.price_col].astype(float)
        mean = close.rolling(window=self.window).mean()
        std = close.rolling(window=self.window).std()

        # Protezione divisione per zero con soglia minima
        std_safe = std.replace(0, np.nan)
        z_score = (close - mean) / std_safe

        # Calcolo posizioni con la logica corretta a isteresi
        raw_signals = compute_zscore_positions(
            z_scores=z_score.to_numpy(),
            entry_threshold=self.threshold,
            exit_threshold=self.exit_threshold,
        )

        signals = pd.Series(raw_signals, index=df.index, dtype=int)
        # Periodo di warm-up iniziale dove std o mean sono NaN -> segnale 0 (HOLD)
        signals[z_score.isna()] = 0

        return self._format_signals(signals)


# =====================================================================
# 2. OTTIMIZZATORE Z-SCORE SU SINGOLO BLOCCO (Train + Val)
# =====================================================================
class ZScoreOptimizer:
    """
    Ottimizzatore Z-Score su singolo blocco:
    - Unisce train_df e val_df per preservare il warm-up iniziale del rolling.
    - Grid search sulle combinazioni (window, threshold) con exit_threshold per mean reversion.
    - Applica shift(1) sui segnali per evitare look-ahead bias.
    - Valuta il rendimento cumulato della strategia unicamente su val_df.
    """

    def __init__(
        self,
        price_col: str = "close",
        window_range: range = range(10, 81, 5),
        threshold_range: Optional[List[float]] = None,
        exit_threshold: float = 0.0,
    ):
        self.price_col = price_col
        self.window_range = window_range
        self.threshold_range = (
            threshold_range
            if threshold_range is not None
            else [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
        )
        self.exit_threshold = exit_threshold
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

        param_grid = list(itertools.product(self.window_range, self.threshold_range))

        for w, thresh in param_grid:
            mean = close.rolling(window=w).mean()
            std = close.rolling(window=w).std()
            std_safe = std.replace(0, np.nan)
            z = (close - mean) / std_safe

            # Applicazione della logica corretta con uscita a ritorno alla media
            pos_array = compute_zscore_positions(
                z_scores=z.to_numpy(),
                entry_threshold=float(thresh),
                exit_threshold=self.exit_threshold,
            )

            # SHIFT(1): il segnale generato a chiusura t opera sui rendimenti di t+1
            pos_series = (
                pd.Series(pos_array, index=full_df.index, dtype=float)
                .shift(1)
                .reindex(val_indices)
                .fillna(0.0)
            )

            strat_ret = pos_series * val_returns
            cum_score = float((1.0 + strat_ret).prod() - 1.0)

            if not np.isnan(cum_score) and cum_score > best_score:
                best_score = cum_score
                best_params = {
                    "window": int(w),
                    "threshold": float(thresh),
                    "exit_threshold": float(self.exit_threshold),
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
# 3. ORCHESTRATORE WALK-FORWARD Z-SCORE (5 Anni Train -> 1 Anno Val)
# =====================================================================
class ZScoreWalkForwardTrainer:
    """
    Gestisce il Walk-Forward temporale:
    - Finestra mobile: 5 anni di train -> ottimizzazione sull'anno successivo di validation.
    - Colleziona la migliore coppia (window, threshold) per ciascun ciclo.
    - Seleziona la combinazione più frequente (Moda statistica).
    """

    def __init__(
        self,
        optimizer: ZScoreOptimizer,
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
        Esegue il loop rolling sul dataset.

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
            f"Inizio Walk-Forward Z-Score: {len(years)} anni ({years[0]} - {years[-1]})\n"
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
                "window": best["window"],
                "threshold": best["threshold"],
                "exit_threshold": best["exit_threshold"],
                "val_return_pct": best["val_return_pct"],
            }
            self.history.append(res)
            param_tuples.append((best["window"], best["threshold"]))

            print(
                f"Train: [{res['train_years']}] -> Val {res['val_year']} | "
                f"Miglior coppia: Window={res['window']}, Threshold={res['threshold']} "
                f"(Rendimento Val: {res['val_return_pct']}%)"
            )

        # Selezione moda
        counts = Counter(param_tuples)
        most_common_pair, freq = counts.most_common(1)[0]

        self.selected_params = {
            "window": most_common_pair[0],
            "threshold": most_common_pair[1],
            "exit_threshold": self.optimizer.exit_threshold,
            "price_col": self.price_col,
            "frequency": f"{freq}/{len(param_tuples)} finestre",
            "all_combinations_tested": len(param_tuples),
        }

        print("\n" + "=" * 55)
        print(
            f"PARAMETRI Z-SCORE SELEZIONATI (MODA): Window={self.selected_params['window']}, "
            f"Threshold={self.selected_params['threshold']}, ExitThreshold={self.selected_params['exit_threshold']}"
        )
        print(f"Frequenza di apparizione: {self.selected_params['frequency']}")
        print("=" * 55)

        return self.selected_params

    def save_final_agent_config(self, filepath: Union[str, Path]) -> None:
        """Salva i parametri della moda in un JSON direttamente leggibile da ZScoreAgent."""
        if self.selected_params is None:
            raise ValueError("Nessun parametro selezionato. Esegui prima .run().")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "window": self.selected_params["window"],
                    "threshold": self.selected_params["threshold"],
                    "exit_threshold": self.selected_params["exit_threshold"],
                    "price_col": self.selected_params["price_col"],
                },
                f,
                indent=4,
            )
        print(f"Configurazione Z-Score salvata con successo in: {path}")