import itertools
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd


# =====================================================================
# 1. AGENTE INFERENZIALE Z-SCORE
# =====================================================================
class ZScoreAgent:
    """
    Agente inferenziale Mean-Reversion basato su Z-Score:
    - Z < -threshold: prezzo anormalmente basso rispetto alla media mobile -> BUY (1)
    - Z > +threshold: prezzo anormalmente alto rispetto alla media mobile -> SELL (-1)
    - Altrimenti: prezzo nel range di tolleranza -> HOLD (0)
    """

    def __init__(
        self,
        window: int | None = None,
        threshold: float | None = None,
        price_col: str = "close",
        output_format: str = "numeric",  # 'numeric' (-1, 0, 1) o 'string' ('SELL', 'HOLD', 'BUY')
    ):
        self.window = window
        self.threshold = threshold
        self.price_col = price_col
        self.output_format = output_format

    def load_params(self, filepath: str | Path):
        """Carica i parametri ottimali da file JSON."""
        path = Path(filepath)
        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)
        self.window = int(params["window"])
        self.threshold = float(params["threshold"])
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Calcola lo Z-Score e genera i segnali di mean reversion."""
        if self.window is None or self.threshold is None:
            raise ValueError("Parametri non impostati. Inizializzali o usa .load_params().")

        close = df[self.price_col]
        mean = close.rolling(window=self.window).mean()
        std = close.rolling(window=self.window).std()

        # Calcolo corretto Z-Score: (Prezzo - Media) / Deviazione Standard
        z_score = (close - mean) / std

        # Logica Mean-Reversion
        raw_signal = np.where(z_score < -self.threshold, 1, 0)
        raw_signal = np.where(z_score > self.threshold, -1, raw_signal)

        signals = pd.Series(raw_signal, index=df.index, dtype=int)
        # Periodo di warm-up iniziale dove std e mean sono NaN
        signals[z_score.isna()] = 0

        if self.output_format == "string":
            return signals.map({1: "BUY", -1: "SELL", 0: "HOLD"})
        return signals


# =====================================================================
# 2. OTTIMIZZATORE Z-SCORE SU SINGOLO BLOCCO (Train + Val)
# =====================================================================
class ZScoreOptimizer:
    """
    Ottimizzatore Z-Score su singolo blocco:
    - Unisce train_df e val_df per preservare il warm-up iniziale del rolling.
    - Grid search sulle combinazioni (window, threshold).
    - Applica shift(1) sui segnali per evitare look-ahead bias.
    - Valuta il rendimento cumulato della strategia unicamente su val_df.
    """

    def __init__(
        self,
        price_col: str = "close",
        window_range: range = range(10, 81, 5),
        threshold_range: list[float] | None = None,
    ):
        self.price_col = price_col
        self.window_range = window_range
        self.threshold_range = (
            threshold_range
            if threshold_range is not None
            else [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
        )
        self.best_params = None

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
        val_indices = val_df.index
        # Concateniamo per non perdere i primi N valori di validation
        full_df = pd.concat([train_df, val_df])
        close = full_df[self.price_col]
        val_returns = val_df[self.price_col].pct_change().fillna(0.0)

        best_score = -np.inf
        best_params = None

        param_grid = list(itertools.product(self.window_range, self.threshold_range))

        for w, thresh in param_grid:
            mean = close.rolling(window=w).mean()
            std = close.rolling(window=w).std()

            # Correzione Z-Score con sottrazione
            z = (close - mean) / std

            # Posizioni operative: 1 (BUY), -1 (SELL), 0 (HOLD/FLAT)
            pos = np.where(z < -thresh, 1.0, 0.0)
            pos = np.where(z > thresh, -1.0, pos)

            # SHIFT(1): il segnale di fine giornata opera sui rendimenti del giorno successivo
            pos_series = (
                pd.Series(pos, index=full_df.index)
                .shift(1)
                .reindex(val_indices)
                .fillna(0.0)
            )

            # Correzione moltiplicazione per calcolo rendimenti composti
            strat_ret = pos_series * val_returns
            cum_score = (1.0 + strat_ret).prod() - 1.0

            if not np.isnan(cum_score) and cum_score > best_score:
                best_score = cum_score
                best_params = {
                    "window": int(w),
                    "threshold": float(thresh),
                    "val_return_pct": round(float(cum_score) * 100, 2),
                }

        self.best_params = best_params
        return best_params

    def save_params(self, filepath: str | Path):
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
        price_col: str = "close"
    ):
        self.optimizer = optimizer
        self.train_window_years = train_window_years
        self.price_col = price_col
        self.history: list[dict] = []
        self.selected_params: dict | None = None

    def run(self, df: pd.DataFrame) -> dict:
        data = df.copy()
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("L'indice del DataFrame deve essere un pd.DatetimeIndex.")

        data["year"] = data.index.year
        years = sorted(data["year"].unique())

        if len(years) <= self.train_window_years:
            raise ValueError(
                f"Servono almeno {self.train_window_years + 1} anni di dati. Presenti: {len(years)}."
            )

        self.history = []
        param_tuples = []

        print(f"Inizio Walk-Forward Z-Score: {len(years)} anni totali ({years[0]} - {years[-1]})\n")

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
                "val_return_pct": best["val_return_pct"],
            }
            self.history.append(res)
            param_tuples.append((best["window"], best["threshold"]))

            print(
                f"Train: [{res['train_years']}] -> Test {res['val_year']} | "
                f"Miglior coppia: Window={res['window']}, Threshold={res['threshold']} "
                f"(Rendimento Val: {res['val_return_pct']}%)"
            )

        # -------------------------------------------------------------
        # Selezione della combinazione più frequente (Moda)
        # -------------------------------------------------------------
        counts = Counter(param_tuples)
        most_common_pair, freq = counts.most_common(1)[0]

        self.selected_params = {
            "window": most_common_pair[0],
            "threshold": most_common_pair[1],
            "frequency": f"{freq}/{len(param_tuples)} finestre",
            "all_combinations_tested": len(param_tuples)
        }

        print("\n" + "=" * 55)
        print(f"PARAMETRI Z-SCORE SELEZIONATI (MODA): Window={self.selected_params['window']}, Threshold={self.selected_params['threshold']}")
        print(f"Frequenza di apparizione: {self.selected_params['frequency']}")
        print("=" * 55)

        return self.selected_params

    def save_final_agent_config(self, filepath: str | Path):
        """Salva i parametri della moda in un JSON direttamente leggibile da ZScoreAgent."""
        if self.selected_params is None:
            raise ValueError("Nessun parametro selezionato. Esegui prima .run().")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "window": self.selected_params["window"],
                "threshold": self.selected_params["threshold"]
            }, f, indent=4)
        print(f"Configurazione Z-Score salvata con successo in: {path}")