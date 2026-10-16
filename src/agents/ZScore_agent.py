"""
ZScore_agent.py

Agente statistico basato su Z-Score rolling con supporto per:
- Mean-Reversion a isteresi (ritorno alla media su mercati laterali)
- Momentum / Breakout a isteresi (inseguimento trend su regimi direzionali)
- Calcolo congiunto di Segnali discreti {-1, 0, 1} e Confidenza continua [0.0, 1.0]
- Ottimizzazione Walk-Forward (5y IS -> 1y OOS) orientata alla stabilita e penalizzazione del Max Drawdown.
"""

import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .base_agent import BaseAgent, calculate_performance_metrics, compute_objective_score


# =====================================================================
# LOGICA DI TRADING SU Z-SCORE A ISTERESI
# =====================================================================
def compute_zscore_positions(
    z_scores: np.ndarray,
    entry_threshold: float,
    exit_threshold: float = 0.0,
    mode: str = "momentum",
) -> np.ndarray:
    """
    Calcola la serie delle posizioni operative basate su Z-score con isteresi.

    Modalita:
    - mode == 'momentum' (default per obbligazionario in trend macro):
        * Z > +entry_threshold: Breakout rialzista -> BUY (+1)
        * Z <= +exit_threshold: Rientro / esaurimento -> FLAT (0)
        * Z < -entry_threshold: Breakdown ribassista -> SELL (-1)
        * Z >= -exit_threshold: Rientro / esaurimento -> FLAT (0)
    - mode == 'mean_reversion':
        * Z < -entry_threshold: Ipervenduto anomalo -> BUY (+1)
        * Z >= -exit_threshold: Ritorno alla media -> FLAT (0)
        * Z > +entry_threshold: Ipercomprato anomalo -> SELL (-1)
        * Z <= +exit_threshold: Ritorno alla media -> FLAT (0)

    Parametri:
        z_scores: Array numpy con i valori dello Z-score.
        entry_threshold: Soglia di deviazione standard per l'ingresso.
        exit_threshold: Soglia di rientro verso la media per l'uscita (default 0.0).
        mode: 'momentum' oppure 'mean_reversion'.

    Ritorna:
        np.ndarray di interi con valori in {-1, 0, 1}.
    """
    n = len(z_scores)
    positions = np.zeros(n, dtype=int)
    current_pos = 0

    if mode == "momentum":
        for i in range(n):
            z = z_scores[i]
            if np.isnan(z):
                current_pos = 0
            elif current_pos == 0:
                if z > entry_threshold:
                    current_pos = 1
                elif z < -entry_threshold:
                    current_pos = -1
            elif current_pos == 1:
                if z <= exit_threshold:
                    current_pos = 0
                    if z < -entry_threshold:
                        current_pos = -1
            elif current_pos == -1:
                if z >= -exit_threshold:
                    current_pos = 0
                    if z > entry_threshold:
                        current_pos = 1
            positions[i] = current_pos

    else:  # mean_reversion
        for i in range(n):
            z = z_scores[i]
            if np.isnan(z):
                current_pos = 0
            elif current_pos == 0:
                if z < -entry_threshold:
                    current_pos = 1
                elif z > entry_threshold:
                    current_pos = -1
            elif current_pos == 1:
                if z >= -exit_threshold:
                    current_pos = 0
                    if z > entry_threshold:
                        current_pos = -1
            elif current_pos == -1:
                if z <= exit_threshold:
                    current_pos = 0
                    if z < -entry_threshold:
                        current_pos = 1
            positions[i] = current_pos

    return positions


# =====================================================================
# 1. AGENTE INFERENZIALE Z-SCORE
# =====================================================================
class ZScoreAgent(BaseAgent):
    """
    Agente inferenziale basato su Z-Score rolling con supporto per modalita
    momentum (breakout) o mean-reversion, e calcolo della confidenza.
    """

    def __init__(
        self,
        window: Optional[int] = None,
        threshold: Optional[float] = None,
        exit_threshold: float = 0.0,
        mode: str = "momentum",
        price_col: str = "close",
        output_format: str = "numeric",  # 'numeric' (-1, 0, 1) o 'string' ('SELL', 'HOLD', 'BUY')
    ):
        super().__init__(name="ZScoreAgent", output_format=output_format)
        self.window = window
        self.threshold = threshold
        self.exit_threshold = exit_threshold
        self.mode = mode
        self.price_col = price_col

        if window is not None and threshold is not None:
            self.params = {
                "window": int(window),
                "threshold": float(threshold),
                "exit_threshold": float(exit_threshold),
                "mode": str(mode),
                "price_col": price_col,
            }

    def _apply_params(self, params: Dict[str, Any]) -> None:
        """Aggiorna i parametri dell'agente dopo il caricamento da JSON."""
        self.window = int(params["window"])
        self.threshold = float(params["threshold"])
        self.exit_threshold = float(params.get("exit_threshold", 0.0))
        self.mode = str(params.get("mode", "momentum"))
        if "price_col" in params:
            self.price_col = str(params["price_col"])

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Calcola lo Z-Score e genera i segnali discreti {-1, 0, 1}."""
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

        std_safe = std.replace(0, np.nan)
        z_score = (close - mean) / std_safe

        raw_signals = compute_zscore_positions(
            z_scores=z_score.to_numpy(),
            entry_threshold=self.threshold,
            exit_threshold=self.exit_threshold,
            mode=self.mode,
        )

        signals = pd.Series(raw_signals, index=df.index, dtype=int)
        signals[z_score.isna()] = 0

        return self._format_signals(signals)

    def predict_with_confidence(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calcola congiuntamente i segnali operativi e la relativa confidenza.

        La confidenza e funzione dell'intensita della deviazione standard:
        - Se in posizione attiva (|signal| == 1), la confidenza parte da 0.50 e cresce verso 0.95+.
        - Durante il flat o warm-up (signal == 0), la confidenza e 0.0.
        """
        signals = self.predict(df)

        close = df[self.price_col].astype(float)
        mean = close.rolling(window=self.window).mean()
        std = close.rolling(window=self.window).std().replace(0, np.nan)
        z = ((close - mean) / std).abs().fillna(0.0)

        thresh = max(float(self.threshold), 0.5)
        conf_values = np.where(
            signals != 0,
            0.50 + 0.45 * np.tanh(z / thresh),
            0.0,
        )
        confidence = pd.Series(conf_values, index=df.index, dtype=float).round(4)
        return signals, confidence


# =====================================================================
# 2. OTTIMIZZATORE Z-SCORE SU SINGOLO BLOCCO (Train IS + Val OOS)
# =====================================================================
class ZScoreOptimizer:
    """
    Ottimizzatore Z-Score su singolo blocco temporale:
    - Esegue la grid search su (window, threshold, mode) RIGOROSAMENTE sul Train (IS, 5 anni).
    - Valuta secondo una funzione obiettivo basata sulla stabilita/rischio ('sharpe_mdd').
    - Valuta le metriche Out-Of-Sample sull'anno successivo di Validation (OOS) senza re-ottimizzare.
    - Confronta le performance In-Sample e Out-Of-Sample per rilevare l'overfitting.
    """

    def __init__(
        self,
        price_col: str = "close",
        window_range: Optional[Union[range, List[int]]] = None,
        threshold_range: Optional[List[float]] = None,
        mode_range: Optional[List[str]] = None,
        exit_threshold: float = 0.0,
        objective_metric: str = "sharpe_mdd",
    ):
        self.price_col = price_col
        self.window_range = window_range if window_range is not None else [15, 20, 25, 30, 40, 50, 60]
        self.threshold_range = (
            threshold_range
            if threshold_range is not None
            else [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        )
        self.mode_range = mode_range if mode_range is not None else ["momentum", "mean_reversion"]
        self.exit_threshold = exit_threshold
        self.objective_metric = objective_metric
        self.best_params: Optional[Dict[str, Any]] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        warmup_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Esegue la grid search sui soli dati di Train (In-Sample) e valuta la performance
        sia su Train che su Validation (Out-Of-Sample).
        """
        dfs_to_concat = [df for df in [warmup_df, train_df, val_df] if df is not None and not df.empty]
        full_df = pd.concat(dfs_to_concat)

        if self.price_col not in full_df.columns:
            raise KeyError(
                f"Colonna '{self.price_col}' non presente nei dati di train/validation."
            )

        train_indices = train_df.index
        val_indices = val_df.index if val_df is not None else None

        close = full_df[self.price_col].astype(float)
        full_returns = close.pct_change().fillna(0.0)

        best_score = -np.inf
        best_w = None
        best_thresh = None
        best_mode = None
        best_is_metrics = None

        param_grid = list(itertools.product(self.window_range, self.threshold_range, self.mode_range))

        for w, thresh, m in param_grid:
            mean = close.rolling(window=w).mean()
            std = close.rolling(window=w).std()
            std_safe = std.replace(0, np.nan)
            z = (close - mean) / std_safe

            pos_array = compute_zscore_positions(
                z_scores=z.to_numpy(),
                entry_threshold=float(thresh),
                exit_threshold=self.exit_threshold,
                mode=m,
            )

            # SHIFT(1): il segnale generato a chiusura t opera sui rendimenti di t+1
            pos_series = pd.Series(pos_array, index=full_df.index, dtype=float).shift(1)

            # Valutazione RIGOROSA solo su periodo di TRAINING (In-Sample)
            train_pos = pos_series.reindex(train_indices).fillna(0.0)
            train_strat_ret = train_pos * full_returns.reindex(train_indices).fillna(0.0)

            is_metrics = calculate_performance_metrics(train_strat_ret)
            score = compute_objective_score(is_metrics, metric_type=self.objective_metric)

            if not np.isnan(score) and score > best_score:
                best_score = score
                best_w = w
                best_thresh = thresh
                best_mode = m
                best_is_metrics = is_metrics

        if best_w is None or best_thresh is None or best_mode is None:
            raise RuntimeError("Nessuna combinazione valida trovata durante l'ottimizzazione.")

        # Valutazione Out-Of-Sample (Validation 1 anno successivo) con i parametri ottimi scelti
        oos_metrics = None
        if val_df is not None and not val_df.empty:
            mean_opt = close.rolling(window=best_w).mean()
            std_opt = close.rolling(window=best_w).std().replace(0, np.nan)
            z_opt = (close - mean_opt) / std_opt

            pos_opt = compute_zscore_positions(
                z_scores=z_opt.to_numpy(),
                entry_threshold=float(best_thresh),
                exit_threshold=self.exit_threshold,
                mode=best_mode,
            )
            pos_series_opt = pd.Series(pos_opt, index=full_df.index, dtype=float).shift(1)

            val_pos = pos_series_opt.reindex(val_indices).fillna(0.0)
            val_strat_ret = val_pos * full_returns.reindex(val_indices).fillna(0.0)
            oos_metrics = calculate_performance_metrics(val_strat_ret)

        is_ret_pct = round(best_is_metrics["cumulative_return"] * 100, 2)
        is_sharpe = round(best_is_metrics["sharpe_ratio"], 2)
        is_mdd_pct = round(best_is_metrics["max_drawdown"] * 100, 2)

        oos_ret_pct = round(oos_metrics["cumulative_return"] * 100, 2) if oos_metrics else None
        oos_sharpe = round(oos_metrics["sharpe_ratio"], 2) if oos_metrics else None
        oos_mdd_pct = round(oos_metrics["max_drawdown"] * 100, 2) if oos_metrics else None

        overfitting_sharpe_drop = (
            round(is_sharpe - oos_sharpe, 2) if (oos_sharpe is not None) else None
        )

        self.best_params = {
            "window": int(best_w),
            "threshold": float(best_thresh),
            "exit_threshold": float(self.exit_threshold),
            "mode": str(best_mode),
            "price_col": self.price_col,
            "objective_metric": self.objective_metric,
            "train_score": round(float(best_score), 4),
            "is_return_pct": is_ret_pct,
            "is_sharpe": is_sharpe,
            "is_mdd_pct": is_mdd_pct,
            "oos_return_pct": oos_ret_pct,
            "oos_sharpe": oos_sharpe,
            "oos_mdd_pct": oos_mdd_pct,
            "overfitting_sharpe_drop": overfitting_sharpe_drop,
            "train_metrics": best_is_metrics,
            "val_metrics": oos_metrics,
            "val_return_pct": oos_ret_pct,
        }
        return self.best_params

    def save_params(self, filepath: Union[str, Path]) -> None:
        """Salva i parametri ottimali su file JSON."""
        if self.best_params is None:
            raise ValueError("Nessun parametro trovato. Esegui prima .fit().")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.best_params, f, indent=4)


# =====================================================================
# 3. ORCHESTRATORE WALK-FORWARD Z-SCORE (5 Anni Train IS -> 1 Anno Val OOS)
# =====================================================================
class ZScoreWalkForwardTrainer:
    """
    Gestisce il Walk-Forward temporale per Z-Score:
    - Finestra mobile: 5 anni di train (IS) -> test sull'anno successivo di validation (OOS).
    - Ottimizza sul training con la metrica di stabilita/rischio desiderata ('sharpe_mdd').
    - Rileva l'overfitting confrontando In-Sample e Out-Of-Sample.
    - Seleziona la combinazione (window, threshold, mode) piu frequente (Moda statistica).
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
            f"\nInizio Walk-Forward Z-Score: {len(years)} anni ({years[0]} - {years[-1]}) | "
            f"Metrica: '{self.optimizer.objective_metric}'"
        )
        print("-" * 120)

        for i in range(len(years) - self.train_window_years):
            train_span = years[i : i + self.train_window_years]
            val_year = years[i + self.train_window_years]

            warmup_years = years[:i]
            warmup_df = data[data["year"].isin(warmup_years)] if len(warmup_years) > 0 else None
            train_df = data[data["year"].isin(train_span)]
            val_df = data[data["year"] == val_year]

            best = self.optimizer.fit(train_df, val_df, warmup_df=warmup_df)

            res = {
                "train_years": f"{train_span[0]}-{train_span[-1]}",
                "val_year": int(val_year),
                "window": best["window"],
                "threshold": best["threshold"],
                "exit_threshold": best["exit_threshold"],
                "mode": best["mode"],
                "is_return_pct": best["is_return_pct"],
                "is_sharpe": best["is_sharpe"],
                "is_mdd_pct": best["is_mdd_pct"],
                "oos_return_pct": best["oos_return_pct"],
                "oos_sharpe": best["oos_sharpe"],
                "oos_mdd_pct": best["oos_mdd_pct"],
                "overfitting_sharpe_drop": best["overfitting_sharpe_drop"],
                "val_return_pct": best["val_return_pct"],
            }
            self.history.append(res)
            param_tuples.append((best["window"], best["threshold"], best["mode"]))

            is_sh = res["is_sharpe"]
            oos_sh = res["oos_sharpe"]
            overfit_alert = " [OVERFIT]" if (is_sh > 0.4 and oos_sh is not None and oos_sh < 0) else ""

            print(
                f"Finestra {i+1:02d} | Train: [{res['train_years']}] -> Test: {res['val_year']} | "
                f"Ottimo: Win={res['window']:2d}, Thresh={res['threshold']:.2f}, Mode={res['mode'][:3]} | "
                f"IS (5y): Ret={res['is_return_pct']:+6.1f}%, Sh={res['is_sharpe']:+.2f}, MDD={res['is_mdd_pct']:4.1f}% | "
                f"OOS (1y): Ret={res['oos_return_pct']:+6.1f}%, Sh={res['oos_sharpe']:+.2f}, MDD={res['oos_mdd_pct']:4.1f}%{overfit_alert}"
            )

        # Selezione moda
        counts = Counter(param_tuples)
        most_common_tuple, freq = counts.most_common(1)[0]

        is_sharpes = [h["is_sharpe"] for h in self.history if h["is_sharpe"] is not None]
        oos_sharpes = [h["oos_sharpe"] for h in self.history if h["oos_sharpe"] is not None]
        is_rets = [h["is_return_pct"] for h in self.history if h["is_return_pct"] is not None]
        oos_rets = [h["oos_return_pct"] for h in self.history if h["oos_return_pct"] is not None]
        is_mdds = [h["is_mdd_pct"] for h in self.history if h["is_mdd_pct"] is not None]
        oos_mdds = [h["oos_mdd_pct"] for h in self.history if h["oos_mdd_pct"] is not None]

        mean_is_sharpe = round(float(np.mean(is_sharpes)), 2)
        mean_oos_sharpe = round(float(np.mean(oos_sharpes)), 2)
        mean_is_ret = round(float(np.mean(is_rets)), 2)
        mean_oos_ret = round(float(np.mean(oos_rets)), 2)
        mean_is_mdd = round(float(np.mean(is_mdds)), 2)
        mean_oos_mdd = round(float(np.mean(oos_mdds)), 2)
        oos_positive_years = sum(1 for r in oos_rets if r > 0)
        oos_win_rate = round(float(oos_positive_years / len(oos_rets) * 100), 1)

        self.selected_params = {
            "window": most_common_tuple[0],
            "threshold": most_common_tuple[1],
            "exit_threshold": self.optimizer.exit_threshold,
            "mode": most_common_tuple[2],
            "price_col": self.price_col,
            "frequency": f"{freq}/{len(param_tuples)} finestre",
            "all_combinations_tested": len(param_tuples),
            "objective_metric": self.optimizer.objective_metric,
            "summary_metrics": {
                "mean_is_return_pct": mean_is_ret,
                "mean_oos_return_pct": mean_oos_ret,
                "mean_is_sharpe": mean_is_sharpe,
                "mean_oos_sharpe": mean_oos_sharpe,
                "mean_is_mdd_pct": mean_is_mdd,
                "mean_oos_mdd_pct": mean_oos_mdd,
                "oos_positive_years": oos_positive_years,
                "total_oos_years": len(oos_rets),
                "oos_win_rate_pct": oos_win_rate,
            },
        }

        print("\n" + "=" * 70)
        print("          RIEPILOGO OTTIMIZZAZIONE & DIAGNOSTICA OVERFITTING")
        print("=" * 70)
        print(
            f"Parametri selezionati (Moda): Window={self.selected_params['window']}, "
            f"Threshold={self.selected_params['threshold']}, Mode={self.selected_params['mode']}, "
            f"ExitThreshold={self.selected_params['exit_threshold']}"
        )
        print(f"Frequenza di apparizione    : {self.selected_params['frequency']}")
        print(f"Rendimento medio cumulato   : In-Sample={mean_is_ret:+.1f}% (5y)  | Out-Of-Sample={mean_oos_ret:+.1f}% (1y)")
        print(f"Sharpe Ratio medio          : In-Sample={mean_is_sharpe:+.2f}       | Out-Of-Sample={mean_oos_sharpe:+.2f}")
        print(f"Max Drawdown medio          : In-Sample={mean_is_mdd:.1f}%          | Out-Of-Sample={mean_oos_mdd:.1f}%")
        print(f"Anni OOS con rendimento > 0 : {oos_positive_years}/{len(oos_rets)} ({oos_win_rate}%)")
        print("=" * 70)

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
                    "mode": self.selected_params["mode"],
                    "price_col": self.selected_params["price_col"],
                },
                f,
                indent=4,
            )
        print(f"Configurazione Z-Score salvata con successo in: {path}")