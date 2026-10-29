"""
EMA_EMW_agent.py

Agente statistico Trend-Following basato su Crossover tra Media Mobile Semplice (SMA)
e Media Mobile Esponenziale (EMA).
Eredita da BaseAgent per garantire un'interfaccia standardizzata.
Supporta:
- Predizione congiunta di Segnali {-1, 0, 1} e Confidenza continua [0.0, 1.0]
- Dead-band opzionale per ridurre le false partenze su movimenti laterali
- Ottimizzazione Walk-Forward orientata alla stabilita (Sharpe penalizzato per Max Drawdown)
- Logging dettagliato di tutte le combinazioni testate durante la grid search
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
# 1. AGENTE INFERENZIALE SMA/EMA
# =====================================================================
class SMAEMAAgent(BaseAgent):
    """
    Agente inferenziale Trend-Following basato su Crossover SMA / EMA:
    - EMA (veloce) > SMA (lenta) -> BUY (+1)
    - EMA (veloce) < SMA (lenta) -> SELL (-1)
    - Se dead_band > 0 e |EMA - SMA| / SMA <= dead_band -> HOLD (0)
    - Durante il periodo iniziale di warm-up -> HOLD (0)
    """

    def __init__(
        self,
        sma_window: Optional[int] = None,
        ema_window: Optional[int] = None,
        dead_band: float = 0.0,
        price_col: Optional[str] = None,
        target_asset: Optional[str] = None,
        output_format: str = "numeric",  # 'numeric' (-1, 0, 1) o 'string' ('SELL', 'HOLD', 'BUY')
    ):
        super().__init__(
            name="SMAEMAAgent",
            target_asset=target_asset,
            price_col=price_col,
            output_format=output_format,
        )
        self.sma_window = sma_window
        self.ema_window = ema_window
        self.dead_band = float(dead_band)
        self.target_asset = target_asset.upper() if target_asset else None
        self.price_col = price_col or (f"{self.target_asset}_close" if self.target_asset else "close")

        if sma_window is not None and ema_window is not None:
            self.params = {
                "sma_window": int(sma_window),
                "ema_window": int(ema_window),
                "dead_band": float(dead_band),
                "price_col": self.price_col,
                "target_asset": self.target_asset,
            }

    def _apply_params(self, params: Dict[str, Any]) -> None:
        """Aggiorna i parametri interni dell'agente dopo il caricamento da JSON."""
        super()._apply_params(params)
        self.sma_window = int(params["sma_window"])
        self.ema_window = int(params["ema_window"])
        self.dead_band = float(params.get("dead_band", 0.0))
        if "target_asset" in params:
            self.target_asset = str(params["target_asset"]).upper()
        if "price_col" in params:
            self.price_col = str(params["price_col"])
        self.params = {
            "sma_window": self.sma_window,
            "ema_window": self.ema_window,
            "dead_band": self.dead_band,
            "price_col": self.price_col,
            "target_asset": self.target_asset,
        }

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Calcola SMA ed EMA sui prezzi e restituisce i segnali operativi uniformati."""
        if self.sma_window is None or self.ema_window is None:
            raise ValueError(
                "Parametri non impostati. Inizializzali nel costruttore o usa .load_params()."
            )

        price_column = self.resolve_price_col(df, price_col=self.price_col, target_asset=self.target_asset)
        close = df[price_column].astype(float)
        sma = close.rolling(window=self.sma_window).mean()
        ema = close.ewm(span=self.ema_window, adjust=False).mean()

        if self.dead_band > 0.0:
            diff_pct = (ema - sma) / sma.replace(0, np.nan)
            raw_signal = np.where(
                diff_pct > self.dead_band,
                1,
                np.where(diff_pct < -self.dead_band, -1, 0),
            )
        else:
            raw_signal = np.where(ema > sma, 1, -1)

        signals = pd.Series(raw_signal, index=df.index, dtype=int)
        # Durante il warm-up iniziale la SMA è NaN: impostiamo segnale a 0 (HOLD)
        signals[sma.isna()] = 0

        return self._format_signals(signals)

    def predict_with_confidence(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calcola sia i segnali operativi discreti sia la stima di confidenza / probabilità associata.

        La confidenza è funzione della divergenza normalizzata tra EMA e SMA:
        - Appena avvenuto il crossover (diff ~ 0), la confidenza è circa 0.50 (incertezza iniziale).
        - Con trend consolidato (diff elevata rispetto alla volatilità), la confidenza sale verso 0.95+.
        - Durante il warm-up o flat (segnale 0), la confidenza è 0.0.
        """
        signals = self.predict(df)

        price_column = self.resolve_price_col(df, price_col=self.price_col, target_asset=self.target_asset)
        close = df[price_column].astype(float)
        sma = close.rolling(window=self.sma_window).mean()
        ema = close.ewm(span=self.ema_window, adjust=False).mean()

        diff = (ema - sma).abs()
        vol = close.rolling(window=self.sma_window).std().replace(0, np.nan)
        norm_diff = (diff / vol).fillna(0.0)

        conf_values = np.where(
            signals != 0,
            0.50 + 0.45 * np.tanh(norm_diff),
            0.0,
        )
        confidence = pd.Series(conf_values, index=df.index, dtype=float).round(4)
        return signals, confidence


# Alias per retrocompatibilità
EMA_EMW_Agent = SMAEMAAgent


# =====================================================================
# 2. OTTIMIZZATORE SU SINGOLA FINESTRA (Train IS + Val OOS)
# =====================================================================
class SMAEMAOptimizer:
    """
    Ottimizzatore su singolo blocco temporale:
    - Ottimizza i parametri (sma_window, ema_window) RIGOROSAMENTE sul blocco di Train (In-Sample, tipicamente 5 anni).
    - Supporta una funzione obiettivo mirata alla stabilità/rischio (Sharpe ratio penalizzato per Max Drawdown, Calmar, o Sharpe puro).
    - Valuta poi i parametri ottimi sia sul periodo di Train (In-Sample) sia sul periodo successivo di Validation (Out-Of-Sample, 1 anno).
    - Registra i risultati completi di TUTTE le combinazioni testate durante la grid search.
    """

    def __init__(
        self,
        price_col: str = "close",
        sma_range: range = range(20, 101, 5),
        ema_range: range = range(5, 51, 5),
        dead_band: float = 0.0,
        objective_metric: str = "sharpe_mdd",
    ):
        self.price_col = price_col
        self.sma_range = sma_range
        self.ema_range = ema_range
        self.dead_band = dead_band
        self.objective_metric = objective_metric
        self.best_params: Optional[Dict[str, Any]] = None
        self.combinations_log: List[Dict[str, Any]] = []

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        warmup_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Esegue la grid search sui soli dati di Train (In-Sample) e valuta la performance
        sia su Train che su Validation (Out-Of-Sample) per tutte le combinazioni.
        """
        dfs_to_concat = [df for df in [warmup_df, train_df, val_df] if df is not None and not df.empty]
        full_df = pd.concat(dfs_to_concat)

        price_col_resolved = self.price_col
        if price_col_resolved not in full_df.columns:
            for c in full_df.columns:
                if str(c).lower().endswith("_close") or str(c).lower() == "close":
                    price_col_resolved = c
                    break
            if price_col_resolved not in full_df.columns:
                raise KeyError(
                    f"Colonna '{self.price_col}' non presente nei dati di train/validation. Colonne: {list(full_df.columns)}"
                )

        train_indices = train_df.index
        val_indices = val_df.index if val_df is not None else None

        close = full_df[price_col_resolved].astype(float)
        full_returns = close.pct_change().fillna(0.0)

        best_score = -np.inf
        best_sma = None
        best_ema = None
        best_is_metrics = None
        self.combinations_log = []

        # Vincolo tecnico: EMA (veloce) deve avere finestra minore di SMA (lenta)
        param_grid = [
            (s, e) for s, e in itertools.product(self.sma_range, self.ema_range) if e < s
        ]

        for sma_w, ema_w in param_grid:
            sma = close.rolling(window=sma_w).mean()
            ema = close.ewm(span=ema_w, adjust=False).mean()

            if self.dead_band > 0.0:
                diff_pct = (ema - sma) / sma.replace(0, np.nan)
                pos = np.where(
                    diff_pct > self.dead_band,
                    1.0,
                    np.where(diff_pct < -self.dead_band, -1.0, 0.0),
                )
            else:
                pos = np.where(ema > sma, 1.0, -1.0)

            # SHIFT(1): il segnale calcolato a chiusura t opera sui rendimenti di t+1
            pos_series = pd.Series(pos, index=full_df.index).shift(1)

            # Valutazione RIGOROSA solo su periodo di TRAINING (In-Sample)
            train_pos = pos_series.reindex(train_indices).fillna(0.0)
            train_strat_ret = train_pos * full_returns.reindex(train_indices).fillna(0.0)

            is_metrics = calculate_performance_metrics(train_strat_ret)
            score = compute_objective_score(is_metrics, metric_type=self.objective_metric)

            # Valutazione OOS per questa specifica combinazione
            oos_metrics = None
            if val_indices is not None and len(val_indices) > 0:
                val_pos = pos_series.reindex(val_indices).fillna(0.0)
                val_strat_ret = val_pos * full_returns.reindex(val_indices).fillna(0.0)
                oos_metrics = calculate_performance_metrics(val_strat_ret)

            comb_record: Dict[str, Any] = {
                "sma_window": int(sma_w),
                "ema_window": int(ema_w),
                "dead_band": float(self.dead_band),
                "is_score": round(float(score), 4),
                "is_return_pct": round(float(is_metrics["cumulative_return"] * 100), 2),
                "is_annualized_ret_pct": round(float(is_metrics["annualized_return"] * 100), 2),
                "is_volatility_pct": round(float(is_metrics["annualized_volatility"] * 100), 2),
                "is_sharpe": round(float(is_metrics["sharpe_ratio"]), 2),
                "is_mdd_pct": round(float(is_metrics["max_drawdown"] * 100), 2),
                "is_profit_factor": round(float(is_metrics["profit_factor"]), 2),
                "is_calmar_ratio": round(float(is_metrics["calmar_ratio"]), 2),
            }

            if oos_metrics is not None:
                comb_record.update({
                    "oos_return_pct": round(float(oos_metrics["cumulative_return"] * 100), 2),
                    "oos_annualized_ret_pct": round(float(oos_metrics["annualized_return"] * 100), 2),
                    "oos_volatility_pct": round(float(oos_metrics["annualized_volatility"] * 100), 2),
                    "oos_sharpe": round(float(oos_metrics["sharpe_ratio"]), 2),
                    "oos_mdd_pct": round(float(oos_metrics["max_drawdown"] * 100), 2),
                    "oos_profit_factor": round(float(oos_metrics["profit_factor"]), 2),
                    "oos_calmar_ratio": round(float(oos_metrics["calmar_ratio"]), 2),
                })

            self.combinations_log.append(comb_record)

            if not np.isnan(score) and score > best_score:
                best_score = score
                best_sma = sma_w
                best_ema = ema_w
                best_is_metrics = is_metrics

        if best_sma is None or best_ema is None:
            raise RuntimeError("Nessuna combinazione valida trovata durante l'ottimizzazione.")

        # Estrazione metriche OOS per il miglior parametro scelto
        best_oos_metrics = None
        if val_df is not None and not val_df.empty:
            best_match = next(
                (c for c in self.combinations_log if c["sma_window"] == best_sma and c["ema_window"] == best_ema),
                None,
            )
            if best_match and "oos_return_pct" in best_match:
                best_oos_metrics = {
                    "cumulative_return": best_match["oos_return_pct"] / 100.0,
                    "sharpe_ratio": best_match["oos_sharpe"],
                    "max_drawdown": best_match["oos_mdd_pct"] / 100.0,
                }

        is_ret_pct = round(best_is_metrics["cumulative_return"] * 100, 2)
        is_sharpe = round(best_is_metrics["sharpe_ratio"], 2)
        is_mdd_pct = round(best_is_metrics["max_drawdown"] * 100, 2)

        oos_ret_pct = round(best_oos_metrics["cumulative_return"] * 100, 2) if best_oos_metrics else None
        oos_sharpe = round(best_oos_metrics["sharpe_ratio"], 2) if best_oos_metrics else None
        oos_mdd_pct = round(best_oos_metrics["max_drawdown"] * 100, 2) if best_oos_metrics else None

        overfitting_sharpe_drop = (
            round(is_sharpe - oos_sharpe, 2) if (oos_sharpe is not None) else None
        )

        self.best_params = {
            "sma_window": int(best_sma),
            "ema_window": int(best_ema),
            "dead_band": float(self.dead_band),
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
            "val_metrics": best_oos_metrics,
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
# 3. ORCHESTRATORE WALK-FORWARD (5 Anni Train IS -> 1 Anno Val OOS)
# =====================================================================
class WalkForwardTrainer:
    """
    Gestisce il ciclo di Walk-Forward:
    - Suddivide il DataFrame anno per anno.
    - Per ogni step estrae una finestra di 5 anni di training (IS) e l'anno successivo di validation (OOS).
    - Ottimizza sul training con la metrica di stabilità/rischio desiderata.
    - Raccoglie e preserva lo storico di TUTTE le combinazioni testate in ogni singola finestra.
    - Esporta il log completo e una sintesi aggregata per ciascuna coppia di parametri.
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
        self.all_combinations_history: List[Dict[str, Any]] = []
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
        self.all_combinations_history = []
        param_tuples = []

        print(
            f"\nInizio Walk-Forward SMA/EMA: {len(years)} anni ({years[0]} - {years[-1]}) | "
            f"Metrica: '{self.optimizer.objective_metric}'"
        )
        print("-" * 115)

        for i in range(len(years) - self.train_window_years):
            train_span = years[i : i + self.train_window_years]
            val_year = years[i + self.train_window_years]

            warmup_years = years[:i]
            warmup_df = data[data["year"].isin(warmup_years)] if len(warmup_years) > 0 else None
            train_df = data[data["year"].isin(train_span)]
            val_df = data[data["year"] == val_year]

            best = self.optimizer.fit(train_df, val_df, warmup_df=warmup_df)

            # Raccoglie i risultati di tutte le combinazioni per questa finestra
            for c in self.optimizer.combinations_log:
                rec = dict(c)
                rec["fold"] = i + 1
                rec["train_years"] = f"{train_span[0]}-{train_span[-1]}"
                rec["val_year"] = int(val_year)
                rec["is_best_in_fold"] = bool(
                    c["sma_window"] == best["sma_window"] and c["ema_window"] == best["ema_window"]
                )
                self.all_combinations_history.append(rec)

            res = {
                "train_years": f"{train_span[0]}-{train_span[-1]}",
                "val_year": int(val_year),
                "sma_window": best["sma_window"],
                "ema_window": best["ema_window"],
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
            param_tuples.append((best["sma_window"], best["ema_window"]))

            is_sh = res["is_sharpe"]
            oos_sh = res["oos_sharpe"]
            overfit_alert = " [OVERFIT]" if (is_sh > 0.4 and oos_sh is not None and oos_sh < 0) else ""

            print(
                f"Finestra {i+1:02d} | Train: [{res['train_years']}] -> Test: {res['val_year']} | "
                f"Ottimo: SMA={res['sma_window']:2d}, EMA={res['ema_window']:2d} | "
                f"IS (5y): Ret={res['is_return_pct']:+6.1f}%, Sh={res['is_sharpe']:+.2f}, MDD={res['is_mdd_pct']:4.1f}% | "
                f"OOS (1y): Ret={res['oos_return_pct']:+6.1f}%, Sh={res['oos_sharpe']:+.2f}, MDD={res['oos_mdd_pct']:4.1f}%{overfit_alert}"
            )

        # Selezione moda
        counts = Counter(param_tuples)
        most_common_pair, freq = counts.most_common(1)[0]

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
            "sma_window": most_common_pair[0],
            "ema_window": most_common_pair[1],
            "dead_band": self.optimizer.dead_band,
            "price_col": self.price_col,
            "frequency": f"{freq}/{len(param_tuples)} finestre",
            "all_combinations_tested": len(param_tuples),
            "total_combination_evaluations": len(self.all_combinations_history),
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

        print("\n" + "=" * 65)
        print("          RIEPILOGO OTTIMIZZAZIONE & DIAGNOSTICA OVERFITTING")
        print("=" * 65)
        print(f"Parametri selezionati (Moda): SMA={self.selected_params['sma_window']}, EMA={self.selected_params['ema_window']}")
        print(f"Frequenza di apparizione    : {self.selected_params['frequency']}")
        print(f"Combinazioni totali loggate : {len(self.all_combinations_history):,} record")
        print(f"Rendimento medio cumulato   : In-Sample={mean_is_ret:+.1f}% (5y)  | Out-Of-Sample={mean_oos_ret:+.1f}% (1y)")
        print(f"Sharpe Ratio medio          : In-Sample={mean_is_sharpe:+.2f}       | Out-Of-Sample={mean_oos_sharpe:+.2f}")
        print(f"Max Drawdown medio          : In-Sample={mean_is_mdd:.1f}%          | Out-Of-Sample={mean_oos_mdd:.1f}%")
        print(f"Anni OOS con rendimento > 0 : {oos_positive_years}/{len(oos_rets)} ({oos_win_rate}%)")
        print("=" * 65)

        return self.selected_params

    def get_combinations_dataframe(self) -> pd.DataFrame:
        """Restituisce il DataFrame con il log completo di ogni singola combinazione testata in ogni finestra."""
        return pd.DataFrame(self.all_combinations_history)

    def get_combinations_summary(self) -> pd.DataFrame:
        """
        Restituisce un DataFrame riassuntivo che aggrega le performance di tutte le combinazioni
        (sma_window, ema_window) calcolate su tutte le finestre del Walk-Forward.
        """
        df_all = self.get_combinations_dataframe()
        if df_all.empty:
            return pd.DataFrame()

        summary = (
            df_all.groupby(["sma_window", "ema_window"])
            .agg(
                folds_tested=("fold", "count"),
                times_best=("is_best_in_fold", "sum"),
                mean_is_score=("is_score", "mean"),
                mean_is_return_pct=("is_return_pct", "mean"),
                mean_is_sharpe=("is_sharpe", "mean"),
                mean_is_mdd_pct=("is_mdd_pct", "mean"),
                mean_oos_return_pct=("oos_return_pct", "mean"),
                mean_oos_sharpe=("oos_sharpe", "mean"),
                mean_oos_mdd_pct=("oos_mdd_pct", "mean"),
                positive_oos_years=("oos_return_pct", lambda s: int((s > 0).sum())),
            )
            .reset_index()
        )

        total_folds = len(self.history) if self.history else 1
        summary["selection_pct"] = (summary["times_best"] / total_folds * 100).round(1)
        summary["oos_win_rate_pct"] = (summary["positive_oos_years"] / summary["folds_tested"] * 100).round(1)

        for col in [
            "mean_is_score", "mean_is_return_pct", "mean_is_sharpe", "mean_is_mdd_pct",
            "mean_oos_return_pct", "mean_oos_sharpe", "mean_oos_mdd_pct"
        ]:
            summary[col] = summary[col].round(2)

        # Ordina per frequenza di selezione e poi per Sharpe OOS medio
        summary = summary.sort_values(
            by=["times_best", "mean_oos_sharpe", "mean_is_score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        return summary

    def save_combinations_log(self, filepath: Union[str, Path]) -> None:
        """Esporta l'elenco completo di ogni combinazione testata su file CSV."""
        df_log = self.get_combinations_dataframe()
        if df_log.empty:
            raise ValueError("Nessun dato di combinazioni presente. Esegui prima .run().")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df_log.to_csv(path, index=False)
        print(f"Log dettagliato combinazioni salvato in: {path} ({len(df_log)} righe)")

    def save_combinations_summary(self, filepath: Union[str, Path]) -> None:
        """Esporta la tabella riassuntiva aggregata per ciascuna combinazione su file CSV."""
        df_sum = self.get_combinations_summary()
        if df_sum.empty:
            raise ValueError("Nessun dato presente. Esegui prima .run().")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df_sum.to_csv(path, index=False)
        print(f"Riepilogo combinazioni aggregato salvato in: {path} ({len(df_sum)} righe)")

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
                    "dead_band": self.selected_params.get("dead_band", 0.0),
                    "price_col": self.selected_params["price_col"],
                },
                f,
                indent=4,
            )
        print(f"Configurazione salvata con successo in: {path}")