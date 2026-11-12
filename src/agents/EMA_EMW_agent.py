"""Statistical trend-following agent based on a crossover between a simple
moving average (SMA) and an exponential moving average (EMA).

The agent inherits from BaseAgent to provide a standard interface and supports:
- joint prediction of signals {-1, 0, 1} and a continuous confidence in [0.0, 1.0];
- an optional dead band that reduces false entries in sideways markets;
- stability-oriented walk-forward optimisation (Sharpe ratio penalised by
  the maximum drawdown);
- detailed logging of every combination tested during the grid search.
"""

import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.evaluator import calculate_performance_metrics, compute_objective_score


# =====================================================================
# 1. SMA/EMA INFERENCE AGENT
# =====================================================================
class SMAEMAAgent(BaseAgent):
    """Trend-following inference agent based on an SMA/EMA crossover.

    - fast EMA > slow SMA -> BUY (+1)
    - fast EMA < slow SMA -> SELL (-1)
    - if dead_band > 0 and |EMA - SMA| / SMA <= dead_band -> HOLD (0)
    - during the initial warm-up period -> HOLD (0)
    """

    def __init__(
        self,
        sma_window: Optional[int] = None,
        ema_window: Optional[int] = None,
        dead_band: float = 0.0,
        price_col: Optional[str] = None,
        target_asset: Optional[str] = None,
        output_format: str = "numeric",  # 'numeric' (-1, 0, 1) or 'string' ('SELL', 'HOLD', 'BUY')
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
        """Update the internal agent parameters after loading them from JSON."""
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
        """Compute the SMA and EMA of prices and return the standardised trading signals.
        """
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
        # During the initial warm-up the SMA is NaN: the signal is set to 0 (HOLD).
        signals[sma.isna()] = 0

        return self._format_signals(signals)

    def predict_with_confidence(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Compute both the discrete signals and the associated confidence estimate.

        Confidence is a function of the normalised divergence between EMA and SMA:
        - right after a crossover (difference close to 0), confidence is about 0.50;
        - in an established trend (large difference relative to volatility),
          confidence rises towards 0.95 or more;
        - during warm-up or when flat (signal 0), confidence is 0.0.
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


# Alias kept for backward compatibility.
EMA_EMW_Agent = SMAEMAAgent


# =====================================================================
# 2. SINGLE-WINDOW OPTIMISER (in-sample training + out-of-sample validation)
# =====================================================================
class SMAEMAOptimizer:
    """Optimiser for a single time block.

    - Optimises (sma_window, ema_window) STRICTLY on the training block
      (in-sample, typically 5 years).
    - Supports a stability/risk-oriented objective (Sharpe ratio penalised by
      the maximum drawdown, Calmar ratio, or plain Sharpe ratio).
    - Evaluates the optimal parameters on both the training period and the
      following validation period (out-of-sample, 1 year).
    - Records the full results of ALL combinations tested in the grid search.
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
        """Run the grid search on the training data only and evaluate every combination.

        Performance is reported on both the training (in-sample) and validation
        (out-of-sample) periods for all combinations.
        """
        dfs_to_concat = [df for df in [warmup_df, train_df, val_df] if df is not None and not df.empty]
        full_df = pd.concat(dfs_to_concat)
        if (
            not isinstance(full_df.index, pd.DatetimeIndex)
            or full_df.index.hasnans
            or not full_df.index.is_monotonic_increasing
            or not full_df.index.is_unique
        ):
            raise ValueError("Warm-up, training e validation devono essere ordinati e senza date duplicate.")

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

        # Technical constraint: the fast EMA window must be shorter than the slow SMA window.
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
            # No position until both moving averages are available.
            valid_indicators = sma.notna() & ema.notna()
            pos = np.where(valid_indicators, pos, 0.0)

            # SHIFT(1): the signal computed at close t applies to the returns of t+1.
            pos_series = pd.Series(pos, index=full_df.index).shift(1)

            # STRICT evaluation on the TRAINING (in-sample) period only.
            train_pos = pos_series.reindex(train_indices).fillna(0.0)
            train_strat_ret = train_pos * full_returns.reindex(train_indices).fillna(0.0)

            is_metrics = calculate_performance_metrics(train_strat_ret)
            score = compute_objective_score(is_metrics, metric_type=self.objective_metric)

            # Out-of-sample evaluation for this combination.
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

        # Out-of-sample metrics of the selected parameters.
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
        """Save the optimal parameters to a JSON file."""
        if self.best_params is None:
            raise ValueError("Nessun parametro trovato. Esegui prima .fit().")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.best_params, f, indent=4)


# =====================================================================
# 3. WALK-FORWARD LOOP (5-year in-sample training -> 1-year out-of-sample validation)
# =====================================================================
class WalkForwardTrainer:
    """Run the walk-forward cycle.

    - Splits the DataFrame year by year.
    - At each step, uses a 5-year training window (in-sample) followed by one
      validation year (out-of-sample).
    - Optimises on the training window with the chosen stability/risk metric.
    - Keeps the history of ALL combinations tested in every window.
    - Exports the full log and an aggregated summary for each parameter pair.
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

    def run(self, df: pd.DataFrame, max_year: Optional[int] = None, initial_warmup_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run the rolling loop over the full historical dataset.

        Args:
            df: Historical DataFrame with a DatetimeIndex.
            max_year: Latest year allowed (e.g. 2021 to respect the OOS lock).
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
            past_train_df = data[data["year"].isin(warmup_years)]

            history_parts = [
                x for x in (initial_warmup_df, past_train_df)
                if x is not None and not x.empty
            ]

            fold_warmup_df = (
                pd.concat(history_parts).sort_index()
                if history_parts else None
            )

            train_df = data[data["year"].isin(train_span)]
            val_df = data[data["year"] == val_year]

            best = self.optimizer.fit(
                train_df,
                val_df,
                warmup_df=fold_warmup_df,
            )
            # Collect the results of every combination for this window.
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

        # Select the mode.
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
        """Return the full log of every combination tested in every window."""
        return pd.DataFrame(self.all_combinations_history)

    def get_combinations_summary(self) -> pd.DataFrame:
        """Return a summary of all (sma_window, ema_window) combinations across walk-forward windows.
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

        # Sort by selection frequency, then by mean out-of-sample Sharpe ratio.
        summary = summary.sort_values(
            by=["times_best", "mean_oos_sharpe", "mean_is_score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        return summary

    def save_combinations_log(self, filepath: Union[str, Path]) -> None:
        """Export the full list of tested combinations to CSV."""
        df_log = self.get_combinations_dataframe()
        if df_log.empty:
            raise ValueError("Nessun dato di combinazioni presente. Esegui prima .run().")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df_log.to_csv(path, index=False)
        print(f"Log dettagliato combinazioni salvato in: {path} ({len(df_log)} righe)")

    def save_combinations_summary(self, filepath: Union[str, Path]) -> None:
        """Export the aggregated summary of each combination to CSV."""
        df_sum = self.get_combinations_summary()
        if df_sum.empty:
            raise ValueError("Nessun dato presente. Esegui prima .run().")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df_sum.to_csv(path, index=False)
        print(f"Riepilogo combinazioni aggregato salvato in: {path} ({len(df_sum)} righe)")

    def save_final_agent_config(self, filepath: Union[str, Path]) -> None:
        """Save the final (modal) parameters to a JSON file readable by SMAEMAAgent."""
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