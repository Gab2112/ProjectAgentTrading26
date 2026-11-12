"""Statistical agent based on a rolling Z-score.

Features:
- mean reversion with hysteresis (reversion to the mean in sideways markets);
- momentum/breakout with hysteresis (trend following in directional regimes);
- an optional stop-loss based on extreme deviations, for risk protection;
- joint computation of discrete signals {-1, 0, 1} and a continuous confidence in [0.0, 1.0];
- stability-oriented walk-forward optimisation (5-year in-sample -> 1-year
  out-of-sample) with a maximum drawdown penalty;
- detailed and summary logging of every combination tested in the grid search.
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
# Z-SCORE TRADING LOGIC WITH HYSTERESIS
# =====================================================================
def compute_zscore_positions(
    z_scores: np.ndarray,
    entry_threshold: float,
    exit_threshold: float = 0.0,
    stop_threshold: Optional[float] = None,
    mode: str = "momentum",
) -> np.ndarray:
    """Compute the Z-score based positions with hysteresis and an optional stop.

    Modes:
    - ``mode == 'momentum'`` (trend following / breakout):
        * Z > +entry_threshold: upside breakout -> BUY (+1)
        * Z <= +exit_threshold: pull-back / exhaustion -> FLAT (0)
        * Z < -entry_threshold: downside breakdown -> SELL (-1)
        * Z >= -exit_threshold: pull-back / exhaustion -> FLAT (0)
        * With stop_threshold active: stop if Z reverses beyond -stop_threshold
          (when long) or +stop_threshold (when short).
    - ``mode == 'mean_reversion'``:
        * Z < -entry_threshold: abnormally oversold -> BUY (+1)
        * Z >= -exit_threshold: reversion to the mean -> FLAT (0)
        * Z > +entry_threshold: abnormally overbought -> SELL (-1)
        * Z <= +exit_threshold: reversion to the mean -> FLAT (0)
        * With stop_threshold active: stop if Z deviates beyond -stop_threshold
          (e.g. -3.5) to cut losses in adverse trends.

    Args:
        z_scores: NumPy array of Z-score values.
        entry_threshold: Standard-deviation threshold for entries.
        exit_threshold: Threshold of reversion towards the mean for exits (default 0.0).
        stop_threshold: Optional stop-loss threshold for extreme deviations.
        mode: ``'momentum'`` or ``'mean_reversion'``.

    Returns:
        np.ndarray of integers in {-1, 0, 1}.
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
                # Normal exit or stop.
                if z <= exit_threshold:
                    current_pos = 0
                    if z < -entry_threshold:
                        current_pos = -1
                elif stop_threshold is not None and z < -stop_threshold:
                    current_pos = 0
            elif current_pos == -1:
                # Normal exit or stop.
                if z >= -exit_threshold:
                    current_pos = 0
                    if z > entry_threshold:
                        current_pos = 1
                elif stop_threshold is not None and z > stop_threshold:
                    current_pos = 0
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
                # Normal exit on reversion to the mean, or stop-loss.
                if stop_threshold is not None and z < -stop_threshold:
                    current_pos = 0  # Stop-loss on a persistent decline.
                elif z >= -exit_threshold:
                    current_pos = 0
                    if z > entry_threshold:
                        current_pos = -1
            elif current_pos == -1:
                # Normal exit, or stop-loss on a sharp rally.
                if stop_threshold is not None and z > stop_threshold:
                    current_pos = 0  # Stop-loss.
                elif z <= exit_threshold:
                    current_pos = 0
                    if z < -entry_threshold:
                        current_pos = 1
            positions[i] = current_pos

    return positions


# =====================================================================
# 1. Z-SCORE INFERENCE AGENT
# =====================================================================
class ZScoreAgent(BaseAgent):
    """Inference agent based on a rolling Z-score.

    Supports momentum (breakout) or mean-reversion modes, an optional
    stop-loss and confidence estimation.
    """

    def __init__(
        self,
        window: Optional[int] = None,
        threshold: Optional[float] = None,
        exit_threshold: float = 0.0,
        stop_threshold: Optional[float] = None,
        mode: str = "momentum",
        price_col: Optional[str] = None,
        target_asset: Optional[str] = None,
        output_format: str = "numeric",  # 'numeric' (-1, 0, 1) or 'string' ('SELL', 'HOLD', 'BUY')
    ):
        super().__init__(
            name="ZScoreAgent",
            target_asset=target_asset,
            price_col=price_col,
            output_format=output_format,
        )
        self.window = window
        self.threshold = threshold
        self.exit_threshold = exit_threshold
        self.stop_threshold = float(stop_threshold) if stop_threshold is not None else None
        self.mode = mode
        self.target_asset = target_asset.upper() if target_asset else None
        self.price_col = price_col or (f"{self.target_asset}_close" if self.target_asset else "close")

        if window is not None and threshold is not None:
            self.params = {
                "window": int(window),
                "threshold": float(threshold),
                "exit_threshold": float(exit_threshold),
                "stop_threshold": self.stop_threshold,
                "mode": str(mode),
                "price_col": self.price_col,
                "target_asset": self.target_asset,
            }

    def _apply_params(self, params: Dict[str, Any]) -> None:
        """Update the agent parameters after loading them from JSON."""
        super()._apply_params(params)
        self.window = int(params["window"])
        self.threshold = float(params["threshold"])
        self.exit_threshold = float(params.get("exit_threshold", 0.0))
        self.stop_threshold = (
            float(params["stop_threshold"])
            if params.get("stop_threshold") is not None
            else None
        )
        self.mode = str(params.get("mode", "momentum"))
        if "target_asset" in params:
            self.target_asset = str(params["target_asset"]).upper()
        if "price_col" in params:
            self.price_col = str(params["price_col"])
        self.params = {
            "window": self.window,
            "threshold": self.threshold,
            "exit_threshold": self.exit_threshold,
            "stop_threshold": self.stop_threshold,
            "mode": self.mode,
            "price_col": self.price_col,
            "target_asset": self.target_asset,
        }

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Compute the Z-score and generate the discrete signals {-1, 0, 1}."""
        if self.window is None or self.threshold is None:
            raise ValueError(
                "Parametri non impostati. Inizializzali nel costruttore o usa .load_params()."
            )

        price_column = self.resolve_price_col(df, price_col=self.price_col, target_asset=self.target_asset)
        close = df[price_column].astype(float)
        mean = close.rolling(window=self.window).mean()
        std = close.rolling(window=self.window).std()

        std_safe = std.replace(0, np.nan)
        z_score = (close - mean) / std_safe

        raw_signals = compute_zscore_positions(
            z_scores=z_score.to_numpy(),
            entry_threshold=self.threshold,
            exit_threshold=self.exit_threshold,
            stop_threshold=self.stop_threshold,
            mode=self.mode,
        )

        signals = pd.Series(raw_signals, index=df.index, dtype=int)
        signals[z_score.isna()] = 0

        return self._format_signals(signals)

    def predict_with_confidence(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Compute the trading signals together with their confidence.

        Confidence is a function of the size of the standardised deviation:
        - in an active position (|signal| == 1), confidence starts at 0.50 and
          rises towards 0.95 or more;
        - when flat or during warm-up (signal == 0), confidence is 0.0.
        """
        signals = self.predict(df)

        price_column = self.resolve_price_col(df, price_col=self.price_col, target_asset=self.target_asset)
        close = df[price_column].astype(float)
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
# 2. SINGLE-BLOCK Z-SCORE OPTIMISER (in-sample training + out-of-sample validation)
# =====================================================================
class ZScoreOptimizer:
    """Z-score optimiser for a single time block.

    - Runs the grid search over (window, threshold, mode) STRICTLY on the
      training block (in-sample, 5 years).
    - Scores each combination with a stability/risk objective (``'sharpe_mdd'``).
    - Evaluates out-of-sample metrics on the following validation year,
      without re-optimising.
    - Compares in-sample and out-of-sample performance to detect overfitting.
    - Records the full results of ALL combinations tested in the grid search.
    """

    def __init__(
        self,
        price_col: str = "close",
        window_range: Optional[Union[range, List[int]]] = None,
        threshold_range: Optional[List[float]] = None,
        mode_range: Optional[List[str]] = None,
        exit_threshold: float = 0.0,
        stop_threshold: Optional[float] = None,
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
        self.stop_threshold = stop_threshold
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
        (out-of-sample) periods for each combination.
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
        best_w = None
        best_thresh = None
        best_mode = None
        best_is_metrics = None
        self.combinations_log = []

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
                stop_threshold=self.stop_threshold,
                mode=m,
            )

            # SHIFT(1): the signal generated at close t applies to the returns of t+1.
            pos_series = pd.Series(pos_array, index=full_df.index, dtype=float).shift(1)

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
                "window": int(w),
                "threshold": float(thresh),
                "mode": str(m),
                "exit_threshold": float(self.exit_threshold),
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
                best_w = w
                best_thresh = thresh
                best_mode = m
                best_is_metrics = is_metrics

        if best_w is None or best_thresh is None or best_mode is None:
            raise RuntimeError("Nessuna combinazione valida trovata durante l'ottimizzazione.")

        # Out-of-sample metrics of the selected parameters.
        best_oos_metrics = None
        if val_df is not None and not val_df.empty:
            best_match = next(
                (
                    c for c in self.combinations_log
                    if c["window"] == best_w and c["threshold"] == best_thresh and c["mode"] == best_mode
                ),
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
            "window": int(best_w),
            "threshold": float(best_thresh),
            "exit_threshold": float(self.exit_threshold),
            "stop_threshold": self.stop_threshold,
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
# 3. Z-SCORE WALK-FORWARD LOOP (5-year in-sample training -> 1-year out-of-sample validation)
# =====================================================================
class ZScoreWalkForwardTrainer:
    """Run the walk-forward procedure for the Z-score agent.

    - Rolling window: 5 years of training (in-sample) followed by one validation
      year (out-of-sample).
    - Optimises on the training window with the chosen stability/risk metric
      (``'sharpe_mdd'``).
    - Detects overfitting by comparing in-sample and out-of-sample results.
    - Keeps the full log of ALL evaluated combinations.
    - Selects the most frequent (window, threshold, mode) combination (statistical mode).
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
        self.all_combinations_history: List[Dict[str, Any]] = []
        self.selected_params: Optional[Dict[str, Any]] = None

    def run(self, df: pd.DataFrame, max_year: Optional[int] = None, initial_warmup_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run the rolling loop over the dataset.

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
            f"\nInizio Walk-Forward Z-Score: {len(years)} anni ({years[0]} - {years[-1]}) | "
            f"Metrica: '{self.optimizer.objective_metric}'"
        )
        print("-" * 120)

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
                    c["window"] == best["window"]
                    and c["threshold"] == best["threshold"]
                    and c["mode"] == best["mode"]
                )
                self.all_combinations_history.append(rec)

            res = {
                "train_years": f"{train_span[0]}-{train_span[-1]}",
                "val_year": int(val_year),
                "window": best["window"],
                "threshold": best["threshold"],
                "exit_threshold": best["exit_threshold"],
                "stop_threshold": best.get("stop_threshold"),
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

        # Select the mode.
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
            "stop_threshold": self.optimizer.stop_threshold,
            "mode": most_common_tuple[2],
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

        print("\n" + "=" * 70)
        print("          RIEPILOGO OTTIMIZZAZIONE & DIAGNOSTICA OVERFITTING")
        print("=" * 70)
        print(
            f"Parametri selezionati (Moda): Window={self.selected_params['window']}, "
            f"Threshold={self.selected_params['threshold']}, Mode={self.selected_params['mode']}, "
            f"ExitThreshold={self.selected_params['exit_threshold']}"
        )
        print(f"Frequenza di apparizione    : {self.selected_params['frequency']}")
        print(f"Combinazioni totali loggate : {len(self.all_combinations_history):,} record")
        print(f"Rendimento medio cumulato   : In-Sample={mean_is_ret:+.1f}% (5y)  | Out-Of-Sample={mean_oos_ret:+.1f}% (1y)")
        print(f"Sharpe Ratio medio          : In-Sample={mean_is_sharpe:+.2f}       | Out-Of-Sample={mean_oos_sharpe:+.2f}")
        print(f"Max Drawdown medio          : In-Sample={mean_is_mdd:.1f}%          | Out-Of-Sample={mean_oos_mdd:.1f}%")
        print(f"Anni OOS con rendimento > 0 : {oos_positive_years}/{len(oos_rets)} ({oos_win_rate}%)")
        print("=" * 70)

        return self.selected_params

    def get_combinations_dataframe(self) -> pd.DataFrame:
        """Return the full log of every combination tested in every window."""
        return pd.DataFrame(self.all_combinations_history)

    def get_combinations_summary(self) -> pd.DataFrame:
        """Return a summary of all (window, threshold, mode) combinations across walk-forward windows.
        """
        df_all = self.get_combinations_dataframe()
        if df_all.empty:
            return pd.DataFrame()

        summary = (
            df_all.groupby(["window", "threshold", "mode"])
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
        """Save the modal parameters to a JSON file readable by ZScoreAgent."""
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
                    "stop_threshold": self.selected_params.get("stop_threshold"),
                    "mode": self.selected_params["mode"],
                    "price_col": self.selected_params["price_col"],
                },
                f,
                indent=4,
            )
        print(f"Configurazione Z-Score salvata con successo in: {path}")