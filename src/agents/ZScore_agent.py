import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd

class ZScoreAgent:
  """Agente puramente inferenziale per Z-Score."""

  def __init__(
      self,
      window: int | None = None,
      threshold: float | None = None,
      price_col: str = "close",
      output_format: str = "numeric",
  ):
    self.window = window
    self.threshold = threshold
    self.price_col = price_col
    self.output_format = output_format

  def load_params(self, filepath: str | Path):
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as f:
      params = json.load(f)
    self.window = int(params["window"])
    self.threshold = float(params["threshold"])
    return self

  def predict(self, df: pd.DataFrame) -> pd.Series:
    if self.window is None or self.threshold is None:
      raise ValueError(
          "Parametri non impostati. Inizializzali o usa .load_params()."
      )

    close = df[self.price_col]
    mean = close.rolling(window=self.window).mean()
    std = close.rolling(window=self.window).std()
    z_score = (close - mean) / std

    # Mean reversion: compra sotto -threshold, vendi sopra +threshold
    raw_signal = np.where(z_score < -self.threshold, 1, 0)
    raw_signal = np.where(z_score > self.threshold, -1, raw_signal)

    signals = pd.Series(raw_signal, index=df.index, dtype=int)
    signals[z_score.isna()] = 0

    if self.output_format == "string":
      return signals.map({1: "BUY", -1: "SELL", 0: "HOLD"})
    return signals


class ZScoreOptimizer:
  """Ottimizzatore su Validation per Z-Score."""

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
    full_df = pd.concat([train_df, val_df])
    close = full_df[self.price_col]
    val_returns = val_df[self.price_col].pct_change().fillna(0.0)

    best_score = -np.inf
    best_params = None

    param_grid = list(
        itertools.product(self.window_range, self.threshold_range)
    )

    for w, thresh in param_grid:
      mean = close.rolling(window=w).mean()
      std = close.rolling(window=w).std()
      z = (close - mean) / std

      pos = np.where(z < -thresh, 1.0, 0.0)
      pos = np.where(z > thresh, -1.0, pos)
      pos_series = (
          pd.Series(pos, index=full_df.index)
          .shift(1)
          .reindex(val_indices)
          .fillna(0.0)
      )

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
    if self.best_params is None:
      raise ValueError("Nessun parametro trovato. Esegui prima .fit().")
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
      json.dump(self.best_params, f, indent=4)