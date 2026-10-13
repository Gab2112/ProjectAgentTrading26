import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd


class SMAEMAAgent:
  """Agente puramente inferenziale: prende i parametri (o li carica da JSON)

  ed emette la pd.Series di segnali (1: BUY, -1: SELL, 0: HOLD).
  """

  def __init__(
      self,
      sma_window: int | None = None,
      ema_window: int | None = None,
      price_col: str = "close",
      output_format: str = "numeric",
  ):
    self.sma_window = sma_window
    self.ema_window = ema_window
    self.price_col = price_col
    self.output_format = output_format

  def load_params(self, filepath: str | Path):
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as f:
      params = json.load(f)
    self.sma_window = int(params["sma_window"])
    self.ema_window = int(params["ema_window"])
    return self

  def predict(self, df: pd.DataFrame) -> pd.Series:
    if self.sma_window is None or self.ema_window is None:
      raise ValueError(
          "Parametri non impostati. Inizializzali o usa .load_params()."
      )

    close = df[self.price_col]
    sma = close.rolling(window=self.sma_window).mean()
    ema = close.ewm(span=self.ema_window, adjust=False).mean()

    raw_signal = np.where(ema > sma, 1, -1)
    signals = pd.Series(raw_signal, index=df.index, dtype=int)
    signals[sma.isna()] = 0  # Warm-up period iniziale

    if self.output_format == "string":
      return signals.map({1: "BUY", -1: "SELL", 0: "HOLD"})
    return signals


class SMAEMAOptimizer:
  """Ottimizzatore: calcola gli indicatori congiuntamente (train + val per continuità)

  e massimizza la metrica target solo sulle osservazioni di validation.
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
    self.best_params = None

  def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    val_indices = val_df.index
    # Concateniamo per non perdere i primi N valori di validation a causa del rolling window
    full_df = pd.concat([train_df, val_df])
    close = full_df[self.price_col]
    val_returns = val_df[self.price_col].pct_change().fillna(0.0)

    best_score = -np.inf
    best_params = None

    param_grid = [
        (s, e)
        for s, e in itertools.product(self.sma_range, self.ema_range)
        if e < s
    ]

    for sma_w, ema_w in param_grid:
      sma = close.rolling(window=sma_w).mean()
      ema = close.ewm(span=ema_w, adjust=False).mean()

      pos = np.where(ema > sma, 1.0, -1.0)
      # Shift(1) per evitare look-ahead bias
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
            "sma_window": int(sma_w),
            "ema_window": int(ema_w),
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