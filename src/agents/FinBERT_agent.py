from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class FinBERTAgent:

  def __init__(
      self,
      model_name: str = "ProsusAI/finbert",
      confidence_threshold: float = 0.5,
      text_col: str = "headline",
      output_format: str = "string",  # 'string' ('BUY'/'SELL'/'HOLD') oppure 'numeric' (1/-1/0)
      device: str | None = None,
  ):
    self.model_name = model_name
    self.confidence_threshold = confidence_threshold
    self.text_col = text_col
    self.output_format = output_format
    self.device = (
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(
        f"[FinBERTAgent] Caricamento '{self.model_name}' su {self.device}..."
    )
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    self.model = AutoModelForSequenceClassification.from_pretrained(
        self.model_name
    ).to(self.device)
    self.model.eval()

  def get_signals(
      self, df: pd.DataFrame, batch_size: int = 32
  ) -> pd.Series:
    """Restituisce una pd.Series con i segnali allineati all'indice temporale del DataFrame.

    - Se il testo è vuoto o assente (NaN) -> 'HOLD' (0)
    - Se P(positive) >= threshold       -> 'BUY'  (1)
    - Se P(negative) >= threshold       -> 'SELL' (-1)
    - Altrimenti                        -> 'HOLD' (0)
    """
    signals = pd.Series("HOLD", index=df.index, dtype="object")

    # Maschera per elaborare solo le righe con testo presente
    has_text_mask = df[self.text_col].notna() & (
        df[self.text_col].astype(str).str.strip() != ""
    )
    valid_texts = df.loc[has_text_mask, self.text_col].astype(str).tolist()

    if not valid_texts:
      if self.output_format == "numeric":
        return pd.Series(0, index=df.index, dtype=int)
      return signals

    computed_signals = []

    with torch.no_grad():
      for i in range(0, len(valid_texts), batch_size):
        batch = valid_texts[i : i + batch_size]
        inputs = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for p in probs:
          pos_p, neg_p = float(p[0]), float(p[1])

          if pos_p > neg_p and pos_p >= self.confidence_threshold:
            computed_signals.append("BUY")
          elif neg_p > pos_p and neg_p >= self.confidence_threshold:
            computed_signals.append("SELL")
          else:
            computed_signals.append("HOLD")

    # Inserimento dei segnali calcolati solo dove c'era testo valido
    signals.loc[has_text_mask] = computed_signals

    if self.output_format == "numeric":
      mapping = {"BUY": 1, "SELL": -1, "HOLD": 0}
      return signals.map(mapping).astype(int)

    return signals

  def run(
      self,
      train_df: pd.DataFrame | None = None,
      test_df: pd.DataFrame = None,
      filepath: str | Path | None = None,
      force_retrain: bool = False,
  ) -> pd.Series:
    """Interfaccia coerente con gli altri due agenti:

    restituisce il vettore/Series di segnali sul test_df.
    """
    if test_df is None:
      raise ValueError("È necessario passare test_df per estrarre i segnali.")

    return self.get_signals(test_df)