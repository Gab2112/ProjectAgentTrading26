"""
FinBERT_agent.py

Agente emotivo / sentiment basato su FinBERT (ProsusAI/finbert o altro modello HuggingFace).
Eredita da BaseAgent per garantire un'interfaccia standardizzata.

Miglioramenti introdotti:
- Risolto il bug dell'indicizzazione fissa (p[0], p[1]) tramite mappatura dinamica con label2id.
- Gestione corretta della probabilità neutrale per evitare falsi positivi quando il testo è prevalentemente neutro.
- Aggiunto parametro 'invert_sentiment' (utile per ETF obbligazionari, dove notizie macro 'positive'
  sull'economia portano a rialzi dei tassi e calo dei prezzi dei bond).
- Metodo predict() standardizzato e compatibile con BaseAgent (con get_signals e run mantenuti per retrocompatibilità).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base_agent import BaseAgent


class FinBERTAgent(BaseAgent):
    """
    Agente di Sentiment Analysis basato su Transformer pre-addestrato FinBERT:
    - Elabora testi/headline finanziarie.
    - Calcola probabilità softmax su positivo, negativo e neutrale.
    - Genera segnali di BUY (+1), SELL (-1) o HOLD (0).
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        confidence_threshold: float = 0.5,
        text_col: str = "headline",
        invert_sentiment: bool = False,
        output_format: str = "numeric",  # 'numeric' (-1, 0, 1) o 'string' ('SELL', 'HOLD', 'BUY')
        device: Optional[str] = None,
    ):
        super().__init__(name="FinBERTAgent", output_format=output_format)
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.text_col = text_col
        self.invert_sentiment = invert_sentiment
        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.params = {
            "model_name": self.model_name,
            "confidence_threshold": float(self.confidence_threshold),
            "text_col": self.text_col,
            "invert_sentiment": bool(self.invert_sentiment),
        }

        print(f"[FinBERTAgent] Caricamento '{self.model_name}' su {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.device)
        self.model.eval()

        # Mappatura dinamica delle label dal config del modello
        config_label2id = getattr(self.model.config, "label2id", {})
        norm_label2id = {str(k).lower(): v for k, v in config_label2id.items()}
        self.pos_idx = norm_label2id.get("positive", 0)
        self.neg_idx = norm_label2id.get("negative", 1)
        self.neu_idx = norm_label2id.get("neutral", 2)

    def _apply_params(self, params: Dict[str, Any]) -> None:
        """Aggiorna i parametri dell'agente dopo il caricamento da JSON."""
        if "confidence_threshold" in params:
            self.confidence_threshold = float(params["confidence_threshold"])
        if "text_col" in params:
            self.text_col = str(params["text_col"])
        if "invert_sentiment" in params:
            self.invert_sentiment = bool(params["invert_sentiment"])

    def predict(
        self, df: pd.DataFrame, batch_size: int = 32
    ) -> pd.Series:
        """
        Calcola i segnali di trading a partire dalle headline testuali nel DataFrame.

        Parametri:
            df (pd.DataFrame): DataFrame contenente la colonna di testo self.text_col.
            batch_size (int): Dimensione dei batch per l'inferenza del transformer.

        Ritorna:
            pd.Series: Serie dei segnali (-1, 0, 1) allineata all'indice temporale del DataFrame.
        """
        if self.text_col not in df.columns:
            # Se la colonna di testo non è presente nel dataset, restituisce una serie di HOLD (0)
            signals = pd.Series(0, index=df.index, dtype=int)
            return self._format_signals(signals)

        signals = pd.Series(0, index=df.index, dtype=int)

        # Maschera per elaborare solo le righe con testo presente e valido
        has_text_mask = df[self.text_col].notna() & (
            df[self.text_col].astype(str).str.strip() != ""
        )
        valid_texts = df.loc[has_text_mask, self.text_col].astype(str).tolist()

        if not valid_texts:
            return self._format_signals(signals)

        computed_signals: List[int] = []

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
                    pos_p = float(p[self.pos_idx])
                    neg_p = float(p[self.neg_idx])
                    neu_p = (
                        float(p[self.neu_idx])
                        if self.neu_idx < len(p)
                        else 0.0
                    )

                    # Verifica che il sentiment direzionale domini il neutrale e superi la soglia
                    if pos_p > neg_p and pos_p > neu_p and pos_p >= self.confidence_threshold:
                        sig = -1 if self.invert_sentiment else 1
                    elif neg_p > pos_p and neg_p > neu_p and neg_p >= self.confidence_threshold:
                        sig = 1 if self.invert_sentiment else -1
                    else:
                        sig = 0

                    computed_signals.append(sig)

        signals.loc[has_text_mask] = computed_signals
        return self._format_signals(signals)

    def get_signals(
        self, df: pd.DataFrame, batch_size: int = 32
    ) -> pd.Series:
        """Alias per retrocompatibilità con la versione precedente."""
        return self.predict(df, batch_size=batch_size)

    def run(
        self,
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        filepath: Optional[Union[str, Path]] = None,
        force_retrain: bool = False,
    ) -> pd.Series:
        """Metodo di interfaccia unificata per retrocompatibilità."""
        if test_df is None:
            raise ValueError("È necessario passare test_df per estrarre i segnali.")
        return self.predict(test_df)