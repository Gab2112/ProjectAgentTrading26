"""
base_agent.py

Classe base astratta per tutti gli agenti di trading del framework ProjectAgentTrading26.
Definisce l'interfaccia standard e condivisa per:
- Inferenza e generazione segnali: predict() e generate_signals()
- Convenzione dei segnali: 1 (BUY/LONG), -1 (SELL/SHORT), 0 (HOLD/FLAT)
- Serializzazione e caricamento parametri: load_params(), save_params()
- Formattazione output uniforme (numeric [-1, 0, 1] o string ['SELL', 'HOLD', 'BUY'])
"""

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd


class BaseAgent(ABC):
    """
    Classe base astratta per tutti gli agenti di trading.

    Tutti gli agenti derivati devono implementare il metodo astratto:
        predict(df: pd.DataFrame) -> pd.Series
    """

    def __init__(
        self,
        name: str = "BaseAgent",
        output_format: str = "numeric",  # 'numeric' (-1, 0, 1) o 'string' ('SELL', 'HOLD', 'BUY')
    ):
        self.name = name
        self.output_format = output_format
        self.params: Dict[str, Any] = {}

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcola e restituisce la serie dei segnali operativi allineata all'indice del DataFrame.

        Parametri:
            df (pd.DataFrame): DataFrame contenente i dati storici (prezzi, feature o testi).

        Ritorna:
            pd.Series: Serie temporale con i segnali generati:
                       -1 (SELL/SHORT), 0 (HOLD/FLAT), 1 (BUY/LONG)
                       oppure le corrispondenti stringhe se output_format='string'.
        """
        pass

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Alias intuitivo per il metodo predict().
        Garantisce coerenza con le convenzioni di signal_generation della pipeline.
        """
        return self.predict(df)

    def load_params(self, filepath: Union[str, Path]) -> "BaseAgent":
        """
        Carica i parametri ottimali da un file JSON e aggiorna lo stato dell'agente.

        Parametri:
            filepath: Percorso del file JSON contenente i parametri.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File parametri non trovato: {path}")

        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)

        self.params = params
        self._apply_params(params)
        return self

    def save_params(self, filepath: Union[str, Path]) -> None:
        """
        Salva i parametri attuali dell'agente in un file JSON.

        Parametri:
            filepath: Percorso di destinazione del file JSON.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.params, f, indent=4)

    def _apply_params(self, params: Dict[str, Any]) -> None:
        """
        Hook opzionale: aggiorna gli attributi specifici dell'agente derivato
        a partire dal dizionario dei parametri caricato.
        """
        pass

    def _format_signals(self, signals: pd.Series) -> pd.Series:
        """
        Normalizza e valida la serie dei segnali:
        - Sostituisce eventuali NaN con 0 (HOLD/FLAT)
        - Converte in interi {-1, 0, 1}
        - Mappa nei valori stringa se output_format == 'string'
        """
        formatted = signals.fillna(0).astype(int)

        # Controllo di conformità sui valori ammessi dal progetto (-1, 0, 1)
        valid_values = {-1, 0, 1}
        observed = set(formatted.unique())
        if not observed.issubset(valid_values):
            raise ValueError(
                f"[{self.name}] Valori non ammessi rilevati nei segnali: {observed}. "
                f"I valori devono appartenere a {-1, 0, 1}."
            )

        if self.output_format == "string":
            mapping = {1: "BUY", -1: "SELL", 0: "HOLD"}
            return formatted.map(mapping)

        return formatted

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', output_format='{self.output_format}', params={self.params})"
