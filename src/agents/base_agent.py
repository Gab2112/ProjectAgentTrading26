"""
base_agent.py

Classe base astratta per tutti gli agenti di trading del framework ProjectAgentTrading26.
Definisce l'interfaccia standard e condivisa per:
- Inferenza e generazione segnali: predict() e generate_signals()
- Convenzione dei segnali: 1 (BUY/LONG), -1 (SELL/SHORT), 0 (HOLD/FLAT)
- Accesso ai dati tramite src/data_loader.py (load_data)
- Risoluzione automatica delle colonne di prezzo per asset specifici
- Serializzazione e caricamento parametri: load_params(), save_params()
- Formattazione output uniforme (numeric [-1, 0, 1] o string ['SELL', 'HOLD', 'BUY'])
"""

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.data_loader import load_agent_dataset, prepare_asset_data
from src.utils import ASSETS


class BaseAgent(ABC):
    """
    Classe base astratta per tutti gli agenti di trading.

    Tutti gli agenti derivati devono implementare il metodo astratto:
        predict(df: pd.DataFrame) -> pd.Series
    """

    DEFAULT_ASSETS = ASSETS

    def __init__(
        self,
        name: str = "BaseAgent",
        target_asset: Optional[str] = None,
        price_col: Optional[str] = None,
        output_format: str = "numeric",  # 'numeric' (-1, 0, 1) o 'string' ('SELL', 'HOLD', 'BUY')
    ):
        self.name = name
        self.target_asset = target_asset.upper() if target_asset else None
        self.price_col = price_col or (f"{self.target_asset}_close" if self.target_asset else "close")
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

    def predict_with_confidence(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calcola sia i segnali operativi discreti {-1, 0, 1} sia la stima di confidenza / probabilità [0.0, 1.0].
        Di default assegna confidenza 1.0 ai segnali attivi (-1, 1) e 0.0 a HOLD (0).
        Le classi derivate possono specializzare il calcolo con modelli probabilistici o metriche continue.
        """
        signals = self.predict(df)
        confidence = pd.Series(np.where(signals != 0, 1.0, 0.0), index=df.index, dtype=float)
        return signals, confidence

    def resolve_price_col(
        self,
        df: pd.DataFrame,
        price_col: Optional[str] = None,
        target_asset: Optional[str] = None,
    ) -> str:
        """
        Risolve in modo intelligente e tollerante la colonna dei prezzi nel DataFrame.
        Supporta sia DataFrame con colonne generiche ('close', 'Close') sia
        DataFrame multi-asset con prefisso ('SPY_close', 'IEF_close', ecc.).
        """
        candidate_col = price_col or self.price_col
        asset = (target_asset or self.target_asset or "").upper()

        # 1. Se il candidato è già esattamente presente
        if candidate_col and candidate_col in df.columns:
            return candidate_col

        # 2. Se è specificato un asset target, cerca la combinazione f"{asset}_close"
        if asset:
            for suffix in ["_close", "_Close", "_CLOSE"]:
                prefixed = f"{asset}{suffix}"
                if prefixed in df.columns:
                    return prefixed
            for suffix in ["_close", "_Close"]:
                prefixed = f"{asset.lower()}{suffix}"
                if prefixed in df.columns:
                    return prefixed

        # 3. Controllo colonne generiche 'close' o 'Close'
        for col_name in ["close", "Close", "Adj Close", "adj_close"]:
            if col_name in df.columns:
                return col_name

        # 4. Se df ha una sola colonna che finisce per '_close'
        close_like = [c for c in df.columns if str(c).lower().endswith("_close")]
        if len(close_like) == 1:
            return close_like[0]

        raise KeyError(
            f"[{self.name}] Impossibile determinare la colonna del prezzo nel DataFrame. "
            f"Candidato cercato: '{candidate_col}', target_asset: '{asset}'. "
            f"Colonne disponibili: {list(df.columns)}"
        )

    # =====================================================================
    # CARICAMENTO DATI (delegato a src/data_loader.py)
    # =====================================================================


    def load_data(
        self,
        asset: Optional[str] = None,
        frequency: str = "weekly",
        include_macro: bool = True,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        root_dir: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Metodo di istanza per caricare direttamente i dati preparati per l'agente.
        Se target_asset è configurato nell'agente, adatta automaticamente le colonne.
        """
        target = (asset or self.target_asset or "").upper()
        df = load_agent_dataset(
            frequency=frequency,
            include_macro=include_macro,
            start_date=start_date,
            end_date=end_date,
            root_dir=root_dir,
        )

        if target:
            return prepare_asset_data(df, asset=target, price_col=self.price_col)
        return df

    # =====================================================================
    # GESTIONE PARAMETRI E SERIALIZZAZIONE
    # =====================================================================
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
        if "target_asset" in params:
            self.target_asset = str(params["target_asset"]).upper()
        if "price_col" in params:
            self.price_col = str(params["price_col"])

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
        asset_str = f", target_asset='{self.target_asset}'" if self.target_asset else ""
        return f"{self.__class__.__name__}(name='{self.name}'{asset_str}, output_format='{self.output_format}', params={self.params})"
