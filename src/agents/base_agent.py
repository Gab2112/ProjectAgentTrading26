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
from typing import Any, Dict, Optional, Tuple, Union
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

    def predict_with_confidence(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calcola sia i segnali operativi discreti {-1, 0, 1} sia la stima di confidenza / probabilità [0.0, 1.0].
        Di default assegna confidenza 1.0 ai segnali attivi (-1, 1) e 0.0 a HOLD (0).
        Le classi derivate possono specializzare il calcolo con modelli probabilistici o metriche continue.
        """
        signals = self.predict(df)
        confidence = pd.Series(np.where(signals != 0, 1.0, 0.0), index=df.index, dtype=float)
        return signals, confidence

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


# =====================================================================
# FUNZIONI DI UTILITÀ PER METRICHE DI PERFORMANCE E STABILITÀ
# =====================================================================
def calculate_performance_metrics(
    returns: pd.Series,
    annualization_factor: int = 52,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """
    Calcola le metriche complete di performance e rischio/stabilità per una serie di rendimenti periodici.

    Parametri:
        returns: Serie di rendimenti periodici (settimanali).
        annualization_factor: Fattore di annualizzazione (default 52 per frequenza settimanale).
        risk_free_rate: Tasso privo di rischio annualizzato (default 0.0).

    Ritorna:
        Dizionario con metriche: cumulative_return, annualized_return, annualized_volatility,
        sharpe_ratio, sortino_ratio, max_drawdown, profit_factor, calmar_ratio.
    """
    clean_ret = returns.fillna(0.0)
    n = len(clean_ret)
    if n == 0:
        return {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "calmar_ratio": 0.0,
        }

    cum_ret = float((1.0 + clean_ret).prod() - 1.0)
    if cum_ret <= -1.0:
        ann_ret = -1.0
    else:
        ann_ret = float((1.0 + cum_ret) ** (annualization_factor / n) - 1.0)

    vol = float(clean_ret.std(ddof=1) * np.sqrt(annualization_factor)) if n > 1 else 0.0

    excess_ret = ann_ret - risk_free_rate
    if vol > 1e-6:
        sharpe = float(excess_ret / vol)
    else:
        sharpe = 0.0 if excess_ret >= 0 else -99.0

    # Maximum Drawdown
    wealth = (1.0 + clean_ret).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    mdd = abs(float(drawdown.min()))

    # Profit Factor
    gains = float(clean_ret[clean_ret > 0].sum())
    losses = float(abs(clean_ret[clean_ret < 0].sum()))
    profit_factor = float(gains / losses) if losses > 1e-6 else (10.0 if gains > 0 else 1.0)

    # Calmar Ratio
    calmar = float(ann_ret / max(mdd, 0.01))

    # Sortino Ratio (penalizza esclusivamente la volatilita ribassista, premiando l'upside)
    downside_ret = clean_ret[clean_ret < 0]
    if len(downside_ret) > 1:
        downside_vol = float(downside_ret.std(ddof=1) * np.sqrt(annualization_factor))
    else:
        downside_vol = vol
    sortino = float(excess_ret / max(downside_vol, 1e-6)) if downside_vol > 1e-6 else (0.0 if excess_ret >= 0 else -99.0)

    return {
        "cumulative_return": cum_ret,
        "annualized_return": ann_ret,
        "annualized_volatility": vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": mdd,
        "profit_factor": profit_factor,
        "calmar_ratio": calmar,
    }


def compute_objective_score(
    metrics: Dict[str, float],
    metric_type: str = "profit_oriented",
) -> float:
    """
    Calcola lo score di ottimizzazione per la selezione dei parametri.

    Opzioni supportate:
    - 'profit_oriented' / 'profit' (default): atteggiamento orientato al profitto.
        Il rendimento cumulato e il driver principale; viene premiata la qualita
        del rendimento (Sharpe/Sortino positivo) con una penalita controllata
        sul Max Drawdown (fattore 0.20 anziche 1.0). In questo modo la strategia
        accetta drawdown fisiologici pur di cavalcare i grandi trend di profitto.
        Score = cum_ret * (1 + max(0, sharpe)) * (1 - 0.20 * MDD) se cum_ret > 0.
    - 'sharpe_mdd': Sharpe ratio fortemente penalizzato per il Max Drawdown (conservativo).
        Score = Sharpe * (1 - MDD) se Sharpe >= 0.
    - 'sortino': Sortino ratio puro (penalizza solo la volatilita ribassista).
    - 'sharpe': Sharpe ratio annualizzato standard.
    - 'calmar': Calmar ratio (rendimento annualizzato / max drawdown).
    - 'return': Rendimento cumulato puro (nessuna penalizzazione del rischio).
    """
    sharpe = float(metrics.get("sharpe_ratio", 0.0))
    mdd = float(metrics.get("max_drawdown", 0.0))
    ann_ret = float(metrics.get("annualized_return", 0.0))
    cum_ret = float(metrics.get("cumulative_return", 0.0))
    sortino = float(metrics.get("sortino_ratio", sharpe))

    if metric_type in ("profit_oriented", "profit"):
        if cum_ret > 0:
            quality_boost = 1.0 + max(0.0, min(sharpe, 3.0))
            drawdown_penalty = 1.0 - 0.20 * min(mdd, 0.99)
            return float(cum_ret * quality_boost * drawdown_penalty)
        else:
            return float(cum_ret * (1.0 + min(mdd, 1.0)))
    elif metric_type == "sharpe_mdd":
        if sharpe >= 0:
            return float(sharpe * (1.0 - min(mdd, 0.99)))
        else:
            return float(sharpe * (1.0 + min(mdd, 1.0)))
    elif metric_type == "sortino":
        return sortino
    elif metric_type == "sharpe":
        return sharpe
    elif metric_type == "calmar":
        return float(ann_ret / max(mdd, 0.02))
    elif metric_type == "return":
        return cum_ret
    else:
        raise ValueError(f"Metrica obiettivo '{metric_type}' non riconosciuta.")
