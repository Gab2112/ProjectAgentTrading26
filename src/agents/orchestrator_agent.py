"""
orchestrator_agent.py

Agente Orchestratore basato su Filtro di Kalman (Kalman Filter) per l'Ensemble Dinamico di Segnali.

Ruolo nell'architettura ProjectAgentTrading26:
- Riceve in input una serie di segnali operativi provenienti da diversi agenti specializzati
  (es. Agente Statistico SMA/EMA & Z-Score, Agente Macro Yield Curve, Agente Sentiment FinBERT, ecc.).
- Ciascun agente fornisce un DataFrame con esattamente due colonne: 'signal' ({-1, 0, 1}) e 'confidence' ([0.0, 1.0]).
- NON richiede una fase di addestramento esplicita / statica: utilizza un Filtro di Kalman ricorsivo (Dynamic State-Space Model)
  per aggiornare costantemente e in tempo reale i pesi ottimali degli agenti all'arrivo dei rendimenti effettivi di mercato.
- Stima lo stato latente beta_t (affidabilità e capacità predittiva di ciascun agente) e la matrice di incertezza / covarianza P_t.
- Produce in output un unico DataFrame finale consolidato con esattamente due colonne:
    1. 'signal': segnale discreto {-1, 0, 1}
    2. 'confidence': confidenza / probabilità continua [0.0, 1.0]
  destinato alle decisioni finali di asset allocation, position sizing e risk management.
- Mantiene anche la classe StaticOrchestratorAgent / OrchestratorOptimizer per consentire il confronto
  diretto tra l'ottimizzazione statica In-Sample e l'ottimizzazione dinamica continua con Filtro di Kalman.
"""

import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Risoluzione robusta dei percorsi per esecuzione diretta o come modulo
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from src.agents.base_agent import BaseAgent, calculate_performance_metrics, compute_objective_score
    from src.agents.macro_yield_curve_agent import MacroYieldCurveAgent
    from src.agents.EMA_EMW_agent import SMAEMAAgent
    from src.agents.ZScore_agent import ZScoreAgent
except ImportError:
    from .base_agent import BaseAgent, calculate_performance_metrics, compute_objective_score
    from .macro_yield_curve_agent import MacroYieldCurveAgent
    from .EMA_EMW_agent import SMAEMAAgent
    from .ZScore_agent import ZScoreAgent


# =====================================================================
# FUNZIONI DI SUPPORTO PER ALLINEAMENTO E VALIDAZIONE SEGNALI
# =====================================================================
def validate_signal_dataframe(df: pd.DataFrame, source_name: str = "SignalSource") -> pd.DataFrame:
    """
    Verifica che il DataFrame contenga le colonne richieste ['signal', 'confidence']
    e che i valori siano conformi alle convenzioni di progetto:
    - signal in {-1, 0, 1}
    - confidence in [0.0, 1.0]
    Normalizza l'indice a DatetimeIndex e gestisce eventuali valori mancanti.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"[{source_name}] L'input deve essere un pd.DataFrame, ricevuto: {type(df)}")

    cols_map = {col.lower(): col for col in df.columns}
    if "signal" not in cols_map or "confidence" not in cols_map:
        raise KeyError(
            f"[{source_name}] Il DataFrame deve contenere le colonne 'signal' e 'confidence'. "
            f"Colonne presenti: {list(df.columns)}"
        )

    clean_df = pd.DataFrame(index=df.index)
    clean_df["signal"] = df[cols_map["signal"]].fillna(0).astype(int)
    clean_df["confidence"] = df[cols_map["confidence"]].fillna(0.0).astype(float).clip(0.0, 1.0)

    # Verifica valori discreti ammessi
    valid_signals = {-1, 0, 1}
    obs_signals = set(clean_df["signal"].unique())
    if not obs_signals.issubset(valid_signals):
        clean_df["signal"] = np.sign(clean_df["signal"]).astype(int)

    if not isinstance(clean_df.index, pd.DatetimeIndex):
        try:
            clean_df.index = pd.to_datetime(clean_df.index)
        except Exception:
            pass

    return clean_df[["signal", "confidence"]]


def align_signals(
    signals_dict: Dict[str, pd.DataFrame],
    fill_method: str = "zeros",
) -> Tuple[Dict[str, pd.DataFrame], pd.DatetimeIndex]:
    """
    Allinea una collezione di DataFrame di segnali su un indice temporale comune.
    """
    if not signals_dict:
        raise ValueError("Il dizionario dei segnali in input è vuoto.")

    validated = {}
    for name, df in signals_dict.items():
        validated[name] = validate_signal_dataframe(df, source_name=name)

    if fill_method == "inner":
        common_idx = validated[list(validated.keys())[0]].index
        for name in list(validated.keys())[1:]:
            common_idx = common_idx.intersection(validated[name].index)
        common_idx = common_idx.sort_values()
        aligned = {name: df.reindex(common_idx) for name, df in validated.items()}
    else:  # 'zeros'
        union_idx = validated[list(validated.keys())[0]].index
        for name in list(validated.keys())[1:]:
            union_idx = union_idx.union(validated[name].index)
        union_idx = union_idx.sort_values()
        aligned = {}
        for name, df in validated.items():
            reindexed = df.reindex(union_idx)
            reindexed["signal"] = reindexed["signal"].fillna(0).astype(int)
            reindexed["confidence"] = reindexed["confidence"].fillna(0.0).astype(float)
            aligned[name] = reindexed

    return aligned, aligned[list(aligned.keys())[0]].index


# =====================================================================
# 1. AGENTE ORCHESTRATORE DINAMICO CON FILTRO DI KALMAN
# =====================================================================
class KalmanOrchestratorAgent(BaseAgent):
    """
    Agente Orchestratore basato su Filtro di Kalman (Dynamic State-Space Model).

    Caratteristiche:
    - Nessun training statico offline richiesto: i pesi vengono aggiornati ricorsivamente
      a ogni osservazione settimanale con la formula bayesiana di Kalman.
    - Lo stato latente beta_t rappresenta la sensibilità / efficacia predittiva di ciascun agente.
    - Modello di misura: y_t = R_t = sum_k beta_{k,t} * x_{k,t-1} + epsilon_t, dove x_k = signal_k * confidence_k.
    - Matrice di covarianza dello stato P_t: quantifica l'incertezza sulla stima dei pesi.
      Minore è la traccia di P_t (alta precisione), maggiore è la confidenza assegnata al segnale finale.
    - Forgetting factor (lambda): permette al filtro di adattarsi agilmente ai cambi di regime di mercato
      senza accumulare memoria infinita.
    - Output: DataFrame con esattamente 2 colonne: 'signal' ({-1, 0, 1}) e 'confidence' ([0.0, 1.0]).
    """

    def __init__(
        self,
        forgetting_factor: float = 0.99,
        threshold_buy: float = 0.25,
        threshold_sell: float = 0.25,
        consensus_boost: float = 0.15,
        conflict_penalty: float = 0.25,
        initial_variance: float = 1.0,
        observation_noise: float = 0.001,
        weight_mode: str = "non_negative",  # 'non_negative', 'softmax', 'unconstrained'
        name: str = "KalmanOrchestratorAgent",
        output_format: str = "numeric",
    ):
        super().__init__(name=name, output_format=output_format)
        self.forgetting_factor = float(forgetting_factor)
        self.threshold_buy = float(threshold_buy)
        self.threshold_sell = float(threshold_sell)
        self.consensus_boost = float(consensus_boost)
        self.conflict_penalty = float(conflict_penalty)
        self.initial_variance = float(initial_variance)
        self.observation_noise = float(observation_noise)
        self.weight_mode = weight_mode

        # Stato interno del filtro
        self.k_agents: int = 0
        self.agent_names: List[str] = []
        self.beta: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.V: float = self.observation_noise
        self.last_convictions: Optional[np.ndarray] = None

        self.params = {
            "forgetting_factor": self.forgetting_factor,
            "threshold_buy": self.threshold_buy,
            "threshold_sell": self.threshold_sell,
            "consensus_boost": self.consensus_boost,
            "conflict_penalty": self.conflict_penalty,
            "initial_variance": self.initial_variance,
            "observation_noise": self.observation_noise,
            "weight_mode": self.weight_mode,
        }

    def _apply_params(self, params: Dict[str, Any]) -> None:
        """Aggiorna i parametri dal dizionario caricato."""
        if "forgetting_factor" in params:
            self.forgetting_factor = float(params["forgetting_factor"])
        if "threshold_buy" in params:
            self.threshold_buy = float(params["threshold_buy"])
        if "threshold_sell" in params:
            self.threshold_sell = float(params["threshold_sell"])
        if "threshold" in params and "threshold_buy" not in params:
            self.threshold_buy = float(params["threshold"])
            self.threshold_sell = float(params["threshold"])
        if "consensus_boost" in params:
            self.consensus_boost = float(params["consensus_boost"])
        if "conflict_penalty" in params:
            self.conflict_penalty = float(params["conflict_penalty"])
        if "weight_mode" in params:
            self.weight_mode = str(params["weight_mode"])

    def _init_state(self, k: int, agent_names: List[str]) -> None:
        """Inizializza lo stato latente e la covarianza del filtro di Kalman."""
        self.k_agents = k
        self.agent_names = list(agent_names)
        # Prior informativo iniziale: equiponderazione neutra tra gli agenti
        self.beta = np.ones(k, dtype=float) / k
        self.P = np.eye(k, dtype=float) * self.initial_variance
        self.V = float(self.observation_noise)
        self.last_convictions = np.zeros(k, dtype=float)

    def _convert_beta_to_weights(self, beta: np.ndarray) -> np.ndarray:
        """
        Trasforma lo stato beta in pesi normalizzati sum(w) = 1.0.
        """
        if self.weight_mode == "softmax":
            exp_b = np.exp(beta - np.max(beta))
            return exp_b / np.sum(exp_b)
        elif self.weight_mode == "unconstrained":
            # Pesi non vincolati (possono invertire agenti se negativi)
            denom = np.sum(np.abs(beta))
            return beta / denom if denom > 1e-6 else np.ones(len(beta)) / len(beta)
        else:  # 'non_negative' (default)
            # Solo contributi positivi: azzera gli agenti con alfa negativo o rumore
            pos_b = np.maximum(beta, 0.0)
            sum_pos = np.sum(pos_b)
            if sum_pos > 1e-6:
                return pos_b / sum_pos
            else:
                return np.ones(len(beta)) / len(beta)

    def update_step(self, realized_return: float, prev_convictions: np.ndarray) -> None:
        """
        Esegue il Measurement Update del Filtro di Kalman all'arrivo del rendimento realizzato y_t.
        """
        if self.beta is None or self.P is None:
            raise RuntimeError("Filtro di Kalman non inizializzato.")

        h = prev_convictions
        # Se tutti i segnali precedenti erano nulli, non c'è informazione da aggiornare
        if np.all(h == 0.0):
            return

        y = float(realized_return)
        y_hat = float(np.dot(h, self.beta))
        e = y - y_hat  # Innovazione (errore di predizione)

        # Varianza dell'innovazione: F_t = h' * P * h + V
        F = float(np.dot(h, np.dot(self.P, h)) + self.V)

        # Guadagno di Kalman: K_t = (P * h) / F_t
        k_gain = np.dot(self.P, h) / F

        # Aggiornamento dello stato: beta_t = beta_{t|t-1} + K_t * e_t
        self.beta = self.beta + k_gain * e

        # Aggiornamento della covarianza: P_t = (I - K_t * h') * P_{t|t-1}
        self.P = self.P - np.outer(k_gain, h) @ self.P

        # Aggiornamento online adattivo della varianza di misura V
        self.V = 0.95 * self.V + 0.05 * (e**2)

    def predict_step(self, current_convictions: np.ndarray, current_signals: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Esegue il Predict Step del Filtro di Kalman e calcola (signal, confidence, weights) per il periodo corrente.
        """
        if self.beta is None or self.P is None:
            raise RuntimeError("Filtro di Kalman non inizializzato.")

        # Predizione dello stato con forgetting factor: P_{t|t-1} = P_{t-1} / lambda
        self.P = self.P / self.forgetting_factor

        # Pesi dinamici correnti
        w = self._convert_beta_to_weights(self.beta)

        # Calcolo del punteggio grezzo aggregato: score = sum_k w_k * (signal_k * conf_k)
        score_raw = float(np.dot(current_convictions, w))

        # Analisi consenso e conflitto
        active_mask = current_signals != 0
        n_active = np.sum(active_mask)

        if n_active >= 2:
            signs = current_signals[active_mask]
            n_long = np.sum(signs == 1)
            n_short = np.sum(signs == -1)

            if n_long > 0 and n_short > 0:
                # Conflitto diretto tra agenti: penalizzazione
                score_adj = score_raw * (1.0 - self.conflict_penalty)
            elif (n_long > 0 and n_short == 0) or (n_short > 0 and n_long == 0):
                # Consenso unanime: boost di sinergia
                score_adj = np.sign(score_raw) * min(1.0, abs(score_raw) * (1.0 + self.consensus_boost))
            else:
                score_adj = score_raw
        else:
            score_adj = score_raw

        # Discretizzazione del segnale
        if score_adj >= self.threshold_buy:
            sig = 1
        elif score_adj <= -self.threshold_sell:
            sig = -1
        else:
            sig = 0

        # Calcolo confidenza finale
        if sig != 0:
            th = self.threshold_buy if sig == 1 else self.threshold_sell
            score_strength = min(1.0, abs(score_adj) / max(th, 0.05))

            # Precisione del filtro: traccia della covarianza P (minore incertezza -> maggiore precisione)
            trace_p = float(np.trace(self.P))
            precision_factor = 1.0 / (1.0 + 0.3 * np.sqrt(max(0.0, trace_p)))

            c = 0.50 + 0.50 * score_strength * precision_factor
            conf = min(1.0, max(0.50, round(float(c), 4)))
        else:
            conf = 0.0

        return sig, conf, w

    def combine(
        self,
        signals_input: Union[Dict[str, pd.DataFrame], List[pd.DataFrame]],
        returns_or_prices: Optional[Union[pd.Series, pd.DataFrame]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Elabora l'intera serie storica (o streaming) tramite il Filtro di Kalman adattivo.
        Esegue l'aggiornamento costante e dinamico dei pesi senza alcuna fase di addestramento esplicita.

        Parametri:
            signals_input: Dizionario {nome: df} con colonne ['signal', 'confidence'].
            returns_or_prices: Serie o DataFrame con rendimenti o prezzi dell'asset target.

        Ritorna:
            Tupla:
            1. df_final: DataFrame con esattamente 2 colonne: 'signal' ({-1, 0, 1}) e 'confidence' ([0.0, 1.0]).
            2. df_weights: DataFrame con la traiettoria temporale dei pesi dinamici assegnati a ciascun agente.
        """
        # Normalizzazione dizionario
        if isinstance(signals_input, list):
            signals_dict = {f"agent_{i+1}": df for i, df in enumerate(signals_input)}
        else:
            signals_dict = signals_input

        aligned_signals, common_idx = align_signals(signals_dict, fill_method="zeros")
        agent_names = list(aligned_signals.keys())
        k = len(agent_names)
        n_steps = len(common_idx)

        # Gestione rendimenti dell'asset
        if returns_or_prices is not None:
            if isinstance(returns_or_prices, pd.DataFrame):
                col = [c for c in returns_or_prices.columns if "return" in c.lower() or "close" in c.lower()][0]
                ret_series = returns_or_prices[col]
            else:
                ret_series = returns_or_prices

            if ret_series.min() >= 0.0 and ret_series.mean() > 5.0:
                asset_ret = ret_series.pct_change().fillna(0.0)
            else:
                asset_ret = ret_series.fillna(0.0)

            asset_ret = asset_ret.reindex(common_idx).fillna(0.0).to_numpy()
        else:
            # Se i rendimenti non sono forniti, non esegue il feedback di mercato
            asset_ret = np.zeros(n_steps, dtype=float)

        # Inizializzazione dello stato del filtro
        self._init_state(k, agent_names)

        # Matrici segnali e confidenze
        sig_mat = np.column_stack([aligned_signals[name]["signal"].to_numpy() for name in agent_names])
        conf_mat = np.column_stack([aligned_signals[name]["confidence"].to_numpy() for name in agent_names])
        conv_mat = sig_mat * conf_mat

        final_signals = np.zeros(n_steps, dtype=int)
        final_conf = np.zeros(n_steps, dtype=float)
        weights_history = np.zeros((n_steps, k), dtype=float)

        # Ciclo temporale ricorsivo del Filtro di Kalman
        for t in range(n_steps):
            # 1. Update step: all'arrivo del rendimento realizzato a t, aggiorna lo stato
            #    usando il segnale noto a t-1 (nessun look-ahead bias)
            if t > 0 and returns_or_prices is not None:
                self.update_step(asset_ret[t], conv_mat[t - 1])

            # 2. Predict step: predice lo stato e calcola segnale, confidenza e pesi per t
            cur_conv = conv_mat[t]
            cur_sig = sig_mat[t]
            sig_t, conf_t, w_t = self.predict_step(cur_conv, cur_sig)

            final_signals[t] = sig_t
            final_conf[t] = conf_t
            weights_history[t] = w_t

        df_final = pd.DataFrame(
            {
                "signal": final_signals,
                "confidence": final_conf,
            },
            index=common_idx,
        )

        df_weights = pd.DataFrame(
            weights_history,
            index=common_idx,
            columns=agent_names,
        )

        return df_final, df_weights

    def predict(self, signals_dict: Dict[str, pd.DataFrame]) -> pd.Series:
        """Restituisce la serie dei segnali operativi discreti {-1, 0, 1}."""
        df_res, _ = self.combine(signals_dict)
        return self._format_signals(df_res["signal"])

    def predict_with_confidence(
        self,
        signals_input: Union[Dict[str, pd.DataFrame], List[pd.DataFrame]],
        returns_or_prices: Optional[Union[pd.Series, pd.DataFrame]] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """Restituisce la tupla (signal, confidence) allineata."""
        df_res, _ = self.combine(signals_input, returns_or_prices)
        return df_res["signal"], df_res["confidence"]


# =====================================================================
# 2. AGENTE ORCHESTRATORE STATICO (Per confronto)
# =====================================================================
class StaticOrchestratorAgent(BaseAgent):
    """
    Agente Orchestratore Statico (basato su pesi fissi ottimizzati In-Sample).
    Mantenuto nel modulo per consentire il confronto comparativo diretto con il Filtro di Kalman.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        threshold_buy: float = 0.30,
        threshold_sell: float = 0.30,
        consensus_boost: float = 0.15,
        conflict_penalty: float = 0.25,
        min_active_agents: int = 1,
        name: str = "StaticOrchestratorAgent",
        output_format: str = "numeric",
    ):
        super().__init__(name=name, output_format=output_format)
        self.weights = weights or {}
        self.threshold_buy = float(threshold_buy)
        self.threshold_sell = float(threshold_sell)
        self.consensus_boost = float(consensus_boost)
        self.conflict_penalty = float(conflict_penalty)
        self.min_active_agents = int(min_active_agents)

        self.params = {
            "weights": self.weights,
            "threshold_buy": self.threshold_buy,
            "threshold_sell": self.threshold_sell,
            "consensus_boost": self.consensus_boost,
            "conflict_penalty": self.conflict_penalty,
            "min_active_agents": self.min_active_agents,
        }

    def combine(
        self,
        signals_input: Union[Dict[str, pd.DataFrame], List[pd.DataFrame]],
    ) -> pd.DataFrame:
        if isinstance(signals_input, list):
            signals_dict = {f"agent_{i+1}": df for i, df in enumerate(signals_input)}
        else:
            signals_dict = signals_input

        aligned_signals, common_idx = align_signals(signals_dict, fill_method="zeros")
        agent_names = list(aligned_signals.keys())
        k_agents = len(agent_names)

        weights_arr = np.zeros(k_agents, dtype=float)
        total_w = sum(self.weights.get(name, 0.0) for name in agent_names)
        if total_w > 1e-6:
            for i, name in enumerate(agent_names):
                weights_arr[i] = self.weights.get(name, 0.0) / total_w
        else:
            weights_arr = np.full(k_agents, 1.0 / k_agents, dtype=float)

        n_steps = len(common_idx)
        signals_mat = np.column_stack([aligned_signals[name]["signal"].to_numpy() for name in agent_names])
        conf_mat = np.column_stack([aligned_signals[name]["confidence"].to_numpy() for name in agent_names])
        conviction_mat = signals_mat * conf_mat

        raw_scores = np.dot(conviction_mat, weights_arr)
        final_signals = np.zeros(n_steps, dtype=int)
        final_conf = np.zeros(n_steps, dtype=float)

        for t in range(n_steps):
            active_mask = signals_mat[t] != 0
            n_active = np.sum(active_mask)
            if n_active < self.min_active_agents:
                final_signals[t] = 0
                final_conf[t] = 0.0
                continue

            active_signs = signals_mat[t, active_mask]
            score_t = raw_scores[t]
            n_long = np.sum(active_signs == 1)
            n_short = np.sum(active_signs == -1)

            if n_long > 0 and n_short > 0:
                adjusted_score = score_t * (1.0 - self.conflict_penalty)
            elif (n_long > 0 and n_short == 0) or (n_short > 0 and n_long == 0):
                adjusted_score = np.sign(score_t) * min(1.0, abs(score_t) * (1.0 + self.consensus_boost)) if n_active >= 2 else score_t
            else:
                adjusted_score = 0.0

            if adjusted_score >= self.threshold_buy:
                sig_out = 1
            elif adjusted_score <= -self.threshold_sell:
                sig_out = -1
            else:
                sig_out = 0

            if sig_out != 0:
                denom = max(1.0 - min(self.threshold_buy, self.threshold_sell), 0.10)
                score_strength = min(1.0, abs(adjusted_score) / denom)
                conf_out = min(1.0, max(0.50, round(float(0.50 + 0.50 * score_strength), 4)))
            else:
                conf_out = 0.0

            final_signals[t] = sig_out
            final_conf[t] = conf_out

        return pd.DataFrame({"signal": final_signals, "confidence": final_conf}, index=common_idx)

    def predict(self, df_or_signals: Dict[str, pd.DataFrame]) -> pd.Series:
        return self._format_signals(self.combine(df_or_signals)["signal"])


# Alias per interoperabilità
OrchestratorAgent = KalmanOrchestratorAgent


# =====================================================================
# 3. OTTIMIZZATORE STATICO & WALK-FORWARD PER CONFRONTO
# =====================================================================
class OrchestratorOptimizer:
    """
    Ottimizzatore statico tradizionale su campione In-Sample (usato come benchmark di confronto).
    """

    def __init__(
        self,
        objective_metric: str = "sharpe_mdd",
        max_train_year: int = 2021,
        annualization_factor: int = 52,
        weight_step: float = 0.10,
        threshold_range: Tuple[float, float, float] = (0.15, 0.55, 0.05),
        consensus_boost: float = 0.15,
        conflict_penalty: float = 0.25,
        min_active_agents: int = 1,
    ):
        self.objective_metric = objective_metric
        self.max_train_year = int(max_train_year)
        self.annualization_factor = int(annualization_factor)
        self.weight_step = float(weight_step)
        self.threshold_range = threshold_range
        self.consensus_boost = float(consensus_boost)
        self.conflict_penalty = float(conflict_penalty)
        self.min_active_agents = int(min_active_agents)

        self.best_params: Dict[str, Any] = {}
        self.in_sample_metrics: Dict[str, float] = {}

    def optimize(
        self,
        signals_dict: Dict[str, pd.DataFrame],
        returns_or_prices: Union[pd.Series, pd.DataFrame],
        asset_name: str = "ASSET",
    ) -> Tuple[StaticOrchestratorAgent, pd.DataFrame]:
        aligned_signals, common_idx = align_signals(signals_dict, fill_method="zeros")
        agent_names = list(aligned_signals.keys())
        k_agents = len(agent_names)

        if isinstance(returns_or_prices, pd.DataFrame):
            col = [c for c in returns_or_prices.columns if "return" in c.lower() or "close" in c.lower()][0]
            series = returns_or_prices[col]
        else:
            series = returns_or_prices

        asset_returns = series.pct_change().fillna(0.0) if (series.min() >= 0.0 and series.mean() > 5.0) else series.fillna(0.0)
        asset_returns = asset_returns.reindex(common_idx).fillna(0.0)

        is_mask = common_idx.year <= self.max_train_year
        is_idx = common_idx[is_mask]
        is_signals = {name: aligned_signals[name].loc[is_idx] for name in agent_names}
        is_ret = asset_returns.loc[is_idx]

        # Griglia pesi discreta
        n_steps = int(round(1.0 / self.weight_step))
        weight_combos = []
        for combo in itertools.product(range(n_steps + 1), repeat=k_agents):
            if sum(combo) == n_steps:
                weight_combos.append([v * self.weight_step for v in combo])

        th_start, th_end, th_step = self.threshold_range
        thresholds = np.round(np.arange(th_start, th_end + 1e-6, th_step), 3)

        tested = []
        best_score = -999.0
        best_record = None

        for w_vec in weight_combos:
            w_dict = {name: float(w_vec[i]) for i, name in enumerate(agent_names)}
            for th in thresholds:
                agent = StaticOrchestratorAgent(
                    weights=w_dict,
                    threshold_buy=float(th),
                    threshold_sell=float(th),
                    consensus_boost=self.consensus_boost,
                    conflict_penalty=self.conflict_penalty,
                )
                df_out = agent.combine(is_signals)
                pos = df_out["signal"].shift(1).fillna(0.0)
                strat_ret = pos * is_ret
                m = calculate_performance_metrics(strat_ret, annualization_factor=self.annualization_factor)
                score = compute_objective_score(m, metric_type=self.objective_metric)

                rec = {
                    "weights": w_dict,
                    "threshold": float(th),
                    "objective_score": round(score, 4),
                    "sharpe_ratio": round(m["sharpe_ratio"], 4),
                    "max_drawdown": round(m["max_drawdown"], 4),
                    "cumulative_return": round(m["cumulative_return"], 4),
                    "profit_factor": round(m["profit_factor"], 4),
                }
                tested.append(rec)
                if score > best_score:
                    best_score = score
                    best_record = rec

        self.best_params = best_record
        self.in_sample_metrics = best_record
        opt_agent = StaticOrchestratorAgent(
            weights=best_record["weights"],
            threshold_buy=best_record["threshold"],
            threshold_sell=best_record["threshold"],
            consensus_boost=self.consensus_boost,
            conflict_penalty=self.conflict_penalty,
            name=f"StaticOrchestrator_{asset_name}",
        )
        return opt_agent, pd.DataFrame(tested).sort_values(by="objective_score", ascending=False)


# =====================================================================
# 4. PIPELINE PRINCIPALE & CONFRONTO INTEGRATO TRA MODELLI
# =====================================================================
def run_orchestrator_pipeline(
    root_dir: Optional[Path] = None,
    assets: Optional[List[str]] = None,
    max_train_year: int = 2021,
    forgetting_factor: float = 0.99,
) -> Dict[str, pd.DataFrame]:
    """
    Esegue la pipeline completa dell'Orchestratore con Filtro di Kalman:
    1. Carica i segnali degli agenti specializzati per ciascun asset ('SHY', 'IEF', 'TLT').
    2. Esegue il Filtro di Kalman adattivo con ottimizzazione ricorsiva continua dei pesi.
    3. Esegue parallelamente l'Orchestratore statico a griglia per un confronto trasparente.
    4. Mostra a terminale la tabella comparativa completa delle performance In-Sample (2004-2021).
    5. Esporta i segnali consolidati finali (esattamente 2 colonne: 'signal' e 'confidence') in data/signals/.
    6. Salva la traiettoria dei pesi dinamici del Filtro di Kalman in data/signals/weights_kalman_{asset}.csv.
    """
    if root_dir is None:
        root_dir = Path(__file__).resolve().parents[2]

    signals_dir = root_dir / "data" / "signals"
    processed_dir = root_dir / "data" / "processed"
    config_dir = root_dir / "src" / "agents" / "config"
    results_dir = root_dir / "data" / "results"

    signals_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if assets is None:
        assets = ["SHY", "IEF", "TLT"]

    features_file = processed_dir / "features_weekly.csv"
    if not features_file.exists():
        features_file = processed_dir / "master_weekly.csv"

    df_features = pd.read_csv(features_file, index_col=0, parse_dates=True)

    # Inizializza l'agente Macro Yield Curve
    macro_agent = MacroYieldCurveAgent()
    macro_multi = macro_agent.predict_multi_asset(df_features)

    final_outputs = {}

    for asset in assets:
        asset_lower = asset.lower()
        asset_upper = asset.upper()

        print("\n" + "#" * 84)
        print(f"### ORCHESTRATORE DINAMICO (FILTRO DI KALMAN) - ASSET: {asset_upper}")
        print("#" * 84)

        # 1. Caricamento segnali statistici
        stat_file = signals_dir / f"signals_statistical_{asset_lower}.csv"
        if not stat_file.exists():
            print(f"  [ATTENZIONE] File {stat_file} non trovato, generazione automatica...")
            price_col = f"{asset_upper}_close"
            sma_a = SMAEMAAgent(price_col=price_col)
            zs_a = ZScoreAgent(price_col=price_col)
            sig_s, conf_s = sma_a.predict_with_confidence(df_features)
            sig_z, conf_z = zs_a.predict_with_confidence(df_features)
            df_stat = pd.DataFrame({"signal": sig_s, "confidence": conf_s}, index=df_features.index)
        else:
            df_stat = pd.read_csv(stat_file, index_col=0, parse_dates=True)

        # 2. Segnali Agente Macro
        macro_sig_col = f"{asset_upper}_signal"
        m_sig = macro_multi[macro_sig_col] if macro_sig_col in macro_multi.columns else pd.Series(0, index=df_features.index)
        macro_score = macro_agent.calculate_macro_score(df_features)
        m_conf = (macro_score.abs() / 2.0).clip(0.0, 1.0).round(4)
        m_conf = pd.Series(np.where(m_sig != 0, np.maximum(0.50, m_conf), 0.0), index=df_features.index)
        df_macro = pd.DataFrame({"signal": m_sig, "confidence": m_conf}, index=df_features.index)

        signals_pool = {
            "Statistical_Agent": df_stat,
            "Macro_YieldCurve_Agent": df_macro,
        }

        # Rendimenti target
        ret_col = f"{asset_upper}_return_1w"
        if ret_col in df_features.columns:
            asset_ret = df_features[ret_col].fillna(0.0)
        else:
            asset_ret = df_features[f"{asset_upper}_close"].pct_change().fillna(0.0)

        # -------------------------------------------------------------
        # A. ESECUZIONE FILTRO DI KALMAN (SENZA FASE DI ADDESTRAMENTO ESPLICITA)
        # -------------------------------------------------------------
        # Soglia mirata in base alla volatilità dell'asset
        th_asset = 0.25 if asset_upper != "TLT" else 0.35
        kalman_agent = KalmanOrchestratorAgent(
            forgetting_factor=forgetting_factor,
            threshold_buy=th_asset,
            threshold_sell=th_asset,
            consensus_boost=0.15,
            conflict_penalty=0.25,
            name=f"KalmanOrchestrator_{asset_upper}",
        )

        df_kalman_signals, df_kalman_weights = kalman_agent.combine(signals_pool, returns_or_prices=asset_ret)

        # -------------------------------------------------------------
        # B. ESECUZIONE ORCHESTRATORE STATICO (BENCHMARK DI CONFRONTO)
        # -------------------------------------------------------------
        static_opt = OrchestratorOptimizer(
            objective_metric="sharpe_mdd",
            max_train_year=max_train_year,
            weight_step=0.05,
            threshold_range=(0.15, 0.50, 0.05),
        )
        static_agent, _ = static_opt.optimize(signals_pool, asset_ret, asset_name=asset_upper)
        df_static_signals = static_agent.combine(signals_pool)

        # -------------------------------------------------------------
        # C. CALCOLO E CONFRONTO DELLE PERFORMANCE IN-SAMPLE (2004-2021)
        # -------------------------------------------------------------
        is_mask = df_features.index.year <= max_train_year
        is_ret = asset_ret.loc[is_mask]

        # 1. Statistical Agent
        pos_stat = df_stat.loc[is_mask, "signal"].shift(1).fillna(0.0)
        m_stat = calculate_performance_metrics(pos_stat * is_ret)

        # 2. Macro Agent
        pos_macro = df_macro.loc[is_mask, "signal"].shift(1).fillna(0.0)
        m_macro = calculate_performance_metrics(pos_macro * is_ret)

        # 3. Static Orchestrator
        pos_static = df_static_signals.loc[is_mask, "signal"].shift(1).fillna(0.0)
        m_static = calculate_performance_metrics(pos_static * is_ret)

        # 4. Kalman Filter Orchestrator
        pos_kalman = df_kalman_signals.loc[is_mask, "signal"].shift(1).fillna(0.0)
        m_kalman = calculate_performance_metrics(pos_kalman * is_ret)

        print("\n" + "-" * 90)
        print(f"  TABELLA COMPARATIVA PERFORMANCE IN-SAMPLE (2004-{max_train_year}) - ASSET: {asset_upper}")
        print("-" * 90)
        print(f"{'Metodo Orchestratore / Agente':<32} | {'Cum Ret':<9} | {'Sharpe':<7} | {'Max DD':<7} | {'Prof Fact':<9} | {'Note':<15}")
        print("-" * 90)
        print(
            f"{'1. Statistical Agent':<32} | {m_stat['cumulative_return']*100:+7.1f}% | "
            f"{m_stat['sharpe_ratio']:>7.2f} | {m_stat['max_drawdown']*100:>5.1f}% | "
            f"{m_stat['profit_factor']:>9.2f} | {'Baseline stat.'}"
        )
        print(
            f"{'2. Macro Yield Curve Agent':<32} | {m_macro['cumulative_return']*100:+7.1f}% | "
            f"{m_macro['sharpe_ratio']:>7.2f} | {m_macro['max_drawdown']*100:>5.1f}% | "
            f"{m_macro['profit_factor']:>9.2f} | {'Baseline macro'}"
        )
        print(
            f"{'3. Static Grid Orchestrator':<32} | {m_static['cumulative_return']*100:+7.1f}% | "
            f"{m_static['sharpe_ratio']:>7.2f} | {m_static['max_drawdown']*100:>5.1f}% | "
            f"{m_static['profit_factor']:>9.2f} | {'Train statico IS'}"
        )
        print("-" * 90)
        print(
            f"{'4. KALMAN FILTER ORCHESTRATOR':<32} | {m_kalman['cumulative_return']*100:+7.1f}% | "
            f"{m_kalman['sharpe_ratio']:>7.2f} | {m_kalman['max_drawdown']*100:>5.1f}% | "
            f"{m_kalman['profit_factor']:>9.2f} | {'ADATTIVO REALE'}"
        )
        print("-" * 90)

        # Salvataggio pesi medi finali del Kalman Filter
        final_w = df_kalman_weights.iloc[-1].to_dict()
        print("Pesi medi assegnati dal Filtro di Kalman sul campione:")
        mean_weights = df_kalman_weights.loc[is_mask].mean()
        for col_name, w_val in mean_weights.items():
            print(f"  - {col_name:<24}: {w_val*100:5.1f}% (ultimo: {final_w[col_name]*100:5.1f}%)")

        # -------------------------------------------------------------
        # D. SALVATAGGIO OUTPUT
        # -------------------------------------------------------------
        # 1. Output finale con esattamente due colonne ['signal', 'confidence']
        assert list(df_kalman_signals.columns) == ["signal", "confidence"]
        out_signals_file = signals_dir / f"signals_orchestrator_{asset_lower}.csv"
        df_kalman_signals.to_csv(out_signals_file)
        print(f"\nSegnale finale Filtro di Kalman salvato in: {out_signals_file}")

        # 2. Salvataggio serie storica pesi dinamici del Kalman
        out_weights_file = signals_dir / f"weights_kalman_{asset_lower}.csv"
        df_kalman_weights.round(4).to_csv(out_weights_file)
        print(f"Traiettoria pesi Filtro di Kalman salvata in: {out_weights_file}")

        # 3. Salvataggio parametri e configurazione JSON
        kalman_config_path = config_dir / f"orchestrator_kalman_params_{asset_lower}.json"
        kalman_payload = {
            "model_type": "KalmanFilterOrchestrator",
            "forgetting_factor": forgetting_factor,
            "threshold": th_asset,
            "weight_mode": kalman_agent.weight_mode,
            "in_sample_metrics": {
                "cumulative_return": round(m_kalman["cumulative_return"], 4),
                "sharpe_ratio": round(m_kalman["sharpe_ratio"], 4),
                "max_drawdown": round(m_kalman["max_drawdown"], 4),
                "profit_factor": round(m_kalman["profit_factor"], 4),
            },
            "mean_weights_is": {k: round(float(v), 4) for k, v in mean_weights.items()},
        }
        with open(kalman_config_path, "w", encoding="utf-8") as f:
            json.dump(kalman_payload, f, indent=4)

        # 4. Salvataggio tabella comparativa CSV per analisi approfondita
        comp_df = pd.DataFrame([
            {"Model": "Statistical_Agent", **m_stat},
            {"Model": "Macro_YieldCurve_Agent", **m_macro},
            {"Model": "Static_Orchestrator", **m_static},
            {"Model": "Kalman_Filter_Orchestrator", **m_kalman},
        ])
        comp_csv = results_dir / f"comparison_orchestrator_kalman_{asset_lower}.csv"
        comp_df.to_csv(comp_csv, index=False)

        final_outputs[asset_upper] = df_kalman_signals

    print("\n" + "=" * 84)
    print("### PIPELINE FILTRO DI KALMAN COMPLETATA CON SUCCESSO PER TUTTI GLI ASSET ###")
    print("=" * 84)

    return final_outputs


if __name__ == "__main__":
    run_orchestrator_pipeline()
