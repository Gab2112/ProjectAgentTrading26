"""
orchestrator_agent.py

Agente Orchestratore ed Ensemble Quantitativo Ottimizzato per la Sintesi di Segnali Multiatore.

Ruolo nell'architettura ProjectAgentTrading26:
- Riceve in input una serie di segnali operativi prodotti da diversi agenti specializzati
  (es. Agente Statistico SMA/EMA & Z-Score, Agente Macro Yield Curve, Agente Sentiment FinBERT, ecc.).
- Ciascun agente fornisce un DataFrame con esattamente due colonne: 'signal' ({-1, 0, 1}) e 'confidence' ([0.0, 1.0]).
- Ottimizza la combinazione pesata w = (w_1, ..., w_K), le soglie decisionali di ingresso/uscita (threshold)
  e le regole di interazione tra agenti (sinergia di consenso e penalizzazione da conflitto)
  RIGOROSAMENTE sul periodo In-Sample (2004-2021, rispettando l'OOS lock del 2022+).
- Massimizza la metrica di stabilità 'sharpe_mdd' (Sharpe Ratio penalizzato per il Maximum Drawdown)
  per garantire curve di equity costanti e limitare il rischio di coda.
- Produce in output un unico DataFrame finale consolidato con esattamente due colonne:
    1. 'signal': segnale discreto {-1, 0, 1}
    2. 'confidence': confidenza continua [0.0, 1.0]
  destinato alle decisioni finali di asset allocation, position sizing e backtesting.
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

    # Verifica valori discreti ammessi {-1, 0, 1}
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
# 1. AGENTE ORCHESTRATORE
# =====================================================================
class OrchestratorAgent(BaseAgent):
    """
    Agente Orchestratore per la combinazione e sintesi ottima di segnali multi-agente.

    Funzionalità:
    - Pesa le indicazioni direzionali di ciascun agente in base all'affidabilità ottimizzata
      e alla rispettiva confidenza istantanea: x_k(t) = signal_k(t) * confidence_k(t).
    - Punteggio continuo aggregato: S_raw(t) = sum_k w_k * x_k(t).
    - Gestione Sinergia di Consenso: se 2 o più agenti attivi concordano sullo stesso verso (tutti long o tutti short),
      applica un boost di convinzione e confidenza (+consensus_boost, default 15%).
    - Gestione Conflitto: se ci sono agenti attivi con segnali opposti contemporaneamente (+1 vs -1),
      applica una penalizzazione netta (-conflict_penalty, default 25%).
    - Discretizzazione: scatta BUY (+1) se S_adj >= threshold_buy, SELL (-1) se S_adj <= -threshold_sell,
      altrimenti FLAT/HOLD (0).
    - Output garantito: DataFrame con esattamente 2 colonne: 'signal' ({-1, 0, 1}) e 'confidence' ([0.0, 1.0]).
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        threshold_buy: float = 0.30,
        threshold_sell: float = 0.30,
        consensus_boost: float = 0.15,
        conflict_penalty: float = 0.25,
        min_active_agents: int = 1,
        name: str = "OrchestratorAgent",
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

    def _apply_params(self, params: Dict[str, Any]) -> None:
        """Aggiorna i parametri dell'orchestratore dopo il caricamento da JSON."""
        if "weights" in params:
            self.weights = {str(k): float(v) for k, v in params["weights"].items()}
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
        if "min_active_agents" in params:
            self.min_active_agents = int(params["min_active_agents"])

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
        """
        Combina la serie di segnali in input e restituisce un DataFrame con esattamente
        due colonne: 'signal' e 'confidence'.
        """
        if isinstance(signals_input, list):
            signals_dict = {f"agent_{i+1}": df for i, df in enumerate(signals_input)}
        elif isinstance(signals_input, dict):
            signals_dict = signals_input
        else:
            raise TypeError(
                f"signals_input deve essere Dict o List di DataFrame, ricevuto: {type(signals_input)}"
            )

        if len(signals_dict) == 0:
            raise ValueError("Nessun segnale fornito all'Orchestratore.")

        aligned_signals, common_idx = align_signals(signals_dict, fill_method="zeros")
        agent_names = list(aligned_signals.keys())
        k_agents = len(agent_names)

        # Risoluzione pesi: se non specificati, usa equiponderazione (1 / K)
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

        # Convinzione direzionale pesata: x_k = signal_k * confidence_k in [-1.0, 1.0]
        conviction_mat = signals_mat * conf_mat

        # Punteggio base continuo: S_raw(t) = sum_k (w_k * x_k(t))
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
            active_conf = conf_mat[t, active_mask]
            active_weights = weights_arr[active_mask]

            score_t = raw_scores[t]
            n_long = np.sum(active_signs == 1)
            n_short = np.sum(active_signs == -1)

            # Caso A: Conflitto diretto (agenti con segnali opposti attivi contemporaneamente)
            if n_long > 0 and n_short > 0:
                adjusted_score = score_t * (1.0 - self.conflict_penalty)
            # Caso B: Accordo unanime tra agenti attivi (tutti long o tutti short)
            elif (n_long > 0 and n_short == 0) or (n_short > 0 and n_long == 0):
                if n_active >= 2:
                    adjusted_score = np.sign(score_t) * min(1.0, abs(score_t) * (1.0 + self.consensus_boost))
                else:
                    adjusted_score = score_t
            else:
                adjusted_score = 0.0

            # Discretizzazione del segnale finale
            if adjusted_score >= self.threshold_buy:
                sig_out = 1
            elif adjusted_score <= -self.threshold_sell:
                sig_out = -1
            else:
                sig_out = 0

            # Calcolo confidenza finale
            if sig_out != 0:
                # Media ponderata della confidenza degli agenti concordi
                agreeing_mask = signals_mat[t] == sig_out
                agreeing_w = weights_arr[agreeing_mask]
                agreeing_c = conf_mat[t, agreeing_mask]

                sum_w = np.sum(agreeing_w)
                mean_agree_conf = float(np.sum(agreeing_w * agreeing_c) / sum_w) if sum_w > 0 else 0.50

                denom = max(1.0 - min(self.threshold_buy, self.threshold_sell), 0.10)
                score_strength = min(1.0, abs(adjusted_score) / denom)

                c_combined = 0.50 + 0.50 * (0.60 * score_strength + 0.40 * mean_agree_conf)
                conf_out = min(1.0, max(0.50, round(float(c_combined), 4)))
            else:
                conf_out = 0.0

            final_signals[t] = sig_out
            final_conf[t] = conf_out

        df_out = pd.DataFrame(
            {
                "signal": final_signals,
                "confidence": final_conf,
            },
            index=common_idx,
        )

        return df_out

    def predict(self, df_or_signals: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> pd.Series:
        """Restituisce la serie dei segnali operativi discreti {-1, 0, 1}."""
        if isinstance(df_or_signals, dict):
            df_combined = self.combine(df_or_signals)
        else:
            raise TypeError("predict richiede un dizionario di segnali {nome: df_segnali}")

        return self._format_signals(df_combined["signal"])

    def predict_with_confidence(
        self,
        signals_input: Union[Dict[str, pd.DataFrame], List[pd.DataFrame]],
    ) -> Tuple[pd.Series, pd.Series]:
        """Restituisce la tupla (signal, confidence) allineata."""
        df_combined = self.combine(signals_input)
        return df_combined["signal"], df_combined["confidence"]


# =====================================================================
# 2. OTTIMIZZATORE QUANTITATIVO DELL'ORCHESTRATORE
# =====================================================================
class OrchestratorOptimizer:
    """
    Ottimizzatore quantitativo per l'Agente Orchestratore.

    Principi metodologici:
    - Ottimizza i pesi degli agenti w = (w_1, ..., w_K) e le soglie decisionali (threshold)
      RIGOROSAMENTE sul periodo In-Sample (fino al max_train_year, default 2021).
    - Rispetta il vincolo del progetto OOS lock (dati 2022+ mai toccati in fase di ottimizzazione).
    - Supporta funzioni obiettivo stabili e penalizzanti il drawdown ('sharpe_mdd', 'sharpe', 'calmar', ecc.).
    - Esegue una grid search sul simplesso dei pesi combinata con campionamento Dirichlet per esplorare
      accuratamente tutte le possibili allocazioni.
    """

    def __init__(
        self,
        objective_metric: str = "sharpe_mdd",
        max_train_year: int = 2021,
        annualization_factor: int = 52,
        weight_step: float = 0.05,
        threshold_range: Tuple[float, float, float] = (0.15, 0.55, 0.05),
        consensus_boost: float = 0.15,
        conflict_penalty: float = 0.25,
        min_active_agents: int = 1,
        random_seed: int = 42,
    ):
        self.objective_metric = objective_metric
        self.max_train_year = int(max_train_year)
        self.annualization_factor = int(annualization_factor)
        self.weight_step = float(weight_step)
        self.threshold_range = threshold_range
        self.consensus_boost = float(consensus_boost)
        self.conflict_penalty = float(conflict_penalty)
        self.min_active_agents = int(min_active_agents)
        self.random_seed = int(random_seed)

        self.tested_combinations: List[Dict[str, Any]] = []
        self.best_params: Dict[str, Any] = {}
        self.best_score: float = -999.0
        self.in_sample_metrics: Dict[str, float] = {}

    def _generate_weight_candidates(self, k: int, step: float = 0.05) -> np.ndarray:
        """
        Genera combinazioni discrete di pesi sul simplesso sum(w) = 1.0 con w_i >= 0.
        """
        if k == 1:
            return np.array([[1.0]])

        n_steps = int(round(1.0 / step))
        candidates = []
        for combo in itertools.product(range(n_steps + 1), repeat=k):
            if sum(combo) == n_steps:
                candidates.append([val * step for val in combo])

        candidates_arr = np.array(candidates, dtype=float)

        # Aggiunge pesi equiponderati e monopolari
        equal_w = np.full((1, k), 1.0 / k)
        eye_w = np.eye(k)
        candidates_arr = np.vstack([candidates_arr, equal_w, eye_w])
        candidates_arr = np.unique(np.round(candidates_arr, 4), axis=0)

        # Se k > 2 o campioni ridotti, aggiunge campionamento Dirichlet
        if k > 2 or len(candidates_arr) < 50:
            rng = np.random.default_rng(self.random_seed)
            dirichlet_samples = rng.dirichlet(np.ones(k), size=200)
            candidates_arr = np.vstack([candidates_arr, dirichlet_samples])
            candidates_arr = np.unique(np.round(candidates_arr, 4), axis=0)

        return candidates_arr

    def optimize(
        self,
        signals_dict: Dict[str, pd.DataFrame],
        returns_or_prices: Union[pd.Series, pd.DataFrame],
        asset_name: str = "ASSET",
    ) -> Tuple[OrchestratorAgent, pd.DataFrame]:
        """
        Esegue l'ottimizzazione completa dei pesi e delle soglie sui dati In-Sample.
        """
        if not signals_dict:
            raise ValueError("Dizionario signals_dict vuoto.")

        aligned_signals, common_idx = align_signals(signals_dict, fill_method="zeros")
        agent_names = list(aligned_signals.keys())
        k_agents = len(agent_names)

        # Calcolo rendimenti periodici dell'asset
        if isinstance(returns_or_prices, pd.DataFrame):
            col = [c for c in returns_or_prices.columns if "return" in c.lower() or "close" in c.lower()][0]
            series = returns_or_prices[col]
        else:
            series = returns_or_prices

        if series.min() >= 0.0 and series.mean() > 5.0:
            asset_returns = series.pct_change().fillna(0.0)
        else:
            asset_returns = series.fillna(0.0)

        asset_returns = asset_returns.reindex(common_idx).fillna(0.0)

        # Maschera In-Sample rigorosa (fino al max_train_year, default 2021)
        if isinstance(common_idx, pd.DatetimeIndex):
            is_mask = common_idx.year <= self.max_train_year
        else:
            is_mask = np.ones(len(common_idx), dtype=bool)

        if not np.any(is_mask):
            raise ValueError(f"Nessun dato temporale In-Sample trovato entro l'anno {self.max_train_year}.")

        is_idx = common_idx[is_mask]
        is_signals = {name: aligned_signals[name].loc[is_idx] for name in agent_names}
        is_asset_ret = asset_returns.loc[is_idx]

        print(f"\n" + "=" * 84)
        print(f"  OTTIMIZZAZIONE ORCHESTRATORE - ASSET: {asset_name.upper()} | OOS LOCK <= {self.max_train_year}")
        print(f"  Agenti Coinvolti: {k_agents} ({', '.join(agent_names)})")
        print(f"  Campione In-Sample: {is_idx[0].strftime('%Y-%m-%d')} -> {is_idx[-1].strftime('%Y-%m-%d')} ({len(is_idx)} settimane)")
        print(f"  Metrica Obiettivo: '{self.objective_metric}'")
        print("=" * 84)

        # Calcolo performance baseline dei singoli agenti In-Sample
        agent_benchmarks = {}
        for name in agent_names:
            sig = is_signals[name]["signal"]
            pos = sig.shift(1).fillna(0.0)
            strat_ret = pos * is_asset_ret
            m = calculate_performance_metrics(strat_ret, annualization_factor=self.annualization_factor)
            obj = compute_objective_score(m, metric_type=self.objective_metric)
            agent_benchmarks[name] = {**m, "objective_score": obj}

        weight_candidates = self._generate_weight_candidates(k_agents, step=self.weight_step)
        th_start, th_end, th_step = self.threshold_range
        threshold_candidates = np.round(np.arange(th_start, th_end + 1e-6, th_step), 3)

        total_combos = len(weight_candidates) * len(threshold_candidates)
        print(f"  Combinazioni generate: {total_combos} ({len(weight_candidates)} vettori pesi x {len(threshold_candidates)} soglie)")

        self.tested_combinations = []
        best_score = -999.0
        best_combo = None

        for w_vec in weight_candidates:
            w_dict = {name: float(w_vec[i]) for i, name in enumerate(agent_names)}
            for th in threshold_candidates:
                agent_temp = OrchestratorAgent(
                    weights=w_dict,
                    threshold_buy=float(th),
                    threshold_sell=float(th),
                    consensus_boost=self.consensus_boost,
                    conflict_penalty=self.conflict_penalty,
                    min_active_agents=self.min_active_agents,
                )
                df_res = agent_temp.combine(is_signals)
                sig = df_res["signal"]

                # Timing: shift(1) per applicazione al periodo successivo (evita look-ahead bias)
                pos = sig.shift(1).fillna(0.0)
                strat_ret = pos * is_asset_ret

                metrics = calculate_performance_metrics(strat_ret, annualization_factor=self.annualization_factor)
                score = compute_objective_score(metrics, metric_type=self.objective_metric)
                num_trades = int((pos.diff().fillna(0) != 0).sum())

                record = {
                    "weights": w_dict,
                    "threshold": float(th),
                    "objective_score": round(score, 4),
                    "sharpe_ratio": round(metrics["sharpe_ratio"], 4),
                    "max_drawdown": round(metrics["max_drawdown"], 4),
                    "cumulative_return": round(metrics["cumulative_return"], 4),
                    "annualized_return": round(metrics["annualized_return"], 4),
                    "profit_factor": round(metrics["profit_factor"], 4),
                    "calmar_ratio": round(metrics["calmar_ratio"], 4),
                    "trades": num_trades,
                }
                self.tested_combinations.append(record)

                if score > best_score:
                    best_score = score
                    best_combo = record

        self.best_score = best_score
        self.best_params = best_combo

        best_agent = OrchestratorAgent(
            weights=best_combo["weights"],
            threshold_buy=best_combo["threshold"],
            threshold_sell=best_combo["threshold"],
            consensus_boost=self.consensus_boost,
            conflict_penalty=self.conflict_penalty,
            min_active_agents=self.min_active_agents,
            name=f"Orchestrator_{asset_name.upper()}",
        )
        self.in_sample_metrics = {
            "sharpe_ratio": best_combo["sharpe_ratio"],
            "max_drawdown": best_combo["max_drawdown"],
            "cumulative_return": best_combo["cumulative_return"],
            "annualized_return": best_combo["annualized_return"],
            "profit_factor": best_combo["profit_factor"],
            "calmar_ratio": best_combo["calmar_ratio"],
            "objective_score": best_combo["objective_score"],
            "trades": best_combo["trades"],
        }

        df_summary = pd.DataFrame(self.tested_combinations)
        df_summary = df_summary.sort_values(by="objective_score", ascending=False).reset_index(drop=True)

        print("\n" + "-" * 88)
        print(f"  CONFRONTO PERFORMANCE IN-SAMPLE (2004-{self.max_train_year}) - ASSET: {asset_name.upper()}")
        print("-" * 88)
        print(f"{'Modello / Agente':<28} | {'Cum Ret':<9} | {'Sharpe':<7} | {'Max DD':<7} | {'Prof Fact':<9} | {'Score Obj':<9}")
        print("-" * 88)

        for name, bm in agent_benchmarks.items():
            print(
                f"{name:<28} | {bm['cumulative_return']*100:+7.1f}% | "
                f"{bm['sharpe_ratio']:>7.2f} | {bm['max_drawdown']*100:>5.1f}% | "
                f"{bm['profit_factor']:>9.2f} | {bm['objective_score']:>9.2f}"
            )

        print("-" * 88)
        print(
            f"{'ORCHESTRATORE (OTTIMIZZATO)':<28} | {best_combo['cumulative_return']*100:+7.1f}% | "
            f"{best_combo['sharpe_ratio']:>7.2f} | {best_combo['max_drawdown']*100:>5.1f}% | "
            f"{best_combo['profit_factor']:>9.2f} | {best_combo['objective_score']:>9.2f}  <-- OTTIMO"
        )
        print("-" * 88)

        print("\nPesi Ottimali Assegnati dall'Orchestratore:")
        for name, w in best_combo["weights"].items():
            print(f"  - {name:<24}: {w*100:5.1f}%")
        print(f"  - Soglia Decisionale (Threshold) : {best_combo['threshold']:.2f}")

        return best_agent, df_summary

    def save_final_config(self, filepath: Union[str, Path]) -> None:
        """Salva i parametri ottimi in formato JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "best_params": self.best_params,
            "in_sample_metrics": self.in_sample_metrics,
            "objective_metric": self.objective_metric,
            "max_train_year": self.max_train_year,
            "annualization_factor": self.annualization_factor,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
        print(f"Configurazione Orchestratore salvata in: {path}")


# =====================================================================
# 3. ORCHESTRATORE WALK-FORWARD (Validazione di Robustezza)
# =====================================================================
class WalkForwardOrchestrator:
    """
    Esegue la validazione Walk-Forward (es. 5 anni di training In-Sample -> 1 anno di test Out-Of-Sample)
    per verificare la stabilità temporale dei pesi dell'Orchestratore attraverso i cicli di mercato.
    """

    def __init__(
        self,
        optimizer: OrchestratorOptimizer,
        train_window_years: int = 5,
        test_window_years: int = 1,
    ):
        self.optimizer = optimizer
        self.train_window_years = train_window_years
        self.test_window_years = test_window_years
        self.fold_results: List[Dict[str, Any]] = []

    def run(
        self,
        signals_dict: Dict[str, pd.DataFrame],
        returns_or_prices: Union[pd.Series, pd.DataFrame],
        max_year: int = 2021,
    ) -> pd.DataFrame:
        aligned_signals, common_idx = align_signals(signals_dict, fill_method="zeros")
        years = sorted(list(set(common_idx.year)))
        valid_years = [y for y in years if y <= max_year]

        if len(valid_years) < self.train_window_years + self.test_window_years:
            raise ValueError("Anni disponibili insufficienti per il Walk-Forward configurato.")

        self.fold_results = []
        start_idx = 0
        while start_idx + self.train_window_years < len(valid_years):
            train_years = valid_years[start_idx : start_idx + self.train_window_years]
            test_year = valid_years[start_idx + self.train_window_years]

            print(f"\n[Walk-Forward Orchestrator] Fold Train: {train_years[0]}-{train_years[-1]} -> Test OOS: {test_year}")

            train_mask = common_idx.year.isin(train_years)
            test_mask = common_idx.year == test_year

            train_signals = {name: df.loc[train_mask] for name, df in aligned_signals.items()}
            test_signals = {name: df.loc[test_mask] for name, df in aligned_signals.items()}

            if isinstance(returns_or_prices, pd.DataFrame):
                col = [c for c in returns_or_prices.columns if "return" in c.lower() or "close" in c.lower()][0]
                ret_series = returns_or_prices[col]
            else:
                ret_series = returns_or_prices

            ret_series = ret_series.reindex(common_idx).fillna(0.0)
            train_ret = ret_series.loc[train_mask]
            test_ret = ret_series.loc[test_mask]

            fold_opt = OrchestratorOptimizer(
                objective_metric=self.optimizer.objective_metric,
                max_train_year=train_years[-1],
                annualization_factor=self.optimizer.annualization_factor,
                weight_step=self.optimizer.weight_step,
                threshold_range=self.optimizer.threshold_range,
                consensus_boost=self.optimizer.consensus_boost,
                conflict_penalty=self.optimizer.conflict_penalty,
                min_active_agents=self.optimizer.min_active_agents,
            )
            opt_agent, _ = fold_opt.optimize(train_signals, train_ret, asset_name=f"WF_{test_year}")

            df_test = opt_agent.combine(test_signals)
            pos_test = df_test["signal"].shift(1).fillna(0.0)
            strat_test = pos_test * test_ret
            m_oos = calculate_performance_metrics(strat_test)

            self.fold_results.append({
                "test_year": test_year,
                "train_years": f"{train_years[0]}-{train_years[-1]}",
                "best_weights": opt_agent.weights,
                "threshold": opt_agent.threshold_buy,
                "oos_sharpe": m_oos["sharpe_ratio"],
                "oos_cum_ret": m_oos["cumulative_return"],
                "oos_max_dd": m_oos["max_drawdown"],
            })

            start_idx += self.test_window_years

        return pd.DataFrame(self.fold_results)


# =====================================================================
# 4. ENTRY POINT: Pipeline Completa dell'Orchestratore
# =====================================================================
def run_orchestrator_pipeline(
    root_dir: Optional[Path] = None,
    assets: Optional[List[str]] = None,
    max_train_year: int = 2021,
    objective_metric: str = "sharpe_mdd",
) -> Dict[str, pd.DataFrame]:
    """
    Esegue l'intero workflow quantitativo dell'agente orchestratore:
    1. Carica i segnali generati dai singoli agenti specializzati per ciascun asset ('SHY', 'IEF', 'TLT').
    2. Ottimizza la combinazione di pesi e soglie sul campione In-Sample (2004-2021).
    3. Salva la configurazione JSON ottimale in src/agents/config/orchestrator_params_{asset}.json.
    4. Genera per ciascun asset il DataFrame finale con esattamente due colonne: 'signal' e 'confidence'.
    5. Esporta i CSV finali in data/signals/signals_orchestrator_{asset}.csv.
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

    if not features_file.exists():
        raise FileNotFoundError(f"File dati non trovato in: {features_file}")

    df_features = pd.read_csv(features_file, index_col=0, parse_dates=True)

    # Inizializza l'agente Macro Yield Curve
    macro_agent = MacroYieldCurveAgent()
    macro_multi = macro_agent.predict_multi_asset(df_features)

    final_outputs = {}

    for asset in assets:
        asset_lower = asset.lower()
        asset_upper = asset.upper()

        print("\n" + "#" * 84)
        print(f"### ESECUZIONE ORCHESTRATORE OTTIMIZZATO PER ASSET: {asset_upper}")
        print("#" * 84)

        # 1. Caricamento segnali statistici
        stat_file = signals_dir / f"signals_statistical_{asset_lower}.csv"
        if not stat_file.exists():
            print(f"  [ATTENZIONE] File segnali statistici non trovato: {stat_file}. Generazione al volo...")
            price_col = f"{asset_upper}_close"
            sma_a = SMAEMAAgent(price_col=price_col)
            zs_a = ZScoreAgent(price_col=price_col)
            sig_s, conf_s = sma_a.predict_with_confidence(df_features)
            sig_z, conf_z = zs_a.predict_with_confidence(df_features)
            df_stat = pd.DataFrame({"signal": sig_s, "confidence": conf_s}, index=df_features.index)
        else:
            df_stat = pd.read_csv(stat_file, index_col=0, parse_dates=True)

        # 2. Segnali Agente Macro / Regimes
        macro_sig_col = f"{asset_upper}_signal"
        m_sig = macro_multi[macro_sig_col] if macro_sig_col in macro_multi.columns else pd.Series(0, index=df_features.index)
        macro_score = macro_agent.calculate_macro_score(df_features)
        m_conf = (macro_score.abs() / 2.0).clip(0.0, 1.0).round(4)
        m_conf = pd.Series(np.where(m_sig != 0, np.maximum(0.50, m_conf), 0.0), index=df_features.index)
        df_macro = pd.DataFrame({"signal": m_sig, "confidence": m_conf}, index=df_features.index)

        # 3. Assemblaggio del pool di agenti
        signals_pool = {
            "Statistical_Agent": df_stat,
            "Macro_YieldCurve_Agent": df_macro,
        }

        # Serie dei rendimenti dell'asset target
        ret_col = f"{asset_upper}_return_1w"
        if ret_col in df_features.columns:
            asset_ret = df_features[ret_col]
        else:
            asset_ret = df_features[f"{asset_upper}_close"].pct_change().fillna(0.0)

        # 4. Ottimizzazione Orchestratore
        optimizer = OrchestratorOptimizer(
            objective_metric=objective_metric,
            max_train_year=max_train_year,
            weight_step=0.05,
            threshold_range=(0.15, 0.50, 0.05),
        )
        opt_orchestrator, df_summary = optimizer.optimize(
            signals_dict=signals_pool,
            returns_or_prices=asset_ret,
            asset_name=asset_upper,
        )

        # 5. Salvataggio della configurazione ottimale JSON
        config_path = config_dir / f"orchestrator_params_{asset_lower}.json"
        optimizer.save_final_config(config_path)

        # Salvataggio log combinazioni
        summary_csv = results_dir / f"summary_combinations_orchestrator_{asset_lower}.csv"
        df_summary.to_csv(summary_csv, index=False)
        print(f"Riepilogo combinazioni salvato in: {summary_csv}")

        # 6. Generazione del DataFrame FINALE con esattamente 2 colonne: 'signal' e 'confidence'
        df_final_signals = opt_orchestrator.combine(signals_pool)

        assert list(df_final_signals.columns) == ["signal", "confidence"], (
            f"Errore formato: atteso ['signal', 'confidence'], ottenuto {list(df_final_signals.columns)}"
        )

        output_csv = signals_dir / f"signals_orchestrator_{asset_lower}.csv"
        df_final_signals.to_csv(output_csv)
        print(f"Segnali finali Orchestratore salvati in: {output_csv}")
        print(f"Distribuzione segnali: {dict(df_final_signals['signal'].value_counts())}")
        print(f"Confidenza media sui segnali attivi: {df_final_signals.loc[df_final_signals['signal'] != 0, 'confidence'].mean():.4f}")

        final_outputs[asset_upper] = df_final_signals

    print("\n" + "=" * 84)
    print("### PIPELINE AGENTE ORCHESTRATORE COMPLETATA CON SUCCESSO PER TUTTI GLI ASSET ###")
    print("=" * 84)
    return final_outputs


if __name__ == "__main__":
    run_orchestrator_pipeline()
