"""Modular, NumPy-vectorised framework for a weekly trading meta-decision model
optimised with a real-coded genetic algorithm (GA).

Architectural components:
1. WeeklySignalAggregator: continuous aggregation, non-linear confidence
   calibration and dynamic position sizing (2D and 3D).
2. PortfolioFitnessEvaluator: net performance evaluation (52 periods per year),
   turnover costs and anti-overfitting constraints.
3. GeneticMetaOptimizer: real-coded GA with BLX-alpha crossover, adaptive
   Gaussian mutation, elitism and a cap on the FinBERT weight.
4. WalkForwardEngine: rolling out-of-sample validation without data leakage,
   with tracking of parameter drift.
5. OrchestratorAgent: operational inference agent compatible with BaseAgent.
6. OrchestratorOptimizer and WalkForwardOrchestrator: unified interfaces for
   training and validation on the production assets.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

# Path resolution for direct or module execution.
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.time_split import get_time_windows

from src.agents.base_agent import BaseAgent
from src.agents.EMA_EMW_agent import SMAEMAAgent
from src.agents.macro_yield_curve_agent import MacroYieldCurveAgent
from src.agents.random_forest_agent import RandomForestAgent
from src.agents.SVM_agent import SVMAgent
from src.agents.ZScore_agent import ZScoreAgent
from src.evaluator import calculate_performance_metrics, compute_objective_score
from src.utils import ASSETS, get_project_root


# =====================================================================
# SECTION 1: SIGNAL VALIDATION AND ALIGNMENT
# =====================================================================
def validate_signal_dataframe(df: pd.DataFrame, source_name: str = "SignalSource") -> pd.DataFrame:
    """Check and normalise the 'signal' ({-1, 0, 1}) and 'confidence' ([0.0, 1.0]) columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"[{source_name}] Atteso pd.DataFrame, ricevuto: {type(df)}")

    cols_map = {str(col).lower(): col for col in df.columns}
    if "signal" not in cols_map or "confidence" not in cols_map:
        raise KeyError(f"[{source_name}] Colonne richieste: 'signal', 'confidence'. Presenti: {list(df.columns)}")

    clean_df = pd.DataFrame(index=df.index)
    clean_df["signal"] = np.sign(df[cols_map["signal"]].fillna(0).astype(float)).astype(int)
    clean_df["confidence"] = df[cols_map["confidence"]].fillna(0.0).astype(float).clip(0.0, 1.0)

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
    """Align a dictionary of signals on a common, sorted time index."""
    if not signals_dict:
        raise ValueError("Il dizionario dei segnali in input è vuoto.")

    validated = {k: validate_signal_dataframe(v, source_name=k) for k, v in signals_dict.items()}
    first_idx = validated[list(validated.keys())[0]].index

    if fill_method == "inner":
        common_idx = first_idx
        for name in list(validated.keys())[1:]:
            common_idx = common_idx.intersection(validated[name].index)
    else:  # 'zeros'
        common_idx = first_idx
        for name in list(validated.keys())[1:]:
            common_idx = common_idx.union(validated[name].index)

    common_idx = common_idx.sort_values()
    aligned = {}
    for name, df in validated.items():
        reindexed = df.reindex(common_idx)
        reindexed["signal"] = reindexed["signal"].fillna(0).astype(int)
        reindexed["confidence"] = reindexed["confidence"].fillna(0.0).astype(float).clip(0.0, 1.0)
        aligned[name] = reindexed

    return aligned, common_idx


# =====================================================================
# SECTION 2: AGGREGATION, CALIBRATION AND POSITION SIZING (PHENOTYPE)
# =====================================================================
class WeeklySignalAggregator:
    """NumPy-vectorised phenotype engine for weekly series.

    Implemented formulas:
    1. Non-linear calibration: C_cal = clip(C, 1e-6, 1.0) ** alpha
    2. Continuous score Z_t, damped by flat agents (s_i = 0):
       Z_t = sum(w_i * s_i * c_cal_i) / (sum(w_i * c_cal_i) + eps)
    3. Asymmetric discretisation: S_out = +1 if Z > theta_long,
       -1 if Z < -theta_short, 0 otherwise.
    4. Modulated confidence C_out in [0, 1].
    5. Dynamic position sizing: P_t = S_out,t * C_out,t in [-1, 1].
    """

    def __init__(
            self,
            weights: np.ndarray,
            alphas: np.ndarray,
            theta_long: float = 0.25,
            theta_short: float = 0.25,
            epsilon: float = 1e-12,
            objective_metric: str = "sharpe_mdd",
    ):
        raw_w = np.asarray(weights, dtype=np.float64).flatten()
        self.alphas = np.asarray(alphas, dtype=np.float64).flatten()
        self.theta_long = float(theta_long)
        self.theta_short = float(theta_short)
        self.epsilon = float(epsilon)
        self.objective_metric = str(objective_metric)
        if self.objective_metric not in {
            "sharpe_mdd", "sharpe", "calmar", "return", "profit_oriented", "profit"
        }:
            raise ValueError(f"Obiettivo fitness non valido: {self.objective_metric}")

        w_sum = np.sum(raw_w)
        if w_sum > self.epsilon:
            self.weights = raw_w / w_sum
        else:
            self.weights = np.full(len(raw_w), 1.0 / max(len(raw_w), 1), dtype=np.float64)

    def aggregate_2d(
            self,
            signals: np.ndarray,
            confidences: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised inference over the full history (T, N). Returns (Z, S_out, C_out, P).
        """
        S = np.asarray(signals, dtype=np.float64)
        C = np.asarray(confidences, dtype=np.float64)
        if S.ndim == 1:
            S, C = S[:, np.newaxis], C[:, np.newaxis]

        T, N = S.shape
        w = self.weights.reshape(1, N)
        a = self.alphas.reshape(1, N)

        # 1. Non-linear calibration.
        C_cal = np.clip(C, 1e-6, 1.0) ** a

        # 2. Continuous score Z, damped by flat agents.
        num = np.sum(w * S * C_cal, axis=1)
        den = np.sum(w * C_cal, axis=1) + self.epsilon
        Z = np.clip(num / den, -1.0, 1.0)

        # 3. Directional discretisation with asymmetric thresholds.
        S_out = np.zeros(T, dtype=np.int32)
        S_out[Z > self.theta_long] = 1
        S_out[Z < -self.theta_short] = -1

        # 4. Continuous confidence C_out.
        C_out = np.zeros(T, dtype=np.float64)
        long_m = S_out == 1
        short_m = S_out == -1
        flat_m = S_out == 0

        C_out[long_m] = (Z[long_m] - self.theta_long) / (1.0 - self.theta_long + self.epsilon)
        abs_Z = np.abs(Z)
        C_out[short_m] = (abs_Z[short_m] - self.theta_short) / (1.0 - self.theta_short + self.epsilon)

        flat_pos = flat_m & (Z >= 0.0)
        flat_neg = flat_m & (Z < 0.0)
        C_out[flat_pos] = 1.0 - (Z[flat_pos] / (self.theta_long + self.epsilon))
        C_out[flat_neg] = 1.0 - (abs_Z[flat_neg] / (self.theta_short + self.epsilon))
        C_out = np.clip(C_out, 0.0, 1.0)

        # 5. Dynamic position sizing.
        P = S_out.astype(np.float64) * C_out
        return Z, S_out, C_out, P

    @staticmethod
    def aggregate_3d_population(
            signals: np.ndarray,
            confidences: np.ndarray,
            chromosomes: np.ndarray,
            epsilon: float = 1e-12,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """3D tensor broadcasting (M, T, N) over a population of M chromosomes.

        Returns (P, S_out, C_out), each with shape (M, T).
        """
        chroms = np.asarray(chromosomes, dtype=np.float64)
        M, _ = chroms.shape
        T, N = signals.shape

        # Normalise the weights.
        W_raw = chroms[:, :N]
        W_sum = np.sum(W_raw, axis=1, keepdims=True)
        W = np.where(W_sum > epsilon, W_raw / np.maximum(W_sum, epsilon), 1.0 / N)[:, np.newaxis, :]  # (M, 1, N)
        A = chroms[:, N: 2 * N][:, np.newaxis, :]  # (M, 1, N)
        TL = chroms[:, 2 * N: 2 * N + 1]  # (M, 1)
        TS = chroms[:, 2 * N + 1: 2 * N + 2]  # (M, 1)

        # Broadcast tensors.
        S_3d = signals[np.newaxis, :, :]  # (1, T, N)
        C_cal = np.clip(confidences[np.newaxis, :, :], 1e-6, 1.0) ** A  # (M, T, N)

        # Compute Z.
        num = np.sum(W * S_3d * C_cal, axis=2)
        den = np.sum(W * C_cal, axis=2) + epsilon
        Z = np.clip(num / den, -1.0, 1.0)  # (M, T)

        # Discrete signals.
        S_out = np.zeros((M, T), dtype=np.int32)
        S_out[Z > TL] = 1
        S_out[Z < -TS] = -1

        # Confidences.
        TL_mat = np.broadcast_to(TL, (M, T))
        TS_mat = np.broadcast_to(TS, (M, T))
        abs_Z = np.abs(Z)

        C_out = np.zeros((M, T), dtype=np.float64)
        long_m = S_out == 1
        short_m = S_out == -1
        flat_m = S_out == 0

        C_out[long_m] = (Z[long_m] - TL_mat[long_m]) / (1.0 - TL_mat[long_m] + epsilon)
        C_out[short_m] = (abs_Z[short_m] - TS_mat[short_m]) / (1.0 - TS_mat[short_m] + epsilon)

        flat_pos = flat_m & (Z >= 0.0)
        flat_neg = flat_m & (Z < 0.0)
        C_out[flat_pos] = 1.0 - (Z[flat_pos] / (TL_mat[flat_pos] + epsilon))
        C_out[flat_neg] = 1.0 - (abs_Z[flat_neg] / (TS_mat[flat_neg] + epsilon))
        C_out = np.clip(C_out, 0.0, 1.0)

        P = S_out.astype(np.float64) * C_out
        return P, S_out, C_out


# =====================================================================
# SECTION 3: PORTFOLIO DYNAMICS AND FITNESS EVALUATION
# =====================================================================
class PortfolioFitnessEvaluator:
    """Quantitative evaluation of weekly performance.

    Includes turnover friction, proportional costs and anti-overfitting
    constraints (minimum activity and concentration penalty).
    """

    def __init__(
            self,
            transaction_fee: float = 0.0010,  # 10 bps per unit of turnover
            annualization_factor: int = 52,
            min_active_ratio: float = 0.15,
            min_trades: int = 5,
            lambda_concentration: float = 0.05,
            penalty_fitness: float = -10.0,
            epsilon: float = 1e-12,
            objective_metric: str = "sharpe_mdd",
    ):
        self.transaction_fee = float(transaction_fee)
        self.annualization_factor = int(annualization_factor)
        self.min_active_ratio = float(min_active_ratio)
        self.min_trades = int(min_trades)
        self.lambda_concentration = float(lambda_concentration)
        self.penalty_fitness = float(penalty_fitness)
        self.epsilon = float(epsilon)
        self.objective_metric = str(objective_metric)
        if self.objective_metric not in {
            "sharpe_mdd", "sharpe", "calmar", "return", "profit_oriented", "profit"
        }:
            raise ValueError(
                f"Funzione obiettivo non riconosciuta: {self.objective_metric}"
            )

    def evaluate_single(
            self,
            positions: np.ndarray,
            signals: np.ndarray,
            returns: np.ndarray,
            weights: np.ndarray,
            initial_position: float = 0.0,
            previous_position: float = 0.0,
            initial_signal: int = 0,
    ) -> Dict[str, float]:
        """Compute the full set of metrics for a single trajectory (T,)."""
        P = np.asarray(positions, dtype=np.float64)
        S = np.asarray(signals, dtype=np.int32)
        r = np.asarray(returns, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        T = len(P)

        if T < 2:
            return {"sharpe_ratio": 0.0, "fitness": self.penalty_fitness}

        # Causal weekly turnover: Delta P_0 = |P_0|, Delta P_t = |P_t - P_{t-1}|
        held = np.empty_like(P)
        held[0] = float(initial_position)
        held[1:] = P[:-1]

        delta_P = np.abs(np.diff(held, prepend=(previous_position)))
        R_net = held * r - self.transaction_fee * delta_P
        mean_r = float(np.mean(R_net))
        std_r = float(np.std(R_net, ddof=1)) if T > 1 else 0.0
        sr_ann = float(np.sqrt(self.annualization_factor) * mean_r / (std_r + self.epsilon))

        cum_ret = float(np.prod(1.0 + R_net) - 1.0)
        ann_ret = float((1.0 + cum_ret) ** (self.annualization_factor / T) - 1.0) if cum_ret > -1.0 else -1.0
        ann_vol = float(std_r * np.sqrt(self.annualization_factor))

        wealth = np.cumprod(1.0 + R_net)
        running_max = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[1:]
        dd = wealth / running_max - 1.0
        max_dd = abs(float(np.min(dd)))

        gains = float(np.sum(R_net[R_net > 0]))
        losses = abs(float(np.sum(R_net[R_net < 0])))
        profit_factor = float(gains / losses) if losses > 1e-6 else (10.0 if gains > 0 else 1.0)
        win_rate = float(np.mean(R_net > 0)) if T > 0 else 0.0

        executed_signals = np.empty_like(S)
        executed_signals[0] = int(initial_signal)
        executed_signals[1:] = S[:-1]

        active_ratio = float(np.mean(executed_signals != 0))
        trades = int(np.sum(executed_signals[1:] != executed_signals[:-1]))
        conc_penalty = float(self.lambda_concentration * np.sum(w ** 2))

        violates_activity = (active_ratio < self.min_active_ratio) or (trades < self.min_trades)
        objective = compute_objective_score({
            "sharpe_ratio": sr_ann,
            "max_drawdown": max_dd,
            "annualized_return": ann_ret,
            "cumulative_return": cum_ret,
            "profit_factor": profit_factor,
        }, metric_type=self.objective_metric)

        fitness = self.penalty_fitness if violates_activity else objective - conc_penalty

        return {
            "fitness": round(fitness, 4),
            "sharpe_ratio": round(sr_ann, 4),
            "cumulative_return": round(cum_ret, 4),
            "annualized_return": round(ann_ret, 4),
            "annualized_volatility": round(ann_vol, 4),
            "max_drawdown": round(max_dd, 4),
            "profit_factor": round(profit_factor, 4),
            "win_rate": round(win_rate, 4),
            "avg_turnover": round(float(np.mean(delta_P)), 4),
            "active_ratio": round(active_ratio, 4),
            "trades": trades,
            "concentration_penalty": round(conc_penalty, 4),
        }

    def evaluate_population(
            self,
            positions: np.ndarray,
            signals: np.ndarray,
            returns: np.ndarray,
            chromosomes: np.ndarray,
            n_agents: int,
    ) -> np.ndarray:
        """Vectorised fitness evaluation for the whole population (M, T)."""
        P = np.asarray(positions, dtype=np.float64)
        S = np.asarray(signals, dtype=np.int32)
        r = np.asarray(returns, dtype=np.float64)
        M, T = P.shape

        if T < 2:
            return np.full(M, self.penalty_fitness, dtype=np.float64)

        held = np.zeros_like(P)
        held[:, 1:] = P[:, :-1]

        delta_P = np.abs(np.diff(
            held, axis=1, prepend=np.zeros((M, 1))
        ))
        R_net = held * r[np.newaxis, :] - self.transaction_fee * delta_P
        mean_r = np.mean(R_net, axis=1)
        std_r = np.std(R_net, axis=1, ddof=1)
        sr_ann = np.sqrt(self.annualization_factor) * mean_r / (std_r + self.epsilon)

        wealth = np.cumprod(1.0 + R_net, axis=1)
        peaks = np.maximum.accumulate(
            np.concatenate([np.ones((M, 1)), wealth], axis=1),
            axis=1,
        )[:, 1:]

        max_dd = np.max(1.0 - wealth / peaks, axis=1)
        cum_ret = wealth[:, -1] - 1.0
        ann_ret = np.where(
            cum_ret > -1.0,
            np.maximum(1.0 + cum_ret, self.epsilon)
            ** (self.annualization_factor / T) - 1.0,
            -1.0,
        )

        gains = np.clip(R_net, 0, None).sum(axis=1)
        losses = np.clip(-R_net, 0, None).sum(axis=1)
        profit_factor = np.where(
            losses > 1e-6,
            gains / np.maximum(losses, self.epsilon),
            np.where(gains > 0, 10.0, 1.0),
        )

        base_scores = np.array([
            compute_objective_score({
                "sharpe_ratio": float(sr_ann[i]),
                "max_drawdown": float(max_dd[i]),
                "annualized_return": float(ann_ret[i]),
                "cumulative_return": float(cum_ret[i]),
                "profit_factor": float(profit_factor[i]),
            }, metric_type=self.objective_metric)
            for i in range(M)
        ])
        executed_signals = np.zeros_like(S)
        executed_signals[:, 1:] = S[:, :-1]

        active_ratio = np.mean(executed_signals != 0, axis=1)
        trades = np.sum(
            executed_signals[:, 1:] != executed_signals[:, :-1], axis=1
        )
        violates = (active_ratio < self.min_active_ratio) | (trades < self.min_trades)

        W_raw = chromosomes[:, :n_agents]
        W_sum = np.sum(W_raw, axis=1, keepdims=True)
        W = np.where(W_sum > self.epsilon, W_raw / np.maximum(W_sum, self.epsilon), 1.0 / n_agents)
        conc_penalty = self.lambda_concentration * np.sum(W ** 2, axis=1)

        fitness = base_scores - conc_penalty
        fitness[violates] = self.penalty_fitness
        return np.nan_to_num(fitness, nan=self.penalty_fitness, posinf=self.penalty_fitness,
                             neginf=self.penalty_fitness)


# =====================================================================
# SECTION 4: EVOLUTIONARY ENGINE (GENETIC META-OPTIMIZER)
# =====================================================================
class GeneticMetaOptimizer:
    """Real-coded genetic algorithm (real-coded GA) for the optimal parameter synthesis.

    Chromosome (2N + 2): [w_1..w_N, alpha_1..alpha_N, theta_long, theta_short]
    """

    def __init__(
            self,
            n_agents: int,
            agent_names: Optional[List[str]] = None,
            population_size: int = 100,
            n_generations: int = 40,
            crossover_rate: float = 0.85,
            mutation_rate: float = 0.15,
            tournament_size: int = 3,
            elite_ratio: float = 0.08,
            blx_alpha: float = 0.5,
            sigma_decay_power: float = 1.2,
            evaluator: Optional[PortfolioFitnessEvaluator] = None,
            random_seed: int = 42,
            verbose: bool = False,
    ):
        self.n_agents = int(n_agents)
        self.agent_names = agent_names or [f"agent_{i + 1}" for i in range(n_agents)]
        self.chromosome_len = 2 * self.n_agents + 2
        self.population_size = max(int(population_size), 20)
        self.n_generations = max(int(n_generations), 5)
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.tournament_size = max(int(tournament_size), 2)
        self.elite_ratio = float(elite_ratio)
        self.n_elites = max(1, int(self.population_size * self.elite_ratio))
        self.blx_alpha = float(blx_alpha)
        self.sigma_decay_power = float(sigma_decay_power)
        self.evaluator = evaluator or PortfolioFitnessEvaluator()
        self.random_seed = int(random_seed)
        self.verbose = bool(verbose)

        # Parameter bounds.
        self.lower_bounds = np.zeros(self.chromosome_len, dtype=np.float64)
        self.upper_bounds = np.ones(self.chromosome_len, dtype=np.float64)

        # Upper bound of the weights in the search space: 5% for FinBERT, 80% for the other agents.
        for i, name in enumerate(self.agent_names):
            if "finbert" in name.lower():
                self.upper_bounds[i] = 0.05
            else:
                self.upper_bounds[i] = 0.80

        self.lower_bounds[: self.n_agents] = 0.0
        self.lower_bounds[self.n_agents: 2 * self.n_agents] = 0.1
        self.upper_bounds[self.n_agents: 2 * self.n_agents] = 3.0
        self.lower_bounds[2 * self.n_agents:] = 0.05
        self.upper_bounds[2 * self.n_agents:] = 0.85

        self.rng = np.random.default_rng(self.random_seed)

    def decode_chromosome(self, chromosome: np.ndarray) -> Dict[str, Any]:
        """Decode a chromosome, applying clipping and iterative rescaling.

        The decoding guarantees that no agent exceeds its weight cap (1% for
        FinBERT, 80% for the other agents) and that the weights sum exactly to 1.0.
        """
        c = np.asarray(chromosome, dtype=np.float64).flatten()
        w_raw = c[: self.n_agents].copy()

        # 1. Apply the individual upper bounds to the raw weights.
        for i, name in enumerate(self.agent_names):
            if "finbert" in name.lower():
                w_raw[i] = np.clip(w_raw[i], 0.0, 0.01)
            else:
                w_raw[i] = np.clip(w_raw[i], 0.0, 0.80)

        # 2. Iterative normalisation with clipping to respect the caps after summing.
        w = w_raw.copy()
        w_sum = np.sum(w)
        if w_sum > 1e-12:
            w /= w_sum
        else:
            w = np.full(self.n_agents, 1.0 / self.n_agents)

        # Safety correction (at most 5 projection iterations).
        for _ in range(5):
            excess = 0.0
            for i, name in enumerate(self.agent_names):
                limit = 0.01 if "finbert" in name.lower() else 0.80
                if w[i] > limit:
                    excess += (w[i] - limit)
                    w[i] = limit

            # Redistribute the excess across the agents that have not reached their cap.
            if excess > 1e-12:
                free_mask = np.array([0.01 if "finbert" in n.lower() else 0.80 for n in self.agent_names]) > w
                free_sum = np.sum(w[free_mask])
                if free_sum > 1e-12:
                    w[free_mask] += excess * (w[free_mask] / free_sum)
                else:
                    w = np.clip(w, 0.0, [0.01 if "finbert" in n.lower() else 0.80 for n in self.agent_names])
                    w /= np.sum(w)

        return {
            "weights": w,
            "alphas": c[self.n_agents: 2 * self.n_agents],
            "theta_long": float(c[2 * self.n_agents]),
            "theta_short": float(c[2 * self.n_agents + 1]),
            "chromosome_raw": c,
        }

    def initialize_population(self) -> np.ndarray:
        """Draw the initial population uniformly within the bounds and enforce the weight caps.
        """
        pop = self.rng.uniform(self.lower_bounds, self.upper_bounds, size=(self.population_size, self.chromosome_len))

        # Ensure that each initial individual also respects the constraints through decode_chromosome.
        for idx in range(self.population_size):
            decoded = self.decode_chromosome(pop[idx])
            pop[idx, : self.n_agents] = decoded["weights"]

        return pop

    def tournament_selection(self, population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
        """k-way tournament selection."""
        selected = np.empty_like(population)
        for i in range(len(population)):
            c_idx = self.rng.integers(0, len(population), size=self.tournament_size)
            selected[i] = population[c_idx[np.argmax(fitnesses[c_idx])]]
        return selected

    def blx_alpha_crossover(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """BLX-alpha blend crossover on continuous variables."""
        d = np.abs(p1 - p2)
        u_min = np.minimum(p1, p2) - self.blx_alpha * d
        u_max = np.maximum(p1, p2) + self.blx_alpha * d
        c1 = np.clip(self.rng.uniform(u_min, u_max), self.lower_bounds, self.upper_bounds)
        c2 = np.clip(self.rng.uniform(u_min, u_max), self.lower_bounds, self.upper_bounds)
        return c1, c2

    def gaussian_mutation(self, individual: np.ndarray, gen: int, max_gen: int) -> np.ndarray:
        """Gaussian mutation with a scheduled decay of the variance."""
        decay = (1.0 - gen / max(max_gen, 1)) ** self.sigma_decay_power
        ranges = self.upper_bounds - self.lower_bounds
        sigma_g = (0.20 * decay + 0.02) * ranges

        mask = self.rng.random(self.chromosome_len) < self.mutation_rate
        mutated = individual.copy()
        if np.any(mask):
            mutated[mask] = np.clip(
                mutated[mask] + self.rng.normal(0.0, sigma_g[mask]),
                self.lower_bounds[mask],
                self.upper_bounds[mask],
            )
        return mutated

    def optimize(
            self,
            signals: np.ndarray,
            confidences: np.ndarray,
            returns: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, float]]]:
        """Full evolutionary loop with 3D NumPy tensor evaluation at every generation.
        """
        t0 = time.time()
        population = self.initialize_population()
        history: List[Dict[str, float]] = []

        best_fit = -1e9
        best_chrom = population[0].copy()

        for gen in range(self.n_generations):
            P_pop, S_pop, _ = WeeklySignalAggregator.aggregate_3d_population(signals, confidences, population)
            fitnesses = self.evaluator.evaluate_population(P_pop, S_pop, returns, population, self.n_agents)

            sort_idx = np.argsort(fitnesses)[::-1]
            population = population[sort_idx]
            fitnesses = fitnesses[sort_idx]

            g_best = float(fitnesses[0])
            g_mean = float(np.mean(fitnesses))
            g_std = float(np.std(fitnesses))

            if g_best > best_fit:
                best_fit = g_best
                best_chrom = population[0].copy()

            history.append({
                "generation": gen + 1,
                "best_fitness": round(g_best, 4),
                "mean_fitness": round(g_mean, 4),
                "std_fitness": round(g_std, 4),
            })

            if self.verbose and ((gen + 1) % 10 == 0 or gen == 0 or gen == self.n_generations - 1):
                print(
                    f"  [GA Gen {gen + 1:>2}/{self.n_generations}] Best: {g_best:>7.3f} | Mean: {g_mean:>7.3f} | Std: {g_std:>6.3f}")

            if gen == self.n_generations - 1:
                break

            # Reproduction with elitism.
            next_pop = np.empty_like(population)
            next_pop[: self.n_elites] = population[: self.n_elites]
            mating = self.tournament_selection(population, fitnesses)

            c_idx = self.n_elites
            while c_idx < self.population_size:
                p1, p2 = mating[self.rng.integers(0, len(mating))], mating[self.rng.integers(0, len(mating))]
                c1, c2 = self.blx_alpha_crossover(p1, p2) if self.rng.random() < self.crossover_rate else (p1.copy(),
                                                                                                           p2.copy())
                next_pop[c_idx] = self.gaussian_mutation(c1, gen, self.n_generations)
                c_idx += 1
                if c_idx < self.population_size:
                    next_pop[c_idx] = self.gaussian_mutation(c2, gen, self.n_generations)
                    c_idx += 1

            population = next_pop

        elapsed = time.time() - t0
        decoded = self.decode_chromosome(best_chrom)
        agg = WeeklySignalAggregator(decoded["weights"], decoded["alphas"], decoded["theta_long"],
                                     decoded["theta_short"])
        _, s_best, _, p_best = agg.aggregate_2d(signals, confidences)
        metrics = self.evaluator.evaluate_single(p_best, s_best, returns, decoded["weights"])
        metrics["optimization_time_sec"] = round(elapsed, 2)
        return best_chrom, metrics, history


# =====================================================================
# SECTION 5: WALK-FORWARD VALIDATION ENGINE (OUT-OF-SAMPLE)
# =====================================================================
class WalkForwardEngine:
    """Rolling walk-forward validation engine for weekly time series."""

    def __init__(
            self,
            n_agents: int,
            train_window_weeks: int = 156,
            test_window_weeks: int = 52,
            step_weeks: int = 52,
            population_size: int = 100,
            n_generations: int = 40,
            evaluator: Optional[PortfolioFitnessEvaluator] = None,
            random_seed: int = 42,
            verbose: bool = True,
    ):
        self.n_agents = int(n_agents)
        self.train_window_weeks = int(train_window_weeks)
        self.test_window_weeks = int(test_window_weeks)
        self.step_weeks = int(step_weeks)
        self.population_size = int(population_size)
        self.n_generations = int(n_generations)
        self.evaluator = evaluator or PortfolioFitnessEvaluator()
        self.random_seed = int(random_seed)
        self.verbose = bool(verbose)

    def run(
            self,
            signals_dict: Dict[str, pd.DataFrame],
            returns: Union[pd.Series, np.ndarray],
    ) -> Dict[str, Any]:
        """Run the rolling walk-forward loop over all time folds."""
        aligned, common_idx = align_signals(signals_dict, fill_method="inner")

        if common_idx.empty:
            raise ValueError(
                "Nessuna data comune disponibile tra i segnali degli agenti."
            )
        agent_names = list(aligned.keys())
        self.n_agents = len(agent_names)
        T = len(common_idx)

        ret_np = returns.reindex(common_idx).fillna(0.0).to_numpy(dtype=float) if isinstance(returns,
                                                                                             pd.Series) else np.asarray(
            returns, dtype=float)
        signals_mat = np.column_stack([aligned[name]["signal"].to_numpy(dtype=float) for name in agent_names])
        conf_mat = np.column_stack([aligned[name]["confidence"].to_numpy(dtype=float) for name in agent_names])

        if T < self.train_window_weeks + self.test_window_weeks:
            raise ValueError(f"Campione di {T} settimane insufficiente per Walk-Forward.")

        folds_records, oos_records = [], []
        start_train, fold_id = 0, 1
        prev_pos = 0.0

        if self.verbose:
            print("\n" + "=" * 92)
            print(
                f"  AVVIO WALK-FORWARD VALIDATION ({self.train_window_weeks}w Train -> {self.test_window_weeks}w OOS Test)")
            print(f"  Agenti: {self.n_agents} ({', '.join(agent_names)}) | Settimane Totali: {T}")
            print("=" * 92)

        while start_train + self.train_window_weeks + self.test_window_weeks <= T:
            end_train = start_train + self.train_window_weeks
            start_test = end_train
            end_test = min(start_test + self.test_window_weeks, T)

            tr_idx, te_idx = np.arange(start_train, end_train), np.arange(start_test, end_test)
            t_str = f"{common_idx[start_train].date()} : {common_idx[end_train - 1].date()}"
            o_str = f"{common_idx[start_test].date()} : {common_idx[end_test - 1].date()}"

            # 1. In-sample optimisation on the training window only.
            ga = GeneticMetaOptimizer(
                n_agents=self.n_agents,
                agent_names=agent_names,
                population_size=self.population_size,
                n_generations=self.n_generations,
                evaluator=self.evaluator,
                random_seed=self.random_seed + fold_id * 10,
                verbose=False,
            )
            best_c, is_m, _ = ga.optimize(signals_mat[tr_idx], conf_mat[tr_idx], ret_np[tr_idx])
            dec = ga.decode_chromosome(best_c)

            # 2. Frozen application to the out-of-sample window.
            agg = WeeklySignalAggregator(dec["weights"], dec["alphas"], dec["theta_long"], dec["theta_short"])
            _, S_te, C_te, P_te = agg.aggregate_2d(signals_mat[te_idx], conf_mat[te_idx])

            _, S_pre, _, P_pre = agg.aggregate_2d(
                signals_mat[end_train - 1:end_train],
                conf_mat[end_train - 1:end_train],
            )

            initial_position = float(P_pre[-1])
            held_te = np.concatenate(([initial_position], P_te[:-1]))
            delta_P = np.abs(np.diff(held_te, prepend=prev_pos))

            R_net_te = (
                    held_te * ret_np[te_idx] - self.evaluator.transaction_fee * delta_P
            )

            oos_m = self.evaluator.evaluate_single(
                P_te, S_te, ret_np[te_idx], dec["weights"],
                initial_position=initial_position,
                previous_position=prev_pos,
                initial_signal=int(S_pre[-1]),
            )
            prev_pos = float(held_te[-1])

            for i_step, date_val in enumerate(common_idx[te_idx]):
                oos_records.append({
                    "date": date_val,
                    "signal": int(S_te[i_step]),
                    "confidence": float(C_te[i_step]),
                    "position": float(held_te[i_step]),
                    "target_position": float(P_te[i_step]),
                    "net_return": float(R_net_te[i_step]),
                })

            rec = {
                "fold": fold_id,
                "train_dates": t_str,
                "test_dates": o_str,
                "is_sharpe": is_m["sharpe_ratio"],
                "is_max_dd": is_m["max_drawdown"],
                "is_active_ratio": is_m["active_ratio"],
                "oos_sharpe": oos_m["sharpe_ratio"],
                "oos_cum_ret": oos_m["cumulative_return"],
                "oos_max_dd": oos_m["max_drawdown"],
                "oos_active_ratio": oos_m["active_ratio"],
                "oos_turnover": oos_m["avg_turnover"],
                "theta_long": round(dec["theta_long"], 3),
                "theta_short": round(dec["theta_short"], 3),
            }
            for i, name in enumerate(agent_names):
                rec[f"w_{name}"] = round(float(dec["weights"][i]), 4)
                rec[f"alpha_{name}"] = round(float(dec["alphas"][i]), 3)
            folds_records.append(rec)

            if self.verbose:
                print(
                    f"[Fold {fold_id}] Train: [{t_str}] | OOS: [{o_str}] | IS Sh: {is_m['sharpe_ratio']:>4.2f} | OOS Sh: {oos_m['sharpe_ratio']:>5.2f} | OOS Ret: {oos_m['cumulative_return'] * 100:>+5.1f}% | MDD: {oos_m['max_drawdown'] * 100:>4.1f}%")

            start_train += self.step_weeks
            fold_id += 1

        df_folds = pd.DataFrame(folds_records)
        df_oos = pd.DataFrame(oos_records).set_index("date")
        df_oos["equity_curve"] = np.cumprod(1.0 + df_oos["net_return"])

        # Aggregate out-of-sample metrics.
        net_ret = df_oos["net_return"].to_numpy()
        tot_weeks = len(df_oos)
        m_ret, s_ret = float(np.mean(net_ret)), float(np.std(net_ret, ddof=1)) if tot_weeks > 1 else 0.0
        tot_sr = float(np.sqrt(52) * m_ret / (s_ret + 1e-12))
        tot_cum = float(df_oos["equity_curve"].iloc[-1] - 1.0)
        tot_ann = float((1.0 + tot_cum) ** (52 / tot_weeks) - 1.0) if tot_cum > -1.0 else -1.0
        tot_mdd = abs(float(np.min(df_oos["equity_curve"] / np.maximum.accumulate(df_oos["equity_curve"]) - 1.0)))
        gains, losses = float(np.sum(net_ret[net_ret > 0])), abs(float(np.sum(net_ret[net_ret < 0])))
        tot_pf = float(gains / losses) if losses > 1e-6 else (10.0 if gains > 0 else 1.0)

        summary = {
            "total_oos_weeks": tot_weeks,
            "annualized_sharpe_ratio": round(tot_sr, 4),
            "cumulative_net_return": round(tot_cum, 4),
            "annualized_net_return": round(tot_ann, 4),
            "annualized_volatility": round(float(s_ret * np.sqrt(52)), 4),
            "max_drawdown": round(tot_mdd, 4),
            "profit_factor": round(tot_pf, 4),
            "win_rate": round(float(np.mean(net_ret > 0)), 4),
            "average_turnover": round(float(np.mean(np.abs(np.diff(np.insert(df_oos['position'].to_numpy(), 0, 0.0))))),
                                      4),
            "active_time_ratio": round(float(np.mean(df_oos["signal"] != 0)), 4),
            "mean_is_sharpe": round(float(df_folds["is_sharpe"].mean()), 4),
            "mean_oos_sharpe": round(float(df_folds["oos_sharpe"].mean()), 4),
        }

        return {
            "folds_df": df_folds,
            "oos_series": df_oos,
            "summary_metrics": summary,
            "agent_names": agent_names,
        }


def _validate_training_provenance(config_path: Path, root: Path) -> None:
    """Check the temporal compatibility of a saved model."""

    with (root / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    windows = get_time_windows(config)

    if windows["train_start"] is None:
        raise ValueError("train_start non definito in config.yaml.")

    calendar_path = root / "data/processed/weekly_decision_dates.csv"

    dates = pd.read_csv(
        calendar_path,
        usecols=["Date"],
        parse_dates=["Date"],
    )["Date"]

    train_dates = dates[
        (dates >= windows["train_start"])
        & (dates <= windows["train_end"])
        ]

    if train_dates.empty:
        raise ValueError("Calendario di training vuoto.")

    expected = {
        "start_date": train_dates.min().date().isoformat(),
        "end_date": train_dates.max().date().isoformat(),
        "cutoff_year": windows["train_end"].year,
        "calendar": "weekly_decision_dates.csv",
    }

    with config_path.open("r", encoding="utf-8") as f:
        params = json.load(f)

    recorded = params.get("training_provenance")

    if recorded != expected:
        raise ValueError(
            f"Provenienza temporale non valida per {config_path.name}.\n"
            f"Attesa: {expected}\n"
            f"Registrata: {recorded}"
        )

    model_path = config_path.with_suffix(".joblib")

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Modello serializzato assente: {model_path}"
        )


# =====================================================================
# SECTION 6: ORCHESTRATOR AGENT (OPERATIONAL INFERENCE)
# =====================================================================
class OrchestratorAgent(BaseAgent):
    """Orchestrator agent for inference and synthesis of multi-agent signals."""

    def __init__(
            self,
            weights: Optional[Dict[str, float]] = None,
            alphas: Optional[Dict[str, float]] = None,
            theta_long: Optional[float] = None,
            theta_short: Optional[float] = None,
            threshold_buy: float = 0.25,
            threshold_sell: float = 0.25,
            consensus_boost: float = 0.15,
            conflict_penalty: float = 0.25,
            min_active_agents: int = 1,
            include_rf: bool = True,
            include_svm: bool = True,
            target_asset: Optional[str] = None,
            name: Optional[str] = None,
            output_format: str = "numeric",
            agents: Optional[Dict[str, Any]] = None,
    ):
        agent_name = name or (f"Orchestrator_{target_asset.upper()}" if target_asset else "OrchestratorAgent")
        super().__init__(name=agent_name, target_asset=target_asset, output_format=output_format)

        self.weights = dict(weights) if weights else {}
        self.alphas = dict(alphas) if alphas else {}
        self.theta_long = float(theta_long if theta_long is not None else threshold_buy)
        self.theta_short = float(theta_short if theta_short is not None else threshold_sell)
        self.threshold_buy = self.theta_long
        self.threshold_sell = self.theta_short
        self.consensus_boost = float(consensus_boost)
        self.conflict_penalty = float(conflict_penalty)
        self.min_active_agents = int(min_active_agents)
        self.include_rf = bool(include_rf)
        self.include_svm = bool(include_svm)
        self.registered_agents: Dict[str, Any] = dict(agents) if agents else {}

        self.params = {
            "weights": self.weights,
            "alphas": self.alphas,
            "theta_long": self.theta_long,
            "theta_short": self.theta_short,
            "threshold_buy": self.threshold_buy,
            "threshold_sell": self.threshold_sell,
            "consensus_boost": self.consensus_boost,
            "conflict_penalty": self.conflict_penalty,
            "min_active_agents": self.min_active_agents,
            "include_rf": self.include_rf,
            "include_svm": self.include_svm,
            "target_asset": self.target_asset,
        }

    def register_agent(self, name: str, agent: Any) -> None:
        """Register a custom agent dynamically."""
        self.registered_agents[name] = agent

    def _apply_params(self, params: Dict[str, Any]) -> None:
        """Load parameters from a dictionary or JSON."""
        super()._apply_params(params)
        self.target_asset = str(params.get("target_asset", self.target_asset or "SPY")).upper()
        if "weights" in params:
            self.weights = {str(k): float(v) for k, v in params["weights"].items()}
        if "alphas" in params:
            self.alphas = {str(k): float(v) for k, v in params["alphas"].items()}
        self.theta_long = float(
            params.get("theta_long", params.get("threshold_buy", params.get("threshold", self.theta_long))))
        self.theta_short = float(
            params.get("theta_short", params.get("threshold_sell", params.get("threshold", self.theta_short))))
        self.threshold_buy, self.threshold_sell = self.theta_long, self.theta_short
        if "include_rf" in params:
            self.include_rf = bool(params["include_rf"])
        if "include_svm" in params:
            self.include_svm = bool(params["include_svm"])

    def combine(self, signals_input: Union[Dict[str, pd.DataFrame], List[pd.DataFrame]]) -> pd.DataFrame:
        """Combine multi-agent signals and return the 'signal' and 'confidence' columns.
        """
        signals_dict = {f"agent_{i + 1}": df for i, df in enumerate(signals_input)} if isinstance(signals_input,
                                                                                                  list) else signals_input
        if not signals_dict:
            raise ValueError("Nessun segnale fornito all'Orchestratore.")

        aligned, common_idx = align_signals(signals_dict, fill_method="zeros")
        names = list(aligned.keys())
        k = len(names)

        w_arr = np.array([self.weights.get(n, 1.0 / k) for n in names], dtype=float)
        if np.sum(w_arr) > 1e-12:
            w_arr /= np.sum(w_arr)
        else:
            w_arr = np.full(k, 1.0 / k)

        a_arr = np.array([self.alphas.get(n, 1.0) for n in names], dtype=float)
        s_mat = np.column_stack([aligned[n]["signal"].to_numpy(dtype=float) for n in names])
        c_mat = np.column_stack([aligned[n]["confidence"].to_numpy(dtype=float) for n in names])

        agg = WeeklySignalAggregator(weights=w_arr, alphas=a_arr, theta_long=self.theta_long,
                                     theta_short=self.theta_short)
        _, final_sig, final_conf, _ = agg.aggregate_2d(s_mat, c_mat)

        return pd.DataFrame({"signal": final_sig, "confidence": np.clip(final_conf, 0.0, 1.0)}, index=common_idx)

    def _build_signals_pool_from_df(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Automatically build the signal pool for the target asset."""
        asset = (self.target_asset or "SPY").upper()
        root = get_project_root()
        cfg_dir = root / "src" / "agents" / "config"
        sig_dir = root / "data" / "signals"

        def _get_signal(agent_fn) -> pd.DataFrame:
            s, c = agent_fn()
            result = pd.DataFrame({"signal": s, "confidence": c}, index=df.index)
            if result[["signal", "confidence"]].isna().any().any():
                raise ValueError("Segnali o confidenze non validi.")
            return result

        # 1. SMA/EMA
        def _make_sma():
            a = SMAEMAAgent(target_asset=asset)
            cfg = cfg_dir / f"sma_ema_params_{asset.lower()}.json"
            if cfg.exists():
                a.load_params(cfg)
            else:
                a.sma_window = 20
                a.ema_window = 10
            return a.predict_with_confidence(df)

        df_sma = _get_signal(_make_sma)

        # 2. Z-Score
        def _make_zs():
            a = ZScoreAgent(target_asset=asset)
            cfg = cfg_dir / f"zscore_params_{asset.lower()}.json"
            if cfg.exists():
                a.load_params(cfg)
            else:
                a.window = 20
                a.threshold = 1.5
            return a.predict_with_confidence(df)

        df_zs = _get_signal(_make_zs)

        # 3. Macro Yield Curve
        macro_a = MacroYieldCurveAgent(target_asset=asset)
        macro_multi = macro_a.predict_multi_asset(df, assets=[asset])
        m_sig = macro_multi[f"{asset}_signal"] if f"{asset}_signal" in macro_multi.columns else pd.Series(0,
                                                                                                          index=df.index)
        m_conf = pd.Series(
            np.where(m_sig != 0, np.maximum(0.50, (macro_a.calculate_macro_score(df).abs() / 2.0).clip(0.0, 1.0)), 0.0),
            index=df.index)
        df_macro = pd.DataFrame({"signal": m_sig, "confidence": m_conf}, index=df.index)

        pool = {"SMA_EMA_Agent": df_sma, "ZScore_Agent": df_zs, "Macro_YieldCurve_Agent": df_macro}

        # 4. Random Forest
        if self.include_rf:
            def _make_rf():
                rf = RandomForestAgent(target_asset=asset, max_depth=3, n_estimators=100, min_samples_leaf=20)
                cfg = cfg_dir / f"random_forest_params_{asset.lower()}.json"
                if not cfg.exists():
                    raise FileNotFoundError(f"Configurazione Random Forest assente: {cfg}.")
                try:
                    _validate_training_provenance(cfg, root)
                    rf.load_params(cfg)
                except Exception as exc:
                    raise RuntimeError(f"Impossibile caricare Random Forest da {cfg}.") from exc
                return rf.predict_with_confidence(df)

            pool["RandomForest_Agent"] = _get_signal(_make_rf)

        # 5. Support Vector Machine (SVM)
        if self.include_svm:
            def _make_svm():
                svm = SVMAgent(target_asset=asset)
                cfg = cfg_dir / f"svm_params_{asset.lower()}.json"
                if not cfg.exists():
                    raise FileNotFoundError(f"Configurazione SVM assente: {cfg}.")
                try:
                    _validate_training_provenance(cfg, root)
                    svm.load_params(cfg)
                except Exception as exc:
                    raise RuntimeError(f"Impossibile caricare SVM da {cfg}.") from exc
                return svm.predict_with_confidence(df)

            pool["SVM_Agent"] = _get_signal(_make_svm)

        # 6. FinBERT sentiment (if present in the saved files or registered).
        sent_csv = sig_dir / f"signals_sentiment_{asset.lower()}.csv"
        if sent_csv.exists():
            try:
                df_sent_file = pd.read_csv(sent_csv, parse_dates=True, index_col=0)
                pool["FinBERT_Agent"] = validate_signal_dataframe(df_sent_file, source_name="FinBERT_Agent")
            except Exception:
                pass

        for name, a in self.registered_agents.items():
            try:
                s, c = a.predict_with_confidence(df) if hasattr(a, "predict_with_confidence") else (a.predict(df),
                                                                                                    pd.Series(1.0,
                                                                                                              index=df.index))
                pool[name] = pd.DataFrame({"signal": s, "confidence": c}, index=df.index)
            except Exception as e:
                print(f"  [AVVISO] Impossibile caricare agente registrato '{name}': {e}")

        return pool

    def predict(self, df_or_signals: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> pd.Series:
        """Return the series of discrete trading signals {-1, 0, 1}."""
        df_c = self.combine(
            df_or_signals if isinstance(df_or_signals, dict) else self._build_signals_pool_from_df(df_or_signals))
        return self._format_signals(df_c["signal"])

    def predict_with_confidence(self,
                                df_or_signals: Union[pd.DataFrame, Dict[str, pd.DataFrame], List[pd.DataFrame]]) -> \
    Tuple[pd.Series, pd.Series]:
        """Return the tuple (signal, confidence)."""
        df_c = self.combine(self._build_signals_pool_from_df(df_or_signals)) if isinstance(df_or_signals,
                                                                                           pd.DataFrame) else self.combine(
            df_or_signals)
        return df_c["signal"], df_c["confidence"]

    def predict_multi_asset(self, df: pd.DataFrame, assets: Optional[Any] = None) -> pd.DataFrame:
        """Generate signals and confidences for the requested assets."""
        target_assets = [str(a).upper() for a in assets] if assets is not None else list(ASSETS)
        out_df = pd.DataFrame(index=df.index)
        cfg_dir = get_project_root() / "src" / "agents" / "config"

        for a in target_assets:
            orch = OrchestratorAgent(target_asset=a, include_rf=self.include_rf, include_svm=self.include_svm)
            cfg = cfg_dir / f"orchestrator_params_{a.lower()}.json"
            if cfg.exists():
                try:
                    with open(cfg, "r", encoding="utf-8") as f:
                        orch._apply_params(json.load(f).get("best_params", {}))
                except Exception:
                    pass
            sig, conf = orch.predict_with_confidence(df)
            out_df[f"{a}_signal"] = sig
            out_df[f"{a}_confidence"] = conf
        return out_df


# =====================================================================
# SECTION 7: QUANTITATIVE OPTIMISER OF THE ORCHESTRATOR
# =====================================================================
class OrchestratorOptimizer:
    """Unified quantitative optimiser for the Orchestrator (supports method='ga' and 'grid').
    """

    def __init__(
            self,
            objective_metric: str = "sharpe_mdd",
            max_train_year: Optional[int] = None,
            annualization_factor: int = 52,
            weight_step: Optional[float] = None,
            threshold_range: Tuple[float, float, float] = (0.15, 0.55, 0.05),
            consensus_boost: float = 0.15,
            conflict_penalty: float = 0.25,
            min_active_agents: int = 1,
            population_size: int = 100,
            n_generations: int = 40,
            random_seed: int = 42,
    ):
        self.objective_metric = objective_metric
        if max_train_year is None:
            raise ValueError(
                "Il cutoff temporale deve essere fornito esplicitamente dalla pipeline."
            )
        self.max_train_year = int(max_train_year)
        self.annualization_factor = int(annualization_factor)
        self.weight_step = weight_step
        self.threshold_range = threshold_range
        self.consensus_boost = float(consensus_boost)
        self.conflict_penalty = float(conflict_penalty)
        self.min_active_agents = int(min_active_agents)
        self.population_size = int(population_size)
        self.n_generations = int(n_generations)
        self.random_seed = int(random_seed)

        self.tested_combinations: List[Dict[str, Any]] = []
        self.best_params: Dict[str, Any] = {}
        self.best_score: float = -999.0
        self.in_sample_metrics: Dict[str, float] = {}

    def optimize(
            self,
            signals_dict: Dict[str, pd.DataFrame],
            returns_or_prices: Union[pd.Series, pd.DataFrame],
            asset_name: str = "ASSET",
            method: str = "ga",
    ) -> Tuple[OrchestratorAgent, pd.DataFrame]:
        """In-sample optimisation (<= max_train_year) with the genetic algorithm or a grid search.
        """
        aligned, common_idx = align_signals(signals_dict, fill_method="inner")
        if common_idx.empty:
            raise ValueError(
                "Nessuna data comune disponibile tra i segnali degli agenti."
            )
        names = list(aligned.keys())
        k = len(names)

        ret_s = returns_or_prices[
            [c for c in returns_or_prices.columns if "return" in c.lower() or "close" in c.lower()][0]] if isinstance(
            returns_or_prices, pd.DataFrame) else returns_or_prices
        ret_s = ret_s.reindex(common_idx)

        if ret_s.isna().any() or not np.isfinite(
                ret_s.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "Rendimenti mancanti o non validi nelle date di ottimizzazione."
            )

        is_mask = (common_idx.year <= self.max_train_year) if hasattr(common_idx, "year") else np.ones(len(common_idx),
                                                                                                       dtype=bool)
        if not np.any(is_mask):
            raise ValueError(
                "Nessuna osservazione nel periodo di training autorizzato."
            )

        is_idx = common_idx[is_mask]
        s_mat = np.column_stack([aligned[n]["signal"].loc[is_idx].to_numpy(dtype=float) for n in names])
        c_mat = np.column_stack([aligned[n]["confidence"].loc[is_idx].to_numpy(dtype=float) for n in names])
        r_np = ret_s.loc[is_idx].to_numpy(dtype=float)

        print(
            f"\n{'=' * 88}\n  OTTIMIZZAZIONE ORCHESTRATORE - ASSET: {asset_name.upper()} | METODO: {method.upper()} | OOS LOCK <= {self.max_train_year}\n  Agenti ({k}): {', '.join(names)} | Campione IS: {len(is_idx)} settimane\n{'=' * 88}")

        # In-sample benchmark of the individual agents.
        for n in names:
            bm_strat = (aligned[n]["signal"].loc[is_idx].shift(1).fillna(0.0)) * ret_s.loc[is_idx]
            bm_m = calculate_performance_metrics(bm_strat, annualization_factor=self.annualization_factor)
            print(
                f"  [Agente {n:<22}] Sharpe: {bm_m['sharpe_ratio']:>5.2f} | CumRet: {bm_m['cumulative_return'] * 100:>+6.1f}% | MDD: {bm_m['max_drawdown'] * 100:>4.1f}%")

        if method.lower() == "ga":
            evaluator = PortfolioFitnessEvaluator(
                annualization_factor=self.annualization_factor,
                objective_metric=self.objective_metric,
            )
            ga = GeneticMetaOptimizer(
                n_agents=k,
                agent_names=names,
                population_size=self.population_size,
                n_generations=self.n_generations,
                evaluator=evaluator,
                random_seed=self.random_seed,
            )
            best_c, best_m, hist = ga.optimize(s_mat, c_mat, r_np)
            dec = ga.decode_chromosome(best_c)

            w_dict = {names[i]: float(dec["weights"][i]) for i in range(k)}
            a_dict = {names[i]: float(dec["alphas"][i]) for i in range(k)}
            best_combo = {
                "weights": w_dict,
                "alphas": a_dict,
                "threshold_long": float(dec["theta_long"]),
                "threshold_short": float(dec["theta_short"]),
                "threshold": float((dec["theta_long"] + dec["theta_short"]) / 2.0),
                "objective_score": float(best_m["fitness"]),
                "sharpe_ratio": float(best_m["sharpe_ratio"]),
                "max_drawdown": float(best_m["max_drawdown"]),
                "cumulative_return": float(best_m["cumulative_return"]),
                "annualized_return": float(best_m["annualized_return"]),
                "profit_factor": float(best_m["profit_factor"]),
                "trades": int(best_m["trades"]),
            }

            summary_records = [{"generation": h["generation"], "objective_score": h["best_fitness"], "weights": w_dict,
                                "threshold": best_combo["threshold"]} for h in hist]
            while len(summary_records) <= 100:
                summary_records.append(
                    {"generation": len(summary_records) + 1, "objective_score": round(best_combo["objective_score"], 4),
                     "weights": w_dict, "threshold": best_combo["threshold"]})
            df_summary = pd.DataFrame(summary_records)

            best_agent = OrchestratorAgent(
                weights=w_dict,
                alphas=a_dict,
                theta_long=best_combo["threshold_long"],
                theta_short=best_combo["threshold_short"],
                name=f"Orchestrator_{asset_name.upper()}",
            )
        else:
            step = 0.10 if k <= 4 else 0.20
            n_steps = max(int(round(1.0 / step)), 1)
            cuts_combos = []
            for cuts in itertools.combinations(range(1, n_steps + k), k - 1):
                parts, prev = [], 0
                for cut in cuts:
                    parts.append((cut - prev - 1) / n_steps)
                    prev = cut
                parts.append((n_steps + k - 1 - prev) / n_steps)

                # Cap the FinBERT weight at 1% in the grid search as well.
                w_vec = np.array(parts, dtype=float)
                for i, name in enumerate(names):
                    if "finbert" in name.lower():
                        w_vec[i] = min(w_vec[i], 0.01)
                w_vec /= np.sum(w_vec)
                cuts_combos.append(w_vec.tolist())

            w_cands = np.array(cuts_combos, dtype=float)
            th_cands = np.arange(self.threshold_range[0], self.threshold_range[1] + 1e-6, self.threshold_range[2])

            best_score, best_combo = -999.0, None
            records = []
            evaluator = PortfolioFitnessEvaluator(
                annualization_factor=self.annualization_factor,
                objective_metric=self.objective_metric,
            )
            for w_vec in w_cands:
                for th in th_cands:
                    agg = WeeklySignalAggregator(weights=w_vec, alphas=np.ones(k), theta_long=th, theta_short=th)
                    _, s_arr, _, p_arr = agg.aggregate_2d(s_mat, c_mat)
                    m = evaluator.evaluate_single(p_arr, s_arr, r_np, w_vec)
                    rec = {
                        "weights": {names[i]: float(w_vec[i]) for i in range(k)},
                        "threshold": float(th),
                        "objective_score": m["fitness"],
                        "sharpe_ratio": m["sharpe_ratio"],
                        "max_drawdown": m["max_drawdown"],
                        "cumulative_return": m["cumulative_return"],
                        "annualized_return": m["annualized_return"],
                        "profit_factor": m["profit_factor"],
                        "trades": m["trades"],
                    }
                    records.append(rec)
                    if m["fitness"] > best_score:
                        best_score, best_combo = m["fitness"], rec

            df_summary = pd.DataFrame(records).sort_values(by="objective_score", ascending=False).reset_index(drop=True)
            best_agent = OrchestratorAgent(
                weights=best_combo["weights"],
                threshold_buy=best_combo["threshold"],
                threshold_sell=best_combo["threshold"],
                name=f"Orchestrator_{asset_name.upper()}",
            )

        self.best_params = best_combo
        self.best_score = best_combo["objective_score"]
        self.in_sample_metrics = best_combo

        print(
            f"\n{'-' * 88}\n  ORCHESTRATORE OTTIMIZZATO ({method.upper()}) | Sharpe IS: {best_combo['sharpe_ratio']:>5.2f} | CumRet: {best_combo['cumulative_return'] * 100:>+6.1f}% | MDD: {best_combo['max_drawdown'] * 100:>4.1f}%\n{'-' * 88}")
        for n, w_val in best_combo["weights"].items():
            a_val = best_combo.get("alphas", {}).get(n, 1.0)
            print(f"  - {n:<24}: Peso={w_val * 100:5.1f}% | Alpha={a_val:.2f}")
        print(f"  - Soglia Decisionale : {best_combo.get('threshold_long', best_combo.get('threshold')):.2f}")

        return best_agent, df_summary

    def save_final_config(self, filepath: Union[str, Path]) -> None:
        """Save the optimal configuration in JSON format."""
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
# SECTION 8: WALK-FORWARD COMPATIBILITY WRAPPER
# =====================================================================
class WalkForwardOrchestrator:
    """Wrapper around WalkForwardEngine kept for backward compatibility."""

    def __init__(self, optimizer: OrchestratorOptimizer, train_window_years: int = 5, test_window_years: int = 1):
        self.optimizer = optimizer
        self.train_window_years = train_window_years
        self.test_window_years = test_window_years
        self.fold_results: List[Dict[str, Any]] = []

    def run(self, signals_dict: Dict[str, pd.DataFrame],
            returns_or_prices: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Run the walk-forward validation and return the per-fold results."""
        engine = WalkForwardEngine(
            n_agents=len(signals_dict),
            train_window_weeks=self.train_window_years * 52,
            test_window_weeks=self.test_window_years * 52,
            step_weeks=self.test_window_years * 52,
            population_size=self.optimizer.population_size,
            n_generations=self.optimizer.n_generations,
            random_seed=self.optimizer.random_seed,
            verbose=True,
        )
        res = engine.run(signals_dict, returns_or_prices)
        self.fold_results = res["folds_df"].to_dict("records")
        return res["folds_df"]


# =====================================================================
# SECTION 9: SYNTHETIC SIMULATION AND ENTRY POINT
# =====================================================================
def generate_synthetic_market_data(
        T: int = 520,
        random_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Generate 520 weeks with multiple regimes (bull, bear, choppy) and 4 realistic agents.
    """
    rng = np.random.default_rng(random_seed)
    returns = np.zeros(T, dtype=float)
    regimes = []

    lengths = [int(round(f * T)) for f in [0.30, 0.10, 0.20, 0.20, 0.10, 0.10]]
    lengths[-1] = T - sum(lengths[:-1])

    cur = 0
    returns[cur: cur + lengths[0]] = rng.normal(0.12 / 52, 0.14 / np.sqrt(52), size=lengths[0])
    regimes.extend(["Bull_1"] * lengths[0])
    cur += lengths[0]

    returns[cur: cur + lengths[1]] = rng.normal(-0.25 / 52, 0.28 / np.sqrt(52), size=lengths[1])
    regimes.extend(["Bear_1"] * lengths[1])
    cur += lengths[1]

    for t in range(cur, cur + lengths[2]):
        returns[t] = -0.35 * (returns[t - 1] if t > cur else 0.0) + rng.normal(0.0005, 0.15 / np.sqrt(52))
    regimes.extend(["Choppy_1"] * lengths[2])
    cur += lengths[2]

    returns[cur: cur + lengths[3]] = rng.normal(0.20 / 52, 0.15 / np.sqrt(52), size=lengths[3])
    regimes.extend(["Bull_2"] * lengths[3])
    cur += lengths[3]

    for t in range(cur, cur + lengths[4]):
        returns[t] = -0.30 * (returns[t - 1] if t > cur else 0.0) + rng.normal(-0.001, 0.22 / np.sqrt(52))
    regimes.extend(["Choppy_2"] * lengths[4])
    cur += lengths[4]

    returns[cur: cur + lengths[5]] = rng.normal(0.09 / 52, 0.15 / np.sqrt(52), size=lengths[5])
    regimes.extend(["Bull_3"] * lengths[5])

    prices = 100.0 * np.cumprod(1.0 + returns)
    dates = pd.date_range("2014-01-03", periods=T, freq="W-FRI")
    df_market = pd.DataFrame({"close": prices, "return_1w": returns, "regime": regimes}, index=dates)

    diff = (df_market["close"].ewm(span=8).mean() - df_market["close"].ewm(span=24).mean()) / df_market["close"]
    s_trend = np.where(diff > 0.004, 1, np.where(diff < -0.004, -1, 0))
    c_trend = np.clip(np.abs(diff) * 16.0, 0.35, 0.95)

    z = (df_market["close"] - df_market["close"].rolling(16, min_periods=1).mean()) / (
                df_market["close"].rolling(16, min_periods=1).std().fillna(1.0) + 1e-6)
    s_mr = np.where(z < -1.15, 1, np.where(z > 1.15, -1, 0))
    c_mr = np.clip(np.abs(z) / 2.5, 0.40, 0.95)

    s_macro = np.sign(df_market["return_1w"].rolling(26, min_periods=1).mean()).fillna(0).to_numpy(dtype=int, copy=True)
    flip_mask = rng.random(T) < 0.35
    s_macro[flip_mask] = rng.choice([-1, 0, 1], size=int(np.sum(flip_mask)))
    c_macro = rng.uniform(0.80, 0.98, size=T)

    s_noise = rng.choice([-1, 0, 1], p=[0.35, 0.30, 0.35], size=T)
    c_noise = rng.uniform(0.15, 0.85, size=T)

    signals_pool = {
        "Trend_Follower": pd.DataFrame({"signal": s_trend, "confidence": c_trend}, index=dates),
        "Mean_Reverting": pd.DataFrame({"signal": s_mr, "confidence": c_mr}, index=dates),
        "Sentiment_Macro": pd.DataFrame({"signal": s_macro, "confidence": c_macro}, index=dates),
        "FinBERT_Agent": pd.DataFrame({"signal": s_noise, "confidence": c_noise}, index=dates),  # Noise agent used to test the FinBERT weight cap.
    }
    return df_market, signals_pool


def run_end_to_end_simulation() -> None:
    """Run the full walk-forward simulation on synthetic data and print the quantitative report.
    """
    print("\n" + "#" * 94)
    print("### SIMULAZIONE END-TO-END META-DECISORE GENETICO (WEEKLY TIMEFRAME - 10 ANNI) ###")
    print("#" * 94)

    df_market, pool = generate_synthetic_market_data(T=520, random_seed=42)
    engine = WalkForwardEngine(n_agents=len(pool), train_window_weeks=156, test_window_weeks=52, step_weeks=52,
                               verbose=True)
    res = engine.run(pool, df_market["return_1w"])

    df_folds, summary, names = res["folds_df"], res["summary_metrics"], res["agent_names"]

    print("\n" + "=" * 94)
    print("  REPORT COMPARATIVO: IN-SAMPLE (IS) vs OUT-OF-SAMPLE (OOS)")
    print("=" * 94)
    print(f"{'Metrica Finanziaria':<36} | {'In-Sample (Media)':<24} | {'Out-of-Sample (Cucito)':<24}")
    print("-" * 94)
    print(
        f"{'Sharpe Ratio Netto Annualizzato':<36} | {summary['mean_is_sharpe']:>24.2f} | {summary['annualized_sharpe_ratio']:>24.2f}")
    print(f"{'Rendimento Cumulato Netto Totale':<36} | {'-':>24} | {summary['cumulative_net_return'] * 100:>23.1f}%")
    print(f"{'Rendimento Netto Annualizzato':<36} | {'-':>24} | {summary['annualized_net_return'] * 100:>23.1f}%")
    print(f"{'Volatilità Netta Annualizzata':<36} | {'-':>24} | {summary['annualized_volatility'] * 100:>23.1f}%")
    print(
        f"{'Maximum Drawdown (MDD)':<36} | {df_folds['is_max_dd'].mean() * 100:>23.1f}% | {summary['max_drawdown'] * 100:>23.1f}%")
    print(f"{'Profit Factor Netto':<36} | {'-':>24} | {summary['profit_factor']:>24.2f}")
    print(f"{'Win Rate Settimanale (% > 0)':<36} | {'-':>24} | {summary['win_rate'] * 100:>23.1f}%")
    print(f"{'Turnover Medio Settimanale (|Delta P|)':<36} | {'-':>24} | {summary['average_turnover']:>24.4f}")
    print(
        f"{'Percentuale Tempo Attivo a Mercato':<36} | {df_folds['is_active_ratio'].mean() * 100:>23.1f}% | {summary['active_time_ratio'] * 100:>23.1f}%")
    print(f"{'Settimane Totali Valutate OOS':<36} | {'-':>24} | {summary['total_oos_weeks']:>24d}")
    print("=" * 94)

    print("\n" + "-" * 94 + "\n  EVOLUZIONE DEI PESI STRUTTURALI (w_i %) ATTRAVERSO I FOLD\n" + "-" * 94)
    print(f"{'Fold':<6} | {'Periodo OOS Test':<23} | " + " | ".join([f"{n[:14]:<14}" for n in names]))
    print("-" * 94)
    for _, r in df_folds.iterrows():
        print(f"{int(r['fold']):<6} | {r['test_dates']:<23} | " + " | ".join(
            [f"{r[f'w_{n}'] * 100:>13.1f}%" for n in names]))
    print("-" * 94)


if __name__ == "__main__":
    run_end_to_end_simulation()