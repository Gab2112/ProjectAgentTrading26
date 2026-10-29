"""
random_forest_agent.py

Agente quantitativo di Machine Learning basato su Random Forest (Ensemble di Decision Trees).
Eredita da BaseAgent per garantire piena conformità e coerenza architetturale
con gli altri agenti di ProjectAgentTrading26 (SMA/EMA, Z-Score, Macro Yield Curve, FinBERT).

Razionale Quantitativo e Metodologico:
- Risolve il limite delle relazioni puramente lineari estraendo pattern non lineari
  e interazioni tra feature macroeconomiche (pendenza curva, variazioni tassi, inflazione)
  e dinamiche di prezzo (rendimenti ritardati, volatilità).
- Grazie al Bagging (Bootstrap Aggregation) e al Feature Subsampling (max_features='sqrt'),
  riduce drasticamente la varianza e previene l'overfitting, risultando ideale per il
  dataset a frequenza settimanale (campione In-Sample di ~940 osservazioni).
- Genera nativamente probabilità calibrate di rialzo/ribasso (predict_proba), che si
  mappano in modo naturale nella stima di confidenza continua [0.0, 1.0] e nei segnali
  discreti ammessi {-1, 0, 1}.
- Fornisce la Feature Importance (MDI / Gini Importance), garantendo trasparenza
  ed interpretabilità economica (fondamentale per tesi/ricerche quantitative).
"""

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Risoluzione robusta dei percorsi per esecuzione diretta o come modulo
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from src.agents.base_agent import (
        BaseAgent,
        calculate_performance_metrics,
        compute_objective_score,
    )
except ImportError:
    from .base_agent import (
        BaseAgent,
        calculate_performance_metrics,
        compute_objective_score,
    )


# =====================================================================
# FEATURE MACRO E DI MERCATO DI DEFAULT
# =====================================================================
DEFAULT_FEATURE_COLS = [
    "curve_10y_2y",
    "curve_10y_2y_change_1w",
    "curve_10y_3m",
    "curve_10y_3m_change_1w",
    "curve_30y_10y",
    "curve_30y_10y_change_1w",
    "treasury_10y_change_1w",
    "treasury_2y_change_1w",
    "treasury_3m_change_1w",
    "rates__policy_rate",
    "cpi_yoy",
    "breakeven_5y_change_1w",
    "rate_volatility_13w",
]


# =====================================================================
# 1. AGENTE INFERENZIALE RANDOM FOREST
# =====================================================================
class RandomForestAgent(BaseAgent):
    """
    Agente inferenziale basato su Random Forest Classifier.

    Flusso Operativo:
    1. Riceve in input le feature macroeconomiche e di mercato (features_weekly.csv).
    2. Utilizza un ensemble di alberi decisionali poco profondi (shallow trees: max_depth 2-4)
       e foglie ampie (min_samples_leaf >= 15) per prevenire il memorization noise.
    3. Predice a tempo t la probabilità direzionale a t+1: P(Up) = P(R_{t+1} > 0).
    4. Mappa la probabilità nei segnali di trading discreti {-1, 0, 1}:
       - P(Up) >= prob_threshold               -> BUY (+1)
       - P(Up) <= 1.0 - prob_threshold          -> SELL (-1)
       - (1 - prob_threshold) < P < threshold  -> HOLD / FLAT (0)
    5. Calcola la confidenza continua [0.0, 1.0]:
       - Confidenza proporzionale alla distanza dal baricentro neutrale (0.50):
         c = min(1.0, max(0.50, 0.50 + 2.0 * |P(Up) - 0.50|)) se attivo, 0.0 se FLAT.
    """

    def __init__(
        self,
        target_asset: str = "SPY",
        price_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        n_estimators: int = 150,
        max_depth: Optional[int] = 3,
        min_samples_leaf: int = 20,
        prob_threshold: float = 0.52,
        max_features: Union[str, float] = "sqrt",
        random_state: int = 42,
        name: Optional[str] = None,
        output_format: str = "numeric",  # 'numeric' (-1, 0, 1) o 'string' ('SELL', 'HOLD', 'BUY')
    ):
        agent_name = name or f"RandomForestAgent_{target_asset.upper()}"
        super().__init__(
            name=agent_name,
            target_asset=target_asset,
            price_col=price_col,
            output_format=output_format,
        )

        self.target_asset = target_asset.upper()
        self.price_col = price_col or f"{self.target_asset}_close"
        self.feature_cols = list(feature_cols) if feature_cols else list(DEFAULT_FEATURE_COLS)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth) if max_depth is not None else None
        self.min_samples_leaf = int(min_samples_leaf)
        self.prob_threshold = float(prob_threshold)
        self.max_features = max_features
        self.random_state = int(random_state)

        self.model: Optional[RandomForestClassifier] = None
        self.feature_importances: Dict[str, float] = {}
        self.trained_features: List[str] = []

        self.params: Dict[str, Any] = {
            "target_asset": self.target_asset,
            "price_col": self.price_col,
            "feature_cols": self.feature_cols,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "prob_threshold": self.prob_threshold,
            "max_features": self.max_features,
            "random_state": self.random_state,
        }

    def _apply_params(self, params: Dict[str, Any]) -> None:
        """Aggiorna i parametri dell'agente dopo il caricamento da JSON."""
        if "target_asset" in params:
            self.target_asset = str(params["target_asset"]).upper()
        if "price_col" in params:
            self.price_col = str(params["price_col"])
        else:
            self.price_col = f"{self.target_asset}_close"
        if "feature_cols" in params:
            self.feature_cols = list(params["feature_cols"])
        if "n_estimators" in params:
            self.n_estimators = int(params["n_estimators"])
        if "max_depth" in params:
            self.max_depth = int(params["max_depth"]) if params["max_depth"] is not None else None
        if "min_samples_leaf" in params:
            self.min_samples_leaf = int(params["min_samples_leaf"])
        if "prob_threshold" in params:
            self.prob_threshold = float(params["prob_threshold"])
        if "max_features" in params:
            self.max_features = params["max_features"]
        if "random_state" in params:
            self.random_state = int(params["random_state"])
        if "feature_importances" in params:
            self.feature_importances = {
                str(k): float(v) for k, v in params["feature_importances"].items()
            }
        if "trained_features" in params:
            self.trained_features = list(params["trained_features"])

        self.params.update(params)

    def _extract_xy(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Estrae la matrice delle feature X e il target binario direzionale y:
        y_t = 1 se il rendimento a t+1 è positivo (R_{t+1} > 0), altrimenti 0.
        Garantisce assenza totale di look-ahead bias.
        """
        # Verifica ed eventuale arricchimento con feature macro da BaseAgent
        missing_macro = [c for c in self.feature_cols if c not in df.columns]
        if missing_macro:
            try:
                macro_df = self.load_macro_data(decision_dates=df.index)
                for c in macro_df.columns:
                    if c not in df.columns:
                        df[c] = macro_df[c]
            except Exception:
                pass

        price_col_resolved = self.resolve_price_col(df, price_col=self.price_col, target_asset=self.target_asset)

        # Selezione feature disponibili nel DataFrame
        active_features = [col for col in self.feature_cols if col in df.columns]
        ret_col = f"{self.target_asset}_return_1w"
        if ret_col in df.columns and ret_col not in active_features:
            active_features.append(ret_col)

        if not active_features:
            raise ValueError(
                f"Nessuna feature specificata trovata nel DataFrame. Feature richieste: {self.feature_cols}"
            )

        X = df[active_features].copy()

        # Calcolo rendimenti futuri a t+1 (shift(-1) sul rendimento o prezzo)
        price_series = df[price_col_resolved].astype(float)
        future_returns = price_series.pct_change().shift(-1)
        y = (future_returns > 0.0).astype(int)

        return X, y, future_returns

    def fit(
        self,
        df: pd.DataFrame,
        max_year: Optional[int] = 2021,
    ) -> "RandomForestAgent":
        """
        Addestra il modello Random Forest sui dati In-Sample (fino a max_year).
        Rispetta rigorosamente l'OOS lock del progetto (es. dati >= 2022 mai visti in fase di fit).
        """
        X, y, _ = self._extract_xy(df)

        # Filtro In-Sample temporale rigoroso
        if max_year is not None and isinstance(df.index, pd.DatetimeIndex):
            is_mask = df.index.year <= max_year
        else:
            is_mask = pd.Series(True, index=df.index)

        # Rimuove NaN dovuti al warm-up iniziale o all'ultimo target shift(-1)
        valid_mask = is_mask & X.notna().all(axis=1) & y.notna()

        X_train = X.loc[valid_mask]
        y_train = y.loc[valid_mask]

        if len(X_train) < 50:
            raise ValueError(f"Campione di training insufficiente: {len(X_train)} righe valide.")

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            class_weight="balanced",
            n_jobs=-1,
        )

        self.model.fit(X_train, y_train)
        self.trained_features = list(X_train.columns)

        # Estrazione feature importances
        importances = self.model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        self.feature_importances = {
            self.trained_features[i]: round(float(importances[i]), 4)
            for i in sorted_idx
        }

        self.params["feature_importances"] = self.feature_importances
        self.params["trained_features"] = self.trained_features
        return self

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcola la serie delle probabilità di rialzo P(R_{t+1} > 0).
        Gestisce opportunamente i periodi di warm-up iniziale assegnando probabilità neutrale 0.50.
        """
        if self.model is None:
            raise ValueError(
                f"[{self.name}] Modello Random Forest non addestrato. "
                "Esegui prima .fit() o carica i parametri con .load_params()."
            )

        features = self.trained_features if self.trained_features else self.feature_cols
        missing_cols = [c for c in features if c not in df.columns]
        if missing_cols:
            raise KeyError(f"Colonne feature mancanti nel DataFrame per l'inferenza: {missing_cols}")

        X = df[features].copy()
        valid_mask = X.notna().all(axis=1)

        proba_series = pd.Series(0.50, index=df.index, dtype=float)
        if valid_mask.any():
            X_valid = X.loc[valid_mask]
            # Probabilità della classe positiva (1 = Rialzo)
            preds_proba = self.model.predict_proba(X_valid)[:, 1]
            proba_series.loc[valid_mask] = preds_proba

        return proba_series

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcola i segnali di trading discreti {-1, 0, 1} a partire dalle probabilità stimate.
        - P(Up) >= prob_threshold              -> BUY (+1)
        - P(Up) <= 1.0 - prob_threshold         -> SELL (-1)
        - altrimenti                            -> HOLD (0)
        """
        proba = self.predict_proba(df)

        raw_signals = np.zeros(len(df), dtype=int)
        buy_mask = proba >= self.prob_threshold
        sell_mask = proba <= (1.0 - self.prob_threshold)

        raw_signals[buy_mask] = 1
        raw_signals[sell_mask] = -1

        signals = pd.Series(raw_signals, index=df.index, dtype=int)
        return self._format_signals(signals)

    def predict_with_confidence(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calcola sia i segnali operativi discreti {-1, 0, 1} sia la stima di confidenza continua [0.0, 1.0].

        La confidenza è funzione della convinzione probabilistica rispetto alla soglia neutrale:
        - Segnale HOLD (0) -> confidenza 0.0
        - Segnale BUY (+1) o SELL (-1) -> confidenza calibrata tra 0.50 e 1.00:
            c = min(1.0, max(0.50, 0.50 + 2.0 * |P - 0.50|))
        """
        proba = self.predict_proba(df)
        signals = self.predict(df)

        # Distanza dalla neutralità
        distance = (proba - 0.50).abs()
        conf_scaled = (0.50 + 2.0 * distance).clip(0.50, 1.0).round(4)

        conf_values = np.where(signals != 0, conf_scaled, 0.0)
        confidence = pd.Series(conf_values, index=df.index, dtype=float)

        return signals, confidence

    def get_feature_importances(self) -> pd.Series:
        """Restituisce la serie ordinata decrescente della Feature Importance (Gini Importance)."""
        if not self.feature_importances:
            return pd.Series(dtype=float)
        return pd.Series(self.feature_importances).sort_values(ascending=False)

    def save_params(self, filepath: Union[str, Path]) -> None:
        """
        Salva i parametri e le metriche dell'agente in un file JSON,
        e serializza il modello Random Forest addestrato (.joblib) nel medesimo percorso.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Salvataggio del modello scikit-learn in .joblib
        model_filename = path.with_suffix(".joblib")
        if self.model is not None:
            joblib.dump(self.model, model_filename)
            self.params["model_joblib_file"] = model_filename.name

        # Salvataggio metadati e parametri in JSON
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.params, f, indent=4)

    def load_params(self, filepath: Union[str, Path]) -> "RandomForestAgent":
        """
        Carica i parametri ottimali dal file JSON e il modello serializzato (.joblib).
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File parametri non trovato: {path}")

        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)

        self._apply_params(params)

        # Caricamento del modello serializzato .joblib se presente
        joblib_file = path.with_suffix(".joblib")
        if joblib_file.exists():
            self.model = joblib.load(joblib_file)

        return self


# =====================================================================
# 2. OTTIMIZZATORE SU SINGOLA FINESTRA (Train IS + Val OOS)
# =====================================================================
class RandomForestOptimizer:
    """
    Ottimizzatore su singolo blocco temporale per l'agente Random Forest.

    Principi metodologici:
    - Ottimizza i parametri chiave (max_depth, min_samples_leaf, prob_threshold, n_estimators)
      RIGOROSAMENTE sul blocco di Train (In-Sample).
    - Valuta la qualità del segnale tramite compute_objective_score ('profit_oriented' o 'sharpe_mdd').
    - Testa i parametri ottimali In-Sample sul blocco di Validation (Out-Of-Sample) per misurare
      l'overfitting drop (is_sharpe - oos_sharpe).
    """

    def __init__(
        self,
        target_asset: str = "TLT",
        price_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        depth_candidates: List[Optional[int]] = [2, 3, 4],
        min_leaf_candidates: List[int] = [15, 25, 35],
        threshold_candidates: List[float] = [0.51, 0.52, 0.53, 0.54],
        n_estimators_candidates: List[int] = [100, 150],
        objective_metric: str = "profit_oriented",
        random_state: int = 42,
    ):
        self.target_asset = target_asset.upper()
        self.price_col = price_col or f"{self.target_asset}_close"
        self.feature_cols = feature_cols or DEFAULT_FEATURE_COLS
        self.depth_candidates = depth_candidates
        self.min_leaf_candidates = min_leaf_candidates
        self.threshold_candidates = threshold_candidates
        self.n_estimators_candidates = n_estimators_candidates
        self.objective_metric = objective_metric
        self.random_state = int(random_state)

        self.best_params: Optional[Dict[str, Any]] = None
        self.best_model: Optional[RandomForestClassifier] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        warmup_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Esegue la grid search sui soli dati di Train (In-Sample) e valuta la performance
        sia su Train che su Validation (Out-Of-Sample).
        """
        dfs_to_concat = [df for df in [warmup_df, train_df, val_df] if df is not None and not df.empty]
        if not dfs_to_concat:
            raise ValueError("Nessun DataFrame fornito per l'ottimizzazione.")

        full_df = pd.concat(dfs_to_concat)
        full_df = full_df[~full_df.index.duplicated(keep="last")].sort_index()

        train_indices = train_df.index
        val_indices = val_df.index if (val_df is not None and not val_df.empty) else pd.DatetimeIndex([])

        # Rendimenti dell'asset per backtesting rapido con SHIFT(1)
        close_series = full_df[self.price_col].astype(float)
        full_returns = close_series.pct_change().fillna(0.0)

        best_score = -999.0
        best_combo: Optional[Dict[str, Any]] = None
        best_agent: Optional[RandomForestAgent] = None
        best_is_metrics: Dict[str, float] = {}

        # Grid search su iperparametri parsimoniosi
        for depth in self.depth_candidates:
            for min_leaf in self.min_leaf_candidates:
                for n_est in self.n_estimators_candidates:
                    # Addestra l'agente Random Forest sui soli dati di Train
                    agent = RandomForestAgent(
                        target_asset=self.target_asset,
                        price_col=self.price_col,
                        feature_cols=self.feature_cols,
                        n_estimators=n_est,
                        max_depth=depth,
                        min_samples_leaf=min_leaf,
                        random_state=self.random_state,
                    )

                    try:
                        agent.fit(train_df, max_year=None)
                    except Exception:
                        continue

                    # Ottimizza la soglia di probabilità per questo modello addestrato
                    proba_full = agent.predict_proba(full_df)

                    for th in self.threshold_candidates:
                        raw_sig = np.zeros(len(full_df), dtype=int)
                        raw_sig[proba_full >= th] = 1
                        raw_sig[proba_full <= (1.0 - th)] = -1

                        # Regola cardine: shift(1) per evitare look-ahead bias
                        pos_series = pd.Series(raw_sig, index=full_df.index).shift(1)

                        # Valutazione su In-Sample
                        train_pos = pos_series.reindex(train_indices).fillna(0.0)
                        train_ret = train_pos * full_returns.reindex(train_indices).fillna(0.0)

                        is_metrics = calculate_performance_metrics(train_ret)
                        score = compute_objective_score(is_metrics, metric_type=self.objective_metric)

                        if not np.isnan(score) and score > best_score:
                            best_score = score
                            best_combo = {
                                "max_depth": depth,
                                "min_samples_leaf": min_leaf,
                                "n_estimators": n_est,
                                "prob_threshold": th,
                            }
                            best_agent = agent
                            best_agent.prob_threshold = th
                            best_is_metrics = is_metrics

        if best_combo is None or best_agent is None:
            raise RuntimeError(f"[{self.target_asset}] Nessuna combinazione valida trovata in grid search.")

        # Valutazione Out-Of-Sample (Validation)
        oos_metrics = None
        if val_df is not None and not val_df.empty:
            sig_opt = best_agent.predict(full_df)
            pos_opt = sig_opt.shift(1).fillna(0.0)
            val_pos = pos_opt.reindex(val_indices).fillna(0.0)
            val_ret = val_pos * full_returns.reindex(val_indices).fillna(0.0)
            oos_metrics = calculate_performance_metrics(val_ret)

        is_ret_pct = round(best_is_metrics.get("cumulative_return", 0.0) * 100, 2)
        is_sharpe = round(best_is_metrics.get("sharpe_ratio", 0.0), 2)
        is_mdd_pct = round(best_is_metrics.get("max_drawdown", 0.0) * 100, 2)

        oos_ret_pct = round(oos_metrics["cumulative_return"] * 100, 2) if oos_metrics else None
        oos_sharpe = round(oos_metrics["sharpe_ratio"], 2) if oos_metrics else None
        oos_mdd_pct = round(oos_metrics["max_drawdown"] * 100, 2) if oos_metrics else None

        overfitting_drop = (
            round(is_sharpe - oos_sharpe, 2) if (oos_sharpe is not None) else None
        )

        self.best_model = best_agent.model
        self.best_params = {
            "target_asset": self.target_asset,
            "price_col": self.price_col,
            "max_depth": best_combo["max_depth"],
            "min_samples_leaf": best_combo["min_samples_leaf"],
            "n_estimators": best_combo["n_estimators"],
            "prob_threshold": best_combo["prob_threshold"],
            "objective_metric": self.objective_metric,
            "train_score": round(float(best_score), 4),
            "is_return_pct": is_ret_pct,
            "is_sharpe": is_sharpe,
            "is_mdd_pct": is_mdd_pct,
            "oos_return_pct": oos_ret_pct,
            "oos_sharpe": oos_sharpe,
            "oos_mdd_pct": oos_mdd_pct,
            "overfitting_sharpe_drop": overfitting_drop,
            "train_metrics": best_is_metrics,
            "val_metrics": oos_metrics,
            "feature_importances": best_agent.feature_importances,
            "trained_features": best_agent.trained_features,
        }
        return self.best_params

    def save_params(self, filepath: Union[str, Path]) -> None:
        """Salva i parametri ottimali su file JSON."""
        if self.best_params is None:
            raise ValueError("Nessun parametro trovato. Esegui prima .fit().")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.best_params, f, indent=4)


# =====================================================================
# 3. TRAINER WALK-FORWARD (5 Anni Train IS -> 1 Anno Val OOS)
# =====================================================================
class RandomForestWalkForwardTrainer:
    """
    Gestisce il ciclo di Walk-Forward per Random Forest:
    - Finestra mobile: 5 anni di training In-Sample -> 1 anno di test Validation.
    - Evidenzia l'overfitting confrontando In-Sample e Out-Of-Sample.
    - Seleziona la combinazione di iperparametri più frequente (moda statistica).
    - Addestra il modello finale consolidato su tutto l'In-Sample (fino a max_year).
    """

    def __init__(
        self,
        optimizer: RandomForestOptimizer,
        train_window_years: int = 5,
        target_asset: str = "TLT",
    ):
        self.optimizer = optimizer
        self.train_window_years = train_window_years
        self.target_asset = target_asset.upper()
        self.history: List[Dict[str, Any]] = []
        self.selected_params: Optional[Dict[str, Any]] = None
        self.final_agent: Optional[RandomForestAgent] = None

    def run(self, df: pd.DataFrame, max_year: Optional[int] = 2021) -> Dict[str, Any]:
        """Esegue il ciclo completo di Walk-Forward."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("L'indice del DataFrame deve essere di tipo DatetimeIndex.")

        all_years = sorted(df.index.year.unique())
        if max_year is not None:
            all_years = [y for y in all_years if y <= max_year]

        total_needed = self.train_window_years + 1
        if len(all_years) < total_needed:
            raise ValueError(
                f"Anni insufficienti ({len(all_years)}) per la finestra richiesta ({total_needed})."
            )

        print(
            f"\n[{self.target_asset}] >>> Avvio Walk-Forward Random Forest: "
            f"finestra {self.train_window_years} anni Train -> 1 anno Validation"
        )
        print(f"{'Periodo Train':<14} | {'Val':<5} | {'IS Sharpe':<10} | {'OOS Sharpe':<10} | {'Parametri (d, l, th)':<24}")
        print("-" * 75)

        self.history = []

        for i in range(len(all_years) - self.train_window_years):
            train_years = all_years[i : i + self.train_window_years]
            val_year = all_years[i + self.train_window_years]

            train_df = df[df.index.year.isin(train_years)]
            val_df = df[df.index.year == val_year]

            res = self.optimizer.fit(train_df=train_df, val_df=val_df)

            train_label = f"{train_years[0]}-{train_years[-1]}"
            d = res["max_depth"]
            l = res["min_samples_leaf"]
            th = res["prob_threshold"]
            combo_str = f"d={d}, leaf={l}, th={th:.2f}"

            print(
                f"{train_label:<14} | {val_year:<5} | "
                f"{res['is_sharpe']:>10.2f} | {res['oos_sharpe']:>10.2f} | "
                f"{combo_str:<24}"
            )

            self.history.append(
                {
                    "train_years": train_years,
                    "val_year": val_year,
                    "best_params": res,
                    "max_depth": d,
                    "min_samples_leaf": l,
                    "prob_threshold": th,
                    "n_estimators": res["n_estimators"],
                }
            )

        # Selezione dei parametri più frequenti (moda statistica)
        param_tuples = [
            (h["max_depth"], h["min_samples_leaf"], h["prob_threshold"], h["n_estimators"])
            for h in self.history
        ]
        counts = Counter(param_tuples)
        best_tuple, freq = counts.most_common(1)[0]

        sel_depth, sel_leaf, sel_th, sel_nest = best_tuple

        # Addestramento finale consolidato su tutto il periodo In-Sample
        is_df = df[df.index.year <= max_year] if max_year else df
        final_agent = RandomForestAgent(
            target_asset=self.target_asset,
            price_col=self.optimizer.price_col,
            feature_cols=self.optimizer.feature_cols,
            n_estimators=sel_nest,
            max_depth=sel_depth,
            min_samples_leaf=sel_leaf,
            prob_threshold=sel_th,
            random_state=self.optimizer.random_state,
        )
        final_agent.fit(is_df, max_year=max_year)
        self.final_agent = final_agent

        # Calcolo metriche su tutto l'In-Sample
        sig_is = final_agent.predict(is_df)
        pos_is = sig_is.shift(1).fillna(0.0)
        ret_is = is_df[self.optimizer.price_col].pct_change().fillna(0.0)
        final_metrics = calculate_performance_metrics(pos_is * ret_is)

        self.selected_params = {
            "target_asset": self.target_asset,
            "price_col": self.optimizer.price_col,
            "max_depth": sel_depth,
            "min_samples_leaf": sel_leaf,
            "prob_threshold": sel_th,
            "n_estimators": sel_nest,
            "frequency_count": freq,
            "total_windows": len(self.history),
            "max_train_year": max_year,
            "final_in_sample_metrics": final_metrics,
            "feature_importances": final_agent.feature_importances,
            "trained_features": final_agent.trained_features,
        }

        print("-" * 75)
        print(
            f"Parametri finali consolidati: max_depth={sel_depth}, min_samples_leaf={sel_leaf}, "
            f"prob_threshold={sel_th:.2f}, n_estimators={sel_nest} "
            f"(scelti in {freq}/{len(self.history)} finestre)"
        )
        print(f"In-Sample Sharpe finale: {final_metrics['sharpe_ratio']:.2f} | Cum Ret: {final_metrics['cumulative_return']*100:.1f}%\n")

        return self.selected_params

    def save_final_agent_config(self, filepath: Union[str, Path]) -> None:
        """Salva i parametri consolidati e serializza il modello finale."""
        if self.final_agent is None or self.selected_params is None:
            raise ValueError("Nessun agente finale addestrato. Esegui prima .run().")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serializzazione modello con joblib
        joblib_file = path.with_suffix(".joblib")
        joblib.dump(self.final_agent.model, joblib_file)
        self.selected_params["model_joblib_file"] = joblib_file.name

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.selected_params, f, indent=4)

        print(f"[{self.target_asset}] Configurazione salvata in: {path}")
        print(f"[{self.target_asset}] Modello serializzato salvato in: {joblib_file}")
