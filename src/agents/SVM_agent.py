"""
SVM_agent.py

Agente quantitativo di Machine Learning basato su Support Vector Machine (SVM / Support Vector Classifier).
Eredita da BaseAgent per garantire piena conformità e interoperabilità con gli altri agenti del framework.

========================================================================================
FONDAMENTI TEORICI E RAZIONALE QUANTITATIVO:
========================================================================================
1. Garanzie di Convergenza Globale (Problema Convesso):
   A differenza delle Reti Neurali (superfici di loss non convesse, punti di sella, minimi locali)
   e dei Decision Trees (split greedy euristici sub-ottimali), la SVM risolve un problema
   di Programmazione Quadratica (QP) strettamente convesso con vincoli lineari:
       min_{w, b, xi}  (1/2) ||w||^2 + C * sum_i(xi_i)
       s.t.            y_i (w^T phi(x_i) + b) >= 1 - xi_i,  xi_i >= 0
   Il minimo globale è UNICO ed è teoricamente garantito che l'algoritmo (es. SMO - Sequential
   Minimal Optimization) converga sempre alla soluzione globale ottimale in tempo polinomiale.

2. Structural Risk Minimization (SRM) e Margine Massimo:
   La SVM massimizza il margine geometrico 2 / ||w|| tra le classi, minimizzando la dimensione VC
   (Vapnik-Chervonenkis). Questo offre un limite teorico superiore sull'errore di generalizzazione
   Out-Of-Sample, conferendo una naturale robustezza contro il rumore tipico delle serie settimanali.

3. Kernel RBF (Radial Basis Function) e Spazio di Hilbert (RKHS):
   K(x, x') = exp(-gamma * ||x - x'||^2)
   Proietta implicitamente le feature macro e di prezzo in uno spazio vettoriale a dimensione
   infinita in cui i regimi economici complessi diventano linearmente separabili.

4. Calibrazione di Confidenza (Platt Scaling):
   La distanza algebrica dall'iperpiano di separazione d(x) viene trasformata in probabilità
   calibrata P(Up) tramite scaling logistico sigmoide (Platt Scaling).
   La confidenza continua [0.0, 1.0] riflette la convinzione proporzionale alla distanza
   dal baricentro di indeterminazione (0.50).

5. Standardizzazione Robusta Senza Look-Ahead Bias:
   Poiché la distanza euclidea è sensibile alla scala delle feature, viene impiegato un
   RobustScaler (basato su mediana e range interquartile) addestrato ESCLUSIVAMENTE sui dati
   In-Sample della finestra di training per prevenire qualsiasi data leakage.
"""

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# Risoluzione percorsi per esecuzione autonoma o come package
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.agents.base_agent import BaseAgent
from src.data_loader import load_agent_macro_data
from src.evaluator import calculate_performance_metrics, compute_objective_score

# Feature macro e di mercato standard utilizzate per l'inferenza
DEFAULT_SVM_FEATURE_COLS = [
    "curve_10y_2y",
    "curve_10y_2y_change_1w",
    "curve_10y_3m",
    "curve_10y_3m_change_1w",
    "treasury_10y_change_1w",
    "treasury_2y_change_1w",
    "rates__policy_rate",
    "cpi_yoy",
    "breakeven_5y_change_1w",
    "rate_volatility_13w",
]


# =====================================================================
# 1. AGENTE INFERENZIALE SUPPORT VECTOR MACHINE (SVMAgent)
# =====================================================================
class SVMAgent(BaseAgent):
    """
    Agente inferenziale quantitativo basato su Support Vector Classifier (SVC).
    
    Genera:
    - Segnali discreti: +1 (BUY/LONG), -1 (SELL/SHORT), 0 (HOLD/FLAT)
    - Confidenza continua: c_t in [0.0, 1.0]
    """

    def __init__(
        self,
        target_asset: str = "SPY",
        price_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        C: float = 1.0,
        gamma: Union[str, float] = "scale",
        kernel: str = "rbf",
        prob_threshold: float = 0.52,
        dead_band: float = 0.0,
        scaler_type: str = "robust",  # 'robust' o 'standard'
        random_state: int = 42,
        name: Optional[str] = None,
        output_format: str = "numeric",
    ):
        agent_name = name or f"SVMAgent_{target_asset.upper()}"
        super().__init__(
            name=agent_name,
            target_asset=target_asset,
            price_col=price_col,
            output_format=output_format,
        )

        self.target_asset = target_asset.upper()
        self.price_col = price_col or f"{self.target_asset}_close"
        self.feature_cols = list(feature_cols) if feature_cols else list(DEFAULT_SVM_FEATURE_COLS)
        self.C = float(C)
        self.gamma = gamma if isinstance(gamma, str) else float(gamma)
        self.kernel = str(kernel).lower()
        self.prob_threshold = float(prob_threshold)
        self.dead_band = float(dead_band)
        self.scaler_type = str(scaler_type).lower()
        self.random_state = int(random_state)

        self.model: Optional[SVC] = None
        self.scaler: Optional[Union[RobustScaler, StandardScaler]] = None
        self.trained_features: List[str] = []
        self.support_vector_stats: Dict[str, Any] = {}

        self.params: Dict[str, Any] = {
            "target_asset": self.target_asset,
            "price_col": self.price_col,
            "feature_cols": self.feature_cols,
            "C": self.C,
            "gamma": self.gamma,
            "kernel": self.kernel,
            "prob_threshold": self.prob_threshold,
            "dead_band": self.dead_band,
            "scaler_type": self.scaler_type,
            "random_state": self.random_state,
        }

    def _apply_params(self, params: Dict[str, Any]) -> None:
        """Carica i parametri aggiornati dopo lettura da JSON."""
        super()._apply_params(params)
        if "target_asset" in params:
            self.target_asset = str(params["target_asset"]).upper()
        if "price_col" in params:
            self.price_col = str(params["price_col"])
        if "feature_cols" in params:
            self.feature_cols = list(params["feature_cols"])
        if "C" in params:
            self.C = float(params["C"])
        if "gamma" in params:
            self.gamma = params["gamma"] if isinstance(params["gamma"], str) else float(params["gamma"])
        if "kernel" in params:
            self.kernel = str(params["kernel"]).lower()
        if "prob_threshold" in params:
            self.prob_threshold = float(params["prob_threshold"])
        if "dead_band" in params:
            self.dead_band = float(params["dead_band"])
        if "scaler_type" in params:
            self.scaler_type = str(params["scaler_type"]).lower()
        if "trained_features" in params:
            self.trained_features = list(params["trained_features"])
        if "support_vector_stats" in params:
            self.support_vector_stats = dict(params["support_vector_stats"])

        self.params.update(params)

    def _extract_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Estrae la matrice delle feature X e la variabile target binaria:
        y_t = 1 se il rendimento settimanale futuro (t -> t+1) è positivo, 0 altrimenti.
        Allineamento rigoroso senza look-ahead bias.
        """
        # Risoluzione colonne prezzo
        price_col_resolved = self.resolve_price_col(df, price_col=self.price_col, target_asset=self.target_asset)

        # Arricchimento con feature macro se assenti
        missing_macro = [c for c in self.feature_cols if c not in df.columns]
        if missing_macro:
            try:
                macro_df = load_agent_macro_data(decision_dates=df.index)
                for c in macro_df.columns:
                    if c not in df.columns:
                        df[c] = macro_df[c]
            except Exception:
                pass

        active_features = [col for col in self.feature_cols if col in df.columns]
        ret_col = f"{self.target_asset}_return_1w"
        if ret_col in df.columns and ret_col not in active_features:
            active_features.append(ret_col)

        if not active_features:
            raise ValueError(f"Nessuna feature trovata nel DataFrame tra: {self.feature_cols}")

        if not df.index.is_monotonic_increasing or not df.index.is_unique:
            raise ValueError("Le date delle feature devono essere ordinate e univoche.")
        # Rimosso bfill() perchè potrebbe recuperare valori successivi alla data della previsione
        X = df[active_features].copy().ffill()
        price_series = df[price_col_resolved].astype(float)
        future_returns = price_series.shift(-1).div(price_series).sub(1.0)
        future_returns = future_returns.replace([np.inf, -np.inf], np.nan)
        y = (future_returns > 0.0).astype(float).where(future_returns.notna())

        return X, y, future_returns

    def fit(self, df: pd.DataFrame, max_year: Optional[int]) -> "SVMAgent":
        """
        Addestra il modello SVM con vincolo temporale In-Sample (<= max_year).
        La standardizzazione (scaler) è fittata ESCLUSIVAMENTE sui dati In-Sample.
        """
        if (
            not isinstance(df.index, pd.DatetimeIndex)
            or df.index.hasnans
            or not df.index.is_monotonic_increasing
            or not df.index.is_unique
        ):
            raise ValueError("Indice temporale del training non valido.")

        train_df = (
            df.loc[df.index.year <= max_year].copy()
            if max_year is not None
            else df.copy()
        )

        X, y, _ = self._extract_xy(train_df)

        valid_mask = X.notna().all(axis=1) & y.notna()
        X_train = X.loc[valid_mask]
        y_train = y.loc[valid_mask].astype(int)

        # Esclusione dell'ultimo record privo di forward return (NaN)
        valid_mask = ~y_train.isna() & ~X_train.isna().any(axis=1)
        X_train = X_train.loc[valid_mask]
        y_train = y_train.loc[valid_mask]

        if len(X_train) < 30:
            raise ValueError(f"Campione In-Sample insufficiente ({len(X_train)} righe) per il training SVM.")
        if y_train.nunique() != 2:
            raise ValueError("Il training SVM richiede entrambe le classi del target.")
        self.trained_features = list(X_train.columns)

        # Inizializzazione e fit del Trasformatore di Scala In-Sample
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = RobustScaler()

        X_train_scaled = self.scaler.fit_transform(X_train)

        # Inizializzazione SVC con probabilità calibrate (Platt Scaling)
        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            probability=True,
            random_state=self.random_state,
        )
        self.model.fit(X_train_scaled, y_train)

        # Statistiche sui Support Vector per diagnostica di generalizzazione
        n_sv = int(np.sum(self.model.n_support_))
        sv_ratio = round(n_sv / len(X_train), 4)
        self.support_vector_stats = {
            "n_support_vectors": n_sv,
            "support_vector_ratio": sv_ratio,
            "n_features": len(self.trained_features),
            "n_samples_train": len(X_train),
        }

        self.params["trained_features"] = self.trained_features
        self.params["support_vector_stats"] = self.support_vector_stats
        return self

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcola la probabilità stimata di rialzo P(Up) applicando lo scaler fittato.
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Modello SVM non addestrato. Esegui prima .fit() o .load_params().")

        missing = [c for c in self.trained_features if c not in df.columns]
        if missing:
            raise KeyError(f"Feature mancanti nel DataFrame di inferenza: {missing}")

        X = df[self.trained_features].copy().ffill()
        valid = X.notna().all(axis=1)

        proba = pd.Series(0.50, index=df.index, dtype=float)
        if valid.any():
            X_scaled = self.scaler.transform(X.loc[valid])
            proba.loc[valid] = self.model.predict_proba(X_scaled)[:, 1]

        return proba

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Restituisce i segnali direzionali discreti {-1, 0, 1}:
        - P(Up) >= prob_threshold               -> BUY (+1)
        - P(Up) <= 1.0 - prob_threshold          -> SELL (-1)
        - Altrimenti                            -> HOLD / FLAT (0)
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
        Restituisce congiuntamente (signal, confidence):
        - Segnale FLAT (0) -> confidenza 0.0
        - Segnale ATTIVO (+1 / -1) -> confidenza continua calibrata in [0.50, 1.00]
          proporzionale alla convinzione rispetto al centro neutrale 0.50.
        """
        proba = self.predict_proba(df)
        signals = self.predict(df)

        distance = (proba - 0.50).abs()
        conf_scaled = (0.50 + 2.0 * distance).clip(0.50, 1.0).round(4)
        conf_values = np.where(signals != 0, conf_scaled, 0.0)

        confidence = pd.Series(conf_values, index=df.index, dtype=float)
        return signals, confidence

    def save_params(self, filepath: Union[str, Path]) -> None:
        """Salva i parametri in formato JSON e serializza modello + scaler (.joblib)."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib_path = path.with_suffix(".joblib")
        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "trained_features": self.trained_features,
            "support_vector_stats": self.support_vector_stats,
        }
        joblib.dump(payload, joblib_path)
        self.params["model_joblib_file"] = joblib_path.name

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.params, f, indent=4)

    def load_params(self, filepath: Union[str, Path]) -> "SVMAgent":
        """Carica la configurazione JSON e deserializza modello + scaler (.joblib)."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File parametri non trovato: {path}")

        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)

        self._apply_params(params)

        joblib_path = path.with_suffix(".joblib")
        if joblib_path.exists():
            payload = joblib.load(joblib_path)
            self.model = payload.get("model")
            self.scaler = payload.get("scaler")
            self.trained_features = payload.get("trained_features", [])
            self.support_vector_stats = payload.get("support_vector_stats", {})

        return self


# =====================================================================
# 2. OTTIMIZZATORE SU SINGOLA FINESTRA (Train IS + Val OOS)
# =====================================================================
class SVMOptimizer:
    """
    Ottimizzatore su singolo fold temporale per SVMAgent.
    Esplora la griglia di iperparametri controllando la regolarizzazione C
    e la larghezza del kernel RBF gamma.
    """

    def __init__(
        self,
        target_asset: str = "SPY",
        price_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        C_candidates: List[float] = [0.1, 1.0, 5.0],
        gamma_candidates: List[Union[str, float]] = ["scale", 0.01, 0.05],
        threshold_candidates: List[float] = [0.51, 0.52, 0.53],
        objective_metric: str = "sharpe_mdd",
        random_state: int = 42,
    ):
        self.target_asset = target_asset.upper()
        self.price_col = price_col or f"{self.target_asset}_close"
        self.feature_cols = feature_cols or DEFAULT_SVM_FEATURE_COLS
        self.C_candidates = C_candidates
        self.gamma_candidates = gamma_candidates
        self.threshold_candidates = threshold_candidates
        self.objective_metric = objective_metric
        self.random_state = int(random_state)

        self.best_params: Optional[Dict[str, Any]] = None
        self.best_agent: Optional[SVMAgent] = None
        self.combinations_log: List[Dict[str, Any]] = []

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Esegue la grid search sul blocco di Train e valida sul blocco di Validation."""
        full_df = pd.concat([train_df, val_df]) if val_df is not None and not val_df.empty else train_df
        full_df = full_df[~full_df.index.duplicated(keep="last")].sort_index()

        train_indices = train_df.index
        val_indices = val_df.index if val_df is not None and not val_df.empty else pd.DatetimeIndex([])

        close_series = full_df[self.price_col].astype(float)
        full_returns = close_series.pct_change().fillna(0.0)

        best_score = -999.0
        best_combo: Optional[Dict[str, Any]] = None
        best_agent: Optional[SVMAgent] = None
        best_is_metrics: Dict[str, float] = {}

        for c_val in self.C_candidates:
            for g_val in self.gamma_candidates:
                agent = SVMAgent(
                    target_asset=self.target_asset,
                    price_col=self.price_col,
                    feature_cols=self.feature_cols,
                    C=c_val,
                    gamma=g_val,
                    random_state=self.random_state,
                )
                try:
                    agent.fit(train_df, max_year=None)
                except Exception:
                    continue

                proba = agent.predict_proba(full_df)

                for th in self.threshold_candidates:
                    raw_sig = np.zeros(len(full_df), dtype=int)
                    raw_sig[proba >= th] = 1
                    raw_sig[proba <= (1.0 - th)] = -1

                    pos_series = pd.Series(raw_sig, index=full_df.index, dtype=float).shift(1)
                    train_pos = pos_series.reindex(train_indices).fillna(0.0)
                    train_ret = train_pos * full_returns.reindex(train_indices).fillna(0.0)

                    is_metrics = calculate_performance_metrics(train_ret)
                    score = compute_objective_score(is_metrics, metric_type=self.objective_metric)

                    if score > best_score:
                        best_score = score
                        best_combo = {
                            "C": c_val,
                            "gamma": g_val,
                            "prob_threshold": th,
                        }
                        best_agent = agent
                        best_is_metrics = is_metrics

        if best_agent is None or best_combo is None:
            raise RuntimeError(f"[{self.target_asset}] Nessuna combinazione valida trovata per SVMAgent.")

        # Valutazione Out-Of-Sample congelata
        oos_metrics = None
        if val_df is not None and not val_df.empty:
            best_agent.prob_threshold = best_combo["prob_threshold"]
            sig_opt = best_agent.predict(full_df)
            val_pos = sig_opt.shift(1).reindex(val_indices).fillna(0.0)
            val_ret = val_pos * full_returns.reindex(val_indices).fillna(0.0)
            oos_metrics = calculate_performance_metrics(val_ret)

        is_sharpe = round(best_is_metrics.get("sharpe_ratio", 0.0), 2)
        oos_sharpe = round(oos_metrics.get("sharpe_ratio", 0.0), 2) if oos_metrics else None

        self.best_agent = best_agent
        self.best_params = {
            "target_asset": self.target_asset,
            "price_col": self.price_col,
            "C": best_combo["C"],
            "gamma": best_combo["gamma"],
            "prob_threshold": best_combo["prob_threshold"],
            "objective_metric": self.objective_metric,
            "train_score": round(float(best_score), 4),
            "is_sharpe": is_sharpe,
            "is_return_pct": round(best_is_metrics.get("cumulative_return", 0.0) * 100, 2),
            "is_mdd_pct": round(best_is_metrics.get("max_drawdown", 0.0) * 100, 2),
            "oos_sharpe": oos_sharpe,
            "oos_return_pct": round(oos_metrics.get("cumulative_return", 0.0) * 100, 2) if oos_metrics else None,
            "oos_mdd_pct": round(oos_metrics.get("max_drawdown", 0.0) * 100, 2) if oos_metrics else None,
            "overfitting_sharpe_drop": round(is_sharpe - oos_sharpe, 2) if oos_sharpe is not None else None,
            "support_vector_stats": best_agent.support_vector_stats,
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
class SVMWalkForwardTrainer:
    """
    Gestisce la validazione Walk-Forward rolling (5y Train -> 1y Test)
    e la selezione robusta della combinazione modale per SVMAgent.
    """

    def __init__(
        self,
        optimizer: SVMOptimizer,
        train_window_years: int = 5,
        target_asset: str = "SPY",
    ):
        self.optimizer = optimizer
        self.train_window_years = train_window_years
        self.target_asset = target_asset.upper()
        self.history: List[Dict[str, Any]] = []
        self.selected_params: Optional[Dict[str, Any]] = None
        self.final_agent: Optional[SVMAgent] = None
        self.oof_signals = pd.DataFrame(columns=["signal", "confidence"])

    def run(self, df: pd.DataFrame, max_year: int) -> Dict[str, Any]:
        """Esegue il ciclo completo di Walk-Forward."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("L'indice del DataFrame deve essere di tipo DatetimeIndex.")

        all_years = sorted(df.index.year.unique())
        if max_year is not None:
            all_years = [y for y in all_years if y <= max_year]

        total_needed = self.train_window_years + 1
        if len(all_years) < total_needed:
            raise ValueError(f"Anni insufficienti ({len(all_years)}) per la finestra richiesta ({total_needed}).")

        print(f"\n[{self.target_asset}] >>> Avvio Walk-Forward SVMAgent ({self.train_window_years}y Train -> 1y Val)")
        print(f"{'Periodo Train':<14} | {'Val':<5} | {'IS Sharpe':<10} | {'OOS Sharpe':<10} | {'Parametri (C, gamma, th)':<26}")
        print("-" * 75)

        self.history = []
        self.oof_signals = pd.DataFrame(columns=["signal", "confidence"])
        oof_chunks = []

        for i in range(len(all_years) - self.train_window_years):
            train_years = all_years[i : i + self.train_window_years]
            val_year = all_years[i + self.train_window_years]

            train_df = df[df.index.year.isin(train_years)]
            val_df = df[df.index.year == val_year]

            res = self.optimizer.fit(train_df=train_df, val_df=val_df)
            if train_df.index.max() >= val_df.index.min():
                raise ValueError("Fold OOF non cronologico.")

            fold_agent = self.optimizer.best_agent
            if fold_agent is None:
                raise RuntimeError("Modello SVM del fold non disponibile.")

            sig, conf = fold_agent.predict_with_confidence(val_df)
            oof_chunks.append(pd.DataFrame(
                {"signal": sig, "confidence": conf}, index=val_df.index
            ))

            c_val, g_val, th = res["C"], res["gamma"], res["prob_threshold"]
            train_lbl = f"{train_years[0]}-{train_years[-1]}"
            combo_str = f"C={c_val}, g={g_val}, th={th:.2f}"

            print(f"{train_lbl:<14} | {val_year:<5} | {res['is_sharpe']:>10.2f} | {res['oos_sharpe']:>10.2f} | {combo_str:<26}")

            self.history.append({
                "train_years": train_years,
                "val_year": val_year,
                "best_params": res,
                "C": c_val,
                "gamma": g_val,
                "prob_threshold": th,
            })

        self.oof_signals = pd.concat(oof_chunks).sort_index()
        if not self.oof_signals.index.is_unique:
            raise ValueError("Date OOF SVM duplicate.")

        # Selezione dei parametri più frequenti (moda statistica)
        param_tuples = [(h["C"], h["gamma"], h["prob_threshold"]) for h in self.history]
        best_tuple, _ = Counter(param_tuples).most_common(1)[0]
        sel_c, sel_g, sel_th = best_tuple

        # Addestramento consolidato sul periodo In-Sample
        is_df = df[df.index.year <= max_year] if max_year else df
        final_agent = SVMAgent(
            target_asset=self.target_asset,
            price_col=self.optimizer.price_col,
            feature_cols=self.optimizer.feature_cols,
            C=sel_c,
            gamma=sel_g,
            prob_threshold=sel_th,
            random_state=self.optimizer.random_state,
        )
        final_agent.fit(is_df, max_year=max_year)
        self.final_agent = final_agent

        self.selected_params = {
            "target_asset": self.target_asset,
            "C": sel_c,
            "gamma": sel_g,
            "prob_threshold": sel_th,
            "support_vector_stats": final_agent.support_vector_stats,
            "trained_features": final_agent.trained_features,
            "history": self.history,
        }
        return self.selected_params

    def save_final_agent_config(self, filepath: Union[str, Path]) -> None:
        """Salva la configurazione finale consolidata dell'agente."""
        if self.final_agent is None:
            raise ValueError("Nessun agente addestrato. Esegui prima .run().")
        self.final_agent.save_params(filepath)


if __name__ == "__main__":
    print(
        "Utilizzare la pipeline di training centralizzata:\n"
        "python3 src/agents/train.py SPY --train-svm --skip-rf --skip-orch"
    )
