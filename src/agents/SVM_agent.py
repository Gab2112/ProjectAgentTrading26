"""Machine-learning agent based on a Support Vector Machine (SVM) for the
multi-asset project (SPY, IEF, GLD, SHY).

For each ETF and each weekly decision date, the agent predicts whether the
price will be higher in ``horizon_weeks`` weeks (binary classification) and,
like the other agents, produces a signal in {-1, 0, +1} and a confidence in
[0, 1]. Settings are defined in ``config.yaml`` (``strategies.C.svm_agent``).

Methodological safeguards
-------------------------
1. Backward-looking variables only (``src/features.build_asset_features``):
   ETF returns and momentum, momentum of the other ETFs and, if requested,
   Treasury rate changes from the regime module. No backward filling:
   incomplete rows are neither used nor predicted.
2. Annual walk-forward: predictions for year Y use a model trained only on
   weeks whose label is known before the start of Y (purge equal to the
   forecast horizon). These out-of-fold predictions feed the orchestrator.
3. Hyperparameters are chosen for each year with an internal temporal
   cross-validation (TimeSeriesSplit with gap = horizon), using balanced
   accuracy. No "mode" is computed over the years being evaluated.
4. The StandardScaler is fitted inside the pipeline, on training data only.
5. The score is the ``decision_function`` divided by its standard deviation
   on the training set (not centred), with a neutral band. This avoids Platt
   probabilities, which calibrate on non-temporal folds, and reduces the
   tendency to always predict an up move.
6. The final model (for dates after the training period) is trained only up
   to the cutoff passed to ``SVMWalkForwardTrainer.run``.

Usage through the central training pipeline:
    python3 -m src.agents.train SPY
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.agents.base_agent import BaseAgent
from src.data_loader import load_weekly_yields
from src.features import build_asset_features, build_cluster_features
from src.utils import ASSETS, ROOT_DIR, load_config


MACRO_FEATURE_COLUMNS = (
    "treasury_2y_change_13w",
    "curve_10y_2y_change_13w",
    "rate_volatility_13w",
)


# =============================================================================
# Settings and data
# =============================================================================

def get_svm_settings(config: dict) -> dict:
    """Read ``strategies.C.svm_agent`` from config.yaml and check its keys."""
    settings = config.get("strategies", {}).get("C", {}).get("svm_agent")
    if not isinstance(settings, dict):
        raise KeyError("Sezione 'strategies.C.svm_agent' mancante nel config.yaml.")
    required = (
        "horizon_weeks", "first_test_year", "min_train_weeks", "cv_splits",
        "C_values", "gamma_values", "neutral_band", "long_only", "use_macro_features",
    )
    missing = [key for key in required if key not in settings]
    if missing:
        raise KeyError("Impostazioni dell'agente SVM mancanti: " + ", ".join(missing))
    return dict(settings)


def weekly_closes_from_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the closing prices of the four ETFs (``<ETF>_close`` columns)."""
    if (
        not isinstance(df.index, pd.DatetimeIndex)
        or df.index.hasnans
        or not df.index.is_monotonic_increasing
        or not df.index.is_unique
    ):
        raise ValueError("Indice temporale non valido: date ordinate e univoche richieste.")
    columns = [f"{asset}_close" for asset in ASSETS]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Prezzi mancanti per l'agente SVM: {missing}")
    closes = df[columns].astype(float).copy()
    closes.columns = list(ASSETS)
    return closes


def load_macro_features(
    decision_dates: pd.DatetimeIndex,
    config: dict,
    root_dir: Path,
) -> pd.DataFrame:
    """Rate changes from the regime module, aligned to the decision dates."""
    # Local import: regime_clustering sets the matplotlib backend.
    from src.regime_clustering import align_to_decision_dates, get_clustering_settings

    weekly = build_cluster_features(
        load_weekly_yields(config, root_dir), get_clustering_settings(config)
    )
    # One-week lag prevents Thursday decisions from using Friday macro releases.
    return align_to_decision_dates(
        weekly[list(MACRO_FEATURE_COLUMNS)].shift(1), decision_dates
    )


# =============================================================================
# Prediction target
# =============================================================================


def build_direction_target(weekly_close: pd.Series, horizon: int) -> pd.Series:
    """Binary label: 1 if the price ``horizon`` decisions ahead is higher than the current price, else 0.

    NaN where the future is not available. The label uses FUTURE information
    and is used only for training and evaluation.
    """
    future_return = weekly_close.shift(-horizon) / weekly_close - 1.0
    label = (future_return > 0).astype(float)
    return label.where(future_return.notna())


# =============================================================================
# Model
# =============================================================================

@dataclass(frozen=True)
class SVMCandidate:
    """One SVM hyperparameter configuration from the search grid."""
    kernel: str
    C: float
    gamma: Any = "scale"
    class_weight: Optional[str] = None

    def build(self) -> Pipeline:
        """Build the scikit-learn pipeline (StandardScaler followed by SVC)."""
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svc",
                    SVC(
                        kernel=self.kernel,
                        C=self.C,
                        gamma=self.gamma,
                        class_weight=self.class_weight,
                    ),
                ),
            ]
        )

    def as_dict(self) -> Dict[str, Any]:
        """Return the configuration as a dictionary for logging and serialisation."""
        return {"kernel": self.kernel, "C": self.C, "gamma": self.gamma}


def build_candidates(settings: dict) -> List[SVMCandidate]:
    """Small hyperparameter grid fixed in the configuration (simpler models first)."""
    weight = settings.get("class_weight")
    weight = None if weight in (None, "none") else str(weight)
    candidates = [SVMCandidate("linear", float(c), "scale", weight) for c in settings["C_values"]]
    for c in settings["C_values"]:
        for gamma in settings["gamma_values"]:
            candidates.append(SVMCandidate("rbf", float(c), gamma, weight))
    return candidates


def select_candidate(
    X: pd.DataFrame,
    y: pd.Series,
    candidates: List[SVMCandidate],
    n_splits: int,
    gap: int,
) -> Tuple[SVMCandidate, float]:
    """Select the hyperparameters with the highest mean balanced accuracy.

    The score comes from a TEMPORAL cross-validation in which each fold
    validates only on dates after its training dates, with a gap equal to the
    horizon. Ties are broken in favour of the candidate listed first (the
    simpler one).
    """
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    best, best_score = candidates[0], -np.inf

    for candidate in candidates:
        scores = []
        for train_idx, test_idx in splitter.split(X):
            y_train = y.iloc[train_idx]
            if y_train.nunique() < 2:
                continue
            model = candidate.build().fit(X.iloc[train_idx], y_train)
            scores.append(balanced_accuracy_score(y.iloc[test_idx], model.predict(X.iloc[test_idx])))
        if scores and float(np.mean(scores)) > best_score:
            best, best_score = candidate, float(np.mean(scores))

    return best, best_score


def scores_to_signals(
    z_scores: pd.Series,
    neutral_band: float,
    long_only: bool,
) -> Tuple[pd.Series, pd.Series]:
    """Convert the standardised score into a signal and a confidence.

    |z| <= neutral_band -> 0 (uncertain)
    z >  neutral_band   -> +1
    z < -neutral_band   -> -1 (or 0 if long_only)
    confidence = tanh(|z|) for active signals, 0 for null signals.
    """
    signals = pd.Series(0, index=z_scores.index, dtype=int)
    signals[z_scores > neutral_band] = 1
    signals[z_scores < -neutral_band] = 0 if long_only else -1
    signals[z_scores.isna()] = 0

    confidence = np.tanh(z_scores.abs()).where(signals != 0, 0.0).fillna(0.0)
    return signals, confidence.astype(float)


def walk_forward_svm(
    features: pd.DataFrame,
    label: pd.Series,
    settings: dict,
    first_year: int,
    last_year: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward predictions, year by year.

    For year Y:
      - usable samples: rows with complete variables and a label known at the
        start of Y (position + horizon <= first decision of Y);
      - hyperparameters are chosen with TimeSeriesSplit on past samples only;
      - the model is retrained on the past and predicts every week of Y;
      - the score is divided by the standard deviation of the training scores
        (no information from year Y). The mean is not subtracted, so the sign
        of the score remains that of the class predicted by the SVM.

    Returns (weekly predictions, annual history of the selections).
    """
    horizon = int(settings["horizon_weeks"])
    candidates = build_candidates(settings)

    data = features.copy()
    complete = data.notna().all(axis=1)
    positions = pd.Series(np.arange(len(data)), index=data.index)

    predictions = pd.DataFrame(index=data.index, columns=["score", "z_score"], dtype=float)
    history = []

    for year in range(first_year, last_year + 1):
        in_year = (data.index.year == year) & complete
        if not in_year.any():
            continue

        first_position = positions[data.index.year == year].iloc[0]
        usable = complete & label.notna() & (positions + horizon <= first_position)
        X_past, y_past = data.loc[usable], label.loc[usable].astype(int)

        if len(X_past) < int(settings["min_train_weeks"]) or y_past.nunique() < 2:
            continue

        candidate, cv_score = select_candidate(
            X_past, y_past, candidates, int(settings["cv_splits"]), gap=horizon
        )
        model = candidate.build().fit(X_past, y_past)

        train_scores = model.decision_function(X_past)
        std = float(np.std(train_scores))
        std = std if std > 1e-12 else 1.0

        year_scores = model.decision_function(data.loc[in_year])
        predictions.loc[in_year, "score"] = year_scores
        predictions.loc[in_year, "z_score"] = year_scores / std

        realized = label.loc[in_year]
        known = realized.notna()
        hit_rate = (
            float(((year_scores[known.to_numpy()] > 0).astype(int) == realized[known].astype(int)).mean())
            if known.any() else np.nan
        )
        history.append(
            {
                "year": year,
                **candidate.as_dict(),
                "cv_balanced_accuracy": round(cv_score, 4),
                "train_weeks": int(len(X_past)),
                "train_up_share": round(float(y_past.mean()), 4),
                "year_hit_rate": round(hit_rate, 4) if not np.isnan(hit_rate) else np.nan,
                "year_up_share": round(float(realized[known].mean()), 4) if known.any() else np.nan,
            }
        )

    return predictions, pd.DataFrame(history)


# =============================================================================
# Agent (common BaseAgent interface)
# =============================================================================

class SVMAgent(BaseAgent):
    """SVM agent for a single ETF.

    - ``fit_walk_forward``: out-of-fold predictions, year by year (training);
    - ``fit_final``: a single model trained on the full training period, saved
      with ``save_params`` and used by ``predict`` for later dates.

    ``predict`` uses the final model if available; otherwise it uses the
    walk-forward predictions already computed for the requested dates.
    """

    def __init__(
        self,
        target_asset: str = "SPY",
        settings: Optional[dict] = None,
        output_format: str = "numeric",
        root_dir: Optional[Union[str, Path]] = None,
    ):
        super().__init__(
            name=f"SVMAgent_{target_asset.upper()}",
            target_asset=target_asset,
            output_format=output_format,
        )
        self.target_asset = target_asset.upper()
        self.root_dir = Path(root_dir) if root_dir is not None else ROOT_DIR
        self.settings = dict(settings) if settings is not None else get_svm_settings(
            load_config(self.root_dir / "config.yaml")
        )
        self.predictions: Optional[pd.DataFrame] = None
        self.history: Optional[pd.DataFrame] = None
        self.model: Optional[Pipeline] = None
        self.score_scale: float = 1.0
        self.candidate: Optional[Dict[str, Any]] = None
        self.trained_features: List[str] = []
        self.params = {"target_asset": self.target_asset, **self.settings}

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------
    def _macro_for(self, dates: pd.DatetimeIndex) -> Optional[pd.DataFrame]:
        if not self.settings["use_macro_features"]:
            return None
        config = load_config(self.root_dir / "config.yaml")
        return load_macro_features(dates, config, self.root_dir)

    def build_features(
        self,
        weekly_close: pd.DataFrame,
        macro_features: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Build the backward-looking features of the target ETF."""
        return build_asset_features(weekly_close, self.target_asset, macro_features)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit_walk_forward(
        self,
        weekly_close: pd.DataFrame,
        macro_features: Optional[pd.DataFrame],
        first_year: int,
        last_year: int,
    ) -> "SVMAgent":
        """Compute the out-of-fold walk-forward predictions between ``first_year`` and ``last_year``.
        """
        features = self.build_features(weekly_close, macro_features)
        label = build_direction_target(
            weekly_close[self.target_asset], int(self.settings["horizon_weeks"])
        )
        self.predictions, self.history = walk_forward_svm(
            features, label, self.settings, first_year, last_year
        )
        return self

    def fit_final(
        self,
        weekly_close: pd.DataFrame,
        macro_features: Optional[pd.DataFrame],
    ) -> "SVMAgent":
        """Train the final model on every week with a known label."""
        features = self.build_features(weekly_close, macro_features)
        label = build_direction_target(
            weekly_close[self.target_asset], int(self.settings["horizon_weeks"])
        )
        usable = features.notna().all(axis=1) & label.notna()
        X, y = features.loc[usable], label.loc[usable].astype(int)
        if y.nunique() != 2:
            raise ValueError("Il training SVM richiede entrambe le classi del target.")
        if len(X) < int(self.settings["min_train_weeks"]):
            raise ValueError(f"Campione di training insufficiente ({len(X)} settimane).")

        candidate, cv_score = select_candidate(
            X, y, build_candidates(self.settings),
            int(self.settings["cv_splits"]), gap=int(self.settings["horizon_weeks"]),
        )
        self.model = candidate.build().fit(X, y)
        std = float(np.std(self.model.decision_function(X)))
        self.score_scale = std if std > 1e-12 else 1.0
        self.candidate = {**candidate.as_dict(), "cv_balanced_accuracy": round(cv_score, 4)}
        self.trained_features = list(X.columns)
        self.params.update({
            "final_model": self.candidate,
            "score_scale": self.score_scale,
            "trained_features": self.trained_features,
            "n_samples_train": int(len(X)),
            "train_start": X.index.min().date().isoformat(),
            "train_end": X.index.max().date().isoformat(),
        })
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_score(self, df: pd.DataFrame) -> pd.Series:
        """Scaled score: decision_function divided by its standard deviation on the training set.
        """
        if self.model is not None:
            closes = weekly_closes_from_panel(df)
            features = self.build_features(closes, self._macro_for(closes.index))
            features = features.reindex(columns=self.trained_features)
            valid = features.notna().all(axis=1)
            scores = pd.Series(np.nan, index=df.index, dtype=float)
            if valid.any():
                scores.loc[valid] = self.model.decision_function(features.loc[valid]) / self.score_scale
            return scores
        if self.predictions is not None:
            return self.predictions["z_score"].reindex(df.index)
        raise ValueError(f"[{self.name}] Modello non allenato: eseguire fit_final() o fit_walk_forward().")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return the discrete trading signals {-1, 0, +1}."""
        signals, _ = self.predict_with_confidence(df)
        return signals

    def predict_with_confidence(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Return the trading signals and their confidence."""
        signals, confidence = scores_to_signals(
            self.predict_score(df),
            float(self.settings["neutral_band"]),
            bool(self.settings["long_only"]),
        )
        return self._format_signals(signals), confidence

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_params(self, filepath: Union[str, Path]) -> None:
        """Save the parameters to JSON and the final model to a .joblib file alongside it.
        """
        if self.model is None:
            raise ValueError("Nessun modello finale da salvare: eseguire fit_final().")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib_path = path.with_suffix(".joblib")
        joblib.dump({"model": self.model}, joblib_path)
        self.params["model_joblib_file"] = joblib_path.name
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.params, handle, indent=4, default=str)

    def load_params(self, filepath: Union[str, Path]) -> "SVMAgent":
        """Load the parameters from JSON and the final model from the matching .joblib file.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File parametri non trovato: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            params = json.load(handle)
        self.params = params
        self.settings = {k: params[k] for k in self.settings if k in params}
        self.score_scale = float(params["score_scale"])
        self.candidate = params.get("final_model")
        self.trained_features = list(params["trained_features"])
        self.model = joblib.load(path.with_suffix(".joblib"))["model"]
        return self


# =============================================================================
# Hyperparameter selection and walk-forward (interface used by train.py)
# =============================================================================

class SVMOptimizer:
    """Hyperparameter grid with selection by temporal cross-validation."""

    def __init__(self, settings: dict):
        self.settings = dict(settings)
        self.candidates = build_candidates(self.settings)

    def select(self, X: pd.DataFrame, y: pd.Series) -> Tuple[SVMCandidate, float]:
        """Select the best candidate and return it with its cross-validation score."""
        return select_candidate(
            X, y, self.candidates, int(self.settings["cv_splits"]),
            gap=int(self.settings["horizon_weeks"]),
        )


class SVMWalkForwardTrainer:
    """Produce the out-of-fold signals (``oof_signals``) and the final model.

    ``run(df, max_year)``: ``df`` is the weekly training panel with the
    ``<ETF>_close`` columns; no row after ``max_year`` is used.
    """

    def __init__(
        self,
        target_asset: str = "SPY",
        settings: Optional[dict] = None,
        root_dir: Optional[Union[str, Path]] = None,
    ):
        self.target_asset = target_asset.upper()
        self.root_dir = Path(root_dir) if root_dir is not None else ROOT_DIR
        self.settings = dict(settings) if settings is not None else get_svm_settings(
            load_config(self.root_dir / "config.yaml")
        )
        self.optimizer = SVMOptimizer(self.settings)
        self.history: List[Dict[str, Any]] = []
        self.oof_signals = pd.DataFrame(columns=["signal", "confidence"])
        self.selected_params: Optional[Dict[str, Any]] = None
        self.final_agent: Optional[SVMAgent] = None

    def run(self, df: pd.DataFrame, max_year: int) -> Dict[str, Any]:
        """Run the annual walk-forward, build the out-of-fold signals and train the final model.
        """
        closes = weekly_closes_from_panel(df)
        closes = closes.loc[closes.index.year <= int(max_year)]
        first_year = int(self.settings["first_test_year"])
        if closes.empty or first_year > int(max_year):
            raise ValueError("Periodo insufficiente per il walk-forward SVM.")

        oof_agent = SVMAgent(self.target_asset, self.settings, root_dir=self.root_dir)
        macro = oof_agent._macro_for(closes.index)
        oof_agent.fit_walk_forward(closes, macro, first_year, int(max_year))

        scores = oof_agent.predictions["z_score"]
        scores = scores.loc[(scores.index.year >= first_year) & scores.notna()]
        if scores.empty:
            raise ValueError("Nessuna previsione out-of-fold SVM disponibile.")
        signals, confidence = scores_to_signals(
            scores, float(self.settings["neutral_band"]), bool(self.settings["long_only"]),
        )
        self.oof_signals = pd.DataFrame({"signal": signals, "confidence": confidence})
        self.history = oof_agent.history.to_dict("records")

        final_agent = SVMAgent(self.target_asset, self.settings, root_dir=self.root_dir)
        final_agent.fit_final(closes, macro)
        self.final_agent = final_agent

        counts = self.oof_signals["signal"].value_counts(normalize=True)
        print(f"\n[{self.target_asset}] >>> SVM walk-forward annuale (purge {self.settings['horizon_weeks']} sett.)")
        for record in self.history:
            print(f"  {record['year']}: {record['kernel']:<6} C={record['C']:<5} gamma={record['gamma']!s:<6} "
                  f"CV bal.acc={record['cv_balanced_accuracy']:.3f}  hit rate={record['year_hit_rate']}")
        print(f"  Segnali OOF: Long {counts.get(1, 0):.0%} | Flat {counts.get(0, 0):.0%} | Short {counts.get(-1, 0):.0%}")

        self.selected_params = {
            "target_asset": self.target_asset,
            **final_agent.candidate,
            "neutral_band": float(self.settings["neutral_band"]),
            "oof_start": self.oof_signals.index.min().date().isoformat(),
            "oof_end": self.oof_signals.index.max().date().isoformat(),
            "oof_long_share": round(float(counts.get(1, 0.0)), 4),
            "history": self.history,
        }
        return self.selected_params

    def save_final_agent_config(self, filepath: Union[str, Path]) -> None:
        """Save the parameters and the model of the final agent."""
        if self.final_agent is None:
            raise ValueError("Nessun agente addestrato. Esegui prima .run().")
        self.final_agent.save_params(filepath)
