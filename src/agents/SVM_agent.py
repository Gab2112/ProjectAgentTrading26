"""
SVM_agent.py

Agente di Machine Learning basato su Support Vector Machine (SVM) per il
progetto Multi-Asset (SPY, IEF, GLD, SHY).

Per ogni ETF e per ogni data di decisione settimanale prevede se il prezzo
sarà più alto tra ``horizon_weeks`` settimane (classificazione binaria) e
produce, come gli altri agenti, un segnale {-1, 0, +1} e una confidenza in
[0, 1]. Le impostazioni stanno in ``config.yaml`` (``strategies.C.svm_agent``).

Protezioni metodologiche
------------------------
1. Variabili solo retrospettive (``src/features.build_asset_features``):
   rendimenti e momentum dell'ETF, momentum degli altri ETF e, se richiesto,
   variazioni dei tassi Treasury dal modulo dei regimi. Nessun riempimento
   all'indietro: le righe incomplete non vengono usate né previste.
2. Walk-forward annuale: le previsioni dell'anno Y usano un modello allenato
   solo con settimane la cui etichetta è nota prima dell'inizio di Y
   (purge pari all'orizzonte della previsione). Sono previsioni
   out-of-fold, usate dall'orchestratore.
3. Iperparametri scelti per ogni anno con una validazione incrociata
   temporale interna (TimeSeriesSplit con gap = orizzonte), metrica
   balanced accuracy. Nessuna "moda" calcolata sugli anni da valutare.
4. Lo StandardScaler è stimato dentro la pipeline, solo sul training.
5. Il punteggio è la ``decision_function`` divisa per la sua deviazione
   standard sul training (non centrata), con una banda neutra: evita la
   probabilità di Platt, che stima una calibrazione con fold non temporali,
   e riduce la tendenza a prevedere sempre "rialzo".
6. Il modello finale (per le date successive al training) è allenato solo
   fino al cutoff passato a ``SVMWalkForwardTrainer.run``.

Uso tramite la pipeline centralizzata:
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
# Impostazioni e dati
# =============================================================================

def get_svm_settings(config: dict) -> dict:
    """Legge ``strategies.C.svm_agent`` da config.yaml e controlla le chiavi."""
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
    """Estrae i prezzi di chiusura dei quattro ETF (colonne ``<ETF>_close``)."""
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
    """Variazioni dei tassi del modulo dei regimi, allineate alle date di decisione."""
    # Import locale: regime_clustering imposta il backend grafico di matplotlib.
    from src.regime_clustering import align_to_decision_dates, get_clustering_settings

    weekly = build_cluster_features(
        load_weekly_yields(config, root_dir), get_clustering_settings(config)
    )
    # One-week lag prevents Thursday decisions from using Friday macro releases.
    return align_to_decision_dates(
        weekly[list(MACRO_FEATURE_COLUMNS)].shift(1), decision_dates
    )


# =============================================================================
# Obiettivo da prevedere
# =============================================================================


def build_direction_target(weekly_close: pd.Series, horizon: int) -> pd.Series:
    """
    Etichetta binaria: 1 se il prezzo tra `horizon` decisioni è più alto
    di quello attuale, 0 altrimenti. NaN dove il futuro non è disponibile.

    L'etichetta usa informazioni FUTURE: serve solo per allenare/valutare.
    """
    future_return = weekly_close.shift(-horizon) / weekly_close - 1.0
    label = (future_return > 0).astype(float)
    return label.where(future_return.notna())


# =============================================================================
# Modello
# =============================================================================

@dataclass(frozen=True)
class SVMCandidate:
    kernel: str
    C: float
    gamma: Any = "scale"
    class_weight: Optional[str] = None

    def build(self) -> Pipeline:
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
        return {"kernel": self.kernel, "C": self.C, "gamma": self.gamma}


def build_candidates(settings: dict) -> List[SVMCandidate]:
    """Griglia piccola e fissata nel config (prima i modelli più semplici)."""
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
    """
    Sceglie l'iperparametro con la balanced accuracy media più alta in una
    validazione incrociata TEMPORALE (ogni fold valida solo su date successive
    a quelle di allenamento, con un gap pari all'orizzonte).
    A parità vince il candidato elencato prima (più semplice).
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
    """
    Converte il punteggio standardizzato in segnale e confidenza.

    |z| <= neutral_band          -> 0 (incertezza)
    z >  neutral_band            -> +1
    z < -neutral_band            -> -1 (oppure 0 se long_only)
    confidenza = tanh(|z|) per i segnali attivi, 0 per i segnali nulli.
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
    """
    Previsioni walk-forward anno per anno.

    Per l'anno Y:
      - campioni utilizzabili: righe con variabili complete e etichetta
        conosciuta all'inizio di Y (posizione + orizzonte <= prima decisione di Y);
      - scelta dell'iperparametro con TimeSeriesSplit sui soli campioni passati;
      - riallenamento sul passato e previsione di tutte le settimane di Y;
      - il punteggio è diviso per la deviazione standard dei punteggi del
        training (nessuna informazione dell'anno Y). Non si sottrae la media:
        il segno del punteggio resta quello della classe prevista dalla SVM.

    Restituisce (previsioni settimanali, storico annuale delle scelte).
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
# Agente (interfaccia comune BaseAgent)
# =============================================================================

class SVMAgent(BaseAgent):
    """
    Agente SVM per un singolo ETF.

    - ``fit_walk_forward``: previsioni out-of-fold anno per anno (training);
    - ``fit_final``: modello unico allenato su tutto il training, salvato con
      ``save_params`` e usato da ``predict`` sulle date successive.

    ``predict`` usa il modello finale se presente, altrimenti le previsioni
    walk-forward già calcolate per le date richieste.
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
    # Variabili
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
        return build_asset_features(weekly_close, self.target_asset, macro_features)

    # ------------------------------------------------------------------
    # Allenamento
    # ------------------------------------------------------------------
    def fit_walk_forward(
        self,
        weekly_close: pd.DataFrame,
        macro_features: Optional[pd.DataFrame],
        first_year: int,
        last_year: int,
    ) -> "SVMAgent":
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
        """Allena il modello finale su tutte le settimane con etichetta nota."""
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
    # Previsione
    # ------------------------------------------------------------------
    def predict_score(self, df: pd.DataFrame) -> pd.Series:
        """Punteggio scalato: decision_function / deviazione standard sul training."""
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
        signals, _ = self.predict_with_confidence(df)
        return signals

    def predict_with_confidence(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        signals, confidence = scores_to_signals(
            self.predict_score(df),
            float(self.settings["neutral_band"]),
            bool(self.settings["long_only"]),
        )
        return self._format_signals(signals), confidence

    # ------------------------------------------------------------------
    # Salvataggio
    # ------------------------------------------------------------------
    def save_params(self, filepath: Union[str, Path]) -> None:
        """Parametri in JSON e modello finale in un file .joblib accanto."""
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
# Scelta degli iperparametri e walk-forward (interfaccia usata da train.py)
# =============================================================================

class SVMOptimizer:
    """Griglia di iperparametri e scelta con validazione incrociata temporale."""

    def __init__(self, settings: dict):
        self.settings = dict(settings)
        self.candidates = build_candidates(self.settings)

    def select(self, X: pd.DataFrame, y: pd.Series) -> Tuple[SVMCandidate, float]:
        return select_candidate(
            X, y, self.candidates, int(self.settings["cv_splits"]),
            gap=int(self.settings["horizon_weeks"]),
        )


class SVMWalkForwardTrainer:
    """
    Produce i segnali out-of-fold (``oof_signals``) e il modello finale.

    ``run(df, max_year)``: ``df`` è il pannello settimanale di training con le
    colonne ``<ETF>_close``; nessuna riga successiva a ``max_year`` viene usata.
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
        if self.final_agent is None:
            raise ValueError("Nessun agente addestrato. Esegui prima .run().")
        self.final_agent.save_params(filepath)
