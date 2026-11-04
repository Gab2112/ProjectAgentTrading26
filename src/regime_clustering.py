"""
regime_clustering.py

Preparazione del Level B: identificazione dei regimi macro-finanziari con un
Gaussian Mixture Model (GMM).

Il modulo NON modifica i pesi di A1/A2 e non è ancora usato da alcuna
strategia: produce i regimi e alcune diagnostiche, che il Level B potrà
utilizzare come informazione di contesto.

Scelte metodologiche (registrate nel decision log, configurazione congelata):

- variabili costruite solo con informazioni disponibili alla data di
  decisione settimanale:
    * variazione a 13 settimane del Treasury 2Y e della curva 10Y-2Y;
    * volatilità a 13 settimane delle variazioni settimanali del 10Y;
    * z-score a 3 anni (156 settimane) del 2Y e della curva 10Y-2Y;
- i livelli grezzi non sono usati: identificano epoche storiche, non
  regimi ricorrenti;
- GMM con 3 cluster e covarianza diagonale, allenato UNA SOLA VOLTA sul
  training e poi applicato alle settimane successive;
- cluster numerati in ordine crescente della variazione media del 2Y, così
  le etichette restano stabili;
- settimane meno verosimili del 2% più insolito del training segnalate come
  "non riconosciute" (known = False).

Output:
    data/signals/B/regime_clusters_weekly.csv
    data/results/B/regime_clustering/cluster_summary.csv
    data/results/B/regime_clustering/risk_by_cluster_train.csv
    data/results/B/regime_clustering/cluster_timeline.png

Esecuzione dalla radice del repository:
    python3 -m src.regime_clustering
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_daily_market_data, load_weekly_yields
from src.features import build_cluster_features
from src.time_split import build_weekly_decision_dates, get_time_windows, restrict_to_allowed_period
from src.utils import ROOT_DIR, get_tickers, load_config

# Colori categoriali in ordine fisso (un colore per cluster).
CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
UNKNOWN_COLOR = "#b0b0b0"


# =============================================================================
# Impostazioni
# =============================================================================

def get_clustering_settings(config: dict) -> dict:
    """Legge le impostazioni del clustering da strategies.B.clustering."""
    settings = config.get("strategies", {}).get("B", {}).get("clustering")
    if not isinstance(settings, dict):
        raise KeyError("Sezione 'strategies.B.clustering' mancante nel config.yaml.")

    required = (
        "n_clusters", "covariance_type", "unknown_quantile", "order_by",
        "change_weeks", "volatility_weeks", "zscore_weeks", "zscore_min_weeks",
        "features",
    )
    missing = [key for key in required if key not in settings]
    if missing:
        raise KeyError("Impostazioni del clustering mancanti: " + ", ".join(missing))
    if settings["order_by"] not in settings["features"]:
        raise ValueError("'order_by' deve essere una delle variabili del clustering.")
    if not 0 < float(settings["unknown_quantile"]) < 0.5:
        raise ValueError("'unknown_quantile' deve essere compreso tra 0 e 0.5.")

    return settings


# =============================================================================
# Allineamento alle date di decisione
# (dati: src/data_loader.load_weekly_yields, variabili: src/features.build_cluster_features)
# =============================================================================


def align_to_decision_dates(
    weekly_features: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Associa a ogni data di decisione (ultima seduta della settimana) le
    variabili della sua settimana W-FRI.
    """
    week_labels = decision_dates.to_period("W-FRI").end_time.normalize()
    aligned = weekly_features.reindex(week_labels)
    aligned.index = decision_dates
    aligned.index.name = "Date"
    return aligned


# =============================================================================
# Modello
# =============================================================================

def fit_cluster_model(
    train: pd.DataFrame,
    n_clusters: int,
    covariance_type: str,
    seed: int,
) -> tuple[StandardScaler, GaussianMixture]:
    """Standardizza e allena il GMM usando SOLO le settimane di training."""
    scaler = StandardScaler().fit(train)
    model = GaussianMixture(
        n_components=n_clusters,
        covariance_type=covariance_type,
        n_init=5,
        random_state=seed,
    )
    model.fit(scaler.transform(train))
    return scaler, model


def compute_cluster_probabilities(
    data: pd.DataFrame,
    scaler: StandardScaler,
    model: GaussianMixture,
    order_by: str,
) -> pd.DataFrame:
    """
    Probabilità di ogni cluster per ogni settimana e cluster più probabile.

    I cluster sono rinumerati in ordine crescente della media di `order_by`
    (cluster 0 = calo più forte del 2Y): le etichette restano stabili.
    """
    probs = pd.DataFrame(model.predict_proba(scaler.transform(data)), index=data.index)
    order = model.means_[:, list(data.columns).index(order_by)].argsort()
    probs = probs[order]
    probs.columns = [f"p_cluster_{i}" for i in range(len(order))]
    probs["cluster"] = probs.to_numpy().argmax(axis=1)
    return probs


def flag_unknown_weeks(
    data: pd.DataFrame,
    train_index: pd.DatetimeIndex,
    scaler: StandardScaler,
    model: GaussianMixture,
    quantile: float,
) -> pd.DataFrame:
    """
    Segnala le settimane che il modello non riconosce.

    Soglia = quantile `quantile` della log-verosimiglianza delle settimane
    di training: una settimana è "non riconosciuta" se è meno verosimile
    del più insolito `quantile` del periodo di allenamento.
    """
    log_likelihood = pd.Series(model.score_samples(scaler.transform(data)), index=data.index)
    threshold = log_likelihood.loc[train_index].quantile(quantile)
    return pd.DataFrame({"log_likelihood": log_likelihood, "known": log_likelihood >= threshold})


def run_cluster_model(
    features: pd.DataFrame,
    train_end: pd.Timestamp,
    settings: dict,
    seed: int,
) -> pd.DataFrame:
    """Allena il GMM sul training e assegna i cluster a tutte le settimane fornite."""
    columns = list(settings["features"])
    data = features[columns].dropna()
    train = data.loc[data.index <= train_end]
    if len(train) < 10 * int(settings["n_clusters"]):
        raise ValueError("Troppe poche settimane di training per il clustering.")

    scaler, model = fit_cluster_model(
        train, int(settings["n_clusters"]), str(settings["covariance_type"]), seed
    )
    probs = compute_cluster_probabilities(data, scaler, model, str(settings["order_by"]))
    flags = flag_unknown_weeks(data, train.index, scaler, model, float(settings["unknown_quantile"]))
    return probs.join(flags)


# =============================================================================
# Diagnostiche
# =============================================================================

def build_cluster_summary(
    features: pd.DataFrame,
    regimes: pd.DataFrame,
    train_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Media delle variabili per cluster (solo training), settimane per cluster
    dentro e fuori dal training e settimane non riconosciute.
    """
    labeled = features.loc[regimes.index].assign(
        cluster=regimes["cluster"], known=regimes["known"]
    )
    train = labeled.loc[labeled.index <= train_end]
    after = labeled.loc[labeled.index > train_end]

    summary = train.drop(columns="known").groupby("cluster").mean()
    summary["weeks_train"] = train["cluster"].value_counts()
    summary["weeks_after_train"] = after["cluster"].value_counts()
    summary["unknown_after_train"] = after.loc[~after["known"], "cluster"].value_counts()
    summary = summary.fillna({"weeks_after_train": 0, "unknown_after_train": 0})
    summary["share_train_pct"] = summary["weeks_train"] / max(len(train), 1) * 100
    summary["share_after_train_pct"] = summary["weeks_after_train"] / max(len(after), 1) * 100
    return summary.round(4)


def build_risk_summary_train(
    regimes: pd.DataFrame,
    market_daily: pd.DataFrame,
    tickers: list[str],
    train_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Volatilità annualizzata dei rendimenti giornalieri di ogni ETF nella
    settimana SUCCESSIVA alla decisione, per cluster. SOLO TRAINING: entrano
    solo le settimane la cui finestra futura termina entro `train_end`.

    Serve a verificare se i regimi distinguono periodi più o meno rischiosi,
    senza consultare validation o OOS.
    """
    closes = market_daily[[f"{t}_close" for t in tickers]]
    returns = closes.pct_change()
    returns.columns = tickers

    known = regimes.loc[regimes["known"], "cluster"]
    decisions = regimes.index
    rows = []
    for current, following in zip(decisions[:-1], decisions[1:]):
        if following > train_end or current not in known.index:
            continue
        window = returns.loc[(returns.index > current) & (returns.index <= following)]
        window = window.assign(cluster=int(known.loc[current]))
        rows.append(window)

    if not rows:
        raise ValueError("Nessuna settimana di training disponibile per la diagnostica.")

    panel = pd.concat(rows)
    vol = panel.groupby("cluster")[tickers].std() * np.sqrt(252) * 100
    vol.columns = [f"{t}_vol_next_week_pct" for t in tickers]
    vol["daily_obs"] = panel.groupby("cluster").size()
    return vol.round(2)


def plot_cluster_timeline(regimes: pd.DataFrame, train_end: pd.Timestamp, output_path: Path) -> None:
    """Cluster più probabile settimana per settimana; non riconosciute in grigio."""
    clusters = regimes["cluster"]
    known = regimes["known"]
    fig, ax = plt.subplots(figsize=(12, 3))

    for cluster in sorted(clusters.unique()):
        dates = clusters.index[(clusters == cluster) & known]
        ax.scatter(dates, [cluster] * len(dates), s=10,
                   color=CLUSTER_COLORS[int(cluster) % len(CLUSTER_COLORS)],
                   label=f"Cluster {cluster}")
    if (~known).any():
        ax.scatter(clusters.index[~known], clusters[~known], s=10,
                   color=UNKNOWN_COLOR, label="Non riconosciuta")

    ax.axvline(train_end, color="#888888", linewidth=1, linestyle="--")
    ax.text(train_end, ax.get_ylim()[1], "  fine training", va="top", fontsize=8, color="#555555")
    ax.set_yticks(sorted(clusters.unique()))
    ax.set_ylabel("Cluster")
    ax.set_title("Regimi macro-finanziari (GMM, allenato sul solo training)")
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# =============================================================================
# Pipeline
# =============================================================================

def run_regime_clustering(config: dict, root_dir: Path = ROOT_DIR) -> pd.DataFrame:
    """Costruisce le variabili, allena il GMM e salva regimi e diagnostiche."""
    settings = get_clustering_settings(config)
    seed = int(config["project"].get("random_seed", 42))
    train_end = get_time_windows(config)["train_end"]

    # Calendario decisionale: stesso usato da A1/A2 (ultima seduta della settimana).
    market_daily, _ = load_daily_market_data(config, root_dir)
    market_daily = restrict_to_allowed_period(market_daily, config)
    decision_dates = pd.DatetimeIndex(build_weekly_decision_dates(market_daily).index)

    weekly_features = build_cluster_features(load_weekly_yields(config, root_dir), settings)
    features = align_to_decision_dates(weekly_features, decision_dates)
    features = restrict_to_allowed_period(features, config)

    regimes = run_cluster_model(features, train_end, settings, seed)

    signals_dir = root_dir / config["paths"]["signals"] / "B"
    results_dir = root_dir / config["paths"]["results"] / "B" / "regime_clustering"
    signals_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    regimes.to_csv(signals_dir / "regime_clusters_weekly.csv")
    build_cluster_summary(features, regimes, train_end).to_csv(results_dir / "cluster_summary.csv")
    build_risk_summary_train(regimes, market_daily, get_tickers(config), train_end).to_csv(
        results_dir / "risk_by_cluster_train.csv"
    )
    plot_cluster_timeline(regimes, train_end, results_dir / "cluster_timeline.png")

    after = regimes.loc[regimes.index > train_end, "known"]
    print("\n[Regime Clustering] Completato")
    print(f"  Cluster:    {settings['n_clusters']} (GMM {settings['covariance_type']}, "
          f"allenato fino al {train_end.date()})")
    print(f"  Settimane:  {len(regimes):,} dal {regimes.index.min().date()} "
          f"al {regimes.index.max().date()}")
    if len(after):
        print(f"  Non riconosciute dopo il training: {(~after).mean():.0%}")
    print(f"  Regimi:     {signals_dir / 'regime_clusters_weekly.csv'}")
    print(f"  Diagnostica: {results_dir}")
    return regimes


if __name__ == "__main__":
    run_regime_clustering(load_config(), ROOT_DIR)
