"""Identification of macro-financial regimes with a Gaussian Mixture Model (GMM).

Preparatory work for Level B. The module does NOT modify A1/A2 weights and is
not used directly by any strategy: it produces regime labels and diagnostics
that Level B can use as contextual information.

Methodological choices (recorded in the decision log; configuration frozen):

- variables are built only from information available at the weekly
  decision date:
    * 13-week change of the 2Y Treasury yield and of the 10Y-2Y curve;
    * 13-week volatility of weekly changes in the 10Y yield;
    * 3-year (156-week) z-score of the 2Y yield and of the 10Y-2Y curve;
- raw levels are not used, since they identify historical eras rather than
  recurring regimes;
- a 3-cluster GMM with diagonal covariance is fitted ONCE on the training
  period and then applied to subsequent weeks;
- clusters are numbered in ascending order of the mean 2Y change, so that
  labels remain stable;
- weeks less likely than the most unusual 2% of the training period are
  flagged as "unrecognised" (known = False).

Output:
    data/signals/B/regime_clusters_weekly.csv
    data/results/B/regime_clustering/cluster_summary.csv
    data/results/B/regime_clustering/risk_by_cluster_train.csv
    data/results/B/regime_clustering/cluster_timeline.png

Run from the repository root:
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

# Categorical colours in a fixed order (one colour per cluster).
CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
UNKNOWN_COLOR = "#b0b0b0"


# =============================================================================
# Settings
# =============================================================================

def get_clustering_settings(config: dict) -> dict:
    """Read the clustering settings from ``strategies.B.clustering``."""
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
# Alignment to the decision dates
# (data: src/data_loader.load_weekly_yields, variables: src/features.build_cluster_features)
# =============================================================================


def align_to_decision_dates(
    weekly_features: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Attach to each decision date (last session of the week) the variables of its W-FRI week.
    """
    week_labels = decision_dates.to_period("W-FRI").end_time.normalize()
    aligned = weekly_features.reindex(week_labels)
    aligned.index = decision_dates
    aligned.index.name = "Date"
    return aligned


# =============================================================================
# Model
# =============================================================================

def fit_cluster_model(
    train: pd.DataFrame,
    n_clusters: int,
    covariance_type: str,
    seed: int,
) -> tuple[StandardScaler, GaussianMixture]:
    """Standardise the variables and fit the GMM on training weeks ONLY."""
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
    """Return the probability of each cluster for every week and the most likely cluster.

    Clusters are renumbered in ascending order of the mean of ``order_by``
    (cluster 0 = strongest fall in the 2Y yield), so that labels remain stable.
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
    """Flag the weeks that the model does not recognise.

    Threshold = ``quantile`` quantile of the log-likelihood of the training
    weeks: a week is "unrecognised" if it is less likely than the most unusual
    ``quantile`` share of the training period.
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
    """Fit the GMM on the training period and assign clusters to all supplied weeks."""
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
# Diagnostics
# =============================================================================

def build_cluster_summary(
    features: pd.DataFrame,
    regimes: pd.DataFrame,
    train_end: pd.Timestamp,
) -> pd.DataFrame:
    """Summarise the cluster variables and counts.

    Reports the mean of each variable per cluster (training only), the number
    of weeks per cluster inside and outside the training period, and the
    unrecognised weeks.
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
    """Annualised volatility of each ETF's daily returns in the week AFTER the decision, by cluster.

    TRAINING ONLY: only weeks whose forward window ends by ``train_end`` are
    included. The purpose is to check whether the regimes separate riskier and
    calmer periods without consulting validation or OOS data.
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
    """Plot the most likely cluster week by week; unrecognised weeks are shown in grey.
    """
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
    """Build the variables, fit the GMM and save regimes and diagnostics."""
    settings = get_clustering_settings(config)
    seed = int(config["project"].get("random_seed", 42))
    train_end = get_time_windows(config)["train_end"]

    # Decision calendar: the same used by A1/A2 (last session of the week).
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
