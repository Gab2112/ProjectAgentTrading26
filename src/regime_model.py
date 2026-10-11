"""
regime_model.py

Baseline deterministico per l'identificazione dei regimi Macro / Rates.

L'obiettivo di questo modulo è trasformare le feature economiche in stati discreti e 
interpretabili, non ancora generare segnali di trading.

Le quattro dimensioni iniziali sono:

1. inflazione: rising / falling;
2. tassi: rising / falling;
3. curva: normal / inverted;
4. volatilità dei tassi: high / low.

Questa rappresentazione costituisce il benchmark economico semplice che potrà
essere confrontato in seguito con clustering, HMM o altri modelli statistici/ML.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd


# =============================================================================
# Lettura delle feature
# =============================================================================

def load_feature_dataset(config: dict, root_dir: Path) -> pd.DataFrame:
    """Carica il dataset settimanale prodotto da feature_engineering.py."""
    path = root_dir / config["paths"]["processed_data"] / "features_weekly.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {path}"
        )

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()

    return df


# =============================================================================
# Parametri del baseline
# =============================================================================

def _get_regime_parameters(config: dict) -> dict:
    """
    Recupera i parametri del baseline dal config.yaml.

    I fallback permettono al modulo di funzionare anche prima di aver aggiunto
    esplicitamente tutti i parametri al file di configurazione.
    """
    regime_config = config.get("regime", {})

    return {
        "inflation_trend_weeks": int(
            regime_config.get("inflation_trend_weeks", 13)
        ),
        "rate_trend_weeks": int(
            regime_config.get("rate_trend_weeks", 13)
        ),
        "volatility_min_history_weeks": int(
            regime_config.get("volatility_min_history_weeks", 52)
        ),
    }


# =============================================================================
# Costruzione degli stati discreti
# =============================================================================

def add_inflation_state(
    df: pd.DataFrame,
    trend_weeks: int,
) -> pd.DataFrame:
    """
    Classifica l'inflazione come rising o falling.

    Confrontiamo il CPI YoY corrente con quello di circa un trimestre prima.
    Non utilizziamo informazioni future: il confronto è esclusivamente backward.
    """
    regimes = df.copy()

    if "cpi_yoy" not in regimes.columns:
        raise KeyError("Missing required feature: cpi_yoy")

    inflation_change = regimes["cpi_yoy"].diff(trend_weeks)

    regimes["inflation_change_regime_window"] = inflation_change

    regimes["inflation_state"] = pd.Series(
        pd.NA,
        index=regimes.index,
        dtype="object",
    )

    regimes.loc[
        inflation_change >= 0,
        "inflation_state",
    ] = "rising"

    regimes.loc[
        inflation_change < 0,
        "inflation_state",
    ] = "falling"

    return regimes


def add_rate_state(
    df: pd.DataFrame,
    trend_weeks: int,
) -> pd.DataFrame:
    """
    Classifica il regime dei tassi usando il Treasury 2Y.

    Il 2Y è utilizzato come proxy sintetica delle aspettative di politica
    monetaria e del tratto breve/intermedio della curva.
    """
    regimes = df.copy()

    source_column = "rates__treasury_2y"

    if source_column not in regimes.columns:
        raise KeyError(f"Missing required feature: {source_column}")

    rate_change = regimes[source_column].diff(trend_weeks)

    regimes["rate_change_regime_window"] = rate_change

    regimes["rate_state"] = pd.Series(
        pd.NA,
        index=regimes.index,
        dtype="object",
    )

    regimes.loc[
        rate_change >= 0,
        "rate_state",
    ] = "rising"

    regimes.loc[
        rate_change < 0,
        "rate_state",
    ] = "falling"

    return regimes


def add_curve_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifica la curva 10Y-2Y come normal oppure inverted.

    La soglia zero ha una chiara interpretazione economica e non viene
    ottimizzata sui dati.
    """
    regimes = df.copy()

    source_column = "curve_10y_2y"

    if source_column not in regimes.columns:
        raise KeyError(f"Missing required feature: {source_column}")

    regimes["curve_state"] = pd.Series(
        pd.NA,
        index=regimes.index,
        dtype="object",
    )

    regimes.loc[
        regimes[source_column] >= 0,
        "curve_state",
    ] = "normal"

    regimes.loc[
        regimes[source_column] < 0,
        "curve_state",
    ] = "inverted"

    return regimes


def add_volatility_state(
    df: pd.DataFrame,
    min_history_weeks: int,
) -> pd.DataFrame:
    """
    Classifica la volatilità dei tassi come high oppure low.

    La soglia è la mediana storica cumulativa calcolata utilizzando solo le 
    osservazioni disponibili fino alla settimana precedente. 
    Lo shift(1) è essenziale: impedisce alla settimana corrente
    di contribuire alla soglia usata per classificare sé stessa.

    In questo modo non utilizziamo la mediana dell'intero campione, che
    introdurrebbe informazione proveniente dal futuro.
    """
    regimes = df.copy()

    source_column = "rate_volatility_13w"

    if source_column not in regimes.columns:
        raise KeyError(f"Missing required feature: {source_column}")

    historical_threshold = (
        regimes[source_column]
        .expanding(min_periods=min_history_weeks)
        .median()
        .shift(1)
    )

    regimes["rate_volatility_threshold"] = historical_threshold

    regimes["volatility_state"] = pd.Series(
        pd.NA,
        index=regimes.index,
        dtype="object",
    )

    valid = (
        regimes[source_column].notna()
        & historical_threshold.notna()
    )

    regimes.loc[
        valid & (regimes[source_column] > historical_threshold),
        "volatility_state",
    ] = "high"

    regimes.loc[
        valid & (regimes[source_column] <= historical_threshold),
        "volatility_state",
    ] = "low"

    return regimes


# =============================================================================
# Regime combinato
# =============================================================================

def build_combined_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combina le quattro dimensioni in un unico regime discreto.

    Esempio:
        rising_rates_rising_curve_inverted_vol_high

    Manteniamo anche le quattro componenti separate, perché saranno utili
    nell'analisi economica e nei modelli successivi.
    """
    regimes = df.copy()

    state_columns = [
        "inflation_state",
        "rate_state",
        "curve_state",
        "volatility_state",
    ]

    valid = regimes[state_columns].notna().all(axis=1)

    regimes["regime_label"] = pd.Series(
        pd.NA,
        index=regimes.index,
        dtype="object",
    )

    regimes.loc[valid, "regime_label"] = (
        "inflation_"
        + regimes.loc[valid, "inflation_state"]
        + "__rates_"
        + regimes.loc[valid, "rate_state"]
        + "__curve_"
        + regimes.loc[valid, "curve_state"]
        + "__vol_"
        + regimes.loc[valid, "volatility_state"]
    )

    return regimes


# =============================================================================
# Pipeline del baseline
# =============================================================================

def build_regime_dataset(
    features: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Applica sequenzialmente le quattro classificazioni del regime."""
    params = _get_regime_parameters(config)

    regimes = add_inflation_state(
        features,
        trend_weeks=params["inflation_trend_weeks"],
    )

    regimes = add_rate_state(
        regimes,
        trend_weeks=params["rate_trend_weeks"],
    )

    regimes = add_curve_state(regimes)

    regimes = add_volatility_state(
        regimes,
        min_history_weeks=params["volatility_min_history_weeks"],
    )

    regimes = build_combined_regime(regimes)

    return regimes


def build_regime_summary(regimes: pd.DataFrame) -> pd.DataFrame:
    """
    Conta quante settimane appartengono a ciascun regime combinato.

    Il report ci aiuterà a verificare se alcuni stati sono troppo rari per
    essere utilizzati in modo affidabile nella fase di costruzione dei segnali.
    """
    valid_regimes = regimes["regime_label"].dropna()

    summary = (
        valid_regimes
        .value_counts()
        .rename_axis("regime_label")
        .reset_index(name="weeks")
    )

    summary["share_pct"] = (
        summary["weeks"] / summary["weeks"].sum() * 100
    )

    return summary


# =============================================================================
# Entry point
# =============================================================================

def run_regime_model(
    config: dict,
    root_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Esegue il baseline deterministico dei regimi e salva gli output.

    Output:
        data/processed/regimes_weekly.csv
        data/processed/regime_summary.csv
    """
    processed_dir = root_dir / config["paths"]["processed_data"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    features = load_feature_dataset(config, root_dir)
    regimes = build_regime_dataset(features, config)
    summary = build_regime_summary(regimes)

    regimes_path = processed_dir / "regimes_weekly.csv"
    summary_path = processed_dir / "regime_summary.csv"

    regimes.to_csv(regimes_path)
    summary.to_csv(summary_path, index=False)

    valid_count = regimes["regime_label"].notna().sum()
    unique_count = regimes["regime_label"].nunique(dropna=True)

    print("\n[Regime Model] Deterministic baseline completed successfully")
    print(f"  Dataset:        {regimes_path}")
    print(f"  Valid weeks:    {valid_count:,}")
    print(f"  Unique regimes: {unique_count}")
    print(f"  Summary:        {summary_path}")

    return regimes, summary
