"""
dataset.py

Preparazione del dataset specifico per lo Stage 2 - Forecasting.

Questo modulo parte dal dataset generale features_weekly.csv e costruisce:

1. le trasformazioni specifiche richieste dal forecasting;
2. il target futuro a 4 settimane;
3. il Feature Set v1 utilizzato dai modelli predittivi.

Il modulo NON addestra modelli e NON effettua scaling.
"""

from pathlib import Path

import pandas as pd


# =============================================================================
# Feature Set v1
# =============================================================================

FEATURE_COLUMNS_V1 = [
    # Politica monetaria e livello dei tassi
    "rates__policy_rate",
    "rates__treasury_2y",

    # Forma della curva
    "curve_10y_2y",
    "curve_30y_10y",

    # Dinamica recente dei tassi e della curva
    "treasury_2y_change_4w",
    "curve_10y_2y_change_4w",
    "curve_30y_10y_change_4w",

    # Inflazione osservata
    "cpi_yoy",
    "core_cpi_yoy",
    "cpi_yoy_change_13w",
    "core_cpi_yoy_change_13w",

    # Aspettative di inflazione
    "inflation__breakeven_5y",
    "inflation__breakeven_10y",
    "breakeven_5y_change_4w",
    "breakeven_10y_change_4w",

    # Volatilità dei tassi
    "rate_volatility_13w",
]


FEATURE_COLUMNS_V2 = [
    # Stato dei tassi
    "rates__treasury_2y",

    # Dinamica recente dei tassi
    "treasury_2y_change_4w",

    # Forma principale della curva
    "curve_10y_2y",

    # Inflazione osservata
    "core_cpi_yoy",
    "core_cpi_yoy_change_13w",

    # Inflazione attesa
    "inflation__breakeven_10y",

    # Volatilità dei tassi
    "rate_volatility_13w",
]

def load_feature_dataset(
    config: dict,
    root_dir: Path,
) -> pd.DataFrame:
    """Carica il dataset generale prodotto da feature_engineering.py."""

    path = (
        root_dir
        / config["paths"]["processed_data"]
        / "features_weekly.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {path}"
        )

    df = pd.read_csv(
        path,
        parse_dates=["Date"],
    )

    df = (
        df
        .set_index("Date")
        .sort_index()
    )

    return df

def add_forecasting_features(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Costruisce le trasformazioni temporali specifiche del Forecasting v1.

    Tutte queste feature utilizzano esclusivamente informazioni disponibili
    alla data t. Non viene utilizzata alcuna osservazione futura.
    """

    features = df.copy()

    forecasting_config = config["forecasting"]

    rate_window = int(
        forecasting_config["rate_change_weeks"]
    )

    inflation_window = int(
        forecasting_config["inflation_trend_weeks"]
    )

    # -------------------------------------------------------------------------
    # Dinamica del Treasury 2Y
    # -------------------------------------------------------------------------
    # Misura quanto il rendimento a 2 anni è cambiato nelle ultime 4 settimane.
    features["treasury_2y_change_4w"] = (
        features["rates__treasury_2y"]
        .diff(periods=rate_window)
    )

    # -------------------------------------------------------------------------
    # Dinamica della curva 10Y-2Y
    # -------------------------------------------------------------------------
    features["curve_10y_2y_change_4w"] = (
        features["curve_10y_2y"]
        .diff(periods=rate_window)
    )

    # -------------------------------------------------------------------------
    # Dinamica della parte lunga 30Y-10Y
    # -------------------------------------------------------------------------
    features["curve_30y_10y_change_4w"] = (
        features["curve_30y_10y"]
        .diff(periods=rate_window)
    )

    # -------------------------------------------------------------------------
    # Trend dell'inflazione osservata
    # -------------------------------------------------------------------------
    # Confrontiamo il livello attuale del CPI YoY con quello disponibile
    # circa un trimestre prima.
    features["cpi_yoy_change_13w"] = (
        features["cpi_yoy"]
        .diff(periods=inflation_window)
    )

    features["core_cpi_yoy_change_13w"] = (
        features["core_cpi_yoy"]
        .diff(periods=inflation_window)
    )

    # -------------------------------------------------------------------------
    # Dinamica delle aspettative di inflazione
    # -------------------------------------------------------------------------
    features["breakeven_5y_change_4w"] = (
        features["inflation__breakeven_5y"]
        .diff(periods=rate_window)
    )

    features["breakeven_10y_change_4w"] = (
        features["inflation__breakeven_10y"]
        .diff(periods=rate_window)
    )

    return features

def add_forward_target(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Costruisce il target continuo dello Stage 2:

        rendimento futuro TLT a 4 settimane
        meno
        rendimento futuro SHY a 4 settimane.

    Il target utilizza informazioni future ed è quindi destinato
    esclusivamente all'addestramento e alla valutazione dei modelli.
    """

    dataset = df.copy()

    horizon = int(
        config["forecasting"]["horizon_weeks"]
    )

    target_name = config["forecasting"]["target_name"]

    # -------------------------------------------------------------------------
    # Rendimento composto futuro della lunga duration
    # -------------------------------------------------------------------------
    tlt_forward_return = (
        dataset["TLT_close"].shift(-horizon)
        / dataset["TLT_close"]
        - 1.0
    )

    # -------------------------------------------------------------------------
    # Rendimento composto futuro della breve duration
    # -------------------------------------------------------------------------
    shy_forward_return = (
        dataset["SHY_close"].shift(-horizon)
        / dataset["SHY_close"]
        - 1.0
    )

    # -------------------------------------------------------------------------
    # Premio futuro relativo della duration
    # -------------------------------------------------------------------------
    dataset[target_name] = (
        tlt_forward_return
        - shy_forward_return
    )

    return dataset

def build_forecasting_dataset(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Costruisce il Forecasting Dataset v1."""

    dataset = add_forecasting_features(
        df,
        config,
    )

    dataset = add_forward_target(
        dataset,
        config,
    )

    target_name = config["forecasting"]["target_name"]

    required_columns = (
        FEATURE_COLUMNS_V1
        + [target_name]
    )

    # Verifica esplicita delle colonne richieste.
    missing_columns = [
        column
        for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing forecasting columns: "
            + ", ".join(missing_columns)
        )

    # Manteniamo soltanto le feature previste dal disegno sperimentale
    # e il target futuro.
    forecasting_dataset = (
        dataset[required_columns]
        .dropna()
        .copy()
    )

    return forecasting_dataset

def run_forecasting_dataset(
    config: dict,
    root_dir: Path,
) -> pd.DataFrame:
    """Costruisce e salva il Forecasting Dataset v1."""

    processed_dir = (
        root_dir
        / config["paths"]["processed_data"]
    )

    features = load_feature_dataset(
        config,
        root_dir,
    )

    dataset = build_forecasting_dataset(
        features,
        config,
    )

    output_path = (
        processed_dir
        / "forecasting_dataset.csv"
    )

    dataset.to_csv(output_path)

    print("\n[Forecasting Dataset] Completed successfully")
    print(f"  Dataset:  {output_path}")
    print(f"  Rows:     {len(dataset):,}")
    print(f"  Features: {len(FEATURE_COLUMNS_V1)}")
    print(f"  Start:    {dataset.index.min().date()}")
    print(f"  End:      {dataset.index.max().date()}")

    return dataset

