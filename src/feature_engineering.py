"""
feature_engineering.py

Costruzione delle feature settimanali per il progetto.

Questo modulo parte dal dataset processato master_weekly.csv e costruisce
variabili economiche interpretabili per descrivere:

1. andamento degli asset obbligazionari;
2. livello e variazione dei tassi;
3. forma della curva dei rendimenti;
4. inflazione corrente e attesa;
5. volatilità dei tassi.

Le feature vengono costruite utilizzando esclusivamente informazioni disponibili
alla data t. Le variabili target verranno create separatamente nel modulo di 
di forecasting / preparazione del modello.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.data_loader import load_dataset
from src.utils import get_release_lag_days

# =============================================================================
# Lettura del dataset processato
# =============================================================================

def load_master_dataset(config: dict, root_dir: Path) -> pd.DataFrame:
    """Carica il dataset settimanale costruito da data_transformation.py."""
    path = root_dir / config["paths"]["processed_data"] / "master_weekly.csv"

    return load_dataset(path)


def _build_monthly_yoy_feature(
    config: dict,
    root_dir: Path,
    series_id: str,
    feature_name: str,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Calcola il vero YoY a 12 mesi sulla serie mensile originale,
    quindi applica release lag e conversione settimanale.
    """
    path = (
        root_dir
        / config["paths"]["raw_data"]
        / "macro"
        / f"{series_id}.csv"
    )

    monthly = load_dataset(path)

    monthly[series_id] = pd.to_numeric(
        monthly[series_id],
        errors="coerce",
    )

    # Vero confronto con lo stesso mese dell'anno precedente.
    monthly[feature_name] = (
        monthly[series_id]
        .pct_change(periods=12, fill_method=None)
        * 100
    )

    # Il dato viene reso disponibile solo dopo il lag di pubblicazione.
    lag_days = get_release_lag_days(config, series_id)

    monthly.index = (
        monthly.index
        + pd.to_timedelta(lag_days, unit="D")
    )

    # Conversione alla frequenza settimanale.
    weekly = (
        monthly[[feature_name]]
        .resample("W-FRI")
        .last()
        .ffill()
    )

    weekly = weekly.reindex(target_index).ffill()

    return weekly[feature_name]



# =============================================================================
# Feature di mercato
# =============================================================================

def add_market_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Calcola i rendimenti settimanali degli ETF tradabili.

    Il rendimento a una settimana misura la variazione del prezzo di chiusura
    rispetto alla settimana precedente e sarà utile sia come informazione di
    mercato sia, più avanti, per la costruzione del backtest.
    """
    features = df.copy()

    for asset in config["universe"]["traded_assets"].values():
        ticker = asset["ticker"]
        close_column = f"{ticker}_close"

        if close_column not in features.columns:
            raise KeyError(f"Missing market column: {close_column}")

        features[f"{ticker}_return_1w"] = features[close_column].pct_change()

    return features


# =============================================================================
# Feature sui tassi
# =============================================================================

def add_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Costruisce variazioni settimanali dei principali tassi Treasury e del
    policy rate.

    Poiché i rendimenti FRED sono espressi in punti percentuali, una differenza
    di 0.10 equivale a circa 10 basis points.
    """
    features = df.copy()

    rate_columns = {
        "rates__policy_rate": "policy_rate",
        "rates__treasury_3m": "treasury_3m",
        "rates__treasury_2y": "treasury_2y",
        "rates__treasury_5y": "treasury_5y",
        "rates__treasury_10y": "treasury_10y",
        "rates__treasury_30y": "treasury_30y",
    }

    for source_column, feature_name in rate_columns.items():
        if source_column in features.columns:
            features[f"{feature_name}_change_1w"] = (
                features[source_column].diff()
            )

    return features


# =============================================================================
# Feature sulla curva dei rendimenti
# =============================================================================

def add_yield_curve_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Costruisce gli spread principali della curva Treasury.

    Spread positivo:
        curva normalmente inclinata.

    Spread negativo:
        curva invertita.

    Calcoliamo anche la variazione settimanale dello spread, utile per
    distinguere steepening e flattening.
    """
    features = df.copy()

    spread_definitions = {
        "curve_10y_2y": (
            "rates__treasury_10y",
            "rates__treasury_2y",
        ),
        "curve_10y_3m": (
            "rates__treasury_10y",
            "rates__treasury_3m",
        ),
        "curve_30y_10y": (
            "rates__treasury_30y",
            "rates__treasury_10y",
        ),
    }

    for feature_name, (long_rate, short_rate) in spread_definitions.items():
        if long_rate not in features.columns or short_rate not in features.columns:
            raise KeyError(
                f"Missing yield-curve inputs: {long_rate}, {short_rate}"
            )

        features[feature_name] = (
            features[long_rate] - features[short_rate]
        )

        features[f"{feature_name}_change_1w"] = (
            features[feature_name].diff()
        )

    return features


# =============================================================================
# Feature sull'inflazione
# =============================================================================

def add_inflation_features(
    df: pd.DataFrame,
    config: dict,
    root_dir: Path,
) -> pd.DataFrame:    
    """
    Costruisce indicatori di inflazione osservata e attesa.

    CPI e Core CPI:
        variazione percentuale rispetto a circa 52 settimane prima.

    Breakeven inflation:
        livello corrente e variazione settimanale delle aspettative di
        inflazione implicite nel mercato.

    Il CPI nel master dataset è già stato spostato in avanti tramite release lag,
    quindi questa trasformazione utilizza soltanto dati che assumiamo fossero
    disponibili in quel momento.
    """
    features = df.copy()
    features["cpi_yoy"] = _build_monthly_yoy_feature(
        config=config,
        root_dir=root_dir,
        series_id="CPIAUCSL",
        feature_name="cpi_yoy",
        target_index=features.index,
    )

    features["core_cpi_yoy"] = _build_monthly_yoy_feature(
        config=config,
        root_dir=root_dir,
        series_id="CPILFESL",
        feature_name="core_cpi_yoy",
        target_index=features.index,
    )

    for horizon in ["5y", "10y"]:
        column = f"inflation__breakeven_{horizon}"

        if column in features.columns:
            features[f"breakeven_{horizon}_change_1w"] = (
                features[column].diff()
            )

    return features


# =============================================================================
# Feature sulla volatilità dei tassi
# =============================================================================

def add_rate_volatility_features(
    df: pd.DataFrame,
    window: int = 13,
) -> pd.DataFrame:
    """
    Costruisce una proxy della volatilità dei tassi utilizzando la deviazione
    standard mobile delle variazioni settimanali del Treasury 10Y.

    Una finestra di 13 settimane rappresenta circa un trimestre.

    Questa è una soluzione fallback. Se in seguito utilizzeremo MOVE Index
    o una serie Bloomberg equivalente, questa proxy potrà essere sostituita
    o mantenuta come benchmark.
    """
    features = df.copy()

    source_column = "rates__treasury_10y"

    if source_column not in features.columns:
        raise KeyError(f"Missing required column: {source_column}")

    weekly_change = features[source_column].diff()

    features["rate_volatility_13w"] = (
        weekly_change
        .rolling(window=window, min_periods=window)
        .std()
    )

    return features


# =============================================================================
# Selezione del dataset finale delle feature
# =============================================================================

def build_feature_dataset(
    df: pd.DataFrame,
    config: dict,
    root_dir: Path,
) -> pd.DataFrame:    
    """
    Applica sequenzialmente tutti i blocchi di feature engineering.

    Non vengono create variabili future o target, così da mantenere una chiara
    separazione tra informazione disponibile al tempo t e ciò che il modello
    proverà eventualmente a prevedere (nel caso lo dovessimo estendere).
    """
    features = add_market_features(df, config)
    features = add_rate_features(features)
    features = add_yield_curve_features(features)
    features = add_inflation_features(
        features,
        config,
        root_dir,
    )    
    features = add_rate_volatility_features(features)

    return features


def build_feature_quality_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Crea un report sintetico sulla copertura delle feature costruite."""
    summary = pd.DataFrame(
        {
            "non_null": features.notna().sum(),
            "missing": features.isna().sum(),
            "missing_pct": features.isna().mean() * 100,
            "first_valid": [
                features[column].first_valid_index()
                for column in features.columns
            ],
            "last_valid": [
                features[column].last_valid_index()
                for column in features.columns
            ],
        }
    )

    return summary


# =============================================================================
# Entry point del modulo
# =============================================================================

def run_feature_engineering(
    config: dict,
    root_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Esegue l'intero stage di feature engineering e salva gli output.

    Output:
        data/processed/features_weekly.csv
        data/processed/feature_quality_summary.csv
    """
    processed_dir = root_dir / config["paths"]["processed_data"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    master = load_master_dataset(config, root_dir)
    features = build_feature_dataset(
        master,
        config,
        root_dir,
    )
    quality_summary = build_feature_quality_summary(features)

    features_path = processed_dir / "features_weekly.csv"
    quality_path = processed_dir / "feature_quality_summary.csv"

    features.to_csv(features_path)
    quality_summary.to_csv(quality_path)

    print("\n[Feature Engineering] Completed successfully")
    print(f"  Dataset:        {features_path}")
    print(f"  Rows:           {len(features):,}")
    print(f"  Columns:        {len(features.columns):,}")
    print(f"  Start:          {features.index.min().date()}")
    print(f"  End:            {features.index.max().date()}")
    print(f"  Quality report: {quality_path}")

    return features, quality_summary
