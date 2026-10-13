"""
data_transformation.py

Pulizia, allineamento temporale e trasformazione settimanale dei dati grezzi.

Obbiettivi principali
 1. Leggere i CSV salvati in data/raw;
 2. Pulire date, duplicati e missing values;
 3. Applicare un lag prudenziale ai dati macro mensili (da rivalutare post);
 4. Convertire le serie a frequenza settimanale;
 5. Costruire un unico dataset 'master' in data/processed.

Nota metodologica:
per CPI, Core CPI e FEDFUNDS utilizziamo un lag configurabile che approssima 
la data in cui il dato diventa effettivamente disponibile. Aspetto sicuramente
da rivalutare nelle fasi finali del progetto (analizzare il calendario 
ufficiale delle pubblicazioni, i.e. ALFRED).
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.data_loader import load_dataset

# =============================================================================
# Utility di base
# =============================================================================


def _read_csv_with_date_index(path: Path) -> pd.DataFrame:
    """Legge un CSV, converte la colonna Date in indice temporale e ordina i dati."""
    # Controllo del file, conversione della data, rimozione delle date non
    # valide o duplicate e ordinamento sono gestiti da load_dataset().
    return load_dataset(path)


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Converte tutte le colonne del DataFrame in formato numerico quando possibile."""
    numeric_df = df.copy()

    for column in numeric_df.columns:
        numeric_df[column] = pd.to_numeric(
            numeric_df[column],
            errors="coerce",
        )

    return numeric_df


# =============================================================================
# Dati di mercato
# =============================================================================

def transform_market_data(
    config: dict,
    root_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """
    Pulisce i dati giornalieri degli ETF e li converte a frequenza settimanale.

    Regola di aggregazione:
    - Open   -> primo valore della settimana
    - High   -> massimo della settimana
    - Low    -> minimo della settimana
    - Close  -> ultimo valore della settimana
    - Volume -> somma settimanale

    L'indice settimanale viene etichettato con il venerdì (W-FRI).
    """
    raw_market_dir = root_dir / config["paths"]["raw_data"] / "market"

    assets = config["universe"]["traded_assets"]
    weekly_market: Dict[str, pd.DataFrame] = {}

    print("\n[Data Transformation] Market data")

    for _, asset in assets.items():
        ticker = asset["ticker"]
        path = raw_market_dir / f"{ticker}.csv"

        print(f"  Transforming {ticker}...")

        df = _read_csv_with_date_index(path)
        df = _coerce_numeric(df)

        # Manteniamo solo le colonne effettivamente disponibili nel file.
        aggregation_rules = {}

        if "Open" in df.columns:
            aggregation_rules["Open"] = "first"
        if "High" in df.columns:
            aggregation_rules["High"] = "max"
        if "Low" in df.columns:
            aggregation_rules["Low"] = "min"
        if "Close" in df.columns:
            aggregation_rules["Close"] = "last"
        if "Volume" in df.columns:
            aggregation_rules["Volume"] = "sum"

        if "Close" not in aggregation_rules:
            raise KeyError(f"'Close' column not found for {ticker}.")

        # Una settimana viene rappresentata dall'ultima informazione disponibile
        # entro il venerdì, evitando di usare dati della settimana successiva.
        weekly = df.resample("W-FRI").agg(aggregation_rules)

        # Se in una settimana non esiste alcun prezzo di chiusura, la riga non è utile.
        weekly = weekly.dropna(subset=["Close"])

        # Prefisso del ticker per evitare conflitti quando combiniamo più asset.
        weekly = weekly.rename(
            columns={column: f"{ticker}_{column.lower()}" for column in weekly.columns}
        )

        weekly_market[ticker] = weekly

        print(f"    Weekly rows: {len(weekly):,}")

    return weekly_market


# =============================================================================
# Dati macro / rates
# =============================================================================

def _macro_series_mapping(config: dict) -> Dict[str, str]:
    """Costruisce la mappa nome_logico -> FRED series ID dal config.yaml."""
    mapping: Dict[str, str] = {}

    for name, series_id in config["data"]["rates"].items():
        if series_id:
            mapping[f"rates__{name}"] = series_id

    for name, series_id in config["data"]["inflation"].items():
        if series_id:
            mapping[f"inflation__{name}"] = series_id

    return mapping


def _get_release_lag_days(config: dict, series_id: str) -> int:
    """
    Restituisce il lag prudenziale, in giorni di calendario, per la serie.

    I valori devono essere definiti nel file config.yaml.
    Se la sezione non è presente, vengono usati fallback conservativi.

    Spiegazione intuitiva e semplice di questo lag: dobbiamo evitare il 
    cossiddetto 'look-ahead bias'. I.e. il CPI di gennaio descrive l'inflazione
    relativa al mese di gennaio, ma non è disponibile prima di febbraio (non)
    viene pubblicato in tempo reale (essendo un dato macroeconomico richiede
    del tempo per essere determinto). Se nel backtest noi associamo quel valore
    corrispondente alla data di gennaio, il nostro modello barerebbere perchè 
    avrebbe accesso a un'info che, in quel momento storico, un trader non avrebbe.
    Dunque, introduciamo un lag che permette di spostare temporalmente in avanti
    l'informazione dell'indice CPI. I valori inseriti 45 e 35 giorni sono
    arbitrari, al momento sono dei fallback prudenti. Nelle fasi più avanzate
    del progetto analizzeremo le date di rilascio e renderemo il backtest più 
    rigoroso.

    """
    configured_lags = (
        config.get("data", {})
        .get("release_lags_days", {})
    )

    if series_id in configured_lags:
        return int(configured_lags[series_id])

    # Fallback provvisori e volutamente prudenti.
    default_lags = {
        "CPIAUCSL": 45,
        "CPILFESL": 45,
        "FEDFUNDS": 35,
    }

    return default_lags.get(series_id, 0)


def _is_low_frequency_macro(series_id: str) -> bool:
    """Identifica le serie mensili che richiedono un trattamento point-in-time."""
    return series_id in {
        "CPIAUCSL",
        "CPILFESL",
        "FEDFUNDS",
    }


def transform_macro_data(
    config: dict,
    root_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """
    Pulisce e converte le serie macro/rates alla frequenza settimanale.

    Serie giornaliere:
        ultimo valore disponibile della settimana.

    Serie mensili:
        1. si sposta la data in avanti usando un lag prudenziale;
        2. si converte a frequenza settimanale;
        3. si propaga l'ultimo dato pubblicato fino alla nuova release.

    """
    raw_macro_dir = root_dir / config["paths"]["raw_data"] / "macro"

    mapping = _macro_series_mapping(config)
    weekly_macro: Dict[str, pd.DataFrame] = {}

    print("\n[Data Transformation] Macro / rates data")

    for logical_name, series_id in mapping.items():
        path = raw_macro_dir / f"{series_id}.csv"

        print(f"  Transforming {series_id:<10} ({logical_name})...")

        df = _read_csv_with_date_index(path)
        df = _coerce_numeric(df)

        if series_id not in df.columns:
            # FRED sembrerebbe restituire una sola colonna con il nome della serie.
            if len(df.columns) == 1:
                df.columns = [series_id]
            else:
                raise KeyError(
                    f"Unable to identify value column for {series_id}."
                )

        series = df[[series_id]].copy()

        if _is_low_frequency_macro(series_id):
            lag_days = _get_release_lag_days(config, series_id)

            # Trasformiamo la data dell'osservazione in una data di disponibilità
            # approssimata, così da ridurre il rischio di look-ahead bias.
            series.index = series.index + pd.to_timedelta(lag_days, unit="D")
            series.index.name = "Date"

            # Dopo la release, l'ultimo dato noto resta valido fino alla release successiva.
            weekly = series.resample("W-FRI").last().ffill()

            print(f"    Applied release lag: {lag_days} days")
        else:
            # Per yields e breakevens giornalieri usiamo l'ultimo valore disponibile
            # della settimana. Un piccolo forward fill copre festività / missing sporadici.
            weekly = series.resample("W-FRI").last().ffill(limit=2)

        # Usiamo nomi descrittivi e indipendenti dal codice FRED originale.
        weekly = weekly.rename(columns={series_id: logical_name})

        weekly_macro[logical_name] = weekly

        print(f"    Weekly rows: {len(weekly):,}")

    return weekly_macro


# =============================================================================
# Dataset master
# =============================================================================

def build_master_dataset(
    market_data: Dict[str, pd.DataFrame],
    macro_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Combina mercato e macro in un singolo dataset settimanale.

    Il join è esterno durante la costruzione per non perdere informazioni.
    Successivamente eliminiamo solo le settimane in cui manca almeno un prezzo 
    di chiusura degli asset tradabili, perché senza prezzo non possiamo fare 
    backtest su quell'asset.
    """
    frames = list(market_data.values()) + list(macro_data.values())

    if not frames:
        raise ValueError("No transformed datasets available.")

    master = pd.concat(frames, axis=1, join="outer").sort_index()

    # Le variabili macro mensili mantengono l'ultimo valore noto fino alla
    # pubblicazione della nuova osservazione.
    low_frequency_macro = [
        "rates__policy_rate",
        "inflation__cpi",
        "inflation__core_cpi",
    ]

    available_columns = [
        column
        for column in low_frequency_macro
        if column in master.columns
    ]

    master[available_columns] = master[available_columns].ffill()

    # Individuiamo tutte le colonne di chiusura degli strumenti tradabili.
    close_columns = [
        column
        for column in master.columns
        if column.endswith("_close")
    ]

    if not close_columns:
        raise ValueError("No market close columns found in master dataset.")

    # Manteniamo solo le settimane in cui tutti gli asset hanno un Close disponibile.
    master = master.dropna(subset=close_columns)

    master.index.name = "Date"

    return master


def build_data_quality_summary(master: pd.DataFrame) -> pd.DataFrame:
    """Crea un piccolo report di copertura e missing values per ogni colonna."""
    summary = pd.DataFrame(
        {
            "non_null": master.notna().sum(),
            "missing": master.isna().sum(),
            "missing_pct": master.isna().mean() * 100,
            "first_valid": [
                master[column].first_valid_index()
                for column in master.columns
            ],
            "last_valid": [
                master[column].last_valid_index()
                for column in master.columns
            ],
        }
    )

    return summary


def run_data_transformation(
    config: dict,
    root_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Esegue l'intero stage di trasformazione e salva gli output processati.
    """
    processed_dir = root_dir / config["paths"]["processed_data"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    market_data = transform_market_data(config, root_dir)
    macro_data = transform_macro_data(config, root_dir)

    master = build_master_dataset(
        market_data=market_data,
        macro_data=macro_data,
    )

    quality_summary = build_data_quality_summary(master)

    master_path = processed_dir / "master_weekly.csv"
    quality_path = processed_dir / "data_quality_summary.csv"

    master.to_csv(master_path)
    quality_summary.to_csv(quality_path)

    print("\n[Data Transformation] Completed successfully")
    print(f"  Master dataset: {master_path}")
    print(f"  Rows:           {len(master):,}")
    print(f"  Columns:        {len(master.columns):,}")
    print(f"  Start:          {master.index.min().date()}")
    print(f"  End:            {master.index.max().date()}")
    print(f"  Quality report: {quality_path}")

    return master, quality_summary
