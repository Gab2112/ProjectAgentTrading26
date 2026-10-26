"""Preparazione dei dati di mercato e macroeconomici per il progetto Multi-Asset.

Responsabilità
---------------
1. Leggere i CSV dei quattro ETF con controlli rigorosi su date e prezzi.
2. Conservare un pannello GIORNALIERO allineato dei prezzi Open/Close rettificati
   per i backtest dei livelli A1 e A2.
3. Conservare un pannello SETTIMANALE per l'analisi e le future feature del livello B.
4. Escludere i dati macroeconomici durante A1/A2, rendendo i benchmark
   indipendenti dalle serie FRED.
5. Escludere le osservazioni OOS dai file elaborati finché non sono autorizzate.

Input:  data/raw/market/<TICKER>.csv; facoltativamente data/raw/macro/*.csv
Output: data/processed/market_daily.csv
        data/processed/weekly_decision_dates.csv
        data/processed/market_daily_quality.csv
        data/processed/master_weekly.csv
        data/processed/data_quality_summary.csv

Nota: i prezzi Open e Close devono essere rettificati in maniera coerente,
come nei download Yahoo con ``auto_adjust=True``. I prezzi di apertura
rettificati sono una proxy per il backtest, non quotazioni storiche
necessariamente eseguibili.

Questo modulo NON genera pesi di portafoglio e NON calcola rendimenti:
queste responsabilità appartengono ai moduli di strategia e backtesting.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.data_loader import load_dataset
from src.time_split import restrict_to_allowed_period
from src.utils import get_release_lag_days, get_tickers


REQUIRED_MARKET_COLUMNS = ("Open", "Close")
LOW_FREQUENCY_SERIES = frozenset({"CPIAUCSL", "CPILFESL", "FEDFUNDS"})


def _read_csv_with_date_index(path: Path) -> pd.DataFrame:
    """Legge un CSV tramite il loader condiviso, con verifica rigorosa delle date."""
    df = load_dataset(path, strict=True)
    if df.empty:
        raise ValueError(f"Empty dataset: {path}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Expected DatetimeIndex: {path}")
    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Converte le colonne in valori numerici; i valori mancanti restano NaN."""
    result = df.copy()
    for col in result.columns:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def _read_etf_prices(path: Path, ticker: str) -> pd.DataFrame:
    """Verifica la presenza e la validità dei prezzi giornalieri Open/Close."""
    df = _read_csv_with_date_index(path)
    missing = [column for column in REQUIRED_MARKET_COLUMNS if column not in df]
    if missing:
        raise KeyError(f"{ticker}: missing required columns {missing} in {path}")

    df = _coerce_numeric(df)
    prices = df.loc[:, list(REQUIRED_MARKET_COLUMNS)]
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{ticker}: missing or non-finite daily Open/Close prices")
    if (values <= 0).any():
        raise ValueError(f"{ticker}: daily Open/Close prices must be positive")

    return df


def load_daily_market_data(
    config: dict,
    root_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restituisce i prezzi giornalieri allineati e un controllo di copertura.

    Considera soltanto il periodo in cui tutti gli ETF sono negoziati.
    Le differenze nelle date di quotazione iniziale non vengono quindi
    interpretate erroneamente come dati mancanti. Le eventuali sedute
    isolate assenti per un ETF nel periodo comune vengono escluse per TUTTI
    gli ETF e registrate nel report di qualità. I prezzi negoziabili
    non vengono mai riempiti in avanti (forward fill).

    Colonne in uscita: SPY_open, SPY_close, IEF_open, IEF_close, ...
    L'indice contiene date effettive di negoziazione, non etichette W-FRI.
    """
    tickers = get_tickers(config)
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("The investable universe must contain distinct tickers")

    market_dir = root_dir / config["paths"]["raw_data"] / "market"
    raw = {
        ticker: _read_etf_prices(market_dir / f"{ticker}.csv", ticker)
        for ticker in tickers
    }
    common_start = max(df.index.min() for df in raw.values())
    common_end = min(df.index.max() for df in raw.values())
    if common_start >= common_end:
        raise ValueError("No overlapping trading history for the ETF universe")

    scoped = {ticker: df.loc[common_start:common_end] for ticker, df in raw.items()}
    common_dates = scoped[tickers[0]].index
    all_dates = common_dates
    for ticker in tickers[1:]:
        common_dates = common_dates.intersection(scoped[ticker].index)
        all_dates = all_dates.union(scoped[ticker].index)
    common_dates = common_dates.sort_values()

    if len(common_dates) < 2:
        raise ValueError("Fewer than two common trading sessions")
    if len(common_dates) / len(all_dates) < 0.99:
        raise ValueError(
            "More than 1% of sessions are inconsistent across ETFs. "
            "Inspect raw CSVs before proceeding."
        )

    frames = []
    audit_rows = []
    for ticker in tickers:
        df = scoped[ticker]
        daily = df.reindex(common_dates)[["Open", "Close"]]
        daily = daily.rename(
            columns={"Open": f"{ticker}_open", "Close": f"{ticker}_close"}
        )
        frames.append(daily)
        audit_rows.append(
            {
                "ticker": ticker,
                "raw_rows": len(raw[ticker]),
                "overlap_rows": len(df),
                "aligned_rows": len(common_dates),
                "sessions_missing_within_overlap": len(all_dates.difference(df.index)),
                "first_raw_date": raw[ticker].index.min().date().isoformat(),
                "last_raw_date": raw[ticker].index.max().date().isoformat(),
            }
        )

    market_daily = pd.concat(frames, axis=1)
    market_daily.index.name = "Date"
    if market_daily.isna().any().any():
        raise RuntimeError("Market alignment introduced missing Open/Close values")

    audit = pd.DataFrame(audit_rows).set_index("ticker")
    return market_daily, audit


def build_weekly_decision_dates(market_daily: pd.DataFrame) -> pd.DataFrame:
    """Individua l'ULTIMA seduta effettiva di ciascuna settimana.

    Una settimana con festività può terminare di giovedi'. I segnali
    vengono calcolati dopo la chiusura di quella seduta e possono essere
    eseguiti all'apertura della prima seduta successiva disponibile.
    """
    session_dates = pd.Series(market_daily.index, index=market_daily.index)
    dates = session_dates.resample("W-FRI").last().dropna()
    decisions = pd.DataFrame({"decision_date": pd.to_datetime(dates.values)})
    decisions.index = pd.DatetimeIndex(decisions["decision_date"], name="Date")
    # A differenza delle etichette W-FRI, l'indice indica sempre una seduta reale.
    return decisions


def transform_market_data(
    config: dict,
    root_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """Aggrega i dati OHLCV giornalieri nel pannello settimanale storico.

    Le serie OHLCV settimanali servono per le analisi, NON per eseguire
    ordini all'apertura successiva o stimare la covarianza su 126 sedute.
    A1 e A2 devono utilizzare invece market_daily.csv.
    """
    market_dir = root_dir / config["paths"]["raw_data"] / "market"
    weekly_market: Dict[str, pd.DataFrame] = {}
    print("\n[Data Transformation] Weekly market data")

    for ticker in get_tickers(config):
        df = _read_etf_prices(market_dir / f"{ticker}.csv", ticker)
        aggregation_rules = {
            field: reducer
            for field, reducer in {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }.items()
            if field in df.columns
        }
        weekly = df.resample("W-FRI").agg(aggregation_rules)
        weekly = weekly.dropna(subset=["Close"])
        weekly = weekly.rename(
            columns={col: f"{ticker}_{col.lower()}" for col in weekly.columns}
        )
        weekly_market[ticker] = weekly
        print(f"  {ticker}: {len(weekly):,} weekly rows")

    return weekly_market


def _macro_series_mapping(config: dict) -> Dict[str, str]:
    """Mantiene la corrispondenza tra nomi logici e identificativi delle serie FRED."""
    mapping: Dict[str, str] = {}
    for section in ("rates", "inflation"):
        for name, series_id in config.get("data", {}).get(section, {}).items():
            if series_id:
                mapping[f"{section}__{name}"] = series_id
    return mapping


def _is_low_frequency_macro(series_id: str) -> bool:
    return series_id in LOW_FREQUENCY_SERIES


def transform_macro_data(
    config: dict,
    root_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """Prepara le serie macroeconomiche storiche per il futuro livello B.

    ATTENZIONE: i ritardi di pubblicazione fissi sono soltanto approssimazioni.
    Non ricostruiscono le versioni storiche dei dati (vintage) né gli istanti
    esatti di pubblicazione. Il livello B non deve trattare queste variabili
    come feature verificate point-in-time.
    """
    raw_dir = root_dir / config["paths"]["raw_data"] / "macro"
    transformed: Dict[str, pd.DataFrame] = {}

    for logical_name, series_id in _macro_series_mapping(config).items():
        df = _coerce_numeric(_read_csv_with_date_index(raw_dir / f"{series_id}.csv"))
        if series_id not in df.columns:
            if len(df.columns) != 1:
                raise KeyError(f"Cannot identify FRED series {series_id}")
            df.columns = [series_id]
        series = df[[series_id]].copy()

        if _is_low_frequency_macro(series_id):
            lag = get_release_lag_days(config, series_id)
            series.index = series.index + pd.to_timedelta(lag, unit="D")
            series.index.name = "Date"
            weekly = series.resample("W-FRI").last().ffill()
        else:
            weekly = series.resample("W-FRI").last().ffill(limit=2)
        transformed[logical_name] = weekly.rename(columns={series_id: logical_name})

    return transformed


def build_master_dataset(
    market_data: Dict[str, pd.DataFrame],
    macro_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Unisce i dati settimanali di mercato alle eventuali variabili macroeconomiche."""
    frames = list(market_data.values()) + list(macro_data.values())
    if not frames:
        raise ValueError("No transformed datasets available")
    master = pd.concat(frames, axis=1, join="outer").sort_index()

    slow_cols = [
        name
        for name in ("rates__policy_rate", "inflation__cpi", "inflation__core_cpi")
        if name in master.columns
    ]
    if slow_cols:
        master[slow_cols] = master[slow_cols].ffill()

    close_cols = [col for col in master if col.endswith("_close")]
    if not close_cols:
        raise ValueError("No tradable close columns in weekly master")
    master = master.dropna(subset=close_cols)
    master.index.name = "Date"
    return master


def build_data_quality_summary(master: pd.DataFrame) -> pd.DataFrame:
    """Riepiloga copertura, valori mancanti e date valide del dataset settimanale."""
    return pd.DataFrame(
        {
            "non_null": master.notna().sum(),
            "missing": master.isna().sum(),
            "missing_pct": master.isna().mean() * 100,
            "first_valid": [master[c].first_valid_index() for c in master],
            "last_valid": [master[c].last_valid_index() for c in master],
        }
    )


def run_data_transformation(
    config: dict,
    root_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepara i dati giornalieri per il backtest e il dataset settimanale.

    Restituisce la stessa coppia (weekly_master, quality_summary) della
    versione precedente, mantenendo invariata l'interfaccia pubblica
    durante la migrazione graduale.
    """
    out_dir = root_dir / config["paths"]["processed_data"]
    out_dir.mkdir(parents=True, exist_ok=True)

    level = str(config.get("pipeline", {}).get("active_level", "A1")).upper()
    if level not in {"A1", "A2", "B", "C"}:
        raise ValueError(f"Unknown pipeline level: {level}")

    # I livelli B/C richiedono un'integrazione esplicita successiva.
    # La trasformazione macro attuale usa ritardi stimati, non vintage storiche verificate.
    if level in {"B", "C"}:
        raise NotImplementedError(
            "Level B/C data preparation is not implemented yet. "
            "Validate macro release dates/vintages and refactor this stage first."
        )

    daily, daily_audit = load_daily_market_data(config, root_dir)
    daily = restrict_to_allowed_period(daily, config)
    decision_dates = build_weekly_decision_dates(daily)
    weekly_market = transform_market_data(config, root_dir)
    macros = transform_macro_data(config, root_dir) if level in {"B", "C"} else {}

    weekly_master = build_master_dataset(weekly_market, macros)
    weekly_master = restrict_to_allowed_period(weekly_master, config)
    weekly_quality = build_data_quality_summary(weekly_master)

    daily.to_csv(out_dir / "market_daily.csv")
    decision_dates.to_csv(out_dir / "weekly_decision_dates.csv", index_label="Date")
    daily_audit.to_csv(out_dir / "market_daily_quality.csv")
    weekly_master.to_csv(out_dir / "master_weekly.csv")
    weekly_quality.to_csv(out_dir / "data_quality_summary.csv")

    print("\n[Data Transformation] Multi-Asset data ready")
    print(f"  Daily sessions:   {len(daily):,}")
    print(f"  Weekly decisions: {len(decision_dates):,}")
    print(f"  Weekly rows:      {len(weekly_master):,}")
    print(f"  Daily output:     {out_dir / 'market_daily.csv'}")
    print(f"  Weekly output:    {out_dir / 'master_weekly.csv'}")
    print("  OOS disabled:     " + str(not config.get("evaluation", {}).get("evaluate_oos", False)))
    return weekly_master, weekly_quality
