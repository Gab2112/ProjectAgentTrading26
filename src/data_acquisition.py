"""Acquisition and validation of raw data for the multi-asset project.

The module handles two independent sources:

- Yahoo Finance: adjusted daily OHLCV prices for SPY, IEF, GLD and SHY.
- FRED: candidate macroeconomic series for levels B and C.

CSV files are saved in ``data/raw/market`` and ``data/raw/macro``.
No returns, indicators or trading signals are computed here.

To protect the reproducibility of the experiments, existing CSV files are
not overwritten unless the configuration sets
``pipeline.overwrite_raw_data: true``.

During A1 and A2 the macroeconomic download is disabled unless it is
explicitly requested through ``pipeline.download_macro_data: true``.

WARNING: ordinary FRED series may contain historical revisions. Downloading
them does not, on its own, guarantee point-in-time data availability.
"""

from pathlib import Path
import re
from typing import Dict

import numpy as np
import pandas as pd

from src.utils import get_tickers


# Only identifiers that are valid file names are accepted.
_VALID_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_MARKET_PRICE_COLUMNS = ("Open", "High", "Low", "Close")


def _parse_date(value: object, name: str) -> pd.Timestamp:
    """Parse a configuration date and validate its format."""
    try:
        date = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Data non valida per {name}: {value!r}") from exc

    if pd.isna(date) or date.tzinfo is not None or date != date.normalize():
        raise ValueError(f"{name} deve contenere una data senza orario: {value!r}")
    return date


def _resolve_end_date(config: dict) -> str:
    """Return the last date included in the acquisition.

    If ``data.end_date`` is null, the previous UTC day is used to avoid
    including a current, possibly incomplete, session.
    """
    configured = config["data"].get("end_date")
    if configured is None:
        last_day = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
        last_day -= pd.Timedelta(days=1)
    else:
        last_day = _parse_date(configured, "data.end_date")

    start_day = _parse_date(config["data"]["start_date"], "data.start_date")
    if last_day < start_day:
        raise ValueError("data.end_date precede data.start_date")
    return last_day.strftime("%Y-%m-%d")


def _normalize_dates(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalise the date index without introducing future observations."""
    if df.empty:
        raise ValueError(f"Nessun dato disponibile: {source}")

    data = df.copy()
    try:
        index = pd.DatetimeIndex(pd.to_datetime(data.index, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Indice temporale non valido: {source}") from exc

    if index.tz is not None:
        # Keep the local date reported by the exchange instead of converting it to UTC.
        index = index.tz_localize(None)
    index = index.normalize()

    if index.hasnans or index.has_duplicates:
        raise ValueError(f"Date mancanti o duplicate: {source}")
    data.index = index
    data.index.name = "Date"
    return data.sort_index()


def _normalize_market_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Validate adjusted Open/Close prices and keep the OHLCV columns."""
    data = df.copy()

    # yfinance may return MultiIndex columns even for a single ETF.
    if isinstance(data.columns, pd.MultiIndex):
        for level in range(data.columns.nlevels):
            labels = data.columns.get_level_values(level)
            if {"Open", "Close"}.issubset(set(labels)):
                data.columns = labels
                break
        else:
            raise ValueError(f"{ticker}: colonne OHLCV non riconoscibili")

    if data.columns.has_duplicates:
        raise ValueError(f"{ticker}: colonne duplicate")

    missing = [field for field in ("Open", "Close") if field not in data]
    if missing:
        raise ValueError(f"{ticker}: colonne obbligatorie assenti: {missing}")

    fields = [field for field in _MARKET_PRICE_COLUMNS if field in data]
    if "Volume" in data:
        fields.append("Volume")
    data = _normalize_dates(data[fields], ticker)

    for column in fields:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    price_columns = [field for field in _MARKET_PRICE_COLUMNS if field in data]
    prices = data[price_columns].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError(f"{ticker}: prezzi assenti, non finiti o non positivi")

    # Volume is informative but not required by the A1/A2 backtest.
    if "Volume" in data and data["Volume"].notna().any():
        volume = data["Volume"].dropna().to_numpy(dtype=float)
        if not np.isfinite(volume).all() or (volume < 0).any():
            raise ValueError(f"{ticker}: volumi non validi")

    return data


def _normalize_macro_data(df: pd.DataFrame, series_id: str) -> pd.DataFrame:
    """Keep the original FRED frequency without filling missing values."""
    data = _normalize_dates(df, series_id)
    if series_id not in data.columns:
        if len(data.columns) != 1:
            raise ValueError(f"{series_id}: colonna FRED non identificabile")
        data.columns = [series_id]

    data = data[[series_id]].copy()
    data[series_id] = pd.to_numeric(data[series_id], errors="coerce")
    if not np.isfinite(data[series_id].dropna().to_numpy(dtype=float)).all():
        raise ValueError(f"{series_id}: valori macroeconomici non finiti")
    if data[series_id].notna().sum() == 0:
        raise ValueError(f"{series_id}: nessuna osservazione macroeconomica valida")
    return data


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    """Write the CSV through a temporary file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp")
    try:
        df.to_csv(temp)
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def _overwrite_enabled(config: dict) -> bool:
    """Check for explicit permission to replace raw data."""
    enabled = config.get("pipeline", {}).get("overwrite_raw_data", False)
    if not isinstance(enabled, bool):
        raise ValueError("pipeline.overwrite_raw_data deve essere booleano")
    return enabled


def download_market_data(config: dict, root_dir: Path) -> Dict[str, pd.DataFrame]:
    """Download or reuse daily data for the configured ETFs.

    Configuration dates are INCLUSIVE, whereas Yahoo Finance treats ``end``
    as exclusive: one day is therefore added.

    New data is requested with ``auto_adjust=True`` to obtain consistently
    adjusted Open and Close prices. Adjustment does not make prices
    automatically executable in a real market.
    """
    source = config["data"].get("market_source", {})
    if source.get("provider", "yfinance") != "yfinance":
        raise ValueError("Per i dati di mercato è supportato solo yfinance")
    if source.get("auto_adjust", True) is not True:
        raise ValueError("A1/A2 richiedono auto_adjust=True per Open e Close")
    if source.get("interval", "1d") != "1d":
        raise ValueError("Per i dati di mercato è richiesta la frequenza 1d")

    start = _parse_date(config["data"]["start_date"], "data.start_date")
    end = _parse_date(_resolve_end_date(config), "data.end_date")
    market_dir = Path(root_dir) / config["paths"]["raw_data"] / "market"
    market_dir.mkdir(parents=True, exist_ok=True)

    output: Dict[str, pd.DataFrame] = {}
    overwrite = _overwrite_enabled(config)
    print(f"\n[Acquisizione] ETF giornalieri: {start.date()} - {end.date()}")

    for ticker in get_tickers(config):
        path = market_dir / f"{ticker}.csv"

        if path.exists() and not overwrite:
            existing = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
            output[ticker] = _normalize_market_data(existing, ticker)
            print(f"  {ticker}: CSV esistente conservato ({len(existing)} sedute)")
            continue

        # Deferred import: yfinance is not needed when all CSV files already exist.
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("Per scaricare ETF installare yfinance") from exc

        print(f"  {ticker}: download da Yahoo Finance")
        try:
            downloaded = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Download Yahoo Finance fallito: {ticker}") from exc

        data = _normalize_market_data(downloaded, ticker)
        _save_csv(data, path)
        output[ticker] = data
        print(f"  {ticker}: salvate {len(data)} sedute")

    return output


def _get_fred_series(config: dict) -> Dict[str, str]:
    """Map logical names to the FRED series listed in the configuration."""
    series: Dict[str, str] = {}
    for section in ("rates", "inflation", "stress"):
        for name, series_id in config["data"].get(section, {}).items():
            if not series_id:
                continue
            series_id = str(series_id).strip()
            if not _VALID_ID.fullmatch(series_id):
                raise ValueError(f"Identificativo FRED non valido: {series_id!r}")
            series[f"{section}__{name}"] = series_id
    return series


def download_macro_data(config: dict, root_dir: Path) -> Dict[str, pd.DataFrame]:
    """Download FRED series, preserving their original dates and values.

    No forward filling is applied, publication dates are not estimated and
    historical data vintages are not reconstructed.
    """
    source = config["data"].get("macro_source", {})
    if source.get("provider", "FRED").upper() != "FRED":
        raise ValueError("Per le serie macro è supportata solo la fonte FRED")

    start = _parse_date(config["data"]["start_date"], "data.start_date")
    end = _parse_date(_resolve_end_date(config), "data.end_date")
    macro_dir = Path(root_dir) / config["paths"]["raw_data"] / "macro"
    macro_dir.mkdir(parents=True, exist_ok=True)

    output: Dict[str, pd.DataFrame] = {}
    overwrite = _overwrite_enabled(config)
    print("\n[Acquisizione] Serie macroeconomiche FRED")

    for logical_name, series_id in _get_fred_series(config).items():
        path = macro_dir / f"{series_id}.csv"

        if path.exists() and not overwrite:
            existing = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
            output[logical_name] = _normalize_macro_data(existing, series_id)
            print(f"  {series_id}: CSV esistente conservato")
            continue

        # Deferred import: pandas_datareader is not needed for levels A1/A2.
        try:
            from pandas_datareader import data as web
        except ImportError as exc:
            raise ImportError("Per scaricare FRED installare pandas_datareader") from exc

        print(f"  {series_id}: download da FRED")
        try:
            downloaded = web.DataReader(
                series_id,
                "fred",
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
            )
        except Exception as exc:
            raise RuntimeError(f"Download FRED fallito: {series_id}") from exc

        data = _normalize_macro_data(downloaded, series_id)
        _save_csv(data, path)
        output[logical_name] = data
        print(f"  {series_id}: salvate {len(data)} osservazioni")

    return output


def run_data_acquisition(config: dict, root_dir: Path) -> dict:
    """Coordinate data acquisition for the currently selected level."""
    level = str(config.get("pipeline", {}).get("active_level", "A1")).upper()
    if level not in {"A1", "A2", "B", "C"}:
        raise ValueError(f"Livello di ricerca sconosciuto: {level}")

    market_data = download_market_data(config, root_dir)

    # In Level A, the benchmarks must not depend on macroeconomic series.
    macro_flag = config.get("pipeline", {}).get("download_macro_data")
    if macro_flag is not None and not isinstance(macro_flag, bool):
        raise ValueError("pipeline.download_macro_data deve essere booleano")
    acquire_macro = (level in {"B", "C"}) if macro_flag is None else macro_flag

    if acquire_macro:
        macro_data = download_macro_data(config, root_dir)
    else:
        macro_data = {}
        print("[Acquisizione] FRED saltato: non richiesto per A1/A2")

    print("[Acquisizione] Fase completata correttamente")
    return {"market": market_data, "macro": macro_data}
