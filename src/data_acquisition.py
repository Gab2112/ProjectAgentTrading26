"""Raw-data acquisition (step 1 of the pipeline, optional).

Sources
- Yahoo Finance (yfinance): daily OHLCV of SPY, IEF, GLD and SHY, requested
  with ``auto_adjust=True`` so that Open, High, Low and Close are adjusted
  for splits and dividends consistently;
- FRED (pandas_datareader): Treasury yields, inflation, breakeven inflation
  and VIX series listed in ``config.yaml``.

Files are written to ``data/raw/market`` and ``data/raw/macro``. The
repository already contains these files, so the default pipeline does NOT
download anything (``pipeline.download_data: false``) and every result is
reproducible offline. Re-downloading may change the data, because Yahoo
Finance re-computes adjusted prices after each distribution and FRED series
can be revised.

No return, indicator or signal is computed here.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.utils import get_tickers

_VALID_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_MARKET_PRICE_COLUMNS = ("Open", "High", "Low", "Close")


def _parse_date(value: object, name: str) -> pd.Timestamp:
    """Parse a configuration date and validate its format."""
    try:
        date = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid date for {name}: {value!r}") from exc
    if pd.isna(date) or date.tzinfo is not None or date != date.normalize():
        raise ValueError(f"{name} must be a calendar date without a time component: {value!r}")
    return date


def _resolve_end_date(config: dict) -> pd.Timestamp:
    """Last date included in the download.

    When ``data.end_date`` is null the previous UTC day is used, so that an
    incomplete current session is never stored.
    """
    configured = config["data"].get("end_date")
    if configured is None:
        last_day = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None) - pd.Timedelta(days=1)
    else:
        last_day = _parse_date(configured, "data.end_date")
    if last_day < _parse_date(config["data"]["start_date"], "data.start_date"):
        raise ValueError("data.end_date precedes data.start_date")
    return last_day


def _normalize_dates(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalise the date index, keeping the local exchange date."""
    if df.empty:
        raise ValueError(f"No data available: {source}")
    data = df.copy()
    index = pd.DatetimeIndex(pd.to_datetime(data.index, errors="raise"))
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.normalize()
    if index.hasnans or index.has_duplicates:
        raise ValueError(f"Missing or duplicate dates: {source}")
    data.index = index
    data.index.name = "Date"
    return data.sort_index()


def _normalize_market_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Validate the adjusted prices and keep the OHLCV columns."""
    data = df.copy()
    # yfinance may return MultiIndex columns even for a single ticker.
    if isinstance(data.columns, pd.MultiIndex):
        for level in range(data.columns.nlevels):
            labels = data.columns.get_level_values(level)
            if {"Open", "Close"}.issubset(set(labels)):
                data.columns = labels
                break
        else:
            raise ValueError(f"{ticker}: unrecognisable OHLCV columns")

    missing = [field for field in ("Open", "Close") if field not in data]
    if missing:
        raise ValueError(f"{ticker}: missing mandatory columns {missing}")

    fields = [field for field in _MARKET_PRICE_COLUMNS if field in data] + (["Volume"] if "Volume" in data else [])
    data = _normalize_dates(data[fields], ticker)
    for column in fields:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    prices = data[[f for f in _MARKET_PRICE_COLUMNS if f in data]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError(f"{ticker}: missing, non-finite or non-positive prices")
    return data


def _normalize_macro_data(df: pd.DataFrame, series_id: str) -> pd.DataFrame:
    """Keep the native FRED frequency; missing values are not filled."""
    data = _normalize_dates(df, series_id)
    if series_id not in data.columns:
        if len(data.columns) != 1:
            raise ValueError(f"{series_id}: unidentifiable FRED column")
        data.columns = [series_id]
    data = data[[series_id]].copy()
    data[series_id] = pd.to_numeric(data[series_id], errors="coerce")
    if data[series_id].notna().sum() == 0:
        raise ValueError(f"{series_id}: no valid observation")
    return data


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV atomically through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp")
    try:
        df.to_csv(temp)
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def download_market_data(config: dict, root_dir: Path) -> Dict[str, pd.DataFrame]:
    """Download adjusted daily prices for every configured ETF.

    Configuration dates are inclusive, whereas Yahoo Finance treats ``end``
    as exclusive: one day is therefore added.
    """
    import yfinance as yf

    source = config["data"]["market_source"]
    if source.get("provider") != "yfinance" or source.get("auto_adjust") is not True:
        raise ValueError("Market data must come from yfinance with auto_adjust=True.")

    start = _parse_date(config["data"]["start_date"], "data.start_date")
    end = _resolve_end_date(config)
    market_dir = Path(root_dir) / config["paths"]["raw_data"] / "market"
    output: Dict[str, pd.DataFrame] = {}
    print(f"\n[Download] Daily ETF prices: {start.date()} - {end.date()}")

    for ticker in get_tickers(config):
        downloaded = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        data = _normalize_market_data(downloaded, ticker)
        _save_csv(data, market_dir / f"{ticker}.csv")
        output[ticker] = data
        print(f"  {ticker}: {len(data)} sessions saved")
    return output


def download_macro_data(config: dict, root_dir: Path) -> Dict[str, pd.DataFrame]:
    """Download the FRED series listed in ``config.yaml``, preserving dates and values."""
    from pandas_datareader import data as web

    start = _parse_date(config["data"]["start_date"], "data.start_date")
    end = _resolve_end_date(config)
    macro_dir = Path(root_dir) / config["paths"]["raw_data"] / "macro"
    output: Dict[str, pd.DataFrame] = {}
    print("\n[Download] FRED macroeconomic series")

    for section in ("rates", "inflation", "stress"):
        for name, series_id in config["data"].get(section, {}).items():
            series_id = str(series_id).strip()
            if not _VALID_ID.fullmatch(series_id):
                raise ValueError(f"Invalid FRED identifier: {series_id!r}")
            downloaded = web.DataReader(
                series_id, "fred", start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d")
            )
            data = _normalize_macro_data(downloaded, series_id)
            _save_csv(data, macro_dir / f"{series_id}.csv")
            output[f"{section}__{name}"] = data
            print(f"  {series_id}: {len(data)} observations saved")
    return output


def run_data_acquisition(config: dict, root_dir: Path) -> dict:
    """Download every raw dataset, overwriting the files in data/raw."""
    return {
        "market": download_market_data(config, root_dir),
        "macro": download_macro_data(config, root_dir),
    }
