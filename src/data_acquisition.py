"""
data_acquisition.py 

Acquisizione dei dati per il progetto su Macro / Rates Regimes.

This module downloads:
 1. Market data for the configured fixed-income assets from YF.
 2. Macro / rates series from FRED.

 If any of the data sources are unavailable, the module will 
 raise an exception (raccoglierò i dati da bloomberg).

 Raw data are saved under data/raw withouth performing feature
 engineering or strategyt-specific transformations.
"""

from pathlib import Path
from typing import Dict

import pandas as pd
import yfinance as yf
from pandas_datareader import data as web


def _resolve_end_date(config: dict) -> str:
    """Return the configured end date or today's date if it is left open."""
    end_date = config["data"].get("end_date")

    if end_date is None:
        return pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

    return str(end_date)


def download_market_data(config: dict, root_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Download daily OHLCV data for the configured traded assets.

    (OHLCV = Open, High, Low, Close, Volume)
    
    Each asset is saved separately under:
        data/raw/market/<TICKER>.csv
    """
    start_date = str(config["data"]["start_date"])
    end_date = _resolve_end_date(config)

    market_dir = root_dir / config["paths"]["raw_data"] / "market"
    market_dir.mkdir(parents=True, exist_ok=True)

    assets = config["universe"]["traded_assets"]
    market_data: Dict[str, pd.DataFrame] = {}

    print("\n[Data Acquisition] Market data")
    print(f"Period: {start_date} -> {end_date}")

    for bucket, asset in assets.items():
        ticker = asset["ticker"]

        print(f"  Downloading {ticker:<5} ({bucket})...")

        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download market data for {ticker}."
            ) from exc

        if df.empty:
            raise ValueError(f"No market data returned for {ticker}.")

        # yfinance may return a MultiIndex even for a single ticker.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index.name = "Date"
        df.to_csv(market_dir / f"{ticker}.csv")

        market_data[ticker] = df

        print(f"    Saved {len(df):,} rows.")

    return market_data


def _get_fred_series(config: dict) -> Dict[str, str]:
    """Collect the configured FRED series into a flat name -> series-ID mapping."""
    series: Dict[str, str] = {}

    for name, series_id in config["data"]["rates"].items():
        if series_id:
            series[f"rates__{name}"] = series_id

    for name, series_id in config["data"]["inflation"].items():
        if series_id:
            series[f"inflation__{name}"] = series_id

    return series


def download_macro_data(config: dict, root_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Download configured macro / rates series from FRED.

    Each series is kept separately so that its native observation frequency
    is preserved as much as possible.

    Files are saved under:
        data/raw/macro/<SERIES_ID>.csv
    """
    start_date = str(config["data"]["start_date"])
    end_date = _resolve_end_date(config)

    macro_dir = root_dir / config["paths"]["raw_data"] / "macro"
    macro_dir.mkdir(parents=True, exist_ok=True)

    fred_series = _get_fred_series(config)
    macro_data: Dict[str, pd.DataFrame] = {}

    print("\n[Data Acquisition] FRED macro / rates data")

    for logical_name, series_id in fred_series.items():
        print(f"  Downloading {series_id:<10} ({logical_name})...")

        try:
            df = web.DataReader(
                series_id,
                "fred",
                start=start_date,
                end=end_date,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download FRED series {series_id}."
            ) from exc

        if df.empty:
            raise ValueError(f"No FRED data returned for {series_id}.")

        df.index.name = "Date"
        df.to_csv(macro_dir / f"{series_id}.csv")

        macro_data[logical_name] = df

        print(f"    Saved {len(df):,} rows.")

    return macro_data


def run_data_acquisition(config: dict, root_dir: Path) -> dict:
    """Run the complete raw-data acquisition stage."""
    market_data = download_market_data(config, root_dir)
    macro_data = download_macro_data(config, root_dir)

    print("\nData acquisition completed successfully.")

    return {
        "market": market_data,
        "macro": macro_data,
    }