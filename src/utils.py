"""Shared utilities.

- ``ASSETS``: the investable universe, the single definition used by every module;
- ``get_tickers``: tickers in the order declared in ``config.yaml``;
- ``get_release_lag_days``: conventional publication lags of monthly FRED series;
- ``load_config`` and ``get_project_root``: configuration and path resolution.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.yaml"

# Investable universe, in the order used for weights, columns and orders.
ASSETS = ("SPY", "IEF", "GLD", "SHY")

# Admissible ticker characters: each ticker is also used as a file name.
_VALID_TICKER = re.compile(r"^[A-Za-z0-9._-]+$")

# Conventional publication lags, in calendar days, used when a series has no
# entry in ``data.release_lags_days``. They approximate release dates and do
# not reconstruct historical data vintages.
DEFAULT_RELEASE_LAGS_DAYS = {
    "CPIAUCSL": 45,
    "CPILFESL": 45,
    "FEDFUNDS": 35,
}


def get_tickers(config: dict) -> list[str]:
    """Return the investable tickers in the order defined in ``config.yaml``."""
    if not isinstance(config, dict):
        raise TypeError("The configuration must be a dictionary.")

    assets = config.get("universe", {}).get("traded_assets")
    if not isinstance(assets, dict) or not assets:
        raise ValueError("The section 'universe.traded_assets' must list the traded assets.")

    tickers: list[str] = []
    for category, asset in assets.items():
        ticker = asset.get("ticker") if isinstance(asset, dict) else None
        if not isinstance(ticker, str) or not ticker or ticker != ticker.strip():
            raise ValueError(f"Missing or invalid ticker for '{category}'.")
        if not _VALID_TICKER.fullmatch(ticker):
            raise ValueError(f"Invalid ticker: {ticker!r}")
        if ticker != ticker.upper():
            raise ValueError(f"The ticker '{ticker}' must be upper case.")
        if ticker in tickers:
            raise ValueError(f"Duplicate ticker in the investable universe: {ticker}.")
        tickers.append(ticker)
    return tickers


def get_release_lag_days(config: dict, series_id: str) -> int:
    """Return the publication lag of a FRED series, in calendar days.

    The value is read from ``data.release_lags_days``; otherwise the default
    in ``DEFAULT_RELEASE_LAGS_DAYS`` is used, and zero for daily series.
    Negative, fractional or boolean lags are rejected because they would
    introduce look-ahead bias.
    """
    if not isinstance(config, dict):
        raise TypeError("The configuration must be a dictionary.")
    if not isinstance(series_id, str) or not series_id.strip():
        raise ValueError("Invalid FRED series identifier.")

    configured = config.get("data", {}).get("release_lags_days") or {}
    if not isinstance(configured, dict):
        raise TypeError("'data.release_lags_days' must be a dictionary.")

    lag = configured.get(series_id, DEFAULT_RELEASE_LAGS_DAYS.get(series_id, 0))
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 0:
        raise ValueError(
            f"Invalid publication lag for '{series_id}': {lag!r}. "
            "A non-negative integer number of days is required."
        )
    return lag


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load ``config.yaml`` without filling missing parameters."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("The configuration file must contain a YAML mapping.")
    return config


def get_project_root(start_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the repository root (the directory containing config.yaml and src/)."""
    current = Path(start_path).resolve() if start_path is not None else Path(__file__).resolve().parent
    for candidate in [current] + list(current.parents):
        if (candidate / "config.yaml").is_file() and (candidate / "src").is_dir():
            return candidate
    return ROOT_DIR
