"""Shared utilities for the multi-asset pipeline.

This module centralises:
- the investable universe (``ASSETS``, the single definition of the ETFs, and ``get_tickers``);
- loading of ``config.yaml`` (``load_config``) and resolution of the project root;
- the provisional publication lags of macroeconomic series;
- protected writing of frozen artifacts, which are never overwritten.
"""

from pathlib import Path
import re
from typing import Optional, Union

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.yaml"

# Universe of levels A/B/C, in the order used for weights, columns and orders.
# This is the ONLY definition in the code base: every module imports it from here.
ASSETS = ("SPY", "IEF", "GLD", "SHY")

# Allowed ticker characters: each ticker is also used as a CSV file name.
_VALID_TICKER = re.compile(r"^[A-Za-z0-9._-]+$")


# Indicative publication lags, in calendar days, for selected FRED series.
# They do NOT represent a historical release calendar nor the data vintages
# available at each date. Actual release dates and revisions must be verified
# before these series are used in Level B.
DEFAULT_RELEASE_LAGS_DAYS = {
    "CPIAUCSL": 45,
    "CPILFESL": 45,
    "FEDFUNDS": 35,
}


def get_tickers(config: dict) -> list[str]:
    """Return the investable tickers in the order defined in ``config.yaml``.

    The project universe is SPY, IEF, GLD and SHY. This function does not
    enforce those four instruments: checking the universe required by levels
    A1/A2 is the responsibility of the strategy modules.

    It verifies that the configuration entries are valid and that each ticker
    appears only once, avoiding ambiguity between data, weights and orders.
    """
    if not isinstance(config, dict):
        raise TypeError("La configurazione deve essere un dizionario.")

    universe = config.get("universe")
    if not isinstance(universe, dict):
        raise ValueError("Manca la sezione 'universe' nella configurazione.")

    assets = universe.get("traded_assets")
    if not isinstance(assets, dict) or not assets:
        raise ValueError("La sezione 'universe.traded_assets' deve contenere asset.")

    tickers: list[str] = []
    for category, asset in assets.items():
        if not isinstance(asset, dict):
            raise ValueError(f"Configurazione non valida per l'asset '{category}'.")

        ticker = asset.get("ticker")
        if not isinstance(ticker, str) or not ticker or ticker != ticker.strip():
            raise ValueError(f"Ticker mancante o non valido per '{category}'.")
        if not _VALID_TICKER.fullmatch(ticker):
            raise ValueError(f"Ticker non valido: {ticker!r}")
        if ticker != ticker.upper():
            raise ValueError(f"Il ticker '{ticker}' deve essere scritto in maiuscolo.")
        if ticker in tickers:
            raise ValueError(f"Ticker duplicato nell'universo investibile: {ticker}.")

        tickers.append(ticker)

    return tickers


def get_release_lag_days(config: dict, series_id: str) -> int:
    """Return the conventional publication lag of a FRED series, in days.

    The value is read from ``data.release_lags_days`` in ``config.yaml``. If no
    specific value is configured, ``DEFAULT_RELEASE_LAGS_DAYS`` is used; any
    other series keeps the previous behaviour and returns zero.

    A lag of zero does NOT certify that a series was available in real time.
    The fixed 45- and 35-day lags are approximations as well: Level B must rely
    on a verified point-in-time protocol.

    Negative, fractional or boolean lags are rejected, since they could
    otherwise introduce look-ahead errors that are hard to detect.
    """
    if not isinstance(config, dict):
        raise TypeError("La configurazione deve essere un dizionario.")
    if not isinstance(series_id, str) or not series_id.strip():
        raise ValueError("L'identificativo della serie FRED non è valido.")

    data = config.get("data", {})
    if not isinstance(data, dict):
        raise TypeError("La sezione 'data' deve essere un dizionario.")

    configured_lags = data.get("release_lags_days")
    if configured_lags is None:
        configured_lags = {}
    if not isinstance(configured_lags, dict):
        raise TypeError("'data.release_lags_days' deve essere un dizionario.")

    lag = configured_lags.get(
        series_id,
        DEFAULT_RELEASE_LAGS_DAYS.get(series_id, 0),
    )
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 0:
        raise ValueError(
            f"Ritardo di pubblicazione non valido per '{series_id}': {lag!r}. "
            "È richiesto un numero intero di giorni maggiore o uguale a zero."
        )

    return lag


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load YAML without silently filling missing parameters."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File di configurazione non trovato: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Il file di configurazione deve contenere un dizionario YAML.")
    return config


def write_frozen_outputs(files: dict[Path, str]) -> None:
    """Create absent artifacts, never overwrite a frozen existing artifact.

    Check ALL conflicts before creating ANY missing files, so an unexpected
    difference cannot leave a half-updated version directory behind.
    """
    conflicts = []
    for path, content in files.items():
        if path.exists():
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                conflicts.append(str(path))

    if conflicts:
        raise RuntimeError(
            "Frozen Level B artifacts differ; no files were overwritten:\n"
            + "\n".join(conflicts)
        )

    for path, content in files.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")


def get_project_root(start_path: Optional[Union[str, Path]] = None) -> Path:
    """Robustly resolve the root directory of the ProjectAgentTrading repository."""
    if start_path is not None:
        current = Path(start_path).resolve()
    else:
        current = Path(__file__).resolve().parent

    for candidate in [current] + list(current.parents):
        if (candidate / "config.yaml").is_file() and (candidate / "src").is_dir():
            return candidate

    # Standard fallback: the repository root that contains src/.
    return ROOT_DIR
