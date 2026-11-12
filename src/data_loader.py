"""Data loading for the multi-asset project.

Single entry point for reading files:
- ``load_dataset``: generic reader (CSV, TSV, Excel, Parquet, Feather, JSON)
  with strict checks on the time index;
- daily ETF prices (``load_daily_market_data``);
- weekly Treasury yields for regime clustering (``load_weekly_yields``);
- the market and macro dataset used by the agents (``load_agent_*``).

This module reads and aligns data but does NOT build model features:
features are defined in ``src/features.py``.
"""

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.features import build_agent_macro_features
from src.utils import ASSETS, get_project_root, get_release_lag_days, get_tickers, load_config


REQUIRED_MARKET_COLUMNS = ("Open", "Close")


def load_dataset(
    path: str | Path,
    date_column: str | None = "Date",
    strict: bool = True,
    **read_options,
) -> pd.DataFrame:
    """Load a tabular dataset with strict checks on its date index.

    Supports CSV, TSV, TXT, Excel, Parquet, Feather and JSON files and applies
    either strict or permissive date-index validation.
    """
    path = Path(path)
    extension = path.suffix.lower()
    formats = (
        ".csv", ".tsv", ".txt", ".xlsx", ".xls",
        ".parquet", ".feather", ".json",
    )
    if extension not in formats:
        raise ValueError(
            f"Tipo di file non supportato: '{extension}'. "
            f"Formati supportati: {', '.join(sorted(formats))}"
        )

    if not isinstance(strict, bool):
        raise TypeError("Il parametro 'strict' deve essere booleano.")
    if date_column is not None and (not isinstance(date_column, str) or not date_column):
        raise TypeError("'date_column' deve essere una stringa non vuota o None.")
    if not path.is_file():
        raise FileNotFoundError(f"File non trovato: {path}")

    try:
        if extension in {".csv", ".tsv", ".txt"}:
            if extension == ".tsv":
                read_options.setdefault("sep", "\t")
            frame = pd.read_csv(path, **read_options)
        elif extension in {".xlsx", ".xls"}:
            frame = pd.read_excel(path, **read_options)
        elif extension == ".parquet":
            frame = pd.read_parquet(path, **read_options)
        elif extension == ".feather":
            frame = pd.read_feather(path, **read_options)
        else:
            frame = pd.read_json(path, **read_options)
    except Exception as exc:
        raise RuntimeError(f"Errore durante la lettura del file {path}: {exc}") from exc

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Il file {path.name} non contiene un DataFrame.")
    if frame.empty:
        raise ValueError(f"Il file {path.name} non contiene osservazioni.")
    if frame.columns.has_duplicates:
        raise ValueError(f"Colonne duplicate nel file {path.name}.")
    if date_column is None:
        return frame

    if date_column not in frame.columns and frame.index.name == date_column:
        frame = frame.reset_index()
    if date_column not in frame.columns:
        raise KeyError(f"Colonna '{date_column}' assente nel file {path.name}.")

    frame = frame.copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    invalid_dates = frame[date_column].isna()
    if invalid_dates.any():
        if strict:
            raise ValueError(
                f"Rilevate {int(invalid_dates.sum())} date non valide "
                f"nel file {path.name}."
            )
        frame = frame.loc[~invalid_dates].copy()
    if frame.empty:
        raise ValueError(f"Nessuna osservazione con data valida nel file {path.name}.")
    duplicate_dates = frame.duplicated(subset=[date_column], keep=False)
    if duplicate_dates.any():
        if strict:
            raise ValueError(
                f"Rilevate {int(duplicate_dates.sum())} righe con date "
                f"duplicate nel file {path.name}."
            )
        frame = frame.drop_duplicates(subset=[date_column], keep="last")
    result = frame.set_index(date_column).sort_index()
    if not isinstance(result.index, pd.DatetimeIndex):
        raise TypeError(f"Impossibile costruire un indice temporale per {path.name}.")
    return result


def read_csv_with_date_index(path: Path) -> pd.DataFrame:
    """Read a CSV through the shared loader with strict date validation."""
    df = load_dataset(path, strict=True)
    if df.empty:
        raise ValueError(f"Empty dataset: {path}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Expected DatetimeIndex: {path}")
    return df


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to numeric values; missing values remain NaN."""
    result = df.copy()
    for col in result.columns:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def read_etf_prices(path: Path, ticker: str) -> pd.DataFrame:
    """Check that daily Open/Close prices are present and valid."""
    df = read_csv_with_date_index(path)
    missing = [column for column in REQUIRED_MARKET_COLUMNS if column not in df]
    if missing:
        raise KeyError(f"{ticker}: missing required columns {missing} in {path}")

    df = coerce_numeric(df)
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
    """Return aligned daily prices together with a coverage report.

    Only the period in which all ETFs are traded is considered, so that
    different listing dates are not mistaken for missing data. Isolated
    sessions missing for one ETF within the common period are dropped for
    ALL ETFs and recorded in the quality report. Tradable prices are never
    forward-filled.

    Output columns: SPY_open, SPY_close, IEF_open, IEF_close, ...
    The index contains actual trading dates, not W-FRI labels.
    """
    tickers = get_tickers(config)
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("The investable universe must contain distinct tickers")

    market_dir = root_dir / config["paths"]["raw_data"] / "market"
    raw = {
        ticker: read_etf_prices(market_dir / f"{ticker}.csv", ticker)
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


def load_weekly_yields(config: dict, root_dir: Path) -> pd.DataFrame:
    """Load daily Treasury yields (FRED) and resample them to weekly frequency.

    The last available value of each week is used. Daily Treasury yields are
    published at the end of the day, so the value observed on the decision
    date is known before execution at the next session's open.
    """
    rates = config["data"]["rates"]
    series = {
        "treasury_2y": rates["treasury_2y"],
        "treasury_10y": rates["treasury_10y"],
    }
    macro_dir = root_dir / config["paths"]["raw_data"] / "macro"

    columns = {}
    for name, series_id in series.items():
        df = load_dataset(macro_dir / f"{series_id}.csv", strict=True)
        values = pd.to_numeric(df[series_id], errors="coerce")
        columns[name] = values.resample("W-FRI").last().ffill(limit=2)

    weekly = pd.DataFrame(columns)
    weekly.index.name = "Date"
    return weekly


# =============================================================================
# Agent data (weekly market data and derived FRED macro series)
# =============================================================================

def load_agent_market_data(
    frequency: str = "weekly",
    source: Optional[Union[str, Path]] = None,
    assets: Optional[Sequence[str]] = None,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    root_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load historical market data for the basket assets (default: SPY, IEF, GLD, SHY).

    Args:
        frequency: ``'weekly'`` (default) or ``'daily'``.
        source: Optional path to a specific CSV file, or an alias
            (``'master_weekly'``, ``'market_daily'``).
        assets: Tickers to select (e.g. ``['SPY', 'GLD']``). If None, all
            available assets are loaded.
        start_date: Earliest date (inclusive), e.g. ``'2005-05-20'``.
        end_date: Latest date (inclusive).
        root_dir: Project root directory, resolved automatically if omitted.

    Returns:
        pd.DataFrame: Date-indexed DataFrame of historical prices.
    """
    root = get_project_root(root_dir)
    processed_dir = root / "data" / "processed"

    # Resolve the source path.
    if source is not None and Path(source).is_file():
        csv_path = Path(source)
    elif frequency == "daily" or str(source).lower() == "market_daily":
        csv_path = processed_dir / "market_daily.csv"
    else:
        csv_path = processed_dir / "master_weekly.csv"
        if not csv_path.exists():
            fallback = processed_dir / "decision_features_training.csv"
            if fallback.exists():
                csv_path = fallback

    if not csv_path.exists():
        raise FileNotFoundError(
            f"File di dati di mercato non trovato in: {csv_path}. "
            f"Esegui la data transformation o verifica la directory data/processed."
        )

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Apply the date filter.
    if start_date is not None:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df.index <= pd.Timestamp(end_date)]

    # Keep only the requested assets.
    if assets is not None:
        req_assets = [a.upper() for a in assets]
        matched_cols = []
        for col in df.columns:
            col_upper = col.upper()
            if any(col_upper.startswith(f"{a}_") or col_upper == a for a in req_assets):
                matched_cols.append(col)
            elif not any(col_upper.startswith(f"{x}_") for x in ASSETS):
                matched_cols.append(col)  # keep features that are not tied to a specific asset
        if matched_cols:
            df = df[matched_cols]

    return df


def load_agent_macro_data(
    decision_dates: Optional[pd.DatetimeIndex] = None,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    root_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load and summarise FRED macroeconomic series (rates, curve spreads, inflation, VIX).

    Observations are aligned causally to the supplied dates, avoiding look-ahead bias.
    """
    root = get_project_root(root_dir)
    macro_dir = root / "data" / "raw" / "macro"

    config = load_config(root / "config.yaml")

    if not macro_dir.is_dir():
        raise FileNotFoundError(f"Directory dati macro non trovata in: {macro_dir}")

    macro_series: Dict[str, pd.Series] = {}
    for csv_file in macro_dir.glob("*.csv"):
        try:
            s_df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            val_col = s_df.columns[0]
            num_series = pd.to_numeric(s_df[val_col], errors="coerce")
            macro_series[csv_file.stem] = num_series
        except Exception:
            continue

    # Build the daily DataFrame of rates and base indicators.
    aligned_series = {}
    for name, s in macro_series.items():
        s = s.dropna().sort_index().copy()
        if name == "FEDFUNDS":
            s.index += pd.Timedelta(
                days=get_release_lag_days(config, "FEDFUNDS")
            )
        aligned_series[name] = s.rename(name)

    daily_macro = pd.concat(aligned_series.values(), axis=1, sort=False).sort_index()
    daily_macro = daily_macro.ffill(limit=5)

    if "FEDFUNDS" in aligned_series:
        daily_macro["FEDFUNDS"] = aligned_series["FEDFUNDS"].reindex(
            daily_macro.index, method="ffill"
        )

    calc_df = build_agent_macro_features(daily_macro, macro_series, config)

    # Align to decision_dates when provided; otherwise return daily data.
    if decision_dates is not None:
        # Causal alignment with a backward merge_asof.
        dec_df = pd.DataFrame({"Date": decision_dates}).sort_values("Date")
        calc_clean = calc_df.dropna(how="all").reset_index().rename(columns={"index": "Date"})
        aligned = pd.merge_asof(dec_df, calc_clean, on="Date", direction="backward").set_index("Date")
        result = aligned
    else:
        result = calc_df

    if start_date is not None:
        result = result[result.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        result = result[result.index <= pd.Timestamp(end_date)]

    return result


def load_agent_dataset(
    assets: Optional[Sequence[str]] = None,
    frequency: str = "weekly",
    include_macro: bool = True,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    root_dir: Optional[Union[str, Path]] = None,
    save_features_weekly_csv: bool = False,
) -> pd.DataFrame:
    """Load the joint market dataset (SPY, IEF, GLD, SHY) with aligned macro features.

    If ``save_features_weekly_csv=True`` or ``data/processed/features_weekly.csv``
    does not exist, a copy can be saved for backward compatibility with
    existing modules.
    """
    market_df = load_agent_market_data(
        frequency=frequency,
        assets=assets,
        start_date=start_date,
        end_date=end_date,
        root_dir=root_dir,
    )

    if include_macro:
        try:
            macro_df = load_agent_macro_data(
                decision_dates=market_df.index,
                root_dir=root_dir,
            )
            complete_df = market_df.join(macro_df, how="left")
        except Exception as e:
            print(f"[load_agent_dataset] Avviso: caricamento macro fallito ({e}), uso solo dati di mercato.")
            complete_df = market_df
    else:
        complete_df = market_df

    if save_features_weekly_csv:
        root = get_project_root(root_dir)
        target_path = root / "data" / "processed" / "features_weekly.csv"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        complete_df.to_csv(target_path)
        print(f"[data_loader] Esportato dataset compatibile in: {target_path}")

    return complete_df


def prepare_asset_data(
    df: pd.DataFrame,
    asset: str,
    price_col: str = "close",
) -> pd.DataFrame:
    """Extract a standardised single-asset view (e.g. ``'SPY'``) from the multi-asset DataFrame.

    - Creates the canonical columns ``'open'``, ``'high'``, ``'low'``, ``'close'``, ``'volume'``.
    - Keeps the original prefixed columns (``'SPY_close'``, etc.).
    - Keeps every global or macro feature column (e.g. ``'curve_10y_2y'``, ``'vix_level'``).
    """
    asset_u = asset.upper()
    out_df = df.copy()

    # Map the standard OHLCV fields.
    for standard_name in ["open", "high", "low", "close", "volume"]:
        source_col = f"{asset_u}_{standard_name}"
        if source_col in out_df.columns:
            out_df[standard_name] = out_df[source_col]
        elif f"{asset.lower()}_{standard_name}" in out_df.columns:
            out_df[standard_name] = out_df[f"{asset.lower()}_{standard_name}"]

    # Map the canonical return and momentum fields.
    for metric_name in ["return_1w", "momentum_21d", "momentum_63d"]:
        source_col = f"{asset_u}_{metric_name}"
        if source_col in out_df.columns:
            out_df[metric_name] = out_df[source_col]
        elif f"{asset.lower()}_{metric_name}" in out_df.columns:
            out_df[metric_name] = out_df[f"{asset.lower()}_{metric_name}"]

    return out_df
