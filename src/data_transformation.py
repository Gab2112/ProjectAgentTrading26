"""Preparation of market and macroeconomic data for the multi-asset project.

Responsibilities
----------------
1. Orchestrate data preparation: files are read in ``src/data_loader.py``
   and features are computed in ``src/features.py``.
2. Store an aligned DAILY panel of adjusted Open/Close prices for the
   Level A1 and A2 backtests.
3. Store the W-FRI research panel and, separately, the features indexed
   on the ACTUAL decision dates (last trading session of each week).
4. Exclude macroeconomic data during A1/A2, so that the benchmarks are
   independent of FRED series.
5. Exclude OOS observations from processed files until they are authorised.
6. Export market-only TRAINING features on the actual decision dates.
   Optional macro series require explicit date alignment.

Input:  data/raw/market/<TICKER>.csv; optionally data/raw/macro/*.csv
Output: data/processed/market_daily.csv
        data/processed/weekly_decision_dates.csv
        data/processed/market_daily_quality.csv
        data/processed/master_weekly.csv
        data/processed/data_quality_summary.csv
        data/processed/decision_features_training.csv

Note: Open and Close prices must be adjusted consistently, as in Yahoo
downloads with ``auto_adjust=True``. Adjusted opening prices are a proxy
for the backtest, not necessarily executable historical quotes.

This module does NOT generate portfolio weights and does NOT compute
returns: those responsibilities belong to the strategy and backtesting modules.
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.data_loader import (
    coerce_numeric,
    load_daily_market_data,
    read_csv_with_date_index,
    read_etf_prices,
)
from src.features import build_training_decision_features
from src.time_split import build_weekly_decision_dates, restrict_to_allowed_period
from src.utils import get_release_lag_days, get_tickers


LOW_FREQUENCY_SERIES = frozenset({"CPIAUCSL", "CPILFESL", "FEDFUNDS"})


def transform_market_data(
    config: dict,
    root_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """Aggregate daily OHLCV data into the historical weekly panel.

    Weekly OHLCV series are used for analysis, NOT for executing orders at the
    next open or estimating the 126-session covariance. A1 and A2 must use
    market_daily.csv instead.
    """
    market_dir = root_dir / config["paths"]["raw_data"] / "market"
    weekly_market: Dict[str, pd.DataFrame] = {}
    print("\n[Data Transformation] Weekly market data")

    for ticker in get_tickers(config):
        df = read_etf_prices(market_dir / f"{ticker}.csv", ticker)
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
    """Map logical names to FRED series identifiers."""
    mapping: Dict[str, str] = {}
    for section in ("rates", "inflation", "stress"):
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
    """Prepare historical macroeconomic series for Level B.

    WARNING: fixed publication lags are only approximations. They do not
    reconstruct historical data vintages or exact release times. Level B must
    not treat these variables as verified point-in-time features.
    """
    raw_dir = root_dir / config["paths"]["raw_data"] / "macro"
    transformed: Dict[str, pd.DataFrame] = {}

    for logical_name, series_id in _macro_series_mapping(config).items():
        df = coerce_numeric(read_csv_with_date_index(raw_dir / f"{series_id}.csv"))
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
    """Merge weekly market data with the optional macroeconomic variables."""
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
    """Summarise coverage, missing values and valid dates of the weekly dataset."""
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
    """Prepare the daily backtest data and the weekly dataset.

    Returns the pair ``(weekly_master, quality_summary)``, keeping the public
    interface unchanged.
    """
    out_dir = root_dir / config["paths"]["processed_data"]
    out_dir.mkdir(parents=True, exist_ok=True)

    level = str(config.get("pipeline", {}).get("active_level", "A1")).upper()
    if level not in {"A1", "A2", "B", "C"}:
        raise ValueError(f"Unknown pipeline level: {level}")

    # Levels B/C require a separate, explicit integration step.
    # The current macro transformation uses estimated lags, not verified historical vintages.
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
    decision_features = build_training_decision_features(
        config, daily, decision_dates.index
    )

    daily.to_csv(out_dir / "market_daily.csv")
    decision_dates.to_csv(out_dir / "weekly_decision_dates.csv", index_label="Date")
    daily_audit.to_csv(out_dir / "market_daily_quality.csv")
    weekly_master.to_csv(out_dir / "master_weekly.csv")
    weekly_quality.to_csv(out_dir / "data_quality_summary.csv")
    decision_features.to_csv(out_dir / "decision_features_training.csv", index_label="Date")

    print("\n[Data Transformation] Multi-Asset data ready")
    print(f"  Daily sessions:   {len(daily):,}")
    print(f"  Weekly decisions: {len(decision_dates):,}")
    print(f"  Weekly rows:      {len(weekly_master):,}")
    print(f"  Daily output:     {out_dir / 'market_daily.csv'}")
    print(f"  Weekly output:    {out_dir / 'master_weekly.csv'}")
    print(f"  Decision features:{out_dir / 'decision_features_training.csv'}")
    print("  OOS disabled:     " + str(not config.get("evaluation", {}).get("evaluate_oos", False)))
    return weekly_master, weekly_quality


