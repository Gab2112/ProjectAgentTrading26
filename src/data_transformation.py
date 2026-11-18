"""Data preparation and validation (step 2 of the pipeline).

Input:  data/raw/market/<TICKER>.csv   daily OHLCV, split- and dividend-adjusted
        data/raw/macro/<SERIES>.csv    FRED series
Output: data/processed/market_daily.csv              aligned daily adjusted Open/Close
        data/processed/market_daily_quality.csv      coverage report per ETF
        data/processed/price_validation.csv          adjusted-price sanity checks
        data/processed/weekly_decision_dates.csv     last trading session of each week
        data/processed/master_weekly.csv             weekly market panel (OHLCV)
        data/processed/macro_weekly.csv              weekly macro panel, after publication lags
        data/processed/data_quality_summary.csv      coverage of the weekly panel
        data/processed/decision_features_training.csv  in-sample feature snapshot

Price validation. A backtest is meaningful only with prices adjusted for
splits and dividends. ``validate_adjusted_prices`` rejects any daily adjusted
return whose absolute value exceeds ``data.validation.max_abs_daily_return``,
because an unadjusted split would appear as a jump of that kind. Open and
Close must be adjusted consistently, as in Yahoo Finance downloads with
``auto_adjust=True``; the adjusted Open is used as a proxy for the executable
opening price.

Macroeconomic series. Monthly series are shifted by their conventional
publication lag before being aligned to weekly dates. The lags are
approximations and do not reconstruct historical data vintages.

This module does not generate portfolio weights or strategy returns.
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
from src.time_split import build_weekly_decision_dates
from src.utils import get_release_lag_days, get_tickers


LOW_FREQUENCY_SERIES = frozenset({"CPIAUCSL", "CPILFESL", "FEDFUNDS"})


def validate_adjusted_prices(market_daily: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Check that daily adjusted prices contain no split-like discontinuities.

    For every ETF the function reports the largest absolute close-to-close and
    close-to-open return. A value above the configured threshold stops the
    pipeline, since it would indicate an unadjusted corporate action or a data
    error that would generate false signals.
    """
    threshold = float(config["data"]["validation"]["max_abs_daily_return"])
    rows = []
    for ticker in get_tickers(config):
        close = market_daily[f"{ticker}_close"].astype(float)
        open_ = market_daily[f"{ticker}_open"].astype(float)
        close_to_close = close.pct_change(fill_method=None).dropna()
        close_to_open = (open_ / close.shift(1) - 1.0).dropna()
        worst_cc = close_to_close.abs().idxmax()
        worst_co = close_to_open.abs().idxmax()
        rows.append(
            {
                "ticker": ticker,
                "max_abs_close_to_close": float(close_to_close.abs().max()),
                "date_max_close_to_close": worst_cc.date().isoformat(),
                "max_abs_close_to_open": float(close_to_open.abs().max()),
                "date_max_close_to_open": worst_co.date().isoformat(),
                "threshold": threshold,
            }
        )

    report = pd.DataFrame(rows).set_index("ticker")
    report["passed"] = (
        (report["max_abs_close_to_close"] <= threshold)
        & (report["max_abs_close_to_open"] <= threshold)
    )
    if not report["passed"].all():
        failed = report.index[~report["passed"]].tolist()
        raise ValueError(
            f"Possible unadjusted prices for {failed}: a daily return exceeds "
            f"{threshold:.0%}. Inspect data/raw/market before proceeding."
        )
    return report


def transform_market_data(config: dict, root_dir: Path) -> Dict[str, pd.DataFrame]:
    """Aggregate daily OHLCV data into a weekly (W-FRI) descriptive panel.

    The weekly panel is used for descriptive analysis only; execution and
    portfolio accounting always use the daily data in market_daily.csv.
    """
    market_dir = root_dir / config["paths"]["raw_data"] / "market"
    weekly_market: Dict[str, pd.DataFrame] = {}
    print("\n[Data] Weekly market panel")

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
        weekly = df.resample("W-FRI").agg(aggregation_rules).dropna(subset=["Close"])
        weekly = weekly.rename(columns={col: f"{ticker}_{col.lower()}" for col in weekly.columns})
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


def transform_macro_data(config: dict, root_dir: Path) -> Dict[str, pd.DataFrame]:
    """Align FRED series to weekly dates after applying publication lags."""
    raw_dir = root_dir / config["paths"]["raw_data"] / "macro"
    transformed: Dict[str, pd.DataFrame] = {}

    for logical_name, series_id in _macro_series_mapping(config).items():
        df = coerce_numeric(read_csv_with_date_index(raw_dir / f"{series_id}.csv"))
        if series_id not in df.columns:
            if len(df.columns) != 1:
                raise KeyError(f"Cannot identify FRED series {series_id}")
            df.columns = [series_id]
        series = df[[series_id]].copy()

        if series_id in LOW_FREQUENCY_SERIES:
            # A monthly value is dated at the start of the reference month but
            # published several weeks later: shift it to its release date.
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
    """Merge the weekly market panel with the lagged macroeconomic series."""
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
        raise ValueError("No tradable close columns in the weekly panel")
    master = master.dropna(subset=close_cols)
    master.index.name = "Date"
    return master


def build_data_quality_summary(master: pd.DataFrame) -> pd.DataFrame:
    """Summarise coverage, missing values and first/last valid dates."""
    return pd.DataFrame(
        {
            "non_null": master.notna().sum(),
            "missing": master.isna().sum(),
            "missing_pct": master.isna().mean() * 100,
            "first_valid": [master[c].first_valid_index() for c in master],
            "last_valid": [master[c].last_valid_index() for c in master],
        }
    )


def run_data_transformation(config: dict, root_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Validate raw data and write every processed dataset to data/processed.

    The processed files cover the full sample (IS and OOS). The separation of
    periods is enforced downstream: training code truncates its inputs at
    ``validation.train_end`` and the OOS period is evaluated only once.
    """
    out_dir = root_dir / config["paths"]["processed_data"]
    out_dir.mkdir(parents=True, exist_ok=True)

    daily, daily_audit = load_daily_market_data(config, root_dir)
    end_date = config["data"].get("end_date")
    if end_date is not None:
        daily = daily.loc[:pd.Timestamp(end_date)]

    price_report = validate_adjusted_prices(daily, config)
    decision_dates = build_weekly_decision_dates(daily)
    weekly_master = build_master_dataset(transform_market_data(config, root_dir), {})
    weekly_master = weekly_master.loc[:daily.index.max()]
    macro_frames = list(transform_macro_data(config, root_dir).values())
    macro_weekly = pd.concat(macro_frames, axis=1).sort_index() if macro_frames else pd.DataFrame(index=weekly_master.index)
    macro_weekly = macro_weekly.loc[:daily.index.max()]
    macro_weekly.index.name = "Date"
    weekly_quality = build_data_quality_summary(weekly_master)
    decision_features = build_training_decision_features(config, daily, decision_dates.index)

    daily.to_csv(out_dir / "market_daily.csv")
    daily_audit.to_csv(out_dir / "market_daily_quality.csv")
    price_report.to_csv(out_dir / "price_validation.csv")
    decision_dates.to_csv(out_dir / "weekly_decision_dates.csv", index_label="Date")
    weekly_master.to_csv(out_dir / "master_weekly.csv")
    macro_weekly.to_csv(out_dir / "macro_weekly.csv")
    weekly_quality.to_csv(out_dir / "data_quality_summary.csv")
    decision_features.to_csv(out_dir / "decision_features_training.csv", index_label="Date")

    print("\n[Data] Processed datasets written to data/processed")
    print(f"  Daily sessions:     {len(daily):,} ({daily.index.min().date()} -> {daily.index.max().date()})")
    print(f"  Weekly decisions:   {len(decision_dates):,}")
    print(f"  Largest daily move: {price_report['max_abs_close_to_close'].max():.2%} "
          f"(threshold {price_report['threshold'].iloc[0]:.0%}); adjusted-price check passed")
    return weekly_master, weekly_quality


__all__ = [
    "build_data_quality_summary",
    "build_master_dataset",
    "run_data_transformation",
    "transform_macro_data",
    "transform_market_data",
    "validate_adjusted_prices",
]
