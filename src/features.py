"""Costruzione delle feature del progetto Multi-Asset (unica fonte).

Tutte le variabili usate da strategie, clustering e agenti si calcolano qui,
così ogni modello vede le stesse definizioni:
- feature di mercato sulle vere date di decisione (``build_decision_feature_panel``);
- livelli macro per il livello B (``build_macro_feature_levels``);
- variabili del clustering (``build_cluster_features``);
- feature per asset degli agenti ML (``build_asset_features``);
- feature macro derivate degli agenti (``build_agent_macro_features``).

Nessuna funzione legge file: i dati arrivano da ``src/data_loader.py``.
"""

from collections.abc import Mapping
from typing import Optional

import numpy as np
import pandas as pd

from src.macro_availability import align_daily_macro_panel, align_daily_observation_proxy
from src.time_split import build_weekly_decision_dates, get_time_windows
from src.utils import get_release_lag_days


def build_decision_feature_panel(
    market_daily: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
    macro_daily: Mapping[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Market features known after each actual weekly decision Close.

    Uses actual trading-session dates, not W-FRI calendar labels. The returned
    panel includes adjusted Close, weekly realized return since the previous
    *supplied* decision, and 21/63-session momentum. It never reads market
    observations beyond the last decision. Any optional daily FRED inputs are
    date-lagged OBSERVATION PROXIES, not verified historical release vintages.

    This is a feature snapshot, NOT a future-return target or a trade decision.
    The next executable trading time remains the following session's Open.
    """
    if not isinstance(market_daily, pd.DataFrame) or market_daily.empty:
        raise ValueError("Expected a nonempty daily market DataFrame")
    if market_daily.columns.has_duplicates:
        raise ValueError("Duplicate daily market columns")
    if not isinstance(decision_dates, pd.DatetimeIndex) or decision_dates.empty:
        raise ValueError("Expected nonempty DatetimeIndex of decision dates")

    for name, idx in (("market_daily", market_daily.index),
                      ("decision_dates", decision_dates)):
        if (not isinstance(idx, pd.DatetimeIndex)
                or idx.tz is not None or idx.hasnans or idx.has_duplicates
                or not idx.is_monotonic_increasing
                or not idx.equals(idx.normalize())):
            raise ValueError(f"Invalid actual-session dates for {name}")

    if not decision_dates.isin(market_daily.index).all():
        raise ValueError("Decision dates must correspond to trading sessions")

    # A missing decision week would turn 'return_1w' into a multi-week return.
    # Using a Thu signal when Friday also traded would similarly mislabel
    # the week's close. Both situations are rejected instead of silently
    # manufacturing plausible but misdated ML features.
    weekly_dates = build_weekly_decision_dates(
        market_daily.loc[:decision_dates.max()]
    ).index
    expected = weekly_dates[(weekly_dates >= decision_dates.min())
                            & (weekly_dates <= decision_dates.max())]
    if not expected.equals(decision_dates):
        raise ValueError("Decision dates must contain each actual weekly close")

    close_cols = [c for c in market_daily.columns
                  if isinstance(c, str) and c.endswith("_close")]
    if not close_cols:
        raise ValueError("Missing adjusted ETF Close columns")

    # The explicit cutoff makes future-row changes incapable of affecting
    # feature values available at earlier decisions.
    closes = market_daily.loc[:decision_dates.max(), close_cols].astype(float)
    if not np.isfinite(closes.to_numpy()).all() or (closes <= 0).any().any():
        raise ValueError("Invalid adjusted Close prices")

    snapshot = closes.loc[decision_dates].copy()
    snapshot.index.name = "Date"
    weekly_return = snapshot.pct_change(fill_method=None)
    weekly_return.columns = [c[:-6] + "_return_1w" for c in close_cols]
    features = pd.concat([snapshot, weekly_return], axis=1)

    for sessions in (21, 63):
        momentum = closes.pct_change(periods=sessions, fill_method=None)
        momentum = momentum.loc[decision_dates]
        momentum.columns = [c[:-6] + f"_momentum_{sessions}d"
                            for c in close_cols]
        features = pd.concat([features, momentum], axis=1)

    if macro_daily is not None:
        # Observation dates are included for downstream temporal audits.
        macro = align_daily_macro_panel(decision_dates, macro_daily)
        features = features.join(macro, validate="one_to_one")

    return features


def build_training_decision_features(
    config: dict,
    market_daily: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
    macro_daily: Mapping[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Construct one train-only research panel with historical indicator warm-up.

    Explicitly excludes 2017+ validation and 2022+ final holdout from the
    returned rows. The input history can contain an earlier warm-up period.
    Macro inputs must be explicitly requested and are NOT PIT-certified.
    """
    if (config.get("evaluation", {}).get("evaluate_oos") is not False
            or config.get("validation", {}).get("development_only") is not True):
        raise ValueError("Development research features must stay train-only")

    windows = get_time_windows(config)
    start, end = windows["train_start"], windows["train_end"]
    if start is None or end is None:
        raise ValueError("Training boundaries are required")
    if not isinstance(decision_dates, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex of decision dates")

    selected = decision_dates[(decision_dates >= start) & (decision_dates <= end)]
    if selected.empty:
        raise ValueError("No training decisions available")
    return build_decision_feature_panel(
        market_daily.loc[:end], selected, macro_daily=macro_daily
    )


def build_macro_feature_levels(
    decision_dates: pd.DatetimeIndex,
    vix: pd.Series,
    breakeven_10y: pd.Series,
) -> pd.DataFrame:
    """Build candidate features. Not point-in-time verified."""

    vix_data = align_daily_observation_proxy(decision_dates, vix)

    breakeven_data = align_daily_observation_proxy(
        decision_dates, breakeven_10y
    )

    features = pd.DataFrame(index=vix_data.index)

    features["vix_level"] = vix_data["value"]
    features["breakeven_10y"] = breakeven_data["value"]

    # Preserve observation dates for temporal auditing.
    features["vix_observation_date"] = vix_data["observation_date"]
    features["breakeven_observation_date"] = (
        breakeven_data["observation_date"]
    )
    # Four-week changes, measured in index points
    # and percentage points, respectively.
    features["vix_change_4w"] = (
        features["vix_level"].diff(4)
    )

    features["breakeven_change_4w"] = (
        features["breakeven_10y"].diff(4)
    )
    return features


def rolling_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Quanto il valore attuale è alto o basso rispetto alle ultime `window` settimane."""
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / std


def build_cluster_features(weekly_yields: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """
    Costruisce le variabili del clustering sul calendario settimanale W-FRI.

    Tutte le trasformazioni guardano solo indietro (diff, rolling).
    """
    change = int(settings["change_weeks"])
    vol_window = int(settings["volatility_weeks"])
    z_window = int(settings["zscore_weeks"])
    z_min = int(settings["zscore_min_weeks"])

    y2 = weekly_yields["treasury_2y"]
    y10 = weekly_yields["treasury_10y"]
    curve = y10 - y2

    features = pd.DataFrame(index=weekly_yields.index)
    features["treasury_2y_change_13w"] = y2.diff(change)
    features["curve_10y_2y_change_13w"] = curve.diff(change)
    features["rate_volatility_13w"] = y10.diff().rolling(vol_window, min_periods=vol_window).std()
    features["treasury_2y_zscore_3y"] = rolling_zscore(y2, z_window, z_min)
    features["curve_10y_2y_zscore_3y"] = rolling_zscore(curve, z_window, z_min)

    return features


def build_asset_features(
    weekly_close: pd.DataFrame,
    target: str,
    macro_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Costruisce le variabili dell'agente per un ETF, alle date di decisione.

    Tutte le trasformazioni guardano solo indietro (pct_change, rolling).

    Variabili dell'ETF:
        ret_1w, mom_4w, mom_13w, mom_26w, mom_52w,
        vol_13w (deviazione standard dei rendimenti settimanali),
        dist_52w_high (distanza dal massimo delle ultime 52 settimane).
    Variabili degli altri ETF:
        <ALTRO>_mom_13w.
    Variabili macro (opzionali, dal modulo dei regimi):
        variazioni a 13 settimane di 2Y e curva, volatilità dei tassi.
    """
    if target not in weekly_close.columns:
        raise KeyError(f"ETF '{target}' non presente nei prezzi settimanali.")

    close = weekly_close[target].astype(float)
    weekly_ret = close.pct_change()

    features = pd.DataFrame(index=weekly_close.index)
    features["ret_1w"] = weekly_ret
    for weeks in (4, 13, 26, 52):
        features[f"mom_{weeks}w"] = close.pct_change(weeks)
    features["vol_13w"] = weekly_ret.rolling(13, min_periods=13).std()
    features["dist_52w_high"] = close / close.rolling(52, min_periods=52).max() - 1.0

    for other in weekly_close.columns:
        if other != target:
            features[f"{other}_mom_13w"] = weekly_close[other].astype(float).pct_change(13)

    if macro_features is not None:
        features = features.join(macro_features, how="left")

    return features


def build_agent_macro_features(
    daily_macro: pd.DataFrame,
    raw_series: Mapping[str, pd.Series],
    config: dict,
) -> pd.DataFrame:
    """Feature macro derivate usate dagli agenti (curve, variazioni, CPI, VIX, breakeven).

    ``daily_macro``: serie FRED giornaliere allineate (FEDFUNDS gia' ritardata);
    ``raw_series``: serie FRED grezze per ID (il CPI mensile si legge da qui);
    i ritardi di pubblicazione vengono da ``config`` (``get_release_lag_days``).
    """
    # Calcolo indicatori derivati (curve spread, variazioni, inflazione)
    calc_df = pd.DataFrame(index=daily_macro.index)

    # 1. Pendenze di curva
    if "DGS10" in daily_macro and "DGS2" in daily_macro:
        calc_df["curve_10y_2y"] = daily_macro["DGS10"] - daily_macro["DGS2"]
        calc_df["curve_10y_2y_change_1w"] = calc_df["curve_10y_2y"].diff(5)

    if "DGS10" in daily_macro and "DGS3MO" in daily_macro:
        calc_df["curve_10y_3m"] = daily_macro["DGS10"] - daily_macro["DGS3MO"]
        calc_df["curve_10y_3m_change_1w"] = calc_df["curve_10y_3m"].diff(5)

    if "DGS30" in daily_macro and "DGS10" in daily_macro:
        calc_df["curve_30y_10y"] = daily_macro["DGS30"] - daily_macro["DGS10"]
        calc_df["curve_30y_10y_change_1w"] = calc_df["curve_30y_10y"].diff(5)

    # 2. Variazioni tassi 1w (~5 sedute lavorative)
    if "DGS10" in daily_macro:
        calc_df["treasury_10y_change_1w"] = daily_macro["DGS10"].diff(5)
        calc_df["treasury_10y"] = daily_macro["DGS10"]
    if "DGS2" in daily_macro:
        calc_df["treasury_2y_change_1w"] = daily_macro["DGS2"].diff(5)
        calc_df["treasury_2y"] = daily_macro["DGS2"]
    if "DGS3MO" in daily_macro:
        calc_df["treasury_3m_change_1w"] = daily_macro["DGS3MO"].diff(5)

    # 3. Policy rate Fed
    if "FEDFUNDS" in daily_macro:
        calc_df["rates__policy_rate"] = daily_macro["FEDFUNDS"].ffill()

    # 4. Inflazione YoY (CPIAUCSL shifted con ritardo prudenziale di 45 giorni per evitare lookahead)
    if "CPIAUCSL" in raw_series:
        cpi_monthly = raw_series["CPIAUCSL"].dropna().sort_index()
        cpi_yoy = cpi_monthly.pct_change(12, fill_method=None)
        lag = get_release_lag_days(config, "CPIAUCSL")
        cpi_yoy.index = cpi_yoy.index + pd.Timedelta(days=lag)
        calc_df["cpi_yoy"] = cpi_yoy.reindex(calc_df.index, method="ffill")

    # 5. Volatilità VIX
    if "VIXCLS" in daily_macro:
        calc_df["vix_level"] = daily_macro["VIXCLS"]
        calc_df["vix_change_4w"] = daily_macro["VIXCLS"].diff(20)

    # 6. Breakeven inflation
    if "T10YIE" in daily_macro:
        calc_df["breakeven_10y"] = daily_macro["T10YIE"]
    if "T5YIE" in daily_macro:
        calc_df["breakeven_5y_change_1w"] = daily_macro["T5YIE"].diff(5)

    return calc_df
