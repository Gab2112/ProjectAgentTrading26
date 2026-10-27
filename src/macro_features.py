"""Macro-financial feature engineering for Level B."""

import pandas as pd

from src.macro_availability import align_daily_observation_proxy


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

def classify_macro_regimes(
    features: pd.DataFrame,
    min_history: int = 52,
    window_weeks: int | None = None,
) -> pd.DataFrame:
    """Experimental classifier. Not PIT-certified."""

    required = {"vix_level", "breakeven_10y"}
    if not required.issubset(features.columns):
        raise ValueError("Missing macro features")

    if min_history < 1:
        raise ValueError("Invalid minimum history")

    if window_weeks is not None and window_weeks < min_history:
        raise ValueError("Rolling window shorter than minimum history")

    result = pd.DataFrame(index=features.index)

    def past_median(column):
        past = features[column].shift(1)

        if window_weeks is None:
            return past.expanding(
                min_periods=min_history
            ).median()

        return past.rolling(
            window=window_weeks,
            min_periods=min_history
        ).median()

    result["vix_threshold"] = past_median("vix_level")

    result["breakeven_threshold"] = past_median(
        "breakeven_10y"
    )
    valid = result.notna().all(axis=1)

    result["stress_state"] = pd.Series(
        pd.NA, index=features.index, dtype="string"
    )
    result["inflation_comp_state"] = pd.Series(
        pd.NA, index=features.index, dtype="string"
    )

    result.loc[valid, "stress_state"] = (
        features.loc[valid, "vix_level"]
        .gt(result.loc[valid, "vix_threshold"])
        .map({True: "high", False: "low"})
    )

    result.loc[valid, "inflation_comp_state"] = (
        features.loc[valid, "breakeven_10y"]
        .gt(result.loc[valid, "breakeven_threshold"])
        .map({True: "high", False: "low"})
    )

    result["regime"] = result["stress_state"].str.cat(
        result["inflation_comp_state"], sep="_"
    )

    return result