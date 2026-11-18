"""Interest-rate features consumed by the SVM agent.

The variables are computed in ``src/features.build_cluster_features`` from
weekly Treasury yields and use only information available at each weekly
decision date:
- 13-week change of the 2Y yield and of the 10Y-2Y curve;
- 13-week volatility of weekly changes in the 10Y yield;
- 3-year z-scores of the 2Y yield and of the 10Y-2Y curve.

Window lengths are read from ``agents.rate_features`` in ``config.yaml``.
"""

from __future__ import annotations

import pandas as pd


def get_rate_feature_settings(config: dict) -> dict:
    """Read and check the window lengths of the rate features."""
    settings = config.get("agents", {}).get("rate_features")
    if not isinstance(settings, dict):
        raise KeyError("Section 'agents.rate_features' is missing from config.yaml.")
    required = ("change_weeks", "volatility_weeks", "zscore_weeks", "zscore_min_weeks")
    missing = [key for key in required if key not in settings]
    if missing:
        raise KeyError("Missing rate-feature settings: " + ", ".join(missing))
    return settings


def align_to_decision_dates(weekly_features: pd.DataFrame, decision_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Attach to each decision date (last session of the week) the values of its W-FRI week."""
    week_labels = decision_dates.to_period("W-FRI").end_time.normalize()
    aligned = weekly_features.reindex(week_labels)
    aligned.index = decision_dates
    aligned.index.name = "Date"
    return aligned
