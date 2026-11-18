"""Multi-asset agent strategy: signals, allocation and validation tuning.

Signal sources
- In-sample (IS): ``data/signals/signals_orchestrator_<asset>.csv``, written
  by the training step. The Random Forest and SVM inputs are out-of-fold
  predictions; the rule-based agents use the parameters selected by
  walk-forward on the IS period, and the Orchestrator was fitted on these
  same weeks. IS figures are therefore in-sample by construction.
- Out-of-sample (OOS): the trained models and parameters saved in
  ``src/agents/config`` are applied, without any refit, to the feature panel
  of every decision date. Only rows dated on or after ``oos_start`` are used.

Allocation rule (``MultiAssetSignalAggregator``)
At each weekly decision date t, for every asset with Orchestrator signal s
and confidence c:
- direction +1 (long) if s = +1 and c >= min_confidence;
- direction -1 (short) if s = -1, c >= min_confidence and allow_short;
- direction 0 otherwise;
- if spy_default_long, SPY is long unless its signal is a confirmed short
  (s = -1 with c >= min_confidence), in which case it is short when
  allow_short is true and flat otherwise.
Each active asset receives a weight of min(1 / n_active, max_weight_per_asset)
with the sign of its direction; the remainder is cash. Gross exposure is at
most 100%. A new target replaces the previous one only if some weight changes
by at least ``rebalance_threshold``, which limits turnover.

Validation tuning (``tune_allocation``)
``min_confidence``, ``max_weight_per_asset`` and ``allow_short`` are chosen
by grid search on the validation window (inside the IS period), maximising
the daily Sharpe ratio of the net-of-cost backtest. The OOS period is not
read by this function.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.orchestrator_agent import OrchestratorAgent
from src.backtest import run_weight_backtest
from src.data_loader import load_agent_macro_data
from src.evaluator import calculate_metrics
from src.features import build_decision_feature_panel
from src.time_split import period_bounds
from src.utils import ASSETS


class MultiAssetSignalAggregator:
    """Convert per-asset Orchestrator signals into portfolio target weights."""

    def __init__(
        self,
        min_confidence: float,
        max_weight_per_asset: float,
        allow_short: bool,
        spy_default_long: bool,
        rebalance_threshold: float,
    ):
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must lie in [0, 1].")
        if not 0.0 < max_weight_per_asset <= 1.0:
            raise ValueError("max_weight_per_asset must lie in (0, 1].")
        self.min_confidence = float(min_confidence)
        self.max_weight_per_asset = float(max_weight_per_asset)
        self.allow_short = bool(allow_short)
        self.spy_default_long = bool(spy_default_long)
        self.rebalance_threshold = float(rebalance_threshold)

    def directions(self, signals: pd.DataFrame, confidences: pd.DataFrame) -> pd.DataFrame:
        """Direction in {-1, 0, +1} of every asset at every decision date."""
        confirmed = confidences[list(ASSETS)] >= self.min_confidence
        sig = signals[list(ASSETS)]
        long_ = (sig == 1) & confirmed
        short = (sig == -1) & confirmed
        directions = long_.astype(int) - (short & self.allow_short).astype(int)

        if self.spy_default_long:
            spy_short = short["SPY"]
            directions["SPY"] = np.where(spy_short, -1 if self.allow_short else 0, 1)
        return directions.astype(int)

    def target_weights(self, signals: pd.DataFrame, confidences: pd.DataFrame) -> pd.DataFrame:
        """Weekly target weights with the rebalancing threshold applied."""
        directions = self.directions(signals, confidences)
        n_active = directions.abs().sum(axis=1)
        size = np.minimum(1.0 / n_active.replace(0, np.nan), self.max_weight_per_asset).fillna(0.0)
        raw = directions.mul(size, axis=0).astype(float)

        held = pd.DataFrame(0.0, index=raw.index, columns=list(ASSETS))
        current = pd.Series(0.0, index=list(ASSETS))
        for date in raw.index:
            target = raw.loc[date]
            if (target - current).abs().max() >= self.rebalance_threshold or date == raw.index[0]:
                current = target
            held.loc[date] = current
        return held


# =============================================================================
# Signal sources
# =============================================================================


def load_in_sample_signals(config: dict, root_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Orchestrator signals and confidences saved by the training step (IS only)."""
    signals_dir = root_dir / config["paths"]["signals"]
    signals, confidences = {}, {}
    for asset in ASSETS:
        path = signals_dir / f"signals_orchestrator_{asset.lower()}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found: run the training step (pipeline.train_agents: true).")
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        signals[asset] = frame["signal"].astype(int)
        confidences[asset] = frame["confidence"].astype(float)

    signals_df, conf_df = pd.DataFrame(signals).dropna(), pd.DataFrame(confidences).dropna()
    _, train_end = period_bounds(config, "in_sample")
    if signals_df.index.max() > train_end:
        raise ValueError("The in-sample signal files contain dates after train_end.")
    return signals_df, conf_df


def build_feature_panel(market_daily: pd.DataFrame, decision_dates: pd.DatetimeIndex, root_dir: Path) -> pd.DataFrame:
    """Market and macro features on every decision date (backward-looking only)."""
    panel = build_decision_feature_panel(market_daily, decision_dates)
    return panel.join(load_agent_macro_data(decision_dates=panel.index, root_dir=root_dir), how="left")


def generate_signals_with_trained_models(
    feature_panel: pd.DataFrame,
    config: dict,
    root_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the saved agents and Orchestrator (no refit) to every decision date."""
    models_dir = root_dir / config["paths"]["models"]
    signals, confidences = {}, {}
    for asset in ASSETS:
        with (models_dir / f"orchestrator_params_{asset.lower()}.json").open("r", encoding="utf-8") as fh:
            params = json.load(fh)["best_params"]
        orchestrator = OrchestratorAgent(target_asset=asset)
        orchestrator._apply_params(params)
        combined = orchestrator.combine(orchestrator._build_signals_pool_from_df(feature_panel))
        signals[asset] = combined["signal"].astype(int)
        confidences[asset] = combined["confidence"].astype(float)
    return pd.DataFrame(signals), pd.DataFrame(confidences)


# =============================================================================
# Validation tuning of the allocation parameters
# =============================================================================


def make_aggregator(config: dict, params: dict) -> MultiAssetSignalAggregator:
    allocation = config["allocation"]
    return MultiAssetSignalAggregator(
        min_confidence=float(params["min_confidence"]),
        max_weight_per_asset=float(params["max_weight_per_asset"]),
        allow_short=bool(params["allow_short"]),
        spy_default_long=bool(allocation["spy_default_long"]),
        rebalance_threshold=float(allocation["rebalance_threshold"]),
    )


def tune_allocation(
    config: dict,
    signals: pd.DataFrame,
    confidences: pd.DataFrame,
    market_daily: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Grid search of the allocation parameters on the validation window."""
    start, end = period_bounds(config, "validation")
    mask = (signals.index >= start) & (signals.index <= end)
    if not mask.any():
        raise ValueError("No in-sample signal falls inside the validation window.")
    sig, conf = signals.loc[mask], confidences.loc[mask]

    evaluation = config["evaluation"]
    metric = config["allocation"]["selection_metric"]
    grid = config["allocation"]["grid"]
    rows = []
    for min_conf, cap, allow_short in itertools.product(
        grid["min_confidence"], grid["max_weight_per_asset"], grid["allow_short"]
    ):
        params = {"min_confidence": min_conf, "max_weight_per_asset": cap, "allow_short": allow_short}
        weights = make_aggregator(config, params).target_weights(sig, conf)
        nav = run_weight_backtest(weights, market_daily, config, start=sig.index[0], end=end)
        metrics = calculate_metrics(
            nav["daily_return"], evaluation["risk_free_rate"], evaluation["annualization_factor"]
        )
        rows.append({**params, **metrics})

    results = pd.DataFrame(rows)
    # Ties are resolved in grid order, which lists the more conservative values first.
    best = results.loc[results[metric].idxmax()]
    selected = {
        "min_confidence": float(best["min_confidence"]),
        "max_weight_per_asset": float(best["max_weight_per_asset"]),
        "allow_short": bool(best["allow_short"]),
        "validation_start": str(sig.index[0].date()),
        "validation_end": str(end.date()),
        "selection_metric": metric,
        "validation_value": float(best[metric]),
    }
    return selected, results
