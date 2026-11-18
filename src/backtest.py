"""Backtest interface used by every strategy in the project.

All strategies, baselines and per-asset signal tests are simulated with the
same daily engine (``src/execution_engine.py``), under the assumptions
stated in ``config.yaml``:

- Timing. A target allocation decided with information available at the
  close of decision date t (the last trading session of the week) is
  executed at the OPEN of the next trading session. No price observed after
  the close of t is used to form the decision.
- Costs. Commission plus slippage, in basis points of traded notional, are
  charged on every rebalance. Short positions pay an annual borrowing fee
  (ACT/365) and must satisfy a simplified collateral requirement.
- Sizing. Weights are fractions of current NAV; the gross exposure never
  exceeds 100%, so no leverage is used. Unallocated capital is cash with a
  zero return.
- Returns. The NAV is marked to market at every close; the daily return is
  the change in NAV, net of every cost.

``run_signal_backtest`` applies the convention required by the project
specification to a single asset: a weight of +1 when the signal is +1
(long), 0 when it is 0 (flat) and -1 when it is -1 (short), so that the
strategy earns the asset return, zero, or minus the asset return respectively
(net of costs and of the one-session execution lag).
"""

from __future__ import annotations

import pandas as pd

from src.execution_engine import simulate_longshort_daily
from src.utils import ASSETS


def split_open_close(market_daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the daily adjusted Open and Close panels with columns ``ASSETS``."""
    opens = market_daily[[f"{a}_open" for a in ASSETS]].astype(float)
    closes = market_daily[[f"{a}_close" for a in ASSETS]].astype(float)
    opens.columns = list(ASSETS)
    closes.columns = list(ASSETS)
    return opens, closes


def run_weight_backtest(
    target_weights: pd.DataFrame,
    market_daily: pd.DataFrame,
    config: dict,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Simulate a portfolio of weekly target weights over [start, end].

    The simulation starts in cash on ``start``. Decisions dated before
    ``start`` are ignored, so that each evaluation period is an independent
    simulation. The first decision inside the window is executed at the
    following open.
    """
    execution = config["execution"]
    opens, closes = split_open_close(market_daily)
    mask = opens.index >= start
    if end is not None:
        mask &= opens.index <= end
    opens, closes = opens.loc[mask], closes.loc[mask]

    weights = target_weights.reindex(columns=list(ASSETS)).fillna(0.0)
    weights = weights.loc[(weights.index >= opens.index[0]) & (weights.index < opens.index[-1])]
    if weights.empty:
        raise ValueError("No decision falls inside the backtest window.")

    return simulate_longshort_daily(
        open_prices=opens,
        close_prices=closes,
        target_weights=weights,
        initial_capital=float(execution["initial_capital"]),
        commission_bps=float(execution["commission_bps"]),
        slippage_bps=float(execution["slippage_bps"]),
        annual_borrow_rate=float(execution["annual_borrow_rate"]),
        additional_margin_rate=float(execution["additional_margin_rate"]),
    )


def signals_to_single_asset_weights(signals: pd.Series, asset: str) -> pd.DataFrame:
    """Convert a {-1, 0, +1} signal series into full-NAV weights on one asset."""
    values = pd.to_numeric(signals, errors="raise")
    if not values.isin([-1, 0, 1]).all():
        raise ValueError("Signals must take values in {-1, 0, 1}.")
    weights = pd.DataFrame(0.0, index=values.index, columns=list(ASSETS))
    weights[asset] = values.astype(float)
    return weights


def run_signal_backtest(
    signals: pd.Series,
    asset: str,
    market_daily: pd.DataFrame,
    config: dict,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Backtest the discrete signal of one asset with the long/flat/short convention."""
    return run_weight_backtest(
        signals_to_single_asset_weights(signals, asset), market_daily, config, start, end
    )


def buy_and_hold_weights(decision_dates: pd.DatetimeIndex, asset: str) -> pd.DataFrame:
    """Weights of a buy-and-hold position: one initial purchase, never rebalanced."""
    weights = pd.DataFrame(0.0, index=decision_dates[:1], columns=list(ASSETS))
    weights[asset] = 1.0
    return weights
