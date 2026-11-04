"""Training-only execution diagnostic for the four-ETF agent ensemble.

This adapter is deliberately separate from A1/A2 and from agent training.
It converts weekly post-close agent signals into target weights and delegates
trades and cash accounting to the existing daily next-open backtest engine.

IMPORTANT: Orchestrator signals fitted on the same training period are
in-sample diagnostics, even when their RF/SVM inputs are OOF.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.execution_engine import (
    schedule_next_open_executions,
    simulate_longshort_daily,
)
from src.utils import ASSETS


def _validate_dates(index: pd.Index, name: str) -> None:
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.empty
        or index.hasnans
        or index.has_duplicates
        or index.tz is not None
        or not index.is_monotonic_increasing
        or not index.normalize().equals(index)
    ):
        raise ValueError(f"{name}: date mancanti, duplicate o non ordinate.")


def signals_to_target_weights(
    signals_by_asset: Mapping[str, pd.DataFrame],
    asset_budgets: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Convert weekly (signal, confidence) to long/flat portfolio targets.

    Under this diagnostic convention SELL (-1) means exit to cash, NOT short.
    Each asset receives its explicit budget (default 25% of NAV).
    Missing dates are never silently converted to HOLD.
    """
    if not isinstance(signals_by_asset, Mapping) or not signals_by_asset:
        raise ValueError("Servono segnali per almeno un asset.")
    if set(signals_by_asset) - set(ASSETS):
        raise ValueError("L'adattatore supporta soltanto SPY, IEF, GLD e SHY.")

    if asset_budgets is None:
        budgets = {asset: 1.0 / len(ASSETS) for asset in ASSETS}
    else:
        if not isinstance(asset_budgets, Mapping) or set(asset_budgets) - set(ASSETS):
            raise ValueError("Budget per asset non valido.")
        budgets = {asset: float(asset_budgets.get(asset, 0.0)) for asset in ASSETS}

    budget_values = np.asarray(list(budgets.values()), dtype=float)
    if (
        not np.isfinite(budget_values).all()
        or (budget_values < 0.0).any()
        or budget_values.sum() <= 0.0
        or budget_values.sum() > 1.0 + 1e-10
    ):
        raise ValueError("I budget devono essere non negativi e avere somma in (0, 1].")

    common_dates: pd.DatetimeIndex | None = None
    targets: pd.DataFrame | None = None

    for asset, frame in signals_by_asset.items():
        if not isinstance(frame, pd.DataFrame) or not {'signal', 'confidence'}.issubset(frame.columns):
            raise ValueError(f"{asset}: servono colonne signal e confidence.")
        _validate_dates(frame.index, f"Segnali {asset}")
        if common_dates is None:
            common_dates = frame.index
            targets = pd.DataFrame(0.0, index=common_dates, columns=ASSETS)
        elif not frame.index.equals(common_dates):
            raise ValueError("Gli asset devono avere le stesse date di decisione; nessun fill automatico.")

        signals = pd.to_numeric(frame['signal'], errors='raise').to_numpy(dtype=float)
        confidences = pd.to_numeric(frame['confidence'], errors='raise').to_numpy(dtype=float)
        if not np.isfinite(signals).all() or not np.isin(signals, [-1.0, 0.0, 1.0]).all():
            raise ValueError(f"{asset}: segnali non validi.")
        if not np.isfinite(confidences).all() or ((confidences < 0) | (confidences > 1)).any():
            raise ValueError(f"{asset}: confidenza fuori dall'intervallo [0,1].")

        # Long/flat: a SELL liquidates the position. No implicit short-selling.
        targets.loc[:, asset] = np.maximum(signals, 0.0) * confidences * budgets[asset]

    assert targets is not None
    if (targets.abs().sum(axis=1) > 1.0 + 1e-10).any():
        raise ValueError("Il target supera il capitale disponibile.")
    targets.index = pd.DatetimeIndex(targets.index, name='Date')
    return targets


def run_agent_training_execution_diagnostic(
    config: dict,
    market_daily: pd.DataFrame,
    signals_by_asset: Mapping[str, pd.DataFrame],
    asset_budgets: Mapping[str, float] | None = None,
) -> dict:
    """Simulate agent targets at next market Open, only through train_end.

    Returns {target_weights, execution_schedule, daily_backtest, sample}.
    Does NOT optimize parameters, inspect 2017+, or modify existing backtests.
    """
    if config.get('validation', {}).get('development_only') is not True:
        raise ValueError('Diagnostica consentita soltanto in development_only.')
    if config.get('evaluation', {}).get('evaluate_oos') is not False:
        raise ValueError('Non e consentito aprire il periodo OOS.')
    execution = config.get('execution', {})
    if (execution.get('execution_rule') != 'next_available_trading_open'
            or execution.get('signal_frequency') != 'weekly'
            or execution.get('cost_basis') != 'traded_notional'):
        raise ValueError('La convenzione di esecuzione non coincide con config.yaml.')

    validation = config.get('validation', {})
    train_start = pd.Timestamp(validation['train_start'])
    train_end = pd.Timestamp(validation['train_end'])
    if train_start > train_end:
        raise ValueError('Intervallo di training non valido.')

    targets = signals_to_target_weights(signals_by_asset, asset_budgets)
    if targets.index.min() < train_start or targets.index.max() > train_end:
        raise ValueError('Segnali fuori dal training autorizzato.')

    if not isinstance(market_daily, pd.DataFrame):
        raise TypeError('market_daily deve essere un DataFrame.')
    _validate_dates(market_daily.index, 'market_daily')
    cols = [f'{asset}_{field}' for field in ('open', 'close') for asset in ASSETS]
    if not set(cols).issubset(market_daily.columns):
        raise KeyError('Mancano colonne Open/Close degli ETF nel panel.')

    # Do not allow warm-up or post-cutoff rows into the accounting window.
    daily = market_daily.loc[
        (market_daily.index >= targets.index.min())
        & (market_daily.index <= train_end), cols
    ].copy()
    if len(daily) < 2:
        raise ValueError('Meno di due sedute disponibili per eseguire i segnali.')
    if not targets.index.isin(daily.index).all():
        raise ValueError('Una data di decisione non corrisponde ad una seduta.')

    frames = {}
    for field in ('open', 'close'):
        values = daily[[f'{asset}_{field}' for asset in ASSETS]].copy()
        values.columns = list(ASSETS)
        values = values.apply(pd.to_numeric, errors='raise')
        array = values.to_numpy(dtype=float)
        if not np.isfinite(array).all() or (array <= 0.0).any():
            raise ValueError(f'Prezzi {field} mancanti, non finiti o non positivi.')
        frames[field] = values

    capital = float(execution['initial_capital'])
    commission = float(execution['commission_bps'])
    slippage = float(execution['slippage_bps'])
    if (
        not np.isfinite([capital, commission, slippage]).all()
        or capital <= 0.0 or commission < 0.0 or slippage < 0.0
    ):
        raise ValueError('Parametri economici di esecuzione non validi.')

    # A1/A2 and the shared executor remain untouched.
    execution_schedule = schedule_next_open_executions(targets, frames['open'].index)
    daily_backtest = simulate_longshort_daily(
        open_prices=frames['open'],
        close_prices=frames['close'],
        target_weights=targets,
        initial_capital=capital,
        commission_bps=commission,
        slippage_bps=slippage,
        annual_borrow_rate=0.0,  # Long/flat, no shorts.
    )
    return {
        'target_weights': targets,
        'execution_schedule': execution_schedule,
        'daily_backtest': daily_backtest,
        'sample': 'training_diagnostic_only_not_independent_oos',
    }
