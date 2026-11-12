"""Read-only consistency guards for the historical frozen Group 9 release.

This adapter never edits the frozen source or the model training outputs.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import yaml

from src.frozen_satellite_selector import generate_dynamic_growth_matrix
from src.data_loader import load_dataset, load_daily_market_data
from src.time_split import build_weekly_decision_dates
from src.utils import ASSETS, ROOT_DIR, load_config


def verify_release_contract(root: Path = ROOT_DIR) -> dict:
    root = Path(root)
    spec = yaml.safe_load((root / 'release_config.yaml').read_text())
    old = load_config(root / 'config.yaml')
    frozen = __import__('json').loads((root/'reports/development_freeze_2021.json').read_text())
    portfolio = spec['final_portfolio']
    selector = portfolio['selection']
    execution = spec['execution']
    assert portfolio['core_asset'] == 'SPY'
    assert float(portfolio['core_weight']) == float(frozen['core_weight']) == 0.35
    assert float(portfolio['satellite_weight']) == float(frozen['satellite_weight']) == 0.65
    assert tuple(portfolio['satellite_tie_order']) == ('SHY','IEF','GLD','SPY')
    assert selector['minimum_confidence'] == 0.35
    assert selector['hysteresis_buffer'] == 0.015
    assert selector['fallback_volatility'] == 0.10
    assert selector['spy_trend_sma_weekly_observations'] == 30
    assert selector['spy_trend_score_multiplier_min'] == 1.2
    assert selector['spy_trend_score_multiplier_max'] == 2.5
    assert old['validation']['train_end'] == frozen['training_end']
    assert old['validation']['development_only'] is True
    assert old['evaluation']['evaluate_oos'] is False
    assert old['strategies']['C']['sentiment_enabled'] is False
    assert float(execution['starting_usd']) == float(old['execution']['initial_capital'])
    assert float(execution['commission_bps']) == float(old['execution']['commission_bps'])
    assert float(execution['slippage_bps']) == float(old['execution']['slippage_bps'])
    assert float(execution['borrow_rate_annual']) == float(old['strategies']['B']['annual_borrow_rate'])
    assert float(execution['short_margin_buffer']) == float(old['strategies']['B']['additional_margin_rate'])
    assert spec['metrics']['annualization_trading_days'] == old['evaluation']['annualization_factor']
    assert spec['metrics']['assumed_risk_free_rate'] == old['evaluation']['risk_free_rate']
    assert spec['release']['development_end'] == frozen['training_end']
    # These policy values are frozen in implementation; detecting drift avoids
    # treating this declarative contract as editable model hyperparameters.
    source = inspect.getsource(generate_dynamic_growth_matrix)
    checks = (
        'rolling(window=30)',
        'min(2.5, max(1.2,',
        "sig[(sig == -1) & bullish_trend] = 0",
    )
    if not all(marker in source for marker in checks):
        raise RuntimeError('Frozen selector does not match release contract')
    return spec


def verify_signal_calendar(root: Path = ROOT_DIR, *, oos: bool = False) -> int:
    root = Path(root)
    original = load_config(root/'config.yaml')
    if oos:
        full, _ = load_daily_market_data(original, root)
        market = full
    else:
        full = load_dataset(root/'data/processed/market_daily.csv', strict=True)
        market = full.loc[:pd.Timestamp(original['validation']['train_end'])]
    weekly = build_weekly_decision_dates(market).index
    sources = root/('reports/oos_signals' if oos else 'data/signals')
    prefixes = []
    for asset in ASSETS:
        path = sources/f'signals_orchestrator_{asset.lower()}.csv'
        frame = load_dataset(path, strict=True)
        if frame.empty or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError(f'{path}: invalid calendar')
        prefixes.append(frame.index)
    start = max(idx.min() for idx in prefixes)
    end = min(idx.max() for idx in prefixes)
    if any(idx.min() != start or idx.max() != end for idx in prefixes):
        raise ValueError(f'Signal time-span mismatch: {sources}')
    required = weekly[(weekly >= start) & (weekly <= end)]
    if oos:
        first = pd.Timestamp(original['validation']['oos_start'])
        required = required[required >= first]
        if required.empty or required.min() != start or end != weekly[-1]:
            raise ValueError('OOS signal boundaries are incomplete')
    if not len(required):
        raise ValueError('No signal dates in the selected period')
    for asset, dates in zip(ASSETS, prefixes):
        extra = dates.difference(required)
        missing = required.difference(dates)
        if len(extra) or len(missing):
            raise ValueError(f'{asset}: missing {len(missing)} weekly decisions, extra {len(extra)}')
    return len(required)
