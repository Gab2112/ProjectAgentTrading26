"""Compare final policy candidates with a 2021-cutoff development-only protocol.

The inputs are the already-trained 2010--2021 GA signals and the pre-2022 market
history. This is a *reconstructed* development-only selection calculation,
not evidence that the candidate set was historically registered in 2021.
The original, immutable Core-Satellite freeze is retained separately.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data_loader import load_daily_market_data
from src.evaluator import calculate_metrics
from src.execution_engine import simulate_longshort_daily
from src.research_strategy_comparison import (
    OOS_BEGIN, OOS_END, build_comparison, equal_active_targets, sma200_targets,
)
from src.utils import ASSETS, ROOT_DIR, load_config

DEVELOPMENT_BEGIN = '2010-01-08'
DEVELOPMENT_END = '2021-12-31'
POLICY_NAMES = ('Core35_Satellite65', 'Equal_Active_4ETF', 'Regime_Adaptive_SMA200')


def _check_provenance(root: Path) -> None:
    manifest = json.loads((root/'reports/development_freeze_2021.json').read_text())
    if (manifest['training_end'] != DEVELOPMENT_END or
            manifest['selected_strategy'] != 'Core35_Satellite65' or
            manifest['status'] != 'FROZEN_BEFORE_FINAL_OOS'):
        raise RuntimeError('Original 2021 frozen-model provenance not recognized')
    if pd.Timestamp(DEVELOPMENT_END) >= pd.Timestamp(OOS_BEGIN):
        raise ValueError('Overlapping development and post-2021 calendars')
    study = yaml.safe_load((root/'final_policy_config.yaml').read_text())
    if (study['original_frozen_reference'] != 'Core35_Satellite65'
            or tuple(study['concluding_allocation_policies']) !=
            ('Equal_Active_4ETF', 'Regime_Adaptive_SMA200')
            or study['development_window']['last_weekly_decision'] != DEVELOPMENT_END
            or study['historical_evaluation_window']['first_weekly_decision'] != OOS_BEGIN
            or study['development_window']['ranking_metric'] != 'sharpe_ratio'
            or study['original_2021_freeze_not_rewritten'] is not True):
        raise RuntimeError('Final policy-study configuration does not match research contract')


def evaluate_development(root: Path = ROOT_DIR) -> pd.DataFrame:
    """Calculate fixed-policy performance using no 2022+ signals or prices.

    GA ensemble parameters already have their historical fitting cutoff in 2021;
    this routine does *not* refit the agents or tune parameters on 2022--2026.
    Scores are development-period diagnostics, not independent validation.
    """
    root = Path(root)
    _check_provenance(root)
    cfg = load_config(root/'config.yaml')
    market, _ = load_daily_market_data(cfg, root)
    # Strictly restrict every price input to the end of 2021. In particular,
    # the regime indicator cannot access future 2022--2026 closes.
    market = market.loc[:DEVELOPMENT_END].copy()
    folder = root/'data/signals'
    signals = {}
    for asset in ASSETS:
        frame = pd.read_csv(folder/f'signals_orchestrator_{asset.lower()}.csv',
                            index_col='Date', parse_dates=['Date'])
        if frame.empty or frame.index.max() > pd.Timestamp(DEVELOPMENT_END):
            raise ValueError(f'{asset}: missing or future-dated development signals')
        if frame.index.min() != pd.Timestamp(DEVELOPMENT_BEGIN):
            raise ValueError(f'{asset}: unexpected development-signal start')
        signals[asset] = frame
    variants = sma200_targets(market, signals, DEVELOPMENT_BEGIN, DEVELOPMENT_END)
    idx = variants['original'].index
    if (len(idx) != 626 or idx[-1] > pd.Timestamp(DEVELOPMENT_END)
            or not all(frame.index.equals(idx) for frame in signals.values())):
        raise RuntimeError('Development decision-calendar validation failed')
    targets = {
        'Core35_Satellite65': variants['original'],
        'Equal_Active_4ETF': equal_active_targets(signals, tuple(ASSETS)),
        'Regime_Adaptive_SMA200': variants['regime_adaptive_sma200'],
    }
    window = market.loc[idx.min():idx.max()]
    opens = window[[f'{asset}_open' for asset in ASSETS]].copy()
    closes = window[[f'{asset}_close' for asset in ASSETS]].copy()
    opens.columns = closes.columns = list(ASSETS)
    execution = cfg['execution']; strategy_b = cfg['strategies']['B']
    kwargs = dict(initial_capital=float(execution['initial_capital']),
                  commission_bps=float(execution['commission_bps']),
                  slippage_bps=float(execution['slippage_bps']),
                  annual_borrow_rate=float(strategy_b['annual_borrow_rate']),
                  additional_margin_rate=float(strategy_b['additional_margin_rate']))
    rows = []
    for name, target in targets.items():
        ledger = simulate_longshort_daily(opens, closes, target, **kwargs)
        metrics = calculate_metrics(ledger['daily_return'], 0.0, 252)
        rows.append(dict(strategy=name, period='development_2010_2021',
                         first_decision=str(idx.min().date()),
                         last_decision=str(idx.max().date()),
                         decisions=len(idx),
                         cagr=float(metrics['cagr']),
                         sharpe_ratio=float(metrics['sharpe_ratio']),
                         max_drawdown=float(metrics['max_drawdown']),
                         annualized_volatility=float(metrics['annualized_volatility']),
                         mean_gross_exposure=float(ledger['gross_exposure'].mean())))
    return pd.DataFrame(rows).set_index('strategy')


def evaluate_final_policy_comparison(root: Path = ROOT_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return both period tables, including the archived 2022--2026 replay.

    The candidate set and the research comparison were reviewed after the
    historical OOS; this method does not transform that inspection into a
    contemporary, independently registered 2021 policy-selection decision.
    """
    dev = evaluate_development(root)
    later, _, _, _ = build_comparison(root)
    later = later.loc[list(POLICY_NAMES),
                      ['cagr', 'sharpe_ratio', 'max_drawdown',
                       'annualized_volatility', 'mean_gross_exposure']].copy()
    later['period'] = 'historical_evaluation_2022_2026'
    return dev, later


def print_final_policy_report(root: Path = ROOT_DIR) -> None:
    development, history = evaluate_final_policy_comparison(root)
    print('\n=== GROUP 9 | FINAL POLICY RESEARCH COMPARISON ===')
    print('The original Core-Satellite freeze is the protected historical baseline.')
    print('Final comparative policies: Equal Active 4 ETF and Regime-Adaptive SMA200.')
    print('\nDevelopment-only 2010-2021 (GA models already fitted by 2021):')
    print((development[['cagr','sharpe_ratio','max_drawdown']].assign(
        cagr=lambda x:x.cagr*100, max_drawdown=lambda x:x.max_drawdown*100
    )).round(3).to_string())
    winner = development['sharpe_ratio'].idxmax()
    print(f'\nBest 2010-2021 development Sharpe within the three policies: {winner}')
    print('\nMatched 2022-2026 historical evaluation:')
    print((history[['cagr','sharpe_ratio','max_drawdown']].assign(
        cagr=lambda x:x.cagr*100, max_drawdown=lambda x:x.max_drawdown*100
    )).round(3).to_string())
    print('\nMethod: fixed, previously trained agent signals; next-open execution and fees.')
    print('Provenance: the 2021 historical manifest registers Core-Satellite only;')
    print('the comparative policy analysis was reconstructed in November 2026.')
