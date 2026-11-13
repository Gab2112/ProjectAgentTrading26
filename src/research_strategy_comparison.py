"""Non-frozen, post-selection research: Equal Active and SMA200 alternatives.

This module intentionally lives outside the original 47-file scientific freeze.
All frozen sources, model artifacts and official OOS results are read only.
The 2022-2026 comparisons are retrospective research, NOT a new independent OOS.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.agent_backtest import generate_dynamic_growth_matrix
from src.core_satellite import SATELLITE_ORDER, build_core_satellite_targets
from src.data_loader import load_daily_market_data
from src.evaluator import calculate_metrics
from src.execution_engine import simulate_longshort_daily
from src.regime_allocator import build_trend_signal_panel
from src.time_split import build_weekly_decision_dates
from src.utils import ASSETS, ROOT_DIR, load_config

OOS_BEGIN = '2022-01-07'
OOS_END = '2026-10-09'


def equal_active_targets(
    signals: dict[str, pd.DataFrame],
    assets: tuple[str, ...],
    *,
    min_conf: float = 0.20,
    max_weight: float = 0.50,
    rebalance_threshold: float = 0.05,
) -> pd.DataFrame:
    """Gabriele's Equal Active allocation rule (frozen signals, new simulator).

    SPY active unless confidently SHORT; other ETFs active on confident LONG.
    Weight all active ETFs equally up to the 50% cap. If multiple assets are
    active and the capped sum is below one, re-normalize (original policy).
    A new weekly target is adopted only for changes >= 5 percentage points.
    All remaining weight is uninvested cash; short positions are not opened.
    """
    if not assets or not set(assets).issubset(ASSETS) or len(set(assets)) != len(assets):
        raise ValueError('Unsupported Equal Active asset selection')
    if not 0 <= min_conf <= 1 or not 0 < max_weight <= 1 or not 0 <= rebalance_threshold <= 1:
        raise ValueError('Invalid policy parameter')
    idx = signals[assets[0]].index
    if not all(signals[a].index.equals(idx) for a in assets):
        raise ValueError('Equal Active signals need a common weekly calendar')
    w = pd.DataFrame(0.0, index=idx, columns=list(ASSETS))
    for date in idx:
        active = []
        for a in assets:
            s = float(signals[a].at[date, 'signal'])
            c = float(signals[a].at[date, 'confidence'])
            if not s in (-1, 0, 1) or not np.isfinite(c) or not 0 <= c <= 1:
                raise ValueError(f'Invalid signal/confidence for {a} {date}')
            if (a == 'SPY' and not (s < 0 and c >= min_conf)) or (
                a != 'SPY' and s == 1 and c >= min_conf
            ):
                active.append(a)
        if not active:
            continue
        w.loc[date, active] = min(1.0/len(active), max_weight)
        weight_sum = w.loc[date].sum()
        if 0 < weight_sum < 1.0 and len(active) > 1:
            w.loc[date] /= weight_sum
    # Preserve Gabriele's weekly target-level hysteresis rule.
    current = pd.Series(0.0, index=list(ASSETS))
    for date in idx:
        desired = w.loc[date].copy()
        if (desired.sub(current).abs() >= rebalance_threshold).any():
            current = desired
        w.loc[date] = current
    return w


def sma200_targets(
    market_daily: pd.DataFrame,
    signals: dict[str, pd.DataFrame],
    begin: str = OOS_BEGIN,
    end: str = OOS_END,
    *,
    core_weight: float = 0.35,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Reproduce the documented retrospective B/C + SMA200 switch policy.

    Above SMA200: immutable original Core-Satellite target.
    Below SMA200: gate satellite by (21D and 63D positive momentum) AND
    Orchestrator LONG, then reuse the original frozen satellite selector.
    Separate selector histories are retained on each branch of the regime switch.
    """
    boundary = pd.Timestamp(end)
    if not 0 <= core_weight <= 1 or set(signals) != set(ASSETS):
        raise ValueError('Invalid original signal universe or core weight')
    market = market_daily.loc[:boundary].copy()
    original = build_core_satellite_targets(market, signals, begin, end, core_weight=core_weight)
    dates = original.index
    weekly = build_weekly_decision_dates(market).index
    closing = market[[f'{a}_close' for a in ASSETS]].astype(float)
    closing.columns = list(ASSETS)
    closing = closing.loc[:dates[-1]]
    sma200 = closing['SPY'].rolling(200, min_periods=200).mean()
    stress = closing['SPY'].loc[dates].lt(sma200.loc[dates])
    if sma200.loc[dates].isna().any() or stress.isna().any():
        raise ValueError('At least 200 prior closes needed for SMA200')
    stress.name = 'SPY_below_SMA200'
    momentum = build_trend_signal_panel(closing, dates)
    filtered = {}
    for asset in SATELLITE_ORDER:
        frame = signals[asset].loc[dates].copy()
        eligible = momentum[asset].eq(1) & frame['signal'].eq(1)
        frame.loc[~eligible, 'signal'] = 0
        filtered[asset] = frame
    weekly_close = market.loc[weekly, [f'{a}_close' for a in ASSETS]]
    selected = generate_dynamic_growth_matrix(
        filtered,
        df_features=weekly_close,
        default_vol=0.10,
        min_confidence=0.35,
        hysteresis_buffer=0.015,
    )
    defensive = selected.reindex(index=dates, columns=list(ASSETS)).astype(float)
    defensive *= 1.0 - core_weight
    defensive['SPY'] += core_weight
    adaptive = original.copy()
    adaptive.loc[stress] = defensive.loc[stress]
    for name, arr in [('original', original), ('defensive', defensive), ('adaptive', adaptive)]:
        values = arr.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (np.abs(values).sum(axis=1) > 1+1e-10).any():
            raise ValueError(f'{name} invalid target gross exposure')
    return {'original': original, 'defensive_long_flat': defensive,
            'regime_adaptive_sma200': adaptive, 'stress': stress}


def build_comparison(root: Path = ROOT_DIR) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Compute daily returns and metrics for all strategies on the *same* dates.

    Exact official Core NAV and targets are independently reconciled before
    returning comparisons, making accidental changes to the experiment visible.
    """
    root = Path(root)
    cfg = load_config(root/'config.yaml')
    market, _ = load_daily_market_data(cfg, root)
    folder = root/'reports/oos_signals'
    signals = {
        a: pd.read_csv(folder/f'signals_orchestrator_{a.lower()}.csv',
                       index_col='Date', parse_dates=['Date']) for a in ASSETS
    }
    variants = sma200_targets(market, signals)
    idx = variants['original'].index
    archived = pd.read_csv(root/'reports/oos_final_2022_2026/core_satellite_targets.csv',
                           index_col='Date', parse_dates=['Date'])
    if not idx.equals(archived.index) or not np.allclose(variants['original'], archived, atol=1e-11):
        raise RuntimeError('Original Core targets fail archive reconciliation')
    window = market.loc[idx.min():idx.max()]
    opens = window[[f'{a}_open' for a in ASSETS]].copy()
    closes = window[[f'{a}_close' for a in ASSETS]].copy()
    opens.columns = closes.columns = list(ASSETS)
    x = cfg['execution']; b = cfg['strategies']['B']
    kwargs = dict(initial_capital=float(x['initial_capital']),
                  commission_bps=float(x['commission_bps']),
                  slippage_bps=float(x['slippage_bps']),
                  annual_borrow_rate=float(b['annual_borrow_rate']),
                  additional_margin_rate=float(b['additional_margin_rate']))
    targets = {'Core35_Satellite65': variants['original'],
               'Equal_Active_3ETF': equal_active_targets(signals, ('SPY','IEF','GLD')),
               'Equal_Active_4ETF': equal_active_targets(signals, ('SPY','IEF','GLD','SHY')),
               'Regime_Adaptive_SMA200': variants['regime_adaptive_sma200']}
    sixty = pd.DataFrame(0.0, index=idx, columns=list(ASSETS))
    sixty['SPY'] = .60; sixty['IEF'] = .40
    spy = pd.DataFrame(0.0, index=idx[:1], columns=list(ASSETS))
    spy['SPY'] = 1.0
    targets['SPY_BuyHold'] = spy
    targets['Sixty_Forty'] = sixty
    rows = []; frames = {}
    for name, target in targets.items():
        ledger = simulate_longshort_daily(opens, closes, target, **kwargs)
        m = calculate_metrics(ledger['daily_return'], 0.0, 252)
        rows.append(dict(strategy=name, evaluation='official_frozen' if name in ('Core35_Satellite65','SPY_BuyHold','Sixty_Forty') else 'retrospective_post_selection',
                         start=str(ledger.index.min().date()), end=str(ledger.index.max().date()),
                         observations=int(m['observations']), decisions=len(target),
                         cumulative_return=float(m['cumulative_return']),
                         cagr=float(m['cagr']), annualized_volatility=float(m['annualized_volatility']),
                         sharpe_ratio=float(m['sharpe_ratio']), max_drawdown=float(m['max_drawdown']),
                         mean_gross_exposure=float(ledger['gross_exposure'].mean()),
                         turnover_sum=float(ledger['turnover'].sum()),
                         transaction_costs_usd=float(ledger['transaction_cost'].sum()),
                         borrow_costs_usd=float(ledger['borrow_fee'].sum()),
                         final_nav=float(ledger['nav'].iloc[-1])))
        frames[name] = ledger
        if name in ('Core35_Satellite65','SPY_BuyHold','Sixty_Forty'):
            official_name={'Core35_Satellite65':'Core35_Satellite65_daily.csv',
                           'SPY_BuyHold':'SPY_buy_hold_daily.csv',
                           'Sixty_Forty':'Sixty_Forty_daily.csv'}[name]
            official = pd.read_csv(root/'reports/oos_final_2022_2026'/official_name,
                                   index_col='Date', parse_dates=['Date'])
            if not ledger.index.equals(official.index) or not np.allclose(ledger['nav'], official['nav'], rtol=1e-11, atol=1e-6):
                raise RuntimeError(f'{name}: independent NAV reconstruction differs from freeze')
    daily = pd.DataFrame({name:frame['nav'] for name,frame in frames.items()})
    metrics = pd.DataFrame(rows).set_index('strategy')
    meta = {'stress_decisions': int(variants['stress'].sum()),
            'total_decisions': len(idx), 'frozen_equal_active_3_etf_universe':['SPY','IEF','GLD'],
            'equal_active_4_etf_universe':['SPY','IEF','GLD','SHY'],
            'methodology':'POST-SELECTION RESEARCH: models frozen 2021; allocations selected after reviewing 2022-2026; next-open 2+1bps cost; 1% borrow; not new independent OOS',
            'notes':'Gabriele original 2021-start weekly zero-cost setup is distinct and not reproduced here.'}
    return metrics, daily, targets, meta


def run_and_export(root: Path = ROOT_DIR, output_dir: Path | None = None) -> pd.DataFrame:
    """CLI helper; only writes the explicitly supplied research output folder."""
    import json
    from matplotlib import pyplot as plt
    root = Path(root).resolve()
    # The packaged research_outputs/ is indexed by PACKAGE_MANIFEST.json.
    # An ordinary rerun must never overwrite those protected release artifacts.
    m,nav,targets,meta = build_comparison(root)
    if output_dir is None:
        return m  # read-only matched replay
    output_dir = Path(output_dir).resolve()
    if output_dir.is_relative_to(root):
        raise ValueError('Export to a directory OUTSIDE project/ to preserve ZIP manifest integrity')
    output_dir.mkdir(parents=True, exist_ok=True)
    m.to_csv(output_dir/'strategy_comparison_metrics.csv', float_format='%.12g')
    nav.to_csv(output_dir/'strategy_comparison_nav.csv',index_label='Date',float_format='%.12g')
    (output_dir/'evaluation_scope.json').write_text(json.dumps(meta, indent=2)+'\n',encoding='utf-8')
    for name,w in targets.items():
        w.to_csv(output_dir/f'targets_{name}.csv',index_label='Date',float_format='%.12g')
    labels={'Core35_Satellite65':'Core-Satellite (official)',
            'Equal_Active_3ETF':'Equal Active 3 ETF (retrospective)',
            'Equal_Active_4ETF':'Equal Active 4 ETF (retrospective)',
            'Regime_Adaptive_SMA200':'SMA200 adaptive (retrospective)',
            'SPY_BuyHold':'SPY buy & hold (official)',
            'Sixty_Forty':'60/40 SPY-IEF (official)'}
    palette={'Core35_Satellite65':'#9E6471','Equal_Active_3ETF':'#087f8c',
             'Equal_Active_4ETF':'#50a19c','Regime_Adaptive_SMA200':'#F0A432',
             'SPY_BuyHold':'#566F9C','Sixty_Forty':'#839173'}
    fig, ax=plt.subplots(figsize=(11.5,5.4),dpi=170)
    for name in nav.columns:
        ax.plot(nav.index, nav[name]/1000.0, label=labels[name], lw=2.0 if name in ('Core35_Satellite65','Regime_Adaptive_SMA200','Equal_Active_3ETF') else 1.35,
                c=palette[name], alpha=1.0 if name!='Equal_Active_4ETF' else .70)
    ax.set(title='Common 2022-2026 window: original and retrospective allocations',ylabel='Portfolio value ($ thousands)',xlabel='')
    ax.grid(alpha=.18);ax.legend(fontsize=8,loc='upper left',ncols=2)
    fig.autofmt_xdate();fig.tight_layout()
    fig.savefig(output_dir/'comparison_equity.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(11.5,4.7),dpi=170)
    for name in nav.columns:
        dd=nav[name]/nav[name].cummax()-1
        ax.plot(nav.index,dd*100,label=labels[name],color=palette[name],lw=1.8 if name in ('Regime_Adaptive_SMA200','Equal_Active_3ETF','Core35_Satellite65') else 1.15,alpha=.94)
    ax.set(title='Drawdowns on the same trading calendar',ylabel='Drawdown (%)',xlabel='')
    ax.grid(alpha=.18);ax.legend(fontsize=8,loc='lower left',ncols=2)
    fig.autofmt_xdate();fig.tight_layout()
    fig.savefig(output_dir/'comparison_drawdowns.png');plt.close(fig)
    return m
