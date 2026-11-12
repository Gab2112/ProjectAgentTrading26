"""Isolated development-only compatibility runner for historical A/B/C modes.

Legacy stages can write generated files. They are executed only in an
unpublished temporary tree so release files and frozen inputs stay intact.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from src.utils import ROOT_DIR, load_config


def run_development_in_copy(level: str, *, b_version: str = 'v2',
                            config_path: Path | None = None) -> None:
    """Run a historical development level inside a temporary copy of the project.

    The legacy stages write generated files, so they are executed in an
    isolated worker process on a temporary copy; release files and frozen
    inputs in the original tree are never modified.
    """
    root = ROOT_DIR
    with tempfile.TemporaryDirectory(prefix='group9_dev_') as tmp:
        destination = Path(tmp)/'project'

        def ignore(directory: str, names: list[str]) -> set[str]:
            return {'__pycache__', '.venv', '.git', 'outputs', 'figures'}.intersection(names)

        shutil.copytree(root, destination, ignore=ignore)
        cmd = [sys.executable, '-m', 'src.submission_runner',
               '--worker', '--level', level, '--b-version', b_version]
        if config_path:
            path = Path(config_path)
            if not path.is_absolute():
                path = root/path
            if path.resolve().is_relative_to(root.resolve()):
                relative = path.resolve().relative_to(root.resolve())
                cmd += ['--config', str(relative)]
            else:
                raise ValueError('Development YAML must be inside this project')
        print(f'=== Development-only {level} (isolated temporary worktree) ===', flush=True)
        subprocess.run(cmd, cwd=destination, check=True)
    print('Development complete; temporary outputs removed. Frozen archive untouched.')


def run_worker(level: str, b_version: str, config_path: str | None) -> None:
    """Execute one development level inside the isolated temporary copy."""
    from main import run_pipeline, validate_config

    config = load_config(ROOT_DIR/(config_path or 'config.yaml'))
    validate_config(config)

    if level == 'C':
        if config_path:
            raise ValueError('C study must use immutable frozen config.yaml')
        # Reconstruct the required derived weekly trading calendar only.
        from src.data_loader import load_dataset
        from src.time_split import build_weekly_decision_dates
        from src.run_training_comparison import run_training_comparison
        from src.submission_validation import verify_signal_calendar
        import pandas as pd
        end = pd.Timestamp(config['validation']['train_end'])
        market = load_dataset(ROOT_DIR/'data/processed/market_daily.csv', strict=True)
        weeks = build_weekly_decision_dates(market.loc[:end])
        target = ROOT_DIR/'data/processed/weekly_decision_dates.csv'
        weeks.to_csv(target, index_label='Date')
        verify_signal_calendar(ROOT_DIR)
        comparison = run_training_comparison(config, ROOT_DIR)
        view = comparison['summary'][['cagr', 'annualized_volatility',
                                      'sharpe_ratio', 'max_drawdown']].copy()
        for col in ('cagr', 'annualized_volatility', 'max_drawdown'):
            view[col] *= 100
        print('=== IN-SAMPLE LEVEL C / B MATCHED COMPARISON ===')
        print(view.round(3).to_string())
    elif level == 'ALL':
        run_pipeline(level='ALL', config_path=ROOT_DIR/(config_path or 'config.yaml'))
        run_pipeline(level='B', b_version='v1')
        run_pipeline(level='B', b_version='v2')
        run_worker('C', 'v2', None)
    elif level in {'A1','A2','B'}:
        run_pipeline(level=level, b_version=b_version if level == 'B' else None,
                     config_path=ROOT_DIR/(config_path or 'config.yaml'))
    else:
        raise ValueError(f'Unsupported legacy development level: {level}')


def main() -> None:
    """Command-line entry point of the isolated development worker."""
    p = argparse.ArgumentParser(description='Private isolated development worker')
    p.add_argument('--worker', action='store_true', required=True)
    p.add_argument('--level', required=True)
    p.add_argument('--b-version', default='v2')
    p.add_argument('--config', default=None)
    a = p.parse_args()
    run_worker(a.level, a.b_version, a.config)


if __name__ == '__main__':
    main()


# Research-only interfaces. These never call the official final OOS writer.
def _research_output_directory(output_dir: Path | None, root_dir: Path) -> Path | None:
    """Avoid accidentally writing to an archived or frozen project location."""
    if output_dir is None:
        return None
    dest = Path(output_dir).expanduser().resolve()
    project = Path(root_dir).resolve()
    protected = ('data', 'reports', 'src', 'tests', 'docs', 'figures',
                 'config.yaml', 'release_config.yaml', 'PACKAGE_MANIFEST.json')
    if dest == project or any(dest == project/name or
                               (project/name) in dest.parents for name in protected):
        raise ValueError('Output must not be inside an archived project directory')
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def calculate_archived_robustness(root_dir: Path = ROOT_DIR):
    """Describe fixed daily-NAV records over full, 2022 and 2023+ windows.

    The 2023+ result is a *retrospective subset* of a continuous archived
    2022-start simulation. Holdings are NOT reset in January 2023, and no
    new train/test or re-fit procedure is performed.
    """
    import pandas as pd
    from src.evaluator import calculate_metrics

    source = Path(root_dir)/'reports/oos_final_2022_2026'
    scenarios = {
        'Core35_Satellite65': 'Core35_Satellite65_daily.csv',
        'SPY_buy_hold': 'SPY_buy_hold_daily.csv',
        'Sixty_Forty': 'Sixty_Forty_daily.csv',
    }
    intervals = (
        ('full_2022_2026', '2022-01-07', '2026-10-09'),
        ('crisis_2022', '2022-01-07', '2022-12-31'),
        ('retrospective_2023_2026', '2023-01-01', '2026-10-09'),
    )
    rows=[]
    for strategy, filename in scenarios.items():
        raw = pd.read_csv(source/filename, index_col='Date', parse_dates=['Date'])
        if raw.empty or raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
            raise ValueError(f'Invalid archived NAV series for {strategy}')
        for label, start, end in intervals:
            subset = raw.loc[start:end]
            if subset.empty:
                raise ValueError(f'Empty {label} period for {strategy}')
            metric = calculate_metrics(subset['daily_return'], 0.0, 252)
            rows.append(dict(
                period=label, strategy=strategy, observations=len(subset),
                first_date=str(subset.index[0].date()),
                last_date=str(subset.index[-1].date()),
                cagr_pct=float(metric['cagr']*100.0),
                sharpe=float(metric['sharpe_ratio']),
                mdd_pct=float(metric['max_drawdown']*100.0),
                cumulative_pct=float(metric['cumulative_return']*100.0),
            ))
    return pd.DataFrame(rows)


def calculate_regime_adaptive_research(root_dir: Path = ROOT_DIR):
    """Retrospective, same-engine comparison of frozen and adaptive targets.

    OOS data are inspected here ONLY because this is explicitly labeled a
    post-hoc research diagnostic. No parameter fitting or official freeze
    change occurs. Original target weights and daily NAV are reconciled.
    """
    import numpy as np
    import pandas as pd
    from src.data_loader import load_daily_market_data
    from src.evaluator import calculate_metrics
    from src.execution_engine import simulate_longshort_daily
    from src.level_c_strategy import build_regime_adaptive_core_targets
    from src.utils import ASSETS

    root = Path(root_dir)
    config = load_config(root/'config.yaml')
    market, _ = load_daily_market_data(config, root)
    b=config['strategies']['B']
    x=config['execution']
    kwargs=dict(initial_capital=float(x['initial_capital']),
                commission_bps=float(x['commission_bps']),
                slippage_bps=float(x['slippage_bps']),
                annual_borrow_rate=float(b['annual_borrow_rate']),
                additional_margin_rate=float(b['additional_margin_rate']))
    rows=[]
    for sample, folder, begin, finish in (
        ('development_2010_2021',root/'data/signals','2010-01-01','2021-12-31'),
        ('retrospective_2022_2026',root/'reports/oos_signals','2022-01-07','2026-10-09'),
    ):
        signals={a:pd.read_csv(folder/f'signals_orchestrator_{a.lower()}.csv',
                     index_col='Date',parse_dates=['Date']) for a in ASSETS}
        variants=build_regime_adaptive_core_targets(market,signals,begin,finish)
        targets=variants['original']
        dates=targets.index
        if sample.startswith('retrospective_'):
            archived=pd.read_csv(root/'reports/oos_final_2022_2026/core_satellite_targets.csv',
                                 index_col='Date',parse_dates=['Date'])
            if not dates.equals(archived.index) or not np.allclose(targets,archived,atol=1e-11):
                raise RuntimeError('Original frozen OOS targets did not match archive')
        session=market.loc[dates.min():dates.max()]
        opens=session[[f'{a}_open' for a in ASSETS]].copy()
        closes=session[[f'{a}_close' for a in ASSETS]].copy()
        opens.columns=closes.columns=list(ASSETS)
        weights={k:variants[k] for k in ('original','defensive_long_flat','regime_adaptive')}
        sixty=pd.DataFrame(0.0,index=dates,columns=list(ASSETS))
        sixty['SPY']=.60; sixty['IEF']=.40
        weights['60_40_weekly']=sixty
        for name,target in weights.items():
            bt=simulate_longshort_daily(open_prices=opens,close_prices=closes,
                                        target_weights=target,**kwargs)
            if name=='original' and sample.startswith('retrospective_'):
                official=pd.read_csv(root/'reports/oos_final_2022_2026/Core35_Satellite65_daily.csv',
                                     index_col='Date',parse_dates=['Date'])
                if not bt.index.equals(official.index) or not np.allclose(
                        bt['nav'],official['nav'],atol=1e-6,rtol=1e-11):
                    raise RuntimeError('Frozen OOS NAV failed comparison')
            m=calculate_metrics(bt['daily_return'],0.0,252)
            rows.append(dict(sample=sample,strategy=name,decisions=len(dates),
                     stress_decisions=int(variants['stress'].sum()),
                     cagr_pct=float(m['cagr']*100),sharpe=float(m['sharpe_ratio']),
                     mdd_pct=float(m['max_drawdown']*100),
                     mean_gross_pct=float(bt['gross_exposure'].mean()*100),
                     turnover=float(bt['turnover'].sum())))
    return pd.DataFrame(rows)


def run_archived_research_analysis(
    analysis: str, *, output_dir: Path | None = None, root_dir: Path = ROOT_DIR,
):
    """Print a research comparison; optionally save CSV outside frozen paths."""
    if analysis not in ('temporal-robustness','regime-adaptive'):
        raise ValueError('Unknown research analysis')
    output = _research_output_directory(output_dir,root_dir)
    if analysis=='temporal-robustness':
        data=calculate_archived_robustness(root_dir)
    else:
        data=calculate_regime_adaptive_research(root_dir)
    print(f'\n=== EXPLORATORY {analysis.upper()} ===')
    print(data.round(3).to_string(index=False))
    print('NOTE: observed-data diagnostic; not an independent newly selected OOS.')
    if output is not None:
        dest=output/(analysis.replace('-','_')+'.csv')
        data.to_csv(dest,index=False,float_format='%.10g')
        print('Research output:',dest)
    return data
