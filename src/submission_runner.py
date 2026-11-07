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
    p = argparse.ArgumentParser(description='Private isolated development worker')
    p.add_argument('--worker', action='store_true', required=True)
    p.add_argument('--level', required=True)
    p.add_argument('--b-version', default='v2')
    p.add_argument('--config', default=None)
    a = p.parse_args()
    run_worker(a.level, a.b_version, a.config)


if __name__ == '__main__':
    main()
