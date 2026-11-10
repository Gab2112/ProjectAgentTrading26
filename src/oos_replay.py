"""Recreate the locked final OOS backtest on a disposable isolated copy.

This is reproducibility verification, NEVER post-test model selection.
No model .fit(), GA optimization or overwrite of the archived results.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from src.release_pipeline import audit_release
from src.submission_validation import verify_signal_calendar
from src.utils import ROOT_DIR

CSV_NAMES = ('comparison_metrics.csv', 'core_satellite_targets.csv',
             'Core35_Satellite65_daily.csv', 'SPY_buy_hold_daily.csv',
             'Sixty_Forty_daily.csv')


def reproduce_frozen_oos(root: Path = ROOT_DIR) -> None:
    root = Path(root).resolve()
    audit_release(root)
    verify_signal_calendar(root, oos=True)
    reference = root/'reports/oos_final_2022_2026'

    with tempfile.TemporaryDirectory(prefix='group9_frozen_replay_') as tmp:
        destination = Path(tmp)/'project'

        def ignore(folder: str, names: list[str]) -> set[str]:
            if Path(folder).name == 'reports':
                return {'oos_final_2022_2026'}.intersection(names)
            return {'__pycache__', '.venv', '.git', 'outputs'}.intersection(names)

        shutil.copytree(root, destination, ignore=ignore)
        output = subprocess.run(
            [sys.executable, '-m', 'src.final_oos_backtest'],
            cwd=destination, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180, check=False,
        )
        if output.returncode != 0:
            raise RuntimeError('Frozen OOS replay failed on isolated copy:\n'+
                               output.stdout[-2400:]+'\n'+output.stderr[-3000:])
        generated = destination/'reports/oos_final_2022_2026'
        for name in CSV_NAMES:
            expected = pd.read_csv(reference/name, index_col=0)
            observed = pd.read_csv(generated/name, index_col=0)
            try:
                pd.testing.assert_frame_equal(
                    expected, observed, check_exact=False,
                    rtol=1e-10, atol=1e-8,
                )
            except AssertionError as exc:
                raise RuntimeError(f'OOS replay differs from frozen {name}: {exc}') from exc
            print(f'PASS: {name}')
    print('PASS: exact frozen OOS research replay; original archive unchanged.')
