"""Create presentation-ready figures from archived, immutable OOS daily rows."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from src.release_pipeline import audit_release
from src.utils import ROOT_DIR

NAMES = {'Core35_Satellite65': 'Core35-Satellite65',
         'SPY_buy_hold': 'SPY Buy & Hold',
         'Sixty_Forty': '60/40 SPY-IEF'}


def build_final_charts(root: Path = ROOT_DIR, output_dir: Path | None = None) -> list[Path]:
    root = Path(root)
    audit_release(root)
    folder = Path(output_dir) if output_dir is not None else root/'outputs'/'figures'
    if not folder.is_absolute():
        folder = root/folder
    folder.mkdir(parents=True, exist_ok=True)
    nav = pd.DataFrame()
    for key in NAMES:
        path = root/'reports/oos_final_2022_2026'/f'{key}_daily.csv'
        df = pd.read_csv(path, index_col='Date', parse_dates=['Date'])
        nav[NAMES[key]] = 100.0 * df['nav'] / 100000.0
        if not df.index.is_monotonic_increasing:
            raise RuntimeError(f'Invalid NAV dates in {path}')
    if nav.isna().any().any():
        raise RuntimeError('Missing dates between benchmark NAVs')

    outputs=[]
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    nav.plot(ax=ax, lw=1.9)
    ax.axhline(100, linestyle=':', linewidth=0.9, color='gray')
    ax.set(title='Out-of-sample equity curves (7 Jan 2022 - 9 Oct 2026)',
           ylabel='Index (initial capital = 100)', xlabel='')
    ax.grid(alpha=0.22)
    fig.tight_layout()
    file=folder/'oos_equity_curves.png'
    fig.savefig(file, dpi=170)
    outputs.append(file)
    plt.close(fig)

    dd = nav/nav.cummax().clip(lower=100) - 1
    fig, ax = plt.subplots(figsize=(10.5, 4.7))
    (dd*100).plot(ax=ax, lw=1.6)
    ax.set(title='Out-of-sample drawdowns (same daily calendar)',
           ylabel='Drawdown (%)', xlabel='')
    ax.grid(alpha=0.22)
    fig.tight_layout()
    file=folder/'oos_drawdowns.png'
    fig.savefig(file,dpi=170)
    outputs.append(file)
    plt.close(fig)
    return outputs
