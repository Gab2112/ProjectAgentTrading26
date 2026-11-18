"""Charts of the final evaluation (written to data/results/plots).

- ``plot_equity_and_drawdown``: equity curves rebased to 100, with the
  benchmarks on the same chart, and the drawdown of each curve;
- ``plot_signal_backtests``: per-asset long/flat/short signal backtests
  against buy-and-hold of the same asset;
- ``plot_signal_distribution``: share of long, flat and short signals;
- ``plot_validation_grid``: validation Sharpe ratio of every allocation candidate.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

from src.utils import ASSETS

STRATEGY_NAME = "Agent Strategy"
PALETTE = {
    STRATEGY_NAME: "#1b7a4a",
    "SPY Buy & Hold": "#c0392b",
    "Equal Weight Buy & Hold": "#7f8c8d",
    "60/40 SPY-IEF (weekly)": "#e67e22",
    "Equal Weight (weekly)": "#8e44ad",
    "Volatility Control": "#2e86c1",
    "Macro Regime Long/Short": "#95a5a6",
    "Macro Regime Long/Flat": "#d4ac0d",
}
SIGNAL_COLORS = {-1: "#c0392b", 0: "#95a5a6", 1: "#1b7a4a"}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def equity_curve(nav: pd.DataFrame, initial_capital: float, start: pd.Timestamp) -> pd.Series:
    """NAV expressed in units of 100, with a starting point of 100 at the decision date."""
    curve = 100.0 * nav["nav"] / float(initial_capital)
    return pd.concat([pd.Series([100.0], index=[pd.Timestamp(start)]), curve]).sort_index()


def plot_equity_and_drawdown(curves: dict[str, pd.Series], title: str, path: Path) -> None:
    """Equity curves (start = 100), with the benchmarks on the same chart, and drawdowns."""
    fig, (ax_eq, ax_dd) = plt.subplots(2, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for name, equity in curves.items():
        drawdown = equity / equity.cummax() - 1.0
        is_strategy = name == STRATEGY_NAME
        style = dict(color=PALETTE.get(name, "#333333"), linewidth=2.6 if is_strategy else 1.4,
                     alpha=1.0 if is_strategy else 0.85, zorder=5 if is_strategy else 3)
        ax_eq.plot(equity.index, equity, label=name, **style)
        ax_dd.plot(drawdown.index, drawdown * 100.0, **style)

    ax_eq.axhline(100.0, color="#222222", linewidth=0.8, alpha=0.6)
    ax_eq.set_title(f"Equity curves (start = 100) - {title}", fontsize=13, fontweight="bold")
    ax_eq.set_ylabel("Portfolio value")
    ax_eq.grid(True, linestyle="--", alpha=0.4)
    ax_eq.legend(loc="upper left", fontsize=9, frameon=True)
    ax_dd.set_title("Drawdown", fontsize=11, fontweight="bold")
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0, decimals=0))
    ax_dd.grid(True, linestyle="--", alpha=0.4)
    _save(fig, path)


def plot_signal_backtests(curves: dict[str, dict[str, pd.Series]], title: str, path: Path) -> None:
    """One panel per asset: signal strategy against buy-and-hold of the asset (start = 100)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for ax, asset in zip(axes.flat, ASSETS):
        for label, equity in curves[asset].items():
            ax.plot(equity.index, equity, label=label, linewidth=2.0 if "signal" in label else 1.2,
                    color="#1b7a4a" if "signal" in label else "#7f8c8d")
        ax.axhline(100.0, color="#222222", linewidth=0.6, alpha=0.5)
        ax.set_title(asset, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Per-asset signal backtests (+1 long, 0 flat, -1 short) - {title}", fontsize=13, fontweight="bold")
    _save(fig, path)


def plot_signal_distribution(signals: pd.DataFrame, title: str, path: Path) -> None:
    """Share of long, flat and short Orchestrator signals per asset."""
    shares = pd.DataFrame(
        {value: (signals == value).mean() * 100.0 for value in (1, 0, -1)}
    ).loc[list(ASSETS)]
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = pd.Series(0.0, index=shares.index)
    for value, label in ((1, "Long (+1)"), (0, "Flat (0)"), (-1, "Short (-1)")):
        ax.bar(shares.index, shares[value], bottom=bottom, color=SIGNAL_COLORS[value], label=label)
        bottom += shares[value]
    ax.set_ylabel("Share of weeks (%)")
    ax.set_title(f"Orchestrator signal distribution - {title}", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, path)


def plot_validation_grid(results: pd.DataFrame, metric: str, path: Path) -> None:
    """Heat maps of the validation metric for every allocation candidate."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, allow_short in zip(axes, (False, True)):
        table = results[results["allow_short"] == allow_short].pivot(
            index="min_confidence", columns="max_weight_per_asset", values=metric
        )
        image = ax.imshow(table.to_numpy(), cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(table.columns)), [f"{c:.2f}" for c in table.columns])
        ax.set_yticks(range(len(table.index)), [f"{i:.2f}" for i in table.index])
        for (i, j), value in pd.DataFrame(table.to_numpy()).stack().items():
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
        ax.set_xlabel("Max weight per asset")
        ax.set_title(f"allow_short = {allow_short}")
        fig.colorbar(image, ax=ax, shrink=0.8)
    axes[0].set_ylabel("Min confidence")
    fig.suptitle(f"Validation {metric.replace('_', ' ')} of the allocation candidates", fontweight="bold")
    _save(fig, path)
