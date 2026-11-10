#!/usr/bin/env python3
"""Reproducible Group 9 presentation charts, using frozen archive + backtest engine.

Install at: scripts/generate_presentation_charts.py
Run from the repository root:
    python scripts/generate_presentation_charts.py
    python scripts/generate_presentation_charts.py --include-adaptive

This is a post-freeze visualization and transaction-cost sensitivity tool.
It NEVER refits an agent, updates a signal, or writes into frozen data/reports.

The zero-transaction-cost scenario is a fresh replay of IDENTICAL weekly
portfolio targets with commission_bps=0 and slippage_bps=0. The original
1% short-borrow rate stays in force in both scenarios.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_daily_market_data
from src.evaluator import calculate_metrics
from src.execution_engine import simulate_longshort_daily
from src.release_pipeline import audit_release
from src.utils import ASSETS, load_config

NAMES = {
    "Core35_Satellite65": "Core35-Satellite65",
    "SPY_buy_hold": "SPY Buy & Hold",
    "Sixty_Forty": "60/40 SPY-IEF",
}
COLORS = {
    "Core35_Satellite65": "#D76C45",
    "SPY_buy_hold": "#153F64",
    "Sixty_Forty": "#138880",
    "Regime_Adaptive": "#754DA3",
}
FIRST = "2022-01-07"
LAST = "2026-10-09"
RETRO_START = "2023-01-03"


def read_csv_dates(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required release artifact: {path}")
    frame = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    if (frame.empty or frame.index.hasnans or frame.index.has_duplicates
            or not frame.index.is_monotonic_increasing):
        raise ValueError(f"Invalid chronological data in {path}")
    return frame


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def save_figure(fig, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def format_dates(ax) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def metric_record(frame: pd.DataFrame, annualization: int, rf_daily: float) -> dict:
    m = calculate_metrics(frame["daily_return"], rf_daily, annualization)
    return {
        "observations": int(m["observations"]),
        "cagr_pct": m["cagr"] * 100,
        "sharpe": m["sharpe_ratio"],
        "max_drawdown_pct": m["max_drawdown"] * 100,
        "cumulative_return_pct": m["cumulative_return"] * 100,
        "ending_continuous_nav_usd": float(frame["nav"].iloc[-1]),
    }


def archived_inputs(root: Path):
    config = load_config(root / "config.yaml")
    release = root / "reports/oos_final_2022_2026"
    archived = {key: read_csv_dates(release / f"{key}_daily.csv") for key in NAMES}
    target = read_csv_dates(release / "core_satellite_targets.csv")[list(ASSETS)]
    comparison = pd.read_csv(release / "comparison_metrics.csv", index_col="strategy")

    dates = archived["Core35_Satellite65"].index
    if (dates.min() != pd.Timestamp(FIRST) or dates.max() != pd.Timestamp(LAST)
            or not all(x.index.equals(dates) for x in archived.values())):
        raise RuntimeError("Archived OOS calendar does not match the frozen study")
    if (target.index.min() != dates.min() or target.index.max() != dates.max()):
        raise RuntimeError("Frozen Core decision calendar does not match the OOS window")

    market, _audit = load_daily_market_data(config, root)
    # Preserve pre-2022 history for the optional 200-session SMA warm-up.
    oos_market = market.loc[FIRST:LAST]
    if not oos_market.index.equals(dates):
        raise RuntimeError("Raw adjusted Open/Close calendar does not reproduce archived OOS")
    opens = oos_market[[f"{a}_open" for a in ASSETS]].copy()
    closes = oos_market[[f"{a}_close" for a in ASSETS]].copy()
    opens.columns = closes.columns = list(ASSETS)

    # Identical targets to the historical final test, NEVER recomputed forecasts.
    spy = pd.DataFrame(0.0, index=target.index[:1], columns=list(ASSETS))
    spy["SPY"] = 1.0
    sixty = pd.DataFrame(0.0, index=target.index, columns=list(ASSETS))
    sixty["SPY"] = 0.60
    sixty["IEF"] = 0.40
    weights = {
        "Core35_Satellite65": target,
        "SPY_buy_hold": spy,
        "Sixty_Forty": sixty,
    }
    return config, archived, comparison, weights, opens, closes, market


def replay_sensitivity(config, archived, comparison, weights, opens, closes):
    x = config["execution"]
    b = config["strategies"]["B"]
    initial = float(x["initial_capital"])
    kwargs = {
        "initial_capital": initial,
        "annual_borrow_rate": float(b["annual_borrow_rate"]),
        "additional_margin_rate": float(b["additional_margin_rate"]),
    }
    ann = int(config["evaluation"]["annualization_factor"])
    rf_annual = float(config["evaluation"]["risk_free_rate"])
    rf = (1.0 + rf_annual) ** (1.0 / ann) - 1.0
    results = []
    no_cost = {}
    for name, target in weights.items():
        # Strict reconciliation against the original OOS: same engine and costs.
        baseline = simulate_longshort_daily(
            opens, closes, target, commission_bps=float(x["commission_bps"]),
            slippage_bps=float(x["slippage_bps"]), **kwargs,
        )
        np.testing.assert_allclose(
            baseline["nav"].to_numpy(), archived[name]["nav"].to_numpy(),
            rtol=1e-10, atol=1e-5,
            err_msg=f"Frozen OOS NAV failed reconciliation: {name}",
        )
        np.testing.assert_allclose(
            baseline["transaction_cost"].to_numpy(),
            archived[name]["transaction_cost"].to_numpy(),
            rtol=1e-9, atol=1e-6,
            err_msg=f"Frozen transaction-cost accounting differs: {name}",
        )
        original = metric_record(archived[name], ann, rf)
        if not np.isclose(original["cagr_pct"] / 100, comparison.loc[name, "cagr"],
                          atol=1e-10, rtol=1e-10):
            raise RuntimeError(f"Frozen published CAGR mismatch: {name}")

        # Real counterfactual replay, not an additive NAV correction.
        hypothetical = simulate_longshort_daily(
            opens, closes, target, commission_bps=0.0, slippage_bps=0.0, **kwargs,
        )
        if not hypothetical.index.equals(archived[name].index):
            raise RuntimeError("Transaction-cost counterfactual changed the calendar")
        if not np.allclose(hypothetical["transaction_cost"], 0):
            raise RuntimeError("No-transaction-cost simulation charged a trading fee")
        no_cost[name] = hypothetical
        alternative = metric_record(hypothetical, ann, rf)
        results.append({
            "strategy": name,
            "cagr_with_costs_pct": original["cagr_pct"],
            "cagr_no_transaction_costs_pct": alternative["cagr_pct"],
            "delta_cagr_pp": alternative["cagr_pct"] - original["cagr_pct"],
            "sharpe_with_costs": original["sharpe"],
            "sharpe_no_transaction_costs": alternative["sharpe"],
            "mdd_with_costs_pct": original["max_drawdown_pct"],
            "mdd_no_transaction_costs_pct": alternative["max_drawdown_pct"],
            "final_nav_with_costs_usd": original["ending_continuous_nav_usd"],
            "final_nav_no_transaction_costs_usd": alternative["ending_continuous_nav_usd"],
            "final_nav_difference_usd": (
                alternative["ending_continuous_nav_usd"] - original["ending_continuous_nav_usd"]
            ),
            "historical_trading_costs_usd": float(archived[name]["transaction_cost"].sum()),
            "historical_borrow_fees_usd": float(archived[name]["borrow_fee"].sum()),
        })
    return pd.DataFrame(results), no_cost, ann, rf


def plot_main_charts(archive, no_cost, cost_table, output):
    # Figure 1: official, untouched continuous net portfolios.
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for name, df in archive.items():
        ax.plot(df.index, df["nav"] / 1000.0, label=NAMES[name],
                color=COLORS[name], lw=2.1)
    ax.axhline(100.0, color="#999999", ls=":", lw=1)
    ax.set(title="Official frozen OOS | 2022-2026", ylabel="Portfolio value (index, start = 100)")
    format_dates(ax)
    ax.legend(loc="upper left")
    save_figure(fig, output, "01_official_oos_equity")

    # Figure 2: retrospective subset of the SAME NAV, anchored at end-2022.
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for name, df in archive.items():
        prev = df.loc[df.index < RETRO_START].iloc[-1]
        sub = df.loc[RETRO_START:]
        idx = pd.DatetimeIndex([df.index[df.index < RETRO_START][-1]]).append(sub.index)
        vals = np.r_[100.0, sub["nav"].to_numpy() / float(prev["nav"]) * 100.0]
        ax.plot(idx, vals, label=NAMES[name], color=COLORS[name], lw=2.1)
    ax.axhline(100.0, color="#999999", ls=":", lw=1)
    ax.set(title="Retrospective 2023-2026 | Continuous NAV rebased at end-2022",
           ylabel="Portfolio value (30 Dec 2022 = 100)")
    format_dates(ax)
    ax.legend(loc="upper left")
    fig.text(0.12, 0.01, "Historical slice, not a new holdout; no retraining or position reset.",
             fontsize=9, color="#555555")
    save_figure(fig, output, "02_retrospective_2023_equity")

    # Figure 3: drawdowns on the full official OOS window.
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for name, df in archive.items():
        nav = df["nav"]
        dd = 100.0 * (nav / nav.cummax().clip(lower=float(nav.iloc[0])) - 1)
        ax.plot(df.index, dd, label=NAMES[name], color=COLORS[name], lw=1.9)
    ax.set(title="Official OOS | Drawdowns", ylabel="Drawdown (%)")
    format_dates(ax)
    ax.legend(loc="lower left")
    save_figure(fig, output, "03_official_oos_drawdowns")

    # Figure 4: direct comparable cost policy, all models with the same engine.
    names = list(cost_table["strategy"])
    x = np.arange(len(names))
    width = 0.37
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    a = ax.bar(x - width / 2, cost_table["cagr_with_costs_pct"], width,
               label="2 bps commission + 1 bp slippage", color="#153F64")
    b = ax.bar(x + width / 2, cost_table["cagr_no_transaction_costs_pct"], width,
               label="0 bps transaction costs", color="#80ACA9")
    for group in (a, b):
        ax.bar_label(group, fmt="%.2f%%", padding=3, fontsize=10)
    ax.set(xticks=x, xticklabels=[NAMES[n] for n in names],
           ylabel="Annualized return (CAGR, %)",
           title="Trading-cost sensitivity | Same frozen target weights")
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(cost_table["cagr_no_transaction_costs_pct"]) * 1.23)
    fig.text(0.12, 0.015, "Borrowing costs remain at 1% p.a. in both simulations.",
             fontsize=9, color="#555555")
    save_figure(fig, output, "04_transaction_cost_sensitivity")

    # Figure 5: cumulative economic impact of trading frictions for the Core.
    core = "Core35_Satellite65"
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(archive[core].index, archive[core]["nav"] / 1000.0,
            label="Archived Core | Trading costs included", lw=2.2, color=COLORS[core])
    ax.plot(no_cost[core].index, no_cost[core]["nav"] / 1000.0,
            label="Same targets | No transaction costs", lw=2.2,
            color="#153F64", ls="--")
    ax.set(title="Core-Satellite | Trading-cost impact on equity",
           ylabel="Portfolio value (index, start = 100)")
    format_dates(ax)
    ax.legend(loc="upper left")
    save_figure(fig, output, "05_core_cost_counterfactual_equity")

    # Figure 6: actual paid charges in frozen NAV (not the compounded drag).
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    x = np.arange(len(names))
    commissions = cost_table["historical_trading_costs_usd"].to_numpy()
    borrowing = cost_table["historical_borrow_fees_usd"].to_numpy()
    ax.bar(x, commissions, label="Commission + slippage", color="#153F64")
    ax.bar(x, borrowing, bottom=commissions, label="Short borrow fees", color="#D76C45")
    ax.set(xticks=x, xticklabels=[NAMES[n] for n in names],
           ylabel="USD paid over archived OOS", title="Actual trading and borrowing charges")
    ax.legend(loc="upper right")
    save_figure(fig, output, "06_paid_cost_breakdown")


def optional_adaptive_chart(market, opens, closes, config, archive, output):
    from src.level_c_strategy import build_regime_adaptive_core_targets

    signals = {
        asset: read_csv_dates(ROOT / "reports/oos_signals"
                              / f"signals_orchestrator_{asset.lower()}.csv")
        for asset in ASSETS
    }
    variants = build_regime_adaptive_core_targets(market, signals, FIRST, LAST)
    frozen = read_csv_dates(ROOT / "reports/oos_final_2022_2026"
                            / "core_satellite_targets.csv")[list(ASSETS)]
    if not variants["original"].index.equals(frozen.index):
        raise RuntimeError("Adaptive comparison has a different decision calendar")
    np.testing.assert_allclose(variants["original"], frozen, atol=1e-11, rtol=0)
    exe = config["execution"]
    borrow = config["strategies"]["B"]
    bt = simulate_longshort_daily(
        opens, closes, variants["regime_adaptive"],
        initial_capital=float(exe["initial_capital"]),
        commission_bps=float(exe["commission_bps"]),
        slippage_bps=float(exe["slippage_bps"]),
        annual_borrow_rate=float(borrow["annual_borrow_rate"]),
        additional_margin_rate=float(borrow["additional_margin_rate"]),
    )
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(bt.index, bt["nav"] / 1000, color=COLORS["Regime_Adaptive"],
            label="Regime-Adaptive (exploratory)", lw=2.2)
    for name in ("Core35_Satellite65", "Sixty_Forty"):
        ax.plot(archive[name].index, archive[name]["nav"] / 1000,
                color=COLORS[name], label=NAMES[name], lw=2.0)
    ax.set(title="Regime-Adaptive SMA200 | Retrospective comparison",
           ylabel="Portfolio value (index, start = 100)")
    format_dates(ax)
    ax.legend(loc="upper left")
    fig.text(0.12, 0.012, "Post-freeze exploratory policy; not an independently selected new OOS.",
             fontsize=9, color="#555555")
    save_figure(fig, output, "07_regime_adaptive_exploratory")
    return bt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/presentation",
                        help="Must remain within the project's outputs/ directory")
    parser.add_argument("--include-adaptive", action="store_true",
                        help="Also plot the retrospective Regime-Adaptive SMA200 extension")
    args = parser.parse_args()

    output = (ROOT / args.output_dir).resolve()
    if not output.is_relative_to((ROOT / "outputs").resolve()):
        parser.error("Output path must be inside project/outputs/ (frozen paths protected)")

    # Integrity check is READ-ONLY, before running cost counterfactuals.
    audit_release(ROOT)
    config, archived, comparison, weights, opens, closes, market = archived_inputs(ROOT)
    cost_table, no_cost, ann, rf = replay_sensitivity(
        config, archived, comparison, weights, opens, closes,
    )

    output.mkdir(parents=True, exist_ok=True)
    setup_style()
    plot_main_charts(archived, no_cost, cost_table, output)

    metrics = []
    for name, df in archived.items():
        for label, sample in (
            ("official_2022_2026", df),
            ("retrospective_2023_2026", df.loc[RETRO_START:]),
        ):
            metrics.append({"sample": label, "strategy": name,
                            **metric_record(sample, ann, rf)})
    metric_table = pd.DataFrame(metrics)
    metric_table.to_csv(output / "period_metrics.csv", index=False)
    cost_table.to_csv(output / "cost_sensitivity.csv", index=False)

    if args.include_adaptive:
        bt = optional_adaptive_chart(market, opens, closes, config, archived, output)
        pd.DataFrame([{"sample": "retrospective_2022_2026",
                       "strategy": "Regime_Adaptive", **metric_record(bt, ann, rf)}]).to_csv(
            output / "adaptive_metrics.csv", index=False)

    summary = {
        "archived_oos": [FIRST, LAST],
        "retrospective_start": RETRO_START,
        "metrics_engine": "src.evaluator.calculate_metrics (252 sessions/year)",
        "replay_engine": "src.execution_engine.simulate_longshort_daily",
        "original_targets_reused": True,
        "borrowing_retained_in_zero_transaction_cost_scenario": True,
        "official_oos_files_unchanged": True,
        "post_freeze_cost_sensitivity_not_registered_oos": True,
    }
    (output / "chart_methodology.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== OFFICIAL & RETROSPECTIVE PERIOD METRICS ===")
    print(metric_table[["sample", "strategy", "cagr_pct", "sharpe",
                        "max_drawdown_pct"]].round(3).to_string(index=False))
    print("\n=== COUNTERFACTUAL TRANSACTION COST SENSITIVITY ===")
    print(cost_table[["strategy", "cagr_with_costs_pct",
                      "cagr_no_transaction_costs_pct", "delta_cagr_pp",
                      "historical_trading_costs_usd",
                      "historical_borrow_fees_usd"]].round(3).to_string(index=False))
    print(f"\nDone: charts, CSVs and methodology saved under {output}")
    print("Archived models, targets and official OOS results were not modified.")


if __name__ == "__main__":
    main()
