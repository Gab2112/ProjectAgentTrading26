"""One-shot frozen OOS evaluation: Core-Satellite vs benchmarks."""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from src.core_satellite import build_core_satellite_targets
from src.data_loader import load_daily_market_data
from src.evaluator import calculate_metrics
from src.execution_engine import (
    schedule_next_open_executions,
    simulate_longshort_daily,
)
from src.time_split import build_weekly_decision_dates
from src.utils import ASSETS, load_config


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path.cwd()
    config = load_config(root / "config.yaml")

    freeze_path = root / "reports/development_freeze_2021.json"
    audit_path = root / "reports/oos_input_audit.json"
    provenance_path = root / "reports/oos_signals/inference_provenance.json"

    freeze = json.loads(freeze_path.read_text())
    audit = json.loads(audit_path.read_text())
    provenance = json.loads(provenance_path.read_text())

    output_dir = root / "reports/oos_final_2022_2026"

    # No second OOS evaluation using this workflow.
    if output_dir.exists():
        raise FileExistsError(
            "Final OOS output already exists. "
            "Refusing a repeated evaluation."
        )

    # Verify the pre-registered strategy and temporal boundaries.
    assert freeze["status"] == "FROZEN_BEFORE_FINAL_OOS"
    assert freeze["selected_strategy"] == "Core35_Satellite65"
    assert freeze["core_weight"] == 0.35
    assert freeze["satellite_weight"] == 0.65
    assert freeze["training_end"] == "2021-12-31"
    assert freeze["final_oos_start"] == "2022-01-01"
    assert freeze["finbert_enabled"] is False

    assert config["validation"]["train_end"] == freeze["training_end"]
    assert config["validation"]["oos_start"] == freeze["final_oos_start"]
    assert config["validation"]["development_only"] is True
    assert config["evaluation"]["evaluate_oos"] is False
    assert config["strategies"]["C"]["sentiment_enabled"] is False

    assert config["execution"]["execution_rule"] == "next_available_trading_open"
    assert config["execution"]["signal_frequency"] == "weekly"
    assert config["execution"]["cost_basis"] == "traded_notional"
    assert float(config["strategies"]["B"]["cash_interest_rate"]) == 0.0

    assert provenance["model_fit_after_2021"] is False
    assert provenance["finbert_included"] is False
    assert sha(freeze_path) == provenance["frozen_manifest_sha256"]
    assert sha(audit_path) == provenance["input_audit_sha256"]
    assert (
        sha(root / "src/frozen_oos_inference.py")
        == provenance["inference_code_sha256"]
    )

    # Verify all 47 frozen files and all raw inputs.
    for relative, expected in freeze["code_and_model_sha256"].items():
        if sha(root / relative) != expected:
            raise RuntimeError(f"Frozen file changed: {relative}")

    for relative, expected in audit["raw_input_sha256"].items():
        if sha(root / relative) != expected:
            raise RuntimeError(f"Raw input changed: {relative}")

    print("OK: freeze, models, configuration and raw data verified.")

    market, _ = load_daily_market_data(config, root)

    assert (
        market.index.max().date().isoformat()
        == audit["last_common_oos_session"]
    )

    all_decisions = build_weekly_decision_dates(market).index
    oos_start = pd.Timestamp(freeze["final_oos_start"])
    decisions = all_decisions[all_decisions >= oos_start]

    assert len(decisions) == provenance["signals_per_asset"]
    assert decisions.min().date().isoformat() == provenance["first_signal"]
    assert decisions.max().date().isoformat() == provenance["last_signal"]

    signals = {}

    for asset in ASSETS:
        path = (
            root / "reports/oos_signals"
            / f"signals_orchestrator_{asset.lower()}.csv"
        )

        if sha(path) != provenance["output_sha256"][asset]:
            raise RuntimeError(f"Frozen OOS signals changed: {asset}")

        df = pd.read_csv(
            path,
            index_col="Date",
            parse_dates=["Date"],
        )

        if (
            not df.index.equals(decisions)
            or df[["signal", "confidence"]].isna().any().any()
            or not df["signal"].isin([-1, 0, 1]).all()
            or not df["confidence"].between(0, 1).all()
        ):
            raise ValueError(f"Invalid OOS signals: {asset}")

        signals[asset] = df

    # Allocations are built using the pre-registered Core-Satellite rule.
    core = build_core_satellite_targets(
        market_daily=market,
        signals_by_asset=signals,
        train_start=oos_start,
        train_end=decisions.max(),
        core_weight=freeze["core_weight"],
    )

    if not core.index.equals(decisions):
        raise RuntimeError("Core-Satellite decision calendar mismatch")

    # Benchmarks have the same starting capital and trading calendar.
    spy = pd.DataFrame(
        0.0, index=decisions[:1], columns=list(ASSETS)
    )
    spy["SPY"] = 1.0

    sixty_forty = pd.DataFrame(
        0.0, index=decisions, columns=list(ASSETS)
    )
    sixty_forty["SPY"] = 0.60
    sixty_forty["IEF"] = 0.40

    targets = {
        "Core35_Satellite65": core,
        "SPY_buy_hold": spy,
        "Sixty_Forty": sixty_forty,
    }

    # Start all portfolios at the first OOS decision Close.
    # Orders execute only at subsequent actual trading Opens.
    evaluation_market = market.loc[
        decisions.min():decisions.max()
    ].copy()

    opens = evaluation_market[
        [f"{a}_open" for a in ASSETS]
    ].copy()

    closes = evaluation_market[
        [f"{a}_close" for a in ASSETS]
    ].copy()

    opens.columns = list(ASSETS)
    closes.columns = list(ASSETS)

    assert opens.index.equals(closes.index)
    assert opens.index.min() == decisions.min()
    assert opens.index.max() == decisions.max()

    settings = config["execution"]
    borrow = config["strategies"]["B"]

    initial = float(settings["initial_capital"])
    commission = float(settings["commission_bps"])
    slippage = float(settings["slippage_bps"])
    borrow_rate = float(borrow["annual_borrow_rate"])
    margin_rate = float(borrow["additional_margin_rate"])

    annualization = int(config["evaluation"]["annualization_factor"])
    annual_rf = float(config["evaluation"]["risk_free_rate"])
    daily_rf = (1.0 + annual_rf) ** (1.0 / annualization) - 1.0

    records = []
    backtests = {}
    execution_dates = {}

    for name, weights in targets.items():
        schedule = schedule_next_open_executions(
            weights, opens.index
        )

        if name == "Core35_Satellite65":
            # Last signal cannot execute without another market session.
            assert len(schedule) == len(decisions) - 1
            assert schedule.index.min() > decisions.min()
            assert schedule.index.max() <= decisions.max()

        bt = simulate_longshort_daily(
            open_prices=opens,
            close_prices=closes,
            target_weights=weights,
            initial_capital=initial,
            commission_bps=commission,
            slippage_bps=slippage,
            annual_borrow_rate=borrow_rate,
            additional_margin_rate=margin_rate,
        )

        if (
            bt.empty
            or not bt.index.equals(opens.index)
            or not np.isfinite(bt["nav"]).all()
            or not np.isfinite(bt["daily_return"]).all()
            or (bt["nav"] <= 0).any()
            or (bt["margin_shortfall"] > 1e-8).any()
        ):
            raise RuntimeError(f"Invalid OOS backtest: {name}")

        assert not bool(bt["executed"].iloc[0])

        np.testing.assert_allclose(
            float(np.prod(1.0 + bt["daily_return"])),
            float(bt["nav"].iloc[-1] / initial),
            rtol=1e-8,
            atol=1e-9,
        )

        metrics = calculate_metrics(
            bt["daily_return"],
            daily_rf,
            annualization,
        )

        records.append({
            "strategy": name,
            **metrics,
            "final_nav": float(bt["nav"].iloc[-1]),
            "nonzero_trades": int(bt["executed"].sum()),
            "scheduled_rebalances": len(schedule),
            "total_turnover": float(bt["turnover"].sum()),
            "transaction_costs": float(
                bt["transaction_cost"].sum()
            ),
            "borrow_fees": float(bt["borrow_fee"].sum()),
            "mean_gross_exposure": float(
                bt["gross_exposure"].mean()
            ),
        })

        backtests[name] = bt
        execution_dates[name] = {
            "first": schedule.index.min().date().isoformat(),
            "last": schedule.index.max().date().isoformat(),
        }

    summary = pd.DataFrame(records).set_index("strategy")

    # Verify protected inputs again before creating the final report.
    for relative, expected in freeze["code_and_model_sha256"].items():
        assert sha(root / relative) == expected

    for asset in ASSETS:
        path = (
            root / "reports/oos_signals"
            / f"signals_orchestrator_{asset.lower()}.csv"
        )
        assert sha(path) == provenance["output_sha256"][asset]

    # Save a single permanent evaluation record.
    output_dir.mkdir(parents=True, exist_ok=False)

    summary.to_csv(output_dir / "comparison_metrics.csv")
    core.to_csv(output_dir / "core_satellite_targets.csv", index_label="Date")

    for name, bt in backtests.items():
        bt.to_csv(
            output_dir / f"{name}_daily.csv",
            index_label="Date",
        )

    report = {
        "status": "FINAL_OOS_EVALUATED_ONCE",
        "selected_strategy": freeze["selected_strategy"],
        "training_end": freeze["training_end"],
        "first_decision": decisions.min().date().isoformat(),
        "last_decision": decisions.max().date().isoformat(),
        "first_accounted_session": opens.index.min().date().isoformat(),
        "last_accounted_session": opens.index.max().date().isoformat(),
        "last_signal_without_next_open": True,
        "number_of_weekly_decisions": len(decisions),
        "number_of_daily_sessions": len(opens),
        "execution_dates": execution_dates,
        "model_retrained": False,
        "strategy_retuned_on_oos": False,
        "finbert_used": False,
        "macro_point_in_time_verified": False,
        "price_data": "Auto-adjusted ETF Open/Close",
        "accounting": "Shared daily next-open engine",
        "commission_bps": commission,
        "slippage_bps": slippage,
        "annual_short_borrow_rate": borrow_rate,
        "frozen_manifest_sha256": sha(freeze_path),
        "signal_provenance_sha256": sha(provenance_path),
        "backtest_code_sha256": sha(
            root / "src/final_oos_backtest.py"
        ),
    }

    (output_dir / "evaluation_provenance.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    view = summary[[
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "nonzero_trades",
        "final_nav",
    ]].copy()

    for col in ("cagr", "annualized_volatility", "max_drawdown"):
        view[col] *= 100.0

    view.columns = [
        "CAGR_%",
        "Vol_%",
        "Sharpe",
        "Max_DD_%",
        "Trades",
        "Final_NAV",
    ]

    print("\n=== FINAL OOS: 2022-2026 ===")
    print(
        f"Decisions: {decisions.min().date()} "
        f"to {decisions.max().date()}"
    )
    print(view.round(3).to_string())

    print("\nOK: single final OOS evaluation recorded.")
    print("OK: frozen model and signals unchanged.")
    print("Macro PIT vintage status: NOT VERIFIED.")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
