"""One-off diagnostic of the frozen B_v2 strategy on held-out dates.

Run from the repository root after copying this file there:
    python3 holdout_diagnostic_bv2.py --through 2021-12-31
    python3 holdout_diagnostic_bv2.py --through 2025-12-31

No modifications to main.py, config.yaml, frozen B_v1/B_v2 artifacts,
or signal-generation source files. This script does NOT establish
point-in-time validity of FRED macro series.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data_loader import load_dataset
from src.regime_allocator import ASSETS
from src.regime_strategy import build_level_b_v2_training
from src.longshort_backtester import simulate_longshort_daily
from src.evaluator import calculate_metrics

ROOT = Path(__file__).resolve().parent
TRAIN_END = pd.Timestamp("2016-12-31")
VALIDATION_END = pd.Timestamp("2021-12-31")
OOS_START = pd.Timestamp("2022-01-01")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_snapshot(config: dict) -> dict[str, str]:
    result = {}
    for key in ("signals", "results"):
        folder = ROOT / config["paths"][key] / "B"
        for variant in ("regime_longshort", "regime_longflat_v2"):
            base = folder / variant
            if not base.is_dir():
                raise FileNotFoundError(f"Missing frozen directory: {base}")
            for path in base.rglob("*"):
                if path.is_file():
                    result[str(path.relative_to(ROOT))] = digest(path)
    return result


def extended_config(original: dict, as_of: pd.Timestamp) -> dict:
    """Extend a *copy* of training boundaries solely for causal signal replay.

    B_v2 uses deterministic rolling rules, not an estimator fitted on
    validation/OOS returns. All strategy parameters must remain unchanged.
    """
    config = deepcopy(original)
    config["validation"]["train_end"] = as_of.date().isoformat()
    # Preserve chronological constraints inside the existing splitter.
    config["validation"]["validation_end"] = (
        as_of + pd.DateOffset(years=1)
    ).date().isoformat()
    config["validation"]["oos_start"] = (
        as_of + pd.DateOffset(years=1) + pd.Timedelta(days=1)
    ).date().isoformat()
    for key in ("validation_start", "val_start"):
        if key in config["validation"]:
            config["validation"][key] = (
                as_of + pd.Timedelta(days=1)
            ).date().isoformat()
    config["evaluation"]["evaluate_oos"] = False
    config["validation"]["development_only"] = True
    return config


def metric_record(name: str, frame: pd.DataFrame, rf_daily: float,
                  annualization: int) -> dict:
    if frame.empty:
        raise ValueError(f"Empty reporting period: {name}")
    ret = frame["daily_return"].astype(float)
    m = calculate_metrics(ret, rf_daily, annualization)
    growth = float(np.prod(1.0 + ret.to_numpy(dtype=float)))
    return {
        "period": name,
        "first_session": frame.index.min().date().isoformat(),
        "last_session": frame.index.max().date().isoformat(),
        "sessions": len(frame),
        "growth_of_100k": 100000.0 * growth,
        "total_return_pct": 100.0 * (growth - 1.0),
        "cagr_pct": float(m["cagr"]) * 100,
        "annualized_volatility_pct": float(m["annualized_volatility"]) * 100,
        "sharpe": float(m["sharpe_ratio"]),
        "max_drawdown_pct": float(m["max_drawdown"]) * 100,
        "transaction_costs_at_continuous_nav": float(
            frame["transaction_cost"].sum()
        ),
        "executed_rebalances": int(frame["executed"].sum()),
    }


def run(through: pd.Timestamp) -> None:
    if through not in (VALIDATION_END, pd.Timestamp("2025-12-31")):
        raise ValueError("Allowed --through dates: 2021-12-31 or 2025-12-31")

    cfg_path = ROOT / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as stream:
        original = yaml.safe_load(stream)

    assert original["evaluation"]["evaluate_oos"] is False
    assert original["validation"]["development_only"] is True
    assert original["features"]["macro_point_in_time_verified"] is False
    assert pd.Timestamp(original["validation"]["train_end"]) == TRAIN_END
    assert pd.Timestamp(original["validation"]["validation_end"]) == VALIDATION_END
    assert pd.Timestamp(original["validation"]["oos_start"]) == OOS_START

    before = frozen_snapshot(original)
    config_hash_before = digest(cfg_path)

    path = ROOT / original["paths"]["processed_data"] / "market_daily.csv"
    market_all = load_dataset(path, strict=True)
    last = market_all.index.max()
    print(f"Processed prices available: {market_all.index.min().date()} to {last.date()}")

    if last < through - pd.Timedelta(days=7):
        raise RuntimeError(
            f"Insufficient market coverage for requested {through.date()}: "
            f"last session is {last.date()}. No evaluation performed. "
            "Do not change main.py or config.yaml. Obtain a separate "
            "OOS-capable data preparation path first."
        )

    # Temporary in-memory boundary override. Original YAML is never edited.
    candidate = build_level_b_v2_training(extended_config(original, through), ROOT)
    full_weights = candidate["target_weights"]
    if (
        full_weights.empty
        or full_weights.index.max() > through
        or (full_weights.to_numpy(dtype=float) < -1e-10).any()
        or (full_weights.sum(axis=1) > 1 + 1e-10).any()
    ):
        raise RuntimeError("Extended B_v2 signals violate temporal/long-only constraints")

    frozen_weights = load_dataset(
        ROOT / original["paths"]["signals"] / "B"
        / "regime_longflat_v2" / "target_weights.csv",
        strict=True,
    )
    if not frozen_weights.index.isin(full_weights.index).all():
        raise RuntimeError("The historical training decisions changed")
    np.testing.assert_allclose(
        full_weights.reindex(frozen_weights.index)[list(ASSETS)].to_numpy(),
        frozen_weights[list(ASSETS)].to_numpy(),
        rtol=0,
        atol=1e-12,
        err_msg="Frozen training weights changed when future data were made available",
    )
    print("Frozen 2005-2016 target weights: EXACT MATCH")

    # Check an additional prefix before the held-out endpoint.
    if through > pd.Timestamp("2019-12-31"):
        prefix_end = pd.Timestamp("2019-12-31")
        prefix = build_level_b_v2_training(
            extended_config(original, prefix_end), ROOT
        )["target_weights"]
        np.testing.assert_allclose(
            full_weights.reindex(prefix.index)[list(ASSETS)].to_numpy(),
            prefix[list(ASSETS)].to_numpy(),
            rtol=0,
            atol=1e-12,
            err_msg="Later data changed earlier B_v2 target weights",
        )
        print("Causal prefix invariance through 2019: PASS")

    train_start = pd.Timestamp(original["validation"]["train_start"])
    market = market_all.loc[train_start:through].copy()
    if market.empty or not full_weights.index.isin(market.index).all():
        raise RuntimeError("Missing market sessions for extended decision dates")

    opens = market[[f"{asset}_open" for asset in ASSETS]].copy()
    closes = market[[f"{asset}_close" for asset in ASSETS]].copy()
    opens.columns = list(ASSETS)
    closes.columns = list(ASSETS)

    exe = original["execution"]
    b_cfg = original["strategies"]["B"]
    capital = float(exe["initial_capital"])
    bt = simulate_longshort_daily(
        open_prices=opens,
        close_prices=closes,
        target_weights=full_weights,
        initial_capital=capital,
        commission_bps=exe["commission_bps"],
        slippage_bps=exe["slippage_bps"],
        annual_borrow_rate=float(b_cfg["annual_borrow_rate"]),
        additional_margin_rate=float(b_cfg["additional_margin_rate"]),
    )
    if (bt["margin_shortfall"] > 1e-8).any() or not bt["borrow_fee"].eq(0).all():
        raise RuntimeError("Unexpected margin violation or short borrowing in B_v2")

    frozen_daily = load_dataset(
        ROOT / original["paths"]["results"] / "B"
        / "regime_longflat_v2" / "backtest_daily.csv",
        strict=True,
    )
    np.testing.assert_allclose(
        bt.reindex(frozen_daily.index)["nav"].to_numpy(),
        frozen_daily["nav"].to_numpy(),
        rtol=1e-9,
        atol=1e-5,
        err_msg="Continuous backtest did not reproduce frozen training NAV",
    )
    print("Frozen 2005-2016 NAV path: MATCH")

    ann = int(original["evaluation"]["annualization_factor"])
    rf_annual = float(original["evaluation"]["risk_free_rate"])
    rf_daily = (1.0 + rf_annual) ** (1.0 / ann) - 1.0

    results = [metric_record(
        "validation_2017_2021",
        bt.loc[(bt.index > TRAIN_END) & (bt.index <= VALIDATION_END)],
        rf_daily, ann,
    )]
    if through >= OOS_START:
        results.append(metric_record(
            "oos_2022_2025",
            bt.loc[(bt.index >= OOS_START) & (bt.index <= through)],
            rf_daily, ann,
        ))

    print("\nB_v2 | HELD-OUT DIAGNOSTIC | NO PARAMETER CHANGES")
    table = pd.DataFrame(results).set_index("period")
    print(table.round(4).to_string())
    print("Macro point-in-time status: NOT CERTIFIED")

    assert digest(cfg_path) == config_hash_before, "Original config changed"
    assert frozen_snapshot(original) == before, "Frozen B results changed"
    print("Config and frozen B_v1/B_v2 artifacts: UNCHANGED")

    # Maintain a separate, explicit record of inspected held-out periods.
    audit_dir = ROOT / original["paths"]["results"] / "B" / "heldout_diagnostic"
    audit = audit_dir / f"bv2_through_{through.date()}.json"
    if audit.exists():
        raise FileExistsError(f"Holdout diagnostic already exists: {audit}")
    manifest = {
        "description": "One-off held-out diagnostic of previously frozen B_v2",
        "observed_through": through.date().isoformat(),
        "validation_inspected": True,
        "oos_inspected": through >= OOS_START,
        "source_config_sha256": config_hash_before,
        "frozen_artifact_sha256": before,
        "method": "causal deterministic replay, continuous NAV, isolated period metrics",
        "warning": "Macro series point-in-time release/vintage status is not certified",
        "results": results,
    }
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print("Audit:", audit.relative_to(ROOT))
    print("B_v2 HELD-OUT DIAGNOSTIC - PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--through", required=True, choices=("2021-12-31", "2025-12-31"),
        help="Last calendar date to evaluate (2025 requires 2022+ market/macro data)",
    )
    args = parser.parse_args()
    run(pd.Timestamp(args.through))
