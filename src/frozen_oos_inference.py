"""Frozen-model 2022+ signal inference. No fitting or backtesting."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from src.agents.EMA_EMW_agent import SMAEMAAgent
from src.agents.ZScore_agent import ZScoreAgent
from src.agents.macro_yield_curve_agent import MacroYieldCurveAgent
from src.agents.random_forest_agent import RandomForestAgent
from src.agents.SVM_agent import SVMAgent
from src.agents.orchestrator_agent import OrchestratorAgent
from src.data_loader import load_daily_market_data, load_agent_macro_data
from src.features import build_decision_feature_panel
from src.time_split import build_weekly_decision_dates
from src.utils import ASSETS, ROOT_DIR, load_config


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    root = ROOT_DIR
    config = load_config(root / "config.yaml")

    manifest = json.loads(
        (root / "reports/development_freeze_2021.json").read_text()
    )
    audit = json.loads(
        (root / "reports/oos_input_audit.json").read_text()
    )

    assert manifest["status"] == "FROZEN_BEFORE_FINAL_OOS"
    assert manifest["selected_strategy"] == "Core35_Satellite65"
    assert config["validation"]["train_end"] == manifest["training_end"]
    assert config["validation"]["oos_start"] == manifest["final_oos_start"]
    assert config["validation"]["development_only"] is True
    assert config["evaluation"]["evaluate_oos"] is False
    assert config["strategies"]["C"]["sentiment_enabled"] is False

    for relative, expected in manifest["code_and_model_sha256"].items():
        if sha(root / relative) != expected:
            raise RuntimeError(f"Frozen file changed: {relative}")

    for relative, expected in audit["raw_input_sha256"].items():
        if sha(root / relative) != expected:
            raise RuntimeError(f"Raw input changed: {relative}")

    print("OK: frozen artifacts and raw inputs unchanged")

    market, _ = load_daily_market_data(config, root)
    cutoff = market.index.max()

    assert cutoff.date().isoformat() == audit["last_common_oos_session"]

    decisions = build_weekly_decision_dates(market).index
    first_oos = pd.Timestamp(manifest["final_oos_start"])
    oos_dates = decisions[decisions >= first_oos]

    assert len(oos_dates) > 30
    assert oos_dates.max() <= cutoff

    # Historical warm-up is retained for rolling indicators.
    full_panel = build_decision_feature_panel(market, decisions)

    macro_panel = load_agent_macro_data(
        decision_dates=full_panel.index,
        root_dir=root,
    )
    full_panel = full_panel.join(macro_panel, how="left")

    agent_names = (
        "SMA_EMA_Agent",
        "ZScore_Agent",
        "Macro_YieldCurve_Agent",
        "RandomForest_Agent",
        "SVM_Agent",
    )

    def pack(pair, asset, agent):
        signals, confidences = pair

        frame = pd.DataFrame({
            "signal": signals.reindex(full_panel.index),
            "confidence": confidences.reindex(full_panel.index),
        }, index=full_panel.index).loc[oos_dates]

        if frame.isna().any().any():
            raise ValueError(f"{asset}/{agent}: missing predictions")

        if (
            not frame["signal"].isin([-1, 0, 1]).all()
            or not frame["confidence"].between(0, 1).all()
        ):
            raise ValueError(f"{asset}/{agent}: invalid predictions")

        return frame

    outputs = {}
    cfg_dir = root / "src/agents/config"

    for asset in ASSETS:
        suffix = asset.lower()

        other_prefixes = tuple(
            f"{other}_"
            for other in ASSETS
            if other != asset
        )

        cols = [
            col for col in full_panel.columns
            if not col.startswith(other_prefixes)
        ]

        asset_panel = full_panel[cols].copy()

        # Statistical agents with frozen parameters.
        sma = SMAEMAAgent(target_asset=asset)
        sma.load_params(cfg_dir / f"sma_ema_params_{suffix}.json")

        zs = ZScoreAgent(target_asset=asset)
        zs.load_params(cfg_dir / f"zscore_params_{suffix}.json")

        # Previously fitted ML models. No .fit() calls.
        rf = RandomForestAgent(target_asset=asset)
        rf.load_params(cfg_dir / f"random_forest_params_{suffix}.json")

        svm = SVMAgent(target_asset=asset, root_dir=root)
        svm.load_params(cfg_dir / f"svm_params_{suffix}.json")

        if rf.model is None or svm.model is None:
            raise RuntimeError(f"{asset}: frozen ML model not loaded")

        if not set(rf.trained_features).issubset(asset_panel.columns):
            raise RuntimeError(f"{asset}: missing RF features")

        # Reproduce the training macro signal construction.
        macro = MacroYieldCurveAgent(target_asset=asset)

        macro_multi = macro.predict_multi_asset(
            asset_panel, assets=[asset]
        )
        macro_signal = macro_multi[f"{asset}_signal"]
        macro_score = macro.calculate_macro_score(asset_panel)

        macro_conf = pd.Series(
            np.where(
                macro_signal != 0,
                np.maximum(
                    0.50,
                    (macro_score.abs() / 2.0).clip(0, 1).round(4),
                ),
                0.0,
            ),
            index=asset_panel.index,
        )

        # Explicitly fixed pool: FinBERT cannot enter.
        pool = {
            agent_names[0]: pack(
                sma.predict_with_confidence(asset_panel),
                asset, agent_names[0],
            ),
            agent_names[1]: pack(
                zs.predict_with_confidence(asset_panel),
                asset, agent_names[1],
            ),
            agent_names[2]: pack(
                (macro_signal, macro_conf),
                asset, agent_names[2],
            ),
            agent_names[3]: pack(
                rf.predict_with_confidence(asset_panel),
                asset, agent_names[3],
            ),
            agent_names[4]: pack(
                svm.predict_with_confidence(full_panel),
                asset, agent_names[4],
            ),
        }

        # Frozen GA weights and thresholds.
        frozen = json.loads(
            (cfg_dir / f"orchestrator_params_{suffix}.json").read_text()
        )
        params = frozen["best_params"]

        if (
            int(frozen["max_train_year"]) != 2021
            or set(params["weights"]) != set(agent_names)
            or set(params["alphas"]) != set(agent_names)
        ):
            raise RuntimeError(f"{asset}: invalid frozen ensemble")

        orchestrator = OrchestratorAgent(
            weights=params["weights"],
            alphas=params["alphas"],
            theta_long=params["threshold_long"],
            theta_short=params["threshold_short"],
            target_asset=asset,
        )

        result = orchestrator.combine(pool)

        if (
            not result.index.equals(oos_dates)
            or result.isna().any().any()
            or not result["signal"].isin([-1, 0, 1]).all()
            or not result["confidence"].between(0, 1).all()
        ):
            raise ValueError(f"{asset}: invalid final signals")

        outputs[asset] = result

    output_dir = root / "reports/oos_signals"
    paths = {
        asset: output_dir / f"signals_orchestrator_{asset.lower()}.csv"
        for asset in ASSETS
    }
    provenance_path = output_dir / "inference_provenance.json"

    if any(p.exists() for p in [*paths.values(), provenance_path]):
        raise FileExistsError("OOS signals already exist: refusing overwrite")

    output_dir.mkdir(parents=True, exist_ok=True)

    for asset, result in outputs.items():
        result.to_csv(paths[asset], index_label="Date")

    provenance = {
        "sample": "2022+ signals; no OOS backtest executed",
        "model_fit_after_2021": False,
        "finbert_included": False,
        "macro_point_in_time_verified": False,
        "first_signal": oos_dates.min().date().isoformat(),
        "last_signal": oos_dates.max().date().isoformat(),
        "signals_per_asset": len(oos_dates),
        "inference_code_sha256": sha(
            root / "src/frozen_oos_inference.py"
        ),
        "frozen_manifest_sha256": sha(
            root / "reports/development_freeze_2021.json"
        ),
        "input_audit_sha256": sha(
            root / "reports/oos_input_audit.json"
        ),
        "output_sha256": {
            asset: sha(path)
            for asset, path in paths.items()
        },
    }

    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    print(
        f"OK: {len(oos_dates)} weekly signals per ETF, "
        f"{oos_dates.min().date()} -> {oos_dates.max().date()}"
    )
    print("OK: no retraining, no FinBERT, no performance evaluation")
    print(f"Signals saved to: {output_dir}")


if __name__ == "__main__":
    run()
