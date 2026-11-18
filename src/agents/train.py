"""Training of every quantitative agent on the in-sample period (step 3 of the pipeline).

Agents
- SMAEMAAgent: trend following on a simple/exponential moving-average pair;
- ZScoreAgent: momentum or mean-reversion rule on a rolling z-score;
- MacroYieldCurveAgent: rule-based signal from yield-curve slope and inflation;
- RandomForestAgent: classifier of the next-week direction (macro and price features);
- SVMAgent: support-vector classifier of the four-week direction;
- OrchestratorAgent: ensemble that combines the five agents with weights,
  confidence-calibration exponents and decision thresholds found by a
  real-coded genetic algorithm.

Protocol
- Only observations dated on or before ``validation.train_end`` are loaded;
  earlier history (before ``train_start``) is used as indicator warm-up only.
- SMA/EMA, Z-score and Random Forest hyperparameters are selected with a
  rolling walk-forward scheme (``agents.walk_forward_train_years`` of
  training, one year of test, rolled yearly).
- The SVM is refitted every year on all prior data with an inner
  time-series cross-validation (purged by the label horizon).
- The Random Forest and SVM produce OUT-OF-FOLD predictions: each weekly
  prediction comes from a model fitted on earlier data only.
- The Orchestrator is fitted on the out-of-fold signal pool, so that it
  never learns from predictions that saw their own labels. It is fitted in
  two ways: (i) a yearly expanding-window walk-forward, whose test-year
  outputs are out-of-sample with respect to the Orchestrator parameters and
  are the in-sample signals used downstream; (ii) a final fit on the whole
  pool, which is the model applied to the out-of-sample period.

Outputs
- src/agents/config/*.json, *.joblib   final parameters and fitted models;
- data/signals/signals_<agent>_<asset>.csv   in-sample signal history (the
  ML columns are out-of-fold; the Orchestrator file is the walk-forward output);
- data/signals/signals_orchestrator_fullfit_<asset>.csv   final fit applied
  to its own training weeks (in-sample, diagnostic only);
- data/results/training/*.csv   grid-search logs, summaries and the
  Orchestrator walk-forward folds.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.EMA_EMW_agent import SMAEMAAgent, SMAEMAOptimizer, WalkForwardTrainer as SMAEMATrainer
from src.agents.macro_yield_curve_agent import MacroYieldCurveAgent
from src.agents.orchestrator_agent import OrchestratorOptimizer, walk_forward_orchestrator
from src.agents.random_forest_agent import RandomForestOptimizer, RandomForestWalkForwardTrainer
from src.agents.SVM_agent import SVMWalkForwardTrainer
from src.agents.ZScore_agent import ZScoreAgent, ZScoreOptimizer, ZScoreWalkForwardTrainer
from src.data_loader import load_agent_macro_data
from src.evaluator import calculate_performance_metrics, compute_objective_score
from src.features import build_decision_feature_panel
from src.time_split import get_time_windows
from src.utils import ASSETS

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")


def save_training_provenance(config_path: Path, training_df: pd.DataFrame, cutoff_year: int) -> None:
    """Record in the model file the period on which the model was fitted."""
    with config_path.open("r", encoding="utf-8") as f:
        params = json.load(f)

    params["training_provenance"] = {
        "start_date": training_df.index.min().date().isoformat(),
        "end_date": training_df.index.max().date().isoformat(),
        "cutoff_year": int(cutoff_year),
        "calendar": "weekly_decision_dates.csv",
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=4)


def _signal_frame(signal: pd.Series, confidence: pd.Series, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {"signal": signal.reindex(index), "confidence": confidence.reindex(index)},
        index=index,
    )


def _print_in_sample_row(label: str, metrics: dict, objective_metric: str) -> None:
    print(
        f"{label:<28} | {metrics['cumulative_return'] * 100:+8.1f}% | {metrics['sharpe_ratio']:>7.2f} "
        f"| {metrics['max_drawdown'] * 100:>6.1f}% | {metrics['profit_factor']:>9.2f} "
        f"| {compute_objective_score(metrics, objective_metric):>9.2f}"
    )


# =====================================================================
# SINGLE-ASSET TRAINING
# =====================================================================
def train_single_asset(
        asset: str,
        df: pd.DataFrame,
        config: dict,
        config_dir: Path,
        signals_dir: Path,
        results_dir: Path,
        max_train_year: int,
        warmup_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Train every agent for one asset and fit its Orchestrator."""
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        raise ValueError("Training requires a non-empty DatetimeIndex.")
    if not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError("Training dates must be sorted and unique.")
    if (df.index.year > max_train_year).any():
        raise ValueError("The training dataset contains observations after the cut-off.")
    if warmup_df is not None and not warmup_df.empty and warmup_df.index.max() >= df.index.min():
        raise ValueError("The warm-up window overlaps the training window.")

    agents_cfg = config["agents"]
    objective_metric = agents_cfg["objective_metric"]
    wf_years = int(agents_cfg["walk_forward_train_years"])

    asset_upper, asset_lower = asset.upper(), asset.lower()
    price_col = f"{asset_upper}_close"
    if price_col not in df.columns:
        raise KeyError(f"Column '{price_col}' not found in the dataset.")

    # The SVM uses cross-asset momentum; the other agents see only their own asset.
    df_all_assets = df.copy()
    excluded_prefixes = tuple(f"{a}_" for a in ASSETS if a != asset_upper)
    own_cols = [c for c in df.columns if not c.startswith(excluded_prefixes)]
    df = df[own_cols].copy()
    if warmup_df is not None and not warmup_df.empty:
        warmup_df = warmup_df[own_cols].copy()

    print("\n" + "#" * 84)
    print(f"### TRAINING: {asset_upper} ({len(df)} weeks, {df.index.min().date()} -> {df.index.max().date()})")
    print("#" * 84)

    # 1. SMA/EMA trend agent (walk-forward grid search)
    print(f"\n[{asset_upper}] Walk-forward: SMA/EMA agent")
    sma_opt = SMAEMAOptimizer(
        price_col=price_col,
        objective_metric=objective_metric,
        sma_range=list(agents_cfg["sma_ema"]["sma_windows"]),
        ema_range=list(agents_cfg["sma_ema"]["ema_windows"]),
    )
    sma_trainer = SMAEMATrainer(optimizer=sma_opt, train_window_years=wf_years, price_col=price_col)
    sma_res = sma_trainer.run(df, max_year=max_train_year, initial_warmup_df=warmup_df)
    sma_config_file = config_dir / f"sma_ema_params_{asset_lower}.json"
    sma_trainer.save_final_agent_config(sma_config_file)
    sma_trainer.save_combinations_log(results_dir / f"grid_search_combinations_sma_ema_{asset_lower}.csv")
    sma_trainer.save_combinations_summary(results_dir / f"summary_combinations_sma_ema_{asset_lower}.csv")

    # 2. Z-score agent (walk-forward grid search)
    print(f"\n[{asset_upper}] Walk-forward: Z-score agent")
    zs_opt = ZScoreOptimizer(
        price_col=price_col,
        exit_threshold=0.0,
        objective_metric=objective_metric,
        window_range=list(agents_cfg["zscore"]["windows"]),
        threshold_range=list(agents_cfg["zscore"]["thresholds"]),
        mode_range=list(agents_cfg["zscore"]["modes"]),
    )
    zs_trainer = ZScoreWalkForwardTrainer(optimizer=zs_opt, train_window_years=wf_years, price_col=price_col)
    zs_res = zs_trainer.run(df, max_year=max_train_year, initial_warmup_df=warmup_df)
    zs_config_file = config_dir / f"zscore_params_{asset_lower}.json"
    zs_trainer.save_final_agent_config(zs_config_file)
    zs_trainer.save_combinations_log(results_dir / f"grid_search_combinations_zscore_{asset_lower}.csv")
    zs_trainer.save_combinations_summary(results_dir / f"summary_combinations_zscore_{asset_lower}.csv")

    # 3. Signals of the rule-based agents with their final parameters.
    history = pd.concat([warmup_df, df]).sort_index() if warmup_df is not None and not warmup_df.empty else df

    sma_agent = SMAEMAAgent(price_col=price_col, target_asset=asset_upper)
    sma_agent.load_params(sma_config_file)
    sig_sma, conf_sma = sma_agent.predict_with_confidence(history)
    df_sma = _signal_frame(sig_sma, conf_sma, df.index)
    df_sma.to_csv(signals_dir / f"signals_sma_ema_{asset_lower}.csv")

    zs_agent = ZScoreAgent(price_col=price_col, target_asset=asset_upper)
    zs_agent.load_params(zs_config_file)
    sig_zs, conf_zs = zs_agent.predict_with_confidence(history)
    df_zs = _signal_frame(sig_zs, conf_zs, df.index)
    df_zs.to_csv(signals_dir / f"signals_zscore_{asset_lower}.csv")

    # 4. Random Forest (walk-forward, out-of-fold predictions)
    print(f"\n[{asset_upper}] Walk-forward: Random Forest agent")
    rf_cfg = agents_cfg["random_forest"]
    rf_opt = RandomForestOptimizer(
        target_asset=asset_upper,
        price_col=price_col,
        depth_candidates=rf_cfg["depth_candidates"],
        min_leaf_candidates=rf_cfg["min_leaf_candidates"],
        threshold_candidates=rf_cfg["threshold_candidates"],
        n_estimators_candidates=rf_cfg["n_estimators_candidates"],
        objective_metric=objective_metric,
    )
    rf_trainer = RandomForestWalkForwardTrainer(optimizer=rf_opt, train_window_years=wf_years, target_asset=asset_upper)
    rf_res = rf_trainer.run(df, max_year=max_train_year)
    rf_config_file = config_dir / f"random_forest_params_{asset_lower}.json"
    rf_trainer.save_final_agent_config(rf_config_file)
    save_training_provenance(rf_config_file, df, max_train_year)
    df_rf = rf_trainer.oof_signals.copy()
    if df_rf.empty or df_rf[["signal", "confidence"]].isna().any().any():
        raise ValueError("Random Forest out-of-fold predictions are missing or incomplete.")
    df_rf.to_csv(signals_dir / f"signals_random_forest_{asset_lower}.csv")

    # 5. SVM (yearly refit, out-of-fold predictions)
    print(f"\n[{asset_upper}] Walk-forward: SVM agent")
    svm_trainer = SVMWalkForwardTrainer(target_asset=asset_upper, root_dir=PROJECT_ROOT)
    svm_res = svm_trainer.run(df_all_assets, max_year=max_train_year)
    svm_config_file = config_dir / f"svm_params_{asset_lower}.json"
    svm_trainer.save_final_agent_config(svm_config_file)
    save_training_provenance(svm_config_file, df, max_train_year)
    df_svm = svm_trainer.oof_signals.copy()
    if df_svm.empty or df_svm[["signal", "confidence"]].isna().any().any():
        raise ValueError("SVM out-of-fold predictions are missing or incomplete.")
    df_svm.to_csv(signals_dir / f"signals_svm_{asset_lower}.csv")

    # 6. Macro yield-curve agent (rule-based, no fitted parameters)
    macro_agent = MacroYieldCurveAgent(target_asset=asset_upper)
    m_sig = macro_agent.predict_multi_asset(df, assets=[asset_upper])[f"{asset_upper}_signal"]
    m_conf = (macro_agent.calculate_macro_score(df).abs() / 2.0).clip(0.0, 1.0).round(4)
    m_conf = pd.Series(np.where(m_sig != 0, np.maximum(0.50, m_conf), 0.0), index=df.index)
    df_macro = pd.DataFrame({"signal": m_sig, "confidence": m_conf}, index=df.index)
    df_macro.to_csv(signals_dir / f"signals_macro_{asset_lower}.csv")

    # Weeks on which both ML agents have an out-of-fold prediction.
    oof_index = df.index.intersection(df_rf.index).intersection(df_svm.index)
    if oof_index.empty:
        raise ValueError("No common out-of-fold week is available.")

    # 7. Orchestrator (genetic algorithm on the out-of-fold signal pool)
    signals_pool = {
        "SMA_EMA_Agent": df_sma,
        "ZScore_Agent": df_zs,
        "Macro_YieldCurve_Agent": df_macro,
        "RandomForest_Agent": df_rf,
        "SVM_Agent": df_svm,
    }
    signals_pool = {name: frame.loc[oof_index].copy() for name, frame in signals_pool.items()}
    if any(frame[["signal", "confidence"]].isna().any().any() for frame in signals_pool.values()):
        raise ValueError("The out-of-fold signal pool contains missing values.")

    ret_asset = df[price_col].astype(float).pct_change().fillna(0.0)
    orch_opt = OrchestratorOptimizer(
        objective_metric=objective_metric,
        max_train_year=max_train_year,
        threshold_range=tuple(agents_cfg["orchestrator"]["threshold_range"]),
        population_size=int(agents_cfg["orchestrator"]["population_size"]),
        n_generations=int(agents_cfg["orchestrator"]["n_generations"]),
        random_seed=int(config["project"]["random_seed"]),
    )
    best_orch_agent, df_summary_orch = orch_opt.optimize(
        signals_dict=signals_pool,
        returns_or_prices=ret_asset.loc[oof_index],
        asset_name=asset_upper,
        method=agents_cfg["orchestrator_method"],
    )
    orch_opt.save_final_config(config_dir / f"orchestrator_params_{asset_lower}.json")
    df_summary_orch.to_csv(results_dir / f"summary_combinations_orchestrator_{asset_lower}.csv", index=False)
    df_orch_full = best_orch_agent.combine(signals_pool)
    df_orch_full.to_csv(signals_dir / f"signals_orchestrator_fullfit_{asset_lower}.csv")

    # 7b. Yearly expanding-window walk-forward of the Orchestrator.
    orch_cfg = agents_cfg["orchestrator"]
    print(f"\n[{asset_upper}] Orchestrator walk-forward (expanding window, yearly refit)")
    df_orch, wf_folds = walk_forward_orchestrator(
        signals_dict=signals_pool,
        returns=ret_asset.loc[oof_index],
        last_test_year=max_train_year,
        min_train_years=int(orch_cfg["walk_forward_min_train_years"]),
        objective_metric=objective_metric,
        population_size=int(orch_cfg["population_size"]),
        n_generations=int(orch_cfg["n_generations"]),
        random_seed=int(config["project"]["random_seed"]),
    )
    wf_folds.to_csv(results_dir / f"orchestrator_walk_forward_{asset_lower}.csv", index=False)
    df_orch.to_csv(signals_dir / f"signals_orchestrator_{asset_lower}.csv")
    for fold in wf_folds.itertuples():
        print(f"  Test {fold.test_year} | train {fold.train_start} -> {fold.train_end} | "
              f"train Sharpe {fold.train_sharpe:+.2f} | test Sharpe {fold.test_sharpe:+.2f}")

    # 8. In-sample diagnostic table (weekly signal returns, signal lagged one week).
    print("\n" + "-" * 88)
    print(f"  IN-SAMPLE AGENT COMPARISON ({oof_index.min():%Y-%m-%d} - {oof_index.max():%Y-%m-%d}) - {asset_upper}")
    print("  Note: SMA/EMA, Z-score and Orchestrator parameters were selected on this window.")
    print("-" * 88)
    print(f"{'Agent':<28} | {'Cum. ret.':>9} | {'Sharpe':>7} | {'Max DD':>7} | {'Prof. f.':>9} | {'Objective':>9}")
    print("-" * 88)
    for label, frame in (
        ("SMA/EMA", df_sma), ("Z-score", df_zs), ("Macro yield curve", df_macro),
        ("Random Forest (OOF)", df_rf), ("SVM (OOF)", df_svm), ("Orchestrator (full fit)", df_orch_full),
    ):
        strat = (frame["signal"].reindex(df.index).shift(1).fillna(0.0) * ret_asset).loc[oof_index]
        _print_in_sample_row(label, calculate_performance_metrics(strat), objective_metric)
    print(f"  Walk-forward window {df_orch.index.min():%Y-%m-%d} - {df_orch.index.max():%Y-%m-%d}:")
    for label, frame in (("Orchestrator (full fit)", df_orch_full), ("Orchestrator (walk-forward)", df_orch)):
        strat = (frame["signal"].reindex(df.index).shift(1).fillna(0.0) * ret_asset).loc[df_orch.index]
        _print_in_sample_row(label, calculate_performance_metrics(strat), objective_metric)
    print("-" * 88)

    return {
        "asset": asset_upper,
        "sma": sma_res,
        "zscore": zs_res,
        "random_forest": rf_res,
        "svm": svm_res,
        "orchestrator": orch_opt.best_params,
        "signals_orchestrator": df_orch,
        "signals_orchestrator_fullfit": df_orch_full,
    }


# =====================================================================
# MULTI-ASSET TRAINING
# =====================================================================
def run_all_assets(config: dict, root_dir: Path = PROJECT_ROOT, assets: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Train every agent for every asset using in-sample data only."""
    windows = get_time_windows(config)
    train_start, train_end = windows["train_start"], windows["train_end"]
    max_train_year = train_end.year
    if train_end != pd.Timestamp(year=max_train_year, month=12, day=31):
        raise ValueError("train_end must be 31 December: the agents refit on calendar years.")
    assets = list(assets or ASSETS)

    config_dir = root_dir / config["paths"]["models"]
    signals_dir = root_dir / config["paths"]["signals"]
    results_dir = root_dir / config["paths"]["results"] / "training"
    for directory in (config_dir, signals_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)

    processed = root_dir / config["paths"]["processed_data"]
    market_daily = pd.read_csv(processed / "market_daily.csv", index_col="Date", parse_dates=["Date"]).loc[:train_end]
    decision_dates = pd.read_csv(processed / "weekly_decision_dates.csv", index_col="Date", parse_dates=["Date"]).index
    decision_dates = decision_dates[decision_dates <= train_end]

    df = build_decision_feature_panel(market_daily, decision_dates)
    df = df.join(load_agent_macro_data(decision_dates=df.index, root_dir=root_dir), how="left")
    if df.index.max() > train_end:
        raise RuntimeError("Out-of-sample observations reached the training dataset.")

    warmup_df = df.loc[df.index < train_start].copy()
    train_df = df.loc[(df.index >= train_start) & (df.index <= train_end)].copy()
    if train_df.empty:
        raise ValueError("No data in the configured training period.")

    print(f"[Train] Dataset: {len(train_df)} weeks ({train_df.index[0].date()} -> {train_df.index[-1].date()}), "
          f"warm-up: {len(warmup_df)} weeks")
    print(f"[Train] Assets: {assets} | objective: {config['agents']['objective_metric']}")

    all_results = {
        asset.upper(): train_single_asset(
            asset=asset,
            df=train_df,
            config=config,
            warmup_df=warmup_df,
            config_dir=config_dir,
            signals_dir=signals_dir,
            results_dir=results_dir,
            max_train_year=max_train_year,
        )
        for asset in assets
    }

    print("\n" + "=" * 100)
    print("  SELECTED PARAMETERS")
    print("=" * 100)
    for asset in assets:
        r = all_results[asset.upper()]
        svm = r.get("svm") or {}
        print(
            f"{asset:<5} | SMA={r['sma']['sma_window']}, EMA={r['sma']['ema_window']} "
            f"| Z: W={r['zscore']['window']}, Th={r['zscore']['threshold']:.1f} "
            f"| RF: depth={r['random_forest']['max_depth']}, leaf={r['random_forest']['min_samples_leaf']}, "
            f"th={r['random_forest']['prob_threshold']:.2f} "
            f"| SVM: {svm.get('kernel', 'n/a')} C={svm.get('C', 'n/a')} "
            f"| Orchestrator: theta_long={r['orchestrator']['threshold_long']:.2f}"
        )
    print("=" * 100)
    return all_results


if __name__ == "__main__":
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as fh:
        run_all_assets(yaml.safe_load(fh))
