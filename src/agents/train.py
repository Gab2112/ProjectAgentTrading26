"""
train.py

Script di training e tuning dei parametri per tutti gli agenti quantitativi di ProjectAgentTrading26:
- Agenti Statistici:
    * SMAEMAAgent (Trend-Following)
    * ZScoreAgent (Breakout/Momentum & Mean-Reversion con Isteresi)
- Agente Machine Learning:
    * RandomForestAgent (Pattern non lineari, interazioni macro e feature price-lagged)
- Agente Macro:
    * MacroYieldCurveAgent (Inversione e pendenza della curva dei rendimenti, regimi macro)
- Agente Orchestratore:
    * OrchestratorAgent (Ensemble quantitativo con pesi sul simplesso, boost di consenso e penalizzazione conflitto)
- Training e tuning limitati al periodo definito in config.yaml.
- Warm-up precedente al training escluso dalle metriche di performance.

Funzionalità principali:
- Supporta l'universo multi-asset a 4 classi: SPY (Azionario), IEF (Obbligazionario 7-10y), GLD (Oro), SHY (Liquidità/Breve termine).
- Esegue l'ottimizzazione Walk-Forward rolling (5 anni di training In-Sample -> 1 anno di test Validation Out-Of-Sample).

- Funzione obiettivo di stabilità e penalizzazione del Max Drawdown ('sharpe_mdd').
- Salva tutti i file di configurazione JSON e i modelli serializzati (.joblib) in src/agents/config/:
    * sma_ema_params_{asset}.json
    * zscore_params_{asset}.json
    * random_forest_params_{asset}.json (e .joblib)
    * orchestrator_params_{asset}.json
- Esporta i log dettagliati di tutte le combinazioni testate in data/results/:
    * grid_search_combinations_sma_ema_{asset}.csv
    * summary_combinations_sma_ema_{asset}.csv
    * grid_search_combinations_zscore_{asset}.csv
    * summary_combinations_zscore_{asset}.csv
    * summary_combinations_orchestrator_{asset}.csv
- Salva i DataFrame dei segnali pronti per backtesting e allocation in data/signals/:
    * signals_statistical_{asset}.csv
    * signals_random_forest_{asset}.csv
    * signals_orchestrator_{asset}.csv
"""

import argparse
import sys
import numpy as np
import pandas as pd
import yaml
import json

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Risoluzione robusta dei percorsi
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.time_split import get_time_windows
from src.features import build_decision_feature_panel

from src.agents.EMA_EMW_agent import SMAEMAAgent, SMAEMAOptimizer, WalkForwardTrainer as SMAEMATrainer
from src.agents.ZScore_agent import ZScoreAgent, ZScoreOptimizer, ZScoreWalkForwardTrainer
from src.agents.macro_yield_curve_agent import MacroYieldCurveAgent
from src.agents.random_forest_agent import (
    RandomForestAgent,
    RandomForestOptimizer,
    RandomForestWalkForwardTrainer,
)
from src.agents.SVM_agent import SVMOptimizer, SVMWalkForwardTrainer
from src.agents.orchestrator_agent import (
    OrchestratorAgent,
    OrchestratorOptimizer,
)
from src.data_loader import load_agent_macro_data
from src.evaluator import calculate_performance_metrics, compute_objective_score
from src.utils import ASSETS


def save_training_provenance(
    config_path: Path,
    training_df: pd.DataFrame,
    cutoff_year: int,
) -> None:
    """Registra il periodo temporale utilizzato per il training."""

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


# =====================================================================
# PIPELINE DI TRAINING PER SINGOLO ASSET
# =====================================================================
def train_single_asset(
    asset: str,
    df: pd.DataFrame,
    config_dir: Path,
    signals_dir: Path,
    results_dir: Path,
    max_train_year: int,
    objective_metric: str = "sharpe_mdd",
    train_rf: bool = True,
    train_svm: bool = True,
    train_orchestrator: bool = True,
    orch_method: str = "ga",
    warmup_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Esegue il training Walk-Forward completo per un asset specifico ('SPY', 'IEF', 'GLD', 'SHY'):
    1. Agente SMA/EMA (Trend-Following) -> config JSON & log combinazioni.
    2. Agente Z-Score (Momentum/Mean-Reversion) -> config JSON & log combinazioni.
    3. Ensemble Statistico -> signals_statistical_{asset}.csv.
    4. Agente Random Forest (Machine Learning) -> config JSON, modello .joblib & signals_random_forest_{asset}.csv.
    5. Agente Macro Yield Curve -> segnali macro allineati.
    6. Agente Orchestratore (Ensemble Quantitativo) -> orchestrator_params_{asset}.json & signals_orchestrator_{asset}.csv.
    """
    # Verifica delle condizioni temporali del dataset
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        raise ValueError("Il training richiede un DatetimeIndex non vuoto.")

    if not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError("Le date di training devono essere ordinate e univoche.")

    if (df.index.year > max_train_year).any():
        raise ValueError("Il dataset contiene osservazioni successive al cutoff.")

    if warmup_df is not None and not warmup_df.empty:
        if warmup_df.index.max() >= df.index.min():
            raise ValueError("Il warm-up si sovrappone al training.")

    asset_upper = asset.upper()
    asset_lower = asset.lower()
    price_col = f"{asset_upper}_close"
    if price_col not in df.columns:
        raise KeyError(f"Colonna '{price_col}' non trovata nel dataset.")

    # Filtro colonne: mantiene solo l'asset target corrente e le feature macro/globali.
    # Esclude le colonne con prefisso degli altri 3 asset per eliminare zavorra e rumore.
    other_assets = [a for a in ASSETS if a != asset_upper]
    excluded_prefixes = tuple(f"{a}_" for a in other_assets)
    filtered_cols = [c for c in df.columns if not c.startswith(excluded_prefixes)]
    excluded_count = len(df.columns) - len(filtered_cols)
    df = df[filtered_cols].copy()
    if warmup_df is not None and not warmup_df.empty:
        warmup_df = warmup_df[filtered_cols].copy()

    print(f"[{asset_upper}] Colonne dataset filtrate: {len(filtered_cols)} (escluse {excluded_count} colonne degli altri asset)")

    print("\n" + "#" * 84)
    print(df.head())
    print(df.info())
    print(f"### AVVIO TRAINING & OTTIMIZZAZIONE COMPLETA: {asset_upper} ({price_col})")
    print("#" * 84)

    # -----------------------------------------------------------------
    # 1. OTTIMIZZAZIONE SMA/EMA (Trend-Following)
    # -----------------------------------------------------------------
    print(f"\n[{asset_upper}] >>> Walk-Forward: SMA/EMA Agent...")
    sma_opt = SMAEMAOptimizer(price_col=price_col, objective_metric=objective_metric)
    sma_trainer = SMAEMATrainer(optimizer=sma_opt, train_window_years=5, price_col=price_col)
    sma_res = sma_trainer.run(df, max_year=max_train_year, initial_warmup_df=warmup_df,)

    sma_config_file = config_dir / f"sma_ema_params_{asset_lower}.json"
    sma_trainer.save_final_agent_config(sma_config_file)

    sma_comb_log_file = results_dir / f"grid_search_combinations_sma_ema_{asset_lower}.csv"
    sma_trainer.save_combinations_log(sma_comb_log_file)

    sma_comb_sum_file = results_dir / f"summary_combinations_sma_ema_{asset_lower}.csv"
    sma_trainer.save_combinations_summary(sma_comb_sum_file)

    sma_sum_df = sma_trainer.get_combinations_summary()
    print(f"\n[{asset_upper}] Top 3 Combinazioni SMA/EMA per performance OOS aggregata:")
    print(
        sma_sum_df.head(3)[
            ["sma_window", "ema_window", "times_best", "selection_pct", "mean_oos_sharpe", "mean_oos_return_pct", "oos_win_rate_pct"]
        ].to_string(index=False)
    )

    # -----------------------------------------------------------------
    # 2. OTTIMIZZAZIONE Z-SCORE (Momentum / Mean-Reversion con Isteresi)
    # -----------------------------------------------------------------
    print(f"\n[{asset_upper}] >>> Walk-Forward: Z-Score Agent...")
    zs_opt = ZScoreOptimizer(price_col=price_col, exit_threshold=0.0, objective_metric=objective_metric)
    zs_trainer = ZScoreWalkForwardTrainer(optimizer=zs_opt, train_window_years=5, price_col=price_col)
    zs_res = zs_trainer.run(df, max_year=max_train_year, initial_warmup_df=warmup_df,)

    zs_config_file = config_dir / f"zscore_params_{asset_lower}.json"
    zs_trainer.save_final_agent_config(zs_config_file)

    zs_comb_log_file = results_dir / f"grid_search_combinations_zscore_{asset_lower}.csv"
    zs_trainer.save_combinations_log(zs_comb_log_file)

    zs_comb_sum_file = results_dir / f"summary_combinations_zscore_{asset_lower}.csv"
    zs_trainer.save_combinations_summary(zs_comb_sum_file)

    zs_sum_df = zs_trainer.get_combinations_summary()
    print(f"\n[{asset_upper}] Top 3 Combinazioni Z-Score per performance OOS aggregata:")
    print(
        zs_sum_df.head(3)[
            ["window", "threshold", "mode", "times_best", "selection_pct", "mean_oos_sharpe", "mean_oos_return_pct", "oos_win_rate_pct"]
        ].to_string(index=False)
    )

    # Storico completo per inizializzare gli indicatori
    signal_history = (
        pd.concat([warmup_df, df]).sort_index()
        if warmup_df is not None and not warmup_df.empty
        else df
    )
    # -----------------------------------------------------------------
    # 3. ISTANZIAZIONE E SALVATAGGIO SEGNALI DEI DUE AGENTI STATISTICI SEPARATI
    # -----------------------------------------------------------------
    sma_agent = SMAEMAAgent(price_col=price_col, target_asset=asset_upper)
    sma_agent.load_params(sma_config_file)
    sig_sma, conf_sma = sma_agent.predict_with_confidence(signal_history)
    sig_sma = sig_sma.reindex(df.index)
    conf_sma = conf_sma.reindex(df.index)
    df_sma = pd.DataFrame({"signal": sig_sma, "confidence": conf_sma}, index=df.index)
    sma_csv = signals_dir / f"signals_sma_ema_{asset_lower}.csv"
    df_sma.to_csv(sma_csv)
    print(f"\n[{asset_upper}] Segnali SMA/EMA salvati in: {sma_csv}")

    zs_agent = ZScoreAgent(price_col=price_col, target_asset=asset_upper)
    zs_agent.load_params(zs_config_file)
    sig_zs, conf_zs = zs_agent.predict_with_confidence(signal_history)
    sig_zs = sig_zs.reindex(df.index)
    conf_zs = conf_zs.reindex(df.index)
    df_zs = pd.DataFrame({"signal": sig_zs, "confidence": conf_zs}, index=df.index)
    zs_csv = signals_dir / f"signals_zscore_{asset_lower}.csv"
    df_zs.to_csv(zs_csv)
    print(f"[{asset_upper}] Segnali Z-Score salvati in: {zs_csv}")

    # -----------------------------------------------------------------
    # 4. AGENTE MACHINE LEARNING RANDOM FOREST
    # -----------------------------------------------------------------
    rf_res = None
    df_rf = None
    rf_config_file = config_dir / f"random_forest_params_{asset_lower}.json"
    rf_csv = signals_dir / f"signals_random_forest_{asset_lower}.csv"
    if train_rf:
        print(f"\n[{asset_upper}] >>> Walk-Forward: Random Forest Agent...")
        rf_opt = RandomForestOptimizer(
            target_asset=asset_upper,
            price_col=price_col,
            depth_candidates=[2, 3],
            min_leaf_candidates=[20, 30],
            threshold_candidates=[0.51, 0.52, 0.53],
            n_estimators_candidates=[100],
            objective_metric=objective_metric,
        )
        rf_trainer = RandomForestWalkForwardTrainer(
            optimizer=rf_opt,
            train_window_years=5,
            target_asset=asset_upper,
        )
        rf_res = rf_trainer.run(df, max_year=max_train_year)
        rf_trainer.save_final_agent_config(rf_config_file)
        save_training_provenance(rf_config_file, df, max_train_year)

        df_rf = rf_trainer.oof_signals.copy()
        if df_rf.empty or df_rf[["signal", "confidence"]].isna().any().any():
            raise ValueError("Predizioni OOF Random Forest assenti o incomplete.")
        df_rf.to_csv(rf_csv)
        print(f"[{asset_upper}] Segnali Random Forest salvati in: {rf_csv}")
    else:
        print(
            f"[{asset_upper}] Random Forest escluso: "
            "nessuna cache preesistente viene utilizzata "
            "senza verifica del protocollo temporale."
        )

    # -----------------------------------------------------------------
    # 5. AGENTE MACHINE LEARNING SVM (Support Vector Classifier)
    # -----------------------------------------------------------------
    svm_res = None
    df_svm = None
    svm_config_file = config_dir / f"svm_params_{asset_lower}.json"
    svm_csv = signals_dir / f"signals_svm_{asset_lower}.csv"
    if train_svm:
        print(f"\n[{asset_upper}] >>> Walk-Forward: SVM Agent (Support Vector Classifier)...")
        # Impostazioni in config.yaml (strategies.C.svm_agent): walk-forward
        # annuale con purge, iperparametri scelti con CV temporale interna.
        svm_opt = SVMOptimizer(
            target_asset=asset_upper,
            price_col=price_col,
            objective_metric=objective_metric,
        )
        svm_trainer = SVMWalkForwardTrainer(
            optimizer=svm_opt,
            train_window_years=5,
            target_asset=asset_upper,
        )
        svm_res = svm_trainer.run(df, max_year=max_train_year)
        svm_trainer.save_final_agent_config(svm_config_file)
        save_training_provenance(svm_config_file, df, max_train_year)
        df_svm = svm_trainer.oof_signals.copy()
        if df_svm.empty or df_svm[["signal", "confidence"]].isna().any().any():
            raise ValueError("Predizioni OOF SVM assenti o incomplete.")
        df_svm.to_csv(svm_csv)
        print(f"[{asset_upper}] Segnali SVM salvati in: {svm_csv}")
    else:
        print(
            f"[{asset_upper}] SVM escluso: "
            "nessuna cache preesistente viene utilizzata "
            "senza verifica del protocollo temporale."
        )

    # -----------------------------------------------------------------
    # 6. AGENTE MACRO YIELD CURVE
    # -----------------------------------------------------------------
    macro_agent = MacroYieldCurveAgent(target_asset=asset_upper)
    macro_multi = macro_agent.predict_multi_asset(df, assets=[asset_upper])
    m_sig = macro_multi[f"{asset_upper}_signal"]
    macro_score = macro_agent.calculate_macro_score(df)
    m_conf = (macro_score.abs() / 2.0).clip(0.0, 1.0).round(4)
    m_conf = pd.Series(np.where(m_sig != 0, np.maximum(0.50, m_conf), 0.0), index=df.index)
    df_macro = pd.DataFrame({"signal": m_sig, "confidence": m_conf}, index=df.index)
    macro_csv = signals_dir / f"signals_macro_{asset_lower}.csv"
    df_macro.to_csv(macro_csv)
    print(f"[{asset_upper}] Segnali Macro Yield Curve salvati in: {macro_csv}")

    oof_index = df.index
    for ml_signals in (df_rf, df_svm):
        if ml_signals is not None:
            if ml_signals.empty or not ml_signals.index.is_unique:
                raise ValueError("Indice delle predizioni OOF non valido.")
            oof_index = oof_index.intersection(ml_signals.index)

    if oof_index.empty:
        raise ValueError("Nessuna settimana OOF comune disponibile.")
    # -----------------------------------------------------------------
    # 7. AGENTE ORCHESTRATORE (Ensemble Multi-Agente)
    # -----------------------------------------------------------------
    orch_res = None
    df_orch = None
    if train_orchestrator:
        signals_pool = {
            "SMA_EMA_Agent": df_sma,
            "ZScore_Agent": df_zs,
            "Macro_YieldCurve_Agent": df_macro,
        }
        if df_rf is not None:
            signals_pool["RandomForest_Agent"] = df_rf
        if df_svm is not None:
            signals_pool["SVM_Agent"] = df_svm
        signals_pool = {
            name: signals.loc[oof_index].copy()
            for name, signals in signals_pool.items()
        }

        if any(s[["signal", "confidence"]].isna().any().any()
            for s in signals_pool.values()):
            raise ValueError("Il pool OOF contiene segnali mancanti.")

        print(f"\n[{asset_upper}] >>> Ottimizzazione Agente Orchestratore ({len(signals_pool)} Agenti: {', '.join(signals_pool.keys())})...")

        ret_asset = df[price_col].astype(float).pct_change().fillna(0.0).loc[oof_index]
        orch_opt = OrchestratorOptimizer(
            objective_metric=objective_metric,
            max_train_year=max_train_year,
            weight_step=None,  # Adattivo automatico
            threshold_range=(0.15, 0.50, 0.05),
        )
        best_orch_agent, df_summary_orch = orch_opt.optimize(
            signals_dict=signals_pool,
            returns_or_prices=ret_asset,
            asset_name=asset_upper,
            method=orch_method,
        )

        orch_config_file = config_dir / f"orchestrator_params_{asset_lower}.json"
        orch_opt.save_final_config(orch_config_file)

        orch_summary_file = results_dir / f"summary_combinations_orchestrator_{asset_lower}.csv"
        df_summary_orch.to_csv(orch_summary_file, index=False)

        df_orch = best_orch_agent.combine(signals_pool)
        orch_csv = signals_dir / f"signals_orchestrator_{asset_lower}.csv"
        df_orch.to_csv(orch_csv)
        print(f"[{asset_upper}] Segnali finali Orchestratore salvati in: {orch_csv}")
        orch_res = orch_opt.best_params

    # -----------------------------------------------------------------
    # 7. TABELLA COMPARATIVA PERFORMANCE IN-SAMPLE (<= max_train_year)
    # -----------------------------------------------------------------
    train_mask = df.index.isin(oof_index)
    ret_asset = df[price_col].astype(float).pct_change().fillna(0.0)

    pos_sma = sig_sma.shift(1).fillna(0.0)
    pos_zs = sig_zs.shift(1).fillna(0.0)
    pos_macro = df_macro["signal"].shift(1).fillna(0.0)

    strat_sma = (pos_sma * ret_asset)[train_mask]
    strat_zs = (pos_zs * ret_asset)[train_mask]
    strat_macro = (pos_macro * ret_asset)[train_mask]

    m_sma = calculate_performance_metrics(strat_sma)
    m_zs = calculate_performance_metrics(strat_zs)
    m_macro = calculate_performance_metrics(strat_macro)

    print("\n" + "-" * 88)
    print(
        f"  CONFRONTO FINALE IN-SAMPLE "
        f"({oof_index.min():%Y-%m-%d} - {oof_index.max():%Y-%m-%d}) "
        f"- ASSET: {asset_upper}"
    )
    print("-" * 88)
    print(f"{'Modello / Agente':<28} | {'Cum Ret':<9} | {'Sharpe':<7} | {'Max DD':<7} | {'Prof Fact':<9} | {'Score Obj':<9}")
    print("-" * 88)
    print(f"{'SMA/EMA Agent':<28} | {m_sma['cumulative_return']*100:+7.1f}% | {m_sma['sharpe_ratio']:>7.2f} | {m_sma['max_drawdown']*100:>5.1f}% | {m_sma['profit_factor']:>9.2f} | {compute_objective_score(m_sma, objective_metric):>9.2f}")
    print(f"{'Z-Score Agent':<28} | {m_zs['cumulative_return']*100:+7.1f}% | {m_zs['sharpe_ratio']:>7.2f} | {m_zs['max_drawdown']*100:>5.1f}% | {m_zs['profit_factor']:>9.2f} | {compute_objective_score(m_zs, objective_metric):>9.2f}")
    print(f"{'Macro Yield Curve Agent':<28} | {m_macro['cumulative_return']*100:+7.1f}% | {m_macro['sharpe_ratio']:>7.2f} | {m_macro['max_drawdown']*100:>5.1f}% | {m_macro['profit_factor']:>9.2f} | {compute_objective_score(m_macro, objective_metric):>9.2f}")

    if df_rf is not None:
        pos_rf = df_rf["signal"].shift(1).fillna(0.0)
        strat_rf = (pos_rf * ret_asset)[train_mask]
        m_rf = calculate_performance_metrics(strat_rf)
        print(f"{'Random Forest Agent':<28} | {m_rf['cumulative_return']*100:+7.1f}% | {m_rf['sharpe_ratio']:>7.2f} | {m_rf['max_drawdown']*100:>5.1f}% | {m_rf['profit_factor']:>9.2f} | {compute_objective_score(m_rf, objective_metric):>9.2f}")

    if df_svm is not None:
        pos_svm = df_svm["signal"].shift(1).fillna(0.0)
        strat_svm = (pos_svm * ret_asset)[train_mask]
        m_svm = calculate_performance_metrics(strat_svm)
        print(f"{'SVM Agent':<28} | {m_svm['cumulative_return']*100:+7.1f}% | {m_svm['sharpe_ratio']:>7.2f} | {m_svm['max_drawdown']*100:>5.1f}% | {m_svm['profit_factor']:>9.2f} | {compute_objective_score(m_svm, objective_metric):>9.2f}")

    if df_orch is not None:
        pos_orch = df_orch["signal"].shift(1).fillna(0.0)
        strat_orch = (pos_orch * ret_asset)[train_mask]
        m_orch = calculate_performance_metrics(strat_orch)
        print("-" * 88)
        print(f"{'ORCHESTRATORE (OTTIMIZZATO)':<28} | {m_orch['cumulative_return']*100:+7.1f}% | {m_orch['sharpe_ratio']:>7.2f} | {m_orch['max_drawdown']*100:>5.1f}% | {m_orch['profit_factor']:>9.2f} | {compute_objective_score(m_orch, objective_metric):>9.2f}  <-- OTTIMO")

    print("-" * 88)

    return {
        "asset": asset_upper,
        "sma": sma_res,
        "zscore": zs_res,
        "random_forest": rf_res,
        "svm": svm_res,
        "orchestrator": orch_res,
        "signals_sma_ema": df_sma,
        "signals_zscore": df_zs,
        "signals_random_forest": df_rf,
        "signals_svm": df_svm,
        "signals_orchestrator": df_orch,
    }


# =====================================================================
# MAIN PIPELINE MULTI-ASSET
# =====================================================================
def run_all_assets(
    assets: Optional[List[str]] = None,
    max_train_year: Optional[int] = None,
    objective_metric: str = "sharpe_mdd",
    train_rf: bool = True,
    train_svm: bool = True,
    train_orchestrator: bool = True,
    orch_method: str = "ga",
) -> Dict[str, Dict[str, Any]]:
    """
    Esegue il workflow completo di training su tutti gli asset specificati
    (default: universe a 4 asset SPY, IEF, GLD, SHY).
    """
    # Caricamento del protocollo temporale centralizzato
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    time_windows = get_time_windows(config)

    train_start = time_windows["train_start"]
    train_end = time_windows["train_end"]
    if max_train_year is not None and max_train_year != train_end.year:
        raise ValueError("Il cutoff di training deve coincidere con config.yaml")

    max_train_year = train_end.year
    if assets is None:
        assets = list(ASSETS)

    config_dir = CURRENT_DIR / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    signals_dir = PROJECT_ROOT / "data" / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)

    results_dir = PROJECT_ROOT / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Prezzi giornalieri del progetto
    market_daily = pd.read_csv(
        PROJECT_ROOT / "data/processed/market_daily.csv",
        index_col="Date",
        parse_dates=["Date"],
    ).loc[:train_end]

    # Calendario decisionale settimanale condiviso
    decision_dates = pd.read_csv(
        PROJECT_ROOT / "data/processed/weekly_decision_dates.csv",
        index_col="Date",
        parse_dates=["Date"],
    ).index

    decision_dates = decision_dates[decision_dates <= train_end]

    # Feature di mercato allineate alle date decisionali
    df = build_decision_feature_panel(market_daily, decision_dates)

    # Integrazione delle feature macroeconomiche
    macro_df = load_agent_macro_data(
        decision_dates=df.index,
        root_dir=PROJECT_ROOT,
    )
    df = df.join(macro_df, how="left")
    # Separazione tra warm-up storico e training ufficiale
    warmup_df = df.loc[df.index < train_start].copy()
    train_df = df.loc[
        (df.index >= train_start) & (df.index <= train_end)
    ].copy()

    if train_df.empty:
        raise ValueError("Nessun dato nel periodo di training configurato.")
    print(f"[Train] Dataset caricato: {len(df)} settimane dal {df.index[0].date()} al {df.index[-1].date()}")
    print(f"[Train] Universo Asset da ottimizzare: {assets}")
    print(f"[Train] Metrica obiettivo: '{objective_metric}'")
    print(f"[Train] Vincolo OOS Lock: training e tuning rigorosamente entro il {max_train_year}")
    print(f"[Train] Metodo Ottimizzazione Orchestratore: '{orch_method.upper()}'")

    all_results = {}

    for asset in assets:

        res = train_single_asset(
            asset=asset,
            df=train_df,
            warmup_df=warmup_df,
            config_dir=config_dir,
            signals_dir=signals_dir,
            results_dir=results_dir,
            max_train_year=max_train_year,
            objective_metric=objective_metric,
            train_rf=train_rf,
            train_svm=train_svm,
            train_orchestrator=train_orchestrator,
            orch_method=orch_method,
        )
        all_results[asset.upper()] = res

    # -----------------------------------------------------------------
    # RIEPILOGO FINALE GENERALE
    # -----------------------------------------------------------------
    print("\n" + "=" * 112)
    print("                              RIEPILOGO FINALE CONFIGURAZIONI & PARAMETRI")
    print("=" * 112)
    header = f"{'Asset':<6} | {'SMA/EMA Opt':<14} | {'ZScore Opt':<18} | {'Random Forest':<18} | {'SVM Opt':<20} | {'Orchestrator Best':<20}"
    print(header)
    print("-" * 112)

    for asset in assets:
        r = all_results[asset.upper()]
        sma_s = f"S={r['sma']['sma_window']}, E={r['sma']['ema_window']}"
        zs_s = f"W={r['zscore']['window']}, Th={r['zscore']['threshold']:.1f}"
        rf_s = (
            f"d={r['random_forest']['max_depth']}, l={r['random_forest']['min_samples_leaf']}, th={r['random_forest']['prob_threshold']:.2f}"
            if r.get("random_forest")
            else "N/A"
        )
        svm_res = r.get("svm")
        if svm_res:
            kernel = svm_res.get("kernel", "N/A")
            c_val = svm_res.get("C", "N/A")
            gamma = svm_res.get("gamma", "N/A")
            svm_s = f"{kernel} C={c_val}, g={gamma}"
        else:
            svm_s = "N/A"

        if r.get("orchestrator"):
            orch_s = f"Th={r['orchestrator']['threshold']:.2f}, Sh={r['orchestrator']['sharpe_ratio']:.2f}"
        else:
            orch_s = "N/A"
        print(f"{asset:<6} | {sma_s:<14} | {zs_s:<18} | {rf_s:<18} | {svm_s:<20} | {orch_s:<20}")

    print("=" * 112)
    print("\n[OK] Configurazione e salvataggio completati. Tutti i parametri sono pronti per la predizione:")
    print(f"  - Config JSON salvati in : {config_dir.resolve()}")
    print(f"  - Segnali CSV salvati in : {signals_dir.resolve()}")
    print(f"  - Log e summary salvati in: {results_dir.resolve()}\n")

    return all_results


# =====================================================================
# CLI ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script di training e tuning parametri per gli agenti quantitativi.")
    parser.add_argument(
        "target",
        nargs="?",
        default="ALL",
        help="Asset target ('ALL', 'SPY', 'IEF', 'GLD', 'SHY' o 'LEGACY_TREASURY'). Default: 'ALL'",
    )
    parser.add_argument(
        "metric",
        nargs="?",
        default="sharpe_mdd",
        help="Metrica obiettivo per l'ottimizzazione ('sharpe_mdd', 'sharpe', 'calmar', 'return'). Default: 'sharpe_mdd'",
    )
    parser.add_argument("--skip-rf", action="store_true", help="Salta l'addestramento dell'agente Random Forest")
    parser.add_argument("--skip-svm", action="store_true", help="Salta l'addestramento dell'agente SVM")
    parser.add_argument("--train-svm", action="store_true", default=None, help="Forza l'addestramento dell'agente SVM")
    parser.add_argument("--skip-orch", action="store_true", help="Salta l'ottimizzazione dell'Orchestratore")
    parser.add_argument(
        "--orch-method",
        choices=["ga", "grid"],
        default="ga",
        help="Metodo di ottimizzazione per l'Orchestratore: 'ga' (Algoritmo Genetico) o 'grid'. Default: 'ga'",
    )

    args = parser.parse_args()

    if args.target.upper() == "ALL":
        target_assets = list(ASSETS)
    elif args.target.upper() == "LEGACY_TREASURY":
        target_assets = ["SHY", "IEF", "TLT"]
    else:
        target_assets = [args.target.upper()]

    train_svm_flag = True
    if args.skip_svm:
        train_svm_flag = False
    elif args.train_svm is not None:
        train_svm_flag = args.train_svm

    run_all_assets(
        assets=target_assets,
        objective_metric=args.metric,
        train_rf=not args.skip_rf,
        train_svm=train_svm_flag,
        train_orchestrator=not args.skip_orch,
        orch_method=args.orch_method,
    )
