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

Funzionalità principali:
- Supporta l'universo multi-asset a 4 classi: SPY (Azionario), IEF (Obbligazionario 7-10y), GLD (Oro), SHY (Liquidità/Breve termine).
- Esegue l'ottimizzazione Walk-Forward rolling (5 anni di training In-Sample -> 1 anno di test Validation Out-Of-Sample).
- Rispetta rigorosamente il vincolo OOS lock del progetto: training e selezione limitati al periodo <= 2021 (max_train_year = 2021).
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
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Risoluzione robusta dei percorsi
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.agents.base_agent import (
    BaseAgent,
    DEFAULT_ASSETS,
    calculate_performance_metrics,
    compute_objective_score,
)
from src.agents.EMA_EMW_agent import SMAEMAAgent, SMAEMAOptimizer, WalkForwardTrainer as SMAEMATrainer
from src.agents.ZScore_agent import ZScoreAgent, ZScoreOptimizer, ZScoreWalkForwardTrainer
from src.agents.macro_yield_curve_agent import MacroYieldCurveAgent
from src.agents.random_forest_agent import (
    RandomForestAgent,
    RandomForestOptimizer,
    RandomForestWalkForwardTrainer,
)
from src.agents.orchestrator_agent import (
    OrchestratorAgent,
    OrchestratorOptimizer,
)


# =====================================================================
# COMBINATORE STATISTICO: ENSEMBLE DI SEGNALI E CONFIDENZA
# =====================================================================
def combine_statistical_signals(
    signals_sma: pd.Series,
    conf_sma: pd.Series,
    signals_zs: pd.Series,
    conf_zs: pd.Series,
) -> pd.DataFrame:
    """
    Combina i segnali e le confidenze dei due agenti statistici (SMA/EMA e Z-Score)
    in un unico DataFrame con esattamente 2 colonne: 'signal' e 'confidence'.

    Regole dell'Ensemble Statistico:
    1. Accordo pieno (stesso segno attivo):
       - Segnale confermato da entrambi gli agenti.
       - Confidenza combinata con boost di sinergia (+15%), max 1.0.
    2. Segnale singolo attivo (uno attivo, l'altro flat a 0):
       - Segnale dell'agente attivo mantenuto.
       - Confidenza scalata all'85% (assenza di seconda conferma).
    3. Disaccordo diretto (+1 vs -1):
       - Se la differenza di confidenza e >= 0.20, prevale l'agente piu convinto
         con confidenza netta pari al delta (|c1 - c2|).
       - Se la differenza e < 0.20, situazione di stallo: segnale 0 (FLAT), confidenza 0.0.
    4. Entrambi flat (0, 0):
       - Segnale 0 (FLAT), confidenza 0.0.
    """
    n = len(signals_sma)
    out_sig = np.zeros(n, dtype=int)
    out_conf = np.zeros(n, dtype=float)

    s1 = signals_sma.to_numpy()
    c1 = conf_sma.to_numpy()
    s2 = signals_zs.to_numpy()
    c2 = conf_zs.to_numpy()

    for i in range(n):
        sig1, conf1 = s1[i], c1[i]
        sig2, conf2 = s2[i], c2[i]

        if sig1 == sig2 and sig1 != 0:
            out_sig[i] = sig1
            out_conf[i] = min(1.0, round(float((conf1 + conf2) / 2.0 * 1.15), 4))
        elif sig1 != 0 and sig2 == 0:
            out_sig[i] = sig1
            out_conf[i] = round(float(conf1 * 0.85), 4)
        elif sig1 == 0 and sig2 != 0:
            out_sig[i] = sig2
            out_conf[i] = round(float(conf2 * 0.85), 4)
        elif sig1 != 0 and sig2 != 0 and sig1 != sig2:
            diff = abs(conf1 - conf2)
            if diff >= 0.20:
                out_sig[i] = sig1 if conf1 > conf2 else sig2
                out_conf[i] = round(float(diff), 4)
            else:
                out_sig[i] = 0
                out_conf[i] = 0.0
        else:
            out_sig[i] = 0
            out_conf[i] = 0.0

    df_out = pd.DataFrame(
        {
            "signal": out_sig,
            "confidence": out_conf,
        },
        index=signals_sma.index,
    )
    return df_out


# =====================================================================
# PIPELINE DI TRAINING PER SINGOLO ASSET
# =====================================================================
def train_single_asset(
    asset: str,
    df: pd.DataFrame,
    config_dir: Path,
    signals_dir: Path,
    results_dir: Path,
    max_train_year: int = 2021,
    objective_metric: str = "sharpe_mdd",
    train_rf: bool = True,
    train_orchestrator: bool = True,
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
    asset_upper = asset.upper()
    asset_lower = asset.lower()
    price_col = f"{asset_upper}_close"
    if price_col not in df.columns:
        raise KeyError(f"Colonna '{price_col}' non trovata nel dataset.")

    print("\n" + "#" * 84)
    print(f"### AVVIO TRAINING & OTTIMIZZAZIONE COMPLETA: {asset_upper} ({price_col})")
    print("#" * 84)

    # -----------------------------------------------------------------
    # 1. OTTIMIZZAZIONE SMA/EMA (Trend-Following)
    # -----------------------------------------------------------------
    print(f"\n[{asset_upper}] >>> Walk-Forward: SMA/EMA Agent...")
    sma_opt = SMAEMAOptimizer(price_col=price_col, objective_metric=objective_metric)
    sma_trainer = SMAEMATrainer(optimizer=sma_opt, train_window_years=5, price_col=price_col)
    sma_res = sma_trainer.run(df, max_year=max_train_year)

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
    zs_res = zs_trainer.run(df, max_year=max_train_year)

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

    # -----------------------------------------------------------------
    # 3. ISTANZIAZIONE ED ENSEMBLE STATISTICO
    # -----------------------------------------------------------------
    sma_agent = SMAEMAAgent(price_col=price_col, target_asset=asset_upper)
    sma_agent.load_params(sma_config_file)
    sig_sma, conf_sma = sma_agent.predict_with_confidence(df)

    zs_agent = ZScoreAgent(price_col=price_col, target_asset=asset_upper)
    zs_agent.load_params(zs_config_file)
    sig_zs, conf_zs = zs_agent.predict_with_confidence(df)

    df_stat = combine_statistical_signals(
        signals_sma=sig_sma,
        conf_sma=conf_sma,
        signals_zs=sig_zs,
        conf_zs=conf_zs,
    )
    stat_csv = signals_dir / f"signals_statistical_{asset_lower}.csv"
    df_stat.to_csv(stat_csv)
    print(f"\n[{asset_upper}] Segnali statistici salvati in: {stat_csv}")

    # -----------------------------------------------------------------
    # 4. AGENTE MACHINE LEARNING RANDOM FOREST
    # -----------------------------------------------------------------
    rf_res = None
    df_rf = None
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

        rf_config_file = config_dir / f"random_forest_params_{asset_lower}.json"
        rf_trainer.save_final_agent_config(rf_config_file)

        sig_rf, conf_rf = rf_trainer.final_agent.predict_with_confidence(df)
        df_rf = pd.DataFrame({"signal": sig_rf, "confidence": conf_rf}, index=df.index)
        rf_csv = signals_dir / f"signals_random_forest_{asset_lower}.csv"
        df_rf.to_csv(rf_csv)
        print(f"[{asset_upper}] Segnali Random Forest salvati in: {rf_csv}")

    # -----------------------------------------------------------------
    # 5. AGENTE MACRO YIELD CURVE
    # -----------------------------------------------------------------
    macro_agent = MacroYieldCurveAgent(target_asset=asset_upper)
    macro_multi = macro_agent.predict_multi_asset(df, assets=[asset_upper])
    m_sig = macro_multi[f"{asset_upper}_signal"]
    macro_score = macro_agent.calculate_macro_score(df)
    m_conf = (macro_score.abs() / 2.0).clip(0.0, 1.0).round(4)
    m_conf = pd.Series(np.where(m_sig != 0, np.maximum(0.50, m_conf), 0.0), index=df.index)
    df_macro = pd.DataFrame({"signal": m_sig, "confidence": m_conf}, index=df.index)

    # -----------------------------------------------------------------
    # 6. AGENTE ORCHESTRATORE (Ensemble Ottimizzato)
    # -----------------------------------------------------------------
    orch_res = None
    df_orch = None
    if train_orchestrator:
        print(f"\n[{asset_upper}] >>> Ottimizzazione Agente Orchestratore...")
        signals_pool = {
            "Statistical_Agent": df_stat,
            "Macro_YieldCurve_Agent": df_macro,
        }
        if df_rf is not None:
            signals_pool["RandomForest_Agent"] = df_rf

        ret_asset = df[price_col].astype(float).pct_change().fillna(0.0)

        orch_opt = OrchestratorOptimizer(
            objective_metric=objective_metric,
            max_train_year=max_train_year,
            weight_step=0.05,
            threshold_range=(0.15, 0.50, 0.05),
        )
        best_orch_agent, df_summary_orch = orch_opt.optimize(
            signals_dict=signals_pool,
            returns_or_prices=ret_asset,
            asset_name=asset_upper,
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
    train_mask = df.index.year <= max_train_year
    ret_asset = df[price_col].astype(float).pct_change().fillna(0.0)

    pos_sma = sig_sma.shift(1).fillna(0.0)
    pos_zs = sig_zs.shift(1).fillna(0.0)
    pos_stat = df_stat["signal"].shift(1).fillna(0.0)
    pos_macro = df_macro["signal"].shift(1).fillna(0.0)

    strat_sma = (pos_sma * ret_asset)[train_mask]
    strat_zs = (pos_zs * ret_asset)[train_mask]
    strat_stat = (pos_stat * ret_asset)[train_mask]
    strat_macro = (pos_macro * ret_asset)[train_mask]

    m_sma = calculate_performance_metrics(strat_sma)
    m_zs = calculate_performance_metrics(strat_zs)
    m_stat = calculate_performance_metrics(strat_stat)
    m_macro = calculate_performance_metrics(strat_macro)

    print("\n" + "-" * 88)
    print(f"  CONFRONTO FINALE IN-SAMPLE (2004-{max_train_year}) - ASSET: {asset_upper}")
    print("-" * 88)
    print(f"{'Modello / Agente':<28} | {'Cum Ret':<9} | {'Sharpe':<7} | {'Max DD':<7} | {'Prof Fact':<9} | {'Score Obj':<9}")
    print("-" * 88)
    print(f"{'SMA/EMA Agent':<28} | {m_sma['cumulative_return']*100:+7.1f}% | {m_sma['sharpe_ratio']:>7.2f} | {m_sma['max_drawdown']*100:>5.1f}% | {m_sma['profit_factor']:>9.2f} | {compute_objective_score(m_sma, objective_metric):>9.2f}")
    print(f"{'Z-Score Agent':<28} | {m_zs['cumulative_return']*100:+7.1f}% | {m_zs['sharpe_ratio']:>7.2f} | {m_zs['max_drawdown']*100:>5.1f}% | {m_zs['profit_factor']:>9.2f} | {compute_objective_score(m_zs, objective_metric):>9.2f}")
    print(f"{'Ensemble Statistico':<28} | {m_stat['cumulative_return']*100:+7.1f}% | {m_stat['sharpe_ratio']:>7.2f} | {m_stat['max_drawdown']*100:>5.1f}% | {m_stat['profit_factor']:>9.2f} | {compute_objective_score(m_stat, objective_metric):>9.2f}")
    print(f"{'Macro Yield Curve Agent':<28} | {m_macro['cumulative_return']*100:+7.1f}% | {m_macro['sharpe_ratio']:>7.2f} | {m_macro['max_drawdown']*100:>5.1f}% | {m_macro['profit_factor']:>9.2f} | {compute_objective_score(m_macro, objective_metric):>9.2f}")

    if df_rf is not None:
        pos_rf = df_rf["signal"].shift(1).fillna(0.0)
        strat_rf = (pos_rf * ret_asset)[train_mask]
        m_rf = calculate_performance_metrics(strat_rf)
        print(f"{'Random Forest Agent':<28} | {m_rf['cumulative_return']*100:+7.1f}% | {m_rf['sharpe_ratio']:>7.2f} | {m_rf['max_drawdown']*100:>5.1f}% | {m_rf['profit_factor']:>9.2f} | {compute_objective_score(m_rf, objective_metric):>9.2f}")

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
        "orchestrator": orch_res,
        "signals_statistical": df_stat,
        "signals_random_forest": df_rf,
        "signals_orchestrator": df_orch,
    }


# =====================================================================
# MAIN PIPELINE MULTI-ASSET
# =====================================================================
def run_all_assets(
    assets: Optional[List[str]] = None,
    max_train_year: int = 2021,
    objective_metric: str = "sharpe_mdd",
    train_rf: bool = True,
    train_orchestrator: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Esegue il workflow completo di training su tutti gli asset specificati
    (default: universe a 4 asset SPY, IEF, GLD, SHY).
    """
    if assets is None:
        assets = list(DEFAULT_ASSETS)

    config_dir = CURRENT_DIR / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    signals_dir = PROJECT_ROOT / "data" / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)

    results_dir = PROJECT_ROOT / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Caricamento centralizzato dataset con feature macro e prezzi di tutti gli asset
    df = BaseAgent.load_complete_dataset(root_dir=PROJECT_ROOT, save_features_weekly_csv=True)
    print(f"[Train] Dataset caricato: {len(df)} settimane dal {df.index[0].date()} al {df.index[-1].date()}")
    print(f"[Train] Universo Asset da ottimizzare: {assets}")
    print(f"[Train] Metrica obiettivo: '{objective_metric}'")
    print(f"[Train] Vincolo OOS Lock: training e tuning rigorosamente entro il {max_train_year}")

    all_results = {}

    for asset in assets:
        res = train_single_asset(
            asset=asset,
            df=df,
            config_dir=config_dir,
            signals_dir=signals_dir,
            results_dir=results_dir,
            max_train_year=max_train_year,
            objective_metric=objective_metric,
            train_rf=train_rf,
            train_orchestrator=train_orchestrator,
        )
        all_results[asset.upper()] = res

    # -----------------------------------------------------------------
    # RIEPILOGO FINALE GENERALE
    # -----------------------------------------------------------------
    print("\n" + "=" * 90)
    print("                    RIEPILOGO FINALE CONFIGURAZIONI & PARAMETRI")
    print("=" * 90)
    header = f"{'Asset':<6} | {'SMA/EMA Opt':<14} | {'ZScore Opt':<18} | {'Random Forest':<18} | {'Orchestrator Best':<20}"
    print(header)
    print("-" * 90)

    for asset in assets:
        r = all_results[asset.upper()]
        sma_s = f"S={r['sma']['sma_window']}, E={r['sma']['ema_window']}"
        zs_s = f"W={r['zscore']['window']}, Th={r['zscore']['threshold']:.1f}"
        rf_s = (
            f"d={r['random_forest']['max_depth']}, l={r['random_forest']['min_samples_leaf']}, th={r['random_forest']['prob_threshold']:.2f}"
            if r.get("random_forest")
            else "N/A"
        )
        if r.get("orchestrator"):
            orch_s = f"Th={r['orchestrator']['threshold']:.2f}, Sh={r['orchestrator']['sharpe_ratio']:.2f}"
        else:
            orch_s = "N/A"
        print(f"{asset:<6} | {sma_s:<14} | {zs_s:<18} | {rf_s:<18} | {orch_s:<20}")

    print("=" * 90)
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
    parser.add_argument("--skip-orch", action="store_true", help="Salta l'ottimizzazione dell'Orchestratore")

    args = parser.parse_args()

    if args.target.upper() == "ALL":
        target_assets = list(DEFAULT_ASSETS)
    elif args.target.upper() == "LEGACY_TREASURY":
        target_assets = ["SHY", "IEF", "TLT"]
    else:
        target_assets = [args.target.upper()]

    run_all_assets(
        assets=target_assets,
        max_train_year=2021,
        objective_metric=args.metric,
        train_rf=not args.skip_rf,
        train_orchestrator=not args.skip_orch,
    )
