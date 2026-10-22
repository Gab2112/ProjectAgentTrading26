"""
main.py

Entry point principale del progetto ProjectAgentTrading26.

Modalita supportate:
1. 'oos' / '--oos' (default):
   Esegue la previsione Out-Of-Sample (2022-2026) degli agenti multi-agente
   (SMA/EMA, Z-Score, Macro Yield Curve, FinBERT Sentiment e Orchestratore),
   caricando i parametri ottimali addestrati salvati in src/agents/config/.
   Include un audit preventivo della disponibilita di notizie e segnali FinBERT nel periodo 2022-2026.
   Salva i segnali OOS in data/signals/oos_signals_orchestrator_{asset}.csv e il riepilogo
   performance in data/results/oos_performance_summary.csv.

2. 'pipeline' / '--pipeline':
   Esegue la pipeline deterministica end-to-end baseline (fasi 0-7: data transformation,
   feature engineering, regime model, signal generation, portfolio construction, backtest, evaluation).

3. 'check_finbert' / '--check-finbert':
   Esegue esclusivamente il controllo e la diagnostica sul dataset notizie e segnali FinBERT 2022-2026.
"""

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Risoluzione robusta dei percorsi
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

CONFIG_PATH = ROOT_DIR / "config.yaml"

# Moduli degli agenti multi-agente
from src.agents.base_agent import calculate_performance_metrics
from src.agents.EMA_EMW_agent import SMAEMAAgent
from src.agents.ZScore_agent import ZScoreAgent
from src.agents.macro_yield_curve_agent import MacroYieldCurveAgent
from src.agents.orchestrator_agent import OrchestratorAgent


# =============================================================================
# Configurazione
# =============================================================================

def load_config(path: Path) -> dict:
    """Carica il file centrale di configurazione YAML."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a dictionary.")

    return config


def validate_config(config: dict) -> None:
    """Controlla che le sezioni minime del config siano presenti."""
    required_sections = [
        "project",
        "research",
        "universe",
        "frequency",
        "data",
        "validation",
        "execution",
        "agents",
        "paths",
    ]

    missing = [section for section in required_sections if section not in config]
    if missing:
        raise KeyError("Missing required config sections: " + ", ".join(missing))


# =============================================================================
# Ambiente
# =============================================================================

def initialize_environment(config: dict) -> None:
    """Imposta i seed casuali e crea le cartelle configurate."""
    seed = int(config["project"]["random_seed"])
    random.seed(seed)
    np.random.seed(seed)

    for relative_path in config["paths"].values():
        directory = ROOT_DIR / relative_path
        directory.mkdir(parents=True, exist_ok=True)


def print_project_summary(config: dict) -> None:
    """Stampa una sintesi della configurazione attiva."""
    assets = config["universe"]["traded_assets"]
    pipeline_config = config.get("pipeline", {})
    run_download = bool(pipeline_config.get("run_data_acquisition", False))
    evaluate_oos = bool(config.get("evaluation", {}).get("evaluate_oos", False))

    print("\n======================================================")
    print("            PROJECTAGENTTRADING26")
    print("======================================================")
    print(f"Version:            {config['project']['version']}")
    print(f"Research candidate: {config['research']['working_candidate']}")
    print(f"Market scope:       {config['research']['market_scope']}")
    print(f"Strategy frequency: {config['frequency']['strategy']}")
    print(f"Agents enabled:     {config['agents']['enabled']}")
    print(f"Download raw data:  {run_download}")
    print(f"OOS evaluation:     {evaluate_oos}")
    print("------------------------------------------------------")
    print("Traded universe:")
    for bucket, asset in assets.items():
        print(f"  {bucket:<24} {asset['ticker']:<6} ({asset['role']})")
    print("======================================================\n")


# =============================================================================
# 1. CONTROLLO E DIAGNOSTICA DATAFRAME FINBERT / NOTIZIE OOS (2022-2026)
# =============================================================================

def check_finbert_oos_news(
    root_dir: Optional[Path] = None,
    oos_start_year: int = 2022,
) -> Dict[str, Any]:
    """
    Verifica se il dataset delle notizie finanziarie e i segnali FinBERT contengono
    copertura temporale e segnali operativi per il periodo Out-Of-Sample (2022-2026).

    Controlli eseguiti:
    1. File grezzo notizie: data/raw/news/financial_news.csv
       - Esistenza del file
       - Range temporale complessivo
       - Conteggio notizie nel periodo >= oos_start_year (2022-2026)
       - Ripartizione delle notizie per anno
    2. File segnali FinBERT Sentiment per i 3 asset:
       - signals_sentiment_shy.csv
       - signals_sentiment_ief.csv
       - signals_sentiment_tlt.csv
       - Conteggio osservazioni OOS e numero di segnali attivi (signal != 0)
    """
    if root_dir is None:
        root_dir = ROOT_DIR

    news_path = root_dir / "data" / "raw" / "news" / "financial_news.csv"
    signals_dir = root_dir / "data" / "signals"

    print("\n" + "=" * 80)
    print(f"  AUDIT E VERIFICA DATI FINBERT / NOTIZIE PER IL PERIODO OOS ({oos_start_year}-2026)")
    print("=" * 80)

    report = {
        "news_file_exists": news_path.exists(),
        "total_news_count": 0,
        "oos_news_count": 0,
        "news_by_year": {},
        "sentiment_signals": {},
        "is_ready_for_oos": False,
    }

    # 1. Controllo file notizie grezze
    if news_path.exists():
        df_news = pd.read_csv(news_path)
        report["total_news_count"] = len(df_news)

        date_col = [c for c in df_news.columns if "date" in c.lower() or "time" in c.lower()]
        if date_col:
            dcol = date_col[0]
            df_news[dcol] = pd.to_datetime(df_news[dcol])
            min_date = df_news[dcol].min().strftime("%Y-%m-%d")
            max_date = df_news[dcol].max().strftime("%Y-%m-%d")

            news_oos = df_news[df_news[dcol].dt.year >= oos_start_year]
            report["oos_news_count"] = len(news_oos)
            year_counts = dict(df_news[dcol].dt.year.value_counts().sort_index())
            report["news_by_year"] = year_counts

            print(f"  [Notizie Grezze] File trovato: {news_path.relative_to(root_dir)}")
            print(f"    - Totale notizie nel database : {len(df_news)}")
            print(f"    - Finestra temporale notizie   : dal {min_date} al {max_date}")
            print(f"    - Notizie periodo OOS ({oos_start_year}+) : {len(news_oos)} su {len(df_news)} "
                  f"({len(news_oos)/len(df_news)*100:.1f}%)")
            print("    - Distribuzione notizie OOS per anno:")
            for yr, count in sorted(year_counts.items()):
                if yr >= oos_start_year:
                    print(f"        * Anno {yr}: {count} notizie")
        else:
            print(f"  [WARN] Nessuna colonna data rilevata in {news_path}")
    else:
        print(f"  [ERRORE] File notizie non trovato in: {news_path}")

    # 2. Controllo file segnali FinBERT Sentiment per SHY, IEF, TLT
    print(f"\n  [Segnali FinBERT] Verifica serie temporali in: {signals_dir.relative_to(root_dir)}")
    all_signals_ok = True

    for ticker in ["SHY", "IEF", "TLT"]:
        sig_file = signals_dir / f"signals_sentiment_{ticker.lower()}.csv"
        if sig_file.exists():
            df_sig = pd.read_csv(sig_file, parse_dates=["Date"], index_col="Date")
            oos_sig = df_sig[df_sig.index.year >= oos_start_year]
            n_oos = len(oos_sig)
            active_counts = dict(oos_sig["signal"].value_counts())
            n_active = int((oos_sig["signal"] != 0).sum())

            report["sentiment_signals"][ticker] = {
                "file_exists": True,
                "oos_periods": n_oos,
                "active_signals": n_active,
                "distribution": active_counts,
            }

            print(f"    - {ticker}: {n_oos} settimane OOS ({oos_sig.index.min().strftime('%Y-%m-%d')} -> "
                  f"{oos_sig.index.max().strftime('%Y-%m-%d')}) | Segnali attivi: {n_active} {active_counts}")
        else:
            all_signals_ok = False
            report["sentiment_signals"][ticker] = {"file_exists": False}
            print(f"    - [WARN] File segnali sentiment per {ticker} non trovato: {sig_file.name}")

    report["is_ready_for_oos"] = report["oos_news_count"] > 0 and all_signals_ok

    print("\n  [ESITO CONTROLLO FINBERT]")
    if report["is_ready_for_oos"]:
        print("  -> OK! Il dataset FinBERT dispone di notizie e segnali attivi per l'OOS 2022-2026.")
        print("     L'agente FinBERT puo essere impiegato a pieno regime nelle previsioni OOS.")
    else:
        print("  -> ATTENZIONE: Mancano alcuni file o notizie per il periodo OOS.")
    print("=" * 80 + "\n")

    return report


# =============================================================================
# 2. PREVISIONE OUT-OF-SAMPLE (2022-2026) CON PARAMETRI ADDESTRATI
# =============================================================================

def run_oos_prediction(
    config: Optional[dict] = None,
    root_dir: Optional[Path] = None,
    assets: Optional[List[str]] = None,
    oos_start: str = "2022-01-01",
) -> Dict[str, pd.DataFrame]:
    """
    Esegue la sola previsione sul set di test Out-Of-Sample (2022-2026):
    1. Esegue il controllo preventivo sul dataset notizie e segnali FinBERT.
    2. Carica i file di configurazione con i parametri ottimali addestrati da src/agents/config/:
       - sma_ema_params_{asset}.json
       - zscore_params_{asset}.json
       - macro_yield_params_{asset}.json
       - orchestrator_params_{asset}.json
    3. Calcola l'inferenza completa (permettendo il naturale warm-up degli indicatori rolling)
       e isola rigorosamente la porzione Out-Of-Sample (dal oos_start in poi).
    4. Genera e salva per ciascun asset:
       - data/signals/oos_signals_orchestrator_{asset}.csv
       - data/signals/oos_signals_sma_ema_{asset}.csv
       - data/signals/oos_signals_zscore_{asset}.csv
       - data/signals/oos_signals_macro_{asset}.csv
    5. Valuta e confronta le performance della strategia multi-agente rispetto
       al benchmark Buy & Hold sul medesimo orizzonte OOS (2022-2026).
    6. Salva un riepilogo comparativo in data/results/oos_performance_summary.csv.
    """
    if root_dir is None:
        root_dir = ROOT_DIR

    if config is None:
        if CONFIG_PATH.exists():
            config = load_config(CONFIG_PATH)
        else:
            config = {}

    if assets is None:
        if "universe" in config and "traded_assets" in config["universe"]:
            assets = [item["ticker"] for item in config["universe"]["traded_assets"].values()]
        else:
            assets = ["SHY", "IEF", "TLT"]

    # Data di inizio OOS da config se presente
    if "validation" in config and "oos_start" in config["validation"]:
        oos_start = str(config["validation"]["oos_start"])

    # 1. Controllo preventivo notizie e segnali FinBERT
    check_finbert_oos_news(root_dir=root_dir, oos_start_year=pd.to_datetime(oos_start).year)

    config_dir = root_dir / "src" / "agents" / "config"
    signals_dir = root_dir / "data" / "signals"
    results_dir = root_dir / "data" / "results"
    processed_dir = root_dir / "data" / "processed"

    signals_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    features_file = processed_dir / "features_weekly.csv"
    if not features_file.exists():
        features_file = processed_dir / "master_weekly.csv"
    if not features_file.exists():
        raise FileNotFoundError(
            f"Dataset delle feature non trovato in: {features_file}. "
            "Esegui prima la preparazione dati o la pipeline."
        )

    df_features = pd.read_csv(features_file, parse_dates=["Date"], index_col="Date")
    oos_dt = pd.to_datetime(oos_start)
    oos_mask = df_features.index >= oos_dt

    if not np.any(oos_mask):
        raise ValueError(f"Nessuna data presente nel dataset successiva al {oos_start}.")

    oos_dates = df_features.index[oos_mask]
    print("=" * 88)
    print("  AVVIO PREVISIONE OUT-OF-SAMPLE (OOS) MULTI-AGENTE")
    print(f"  Orizzonte OOS: dal {oos_dates[0].strftime('%Y-%m-%d')} al {oos_dates[-1].strftime('%Y-%m-%d')} "
          f"({len(oos_dates)} settimane)")
    print(f"  Asset in analisi: {assets}")
    print(f"  Directory parametri addestrati: {config_dir.relative_to(root_dir)}")
    print("=" * 88)

    oos_results_summary: List[Dict[str, Any]] = []
    oos_output_dfs: Dict[str, pd.DataFrame] = {}

    for asset in assets:
        asset_lower = asset.lower()
        asset_upper = asset.upper()
        price_col = f"{asset_upper}_close"

        if price_col not in df_features.columns:
            print(f"  [ERRORE] Colonna '{price_col}' non trovata per {asset_upper}. Skip.")
            continue

        print(f"\n[{asset_upper}] >>> Generazione Previsione OOS con Parametri Addestrati...")

        # -------------------------------------------------------------
        # A. AGENTE 1: SMA/EMA Trend-Following
        # -------------------------------------------------------------
        sma_cfg_file = config_dir / f"sma_ema_params_{asset_lower}.json"
        if not sma_cfg_file.exists():
            raise FileNotFoundError(f"Parametri SMA/EMA non trovati per {asset_upper}: {sma_cfg_file}")

        sma_agent = SMAEMAAgent(price_col=price_col)
        sma_agent.load_params(sma_cfg_file)
        sig_sma_full, conf_sma_full = sma_agent.predict_with_confidence(df_features)

        # -------------------------------------------------------------
        # B. AGENTE 2: Z-Score Mean-Reversion / Momentum
        # -------------------------------------------------------------
        zs_cfg_file = config_dir / f"zscore_params_{asset_lower}.json"
        if not zs_cfg_file.exists():
            raise FileNotFoundError(f"Parametri Z-Score non trovati per {asset_upper}: {zs_cfg_file}")

        zs_agent = ZScoreAgent(price_col=price_col)
        zs_agent.load_params(zs_cfg_file)
        sig_zs_full, conf_zs_full = zs_agent.predict_with_confidence(df_features)

        # -------------------------------------------------------------
        # C. AGENTE 3: Macro Yield Curve & Inflazione
        # -------------------------------------------------------------
        macro_cfg_file = config_dir / f"macro_yield_params_{asset_lower}.json"
        if not macro_cfg_file.exists():
            raise FileNotFoundError(f"Parametri Macro non trovati per {asset_upper}: {macro_cfg_file}")

        macro_agent = MacroYieldCurveAgent()
        macro_agent.load_params(macro_cfg_file)
        macro_multi = macro_agent.predict_multi_asset(df_features)
        macro_score = macro_agent.calculate_macro_score(df_features)

        sig_col = f"{asset_upper}_signal"
        m_sig = macro_multi[sig_col] if sig_col in macro_multi.columns else pd.Series(0, index=df_features.index)
        m_conf_raw = (macro_score.abs() / 2.0).clip(0.0, 1.0).round(4)
        m_conf_full = pd.Series(
            np.where(m_sig != 0, np.maximum(0.50, m_conf_raw), 0.0),
            index=df_features.index,
        )

        # -------------------------------------------------------------
        # D. AGENTE 4: FinBERT Sentiment
        # -------------------------------------------------------------
        sent_csv = signals_dir / f"signals_sentiment_{asset_lower}.csv"
        if not sent_csv.exists():
            raise FileNotFoundError(f"File segnali sentiment FinBERT non trovato: {sent_csv}")

        df_sent_full = pd.read_csv(sent_csv, parse_dates=["Date"], index_col="Date")

        # -------------------------------------------------------------
        # E. COMBINAZIONE TRAMITE ORCHESTRATORE OTTIMIZZATO
        # -------------------------------------------------------------
        orch_cfg_file = config_dir / f"orchestrator_params_{asset_lower}.json"
        if not orch_cfg_file.exists():
            raise FileNotFoundError(f"Parametri Orchestratore non trovati per {asset_upper}: {orch_cfg_file}")

        with open(orch_cfg_file, "r", encoding="utf-8") as f:
            orch_cfg = json.load(f)

        best_params = orch_cfg.get("best_params", {})
        weights = best_params.get("weights", {})
        threshold = float(best_params.get("threshold", 0.15))

        signals_pool_full = {
            "SMA_EMA_Agent": pd.DataFrame({"signal": sig_sma_full, "confidence": conf_sma_full}),
            "ZScore_Agent": pd.DataFrame({"signal": sig_zs_full, "confidence": conf_zs_full}),
            "Macro_YieldCurve": pd.DataFrame({"signal": m_sig, "confidence": m_conf_full}),
            "FinBERT_Sentiment": df_sent_full[["signal", "confidence"]].copy(),
        }

        max_w = float(orch_cfg.get("max_weight", 0.35))
        orchestrator = OrchestratorAgent(
            weights=weights,
            threshold_buy=threshold,
            threshold_sell=threshold,
            consensus_boost=0.20,
            conflict_penalty=0.20,
            min_active_agents=1,
            max_weight=max_w,
            name=f"Orchestrator_{asset_upper}",
        )

        df_orchestrator_full = orchestrator.combine(signals_pool_full)

        # -------------------------------------------------------------
        # F. ISOLAMENTO DELLA SOLA FINESTRA OUT-OF-SAMPLE (2022-2026)
        # -------------------------------------------------------------
        df_oos_orch = df_orchestrator_full.loc[oos_mask, ["signal", "confidence"]].copy()
        df_oos_sma = signals_pool_full["SMA_EMA_Agent"].loc[oos_mask, ["signal", "confidence"]].copy()
        df_oos_zs = signals_pool_full["ZScore_Agent"].loc[oos_mask, ["signal", "confidence"]].copy()
        df_oos_macro = signals_pool_full["Macro_YieldCurve"].loc[oos_mask, ["signal", "confidence"]].copy()
        df_oos_sent = signals_pool_full["FinBERT_Sentiment"].loc[oos_mask, ["signal", "confidence"]].copy()

        # Salvataggio CSV dei segnali OOS
        out_orch_csv = signals_dir / f"oos_signals_orchestrator_{asset_lower}.csv"
        df_oos_orch.to_csv(out_orch_csv)
        df_oos_sma.to_csv(signals_dir / f"oos_signals_sma_ema_{asset_lower}.csv")
        df_oos_zs.to_csv(signals_dir / f"oos_signals_zscore_{asset_lower}.csv")
        df_oos_macro.to_csv(signals_dir / f"oos_signals_macro_{asset_lower}.csv")

        print(f"  [OK] Segnali OOS Orchestratore salvati in: {out_orch_csv.relative_to(root_dir)}")

        # -------------------------------------------------------------
        # G. CALCOLO E CONFRONTO DELLE PERFORMANCE OOS
        # -------------------------------------------------------------
        ret_series = df_features[price_col].astype(float).pct_change().fillna(0.0)
        ret_oos = ret_series.loc[oos_mask]

        # Posizione della strategia: shift(1) per evitare look-ahead bias
        pos_orch = df_oos_orch["signal"].shift(1).fillna(0.0)
        strat_oos_ret = pos_orch * ret_oos

        m_strat = calculate_performance_metrics(strat_oos_ret)
        m_bnh = calculate_performance_metrics(ret_oos)

        trades_count = int((pos_orch.diff().fillna(0) != 0).sum())
        alpha_cum = m_strat["cumulative_return"] - m_bnh["cumulative_return"]

        # Distribuzione dei segnali OOS generati
        sig_dist = dict(df_oos_orch["signal"].value_counts())
        n_long = sig_dist.get(1, 0)
        n_short = sig_dist.get(-1, 0)
        n_flat = sig_dist.get(0, 0)

        record = {
            "Asset": asset_upper,
            "Strat_CumRet": round(m_strat["cumulative_return"] * 100, 2),
            "BnH_CumRet": round(m_bnh["cumulative_return"] * 100, 2),
            "Alpha_CumRet": round(alpha_cum * 100, 2),
            "Strat_AnnRet": round(m_strat["annualized_return"] * 100, 2),
            "BnH_AnnRet": round(m_bnh["annualized_return"] * 100, 2),
            "Strat_Sharpe": round(m_strat["sharpe_ratio"], 2),
            "BnH_Sharpe": round(m_bnh["sharpe_ratio"], 2),
            "Strat_MaxDD": round(m_strat["max_drawdown"] * 100, 2),
            "BnH_MaxDD": round(m_bnh["max_drawdown"] * 100, 2),
            "Strat_ProfFact": round(m_strat["profit_factor"], 2),
            "Trades": trades_count,
            "Long_Weeks": n_long,
            "Short_Weeks": n_short,
            "Flat_Weeks": n_flat,
            "Weights_Used": str({k: round(v, 2) for k, v in weights.items()}),
            "Threshold": threshold,
        }
        oos_results_summary.append(record)
        oos_output_dfs[asset_upper] = df_oos_orch

        # Stampa a video dettagliata per l'asset
        print(f"  Pesi caricati : {weights}")
        print(f"  Soglia usata  : {threshold}")
        print(f"  Distribuzione : Long={n_long}w, Short={n_short}w, Flat={n_flat}w ({len(df_oos_orch)} settimane)")
        print(f"  Confronto OOS : Strategia={m_strat['cumulative_return']*100:+6.1f}% | "
              f"Buy&Hold={m_bnh['cumulative_return']*100:+6.1f}% | Alpha={alpha_cum*100:+6.1f}%")
        print(f"  Sharpe / MDD  : Strat Sh={m_strat['sharpe_ratio']:+5.2f} (MDD={m_strat['max_drawdown']*100:4.1f}%) | "
              f"B&H Sh={m_bnh['sharpe_ratio']:+5.2f} (MDD={m_bnh['max_drawdown']*100:4.1f}%)")

    # -----------------------------------------------------------------
    # H. TABELLA COMPARATIVA FINALE E SALVATAGGIO REPORT
    # -----------------------------------------------------------------
    df_summary = pd.DataFrame(oos_results_summary)
    summary_csv = results_dir / "oos_performance_summary.csv"
    df_summary.to_csv(summary_csv, index=False)

    print("\n" + "=" * 92)
    print("           RIEPILOGO COMPARATIVO OUT-OF-SAMPLE (2022-2026): STRATEGIA vs BUY & HOLD")
    print("=" * 92)
    print(f"{'Asset':<6} | {'Strat Cum':<10} | {'B&H Cum':<10} | {'Alpha':<9} | {'Strat Sh':<9} | {'B&H Sh':<8} | {'Strat MDD':<10} | {'Trades':<6}")
    print("-" * 92)
    for _, r in df_summary.iterrows():
        print(
            f"{r['Asset']:<6} | "
            f"{r['Strat_CumRet']:+9.1f}% | "
            f"{r['BnH_CumRet']:+9.1f}% | "
            f"{r['Alpha_CumRet']:+8.1f}% | "
            f"{r['Strat_Sharpe']:>9.2f} | "
            f"{r['BnH_Sharpe']:>8.2f} | "
            f"{r['Strat_MaxDD']:>9.1f}% | "
            f"{r['Trades']:>6d}"
        )
    print("=" * 92)
    print(f"\n[OK] Riepilogo performance OOS salvato in: {summary_csv.relative_to(root_dir)}")

    return oos_output_dfs


# =============================================================================
# Pipeline completa deterministica baseline
# =============================================================================

def run_pipeline() -> None:
    """Esegue la pipeline completa del baseline Macro / Rates Regimes."""
    from src.data_acquisition import run_data_acquisition
    from src.data_transformation import run_data_transformation
    from src.feature_engineering import run_feature_engineering
    from src.regime_model import run_regime_model
    from src.signal_generation import run_signal_generation
    from src.portfolio import run_portfolio
    from src.backtester import run_backtest
    from src.evaluator import run_evaluation

    config = load_config(CONFIG_PATH)

    validate_config(config)
    initialize_environment(config)
    print_project_summary(config)

    pipeline_config = config.get("pipeline", {})
    run_download = bool(pipeline_config.get("run_data_acquisition", False))

    if run_download:
        print("[0/7] Data acquisition")
        run_data_acquisition(config, ROOT_DIR)
    else:
        print("[0/7] Data acquisition skipped (using local raw CSVs)")

    print("\n[1/7] Data transformation")
    run_data_transformation(config, ROOT_DIR)

    print("\n[2/7] Feature engineering")
    run_feature_engineering(config, ROOT_DIR)

    print("\n[3/7] Regime model")
    run_regime_model(config, ROOT_DIR)

    print("\n[4/7] Signal generation")
    run_signal_generation(config, ROOT_DIR)

    print("\n[5/7] Portfolio construction")
    run_portfolio(config, ROOT_DIR)

    print("\n[6/7] Backtest")
    run_backtest(config, ROOT_DIR)

    print("\n[7/7] Evaluation")
    run_evaluation(config, ROOT_DIR)

    print("\n======================================================")
    print("     END-TO-END PIPELINE COMPLETED SUCCESSFULLY")
    print("======================================================")


# =============================================================================
# ENTRY POINT PRINCIPALE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ProjectAgentTrading26 - Pipeline Principale & Previsione Out-Of-Sample."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="oos",
        choices=["oos", "pipeline", "check_finbert", "all"],
        help="Modalita di esecuzione: 'oos' (previsione OOS 2022-2026 con agenti addestrati, DEFAULT), "
             "'pipeline' (pipeline deterministica baseline), 'check_finbert' (audit notizie FinBERT), "
             "'all' (esegue sia pipeline che previsione OOS).",
    )
    parser.add_argument(
        "--oos",
        "--oos-only",
        "--predict",
        dest="flag_oos",
        action="store_true",
        help="Forza l'esecuzione della sola previsione OOS (2022-2026).",
    )
    parser.add_argument(
        "--pipeline",
        dest="flag_pipeline",
        action="store_true",
        help="Forza l'esecuzione della pipeline baseline completa.",
    )
    parser.add_argument(
        "--check-finbert",
        dest="flag_check_finbert",
        action="store_true",
        help="Esegue esclusivamente l'audit del dataset FinBERT OOS.",
    )

    args = parser.parse_args()

    # Risoluzione della modalita richiesta
    selected_mode = args.mode.lower()
    if args.flag_check_finbert:
        selected_mode = "check_finbert"
    elif args.flag_pipeline:
        selected_mode = "pipeline"
    elif args.flag_oos:
        selected_mode = "oos"

    if selected_mode == "check_finbert":
        check_finbert_oos_news(ROOT_DIR)
    elif selected_mode == "pipeline":
        run_pipeline()
    elif selected_mode == "all":
        run_pipeline()
        run_oos_prediction(root_dir=ROOT_DIR)
    else:  # default 'oos'
        run_oos_prediction(root_dir=ROOT_DIR)
