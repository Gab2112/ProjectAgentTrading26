"""
train.py

Script di training e tuning dei parametri per gli agenti statistici:
- SMAEMAAgent (Trend-Following)
- ZScoreAgent (Breakout/Momentum & Mean-Reversion con Isteresi)

Funzionalita principali:
- Esegue l'ottimizzazione Walk-Forward su tutti i 3 asset dell'universo (SHY, IEF, TLT).
- Finestra mobile: 5 anni di training (In-Sample) -> test sull'anno successivo (Out-Of-Sample).
- Funzione obiettivo di stabilita e penalizzazione del Max Drawdown ('sharpe_mdd').
- Rispetta il vincolo OOS lock: training limitato fino al 2021 (max_train_year = 2021).
- Salva i file JSON di configurazione in src/agents/config/.
- Genera per ciascuno dei 3 indici un DataFrame dedicato con esattamente 2 colonne:
    1. 'signal' (discreto: {-1, 0, 1})
    2. 'confidence' (continuo: probabilita / confidenza in [0.0, 1.0])
- Salva i 3 DataFrame in data/signals/signals_statistical_{asset}.csv.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Risoluzione robusta dei percorsi
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.agents.base_agent import calculate_performance_metrics
from src.agents.EMA_EMW_agent import SMAEMAAgent, SMAEMAOptimizer, WalkForwardTrainer as SMAEMATrainer
from src.agents.ZScore_agent import ZScoreAgent, ZScoreOptimizer, ZScoreWalkForwardTrainer


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
            # Pieno accordo tra i due agenti statistici
            out_sig[i] = sig1
            out_conf[i] = min(1.0, round(float((conf1 + conf2) / 2.0 * 1.15), 4))
        elif sig1 != 0 and sig2 == 0:
            # Solo SMA/EMA attivo
            out_sig[i] = sig1
            out_conf[i] = round(float(conf1 * 0.85), 4)
        elif sig1 == 0 and sig2 != 0:
            # Solo Z-Score attivo
            out_sig[i] = sig2
            out_conf[i] = round(float(conf2 * 0.85), 4)
        elif sig1 != 0 and sig2 != 0 and sig1 != sig2:
            # Disaccordo diretto (+1 vs -1)
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
    max_train_year: int = 2021,
    objective_metric: str = "sharpe_mdd",
) -> Tuple[Dict, Dict, pd.DataFrame]:
    """
    Esegue il training Walk-Forward per un asset specifico (es. 'SHY', 'IEF', 'TLT'),
    salva le configurazioni ottimali e restituisce il DataFrame (signal, confidence).
    """
    price_col = f"{asset}_close"
    if price_col not in df.columns:
        raise KeyError(f"Colonna '{price_col}' non trovata nel dataset.")

    print("\n" + "#" * 80)
    print(f"### AVVIO TRAINING & OTTIMIZZAZIONE WALK-FORWARD: {asset} ({price_col})")
    print("#" * 80)

    # 1. Ottimizzazione SMA/EMA (Trend-Following)
    print(f"\n[{asset}] >>> Walk-Forward: SMA/EMA Agent...")
    sma_opt = SMAEMAOptimizer(price_col=price_col, objective_metric=objective_metric)
    sma_trainer = SMAEMATrainer(optimizer=sma_opt, train_window_years=5, price_col=price_col)
    sma_res = sma_trainer.run(df, max_year=max_train_year)

    sma_config_file = config_dir / f"sma_ema_params_{asset.lower()}.json"
    sma_trainer.save_final_agent_config(sma_config_file)

    # 2. Ottimizzazione Z-Score (Momentum / Mean-Reversion con Isteresi)
    print(f"\n[{asset}] >>> Walk-Forward: Z-Score Agent...")
    zs_opt = ZScoreOptimizer(price_col=price_col, exit_threshold=0.0, objective_metric=objective_metric)
    zs_trainer = ZScoreWalkForwardTrainer(optimizer=zs_opt, train_window_years=5, price_col=price_col)
    zs_res = zs_trainer.run(df, max_year=max_train_year)

    zs_config_file = config_dir / f"zscore_params_{asset.lower()}.json"
    zs_trainer.save_final_agent_config(zs_config_file)

    # 3. Istanziazione Agenti Ottimizzati e Predizione con Confidenza
    sma_agent = SMAEMAAgent(price_col=price_col)
    sma_agent.load_params(sma_config_file)
    sig_sma, conf_sma = sma_agent.predict_with_confidence(df)

    zs_agent = ZScoreAgent(price_col=price_col)
    zs_agent.load_params(zs_config_file)
    sig_zs, conf_zs = zs_agent.predict_with_confidence(df)

    # 4. Generazione DataFrame con le 2 colonne richieste: 'signal' e 'confidence'
    df_signals = combine_statistical_signals(
        signals_sma=sig_sma,
        conf_sma=conf_sma,
        signals_zs=sig_zs,
        conf_zs=conf_zs,
    )

    # Verifica formale vincoli sulle colonne
    assert list(df_signals.columns) == ["signal", "confidence"], (
        f"Errore formato colonne: atteso ['signal', 'confidence'], ottenuto {list(df_signals.columns)}"
    )

    # 5. Salvataggio DataFrame segnali
    output_csv = signals_dir / f"signals_statistical_{asset.lower()}.csv"
    df_signals.to_csv(output_csv)
    print(f"[{asset}] Segnali statistici salvati in: {output_csv}")

    # 6. Calcolo metriche storiche comparative sul periodo IS (fino al 2021)
    ret_asset = df[price_col].astype(float).pct_change().fillna(0.0)
    train_mask = df.index.year <= max_train_year

    pos_sma = sig_sma.shift(1).fillna(0.0)
    pos_zs = sig_zs.shift(1).fillna(0.0)
    pos_comb = df_signals["signal"].shift(1).fillna(0.0)

    strat_sma = (pos_sma * ret_asset)[train_mask]
    strat_zs = (pos_zs * ret_asset)[train_mask]
    strat_comb = (pos_comb * ret_asset)[train_mask]

    m_sma = calculate_performance_metrics(strat_sma)
    m_zs = calculate_performance_metrics(strat_zs)
    m_comb = calculate_performance_metrics(strat_comb)

    t_sma = int((pos_sma.diff().fillna(0) != 0).sum())
    t_zs = int((pos_zs.diff().fillna(0) != 0).sum())
    t_comb = int((pos_comb.diff().fillna(0) != 0).sum())

    print("\n" + "-" * 78)
    print(f"  PERFORMANCE IN-SAMPLE (2004-{max_train_year}) - ASSET: {asset}")
    print("-" * 78)
    print(f"{'Agente / Ensemble':<24} | {'Cum Ret':<10} | {'Sharpe':<8} | {'Max DD':<8} | {'Prof Fact':<10} | {'Trades':<6}")
    print("-" * 78)
    print(
        f"{'SMA/EMA Agent':<24} | {m_sma['cumulative_return']*100:+8.1f}% | "
        f"{m_sma['sharpe_ratio']:>8.2f} | {m_sma['max_drawdown']*100:>6.1f}% | {m_sma['profit_factor']:>10.2f} | {t_sma:>6d}"
    )
    print(
        f"{'Z-Score Agent':<24} | {m_zs['cumulative_return']*100:+8.1f}% | "
        f"{m_zs['sharpe_ratio']:>8.2f} | {m_zs['max_drawdown']*100:>6.1f}% | {m_zs['profit_factor']:>10.2f} | {t_zs:>6d}"
    )
    print(
        f"{'Ensemble Statistico':<24} | {m_comb['cumulative_return']*100:+8.1f}% | "
        f"{m_comb['sharpe_ratio']:>8.2f} | {m_comb['max_drawdown']*100:>6.1f}% | {m_comb['profit_factor']:>10.2f} | {t_comb:>6d}"
    )
    print("-" * 78)

    return sma_res, zs_res, df_signals


# =====================================================================
# MAIN ORCHESTRATOR
# =====================================================================
def run_all_assets(
    assets: Optional[List[str]] = None,
    max_train_year: int = 2021,
    objective_metric: str = "sharpe_mdd",
) -> Dict[str, pd.DataFrame]:
    """
    Esegue il training Walk-Forward completo per tutti gli asset specificati
    e restituisce un dizionario contenente i 3 DataFrame (uno per ciascun indice).

    Ritorna:
        Dict[str, pd.DataFrame]: {'SHY': df_shy, 'IEF': df_ief, 'TLT': df_tlt}
        ciascuno con esattamente 2 colonne: ['signal', 'confidence'].
    """
    if assets is None:
        assets = ["SHY", "IEF", "TLT"]

    config_dir = CURRENT_DIR / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    signals_dir = PROJECT_ROOT / "data" / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)

    data_path = PROJECT_ROOT / "data" / "processed" / "features_weekly.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset delle feature non trovato in: {data_path}. "
            f"Assicurati di aver generato features_weekly.csv."
        )

    df = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")
    print(f"[Train] Dataset caricato: {len(df)} righe dal {df.index[0].date()} al {df.index[-1].date()}")
    print(f"[Train] Asset da ottimizzare: {assets}")
    print(f"[Train] Metrica obiettivo: '{objective_metric}'")
    print(f"[Train] Vincolo OOS Lock: train fino al {max_train_year}")

    results = {}
    signal_dataframes = {}

    for asset in assets:
        sma_res, zs_res, df_sig = train_single_asset(
            asset=asset,
            df=df,
            config_dir=config_dir,
            signals_dir=signals_dir,
            max_train_year=max_train_year,
            objective_metric=objective_metric,
        )
        results[asset] = {"sma": sma_res, "zscore": zs_res}
        signal_dataframes[asset] = df_sig

    # Riepilogo finale comparativo
    print("\n" + "=" * 85)
    print("                 RIEPILOGO FINALE WALK-FORWARD PER I 3 INDICI")
    print("=" * 85)
    print(f"{'Asset':<6} | {'SMA/EMA Opt':<18} | {'SMA OOS Sh':<10} | {'ZScore Opt':<24} | {'ZS OOS Sh':<10}")
    print("-" * 85)
    for asset in assets:
        sma_r = results[asset]["sma"]
        zs_r = results[asset]["zscore"]
        sma_opt_str = f"S={sma_r['sma_window']}, E={sma_r['ema_window']}"
        sma_sh = f"{sma_r['summary_metrics']['mean_oos_sharpe']:+.2f}"
        zs_opt_str = f"W={zs_r['window']}, Th={zs_r['threshold']:.2f}, {zs_r['mode'][:3]}"
        zs_sh = f"{zs_r['summary_metrics']['mean_oos_sharpe']:+.2f}"
        print(f"{asset:<6} | {sma_opt_str:<18} | {sma_sh:<10} | {zs_opt_str:<24} | {zs_sh:<10}")
    print("=" * 85)

    print("\n[OK] Tutti i 3 DataFrame dei segnali sono stati generati con successo:")
    for asset, sig_df in signal_dataframes.items():
        print(
            f"  - {asset}: shape={sig_df.shape}, colonne={list(sig_df.columns)}, "
            f"distribuzione segnali={dict(sig_df['signal'].value_counts())}"
        )

    return signal_dataframes


if __name__ == "__main__":
    target_arg = sys.argv[1] if len(sys.argv) > 1 else "ALL"
    metric_arg = sys.argv[2] if len(sys.argv) > 2 else "sharpe_mdd"

    if target_arg.upper() == "ALL":
        target_assets = ["SHY", "IEF", "TLT"]
    else:
        target_assets = [target_arg.upper()]

    run_all_assets(assets=target_assets, max_train_year=2021, objective_metric=metric_arg)