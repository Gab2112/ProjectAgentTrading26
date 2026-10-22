"""
train.py

Script di training e tuning dei parametri per tutti gli agenti del framework:
- SMAEMAAgent          (Trend-Following basato su Crossover SMA/EMA)
- ZScoreAgent          (Breakout/Momentum & Mean-Reversion con Isteresi)
- MacroYieldCurveAgent (Macro Quantitativo: Yield Curve, Tassi e Inflazione)
- OrchestratorAgent    (Ensemble Multi-Agente: pesi e soglie decisionali ottimizzate)

Funzionalita principali:
- Esegue l'ottimizzazione Walk-Forward su tutti i 3 asset (SHY, IEF, TLT).
- Finestra mobile: 5 anni di training In-Sample -> 1 anno di test Out-Of-Sample.
- Funzione obiettivo: 'profit_oriented' (priorita al rendimento cumulato e alla stabilita di profitto).
- Rispetta il vincolo OOS lock: training limitato fino al 2021 (max_train_year = 2021).
- Salva tutti i file JSON di configurazione in src/agents/config/.
- Genera per ciascuno dei 3 indici i seguenti DataFrame (2 colonne: 'signal', 'confidence'):
    * data/signals/signals_sma_ema_{asset}.csv       (SMA/EMA Agent)
    * data/signals/signals_zscore_{asset}.csv        (Z-Score Agent)
    * data/signals/signals_macro_{asset}.csv         (MacroYieldCurveAgent)
    * data/signals/signals_orchestrator_{asset}.csv  (Orchestratore: 4 agenti indipendenti con FinBERT)
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Risoluzione robusta dei percorsi
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.agents.base_agent import calculate_performance_metrics, compute_objective_score
from src.agents.EMA_EMW_agent import SMAEMAAgent, SMAEMAOptimizer, WalkForwardTrainer as SMAEMATrainer
from src.agents.ZScore_agent import ZScoreAgent, ZScoreOptimizer, ZScoreWalkForwardTrainer
from src.agents.macro_yield_curve_agent import MacroYieldCurveAgent
from src.agents.orchestrator_agent import (
    OrchestratorAgent,
    OrchestratorOptimizer,
    align_signals,
)


# =====================================================================
# PIPELINE DI TRAINING PER SINGOLO ASSET (SMA/EMA + ZScore)
# =====================================================================
def train_single_asset(
    asset: str,
    df: pd.DataFrame,
    config_dir: Path,
    signals_dir: Path,
    max_train_year: int = 2021,
    objective_metric: str = "profit_oriented",
) -> Tuple[Dict, Dict, pd.DataFrame, pd.DataFrame]:
    """
    Esegue il training Walk-Forward per un asset specifico (es. 'SHY', 'IEF', 'TLT'),
    salva le configurazioni ottimali e i CSV dei segnali individuali.

    Ritorna:
        sma_res   : dizionario con i risultati walk-forward SMA/EMA.
        zs_res    : dizionario con i risultati walk-forward Z-Score.
        df_sma    : DataFrame (signal, confidence) del SMA/EMA Agent.
        df_zscore : DataFrame (signal, confidence) dello Z-Score Agent.
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

    # 3. Predizione con confidenza sui parametri ottimizzati
    sma_agent = SMAEMAAgent(price_col=price_col)
    sma_agent.load_params(sma_config_file)
    sig_sma, conf_sma = sma_agent.predict_with_confidence(df)

    zs_agent = ZScoreAgent(price_col=price_col)
    zs_agent.load_params(zs_config_file)
    sig_zs, conf_zs = zs_agent.predict_with_confidence(df)

    # 4. DataFrame individuali (signal, confidence) per ciascun agente
    df_sma = pd.DataFrame(
        {"signal": sig_sma.astype(int), "confidence": conf_sma},
        index=df.index,
    )
    df_zscore = pd.DataFrame(
        {"signal": sig_zs.astype(int), "confidence": conf_zs},
        index=df.index,
    )

    # 5. Salvataggio CSV individuali
    sma_csv = signals_dir / f"signals_sma_ema_{asset.lower()}.csv"
    zs_csv  = signals_dir / f"signals_zscore_{asset.lower()}.csv"
    df_sma.to_csv(sma_csv)
    df_zscore.to_csv(zs_csv)
    print(f"[{asset}] Segnali SMA/EMA  salvati in: {sma_csv}")
    print(f"[{asset}] Segnali Z-Score  salvati in: {zs_csv}")

    # 6. Metriche IS comparative (solo i due agenti singoli)
    ret_asset  = df[price_col].astype(float).pct_change().fillna(0.0)
    train_mask = df.index.year <= max_train_year

    pos_sma = sig_sma.shift(1).fillna(0.0)
    pos_zs  = sig_zs.shift(1).fillna(0.0)

    m_sma = calculate_performance_metrics((pos_sma * ret_asset)[train_mask])
    m_zs  = calculate_performance_metrics((pos_zs  * ret_asset)[train_mask])

    t_sma = int((pos_sma.diff().fillna(0) != 0).sum())
    t_zs  = int((pos_zs.diff().fillna(0)  != 0).sum())

    print("\n" + "-" * 78)
    print(f"  PERFORMANCE IN-SAMPLE (2004-{max_train_year}) - ASSET: {asset}")
    print("-" * 78)
    print(f"{'Agente':<24} | {'Cum Ret':<10} | {'Sharpe':<8} | {'Max DD':<8} | {'Prof Fact':<10} | {'Trades':<6}")
    print("-" * 78)
    print(
        f"{'SMA/EMA Agent':<24} | {m_sma['cumulative_return']*100:+8.1f}% | "
        f"{m_sma['sharpe_ratio']:>8.2f} | {m_sma['max_drawdown']*100:>6.1f}% | "
        f"{m_sma['profit_factor']:>10.2f} | {t_sma:>6d}"
    )
    print(
        f"{'Z-Score Agent':<24} | {m_zs['cumulative_return']*100:+8.1f}% | "
        f"{m_zs['sharpe_ratio']:>8.2f} | {m_zs['max_drawdown']*100:>6.1f}% | "
        f"{m_zs['profit_factor']:>10.2f} | {t_zs:>6d}"
    )
    print("-" * 78)

    return sma_res, zs_res, df_sma, df_zscore


# =====================================================================
# MACRO YIELD CURVE AGENT — OTTIMIZZATORE E WALK-FORWARD TRAINER
# =====================================================================
class MacroYieldCurveOptimizer:
    """
    Ottimizzatore quantitativo per MacroYieldCurveAgent.

    Parametri ottimizzati tramite grid search In-Sample (fino al max_train_year):
    - slope_window    : finestra (in settimane) per il calcolo del delta della pendenza di curva.
                        Valori esplorati: [2, 4, 8, 13, 26].
    - rate_window     : finestra di smoothing (in settimane) sulla variazione dei tassi.
                        Valori esplorati: [2, 4, 8, 13].
    - score_threshold : soglia del macro-score per la generazione del segnale direzionale.
                        Valori esplorati: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].

    Il MacroYieldCurveAgent costruisce internamente uno score macro composito pesato:
        score(t) = 0.50 * rate_component(t)
                 + 0.35 * curve_regime_component(t)
                 + 0.15 * inflation_component(t)
                 + inversion_bonus(t)

    dove:
      - rate_component    = -1 * (smoothed_rate_change / rolling_std_rate)
                            Positivo se i tassi a lungo scendono (BUY TLT).
      - curve_component   = +1.0 (o +1.5) in Bull regime, -1.0 (o -1.5) in Bear regime,
                            con segno determinato da delta_slope normalizzato.
      - inflation_component = +0.5 se l'inflazione YoY e in calo (13w), -0.5 se in salita.
      - inversion_bonus   = +0.5 aggiuntivo se la curva 10Y-2Y e gia invertita (slope < 0).

    Segnale finale: BUY(+1) se score >= score_threshold, SELL(-1) se score <= -score_threshold,
                    HOLD(0) altrimenti.

    L'ottimizzazione seleziona la combinazione (slope_window, rate_window, score_threshold)
    che massimizza la metrica obiettivo (default: 'profit_oriented') sull'intero campione In-Sample.
    I parametri ottimi vengono poi usati per generare i segnali sull'intero dataset.
    """

    SLOPE_WINDOWS: List[int] = [2, 4, 8, 13, 26]
    RATE_WINDOWS: List[int] = [2, 4, 8, 13]
    SCORE_THRESHOLDS: List[float] = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0]

    def __init__(
        self,
        objective_metric: str = "profit_oriented",
        max_train_year: int = 2021,
        annualization_factor: int = 52,
        slope_col: str = "curve_10y_2y",
        rate_change_col: str = "treasury_10y_change_1w",
        inflation_col: str = "cpi_yoy",
    ):
        self.objective_metric = objective_metric
        self.max_train_year = max_train_year
        self.annualization_factor = annualization_factor
        self.slope_col = slope_col
        self.rate_change_col = rate_change_col
        self.inflation_col = inflation_col

        self.best_params: Dict[str, Any] = {}
        self.best_score: float = -999.0
        self.in_sample_metrics: Dict[str, float] = {}

    def optimize(
        self,
        df: pd.DataFrame,
        asset: str = "TLT",
    ) -> Tuple["MacroYieldCurveAgent", Dict[str, Any]]:
        """
        Esegue la grid search completa dei 3 iperparametri sul campione In-Sample.

        L'agente macroeconomico non e specifico per singolo asset nel suo score interno:
        lo stesso score e applicato alla serie dei rendimenti dell'asset target per valutare
        quale combinazione di parametri produce la strategia piu stabile e profittevole.

        Ritorna:
            - Il MacroYieldCurveAgent istanziato con i parametri ottimi.
            - Il dizionario con i best_params e le metriche IS ottimali.
        """
        price_col = f"{asset}_close"
        if price_col not in df.columns:
            raise KeyError(f"Colonna '{price_col}' non trovata nel DataFrame.")

        # Rendimenti periodici dell'asset
        asset_returns = df[price_col].astype(float).pct_change().fillna(0.0)

        # Maschera In-Sample rigorosa (OOS lock al max_train_year)
        is_mask = df.index.year <= self.max_train_year
        df_is = df.loc[is_mask]
        ret_is = asset_returns.loc[is_mask]

        print(f"\n{'='*80}")
        print(f"  OTTIMIZZAZIONE MacroYieldCurveAgent - ASSET: {asset} | OOS LOCK <= {self.max_train_year}")
        print(f"  Campione IS: {df_is.index[0].date()} -> {df_is.index[-1].date()} ({len(df_is)} settimane)")
        print(f"  Metrica: '{self.objective_metric}' | Grid: {len(self.SLOPE_WINDOWS)}x{len(self.RATE_WINDOWS)}x{len(self.SCORE_THRESHOLDS)} = "
              f"{len(self.SLOPE_WINDOWS)*len(self.RATE_WINDOWS)*len(self.SCORE_THRESHOLDS)} combinazioni")
        print("="*80)

        best_score = -999.0
        best_combo: Dict[str, Any] = {}
        results_list: List[Dict[str, Any]] = []

        for sw in self.SLOPE_WINDOWS:
            for rw in self.RATE_WINDOWS:
                for th in self.SCORE_THRESHOLDS:
                    agent = MacroYieldCurveAgent(
                        slope_col=self.slope_col,
                        rate_change_col=self.rate_change_col,
                        inflation_col=self.inflation_col,
                        slope_window=sw,
                        rate_window=rw,
                        score_threshold=th,
                    )

                    # Calcolo score e segnali sul solo campione IS
                    try:
                        score_series = agent.calculate_macro_score(df_is)
                        raw_sig = np.where(
                            score_series >= th, 1,
                            np.where(score_series <= -th, -1, 0)
                        )
                        sig_is = pd.Series(raw_sig, index=df_is.index, dtype=int)
                        sig_is[score_series.isna()] = 0
                    except (KeyError, Exception):
                        continue

                    pos = sig_is.shift(1).fillna(0.0)
                    strat_ret = pos * ret_is

                    metrics = calculate_performance_metrics(
                        strat_ret, annualization_factor=self.annualization_factor
                    )
                    score = compute_objective_score(metrics, metric_type=self.objective_metric)
                    n_trades = int((pos.diff().fillna(0) != 0).sum())

                    record = {
                        "slope_window": sw,
                        "rate_window": rw,
                        "score_threshold": th,
                        "objective_score": round(score, 4),
                        "sharpe_ratio": round(metrics["sharpe_ratio"], 4),
                        "max_drawdown": round(metrics["max_drawdown"], 4),
                        "cumulative_return": round(metrics["cumulative_return"], 4),
                        "annualized_return": round(metrics["annualized_return"], 4),
                        "profit_factor": round(metrics["profit_factor"], 4),
                        "calmar_ratio": round(metrics["calmar_ratio"], 4),
                        "trades": n_trades,
                    }
                    results_list.append(record)

                    if score > best_score:
                        best_score = score
                        best_combo = record

        self.best_score = best_score
        self.best_params = best_combo
        self.in_sample_metrics = {
            k: best_combo[k]
            for k in ["sharpe_ratio", "max_drawdown", "cumulative_return",
                      "annualized_return", "profit_factor", "calmar_ratio",
                      "objective_score", "trades"]
        }

        # Agente ottimale
        best_agent = MacroYieldCurveAgent(
            slope_col=self.slope_col,
            rate_change_col=self.rate_change_col,
            inflation_col=self.inflation_col,
            slope_window=int(best_combo["slope_window"]),
            rate_window=int(best_combo["rate_window"]),
            score_threshold=float(best_combo["score_threshold"]),
        )

        # Stampa riepilogo top-5
        df_res = pd.DataFrame(results_list).sort_values("objective_score", ascending=False)
        print(f"\n  Top-5 combinazioni MacroYieldCurve per {asset}:")
        print(f"  {'SW':>4} {'RW':>4} {'Th':>5} {'Sharpe':>7} {'MDD':>7} {'Score':>8}")
        print(f"  {'-'*48}")
        for _, row in df_res.head(5).iterrows():
            print(
                f"  {int(row['slope_window']):>4} {int(row['rate_window']):>4} "
                f"{row['score_threshold']:>5.2f} {row['sharpe_ratio']:>7.3f} "
                f"{row['max_drawdown']*100:>6.1f}% {row['objective_score']:>8.4f}"
            )
        print(
            f"\n  [OTTIMO] slope_window={int(best_combo['slope_window'])}, "
            f"rate_window={int(best_combo['rate_window'])}, "
            f"score_threshold={best_combo['score_threshold']:.2f} "
            f"-> Score={best_score:.4f}"
        )

        return best_agent, best_combo

    def save_config(self, filepath: Union[str, Path]) -> None:
        """Salva i parametri ottimi dell'agente macro in formato JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "slope_col": self.slope_col,
            "rate_change_col": self.rate_change_col,
            "inflation_col": self.inflation_col,
            "slope_window": int(self.best_params.get("slope_window", 4)),
            "rate_window": int(self.best_params.get("rate_window", 4)),
            "score_threshold": float(self.best_params.get("score_threshold", 0.5)),
            "objective_metric": self.objective_metric,
            "max_train_year": self.max_train_year,
            "in_sample_metrics": self.in_sample_metrics,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
        print(f"  Configurazione MacroYieldCurveAgent salvata in: {path}")


# =====================================================================
# PIPELINE DI TRAINING MACRO YIELD CURVE PER SINGOLO ASSET
# =====================================================================
def train_macro_asset(
    asset: str,
    df: pd.DataFrame,
    config_dir: Path,
    signals_dir: Path,
    max_train_year: int = 2021,
    objective_metric: str = "profit_oriented",
) -> Tuple["MacroYieldCurveAgent", pd.DataFrame]:
    """
    Esegue l'ottimizzazione del MacroYieldCurveAgent per un singolo asset,
    salva la configurazione JSON ottimale e il CSV dei segnali.

    Cosa viene ottimizzato:
    ========================
    Il MacroYieldCurveAgent non fa previsioni sui prezzi, ma interpreta il REGIME
    macroeconomico corrente combinando tre fonti di segnale:

    1. VARIAZIONE DEI TASSI (rate_component, peso 50%):
       - Confronta la variazione settimanale del Treasury 10Y con la sua deviazione
         standard rolling (finestra 26 settimane) per ottenere un z-score normalizzato.
       - La finestra di smoothing 'rate_window' controlla quanto levigare la variazione
         prima di normalizzarla: valori piccoli (2-4w) -> reattivita, valori grandi (13w)
         -> stabilita e meno segnali spuri.
       - Segno invertito: tassi in calo -> score positivo (bull bond market).

    2. REGIME DI CURVA (curve_component, peso 35%):
       - Calcola il delta della pendenza (spread 10Y-2Y) negli ultimi 'slope_window' periodi.
       - Identifica i 4 regimi classici di curva:
           * Bull Flattening  (rate down, slope decreases): massimo rialzo per TLT.
           * Bull Steepening  (rate down, slope increases): rialzo moderato per TLT.
           * Bear Flattening  (rate up,   slope decreases): stretta Fed, short TLT.
           * Bear Steepening  (rate up,   slope increases): inflazione/risk premium, short TLT.
       - 'slope_window' piccolo (2-4w) e piu reattivo; 13-26w filtra meglio il rumore.
       - Bonus aggiuntivo (+0.5) se la curva e gia invertita in senso assoluto (slope < 0).

    3. INFLAZIONE (inflation_component, peso 15%):
       - Variazione YoY del CPI a 13 settimane (1 trimestre): se scende -> +0.5, se sale -> -0.5.
       - Non viene ottimizzata perche contribuisce marginalmente e ha un unico regime.

    La soglia 'score_threshold' controlla la selettivita del segnale:
       - Alta (0.7-1.0): solo regimi macro molto chiari generano segnale -> meno trade, piu precisione.
       - Bassa (0.3-0.5): piu segnali, piu reattivita al contesto macro, rischio piu rumore.

    Ritorna:
        - L'istanza MacroYieldCurveAgent con i parametri ottimi.
        - Il DataFrame dei segnali macro (signal, confidence) sull'intero dataset.
    """
    print("\n" + "#" * 80)
    print(f"### AVVIO TRAINING MacroYieldCurveAgent: {asset}")
    print("#" * 80)

    optimizer = MacroYieldCurveOptimizer(
        objective_metric=objective_metric,
        max_train_year=max_train_year,
    )
    best_agent, best_combo = optimizer.optimize(df, asset=asset)

    # Salvataggio configurazione JSON
    config_file = config_dir / f"macro_yield_params_{asset.lower()}.json"
    optimizer.save_config(config_file)

    # Generazione segnali sull'intero dataset con i parametri ottimi
    macro_multi = best_agent.predict_multi_asset(df)
    macro_score = best_agent.calculate_macro_score(df)

    sig_col = f"{asset}_signal"
    m_sig = macro_multi[sig_col] if sig_col in macro_multi.columns else pd.Series(0, index=df.index)
    m_conf_raw = (macro_score.abs() / 2.0).clip(0.0, 1.0).round(4)
    m_conf = pd.Series(
        np.where(m_sig != 0, np.maximum(0.50, m_conf_raw), 0.0),
        index=df.index,
    )

    df_macro_signals = pd.DataFrame(
        {"signal": m_sig.astype(int), "confidence": m_conf},
        index=df.index,
    )

    assert list(df_macro_signals.columns) == ["signal", "confidence"], (
        f"Errore formato colonne macro: atteso ['signal', 'confidence'], "
        f"ottenuto {list(df_macro_signals.columns)}"
    )

    output_csv = signals_dir / f"signals_macro_{asset.lower()}.csv"
    df_macro_signals.to_csv(output_csv)
    print(f"[{asset}] Segnali MacroYieldCurve salvati in: {output_csv}")

    # Riepilogo prestazioni IS sui segnali finali
    ret_asset = df[f"{asset}_close"].astype(float).pct_change().fillna(0.0)
    is_mask = df.index.year <= max_train_year
    pos = m_sig.shift(1).fillna(0.0)
    strat_ret = (pos * ret_asset)[is_mask]
    m = calculate_performance_metrics(strat_ret)
    n_trades = int((pos.diff().fillna(0) != 0).sum())

    print(f"\n  [IS Performance MacroYieldCurve - {asset}]")
    print(f"  Cum Ret: {m['cumulative_return']*100:+.1f}% | Sharpe: {m['sharpe_ratio']:.3f} | "
          f"MDD: {m['max_drawdown']*100:.1f}% | Trades: {n_trades}")
    print(f"  Distribuzione segnali: {dict(df_macro_signals['signal'].value_counts())}")

    return best_agent, df_macro_signals


# =====================================================================
# PIPELINE DI TRAINING DELL'ORCHESTRATORE PER SINGOLO ASSET
# =====================================================================
def train_orchestrator_asset(
    asset: str,
    df: pd.DataFrame,
    df_sma: pd.DataFrame,
    df_zscore: pd.DataFrame,
    df_macro: pd.DataFrame,
    config_dir: Path,
    signals_dir: Path,
    results_dir: Path,
    max_train_year: int = 2021,
    objective_metric: str = "profit_oriented",
    max_weight: float = 0.3,
) -> Tuple["OrchestratorAgent", pd.DataFrame]:
    """
    Ottimizza i parametri dell'OrchestratorAgent per un singolo asset e salva i risultati.

    Cosa viene ottimizzato:
    ========================
    L'OrchestratorAgent riceve in input i segnali gia prodotti da 4 agenti SEPARATI e
    INDIPENDENTI — SMA/EMA, Z-Score, MacroYieldCurve e FinBERT Sentiment —
    e deve trovare la combinazione ottima dei seguenti parametri:

    1. PESI DEGLI AGENTI (w_1, ..., w_K):
       - Ogni agente ha il proprio peso indipendente (SMA/EMA, Z-Score, Macro, FinBERT).
       - VINCOLO DI DIVERSIFICAZIONE: ciascun agente puo avere al massimo il 35% del peso (max_weight = 0.35).
         Questo previene la concentrazione eccessiva su un singolo agente (es. FinBERT dormiente).
       - Esplorazione: grid search sul simplesso discreto con step 0.05 + campionamento
         Dirichlet vincolato (w_i <= 0.35 per ogni i).
       - I pesi vengono normalizzati a somma 1: w_k / sum(w).
       - Ogni agente contribuisce con: x_k(t) = signal_k(t) * confidence_k(t) in [-1, 1].
       - Score aggregato: S_raw(t) = sum_k w_k * x_k(t).

    2. SOGLIA DECISIONALE (threshold_buy = threshold_sell):
       - Controlla la selettivita del segnale finale dell'orchestratore.
       - Griglia: da 0.10 a 0.45 con step 0.05.
       - Valori piu reattivi (0.10 - 0.25) permettono di cavalcare i trend ed evitare
         di rimanere flat/cash in modo eccessivamente prudente.

    3. LOGICA DI INTERAZIONE:
       - consensus_boost = 0.20: se gli agenti attivi concordano, lo score viene amplificato
         del 20% per accelerare l'ingresso in posizione con convinzione.
       - conflict_penalty = 0.20: se ci sono segnali discordanti, modera l'esposizione senza
         paralizzare la strategia.
       - min_active_agents = 1: basta che almeno 1 agente sia attivo per generare un segnale.

    La funzione obiettivo 'profit_oriented' premia direttamente il rendimento cumulato e la
    profittabilita, accettando un livello di drawdown fisiologico per cogliere i grandi cicli di mercato.

    I segnali FinBERT Sentiment vengono caricati automaticamente da:
        data/signals/signals_sentiment_{asset}.csv
    Se il file non esiste viene emesso un warning e l'orchestratore gira sui 3 agenti rimanenti.

    Parametri:
        df_sma    : DataFrame (signal, confidence) del SMA/EMA Agent.
        df_zscore : DataFrame (signal, confidence) dello Z-Score Agent.
        df_macro  : DataFrame (signal, confidence) da MacroYieldCurveAgent.
        max_weight: Limite massimo per il peso di ogni agente (default 0.35, ossia 35%).

    Ritorna:
        - L'istanza OrchestratorAgent ottimizzata.
        - Il DataFrame finale dei segnali orchestratore (signal, confidence).
    """
    print("\n" + "#" * 80)
    print(f"### AVVIO TRAINING OrchestratorAgent: {asset} (Max Weight per agente: {max_weight*100:.0f}%)")
    print("#" * 80)

    # Assemblaggio pool: tutti e 4 gli agenti SEPARATI e INDIPENDENTI
    signals_pool: Dict[str, pd.DataFrame] = {
        "SMA_EMA_Agent":    df_sma,
        "ZScore_Agent":     df_zscore,
        "Macro_YieldCurve": df_macro,
    }

    # FinBERT Sentiment — caricato sempre, automaticamente
    sentiment_file = signals_dir / f"signals_sentiment_{asset.lower()}.csv"
    if sentiment_file.exists():
        df_sent = pd.read_csv(sentiment_file, index_col=0, parse_dates=True)
        signals_pool["FinBERT_Sentiment"] = df_sent[["signal", "confidence"]].copy()
        print(f"  [INFO] FinBERT Sentiment caricato: {sentiment_file}")
    else:
        print(f"  [WARN] File FinBERT non trovato ({sentiment_file.name}), "
              f"orchestratore su 3 agenti.")

    # Serie rendimenti dell'asset
    price_col = f"{asset}_close"
    if f"{asset}_return_1w" in df.columns:
        asset_ret = df[f"{asset}_return_1w"]
    else:
        asset_ret = df[price_col].astype(float).pct_change().fillna(0.0)

    agent_names_str = ", ".join(signals_pool.keys())
    print(f"  Agenti nel pool: {agent_names_str}")

    # Ottimizzazione sul campione IS con OrchestratorOptimizer
    optimizer = OrchestratorOptimizer(
        objective_metric=objective_metric,
        max_train_year=max_train_year,
        weight_step=0.05,
        max_weight=max_weight,
        threshold_range=(0.10, 0.45, 0.05),
        consensus_boost=0.20,
        conflict_penalty=0.20,
        min_active_agents=1,
    )
    opt_orchestrator, df_summary = optimizer.optimize(
        signals_dict=signals_pool,
        returns_or_prices=asset_ret,
        asset_name=asset,
    )

    # Salvataggio configurazione JSON
    config_path = config_dir / f"orchestrator_params_{asset.lower()}.json"
    optimizer.save_final_config(config_path)

    # Salvataggio log di tutte le combinazioni testate
    summary_csv = results_dir / f"summary_combinations_orchestrator_{asset.lower()}.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"  Log combinazioni orchestratore salvato in: {summary_csv}")

    # Generazione segnali finali sull'intero dataset con i parametri ottimi
    df_final = opt_orchestrator.combine(signals_pool)

    assert list(df_final.columns) == ["signal", "confidence"], (
        f"Errore formato colonne orchestratore: atteso ['signal', 'confidence'], "
        f"ottenuto {list(df_final.columns)}"
    )

    output_csv = signals_dir / f"signals_orchestrator_{asset.lower()}.csv"
    df_final.to_csv(output_csv)
    print(f"[{asset}] Segnali Orchestratore salvati in: {output_csv}")
    print(f"  Distribuzione segnali: {dict(df_final['signal'].value_counts())}")
    print(f"  Confidenza media (segnali attivi): "
          f"{df_final.loc[df_final['signal'] != 0, 'confidence'].mean():.4f}")

    return opt_orchestrator, df_final


# =====================================================================
# MAIN ORCHESTRATOR — PIPELINE COMPLETA PER TUTTI GLI ASSET
# =====================================================================
def run_all_assets(
    assets: Optional[List[str]] = None,
    max_train_year: int = 2021,
    objective_metric: str = "profit_oriented",
    max_weight: float = 0.35,
    train_macro: bool = True,
    train_orchestrator: bool = True,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Esegue il training Walk-Forward completo per tutti gli asset specificati
    e restituisce un dizionario con tutti i DataFrame di segnali generati.

    I segnali FinBERT Sentiment vengono inclusi nell'orchestratore automaticamente
    se i file data/signals/signals_sentiment_{asset}.csv esistono.

    Ritorna:
        Dict[str, Dict[str, pd.DataFrame]]:
            {
              'SHY': {
                  'sma':          df_sma,
                  'zscore':       df_zscore,
                  'macro':        df_macro,        # se train_macro=True
                  'orchestrator': df_orch,          # se train_orchestrator=True
              },
              'IEF': {...},
              'TLT': {...},
            }
    """
    if assets is None:
        assets = ["SHY", "IEF", "TLT"]

    config_dir = CURRENT_DIR / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    signals_dir = PROJECT_ROOT / "data" / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)

    results_dir = PROJECT_ROOT / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"[Train] Vincolo OOS Lock: training fino al {max_train_year}")
    print(f"[Train] Limite massimo peso per agente nell'Orchestratore: {max_weight*100:.0f}%")
    print(f"[Train] Agenti: SMA/EMA, ZScore"
          + (", MacroYieldCurve" if train_macro else "")
          + (", FinBERT Sentiment (se disponibile)" if train_orchestrator else "")
          + (", Orchestratore" if train_orchestrator else ""))

    all_outputs: Dict[str, Dict[str, pd.DataFrame]] = {}

    for asset in assets:
        all_outputs[asset] = {}

        # --- FASE 1: SMA/EMA + ZScore (Training Walk-Forward separato per ciascuno) ---
        sma_res, zs_res, df_sma, df_zscore = train_single_asset(
            asset=asset,
            df=df,
            config_dir=config_dir,
            signals_dir=signals_dir,
            max_train_year=max_train_year,
            objective_metric=objective_metric,
        )
        all_outputs[asset]["sma"] = df_sma
        all_outputs[asset]["zscore"] = df_zscore

        df_macro = None
        if train_macro:
            # --- FASE 2: MacroYieldCurveAgent ---
            _, df_macro = train_macro_asset(
                asset=asset,
                df=df,
                config_dir=config_dir,
                signals_dir=signals_dir,
                max_train_year=max_train_year,
                objective_metric=objective_metric,
            )
            all_outputs[asset]["macro"] = df_macro

        if train_orchestrator:
            # --- FASE 3: OrchestratorAgent ---
            # SMA/EMA e Z-Score entrano come agenti SEPARATI (pesi indipendenti).
            # Se il macro non e stato addestrato ora, prova a caricare il CSV esistente.
            if df_macro is None:
                macro_csv = signals_dir / f"signals_macro_{asset.lower()}.csv"
                if macro_csv.exists():
                    df_macro = pd.read_csv(macro_csv, index_col=0, parse_dates=True)[["signal", "confidence"]]
                    print(f"  [INFO] Segnali macro caricati da file esistente: {macro_csv}")
                else:
                    print(f"  [WARN] Segnali macro non disponibili per {asset}. "
                          f"Orchestratore su solo SMA/EMA e ZScore.")
                    df_macro = pd.DataFrame(
                        {"signal": 0, "confidence": 0.0}, index=df.index
                    )

            _, df_orch = train_orchestrator_asset(
                asset=asset,
                df=df,
                df_sma=df_sma,
                df_zscore=df_zscore,
                df_macro=df_macro,
                config_dir=config_dir,
                signals_dir=signals_dir,
                results_dir=results_dir,
                max_train_year=max_train_year,
                objective_metric=objective_metric,
                max_weight=max_weight,
            )
            all_outputs[asset]["orchestrator"] = df_orch

    # =====================================================================
    # RIEPILOGO FINALE
    # =====================================================================
    print("\n" + "=" * 85)
    print("                 RIEPILOGO FINALE WALK-FORWARD PER I 3 INDICI")
    print("=" * 85)
    print(f"{'Asset':<6} | {'SMA/EMA Opt':<18} | {'SMA OOS Sh':<10} | {'ZScore Opt':<24} | {'ZS OOS Sh':<10}")
    print("-" * 85)
    # Queste info non sono piu disponibili direttamente, stampiamo solo distribuzione segnali
    for asset in assets:
        print(f"\n  Asset: {asset}")
        for layer, df_s in all_outputs[asset].items():
            dist = dict(df_s["signal"].value_counts())
            print(f"    [{layer:>13}] signal={df_s.shape} distribuzione={dist}")
    print("=" * 85)

    print("\n[OK] Training completato. File salvati:")
    for asset in assets:
        print(f"  - data/signals/signals_sma_ema_{asset.lower()}.csv")
        print(f"  - data/signals/signals_zscore_{asset.lower()}.csv")
        print(f"  - src/agents/config/sma_ema_params_{asset.lower()}.json")
        print(f"  - src/agents/config/zscore_params_{asset.lower()}.json")
        if train_macro:
            print(f"  - data/signals/signals_macro_{asset.lower()}.csv")
            print(f"  - src/agents/config/macro_yield_params_{asset.lower()}.json")
        if train_orchestrator:
            print(f"  - data/signals/signals_orchestrator_{asset.lower()}.csv")
            print(f"  - src/agents/config/orchestrator_params_{asset.lower()}.json")

    return all_outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Training e ottimizzazione Walk-Forward di tutti gli agenti di trading."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="ALL",
        help="Asset da ottimizzare: SHY, IEF, TLT o ALL (default: ALL).",
    )
    parser.add_argument(
        "--metric",
        default="profit_oriented",
        choices=["profit_oriented", "profit", "sharpe", "sharpe_mdd", "calmar", "sortino", "return"],
        help="Metrica obiettivo per l'ottimizzazione (default: profit_oriented).",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.35,
        help="Limite massimo per il peso di ciascun agente nell'Orchestratore (default: 0.35, ossia 35%%).",
    )
    parser.add_argument(
        "--no-macro",
        action="store_true",
        help="Salta il training del MacroYieldCurveAgent.",
    )
    parser.add_argument(
        "--no-orchestrator",
        action="store_true",
        help="Salta il training dell'OrchestratorAgent.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2021,
        help="Anno massimo per il campione In-Sample (OOS lock, default: 2021).",
    )

    args = parser.parse_args()

    target_assets = ["SHY", "IEF", "TLT"] if args.target.upper() == "ALL" else [args.target.upper()]

    run_all_assets(
        assets=target_assets,
        max_train_year=args.year,
        objective_metric=args.metric,
        max_weight=args.max_weight,
        train_macro=not args.no_macro,
        train_orchestrator=not args.no_orchestrator,
    )
