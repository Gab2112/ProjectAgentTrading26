"""
evaluator.py 

Valutazione del baseline Macro / Rates Regimes.

Questo modulo misura le performance della strategia e dei benchmark 
mantenendo separati:

 1. TRAIN;
 2. VALIDATION;
 3. OUT-OF-SAMPLE (OOS).

L'OOS è volutamente disattivato di default. Finchè il modello non
viene congelato, il progetto deve evitare di utilizzare l'OOS per
scegliere feature, soglie, modelli o parametri.

Metriche principali:
 - cumulative return;
 - annualized return (CAGR);
 - annualized volativility;
 - Sharpe ratio;
 - maximum drawdown (MDD).
 - Profit Factor.

Per lo Sharpe utilizziamo come proxy del risk-free il Treasury 3M disponibile 
nel dataset.  Il rendimento settimanale risk-free viene ricavato dal rendimento
annualizzato osservato nella stessa settimana.

Il benchmark principale è un vero equal-weighted BUY & HOLD: all'inizio di ciascun
periodo di valutazione assegniamo 1/3 a  SHY, IEF e TLT e lasciamo poi i pesi 
evolvere naturalmente con i prezzi.
 """

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.data_loader import load_dataset
from src.utils import get_tickers
from src.time_split import (
    get_time_boundaries,
    is_oos_enabled,
    split_time_periods,
)

# =============================================================================
# Lettura del backtest
# =============================================================================

def load_backtest_dataset(config: dict, root_dir: Path) -> pd.DataFrame:
    """Carica il dataset prodotto da backtester.py."""
    path = (
        root_dir
        / config["paths"]["results"]
        / "backtest_weekly.csv"
    )

    return load_dataset(path)


# =============================================================================
# Configurazione della valutazione
# =============================================================================

def _get_evaluation_parameters(config: dict) -> dict:
    """
    Recupera i parametri di valutazione.

    I confini temporali e il blocco OOS sono gestiti
    esclusivamente dal modulo time_split.py.
    """
    train_end, validation_end, oos_start = get_time_boundaries(config)

    annualization_factor = int(
        config.get("evaluation", {}).get("annualization_factor", 52)
    )

    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive.")

    return {
        "train_end": train_end,
        "validation_end": validation_end,
        "oos_start": oos_start,
        "annualization_factor": annualization_factor,
        "evaluate_oos": is_oos_enabled(config),
    }

# =============================================================================
# Individuazione dell'inizio effettivo della strategia
# =============================================================================

def get_first_active_week(
    backtest: pd.DataFrame,
    tickers: list[str],
) -> pd.Timestamp:
    """
    Restituisce la prima settimana in cui il portafoglio ha una posizione attiva.

    Le settimane precedenti vengono escluse dalla valutazione: la strategia
    non esisteva ancora perché le feature/regimi richiedevano storico minimo.
    """
    applied_columns = [
        f"{ticker}_applied_weight"
        for ticker in tickers
    ]

    active = (
        backtest[applied_columns]
        .abs()
        .sum(axis=1)
        > 0
    )

    if not active.any():
        raise ValueError("No active strategy weeks found.")

    return backtest.index[active][0]


# =============================================================================
# Split cronologico
# =============================================================================

def split_evaluation_periods(
    backtest: pd.DataFrame,
    config: dict,
) -> Dict[str, pd.DataFrame]:
    """
    Prepara i dati della strategia per la valutazione.

    Esclude le settimane precedenti alla prima posizione
    effettivamente investita e delega lo split temporale
    al modulo centralizzato.
    """

    first_active = get_first_active_week(
        backtest,
        get_tickers(config),
    )

    data = backtest.loc[
        backtest.index >= first_active
    ].copy()

    return split_time_periods(data, config)


# =============================================================================
# Risk-free rate
# =============================================================================

def build_weekly_risk_free(
    period: pd.DataFrame,
    annualization_factor: int,
) -> pd.Series:
    """
    Trasforma il Treasury 3M annualizzato in una proxy di rendimento settimanale.

    Esempio concettuale:
        rendimento annuale 5%
        -> rendimento equivalente per una settimana.

    Il Treasury 3M è una proxy pratica del tasso privo di rischio, non una
    misura perfetta. L'assunzione verrà documentata nella metodologia finale.
    """
    source_column = "rates__treasury_3m"

    if source_column not in period.columns:
        raise KeyError(
            f"Missing risk-free proxy: {source_column}"
        )

    annual_rate = (
        period[source_column]
        .ffill()
        / 100.0
    )

    weekly_rate = (
        (1.0 + annual_rate)
        ** (1.0 / annualization_factor)
        - 1.0
    )

    return weekly_rate


# =============================================================================
# Benchmark equal-weight buy-and-hold
# =============================================================================

def build_buy_and_hold_benchmark(
    period: pd.DataFrame,
    tickers: list[str],
) -> pd.Series:
    """
    Costruisce un vero portafoglio equal-weight buy-and-hold.

    All'inizio del periodo:
        SHY = 1/3
        IEF = 1/3
        TLT = 1/3

    In seguito non viene effettuato alcun ribilanciamento: ciascuna componente
    cresce secondo il proprio rendimento cumulato e i pesi possono quindi
    allontanarsi da 1/3.
    """
    return_columns = [
        f"{ticker}_return_1w"
        for ticker in tickers
    ]

    missing = [
        column
        for column in return_columns
        if column not in period.columns
    ]

    if missing:
        raise KeyError(
            "Missing benchmark columns: "
            + ", ".join(missing)
        )

    asset_wealth = (
        1.0
        + period[return_columns].fillna(0.0)
    ).cumprod()

    benchmark_wealth = (
        asset_wealth.mean(axis=1)
    )

    benchmark_return = (
        benchmark_wealth
        .pct_change(fill_method=None)
        .fillna(benchmark_wealth.iloc[0] - 1.0)
    )

    return benchmark_return


# =============================================================================
# Metriche
# =============================================================================

def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calcola il maximum drawdown della curva cumulata."""
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0

    return float(drawdown.min())


def calculate_profit_factor(returns: pd.Series) -> float:
    """
    Profit Factor = somma rendimenti positivi / valore assoluto delle perdite.

    È una versione period-return del Profit Factor, coerente con il backtest
    settimanale del progetto.
    """
    returns = returns.dropna()

    gains = returns.loc[returns > 0].sum()
    losses = returns.loc[returns < 0].sum()

    if losses == 0:
        return float("nan")

    return float(gains / abs(losses))


def calculate_metrics(
    returns: pd.Series,
    risk_free: pd.Series,
    annualization_factor: int,
) -> dict:
    """Calcola le principali metriche richieste per una serie di rendimenti."""
    aligned = pd.concat(
        [
            returns.rename("returns"),
            risk_free.rename("risk_free"),
        ],
        axis=1,
    ).dropna()

    if aligned.empty:
        raise ValueError("No observations available for metric calculation.")

    r = aligned["returns"]
    rf = aligned["risk_free"]

    observations = len(r)

    cumulative_return = (
        (1.0 + r).prod() - 1.0
    )

    annualized_return = (
        (1.0 + cumulative_return)
        ** (annualization_factor / observations)
        - 1.0
    )

    annualized_volatility = (
        r.std(ddof=1)
        * np.sqrt(annualization_factor)
    )

    excess_return = r - rf

    excess_volatility = excess_return.std(ddof=1)

    if excess_volatility == 0 or np.isnan(excess_volatility):
        sharpe_ratio = float("nan")
    else:
        sharpe_ratio = float(
            excess_return.mean()
            / excess_volatility
            * np.sqrt(annualization_factor)
        )

    return {
        "observations": observations,
        "cumulative_return": float(cumulative_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": calculate_max_drawdown(r),
        "profit_factor": calculate_profit_factor(r),
    }


# =============================================================================
# Valutazione di un singolo periodo
# =============================================================================

def evaluate_period(
    period_name: str,
    period: pd.DataFrame,
    annualization_factor: int,
    tickers: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Valuta strategia e benchmark all'interno di un singolo periodo.
    """
    risk_free = build_weekly_risk_free(
        period,
        annualization_factor,
    )

    buy_hold_return = build_buy_and_hold_benchmark(period, tickers)

    series_map = {
        "strategy": period["strategy_net_return"],
        "equal_weight_buy_and_hold": buy_hold_return,
    }

    for ticker in tickers:
        series_map[f"{ticker}_buy_and_hold"] = period[f"benchmark_{ticker}_return"]

    metric_rows = []

    curve_data = pd.DataFrame(
        index=period.index
    )

    curve_data["risk_free_weekly"] = risk_free

    for name, returns in series_map.items():
        metrics = calculate_metrics(
            returns=returns,
            risk_free=risk_free,
            annualization_factor=annualization_factor,
        )

        metric_rows.append(
            {
                "period": period_name,
                "portfolio": name,
                "start_date": period.index.min(),
                "end_date": period.index.max(),
                **metrics,
            }
        )

        curve_data[f"{name}_index_100"] = (
            100.0
            * (1.0 + returns.fillna(0.0))
            .cumprod()
        )

    curve_data["period"] = period_name

    metrics_df = pd.DataFrame(metric_rows)

    return metrics_df, curve_data


# =============================================================================
# Pipeline di valutazione
# =============================================================================

def run_evaluation(
    config: dict,
    root_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Esegue la valutazione TRAIN / VALIDATION e, solo quando esplicitamente
    autorizzato nel config, anche OOS.

    Output:
        data/results/evaluation_metrics.csv
        data/results/evaluation_curves.csv
    """
    results_dir = root_dir / config["paths"]["results"]
    results_dir.mkdir(parents=True, exist_ok=True)

    backtest = load_backtest_dataset(config, root_dir)
    params = _get_evaluation_parameters(config)

    periods = split_evaluation_periods(
        backtest,
        config,
    )

    metric_frames = []
    curve_frames = []

    for period_name, period in periods.items():
        metrics, curves = evaluate_period(
            period_name=period_name,
            period=period,
            annualization_factor=params["annualization_factor"],
            tickers=get_tickers(config),
        )

        metric_frames.append(metrics)
        curve_frames.append(curves)

    metrics_df = pd.concat(
        metric_frames,
        ignore_index=True,
    )

    curves_df = pd.concat(
        curve_frames,
        axis=0,
    )

    metrics_path = (
        results_dir
        / "evaluation_metrics.csv"
    )

    curves_path = (
        results_dir
        / "evaluation_curves.csv"
    )

    metrics_df.to_csv(metrics_path, index=False)
    curves_df.to_csv(curves_path)

    print("\n[Evaluator] Development evaluation completed successfully")
    print(f"  Metrics:       {metrics_path}")
    print(f"  Curves:        {curves_path}")
    print(
        "  OOS evaluated: "
        f"{params['evaluate_oos']}"
    )

    print("\n  Periods evaluated:")

    for period_name, period in periods.items():
        print(
            f"    {period_name:<10} "
            f"{period.index.min().date()} -> "
            f"{period.index.max().date()} "
            f"({len(period):,} weeks)"
        )

    if not params["evaluate_oos"]:
        print(
            "\n  OOS remains locked. "
            "Do not inspect it until the model is frozen."
        )

    return metrics_df, curves_df
