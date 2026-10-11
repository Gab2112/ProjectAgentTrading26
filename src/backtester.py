"""
backtester.py 

Backtest settimanale della baseline Macro / Rates Regimes.

Questo modulo applica i target weights prodotti dal portafoglio.py ai
rendimenti degli asset SHY, IEF e TLT.

Principio temporale fondamentale da rispettare:
 - il portafoglio deciso alla settimana t utilizza info disponibili a t;
 - quel portafoglio NON può guadagnare il rendimento già osservato nella stessa settimana;
 - i pesi vengono quindi applicati dal periodo successivo tramite shift(1).

 In questo modo:

 info disponibili a t -> target weight a t -> esecuzione nel periodo t+1 
 -> rendimento realizzato in t+1

Il modulo supporta inoltre costi di transazione e slittamento di prezzo 
(come richiesto dalle istruzioni) configurabili in punti base.
Se nel config.yaml tali valori sono nulll, la baseline usa 0 bps.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd


# =============================================================================
# Lettura dei pesi di portafoglio
# =============================================================================

def load_portfolio_dataset(config: dict, root_dir: Path) -> pd.DataFrame:
    """Carica il dataset settimanale prodotto da portfolio.py."""
    path = (
        root_dir
        / config["paths"]["signals"]
        / "portfolio_weights_weekly.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Portfolio dataset not found: {path}"
        )

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()

    return df


# =============================================================================
# Parametri di esecuzione
# =============================================================================

def _get_execution_parameters(config: dict) -> dict:
    """
    Recupera i parametri di esecuzione dal config.yaml.

    Se commissioni o slippage sono null, vengono interpretati come 0 bps.
    """
    execution_config = config.get("execution", {})

    initial_capital = float(
        execution_config.get("initial_capital", 100000.0)
    )

    commission_bps = execution_config.get("commission_bps")
    slippage_bps = execution_config.get("slippage_bps")

    commission_bps = (
        0.0 if commission_bps is None else float(commission_bps)
    )

    slippage_bps = (
        0.0 if slippage_bps is None else float(slippage_bps)
    )

    execution_rule = (
        execution_config.get("execution_rule")
        or "next_available_period"
    )

    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")

    if commission_bps < 0 or slippage_bps < 0:
        raise ValueError(
            "commission_bps and slippage_bps cannot be negative."
        )

    if execution_rule != "next_available_period":
        raise ValueError(
            "Unsupported execution_rule: "
            f"{execution_rule}"
        )

    return {
        "initial_capital": initial_capital,
        "commission_bps": commission_bps,
        "slippage_bps": slippage_bps,
        "execution_rule": execution_rule,
    }


# =============================================================================
# Applicazione temporale dei pesi
# =============================================================================

def apply_execution_lag(portfolio: pd.DataFrame) -> pd.DataFrame:
    """
    Sposta i target weights di una settimana in avanti.

    I target weights calcolati alla data t diventano quindi le posizioni
    effettivamente applicate al rendimento tra t e t+1.

    Questo passaggio è essenziale per evitare look-ahead bias.
    """
    backtest = portfolio.copy()

    for asset in ["SHY", "IEF", "TLT"]:
        target_column = f"{asset}_weight"
        applied_column = f"{asset}_applied_weight"

        if target_column not in backtest.columns:
            raise KeyError(
                f"Missing portfolio column: {target_column}"
            )

        backtest[applied_column] = (
            backtest[target_column]
            .shift(1)
            .fillna(0.0)
        )

    return backtest


# =============================================================================
# Calcolo dei rendimenti del portafoglio
# =============================================================================

def calculate_strategy_returns(
    backtest: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Calcola rendimento lordo, turnover, costi e rendimento netto.

    Rendimento lordo:
        somma(weight_i * return_i)

    Turnover:
        somma delle variazioni assolute dei pesi applicati.

    Costi:
        turnover * costo totale in basis points.

    Rendimento netto:
        rendimento lordo - costi.
    """
    result = backtest.copy()
    params = _get_execution_parameters(config)

    asset_returns = []
    applied_weights = []

    for asset in ["SHY", "IEF", "TLT"]:
        return_column = f"{asset}_return_1w"
        weight_column = f"{asset}_applied_weight"

        if return_column not in result.columns:
            raise KeyError(
                f"Missing weekly return column: {return_column}"
            )

        asset_returns.append(return_column)
        applied_weights.append(weight_column)

    # -------------------------------------------------------------------------
    # Rendimento lordo della strategia.
    # -------------------------------------------------------------------------
    result["strategy_gross_return"] = 0.0

    for return_column, weight_column in zip(
        asset_returns,
        applied_weights,
    ):
        result["strategy_gross_return"] += (
            result[weight_column]
            * result[return_column].fillna(0.0)
        )

    # -------------------------------------------------------------------------
    # Turnover del portafoglio.
    # -------------------------------------------------------------------------
    # Misuriamo quanto cambiano le posizioni rispetto alla settimana precedente.
    # Esempio: passare da TLT +1 a IEF +1 implica turnover pari a 2.
    result["turnover"] = 0.0

    for weight_column in applied_weights:
        result["turnover"] += (
            result[weight_column]
            .diff()
            .abs()
            .fillna(result[weight_column].abs())
        )

    # -------------------------------------------------------------------------
    # Costi di transazione.
    # -------------------------------------------------------------------------
    total_cost_bps = (
        params["commission_bps"]
        + params["slippage_bps"]
    )

    result["transaction_cost"] = (
        result["turnover"]
        * total_cost_bps
        / 10000.0
    )

    result["strategy_net_return"] = (
        result["strategy_gross_return"]
        - result["transaction_cost"]
    )

    return result


# =============================================================================
# Benchmark
# =============================================================================

def add_benchmarks(backtest: pd.DataFrame) -> pd.DataFrame:
    """
    Costruisce benchmark semplici e trasparenti.

    1. Equal-weight buy-and-hold di SHY, IEF e TLT.
    2. Rendimenti buy-and-hold individuali dei tre ETF.

    Il benchmark equal-weight assegna un terzo del capitale a ciascun ETF.
    """
    result = backtest.copy()

    required_returns = [
        "SHY_return_1w",
        "IEF_return_1w",
        "TLT_return_1w",
    ]

    missing = [
        column
        for column in required_returns
        if column not in result.columns
    ]

    if missing:
        raise KeyError(
            f"Missing benchmark return columns: {', '.join(missing)}"
        )

    result["benchmark_equal_weight_return"] = (
        result[required_returns]
        .mean(axis=1)
    )

    result["benchmark_SHY_return"] = result["SHY_return_1w"]
    result["benchmark_IEF_return"] = result["IEF_return_1w"]
    result["benchmark_TLT_return"] = result["TLT_return_1w"]

    return result


# =============================================================================
# Equity curves
# =============================================================================

def add_equity_curves(
    backtest: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Costruisce equity curve monetarie e indici normalizzati a 100.

    Le curve complete vengono salvate qui; l'evaluator deciderà in seguito
    quale intervallo utilizzare per IS / validation / OOS.
    """
    result = backtest.copy()
    params = _get_execution_parameters(config)

    initial_capital = params["initial_capital"]

    result["strategy_equity"] = (
        initial_capital
        * (1.0 + result["strategy_net_return"])
        .cumprod()
    )

    result["benchmark_equal_weight_equity"] = (
        initial_capital
        * (1.0 + result["benchmark_equal_weight_return"].fillna(0.0))
        .cumprod()
    )

    # Indici base 100, utili per confrontare visivamente strategia e benchmark.
    result["strategy_index_100"] = (
        100.0
        * (1.0 + result["strategy_net_return"])
        .cumprod()
    )

    result["benchmark_equal_weight_index_100"] = (
        100.0
        * (1.0 + result["benchmark_equal_weight_return"].fillna(0.0))
        .cumprod()
    )

    return result


# =============================================================================
# Controlli di coerenza
# =============================================================================

def validate_backtest(backtest: pd.DataFrame) -> None:
    """
    Esegue alcuni controlli minimi prima di salvare i risultati.
    """
    required_columns = [
        "strategy_gross_return",
        "strategy_net_return",
        "turnover",
        "transaction_cost",
        "strategy_equity",
    ]

    for column in required_columns:
        if column not in backtest.columns:
            raise KeyError(
                f"Backtest output missing column: {column}"
            )

    # Una perdita settimanale inferiore a -100% sarebbe impossibile
    # con il livello di leva attuale e segnala quasi certamente un errore.
    if (backtest["strategy_net_return"] <= -1.0).any():
        raise ValueError(
            "Strategy return <= -100% detected."
        )

    if (backtest["strategy_equity"] <= 0).any():
        raise ValueError(
            "Non-positive strategy equity detected."
        )


# =============================================================================
# Pipeline del backtest
# =============================================================================

def run_backtest(
    config: dict,
    root_dir: Path,
) -> Tuple[pd.DataFrame, dict]:
    """
    Esegue il backtest completo del baseline e salva l'output.

    Output:
        data/results/backtest_weekly.csv

    Le metriche di performance vere e proprie verranno calcolate
    separatamente in evaluator.py.
    """
    results_dir = root_dir / config["paths"]["results"]
    results_dir.mkdir(parents=True, exist_ok=True)

    portfolio = load_portfolio_dataset(config, root_dir)

    backtest = apply_execution_lag(portfolio)
    backtest = calculate_strategy_returns(backtest, config)
    backtest = add_benchmarks(backtest)
    backtest = add_equity_curves(backtest, config)

    validate_backtest(backtest)

    output_path = results_dir / "backtest_weekly.csv"
    backtest.to_csv(output_path)

    params = _get_execution_parameters(config)

    # La prima settimana realmente investita è quella in cui almeno un peso
    # applicato è diverso da zero.
    applied_columns = [
        "SHY_applied_weight",
        "IEF_applied_weight",
        "TLT_applied_weight",
    ]

    active = (
        backtest[applied_columns]
        .abs()
        .sum(axis=1)
        > 0
    )

    first_active = (
        backtest.index[active][0]
        if active.any()
        else None
    )

    summary = {
        "rows": len(backtest),
        "first_active_week": first_active,
        "final_equity": float(
            backtest["strategy_equity"].iloc[-1]
        ),
        "total_turnover": float(
            backtest["turnover"].sum()
        ),
        "commission_bps": params["commission_bps"],
        "slippage_bps": params["slippage_bps"],
    }

    print("\n[Backtester] Baseline completed successfully")
    print(f"  Dataset:          {output_path}")
    print(f"  Rows:             {len(backtest):,}")

    if first_active is not None:
        print(
            "  First active week: "
            f"{first_active.date()}"
        )

    print(
        "  Final equity:      "
        f"{summary['final_equity']:,.2f}"
    )
    print(
        "  Total turnover:    "
        f"{summary['total_turnover']:.2f}"
    )
    print(
        "  Trading costs:     "
        f"{params['commission_bps'] + params['slippage_bps']:.2f} bps"
    )

    return backtest, summary
