"""Evaluation of the multi-asset strategies A1 and A2.

The module reads the daily backtests produced by ``backtester.py``.
It does not generate signals, rebuild benchmarks or simulate trades.

Conventions:
- evaluation on the training period only, without validation or OOS data;
- daily returns net of transaction costs;
- annualisation over 252 sessions (configurable);
- constant annual risk-free rate, zero by default;
- CAGR, Sharpe, Sortino, drawdown, turnover and costs for each strategy.

The OOS test is never opened in this module.

Public interface:
    run_evaluation(config, root_dir, level=None, backtests=None)

Output per level:
    data/results/A1/evaluation_metrics.csv
    data/results/A1/evaluation_curves.csv
    data/results/A2/evaluation_metrics.csv
    data/results/A2/evaluation_curves.csv

Each strategy also receives its own summary and equity curve.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict

import numpy as np
import pandas as pd

from src.data_loader import load_dataset
from src.time_split import get_time_windows, is_oos_enabled


LIVELLI_IMPLEMENTATI = frozenset({"A1", "A2"})
NOME_SICURO = re.compile(r"^[A-Za-z0-9_-]+$")
COLONNE_OBBLIGATORIE = (
    "nav",
    "rendimento_netto",
    "turnover",
    "costo_transazione",
    "operazione_eseguita",
)


# =============================================================================
# Configuration and level selection
# =============================================================================


def _seleziona_livello(config: dict, level: str | None) -> tuple[str, list[str]]:
    """Select only the strategies that are implemented and enabled in the YAML file."""
    selezionato = str(
        level if level is not None
        else config.get("pipeline", {}).get("active_level", "A1")
    ).upper()

    if selezionato not in LIVELLI_IMPLEMENTATI:
        raise NotImplementedError(
            f"Valutazione del livello {selezionato} non implementata. "
            "Sono disponibili soltanto A1 e A2."
        )

    strategie = config.get("strategies", {})
    impostazioni = strategie.get(selezionato, {})
    if not isinstance(impostazioni, dict) or impostazioni.get("enabled") is not True:
        raise ValueError(f"Il livello {selezionato} non e' abilitato nel config.yaml.")

    if config.get("pipeline", {}).get("legacy_treasury_enabled") is True:
        raise ValueError("La pipeline Treasury precedente non deve essere abilitata.")

    if selezionato == "A1":
        benchmark = config.get("benchmarks", {})
        if not isinstance(benchmark, dict) or not benchmark:
            raise ValueError("Mancano i benchmark A1 nel config.yaml.")
        nomi = impostazioni.get("benchmark_names", list(benchmark))
        if not isinstance(nomi, list) or not nomi:
            raise ValueError("La lista dei benchmark A1 non e' valida.")
        if any(nome not in benchmark for nome in nomi):
            raise KeyError("Un benchmark A1 non e' definito nel config.yaml.")
    else:
        nomi = ["volatility_control"]

    if len(nomi) != len(set(nomi)):
        raise ValueError("I nomi delle strategie non possono essere duplicati.")
    if any(not isinstance(nome, str) or not NOME_SICURO.fullmatch(nome) for nome in nomi):
        raise ValueError("I nomi delle strategie non sono validi.")

    return selezionato, list(nomi)


def _parametri_valutazione(config: dict) -> tuple[int, float, pd.Timestamp, pd.Timestamp, float]:
    """Read the evaluation parameters and enforce the use of training data only."""
    impostazioni = config.get("evaluation", {})
    if not isinstance(impostazioni, dict):
        raise TypeError("La sezione 'evaluation' deve essere un dizionario.")

    if is_oos_enabled(config):
        raise ValueError("Accesso OOS non autorizzato durante la valutazione A1/A2.")

    finestre = get_time_windows(config)
    inizio, fine = finestre["train_start"], finestre["train_end"]
    if inizio is None:
        raise ValueError("Impostare 'validation.train_start' per il progetto Multi-Asset.")

    fattore = impostazioni.get("annualization_factor", 252)
    if isinstance(fattore, bool) or not isinstance(fattore, (int, np.integer)):
        raise ValueError("annualization_factor deve essere un intero positivo.")
    if fattore <= 1:
        raise ValueError("annualization_factor deve essere maggiore di uno.")

    # Backtester returns are daily in this version.
    if impostazioni.get("returns_frequency", "daily") != "daily":
        raise ValueError("Il backtester Multi-Asset produce rendimenti giornalieri.")

    tasso_annuo = impostazioni.get(
        "risk_free_rate_annual", impostazioni.get("risk_free_rate", 0.0)
    )
    try:
        tasso_annuo = float(tasso_annuo)
    except (TypeError, ValueError) as exc:
        raise ValueError("Il tasso risk-free deve essere numerico.") from exc
    if not np.isfinite(tasso_annuo) or tasso_annuo <= -1:
        raise ValueError("Il tasso risk-free annuo non e' valido.")

    capitale = config.get("execution", {}).get("initial_capital", 100000.0)
    try:
        capitale = float(capitale)
    except (TypeError, ValueError) as exc:
        raise ValueError("Il capitale iniziale deve essere numerico.") from exc
    if not np.isfinite(capitale) or capitale <= 0:
        raise ValueError("Il capitale iniziale deve essere positivo e finito.")

    return int(fattore), tasso_annuo, inizio, fine, capitale


# =============================================================================
# Loading and validation of backtest results
# =============================================================================


def load_backtest_dataset(
    config: dict,
    root_dir: Path,
    level: str,
    portfolio: str,
) -> pd.DataFrame:
    """Load the backtest of a single strategy."""
    path = (
        Path(root_dir)
        / config["paths"]["results"]
        / level
        / portfolio
        / "backtest_daily.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Backtest non trovato: {path}. "
            f"Eseguire prima run_backtest(..., level='{level}')."
        )
    return load_dataset(path, strict=True)


def _convalida_backtest(
    backtest: pd.DataFrame,
    inizio: pd.Timestamp,
    fine: pd.Timestamp,
    capitale_iniziale: float,
    nome: str,
) -> pd.DataFrame:
    """Validate inputs, calendar, returns and NAV consistency.

    The date check precedes any computation: no observation after the
    training period may enter the evaluation.
    """
    if not isinstance(backtest, pd.DataFrame) or backtest.empty:
        raise ValueError(f"{nome}: backtest assente o vuoto.")

    indice = backtest.index
    if not isinstance(indice, pd.DatetimeIndex):
        raise TypeError(f"{nome}: e' richiesto un DatetimeIndex.")
    if indice.hasnans or indice.tz is not None or indice.has_duplicates:
        raise ValueError(f"{nome}: date mancanti, duplicate o con fuso orario.")
    if not indice.is_monotonic_increasing:
        raise ValueError(f"{nome}: date non ordinate.")
    if indice.min() < inizio or indice.max() > fine:
        raise ValueError(
            f"{nome}: il backtest contiene date esterne al training. "
            "La validation e l'OOS non sono autorizzati in questa fase."
        )

    mancanti = [colonna for colonna in COLONNE_OBBLIGATORIE if colonna not in backtest]
    if mancanti:
        raise KeyError(f"{nome}: colonne mancanti: {', '.join(mancanti)}")
    if len(backtest) < 2:
        raise ValueError(f"{nome}: servono almeno due sedute per le metriche.")

    risultato = backtest.copy(deep=True)
    for colonna in COLONNE_OBBLIGATORIE[:-1]:
        risultato[colonna] = pd.to_numeric(risultato[colonna], errors="raise")
        if not np.isfinite(risultato[colonna].to_numpy(dtype=float)).all():
            raise ValueError(f"{nome}: valori non finiti in '{colonna}'.")

    eseguite = risultato["operazione_eseguita"]
    if not eseguite.isin([True, False]).all():
        raise ValueError(f"{nome}: 'operazione_eseguita' deve essere booleano.")
    risultato["operazione_eseguita"] = eseguite.astype(bool)

    if (risultato["nav"] <= 0).any():
        raise ValueError(f"{nome}: NAV non positivo.")
    if (risultato["rendimento_netto"] <= -1).any():
        raise ValueError(f"{nome}: rendimento netto minore o uguale a -100%.")
    if (risultato["turnover"] < 0).any() or (risultato["costo_transazione"] < 0).any():
        raise ValueError(f"{nome}: turnover o costi negativi.")
    if not risultato["operazione_eseguita"].any():
        raise ValueError(f"{nome}: nessuna operazione eseguita.")

    # The first row compares the NAV with the initial capital;
    # subsequent rows use the NAV of the previous session.
    nav_precedente = risultato["nav"].shift(1)
    nav_precedente.iloc[0] = capitale_iniziale
    rendimenti_attesi = risultato["nav"] / nav_precedente - 1.0
    if not np.allclose(
        rendimenti_attesi.to_numpy(dtype=float),
        risultato["rendimento_netto"].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-9,
    ):
        raise ValueError(f"{nome}: i rendimenti netti non sono coerenti con il NAV.")

    return risultato


# =============================================================================
# Financial performance metrics
# =============================================================================


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Compute the maximum drawdown, including the initial wealth."""
    r = pd.to_numeric(returns, errors="raise").dropna().to_numpy(dtype=float)
    if not len(r) or not np.isfinite(r).all() or np.any(r <= -1):
        raise ValueError("Serie dei rendimenti non valida per il maximum drawdown.")

    ricchezza = np.concatenate(([1.0], np.cumprod(1.0 + r)))
    massimo_precedente = np.maximum.accumulate(ricchezza)
    return float(np.min(ricchezza / massimo_precedente - 1.0))


def calculate_profit_factor(returns: pd.Series) -> float:
    """Ratio between positive and absolute negative daily returns.

    This measure is based on PERIODIC returns, not on the profit factor of
    individual trades, and must not be read as a count of profitable trades.
    """
    r = pd.to_numeric(returns, errors="raise").dropna().to_numpy(dtype=float)
    if not len(r) or not np.isfinite(r).all():
        raise ValueError("Serie dei rendimenti non valida per il profit factor.")
    profitti = r[r > 0].sum()
    perdite = -r[r < 0].sum()
    return float(profitti / perdite) if perdite > 0 else float("nan")


def calculate_metrics(
    returns: pd.Series,
    risk_free: pd.Series | float,
    annualization_factor: int,
) -> dict:
    """Compute metrics on daily returns net of transaction costs.

    CAGR: (final wealth / initial wealth) ** (252 / n) - 1.
    Sharpe: mean excess return / standard deviation (ddof=1).
    Sortino: mean excess return / downside deviation relative to the risk-free rate.
    Volatility is the annualised sample standard deviation.
    """
    r = pd.to_numeric(returns, errors="raise").astype(float)
    if isinstance(risk_free, pd.Series):
        rf = pd.to_numeric(risk_free.reindex(r.index), errors="raise").astype(float)
    else:
        rf = pd.Series(float(risk_free), index=r.index, dtype=float)

    if not r.index.equals(rf.index) or r.isna().any() or rf.isna().any():
        raise ValueError("Rendimenti e risk-free devono avere le stesse date, senza NaN.")
    rv = r.to_numpy(dtype=float)
    rfv = rf.to_numpy(dtype=float)
    if len(rv) < 2 or not np.isfinite(rv).all() or not np.isfinite(rfv).all():
        raise ValueError("Sono necessari almeno due rendimenti e risk-free finiti.")
    if np.any(rv <= -1):
        raise ValueError("Rendimenti inferiori o uguali a -100% non ammessi.")
    if annualization_factor <= 1:
        raise ValueError("Fattore di annualizzazione non valido.")

    numero_sedute = len(rv)
    ricchezza_finale = float(np.prod(1.0 + rv))
    rendimento_cumulato = ricchezza_finale - 1.0
    cagr = ricchezza_finale ** (annualization_factor / numero_sedute) - 1.0
    volatilita = float(np.std(rv, ddof=1) * np.sqrt(annualization_factor))

    excess = rv - rfv
    std_excess = float(np.std(excess, ddof=1))
    sharpe = (
        float(np.mean(excess) / std_excess * np.sqrt(annualization_factor))
        if std_excess > 0 else float("nan")
    )

    # Positive observations also enter the Sortino denominator with a zero
    # contribution: this is the downside deviation relative to the risk-free target.
    downside = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)))
    sortino = (
        float(np.mean(excess) / downside * np.sqrt(annualization_factor))
        if downside > 0 else float("nan")
    )

    return {
        "observations": int(numero_sedute),
        "cumulative_return": float(rendimento_cumulato),
        "cagr": float(cagr),
        "annualized_volatility": volatilita,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": calculate_max_drawdown(r),
        "profit_factor_daily_returns": calculate_profit_factor(r),
    }


def evaluate_period(
    period_name: str,
    period: pd.DataFrame,
    annualization_factor: int,
    risk_free_annual: float = 0.0,
    initial_capital: float = 100000.0,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate a daily backtest and build its equity curve.

    The period is always the training period in the current implementation.
    Each strategy is evaluated on its actual trades, without building
    alternative theoretical prices.
    """
    if period_name != "train":
        raise NotImplementedError("La valutazione e' attualmente limitata al training.")

    tasso_giornaliero = (1.0 + risk_free_annual) ** (1.0 / annualization_factor) - 1.0
    rf = pd.Series(tasso_giornaliero, index=period.index)
    metriche = calculate_metrics(period["rendimento_netto"], rf, annualization_factor)

    # The NAV includes costs and the weights actually held at each close.
    nav = period["nav"].to_numpy(dtype=float)
    massimo = np.maximum.accumulate(np.concatenate(([initial_capital], nav)))[1:]
    drawdown = nav / massimo - 1.0
    curve = pd.DataFrame({
        "nav": period["nav"],
        "indice_base_100": 100.0 * period["nav"] / initial_capital,
        "drawdown": drawdown,
        "rendimento_netto": period["rendimento_netto"],
        "turnover": period["turnover"],
        "costo_transazione": period["costo_transazione"],
        "operazione_eseguita": period["operazione_eseguita"],
    }, index=period.index)

    operazioni = period["operazione_eseguita"]
    metriche.update({
        "initial_capital": float(initial_capital),
        "final_nav": float(nav[-1]),
        "total_turnover": float(period["turnover"].sum()),
        "total_transaction_costs": float(period["costo_transazione"].sum()),
        "executed_rebalances": int(operazioni.sum()),
        "first_execution_date": period.index[operazioni][0].date().isoformat(),
        "last_execution_date": period.index[operazioni][-1].date().isoformat(),
    })

    # Consistency check between cumulative return and final capital.
    rapporto_nav = nav[-1] / initial_capital - 1.0
    if not np.isclose(metriche["cumulative_return"], rapporto_nav, rtol=1e-7, atol=1e-8):
        raise RuntimeError("Il rendimento cumulato non coincide con il NAV finale.")
    if not np.isclose(metriche["max_drawdown"], float(drawdown.min()), rtol=1e-7, atol=1e-8):
        raise RuntimeError("Il massimo drawdown non coincide con la curva NAV.")

    return metriche, curve


# =============================================================================
# Evaluation pipeline and output files
# =============================================================================


def run_evaluation(
    config: dict,
    root_dir: Path,
    level: str | None = None,
    backtests: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every strategy of the requested level, on the training period only.

    If ``backtests`` is provided, the DataFrames returned by ``run_backtest``
    are evaluated directly; otherwise the ``backtest_daily.csv`` files are read.

    A1 and A2 output files are kept separate and do not overwrite the
    historical results of the earlier Treasury project.
    """
    livello, nomi = _seleziona_livello(config, level)
    fattore, rf_annuo, inizio, fine, capitale = _parametri_valutazione(config)
    root_dir = Path(root_dir)

    if backtests is not None:
        if not isinstance(backtests, dict) or set(backtests) != set(nomi):
            raise ValueError("I backtest forniti non coincidono con le strategie del livello.")

    righe_metriche = []
    curve_per_strategia = []
    risultati_pronti = []

    # ALL backtests are validated and evaluated first; files are written only
    # afterwards, avoiding partial output if one strategy is invalid.
    for nome in nomi:
        originale = (
            backtests[nome] if backtests is not None
            else load_backtest_dataset(config, root_dir, livello, nome)
        )
        dataset = _convalida_backtest(originale, inizio, fine, capitale, nome)
        metriche, curve = evaluate_period(
            "train", dataset, fattore, rf_annuo, capitale
        )

        intestazione = {
            "period": "train",
            "level": livello,
            "portfolio": nome,
            "start_date": dataset.index.min().date().isoformat(),
            "end_date": dataset.index.max().date().isoformat(),
            "risk_free_rate_annual": float(rf_annuo),
        }
        riga = {**intestazione, **metriche}
        righe_metriche.append(riga)

        curva_con_etichetta = curve.reset_index(names="Date")
        curva_con_etichetta.insert(1, "period", "train")
        curva_con_etichetta.insert(2, "level", livello)
        curva_con_etichetta.insert(3, "portfolio", nome)
        curve_per_strategia.append(curva_con_etichetta)
        risultati_pronti.append((nome, riga, curve))

    metrics_df = pd.DataFrame(righe_metriche)
    curves_df = pd.concat(curve_per_strategia, ignore_index=True)

    output_level = root_dir / config["paths"]["results"] / livello
    output_level.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_level / "evaluation_metrics.csv", index=False)
    curves_df.to_csv(output_level / "evaluation_curves.csv", index=False)

    for nome, metriche, curva in risultati_pronti:
        output_strategia = output_level / nome
        output_strategia.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([metriche]).to_csv(
            output_strategia / "evaluation_metrics.csv", index=False
        )
        curva.to_csv(output_strategia / "equity_curve.csv", index_label="Date")

    print(f"\n[Evaluator] Livello {livello} | solo TRAINING")
    for riga in righe_metriche:
        print(
            f"  {riga['portfolio']:<29} "
            f"CAGR {riga['cagr'] * 100:7.2f}% | "
            f"Vol {riga['annualized_volatility'] * 100:6.2f}% | "
            f"Sharpe {riga['sharpe_ratio']:6.3f} | "
            f"MDD {riga['max_drawdown'] * 100:7.2f}%"
        )
    print(f"  Metriche: {output_level / 'evaluation_metrics.csv'}")
    print("  Validation e OOS non valutati.")

    return metrics_df, curves_df


# =====================================================================
# PERFORMANCE AND STABILITY METRIC UTILITIES
# =====================================================================
def calculate_performance_metrics(
    returns: pd.Series,
    annualization_factor: int = 52,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """Compute performance and risk/stability metrics for a series of periodic returns.

    Args:
        returns: Periodic (weekly) returns.
        annualization_factor: Annualisation factor (default 52 for weekly data).
        risk_free_rate: Annualised risk-free rate (default 0.0).

    Returns:
        Dictionary with cumulative_return, annualized_return, annualized_volatility,
        sharpe_ratio, max_drawdown, profit_factor and calmar_ratio.
    """
    clean_ret = returns.fillna(0.0)
    n = len(clean_ret)
    if n == 0:
        return {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "calmar_ratio": 0.0,
        }

    cum_ret = float((1.0 + clean_ret).prod() - 1.0)
    if cum_ret <= -1.0:
        ann_ret = -1.0
    else:
        ann_ret = float((1.0 + cum_ret) ** (annualization_factor / n) - 1.0)

    vol = float(clean_ret.std(ddof=1) * np.sqrt(annualization_factor)) if n > 1 else 0.0

    excess_ret = ann_ret - risk_free_rate
    if vol > 1e-6:
        sharpe = float(excess_ret / vol)
    else:
        sharpe = 0.0 if excess_ret >= 0 else -99.0

    # Maximum Drawdown
    wealth = (1.0 + clean_ret).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    mdd = abs(float(drawdown.min()))

    # Profit Factor
    gains = float(clean_ret[clean_ret > 0].sum())
    losses = float(abs(clean_ret[clean_ret < 0].sum()))
    profit_factor = float(gains / losses) if losses > 1e-6 else (10.0 if gains > 0 else 1.0)

    # Calmar Ratio
    calmar = float(ann_ret / max(mdd, 0.01))

    return {
        "cumulative_return": cum_ret,
        "annualized_return": ann_ret,
        "annualized_volatility": vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
        "profit_factor": profit_factor,
        "calmar_ratio": calmar,
    }


def compute_objective_score(
    metrics: Dict[str, float],
    metric_type: str = "sharpe_mdd",
) -> float:
    """Compute the optimisation score, balancing return against stability and risk.

    Supported options:
    - ``'sharpe_mdd'`` (default): Sharpe ratio penalised by the maximum drawdown,
      rewarding smooth equity curves with limited maximum losses.
      Score = Sharpe * (1 - MDD) if Sharpe >= 0, otherwise Sharpe * (1 + MDD).
    - ``'sharpe'``: plain annualised Sharpe ratio.
    - ``'calmar'``: Calmar ratio (annualised return / maximum drawdown).
    - ``'return'``: plain cumulative return, with no risk penalty.
    """
    sharpe = metrics["sharpe_ratio"]
    mdd = metrics["max_drawdown"]
    ann_ret = metrics["annualized_return"]

    if metric_type == "sharpe_mdd":
        if sharpe >= 0:
            return float(sharpe * (1.0 - min(mdd, 0.99)))
        else:
            return float(sharpe * (1.0 + min(mdd, 1.0)))
    elif metric_type == "sharpe":
        return float(sharpe)
    elif metric_type == "calmar":
        return float(ann_ret / max(mdd, 0.02))
    elif metric_type == "return":
        return float(metrics["cumulative_return"])
    elif metric_type in ("profit_oriented", "profit"):
        pf = metrics.get("profit_factor", 1.0)
        return float(pf * (1.0 - min(mdd, 0.99)))
    else:
        raise ValueError(f"Metrica obiettivo '{metric_type}' non riconosciuta.")
