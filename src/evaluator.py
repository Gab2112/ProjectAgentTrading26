"""Valutazione delle strategie Multi-Asset A1 e A2.

Il modulo legge i backtest giornalieri prodotti da ``backtester.py``.
Non genera segnali, non ricostruisce i benchmark e non simula operazioni.

Convenzioni adottate in questa fase:
- valutazione esclusivamente sul training, senza usare validation o OOS;
- rendimenti giornalieri netti dei costi di transazione;
- annualizzazione su 252 sedute (configurabile);
- tasso risk-free annuo costante, inizialmente pari a zero;
- CAGR, Sharpe, Sortino, drawdown, turnover e costi per strategia.

La validation dovra' essere implementata in una fase distinta, dopo aver
congelato le regole di A1/A2. Il test OOS non viene mai aperto qui.

Interfaccia pubblica:
    run_evaluation(config, root_dir, level=None, backtests=None)

Output per livello:
    data/results/A1/evaluation_metrics.csv
    data/results/A1/evaluation_curves.csv
    data/results/A2/evaluation_metrics.csv
    data/results/A2/evaluation_curves.csv

Ogni strategia riceve anche il proprio riepilogo e la propria equity curve.
"""

from __future__ import annotations

from pathlib import Path
import re

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
# Configurazione e selezione del livello
# =============================================================================


def _seleziona_livello(config: dict, level: str | None) -> tuple[str, list[str]]:
    """Seleziona soltanto strategie implementate e abilitate nel YAML."""
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
    """Legge i parametri e impone esplicitamente l'uso del solo training."""
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

    # In questa versione i rendimenti del backtester sono giornalieri.
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
# Lettura e validazione dei risultati di backtest
# =============================================================================


def load_backtest_dataset(
    config: dict,
    root_dir: Path,
    level: str,
    portfolio: str,
) -> pd.DataFrame:
    """Legge il backtest di una singola strategia, senza usare file Treasury."""
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
    """Controlla input, calendario, rendimenti e coerenza del NAV.

    Il controllo sulle date precede qualsiasi calcolo: nessuna osservazione
    successiva al training deve entrare nella valutazione.
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

    # La prima riga confronta il NAV con il capitale investito inizialmente;
    # le righe successive utilizzano il NAV della seduta precedente.
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
# Metriche economico-finanziarie
# =============================================================================


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calcola il massimo drawdown includendo la ricchezza iniziale."""
    r = pd.to_numeric(returns, errors="raise").dropna().to_numpy(dtype=float)
    if not len(r) or not np.isfinite(r).all() or np.any(r <= -1):
        raise ValueError("Serie dei rendimenti non valida per il maximum drawdown.")

    ricchezza = np.concatenate(([1.0], np.cumprod(1.0 + r)))
    massimo_precedente = np.maximum.accumulate(ricchezza)
    return float(np.min(ricchezza / massimo_precedente - 1.0))


def calculate_profit_factor(returns: pd.Series) -> float:
    """Rapporto fra rendimenti giornalieri positivi e negativi in valore assoluto.

    E' una misura basata sui rendimenti PERIODICI, non il profit factor dei
    singoli trade. Non va interpretata come numero di operazioni profittevoli.
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
    """Calcola le metriche sui rendimenti giornalieri netti di transazione.

    CAGR: (ricchezza finale / iniziale) ** (252 / n) - 1.
    Sharpe: media dei rendimenti in eccesso / deviazione standard (ddof=1).
    Sortino: media degli eccessi / semideviazione negativa rispetto al risk-free.
    La volatilita' e' la deviazione standard campionaria annualizzata.
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

    # Nel denominatore del Sortino entrano anche le osservazioni positive,
    # con contributo zero: e' la semideviazione rispetto al target risk-free.
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
    """Valuta un backtest giornaliero e costruisce la sua equity curve.

    Il periodo e' sempre training nell'implementazione attuale. Ogni
    strategia viene valutata usando le sue operazioni effettive, senza
    costruire prezzi teorici alternativi.
    """
    if period_name != "train":
        raise NotImplementedError("La valutazione e' attualmente limitata al training.")

    tasso_giornaliero = (1.0 + risk_free_annual) ** (1.0 / annualization_factor) - 1.0
    rf = pd.Series(tasso_giornaliero, index=period.index)
    metriche = calculate_metrics(period["rendimento_netto"], rf, annualization_factor)

    # Il NAV include i costi e i pesi realmente detenuti alle chiusure.
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

    # Verifica di coerenza fra rendimento cumulato e capitale finale.
    rapporto_nav = nav[-1] / initial_capital - 1.0
    if not np.isclose(metriche["cumulative_return"], rapporto_nav, rtol=1e-7, atol=1e-8):
        raise RuntimeError("Il rendimento cumulato non coincide con il NAV finale.")
    if not np.isclose(metriche["max_drawdown"], float(drawdown.min()), rtol=1e-7, atol=1e-8):
        raise RuntimeError("Il massimo drawdown non coincide con la curva NAV.")

    return metriche, curve


# =============================================================================
# Pipeline di valutazione e file di output
# =============================================================================


def run_evaluation(
    config: dict,
    root_dir: Path,
    level: str | None = None,
    backtests: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Valuta tutte le strategie del livello richiesto, solo sul training.

    Se ``backtests`` e' fornito, valuta direttamente i DataFrame restituiti
    da ``run_backtest``. Altrimenti legge i rispettivi ``backtest_daily.csv``.

    I file di output di A1 e A2 sono separati e non sovrascrivono i risultati
    storici del vecchio progetto Treasury.
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

    # Prima si convalidano e si valutano TUTTI i backtest; solo dopo si
    # scrivono i file, evitando output parziali se una strategia e' errata.
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
