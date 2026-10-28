"""Esecuzione e registrazione dei backtest del progetto Multi-Asset.

Il modulo coordina le strategie A1 e A2 utilizzando un UNICO motore di
simulazione, ``simula_portafoglio`` in ``longshort_backtester.py``.
Non implementa un secondo algoritmo di valorizzazione del portafoglio.

Responsabilita':
1. leggere prezzi Open/Close rettificati dal pannello giornaliero;
2. leggere i pesi target prodotti da ``portfolio.py``;
3. impedire che validation e OOS entrino nel backtest di training;
4. eseguire gli ordini alla prima apertura successiva alla decisione;
5. salvare NAV, rendimenti, turnover, costi e pesi effettivi per strategia.

Le metriche finanziarie (CAGR, Sharpe, drawdown) spettano a ``evaluator.py``.
La scelta della strategia avviene tramite ``level`` o ``pipeline.active_level``.

Interfaccia pubblica:
    run_backtest(config, root_dir, level=None, portfolios=None)

Durante questa fase sono implementati soltanto i livelli A1 e A2,
esclusivamente sul training. Non vengono mai eseguiti B, C o OOS.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.data_transformation import load_dataset
from src.time_split import get_time_windows, is_oos_enabled
from src.utils import get_tickers


SUPPORTED_LEVELS = frozenset({"A1", "A2"})
LEVEL_A_ASSETS = frozenset({"SPY", "IEF", "GLD", "SHY"})
SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def _select_level(config: dict, level: str | None) -> str:
    """Identifica il livello richiesto e impedisce esecuzioni non implementate."""
    selected = str(
        level if level is not None else config.get("pipeline", {}).get("active_level", "A1")
    ).upper()

    if selected not in SUPPORTED_LEVELS:
        raise NotImplementedError(
            f"Backtest del livello {selected} non implementato. "
            "Al momento sono disponibili soltanto A1 e A2."
        )

    settings = config.get("strategies", {}).get(selected)
    if not isinstance(settings, dict) or settings.get("enabled") is not True:
        raise ValueError(f"Il livello {selected} non e' abilitato nel config.yaml.")

    if config.get("pipeline", {}).get("legacy_treasury_enabled") is True:
        raise ValueError("La pipeline Treasury precedente non deve essere abilitata.")

    return selected


def _strategy_names(config: dict, level: str) -> list[str]:
    """Ricava i nomi delle strategie, senza dipendere dai file gia' presenti."""
    if level == "A1":
        benchmarks = config.get("benchmarks", {})
        if not isinstance(benchmarks, dict) or not benchmarks:
            raise ValueError("Mancano i benchmark A1 nel config.yaml.")
        names = config["strategies"]["A1"].get("benchmark_names", list(benchmarks))
        if not isinstance(names, list) or not names:
            raise ValueError("La lista dei benchmark A1 non e' valida.")
        if any(name not in benchmarks for name in names):
            raise KeyError("La selezione A1 contiene benchmark non definiti nel YAML.")
    else:
        names = ["volatility_control"]

    if len(set(names)) != len(names):
        raise ValueError("I nomi delle strategie non possono essere duplicati.")
    if any(not isinstance(name, str) or not SAFE_NAME.fullmatch(name) for name in names):
        raise ValueError("I nomi delle strategie devono contenere soltanto lettere, numeri, '_' e '-'.")

    return list(names)


def _execution_parameters(config: dict) -> dict[str, float]:
    """Legge costi e capitale iniziale, validando la convenzione esecutiva."""
    execution = config.get("execution", {})
    if not isinstance(execution, dict):
        raise TypeError("La sezione 'execution' deve essere un dizionario.")

    if execution.get("execution_rule") != "next_available_trading_open":
        raise ValueError(
            "A1/A2 richiedono execution_rule: next_available_trading_open."
        )
    if execution.get("signal_frequency") != "weekly":
        raise ValueError("A1/A2 richiedono signal_frequency: weekly.")

    keys = {
        "initial_capital": 100000.0,
        "commission_bps": 2.0,
        "slippage_bps": 1.0,
    }
    values = {}
    for key, default in keys.items():
        try:
            value = float(execution.get(key, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Parametro di esecuzione non numerico: {key}.") from exc
        if not np.isfinite(value) or value < 0 or (key == "initial_capital" and value == 0):
            raise ValueError(f"Parametro di esecuzione non valido: {key}.")
        values[key] = value

    return values


def _validate_index(frame: pd.DataFrame, label: str) -> None:
    """Verifica le date prima di applicare ordinamenti o filtri temporali."""
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"{label}: e' richiesto un DatetimeIndex.")
    if index.hasnans or index.tz is not None or index.has_duplicates:
        raise ValueError(f"{label}: date mancanti, duplicate o con fuso orario.")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{label}: le date non sono in ordine crescente.")


def load_market_prices(
    config: dict, root_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Legge i prezzi e seleziona esclusivamente le sedute di training.

    I dati precedenti a ``train_start`` servono a ``portfolio.py`` per gli
    indicatori, ma NON entrano nel periodo contabile del backtest.
    Le osservazioni di validation e OOS vengono escluse prima della
    chiamata al motore di simulazione.
    """
    tickers = get_tickers(config)
    if len(tickers) != 4 or set(tickers) != LEVEL_A_ASSETS:
        raise ValueError("Il backtest A1/A2 richiede SPY, IEF, GLD e SHY.")

    # Impedisci lo sblocco accidentale del test finale in questa fase.
    if is_oos_enabled(config):
        raise ValueError("Backtest OOS non autorizzato: questa pipeline e' limitata al training.")

    windows = get_time_windows(config)
    start, end = windows["train_start"], windows["train_end"]
    if start is None:
        raise ValueError("Occorre impostare 'validation.train_start' nel config.yaml.")

    path = Path(root_dir) / config["paths"]["processed_data"] / "market_daily.csv"
    panel = load_dataset(path, strict=True)
    _validate_index(panel, "market_daily.csv")
    panel = panel.loc[(panel.index >= start) & (panel.index <= end)].copy()
    if len(panel) < 2:
        raise ValueError("Servono almeno due sedute valide nel training.")

    fields = ("open", "close")
    missing = [
        f"{ticker}_{field}"
        for ticker in tickers
        for field in fields
        if f"{ticker}_{field}" not in panel.columns
    ]
    if missing:
        raise KeyError("Colonne prezzi mancanti: " + ", ".join(missing))

    prices = {}
    for field in fields:
        columns = [f"{ticker}_{field}" for ticker in tickers]
        frame = panel.loc[:, columns].copy()
        frame.columns = tickers
        for ticker in tickers:
            frame[ticker] = pd.to_numeric(frame[ticker], errors="raise")
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"I prezzi {field} contengono valori mancanti o non positivi.")
        prices[field] = frame

    return prices["open"], prices["close"]


def load_target_weights(
    config: dict, root_dir: Path, level: str, name: str,
) -> pd.DataFrame:
    """Carica i pesi generati da ``portfolio.py`` per una singola strategia."""
    path = (
        Path(root_dir)
        / config["paths"]["signals"]
        / level
        / name
        / "target_weights.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Pesi target non trovati: {path}. "
            f"Eseguire prima run_portfolio(..., level='{level}')."
        )
    return load_dataset(path, strict=True)


def _validate_targets(
    targets: pd.DataFrame,
    prices: pd.DataFrame,
    config: dict,
    name: str,
) -> pd.DataFrame:
    """Convalida pesi, calendario e rispetto dei confini di training."""
    if not isinstance(targets, pd.DataFrame):
        raise TypeError(f"{name}: i pesi target devono essere un DataFrame.")
    _validate_index(targets, f"Target {name}")
    if targets.empty:
        raise ValueError(f"{name}: non ci sono decisioni di portafoglio.")

    tickers = list(prices.columns)
    if set(targets.columns) != set(tickers) or len(targets.columns) != len(tickers):
        raise ValueError(f"{name}: i pesi devono contenere esattamente {tickers}.")

    windows = get_time_windows(config)
    if targets.index.min() < windows["train_start"] or targets.index.max() > windows["train_end"]:
        raise ValueError(f"{name}: i target devono appartenere soltanto al training.")
    if not targets.index.isin(prices.index).all():
        raise ValueError(f"{name}: alcune decisioni non corrispondono a sedute di mercato.")

    # Non riordinare silenziosamente i pesi: l'ordine canonico e' quello dei prezzi.
    ordered = targets.loc[:, tickers].copy()
    for ticker in tickers:
        ordered[ticker] = pd.to_numeric(ordered[ticker], errors="raise")
    values = ordered.to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or np.any(values < -1e-7)
        or np.any(values > 1 + 1e-7)
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-7, rtol=0)
    ):
        raise ValueError(f"{name}: pesi non validi (richiesti long-only e somma 100%).")

    # Il motore gestira' soltanto le decisioni con almeno una seduta successiva.
    return ordered


def _validate_simulation(
    result: pd.DataFrame,
    opens: pd.DataFrame,
    targets: pd.DataFrame,
    name: str,
) -> dict:
    """Verifica gli invarianti contabili essenziali dopo il backtest."""
    if not result.index.equals(opens.index):
        raise RuntimeError(f"{name}: il motore ha alterato il calendario delle sedute.")

    required = [
        "nav", "rendimento_lordo", "rendimento_netto", "turnover",
        "costo_transazione", "operazione_eseguita",
    ]
    missing = [column for column in required if column not in result.columns]
    if missing:
        raise RuntimeError(f"{name}: colonne di simulazione mancanti: {missing}.")

    for field in required[:-1]:
        series = pd.to_numeric(result[field], errors="raise")
        if not np.isfinite(series.to_numpy(dtype=float)).all():
            raise RuntimeError(f"{name}: valori non finiti nel campo '{field}'.")

    if (result["nav"] <= 0).any():
        raise RuntimeError(f"{name}: NAV non positivo.")
    if (result["turnover"] < 0).any() or (result["costo_transazione"] < 0).any():
        raise RuntimeError(f"{name}: turnover o costi negativi.")
    if (result["rendimento_netto"] <= -1).any():
        raise RuntimeError(f"{name}: rendimento giornaliero minore o uguale a -100%.")

    # Un segnale all'ultima chiusura del training NON viene eseguito nella
    # validation: questa omissione e' deliberata e contabilizzata nel report.
    executable = int(targets.index.isin(opens.index[:-1]).sum())
    trades = int(result["operazione_eseguita"].sum())
    if executable == 0 or trades != executable:
        raise RuntimeError(
            f"{name}: decisioni eseguibili={executable}, operazioni={trades}."
        )

    return {
        "decisioni_ricevute": int(len(targets)),
        "decisioni_non_eseguibili": int(len(targets) - executable),
        "operazioni_eseguite": trades,
        "prima_operazione": result.index[result["operazione_eseguita"]][0].date().isoformat(),
        "ultima_operazione": result.index[result["operazione_eseguita"]][-1].date().isoformat(),
    }


def run_backtest(
    config: dict,
    root_dir: Path,
    level: str | None = None,
    portfolios: dict[str, pd.DataFrame] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """Esegue i backtest di un livello e salva un output per strategia.

    Se ``portfolios`` e' fornito, usa i DataFrame gia' prodotti da
    ``run_portfolio``. Altrimenti legge i file ``target_weights.csv``.

    Output:
        data/results/A1/<benchmark>/backtest_daily.csv
        data/results/A1/<benchmark>/execution_summary.json
        data/results/A2/volatility_control/backtest_daily.csv
        data/results/A2/volatility_control/execution_summary.json

    Restituisce due dizionari: risultati giornalieri e riepiloghi operativi.
    Non calcola indici di performance finanziaria: li produrra' l'evaluator.
    """
    from src.longshort_backtester import simula_portafoglio

    root_dir = Path(root_dir)
    selected = _select_level(config, level)
    names = _strategy_names(config, selected)
    execution = _execution_parameters(config)
    opens, closes = load_market_prices(config, root_dir)

    if portfolios is not None:
        if not isinstance(portfolios, dict) or set(portfolios) != set(names):
            raise ValueError("I portafogli forniti non coincidono con le strategie del livello.")

    daily_results: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict] = {}

    for name in names:
        targets = (
            portfolios[name]
            if portfolios is not None
            else load_target_weights(config, root_dir, selected, name)
        )
        targets = _validate_targets(targets, opens, config, name)

        # Tutta la contabilità e' demandata a un unico motore gia' esistente.
        result = simula_portafoglio(
            aperture=opens,
            chiusure=closes,
            pesi_target=targets,
            capitale_iniziale=execution["initial_capital"],
            commission_bps=execution["commission_bps"],
            slippage_bps=execution["slippage_bps"],
        )
        checks = _validate_simulation(result, opens, targets, name)
        summary = {
            "livello": selected,
            "strategia": name,
            "periodo": "training",
            "inizio_training": opens.index[0].date().isoformat(),
            "fine_training": opens.index[-1].date().isoformat(),
            "capitale_iniziale": execution["initial_capital"],
            "nav_finale": float(result["nav"].iloc[-1]),
            "turnover_totale": float(result["turnover"].sum()),
            "costi_transazione_totali": float(result["costo_transazione"].sum()),
            "commission_bps": execution["commission_bps"],
            "slippage_bps": execution["slippage_bps"],
            **checks,
        }

        # Ogni strategia ha un percorso distinto, senza sovrascrivere altre
        # strategie o i risultati della vecchia pipeline Treasury.
        output_dir = root_dir / config["paths"]["results"] / selected / name
        output_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_dir / "backtest_daily.csv", index_label="Date")
        (output_dir / "execution_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        daily_results[name] = result
        summaries[name] = summary
        print(
            f"[Backtester] {selected}/{name}: "
            f"{checks['operazioni_eseguite']} operazioni, "
            f"NAV finale {summary['nav_finale']:,.2f}, "
            f"costi {summary['costi_transazione_totali']:,.2f}"
        )
        if checks["decisioni_non_eseguibili"]:
            print(
                f"  {checks['decisioni_non_eseguibili']} decisione/i a fine "
                "training non eseguita/e, per non entrare nella validation."
            )

    return daily_results, summaries
