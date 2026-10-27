"""Punto di ingresso unico del progetto Regime-Aware Multi-Asset Allocation.

Livelli attualmente eseguibili:
- A1: benchmark Buy & Hold e portafogli a ribilanciamento settimanale;
- A2: allocazione Equal Weight con controllo della volatilità all'8%.

Il livello B (regimi macro-finanziari) e il livello C (ML/agenti AI) sono
previsti nell'architettura ma NON ancora implementati. Non vengono creati
risultati provvisori o fittizi per questi livelli.

La pipeline condivisa esegue:
1. eventuale acquisizione dei CSV grezzi, se autorizzata nel YAML;
2. trasformazione giornaliera e calendario decisionale settimanale;
3. costruzione dei pesi target per ciascun livello selezionato;
4. backtest con ordini alla prima apertura successiva;
5. valutazione sui soli dati di training.

Uso dalla radice del repository:
    python main.py                 # Usa pipeline.active_level nel YAML
    python main.py --level A1      # Esegue soltanto A1
    python main.py --level A2      # Esegue soltanto A2
    python main.py --level ALL     # Esegue A1 e A2, dati preparati una volta

I dati di validation e OOS NON vengono valutati da questa versione.
Il precedente script run_level_a1.py resta utilizzabile solo come confronto
indipendente e può essere archiviato una volta verificata l'equivalenza.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import pandas as pd
import yaml

from src.data_transformation import run_data_transformation
from src.portfolio import run_portfolio
from src.backtester import run_backtest
from src.evaluator import run_evaluation
from src.time_split import get_time_windows, is_oos_enabled
from src.utils import get_tickers


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
LIVELLI_IMPLEMENTATI = ("A1", "A2")
UNIVERSO_A = frozenset({"SPY", "IEF", "GLD", "SHY"})


def load_config(path: Path) -> dict:
    """Carica il file YAML senza sostituire parametri mancanti in silenzio."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File di configurazione non trovato: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Il file di configurazione deve contenere un dizionario YAML.")
    return config


def validate_config(config: dict) -> None:
    """Verifica i vincoli comuni prima di accedere ai dataset finanziari.

    Le verifiche specifiche di pesi, ottimizzatore e prezzi rimangono nei
    rispettivi moduli. Qui controlliamo che la ricerca sia quella Multi-Asset
    e che nessun periodo futuro possa essere esposto accidentalmente.
    """
    richieste = (
        "project", "research", "universe", "frequency", "data", "portfolio",
        "benchmarks", "strategies", "validation", "evaluation", "execution",
        "paths", "pipeline",
    )
    mancanti = [chiave for chiave in richieste if chiave not in config]
    if mancanti:
        raise KeyError("Sezioni YAML mancanti: " + ", ".join(mancanti))

    if config["research"].get("working_candidate") != "regime_aware_multi_asset_allocation":
        raise ValueError(
            "Il YAML non descrive la ricerca Multi-Asset. "
            "Verificare 'research.working_candidate'."
        )

    tickers = get_tickers(config)
    if len(tickers) != 4 or set(tickers) != UNIVERSO_A:
        raise ValueError("A1/A2 richiedono esattamente SPY, IEF, GLD e SHY.")

    frequenze = config["frequency"]
    if frequenze.get("strategy") != "weekly" or frequenze.get("rebalance") != "weekly":
        raise ValueError("La versione A1/A2 attuale usa decisioni settimanali.")

    pipeline = config["pipeline"]
    if pipeline.get("legacy_treasury_enabled", False) is not False:
        raise ValueError("La precedente pipeline Treasury deve essere disabilitata.")
    if not isinstance(pipeline.get("run_data_acquisition", False), bool):
        raise ValueError("'pipeline.run_data_acquisition' deve essere booleano.")

    valutazione = config["evaluation"]
    if valutazione.get("evaluate_oos") is not False:
        raise ValueError("La valutazione OOS deve restare disabilitata durante A1/A2.")
    if config["validation"].get("development_only") is not True:
        raise ValueError("Impostare 'validation.development_only: true' durante lo sviluppo.")
    if is_oos_enabled(config):
        raise ValueError("Accesso OOS non consentito nella pipeline A1/A2.")

    periodi = get_time_windows(config)
    if periodi["train_start"] is None:
        raise ValueError("Impostare 'validation.train_start' nel config.yaml.")
    if pd.Timestamp(config["data"]["start_date"]) >= periodi["train_start"]:
        raise ValueError(
            "'data.start_date' deve precedere l'inizio del training, "
            "per consentire il riscaldamento degli indicatori A2."
        )

    if config["execution"].get("execution_rule") != "next_available_trading_open":
        raise ValueError("L'esecuzione deve avvenire all'apertura successiva al segnale.")
    if config["execution"].get("signal_frequency") != "weekly":
        raise ValueError("Il segnale deve essere settimanale per A1/A2.")


def initialize_environment(config: dict, root_dir: Path = ROOT_DIR) -> None:
    """Imposta il seme casuale e crea soltanto le directory degli output.

    La directory dei dati grezzi NON viene creata implicitamente: se i
    CSV sono assenti, un errore esplicito evita backtest su input sbagliati.
    """
    seme = int(config["project"].get("random_seed", 42))
    random.seed(seme)
    np.random.seed(seme)

    for chiave in ("processed_data", "signals", "results"):
        (Path(root_dir) / config["paths"][chiave]).mkdir(parents=True, exist_ok=True)


def _select_levels(config: dict, level: str | None) -> list[str]:
    """Risolve il livello richiesto, rifiutando B e C non ancora sviluppati."""
    richiesto = str(
        level if level is not None
        else config["pipeline"].get("active_level", "A1")
    ).upper()

    if richiesto == "ALL":
        livelli = list(LIVELLI_IMPLEMENTATI)
    elif richiesto in LIVELLI_IMPLEMENTATI:
        livelli = [richiesto]
    elif richiesto in ("B", "C"):
        raise NotImplementedError(
            f"Livello {richiesto} non ancora implementato. "
            "I moduli dei regimi e degli agenti non sono integrati."
        )
    else:
        raise ValueError(f"Livello non riconosciuto: {richiesto!r}.")

    for nome in livelli:
        impostazioni = config["strategies"].get(nome)
        if not isinstance(impostazioni, dict) or impostazioni.get("enabled") is not True:
            raise ValueError(f"Il livello {nome} non è abilitato nel config.yaml.")

    return livelli


def print_project_summary(config: dict, levels: list[str]) -> None:
    """Stampa la configurazione effettivamente usata per il backtest."""
    periodi = get_time_windows(config)
    print("\n=======================================================")
    print("  PROJECT AGENT TRADING 26 | MULTI-ASSET")
    print("=======================================================")
    print(f"Ricerca:            {config['research']['working_candidate']}")
    print(f"Livelli selezionati: {', '.join(levels)}")
    print(f"ETF:                {', '.join(get_tickers(config))}")
    print(f"Decisioni:          {config['frequency']['strategy']}")
    print(f"Training:           {periodi['train_start'].date()} - {periodi['train_end'].date()}")
    print(f"Validation:         NON valutata")
    print(f"OOS:                BLOCCATO")
    print(f"Acquisizione dati:  {config['pipeline'].get('run_data_acquisition', False)}")
    print("=======================================================\n")


def run_pipeline(
    level: str | None = None,
    config_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Esegue la pipeline A1/A2 e restituisce le metriche per livello.

    Un'esecuzione ``ALL`` trasforma i dati una sola volta e riutilizza il
    medesimo calendario di mercato, lo stesso motore e gli stessi costi.
    I livelli B/C restano non eseguibili fino alla loro implementazione.
    """
    percorso = Path(config_path) if config_path is not None else CONFIG_PATH
    if not percorso.is_absolute():
        percorso = ROOT_DIR / percorso

    originale = load_config(percorso)
    validate_config(originale)
    livelli = _select_levels(originale, level)
    initialize_environment(originale)
    print_project_summary(originale, livelli)

    # La configurazione usata dal codice è una copia: il YAML sul disco
    # non viene modificato scegliendo --level A1, A2 oppure ALL.
    config = deepcopy(originale)
    config["pipeline"]["active_level"] = livelli[0]

    if config["pipeline"].get("run_data_acquisition", False):
        # Import differito: per usare i CSV locali non servono yfinance o FRED.
        from src.data_acquisition import run_data_acquisition

        print("[1/2] Acquisizione dei dati grezzi")
        run_data_acquisition(config, ROOT_DIR)
    else:
        print("[1/2] Acquisizione saltata: si usano i CSV locali.")

    print("[2/2] Preparazione dei prezzi giornalieri e settimanali")
    run_data_transformation(config, ROOT_DIR)

    metriche_per_livello: dict[str, pd.DataFrame] = {}
    for corrente in livelli:
        # Ogni livello usa gli stessi dati, ma produce segnali e report
        # in cartelle separate per consentire un confronto riproducibile.
        config["pipeline"]["active_level"] = corrente
        print(f"\n======== LIVELLO {corrente} | SOLO TRAINING ========")

        print(f"[{corrente} 1/3] Costruzione dei pesi target")
        portafogli = run_portfolio(config, ROOT_DIR, level=corrente)

        print(f"[{corrente} 2/3] Simulazione dei portafogli")
        backtests, _ = run_backtest(
            config, ROOT_DIR, level=corrente, portfolios=portafogli
        )

        print(f"[{corrente} 3/3] Valutazione delle performance")
        metriche, _ = run_evaluation(
            config, ROOT_DIR, level=corrente, backtests=backtests
        )
        metriche_per_livello[corrente] = metriche

    if len(livelli) > 1:
        # Il confronto aggregato usa solo metriche già calcolate dai singoli
        # evaluator, evitando di introdurre una seconda formula di Sharpe/CAGR.
        confronto = pd.concat(
            [metriche_per_livello[corrente] for corrente in livelli],
            ignore_index=True,
        )
        percorso_confronto = (
            ROOT_DIR / config["paths"]["results"] / "multiasset_level_a_comparison.csv"
        )
        confronto.to_csv(percorso_confronto, index=False)
        print(f"\nConfronto A1/A2: {percorso_confronto}")

    print("\nPipeline Multi-Asset completata sul solo training.")
    print("Validation e OOS non sono stati usati per pesi, backtest o metriche.\n")
    return metriche_per_livello


def main(argv: list[str] | None = None) -> None:
    """Legge gli argomenti da terminale e avvia la pipeline condivisa."""
    parser = argparse.ArgumentParser(
        description="Pipeline unica Multi-Asset: benchmark A1 e allocazione A2."
    )
    parser.add_argument(
        "--level", type=str.upper, choices=("A1", "A2", "ALL", "B", "C"),
        default=None, help="Livello da eseguire (default: pipeline.active_level).",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="File YAML alternativo, relativo alla radice del repository.",
    )
    args = parser.parse_args(argv)
    try:
        run_pipeline(level=args.level, config_path=args.config)
    except NotImplementedError as exc:
        # Il livello B/C è un percorso pianificato, non un errore interno.
        parser.exit(2, f"Livello non disponibile: {exc}\n")


if __name__ == "__main__":
    main()
