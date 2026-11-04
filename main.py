"""Entry point for the Regime-Aware Multi-Asset Allocation project.

Available levels:
- A1: buy-and-hold and weekly-rebalanced benchmarks;
- A2: volatility-constrained allocation;
- B: macro-regime strategy, with independently frozen v1 and v2 variants.

All development runs are TRAINING ONLY. Validation and OOS stay blocked.

Usage:
    python main.py --level A1
    python main.py --level A2
    python main.py --level ALL             # A1 and A2 only
    python main.py --level B               # legacy default: v1
    python main.py --level B --b-version v1
    python main.py --level B --b-version v2

The Level B runner verifies existing frozen files before writing. If an
existing artifact disagrees with the calculated result, execution fails
rather than overwriting the artifact. Legacy root-level B metrics are left
untouched; the authoritative outputs are stored in each version's folder.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import pandas as pd

from src.backtester import run_backtest
from src.data_transformation import run_data_transformation
from src.evaluator import run_evaluation
from src.portfolio import run_portfolio
from src.regime_strategy import B_VERSIONS, run_level_b
from src.time_split import get_time_windows, is_oos_enabled
from src.utils import ASSETS, CONFIG_PATH, ROOT_DIR, get_tickers, load_config

LIVELLI_IMPLEMENTATI = ("A1", "A2", "B")


def validate_config(config: dict) -> None:
    """Validate common project and training-only safeguards."""
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
    if len(tickers) != 4 or set(tickers) != set(ASSETS):
        raise ValueError("I livelli Multi-Asset richiedono SPY, IEF, GLD e SHY.")

    frequenze = config["frequency"]
    if frequenze.get("strategy") != "weekly" or frequenze.get("rebalance") != "weekly":
        raise ValueError("La pipeline corrente utilizza decisioni settimanali.")

    pipeline = config["pipeline"]
    if pipeline.get("legacy_treasury_enabled", False) is not False:
        raise ValueError("La precedente pipeline Treasury deve essere disabilitata.")
    if not isinstance(pipeline.get("run_data_acquisition", False), bool):
        raise ValueError("'pipeline.run_data_acquisition' deve essere booleano.")

    if config["evaluation"].get("evaluate_oos") is not False:
        raise ValueError("La valutazione OOS deve restare disabilitata durante lo sviluppo.")
    if config["validation"].get("development_only") is not True:
        raise ValueError("Impostare 'validation.development_only: true' durante lo sviluppo.")
    if is_oos_enabled(config):
        raise ValueError("Accesso OOS non consentito durante lo sviluppo.")

    periodi = get_time_windows(config)
    if periodi["train_start"] is None:
        raise ValueError("Impostare 'validation.train_start' nel config.yaml.")
    if pd.Timestamp(config["data"]["start_date"]) >= periodi["train_start"]:
        raise ValueError(
            "'data.start_date' deve precedere l'inizio del training, "
            "per consentire il riscaldamento degli indicatori."
        )

    if config["execution"].get("execution_rule") != "next_available_trading_open":
        raise ValueError("L'esecuzione deve avvenire all'apertura successiva al segnale.")
    if config["execution"].get("signal_frequency") != "weekly":
        raise ValueError("Il segnale deve essere settimanale.")


def initialize_environment(config: dict, root_dir: Path = ROOT_DIR) -> None:
    """Set random seeds and create output roots, not missing raw-data paths."""
    seme = int(config["project"].get("random_seed", 42))
    random.seed(seme)
    np.random.seed(seme)

    for chiave in ("processed_data", "signals", "results"):
        (Path(root_dir) / config["paths"][chiave]).mkdir(parents=True, exist_ok=True)


def _select_levels(config: dict, level: str | None) -> list[str]:
    """Select only enabled project levels. ALL remains A1+A2."""
    requested = str(
        level if level is not None
        else config["pipeline"].get("active_level", "A1")
    ).upper()

    if requested == "ALL":
        levels = ["A1", "A2"]
    elif requested in LIVELLI_IMPLEMENTATI:
        levels = [requested]
    elif requested == "C":
        raise NotImplementedError("Level C is not yet implemented.")
    else:
        raise ValueError(f"Unknown strategy level: {requested!r}")

    for name in levels:
        settings = config["strategies"].get(name)
        if not isinstance(settings, dict):
            raise ValueError(f"Missing configuration for level {name}")
        if settings.get("enabled") is not True:
            raise ValueError(f"Level {name} is not enabled in config.yaml")
    return levels


def print_project_summary(config: dict, levels: list[str]) -> None:
    """Print effective configuration and forbidden evaluation periods."""
    periodi = get_time_windows(config)
    print("\n=======================================================")
    print("  PROJECT AGENT TRADING 26 | MULTI-ASSET")
    print("=======================================================")
    print(f"Ricerca:            {config['research']['working_candidate']}")
    print(f"Livelli selezionati: {', '.join(levels)}")
    print(f"ETF:                {', '.join(get_tickers(config))}")
    print(f"Decisioni:          {config['frequency']['strategy']}")
    print(f"Training:           {periodi['train_start'].date()} - {periodi['train_end'].date()}")
    print("Validation:         NON valutata")
    print("OOS:                BLOCCATO")
    print(f"Acquisizione dati:  {config['pipeline'].get('run_data_acquisition', False)}")
    print("=======================================================\n")


def run_pipeline(
    level: str | None = None,
    config_path: Path | None = None,
    b_version: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Run A1/A2 or the selected frozen B variant, training only."""
    percorso = Path(config_path) if config_path is not None else CONFIG_PATH
    if not percorso.is_absolute():
        percorso = ROOT_DIR / percorso

    originale = load_config(percorso)
    validate_config(originale)
    livelli = _select_levels(originale, level)

    if b_version is not None and livelli != ["B"]:
        raise ValueError("--b-version can only be used with --level B")
    chosen_b_version = "v1" if b_version is None else str(b_version).lower()
    if chosen_b_version not in B_VERSIONS:
        raise ValueError(f"Invalid --b-version: {chosen_b_version!r}")

    initialize_environment(originale)
    print_project_summary(originale, livelli)
    if livelli == ["B"]:
        print(f"[Level B] Selected version: {chosen_b_version}")

    config = deepcopy(originale)
    config["pipeline"]["active_level"] = livelli[0]

    if config["pipeline"].get("run_data_acquisition", False):
        from src.data_acquisition import run_data_acquisition
        print("[1/2] Acquisizione dei dati grezzi")
        run_data_acquisition(config, ROOT_DIR)
    else:
        print("[1/2] Acquisizione saltata: si usano i CSV locali.")

    print("[2/2] Preparazione dei prezzi giornalieri e settimanali")
    # Existing transformation intentionally rejects direct B and C mode.
    # Reuse A1's price preparation, then build B macro features separately.
    if livelli[0] == "B":
        config["pipeline"]["active_level"] = "A1"
    run_data_transformation(config, ROOT_DIR)

    metriche_per_livello: dict[str, pd.DataFrame] = {}
    for corrente in livelli:
        config["pipeline"]["active_level"] = corrente
        print(f"\n======== LIVELLO {corrente} | SOLO TRAINING ========")

        if corrente == "B":
            metriche_per_livello[corrente] = run_level_b(
                config, ROOT_DIR, b_version=chosen_b_version
            )
            continue

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
    """Parse command line and launch a training-only pipeline."""
    parser = argparse.ArgumentParser(
        description="Multi-Asset pipeline: A1, A2 and frozen Level B variants."
    )
    parser.add_argument(
        "--level", type=str.upper, choices=("A1", "A2", "ALL", "B", "C"),
        default=None, help="Level to execute (default: pipeline.active_level).",
    )
    parser.add_argument(
        "--b-version", choices=B_VERSIONS, default=None,
        help="Level B version: v1 or v2. Default v1 for backward compatibility.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Alternative YAML configuration, relative to the repository root.",
    )
    args = parser.parse_args(argv)
    try:
        run_pipeline(
            level=args.level,
            config_path=args.config,
            b_version=args.b_version,
        )
    except NotImplementedError as exc:
        parser.exit(2, f"Livello non disponibile: {exc}\n")


if __name__ == "__main__":
    main()
