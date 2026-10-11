"""
main.py

Entry point principale del progetto ProjectAgentTrading26.

La pipeline esegue in sequenza:

1. data transformation;
2. feature engineering;
3. regime model;
4. signal generation;
5. portfolio construction;
6. backtest;
7. evaluation.

Per default NON riscarica i dati da Internet: il progetto utilizza i CSV
già presenti in data/raw, così l'esecuzione rimane riproducibile a partire
dal contenuto del repository.

Il download può essere riattivato esplicitamente dal config.yaml.
"""

from pathlib import Path
import random

import numpy as np
import yaml

from src.data_acquisition import run_data_acquisition
from src.data_transformation import run_data_transformation
from src.feature_engineering import run_feature_engineering
from src.regime_model import run_regime_model
from src.signal_generation import run_signal_generation
from src.portfolio import run_portfolio
from src.backtester import run_backtest
from src.evaluator import run_evaluation


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.yaml"


# =============================================================================
# Configurazione
# =============================================================================

def load_config(path: Path) -> dict:
    """Carica il file centrale di configurazione YAML."""
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration file must contain a dictionary."
        )

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

    missing = [
        section
        for section in required_sections
        if section not in config
    ]

    if missing:
        raise KeyError(
            "Missing required config sections: "
            + ", ".join(missing)
        )


# =============================================================================
# Ambiente
# =============================================================================

def initialize_environment(config: dict) -> None:
    """
    Imposta i seed casuali e crea le cartelle configurate.

    Questo rende riproducibili gli eventuali componenti stocastici che verranno
    aggiunti successivamente al progetto.
    """
    seed = int(config["project"]["random_seed"])

    random.seed(seed)
    np.random.seed(seed)

    for relative_path in config["paths"].values():
        directory = ROOT_DIR / relative_path
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def print_project_summary(config: dict) -> None:
    """Stampa una sintesi della configurazione attiva."""
    assets = config["universe"]["traded_assets"]
    pipeline_config = config.get("pipeline", {})

    run_download = bool(
        pipeline_config.get(
            "run_data_acquisition",
            False,
        )
    )

    evaluate_oos = bool(
        config.get("evaluation", {})
        .get("evaluate_oos", False)
    )

    print("\n======================================================")
    print("            PROJECTAGENTTRADING26")
    print("======================================================")
    print(
        f"Version:            "
        f"{config['project']['version']}"
    )
    print(
        f"Research candidate: "
        f"{config['research']['working_candidate']}"
    )
    print(
        f"Market scope:       "
        f"{config['research']['market_scope']}"
    )
    print(
        f"Strategy frequency: "
        f"{config['frequency']['strategy']}"
    )
    print(
        f"Agents enabled:     "
        f"{config['agents']['enabled']}"
    )
    print(
        f"Download raw data:  "
        f"{run_download}"
    )
    print(
        f"OOS evaluation:     "
        f"{evaluate_oos}"
    )
    print("------------------------------------------------------")
    print("Traded universe:")

    for bucket, asset in assets.items():
        print(
            f"  {bucket:<24} "
            f"{asset['ticker']:<6} "
            f"({asset['role']})"
        )

    print("======================================================\n")


# =============================================================================
# Pipeline completa
# =============================================================================

def run_pipeline() -> None:
    """Esegue la pipeline completa del baseline Macro / Rates Regimes."""
    config = load_config(CONFIG_PATH)

    validate_config(config)
    initialize_environment(config)
    print_project_summary(config)

    pipeline_config = config.get("pipeline", {})

    run_download = bool(
        pipeline_config.get(
            "run_data_acquisition",
            False,
        )
    )

    # -------------------------------------------------------------------------
    # Stage 0 - Raw data
    # -------------------------------------------------------------------------
    # Per default utilizziamo i CSV già salvati nel repository.
    # Il download viene effettuato solo quando esplicitamente richiesto.
    if run_download:
        print("[0/7] Data acquisition")
        run_data_acquisition(
            config,
            ROOT_DIR,
        )
    else:
        print(
            "[0/7] Data acquisition skipped "
            "(using local raw CSVs)"
        )

    # -------------------------------------------------------------------------
    # Stage 1 - Data transformation
    # -------------------------------------------------------------------------
    print("\n[1/7] Data transformation")
    run_data_transformation(
        config,
        ROOT_DIR,
    )

    # -------------------------------------------------------------------------
    # Stage 2 - Feature engineering
    # -------------------------------------------------------------------------
    print("\n[2/7] Feature engineering")
    run_feature_engineering(
        config,
        ROOT_DIR,
    )

    # -------------------------------------------------------------------------
    # Stage 3 - Regime model
    # -------------------------------------------------------------------------
    print("\n[3/7] Regime model")
    run_regime_model(
        config,
        ROOT_DIR,
    )

    # -------------------------------------------------------------------------
    # Stage 4 - Signal generation
    # -------------------------------------------------------------------------
    print("\n[4/7] Signal generation")
    run_signal_generation(
        config,
        ROOT_DIR,
    )

    # -------------------------------------------------------------------------
    # Stage 5 - Portfolio construction
    # -------------------------------------------------------------------------
    print("\n[5/7] Portfolio construction")
    run_portfolio(
        config,
        ROOT_DIR,
    )

    # -------------------------------------------------------------------------
    # Stage 6 - Backtest
    # -------------------------------------------------------------------------
    print("\n[6/7] Backtest")
    run_backtest(
        config,
        ROOT_DIR,
    )

    # -------------------------------------------------------------------------
    # Stage 7 - Evaluation
    # -------------------------------------------------------------------------
    print("\n[7/7] Evaluation")
    run_evaluation(
        config,
        ROOT_DIR,
    )

    print("\n======================================================")
    print("     END-TO-END PIPELINE COMPLETED SUCCESSFULLY")
    print("======================================================")
    print(
        "Development evaluation completed. "
        "OOS remains controlled by config.yaml."
    )


if __name__ == "__main__":
    run_pipeline()
