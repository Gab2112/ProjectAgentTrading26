"""
main.py 

Entry point for the Project research pipeline.

The project is currently configured around a provisional
Macro / Rates Regimes systematic trading architecture, 
which is a work in progress (it needs to be confirmed by the
whole group). The pipeline is designed to be modular and 
flexible (as decided by the group).

"""

from pathlib import Path
import random

import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.yaml"

def load_config(path: Path) -> dict:
    """Load the central YAML configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration file must contain a dictionary, got {type(config)}")

    return config

def validate_config(config: dict) -> None:
    """Check that the minimum required configuration sections exist."""
    required_sections= [
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
            f"Missing required config sections: {', '.join(missing)}"
        )

def initialize_environment(config: dict) -> None:
    """Set random seeds and create configured data directories."""
    seed = config["project"]["random_seed"]

    random.seed(seed)
    np.random.seed(seed)

    for relative_path in config["paths"].values():
        directory = ROOT_DIR / relative_path
        directory.mkdir(parents=True, exist_ok=True)


def print_project_summary(config: dict) -> None:
    """ Print the active research configuration. """

    assets = config["universe"]["traded_assets"]

    print("\n======================================================")
    print("            PROJECTAGENTTRADING26")
    print("======================================================")
    print(f"Version:           {config['project']['version']}")
    print(f"Research candidate:{' ':1}{config['research']['working_candidate']}")
    print(f"Market scope:      {config['research']['market_scope']}")
    print(f"Strategy frequency:{' ':1}{config['frequency']['strategy']}")
    print(f"Agents enabled:    {config['agents']['enabled']}")
    print("------------------------------------------------------")
    print("Traded universe:")

    for bucket, asset in assets.items():
        print(
            f"  {bucket:<24} "
            f"{asset['ticker']:<6} "
            f"({asset['role']})"
        )

    print("======================================================\n")

    
def run_pipeline() -> None:
    # =========================================================================
    # Initialize the configured research pipelinee
    # =========================================================================
    
    config = load_config(CONFIG_PATH)

    validate_config(config)
    initialize_environment(config)
    print_project_summary(config)

    print("Architecture bootstrap completed successfully.")
    print("Next stage: Macro / Rates data acquisition.")

if __name__ == "__main__":
    run_pipeline()
