"""Group 9 reproducible systematic-trading research interface.

Default: safely display the already-frozen final OOS evaluation.

Examples:
    python main.py                                # final OOS results (read-only)
    python main.py --release-action audit         # all available integrity checks
    python main.py --release-action replay        # locked OOS replay in temporary copy
    python main.py --release-action charts        # equity and drawdown charts
    python main.py --level CORE                   # development Core-Satellite vs benchmarks
    python main.py --level C                      # historical C_v1/C_v2 development only
    python main.py --level B --b-version v2       # historical B development only
    python main.py --level ALL                    # A1/A2/B/C/CORE development, no OOS

Historical config.yaml is frozen. release_config.yaml describes the immutable
final candidate without rewriting pre-OOS model selection or parameters.
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
        raise NotImplementedError("The legacy A1/A2 pipeline excludes Level C; use main.py --level C for the isolated historical comparison.")
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


def _run_core_development() -> None:
    from src.release_pipeline import audit_release
    from src.submission_validation import verify_signal_calendar
    from src.core_satellite import run_core_satellite_training
    audit_release(ROOT_DIR)
    verify_signal_calendar(ROOT_DIR)
    cfg = load_config(ROOT_DIR/'config.yaml')
    validate_config(cfg)
    result = run_core_satellite_training(cfg, ROOT_DIR)
    view = result['summary'][['cagr', 'annualized_volatility',
                              'sharpe_ratio', 'max_drawdown', 'trades']].copy()
    for column in ('cagr', 'annualized_volatility', 'max_drawdown'):
        view[column] *= 100.0
    print('\n=== FROZEN CORE-SATELLITE DEVELOPMENT: 2010-2021 ===')
    print(view.round(3).to_string())
    print('Development is in-sample. No model training or OOS evaluation.')


def main(argv: list[str] | None = None) -> None:
    """Unified release interface. Never trains or retunes frozen models."""
    parser = argparse.ArgumentParser(
        description='Group 9: frozen Core-Satellite evaluation, replay and development diagnostics.'
    )
    parser.add_argument('--level', type=str.upper,
                        choices=('A1', 'A2', 'B', 'C', 'CORE', 'ALL'),
                        help='Historical training level (CORE = final candidate development). '
                             'ALL = all development comparisons only.')
    parser.add_argument('--b-version', choices=B_VERSIONS, default=None,
                        help='Select historical B_v1 or B_v2 (default: v2).')
    parser.add_argument('--config', type=Path,
                        help='Alternative configuration for A1/A2/B ONLY (in isolated run).')
    parser.add_argument('--release-action',
                        choices=('audit', 'frozen-results', 'replay', 'charts', 'verify-models'),
                        help='Read-only archive functions, OOS reproducibility replay or plots.')
    parser.add_argument('--core-development', action='store_true',
                        help='Legacy alias for --level CORE; does not run final OOS.')
    parser.add_argument('--output-dir', type=Path,
                        help='Where to write generated figures (charts only).')
    args = parser.parse_args(argv)
    if args.core_development and any([args.level, args.b_version, args.config,
                                       args.release_action, args.output_dir]):
        parser.error('--core-development is exclusive')
    if args.release_action and any([args.level, args.b_version, args.config,
                                    args.core_development]):
        parser.error('Archive action cannot be combined with development flags')
    if args.output_dir is not None and args.release_action != 'charts':
        parser.error('--output-dir requires --release-action charts')
    if args.b_version and args.level not in ('B', 'ALL'):
        parser.error('--b-version requires --level B or ALL')
    if args.config and args.level not in ('A1','A2','B'):
        parser.error('Alternative YAML is available only for A1/A2/B')

    from src.submission_validation import verify_release_contract
    from src.release_pipeline import audit_release, show_frozen_results
    verify_release_contract(ROOT_DIR)

    if args.release_action == 'audit':
        audit_release(ROOT_DIR)
        return
    if args.release_action == 'frozen-results' or (not args.level and
        not args.core_development and not args.release_action):
        show_frozen_results(ROOT_DIR)
        return
    if args.release_action == 'verify-models':
        audit_release(ROOT_DIR)
        from src.orchestrator_loader import verify_all_orchestrators
        print('PASS: correctly loaded', verify_all_orchestrators(ROOT_DIR),
              'frozen GA ensembles, including asymmetric thresholds.')
        return
    if args.release_action == 'replay':
        from src.oos_replay import reproduce_frozen_oos
        reproduce_frozen_oos(ROOT_DIR)
        return
    if args.release_action == 'charts':
        from src.final_charts import build_final_charts
        for image in build_final_charts(ROOT_DIR, args.output_dir):
            print('Saved:', image)
        return
    if args.core_development or args.level == 'CORE':
        _run_core_development()
        return
    if args.level == 'ALL':
        from src.submission_runner import run_development_in_copy
        audit_release(ROOT_DIR)
        run_development_in_copy('ALL')
        _run_core_development()
        return
    from src.submission_runner import run_development_in_copy
    audit_release(ROOT_DIR)
    run_development_in_copy(args.level, b_version=args.b_version or 'v2',
                            config_path=args.config)


if __name__ == '__main__':
    main()
