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
import json
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
LIVELLI_IMPLEMENTATI = ("A1", "A2", "B")
UNIVERSO_A = frozenset({"SPY", "IEF", "GLD", "SHY"})
B_VERSIONS = ("v1", "v2")


def load_config(path: Path) -> dict:
    """Load YAML without silently filling missing parameters."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File di configurazione non trovato: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Il file di configurazione deve contenere un dizionario YAML.")
    return config


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
    if len(tickers) != 4 or set(tickers) != UNIVERSO_A:
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


def _write_frozen_outputs(files: dict[Path, str]) -> None:
    """Create absent artifacts, never overwrite a frozen existing artifact.

    Check ALL conflicts before creating ANY missing files, so an unexpected
    difference cannot leave a half-updated version directory behind.
    """
    conflicts = []
    for path, content in files.items():
        if path.exists():
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                conflicts.append(str(path))

    if conflicts:
        raise RuntimeError(
            "Frozen Level B artifacts differ; no files were overwritten:\n"
            + "\n".join(conflicts)
        )

    for path, content in files.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")


def _run_level_b(
    config: dict,
    root_dir: Path,
    b_version: str = "v1",
) -> pd.DataFrame:
    """Reproduce a frozen Level B candidate on training only.

    Both versions reuse the same execution engine and risk calculations.
    Frozen B_v1 and B_v2 outputs are separately stored and compared against
    generated contents before any missing files can be created.
    """
    from src.data_transformation import load_dataset
    from src.regime_allocator import ASSETS
    from src.regime_strategy import (
        build_level_b_training,
        build_level_b_v2_training,
    )
    from src.longshort_backtester import simulate_longshort_daily
    from src.evaluator import calculate_metrics

    if b_version not in B_VERSIONS:
        raise ValueError(f"Unsupported Level B version: {b_version!r}")
    if (
        config["evaluation"]["evaluate_oos"] is not False
        or config["validation"]["development_only"] is not True
    ):
        raise ValueError("Level B must remain training-only")
    if config["features"].get("macro_point_in_time_verified") is not False:
        raise ValueError("Level B macro PIT status must remain unverified")

    b_config = config["strategies"]["B"]
    if float(b_config["cash_interest_rate"]) != 0.0:
        raise ValueError("Nonzero cash interest is not implemented")

    windows = get_time_windows(config)
    start, end = windows["train_start"], windows["train_end"]

    build = (
        build_level_b_training if b_version == "v1"
        else build_level_b_v2_training
    )
    strategy = build(config, root_dir)
    weights = strategy["target_weights"]

    if weights.empty or weights.index.min() < start or weights.index.max() > end:
        raise ValueError("Level B target weights exceed training boundaries")
    if b_version == "v2" and (weights.to_numpy(dtype=float) < -1e-10).any():
        raise ValueError("B_v2 must not contain Short positions")

    market = load_dataset(
        root_dir / config["paths"]["processed_data"] / "market_daily.csv",
        strict=True,
    ).loc[start:end]

    if not weights.index.isin(market.index).all():
        raise ValueError("Decision dates are absent from training market data")

    opens = market[[f"{a}_open" for a in ASSETS]].copy()
    closes = market[[f"{a}_close" for a in ASSETS]].copy()
    opens.columns = list(ASSETS)
    closes.columns = list(ASSETS)

    execution = config["execution"]
    capital = float(execution["initial_capital"])

    result = simulate_longshort_daily(
        open_prices=opens,
        close_prices=closes,
        target_weights=weights,
        initial_capital=capital,
        commission_bps=execution["commission_bps"],
        slippage_bps=execution["slippage_bps"],
        annual_borrow_rate=float(b_config["annual_borrow_rate"]),
        additional_margin_rate=float(b_config["additional_margin_rate"]),
    )

    if (
        result.index.min() < start
        or result.index.max() > end
        or not np.isfinite(result["nav"]).all()
        or (result["nav"] <= 0).any()
        or (result["margin_shortfall"] > 1e-8).any()
    ):
        raise RuntimeError("Level B accounting or temporal checks failed")

    np.testing.assert_allclose(
        (1.0 + result["daily_return"]).prod(),
        result["nav"].iloc[-1] / capital,
        rtol=1e-8,
    )

    ann = int(config["evaluation"]["annualization_factor"])
    rf_annual = float(config["evaluation"]["risk_free_rate"])
    rf_daily = (1.0 + rf_annual) ** (1.0 / ann) - 1.0
    metrics = calculate_metrics(result["daily_return"], rf_daily, ann)

    name = "regime_longshort" if b_version == "v1" else "regime_longflat_v2"
    signals_dir = root_dir / config["paths"]["signals"] / "B" / name
    results_dir = root_dir / config["paths"]["results"] / "B" / name

    if b_version == "v1":
        # Preserve EXACT B5.4 record layout for frozen artifact comparisons.
        record = {
            "period": "train",
            "level": "B",
            "portfolio": "regime_longshort",
            "start_date": result.index.min().date().isoformat(),
            "end_date": result.index.max().date().isoformat(),
            **metrics,
            "initial_capital": capital,
            "final_nav": float(result["nav"].iloc[-1]),
            "total_turnover": float(result["turnover"].sum()),
            "total_transaction_costs": float(result["transaction_cost"].sum()),
            "total_borrow_fees": float(result["borrow_fee"].sum()),
            "executed_rebalances": int(result["executed"].sum()),
        }
        assumptions = {
            "strategy_version": "B_v1_baseline",
            "period": "training_only",
            "macro_point_in_time_verified": False,
            "annual_borrow_rate": float(b_config["annual_borrow_rate"]),
            "additional_margin_rate": float(b_config["additional_margin_rate"]),
            "cash_interest_rate": 0.0,
            "commission_bps": float(execution["commission_bps"]),
            "slippage_bps": float(execution["slippage_bps"]),
            "validation_used": False,
            "oos_used": False,
        }
        files = {
            signals_dir / "target_weights.csv": weights.to_csv(index_label="Date"),
            signals_dir / "signals.csv": strategy["signals"].to_csv(index_label="Date"),
            signals_dir / "regimes.csv": strategy["regimes"].rename("regime").to_csv(index_label="Date"),
            signals_dir / "risk_diagnostics.csv": strategy["risk"].to_csv(index_label="Date"),
            results_dir / "target_weights.csv": weights.to_csv(index_label="Date"),
            results_dir / "backtest_daily.csv": result.to_csv(index_label="Date"),
            results_dir / "evaluation_metrics.csv": pd.DataFrame([record]).to_csv(index=False),
            results_dir / "assumptions.json": json.dumps(assumptions, indent=2) + "\n",
        }
    else:
        # Preserve EXACT B6.7 schema and JSON sorting, not the v1 schema.
        record = {
            "portfolio": "regime_longflat_v2",
            "period": "train",
            **metrics,
            "initial_capital": capital,
            "final_nav": float(result["nav"].iloc[-1]),
            "transaction_costs": float(result["transaction_cost"].sum()),
            "borrow_fees": float(result["borrow_fee"].sum()),
            "executed_rebalances": int(result["executed"].sum()),
        }
        assumptions = {
            "version": "B_v2_candidate",
            "status": "selected_on_training_not_validated",
            "period": "training_only",
            "macro_point_in_time_verified": False,
            "momentum_windows_days": [21, 63],
            "regime_lookback_weeks": 52,
            "regime_based_budgets": True,
            "macro_directional_filter": False,
            "short_positions": False,
            "volatility_window_days": 126,
            "target_volatility_annual": 0.08,
            "commission_bps": float(execution["commission_bps"]),
            "slippage_bps": float(execution["slippage_bps"]),
            "cash_interest_rate": 0.0,
            "validation_used": False,
            "oos_used": False,
        }
        files = {
            signals_dir / "signals.csv": strategy["signals"].to_csv(index_label="Date"),
            signals_dir / "regimes.csv": strategy["regimes"].rename("regime").to_csv(index_label="Date"),
            signals_dir / "budgets.csv": strategy["budgets"].to_csv(index_label="Date"),
            signals_dir / "risk_diagnostics.csv": strategy["risk"].to_csv(index_label="Date"),
            signals_dir / "target_weights.csv": weights.to_csv(index_label="Date"),
            results_dir / "backtest_daily.csv": result.to_csv(index_label="Date"),
            results_dir / "evaluation_metrics.csv": pd.DataFrame([record]).to_csv(index=False),
            results_dir / "assumptions.json": json.dumps(assumptions, indent=2, sort_keys=True) + "\n",
        }

    _write_frozen_outputs(files)

    print(f"\n[Level B / {b_version}] Training-only backtest reproduced")
    print(f"  Decisions:       {len(weights)}")
    print(f"  Executions:      {record['executed_rebalances']}")
    print(f"  Final NAV:       {record['final_nav']:,.2f}")
    print(f"  CAGR:            {metrics['cagr'] * 100:.2f}%")
    print(f"  Sharpe:          {metrics['sharpe_ratio']:.4f}")
    print(f"  Max drawdown:    {metrics['max_drawdown'] * 100:.2f}%")
    print("  Frozen outputs:  verified, never overwritten")
    print("  Macro PIT:       NOT VERIFIED")
    print("  Validation/OOS:  NOT USED")

    return pd.DataFrame([record])


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
            metriche_per_livello[corrente] = _run_level_b(
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
