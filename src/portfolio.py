"""
portfolio.py

Costruzione dei pesi di portafoglio per la baseline Macro / Rates Regimes.

Questo modulo trasforma i segnali discreti generati da signal_generation.py
in pesi effettivi di portafoglio per SHY, IEF e TLT.

Principio del baseline:
- ogni segnale indica la direzione desiderata (+1 / 0 / -1);
- il position sizing determina invece quanto capitale assegnare;
- quando più asset sono attivi nello stesso momento, il capitale viene
  distribuito in parti uguali tra le posizioni attive; (si potrebbe 
  modificare in futuro per pesi differenti )
- l'esposizione lorda complessiva viene normalizzata a 1.0 di default.

Esempi:
1. TLT = +1
   -> TLT weight = +1.00

2. IEF = +1
   -> IEF weight = +1.00

3. SHY = +1 e TLT = -1
   -> SHY weight = +0.50
   -> TLT weight = -0.50

Le weight prodotte rappresentano TARGET di portafoglio alla data t.
L'esecuzione effettiva verrà gestita nel backtester, applicando le posizioni
dal periodo successivo per evitare look-ahead bias.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd


# =============================================================================
# Lettura dei segnali
# =============================================================================

def load_signal_dataset(config: dict, root_dir: Path) -> pd.DataFrame:
    """Carica il dataset settimanale prodotto da signal_generation.py."""
    path = root_dir / config["paths"]["signals"] / "signals_weekly.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Signal dataset not found: {path}"
        )

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()

    return df


# =============================================================================
# Parametri di portafoglio
# =============================================================================

def _get_portfolio_parameters(config: dict) -> dict:
    """
    Recupera i parametri di portafoglio dal config.yaml.

    Per il baseline utilizziamo:
    - allocazione equal-weight tra le posizioni attive;
    - esposizione lorda massima pari a 1.0;
    - nessuna leva aggiuntiva.
    """
    portfolio_config = config.get("portfolio", {})

    allocation_method = (
        portfolio_config.get("allocation_method")
        or "equal_active_signal"
    )

    leverage = portfolio_config.get("leverage")

    if leverage is None:
        leverage = 1.0

    leverage = float(leverage)

    if leverage <= 0:
        raise ValueError("Portfolio leverage must be positive.")

    return {
        "allocation_method": allocation_method,
        "leverage": leverage,
    }


# =============================================================================
# Costruzione dei pesi
# =============================================================================

def build_portfolio_weights(
    signals: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Converte i segnali discreti in target weights.

    Metodo baseline: equal_active_signal.

    Per ogni settimana:
        peso_i = segnale_i / somma(|segnali attivi|)

    Successivamente i pesi vengono moltiplicati per il livello di leverage
    configurato. Con leverage = 1.0, la gross exposure massima è 100%.

    Se nessun segnale è disponibile/attivo, tutti i pesi sono pari a zero.
    """
    portfolio = signals.copy()
    params = _get_portfolio_parameters(config)

    if params["allocation_method"] != "equal_active_signal":
        raise ValueError(
            "Unsupported allocation_method: "
            f"{params['allocation_method']}"
        )

    signal_columns = [
        "SHY_signal",
        "IEF_signal",
        "TLT_signal",
    ]

    missing = [
        column
        for column in signal_columns
        if column not in portfolio.columns
    ]

    if missing:
        raise KeyError(
            f"Missing signal columns: {', '.join(missing)}"
        )

    # Copia dei segnali in una matrice numerica separata.
    signal_matrix = portfolio[signal_columns].astype(float)

    # Numero equivalente di posizioni attive, misurato come somma dei valori
    # assoluti dei segnali. Esempio: +1 SHY e -1 TLT -> gross signal = 2.
    gross_signal = signal_matrix.abs().sum(axis=1)

    # Evitiamo divisioni per zero nelle settimane in cui tutti i segnali sono flat.
    denominator = gross_signal.replace(0, pd.NA)

    weight_matrix = signal_matrix.div(denominator, axis=0)

    # Se non esiste alcuna posizione attiva, il portafoglio rimane interamente flat.
    weight_matrix = weight_matrix.fillna(0.0)

    # Applichiamo l'eventuale leva configurata. Nel baseline il valore è 1.0.
    weight_matrix = weight_matrix * params["leverage"]

    rename_map = {
        "SHY_signal": "SHY_weight",
        "IEF_signal": "IEF_weight",
        "TLT_signal": "TLT_weight",
    }

    weight_matrix = weight_matrix.rename(columns=rename_map)

    for column in weight_matrix.columns:
        portfolio[column] = weight_matrix[column]

    # -------------------------------------------------------------------------
    # Metriche di esposizione utili per i controlli successivi.
    # -------------------------------------------------------------------------
    weight_columns = list(rename_map.values())

    portfolio["gross_exposure"] = (
        portfolio[weight_columns]
        .abs()
        .sum(axis=1)
    )

    portfolio["net_exposure"] = (
        portfolio[weight_columns]
        .sum(axis=1)
    )

    # Controllo numerico: la gross exposure non deve superare la leva prevista.
    tolerance = 1e-12
    max_allowed = params["leverage"] + tolerance

    if (portfolio["gross_exposure"] > max_allowed).any():
        raise ValueError(
            "Gross exposure exceeds configured leverage."
        )

    return portfolio


# =============================================================================
# Report sintetico
# =============================================================================

def build_portfolio_summary(portfolio: pd.DataFrame) -> pd.DataFrame:
    """
    Crea un report sintetico sui pesi e sulle esposizioni del baseline.
    """

    # Consideriamo solo le settimane in cui il regime macro duration 
    # score sono effettivamente disponibili.
    valid = portfolio["duration_score"].notna()
    portfolio = portfolio.loc[valid]

    rows = []

    for asset in ["SHY", "IEF", "TLT"]:
        column = f"{asset}_weight"

        rows.append(
            {
                "asset": asset,
                "mean_weight": portfolio[column].mean(),
                "mean_abs_weight": portfolio[column].abs().mean(),
                "min_weight": portfolio[column].min(),
                "max_weight": portfolio[column].max(),
                "active_weeks": int(
                    (portfolio[column] != 0).sum()
                ),
            }
        )

    summary = pd.DataFrame(rows)

    return summary


# =============================================================================
# Pipeline del modulo
# =============================================================================

def run_portfolio(
    config: dict,
    root_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Esegue la costruzione del portafoglio e salva gli output.

    Output:
        data/signals/portfolio_weights_weekly.csv
        data/signals/portfolio_summary.csv
    """
    signals_dir = root_dir / config["paths"]["signals"]
    signals_dir.mkdir(parents=True, exist_ok=True)

    signals = load_signal_dataset(config, root_dir)
    portfolio = build_portfolio_weights(signals, config)
    summary = build_portfolio_summary(portfolio)

    portfolio_path = signals_dir / "portfolio_weights_weekly.csv"
    summary_path = signals_dir / "portfolio_summary.csv"

    portfolio.to_csv(portfolio_path)
    summary.to_csv(summary_path, index=False)

    valid = portfolio["duration_score"].notna()

    print("\n[Portfolio] Baseline completed successfully")
    print(f"  Dataset:          {portfolio_path}")
    print(f"  Valid weeks:      {int(valid.sum()):,}")
    print(
        "  Max gross exp.:   "
        f"{portfolio.loc[valid, 'gross_exposure'].max():.2f}"
    )
    print(
        "  Mean gross exp.:  "
        f"{portfolio.loc[valid, 'gross_exposure'].mean():.2f}"
    )
    print(f"  Summary:          {summary_path}")

    return portfolio, summary
