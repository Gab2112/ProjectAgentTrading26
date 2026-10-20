
"""
time_split.py

Gestione centralizzata dei periodi temporali della ricerca.

Le date sono definite esclusivamente in config.yaml.

Durante lo sviluppo, il periodo OOS non viene reso disponibile
ai moduli downstream della pipeline.

Train e validation sono separati logicamente, senza creare
dataset fisici distinti.
"""

import pandas as pd


def get_time_boundaries(
    config: dict,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Legge e valida i confini temporali della ricerca."""

    validation = config.get("validation", {})
    names = ("train_end", "validation_end", "oos_start")

    missing = [
        name for name in names
        if validation.get(name) is None
    ]

    if missing:
        raise ValueError(
            "Missing temporal boundaries: " + ", ".join(missing)
        )

    try:
        train_end, validation_end, oos_start = (
            pd.Timestamp(validation[name])
            for name in names
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid temporal boundaries.") from exc

    dates = (train_end, validation_end, oos_start)

    if any(pd.isna(date) for date in dates):
        raise ValueError("Temporal boundaries cannot be NaT.")

    if not (train_end < validation_end < oos_start):
        raise ValueError(
            "Expected train_end < validation_end < oos_start."
        )

    return dates


def is_oos_enabled(config: dict) -> bool:
    """Verifica se l'accesso al periodo OOS è autorizzato."""

    enabled = config.get("evaluation", {}).get(
        "evaluate_oos", False
    )

    if not isinstance(enabled, bool):
        raise ValueError("evaluate_oos must be a boolean.")

    return enabled


def restrict_to_allowed_period(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Limita i dati al periodo autorizzato.

    OOS bloccato: osservazioni fino a validation_end.
    OOS abilitato: intero dataset disponibile.

    Non applica lo split train/validation e non modifica
    l'ordine delle osservazioni.
    """

    _, validation_end, _ = get_time_boundaries(config)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Dataset must have a DatetimeIndex.")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Dataset dates must be sorted.")

    if is_oos_enabled(config):
        allowed = df.copy()
    else:
        allowed = df.loc[df.index <= validation_end].copy()

    if allowed.empty:
        raise ValueError("No observations in the allowed period.")

    return allowed


def split_time_periods(
    df: pd.DataFrame,
    config: dict,
) -> dict[str, pd.DataFrame]:
    """
    Restituisce i periodi logici train e validation.

    Restituisce anche OOS soltanto quando esplicitamente
    autorizzato dal config.
    """

    train_end, validation_end, oos_start = (
        get_time_boundaries(config)
    )

    data = restrict_to_allowed_period(df, config)

    periods = {
        "train": data.loc[data.index <= train_end].copy(),
        "validation": data.loc[
            (data.index > train_end)
            & (data.index <= validation_end)
        ].copy(),
    }

    if is_oos_enabled(config):
        periods["oos"] = data.loc[
            data.index >= oos_start
        ].copy()

    for name, period in periods.items():
        if period.empty:
            raise ValueError(f"Empty temporal period: {name}")

    return periods
