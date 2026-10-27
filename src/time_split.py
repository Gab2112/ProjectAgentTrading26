"""Gestione centralizzata dei periodi temporali del progetto Multi-Asset.

Le date e le autorizzazioni si leggono esclusivamente da ``config.yaml``.
Le funzioni esistenti mantengono le loro interfacce per consentire una
migrazione graduale degli altri moduli.

Regole fondamentali:
- il training inizia da ``train_start`` quando questa data e' definita;
- la validation inizia da ``validation_start`` quando e' definita;
- i dati precedenti al training possono restare disponibili per il
  riscaldamento degli indicatori (ad esempio 126 sedute di covarianza);
- il periodo OOS e' nascosto durante lo sviluppo e richiede una doppia
  autorizzazione esplicita prima di essere reso accessibile.

Questo modulo separa i periodi, ma NON impedisce da solo un utilizzo
improprio della validation nella selezione dei parametri.
"""

import pandas as pd


def _parse_boundary(
    validation: dict,
    name: str,
    *,
    required: bool,
) -> pd.Timestamp | None:
    """Converte una data del file YAML e ne controlla il formato."""
    value = validation.get(name)

    if value is None:
        if required:
            raise ValueError(f"Confine temporale obbligatorio mancante: {name}")
        return None

    try:
        boundary = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Data non valida per {name}: {value!r}") from exc

    if pd.isna(boundary):
        raise ValueError(f"La data {name} non puo' essere NaT.")
    if boundary.tzinfo is not None:
        raise ValueError(f"La data {name} deve essere priva di fuso orario.")
    if boundary != boundary.normalize():
        raise ValueError(f"La data {name} deve indicare un giorno, senza orario.")

    return boundary


def get_time_windows(config: dict) -> dict[str, pd.Timestamp | None]:
    """Legge e valida tutti i confini temporali, incluse le date iniziali.

    Per compatibilita' con la configurazione precedente:
    - ``train_start`` e' facoltativa: se assente, il training comprende
      tutte le osservazioni disponibili fino a ``train_end``;
    - ``validation_start`` e' facoltativa: se assente, coincide con il
      primo giorno di calendario successivo a ``train_end``.

    I confini sono inclusivi. Non si presume che i giorni siano sedute
    di borsa: i filtri si applicano alle date effettive del dataset.
    """
    validation = config.get("validation", {})
    if not isinstance(validation, dict):
        raise TypeError("La sezione 'validation' deve essere un dizionario.")

    train_start = _parse_boundary(validation, "train_start", required=False)
    train_end = _parse_boundary(validation, "train_end", required=True)
    validation_start = _parse_boundary(
        validation, "validation_start", required=False
    )
    validation_end = _parse_boundary(validation, "validation_end", required=True)
    oos_start = _parse_boundary(validation, "oos_start", required=True)

    if validation_start is None:
        validation_start = train_end + pd.Timedelta(days=1)

    if train_start is not None and train_start > train_end:
        raise ValueError("'train_start' deve precedere o coincidere con 'train_end'.")
    if not (train_end < validation_start <= validation_end < oos_start):
        raise ValueError(
            "Ordine delle date non valido: e' necessario "
            "train_end < validation_start <= validation_end < oos_start."
        )

    return {
        "train_start": train_start,
        "train_end": train_end,
        "validation_start": validation_start,
        "validation_end": validation_end,
        "oos_start": oos_start,
    }


def get_time_boundaries(
    config: dict,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Restituisce i tre confini storici, mantenendo l'interfaccia esistente."""
    windows = get_time_windows(config)
    return (
        windows["train_end"],
        windows["validation_end"],
        windows["oos_start"],
    )


def is_oos_enabled(config: dict) -> bool:
    """Autorizza l'OOS solo al termine dello sviluppo.

    L'accesso richiede ``evaluation.evaluate_oos: true`` e, se presente,
    ``validation.development_only: false``. Una configurazione
    contraddittoria genera un errore anziche' sbloccare il test finale.
    """
    evaluation = config.get("evaluation", {})
    validation = config.get("validation", {})

    if not isinstance(evaluation, dict) or not isinstance(validation, dict):
        raise TypeError("Le sezioni 'evaluation' e 'validation' devono essere dizionari.")

    enabled = evaluation.get("evaluate_oos", False)
    development_only = validation.get("development_only", False)

    if not isinstance(enabled, bool):
        raise ValueError("'evaluate_oos' deve essere un valore booleano.")
    if not isinstance(development_only, bool):
        raise ValueError("'development_only' deve essere un valore booleano.")
    if enabled and development_only:
        raise ValueError(
            "Accesso OOS negato: 'development_only' e' ancora true. "
            "Il test finale va sbloccato solo dopo aver fissato il modello."
        )

    return enabled


def _validate_time_index(df: pd.DataFrame) -> None:
    """Controlla che le date siano utilizzabili senza ambiguita'."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Il dataset deve utilizzare un DatetimeIndex.")
    if df.index.hasnans:
        raise ValueError("L'indice temporale contiene date mancanti (NaT).")
    if df.index.tz is not None:
        raise ValueError("L'indice temporale deve essere privo di fuso orario.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Le date del dataset devono essere in ordine crescente.")
    if df.index.has_duplicates:
        raise ValueError("Il dataset contiene date duplicate.")


def restrict_to_allowed_period(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Esclude i dati OOS fino alla loro autorizzazione esplicita.

    Durante lo sviluppo conserva le osservazioni fino a ``validation_end``.
    Le date antecedenti a ``train_start`` rimangono disponibili per
    calcolare indicatori con una finestra storica, senza entrare nelle
    metriche di training calcolate da ``split_time_periods``.

    Questa funzione non applica la suddivisione train/validation.
    """
    windows = get_time_windows(config)
    _validate_time_index(df)

    if is_oos_enabled(config):
        allowed = df.copy()
    else:
        allowed = df.loc[df.index <= windows["validation_end"]].copy()

    if allowed.empty:
        raise ValueError("Nessuna osservazione nel periodo autorizzato.")

    return allowed


def split_time_periods(
    df: pd.DataFrame,
    config: dict,
) -> dict[str, pd.DataFrame]:
    """Suddivide il dataset in training, validation ed eventuale OOS.

    I periodi rispettano le date iniziali e finali definite nel YAML.
    Nessuna osservazione puo' appartenere a due periodi. Eventuali date
    comprese in un intervallo non assegnato restano fuori dagli split.
    L'OOS viene restituito soltanto se autorizzato esplicitamente.
    """
    windows = get_time_windows(config)
    data = restrict_to_allowed_period(df, config)

    train_mask = data.index <= windows["train_end"]
    if windows["train_start"] is not None:
        train_mask &= data.index >= windows["train_start"]

    validation_mask = (
        (data.index >= windows["validation_start"])
        & (data.index <= windows["validation_end"])
    )

    periods = {
        "train": data.loc[train_mask].copy(),
        "validation": data.loc[validation_mask].copy(),
    }

    if is_oos_enabled(config):
        periods["oos"] = data.loc[
            data.index >= windows["oos_start"]
        ].copy()

    for name, period in periods.items():
        if period.empty:
            raise ValueError(f"Periodo temporale senza osservazioni: {name}")

    return periods
