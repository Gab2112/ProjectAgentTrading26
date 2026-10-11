"""
models.py

Gestione del training e dei modelli dello Stage 2 - Forecasting.

Il modulo si occupa di:

1. caricare il forecasting_dataset.csv;
2. selezionare il Feature Set richiesto dal config;
3. verificare la presenza delle feature e del target;
4. costruire Train e Validation;
5. applicare il purge temporale coerente con l'orizzonte del target;
6. costruire i naive benchmark;
7. standardizzare le feature utilizzando esclusivamente il Train;
8. addestrare Linear Regression e Ridge Regression;
9. salvare le previsioni per la successiva fase di evaluation.

L'OOS non viene utilizzato in questa fase.
"""

from pathlib import Path

import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .dataset import (
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_V2,
)


# =============================================================================
# Selezione del Feature Set
# =============================================================================

def get_feature_columns(
    config: dict,
) -> list[str]:
    """
    Restituisce il Feature Set richiesto dal config.

    Feature Set disponibili:

    v1:
        set iniziale più ampio con 16 variabili macro/rates;

    v2:
        set più parsimonioso costruito dopo la diagnostica del modello,
        con minore ridondanza tra le variabili.

    La selezione viene effettuata esclusivamente attraverso config.yaml.
    """

    feature_set = (
        config
        .get("forecasting", {})
        .get("feature_set", "v1")
    )

    if feature_set == "v1":
        return FEATURE_COLUMNS_V1

    if feature_set == "v2":
        return FEATURE_COLUMNS_V2

    raise ValueError(
        f"Unknown forecasting feature set: {feature_set}"
    )


# =============================================================================
# Lettura del dataset di forecasting
# =============================================================================

def load_forecasting_dataset(
    config: dict,
    root_dir: Path,
) -> pd.DataFrame:
    """
    Carica il Forecasting Dataset costruito da dataset.py.
    """

    path = (
        root_dir
        / config["paths"]["processed_data"]
        / "forecasting_dataset.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Forecasting dataset not found: {path}"
        )

    df = pd.read_csv(
        path,
        parse_dates=["Date"],
    )

    df = (
        df
        .set_index("Date")
        .sort_index()
    )

    return df


# =============================================================================
# Controllo delle colonne
# =============================================================================

def validate_forecasting_columns(
    df: pd.DataFrame,
    config: dict,
) -> None:
    """
    Verifica che il dataset contenga tutte le informazioni necessarie
    per il Feature Set attualmente selezionato.

    In questo modo il controllo cambia automaticamente quando nel config
    passiamo, per esempio, da feature_set: v1 a feature_set: v2.
    """

    target_name = config["forecasting"]["target_name"]

    feature_columns = get_feature_columns(
        config
    )

    required_columns = (
        feature_columns
        + [target_name]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing forecasting columns: "
            + ", ".join(missing_columns)
        )


# =============================================================================
# Train / Validation con purge temporale
# =============================================================================

def split_development_data(
    df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide il dataset in Train e Validation applicando un purge temporale.

    Il target utilizza informazioni future fino a t + horizon.

    Per esempio, con horizon = 4:

        osservazione al tempo t
        -> target costruito utilizzando il prezzo a t+4

    Di conseguenza, le ultime osservazioni del Train non possono essere
    utilizzate se il loro target entra già nella Validation.

    Lo stesso principio viene applicato al confine Validation / OOS.

    L'OOS non viene restituito e non viene utilizzato.
    """

    horizon = int(
        config["forecasting"]["horizon_weeks"]
    )

    train_end = pd.Timestamp(
        config["validation"]["train_end"]
    )

    validation_end = pd.Timestamp(
        config["validation"]["validation_end"]
    )

    oos_start = pd.Timestamp(
        config["validation"]["oos_start"]
    )

    # -------------------------------------------------------------------------
    # Per ogni osservazione individuiamo la data finale utilizzata dal target.
    #
    # Lo shift utilizza lo stesso ordinamento temporale del forecasting dataset.
    # -------------------------------------------------------------------------
    observation_dates = pd.Series(
        df.index,
        index=df.index,
    )

    target_end_dates = observation_dates.shift(
        -horizon
    )

    # -------------------------------------------------------------------------
    # TRAIN
    #
    # Condizioni:
    #
    # 1. osservazione entro train_end;
    # 2. anche la fine del target deve rimanere dentro il Train.
    #
    # Questo crea il purge temporale prima della Validation.
    # -------------------------------------------------------------------------
    train_mask = (
        (observation_dates <= train_end)
        & (target_end_dates <= train_end)
    )

    train = df.loc[
        train_mask
    ].copy()

    # -------------------------------------------------------------------------
    # VALIDATION
    #
    # Anche il target delle osservazioni di Validation deve terminare
    # prima del confine OOS.
    # -------------------------------------------------------------------------
    validation_mask = (
        (observation_dates > train_end)
        & (observation_dates <= validation_end)
        & (target_end_dates <= validation_end)
        & (target_end_dates < oos_start)
    )

    validation = df.loc[
        validation_mask
    ].copy()

    if train.empty:
        raise ValueError(
            "Training dataset is empty after temporal purge."
        )

    if validation.empty:
        raise ValueError(
            "Validation dataset is empty after temporal purge."
        )

    return (
        train,
        validation,
    )


# =============================================================================
# Naive forecasting benchmarks
# =============================================================================

def build_naive_predictions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Costruisce due benchmark predittivi semplici.

    1. naive_zero:
       assume che il rendimento relativo futuro TLT-SHY sia sempre zero;

    2. naive_mean:
       utilizza come previsione costante la media del target osservata
       esclusivamente nel Train.

    La media non viene mai ricalcolata sulla Validation.
    """

    target_name = config["forecasting"]["target_name"]

    # -------------------------------------------------------------------------
    # Media storica del target calcolata esclusivamente sul Train.
    # -------------------------------------------------------------------------
    train_target_mean = float(
        train[target_name].mean()
    )

    # -------------------------------------------------------------------------
    # Predizioni sul Train.
    # -------------------------------------------------------------------------
    train_predictions = pd.DataFrame(
        index=train.index
    )

    train_predictions["actual"] = (
        train[target_name]
    )

    train_predictions["naive_zero"] = 0.0

    train_predictions["naive_mean"] = (
        train_target_mean
    )

    # -------------------------------------------------------------------------
    # Predizioni sulla Validation.
    #
    # La media utilizzata rimane quella stimata sul Train.
    # -------------------------------------------------------------------------
    validation_predictions = pd.DataFrame(
        index=validation.index
    )

    validation_predictions["actual"] = (
        validation[target_name]
    )

    validation_predictions["naive_zero"] = 0.0

    validation_predictions["naive_mean"] = (
        train_target_mean
    )

    return (
        train_predictions,
        validation_predictions,
    )


# =============================================================================
# Preparazione dei dati per i modelli statistici
# =============================================================================

def prepare_model_inputs(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: dict,
):
    """
    Prepara matrici X e vettori y per Train e Validation.

    Il Feature Set viene selezionato dinamicamente dal config.

    Lo StandardScaler viene stimato esclusivamente sul Train.

    La Validation viene trasformata utilizzando esattamente media e deviazione
    standard apprese sul Train, senza effettuare alcun nuovo fit.
    """

    target_name = config["forecasting"]["target_name"]

    feature_columns = get_feature_columns(
        config
    )

    # -------------------------------------------------------------------------
    # Separazione tra predittori e target.
    # -------------------------------------------------------------------------
    X_train = train[
        feature_columns
    ].copy()

    y_train = train[
        target_name
    ].copy()

    X_validation = validation[
        feature_columns
    ].copy()

    y_validation = validation[
        target_name
    ].copy()

    # -------------------------------------------------------------------------
    # Scaling.
    #
    # fit_transform() viene utilizzato esclusivamente sul Train.
    # -------------------------------------------------------------------------
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(
            X_train
        ),
        index=X_train.index,
        columns=X_train.columns,
    )

    # -------------------------------------------------------------------------
    # La Validation utilizza lo scaler già stimato.
    #
    # Nessuna informazione della Validation entra nella stima della
    # trasformazione.
    # -------------------------------------------------------------------------
    X_validation_scaled = pd.DataFrame(
        scaler.transform(
            X_validation
        ),
        index=X_validation.index,
        columns=X_validation.columns,
    )

    return (
        X_train_scaled,
        y_train,
        X_validation_scaled,
        y_validation,
        scaler,
    )


# =============================================================================
# Modelli statistici
# =============================================================================

def build_statistical_predictions(
    train_predictions: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    config: dict,
):
    """
    Addestra:

    1. Linear Regression;
    2. Ridge Regression per tutti gli alpha dichiarati nel config.

    Tutti i modelli vengono stimati esclusivamente sul Train.

    La Validation viene utilizzata soltanto per produrre previsioni da
    valutare successivamente nel modulo evaluation.py.
    """

    train_output = (
        train_predictions.copy()
    )

    validation_output = (
        validation_predictions.copy()
    )

    # =========================================================================
    # Linear Regression
    # =========================================================================

    linear_model = LinearRegression()

    linear_model.fit(
        X_train,
        y_train,
    )

    train_output["linear_regression"] = (
        linear_model.predict(
            X_train
        )
    )

    validation_output["linear_regression"] = (
        linear_model.predict(
            X_validation
        )
    )

    # =========================================================================
    # Ridge Regression
    # =========================================================================
    #
    # Gli alpha sono dichiarati preventivamente nel config.yaml.
    #
    # Evitiamo quindi di inventare nuovi valori dopo aver osservato i risultati
    # della Validation.
    # =========================================================================

    ridge_alphas = (
        config["forecasting"]["ridge_alphas"]
    )

    ridge_models = {}

    for alpha in ridge_alphas:

        alpha = float(
            alpha
        )

        ridge_model = Ridge(
            alpha=alpha
        )

        ridge_model.fit(
            X_train,
            y_train,
        )

        column_name = (
            f"ridge_alpha_{alpha:g}"
        )

        train_output[column_name] = (
            ridge_model.predict(
                X_train
            )
        )

        validation_output[column_name] = (
            ridge_model.predict(
                X_validation
            )
        )

        # ---------------------------------------------------------------------
        # Conserviamo il modello in memoria per eventuali analisi successive.
        # ---------------------------------------------------------------------
        ridge_models[alpha] = (
            ridge_model
        )

    return (
        train_output,
        validation_output,
        linear_model,
        ridge_models,
    )


# =============================================================================
# Entry point: verifica dello split
# =============================================================================

def run_development_split(
    config: dict,
    root_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carica il dataset e costruisce Train / Validation.

    Questa funzione serve principalmente per verificare che il purge temporale
    sia corretto prima dell'addestramento dei modelli.
    """

    dataset = load_forecasting_dataset(
        config,
        root_dir,
    )

    validate_forecasting_columns(
        dataset,
        config,
    )

    train, validation = split_development_data(
        dataset,
        config,
    )

    feature_set = (
        config["forecasting"].get(
            "feature_set",
            "v1",
        )
    )

    print(
        "\n[Forecasting Models] "
        "Development split completed successfully"
    )

    print(
        f"\n  Feature Set: {feature_set}"
    )

    print("\n  TRAIN")
    print(
        f"    Rows:  {len(train):,}"
    )
    print(
        f"    Start: {train.index.min().date()}"
    )
    print(
        f"    End:   {train.index.max().date()}"
    )

    print("\n  VALIDATION")
    print(
        f"    Rows:  {len(validation):,}"
    )
    print(
        f"    Start: {validation.index.min().date()}"
    )
    print(
        f"    End:   {validation.index.max().date()}"
    )

    print(
        "\n  OOS remains locked."
    )

    return (
        train,
        validation,
    )


# =============================================================================
# Entry point: naive benchmark
# =============================================================================

def run_naive_benchmarks(
    config: dict,
    root_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Costruisce Train / Validation e genera i naive forecasting benchmarks.
    """

    dataset = load_forecasting_dataset(
        config,
        root_dir,
    )

    validate_forecasting_columns(
        dataset,
        config,
    )

    train, validation = split_development_data(
        dataset,
        config,
    )

    train_predictions, validation_predictions = (
        build_naive_predictions(
            train,
            validation,
            config,
        )
    )

    target_name = (
        config["forecasting"]["target_name"]
    )

    print(
        "\n[Forecasting Models] "
        "Naive benchmarks completed successfully"
    )

    print(
        "\n  Training target mean:"
    )
    print(
        f"    {train[target_name].mean():.6f}"
    )

    print(
        "\n  Validation target mean:"
    )
    print(
        f"    {validation[target_name].mean():.6f}"
    )

    print(
        "\n  Forecasts available:"
    )
    print(
        "    naive_zero"
    )
    print(
        "    naive_mean"
    )

    print(
        "\n  OOS remains locked."
    )

    return (
        train_predictions,
        validation_predictions,
    )


# =============================================================================
# Entry point: modelli statistici
# =============================================================================

def run_statistical_models(
    config: dict,
    root_dir: Path,
):
    """
    Esegue lo Stage 2 fino alla produzione delle previsioni dei primi
    modelli statistici.

    Vengono costruiti:

    - naive zero;
    - naive historical mean;
    - Linear Regression;
    - Ridge Regression per tutti gli alpha predefiniti.

    Il Feature Set utilizzato viene letto dal config.yaml.

    L'OOS rimane completamente escluso.
    """

    dataset = load_forecasting_dataset(
        config,
        root_dir,
    )

    validate_forecasting_columns(
        dataset,
        config,
    )

    feature_columns = get_feature_columns(
        config
    )

    feature_set = (
        config["forecasting"].get(
            "feature_set",
            "v1",
        )
    )

    # -------------------------------------------------------------------------
    # Train / Validation con purge temporale.
    # -------------------------------------------------------------------------
    train, validation = split_development_data(
        dataset,
        config,
    )

    # -------------------------------------------------------------------------
    # Benchmark predittivi.
    # -------------------------------------------------------------------------
    (
        train_predictions,
        validation_predictions,
    ) = build_naive_predictions(
        train,
        validation,
        config,
    )

    # -------------------------------------------------------------------------
    # Preparazione delle feature standardizzate.
    # -------------------------------------------------------------------------
    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        scaler,
    ) = prepare_model_inputs(
        train,
        validation,
        config,
    )

    # -------------------------------------------------------------------------
    # Linear Regression + Ridge grid.
    # -------------------------------------------------------------------------
    (
        train_predictions,
        validation_predictions,
        linear_model,
        ridge_models,
    ) = build_statistical_predictions(
        train_predictions,
        validation_predictions,
        X_train,
        y_train,
        X_validation,
        config,
    )

    # -------------------------------------------------------------------------
    # Salvataggio delle previsioni.
    #
    # Gli output Forecasting rimangono separati dai risultati del baseline
    # deterministico dello Stage 1.
    # -------------------------------------------------------------------------
    results_dir = (
        root_dir
        / config["paths"]["results"]
        / "forecasting"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = (
        results_dir
        / "model_predictions_train.csv"
    )

    validation_path = (
        results_dir
        / "model_predictions_validation.csv"
    )

    train_predictions.to_csv(
        train_path
    )

    validation_predictions.to_csv(
        validation_path
    )

    # -------------------------------------------------------------------------
    # Output sintetico.
    # -------------------------------------------------------------------------
    print(
        "\n[Forecasting Models] "
        "Statistical models completed successfully"
    )

    print(
        f"\n  Feature Set:            {feature_set}"
    )

    print(
        f"  Number of features:     {len(feature_columns)}"
    )

    print(
        f"  Train observations:     {len(train_predictions):,}"
    )

    print(
        f"  Validation observations:{len(validation_predictions):,}"
    )

    print(
        "\n  Features:"
    )

    for feature in feature_columns:
        print(
            f"    {feature}"
        )

    print(
        "\n  Models:"
    )

    for column in train_predictions.columns:

        if column != "actual":
            print(
                f"    {column}"
            )

    print(
        "\n  Predictions saved:"
    )

    print(
        f"    {train_path}"
    )

    print(
        f"    {validation_path}"
    )

    print(
        "\n  OOS remains locked."
    )

    return (
        train_predictions,
        validation_predictions,
        scaler,
        linear_model,
        ridge_models,
    )