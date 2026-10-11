"""
diagnostics.py

Diagnostica dello Stage 2 - Forecasting.

Questo modulo analizza il Feature Set selezionato nel config utilizzando,
per lo sviluppo del modello, esclusivamente il periodo di Train.

Le analisi principali sono:

1. stabilità dei coefficienti all'interno del Train;
2. multicollinearità tra le feature;
3. expanding-window pseudo-OOS all'interno del Train.

La Validation non viene utilizzata per prendere nuove decisioni di sviluppo.
L'OOS rimane completamente escluso.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .evaluation import evaluate_prediction_table

from .models import (
    get_feature_columns,
    load_forecasting_dataset,
    split_development_data,
    validate_forecasting_columns,
)


# =============================================================================
# Stabilità dei coefficienti all'interno del Train
# =============================================================================

def build_coefficient_stability_summary(
    train: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Divide cronologicamente il Train in due metà e stima una Linear Regression
    separatamente sui due periodi.

    Lo scopo non è produrre nuove previsioni, ma verificare se segno e
    intensità delle relazioni rimangono relativamente stabili nel tempo.

    Il Feature Set utilizzato viene letto dal config.
    """

    target_name = config["forecasting"]["target_name"]

    feature_columns = get_feature_columns(
        config
    )

    midpoint = len(train) // 2

    first_half = train.iloc[
        :midpoint
    ].copy()

    second_half = train.iloc[
        midpoint:
    ].copy()

    # -------------------------------------------------------------------------
    # Utilizziamo una scala comune costruita sull'intero Train.
    #
    # Questa operazione è accettabile perché si tratta di una diagnostica
    # descrittiva interna al Train, non di una previsione pseudo-OOS.
    # -------------------------------------------------------------------------
    scaler = StandardScaler()

    scaler.fit(
        train[feature_columns]
    )

    X_first = scaler.transform(
        first_half[feature_columns]
    )

    X_second = scaler.transform(
        second_half[feature_columns]
    )

    X_full = scaler.transform(
        train[feature_columns]
    )

    y_first = first_half[
        target_name
    ]

    y_second = second_half[
        target_name
    ]

    y_full = train[
        target_name
    ]

    # -------------------------------------------------------------------------
    # Stima delle regressioni.
    # -------------------------------------------------------------------------
    model_first = LinearRegression()
    model_second = LinearRegression()
    model_full = LinearRegression()

    model_first.fit(
        X_first,
        y_first,
    )

    model_second.fit(
        X_second,
        y_second,
    )

    model_full.fit(
        X_full,
        y_full,
    )

    summary = pd.DataFrame(
        {
            "feature": feature_columns,
            "coef_first_half": model_first.coef_,
            "coef_second_half": model_second.coef_,
            "coef_full_train": model_full.coef_,
        }
    )

    # -------------------------------------------------------------------------
    # Stabilità del segno.
    # -------------------------------------------------------------------------
    summary["same_sign_between_halves"] = (
        np.sign(summary["coef_first_half"])
        == np.sign(summary["coef_second_half"])
    )

    summary["coefficient_change"] = (
        summary["coef_second_half"]
        - summary["coef_first_half"]
    )

    summary["absolute_coefficient_change"] = (
        summary["coefficient_change"].abs()
    )

    return (
        summary
        .sort_values(
            "absolute_coefficient_change",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# =============================================================================
# Diagnostica della multicollinearità
# =============================================================================

def build_multicollinearity_diagnostics(
    train: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, float]:
    """
    Analizza la dipendenza lineare tra le feature utilizzando esclusivamente
    il periodo di Train.

    Vengono prodotti:

    1. correlazioni pairwise tra le feature;
    2. condition number della matrice standardizzata.

    Un condition number elevato suggerisce che alcune variabili contengano
    informazioni fortemente sovrapposte.
    """

    feature_columns = get_feature_columns(
        config
    )

    X_train = train[
        feature_columns
    ].copy()

    # -------------------------------------------------------------------------
    # Correlazioni pairwise.
    #
    # Utilizziamo soltanto il triangolo superiore della matrice per evitare
    # duplicati.
    # -------------------------------------------------------------------------
    correlation_matrix = (
        X_train.corr()
    )

    upper_triangle = correlation_matrix.where(
        np.triu(
            np.ones(
                correlation_matrix.shape
            ),
            k=1,
        ).astype(bool)
    )

    correlation_pairs = (
        upper_triangle
        .stack()
        .reset_index()
    )

    correlation_pairs.columns = [
        "feature_1",
        "feature_2",
        "correlation",
    ]

    correlation_pairs[
        "absolute_correlation"
    ] = (
        correlation_pairs[
            "correlation"
        ].abs()
    )

    correlation_pairs = (
        correlation_pairs
        .sort_values(
            "absolute_correlation",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Condition number.
    #
    # Le feature vengono standardizzate prima del calcolo per evitare che
    # differenti scale numeriche distorcano la misura.
    # -------------------------------------------------------------------------
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_train
    )

    condition_number = float(
        np.linalg.cond(
            X_scaled
        )
    )

    return (
        correlation_pairs,
        condition_number,
    )


# =============================================================================
# Expanding-window pseudo-OOS all'interno del Train
# =============================================================================

def build_expanding_window_diagnostics(
    train: pd.DataFrame,
    config: dict,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Esegue una valutazione pseudo-out-of-sample esclusivamente all'interno
    del periodo di Train.

    Esempio:

        2005-2009 -> predict 2010
        2005-2010 -> predict 2011
        2005-2011 -> predict 2012
        ...

    Lo scaler e i modelli vengono ristimati da zero a ogni iterazione
    utilizzando esclusivamente informazioni precedenti.

    Viene inoltre applicato il purge coerente con l'orizzonte futuro
    del target.
    """

    target_name = config["forecasting"]["target_name"]

    feature_columns = get_feature_columns(
        config
    )

    horizon = int(
        config["forecasting"]["horizon_weeks"]
    )

    first_test_year = int(
        config["forecasting"][
            "expanding_first_test_year"
        ]
    )

    ridge_alphas = [
        float(alpha)
        for alpha in config[
            "forecasting"
        ]["ridge_alphas"]
    ]

    # -------------------------------------------------------------------------
    # Per ogni osservazione ricostruiamo la data finale utilizzata dal target.
    #
    # Questo permette di applicare il purge anche tra un fold e il successivo.
    # -------------------------------------------------------------------------
    observation_dates = pd.Series(
        train.index,
        index=train.index,
    )

    target_end_dates = (
        observation_dates.shift(
            -horizon
        )
    )

    prediction_blocks = []

    last_available_year = int(
        train.index.max().year
    )

    # =========================================================================
    # Un fold per ogni anno pseudo-OOS
    # =========================================================================

    for test_year in range(
        first_test_year,
        last_available_year + 1,
    ):

        test_start = pd.Timestamp(
            f"{test_year}-01-01"
        )

        test_end = pd.Timestamp(
            f"{test_year}-12-31"
        )

        # ---------------------------------------------------------------------
        # TRAIN DEL FOLD
        #
        # Non basta che l'osservazione appartenga al passato.
        #
        # Anche il target associato deve terminare prima dell'inizio
        # dell'anno di test.
        # ---------------------------------------------------------------------
        fit_mask = (
            (observation_dates < test_start)
            & (target_end_dates < test_start)
        )

        fold_train = train.loc[
            fit_mask
        ].copy()

        # ---------------------------------------------------------------------
        # TEST DEL FOLD
        # ---------------------------------------------------------------------
        test_mask = (
            (train.index >= test_start)
            & (train.index <= test_end)
        )

        fold_test = train.loc[
            test_mask
        ].copy()

        if fold_train.empty or fold_test.empty:
            continue

        # ---------------------------------------------------------------------
        # Matrici X e y.
        # ---------------------------------------------------------------------
        X_train = fold_train[
            feature_columns
        ]

        y_train = fold_train[
            target_name
        ]

        X_test = fold_test[
            feature_columns
        ]

        y_test = fold_test[
            target_name
        ]

        # ---------------------------------------------------------------------
        # Scaling stimato esclusivamente sui dati precedenti del singolo fold.
        # ---------------------------------------------------------------------
        scaler = StandardScaler()

        X_train_scaled = (
            scaler.fit_transform(
                X_train
            )
        )

        X_test_scaled = (
            scaler.transform(
                X_test
            )
        )

        # ---------------------------------------------------------------------
        # Prediction table del fold.
        # ---------------------------------------------------------------------
        fold_predictions = pd.DataFrame(
            index=fold_test.index
        )

        fold_predictions[
            "actual"
        ] = y_test

        # ---------------------------------------------------------------------
        # Naive zero.
        # ---------------------------------------------------------------------
        fold_predictions[
            "naive_zero"
        ] = 0.0

        # ---------------------------------------------------------------------
        # Historical mean calcolata esclusivamente sul passato del fold.
        # ---------------------------------------------------------------------
        historical_mean = float(
            y_train.mean()
        )

        fold_predictions[
            "naive_mean"
        ] = historical_mean

        # ---------------------------------------------------------------------
        # Linear Regression.
        # ---------------------------------------------------------------------
        linear_model = (
            LinearRegression()
        )

        linear_model.fit(
            X_train_scaled,
            y_train,
        )

        fold_predictions[
            "linear_regression"
        ] = (
            linear_model.predict(
                X_test_scaled
            )
        )

        # ---------------------------------------------------------------------
        # Ridge Regression.
        # ---------------------------------------------------------------------
        for alpha in ridge_alphas:

            ridge_model = Ridge(
                alpha=alpha
            )

            ridge_model.fit(
                X_train_scaled,
                y_train,
            )

            column_name = (
                f"ridge_alpha_{alpha:g}"
            )

            fold_predictions[
                column_name
            ] = (
                ridge_model.predict(
                    X_test_scaled
                )
            )

        fold_predictions[
            "test_year"
        ] = test_year

        prediction_blocks.append(
            fold_predictions
        )

    if not prediction_blocks:
        raise ValueError(
            "No expanding-window folds were created."
        )

    # =========================================================================
    # Unione delle previsioni pseudo-OOS
    # =========================================================================

    all_predictions = pd.concat(
        prediction_blocks
    ).sort_index()

    # -------------------------------------------------------------------------
    # Metriche aggregate.
    # -------------------------------------------------------------------------
    pooled_prediction_table = (
        all_predictions
        .drop(
            columns=["test_year"]
        )
    )

    pooled_metrics = (
        evaluate_prediction_table(
            pooled_prediction_table
        )
    )

    pooled_metrics = (
        pooled_metrics
        .sort_values(
            "rmse"
        )
        .reset_index(
            drop=True
        )
    )

    # -------------------------------------------------------------------------
    # Metriche anno per anno.
    # -------------------------------------------------------------------------
    yearly_metric_blocks = []

    for (
        test_year,
        year_data,
    ) in all_predictions.groupby(
        "test_year"
    ):

        year_predictions = (
            year_data
            .drop(
                columns=[
                    "test_year"
                ]
            )
        )

        year_metrics = (
            evaluate_prediction_table(
                year_predictions
            )
        )

        year_metrics.insert(
            0,
            "test_year",
            int(test_year),
        )

        yearly_metric_blocks.append(
            year_metrics
        )

    yearly_metrics = pd.concat(
        yearly_metric_blocks,
        ignore_index=True,
    )

    return (
        all_predictions,
        pooled_metrics,
        yearly_metrics,
    )


# =============================================================================
# Entry point
# =============================================================================

def run_train_only_diagnostics(
    config: dict,
    root_dir: Path,
):
    """
    Esegue le diagnostiche utilizzate per lo sviluppo del Feature Set corrente.

    Vengono utilizzati esclusivamente dati appartenenti al Train.

    La Validation non viene utilizzata per decidere modifiche al modello.
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

    # -------------------------------------------------------------------------
    # Recuperiamo il Train con lo stesso purge utilizzato dai modelli.
    #
    # La Validation viene costruita dalla funzione comune ma non viene
    # utilizzata nelle diagnostiche che seguono.
    # -------------------------------------------------------------------------
    train, _ = split_development_data(
        dataset,
        config,
    )

    feature_set = (
        config["forecasting"]
        .get(
            "feature_set",
            "v1",
        )
    )

    feature_columns = (
        get_feature_columns(
            config
        )
    )

    # -------------------------------------------------------------------------
    # Stabilità dei coefficienti.
    # -------------------------------------------------------------------------
    coefficient_stability = (
        build_coefficient_stability_summary(
            train,
            config,
        )
    )

    # -------------------------------------------------------------------------
    # Multicollinearità.
    # -------------------------------------------------------------------------
    (
        correlation_pairs,
        condition_number,
    ) = build_multicollinearity_diagnostics(
        train,
        config,
    )

    # -------------------------------------------------------------------------
    # Expanding-window pseudo-OOS interno al Train.
    # -------------------------------------------------------------------------
    (
        expanding_predictions,
        expanding_pooled_metrics,
        expanding_yearly_metrics,
    ) = build_expanding_window_diagnostics(
        train,
        config,
    )

    # =========================================================================
    # Salvataggio
    # =========================================================================
    #
    # Ogni Feature Set riceve una propria cartella.
    #
    # In questo modo V2 non sovrascrive gli output diagnostici di V1.
    # =========================================================================

    results_dir = (
        root_dir
        / config["paths"]["results"]
        / "forecasting"
        / "diagnostics"
        / feature_set
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    coefficient_stability.to_csv(
        results_dir
        / "coefficient_stability.csv",
        index=False,
    )

    correlation_pairs.to_csv(
        results_dir
        / "feature_correlations.csv",
        index=False,
    )

    expanding_predictions.to_csv(
        results_dir
        / "expanding_predictions.csv",
    )

    expanding_pooled_metrics.to_csv(
        results_dir
        / "expanding_pooled_metrics.csv",
        index=False,
    )

    expanding_yearly_metrics.to_csv(
        results_dir
        / "expanding_yearly_metrics.csv",
        index=False,
    )

    # =========================================================================
    # Output sintetico
    # =========================================================================

    print(
        "\n[Forecasting Diagnostics] "
        "Train-only diagnostics completed successfully"
    )

    print(
        f"\n  Feature Set:        {feature_set}"
    )

    print(
        f"  Number of features: {len(feature_columns)}"
    )

    print(
        f"  Train observations: {len(train):,}"
    )

    print(
        "\nFEATURES"
    )

    for feature in feature_columns:
        print(
            f"  {feature}"
        )

    # -------------------------------------------------------------------------
    # Stabilità coefficienti.
    # -------------------------------------------------------------------------
    sign_stability = float(
        coefficient_stability[
            "same_sign_between_halves"
        ].mean()
    )

    print(
        "\nLINEAR COEFFICIENT STABILITY"
    )

    print(
        f"  Same sign between Train halves: "
        f"{sign_stability:.2%}"
    )

    print(
        "\n  Largest coefficient changes:"
    )

    print(
        coefficient_stability[
            [
                "feature",
                "coef_first_half",
                "coef_second_half",
                "same_sign_between_halves",
            ]
        ]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Multicollinearità.
    # -------------------------------------------------------------------------
    print(
        "\nMULTICOLLINEARITY"
    )

    print(
        f"  Standardized condition number: "
        f"{condition_number:.2f}"
    )

    print(
        "\n  Highest absolute feature correlations:"
    )

    print(
        correlation_pairs[
            [
                "feature_1",
                "feature_2",
                "correlation",
            ]
        ]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Expanding-window.
    # -------------------------------------------------------------------------
    print(
        "\nEXPANDING-WINDOW "
        "PSEUDO-OOS WITHIN TRAIN"
    )

    print(
        expanding_pooled_metrics
        .to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    print(
        f"\n  Results saved in: {results_dir}"
    )

    print(
        "\n  Validation not used for "
        "new model-development decisions."
    )

    print(
        "  OOS remains locked."
    )

    return (
        coefficient_stability,
        correlation_pairs,
        expanding_pooled_metrics,
        expanding_yearly_metrics,
    )