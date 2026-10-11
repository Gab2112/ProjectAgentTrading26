"""
evaluation.py

Valutazione delle previsioni prodotte nello Stage 2 - Forecasting.

Questo modulo misura la qualità predittiva dei modelli prima della
trasformazione delle previsioni in segnali di trading.

Metriche principali:

1. MAE;
2. RMSE;
3. R²;
4. Relative R² rispetto al naive historical-mean forecast;
5. Pearson correlation;
6. Spearman correlation;
7. Directional Accuracy.

L'OOS non viene utilizzato.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .models import (
    build_naive_predictions,
    load_forecasting_dataset,
    split_development_data,
    validate_forecasting_columns,
)

# =============================================================================
# Lettura delle previsioni dei modelli
# =============================================================================

def load_model_predictions(
    config: dict,
    root_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carica le previsioni già prodotte da models.py.

    La fase di evaluation non riaddestra i modelli:
    utilizza esclusivamente gli output già salvati.
    """

    results_dir = (
        root_dir
        / config["paths"]["results"]
        / "forecasting"
    )

    train_path = (
        results_dir
        / "model_predictions_train.csv"
    )

    validation_path = (
        results_dir
        / "model_predictions_validation.csv"
    )

    if not train_path.exists():
        raise FileNotFoundError(
            f"Train predictions not found: {train_path}"
        )

    if not validation_path.exists():
        raise FileNotFoundError(
            f"Validation predictions not found: {validation_path}"
        )

    train_predictions = pd.read_csv(
        train_path,
        parse_dates=["Date"],
    ).set_index("Date")

    validation_predictions = pd.read_csv(
        validation_path,
        parse_dates=["Date"],
    ).set_index("Date")

    return (
        train_predictions,
        validation_predictions,
    )

# =============================================================================
# Metriche elementari
# =============================================================================

def mean_absolute_error(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    """Errore assoluto medio."""

    return float(
        np.mean(
            np.abs(actual - predicted)
        )
    )


def root_mean_squared_error(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    """Radice dell'errore quadratico medio."""

    return float(
        np.sqrt(
            np.mean(
                (actual - predicted) ** 2
            )
        )
    )


def r_squared(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    """
    Calcola l'R² rispetto alla media effettiva del periodo analizzato.
    """

    residual_sum_squares = float(
        np.sum(
            (actual - predicted) ** 2
        )
    )

    total_sum_squares = float(
        np.sum(
            (actual - actual.mean()) ** 2
        )
    )

    if total_sum_squares == 0:
        return np.nan

    return (
        1.0
        - residual_sum_squares / total_sum_squares
    )


def relative_r_squared(
    actual: pd.Series,
    predicted: pd.Series,
    benchmark: pd.Series,
) -> float:
    """
    Misura quanto il modello migliora rispetto al benchmark predittivo.

    Valore positivo:
        il modello produce errori quadratici inferiori al benchmark.

    Valore negativo:
        il benchmark è migliore del modello.
    """

    model_error = float(
        np.sum(
            (actual - predicted) ** 2
        )
    )

    benchmark_error = float(
        np.sum(
            (actual - benchmark) ** 2
        )
    )

    if benchmark_error == 0:
        return np.nan

    return (
        1.0
        - model_error / benchmark_error
    )

def directional_accuracy(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    """
    Percentuale di osservazioni in cui forecast e target realizzato
    hanno lo stesso segno.

    Se tutte le previsioni sono zero, il modello non esprime una
    direzione e la metrica viene considerata non definita.
    """

    if (predicted == 0).all():
        return np.nan

    actual_direction = np.sign(actual)
    predicted_direction = np.sign(predicted)

    return float(
        np.mean(
            actual_direction == predicted_direction
        )
    )

def safe_pearson_correlation(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    """
    Calcola la correlazione Pearson.

    Una previsione costante, come i naive benchmark, non possiede
    varianza e quindi la correlazione non è definita.
    """

    if predicted.nunique() <= 1:
        return np.nan

    return float(
        actual.corr(
            predicted,
            method="pearson",
        )
    )


def safe_spearman_correlation(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    """
    Calcola la correlazione di rango Spearman.

    Anche in questo caso una previsione costante non produce
    una correlazione statisticamente definita.
    """

    if predicted.nunique() <= 1:
        return np.nan

    return float(
        actual.corr(
            predicted,
            method="spearman",
        )
    )


# =============================================================================
# Valutazione di una tabella di previsioni
# =============================================================================

def evaluate_prediction_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Valuta tutte le colonne di previsione presenti nella tabella.

    La struttura attesa è:

        actual
        naive_zero
        naive_mean
        eventuali modelli futuri...

    naive_mean viene utilizzato come benchmark del Relative R².
    """

    if "actual" not in predictions.columns:
        raise KeyError(
            "Prediction table must contain an 'actual' column."
        )

    if "naive_mean" not in predictions.columns:
        raise KeyError(
            "Prediction table must contain a 'naive_mean' benchmark."
        )

    actual = predictions["actual"]
    benchmark = predictions["naive_mean"]

    model_columns = [
        column
        for column in predictions.columns
        if column != "actual"
    ]

    records = []

    for model_name in model_columns:
        predicted = predictions[model_name]

        records.append(
            {
                "model": model_name,
                "mae": mean_absolute_error(
                    actual,
                    predicted,
                ),
                "rmse": root_mean_squared_error(
                    actual,
                    predicted,
                ),
                "r2": r_squared(
                    actual,
                    predicted,
                ),
                "relative_r2_vs_naive_mean": relative_r_squared(
                    actual,
                    predicted,
                    benchmark,
                ),
                "pearson": safe_pearson_correlation(
                    actual,
                    predicted,
                ),
                "spearman": safe_spearman_correlation(
                    actual,
                    predicted,
                ),
                "directional_accuracy": directional_accuracy(
                    actual,
                    predicted,
                ),
            }
        )

    metrics = pd.DataFrame(records)

    return metrics


# =============================================================================
# Valutazione dei naive benchmark
# =============================================================================

def run_naive_evaluation(
    config: dict,
    root_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Valuta i naive benchmark separatamente su Train e Validation.

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

    train, validation = split_development_data(
        dataset,
        config,
    )

    train_predictions, validation_predictions = build_naive_predictions(
        train,
        validation,
        config,
    )

    train_metrics = evaluate_prediction_table(
        train_predictions
    )

    validation_metrics = evaluate_prediction_table(
        validation_predictions
    )

    # -------------------------------------------------------------------------
    # Salvataggio delle metriche.
    # Manteniamo gli output Forecasting separati dai risultati della Stage 1.
    # -------------------------------------------------------------------------
    forecasting_results_dir = (
        root_dir
        / config["paths"]["results"]
        / "forecasting"
    )

    forecasting_results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = (
        forecasting_results_dir
        / "naive_metrics_train.csv"
    )

    validation_path = (
        forecasting_results_dir
        / "naive_metrics_validation.csv"
    )

    train_metrics.to_csv(
        train_path,
        index=False,
    )

    validation_metrics.to_csv(
        validation_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Informazione utile per interpretare la directional accuracy.
    # -------------------------------------------------------------------------
    train_positive_rate = float(
        (train_predictions["actual"] > 0).mean()
    )

    validation_positive_rate = float(
        (validation_predictions["actual"] > 0).mean()
    )

    print("\n[Forecasting Evaluation] Naive benchmarks evaluated successfully")

    print("\n  TRAIN")
    print(train_metrics.to_string(index=False))
    print(
        f"\n  Actual target > 0: "
        f"{train_positive_rate:.2%}"
    )

    print("\n  VALIDATION")
    print(validation_metrics.to_string(index=False))
    print(
        f"\n  Actual target > 0: "
        f"{validation_positive_rate:.2%}"
    )

    print("\n  OOS remains locked.")

    return train_metrics, validation_metrics


# =============================================================================
# Valutazione dei modelli statistici
# =============================================================================

def run_statistical_evaluation(
    config: dict,
    root_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Valuta naive benchmark, Linear Regression e Ridge Regression
    utilizzando le previsioni già prodotte da models.py.

    Train e Validation vengono valutati separatamente.
    L'OOS rimane completamente escluso.
    """

    (
        train_predictions,
        validation_predictions,
    ) = load_model_predictions(
        config,
        root_dir,
    )

    # -------------------------------------------------------------------------
    # Calcolo delle metriche.
    # La stessa funzione viene applicata a tutti i modelli, così da garantire
    # un confronto coerente sulle stesse osservazioni.
    # -------------------------------------------------------------------------
    train_metrics = evaluate_prediction_table(
        train_predictions
    )

    validation_metrics = evaluate_prediction_table(
        validation_predictions
    )

    # -------------------------------------------------------------------------
    # Ordiniamo soltanto la tabella stampata per RMSE crescente.
    # Non stiamo modificando o selezionando i modelli in modo automatico.
    # -------------------------------------------------------------------------
    train_metrics = (
        train_metrics
        .sort_values("rmse")
        .reset_index(drop=True)
    )

    validation_metrics = (
        validation_metrics
        .sort_values("rmse")
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Salvataggio dei risultati.
    # -------------------------------------------------------------------------
    results_dir = (
        root_dir
        / config["paths"]["results"]
        / "forecasting"
    )

    train_path = (
        results_dir
        / "model_metrics_train.csv"
    )

    validation_path = (
        results_dir
        / "model_metrics_validation.csv"
    )

    train_metrics.to_csv(
        train_path,
        index=False,
    )

    validation_metrics.to_csv(
        validation_path,
        index=False,
    )

    print("\n[Forecasting Evaluation] Statistical models evaluated successfully")

    print("\n  TRAIN")
    print(
        train_metrics.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n  VALIDATION")
    print(
        validation_metrics.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n  Tables are ordered by RMSE only for readability.")
    print("  Model selection has not been finalized.")
    print("\n  OOS remains locked.")

    return (
        train_metrics,
        validation_metrics,
    )