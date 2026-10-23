"""
First forecasting experiment: Ridge vs expanding historical mean.

TRAIN ONLY. No validation or OOS access.
No portfolio construction or trading decisions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_dataset


INITIAL_TRAIN = 156
RIDGE_ALPHA = 10.0

FEATURES = [
    "treasury_30y_change_13w",
    "curve_10y_2y",
    "cpi_yoy",
    "rate_volatility_13w",
]

TARGET = "target_next_week_bps"


def main():
    config = yaml.safe_load(
        Path("config.yaml").read_text(encoding="utf-8")
    )

    if config["evaluation"]["evaluate_oos"] is not False:
        raise ValueError("OOS must remain locked.")

    train_end = pd.Timestamp(
        config["validation"]["train_end"]
    )

    df = load_dataset(
        "data/processed/features_weekly.csv"
    ).loc[:train_end].copy()

    X = pd.DataFrame(
        {
            "treasury_30y_change_13w":
                df["rates__treasury_30y"].diff(13),
            "curve_10y_2y":
                df["curve_10y_2y"],
            "cpi_yoy":
                df["cpi_yoy"],
            "rate_volatility_13w":
                df["rate_volatility_13w"],
        },
        index=df.index,
    )

    # At decision date t, forecast return over (t, t+1].
    y = (
        df["TLT_return_1w"]
        - df["SHY_return_1w"]
    ).shift(-1) * 10000

    dataset = (
        X.join(y.rename(TARGET))
        .dropna()
        .sort_index()
    )

    if len(dataset) <= INITIAL_TRAIN:
        raise ValueError("Insufficient training observations.")

    if dataset.index.max() > train_end:
        raise ValueError("Data beyond train boundary.")

    predictions = []
    last_model = None

    for i in range(INITIAL_TRAIN, len(dataset)):
        history = dataset.iloc[:i]
        current = dataset.iloc[[i]]

        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=RIDGE_ALPHA),
        )

        model.fit(
            history[FEATURES],
            history[TARGET],
        )

        ridge_prediction = float(
            model.predict(current[FEATURES])[0]
        )

        benchmark_prediction = float(
            history[TARGET].mean()
        )

        predictions.append(
            {
                "date": current.index[0],
                "actual": float(current[TARGET].iloc[0]),
                "ridge": ridge_prediction,
                "historical_mean": benchmark_prediction,
            }
        )

        last_model = model

    result = pd.DataFrame(predictions).set_index("date")

    actual = result["actual"].to_numpy()
    ridge = result["ridge"].to_numpy()
    baseline = result["historical_mean"].to_numpy()

    ridge_errors = actual - ridge
    baseline_errors = actual - baseline

    ridge_mse = np.mean(ridge_errors ** 2)
    baseline_mse = np.mean(baseline_errors ** 2)

    ridge_rmse = np.sqrt(ridge_mse)
    baseline_rmse = np.sqrt(baseline_mse)

    ridge_mae = np.mean(np.abs(ridge_errors))
    baseline_mae = np.mean(np.abs(baseline_errors))

    ridge_hit = np.mean(np.sign(ridge) == np.sign(actual))
    baseline_hit = np.mean(
        np.sign(baseline) == np.sign(actual)
    )

    relative_r2 = 1 - ridge_mse / baseline_mse

    print("\nRIDGE WALK-FORWARD | TRAIN ONLY")
    print("=" * 55)

    print("Complete observations:", len(dataset))
    print("Initial training size:", INITIAL_TRAIN)
    print("Predictions:", len(result))
    print("First prediction:", result.index.min().date())
    print("Last prediction:", result.index.max().date())
    print("Ridge alpha:", RIDGE_ALPHA)

    print("\nPREDICTIVE PERFORMANCE")
    print("-" * 55)
    print(f"{'Metric':<24}{'Ridge':>14}{'Mean baseline':>16}")

    print(
        f"{'RMSE (bps)':<24}"
        f"{ridge_rmse:>14.3f}"
        f"{baseline_rmse:>16.3f}"
    )
    print(
        f"{'MAE (bps)':<24}"
        f"{ridge_mae:>14.3f}"
        f"{baseline_mae:>16.3f}"
    )
    print(
        f"{'Directional accuracy':<24}"
        f"{ridge_hit:>13.2%}"
        f"{baseline_hit:>15.2%}"
    )

    print(
        "\nR2 vs expanding historical mean:",
        f"{relative_r2:.5f}",
    )

    print("\nFINAL FIT | STANDARDIZED COEFFICIENTS")
    print("-" * 55)

    coefficients = last_model.named_steps["ridge"].coef_

    for feature, coefficient in zip(FEATURES, coefficients):
        print(f"{feature:<32}{coefficient:>12.4f} bps")

    print("\nThese coefficients belong to the final walk-forward fit.")
    print("They are descriptive, not evidence of causal effects.")

    print("\nTRAIN ONLY: OK")
    print("VALIDATION: NOT USED")
    print("OOS: LOCKED")
    print("No trading strategy executed.")
    print("No output files generated.")


if __name__ == "__main__":
    main()
