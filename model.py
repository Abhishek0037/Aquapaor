"""
Simple linear regression pipeline for water level forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline


FEATURE_COLUMNS = ["Rainfall", "Temperature"]
TARGET_COLUMN = "Water_Level"
DATE_COLUMN = "Date"
REGION_COLUMN = "Region"


def load_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and parse dates."""
    df = pd.read_csv(csv_path)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    return df


@dataclass(frozen=True)
class ModelResult:
    model: Pipeline
    metrics: dict[str, float | None]


def validate_columns(df: pd.DataFrame) -> None:
    required = {DATE_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN}
    missing = required - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"CSV is missing required column(s): {missing_list}")


def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse Date and coerce numeric columns.

    Note: We allow missing values; the model pipeline imputes missing feature values.
    """
    work = df.copy()
    work[DATE_COLUMN] = pd.to_datetime(work[DATE_COLUMN], errors="coerce")
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    return work


def available_regions(df: pd.DataFrame) -> list[str]:
    """Return region list; falls back to a single 'All' if Region column is absent."""
    if REGION_COLUMN not in df.columns:
        return ["All"]
    regions = (
        df[REGION_COLUMN]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    regions = sorted(regions)
    if "All" not in regions:
        regions = ["All"] + regions
    return regions


def filter_by_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Keep rows for selected region; if no Region column, returns all rows."""
    if REGION_COLUMN not in df.columns or region == "All":
        return df.copy().reset_index(drop=True)
    return df[df[REGION_COLUMN].astype(str) == str(region)].copy().reset_index(drop=True)


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LinearRegression()),
        ]
    )


def time_split(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by time (sorted by Date). Returns (train_df, test_df).
    If too few rows, test_df may be empty.
    """
    ordered = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    n = len(ordered)
    if n < 5:
        return ordered, ordered.iloc[0:0]
    split = max(1, int(round(n * (1 - test_ratio))))
    split = min(split, n - 1)
    return ordered.iloc[:split].copy(), ordered.iloc[split:].copy()


def fit_and_evaluate(df: pd.DataFrame) -> ModelResult:
    """
    Fit a simple Pipeline and compute lightweight metrics with a time-based split.

    Returns a final model fitted on the full df (for plotting/prediction),
    plus metrics computed on a held-out tail split when possible.
    """
    validate_columns(df)
    work = normalize_types(df)
    work = work.dropna(subset=[DATE_COLUMN]).copy()
    if work.empty:
        raise ValueError("No valid dates found in the Date column.")

    # sklearn LinearRegression cannot fit with NaN targets; for this prototype
    # we impute missing Water_Level values using the median of the selected slice.
    target_median = float(work[TARGET_COLUMN].median())
    if math.isnan(target_median):
        raise ValueError("All Water_Level values are missing in the selected slice.")
    work[TARGET_COLUMN] = work[TARGET_COLUMN].fillna(target_median)

    train_df, test_df = time_split(work)
    pipe = build_pipeline()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    pipe.fit(X_train, y_train)

    metrics: dict[str, float | None] = {"mae": None, "r2": None}
    if not test_df.empty:
        X_test = test_df[FEATURE_COLUMNS]
        y_test = test_df[TARGET_COLUMN]
        y_pred = pipe.predict(X_test)
        if len(y_test) >= 1:
            metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
        if len(y_test) >= 2:
            r2 = float(r2_score(y_test, y_pred))
            metrics["r2"] = r2 if math.isfinite(r2) else None

    final_model = build_pipeline()
    final_model.fit(work[FEATURE_COLUMNS], work[TARGET_COLUMN])
    return ModelResult(model=final_model, metrics=metrics)


def make_predictions(df: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    """Return a DataFrame with Date, Actual, and Predicted water levels."""
    work = normalize_types(df)
    work = work.dropna(subset=[DATE_COLUMN]).sort_values(DATE_COLUMN).reset_index(drop=True)
    preds = model.predict(work[FEATURE_COLUMNS])
    out = pd.DataFrame(
        {
            DATE_COLUMN: work[DATE_COLUMN].values,
            "Actual_Water_Level": work[TARGET_COLUMN].values,
            "Predicted_Water_Level": preds,
        }
    )
    return out


def predict_next(model: Pipeline, rainfall: float, temperature: float) -> float:
    """Predict a single next-step water level from user inputs."""
    X = pd.DataFrame([{"Rainfall": rainfall, "Temperature": temperature}])
    return float(model.predict(X)[0])


def to_download_csv(pred_df: pd.DataFrame) -> bytes:
    return pred_df.to_csv(index=False).encode("utf-8")


def model_debug_info(model: Pipeline) -> dict[str, Any]:
    """Return simple, UI-friendly model details."""
    lr = model.named_steps.get("model")
    if not hasattr(lr, "coef_"):
        return {}
    return {
        "coefficients": dict(zip(FEATURE_COLUMNS, [float(x) for x in lr.coef_])),  # type: ignore[attr-defined]
        "intercept": float(lr.intercept_),  # type: ignore[attr-defined]
    }
