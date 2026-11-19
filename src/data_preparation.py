"""Loading, cleaning, feature engineering and splitting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, RAW_DATA_PATH, TARGET, TEST_SIZE, VAL_SIZE

CONTINUOUS_FEATURES = [
    "age",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "age_bmi",
    "glucose_hba1c",
]
BINARY_FEATURES = ["hypertension", "heart_disease", "prediabetes_flag"]
NUMERIC_FEATURES = CONTINUOUS_FEATURES + BINARY_FEATURES
CATEGORICAL_FEATURES = ["gender", "smoking_history"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Laboratory markers that are also used to *diagnose* diabetes. Used for the
# "no-lab" ablation, which asks how well risk can be estimated before a blood test.
LAB_FEATURES = ["HbA1c_level", "blood_glucose_level", "glucose_hba1c", "prediabetes_flag"]


def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV, with a helpful error if it has not been downloaded."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See data/README.md for download instructions."
        )
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning decisions documented in notebook 01.

    1. Standardise text columns (lower-case, stripped).
    2. Drop exact duplicate rows.
    3. Drop the very small ``gender == "other"`` group (18 rows, no positive cases).
    """
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.lower().str.strip()
    df = df.drop_duplicates()
    df = df[df["gender"] != "other"]
    return df.reset_index(drop=True)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the model input features engineered during the capstone.

    These are row-wise transformations, so they do not leak information
    between the train / validation / test partitions.
    """
    df = df.copy()
    df["age_bmi"] = df["age"].astype(float) * df["bmi"].astype(float)
    df["glucose_hba1c"] = df["blood_glucose_level"].astype(float) * df["HbA1c_level"].astype(float)
    df["prediabetes_flag"] = df["HbA1c_level"].between(5.7, 6.5, inclusive="left").astype(int)
    return df


def add_clinical_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinically labelled bins used for EDA and subgroup analysis (not model inputs)."""
    df = df.copy()
    df["hba1c_cat"] = pd.cut(
        df["HbA1c_level"], bins=[0, 5.7, 6.5, float("inf")], right=False,
        labels=["normal (<5.7)", "prediabetes (5.7-6.4)", "diabetes (>=6.5)"],
    )
    df["bmi_cat"] = pd.cut(
        df["bmi"], bins=[0, 18.5, 25, 30, float("inf")], right=False,
        labels=["underweight", "normal", "overweight", "obese"],
    )
    df["glucose_cat"] = pd.cut(
        df["blood_glucose_level"], bins=[0, 100, 126, float("inf")], right=False,
        labels=["normal (<100)", "prediabetes (100-125)", "high (>=126)"],
    )
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 18, 40, 60, float("inf")], right=False,
        labels=["<18", "18-39", "40-59", "60+"],
    )
    return df


def prepare_dataset(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load, clean and feature-engineer the dataset in one call."""
    return add_engineered_features(clean_data(load_raw_data(path)))


def split_data(
    df: pd.DataFrame,
    features: list[str] | None = None,
    val_size: float = VAL_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Stratified train / validation / test split.

    Returns ``X_train, X_val, X_test, y_train, y_val, y_test``.
    """
    features = MODEL_FEATURES if features is None else features
    X, y = df[features], df[TARGET]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val,
        stratify=y_train_val, random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
