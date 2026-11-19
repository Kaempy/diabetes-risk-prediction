import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_df():
    """Small synthetic frame with the same schema as the Kaggle dataset."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.binomial(1, 0.15, n)
    return pd.DataFrame({
        "gender": rng.choice(["Female", "Male", " female "], n),
        "age": rng.uniform(1, 80, n).round(),
        "hypertension": rng.binomial(1, 0.1, n),
        "heart_disease": rng.binomial(1, 0.05, n),
        "smoking_history": rng.choice(["never", "No Info", "former", "current"], n),
        "bmi": rng.normal(27, 6, n).clip(12, 90),
        "HbA1c_level": np.where(y == 1, rng.uniform(6, 9, n), rng.uniform(3.5, 6.5, n)).round(1),
        "blood_glucose_level": np.where(y == 1, rng.integers(126, 300, n), rng.integers(80, 160, n)),
        "diabetes": y,
    })
