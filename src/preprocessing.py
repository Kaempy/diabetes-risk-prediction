"""Leakage-safe preprocessing: every statistic is learned from training data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_preparation import BINARY_FEATURES, CATEGORICAL_FEATURES, CONTINUOUS_FEATURES


class IQRCapper(BaseEstimator, TransformerMixin):
    """Clip each column to ``[Q1 - k*IQR, Q3 + k*IQR]`` learned during ``fit``.

    Because it is a scikit-learn transformer, it can sit inside a pipeline and is
    re-fitted on each training fold, so validation / test rows never influence
    the clipping limits.
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor(
    continuous_features: list[str] | None = None,
    binary_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    cap_outliers: bool = True,
) -> ColumnTransformer:
    """Build the preprocessing ColumnTransformer used by every model."""
    continuous_features = CONTINUOUS_FEATURES if continuous_features is None else continuous_features
    binary_features = BINARY_FEATURES if binary_features is None else binary_features
    categorical_features = CATEGORICAL_FEATURES if categorical_features is None else categorical_features

    continuous_steps = [("scaler", StandardScaler())]
    if cap_outliers:
        continuous_steps.insert(0, ("capper", IQRCapper()))

    transformers = [
        ("num", Pipeline(continuous_steps), continuous_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
         categorical_features),
    ]
    if binary_features:
        transformers.insert(1, ("bin", "passthrough", binary_features))

    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def transformed_frame(preprocessor: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    """Transform ``X`` with a fitted preprocessor and return a labelled DataFrame."""
    return pd.DataFrame(
        preprocessor.transform(X),
        columns=preprocessor.get_feature_names_out(),
        index=X.index,
    )
