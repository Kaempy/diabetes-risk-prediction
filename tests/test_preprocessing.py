import numpy as np
import pandas as pd

from src.data_preparation import add_engineered_features, clean_data, split_data
from src.preprocessing import IQRCapper, build_preprocessor, transformed_frame


def test_iqr_capper_learns_bounds_from_fit_data_only():
    train = np.array([[1.0], [2.0], [3.0], [4.0]])
    capper = IQRCapper().fit(train)
    # Q1=1.75, Q3=3.25, IQR=1.5 -> bounds [-0.5, 5.5]
    np.testing.assert_allclose(capper.transform(np.array([[-10.0], [100.0], [3.0]])),
                               [[-0.5], [5.5], [3.0]])


def test_preprocessor_output_is_finite_and_named(raw_df):
    df = add_engineered_features(clean_data(raw_df))
    X_train, X_val, *_ = split_data(df)
    pre = build_preprocessor().fit(X_train)
    out = transformed_frame(pre, X_val)

    assert np.isfinite(out.to_numpy()).all()
    assert "HbA1c_level" in out.columns
    assert any(c.startswith("smoking_history_") for c in out.columns)


def test_preprocessor_ignores_unseen_categories(raw_df):
    df = add_engineered_features(clean_data(raw_df))
    X_train, X_val, *_ = split_data(df)
    pre = build_preprocessor().fit(X_train)
    unseen = X_val.iloc[:1].assign(smoking_history="brand new category")
    assert transformed_frame(pre, unseen).shape[0] == 1
