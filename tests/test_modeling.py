import pytest

from src.data_preparation import add_engineered_features, clean_data, split_data
from src.modeling import build_pipeline, compare_models, select_threshold


@pytest.fixture
def splits(raw_df):
    return split_data(add_engineered_features(clean_data(raw_df)))


@pytest.mark.parametrize("strategy", ["unweighted", "class-weighted", "SMOTE"])
def test_pipeline_fits_and_predicts_probabilities(splits, strategy):
    X_train, X_val, _, y_train, _, _ = splits
    pipe = build_pipeline("Logistic Regression", strategy).fit(X_train, y_train)
    proba = pipe.predict_proba(X_val)[:, 1]
    assert proba.shape == (len(X_val),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_smote_is_only_in_smote_pipeline():
    assert "smote" in build_pipeline("Random Forest", "SMOTE").named_steps
    assert "smote" not in build_pipeline("Random Forest", "unweighted").named_steps


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        build_pipeline("Random Forest", "oversample-everything")


def test_compare_models_includes_baseline(splits):
    X_train, X_val, _, y_train, y_val, _ = splits
    results, fitted = compare_models(X_train, y_train, X_val, y_val,
                                     model_names=["Logistic Regression"], strategies=["unweighted"])
    assert "Baseline (prior)" in results["model"].tolist()
    assert ("Logistic Regression", "unweighted") in fitted


def test_select_threshold_recall_target():
    y = [0, 0, 0, 1, 1, 1, 1]
    p = [0.1, 0.2, 0.3, 0.25, 0.6, 0.7, 0.9]
    t = select_threshold(y, p, strategy="recall", min_recall=1.0)
    assert t == 0.25
