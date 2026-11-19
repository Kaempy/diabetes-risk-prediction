"""Model definitions, imbalance strategies, training and threshold selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

from .config import RANDOM_STATE
from .evaluation import compute_metrics
from .preprocessing import build_preprocessor

MODEL_NAMES = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
STRATEGIES = ["unweighted", "class-weighted", "SMOTE"]


def make_estimator(name: str, class_weighted: bool = False, random_state: int = RANDOM_STATE):
    """Return an unfitted classifier by display name."""
    cw = "balanced" if class_weighted else None
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, class_weight=cw, random_state=random_state)
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, max_features="sqrt",
            class_weight=cw, n_jobs=-1, random_state=random_state,
        )
    if name == "Gradient Boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
            early_stopping=True, validation_fraction=0.1,
            class_weight=cw, random_state=random_state,
        )
    if name == "Baseline":
        return DummyClassifier(strategy="prior")
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(
    model_name: str,
    strategy: str = "unweighted",
    preprocessor=None,
    random_state: int = RANDOM_STATE,
) -> ImbPipeline:
    """Preprocessing -> (optional SMOTE) -> classifier.

    SMOTE lives *inside* the pipeline, so it is only ever applied to training
    data and never to validation or test rows.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    preprocessor = build_preprocessor() if preprocessor is None else preprocessor
    steps = [("preprocess", preprocessor)]
    if strategy == "SMOTE":
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append((
        "model",
        make_estimator(model_name, class_weighted=strategy == "class-weighted",
                       random_state=random_state),
    ))
    return ImbPipeline(steps)


def compare_models(X_train, y_train, X_val, y_val, model_names=None, strategies=None):
    """Fit every model x strategy on train and score it on validation.

    Returns a tidy results table (sorted by PR-AUC) and a dict of fitted pipelines.
    """
    model_names = MODEL_NAMES if model_names is None else model_names
    strategies = STRATEGIES if strategies is None else strategies
    rows, fitted = [], {}

    baseline = build_pipeline("Baseline").fit(X_train, y_train)
    rows.append({"model": "Baseline (prior)", "strategy": "-",
                 **compute_metrics(y_val, baseline.predict_proba(X_val)[:, 1])})

    for name in model_names:
        for strategy in strategies:
            pipe = build_pipeline(name, strategy).fit(X_train, y_train)
            proba = pipe.predict_proba(X_val)[:, 1]
            rows.append({"model": name, "strategy": strategy, **compute_metrics(y_val, proba)})
            fitted[(name, strategy)] = pipe

    results = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    return results, fitted


def calibrate(pipeline: ImbPipeline, X_train, y_train, method: str = "isotonic", cv: int = 5):
    """Wrap an (unfitted) pipeline in cross-validated probability calibration."""
    return CalibratedClassifierCV(pipeline, method=method, cv=cv).fit(X_train, y_train)


def select_threshold(y_true, y_proba, strategy: str = "f1", min_recall: float = 0.90) -> float:
    """Choose a decision threshold on *validation* data.

    ``strategy="f1"`` maximises F1; ``strategy="recall"`` returns the highest
    threshold that still achieves ``min_recall`` (a screening-oriented rule).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision, recall = precision[:-1], recall[:-1]  # align with thresholds
    if strategy == "f1":
        f1 = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros_like(precision), where=(precision + recall) > 0)
        return float(thresholds[np.argmax(f1)])
    if strategy == "recall":
        ok = np.where(recall >= min_recall)[0]
        return float(thresholds[ok[-1]]) if len(ok) else float(thresholds[0])
    raise ValueError("strategy must be 'f1' or 'recall'")
