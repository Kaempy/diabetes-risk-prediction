"""Global and local model explanations (SHAP, permutation importance, odds ratios)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .preprocessing import transformed_frame


def logistic_odds_ratios(pipeline) -> pd.DataFrame:
    """Coefficients and odds ratios of a fitted logistic-regression pipeline.

    Continuous inputs are standardised, so odds ratios are *per one standard
    deviation* increase, holding the other features constant.
    """
    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    df = pd.DataFrame({
        "feature": pre.get_feature_names_out(),
        "coefficient": model.coef_[0],
    })
    df["odds_ratio"] = np.exp(df["coefficient"])
    return df.sort_values("coefficient", key=np.abs, ascending=False).reset_index(drop=True)


def permutation_importance_table(model, X, y, scoring: str = "average_precision",
                                 n_repeats: int = 10, random_state: int = 42) -> pd.DataFrame:
    """Permutation importance on the *original* predictors (model-agnostic)."""
    result = permutation_importance(model, X, y, scoring=scoring, n_repeats=n_repeats,
                                    random_state=random_state, n_jobs=-1)
    return (pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True))


def shap_explanation(pipeline, X_background: pd.DataFrame, X_explain: pd.DataFrame,
                     background_size: int = 200, random_state: int = 42):
    """SHAP values for a fitted tree-based pipeline, on the transformed feature space.

    Values are in log-odds units (the model's raw margin).
    """
    import shap

    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    background = transformed_frame(pre, X_background.sample(
        min(background_size, len(X_background)), random_state=random_state))
    X_t = transformed_frame(pre, X_explain)
    try:
        # Tree-path-dependent mode is exact for sklearn tree ensembles (the
        # interventional mode fails SHAP's additivity check on HistGradientBoosting).
        explainer = shap.TreeExplainer(model)
    except Exception:  # non-tree model (e.g. logistic regression)
        explainer = shap.Explainer(model, background)
    return explainer(X_t)
