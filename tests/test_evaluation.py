import numpy as np
import pytest

from src.evaluation import assign_risk_groups, compute_metrics, risk_group_table


def test_compute_metrics_confusion_counts():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], threshold=0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["sensitivity"] == 0.5 and m["specificity"] == 0.5
    assert m["roc_auc"] == pytest.approx(0.75)


def test_risk_groups_boundaries():
    groups = assign_risk_groups([0.0, 0.019, 0.02, 0.199, 0.2, 1.0], cutoffs=(0.02, 0.20))
    assert list(groups) == ["Low", "Low", "Moderate", "Moderate", "High", "High"]


def test_risk_group_table_shares_sum_to_one():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=500)
    y = rng.binomial(1, p)
    table = risk_group_table(y, p)
    assert table["individuals"].sum() == 500
    assert table["share_of_all_cases"].sum() == pytest.approx(1.0)
