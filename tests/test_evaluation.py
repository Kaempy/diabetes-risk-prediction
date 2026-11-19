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


def test_net_benefit_formula():
    from src.evaluation import net_benefit
    y = [1, 1, 0, 0, 0]
    # TP=2, FP=1, n=5, pt=0.2 -> 2/5 - 1/5 * 0.25 = 0.35
    assert net_benefit(y, [1, 1, 1, 0, 0], 0.2) == pytest.approx(0.35)


def test_decision_curve_reference_strategies():
    from src.evaluation import decision_curve
    y = np.array([1, 0, 0, 0])
    p = np.array([0.9, 0.1, 0.2, 0.05])
    rule = np.array([True, True, False, False])
    curve = decision_curve(y, {"model": p, "rule": rule}, thresholds=[0.25])
    row = curve.loc[0.25]
    assert row["Treat none"] == 0
    assert row["Treat all"] == pytest.approx(0.25 - 0.75 / 3)       # prevalence - (1-prev) * odds
    assert row["model"] == pytest.approx(0.25)                      # perfect separation at 0.25
    assert row["rule"] == pytest.approx(0.25 - 0.25 / 3)
