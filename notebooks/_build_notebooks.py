"""Generate the project notebooks from source.

Run from the project root:  python notebooks/_build_notebooks.py
Then execute them:          jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

SETUP = """\
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", 30)

from src.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, TARGET
from src.evaluation import save_fig"""


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(nb, HERE / name)
    print("wrote", name)


# =============================================================================
# 01 — Data exploration
# =============================================================================
write("01_data_exploration.ipynb", [
    md("""
# 01 · Data Exploration

**Goal:** understand the structure, quality and signal in the data before modelling.

**Dataset:** [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) — 100,000 patient records with demographic, comorbidity, lifestyle and laboratory features, and a binary `diabetes` label.

| Column | Description |
|---|---|
| `gender` | Female / Male / Other |
| `age` | Age in years (0.08 – 80) |
| `hypertension`, `heart_disease` | 0/1 comorbidity flags |
| `smoking_history` | never, former, current, not current, ever, No Info |
| `bmi` | Body-mass index (kg/m²) |
| `HbA1c_level` | Glycated haemoglobin (%) — ~3-month average blood glucose |
| `blood_glucose_level` | Blood glucose (mg/dL) |
| `diabetes` | **Target** — 1 = diabetes |
"""),
    code(SETUP),
    code("""
from src.data_preparation import load_raw_data, clean_data, add_engineered_features, add_clinical_bins

raw = load_raw_data()
print(raw.shape)
raw.head()
"""),
    md("## 1. Data quality"),
    code("""
quality = pd.DataFrame({
    "dtype": raw.dtypes.astype(str),
    "missing": raw.isna().sum(),
    "unique": raw.nunique(),
})
print(f"Duplicate rows: {raw.duplicated().sum():,}")
quality
"""),
    code("""
for col in ["gender", "smoking_history"]:
    print(raw[col].value_counts(), "\\n")
print("Diabetes among gender == 'Other':", raw.loc[raw.gender == "Other", TARGET].value_counts().to_dict())
"""),
    md("""
**Findings**

* No missing values in the conventional sense, but **`smoking_history = "No Info"` (~36%)** is effectively a missing-value category. We keep it as its own level rather than imputing, because "not recorded" may itself carry information.
* **3,854 exact duplicate rows** are dropped.
* `gender == "Other"` has only **18 rows, none with diabetes**. It is too small to learn from, so these rows are removed (documented as a limitation).
* `age` has a minimum of 0.08 — infants are present, which matters when interpreting age effects.
"""),
    code("""
df = add_clinical_bins(add_engineered_features(clean_data(raw)))
print(f"Rows after cleaning: {len(df):,}  (removed {len(raw) - len(df):,})")
print(f"Diabetes prevalence: {df[TARGET].mean():.2%}")
df.describe().T
"""),
    md("## 2. Target distribution"),
    code("""
fig, ax = plt.subplots(figsize=(6, 4))
counts = df[TARGET].value_counts().sort_index()
ax.bar(["No diabetes", "Diabetes"], counts.values, color=["#4C72B0", "#DD8452"])
for i, v in enumerate(counts.values):
    ax.text(i, v, f"{v:,}\\n({v / len(df):.1%})", ha="center", va="bottom")
ax.set(ylabel="Patients", title="Class balance after cleaning", ylim=(0, counts.max() * 1.15))
save_fig(fig, FIGURES_DIR / "eda_target_distribution.png");
"""),
    md("""
Only **~8.8%** of patients have diabetes. Accuracy is therefore a misleading metric (predicting "no diabetes" for everyone already scores 91%), so modelling focuses on **ROC-AUC, PR-AUC, sensitivity, precision and calibration**.
"""),
    md("## 3. Laboratory markers"),
    code("""
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for col, (ax_hist, ax_box) in zip(["HbA1c_level", "blood_glucose_level"], [axes[:, 0], axes[:, 1]]):
    sns.histplot(data=df, x=col, hue=TARGET, stat="density", common_norm=False,
                 discrete=False, bins=30, ax=ax_hist, palette=["#4C72B0", "#DD8452"])
    ax_hist.set_title(f"Distribution of {col} by diabetes status")
    sns.boxplot(data=df, x=TARGET, y=col, ax=ax_box, palette=["#4C72B0", "#DD8452"], hue=TARGET, legend=False)
    ax_box.set_xticks([0, 1], ["No diabetes", "Diabetes"])
    ax_box.set(xlabel="", title=f"{col} by diabetes status")
fig.tight_layout()
save_fig(fig, FIGURES_DIR / "eda_lab_markers.png");
"""),
    code("""
df.groupby(TARGET)[["age", "bmi", "HbA1c_level", "blood_glucose_level"]].median().rename(
    index={0: "No diabetes", 1: "Diabetes"})
"""),
    md("""
Both markers are recorded at a small number of discrete values (18 HbA1c levels, 18 glucose levels), which suggests the data are partly **synthetic or rounded**. Diabetes prevalence by clinical category reveals something important:
"""),
    code("""
lab_tables = {
    col: df.groupby(col, observed=True)[TARGET].agg(patients="size", diabetes_rate="mean")
    for col in ["hba1c_cat", "glucose_cat"]
}
for col, t in lab_tables.items():
    display(t)

print("Diabetes rate when HbA1c >= 6.8:", round(df.loc[df.HbA1c_level >= 6.8, TARGET].mean(), 3))
print("Diabetes rate when glucose >= 220:", round(df.loc[df.blood_glucose_level >= 220, TARGET].mean(), 3))
"""),
    md("""
**Key insight — the label is partly determined by the lab values.**

* **No one** with HbA1c < 5.7% or glucose < 126 mg/dL is labelled diabetic.
* **Everyone** with HbA1c ≥ 6.8% or glucose ≥ 220 mg/dL is labelled diabetic.

These are essentially the clinical diagnostic thresholds, so a model with access to HbA1c and glucose is partly *re-discovering the diagnostic rule* rather than predicting future disease. This explains the very high discrimination seen later, and motivates the **"no-lab" ablation** in notebook 03, which estimates risk from demographics and history alone (a more realistic pre-screening scenario).
"""),
    md("## 4. Demographic and clinical risk factors"),
    code("""
factors = ["age_group", "bmi_cat", "hypertension", "heart_disease", "gender", "smoking_history"]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
overall = df[TARGET].mean()
for col, ax in zip(factors, axes.ravel()):
    rates = df.groupby(col, observed=True)[TARGET].mean().mul(100)
    rates.plot.bar(ax=ax, color="#DD8452", edgecolor="black")
    ax.axhline(overall * 100, ls="--", color="grey", lw=1, label=f"Overall {overall:.1%}")
    ax.set(title=f"Diabetes rate by {col}", xlabel="", ylabel="Diabetes rate (%)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)
fig.tight_layout()
save_fig(fig, FIGURES_DIR / "eda_risk_factors.png");
"""),
    md("""
* **Age** is the strongest non-lab factor: prevalence rises from **0.5%** in under-18s to **~20%** in the 60+ group. A few young diabetics appear as outliers and may represent type 1 diabetes.
* **BMI:** prevalence increases monotonically from underweight (0.8%) to obese (18%).
* **Hypertension** (28% vs 7%) and **heart disease** (32% vs 8%) roughly quadruple the diabetes rate — consistent with known cardiometabolic clustering.
* **Gender:** slightly higher in males (10.1% vs 7.9%), a modest effect.
* **Smoking:** *former* smokers have the highest rate (17%); the *No Info* group has the lowest (4.4%), likely because it is younger — a reminder that these are associations, not causal effects.
"""),
    md("## 5. Correlations"),
    code("""
num_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level", "hypertension", "heart_disease", TARGET]
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
ax.set_title("Pearson correlation of numeric features")
save_fig(fig, FIGURES_DIR / "eda_correlation_heatmap.png");
"""),
    md("""
Blood glucose (r ≈ 0.42) and HbA1c (r ≈ 0.41) have the strongest linear association with diabetes, followed by age (0.27), BMI (0.22), hypertension (0.20) and heart disease (0.17). Predictors are only weakly correlated with each other, so multicollinearity is limited among the raw features (the engineered interaction terms are, by construction, correlated with their components).

## EDA summary

1. **Clean, mostly complete data** — duplicates and the tiny "Other" gender group removed; "No Info" smoking kept as a category.
2. **Imbalanced target (8.8%)** — evaluate with PR-AUC, sensitivity/precision and calibration, not accuracy.
3. **HbA1c and glucose dominate**, and partly *define* the label — interpret headline performance with care.
4. **Age, BMI, hypertension and heart disease** are meaningful secondary risk factors and form the basis of a lab-free screening model.
"""),
])

# =============================================================================
# 02 — Data preparation
# =============================================================================
write("02_data_preparation.ipynb", [
    md("""
# 02 · Data Preparation

This notebook documents every transformation between the raw CSV and the model input, and shows how the pipeline avoids **data leakage**. All logic lives in [`src/data_preparation.py`](../src/data_preparation.py) and [`src/preprocessing.py`](../src/preprocessing.py) so notebooks, tests and any future scripts share one implementation.
"""),
    code(SETUP),
    md("## 1. Cleaning"),
    code("""
from src.data_preparation import (
    load_raw_data, clean_data, add_engineered_features, split_data,
    CONTINUOUS_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, MODEL_FEATURES,
)

raw = load_raw_data()
clean = clean_data(raw)

log = pd.DataFrame([
    ("Raw data", len(raw)),
    ("Drop exact duplicates", len(raw.drop_duplicates())),
    ("Drop gender == 'Other' (18 rows, 0 positives)", len(clean)),
], columns=["step", "rows"])
log["removed"] = -log["rows"].diff().fillna(0).astype(int)
log
"""),
    md("""
## 2. Feature engineering

Three features engineered during the original capstone are kept as model inputs. They are **row-wise** transformations, so they cannot leak information across partitions:

| Feature | Definition | Rationale |
|---|---|---|
| `age_bmi` | age × BMI | Older *and* heavier patients may carry compounded risk |
| `glucose_hba1c` | glucose × HbA1c | Combined short- and long-term glycaemic burden |
| `prediabetes_flag` | 1 if 5.7 ≤ HbA1c < 6.5 | Clinical prediabetes range |

Clinically labelled bins (`hba1c_cat`, `bmi_cat`, `glucose_cat`, `age_group`) are used **only for EDA and subgroup analysis**, not as model inputs.
"""),
    code("""
df = add_engineered_features(clean)
df[["age", "bmi", "age_bmi", "HbA1c_level", "blood_glucose_level", "glucose_hba1c", "prediabetes_flag", TARGET]].head()
"""),
    md("""
## 3. Train / validation / test split

The original capstone used a single 80/20 train/test split and reported results on the test set used for model choice. Here a **three-way stratified split** is used:

* **Train (70%)** — fit models, preprocessing and SMOTE.
* **Validation (15%)** — compare models, choose the decision threshold.
* **Test (15%)** — touched **once**, for the final unbiased estimate.
"""),
    code("""
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

pd.DataFrame({
    "rows": [len(X_train), len(X_val), len(X_test)],
    "diabetes_cases": [y_train.sum(), y_val.sum(), y_test.sum()],
    "prevalence": [y_train.mean(), y_val.mean(), y_test.mean()],
}, index=["train", "validation", "test"])
"""),
    md("""
## 4. Preprocessing pipeline

| Feature group | Columns | Steps |
|---|---|---|
| Continuous | age, bmi, HbA1c, glucose, age_bmi, glucose_hba1c | IQR outlier capping (1.5 × IQR) → standard scaling |
| Binary | hypertension, heart_disease, prediabetes_flag | passthrough |
| Categorical | gender, smoking_history | one-hot encoding (first level dropped, unseen levels ignored) |

The IQR capper is implemented as a scikit-learn transformer (`IQRCapper`), so the clipping limits are **learned from training data only** — and re-learned inside each cross-validation fold.
"""),
    code("""
from src.preprocessing import build_preprocessor, transformed_frame

preprocessor = build_preprocessor().fit(X_train)
capper = preprocessor.named_transformers_["num"].named_steps["capper"]
pd.DataFrame({"lower_cap": capper.lower_, "upper_cap": capper.upper_}, index=CONTINUOUS_FEATURES)
"""),
    code("""
X_train_t = transformed_frame(preprocessor, X_train)
print("Transformed shape:", X_train_t.shape)
X_train_t.describe().T[["mean", "std", "min", "max"]]
"""),
    md("""
## 5. Handling class imbalance

Three strategies are compared in notebook 03:

1. **Unweighted** — the model sees the natural 8.8% prevalence.
2. **Class-weighted** — errors on the minority class are up-weighted (`class_weight="balanced"`).
3. **SMOTE** — synthetic minority samples are generated.

SMOTE is placed **inside an `imblearn` pipeline**, after preprocessing and before the classifier. This guarantees synthetic samples are generated from training rows only, and that validation and test sets keep the real-world prevalence.
"""),
    code("""
from src.modeling import build_pipeline
build_pipeline("Gradient Boosting", "SMOTE")
"""),
    md("""
## Leakage checklist

- [x] Duplicates removed **before** splitting (no identical patient in train and test).
- [x] Engineered features are row-wise only.
- [x] Outlier caps, scaling and encodings fitted on training data only (inside the pipeline).
- [x] SMOTE applied to training data only (inside the pipeline).
- [x] Model and threshold selection use the validation set; the test set is used once.
- [ ] **Caveat:** HbA1c and glucose are diagnostic criteria for the label itself — addressed with the no-lab ablation in notebook 03.
"""),
])

# =============================================================================
# 03 — Model development
# =============================================================================
write("03_model_development.ipynb", [
    md("""
# 03 · Model Development & Evaluation

1. Compare 3 algorithms × 3 imbalance strategies (+ a baseline) on the **validation** set.
2. Calibrate the best model's probabilities.
3. Choose a clinically motivated decision threshold on validation.
4. Evaluate **once** on the held-out test set.
5. Stratify patients into Low / Moderate / High risk groups.
6. Ablation: how well can we do **without laboratory values**?
"""),
    code(SETUP),
    code("""
import json
import joblib

from src.data_preparation import prepare_dataset, split_data, MODEL_FEATURES, LAB_FEATURES
from src.modeling import compare_models, build_pipeline, calibrate, select_threshold
from src.evaluation import (
    compute_metrics, risk_group_table, plot_roc_pr, plot_calibration,
    plot_confusion, plot_threshold_tradeoff, plot_risk_stratification,
)

df = prepare_dataset()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
print(f"train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")
"""),
    md("""
## 1. Model comparison (validation set)

| Model | Why it is included |
|---|---|
| Logistic Regression | Interpretable baseline; coefficients map to odds ratios |
| Random Forest | Captures non-linearities and interactions (best model in the original capstone) |
| Gradient Boosting (`HistGradientBoostingClassifier`) | Typically the strongest learner on tabular data |

Models are **ranked by PR-AUC** (average precision), which focuses on how well the minority class is ranked and is more informative than ROC-AUC under imbalance. Threshold-based metrics below use the default 0.5 cut-off.
"""),
    code("""
results, fitted = compare_models(X_train, y_train, X_val, y_val)
cols = ["model", "strategy", "roc_auc", "pr_auc", "brier", "sensitivity", "specificity", "precision", "f1"]
results[cols]
"""),
    code("""
plot_df = results[results.model != "Baseline (prior)"]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, metric in zip(axes, ["pr_auc", "brier"]):
    sns.barplot(data=plot_df, x="model", y=metric, hue="strategy", ax=ax)
    ax.set(title=f"Validation {metric.upper().replace('_', '-')}", xlabel="")
    if metric == "pr_auc":
        ax.set_ylim(0.6, 1)
fig.tight_layout()
save_fig(fig, FIGURES_DIR / "validation_model_comparison.png");
"""),
    md("""
**Observations**

* All models are far better than the prevalence baseline (PR-AUC ≈ 0.09).
* **Gradient boosting** gives the best ranking of diabetic patients, with Random Forest close behind; logistic regression lags because the label depends on lab thresholds in a non-linear way.
* **Class weighting and SMOTE barely change ranking (ROC/PR-AUC)** but shift predicted probabilities upwards — they raise sensitivity at 0.5 at the cost of precision and *calibration* (higher Brier score). The same trade-off can be achieved more transparently by **moving the threshold** on a well-calibrated model, which is the approach taken below.
"""),
    md("## 2. Calibrated final model"),
    code("""
best = results.iloc[0]
best_name, best_strategy = best["model"], best["strategy"]
print(f"Best by validation PR-AUC: {best_name} ({best_strategy})")

# Isotonic calibration with 5-fold CV on the training set
final_model = calibrate(build_pipeline(best_name, best_strategy), X_train, y_train)
uncalibrated = fitted[(best_name, best_strategy)]

p_val_raw = uncalibrated.predict_proba(X_val)[:, 1]
p_val = final_model.predict_proba(X_val)[:, 1]

pd.DataFrame({
    "uncalibrated": compute_metrics(y_val, p_val_raw),
    "calibrated": compute_metrics(y_val, p_val),
}).loc[["roc_auc", "pr_auc", "brier"]]
"""),
    code("""
fig = plot_calibration({"Uncalibrated": p_val_raw, "Calibrated (isotonic)": p_val}, y_val)
save_fig(fig, FIGURES_DIR / "validation_calibration_curves.png");
"""),
    md("""
## 3. Choosing the decision threshold

For a **screening** use case, missing a diabetic patient (false negative) is usually costlier than an extra confirmatory test (false positive). The operating threshold is therefore the **highest threshold that keeps validation sensitivity ≥ 90%**. The F1-optimal threshold is reported for comparison.
"""),
    code("""
thr_screen = select_threshold(y_val, p_val, strategy="recall", min_recall=0.90)
thr_f1 = select_threshold(y_val, p_val, strategy="f1")

pd.DataFrame({
    "default (0.50)": compute_metrics(y_val, p_val, 0.5),
    f"F1-optimal ({thr_f1:.2f})": compute_metrics(y_val, p_val, thr_f1),
    f"Screening ({thr_screen:.2f})": compute_metrics(y_val, p_val, thr_screen),
}).loc[["threshold", "sensitivity", "specificity", "precision", "f1", "balanced_accuracy", "tp", "fp", "fn"]]
"""),
    code("""
fig = plot_threshold_tradeoff(y_val, p_val, chosen=thr_screen)
save_fig(fig, FIGURES_DIR / "validation_threshold_tradeoff.png");
"""),
    md("""
## 4. Final evaluation on the held-out test set

The test set is used **once**, with the model, calibration and threshold all fixed in advance.
"""),
    code("""
p_test = final_model.predict_proba(X_test)[:, 1]
test_metrics = compute_metrics(y_test, p_test, thr_screen)
pd.Series(test_metrics).to_frame("test")
"""),
    code("""
lr = fitted[("Logistic Regression", "unweighted")]
rf = fitted[("Random Forest", "unweighted")]
fig = plot_roc_pr({
    f"{best_name} (calibrated)": p_test,
    "Random Forest": rf.predict_proba(X_test)[:, 1],
    "Logistic Regression": lr.predict_proba(X_test)[:, 1],
}, y_test)
save_fig(fig, FIGURES_DIR / "test_roc_pr_curves.png");
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
plot_confusion(y_test, p_test, 0.5, "Default threshold (0.50)", ax=axes[0])
plot_confusion(y_test, p_test, thr_screen, f"Screening threshold ({thr_screen:.2f})", ax=axes[1])
save_fig(fig, FIGURES_DIR / "test_confusion_matrices.png");
"""),
    md("""
At the screening threshold the model catches roughly 9 in 10 diabetic patients. The price is more false positives — acceptable for a *first-line* screen where a positive leads to a cheap confirmatory test, not a diagnosis.

## 5. Risk stratification

Calibrated probabilities can be grouped into actionable tiers:

| Risk group | Predicted probability | Suggested action |
|---|---|---|
| Low | p < 2% | Routine care |
| Moderate | 2% ≤ p < 20% | Lifestyle advice, periodic re-testing |
| High | p ≥ 20% | Confirmatory diagnostic testing |
"""),
    code("""
CUTOFFS = (0.02, 0.20)
risk_table = risk_group_table(y_test, p_test, CUTOFFS)
risk_table
"""),
    code("""
fig = plot_risk_stratification(risk_table)
save_fig(fig, FIGURES_DIR / "test_risk_stratification.png");
"""),
    md("""
Observed diabetes rates rise sharply across tiers and closely match the mean predicted risk in each — evidence that the calibrated probabilities are trustworthy at group level.

## 6. Ablation: a lab-free screening model

Because HbA1c and glucose partly define the label (notebook 01), we retrain the best configuration **without** `HbA1c_level`, `blood_glucose_level`, `glucose_hba1c` and `prediabetes_flag`. This answers a more realistic question: *who should be sent for a blood test in the first place?*
"""),
    code("""
from src.data_preparation import CONTINUOUS_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES
from src.preprocessing import build_preprocessor

nolab_features = [f for f in MODEL_FEATURES if f not in LAB_FEATURES]
nolab_pre = build_preprocessor(
    continuous_features=[f for f in CONTINUOUS_FEATURES if f in nolab_features],
    binary_features=[f for f in BINARY_FEATURES if f in nolab_features],
    categorical_features=CATEGORICAL_FEATURES,
)
nolab_model = calibrate(build_pipeline(best_name, best_strategy, preprocessor=nolab_pre),
                        X_train[nolab_features], y_train)

p_val_nolab = nolab_model.predict_proba(X_val[nolab_features])[:, 1]
thr_nolab = select_threshold(y_val, p_val_nolab, strategy="recall", min_recall=0.90)
p_test_nolab = nolab_model.predict_proba(X_test[nolab_features])[:, 1]

ablation = pd.DataFrame({
    "with lab values": test_metrics,
    "without lab values": compute_metrics(y_test, p_test_nolab, thr_nolab),
}).loc[["roc_auc", "pr_auc", "brier", "threshold", "sensitivity", "specificity", "precision"]]
print("Features used without labs:", nolab_features)
ablation
"""),
    code("""
fig = plot_roc_pr({"With lab values": p_test, "Without lab values": p_test_nolab}, y_test)
fig.suptitle("Ablation: removing HbA1c & glucose (test set)", y=1.02)
save_fig(fig, FIGURES_DIR / "test_ablation_no_lab.png");
"""),
    md("""
Without lab values, discrimination drops substantially but remains well above chance, driven by age, BMI, hypertension and heart disease. To reach 90% sensitivity the lab-free model must flag many more patients — consistent with its role as a cheap **pre-screen** that decides who gets a blood test, after which the full model applies.

## 7. Save artefacts
"""),
    code("""
MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(final_model, MODELS_DIR / "final_model.joblib")
joblib.dump(uncalibrated, MODELS_DIR / "best_pipeline_uncalibrated.joblib")
joblib.dump(fitted[("Logistic Regression", "unweighted")], MODELS_DIR / "logistic_regression.joblib")
joblib.dump(nolab_model, MODELS_DIR / "no_lab_model.joblib")

summary = {
    "best_model": f"{best_name} ({best_strategy}) + isotonic calibration",
    "screening_threshold": round(thr_screen, 4),
    "risk_cutoffs": list(CUTOFFS),
    "test_metrics": {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in test_metrics.items()},
    "test_metrics_no_lab": {k: round(float(v), 4) for k, v in ablation["without lab values"].items()},
    "no_lab_features": nolab_features,
    "risk_groups_test": risk_table.round(4).reset_index().astype({"group": str}).to_dict(orient="records"),
}
(REPORTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
results[cols].round(4).to_csv(REPORTS_DIR / "validation_model_comparison.csv", index=False)
print(json.dumps(summary, indent=2))
"""),
])

# =============================================================================
# 04 — Explainability
# =============================================================================
write("04_explainability.ipynb", [
    md("""
# 04 · Explainability & Subgroup Analysis

A good risk model should be **accurate, calibrated and understandable**. This notebook explains *why* the model predicts what it does, at global and individual level, and checks that performance holds across patient subgroups.

* **Logistic regression odds ratios** — a transparent reference model.
* **Permutation importance** — model-agnostic, on the original predictors.
* **SHAP** — global and local explanations of the gradient-boosting model.
* **Subgroup performance** — by gender, age group and BMI category.
"""),
    code(SETUP),
    code("""
import json
import joblib
import shap

from src.data_preparation import prepare_dataset, split_data, add_clinical_bins
from src.evaluation import compute_metrics
from src.explainability import logistic_odds_ratios, permutation_importance_table, shap_explanation

df = prepare_dataset()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

final_model = joblib.load(MODELS_DIR / "final_model.joblib")
tree_pipeline = joblib.load(MODELS_DIR / "best_pipeline_uncalibrated.joblib")
lr_pipeline = joblib.load(MODELS_DIR / "logistic_regression.joblib")
summary = json.loads((REPORTS_DIR / "metrics.json").read_text())
THRESHOLD = summary["screening_threshold"]
p_test = final_model.predict_proba(X_test)[:, 1]
print(summary["best_model"], "| screening threshold:", THRESHOLD)
"""),
    md("""
## 1. Logistic regression odds ratios

Continuous features are standardised, so each odds ratio is the multiplicative change in the odds of diabetes for a **one-standard-deviation increase**, holding the other features fixed. Interaction terms (`glucose_hba1c`, `age_bmi`) overlap heavily with their components, so individual signs should be read cautiously — they are *partial* effects, not clinical causal claims.
"""),
    code("""
odds = logistic_odds_ratios(lr_pipeline)
odds
"""),
    code("""
fig, ax = plt.subplots(figsize=(8, 6))
o = odds.sort_values("coefficient")
ax.barh(o["feature"], o["odds_ratio"], color=np.where(o["coefficient"] > 0, "#DD8452", "#4C72B0"))
ax.axvline(1, color="black", lw=1)
ax.set_xscale("log")
ax.set(xlabel="Odds ratio (log scale)", title="Logistic regression odds ratios")
fig.tight_layout()
save_fig(fig, FIGURES_DIR / "lr_odds_ratios.png");
"""),
    md("""
## 2. Permutation importance (test set)

How much does test PR-AUC drop when each **original** predictor is randomly shuffled? This is model-agnostic and measured on unseen data.
"""),
    code("""
perm = permutation_importance_table(final_model, X_test, y_test, n_repeats=5)
fig, ax = plt.subplots(figsize=(8, 5))
p = perm.sort_values("importance_mean")
ax.barh(p["feature"], p["importance_mean"], xerr=p["importance_std"], color="#4C72B0")
ax.set(xlabel="Drop in PR-AUC when shuffled", title="Permutation importance (test set)")
fig.tight_layout()
save_fig(fig, FIGURES_DIR / "permutation_importance_test.png")
perm
"""),
    md("""
## 3. SHAP explanations

SHAP values decompose each prediction into additive contributions from each feature (in log-odds units). They are computed on the uncalibrated gradient-boosting pipeline; isotonic calibration is a monotonic re-mapping of its scores, so the ranking and the direction of each feature's effect are unchanged.
"""),
    code("""
X_explain = X_test.sample(2000, random_state=42)
sv = shap_explanation(tree_pipeline, X_train, X_explain)

shap.plots.beeswarm(sv, max_display=14, show=False)
plt.title("SHAP summary (2,000 test patients)")
save_fig(plt.gcf(), FIGURES_DIR / "shap_summary.png")
plt.show()
"""),
    code("""
shap.plots.bar(sv, max_display=14, show=False)
plt.title("Mean |SHAP| — global feature importance")
save_fig(plt.gcf(), FIGURES_DIR / "shap_global_importance.png")
plt.show()
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, feat in zip(axes, ["HbA1c_level", "age"]):
    shap.plots.scatter(sv[:, feat], color=sv[:, "blood_glucose_level"] if feat == "HbA1c_level" else sv[:, "bmi"],
                       ax=ax, show=False)
    ax.set_title(f"SHAP dependence: {feat} (standardised)")
fig.tight_layout()
save_fig(fig, FIGURES_DIR / "shap_dependence.png");
"""),
    md("""
**Reading the SHAP plots**

* **Glycaemic markers dominate.** The engineered `glucose_hba1c` interaction is the single most influential feature, followed by `HbA1c_level` itself; together with glucose they account for most of the model's output — matching permutation importance.
* The HbA1c dependence plot shows **step changes at the clinical thresholds**: low values push strongly towards "no diabetes" and values ≥ ~6.6% push strongly towards "diabetes" — the model has learned the diagnostic cut-offs.
* **Age and BMI** act mainly through the `age_bmi` interaction (third most important); age alone shows a clear rising effect from childhood to middle age.
* `hypertension`, `heart_disease`, `gender` and `smoking_history` contribute small adjustments once the other factors are known. The `smoking_history = no info` level matters slightly more than any real smoking level, likely because it acts as a proxy for younger patients.

## 4. Local explanations — three example patients
"""),
    code("""
p_explain = final_model.predict_proba(X_explain)[:, 1]
examples = {
    "low": int(np.argmin(np.abs(p_explain - 0.01))),
    "moderate": int(np.argmin(np.abs(p_explain - 0.10))),
    "high": int(np.argmin(np.abs(p_explain - 0.70))),
}
for label, i in examples.items():
    print(f"{label:>8}: calibrated risk = {p_explain[i]:.1%}")
    display(X_explain.iloc[[i]])
"""),
    code("""
for label, i in examples.items():
    shap.plots.waterfall(sv[i], max_display=10, show=False)
    plt.title(f"{label.capitalize()}-risk patient (calibrated risk {p_explain[i]:.1%})")
    save_fig(plt.gcf(), FIGURES_DIR / f"shap_waterfall_{label}_risk.png")
    plt.show()
"""),
    md("""
Waterfall plots start at the model's average output and show how each feature pushes this patient's score up (red) or down (blue). Feature values are shown on the standardised scale used by the model. These explanations are what a clinician would see alongside a risk score.

## 5. Subgroup performance

A model can look good overall but under-perform for particular groups. Metrics are computed on the test set at the fixed screening threshold.
"""),
    code("""
test = add_clinical_bins(X_test.assign(**{TARGET: y_test.values}))
test["p"] = p_test

rows = []
for col in ["gender", "age_group", "bmi_cat"]:
    for level, g in test.groupby(col, observed=True):
        if g[TARGET].nunique() < 2:
            continue
        m = compute_metrics(g[TARGET], g["p"], THRESHOLD)
        rows.append({"variable": col, "group": str(level), "n": len(g), "prevalence": g[TARGET].mean(),
                     **{k: m[k] for k in ["roc_auc", "pr_auc", "sensitivity", "specificity", "precision"]}})
subgroups = pd.DataFrame(rows)
subgroups
"""),
    code("""
long = subgroups.melt(id_vars=["variable", "group"], value_vars=["roc_auc", "sensitivity", "specificity"],
                      var_name="metric")
g = sns.catplot(data=long, x="group", y="value", hue="metric", col="variable", kind="bar",
                sharex=False, height=4, aspect=1.1)
g.set_titles("{col_name}")
g.set(ylim=(0.5, 1.02), xlabel="", ylabel="")
g.figure.suptitle("Test performance by subgroup (screening threshold)", y=1.04)
save_fig(g.figure, FIGURES_DIR / "subgroup_performance_test.png");
"""),
    md("""
**Subgroup findings**

* Discrimination (ROC-AUC) is consistently high (≥ 0.95) across genders, age groups and BMI categories.
* **Sensitivity is lower in younger and leaner patients** (e.g. 18–39 and normal-BMI groups), where diabetes is rare. Under-18s have only ~10 test cases, so their estimates are very noisy.
* **Specificity is lower in older (60+) and obese patients**, who more often have borderline lab values — the model flags more of them for follow-up.
* Males have somewhat higher sensitivity and lower specificity than females, in line with their higher prevalence.

These patterns suggest group-specific thresholds or additional features may be needed before any deployment, and that results for small subgroups should be validated on larger, external data.

## Conclusions

1. HbA1c and glucose dominate predictions, followed by age, BMI and cardiovascular comorbidities — consistent with clinical knowledge and with the EDA.
2. Explanations are stable across global (SHAP, permutation) and transparent (logistic regression) methods.
3. Performance is broadly consistent across subgroups, with known gaps in younger and leaner patients.
"""),
])

# =============================================================================
# 05 — External validation on NHANES
# =============================================================================
write("05_external_validation_nhanes.ipynb", [
    md("""
# 05 · External Validation on NHANES 2021–2023

Every result so far comes from one Kaggle dataset with unclear provenance, where the label partly follows the lab values. This notebook asks the question that matters for real use: **do the models still work on real, independently collected patients?**

**Validation data:** [NHANES 2021–2023](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023) (CDC / NCHS). It is a nationally representative US survey that **measures HbA1c and fasting plasma glucose** and asks whether a doctor has diagnosed diabetes. The models are applied **exactly as trained on Kaggle**: no refitting, same thresholds, same risk cut-offs.

**Two outcome definitions**

| Outcome | Definition | Why |
|---|---|---|
| `diabetes_diagnosed` | "Has a doctor ever told you that you have diabetes?" = yes | Independent of the lab values, so there is no circularity |
| `diabetes_total` | Diagnosed **or** HbA1c ≥ 6.5% **or** fasting glucose ≥ 126 mg/dL | ADA definition (diagnosed + undiagnosed), closest to how the Kaggle label behaves |
"""),
    code(SETUP),
    code("""
import json
import joblib
from sklearn.metrics import roc_auc_score

from src.data_preparation import prepare_dataset, add_engineered_features, MODEL_FEATURES
from src.evaluation import compute_metrics, risk_group_table, plot_roc_pr, plot_calibration
from src.nhanes import download_nhanes, load_nhanes_raw, harmonise_nhanes, build_cohorts

download_nhanes()                      # skips files that are already present
nhanes = harmonise_nhanes(load_nhanes_raw())
cohorts = build_cohorts(nhanes)
lab = add_engineered_features(cohorts["lab"])
no_lab = add_engineered_features(cohorts["no_lab"])

final_model = joblib.load(MODELS_DIR / "final_model.joblib")
no_lab_model = joblib.load(MODELS_DIR / "no_lab_model.joblib")
summary = json.loads((REPORTS_DIR / "metrics.json").read_text())
THR_LAB = summary["screening_threshold"]
THR_NOLAB = summary["test_metrics_no_lab"]["threshold"]
NOLAB_FEATURES = summary["no_lab_features"]
CUTOFFS = tuple(summary["risk_cutoffs"])
print(f"Thresholds from Kaggle validation: full model {THR_LAB:.3f}, lab-free model {THR_NOLAB:.3f}")
"""),
    md("""
## 1. Harmonisation and cohorts

| Kaggle column | NHANES source |
|---|---|
| `gender` | `RIAGENDR` |
| `age` | `RIDAGEYR` (top-coded at 80, like Kaggle) |
| `hypertension` | `BPQ020` ever told high blood pressure |
| `heart_disease` | `MCQ160B/C/E` heart failure, coronary heart disease or heart attack |
| `smoking_history` | `SMQ020` / `SMQ040` → never / former / current |
| `bmi` | `BMXBMI` |
| `HbA1c_level` | `LBXGH` |
| `blood_glucose_level` | `LBXGLU` fasting plasma glucose (morning fasting subsample only) |

Only adults (≥ 18) are included. Missing values are not imputed; each model is evaluated on the complete cases it needs.
"""),
    code("""
flow = pd.DataFrame({
    "adults": [len(nhanes), nhanes["diabetes_diagnosed"].mean(), nhanes["diabetes_total"].mean()],
    "lab-free cohort (complete non-lab features)": [len(no_lab), no_lab["diabetes_diagnosed"].mean(), no_lab["diabetes_total"].mean()],
    "lab cohort (+ HbA1c & fasting glucose)": [len(lab), lab["diabetes_diagnosed"].mean(), lab["diabetes_total"].mean()],
}, index=["n", "diagnosed prevalence", "total prevalence"]).T
flow["n"] = flow["n"].astype(int)
flow
"""),
    md("## 2. How different are the populations?"),
    code("""
kaggle = prepare_dataset()
kaggle_adults = kaggle[kaggle.age >= 18]

def profile(d, label):
    return pd.Series({
        "diabetes prevalence": d[label].mean(),
        "median age": d["age"].median(),
        "median BMI": d["bmi"].median(),
        "median HbA1c": d["HbA1c_level"].median(),
        "median glucose": d["blood_glucose_level"].median(),
        "hypertension": d["hypertension"].mean(),
        "heart disease": d["heart_disease"].mean(),
        "current smoker": (d["smoking_history"] == "current").mean(),
        "smoking 'no info'": (d["smoking_history"] == "no info").mean(),
        "distinct HbA1c values": d["HbA1c_level"].nunique(),
    })

pd.DataFrame({
    "Kaggle (adults)": profile(kaggle_adults, TARGET),
    "NHANES lab cohort": profile(lab, "diabetes_total"),
}).round(3)
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, col, rng in zip(axes, ["HbA1c_level", "blood_glucose_level"], [(3.5, 10), (60, 300)]):
    bins = np.linspace(*rng, 50)
    ax.hist(kaggle_adults[col], bins=bins, density=True, alpha=0.55, label="Kaggle")
    ax.hist(lab[col].clip(*rng), bins=bins, density=True, alpha=0.55, label="NHANES")
    ax.set(title=f"{col}: Kaggle vs NHANES", xlabel=col, ylabel="Density")
    ax.legend()
fig.tight_layout()
save_fig(fig, FIGURES_DIR / "external_lab_distributions.png");
"""),
    md("""
* NHANES lab values are **continuous**. Kaggle's take only a handful of discrete levels, which fits the suspicion that the Kaggle data are synthetic or heavily rounded.
* NHANES adults are **older and heavier**, and report **much more hypertension**. Diabetes is about twice as common.
* NHANES glucose is **fasting**. The Kaggle glucose readings run higher and are probably random (non-fasting) measurements, so the glucose feature means something slightly different in the two datasets.

Shifts like these are exactly what external validation is meant to stress-test.

## 3. Full model (with lab values)
"""),
    code("""
p_lab = final_model.predict_proba(lab[MODEL_FEATURES])[:, 1]

def evaluate(y, p, thr, w=None):
    m = compute_metrics(y, p, thr)
    m["roc_auc_weighted"] = roc_auc_score(y, p, sample_weight=w) if w is not None else np.nan
    m["mean_predicted"] = float(np.mean(p))
    m["observed_prevalence"] = float(np.mean(y))
    return m

KEYS = ["roc_auc", "roc_auc_weighted", "pr_auc", "brier", "sensitivity", "specificity",
        "precision", "observed_prevalence", "mean_predicted"]
kaggle_test = pd.Series(summary["test_metrics"])
lab_results = pd.DataFrame({
    "Kaggle test (reference)": kaggle_test,
    "NHANES: diagnosed": evaluate(lab["diabetes_diagnosed"], p_lab, THR_LAB, lab["weight_fasting"]),
    "NHANES: total (ADA)": evaluate(lab["diabetes_total"], p_lab, THR_LAB, lab["weight_fasting"]),
}).reindex(KEYS)
lab_results
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
for label, color in [("diabetes_diagnosed", "#4C72B0"), ("diabetes_total", "#DD8452")]:
    y = lab[label]
    fpr, tpr, _ = roc_curve(y, p_lab)
    axes[0].plot(fpr, tpr, color=color, label=f"{label} (AUC={roc_auc_score(y, p_lab):.3f})")
    prec, rec, _ = precision_recall_curve(y, p_lab)
    axes[1].plot(rec, prec, color=color, label=f"{label} (AP={average_precision_score(y, p_lab):.3f})")
    axes[1].axhline(y.mean(), color=color, ls=":", lw=1)
axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
axes[0].set(xlabel="False positive rate", ylabel="Sensitivity", title="ROC — full model on NHANES")
axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall — full model on NHANES")
for a in axes:
    a.legend(loc="lower right" if a is axes[0] else "upper right", fontsize=9)
fig.tight_layout()
save_fig(fig, FIGURES_DIR / "external_full_model_roc_pr.png");
"""),
    md("""
**Reading the results**

* Discrimination drops from 0.977 on Kaggle to **0.89 for total (ADA) diabetes** and **0.86 for self-reported diagnosis**. Survey-weighted AUCs are slightly higher.
* At the Kaggle threshold, sensitivity falls from 88% to **74%**, while specificity and precision stay high.

So why does a model that looked near-perfect on Kaggle lose ground on real patients?

### The glucose-126 artefact
In the Kaggle data, **not a single diabetic patient has glucose below 126 mg/dL**. The model therefore learned "glucose < 126 ⇒ not diabetic" as a hard rule. Real patients don't follow it: many people with diabetes have fasting glucose below 126, especially when it is treated or diagnosed through HbA1c.
"""),
    code("""
print("Kaggle diabetics with glucose < 126:",
      int(((kaggle[TARGET] == 1) & (kaggle["blood_glucose_level"] < 126)).sum()))

lab_scored = lab.assign(p=p_lab, glucose_band=np.where(lab["blood_glucose_level"] >= 126, "glucose >= 126", "glucose < 126"))
pd.concat({
    outcome: lab_scored[lab_scored[outcome] == 1].groupby("glucose_band").agg(
        cases=("p", "size"),
        median_HbA1c=("HbA1c_level", "median"),
        median_predicted_risk=("p", "median"),
        sensitivity=("p", lambda s: (s >= THR_LAB).mean()),
    )
    for outcome in ["diabetes_diagnosed", "diabetes_total"]
})
"""),
    md("""
Among NHANES diabetics with fasting glucose < 126, the median predicted risk is **0%** and only about a third are caught. Above 126, more than 90% are caught. The model is not wrong about HbA1c. It has learned a rule that holds only in the synthetic data.

### Benchmark: is the model better than a single lab test?
"""),
    code("""
benchmarks = pd.DataFrame({
    outcome: {
        "Full model (Kaggle-trained)": roc_auc_score(lab[outcome], p_lab),
        "HbA1c alone": roc_auc_score(lab[outcome], lab["HbA1c_level"]),
        "Fasting glucose alone": roc_auc_score(lab[outcome], lab["blood_glucose_level"]),
    } for outcome in ["diabetes_diagnosed", "diabetes_total"]
})
benchmarks.round(3)
"""),
    md("""
**This is the most important result of the project.** On real patients, **HbA1c alone (AUC ≈ 0.93) beats the Kaggle-trained model (≈ 0.86–0.89)**. Training on synthetic data with built-in thresholds made the model *worse* than the single test it learned from. Without external validation, the 0.98 Kaggle AUC would have hidden this.

Treated patients, by contrast, are *not* the problem: diagnosed patients on medication are caught more often than untreated ones, because they tend to have higher recorded glucose.
"""),
    code("""
diag = lab_scored[lab_scored["diabetes_diagnosed"] == 1]
treat = diag.groupby("on_diabetes_medication").agg(
    patients=("p", "size"),
    median_HbA1c=("HbA1c_level", "median"),
    median_glucose=("blood_glucose_level", "median"),
    sensitivity=("p", lambda s: (s >= THR_LAB).mean()),
).rename(index={0: "not on medication", 1: "on insulin / tablets"})
treat
"""),
    md("""
## 4. Calibration and risk groups
"""),
    code("""
fig = plot_calibration({"Total (ADA)": p_lab}, lab["diabetes_total"])
ax = fig.axes[0]
from sklearn.calibration import calibration_curve
fp, mp = calibration_curve(lab["diabetes_diagnosed"], p_lab, n_bins=10, strategy="quantile")
ax.plot(mp, fp, "s-", label="Diagnosed")
ax.legend(loc="upper left", fontsize=9)
ax.set_title("Calibration on NHANES (full model)")
save_fig(fig, FIGURES_DIR / "external_calibration.png");
"""),
    code("""
risk_nhanes = risk_group_table(lab["diabetes_total"], p_lab, CUTOFFS)
risk_nhanes
"""),
    md("""
The probabilities **do not transfer**. The model's mean predicted risk is 12% against an observed total prevalence of 19%. Its "Low" group (predicted ≈ 0%) still contains about 5% diabetics, roughly a fifth of all cases. These are mostly the glucose < 126 patients described above. The "Moderate" group is far riskier than predicted.

The **ordering** of the groups is preserved (Low < Moderate < High), but the absolute probabilities and cut-offs would need recalibrating on local data before any use.

## 5. Lab-free model (the screening question)

The lab-free model needs only age, BMI, gender, hypertension, heart disease and smoking. It answers *"who should be sent for a blood test?"*, so it is evaluated on all adults with those fields.
"""),
    code("""
p_nolab = no_lab_model.predict_proba(no_lab[NOLAB_FEATURES])[:, 1]
nolab_results = pd.DataFrame({
    "Kaggle test (reference)": pd.Series(summary["test_metrics_no_lab"]),
    "NHANES: diagnosed": evaluate(no_lab["diabetes_diagnosed"], p_nolab, THR_NOLAB, no_lab["weight_exam"]),
    "NHANES: total (ADA)": evaluate(no_lab["diabetes_total"], p_nolab, THR_NOLAB, no_lab["weight_exam"]),
}).reindex(KEYS)
nolab_results
"""),
    code("""
fig, ax = plt.subplots(figsize=(6, 5.5))
for label, y, p in [("NHANES diagnosed", no_lab["diabetes_diagnosed"], p_nolab),
                    ("NHANES total (ADA)", no_lab["diabetes_total"], p_nolab)]:
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y, p):.3f})")
ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
ax.set(xlabel="False positive rate", ylabel="Sensitivity",
       title=f"Lab-free model on NHANES (Kaggle test AUC={summary['test_metrics_no_lab']['roc_auc']:.3f})")
ax.legend(loc="lower right")
fig.tight_layout()
save_fig(fig, FIGURES_DIR / "external_no_lab_roc.png");
"""),
    md("""
The lab-free model drops more modestly, from **0.83 on Kaggle to 0.78 on NHANES** (0.80 survey-weighted). Age, BMI, hypertension and heart disease are real-world risk factors with no synthetic cut-offs, so this signal **transfers better** than the lab-based rules.

Because NHANES adults are older and more often hypertensive, the Kaggle-tuned threshold flags most people (sensitivity 95%, specificity about 33%). The threshold would need re-tuning for a real screening programme.

## 6. Summary
"""),
    code("""
external = {
    "nhanes_cycle": "2021-2023",
    "cohort_sizes": {"lab": len(lab), "no_lab": len(no_lab)},
    "full_model": lab_results.drop(columns="Kaggle test (reference)").round(4).to_dict(),
    "no_lab_model": nolab_results.drop(columns="Kaggle test (reference)").round(4).to_dict(),
    "sensitivity_by_treatment_diagnosed": treat["sensitivity"].round(4).to_dict(),
    "auc_benchmarks": benchmarks.round(4).to_dict(),
}
(REPORTS_DIR / "metrics_nhanes.json").write_text(json.dumps(external, indent=2, default=float))
pd.DataFrame({
    "Full model — diagnosed": lab_results["NHANES: diagnosed"],
    "Full model — total (ADA)": lab_results["NHANES: total (ADA)"],
    "Lab-free — diagnosed": nolab_results["NHANES: diagnosed"],
    "Lab-free — total (ADA)": nolab_results["NHANES: total (ADA)"],
}).loc[["roc_auc", "roc_auc_weighted", "pr_auc", "sensitivity", "specificity"]].round(3)
"""),
    md("""
**Conclusions**

1. **Ranking partly transfers.** The full model keeps good discrimination on real US data (AUC 0.86–0.89), but well below its Kaggle score (0.98).
2. **The Kaggle model learned a synthetic artefact.** In Kaggle no diabetic has glucose < 126, so the model treats it as a hard rule. On NHANES this makes it miss about two thirds of diabetics below that level, and **HbA1c alone (AUC ≈ 0.93) outperforms it**.
3. **Calibration and thresholds do not transfer.** Predicted risks are too low on NHANES and would need recalibrating locally.
4. **The lab-free model generalises best** (AUC 0.83 → 0.78), because it relies on genuine risk factors rather than synthetic thresholds.
5. **Next step:** train on real data such as NHANES itself (pooling several cycles), with the Kaggle data used, if at all, only for pre-training or comparison.

**Caveats:** survey-weighted AUCs are reported alongside unweighted ones, but no full design-based variance estimation was done. NHANES fasting glucose and the Kaggle glucose measure are not identical. The data are cross-sectional.
"""),
])
