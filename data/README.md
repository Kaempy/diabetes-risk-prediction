# Data

## 1. Training data: Kaggle

This project trains on the **Diabetes Prediction Dataset** (100,000 records, 9 columns) published on Kaggle:

<https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset>

The raw file is not committed to this repository. To reproduce the analysis:

1. Download `diabetes_prediction_dataset.csv` from the link above (a free Kaggle account is required), or with the Kaggle CLI:

   ```bash
   kaggle datasets download -d iammustafatz/diabetes-prediction-dataset -p data/raw --unzip
   ```

2. Make sure the file is located at:

   ```
   data/raw/diabetes_prediction_dataset.csv
   ```

| Column | Type | Description |
|---|---|---|
| `gender` | categorical | Female / Male / Other |
| `age` | numeric | Age in years |
| `hypertension` | binary | 1 = has hypertension |
| `heart_disease` | binary | 1 = has heart disease |
| `smoking_history` | categorical | never, former, current, not current, ever, No Info |
| `bmi` | numeric | Body-mass index (kg/m²) |
| `HbA1c_level` | numeric | Glycated haemoglobin (%) |
| `blood_glucose_level` | numeric | Blood glucose (mg/dL) |
| `diabetes` | binary | **Target** — 1 = diabetes |

## 2. External validation data: NHANES 2021–2023

[NHANES](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023) is the National Health and Nutrition Examination Survey, run by CDC / NCHS. Its data are public domain. Notebook 05 downloads the files automatically into `data/external/nhanes/` (about 10 MB, git-ignored) by calling `src.nhanes.download_nhanes()`.

| NHANES file | Variables | Mapped to |
|---|---|---|
| `DEMO_L` | `RIAGENDR`, `RIDAGEYR`, `WTMEC2YR` | `gender`, `age` (top-coded at 80), exam weight |
| `BMX_L` | `BMXBMI` | `bmi` |
| `GHB_L` | `LBXGH` | `HbA1c_level` |
| `GLU_L` | `LBXGLU`, `WTSAF2YR` | `blood_glucose_level` (fasting subsample), fasting weight |
| `BPQ_L` | `BPQ020` | `hypertension` (ever told high blood pressure) |
| `MCQ_L` | `MCQ160B`, `MCQ160C`, `MCQ160E` | `heart_disease` (heart failure, CHD or heart attack) |
| `SMQ_L` | `SMQ020`, `SMQ040` | `smoking_history` (never / former / current) |
| `DIQ_L` | `DIQ010`, `DIQ050`, `DIQ070` | outcome labels, diabetes medication |

The validation uses two outcome definitions:

- **`diabetes_diagnosed`:** `DIQ010 = 1`, meaning a doctor has told the person they have diabetes. Borderline answers count as no.
- **`diabetes_total`:** diagnosed, or HbA1c ≥ 6.5%, or fasting glucose ≥ 126 mg/dL. This is the ADA definition.

Only adults (18 and over) are included, and each model is evaluated on complete cases.
