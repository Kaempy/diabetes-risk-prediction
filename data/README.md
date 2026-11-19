# Data

This project uses the **Diabetes Prediction Dataset** (100,000 records, 9 columns) published on Kaggle:

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
