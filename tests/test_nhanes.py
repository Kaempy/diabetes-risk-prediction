import numpy as np
import pandas as pd

from src.nhanes import build_cohorts, harmonise_nhanes


def make_raw():
    """Five synthetic NHANES respondents covering the coding edge cases."""
    return pd.DataFrame({
        "SEQN": [1, 2, 3, 4, 5],
        "RIAGENDR": [1.0, 2.0, 2.0, 1.0, 2.0],
        "RIDAGEYR": [45.0, 80.0, 30.0, 60.0, 12.0],
        "WTMEC2YR": [1000.0] * 5,
        "BMXBMI": [31.2, 24.0, np.nan, 28.5, 19.0],
        "LBXGH": [7.1, 5.4, 5.9, 6.0, 5.0],
        "LBXGLU": [140.0, np.nan, 99.0, 130.0, 90.0],
        "WTSAF2YR": [2000.0, np.nan, 2000.0, 0.0, 2000.0],
        "DIQ010": [1.0, 2.0, 3.0, 2.0, 2.0],
        "DIQ050": [1.0, np.nan, np.nan, np.nan, np.nan],
        "DIQ070": [2.0, np.nan, np.nan, np.nan, np.nan],
        "BPQ020": [1.0, 2.0, 9.0, 1.0, 2.0],
        "MCQ160B": [2.0, 2.0, 2.0, 2.0, 2.0],
        "MCQ160C": [1.0, 2.0, 9.0, 2.0, 2.0],
        "MCQ160E": [2.0, 2.0, 2.0, 2.0, 2.0],
        "SMQ020": [1.0, 2.0, 1.0, 1.0, np.nan],
        "SMQ040": [3.0, np.nan, 1.0, 2.0, np.nan],
    })


def test_harmonise_maps_codes_to_kaggle_schema():
    df = harmonise_nhanes(make_raw()).set_index("SEQN")

    assert 5 not in df.index                                   # under-18 excluded
    assert df.loc[1, "gender"] == "male" and df.loc[2, "gender"] == "female"
    assert df.loc[1, "smoking_history"] == "former"
    assert df.loc[2, "smoking_history"] == "never"
    assert df.loc[3, "smoking_history"] == "current"
    assert df.loc[1, "heart_disease"] == 1 and df.loc[2, "heart_disease"] == 0
    assert np.isnan(df.loc[3, "heart_disease"])                 # unknown CHD, no positives
    assert np.isnan(df.loc[3, "hypertension"])                  # "don't know" -> missing
    assert df.loc[1, "on_diabetes_medication"] == 1


def test_outcome_definitions():
    df = harmonise_nhanes(make_raw()).set_index("SEQN")
    assert df["diabetes_diagnosed"].tolist() == [1, 0, 0, 0]    # borderline (3) counts as 0
    # respondent 4: not diagnosed but fasting glucose 130 -> undiagnosed diabetes
    assert df["diabetes_total"].tolist() == [1, 0, 0, 1]


def test_cohorts_are_complete_cases():
    cohorts = build_cohorts(harmonise_nhanes(make_raw()))
    assert set(cohorts["no_lab"]["SEQN"]) == {1, 2, 4}          # 3 lacks BMI / hypertension
    assert set(cohorts["lab"]["SEQN"]) == {1}                   # 2 no glucose, 4 zero weight
