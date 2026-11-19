import pandas as pd

from src.data_preparation import (
    MODEL_FEATURES,
    add_clinical_bins,
    add_engineered_features,
    clean_data,
    split_data,
)


def test_clean_data_standardises_text_and_drops_duplicates_and_other(raw_df):
    dirty = pd.concat([raw_df, raw_df.iloc[:5]], ignore_index=True)
    other = raw_df.iloc[:1].assign(gender="Other")
    cleaned = clean_data(pd.concat([dirty, other], ignore_index=True))

    assert not cleaned.duplicated().any()
    assert set(cleaned["gender"]) <= {"female", "male"}
    assert "no info" in set(cleaned["smoking_history"])


def test_engineered_features(raw_df):
    df = add_engineered_features(clean_data(raw_df))
    row = df.iloc[0]
    assert row["age_bmi"] == row["age"] * row["bmi"]
    assert row["glucose_hba1c"] == row["blood_glucose_level"] * row["HbA1c_level"]
    expected_flag = int(5.7 <= row["HbA1c_level"] < 6.5)
    assert row["prediabetes_flag"] == expected_flag


def test_prediabetes_flag_boundaries():
    df = pd.DataFrame({"age": [1] * 4, "bmi": [1] * 4, "blood_glucose_level": [1] * 4,
                       "HbA1c_level": [5.6, 5.7, 6.4, 6.5]})
    assert add_engineered_features(df)["prediabetes_flag"].tolist() == [0, 1, 1, 0]


def test_clinical_bins_use_clinical_cutoffs():
    df = pd.DataFrame({"HbA1c_level": [5.6, 5.7, 6.5], "bmi": [18.4, 24.9, 30],
                       "blood_glucose_level": [99, 100, 126], "age": [10, 40, 70]})
    out = add_clinical_bins(df)
    assert out["hba1c_cat"].cat.codes.tolist() == [0, 1, 2]
    assert out["bmi_cat"].astype(str).tolist() == ["underweight", "normal", "obese"]
    assert out["glucose_cat"].cat.codes.tolist() == [0, 1, 2]


def test_split_is_stratified_and_disjoint(raw_df):
    df = add_engineered_features(clean_data(raw_df))
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    assert len(X_train) + len(X_val) + len(X_test) == len(df)
    assert not set(X_train.index) & set(X_val.index)
    assert not set(X_train.index) & set(X_test.index)
    assert list(X_train.columns) == MODEL_FEATURES
    for y in (y_train, y_val, y_test):
        assert abs(y.mean() - df["diabetes"].mean()) < 0.05
