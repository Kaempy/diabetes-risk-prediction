"""Download NHANES 2021-2023 and harmonise it to the Kaggle schema for external validation.

NHANES (National Health and Nutrition Examination Survey, CDC/NCHS) measures HbA1c
and fasting plasma glucose in a nationally representative US sample and asks whether
a doctor has ever diagnosed diabetes. Data are public domain.
https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT

NHANES_DIR = PROJECT_ROOT / "data" / "external" / "nhanes"
NHANES_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/{name}.xpt"

# File -> variables used
NHANES_FILES = {
    "DEMO_L": ["SEQN", "RIAGENDR", "RIDAGEYR", "WTMEC2YR"],   # demographics, exam weight
    "BMX_L": ["SEQN", "BMXBMI"],                              # body measures
    "GHB_L": ["SEQN", "LBXGH"],                               # HbA1c (%)
    "GLU_L": ["SEQN", "LBXGLU", "WTSAF2YR"],                  # fasting glucose (mg/dL), fasting weight
    "DIQ_L": ["SEQN", "DIQ010", "DIQ050", "DIQ070"],          # diabetes questionnaire
    "BPQ_L": ["SEQN", "BPQ020"],                              # ever told high blood pressure
    "MCQ_L": ["SEQN", "MCQ160B", "MCQ160C", "MCQ160E"],       # heart failure, CHD, heart attack
    "SMQ_L": ["SEQN", "SMQ020", "SMQ040"],                    # smoking
}

NO_LAB_COLUMNS = ["gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi"]
LAB_COLUMNS = ["HbA1c_level", "blood_glucose_level"]


def download_nhanes(dest: str | Path = NHANES_DIR, overwrite: bool = False) -> Path:
    """Download the NHANES 2021-2023 XPT files used in this project (~10 MB)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in NHANES_FILES:
        path = dest / f"{name}.xpt"
        if overwrite or not path.exists():
            urllib.request.urlretrieve(NHANES_URL.format(name=name), path)
    return dest


def load_nhanes_raw(directory: str | Path = NHANES_DIR) -> pd.DataFrame:
    """Merge the NHANES component files on the respondent id (SEQN)."""
    directory = Path(directory)
    merged = None
    for name, cols in NHANES_FILES.items():
        df = pd.read_sas(directory / f"{name}.xpt", format="xport")[cols]
        merged = df if merged is None else merged.merge(df, on="SEQN", how="left")
    return merged


def _yes_no(series: pd.Series) -> pd.Series:
    """NHANES coding 1 = yes, 2 = no, 7/9 = refused / don't know -> 1 / 0 / NaN."""
    return series.map({1.0: 1.0, 2.0: 0.0})


def harmonise_nhanes(raw: pd.DataFrame, min_age: int = 18) -> pd.DataFrame:
    """Map NHANES variables onto the Kaggle column names and add outcome labels.

    Outcomes
    --------
    ``diabetes_diagnosed``  : self-reported doctor diagnosis (DIQ010 = 1). "Borderline" (3) = 0.
    ``diabetes_total``      : diagnosed OR HbA1c >= 6.5% OR fasting glucose >= 126 mg/dL
                              (ADA criteria, i.e. diagnosed + undiagnosed diabetes).
    """
    out = pd.DataFrame({"SEQN": raw["SEQN"].astype(int)})
    out["gender"] = raw["RIAGENDR"].map({1.0: "male", 2.0: "female"})
    out["age"] = raw["RIDAGEYR"].astype(float)          # top-coded at 80, as in the Kaggle data
    out["hypertension"] = _yes_no(raw["BPQ020"])

    heart = pd.concat([_yes_no(raw[c]) for c in ["MCQ160B", "MCQ160C", "MCQ160E"]], axis=1)
    out["heart_disease"] = np.where(heart.eq(1).any(axis=1), 1.0,
                                    np.where(heart.eq(0).all(axis=1), 0.0, np.nan))

    ever, now = raw["SMQ020"], raw["SMQ040"]
    out["smoking_history"] = np.select(
        [ever.eq(2), ever.eq(1) & now.isin([1.0, 2.0]), ever.eq(1) & now.eq(3)],
        ["never", "current", "former"],
        default="no info",
    )

    out["bmi"] = raw["BMXBMI"]
    out["HbA1c_level"] = raw["LBXGH"]
    out["blood_glucose_level"] = raw["LBXGLU"]

    diagnosed = raw["DIQ010"].map({1.0: 1.0, 2.0: 0.0, 3.0: 0.0})
    out["diabetes_diagnosed"] = diagnosed
    lab_positive = (raw["LBXGH"] >= 6.5) | (raw["LBXGLU"] >= 126)
    out["diabetes_total"] = np.where(diagnosed.isna(), np.nan,
                                     ((diagnosed == 1) | lab_positive).astype(float))
    out["on_diabetes_medication"] = (raw["DIQ050"].eq(1) | raw["DIQ070"].eq(1)).astype(int)

    out["weight_exam"] = raw["WTMEC2YR"]
    out["weight_fasting"] = raw["WTSAF2YR"]
    return out[out["age"] >= min_age].reset_index(drop=True)


def build_cohorts(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Complete-case cohorts for the lab-free and full (lab) models."""
    no_lab = df.dropna(subset=NO_LAB_COLUMNS + ["diabetes_diagnosed"])
    lab = no_lab.dropna(subset=LAB_COLUMNS + ["weight_fasting"])
    lab = lab[lab["weight_fasting"] > 0]
    return {"no_lab": no_lab.reset_index(drop=True), "lab": lab.reset_index(drop=True)}
