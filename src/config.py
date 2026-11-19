"""Project-wide paths and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetes_prediction_dataset.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42
TARGET = "diabetes"

# Train / validation / test proportions (stratified on the target).
VAL_SIZE = 0.15
TEST_SIZE = 0.15
