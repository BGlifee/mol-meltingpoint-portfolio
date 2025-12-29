# src/config.py
from pathlib import Path

# ------------------
# Project Paths
# ------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROC = PROJECT_ROOT / "data" / "processed"
DATA_EXT = PROJECT_ROOT / "data" / "external"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "model_artifacts"

for d in [RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ------------------
# Core Settings
# ------------------
SEED = 42
N_SPLITS = 5

TARGET_COL = "Tm"
SMILES_COL = "SMILES"

DEFAULT_TAG = "default"


# ------------------
# Feature Pipeline Defaults
# ------------------
TABULAR_REDUCER = {
    "clip_z": 5.0,
    "corr_threshold": 0.95,
    "top_n": 300,
}

MORGAN_FP = {
    "n_bits": 512,
    "radius": 2,
}


# ------------------
# Optional Model Defaults
# ------------------
LGBM_DEFAULT = {
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
}

XGB_DEFAULT = {
    "n_estimators": 1200,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "max_depth": 6,
}
