from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parents[2]

TARGET_COL = os.getenv("TARGET_COL", "fire_next_1d")

import re

match = re.search(r"fire_next_(\d+)d", TARGET_COL)

if match:
    DEFAULT_HORIZON_DAYS = int(match.group(1))
else:
    DEFAULT_HORIZON_DAYS = 1

HORIZON_DAYS = int(os.getenv("HORIZON_DAYS", DEFAULT_HORIZON_DAYS))

EXP_NAME = os.getenv(
    "EXP_NAME",
    f"split_2015_2022_val2023_test2024_{TARGET_COL}"
)

EXP_DIR = ROOT_DIR / "experiments" / EXP_NAME

MASTER_PATH = ROOT_DIR / "data" / "processed" / "master" / "master_dataset.nc"

XGB_DIR = EXP_DIR / "processed" / "xgb"
CNN_DIR = EXP_DIR / "processed" / "cnn"

XGB_MODEL_DIR = EXP_DIR / "models" / "xgboost"
CNN_MODEL_DIR = EXP_DIR / "models" / "cnn_bilstm"
ENSEMBLE_MODEL_DIR = EXP_DIR / "models" / "ensemble"

REPORT_DIR = EXP_DIR / "reports"
FIGURE_DIR = EXP_DIR / "figures"
MAP_DIR = EXP_DIR / "maps"


def ensure_experiment_dirs():
    for path in [
        XGB_DIR,
        CNN_DIR,
        XGB_MODEL_DIR,
        CNN_MODEL_DIR,
        ENSEMBLE_MODEL_DIR,
        REPORT_DIR,
        FIGURE_DIR,
        MAP_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def print_experiment_info():
    print("\n========== EXPERIMENT CONFIG ==========")
    print("EXP_NAME    :", EXP_NAME)
    print("TARGET_COL  :", TARGET_COL)
    print("HORIZON_DAYS:", HORIZON_DAYS)
    print("EXP_DIR     :", EXP_DIR)