from pathlib import Path
import json

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]

from src.utils.experiment import (
    XGB_DIR,
    XGB_MODEL_DIR,
    FIGURE_DIR,
    TARGET_COL,
    EXP_NAME,
    HORIZON_DAYS,
    ensure_experiment_dirs,
    print_experiment_info,
)

MODEL_DIR = XGB_MODEL_DIR
OUTPUT_DIR = FIGURE_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_POS = 10000
SAMPLE_NEG = 40000
RANDOM_STATE = 42


def load_model():
    model = xgb.XGBClassifier()
    model.load_model(MODEL_DIR / "xgb_fire_model.json")
    return model


def load_feature_columns():
    with open(MODEL_DIR / "xgb_feature_columns.json", "r") as f:
        return json.load(f)


def build_sample():
    train_df = pd.read_parquet(XGB_DIR / "train.parquet")

    pos = train_df[train_df[TARGET_COL] == 1]
    neg = train_df[train_df[TARGET_COL] == 0]

    pos_sample = pos.sample(
        n=min(len(pos), SAMPLE_POS),
        random_state=RANDOM_STATE
    )

    neg_sample = neg.sample(
        n=min(len(neg), SAMPLE_NEG),
        random_state=RANDOM_STATE
    )

    sample = pd.concat([pos_sample, neg_sample], ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=RANDOM_STATE)

    return sample


def main():
    ensure_experiment_dirs()
    print_experiment_info()
    print("\n========== SHAP XGBOOST ==========")

    model = load_model()
    feature_cols = load_feature_columns()

    sample = build_sample()

    X = sample[feature_cols].astype(np.float32)

    print("SHAP sample:", X.shape)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs = np.abs(shap_values).mean(axis=0)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)

    importance_path = OUTPUT_DIR / "xgb_shap_importance.csv"
    importance.to_csv(importance_path, index=False)

    print("\nTop SHAP features:")
    print(importance.head(20))

    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        show=False,
        max_display=20
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "xgb_shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        show=False,
        max_display=20
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "xgb_shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nSaved:")
    print(importance_path)
    print(OUTPUT_DIR / "xgb_shap_bar.png")
    print(OUTPUT_DIR / "xgb_shap_summary.png")


if __name__ == "__main__":
    main()