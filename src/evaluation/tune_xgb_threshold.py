from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.utils.experiment import (
    XGB_DIR,
    XGB_MODEL_DIR,
    REPORT_DIR,
    TARGET_COL,
    EXP_NAME,
    HORIZON_DAYS,
    ensure_experiment_dirs,
    print_experiment_info,
)

MODEL_DIR = XGB_MODEL_DIR


def load_feature_columns():
    path = MODEL_DIR / "xgb_feature_columns.json"

    with open(path, "r") as f:
        return json.load(f)


def load_model():
    model_path = MODEL_DIR / "xgb_fire_model.json"

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    return model


def prepare_xy(df, feature_cols):
    X = df[feature_cols].astype(np.float32)
    y = df[TARGET_COL].astype(np.int8).values

    return X, y


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(np.int8)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "positive_rate_pred": float(np.mean(y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def find_threshold_for_target_recall(y_true, y_prob, target_recall):
    thresholds = np.linspace(0.01, 0.99, 99)

    candidates = []

    for th in thresholds:
        metrics = compute_metrics(y_true, y_prob, th)

        if metrics["recall"] >= target_recall:
            candidates.append(metrics)

    if not candidates:
        return None

    # Dari semua threshold yang mencapai target recall,
    # pilih yang precision-nya paling tinggi.
    best = max(candidates, key=lambda x: x["precision"])

    return best


def main():
    ensure_experiment_dirs()
    print_experiment_info()
    print("\n========== LOAD DATA ==========")

    val_df = pd.read_parquet(XGB_DIR / "val.parquet")
    test_df = pd.read_parquet(XGB_DIR / "test.parquet")

    feature_cols = load_feature_columns()
    model = load_model()

    X_val, y_val = prepare_xy(val_df, feature_cols)
    X_test, y_test = prepare_xy(test_df, feature_cols)

    print("Val :", X_val.shape)
    print("Test:", X_test.shape)

    print("\n========== PREDICT ==========")

    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    target_recalls = [0.40, 0.50, 0.60, 0.70, 0.80]

    results = {}

    print("\n========== THRESHOLD SEARCH ON VALIDATION ==========")

    for target_recall in target_recalls:
        best_val = find_threshold_for_target_recall(
            y_true=y_val,
            y_prob=val_prob,
            target_recall=target_recall,
        )

        if best_val is None:
            print(f"Target recall {target_recall:.2f}: tidak tercapai")
            continue

        th = best_val["threshold"]

        test_metrics = compute_metrics(
            y_true=y_test,
            y_prob=test_prob,
            threshold=th,
        )

        results[f"target_recall_{target_recall}"] = {
            "validation": best_val,
            "test": test_metrics,
        }

        print("\nTarget recall:", target_recall)
        print("Chosen threshold:", th)
        print("Validation:")
        print(json.dumps(best_val, indent=2))
        print("Test:")
        print(json.dumps(test_metrics, indent=2))

    output_path = REPORT_DIR / "xgb_threshold_recall_tuning.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()