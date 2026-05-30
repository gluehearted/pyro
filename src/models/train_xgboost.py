from pathlib import Path
import json
import sys
import joblib

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    classification_report,
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


def load_xgb_data():
    train_path = XGB_DIR / "train.parquet"
    val_path = XGB_DIR / "val.parquet"
    test_path = XGB_DIR / "test.parquet"

    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

    print("\n========== LOADING XGBOOST DATA ==========")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    print("Train:", train_df.shape)
    print("Val  :", val_df.shape)
    print("Test :", test_df.shape)

    return train_df, val_df, test_df


def prepare_xy(df, target_col=TARGET_COL):
    df = df.copy()

    drop_cols = ["time", target_col]

    feature_cols = [
        col for col in df.columns
        if col not in drop_cols
    ]

    X = df[feature_cols].astype(np.float32)
    y = df[target_col].astype(np.int8)

    return X, y, feature_cols


def find_best_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)

    best_idx = np.nanargmax(f1_scores)

    # precision_recall_curve menghasilkan threshold lebih pendek 1 elemen
    if best_idx >= len(thresholds):
        best_threshold = 0.5
    else:
        best_threshold = float(thresholds[best_idx])

    return best_threshold, {
        "best_f1": float(f1_scores[best_idx]),
        "precision_at_best": float(precision[best_idx]),
        "recall_at_best": float(recall[best_idx]),
    }


def evaluate_model(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(np.int8)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_pred": float(np.mean(y_pred)),
    }

    return metrics, y_pred


def main():
    ensure_experiment_dirs()    
    print_experiment_info()
    train_df, val_df, test_df = load_xgb_data()

    X_train, y_train, feature_cols = prepare_xy(train_df)
    X_val, y_val, _ = prepare_xy(val_df)
    X_test, y_test, _ = prepare_xy(test_df)

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())

    scale_pos_weight = neg / max(pos, 1)

    print("\n========== TRAIN INFO ==========")
    print("Features:", len(feature_cols))
    print("Positive:", pos)
    print("Negative:", neg)
    print("scale_pos_weight:", scale_pos_weight)

    model = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )

    print("\n========== TRAINING XGBOOST ==========")

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_train, y_train),
            (X_val, y_val),
        ],
        verbose=50,
    )

    print("\n========== VALIDATION PREDICTION ==========")

    val_prob = model.predict_proba(X_val)[:, 1]

    best_threshold, threshold_info = find_best_threshold(
        y_true=y_val.values,
        y_prob=val_prob,
    )

    print("Best threshold:", best_threshold)
    print("Threshold info:", threshold_info)

    val_metrics, val_pred = evaluate_model(
        y_true=y_val.values,
        y_prob=val_prob,
        threshold=best_threshold,
    )

    print("\n========== TEST PREDICTION ==========")

    test_prob = model.predict_proba(X_test)[:, 1]

    test_metrics, test_pred = evaluate_model(
        y_true=y_test.values,
        y_prob=test_prob,
        threshold=best_threshold,
    )

    print("\n========== VALIDATION METRICS ==========")
    print(json.dumps(val_metrics, indent=2))

    print("\n========== TEST METRICS ==========")
    print(json.dumps(test_metrics, indent=2))

    print("\n========== TEST CLASSIFICATION REPORT ==========")
    print(
        classification_report(
            y_test.values,
            test_pred,
            digits=4,
            zero_division=0,
        )
    )

    model_path = MODEL_DIR / "xgb_fire_model.json"
    feature_path = MODEL_DIR / "xgb_feature_columns.json"
    threshold_path = MODEL_DIR / "xgb_threshold.json"
    report_path = REPORT_DIR / "xgb_metrics.json"
    pred_path = REPORT_DIR / "xgb_test_predictions.parquet"

    model.save_model(model_path)

    with open(feature_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open(threshold_path, "w") as f:
        json.dump(
            {
                "threshold": best_threshold,
                **threshold_info,
            },
            f,
            indent=2,
        )

    with open(report_path, "w") as f:
        json.dump(
            {
                "experiment": EXP_NAME,
                "target": TARGET_COL,
                "horizon_days": HORIZON_DAYS,
                "validation": val_metrics,
                "test": test_metrics,
                "feature_columns": feature_cols,
                "scale_pos_weight": scale_pos_weight,
                "best_iteration": int(model.best_iteration)
                if model.best_iteration is not None
                else None,
            },
            f,
            indent=2,
        )

    test_output = test_df[["time", "latitude", "longitude", TARGET_COL]].copy()
    test_output["xgb_prob"] = test_prob.astype(np.float32)
    test_output["xgb_pred"] = test_pred.astype(np.int8)
    test_output.to_parquet(pred_path, index=False)

    print("\n========== SAVED ==========")
    print("Model      :", model_path)
    print("Features   :", feature_path)
    print("Threshold  :", threshold_path)
    print("Metrics    :", report_path)
    print("Predictions:", pred_path)


if __name__ == "__main__":
    main()