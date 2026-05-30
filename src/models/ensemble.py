from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.utils.experiment import (
    CNN_DIR,
    XGB_DIR,
    MASTER_PATH,
    XGB_MODEL_DIR,
    ENSEMBLE_MODEL_DIR,
    REPORT_DIR,
    TARGET_COL,
    EXP_NAME,
    HORIZON_DAYS,
    ensure_experiment_dirs,
    print_experiment_info,
)

ENSEMBLE_DIR = ENSEMBLE_MODEL_DIR


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
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_pred": float(np.mean(y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def load_xgb_model():
    model = xgb.XGBClassifier()
    model.load_model(XGB_MODEL_DIR / "xgb_fire_model.json")
    return model


def load_feature_columns():
    with open(XGB_MODEL_DIR / "xgb_feature_columns.json", "r") as f:
        return json.load(f)


def load_master_coords():
    ds = xr.open_dataset(MASTER_PATH)

    latitudes = ds["latitude"].values
    longitudes = ds["longitude"].values

    return latitudes, longitudes


def build_xgb_probability_map(split_name, target_times, latitudes, longitudes):
    print(f"\n========== BUILD XGB PROBABILITY MAP: {split_name.upper()} ==========")

    parquet_path = XGB_DIR / f"{split_name}.parquet"
    output_path = REPORT_DIR / f"xgb_{split_name}_prob_map.npy"

    if output_path.exists():
        print(f"Cached file ditemukan: {output_path}")
        return np.load(output_path)

    model = load_xgb_model()
    feature_cols = load_feature_columns()

    df = pd.read_parquet(parquet_path)
    df["time"] = pd.to_datetime(df["time"]).dt.normalize()

    target_times = pd.to_datetime(target_times).normalize()

    df = df[df["time"].isin(set(target_times))].copy()

    print("Rows after time alignment:", len(df))

    X = df[feature_cols].astype(np.float32)

    probs = model.predict_proba(X)[:, 1].astype(np.float32)

    df["xgb_prob"] = probs

    full_index = pd.MultiIndex.from_product(
        [
            target_times,
            latitudes,
            longitudes,
        ],
        names=["time", "latitude", "longitude"]
    )

    prob_series = (
        df.set_index(["time", "latitude", "longitude"])["xgb_prob"]
        .reindex(full_index)
        .fillna(0.0)
    )

    prob_map = prob_series.values.reshape(
        len(target_times),
        len(latitudes),
        len(longitudes)
    ).astype(np.float32)

    np.save(output_path, prob_map)

    print("Saved:", output_path)

    return prob_map


def search_best_ensemble(val_y, cnn_val, xgb_val):
    print("\n========== SEARCH BEST ENSEMBLE ==========")

    alphas = np.linspace(0.0, 1.0, 11)
    thresholds = np.linspace(0.01, 0.99, 99)

    best_f1 = None
    best_recall60 = None
    best_recall70 = None

    y_true = val_y.reshape(-1).astype(np.int8)

    for alpha in alphas:
        prob = alpha * cnn_val + (1.0 - alpha) * xgb_val
        prob_flat = prob.reshape(-1)

        for th in thresholds:
            metrics = compute_metrics(y_true, prob_flat, th)
            metrics["alpha_cnn"] = float(alpha)
            metrics["alpha_xgb"] = float(1.0 - alpha)

            if best_f1 is None or metrics["f1"] > best_f1["f1"]:
                best_f1 = metrics

            if metrics["recall"] >= 0.60:
                if best_recall60 is None or metrics["precision"] > best_recall60["precision"]:
                    best_recall60 = metrics

            if metrics["recall"] >= 0.70:
                if best_recall70 is None or metrics["precision"] > best_recall70["precision"]:
                    best_recall70 = metrics

    return {
        "best_f1": best_f1,
        "early_warning_recall60": best_recall60,
        "high_sensitivity_recall70": best_recall70,
    }


def main():
    ensure_experiment_dirs()
    print_experiment_info()
    latitudes, longitudes = load_master_coords()

    cnn_val = np.load(REPORT_DIR / "cnn_val_prob.npy").astype(np.float32)
    cnn_test = np.load(REPORT_DIR / "cnn_test_prob.npy").astype(np.float32)

    y_val = np.load(CNN_DIR / "y_val.npy", mmap_mode="r").astype(np.float32)
    y_test = np.load(CNN_DIR / "y_test.npy", mmap_mode="r").astype(np.float32)

    times_val = np.load(CNN_DIR / "times_val.npy", allow_pickle=True)
    times_test = np.load(CNN_DIR / "times_test.npy", allow_pickle=True)

    xgb_val = build_xgb_probability_map(
        split_name="val",
        target_times=times_val,
        latitudes=latitudes,
        longitudes=longitudes,
    )

    xgb_test = build_xgb_probability_map(
        split_name="test",
        target_times=times_test,
        latitudes=latitudes,
        longitudes=longitudes,
    )

    print("\nCNN val :", cnn_val.shape)
    print("XGB val :", xgb_val.shape)
    print("y val   :", y_val.shape)

    search_results = search_best_ensemble(
        val_y=y_val,
        cnn_val=cnn_val,
        xgb_val=xgb_val,
    )

    print("\n========== VALIDATION ENSEMBLE SEARCH RESULT ==========")
    print(json.dumps(search_results, indent=2))

    final_modes = {}

    y_test_flat = y_test.reshape(-1).astype(np.int8)

    for mode_name, cfg in search_results.items():
        if cfg is None:
            continue

        alpha = cfg["alpha_cnn"]
        threshold = cfg["threshold"]

        test_prob = alpha * cnn_test + (1.0 - alpha) * xgb_test
        test_prob_flat = test_prob.reshape(-1)

        test_metrics = compute_metrics(
            y_true=y_test_flat,
            y_prob=test_prob_flat,
            threshold=threshold,
        )

        test_metrics["alpha_cnn"] = alpha
        test_metrics["alpha_xgb"] = 1.0 - alpha

        final_modes[mode_name] = {
            "validation": cfg,
            "test": test_metrics,
        }

        print(f"\n========== MODE: {mode_name} ==========")
        print(json.dumps(final_modes[mode_name], indent=2))

    # Default untuk risk map: pakai mode recall60 kalau ada, kalau tidak best_f1.
    selected_mode = "early_warning_recall60"
    if selected_mode not in final_modes:
        selected_mode = "best_f1"

    selected = final_modes[selected_mode]["test"]

    alpha = selected["alpha_cnn"]
    threshold = selected["threshold"]

    ensemble_test_prob = alpha * cnn_test + (1.0 - alpha) * xgb_test
    ensemble_test_pred = (ensemble_test_prob >= threshold).astype(np.int8)

    np.save(REPORT_DIR / "ensemble_test_prob.npy", ensemble_test_prob.astype(np.float32))
    np.save(REPORT_DIR / "ensemble_test_pred.npy", ensemble_test_pred.astype(np.int8))

    output = {
        "experiment": EXP_NAME,
        "target": TARGET_COL,
        "horizon_days": HORIZON_DAYS,
        "selected_mode": selected_mode,
        "all_modes": final_modes,
    }

    with open(REPORT_DIR / "ensemble_metrics.json", "w") as f:
        json.dump(output, f, indent=2)

    with open(ENSEMBLE_DIR / "ensemble_config.json", "w") as f:
        json.dump(
            {
                "experiment": EXP_NAME,
                "target": TARGET_COL,
                "horizon_days": HORIZON_DAYS,
                "selected_mode": selected_mode,
                "alpha_cnn": alpha,
                "alpha_xgb": 1.0 - alpha,
                "threshold": threshold,
            },
            f,
            indent=2,
        )

    print("\n========== ENSEMBLE COMPLETE ==========")
    print("Selected mode:", selected_mode)
    print("Saved:", REPORT_DIR / "ensemble_metrics.json")
    print("Saved:", REPORT_DIR / "ensemble_test_prob.npy")
    print("Saved:", REPORT_DIR / "ensemble_test_pred.npy")


if __name__ == "__main__":
    main()