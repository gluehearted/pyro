from pathlib import Path
import sys
import json
import numpy as np

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
    REPORT_DIR,
    TARGET_COL,
    EXP_NAME,
    HORIZON_DAYS,
    ensure_experiment_dirs,
    print_experiment_info,
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)


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


def search_threshold(y_true, y_prob):
    thresholds = np.linspace(0.01, 0.99, 99)

    best_f1 = None
    recall_modes = {}

    for th in thresholds:
        metrics = compute_metrics(y_true, y_prob, th)

        if best_f1 is None or metrics["f1"] > best_f1["f1"]:
            best_f1 = metrics

        for target_recall in [0.50, 0.60, 0.70, 0.80]:
            key = f"target_recall_{target_recall}"

            if metrics["recall"] >= target_recall:
                if key not in recall_modes:
                    recall_modes[key] = metrics
                elif metrics["precision"] > recall_modes[key]["precision"]:
                    recall_modes[key] = metrics

    return best_f1, recall_modes


def main():
    ensure_experiment_dirs()
    print_experiment_info()
    cnn_val_prob_path = REPORT_DIR / "cnn_val_prob.npy"
    cnn_test_prob_path = REPORT_DIR / "cnn_test_prob.npy"

    if not cnn_val_prob_path.exists():
        raise FileNotFoundError(
            f"{cnn_val_prob_path} tidak ditemukan. Pastikan train_cnn_bilstm.py sudah selesai."
        )

    val_prob = np.load(cnn_val_prob_path).reshape(-1)
    test_prob = np.load(cnn_test_prob_path).reshape(-1)

    y_val = np.load(CNN_DIR / "y_val.npy", mmap_mode="r").reshape(-1).astype(np.int8)
    y_test = np.load(CNN_DIR / "y_test.npy", mmap_mode="r").reshape(-1).astype(np.int8)

    print("\n========== CNN THRESHOLD TUNING ==========")
    print("Val size :", len(y_val))
    print("Test size:", len(y_test))

    best_f1, recall_modes = search_threshold(y_val, val_prob)

    results = {
        "experiment": EXP_NAME,
        "target": TARGET_COL,
        "horizon_days": HORIZON_DAYS,
        "best_f1_on_validation": best_f1,
        "recall_modes_on_validation": {},
        "test_evaluation_using_selected_thresholds": {},
    }

    print("\nBest F1 threshold on validation:")
    print(json.dumps(best_f1, indent=2))

    test_best_f1 = compute_metrics(
        y_true=y_test,
        y_prob=test_prob,
        threshold=best_f1["threshold"],
    )

    results["test_evaluation_using_selected_thresholds"]["best_f1"] = test_best_f1

    for key, val_metrics in recall_modes.items():
        th = val_metrics["threshold"]

        test_metrics = compute_metrics(
            y_true=y_test,
            y_prob=test_prob,
            threshold=th,
        )

        results["recall_modes_on_validation"][key] = val_metrics
        results["test_evaluation_using_selected_thresholds"][key] = test_metrics

        print(f"\n{key}")
        print("Validation:")
        print(json.dumps(val_metrics, indent=2))
        print("Test:")
        print(json.dumps(test_metrics, indent=2))

    output_path = REPORT_DIR / "cnn_threshold_tuning.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()