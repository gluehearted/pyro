from __future__ import annotations

from pathlib import Path
import json
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# ROOT AND EXPERIMENT CONFIG
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

MASTER_PATH = ROOT_DIR / "data" / "processed" / "master" / "master_dataset.nc"

OUTPUT_DIR = ROOT_DIR / "outputs" / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    "1-day": {
        "name": "split_2015_2022_val2023_test2024_fire_next_1d",
        "target": "fire_next_1d",
        "horizon_days": 1,
    },
    "7-day": {
        "name": "split_2015_2022_val2023_test2024_fire_next_7d",
        "target": "fire_next_7d",
        "horizon_days": 7,
    },
    "14-day": {
        "name": "split_2015_2022_val2023_test2024_fire_next_14d",
        "target": "fire_next_14d",
        "horizon_days": 14,
    },
}

SELECTED_MODE = "early_warning_recall60"


# ============================================================
# PATH HELPERS
# ============================================================

def get_exp_dir(label: str) -> Path:
    return ROOT_DIR / "experiments" / EXPERIMENTS[label]["name"]


def get_report_path(label: str) -> Path:
    return get_exp_dir(label) / "reports" / "ensemble_metrics.json"


def get_prob_path(label: str) -> Path:
    return get_exp_dir(label) / "reports" / "ensemble_test_prob.npy"


def get_pred_path(label: str) -> Path:
    return get_exp_dir(label) / "reports" / "ensemble_test_pred.npy"


def get_y_path(label: str) -> Path:
    return get_exp_dir(label) / "processed" / "cnn" / "y_test.npy"


def get_times_path(label: str) -> Path:
    return get_exp_dir(label) / "processed" / "cnn" / "times_test.npy"


def check_required_files():
    print("\n========== CHECKING REQUIRED FILES ==========")

    missing = []

    for label in EXPERIMENTS:
        paths = [
            get_report_path(label),
            get_prob_path(label),
            get_pred_path(label),
            get_times_path(label),
        ]

        for path in paths:
            if not path.exists():
                missing.append(path)

        # y_test.npy boleh tidak ada karena bisa direkonstruksi dari master dataset.
        y_path = get_y_path(label)
        if not y_path.exists():
            print(f"[INFO] y_test.npy not found for {label}. It will be rebuilt from master dataset.")

    if not MASTER_PATH.exists():
        missing.append(MASTER_PATH)

    if missing:
        print("\nMissing files:")
        for path in missing:
            print(" -", path)

        raise FileNotFoundError(
            "Some required files are missing. "
            "Make sure ensemble probability, prediction, metrics, and times_test files exist."
        )

    print("All required core files are available.")


# ============================================================
# METRICS LOADING
# ============================================================

def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def get_selected_test_metrics(label: str, mode: str = SELECTED_MODE) -> dict:
    """
    Supports both structures:
    1. {"all_modes": {"early_warning_recall60": {"test": {...}}}}
    2. {"test": {...}}
    """

    metrics = load_json(get_report_path(label))

    if "all_modes" in metrics:
        return metrics["all_modes"][mode]["test"]

    if "test" in metrics:
        return metrics["test"]

    raise KeyError(f"Cannot find test metrics in {get_report_path(label)}")


def get_selected_val_metrics(label: str, mode: str = SELECTED_MODE) -> dict:
    metrics = load_json(get_report_path(label))

    if "all_modes" in metrics:
        return metrics["all_modes"][mode]["validation"]

    if "validation" in metrics:
        return metrics["validation"]

    raise KeyError(f"Cannot find validation metrics in {get_report_path(label)}")


def collect_metric_table() -> pd.DataFrame:
    rows = []

    for label, cfg in EXPERIMENTS.items():
        m = get_selected_test_metrics(label)

        rows.append({
            "horizon": label,
            "target": cfg["target"],
            "horizon_days": cfg["horizon_days"],
            "threshold": m["threshold"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "roc_auc": m["roc_auc"],
            "pr_auc": m["pr_auc"],
            "positive_rate_true": m.get("positive_rate_true", np.nan),
            "positive_rate_pred": m.get("positive_rate_pred", np.nan),
        })

    df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "multi_horizon_metrics_summary.csv"
    df.to_csv(out_path, index=False)

    print("\nSaved metric summary:", out_path)
    print(df)

    return df

# ============================================================
# ARRAY LOADING
# ============================================================
def load_master_with_optional_label(target_col: str) -> xr.Dataset:
    """
    Load master dataset. If the selected target is not inside master_dataset.nc,
    try loading external label file such as labels_fire_next_14d.nc.
    """

    ds = xr.open_dataset(MASTER_PATH)

    if target_col not in ds.data_vars:
        extra_label_path = (
            ROOT_DIR
            / "data"
            / "processed"
            / "master"
            / f"labels_{target_col}.nc"
        )

        if extra_label_path.exists():
            print(f"Loading extra label for {target_col}: {extra_label_path}")
            ds_label = xr.open_dataset(extra_label_path)
            ds = xr.merge([ds, ds_label])
        else:
            raise FileNotFoundError(
                f"Target {target_col} not found in master dataset and extra label file not found: "
                f"{extra_label_path}"
            )

    return ds


def rebuild_y_test_from_master(label: str) -> Path:
    """
    Rebuild y_test.npy from master_dataset.nc using times_test.npy.
    This is useful when experiment folder has ensemble outputs but missing CNN y_test.npy.
    """

    cfg = EXPERIMENTS[label]
    target_col = cfg["target"]

    y_path = get_y_path(label)
    times_path = get_times_path(label)
    prob_path = get_prob_path(label)

    if not times_path.exists():
        raise FileNotFoundError(f"Cannot rebuild y_test because times_test.npy is missing: {times_path}")

    if not prob_path.exists():
        raise FileNotFoundError(f"Cannot rebuild y_test because ensemble_test_prob.npy is missing: {prob_path}")

    print(f"\n========== REBUILDING y_test.npy FOR {label} ==========")
    print("Target:", target_col)

    times = np.load(times_path, allow_pickle=True)
    times_pd = pd.to_datetime(times)

    prob = np.load(prob_path, mmap_mode="r")
    n_time = min(len(times_pd), prob.shape[0])

    times_pd = times_pd[:n_time]

    ds = load_master_with_optional_label(target_col)

    y_da = ds[target_col].sel(time=times_pd)
    y_da = y_da.transpose("time", "latitude", "longitude")

    y_arr = np.asarray(y_da.values, dtype=np.float32)

    nan_count = int(np.isnan(y_arr).sum())
    if nan_count > 0:
        print(f"[WARNING] Found {nan_count} NaN values in rebuilt y_test. They will be removed only if fully invalid.")
        y_arr = np.nan_to_num(y_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    y_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(y_path, y_arr)

    print("Saved rebuilt y_test:", y_path)
    print("Shape:", y_arr.shape)
    print("Positive:", float(y_arr.sum()))
    print("Ratio:", float(y_arr.mean()))

    ds.close()

    return y_path

def load_prob_y_times(label: str):
    prob_path = get_prob_path(label)
    y_path = get_y_path(label)
    times_path = get_times_path(label)

    if not y_path.exists():
        rebuild_y_test_from_master(label)

    prob = np.load(prob_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    times = np.load(times_path, allow_pickle=True)

    # Align time dimension just in case.
    min_t = min(prob.shape[0], y.shape[0], len(times))

    prob = np.asarray(prob[:min_t], dtype=np.float32)
    y = np.asarray(y[:min_t], dtype=np.float32)
    times = np.asarray(times[:min_t])

    return prob, y, times


def flatten_prob_y(label: str):
    prob, y, _ = load_prob_y_times(label)

    prob_flat = prob.reshape(-1)
    y_flat = y.reshape(-1)

    mask = np.isfinite(prob_flat) & np.isfinite(y_flat)

    prob_flat = prob_flat[mask].astype(np.float32)
    y_flat = y_flat[mask].astype(np.int32)

    return y_flat, prob_flat


# ============================================================
# FIG. 2 MULTI-HORIZON PREDICTION MAP OUTPUT
# ============================================================

def find_common_reference_date():
    """
    Choose the latest date that exists in all three horizons.
    Usually this becomes 2024-12-10 because 14-day prediction
    cannot use the final 14 days as reference dates.
    """

    last_dates = []

    for label in EXPERIMENTS:
        _, _, times = load_prob_y_times(label)
        times_pd = pd.to_datetime(times)
        last_dates.append(times_pd.max())

    common_date = min(last_dates).normalize()
    return common_date


def find_nearest_time_index(times, target_date):
    times_pd = pd.to_datetime(times)
    target_date = pd.Timestamp(target_date)

    idx = int(np.argmin(np.abs(times_pd - target_date)))
    return idx, times_pd[idx]


def make_fig2_multi_horizon_prediction_maps():
    print("\n========== FIG. 2 MULTI-HORIZON PREDICTION MAP ==========")

    ds = xr.open_dataset(MASTER_PATH)
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    common_date = find_common_reference_date()
    print("Selected comparable reference date:", common_date.date())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    vmin = 0.0
    vmax = 1.0

    for ax, label in zip(axes, EXPERIMENTS.keys()):
        prob, _, times = load_prob_y_times(label)

        idx, actual_date = find_nearest_time_index(times, common_date)

        risk_map = prob[idx]

        horizon = EXPERIMENTS[label]["horizon_days"]
        pred_start = actual_date + pd.Timedelta(days=1)
        pred_end = actual_date + pd.Timedelta(days=horizon)

        if horizon == 1:
            window_text = f"{pred_start.date()}"
        else:
            window_text = f"{pred_start.date()} to {pred_end.date()}"

        im = ax.pcolormesh(
            lon,
            lat,
            risk_map,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(
            f"{label} Fire Risk\nInput: {actual_date.date()}\nWindow: {window_text}",
            fontsize=11,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88)
    cbar.set_label("Fire Risk Probability")

    fig.suptitle(
        "Fig. 2. Multi-Horizon Prediction Map Output",
        fontsize=15,
        fontweight="bold",
    )

    out_png = OUTPUT_DIR / "fig2_multi_horizon_prediction_map.png"
    out_pdf = OUTPUT_DIR / "fig2_multi_horizon_prediction_map.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out_png)
    print("Saved:", out_pdf)


# ============================================================
# FIG. 3 CONFUSION MATRIX COMPARISON
# ============================================================

def make_fig3_confusion_matrix_comparison():
    print("\n========== FIG. 3 CONFUSION MATRIX COMPARISON ==========")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    for ax, label in zip(axes, EXPERIMENTS.keys()):
        m = get_selected_test_metrics(label)
        cm = np.array(m["confusion_matrix"], dtype=np.int64)

        # Expected format:
        # [[TN, FP],
        #  [FN, TP]]
        total = cm.sum()
        cm_pct = cm / total * 100

        # Use fixed colors to match the TP/FN/FP/TN style reference.
        color_correct = "#F6E7C2"  # warm beige for TP/TN
        color_incorrect = "#CFE7F5"  # light blue for FP/FN
        cmap = ListedColormap([color_incorrect, color_correct])
        correct_mask = np.array([[1, 0], [0, 1]], dtype=np.int8)

        im = ax.imshow(correct_mask, cmap=cmap, vmin=0, vmax=1)

        labels = np.array([
            ["TN", "FP"],
            ["FN", "TP"]
        ])

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{labels[i, j]}\n{cm[i, j]:,}\n{cm_pct[i, j]:.2f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                )

        ax.set_title(
            f"{label}\nThreshold={m['threshold']:.2f}",
            fontsize=11,
        )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])

        ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
        ax.grid(which="minor", color="black", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    fig.suptitle(
        "Fig. 3. Confusion Matrix Comparison for Early Warning Mode",
        fontsize=15,
        fontweight="bold",
    )

    out_png = OUTPUT_DIR / "fig3_confusion_matrix_comparison.png"
    out_pdf = OUTPUT_DIR / "fig3_confusion_matrix_comparison.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out_png)
    print("Saved:", out_pdf)


# ============================================================
# FIG. 4 PR-AUC AND ROC-AUC CURVES
# ============================================================

def make_fig4_pr_roc_curves():
    print("\n========== FIG. 4 PR-AUC AND ROC-AUC CURVES ==========")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    ax_pr = axes[0]
    ax_roc = axes[1]

    for label in EXPERIMENTS:
        print("Computing PR/ROC curve:", label)

        y_true, prob = flatten_prob_y(label)

        ap = average_precision_score(y_true, prob)
        auc = roc_auc_score(y_true, prob)

        precision, recall, _ = precision_recall_curve(y_true, prob)
        fpr, tpr, _ = roc_curve(y_true, prob)

        ax_pr.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{label} PR-AUC={ap:.3f}",
        )

        ax_roc.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{label} ROC-AUC={auc:.3f}",
        )

    ax_pr.set_title("Precision-Recall Curves")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.grid(True, alpha=0.3)
    ax_pr.legend()

    ax_roc.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax_roc.set_title("ROC Curves")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.grid(True, alpha=0.3)
    ax_roc.legend()

    fig.suptitle(
        "Fig. 4. PR-AUC and ROC-AUC Curves Across Prediction Horizons",
        fontsize=15,
        fontweight="bold",
    )

    out_png = OUTPUT_DIR / "fig4_pr_roc_curves.png"
    out_pdf = OUTPUT_DIR / "fig4_pr_roc_curves.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out_png)
    print("Saved:", out_pdf)


# ============================================================
# FIG. 5 HORIZON PERFORMANCE COMPARISON BAR CHART
# ============================================================

def make_fig5_horizon_performance_bar_chart(metric_df: pd.DataFrame):
    print("\n========== FIG. 5 HORIZON PERFORMANCE BAR CHART ==========")

    metrics = ["precision", "recall", "f1", "pr_auc"]
    labels = metric_df["horizon"].tolist()

    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)

    for i, metric in enumerate(metrics):
        values = metric_df[metric].values

        ax.bar(
            x + (i - 1.5) * width,
            values,
            width,
            label=metric.upper() if metric != "f1" else "F1-Score",
        )

        for xi, v in zip(x + (i - 1.5) * width, values):
            ax.text(
                xi,
                v + 0.01,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Metric Value")
    ax.set_title("Fig. 5. Horizon Performance Comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))

    out_png = OUTPUT_DIR / "fig5_horizon_performance_bar_chart.png"
    out_pdf = OUTPUT_DIR / "fig5_horizon_performance_bar_chart.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out_png)
    print("Saved:", out_pdf)


# ============================================================
# FIG. 6 SHAP FEATURE IMPORTANCE FOR XGBOOST
# ============================================================

def find_xgb_model_path(label: str) -> Path:
    return get_exp_dir(label) / "models" / "xgboost" / "xgb_fire_model.json"


def find_xgb_features_path(label: str) -> Path:
    return get_exp_dir(label) / "models" / "xgboost" / "xgb_feature_columns.json"


def find_xgb_test_data_path(label: str) -> Path:
    return get_exp_dir(label) / "processed" / "xgb" / "test.parquet"


def make_fig6_shap_feature_importance(
    label: str = "14-day",
    sample_size: int = 20000,
):
    """
    Creates SHAP feature importance for the selected XGBoost model.
    Default uses 14-day because it has the strongest final PR-AUC/F1.
    """

    print("\n========== FIG. 6 SHAP FEATURE IMPORTANCE ==========")
    print("Selected horizon:", label)

    try:
        import shap
        import xgboost as xgb
    except ImportError as e:
        raise ImportError(
            "SHAP or XGBoost is not installed. Run:\n"
            "pip install shap xgboost"
        ) from e

    model_path = find_xgb_model_path(label)
    features_path = find_xgb_features_path(label)
    test_path = find_xgb_test_data_path(label)

    if not model_path.exists():
        raise FileNotFoundError(f"XGBoost model not found: {model_path}")

    if not features_path.exists():
        raise FileNotFoundError(f"Feature column file not found: {features_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"XGBoost test data not found: {test_path}")

    with open(features_path, "r") as f:
        feature_cols = json.load(f)

    df = pd.read_parquet(test_path)

    target_col = EXPERIMENTS[label]["target"]

    available_features = [c for c in feature_cols if c in df.columns]

    if target_col in available_features:
        available_features.remove(target_col)

    X = df[available_features].replace([np.inf, -np.inf], np.nan).fillna(0)

    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
    else:
        X_sample = X.copy()

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    print("SHAP sample shape:", X_sample.shape)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Handle binary classifier output variations.
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[-1]
    else:
        shap_values_to_plot = shap_values

    # Bar summary
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values_to_plot,
        X_sample,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.title(f"Fig. 6. SHAP Feature Importance for XGBoost ({label})")

    out_bar_png = OUTPUT_DIR / f"fig6_shap_feature_importance_{label}.png"
    out_bar_pdf = OUTPUT_DIR / f"fig6_shap_feature_importance_{label}.pdf"

    plt.savefig(out_bar_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_bar_pdf, bbox_inches="tight")
    plt.close()

    print("Saved:", out_bar_png)
    print("Saved:", out_bar_pdf)

    # Beeswarm summary
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values_to_plot,
        X_sample,
        show=False,
        max_display=20,
    )
    plt.title(f"SHAP Summary Plot for XGBoost ({label})")

    out_bee_png = OUTPUT_DIR / f"fig6_shap_summary_beeswarm_{label}.png"
    out_bee_pdf = OUTPUT_DIR / f"fig6_shap_summary_beeswarm_{label}.pdf"

    plt.savefig(out_bee_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_bee_pdf, bbox_inches="tight")
    plt.close()

    print("Saved:", out_bee_png)
    print("Saved:", out_bee_pdf)


# ============================================================
# OPTIONAL: EXPORT TABLES FOR PAPER
# ============================================================

def export_latex_tables(metric_df: pd.DataFrame):
    table_path = OUTPUT_DIR / "multi_horizon_metrics_table_latex.txt"

    cols = [
        "horizon",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    df = metric_df[cols].copy()

    for col in cols[1:]:
        df[col] = df[col].map(lambda x: f"{x:.4f}")

    latex = df.to_latex(index=False)

    with open(table_path, "w") as f:
        f.write(latex)

    print("Saved LaTeX table:", table_path)


# ============================================================
# MAIN
# ============================================================

def main():
    check_required_files()

    metric_df = collect_metric_table()

    make_fig2_multi_horizon_prediction_maps()
    make_fig3_confusion_matrix_comparison()
    make_fig4_pr_roc_curves()
    make_fig5_horizon_performance_bar_chart(metric_df)

    # Use 14-day XGBoost for main SHAP figure because it gives the strongest
    # final PR-AUC/F1 in the current experiments.
    make_fig6_shap_feature_importance(label="14-day", sample_size=20000)

    export_latex_tables(metric_df)

    print("\n========== ALL PAPER FIGURES GENERATED ==========")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()