from pathlib import Path
import os
import sys
import json
import numpy as np
import pandas as pd
import xarray as xr
import folium
from branca.colormap import LinearColormap


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))


from src.utils.experiment import (
    MASTER_PATH,
    CNN_DIR,
    REPORT_DIR,
    ENSEMBLE_MODEL_DIR,
    MAP_DIR,
    TARGET_COL,
    EXP_NAME,
    HORIZON_DAYS,
    ensure_experiment_dirs,
    print_experiment_info,
)

ENSEMBLE_CONFIG_PATH = ENSEMBLE_MODEL_DIR / "ensemble_config.json"

MAP_DIR.mkdir(parents=True, exist_ok=True)


def load_ensemble_config():
    default_config = {
        "selected_mode": "early_warning_recall60",
        "alpha_cnn": 0.30,
        "alpha_xgb": 0.70,
        "threshold": 0.84,
    }

    if ENSEMBLE_CONFIG_PATH.exists():
        with open(ENSEMBLE_CONFIG_PATH, "r") as f:
            return json.load(f)

    return default_config


def risk_category(prob, threshold):
    if prob < 0.30:
        return "Low"
    elif prob < 0.60:
        return "Moderate"
    elif prob < threshold:
        return "High"
    else:
        return "Very High"


def risk_color(prob, threshold):
    category = risk_category(prob, threshold)

    if category == "Low":
        return "#2ecc71"
    elif category == "Moderate":
        return "#f1c40f"
    elif category == "High":
        return "#e67e22"
    return "#e74c3c"


def main():
    ensure_experiment_dirs()
    print_experiment_info()
    config = load_ensemble_config()
    threshold = float(config.get("threshold", 0.84))

    ds = xr.open_dataset(MASTER_PATH)

    latitudes = ds["latitude"].values
    longitudes = ds["longitude"].values

    prob_path = REPORT_DIR / "ensemble_test_prob.npy"
    pred_path = REPORT_DIR / "ensemble_test_pred.npy"
    times_path = CNN_DIR / "times_test.npy"

    if not prob_path.exists():
        raise FileNotFoundError(f"Tidak ditemukan: {prob_path}")

    if not pred_path.exists():
        raise FileNotFoundError(f"Tidak ditemukan: {pred_path}")

    if not times_path.exists():
        raise FileNotFoundError(f"Tidak ditemukan: {times_path}")

    probs = np.load(prob_path)
    preds = np.load(pred_path)
    times = np.load(times_path, allow_pickle=True)

    times_pd = pd.to_datetime(times).normalize()

    # Default: pakai tanggal terakhir test.
    # Bisa diganti dengan:
    # MAP_DATE=2024-09-01 python src/visualization/prediction_map.py
    map_date = os.getenv("MAP_DATE")

    if map_date:
        reference_date = pd.Timestamp(map_date).normalize()
        match = np.where(times_pd == reference_date)[0]

        if len(match) == 0:
            raise ValueError(
                f"Tanggal {map_date} tidak ditemukan. "
                f"Range tersedia: {times_pd[0].date()} sampai {times_pd[-1].date()}"
            )

        idx = int(match[0])
    else:
        idx = len(times_pd) - 1
        reference_date = times_pd[idx]

    prediction_start = reference_date + pd.Timedelta(days=1)
    prediction_end = reference_date + pd.Timedelta(days=HORIZON_DAYS)

    if HORIZON_DAYS == 1:
        title = "Next-Day Fire Risk Prediction"
        prediction_window = str(prediction_start.date())
    else:
        title = f"{HORIZON_DAYS}-Day Fire Risk Prediction"
        prediction_window = f"{prediction_start.date()} to {prediction_end.date()}"

    risk = probs[idx]
    pred = preds[idx]

    print(f"\n========== PYROSENSE {title.upper()} ==========")
    print("Experiment       :", EXP_NAME)
    print("Target           :", TARGET_COL)
    print("Horizon days     :", HORIZON_DAYS)
    print("Reference date   :", reference_date.date())
    print("Prediction window:", prediction_window)
    print("Threshold        :", threshold)
    print("Alpha CNN      :", config.get("alpha_cnn"))
    print("Alpha XGB      :", config.get("alpha_xgb"))
    print("Risk min       :", float(risk.min()))
    print("Risk max       :", float(risk.max()))
    print("Risk mean      :", float(risk.mean()))
    print("Predicted high-risk grids:", int(pred.sum()))

    center_lat = float(np.mean(latitudes))
    center_lon = float(np.mean(longitudes))

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles="CartoDB positron"
    )

    lat_step = abs(float(latitudes[1] - latitudes[0]))
    lon_step = abs(float(longitudes[1] - longitudes[0]))

    rows = []

    for i, lat in enumerate(latitudes):
        for j, lon in enumerate(longitudes):
            prob = float(risk[i, j])
            pred_label = int(pred[i, j])
            category = risk_category(prob, threshold)
            color = risk_color(prob, threshold)

            rows.append({
                "experiment": EXP_NAME,
                "target": TARGET_COL,
                "horizon_days": HORIZON_DAYS,
                "reference_date": str(reference_date.date()),
                "prediction_start": str(prediction_start.date()),
                "prediction_end": str(prediction_end.date()),
                "prediction_window": prediction_window,
                "latitude": float(lat),
                "longitude": float(lon),
                "risk_probability": prob,
                "risk_category": category,
                "predicted_fire_risk": pred_label,
            })

            lat_min = float(lat) - lat_step / 2
            lat_max = float(lat) + lat_step / 2
            lon_min = float(lon) - lon_step / 2
            lon_max = float(lon) + lon_step / 2

            bounds = [
                [min(lat_min, lat_max), min(lon_min, lon_max)],
                [max(lat_min, lat_max), max(lon_min, lon_max)],
            ]

            if category == "Low":
                opacity = 0.15
            elif category == "Moderate":
                opacity = 0.35
            elif category == "High":
                opacity = 0.50
            else:
                opacity = 0.70

            tooltip = folium.Tooltip(
                f"""
                <b>PyroSense Prediction</b><br>
                Target: {TARGET_COL}<br>
                Reference date: {reference_date.date()}<br>
                Prediction window: {prediction_window}<br>
                Latitude: {float(lat):.2f}<br>
                Longitude: {float(lon):.2f}<br>
                Fire risk probability: {prob:.4f}<br>
                Risk category: {category}<br>
                Predicted fire risk: {pred_label}
                """,
                sticky=True
            )

            folium.Rectangle(
                bounds=bounds,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=opacity,
                weight=0.25,
                tooltip=tooltip,
            ).add_to(m)

    colormap = LinearColormap(
        colors=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"],
        vmin=0.0,
        vmax=1.0,
        caption=f"{HORIZON_DAYS}-Day Fire Risk Probability"
    )
    colormap.add_to(m)

    title_html = f"""
    <div style="
        position: fixed;
        top: 10px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
        background-color: white;
        padding: 10px 18px;
        border-radius: 8px;
        border: 1px solid #999;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0 2px 6px rgba(0,0,0,0.25);
    ">
        PyroSense {title}<br>
        Input until {reference_date.date()} → Prediction window: {prediction_window}
    </div>
    """

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background-color: white;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #999;
        font-size: 13px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.25);
    ">
        <b>Risk Category</b><br>
        <span style="color:#2ecc71;">■</span> Low: &lt; 0.30<br>
        <span style="color:#f1c40f;">■</span> Moderate: 0.30–0.60<br>
        <span style="color:#e67e22;">■</span> High: 0.60–{threshold:.2f}<br>
        <span style="color:#e74c3c;">■</span> Very High: ≥ {threshold:.2f}<br>
        <br>
        Target: {TARGET_COL}<br>
        Horizon: {HORIZON_DAYS} day(s)<br>
        Ensemble: {config.get("alpha_cnn", 0.30):.2f} CNN + {config.get("alpha_xgb", 0.70):.2f} XGB
    </div>
    """

    m.get_root().html.add_child(folium.Element(title_html))
    m.get_root().html.add_child(folium.Element(legend_html))

    output_html = MAP_DIR / f"prediction_fire_risk_{TARGET_COL}_{reference_date.date()}.html"
    output_csv = MAP_DIR / f"prediction_fire_risk_{TARGET_COL}_{reference_date.date()}.csv"

    m.save(output_html)

    df = pd.DataFrame(rows)
    df = df.sort_values("risk_probability", ascending=False)
    df.to_csv(output_csv, index=False)

    print("\n========== OUTPUT SAVED ==========")
    print("Map HTML:", output_html)
    print("Risk CSV:", output_csv)


if __name__ == "__main__":
    main()