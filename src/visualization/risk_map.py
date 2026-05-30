from pathlib import Path
import os
import json

import numpy as np
import pandas as pd
import xarray as xr
import folium
from branca.colormap import LinearColormap


ROOT_DIR = Path(__file__).resolve().parents[2]

# Kalau nanti pakai folder experiments, bisa jalankan:
# EXP_NAME=split_2015_2022_val2023_test2024 python src/visualization/risk_map.py
EXP_NAME = os.getenv("EXP_NAME")

MASTER_PATH = ROOT_DIR / "data" / "processed" / "master" / "master_dataset.nc"
CNN_DIR = ROOT_DIR / "data" / "processed" / "cnn"

if EXP_NAME:
    REPORT_DIR = ROOT_DIR / "experiments" / EXP_NAME / "reports"
    ENSEMBLE_CONFIG_PATH = ROOT_DIR / "experiments" / EXP_NAME / "models" / "ensemble" / "ensemble_config.json"
    MAP_DIR = ROOT_DIR / "experiments" / EXP_NAME / "maps"
else:
    REPORT_DIR = ROOT_DIR / "outputs" / "reports"
    ENSEMBLE_CONFIG_PATH = ROOT_DIR / "models" / "ensemble" / "ensemble_config.json"
    MAP_DIR = ROOT_DIR / "outputs" / "maps"

MAP_DIR.mkdir(parents=True, exist_ok=True)


def load_ensemble_config():
    """
    Ambil threshold final ensemble.
    Kalau file config tidak ada, pakai threshold default 0.84
    sesuai hasil early_warning_recall60 kamu.
    """

    default_config = {
        "selected_mode": "early_warning_recall60",
        "alpha_cnn": 0.30,
        "alpha_xgb": 0.70,
        "threshold": 0.84,
    }

    if ENSEMBLE_CONFIG_PATH.exists():
        with open(ENSEMBLE_CONFIG_PATH, "r") as f:
            return json.load(f)

    print("WARNING: ensemble_config.json tidak ditemukan.")
    print("Menggunakan default threshold 0.84.")
    return default_config


def risk_category(prob, threshold):
    """
    Kategori risiko berdasarkan probabilitas ensemble.
    threshold dipakai sebagai batas Very High Risk.
    """

    if prob < 0.30:
        return "Low"
    elif prob < 0.60:
        return "Moderate"
    elif prob < threshold:
        return "High"
    else:
        return "Very High"


def risk_color(prob, threshold):
    """
    Warna grid pada peta.
    """

    category = risk_category(prob, threshold)

    if category == "Low":
        return "#2ecc71"      # green
    elif category == "Moderate":
        return "#f1c40f"      # yellow
    elif category == "High":
        return "#e67e22"      # orange
    else:
        return "#e74c3c"      # red


def load_required_data():
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Master dataset tidak ditemukan: {MASTER_PATH}")

    prob_path = REPORT_DIR / "ensemble_test_prob.npy"
    pred_path = REPORT_DIR / "ensemble_test_pred.npy"
    times_path = CNN_DIR / "times_test.npy"
    y_path = CNN_DIR / "y_test.npy"

    for path in [prob_path, pred_path, times_path, y_path]:
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

    ds = xr.open_dataset(MASTER_PATH)

    latitudes = ds["latitude"].values
    longitudes = ds["longitude"].values

    probs = np.load(prob_path)
    preds = np.load(pred_path)
    times = np.load(times_path, allow_pickle=True)
    y_true = np.load(y_path, mmap_mode="r")

    print("\n========== LOADED DATA ==========")
    print("Probability shape:", probs.shape)
    print("Prediction shape :", preds.shape)
    print("Ground truth shape:", y_true.shape)
    print("Times length     :", len(times))
    print("Latitude count   :", len(latitudes))
    print("Longitude count  :", len(longitudes))

    if probs.shape != preds.shape:
        raise ValueError(f"Shape prob dan pred beda: {probs.shape} vs {preds.shape}")

    if probs.shape != y_true.shape:
        raise ValueError(f"Shape prob dan y_true beda: {probs.shape} vs {y_true.shape}")

    if probs.shape[0] != len(times):
        raise ValueError(f"Jumlah time tidak cocok: prob={probs.shape[0]}, times={len(times)}")

    return latitudes, longitudes, probs, preds, y_true, times


def select_map_index(times):
    """
    Default: tanggal terakhir test.
    Bisa pilih tanggal tertentu dengan:
    MAP_DATE=2024-09-01 python src/visualization/risk_map.py

    Atau pilih index:
    MAP_INDEX=100 python src/visualization/risk_map.py
    """

    map_date = os.getenv("MAP_DATE")
    map_index = os.getenv("MAP_INDEX")

    times_pd = pd.to_datetime(times).normalize()

    if map_index is not None:
        idx = int(map_index)

        if idx < 0:
            idx = len(times_pd) + idx

        if idx < 0 or idx >= len(times_pd):
            raise ValueError(f"MAP_INDEX di luar range: {map_index}")

        return idx, times_pd[idx]

    if map_date is not None:
        target_date = pd.Timestamp(map_date).normalize()

        matches = np.where(times_pd == target_date)[0]

        if len(matches) == 0:
            raise ValueError(
                f"MAP_DATE {map_date} tidak ditemukan dalam times_test.npy. "
                f"Range tersedia: {times_pd[0].date()} sampai {times_pd[-1].date()}"
            )

        idx = int(matches[0])
        return idx, times_pd[idx]

    # Default: tanggal terakhir
    idx = len(times_pd) - 1
    return idx, times_pd[idx]


def save_top_risk_csv(latitudes, longitudes, risk, pred, actual, selected_date, threshold):
    rows = []

    for i, lat in enumerate(latitudes):
        for j, lon in enumerate(longitudes):
            prob = float(risk[i, j])

            rows.append({
                "date": selected_date,
                "latitude": float(lat),
                "longitude": float(lon),
                "risk_probability": prob,
                "risk_category": risk_category(prob, threshold),
                "prediction": int(pred[i, j]),
                "actual_label": int(actual[i, j]),
            })

    df = pd.DataFrame(rows)

    df = df.sort_values(
        "risk_probability",
        ascending=False
    ).reset_index(drop=True)

    output_csv = MAP_DIR / f"top_risk_grid_{selected_date}.csv"
    df.to_csv(output_csv, index=False)

    print("Saved top risk CSV:", output_csv)

    return df


def create_risk_map(latitudes, longitudes, risk, pred, actual, selected_date, threshold):
    center_lat = float(np.mean(latitudes))
    center_lon = float(np.mean(longitudes))

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles="CartoDB positron"
    )

    lat_step = abs(float(latitudes[1] - latitudes[0])) if len(latitudes) > 1 else 0.25
    lon_step = abs(float(longitudes[1] - longitudes[0])) if len(longitudes) > 1 else 0.25

    for i, lat in enumerate(latitudes):
        for j, lon in enumerate(longitudes):
            prob = float(risk[i, j])
            pred_label = int(pred[i, j])
            actual_label = int(actual[i, j])

            category = risk_category(prob, threshold)
            color = risk_color(prob, threshold)

            lat_min = float(lat) - lat_step / 2
            lat_max = float(lat) + lat_step / 2
            lon_min = float(lon) - lon_step / 2
            lon_max = float(lon) + lon_step / 2

            bounds = [
                [min(lat_min, lat_max), min(lon_min, lon_max)],
                [max(lat_min, lat_max), max(lon_min, lon_max)],
            ]

            # Biar peta tidak terlalu gelap, opacity dibedakan.
            if category == "Low":
                opacity = 0.18
            elif category == "Moderate":
                opacity = 0.35
            elif category == "High":
                opacity = 0.50
            else:
                opacity = 0.65

            tooltip = folium.Tooltip(
                f"""
                <b>PyroSense Fire Risk</b><br>
                Date: {selected_date}<br>
                Latitude: {float(lat):.2f}<br>
                Longitude: {float(lon):.2f}<br>
                Probability: {prob:.4f}<br>
                Category: {category}<br>
                Prediction: {pred_label}<br>
                Actual Label: {actual_label}
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
        caption="Fire Risk Probability"
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
        PyroSense Fire Risk Map - {selected_date}
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
        Ensemble: 0.30 CNN + 0.70 XGB
    </div>
    """

    m.get_root().html.add_child(folium.Element(title_html))
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def main():
    print("\n========== PYROSENSE RISK MAP ==========")

    config = load_ensemble_config()
    threshold = float(config.get("threshold", 0.84))

    print("Selected mode:", config.get("selected_mode", "unknown"))
    print("Threshold    :", threshold)
    print("Alpha CNN    :", config.get("alpha_cnn", "unknown"))
    print("Alpha XGB    :", config.get("alpha_xgb", "unknown"))

    latitudes, longitudes, probs, preds, y_true, times = load_required_data()

    idx, selected_time = select_map_index(times)
    selected_date = str(selected_time.date())

    print("\nSelected map index:", idx)
    print("Selected date     :", selected_date)

    risk = probs[idx]
    pred = preds[idx]
    actual = y_true[idx]

    print("\n========== SELECTED DATE SUMMARY ==========")
    print("Risk min :", float(np.min(risk)))
    print("Risk max :", float(np.max(risk)))
    print("Risk mean:", float(np.mean(risk)))
    print("Predicted fire grids:", int(np.sum(pred)))
    print("Actual fire grids   :", int(np.sum(actual)))

    save_top_risk_csv(
        latitudes=latitudes,
        longitudes=longitudes,
        risk=risk,
        pred=pred,
        actual=actual,
        selected_date=selected_date,
        threshold=threshold,
    )

    risk_map = create_risk_map(
        latitudes=latitudes,
        longitudes=longitudes,
        risk=risk,
        pred=pred,
        actual=actual,
        selected_date=selected_date,
        threshold=threshold,
    )

    output_path = MAP_DIR / f"fire_risk_map_{selected_date}.html"
    risk_map.save(output_path)

    print("\n========== RISK MAP COMPLETE ==========")
    print("Saved map:", output_path)


if __name__ == "__main__":
    main()