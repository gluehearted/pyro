from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import xarray as xr


def get_project_root():
    return Path(__file__).resolve().parents[2]


def load_config():
    root = get_project_root()
    config_path = root / "config" / "config.yaml"

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_era5_reference():
    root = get_project_root()
    era5_path = root / "data" / "interim" / "era5" / "era5_daily_clean.nc"

    if not era5_path.exists():
        raise FileNotFoundError(
            f"ERA5 clean file tidak ditemukan: {era5_path}\n"
            "Jalankan preprocessing ERA5 dulu."
        )

    ds = xr.open_dataset(era5_path)

    print("\n========== ERA5 REFERENCE GRID ==========")
    print(ds)

    return ds


def standardize_fire_columns(df):
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    required_cols = ["latitude", "longitude", "acq_date"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom wajib tidak ditemukan: {col}")

    if "frp" not in df.columns:
        df["frp"] = np.nan

    if "confidence" not in df.columns:
        df["confidence"] = np.nan

    return df


def load_modis(path, min_confidence=30):
    print(f"\nLoading MODIS: {path}")

    df = pd.read_csv(path)
    df = standardize_fire_columns(df)

    df["sensor"] = "MODIS"

    df["confidence_numeric"] = pd.to_numeric(
        df["confidence"],
        errors="coerce"
    )

    before = len(df)

    df = df[
        df["confidence_numeric"].isna()
        | (df["confidence_numeric"] >= min_confidence)
    ]

    after = len(df)

    print(f"MODIS rows before filter: {before}")
    print(f"MODIS rows after filter : {after}")

    return df


def load_viirs(path):
    print(f"\nLoading VIIRS: {path}")

    df = pd.read_csv(path)
    df = standardize_fire_columns(df)

    df["sensor"] = "VIIRS"

    before = len(df)

    # VIIRS confidence biasanya: l, n, h
    # l = low, n = nominal, h = high
    if df["confidence"].dtype == object:
        df["confidence"] = df["confidence"].astype(str).str.lower().str.strip()

        df = df[df["confidence"].isin(["n", "h", "nominal", "high"])]
    else:
        df["confidence_numeric"] = pd.to_numeric(
            df["confidence"],
            errors="coerce"
        )
        df = df[
            df["confidence_numeric"].isna()
            | (df["confidence_numeric"] >= 30)
        ]

    after = len(df)

    print(f"VIIRS rows before filter: {before}")
    print(f"VIIRS rows after filter : {after}")

    return df


def clean_fire_dataframe(df, cfg):
    print("\n========== CLEANING FIRE DATA ==========")

    df = df.copy()

    df["date"] = pd.to_datetime(df["acq_date"], errors="coerce").dt.floor("D")

    df = df.dropna(subset=["date", "latitude", "longitude"])

    bbox = cfg["roi"]["combined_bbox"]
    min_lon, min_lat, max_lon, max_lat = bbox

    before_roi = len(df)

    df = df[
        (df["longitude"] >= min_lon)
        & (df["longitude"] <= max_lon)
        & (df["latitude"] >= min_lat)
        & (df["latitude"] <= max_lat)
    ]

    after_roi = len(df)

    print(f"Rows before ROI filter: {before_roi}")
    print(f"Rows after ROI filter : {after_roi}")

    before_dup = len(df)

    subset_cols = ["date", "latitude", "longitude", "sensor"]
    df = df.drop_duplicates(subset=subset_cols)

    after_dup = len(df)

    print(f"Rows before duplicate removal: {before_dup}")
    print(f"Rows after duplicate removal : {after_dup}")

    return df


def assign_to_era5_grid(df, era5_ds):
    print("\n========== SPATIAL BINNING TO ERA5 GRID ==========")

    df = df.copy()

    latitudes = era5_ds["latitude"].values
    longitudes = era5_ds["longitude"].values

    lat_step = abs(float(latitudes[1] - latitudes[0]))
    lon_step = abs(float(longitudes[1] - longitudes[0]))

    lat_descending = latitudes[0] > latitudes[-1]
    lon_ascending = longitudes[0] < longitudes[-1]

    if lat_descending:
        df["lat_idx"] = np.rint(
            (latitudes[0] - df["latitude"]) / lat_step
        ).astype(int)
    else:
        df["lat_idx"] = np.rint(
            (df["latitude"] - latitudes[0]) / lat_step
        ).astype(int)

    if lon_ascending:
        df["lon_idx"] = np.rint(
            (df["longitude"] - longitudes[0]) / lon_step
        ).astype(int)
    else:
        df["lon_idx"] = np.rint(
            (longitudes[0] - df["longitude"]) / lon_step
        ).astype(int)

    df = df[
        (df["lat_idx"] >= 0)
        & (df["lat_idx"] < len(latitudes))
        & (df["lon_idx"] >= 0)
        & (df["lon_idx"] < len(longitudes))
    ]

    df["grid_latitude"] = latitudes[df["lat_idx"].values]
    df["grid_longitude"] = longitudes[df["lon_idx"].values]

    print(f"Rows after grid assignment: {len(df)}")

    return df


def build_fire_grid_dataset(df, era5_ds):
    print("\n========== BUILDING FIRE GRID DATASET ==========")

    times = pd.to_datetime(era5_ds["time"].values)
    latitudes = era5_ds["latitude"].values
    longitudes = era5_ds["longitude"].values

    time_index = {
        pd.Timestamp(t).normalize(): i
        for i, t in enumerate(times)
    }

    n_time = len(times)
    n_lat = len(latitudes)
    n_lon = len(longitudes)

    hotspot_count = np.zeros(
        (n_time, n_lat, n_lon),
        dtype=np.float32
    )

    mean_frp = np.zeros(
        (n_time, n_lat, n_lon),
        dtype=np.float32
    )

    max_frp = np.zeros(
        (n_time, n_lat, n_lon),
        dtype=np.float32
    )

    grouped = (
        df.groupby(["date", "lat_idx", "lon_idx"])
        .agg(
            hotspot_count=("frp", "size"),
            mean_frp=("frp", "mean"),
            max_frp=("frp", "max")
        )
        .reset_index()
    )

    print(f"Aggregated fire grid rows: {len(grouped)}")

    for row in grouped.itertuples(index=False):
        date = pd.Timestamp(row.date).normalize()

        if date not in time_index:
            continue

        t_idx = time_index[date]
        lat_idx = int(row.lat_idx)
        lon_idx = int(row.lon_idx)

        hotspot_count[t_idx, lat_idx, lon_idx] = row.hotspot_count

        if not pd.isna(row.mean_frp):
            mean_frp[t_idx, lat_idx, lon_idx] = row.mean_frp

        if not pd.isna(row.max_frp):
            max_frp[t_idx, lat_idx, lon_idx] = row.max_frp

    ds_fire = xr.Dataset(
        data_vars={
            "hotspot_count": (
                ["time", "latitude", "longitude"],
                hotspot_count
            ),
            "mean_frp": (
                ["time", "latitude", "longitude"],
                mean_frp
            ),
            "max_frp": (
                ["time", "latitude", "longitude"],
                max_frp
            ),
        },
        coords={
            "time": times,
            "latitude": latitudes,
            "longitude": longitudes
        }
    )

    print(ds_fire)

    return ds_fire


def add_fire_history_features(ds_fire):
    print("\n========== ADDING FIRE HISTORY FEATURES ==========")

    ds_fire["hotspot_binary"] = xr.where(
        ds_fire["hotspot_count"] > 0,
        1,
        0
    ).astype("float32")

    for window in [3, 7, 14]:
        ds_fire[f"hotspot_count_{window}d"] = (
            ds_fire["hotspot_count"]
            .rolling(time=window)
            .sum()
            .fillna(0)
            .astype("float32")
        )

    return ds_fire


def save_fire_outputs(df_clean, ds_fire):
    root = get_project_root()

    output_dir = root / "data" / "interim" / "fire"
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_parquet = output_dir / "hotspots_clean.parquet"
    clean_csv = output_dir / "hotspots_clean.csv"
    grid_nc = output_dir / "fire_daily_grid.nc"

    print("\n========== SAVING FIRE OUTPUTS ==========")

    try:
        df_clean.to_parquet(clean_parquet, index=False)
        print(f"Saved clean hotspot parquet: {clean_parquet}")
    except Exception as e:
        print(f"Parquet gagal, fallback ke CSV. Reason: {e}")
        df_clean.to_csv(clean_csv, index=False)
        print(f"Saved clean hotspot CSV: {clean_csv}")

    encoding = {
        var: {
            "zlib": True,
            "complevel": 4
        }
        for var in ds_fire.data_vars
    }

    ds_fire.to_netcdf(
        grid_nc,
        encoding=encoding
    )

    print(f"Saved fire grid NetCDF: {grid_nc}")


def preprocess_fire(cfg, era5_ds):
    modis_path = get_project_root() / cfg["data_sources"]["fires_modis"]["path"]
    viirs_path = get_project_root() / cfg["data_sources"]["fires_viirs"]["path"]

    modis = load_modis(
        modis_path,
        min_confidence=cfg["data_sources"]["fires_modis"].get(
            "confidence_threshold",
            30
        )
    )

    viirs = load_viirs(viirs_path)

    df = pd.concat(
        [modis, viirs],
        ignore_index=True
    )

    df = clean_fire_dataframe(df, cfg)

    df = assign_to_era5_grid(df, era5_ds)

    ds_fire = build_fire_grid_dataset(df, era5_ds)

    ds_fire = add_fire_history_features(ds_fire)

    save_fire_outputs(df, ds_fire)

    return ds_fire


def main():
    cfg = load_config()

    era5_ds = load_era5_reference()

    ds_fire = preprocess_fire(cfg, era5_ds)

    print("\n========== FIRE PREPROCESSING COMPLETE ==========")
    print(ds_fire)


if __name__ == "__main__":
    main()