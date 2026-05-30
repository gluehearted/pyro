from glob import glob
import numpy as np
import xarray as xr


def preprocess_era5(cfg):

    print("\n========== ERA5 PREPROCESSING ==========")

    chunks = cfg["processing"]["chunks"]

    # =========================================================
    # LOAD FILES
    # =========================================================

    instant_files = sorted(
        glob("data/raw/era5/*/*instant*.nc")
    )

    accum_files = sorted(
        glob("data/raw/era5/*/*accum*.nc")
    )

    print(f"Instant files : {len(instant_files)}")
    print(f"Accum files   : {len(accum_files)}")

    # =========================================================
    # OPEN DATASET
    # =========================================================

    ds_instant = xr.open_mfdataset(
        instant_files,
        combine="by_coords",
        parallel=True,
        chunks={
            "time": chunks["time"],
            "latitude": chunks["lat"],
            "longitude": chunks["lon"]
        },
        engine="netcdf4"
    )

    ds_accum = xr.open_mfdataset(
        accum_files,
        combine="by_coords",
        parallel=True,
        chunks={
            "time": chunks["time"],
            "latitude": chunks["lat"],
            "longitude": chunks["lon"]
        },
        engine="netcdf4"
    )

    print("\nInstant variables:")
    print(list(ds_instant.data_vars))

    print("\nAccum variables:")
    print(list(ds_accum.data_vars))

    # =========================================================
    # SELECT VARIABLES
    # =========================================================

    instant_keep = [
        v for v in ["t2m", "d2m", "u10", "v10", "swvl1"]
        if v in ds_instant.data_vars
    ]

    accum_keep = [
        v for v in ["tp", "ssrd"]
        if v in ds_accum.data_vars
    ]

    ds_instant = ds_instant[instant_keep]
    ds_accum = ds_accum[accum_keep]

    # =========================================================
    # MERGE
    # =========================================================

    ds = xr.merge(
        [ds_instant, ds_accum],
        compat="override"
    )

    # =========================================================
    # FIX ERA5 DIMENSIONS
    # =========================================================

    print("\nFixing ERA5 dimensions...")

    # rename valid_time -> time
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    # drop expver dimension if exists
    if "expver" in ds.dims:

        print("Removing expver dimension...")

        ds = ds.isel(expver=0)

    # drop number dimension if exists
    if "number" in ds.dims:

        print("Removing number dimension...")

        ds = ds.isel(number=0)

    # ensure dimension order
    target_dims = [
        d for d in ["time", "latitude", "longitude"]
        if d in ds.dims
    ]

    ds = ds.transpose(*target_dims)

    print("\nDataset dimensions:")
    print(ds.dims)

    print("\nMerged variables:")
    print(list(ds.data_vars))

    # =========================================================
    # UNIT CONVERSION
    # =========================================================

    if "t2m" in ds:
        ds["t2m"] = ds["t2m"] - 273.15

    if "d2m" in ds:
        ds["d2m"] = ds["d2m"] - 273.15

    # precipitation meter -> mm
    if "tp" in ds:
        ds["tp"] = ds["tp"] * 1000

    # =========================================================
    # WIND SPEED
    # =========================================================

    if "u10" in ds and "v10" in ds:

        ds["wind_speed"] = np.sqrt(
            ds["u10"]**2 +
            ds["v10"]**2
        )

    # =========================================================
    # RELATIVE HUMIDITY
    # =========================================================

    if "t2m" in ds and "d2m" in ds:

        t = ds["t2m"]
        td = ds["d2m"]

        rh = 100 * (
            np.exp((17.625 * td)/(243.04 + td)) /
            np.exp((17.625 * t)/(243.04 + t))
        )

        ds["relative_humidity"] = rh.clip(0, 100)

    # =========================================================
    # DAILY RESAMPLING
    # =========================================================

    print("\nResampling to daily...")

    daily_mean_vars = []

    for v in [
        "t2m",
        "d2m",
        "u10",
        "v10",
        "wind_speed",
        "relative_humidity",
        "swvl1"
    ]:
        if v in ds:
            daily_mean_vars.append(v)

    daily_sum_vars = []

    for v in ["tp", "ssrd"]:
        if v in ds:
            daily_sum_vars.append(v)

    ds_mean = ds[daily_mean_vars].resample(time="1D").mean()

    ds_sum = ds[daily_sum_vars].resample(time="1D").sum()

    ds_daily = xr.merge([ds_mean, ds_sum])

    # =========================================================
    # INTERPOLATE MISSING
    # =========================================================

    print("\nInterpolating missing values...")

    # =========================================================
    # RECHUNK TIME FOR INTERPOLATION
    # =========================================================

    print("\nRechunking time dimension...")

    ds_daily = ds_daily.chunk({
        "time": -1,
        "latitude": 128,
        "longitude": 128
    })

    ds_daily = ds_daily.interpolate_na(
        dim="time",
        method="linear"
    )

    # =========================================================
    # ROLLING FEATURES
    # =========================================================

    print("\nCreating rolling features...")

    windows = cfg["feature_engineering"]["rolling_windows"]

    for w in windows:

        if "tp" in ds_daily:

            ds_daily[f"rainfall_sum_{w}d"] = (
                ds_daily["tp"]
                .rolling(time=w)
                .sum()
            )

        if "t2m" in ds_daily:

            ds_daily[f"temperature_mean_{w}d"] = (
                ds_daily["t2m"]
                .rolling(time=w)
                .mean()
            )

        if "wind_speed" in ds_daily:

            ds_daily[f"wind_speed_mean_{w}d"] = (
                ds_daily["wind_speed"]
                .rolling(time=w)
                .mean()
            )

    # =========================================================
    # TIME FEATURES
    # =========================================================

    print("\nCreating temporal features...")

    ds_daily["month"] = ds_daily.time.dt.month
    ds_daily["dayofyear"] = ds_daily.time.dt.dayofyear

    ds_daily["sin_doy"] = np.sin(
        2 * np.pi * ds_daily["dayofyear"] / 365
    )

    ds_daily["cos_doy"] = np.cos(
        2 * np.pi * ds_daily["dayofyear"] / 365
    )

    # =========================================================
    # SORT
    # =========================================================

    ds_daily = ds_daily.sortby("time")

    # =========================================================
    # SAVE
    # =========================================================

    output_path = "data/interim/era5/era5_daily_clean.nc"

    print(f"\nSaving to: {output_path}")

    ds_daily.to_netcdf(output_path)

    print("\nERA5 preprocessing complete.")

    return ds_daily