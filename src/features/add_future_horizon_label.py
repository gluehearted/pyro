from pathlib import Path
import os
import sys

import numpy as np
import xarray as xr


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

MASTER_PATH = ROOT_DIR / "data" / "processed" / "master" / "master_dataset.nc"

TARGET_COL = os.getenv("TARGET_COL", "fire_next_14d")
HORIZON_DAYS = int(os.getenv("HORIZON_DAYS", "14"))

OUTPUT_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "master"
    / f"labels_{TARGET_COL}.nc"
)

SOURCE_FIRE_COL = "hotspot_binary"


def main():
    print("\n========== ADD FUTURE HORIZON LABEL ONLY ==========")
    print("Master path :", MASTER_PATH)
    print("Output path :", OUTPUT_PATH)
    print("Source col  :", SOURCE_FIRE_COL)
    print("Target col  :", TARGET_COL)
    print("Horizon days:", HORIZON_DAYS)

    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Master dataset tidak ditemukan: {MASTER_PATH}")

    ds = xr.open_dataset(MASTER_PATH)

    if SOURCE_FIRE_COL not in ds.data_vars:
        raise ValueError(f"{SOURCE_FIRE_COL} tidak ditemukan di master dataset.")

    print("\nLoading hotspot_binary only...")
    fire = ds[SOURCE_FIRE_COL].fillna(0).values.astype(bool)

    times = ds["time"].values
    latitudes = ds["latitude"].values
    longitudes = ds["longitude"].values

    print("Fire shape:", fire.shape)

    label = np.zeros(fire.shape, dtype=np.float32)

    print("\nBuilding future label...")
    for k in range(1, HORIZON_DAYS + 1):
        print(f"Shift +{k} day")
        label[:-k, :, :] = np.maximum(
            label[:-k, :, :],
            fire[k:, :, :].astype(np.float32)
        )

    # Horizon terakhir tidak valid karena tidak punya data masa depan lengkap.
    label[-HORIZON_DAYS:, :, :] = np.nan

    da = xr.DataArray(
        label,
        coords={
            "time": times,
            "latitude": latitudes,
            "longitude": longitudes,
        },
        dims=("time", "latitude", "longitude"),
        name=TARGET_COL,
    )

    da.attrs["description"] = (
        f"Binary fire occurrence label within t+1 to t+{HORIZON_DAYS} days."
    )
    da.attrs["horizon_days"] = HORIZON_DAYS
    da.attrs["source"] = SOURCE_FIRE_COL

    out = xr.Dataset({TARGET_COL: da})

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("\nSaving label-only NetCDF...")
    encoding = {
        TARGET_COL: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
        }
    }

    out.to_netcdf(OUTPUT_PATH, encoding=encoding)

    print("\n========== DONE ==========")
    print("Saved:", OUTPUT_PATH)
    print("Positive:", float(out[TARGET_COL].sum(skipna=True)))
    print("Ratio   :", float(out[TARGET_COL].mean(skipna=True)))
    print("NaN     :", int(out[TARGET_COL].isnull().sum()))


if __name__ == "__main__":
    main()