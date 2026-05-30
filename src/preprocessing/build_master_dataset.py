from pathlib import Path
import yaml
import xarray as xr
import numpy as np

from generate_labels import (
    generate_fire_labels,
    drop_invalid_label_times
)


def get_project_root():
    return Path(__file__).resolve().parents[2]


def load_config():
    root = get_project_root()
    config_path = root / "config" / "config.yaml"

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_interim_datasets():
    root = get_project_root()

    era5_path = root / "data" / "interim" / "era5" / "era5_daily_clean.nc"
    fire_path = root / "data" / "interim" / "fire" / "fire_daily_grid.nc"
    peat_path = root / "data" / "interim" / "peatland" / "peatland_aligned.nc"

    for path in [era5_path, fire_path, peat_path]:
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

    print("\n========== LOADING ERA5 ==========")
    era5_ds = xr.open_dataset(
        era5_path,
        chunks={"time": 365}
    )
    print(era5_ds)

    print("\n========== LOADING FIRE ==========")
    fire_ds = xr.open_dataset(
        fire_path,
        chunks={"time": 365}
    )
    print(fire_ds)

    print("\n========== LOADING PEATLAND ==========")
    peat_ds = xr.open_dataset(peat_path)
    print(peat_ds)

    return era5_ds, fire_ds, peat_ds


def clean_dataset_metadata(ds):
    """
    Menghapus coordinate tambahan yang tidak dibutuhkan untuk ML.
    """

    drop_names = []

    for name in ["number", "spatial_ref"]:
        if name in ds.coords or name in ds.variables:
            drop_names.append(name)

    if drop_names:
        ds = ds.drop_vars(drop_names, errors="ignore")

    return ds


def validate_alignment(era5_ds, fire_ds, peat_ds):
    print("\n========== VALIDATING ALIGNMENT ==========")

    checks = {
        "fire_time_equal_era5": era5_ds.time.equals(fire_ds.time),
        "fire_lat_equal_era5": era5_ds.latitude.equals(fire_ds.latitude),
        "fire_lon_equal_era5": era5_ds.longitude.equals(fire_ds.longitude),
        "peat_lat_equal_era5": era5_ds.latitude.equals(peat_ds.latitude),
        "peat_lon_equal_era5": era5_ds.longitude.equals(peat_ds.longitude),
    }

    for key, value in checks.items():
        print(f"{key}: {value}")

    if not all(checks.values()):
        raise ValueError(
            "Alignment gagal. Time/latitude/longitude antar dataset belum sama."
        )

    print("Alignment OK.")


def build_master_dataset(cfg):
    era5_ds, fire_ds, peat_ds = load_interim_datasets()

    era5_ds = clean_dataset_metadata(era5_ds)
    fire_ds = clean_dataset_metadata(fire_ds)
    peat_ds = clean_dataset_metadata(peat_ds)

    validate_alignment(era5_ds, fire_ds, peat_ds)

    print("\n========== MERGING DATASETS ==========")

    ds = xr.merge(
        [
            era5_ds,
            fire_ds,
            peat_ds
        ],
        compat="override"
    )

    print("\nMASTER DATASET BEFORE LABEL:")
    print(ds)

    print("\n========== GENERATING LABELS ==========")

    horizons = cfg.get("labeling", {}).get(
        "horizons",
        [1, 3, 7]
    )

    ds = generate_fire_labels(
        ds,
        hotspot_var="hotspot_count",
        horizons=horizons
    )

    max_horizon = max(horizons)
    ds = drop_invalid_label_times(ds, max_horizon=max_horizon)

    print("\nMASTER DATASET AFTER LABEL:")
    print(ds)

    return ds


def save_master_dataset(ds):
    root = get_project_root()

    output_dir = root / "data" / "processed" / "master"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "master_dataset.nc"

    print("\n========== SAVING MASTER DATASET ==========")

    encoding = {
        var: {
            "zlib": True,
            "complevel": 4
        }
        for var in ds.data_vars
    }

    ds.to_netcdf(
        output_path,
        encoding=encoding
    )

    print(f"Saved master dataset: {output_path}")


def main():
    cfg = load_config()

    ds = build_master_dataset(cfg)

    save_master_dataset(ds)

    print("\n========== MASTER DATASET BUILD COMPLETE ==========")


if __name__ == "__main__":
    main()