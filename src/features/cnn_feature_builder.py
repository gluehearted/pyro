from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import xarray as xr
import yaml
from numpy.lib.format import open_memmap


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.features.temporal_split import get_split_ranges

from src.utils.experiment import (
    MASTER_PATH,
    CNN_DIR,
    TARGET_COL,
    EXP_NAME,
    HORIZON_DAYS,
    ensure_experiment_dirs,
    print_experiment_info,
)

OUTPUT_DIR = CNN_DIR
SEQUENCE_LENGTH = 7

# Channel CNN sebaiknya tidak terlalu banyak dulu agar storage tidak meledak.
CNN_FEATURES = [
    "t2m",
    "relative_humidity",
    "wind_speed",
    "swvl1",
    "tp",
    "ssrd",

    "rainfall_sum_3d",
    "rainfall_sum_7d",
    "rainfall_sum_14d",

    "temperature_mean_3d",
    "temperature_mean_7d",
    "temperature_mean_14d",

    "wind_speed_mean_3d",
    "wind_speed_mean_7d",

    "hotspot_count",
    "hotspot_count_3d",
    "hotspot_count_7d",
    "hotspot_count_14d",

    "mean_frp",
    "max_frp",

    "peat_fraction",
    "peat_binary",

    "sin_doy",
    "cos_doy",
]


def load_config() -> dict:
    config_path = ROOT_DIR / "config" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config tidak ditemukan: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_master_dataset() -> xr.Dataset:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Master dataset tidak ditemukan: {MASTER_PATH}\n"
            "Jalankan build_master_dataset.py dulu."
        )

    ds = xr.open_dataset(
        MASTER_PATH,
        chunks={"time": 90}
    )

    extra_label_path = ROOT_DIR / "data" / "processed" / "master" / f"labels_{TARGET_COL}.nc"

    if TARGET_COL not in ds.data_vars and extra_label_path.exists():
        print(f"Loading extra label: {extra_label_path}")
        ds_label = xr.open_dataset(extra_label_path, chunks={"time": 90})
        ds = xr.merge([ds, ds_label])

    print("\n========== MASTER DATASET ==========")
    print(ds)

    return ds


def get_available_features(ds: xr.Dataset, requested_features: list[str]) -> list[str]:
    available = []

    for col in requested_features:
        if col in ds.data_vars:
            available.append(col)
        else:
            print(f"[SKIP] CNN feature tidak ditemukan: {col}")

    if TARGET_COL not in ds.data_vars:
        raise ValueError(f"Target {TARGET_COL} tidak ditemukan.")

    return available


def get_reference_3d_var(ds: xr.Dataset, feature_cols: list[str]) -> str:
    for col in feature_cols:
        dims = set(ds[col].dims)
        if {"time", "latitude", "longitude"}.issubset(dims):
            return col

    raise ValueError(
        "Tidak ada feature dengan dimensi time, latitude, longitude "
        "untuk dijadikan reference grid."
    )


def compute_train_channel_stats(
    ds: xr.Dataset,
    feature_cols: list[str],
    train_start,
    train_end,
    reference_var: str,
) -> dict:
    print("\n========== COMPUTING TRAIN NORMALIZATION STATS ==========")

    ref = ds[reference_var].sel(
        time=slice(train_start, train_end)
    )

    stats = {}

    for col in feature_cols:
        print(f"Computing stats: {col}")

        da = ds[col]

        if "time" in da.dims:
            da = da.sel(time=slice(train_start, train_end))

        # Broadcast static 2D atau time-only feature ke bentuk time × lat × lon.
        da = da.broadcast_like(ref)

        mean = float(da.mean(skipna=True).compute())
        std = float(da.std(skipna=True).compute())

        if not np.isfinite(mean):
            mean = 0.0

        if not np.isfinite(std) or std < 1e-6:
            std = 1.0

        stats[col] = {
            "mean": mean,
            "std": std,
        }

        print(f"  mean={mean:.6f}, std={std:.6f}")

    return stats


def make_feature_array_for_split(
    ds: xr.Dataset,
    feature_cols: list[str],
    start,
    end,
    reference_var: str,
    stats: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return:
    feature_arr: (T, C, H, W)
    times      : (T,)
    """

    ref = ds[reference_var].sel(
        time=slice(start, end)
    )

    times = ref["time"].values
    n_time = ref.sizes["time"]
    n_lat = ref.sizes["latitude"]
    n_lon = ref.sizes["longitude"]
    n_channels = len(feature_cols)

    feature_arr = np.empty(
        (n_time, n_channels, n_lat, n_lon),
        dtype=np.float32
    )

    for ci, col in enumerate(feature_cols):
        print(f"Loading channel {ci + 1}/{n_channels}: {col}")

        da = ds[col]

        if "time" in da.dims:
            da = da.sel(time=slice(start, end))

        da = da.broadcast_like(ref)
        da = da.transpose("time", "latitude", "longitude")

        arr = np.asarray(da.values, dtype=np.float32)

        mean = stats[col]["mean"]
        std = stats[col]["std"]

        arr = (arr - mean) / std
        arr = np.nan_to_num(
            arr,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        ).astype(np.float32)

        feature_arr[:, ci, :, :] = arr

        del arr, da

    return feature_arr, times


def make_label_array_for_split(
    ds: xr.Dataset,
    target_col: str,
    start,
    end,
) -> np.ndarray:
    y = ds[target_col].sel(
        time=slice(start, end)
    )

    y = y.transpose("time", "latitude", "longitude")

    arr = np.asarray(y.values, dtype=np.float32)

    nan_count = int(np.isnan(arr).sum())
    if nan_count > 0:
        raise ValueError(
            f"Masih ada NaN pada label {target_col}: {nan_count}. "
            "Filter valid target times harus dilakukan sebelum membuat label array."
        )

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)

    return arr


def build_sequence_memmap(
    split_name: str,
    feature_arr: np.ndarray,
    label_arr: np.ndarray,
    times: np.ndarray,
    sequence_length: int,
    output_dir: Path,
):
    """
    feature_arr: (T, C, H, W)
    label_arr  : (T, H, W)

    X output   : (N, seq, C, H, W)
    y output   : (N, H, W)
    """

    n_time, n_channels, n_lat, n_lon = feature_arr.shape

    n_samples = n_time - sequence_length + 1

    if n_samples <= 0:
        raise ValueError(
            f"Split {split_name} terlalu pendek untuk sequence_length={sequence_length}"
        )

    x_path = output_dir / f"X_{split_name}.npy"
    y_path = output_dir / f"y_{split_name}.npy"
    times_path = output_dir / f"times_{split_name}.npy"

    print(f"\n========== BUILD CNN {split_name.upper()} ==========")
    print(f"X shape: {(n_samples, sequence_length, n_channels, n_lat, n_lon)}")
    print(f"y shape: {(n_samples, n_lat, n_lon)}")

    X = open_memmap(
        x_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_samples, sequence_length, n_channels, n_lat, n_lon)
    )

    y = open_memmap(
        y_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_samples, n_lat, n_lon)
    )

    sample_times = []

    for sample_idx in range(n_samples):
        start_idx = sample_idx
        end_idx = sample_idx + sequence_length
        target_idx = end_idx - 1

        X[sample_idx] = feature_arr[start_idx:end_idx]
        y[sample_idx] = label_arr[target_idx]

        sample_times.append(times[target_idx])

        if sample_idx % 250 == 0:
            print(f"{split_name}: {sample_idx}/{n_samples}")

    X.flush()
    y.flush()

    np.save(times_path, np.array(sample_times))

    print(f"Saved: {x_path}")
    print(f"Saved: {y_path}")
    print(f"Saved: {times_path}")

    return {
        "samples": int(n_samples),
        "sequence_length": int(sequence_length),
        "channels": int(n_channels),
        "height": int(n_lat),
        "width": int(n_lon),
        "x_path": str(x_path),
        "y_path": str(y_path),
        "times_path": str(times_path),
    }

def pd_timestamp_to_date(value):
    return np.datetime_as_string(value, unit="D")

def filter_valid_target_times(
    ds: xr.Dataset,
    target_col: str,
    start,
    end,
) -> xr.Dataset:
    """
    Untuk horizon seperti fire_next_14d, beberapa tanggal terakhir
    punya label NaN karena future window tidak lengkap.
    Tanggal seperti itu harus dibuang sebelum build CNN-BiLSTM tensor.
    """

    ds_split = ds.sel(time=slice(start, end))

    # Cek tanggal yang target-nya valid.
    # Pakai all() supaya tanggal dianggap valid hanya kalau seluruh grid tidak NaN.
    target_valid = ds_split[target_col].notnull().all(
        dim=("latitude", "longitude")
    )

    # Penting: target_valid masih dask array, jadi harus dihitung dulu.
    target_valid = target_valid.compute()

    valid_times = ds_split["time"].where(target_valid, drop=True)

    ds_split = ds_split.sel(time=valid_times)

    if ds_split.sizes.get("time", 0) == 0:
        raise ValueError(
            f"Tidak ada valid time untuk target {target_col} "
            f"pada periode {start} sampai {end}"
        )

    print(
        "Valid target period:",
        str(ds_split.time.values[0])[:10],
        "→",
        str(ds_split.time.values[-1])[:10]
    )

    return ds_split

def main():
    ensure_experiment_dirs()
    print_experiment_info()
    cfg = load_config()
    ds = load_master_dataset()

    feature_cols = get_available_features(ds, CNN_FEATURES)
    reference_var = get_reference_3d_var(ds, feature_cols)

    split_ranges = get_split_ranges(cfg)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_split = split_ranges["train"]

    stats = compute_train_channel_stats(
        ds=ds,
        feature_cols=feature_cols,
        train_start=train_split.start,
        train_end=train_split.end,
        reference_var=reference_var,
    )

    metadata = {
        "experiment": EXP_NAME,
        "target": TARGET_COL,
        "horizon_days": HORIZON_DAYS,
        "sequence_length": SEQUENCE_LENGTH,
        "feature_columns": feature_cols,
        "reference_var": reference_var,
        "normalization": {
            "method": "standard",
            "fit_on": "train_only",
            "channel_stats": stats,
        },
        "splits": {},
    }

    for split_name, split in split_ranges.items():
        print(f"\n========== PREPARING SPLIT: {split_name.upper()} ==========")
        print(f"Original period: {split.start.date()} → {split.end.date()}")

        ds_split = filter_valid_target_times(
            ds=ds,
            target_col=TARGET_COL,
            start=split.start,
            end=split.end,
        )

        split_start = ds_split.time.values[0]
        split_end = ds_split.time.values[-1]

        feature_arr, times = make_feature_array_for_split(
            ds=ds_split,
            feature_cols=feature_cols,
            start=split_start,
            end=split_end,
            reference_var=reference_var,
            stats=stats,
        )

        label_arr = make_label_array_for_split(
            ds=ds_split,
            target_col=TARGET_COL,
            start=split_start,
            end=split_end,
        )

        split_meta = build_sequence_memmap(
            split_name=split_name,
            feature_arr=feature_arr,
            label_arr=label_arr,
            times=times,
            sequence_length=SEQUENCE_LENGTH,
            output_dir=OUTPUT_DIR,
        )

        metadata["splits"][split_name] = {
            "start": str(pd_timestamp_to_date(split_start)),
            "end": str(pd_timestamp_to_date(split_end)),
            "original_start": str(split.start.date()),
            "original_end": str(split.end.date()),
            **split_meta,
        }

        del ds_split, feature_arr, label_arr

    metadata_path = OUTPUT_DIR / "cnn_metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n========== CNN FEATURE BUILD COMPLETE ==========")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()