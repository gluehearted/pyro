from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import xarray as xr
import yaml

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:
    raise ImportError(
        "pyarrow belum terinstall. Jalankan: pip install pyarrow"
    ) from exc


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.utils.experiment import (
    MASTER_PATH,
    XGB_DIR,
    TARGET_COL,
    EXP_NAME,
    HORIZON_DAYS,
    ensure_experiment_dirs,
    print_experiment_info,
)

from src.features.temporal_split import (
    get_split_ranges,
    iter_time_chunks,
)

OUTPUT_DIR = XGB_DIR

# Untuk train, negatif bisa sangat banyak.
# None  = pakai semua data
# 10    = maksimal 10 negatif untuk setiap 1 positif
TRAIN_NEG_POS_RATIO = 10

CHUNK_DAYS = 30
RANDOM_SEED = 42


DEFAULT_FEATURES = [
    # ERA5 base features
    "t2m",
    "d2m",
    "u10",
    "v10",
    "wind_speed",
    "relative_humidity",
    "swvl1",
    "tp",
    "ssrd",

    # ERA5 rolling features
    "rainfall_sum_3d",
    "rainfall_sum_7d",
    "rainfall_sum_14d",
    "temperature_mean_3d",
    "temperature_mean_7d",
    "temperature_mean_14d",
    "wind_speed_mean_3d",
    "wind_speed_mean_7d",
    "wind_speed_mean_14d",

    # temporal features
    "month",
    "dayofyear",
    "sin_doy",
    "cos_doy",

    # FIRMS features
    "hotspot_count",
    "mean_frp",
    "max_frp",
    "hotspot_binary",
    "hotspot_count_3d",
    "hotspot_count_7d",
    "hotspot_count_14d",

    # static spatial features
    "peat_fraction",
    "peat_binary",
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

    extra_label_path = (
    ROOT_DIR
        / "data"
        / "processed"
        / "master"
        / f"labels_{TARGET_COL}.nc"
    )

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
            print(f"[SKIP] Feature tidak ditemukan di dataset: {col}")

    if TARGET_COL not in ds.data_vars:
        raise ValueError(f"Target {TARGET_COL} tidak ditemukan di master dataset.")

    return available


def downsample_train_negatives(
    df: pd.DataFrame,
    target_col: str,
    ratio: int | None,
    seed: int,
) -> pd.DataFrame:
    if ratio is None:
        return df

    pos = df[df[target_col] == 1]
    neg = df[df[target_col] == 0]

    if len(pos) == 0 or len(neg) == 0:
        return df

    max_neg = min(len(neg), len(pos) * ratio)

    neg_sample = neg.sample(
        n=max_neg,
        random_state=seed
    )

    sampled = pd.concat([pos, neg_sample], ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return sampled


def clean_chunk_dataframe(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> pd.DataFrame:
    df = df.copy()

    df = df.dropna(subset=[target_col])

    df[target_col] = (
        pd.to_numeric(df[target_col], errors="coerce")
        .fillna(0)
        .round()
        .astype(np.int8)
    )

    # Koordinat juga dipakai sebagai fitur tabular.
    numeric_cols = ["latitude", "longitude"] + feature_cols

    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
                .astype(np.float32)
            )

    df["time"] = pd.to_datetime(df["time"])

    ordered_cols = (
        ["time", "latitude", "longitude"]
        + feature_cols
        + [target_col]
    )

    ordered_cols = [c for c in ordered_cols if c in df.columns]

    return df[ordered_cols]


def write_parquet_stream(
    ds: xr.Dataset,
    split_name: str,
    start,
    end,
    feature_cols: list[str],
    target_col: str,
    output_path: Path,
    chunk_days: int,
    train_neg_pos_ratio: int | None,
):
    print(f"\n========== BUILD XGB {split_name.upper()} ==========")
    print(f"Period: {start.date()} → {end.date()}")
    print(f"Output: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    total_rows = 0
    total_pos = 0
    total_neg = 0

    selected_vars = feature_cols + [target_col]

    for chunk_start, chunk_end in iter_time_chunks(start, end, chunk_days):
        print(f"Processing chunk: {chunk_start.date()} → {chunk_end.date()}")

        ds_chunk = ds[selected_vars].sel(
            time=slice(chunk_start, chunk_end)
        ).load()

        if ds_chunk.sizes.get("time", 0) == 0:
            continue

        df = ds_chunk.to_dataframe().reset_index()

        df = clean_chunk_dataframe(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col
        )

        if split_name == "train":
            df = downsample_train_negatives(
                df=df,
                target_col=target_col,
                ratio=train_neg_pos_ratio,
                seed=RANDOM_SEED
            )

        if len(df) == 0:
            continue

        pos_count = int((df[target_col] == 1).sum())
        neg_count = int((df[target_col] == 0).sum())

        total_rows += len(df)
        total_pos += pos_count
        total_neg += neg_count

        table = pa.Table.from_pandas(
            df,
            preserve_index=False
        )

        if writer is None:
            writer = pq.ParquetWriter(
                output_path,
                table.schema,
                compression="snappy"
            )

        writer.write_table(table)

        del df, ds_chunk, table

    if writer is not None:
        writer.close()
    else:
        raise RuntimeError(f"Tidak ada data yang ditulis untuk split: {split_name}")

    print(f"\nFinished {split_name}")
    print(f"Rows     : {total_rows:,}")
    print(f"Positive : {total_pos:,}")
    print(f"Negative : {total_neg:,}")

    return {
        "rows": total_rows,
        "positive": total_pos,
        "negative": total_neg,
        "positive_ratio": total_pos / max(total_rows, 1),
    }


def main():
    ensure_experiment_dirs()
    print_experiment_info()
    cfg = load_config()
    ds = load_master_dataset()

    feature_cols = get_available_features(ds, DEFAULT_FEATURES)

    # latitude dan longitude tidak ada di data_vars, tapi akan muncul dari reset_index().
    model_feature_cols = ["latitude", "longitude"] + feature_cols

    split_ranges = get_split_ranges(cfg)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "experiment": EXP_NAME,
        "horizon_days": HORIZON_DAYS,
        "target": TARGET_COL,
        "feature_columns": model_feature_cols,
        "xarray_feature_vars": feature_cols,
        "train_negative_positive_ratio": TRAIN_NEG_POS_RATIO,
        "chunk_days": CHUNK_DAYS,
        "splits": {},
    }

    for split_name, split in split_ranges.items():
        output_path = OUTPUT_DIR / f"{split_name}.parquet"

        stats = write_parquet_stream(
            ds=ds,
            split_name=split_name,
            start=split.start,
            end=split.end,
            feature_cols=feature_cols,
            target_col=TARGET_COL,
            output_path=output_path,
            chunk_days=CHUNK_DAYS,
            train_neg_pos_ratio=TRAIN_NEG_POS_RATIO,
        )

        metadata["splits"][split_name] = {
            "start": str(split.start.date()),
            "end": str(split.end.date()),
            **stats,
        }

    metadata_path = OUTPUT_DIR / "xgb_metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n========== XGB FEATURE BUILD COMPLETE ==========")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()