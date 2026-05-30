from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

ROOT_DIR = Path(__file__).resolve().parents[2]

CNN_DIR = ROOT_DIR / "data" / "processed" / "cnn"
MASTER_PATH = ROOT_DIR / "data" / "processed" / "master" / "master_dataset.nc"

TARGET_COL = "fire_next_1d"


def repair_split(ds, split_name):
    print(f"\n========== REPAIR {split_name.upper()} LABEL ==========")

    times_path = CNN_DIR / f"times_{split_name}.npy"
    y_path = CNN_DIR / f"y_{split_name}.npy"
    backup_path = CNN_DIR / f"y_{split_name}.backup.npy"

    if not times_path.exists():
        raise FileNotFoundError(f"Times file tidak ditemukan: {times_path}")

    if not y_path.exists():
        raise FileNotFoundError(f"Old label file tidak ditemukan: {y_path}")

    old_y = np.load(y_path, mmap_mode="r")
    times = np.load(times_path, allow_pickle=True)

    times = pd.to_datetime(times).normalize()

    master_y = (
        ds[TARGET_COL]
        .sel(time=times)
        .transpose("time", "latitude", "longitude")
        .values
        .astype(np.float32)
    )

    print("Old y shape     :", old_y.shape)
    print("Master y shape  :", master_y.shape)

    if old_y.shape != master_y.shape:
        raise ValueError(
            f"Shape mismatch untuk {split_name}: old={old_y.shape}, master={master_y.shape}"
        )

    old_pos = float(old_y.sum())
    master_pos = float(master_y.sum())
    diff = float(np.abs(old_y - master_y).sum())

    print("Old positive    :", old_pos)
    print("Master positive :", master_pos)
    print("Difference      :", diff)

    if not backup_path.exists():
        print(f"Backup old label ke: {backup_path}")
        old_y_full = np.array(old_y, dtype=np.float32, copy=True)
        np.save(backup_path, old_y_full)

    print(f"Overwrite label: {y_path}")
    np.save(y_path, master_y)

    repaired_y = np.load(y_path, mmap_mode="r")

    print("Repaired positive:", float(repaired_y.sum()))
    print("Repaired ratio   :", float(repaired_y.mean()))

    return diff


def main():
    print("\n========== LOAD MASTER DATASET ==========")

    ds = xr.open_dataset(MASTER_PATH)

    diffs = {}

    for split_name in ["train", "val", "test"]:
        diffs[split_name] = repair_split(ds, split_name)

    print("\n========== SUMMARY ==========")
    for split_name, diff in diffs.items():
        print(f"{split_name}: diff={diff}")

    print("\nDONE. CNN labels repaired from master_dataset.nc.")


if __name__ == "__main__":
    main()