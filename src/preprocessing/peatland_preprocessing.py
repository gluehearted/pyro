from pathlib import Path
from glob import glob
import yaml
import numpy as np
import xarray as xr
import rasterio
from pyproj import Transformer


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
            f"File ERA5 clean tidak ditemukan: {era5_path}\n"
            "Jalankan preprocessing ERA5 dulu."
        )

    ds = xr.open_dataset(era5_path)

    print("\n========== ERA5 REFERENCE GRID ==========")
    print(ds)

    return ds


def resolve_peatland_paths(cfg):
    root = get_project_root()
    peat_cfg = cfg["data_sources"]["peatlands"]

    paths = []

    if "paths" in peat_cfg:
        for pattern in peat_cfg["paths"]:
            full_pattern = root / pattern
            paths.extend(glob(str(full_pattern), recursive=True))

    elif "path" in peat_cfg:
        single_path = root / peat_cfg["path"]
        paths.append(str(single_path))

    paths = sorted(list(set(paths)))

    if not paths:
        raise FileNotFoundError(
            "Tidak ada file .tif peatland ditemukan.\n"
            "Cek config/config.yaml pada bagian data_sources.peatlands."
        )

    print("\n========== PEATLAND TILE FILES ==========")
    for p in paths:
        print(p)

    print(f"\nTotal peatland tiles: {len(paths)}")

    return paths


def sample_single_tile_to_era5_grid(tif_path, era5_ds, peat_threshold=0):
    latitudes = era5_ds["latitude"].values
    longitudes = era5_ds["longitude"].values

    lon2d, lat2d = np.meshgrid(longitudes, latitudes)

    sampled_grid = np.zeros(
        (len(latitudes), len(longitudes)),
        dtype=np.float32
    )

    with rasterio.open(tif_path) as src:
        print("\nReading tile:")
        print(tif_path)
        print(f"CRS    : {src.crs}")
        print(f"Bounds : {src.bounds}")
        print(f"Shape  : {src.height} x {src.width}")
        print(f"Nodata : {src.nodata}")

        if src.crs is None:
            raise ValueError(
                f"Raster tidak punya CRS: {tif_path}\n"
                "Pastikan file punya CRS atau set CRS terlebih dahulu."
            )

        # Transform ERA5 lon/lat EPSG:4326 ke CRS raster
        transformer = Transformer.from_crs(
            "EPSG:4326",
            src.crs,
            always_xy=True
        )

        xs, ys = transformer.transform(
            lon2d.ravel(),
            lat2d.ravel()
        )

        xs = np.array(xs)
        ys = np.array(ys)

        bounds = src.bounds

        inside = (
            (xs >= bounds.left)
            & (xs <= bounds.right)
            & (ys >= bounds.bottom)
            & (ys <= bounds.top)
        )

        inside_count = int(inside.sum())
        print(f"ERA5 grid points inside tile: {inside_count}")

        if inside_count == 0:
            return sampled_grid

        coords = list(zip(xs[inside], ys[inside]))

        values = []

        for val in src.sample(coords):
            v = val[0]

            if src.nodata is not None and v == src.nodata:
                values.append(0.0)
            elif np.isnan(v):
                values.append(0.0)
            else:
                values.append(float(v))

        values = np.array(values, dtype=np.float32)

        flat = sampled_grid.ravel()
        inside_indices = np.where(inside)[0]
        flat[inside_indices] = values

        sampled_grid = flat.reshape(
            len(latitudes),
            len(longitudes)
        )

    # Binary peat indicator
    sampled_grid = np.where(
        sampled_grid > peat_threshold,
        1.0,
        0.0
    ).astype(np.float32)

    return sampled_grid


def preprocess_peatland(cfg, era5_ds):
    print("\n========== PEATLAND PREPROCESSING ==========")

    peat_paths = resolve_peatland_paths(cfg)

    peat_threshold = cfg["data_sources"]["peatlands"].get(
        "peat_threshold",
        0
    )

    latitudes = era5_ds["latitude"].values
    longitudes = era5_ds["longitude"].values

    combined_binary = np.zeros(
        (len(latitudes), len(longitudes)),
        dtype=np.float32
    )

    for tif_path in peat_paths:
        tile_binary = sample_single_tile_to_era5_grid(
            tif_path=tif_path,
            era5_ds=era5_ds,
            peat_threshold=peat_threshold
        )

        # Gabungkan multi-tile.
        # Kalau salah satu tile bilang grid itu peat, maka dianggap peat.
        combined_binary = np.maximum(
            combined_binary,
            tile_binary
        )

    peat_binary = combined_binary.astype(np.float32)

    # Untuk tahap awal, peat_fraction dibuat sama dengan binary.
    # Nanti bisa di-upgrade menjadi true fraction per grid ERA5.
    peat_fraction = peat_binary.copy()

    ds_peat = xr.Dataset(
        data_vars={
            "peat_fraction": (
                ["latitude", "longitude"],
                peat_fraction
            ),
            "peat_binary": (
                ["latitude", "longitude"],
                peat_binary
            ),
        },
        coords={
            "latitude": latitudes,
            "longitude": longitudes
        }
    )

    print("\n========== PEATLAND ALIGNED DATASET ==========")
    print(ds_peat)

    print("\nPeat binary min:", float(ds_peat["peat_binary"].min()))
    print("Peat binary max:", float(ds_peat["peat_binary"].max()))
    print("Peat grid count :", int(ds_peat["peat_binary"].sum()))

    return ds_peat


def save_peatland_output(ds_peat):
    root = get_project_root()

    output_dir = root / "data" / "interim" / "peatland"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "peatland_aligned.nc"

    print("\n========== SAVING PEATLAND OUTPUT ==========")

    encoding = {
        var: {
            "zlib": True,
            "complevel": 4
        }
        for var in ds_peat.data_vars
    }

    ds_peat.to_netcdf(
        output_path,
        encoding=encoding
    )

    print(f"Saved peatland aligned dataset: {output_path}")


def main():
    cfg = load_config()

    era5_ds = load_era5_reference()

    ds_peat = preprocess_peatland(cfg, era5_ds)

    save_peatland_output(ds_peat)

    print("\n========== PEATLAND PREPROCESSING COMPLETE ==========")


if __name__ == "__main__":
    main()