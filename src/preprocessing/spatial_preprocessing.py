import rioxarray
import xarray as xr


def preprocess_forest_cover(
    forest_path,
    era5_ds
):
    forest = rioxarray.open_rasterio(forest_path)

    forest = forest.squeeze()

    forest = forest.rename({
        "x": "longitude",
        "y": "latitude"
    })

    forest = forest.interp(
        latitude=era5_ds.latitude,
        longitude=era5_ds.longitude,
        method="nearest"
    )

    forest = forest.fillna(0)

    ds = xr.Dataset({
        "forest_cover": forest
    })

    return ds