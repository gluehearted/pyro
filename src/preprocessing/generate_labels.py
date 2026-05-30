import numpy as np
import xarray as xr


def generate_fire_labels(
    ds,
    hotspot_var="hotspot_count",
    horizons=(1, 3, 7)
):
    """
    Membuat label prediksi kebakaran berbasis hotspot masa depan.

    Contoh:
    - fire_next_1d: apakah besok ada hotspot?
    - fire_next_3d: apakah 1-3 hari ke depan ada hotspot?
    - fire_next_7d: apakah 1-7 hari ke depan ada hotspot?
    """

    if hotspot_var not in ds:
        raise ValueError(
            f"Variabel {hotspot_var} tidak ditemukan di dataset."
        )

    hotspot = ds[hotspot_var]

    for horizon in horizons:
        future_layers = []

        for step in range(1, horizon + 1):
            future_layers.append(
                hotspot.shift(time=-step)
            )

        future_sum = xr.concat(
            future_layers,
            dim="lead_time"
        ).sum(dim="lead_time", skipna=False)

        label_name = f"fire_next_{horizon}d"

        ds[label_name] = xr.where(
            future_sum > 0,
            1,
            0
        ).astype("float32")

        # Bagian akhir dataset tidak punya masa depan lengkap.
        # Misal label 7 hari butuh data t+1 sampai t+7.
        # Maka 7 hari terakhir harus dibuat NaN agar tidak misleading.
        ds[label_name][-horizon:] = np.nan

    return ds


def drop_invalid_label_times(ds, max_horizon=7):
    """
    Menghapus time-step paling akhir yang tidak punya label masa depan lengkap.
    """

    if max_horizon > 0:
        ds = ds.isel(time=slice(0, -max_horizon))

    return ds