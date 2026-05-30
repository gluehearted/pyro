from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]

era5_path = ROOT_DIR / "data" / "interim" / "era5" / "era5_daily_clean.nc"
output_dir = ROOT_DIR / "outputs" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

ds = xr.open_dataset(era5_path)

print("\n========== DATASET INFO ==========")
print(ds)

print("\n========== VARIABLES ==========")
print(list(ds.data_vars))

date = "2024-09-01"

plt.figure(figsize=(10, 6))
ds["t2m"].sel(time=date).plot()
plt.title(f"ERA5 2m Temperature - {date}")
plt.savefig(output_dir / f"era5_t2m_{date}.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 6))
ds["relative_humidity"].sel(time=date).plot()
plt.title(f"ERA5 Relative Humidity - {date}")
plt.savefig(output_dir / f"era5_rh_{date}.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 6))
ds["tp"].sel(time=date).plot()
plt.title(f"ERA5 Daily Rainfall - {date}")
plt.savefig(output_dir / f"era5_rainfall_{date}.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 6))
ds["swvl1"].sel(time=date).plot()
plt.title(f"ERA5 Soil Moisture Layer 1 - {date}")
plt.savefig(output_dir / f"era5_soil_moisture_{date}.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved figures to:")
print(output_dir)