import os
import cdsapi

c = cdsapi.Client()

OUTPUT_DIR = "data/raw/era5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS = range(2015, 2025)

for year in YEARS:
    output_file = f"{OUTPUT_DIR}/era5_{year}.nc"

    if os.path.exists(output_file):
        print(f"SKIP: {output_file} sudah ada")
        continue

    print(f"\nDownloading ERA5 {year}...")

    try:
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": [
                    "2m_temperature",
                    "2m_dewpoint_temperature",
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "total_precipitation",
                    "surface_solar_radiation_downwards",
                    "volumetric_soil_water_layer_1",
                ],
                "year": str(year),
                "month": [f"{i:02d}" for i in range(1, 13)],
                "day": [f"{i:02d}" for i in range(1, 32)],
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "area": [8.5, 94.0, -6.5, 120.5],
            },
            output_file,
        )

        print(f"DONE: {output_file}")

    except Exception as e:
        print(f"ERROR saat download tahun {year}: {e}")

        if os.path.exists(output_file):
            print(f"Menghapus file corrupt: {output_file}")
            os.remove(output_file)

        print("Silakan run ulang script nanti.")