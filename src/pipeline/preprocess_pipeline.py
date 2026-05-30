import yaml

from src.preprocessing.era5_preprocessing import preprocess_era5
from src.preprocessing.fire_preprocessing import preprocess_fire
from src.preprocessing.peatland_preprocessing import preprocess_peatland
from src.preprocessing.build_master_dataset import build_master_dataset
from src.preprocessing.generate_labels import generate_labels
from src.preprocessing.spatial_preprocessing import validate_alignment

with open("config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

print("Loading ERA5...")
era5_ds = preprocess_era5(cfg)

print("Loading Fire Data...")
fire_ds = preprocess_fire(cfg, era5_ds)

print("Loading Peatland...")
peat_ds = preprocess_peatland(cfg, era5_ds)

print("Building Master Dataset...")
master_ds = build_master_dataset(
    era5_ds,
    fire_ds,
    peat_ds
)

print("Validating Spatial Alignment...")
validate_alignment(master_ds)

print("Generating Labels...")
master_ds = generate_labels(master_ds)

print(master_ds)

master_ds.to_netcdf(
    "data/processed/master_dataset.nc"
)

print("DONE")