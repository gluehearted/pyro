import yaml
from src.pipeline.preprocess_pipeline import load_era5, preprocess_era5

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

print(cfg)
print(cfg["data_sources"])
print(cfg["data_sources"]["era5"])
ds = load_era5(cfg)

print("\n=== INSTANT VARIABLES ===")
print(list(ds_instant.data_vars))

print("\n=== ACCUM VARIABLES ===")
print(list(ds_accum.data_vars))

daily = preprocess_era5(ds, cfg)

print(daily)
