import yaml

from era5_preprocessing import preprocess_era5


with open("config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

ds = preprocess_era5(cfg)

print(ds)